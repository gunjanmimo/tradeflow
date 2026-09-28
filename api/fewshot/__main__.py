"""
    cd api
    python -m fewshot                         # all adapters, horizon 30 min, K = 10 days
    python -m fewshot --horizon 60 --k-days 5 --adapters static,fomaml
    python -m fewshot --no-chronos            # skip the pretrained base

Trains the static base and the meta-learned base on the training days, then
walks forward over the validation and test days: each day is adapted on the K
days before it and traded with the entry rule "forecast > round-trip cost".
Writes models/fewshot/report.md and report.json.
"""
import argparse
import json
import os
import time

import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "fewshot")
ALL = ("static", "finetune", "fomaml", "chronos_zero", "chronos_fewshot")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m fewshot")
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--k-days", type=int, default=10)
    ap.add_argument("--inner-steps", type=int, default=20)
    ap.add_argument("--inner-lr", type=float, default=0.05)
    ap.add_argument("--adapters", default=",".join(ALL))
    ap.add_argument("--no-chronos", action="store_true")
    ap.add_argument("--meta-steps", type=int, default=3000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    import torch
    from core.config import settings
    from core.costs import DEFAULT_COSTS
    from rl import dataset
    from rl.env import SessionTensors, EnvRules
    from rl.evaluate import summarize, flat_policy, long_policy, random_policy, run
    from rl.train import split_days
    from fewshot import data as fdata, meta, evaluate as fe, chronos as ch

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    adapters = [x.strip() for x in a.adapters.split(",") if x.strip()]
    if a.no_chronos or not ch.available():
        adapters = [x for x in adapters if not x.startswith("chronos")]
    t0 = time.time()
    ss = dataset.build(verbose=False)
    tr_days, va_days, te_days = split_days(ss.day)
    norm = dataset.norm_stats(ss.subset(np.isin(ss.day, tr_days)))
    ds = fdata.build(ss, norm, a.horizon)
    scale = meta.target_scale(ds.y[np.isin(ds.day, tr_days)])
    R = meta.to_rows(ds, scale, device)
    n_tr, n_va = len(tr_days), len(va_days)
    TR, VA, TE = (0, n_tr), (n_tr, n_tr + n_va), (n_tr + n_va, len(R.days))
    rt = DEFAULT_COSTS.round_trip_bps
    print(f"{len(ss)} sessions, {len(R.days)} days (train {n_tr}, val {n_va}, test {len(te_days)}); "
          f"horizon {a.horizon} min; target spread {scale:.1f} bps; entry when forecast > {rt:g} bps")

    models = {}
    if {"static", "finetune"} & set(adapters):
        print("Training the static base...")
        models["static"], _ = meta.train_static(R, TR, VA, device, seed=a.seed, verbose=True)
    if "fomaml" in adapters:
        print("Meta-training (first-order MAML)...")
        models["fomaml"], _ = meta.fomaml(R, TR, VA, device, a.k_days, a.inner_steps, a.inner_lr,
                                          meta_steps=a.meta_steps, seed=a.seed, init=models.get("static"),
                                          verbose=True)
    chron = None
    if any(x.startswith("chronos") for x in adapters):
        print("Chronos-Bolt forecasts (the K days before validation onwards)...")
        first = max(0, n_tr - a.k_days)
        sess = np.flatnonzero(np.isin(ss.day, R.days[first:]))
        fc = ch.forecasts(ss, sess, a.horizon, device, cache_key=f"{len(ss)}-{ss.day[0]}-{ss.day[-1]}")
        chron = np.full(len(ss) * fdata.DECISIONS.size, np.nan, dtype=np.float32)
        chron.reshape(len(ss), -1)[sess] = fc

    rules = EnvRules.live()
    report = {"horizon": a.horizon, "k_days": a.k_days, "inner_steps": a.inner_steps,
              "inner_lr": a.inner_lr, "round_trip_bps": rt, "splits": [n_tr, n_va, len(te_days)],
              "results": {}}
    for name, rng in (("val", VA), ("test", TE)):
        span = R.span(*rng)
        s0, s1 = span.start // fdata.DECISIONS.size, span.stop // fdata.DECISIONS.size
        sub = ss.subset(np.arange(s0, s1))
        data = SessionTensors(sub, norm, device)
        y = ds.y[s0:s1].reshape(-1)
        day_rows = np.repeat(sub.day, fdata.DECISIONS.size)
        res = {}
        for kind in adapters:
            base = models.get("fomaml" if kind == "fomaml" else "static")
            pred = fe.walk_forward(kind, R, rng, scale, a.k_days, a.inner_steps, a.inner_lr, base=base,
                                   chronos_rows=chron, seed=a.seed)
            ic = fe.daily_ic(pred, y, day_rows)
            per = fe.trade(data, pred.reshape(s1 - s0, -1), rules, enter_bps=rt, device=device)
            ev = summarize(kind, per, sub.day)
            res[kind] = {"ic": ic.to_dict(), "trading": ev.to_dict(),
                         "forecast_above_cost_share": round(float((pred > rt).mean()), 4)}
        base_per = {"flat": run(data, np.arange(s1 - s0), rules, flat_policy),
                    "always_long": run(data, np.arange(s1 - s0), rules, long_policy)}
        turn = max((r["trading"]["trades_per_session"] for r in res.values()), default=0.5)
        base_per["random"] = run(data, np.arange(s1 - s0), rules,
                                 random_policy(min(1.0, max(2 * turn / 75, 1 / 75))))
        for k, per in base_per.items():
            res[k] = {"ic": None, "trading": summarize(k, per, sub.day).to_dict()}
        report["results"][name] = res
        del data
        torch.cuda.empty_cache()

    lines = [f"# Few-shot forecasting: walk-forward ({time.strftime('%Y-%m-%d %H:%M')})", "",
             f"Horizon {a.horizon} min. Each day is adapted on the {a.k_days} trading days before it "
             f"({a.inner_steps} SGD steps, lr {a.inner_lr}). Trades: enter when the forecast exceeds the "
             f"{rt:g} bps round trip, exit when it turns negative; same fills, stops, costs and close as live.", ""]
    for name in ("val", "test"):
        lines += [f"## {name}", "",
                  "| model | mean IC | IC t | IC>0 days | bps/day | t | Sharpe | trades/session | time in market |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for k, r in report["results"][name].items():
            ic, tr_ = r["ic"], r["trading"]
            ic_s = (f"{ic['mean_ic']:+.4f} | {ic['t_ic']:+.2f} | {ic['positive_days']:.0%}" if ic else "— | — | —")
            lines.append(f"| {k} | {ic_s} | {tr_['mean_daily_bps']:+.3f} | {tr_['t_daily']:+.2f} | "
                         f"{tr_['sharpe']:+.2f} | {tr_['trades_per_session']:.2f} | {tr_['exposure']:.1%} |")
        lines.append("")
    md = "\n".join(lines)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    with open(os.path.join(OUT_DIR, "report.md"), "w") as f:
        f.write(md)
    print("\n" + md)
    print(f"({time.time() - t0:.0f}s) wrote {OUT_DIR}/report.md")


if __name__ == "__main__":
    main()
