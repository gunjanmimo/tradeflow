"""
Train the PPO trading policy, evaluate it walk-forward, and decide whether it
may trade.

    cd api
    python -m rl.train                         # all cached symbols, GPU if present
    python -m rl.train --iterations 300 --n-envs 4096
    python -m rl.train --warm-start            # continue from the deployed policy
    python -m rl.train --synthetic             # sanity check on a market with a planted edge

Honest evaluation. Sessions are split by DATE:

  train  the oldest 60% of trading days: the only data PPO learns from
  val    the next 20%: picks the checkpoint (best validation Sharpe)
  test   the newest 20%: touched once, at the end, for the report and the gate

Promotion gate (all on data the policy never trained on):
  * test:  mean daily return > 0 with t >= RL_GATE_MIN_T (default 2)
  * test:  Sharpe above the always-long baseline's (beats just holding)
  * val:   mean daily return > 0 (not a one-split fluke)

A policy that fails the gate is still saved and runs in SHADOW mode: it logs
what it would do, it does not trade. A new policy replaces the deployed one
only if it does better on the same test days (champion / challenger).
"""
import argparse
import copy
import json
import os
import time
from typing import Any, Dict, Optional

import numpy as np

from rl import features as F

REPORT_PATH_NAME = "report.json"


def split_days(days: np.ndarray, train: float = 0.6, val: float = 0.2):
    u = np.unique(days)
    a, b = int(len(u) * train), int(len(u) * (train + val))
    return u[:a], u[a:b], u[b:]


def train_on(ss, iterations: int = 200, n_envs: int = 4096, device: Optional[str] = None,
             hidden=(128, 64), seed: int = 0, eval_every: int = 10, patience: int = 8,
             warm_start: Optional[str] = None, rules=None, verbose: bool = True,
             lr: float = 3e-4, entropy: float = 0.01, window: int = F.WINDOW,
             obs_noise: float = 0.0, max_passes: float = 20.0) -> Dict[str, Any]:
    """
    max_passes caps how often, on average, each training session is replayed:
    beyond a few dozen passes PPO memorises the training days' paths.
    """
    """Trains on the train days of `ss`; returns model, norm, splits, curves and evaluations."""
    import torch
    from rl import dataset, evaluate
    from rl.env import SessionTensors, TradingEnv, EnvRules
    from rl.ppo import ActorCritic, PPOConfig, collect, update, explained_variance

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rules = rules or EnvRules()
    tr_days, va_days, te_days = split_days(ss.day)
    idx = {name: np.flatnonzero(np.isin(ss.day, d)) for name, d in
           (("train", tr_days), ("val", va_days), ("test", te_days))}
    norm = dataset.norm_stats(ss.subset(idx["train"]))
    data = SessionTensors(ss, norm, device)
    cfg = PPOConfig(hidden=tuple(hidden), n_envs=n_envs, lr=lr, entropy=entropy,
                    minibatch=min(8192, max(256, n_envs * 75 // 8)), window=window, obs_noise=obs_noise)
    model = ActorCritic(window, cfg.hidden).to(device)
    if warm_start and os.path.exists(warm_start):
        from rl.policy import load_into_torch
        try:
            load_into_torch(model, warm_start)
            if verbose:
                print(f"Warm start from {warm_start}")
        except Exception as e:
            print(f"Warm start skipped ({e}); training from scratch")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, eps=1e-5, weight_decay=cfg.weight_decay)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    env = TradingEnv(data, rules, n_envs, pool=torch.as_tensor(idx["train"]), generator=gen,
                     window=window)

    def val_score(res) -> float:
        return res.sharpe if res.trades_per_session > 0 else 0.0

    curves = []
    best = {"score": -np.inf, "state": copy.deepcopy(model.state_dict()), "iteration": 0}
    stale, t0 = 0, time.time()
    if max_passes:
        iterations = min(iterations, max(1, int(max_passes * len(idx["train"]) / n_envs)))
        if verbose:
            print(f"  {iterations} iterations ({max_passes:g} passes over {len(idx['train'])} training sessions)")
    for it in range(1, iterations + 1):
        frac = (it - 1) / max(iterations - 1, 1)
        for g in opt.param_groups:
            g["lr"] = cfg.lr * (1 - 0.9 * frac)
        ent_coef = cfg.entropy + (cfg.entropy_final - cfg.entropy) * frac
        model.train()
        ro = collect(model, env, cfg)
        logs = update(model, opt, ro, cfg, ent_coef)
        row = {"iteration": it,
               "train_ret_bps": round(float(ro.ep_return.mean()) * 10, 3),
               "train_trades": round(float(ro.stats["trades"].mean()), 3),
               "explained_var": round(explained_variance(ro.val, ro.ret), 3),
               **{k: round(v, 5) for k, v in logs.items()}}
        if it % eval_every == 0 or it == iterations:
            v = evaluate.evaluate(model, data, idx["val"], ss.day[idx["val"]], rules, baselines=False)["ppo"]
            row.update(val_daily_bps=v.mean_daily_bps, val_sharpe=v.sharpe, val_trades=v.trades_per_session)
            score = val_score(v)
            if score > best["score"] + 1e-6:
                best = {"score": score, "state": copy.deepcopy(model.state_dict()), "iteration": it}
                stale = 0
            else:
                stale += 1
            if verbose:
                print(f"  it {it:4d}  train {row['train_ret_bps']:+7.2f} bps/session  trades {row['train_trades']:5.2f}"
                      f"  ent {row['entropy']:.3f}  EV {row['explained_var']:+.2f}  |  val {v.mean_daily_bps:+7.3f} bps/day"
                      f"  Sharpe {v.sharpe:+5.2f}  trades {v.trades_per_session:5.2f}  ({time.time() - t0:.0f}s)")
        curves.append(row)
        if patience and stale >= patience:
            if verbose:
                print(f"  early stop: no validation improvement in {patience} evaluations")
            break
    model.load_state_dict(best["state"])
    ev_val = evaluate.evaluate(model, data, idx["val"], ss.day[idx["val"]], rules)
    ev_test = evaluate.evaluate(model, data, idx["test"], ss.day[idx["test"]], rules)
    return {"model": model.cpu(), "norm": norm, "rules": rules, "cfg": cfg, "curves": curves,
            "best_iteration": best["iteration"], "val": ev_val, "test": ev_test,
            "splits": {k: [int(d[0]), int(d[-1]), int(len(d))] for k, d in
                       (("train", tr_days), ("val", va_days), ("test", te_days))},
            "device": device, "seconds": round(time.time() - t0, 1), "data": data, "idx": idx}


def gate(res: Dict[str, Any], min_t: float) -> Dict[str, Any]:
    te, va = res["test"], res["val"]
    ppo, hold = te["ppo"], te["always_long"]
    checks = {
        "test_positive": ppo.mean_daily_bps > 0,
        "test_significant": ppo.t_daily >= min_t,
        "beats_always_long": ppo.sharpe > hold.sharpe,
        "val_positive": va["ppo"].mean_daily_bps > 0,
        "trades": ppo.trades_per_session > 0,
    }
    checks = {k: bool(v) for k, v in checks.items()}
    return {"approved": all(checks.values()), "checks": checks, "min_t": min_t}


def champion_score(path: str, res: Dict[str, Any], ss) -> Optional[float]:
    """
    The deployed policy's test Sharpe on THIS run's test sessions, scored with
    its own feature normalisation and window. None when there is no deployed
    policy (or it cannot be loaded).
    """
    from rl.policy import load_into_torch
    from rl.ppo import ActorCritic
    from rl.env import SessionTensors
    from rl import evaluate
    if not os.path.exists(path):
        return None
    try:
        z = np.load(path)
        meta = json.loads(bytes(z["meta_json"]).decode())
        window = int((meta.get("obs") or {}).get("window", F.WINDOW))
        champ = ActorCritic(window, tuple(meta["ppo"]["hidden"]))
        load_into_torch(champ, path)
        norm = {k[5:]: z[k] for k in z.files if k.startswith("norm_")}
        test = res["idx"]["test"]
        sub = ss.subset(test)
        data = SessionTensors(sub, norm, res["device"])
        champ.to(res["device"])
        per = evaluate.run(data, np.arange(len(sub)), res["rules"], evaluate.ppo_policy(champ), window=window)
        return evaluate.summarize("champion", per, sub.day).sharpe
    except Exception as e:
        print(f"Champion evaluation failed ({type(e).__name__}: {e}); keeping the deployed policy")
        return float("inf")


def render(report: Dict[str, Any]) -> str:
    L = [f"# PPO policy report ({report['trained_at']})", ""]
    s = report["splits"]
    L += [f"Symbols: {report['n_symbols']}, sessions: {report['n_sessions']}. Days: train {s['train'][2]}, "
          f"validation {s['val'][2]}, test {s['test'][2]} (chronological). Best iteration "
          f"{report['best_iteration']} of {len(report['curves'])}, {report['seconds']}s on {report['device']}.",
          f"Costs {report['rules']['per_side'] * 1e4:.1f} bps per fill; stop {report['rules']['stop_atr']}x ATR, "
          f"target {report['rules']['reward_risk']}R.", ""]
    for name in ("val", "test"):
        L += [f"## {name}", "",
              "| policy | mean bps/day | t | Sharpe | total bps | max DD bps | trades/session | exposure | win% | stops | targets |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in report[name].values():
            L.append(f"| {r['policy']} | {r['mean_daily_bps']:+.3f} | {r['t_daily']:+.2f} | {r['sharpe']:+.2f} "
                     f"| {r['total_bps']:+.1f} | {r['max_drawdown_bps']:.1f} | {r['trades_per_session']:.2f} "
                     f"| {r['exposure']:.1%} | {r['win_rate']:.0f} | {r['stops_per_session']:.2f} "
                     f"| {r['targets_per_session']:.2f} |")
        L.append("")
    g = report["gate"]
    L += ["## Promotion gate", "", f"**{'APPROVED to trade (paper)' if g['approved'] else 'NOT approved: shadow mode only'}**", ""]
    L += [f"- {k}: {'pass' if v else 'FAIL'}" for k, v in g["checks"].items()]
    L += ["", f"Deployed: {report['deployed']} ({report['deploy_reason']})", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m rl.train")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--n-envs", type=int, default=4096)
    ap.add_argument("--hidden", default="64,32")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--window", type=int, default=F.WINDOW, help="bars of per-bar features in the observation")
    ap.add_argument("--obs-noise", type=float, default=0.0, help="training observation noise (normalised units)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--warm-start", action="store_true", help="continue from the deployed policy")
    ap.add_argument("--synthetic", action="store_true", help="train on a market with a planted edge")
    ap.add_argument("--no-deploy", action="store_true", help="evaluate only; never replace the deployed policy")
    ap.add_argument("--min-t", type=float, default=None)
    a = ap.parse_args(argv)

    from core.config import settings
    from rl import dataset
    from rl.env import EnvRules
    from rl.policy import POLICY_PATH, MODEL_DIR, export

    min_t = a.min_t if a.min_t is not None else settings.RL_GATE_MIN_T
    if a.synthetic:
        from rl import synthetic
        ss = synthetic.make(n_sessions=2000, edge_bps=2.0, seed=a.seed)
        print(f"Synthetic market: {len(ss)} sessions with a planted 2 bps/min edge")
    else:
        syms = [s.strip().upper() for s in a.symbols.split(",") if s.strip()] or None
        ss = dataset.build(syms)
    print(f"{len(ss)} sessions, {len(ss.symbols)} symbols, {len(ss.days())} days")
    rules = EnvRules.live()
    res = train_on(ss, iterations=a.iterations, n_envs=a.n_envs, device=a.device,
                   hidden=[int(x) for x in a.hidden.split(",")], seed=a.seed,
                   warm_start=POLICY_PATH if a.warm_start else None, rules=rules,
                   lr=a.lr, entropy=a.entropy, window=a.window, obs_noise=a.obs_noise)
    g = gate(res, min_t)

    report = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "synthetic": a.synthetic, "symbols": ss.symbols, "n_symbols": len(ss.symbols),
        "n_sessions": len(ss), "splits": res["splits"], "best_iteration": res["best_iteration"],
        "seconds": res["seconds"], "device": res["device"], "rules": rules.to_dict(),
        "ppo": res["cfg"].to_dict(), "curves": res["curves"],
        "val": {k: v.to_dict() for k, v in res["val"].items()},
        "test": {k: v.to_dict() for k, v in res["test"].items()},
        "gate": g, "feature_version": dataset.FEATURE_VERSION,
        "obs": {"window": res["cfg"].window, "bar": list(F.BAR_FEATURES), "scalar": list(F.SCALAR_FEATURES),
                "position": list(F.POSITION_FEATURES)},
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    cand_path = os.path.join(MODEL_DIR, f"candidate-{stamp}.npz")
    export(res["model"], res["norm"], report, cand_path)

    challenger = res["test"]["ppo"].sharpe
    champ = None if a.synthetic else champion_score(POLICY_PATH, res, ss)
    if a.synthetic or a.no_deploy:
        deployed, why = False, "synthetic run" if a.synthetic else "--no-deploy"
    elif champ is None:
        deployed, why = True, "no comparable deployed policy"
    elif challenger > champ:
        deployed, why = True, f"test Sharpe {challenger:+.2f} beats the deployed policy's {champ:+.2f}"
    else:
        deployed, why = False, f"deployed policy is better on the same test days ({champ:+.2f} >= {challenger:+.2f})"
    report["deployed"], report["deploy_reason"] = deployed, why
    export(res["model"], res["norm"], report, cand_path)
    if deployed:
        export(res["model"], res["norm"], report, POLICY_PATH)
    md = render(report)
    with open(os.path.join(MODEL_DIR, REPORT_PATH_NAME if not a.synthetic else "report_synthetic.json"), "w") as f:
        json.dump(report, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    with open(os.path.join(MODEL_DIR, "report.md" if not a.synthetic else "report_synthetic.md"), "w") as f:
        f.write(md)
    print("\n" + md)
    print(f"Candidate saved to {cand_path}" + (f"; deployed to {POLICY_PATH}" if deployed else ""))
    return report


if __name__ == "__main__":
    main()
