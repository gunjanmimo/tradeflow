"""
Do earlier or later profit-taking rules make the same entries pay more?

    cd api && python -m backtest.exit_sweep [--strategies a,b] [--symbols X,Y] [--workers 8]

Replays each strategy's entries through backtest/sim.py once per exit variant
(the stop, target, strategy exit and end of day never change; only the
scale-out and trailing settings of engine/profit_manager.py do), on the cached
research bars (python -m research download). The entries are the same in
every variant except where an earlier exit frees the symbol for another entry.

Reported per variant: net P&L, gross and net bps per trade (on the notional),
the t-stat of net bps per trade clustered by day, and net bps on the oldest 60%
and the newest 40% of sessions separately. A rule worth switching to has to be
better on both halves, not on the pooled number alone.

Writes backtest/output/exit_sweep.md.
"""
import argparse
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# name -> settings overrides (engine/profit_manager.py)
VARIANTS = {
    "no scale-out": {"PROFIT_TAKING_ENABLED": False},
    "live: 50% at +1R, trail 1R": {"PROFIT_TAKING_ENABLED": True, "SCALE_OUT_AT_R": 1.0,
                                   "SCALE_OUT_FRACTION": 0.5, "TRAIL_DISTANCE_R": 1.0},
    "50% at +0.75R, trail 1R": {"PROFIT_TAKING_ENABLED": True, "SCALE_OUT_AT_R": 0.75,
                                "SCALE_OUT_FRACTION": 0.5, "TRAIL_DISTANCE_R": 1.0},
    "50% at +0.5R, trail 1R": {"PROFIT_TAKING_ENABLED": True, "SCALE_OUT_AT_R": 0.5,
                               "SCALE_OUT_FRACTION": 0.5, "TRAIL_DISTANCE_R": 1.0},
    "breakeven at +0.5R, sell none, trail 1R": {"PROFIT_TAKING_ENABLED": True, "SCALE_OUT_AT_R": 0.5,
                                                "SCALE_OUT_FRACTION": 0.0, "TRAIL_DISTANCE_R": 1.0},
}
DEFAULT_STRATEGIES = ("zscore_reversion", "connors_rsi2", "supertrend", "donchian_turtle")


def _job(args):
    symbol, strat_names, notional = args
    from core.config import settings
    from engine.strategies import registry
    from research import data
    from backtest import sim
    b = data.load(symbol)
    arr = np.column_stack([b.minute, b.o, b.h, b.l, b.c, b.v])
    out = []
    for name in strat_names:
        strat = registry.get(name)
        if strat is None:
            continue
        tape = sim.Tape(symbol, arr)            # indicator contexts are cached across variants
        for variant, overrides in VARIANTS.items():
            saved = {k: getattr(settings, k) for k in overrides}
            try:
                for k, v in overrides.items():
                    setattr(settings, k, v)
                for t in sim.run(tape, strat, notional):
                    out.append((variant, name, t.entry_minute, t.pnl, t.costs, t.exit_reason))
            finally:
                for k, v in saved.items():
                    setattr(settings, k, v)
        del tape
    return symbol, out


def _clustered_t(x: np.ndarray, day: np.ndarray) -> float:
    from research.edge import _clustered_t as ct
    return ct(x, day)


def summarize(rows, notional: float, holdout_from: int):
    from research.data import ny_day
    pnl = np.array([r[3] for r in rows])
    costs = np.array([r[4] for r in rows])
    day = ny_day(np.array([r[2] for r in rows], dtype=np.int64))
    net_bps = pnl / notional * 1e4
    ho = day >= holdout_from
    return {
        "trades": len(rows), "net": float(pnl.sum()),
        "gross_bps": float(((pnl + costs) / notional * 1e4).mean()),
        "net_bps": float(net_bps.mean()), "t": _clustered_t(net_bps, day),
        "win": float((pnl > 0).mean() * 100),
        "first_net_bps": float(net_bps[~ho].mean()) if (~ho).any() else float("nan"),
        "holdout_net_bps": float(net_bps[ho].mean()) if ho.any() else float("nan"),
        "stops": sum(1 for r in rows if r[5].startswith("stop")),
    }


def render(res, strat_names, n_syms, n_days, costs_rt, notional):
    head = ("| variant | trades | win | net P&L | gross bps | net bps | t | first 60% net bps "
            "| holdout 40% net bps | stops |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    def line(label, s):
        return (f"| {label} | {s['trades']} | {s['win']:.0f}% | ${s['net']:+,.0f} | {s['gross_bps']:+.2f} "
                f"| {s['net_bps']:+.2f} | {s['t']:+.2f} | {s['first_net_bps']:+.2f} "
                f"| {s['holdout_net_bps']:+.2f} | {s['stops']} |")
    lines = [f"# Exit sweep: {len(strat_names)} strategies x {n_syms} symbols, {n_days} sessions", "",
             f"${notional:,.0f} per entry, costs {costs_rt:g} bps per round trip (core/costs.py). "
             "bps are per trade on the notional; t is clustered by day.", "",
             "## All strategies pooled", "", head]
    lines += [line(v, res[("*", v)]) for v in VARIANTS if ("*", v) in res]
    for name in strat_names:
        if not any((name, v) in res for v in VARIANTS):
            continue
        lines += ["", f"## {name}", "", head]
        lines += [line(v, res[(name, v)]) for v in VARIANTS if (name, v) in res]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m backtest.exit_sweep")
    ap.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES))
    ap.add_argument("--symbols", default="")
    ap.add_argument("--notional", type=float, default=1000.0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)

    from core.costs import DEFAULT_COSTS
    from research import data, papers
    syms = [s.strip().upper() for s in a.symbols.split(",") if s.strip()] or \
        [s for s in data.available() if s not in ("SPY", "QQQ")]
    names = [s.strip() for s in a.strategies.split(",") if s.strip()]
    bars = data.load_many(syms)
    holdout_from = papers.holdout_start(bars)
    n_days = len({d for b in bars.values() for d in b.days().tolist()})
    del bars
    print(f"{len(names)} strategies x {len(syms)} symbols x {len(VARIANTS)} exit variants, {n_days} sessions")
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for sym, out in pool.map(_job, [(s, names, a.notional) for s in syms]):
            rows.extend(out)
            print(f"  {sym}: {len(out)} trades ({time.time() - t0:.0f}s)", flush=True)
    by = defaultdict(list)
    for r in rows:
        by[(r[1], r[0])].append(r)
        by[("*", r[0])].append(r)
    res = {k: summarize(v, a.notional, holdout_from) for k, v in by.items()}
    report = render(res, names, len(syms), n_days, DEFAULT_COSTS.round_trip_bps, a.notional)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "exit_sweep.md"), "w") as f:
        f.write(report)
    print("\n" + report)
    print(f"({time.time() - t0:.0f}s) wrote {OUT_DIR}/exit_sweep.md")


if __name__ == "__main__":
    main()
