"""
Backtest the platform's strategies on historical one-minute bars.

    cd api && python -m backtest                      # watchlist, 5 days
    python -m backtest --days 30 --symbols NVDA,AAPL
    python -m backtest --strategies supertrend,macd_trend --half-spread-bps 3

Writes backtest/output/report.md and trades.csv. See backtest/sim.py for how the
live rules are replayed and what a minute bar cannot show.

The report splits every result into what the strategy made BEFORE costs and
what the costs took. A strategy whose gross edge per trade is not well above
the round-trip cost cannot be made profitable by tuning exits or sizing.
"""
import argparse
import csv
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from typing import Dict

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def _job(args):
    symbol, bars, strat_names, notional, costs = args
    from engine.strategies import registry
    from backtest import sim
    tape = sim.Tape(symbol, bars, costs)
    out = []
    for name in strat_names:
        strat = registry.get(name)
        if strat is None:
            continue
        out.extend(sim.run(tape, strat, notional))
    return symbol, out


def stats(trades, notional: float) -> Dict[str, float]:
    pnl = [t.pnl for t in trades]
    gross = [t.pnl + t.costs for t in trades]
    wins = [p for p in pnl if p > 0]
    losses = [p for p in pnl if p <= 0]
    eq, peak, dd = 0.0, 0.0, 0.0
    for t in sorted(trades, key=lambda t: t.exit_minute):
        eq += t.pnl
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    n = len(pnl)
    return {
        "trades": n,
        "net": sum(pnl),
        "win_rate": len(wins) / n * 100 if n else 0.0,
        "avg": sum(pnl) / n if n else 0.0,
        "gross_bps": (sum(gross) / n / notional * 1e4) if n else 0.0,
        "cost_bps": (sum(t.costs for t in trades) / n / notional * 1e4) if n else 0.0,
        "pf": (sum(wins) / -sum(losses)) if losses and sum(losses) < 0 else float("inf") if wins else 0.0,
        "max_dd": dd,
        "costs": sum(t.costs for t in trades),
        "stops": sum(1 for t in trades if t.exit_reason.startswith("stop")),
        "targets": sum(1 for t in trades if t.exit_reason.startswith("target")),
    }


def _row(label, s) -> str:
    pf = "inf" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
    return (f"| {label} | {s['trades']} | {s['win_rate']:.0f}% | ${s['net']:+,.2f} | ${s['avg']:+,.3f} "
            f"| {s['gross_bps']:+.1f} | {s['cost_bps']:.1f} | {pf} | ${s['max_dd']:,.2f} "
            f"| {s['stops']} | {s['targets']} |")


HEAD = ("| | trades | win | net P&L | per trade | gross bps/trade | cost bps/trade | profit factor "
        "| max drawdown | stops | targets |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m backtest")
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--symbols", default="", help="comma list; default: the live default watchlist")
    ap.add_argument("--strategies", default="", help="comma list; default: every strategy")
    ap.add_argument("--notional", type=float, default=1000.0, help="dollars per entry")
    ap.add_argument("--half-spread-bps", type=float, default=None)
    ap.add_argument("--slippage-bps", type=float, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)

    from core import costs as costs_mod
    from core.state import state
    from engine.strategies import registry
    from backtest import data

    symbols = [s.strip().upper() for s in a.symbols.split(",") if s.strip()] or sorted(state.watchlist)
    strat_names = ([s.strip() for s in a.strategies.split(",") if s.strip()]
                   or [s.name for s in registry.available()])
    costs = costs_mod.parse(a.half_spread_bps, a.slippage_bps)

    print(f"Loading {a.days} days of 1-minute bars for {len(symbols)} symbols...")
    bars = data.load_many(symbols, a.days)
    print(f"Replaying {len(strat_names)} strategies on {len(bars)} symbols "
          f"({sum(len(b) for b in bars.values()):,} bars)...")
    t0 = time.time()
    trades = []
    jobs = [(s, b, strat_names, a.notional, costs) for s, b in bars.items()]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for sym, out in pool.map(_job, jobs):
            trades.extend(out)
            print(f"  {sym}: {len(out)} trades")
    print(f"Done in {time.time() - t0:.0f}s")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "trades.csv"), "w", newline="") as f:
        w = None
        for t in trades:
            row = asdict(t)
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(row))
                w.writeheader()
            w.writerow(row)

    lines = [f"# Backtest: {a.days} days, {len(bars)} symbols, ${a.notional:,.0f} per entry", "",
             f"Costs: {costs.half_spread_bps:g} bps half-spread + {costs.slippage_bps:g} bps slippage "
             f"per market fill ({costs.round_trip_bps:g} bps round trip). "
             "No news history: sentiment is neutral.", "",
             "Read `gross bps/trade` against `cost bps/trade`: a strategy needs a gross edge well "
             "above its costs before anything else matters.", "",
             "## All strategies pooled", "", HEAD, _row("all", stats(trades, a.notional)),
             "", "## By strategy, best first", "", HEAD]
    by = defaultdict(list)
    for t in trades:
        by[t.strategy].append(t)
    for name, ts in sorted(by.items(), key=lambda kv: -sum(t.pnl for t in kv[1])):
        lines.append(_row(name, stats(ts, a.notional)))
    live = state.strategy_class_defaults["equity"]
    lines += ["", f"## Live default only ({live})", "", HEAD,
              _row(live, stats([t for t in trades if t.strategy == live], a.notional))]
    report = "\n".join(lines) + "\n"
    with open(os.path.join(OUT_DIR, "report.md"), "w") as f:
        f.write(report)
    print("\n" + report)
    print(f"Wrote {OUT_DIR}/report.md and trades.csv")


if __name__ == "__main__":
    main()
