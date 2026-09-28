"""
Backtest the platform's strategies and exit rules on historical minute bars.

    cd api && python -m backtest                      # watchlist, 5 days
    python -m backtest --days 10 --symbols NVDA,AAPL,BTC/USD
    python -m backtest --strategies momentum_breakout,supertrend --variants before,current

Writes backtest/output/report.md and trades.csv. See backtest/sim.py for how the
live rules are replayed and what a minute bar cannot show.
"""
import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from typing import Dict, List

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def _job(args):
    symbol, bars, strat_names, variants, notional, costs = args
    from engine.strategies import registry
    from backtest import sim
    tape = sim.Tape(symbol, bars, costs)
    klass = registry.asset_class(symbol)
    out = []
    for name in strat_names:
        strat = registry.get(name)
        if strat is None or not registry.is_compatible(name, klass):
            continue
        for v in variants:
            with sim.overrides(sim.VARIANTS[v]):
                for t in sim.run(tape, strat, notional):
                    out.append((v, t))
    return symbol, out


def stats(trades) -> Dict[str, float]:
    pnl = [t.pnl for t in trades]
    wins = [p for p in pnl if p > 0]
    losses = [p for p in pnl if p <= 0]
    eq, peak, dd = 0.0, 0.0, 0.0
    for t in sorted(trades, key=lambda t: t.exit_minute):
        eq += t.pnl
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    return {
        "trades": len(pnl),
        "net": sum(pnl),
        "win_rate": len(wins) / len(pnl) * 100 if pnl else 0.0,
        "avg": sum(pnl) / len(pnl) if pnl else 0.0,
        "pf": (sum(wins) / -sum(losses)) if losses and sum(losses) < 0 else float("inf") if wins else 0.0,
        "max_dd": dd,
        "harvested": sum(t.harvested for t in trades),
        "costs": sum(t.costs for t in trades),
        "stops": sum(1 for t in trades if t.exit_reason.startswith("stop")),
        "rescued": sum(1 for t in trades if t.rescued),
    }


def _row(label, s) -> str:
    pf = "inf" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
    return (f"| {label} | {s['trades']} | {s['win_rate']:.0f}% | ${s['net']:+,.2f} | ${s['avg']:+,.3f} "
            f"| {pf} | ${s['max_dd']:,.2f} | ${s['harvested']:+,.2f} | ${s['costs']:,.2f} | {s['stops']} | {s['rescued']} |")


HEAD = ("| | trades | win | net P&L | per trade | profit factor | max drawdown | harvested | costs paid | stop-outs | rescued |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m backtest")
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--symbols", default="", help="comma list; default: the live default watchlist")
    ap.add_argument("--strategies", default="", help="comma list; default: every compatible strategy")
    ap.add_argument("--variants", default=",".join(["no_harvest", "before", "harvest_any", "recovery", "current"]))
    ap.add_argument("--notional", type=float, default=1000.0, help="dollars per entry")
    ap.add_argument("--stock-half-spread-bps", type=float, default=2.0)
    ap.add_argument("--crypto-half-spread-bps", type=float, default=5.0)
    ap.add_argument("--crypto-fee-bps", type=float, default=25.0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)

    from core.state import state
    from engine.strategies import registry
    from backtest import data, sim

    symbols = [s.strip().upper() for s in a.symbols.split(",") if s.strip()] or sorted(state.watchlist)
    strat_names = ([s.strip() for s in a.strategies.split(",") if s.strip()]
                   or [s.name for s in registry.available()])
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    bad = [v for v in variants if v not in sim.VARIANTS]
    if bad:
        sys.exit(f"unknown variant(s) {bad}; choose from {list(sim.VARIANTS)}")
    costs = sim.Costs(a.stock_half_spread_bps, a.crypto_half_spread_bps, a.crypto_fee_bps)

    print(f"Loading {a.days} days of 1-minute bars for {len(symbols)} symbols...")
    bars = data.load_many(symbols, a.days)
    print(f"Replaying {len(strat_names)} strategies x {len(variants)} exit variants "
          f"on {len(bars)} symbols ({sum(len(b) for b in bars.values()):,} bars)...")
    t0 = time.time()
    results = defaultdict(list)        # variant -> [Trade]
    jobs = [(s, b, strat_names, variants, a.notional, costs) for s, b in bars.items()]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for sym, out in pool.map(_job, jobs):
            for v, t in out:
                results[v].append(t)
            print(f"  {sym}: {len(out)} trades")
    print(f"Done in {time.time() - t0:.0f}s")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "trades.csv"), "w", newline="") as f:
        w = None
        for v, ts in results.items():
            for t in ts:
                row = {"variant": v, **asdict(t)}
                if w is None:
                    w = csv.DictWriter(f, fieldnames=list(row))
                    w.writeheader()
                w.writerow(row)

    lines = [f"# Backtest: {a.days} days, {len(bars)} symbols, ${a.notional:,.0f} per entry", "",
             f"Costs: stock half-spread {a.stock_half_spread_bps} bps; crypto half-spread "
             f"{a.crypto_half_spread_bps} bps + fee {a.crypto_fee_bps} bps per side. "
             "No news history: sentiment is neutral.", "",
             "## Exit rules compared (all strategies pooled)", "", HEAD]
    for v in variants:
        lines.append(_row(v, stats(results[v])))
    for klass in ("equity", "crypto"):
        lines += ["", f"## {klass}: by exit rules", "", HEAD]
        for v in variants:
            lines.append(_row(v, stats([t for t in results[v] if registry.asset_class(t.symbol) == klass])))
    for v in variants:
        lines += ["", f"## Strategies under `{v}`, best first", "", HEAD]
        by = defaultdict(list)
        for t in results[v]:
            by[(t.strategy, registry.asset_class(t.symbol))].append(t)
        for (name, klass), ts in sorted(by.items(), key=lambda kv: -sum(t.pnl for t in kv[1])):
            lines.append(_row(f"{name} ({klass})", stats(ts)))
    live = {"crypto": state.strategy_class_defaults["crypto"], "equity": state.strategy_class_defaults["equity"]}
    lines += ["", f"## Live defaults only ({live['equity']} for stocks, {live['crypto']} for crypto)", "", HEAD]
    for v in variants:
        lines.append(_row(v, stats([t for t in results[v]
                                    if t.strategy == live[registry.asset_class(t.symbol)]])))
    report = "\n".join(lines) + "\n"
    with open(os.path.join(OUT_DIR, "report.md"), "w") as f:
        f.write(report)
    print("\n" + report)
    print(f"Wrote {OUT_DIR}/report.md and trades.csv")


if __name__ == "__main__":
    main()
