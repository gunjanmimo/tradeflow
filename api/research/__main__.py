"""
    cd api
    python -m research download [--days 730] [--symbols AAPL,MSFT]
    python -m research edge [--signals reversal_z,vwap_dev] [--horizons 5,15,30,60]
                            [--half-spread-bps 2] [--slippage-bps 1]
    python -m research papers [--half-spread-bps 2] [--slippage-bps 1]

`edge` writes research/output/edge.md and edge.json; `papers` (published
intraday strategies replayed trade by trade, research/papers.py) writes
research/output/papers.md.
"""
import argparse
import json
import os
import sys
import time

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m research")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="cache one-minute bars")
    d.add_argument("--days", type=int, default=730)
    d.add_argument("--symbols", default="")
    d.add_argument("--workers", type=int, default=4)
    e = sub.add_parser("edge", help="measure each signal's edge after costs")
    e.add_argument("--signals", default="")
    e.add_argument("--horizons", default="5,15,30,60")
    e.add_argument("--symbols", default="")
    e.add_argument("--half-spread-bps", type=float, default=None)
    e.add_argument("--slippage-bps", type=float, default=None)
    pp = sub.add_parser("papers", help="replay published intraday strategies after costs")
    pp.add_argument("--half-spread-bps", type=float, default=None)
    pp.add_argument("--slippage-bps", type=float, default=None)
    a = ap.parse_args(argv)

    from research import data
    syms = [s.strip().upper() for s in a.symbols.split(",") if s.strip()] if getattr(a, "symbols", "") else None

    if a.cmd == "download":
        t0 = time.time()
        got = data.update(syms or data.DEFAULT_UNIVERSE, a.days, workers=a.workers)
        print(f"{sum(got.values()):,} bars for {len(got)} symbols in {time.time() - t0:.0f}s "
              f"-> {data.BAR_DIR}")
        return

    from core import costs as costs_mod
    from research import edge, signals
    costs = costs_mod.parse(a.half_spread_bps, a.slippage_bps)
    if a.cmd == "papers":
        from research import papers
        bars = data.load_many(data.available())
        if not bars:
            sys.exit("No cached bars. Run: python -m research download")
        n_days = len({d for b in bars.values() for d in b.days().tolist()})
        t0 = time.time()
        report = papers.render(papers.study(bars, costs), costs,
                               len([s for s in bars if s not in ("SPY", "QQQ")]), n_days)
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, "papers.md"), "w") as f:
            f.write(report)
        print(report)
        print(f"({time.time() - t0:.0f}s) wrote {OUT_DIR}/papers.md")
        return
    names = [s.strip() for s in a.signals.split(",") if s.strip()] or list(signals.SIGNALS)
    bad = [n for n in names if n not in signals.SIGNALS]
    if bad:
        sys.exit(f"unknown signal(s) {bad}; choose from {list(signals.SIGNALS)}")
    horizons = [int(h) for h in a.horizons.split(",")]
    bars = data.load_many(syms or data.available())
    if not bars:
        sys.exit("No cached bars. Run: python -m research download")
    if syms and data.MARKET not in bars:
        m = data.load(data.MARKET)
        if m is not None:
            bars[data.MARKET] = m
    n_days = len({d for b in bars.values() for d in b.days().tolist()})
    print(f"{len(bars)} symbols, {n_days} sessions, {sum(len(b) for b in bars.values()):,} bars")
    t0 = time.time()
    rows = edge.study(bars, names, horizons, costs)
    report = edge.render(rows, costs, len([s for s in bars if s not in ("SPY", "QQQ")]), n_days)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "edge.md"), "w") as f:
        f.write(report)
    with open(os.path.join(OUT_DIR, "edge.json"), "w") as f:
        json.dump({"costs": costs.to_dict(), "rows": [r.to_dict() for r in rows]}, f, indent=1)
    print(report)
    print(f"({time.time() - t0:.0f}s) wrote {OUT_DIR}/edge.md")


if __name__ == "__main__":
    main()
