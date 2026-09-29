"""
    cd api
    python -m scout rank [--top 25] [--tone]     rank now from the live sources and print it
    python -m scout scorecard [--top 10]         judge the rankings the live scout logged
    python -m scout backtest [--days 750] [--top 10] [--symbols AAPL,MSFT,...]
                                                 the performance component on daily history

`rank` does not touch the watchlist. Without --tone headline tone is not scored
(no Jev/Laya load), so the news component is headline count only.
"""
import argparse
import asyncio
import json
import sys


def _print_ranking(rows, excluded, pool, top):
    print(f"{len(rows)} eligible of {pool} in the pool\n")
    print(f"{'#':>3} {'symbol':<7} {'country':<12} {'score':>5}  {'perf':>4} {'today':>5} {'home':>4} "
          f"{'news':>4} {'disc':>4}   why")
    for r in rows[:top]:
        c = r["components"]
        f = lambda k: f"{c[k]:.2f}" if k in c else "  - "
        print(f"{r['rank']:>3} {r['symbol']:<7} {r.get('country', 'US')[:12]:<12} {r['score']:.3f}  "
              f"{f('performance')} {f('today'):>5} {f('home'):>4} {f('news')} {f('discussion')}   "
              f"{'; '.join(r['reasons'][:3])}")
    by = {}
    for e in excluded:
        key = e["why"].split(" (")[0].split(":")[0]
        by[key] = by.get(key, 0) + 1
    print("\nexcluded: " + ", ".join(f"{k} {v}" for k, v in sorted(by.items(), key=lambda kv: -kv[1])))


async def _rank(args):
    from scout.service import scout
    from scout.sources import sources
    if args.tone:
        from sentiment.router import sentiment_service
        await sentiment_service.initialize()
    rows = await scout.rank_now(score_tone=args.tone, log=False)
    for name, h in sources.health.items():
        print(f"  {name:<22} {'ok ' if h['ok'] else 'DOWN'} {h['count']:>4}  {h['detail']}")
    _print_ranking(rows, scout.excluded, scout.pool_size, args.top)
    from scout.exchanges import exchanges
    print("\nEXCHANGES: most traded today -> US line (score, rank) or why not tradable here")
    for b in exchanges.snapshot(scout.picks, rows, per_board=8, excluded=scout.excluded):
        print(f"  {b['label']:<10} {'' if b['ok'] else 'DOWN ' + str(b['detail'])}"
              f"{b['tradable']} of the top {len(exchanges.boards.get(b['market'], {}).get('rows', []))} tradable")
        for r in b["rows"]:
            dest = (f"-> {r['us_symbol']} ({r['score']:.2f}, #{r['rank']})" if r["us_symbol"] and r["eligible"]
                    else f"-> {r['us_symbol']} ({r['why_not']})" if r["us_symbol"] else f"x {r['why_not']}")
            print(f"      {str(r['symbol']):<9} {(r['name'] or '')[:30]:<30} {r['change_pct'] or 0:+6.2f}%  {dest}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m scout")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rank", help="rank now and print")
    r.add_argument("--top", type=int, default=25)
    r.add_argument("--tone", action="store_true", help="score headline tone (loads Jev/Laya)")
    s = sub.add_parser("scorecard", help="judge the logged live rankings")
    s.add_argument("--top", type=int, default=10)
    b = sub.add_parser("backtest", help="the performance component on daily history")
    b.add_argument("--days", type=int, default=750)
    b.add_argument("--top", type=int, default=10)
    b.add_argument("--symbols", default="")
    args = ap.parse_args(argv)

    if args.cmd == "rank":
        asyncio.run(_rank(args))
    elif args.cmd == "scorecard":
        from scout import evaluate
        from scout.service import _PATH
        recs = evaluate.load_rankings(_PATH)
        if not recs:
            print(f"No rankings logged yet at {_PATH}.")
            return 0
        syms = sorted({s for r in recs for s in (r.get("pool_prices") or {})})
        from datetime import date
        first = min(date.fromisoformat(r["ny_date"]) for r in recs)
        bars = evaluate.fetch_daily_ohlcv(syms, days=(date.today() - first).days + 10, min_bars=1)
        closes = {s: dict(zip(b["day"].tolist(), b["close"].tolist())) for s, b in bars.items()}
        print(json.dumps(evaluate.scorecard(recs, closes, args.top), indent=2))
    elif args.cmd == "backtest":
        from scout import evaluate
        from scout.service import _PATH
        # Default universe: every stock the live scout has ranked eligible (its log),
        # never a hand-picked list.
        syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or sorted(
            {s for r in evaluate.load_rankings(_PATH) for s in (r.get("pool_prices") or {})})
        if not syms:
            print("No discovered stocks yet: run the live scout first, or pass --symbols.")
            return 1
        bars = evaluate.fetch_daily_ohlcv(syms, args.days)
        print(json.dumps(evaluate.backtest(bars, args.top), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
