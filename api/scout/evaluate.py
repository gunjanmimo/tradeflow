"""
Does the scout's ranking pick stocks that go on to beat the rest?

  scorecard   Reads the rankings the live scout logged (data/scout/rankings.jsonl).
              For each one: the return from the price at ranking time to that
              session's close (the day trade) and to the next session's close,
              for the top N against the whole eligible pool. Rankings made the
              same day are averaged first, so t is over days: hours of one day
              share one market and are not independent evidence.

  backtest    The one component with years of history: performance (daily
              bars). Each day it ranks the curated universe on the features the
              scout uses and measures the next session, open to close (a day
              trade) and close to close, top N against the universe. News,
              discussion and today's move cannot be rebuilt from history; the
              scorecard judges those going forward.

Both report basis points, net of the round trip (core/costs.py), and a t-stat.
A ranking worth trading needs a positive net excess and t well above 2 -- on
the newest 40% of days too (holdout), not only overall.
"""
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from core.costs import CostModel
from scout.features import daily_features, MIN_HISTORY
from scout.ranker import performance_scores

HOLDOUT_FRACTION = 0.40


def _t(x: np.ndarray) -> float:
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 and x.std(ddof=1) > 0 else 0.0


def summarize(daily_top: List[float], daily_pool: List[float], cost: CostModel) -> Dict[str, Any]:
    """Per-day mean returns (fractions) of the picks and of the pool -> bps summary."""
    top, pool = np.asarray(daily_top), np.asarray(daily_pool)
    if not len(top):
        return {"days": 0}
    ex = top - pool
    h = max(1, int(len(ex) * HOLDOUT_FRACTION))
    return {
        "days": len(ex),
        "top_bps": round(float(top.mean()) * 1e4, 1),
        "pool_bps": round(float(pool.mean()) * 1e4, 1),
        "excess_bps": round(float(ex.mean()) * 1e4, 1),
        "t_excess": round(_t(ex), 2),
        "net_bps": round(float(top.mean()) * 1e4 - cost.round_trip_bps, 1),
        "hit_rate": round(float((ex > 0).mean()), 3),
        "holdout_excess_bps": round(float(ex[-h:].mean()) * 1e4, 1),
        "holdout_t": round(_t(ex[-h:]), 2),
    }


# ---------------------------------------------------------------------------
# Scorecard of the live rankings
# ---------------------------------------------------------------------------

def load_rankings(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def _day_number(ny_date: str) -> int:
    return int(datetime.strptime(ny_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() // 86400)


def scorecard(records: List[Dict[str, Any]], closes_by_day: Dict[str, Dict[int, float]],
              top_n: int = 10, cost: Optional[CostModel] = None) -> Dict[str, Any]:
    """
    records        logged rankings
    closes_by_day  symbol -> {day number: close}
    Only rankings whose session has closed (a close exists for their day) count.
    """
    cost = cost or CostModel()
    per_day: Dict[str, Dict[str, list]] = {}
    for rec in records:
        day = _day_number(rec["ny_date"])
        order = [r["symbol"] for r in rec.get("top", [])][:top_n]
        prices = rec.get("pool_prices") or {}
        for horizon, offset in (("to_close", 0), ("next_close", 1)):
            outs = {}
            for sym, p in prices.items():
                days = closes_by_day.get(sym) or {}
                later = sorted(d for d in days if d >= day)
                if len(later) > offset and p:
                    outs[sym] = days[later[offset]] / p - 1.0
            picks = [outs[s] for s in order if s in outs]
            if len(picks) >= max(1, top_n // 2) and len(outs) >= 2 * top_n:
                d = per_day.setdefault(rec["ny_date"], {}).setdefault(horizon, [])
                d.append((float(np.mean(picks)), float(np.mean(list(outs.values())))))
    result: Dict[str, Any] = {"rankings": len(records), "top_n": top_n,
                              "round_trip_bps": cost.round_trip_bps}
    for horizon in ("to_close", "next_close"):
        rows = [(day, np.mean([t for t, _ in v[horizon]], axis=0), np.mean([p for _, p in v[horizon]]))
                for day, v in sorted(per_day.items()) if horizon in v]
        result[horizon] = summarize([r[1] for r in rows], [r[2] for r in rows], cost)
    return result


# ---------------------------------------------------------------------------
# Backtest of the performance component
# ---------------------------------------------------------------------------

def fetch_daily_ohlcv(symbols: List[str], days: int,
                      min_bars: int = MIN_HISTORY + 5) -> Dict[str, Dict[str, np.ndarray]]:
    """Adjusted daily bars from Alpaca: symbol -> {day, open, high, low, close, volume}."""
    from alpaca.data.enums import Adjustment
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from core.config import settings
    client = StockHistoricalDataClient(settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY)
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for i in range(0, len(symbols), 100):
        resp = client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=symbols[i:i + 100], timeframe=TimeFrame.Day,
            start=end - timedelta(days=days), end=end, adjustment=Adjustment.ALL))
        for sym, bars in (resp.data or {}).items():
            if len(bars) < min_bars:
                continue
            out[sym] = {
                "day": np.array([int(b.timestamp.timestamp() // 86400) for b in bars]),
                **{k: np.array([float(getattr(b, k) or 0.0) for b in bars])
                   for k in ("open", "high", "low", "close", "volume")},
            }
    return out


def backtest(bars: Dict[str, Dict[str, np.ndarray]], top_n: int = 10,
             cost: Optional[CostModel] = None) -> Dict[str, Any]:
    cost = cost or CostModel()
    all_days = sorted({int(d) for b in bars.values() for d in b["day"]})
    idx = {s: {int(d): i for i, d in enumerate(b["day"])} for s, b in bars.items()}
    res = {"oc": ([], []), "cc": ([], []), "bottom_oc": ([], [])}
    for k in range(len(all_days) - 1):
        today, nxt = all_days[k], all_days[k + 1]
        feats, fwd_oc, fwd_cc = {}, {}, {}
        for s, b in bars.items():
            i, j = idx[s].get(today), idx[s].get(nxt)
            if i is None or j is None or i + 1 < MIN_HISTORY + 1:
                continue
            f = daily_features(b["close"][:i + 1], b["high"][:i + 1], b["low"][:i + 1], b["volume"][:i + 1])
            if f is None or b["open"][j] <= 0:
                continue
            feats[s] = f
            fwd_oc[s] = b["close"][j] / b["open"][j] - 1.0
            fwd_cc[s] = b["close"][j] / b["close"][i] - 1.0
        if len(feats) < 3 * top_n:
            continue
        score = performance_scores(feats)
        order = sorted(score, key=score.get, reverse=True)
        for key, fwd, sel in (("oc", fwd_oc, order[:top_n]), ("cc", fwd_cc, order[:top_n]),
                              ("bottom_oc", fwd_oc, order[-top_n:])):
            res[key][0].append(float(np.mean([fwd[s] for s in sel])))
            res[key][1].append(float(np.mean(list(fwd.values()))))
    return {"symbols": len(bars), "top_n": top_n, "round_trip_bps": cost.round_trip_bps,
            "day_trade_open_to_close": summarize(*res["oc"], cost),
            "hold_close_to_close": summarize(*res["cc"], cost),
            "bottom_n_open_to_close": summarize(*res["bottom_oc"], cost)}
