"""
Per-stock features the ranker scores. Pure functions of arrays and dicts, so the
live service, the backtest and the tests compute them identically.
"""
from typing import Any, Dict, Optional

import numpy as np

MIN_HISTORY = 60          # sessions of daily bars a stock needs to be judged


def daily_features(closes: np.ndarray, highs: Optional[np.ndarray] = None,
                   lows: Optional[np.ndarray] = None,
                   volumes: Optional[np.ndarray] = None) -> Optional[Dict[str, float]]:
    """
    Past-window performance as of the last completed session.

      ret_5d, ret_20d, ret_60d   plain returns over the window
      mom_adj_20d                20-day log return over its own 20-day noise: a
                                 steady climb scores above a lurching one
      above_sma20, sma20_gt_50   trend structure (1.0 / 0.0)
      near_high                  close / highest close of 60 sessions (1.0 = at the high)
      rvol_5d                    mean volume of 5 sessions / of the 60 before
      atr_pct                    14-day average true range, % of price
    """
    c = np.asarray(closes, dtype=float)
    if len(c) < MIN_HISTORY or c[-1] <= 0:
        return None
    r = np.diff(np.log(c[-21:]))
    noise = r.std(ddof=1) * np.sqrt(len(r)) if len(r) > 2 else 0.0
    sma20, sma50 = c[-20:].mean(), c[-50:].mean()
    out = {
        "ret_5d": float(c[-1] / c[-6] - 1.0),
        "ret_20d": float(c[-1] / c[-21] - 1.0),
        "ret_60d": float(c[-1] / c[-61] - 1.0) if len(c) > 60 else float(c[-1] / c[0] - 1.0),
        "mom_adj_20d": float(np.log(c[-1] / c[-21]) / noise) if noise > 0 else 0.0,
        "above_sma20": float(c[-1] > sma20),
        "sma20_gt_50": float(sma20 > sma50),
        "near_high": float(c[-1] / c[-60:].max()),
        "price": float(c[-1]),
    }
    if volumes is not None and len(volumes) >= MIN_HISTORY:
        v = np.asarray(volumes, dtype=float)
        base = v[-65:-5].mean()
        out["rvol_5d"] = float(v[-5:].mean() / base) if base > 0 else 1.0
    if highs is not None and lows is not None and len(highs) >= 15:
        h, lo = np.asarray(highs, dtype=float), np.asarray(lows, dtype=float)
        prev = c[-15:-1]
        tr = np.maximum(h[-14:], prev) - np.minimum(lo[-14:], prev)
        out["atr_pct"] = float(tr.mean() / c[-1] * 100.0)
    else:
        out["atr_pct"] = float(np.abs(np.diff(np.log(c[-15:]))).mean() * 125.0)
    return out


def today_features(snap: Dict[str, Any], today: str,
                   session_frac: Optional[float]) -> Optional[Dict[str, float]]:
    """
    Today's move from an Alpaca snapshot (flattened by scout/sources.py).

    Before the open the daily bar is the previous session's: the change is the
    pre-market price against that close and there is no volume to compare yet.
    With no trade printed today at all (the IEX feed is thin before the open)
    there is nothing to say about today, and the component does not vote.
    Volumes are compared within the snapshot's own feed (the free plan's is IEX
    only), never against consolidated daily volume.
    """
    price = snap.get("price")
    if not price:
        return None
    if snap.get("day_date") == today:
        prev_close = snap.get("prev_close")
        out = {"chg_pct": None, "rvol_today": None, "live_session": 1.0}
        if session_frac and snap.get("prev_volume") and snap.get("day_volume") is not None:
            pace = snap["day_volume"] / max(session_frac, 0.05)
            out["rvol_today"] = float(pace / snap["prev_volume"])
    elif snap.get("price_date") == today:
        # Pre-market: a print today against the last close.
        prev_close = snap.get("day_close")
        out = {"chg_pct": None, "rvol_today": None, "live_session": 0.0}
    else:
        return None                 # no trade yet today: nothing to say about today
    if not prev_close:
        return None
    out["chg_pct"] = float((price / prev_close - 1.0) * 100.0)
    return out
