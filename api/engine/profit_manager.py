"""
Profit-taking for an open position: sell into strength once a trade has earned
it, and never let a trade that was well in profit turn into a loss.

Measured in R, the trade's own initial risk per share (entry - initial stop):

  scale out    at +SCALE_OUT_AT_R: sell SCALE_OUT_FRACTION of the shares and move
               the stop to breakeven (entry + BREAKEVEN_BUFFER_PCT for costs)
  trail        after the scale-out, the stop follows the highest price reached,
               TRAIL_DISTANCE_R behind it, and only ever moves up
  target       the original take-profit still closes what is left

Why R, not "any profit": the stack removed on 2026-09-28 sold half of any winner
on the first uptick, which cut every winner short while losers ran to the stop.
+1R means the trade has moved as far in our favour as the stop is away.

Pure functions plus a small record per symbol (data/position_meta.json) so a
restart neither forgets a scale-out nor re-derives a stop that had been raised.
"""
import json
import logging
import os
import time
from typing import Any, Dict, Optional

from core.config import settings

logger = logging.getLogger("tradeflow.profit")

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_PATH = os.path.join(_DATA_DIR, "position_meta.json")


def risk_per_share(pos: Dict[str, Any]) -> float:
    entry = float(pos.get("avg_entry_price") or 0.0)
    init = float(pos.get("initial_stop") or pos.get("stop_loss") or 0.0)
    return entry - init if entry > 0 and 0 < init < entry else 0.0


def breakeven(entry: float) -> float:
    return entry * (1.0 + settings.BREAKEVEN_BUFFER_PCT / 100.0)


def plan(pos: Dict[str, Any], price: float, highest: float) -> Optional[Dict[str, Any]]:
    """What to do now, or None: {"type": "scale_out", fraction, new_stop} or {"type": "raise_stop", new_stop}."""
    if not settings.PROFIT_TAKING_ENABLED:
        return None
    r = risk_per_share(pos)
    entry = float(pos.get("avg_entry_price") or 0.0)
    stop = float(pos.get("stop_loss") or 0.0)
    if r <= 0 or price <= 0:
        return None
    gain_r = (price - entry) / r
    if not pos.get("scaled_out"):
        if gain_r >= settings.SCALE_OUT_AT_R:
            return {"type": "scale_out", "fraction": settings.SCALE_OUT_FRACTION, "gain_r": round(gain_r, 2),
                    "new_stop": round(max(stop, breakeven(entry)), 4)}
        return None
    new_stop = max(stop, breakeven(entry), max(highest, price) - settings.TRAIL_DISTANCE_R * r)
    if new_stop >= stop + settings.STOP_MIN_STEP_R * r and new_stop < price:
        return {"type": "raise_stop", "gain_r": round(gain_r, 2), "new_stop": round(new_stop, 4)}
    return None


def progress(pos: Dict[str, Any], price: float) -> float:
    """0..1: how far the trade is towards its scale-out (1.0 once scaled)."""
    if pos.get("scaled_out"):
        return 1.0
    r = risk_per_share(pos)
    entry = float(pos.get("avg_entry_price") or 0.0)
    if r <= 0 or price <= entry:
        return 0.0
    return min(1.0, (price - entry) / (settings.SCALE_OUT_AT_R * r))


class PositionMeta:
    """initial_stop, scaled_out and the (possibly raised) stop per symbol, across restarts."""

    def __init__(self):
        self.data: Dict[str, Dict[str, Any]] = {}
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        self._loaded = True
        try:
            with open(_PATH) as f:
                self.data = json.load(f)
        except (OSError, ValueError):
            self.data = {}

    def get(self, symbol: str) -> Optional[Dict[str, Any]]:
        self._load()
        return self.data.get(symbol)

    def put(self, symbol: str, **fields):
        self._load()
        self.data[symbol] = {**self.data.get(symbol, {}), **fields, "at": time.time()}
        self._save()

    def drop_except(self, symbols):
        self._load()
        gone = [s for s in self.data if s not in symbols]
        for s in gone:
            del self.data[s]
        if gone:
            self._save()

    def _save(self):
        try:
            os.makedirs(_DATA_DIR, exist_ok=True)
            tmp = _PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.data, f)
            os.replace(tmp, _PATH)
        except OSError as e:
            logger.warning("Could not save position meta: %s", e)


meta = PositionMeta()
