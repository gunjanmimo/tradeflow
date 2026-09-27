"""
Per-strategy track record from the in-process closed-trade ledger.

Used to weight strategies in the adaptive selector and the quant council. The
statistic is average R-multiple (pnl / initial risk), so results compare across
position sizes and asset classes. Below MEMORY_MIN_SAMPLES closed trades a
strategy gets a neutral weight: absence of data must not read as a verdict.
"""
import time
from typing import Dict

from core.config import settings
from core.state import state

_CACHE_TTL = 30.0
_cache: Dict[str, Dict] = {}
_cache_at = 0.0


def _strategy_of(rec: Dict) -> str:
    return rec.get("entry_strategy") or rec.get("strategy") or "unknown"


def stats() -> Dict[str, Dict]:
    global _cache, _cache_at
    now = time.time()
    if now - _cache_at < _CACHE_TTL:
        return _cache
    agg: Dict[str, Dict] = {}
    for rec in list(state.closed_trades):
        r = rec.get("r_multiple")
        if r is None:
            continue
        a = agg.setdefault(_strategy_of(rec), {"n": 0, "wins": 0, "sum_r": 0.0})
        a["n"] += 1
        a["wins"] += 1 if r > 0 else 0
        a["sum_r"] += float(r)
    out = {}
    for name, a in agg.items():
        out[name] = {
            "n": a["n"],
            "win_rate": round(a["wins"] / a["n"], 3),
            "avg_r": round(a["sum_r"] / a["n"], 3),
            "weight": weight_from(a["n"], a["sum_r"] / a["n"]),
        }
    _cache, _cache_at = out, now
    return out


def weight_from(n: int, avg_r: float) -> float:
    if n < settings.MEMORY_MIN_SAMPLES:
        return 1.0
    return round(min(max(1.0 + 0.5 * avg_r, 0.5), 1.5), 3)


def weight(name: str) -> float:
    return stats().get(name, {}).get("weight", 1.0)


def install(precomputed: Dict[str, Dict]):
    """
    Pins the stats to a precomputed table. Used by the analysis worker process,
    whose own closed-trade ledger is empty: the main process sends its stats().
    """
    global _cache, _cache_at
    _cache, _cache_at = dict(precomputed), float("inf")
