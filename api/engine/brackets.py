"""
Single source of truth for stop-loss / take-profit brackets.

Before this module, four different places invented their own fallback brackets:

    allocation_agent.py   ATR-based (the intended rule)
    decision_engine.py    entry * 0.985 / 1.03
    sentinel_agent.py     entry * 0.985 / 1.03
    executor.py (sync)    entry * 0.98  / 1.04

Whichever ran last won, so a position's actual risk depended on execution order.
After a restart the executor's sync path always won, silently resetting every
position to a flat -2%/+4% regardless of volatility or what was originally planned.

All bracket derivation now goes through here, and every result is rounded at the
asset's own price precision so sub-cent assets are not flattened to 0.00.
"""
import logging
from typing import Dict, Optional, Tuple

from core.config import settings
from core.state import round_price, price_decimals

logger = logging.getLogger("tradeflow.brackets")


def stop_distance_for(price: float, atr: Optional[float]) -> float:
    """
    Volatility-scaled stop distance in absolute price units, clamped to the
    configured relative bounds. Bounds are fractions of price, never dollars.
    """
    price = float(price)
    if price <= 0:
        return 0.0
    atr_val = float(atr) if atr and atr > 0 else price * 0.005
    # Stop width follows the live risk dial: defensive profiles take a wider stop
    # (and correspondingly smaller size) so noise does not shake them out.
    from core.state import state
    distance = atr_val * state.risk_profile.stop_atr_multiple
    distance = max(distance, price * settings.MIN_STOP_DISTANCE_PCT)
    distance = min(distance, price * settings.MAX_STOP_DISTANCE_PCT)
    return distance


def derive(price: float, atr: Optional[float] = None) -> Tuple[float, float, float]:
    """
    Returns (stop_loss, take_profit, stop_distance) for a long entry at `price`.
    Reward:risk is set by TAKE_PROFIT_ATR_MULTIPLE / STOP_LOSS_ATR_MULTIPLE.
    """
    price = float(price)
    distance = stop_distance_for(price, atr)
    from core.state import state
    rr = state.risk_profile.reward_risk_ratio
    stop_loss = round_price(price - distance, price)
    take_profit = round_price(price + distance * rr, price)
    return stop_loss, take_profit, distance


def is_valid(price: float, stop_loss: Optional[float], take_profit: Optional[float]) -> bool:
    """
    A bracket is only usable if it actually brackets the price. This rejects the
    degenerate cases the old code produced: a negative stop (unreachable, so the
    position had no downside protection at all) and a take-profit at or below the
    entry (immediately 'hit', closing the position on its first tick).
    """
    try:
        price = float(price)
        if stop_loss is None or take_profit is None:
            return False
        sl, tp = float(stop_loss), float(take_profit)
    except (TypeError, ValueError):
        return False
    if sl <= 0 or tp <= 0:
        return False
    return sl < price < tp


def ensure(position: Dict, price: Optional[float] = None, atr: Optional[float] = None) -> Dict:
    """
    Guarantees a position dict carries a valid bracket, repairing it if not.

    Called from every path that can observe a position (entry, broker sync,
    sentinel tick), so a position can never end up with the degenerate brackets
    that previously left crypto unprotected.
    """
    entry = float(position.get("avg_entry_price") or price or 0.0)
    if entry <= 0:
        return position

    ref = float(price or position.get("current_price") or entry)
    sl, tp = position.get("stop_loss"), position.get("take_profit")

    if not is_valid(entry, sl, tp):
        new_sl, new_tp, distance = derive(entry, atr)
        if sl is not None or tp is not None:
            logger.warning(
                "Repaired invalid bracket for %s: SL %s -> %s, TP %s -> %s (entry %s)",
                position.get("symbol"), sl, new_sl, tp, new_tp, entry,
            )
        position["stop_loss"] = new_sl
        position["take_profit"] = new_tp
        position["stop_pct"] = round(distance / entry * 100, 3) if entry else None
        qty = float(position.get("qty") or 0.0)
        position["dollar_risk"] = round(abs(entry - new_sl) * qty, 2)
        position["dollar_reward"] = round(abs(new_tp - entry) * qty, 2)
        position["bracket_source"] = "repaired"

    return position


def trail(position: Dict, highest_price: float, entry_price: float,
          activate_at_pct: float = 0.008, give_back_pct: float = 0.007) -> Optional[float]:
    """
    Trailing stop, returned only when it would raise the existing stop.

    Rounded at the asset's own precision -- the previous `round(high * 0.993, 2)`
    evaluated to 0.00 for any sub-cent asset, so the trail silently never engaged.
    """
    if entry_price <= 0 or highest_price <= 0:
        return None
    profit_pct = (highest_price - entry_price) / entry_price
    if profit_pct < activate_at_pct:
        return None
    candidate = round_price(highest_price * (1.0 - give_back_pct), highest_price)
    current = position.get("stop_loss")
    try:
        current_val = float(current) if current is not None else None
    except (TypeError, ValueError):
        current_val = None
    if current_val is not None and candidate <= current_val:
        return None
    # Never trail a stop above the price it is protecting
    if candidate >= highest_price:
        return None
    return candidate
