"""
Exits that are not a trading opinion: the position has to go regardless of what
its strategy thinks. Both are checked by the position's sentinel on every tick
and on its one-second heartbeat, and both bypass the equity minimum hold.

  stale price   The price has not changed for STALE_PRICE_EXIT_SECONDS. A position
                nobody is quoting has no live stop and no live target, and ties
                up budget the manager could use. It is closed at once, unless it
                is already down STALE_PRICE_MAX_LOSS_PCT or more: dumping that
                would lock in a large loss on a stale number, so the stop and the
                other exits keep it.

  end of day    This is a day-trading platform. A stock is closed
                FLATTEN_MINUTES_BEFORE_CLOSE before the market it trades on
                closes, and, if that window was missed, as soon as the after-hours
                session lets an exit through. Crypto has no close.

Each returns a human-readable reason when the position must close, else None.
"""
import time
from typing import Any, Dict, Optional

from core.config import settings
from core.state import state, is_crypto_symbol


def stale_price_exit(symbol: str, pnl_pct: float, since: float,
                     now: Optional[float] = None) -> Optional[str]:
    """
    `pnl_pct` is a fraction (-0.03 = down 3%). `since` is when this position
    started being watched: a symbol that was already quiet before the entry has
    not been stale *while held*, so the clock never starts before it.
    """
    now = time.time() if now is None else now
    moved = max(state.price_moved_at.get(symbol, since), since)
    age = now - moved
    if age < settings.STALE_PRICE_EXIT_SECONDS:
        return None
    loss_pct = -pnl_pct * 100.0
    if loss_pct >= settings.STALE_PRICE_MAX_LOSS_PCT:
        return None
    result = f"P&L {pnl_pct * 100:+.2f}%" if pnl_pct < 0 else f"P&L {pnl_pct * 100:+.2f}% (not at a loss)"
    return (f"No new price for {age:.0f}s (limit {settings.STALE_PRICE_EXIT_SECONDS:.0f}s) and "
            f"{result} is inside the {settings.STALE_PRICE_MAX_LOSS_PCT:.0f}% loss limit: "
            f"closing rather than holding an unpriced position")


def end_of_day_exit(symbol: str) -> Optional[str]:
    if not settings.DAY_TRADE_FLATTEN_ENABLED or is_crypto_symbol(symbol):
        return None
    from core.market_hours import us_session, minutes_to_close, POST
    if us_session() == POST:
        return "US market has closed for the day: closing the day trade in after-hours"
    mins = minutes_to_close(symbol)
    if mins is not None and mins <= settings.FLATTEN_MINUTES_BEFORE_CLOSE:
        return (f"US market closes in {max(mins, 0.0):.1f} min "
                f"(flatten window {settings.FLATTEN_MINUTES_BEFORE_CLOSE:.0f} min): "
                f"closing the day trade")
    return None


def check(symbol: str, pos: Dict[str, Any], pnl_pct: float, watching_since: float,
          now: Optional[float] = None) -> Optional[str]:
    """The first forced-exit reason that applies to this position, or None."""
    since = max(float(watching_since or 0.0), float(pos.get("opened_at") or 0.0))
    return end_of_day_exit(symbol) or stale_price_exit(symbol, pnl_pct, since, now)


def telemetry(symbol: str) -> Dict[str, Any]:
    """What the dashboard shows beside a position: how long it has left."""
    from core.market_hours import minutes_to_close
    mins = minutes_to_close(symbol)
    return {
        "minutes_to_close": None if mins is None else round(mins, 1),
        "flatten_in_min": None if mins is None
        else round(max(0.0, mins - settings.FLATTEN_MINUTES_BEFORE_CLOSE), 1),
    }
