"""
US equity session: which part of the trading day it is right now.

  pre      04:00-09:30 NY  extended hours: limit orders only, thin liquidity
  regular  09:30-16:00 NY
  post     16:00-20:00 NY  extended hours (not traded by the bots)
  closed   overnight, weekends, holidays

The broker clock (cached by the market scheduler, never fetched here) is trusted
when fresh: it knows holidays and early closes. Otherwise wall-clock NY time
decides, which is right on every normal weekday.
"""
import time
import zoneinfo
from datetime import datetime

_NY = zoneinfo.ZoneInfo("America/New_York")
_CLOCK_FRESH_SECONDS = 180.0

PRE, REGULAR, POST, CLOSED = "pre", "regular", "post", "closed"


def _broker_clock():
    """(is_open, next_open_iso) from the scheduler's cache, or None when stale/absent."""
    try:
        from engine.market_scheduler import market_scheduler
    except Exception:
        return None
    cache = getattr(market_scheduler, "_clock_cache", None)
    at = getattr(market_scheduler, "_clock_at", 0.0)
    if not cache or cache[1] is None or time.time() - at > _CLOCK_FRESH_SECONDS:
        return None
    return cache


def _broker_close() -> datetime | None:
    """Today's close from the broker clock, when the market is open and the cache fresh."""
    clock = _broker_clock()
    if clock is None or not clock[0]:
        return None
    try:
        from engine.market_scheduler import market_scheduler
        iso = getattr(market_scheduler, "_next_close_iso", None)
        return datetime.fromisoformat(iso).astimezone(_NY) if iso else None
    except Exception:
        return None


def minutes_to_close(symbol: str, now: datetime | None = None) -> float | None:
    """
    Minutes until the market this symbol trades on closes, or None when it has no
    close to race (crypto trades 24/7) or its regular session is not open.

    Every equity the bots can buy is listed on a US exchange -- ADRs and ETFs
    included, whatever their home market -- so they all close with the US market.
    The broker clock gives the actual close (an early-close day ends at 13:00);
    without it the normal 16:00 NY close is assumed.
    """
    from core.state import is_crypto_symbol
    if is_crypto_symbol(symbol):
        return None
    now_ny = (now or datetime.now(_NY)).astimezone(_NY)
    if us_session(now_ny) != REGULAR:
        return None
    close = _broker_close() if now is None else None
    if close is None:
        close = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)
    return (close - now_ny).total_seconds() / 60.0


def us_session(now: datetime | None = None) -> str:
    now_ny = (now or datetime.now(_NY)).astimezone(_NY)
    if now_ny.weekday() >= 5:
        return CLOSED
    minutes = now_ny.hour * 60 + now_ny.minute

    clock = _broker_clock()
    if clock is not None:
        is_open, next_open = clock
        if is_open:
            return REGULAR
        # Not open during the regular window means a holiday or early close.
        if 570 <= minutes < 960:
            return CLOSED
        # Pre-market only exists on a day the market will open.
        if minutes < 570:
            try:
                if datetime.fromisoformat(next_open).astimezone(_NY).date() != now_ny.date():
                    return CLOSED
            except Exception:
                pass

    if 570 <= minutes < 960:
        return REGULAR
    if 240 <= minutes < 570:
        return PRE
    if 960 <= minutes < 1200:
        return POST
    return CLOSED
