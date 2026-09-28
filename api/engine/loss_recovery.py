"""
Loss recovery: what a losing position does instead of being dumped at the first
touch of its stop or the first bearish headline.

The old behaviour drained the account in three ways:

  * noise stop-outs   the stop fired on a single print at or below it (and the
                      equity bracket leg at the broker did the same), so a
                      wick through the stop that recovered a second later was
                      booked as a full loss.
  * news-only exits   bearish Laya/Jev sentiment alone closed a losing trade,
                      even while the price was holding up.
  * no way back       a position that dipped and came back to its entry had no
                      protection against turning into a loss again.

What happens now, for a position whose stop is still below its entry:

  confirmed stop    Touching the stop starts a clock. The position closes when
                    the price stays at or below the stop for STOP_CONFIRM_SECONDS,
                    or at once when it falls through the disaster stop,
                    STOP_DISASTER_EXTRA_R x the stop distance further down.
                    The broker's bracket leg sits at the disaster stop, so it
                    only acts if the engine is not running. A stop at or above
                    the entry (a trailing stop locking in profit) fires at once.

  confirmed news    Bearish sentiment or the reversal read closes a LOSING
                    position only when the price trend agrees. Without that,
                    the stop keeps protecting it.

  rescue add        One small add, a "micro trade", while the position is
                    between RECOVERY_ADD_MIN_R and RECOVERY_ADD_MAX_R of its
                    stop distance under water and the price has stopped
                    falling (the micro trend is flat or up, no fresh bearish
                    news). It lowers the break-even so a partial bounce gets
                    the money back. The stop NEVER moves down, and the add is
                    sized so that the loss if the stop is then hit is at most
                    RECOVERY_MAX_RISK_MULT x the position's original risk.

  break-even lock   Once a position that went under water by RECOVERY_ADD_MIN_R
                    climbs back to its (possibly lowered) break-even plus
                    RECOVERY_BREAKEVEN_BUFFER_PCT, its stop is raised to that
                    break-even. The loss has been recovered and cannot come back;
                    the position keeps running towards its target.

Short selling is deliberately not part of this. Shorting the same symbol
against a losing long is economically identical to selling it; shorting others
needs a margin account, borrow, and carries unbounded loss, and the executor
treats any short as an oversell to be bought back.
"""
import time
from typing import Any, Dict, Optional, Tuple

from core.config import settings
from core.state import state, TradeDecision, round_price


def _risk_per_share(pos: Dict[str, Any], entry: float, stop: float) -> float:
    """The original stop distance. Kept on the position so an add does not shrink it."""
    r = pos.get("initial_risk_ps")
    if r:
        return float(r)
    r = entry - stop
    if r > 0:
        pos["initial_risk_ps"] = r
    return max(r, 0.0)


def disaster_stop(entry: float, stop: float) -> float:
    """The hard floor under a loss stop: STOP_DISASTER_EXTRA_R of the stop distance below it."""
    dist = entry - stop
    if dist <= 0:
        return stop
    return round_price(max(stop - dist * settings.STOP_DISASTER_EXTRA_R, stop * 0.5), stop)


def broker_stop(entry: float, stop: float) -> float:
    """What the broker's bracket stop leg is set to: the disaster stop when confirming is on."""
    if not settings.STOP_CONFIRM_ENABLED:
        return stop
    return disaster_stop(entry, stop)


def check_stop(pos: Dict[str, Any], price: float, entry: float, stop: float,
               now: Optional[float] = None) -> Tuple[bool, str]:
    """
    (close, reason). Also stamps pos["stop_breached_at"] and pos["stop_state"]
    for the dashboard.
    """
    now = time.time() if now is None else now
    if price > stop:
        pos.pop("stop_breached_at", None)
        pos["stop_state"] = "ok"
        return False, ""
    # A stop at or above the entry locks in profit: take it at once.
    if not settings.STOP_CONFIRM_ENABLED or stop >= entry:
        pos["stop_state"] = "hit"
        return True, f"Stop-loss triggered: Price ${price:,.6g} <= SL ${stop:,.6g}"
    risk_ps = _risk_per_share(pos, entry, stop)
    floor = disaster_stop(stop + risk_ps, stop) if risk_ps > 0 else stop
    if price <= floor:
        pos["stop_state"] = "hit"
        return True, (f"Stop-loss triggered: Price ${price:,.6g} fell through the disaster stop "
                      f"${floor:,.6g} (SL ${stop:,.6g})")
    since = float(pos.setdefault("stop_breached_at", now))
    held = now - since
    if held >= settings.STOP_CONFIRM_SECONDS:
        pos["stop_state"] = "hit"
        return True, (f"Stop-loss triggered: Price ${price:,.6g} stayed <= SL ${stop:,.6g} "
                      f"for {held:.0f}s")
    pos["stop_state"] = f"confirming {held:.0f}/{settings.STOP_CONFIRM_SECONDS:.0f}s"
    return False, ""


def price_confirms_weakness(symbol: str) -> bool:
    """True when the minute-bar trend backs a bearish read: turning or trending down."""
    from engine.trend import board
    r = board.read(symbol)
    if not r.ready:
        return False
    return r.reversal_down or r.direction <= -settings.TREND_TRIM_DIRECTION


def soft_exit_allowed(symbol: str, pnl_pct: float) -> bool:
    """
    Whether the sentiment / reversal soft exit may close this position. On a
    winner it always may (it protects the gain). On a loser it needs the price
    to agree, or the headline alone turns a dip into a realised loss.
    """
    if pnl_pct >= 0 or not settings.RECOVERY_ENABLED:
        return True
    return price_confirms_weakness(symbol)


def breakeven_lock(pos: Dict[str, Any], price: float, entry: float, stop: float) -> Optional[float]:
    """
    The new stop once a position that went under water has come back to its
    break-even plus the buffer, else None. Only ever raises the stop.
    """
    if not settings.RECOVERY_ENABLED or entry <= 0:
        return None
    risk_ps = _risk_per_share(pos, entry, stop)
    if risk_ps > 0 and (entry - price) >= settings.RECOVERY_ADD_MIN_R * risk_ps:
        pos["recovery_mode"] = True
    if not pos.get("recovery_mode") or pos.get("recovered"):
        return None
    lock = round_price(entry * (1.0 + settings.RECOVERY_BREAKEVEN_BUFFER_PCT / 100.0), entry)
    if price < lock * (1.0 + settings.RECOVERY_BREAKEVEN_BUFFER_PCT / 100.0) or lock <= stop:
        return None
    pos["recovered"] = True
    return lock


def rescue_size(pos: Dict[str, Any], price: float, qty: float, entry: float, stop: float) -> float:
    """
    Shares to add so that, if the unchanged stop is then hit, the total loss is at
    most RECOVERY_MAX_RISK_MULT x the original risk, and no more than
    RECOVERY_ADD_FRACTION of the current quantity. 0 when nothing fits.

    Loss at the stop after adding `a` shares at `price`:
        (entry - stop) * qty + (price - stop) * a
    """
    risk_ps = _risk_per_share(pos, entry, stop)
    orig_qty = float(pos.get("initial_qty") or qty)
    budget = settings.RECOVERY_MAX_RISK_MULT * risk_ps * orig_qty
    now_at_risk = (entry - stop) * qty
    per_share = price - stop
    if per_share <= 0 or budget <= now_at_risk:
        return 0.0
    return max(0.0, min((budget - now_at_risk) / per_share, qty * settings.RECOVERY_ADD_FRACTION))


def check_rescue(symbol: str, pos: Dict[str, Any], price: float, qty: float, entry: float,
                 stop: float, news_bearish: bool, strategy_exiting: bool,
                 now: Optional[float] = None, trend=None) -> Optional[TradeDecision]:
    """
    A one-time rescue BUY for a losing position whose fall has stalled, or None.
    `trend` is the symbol's TrendRead; read from the live board when omitted.
    """
    if not (settings.RECOVERY_ENABLED and settings.RECOVERY_ADD_ENABLED):
        return None
    if pos.get("rescued") or pos.get("scaled_in") or qty <= 0 or entry <= 0 or stop >= entry:
        return None
    if news_bearish or strategy_exiting or not state.is_trading_active:
        return None
    risk_ps = _risk_per_share(pos, entry, stop)
    if risk_ps <= 0:
        return None
    depth = (entry - price) / risk_ps
    if not (settings.RECOVERY_ADD_MIN_R <= depth <= settings.RECOVERY_ADD_MAX_R):
        return None
    now = time.time() if now is None else now
    if now - float(pos.get("rescue_attempt_at") or 0.0) < settings.RECOVERY_RETRY_SECONDS:
        return None
    # Do not catch a falling knife: the short-term trend has to have stopped falling.
    if trend is None:
        from engine.trend import board
        trend = board.read(symbol)
    r = trend
    if not r.ready or r.reversal_down or r.micro is None or r.micro < settings.RECOVERY_MIN_MICRO:
        return None
    if r.direction <= -settings.TREND_EXIT_DIRECTION:
        return None
    add = rescue_size(pos, price, qty, entry, stop)
    if add <= 0:
        return None
    pos["rescue_attempt_at"] = now
    pos.setdefault("initial_qty", qty)
    worst = (entry - stop) * qty + (price - stop) * add
    new_entry = (entry * qty + price * add) / (qty + add)
    return TradeDecision(
        symbol=symbol, action="BUY", rescue=True, rescue_qty=add,
        reason=(f"Loss recovery: down {depth:.2f}R and the fall has stalled (micro {r.micro:+.2f}); "
                f"adding {add:.6g} @ ${price:,.6g} lowers break-even ${entry:,.6g} -> ${new_entry:,.6g}. "
                f"Stop stays ${stop:,.6g}; worst case ${worst:,.2f}"),
    )
