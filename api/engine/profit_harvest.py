"""
Profit harvest: take money off the table the moment a position is winning.

Whenever a position shows ANY profit, however small, sell
PROFIT_HARVEST_FRACTION of it. The gain on the sold part is booked as day
income (state.book_harvested_income): it counts in today's profit but is never
traded again and never offsets a loss. The remainder keeps running under its
stop, target and trailing stop.

Two rules keep "any profit" honest:

  * Profit is measured at the bid, the price a market sell actually fills at,
    not the last trade. A one-cent profit on the last print is usually a loss
    once the spread is paid, and would be booked as negative "income".
  * The remainder is harvested again only when the bid beats the price of the
    last harvest -- that is, only on NEW profit. Otherwise a position sitting at
    the same small profit would be halved every retry until nothing was left.

  * Crypto pays Alpaca's taker fee on the buy and again on the harvest sell,
    so its profit must clear both fees; the income booked is net of them.

A stock that Alpaca can trade in fractions is harvested in fractional shares:
a one-share winner up $3.70 sells 0.5 share and banks about $1.85. Stocks that
are not fractionable sell whole shares only, so one share is held whole.

PROFIT_HARVEST_USD raises the bar above zero if wanted (0 = any profit). Each
harvest must bank at least PROFIT_HARVEST_MIN_INCOME (a cent), so a sale never
shows up as $0.00.
Never fires on a loss. A position too small to split is held whole rather than
closed: the stop and trailing stop still protect it.
"""
import time
from typing import Any, Dict, Optional

from core.config import settings
from core.state import state, TradeDecision, is_crypto_symbol


def sell_price(symbol: str, price: float) -> float:
    """What a market sell would fill at now: the bid when there is one, else the last price."""
    tick = state.latest_prices.get(symbol)
    bid = float(getattr(tick, "bid", 0.0) or 0.0) if tick else 0.0
    return bid if 0.0 < bid <= price * 1.5 else price


def harvest_qty(symbol: str, qty: float, price: float, simulated: bool = False) -> float:
    """
    Quantity a harvest may sell, rounded down. Crypto at its own precision. A
    fractionable stock (or a simulated position) in fractional shares to
    HARVEST_FRACTION_DECIMALS, subject to Alpaca's fractional-order minimum
    notional; any other stock in whole shares only.
    """
    import math
    from engine.executor import AlpacaExecutor
    if is_crypto_symbol(symbol):
        return AlpacaExecutor._round_qty(symbol, qty, price)
    whole = float(math.floor(qty))
    if not (simulated or symbol in state.fractionable_symbols):
        return whole
    d = settings.HARVEST_FRACTION_DECIMALS
    frac = math.floor(qty * 10 ** d) / 10 ** d
    if frac != whole and frac * price < settings.HARVEST_MIN_FRACTIONAL_NOTIONAL:
        return whole
    return frac


def net_unit_profit(symbol: str, fill: float, entry: float) -> float:
    """Profit per unit sold at `fill`, after crypto fees on both the buy and this sell."""
    if not is_crypto_symbol(symbol):
        return fill - entry
    fee = settings.CRYPTO_TAKER_FEE_BPS / 1e4
    return fill * (1.0 - fee) - entry * (1.0 + fee)


def check(symbol: str, pos: Dict[str, Any], price: float, qty: float, entry: float,
          now: Optional[float] = None) -> Optional[TradeDecision]:
    """A harvest SELL for this position, or None."""
    if not settings.PROFIT_HARVEST_ENABLED or qty <= 0 or entry <= 0:
        return None
    fill = sell_price(symbol, price)
    profit = net_unit_profit(symbol, fill, entry) * qty
    if profit <= 0:
        return None
    if settings.PROFIT_HARVEST_USD > 0 and profit + 1e-9 < settings.PROFIT_HARVEST_USD:
        return None
    # Only new profit: the bid has to beat where the last harvest sold.
    if fill <= float(pos.get("harvest_last_price") or 0.0):
        return None
    if pos.get("harvest_unsplittable"):
        return None
    now = time.time() if now is None else now
    if now - float(pos.get("harvest_attempt_at") or 0.0) < settings.PROFIT_HARVEST_RETRY_SECONDS:
        return None

    fraction = min(max(settings.PROFIT_HARVEST_FRACTION, 0.0), 1.0)
    simulated = pos.get("mode") in ("SIMULATED", "PAPER_SIMULATED")
    sell = harvest_qty(symbol, qty * fraction, price, simulated)
    if sell <= 0 or qty - sell <= 1e-9:
        pos["harvest_unsplittable"] = True
        state.log_event("PROFIT_HARVEST", f"{symbol}: {qty} is too small to sell {fraction:.0%} of; "
                                          f"holding it whole under its stop (profit ${profit:+,.2f})")
        return None

    # The part actually sold must bank at least a cent: no $0.00 harvests.
    income = net_unit_profit(symbol, fill, entry) * sell
    if income + 1e-9 < settings.PROFIT_HARVEST_MIN_INCOME:
        return None

    pos["harvest_attempt_at"] = now
    return TradeDecision(
        symbol=symbol, action="SELL", fraction=fraction, harvest=True,
        reason=(f"Profit harvest: unrealised ${profit:+,.2f} at the bid ${fill:,.6g}"
                + (" after fees" if is_crypto_symbol(symbol) else "")
                + f"; selling {sell:g} of {qty:g} to bank ${income:,.2f} as day income"),
    )
