"""
Fill reconciliation: book every close at the price the broker actually filled.

When the engine closes a position it books the P&L at once, from the last mark,
so the daily-loss halt reacts immediately. That number is provisional: the
market order fills a moment later at the bid, not at the mark, and a bracket
leg closed at the broker is only noticed at the next sync. Booking marks and
never correcting them is how the ledger showed -$1.6k while the paper account
lost $13k in two days.

Every close is therefore tracked until its fill is known:

  own close / trim   by the order id the broker returned
  broker close       (bracket stop/target, manual close) by the symbol's filled
                     sell orders since the position was opened

The difference between the fill and the provisional number is then booked
(state.revise_realized_pnl), so the budget, the stair ladder, the daily ledger
and the halt all converge on what the account really made or lost.

Broker calls run in the sync thread; bookings are applied on the event loop.
"""
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("tradeflow.fills")

GIVE_UP_SECONDS = 900.0        # a fill not found by then is left as booked
LOOKBACK_SLACK_SECONDS = 120.0


class FillReconciler:
    def __init__(self):
        self._checks: List[Dict[str, Any]] = []
        self._done: List[tuple] = []          # (check, fill_qty, fill_price) ready to book
        self._consumed: set = set()           # broker order ids already matched to a close
        self._lock = threading.Lock()
        self.reconciled = 0
        self.total_correction = 0.0
        self.last_error: Optional[str] = None

    # ------------------------------------------------------------------
    def track(self, symbol: str, qty: float, entry: float, provisional_pnl: float,
              record: Optional[Dict[str, Any]] = None, order_id: Optional[str] = None,
              since: Optional[float] = None, on_revised: Optional[Callable] = None):
        """
        Registers a close whose P&L was booked provisionally. With order_id the
        fill is read from that order; without, from the symbol's filled sells
        after `since` (a close the broker made on its own).
        """
        if qty <= 0 or entry <= 0:
            return
        with self._lock:
            self._checks.append({
                "symbol": symbol, "qty": float(qty), "entry": float(entry),
                "provisional": float(provisional_pnl), "record": record,
                "order_id": str(order_id) if order_id else None,
                "since": float(since or time.time()), "at": time.time(),
                "on_revised": on_revised,
            })
            if order_id:
                self._consumed.add(str(order_id))

    @property
    def pending(self) -> int:
        return len(self._checks)

    # ------------------------------------------------------------------
    # Sync thread
    # ------------------------------------------------------------------
    def reconcile_sync(self, client) -> None:
        """Looks up fills for every open check. Blocking; call from the sync thread."""
        if client is None:
            return
        with self._lock:
            checks = list(self._checks)
        now = time.time()
        for chk in checks:
            try:
                got = self._by_order(client, chk) if chk["order_id"] else self._by_symbol(client, chk)
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {str(e)[:160]}"
                logger.debug("Fill lookup for %s failed: %s", chk["symbol"], self.last_error)
                got = None
            with self._lock:
                if got is not None:
                    self._checks.remove(chk)
                    self._done.append((chk, got[0], got[1]))
                elif now - chk["at"] > GIVE_UP_SECONDS:
                    self._checks.remove(chk)
                    logger.warning("No broker fill found for %s close after %.0fs; keeping the "
                                   "provisional P&L %.2f", chk["symbol"], now - chk["at"], chk["provisional"])

    @staticmethod
    def _terminal(order) -> bool:
        status = str(getattr(order, "status", "")).lower()
        return any(k in status for k in ("filled", "canceled", "cancelled", "expired", "rejected",
                                          "done_for_day", "replaced"))

    def _by_order(self, client, chk) -> Optional[tuple]:
        o = client.get_order_by_id(chk["order_id"])
        filled = float(getattr(o, "filled_qty", 0) or 0)
        price = float(getattr(o, "filled_avg_price", 0) or 0)
        status = str(getattr(o, "status", "")).lower()
        if filled > 0 and price > 0 and (status.endswith("filled") and "partially" not in status):
            return filled, price
        if self._terminal(o) and "partially" not in status:
            # Cancelled/expired with a partial fill: book what did fill. None at
            # all: the provisional booking was wrong, the position is still open
            # and the sync re-adds it; nothing was realised.
            return (filled, price) if filled > 0 and price > 0 else (0.0, 0.0)
        return None

    def _by_symbol(self, client, chk) -> Optional[tuple]:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        after = datetime.fromtimestamp(chk["since"] - LOOKBACK_SLACK_SECONDS, tz=timezone.utc)
        orders = client.get_orders(GetOrdersRequest(
            status=QueryOrderStatus.CLOSED, symbols=[chk["symbol"]], after=after, limit=100))
        need, qty, cash, used = chk["qty"], 0.0, 0.0, []
        for o in sorted(orders, key=lambda o: getattr(o, "filled_at", None) or datetime.min.replace(tzinfo=timezone.utc)):
            oid = str(o.id)
            if oid in self._consumed or "sell" not in str(o.side).lower():
                continue
            f = float(getattr(o, "filled_qty", 0) or 0)
            p = float(getattr(o, "filled_avg_price", 0) or 0)
            if f <= 0 or p <= 0:
                continue
            take = min(f, need - qty)
            qty += take
            cash += take * p
            used.append(oid)
            if qty >= need - 1e-9:
                break
        if qty >= need - 1e-9:
            with self._lock:
                self._consumed.update(used)
            return qty, cash / qty
        return None

    # ------------------------------------------------------------------
    # Event loop
    # ------------------------------------------------------------------
    def apply(self) -> int:
        """Books every fill found since the last call. Returns how many."""
        from core.state import state
        with self._lock:
            done, self._done = self._done, []
        for chk, qty, price in done:
            exact = round((price - chk["entry"]) * qty, 2) if qty > 0 else 0.0
            old = chk["provisional"]
            state.revise_realized_pnl(chk["symbol"], old, exact)
            self.reconciled += 1
            self.total_correction = round(self.total_correction + exact - old, 2)
            rec = chk.get("record")
            if rec is not None:
                rec["pnl"] = exact
                rec["provisional_pnl"] = old
                rec["pnl_source"] = "broker_fill"
                if qty > 0:
                    rec["price"] = price
                risk = float(rec.get("initial_risk") or 0.0)
                rec["r_multiple"] = round(exact / risk, 3) if risk > 0 else None
            if abs(exact - old) >= 0.01:
                state.log_event("FILL_RECONCILED",
                    f"{chk['symbol']}: filled {qty:g} @ ${price:,.4g}; P&L ${old:+,.2f} (mark) -> "
                    f"${exact:+,.2f} (fill), correction ${exact - old:+,.2f}")
            cb = chk.get("on_revised")
            if cb is not None:
                try:
                    cb(chk["symbol"], exact, rec)
                except Exception as e:
                    logger.debug("on_revised callback failed: %s", e)
        return len(done)

    def status(self) -> Dict[str, Any]:
        return {"pending": self.pending, "reconciled": self.reconciled,
                "total_correction": self.total_correction, "last_error": self.last_error}


fills = FillReconciler()
