"""
Accounting and live/backtest parity.

  * a close is re-booked at the broker's fill: the budget, the daily ledger and
    the halt see what the account really lost (engine/fills.py)
  * the daily-loss halt also trips on the broker account's own day loss
  * only completed one-minute bars add indicator samples; quotes move bid/ask
  * ATR comes from the true range of bars, the same way live and in backtests
  * the trading day is the New York date
"""
import types
from datetime import datetime, timezone

import numpy as np
import pytest

from core.config import settings
from core.state import state, ny_date
from core.pnl_ledger import pnl_ledger
from core.capital_plan import capital_plan
from engine.fills import FillReconciler
from engine.quant_matrix import QuantMatrix


@pytest.fixture
def books(monkeypatch):
    monkeypatch.setattr(capital_plan.plan, "mode", "classic")
    monkeypatch.setattr(capital_plan.plan, "classic_realized", 0.0)
    monkeypatch.setattr(state, "realized_pnl_today", 0.0)
    monkeypatch.setattr(state, "harvested_today", 0.0)
    monkeypatch.setattr(state, "allocated_capital", 10_000.0)
    monkeypatch.setattr(state, "trading_day", ny_date())
    yield


def _order(oid, side, qty, price, status="filled", filled_at=None):
    return types.SimpleNamespace(id=oid, side=f"OrderSide.{side.upper()}", filled_qty=qty,
                                 filled_avg_price=price, status=f"OrderStatus.{status.upper()}",
                                 filled_at=filled_at or datetime.now(timezone.utc))


class _Client:
    def __init__(self, by_id=None, closed=None):
        self.by_id = by_id or {}
        self.closed = closed or []

    def get_order_by_id(self, oid):
        return self.by_id[oid]

    def get_orders(self, req):
        return list(self.closed)


# ---- fill reconciliation ---------------------------------------------------------

def test_own_close_is_rebooked_at_the_fill(books):
    f = FillReconciler()
    state.book_realized_pnl("AAPL", +5.0)            # mark said +$5 on 10 shares from 100
    rec = {"pnl": 5.0, "initial_risk": 10.0}
    f.track("AAPL", 10, 100.0, 5.0, rec, order_id="o1")
    f.reconcile_sync(_Client(by_id={"o1": _order("o1", "sell", 10, 99.70)}))
    assert f.apply() == 1
    assert rec["pnl"] == pytest.approx(-3.0) and rec["pnl_source"] == "broker_fill"
    assert rec["r_multiple"] == pytest.approx(-0.3)
    assert state.realized_pnl_today == pytest.approx(-3.0)
    assert capital_plan.plan.classic_realized == pytest.approx(-3.0)
    day = pnl_ledger.roll()
    assert day["realized_pnl"] == pytest.approx(-3.0)
    # the fill turned a booked win into a loss: counted once, as a loss
    assert day["closed_trades"] == 1 and day["winning_trades"] == 0 and day["losing_trades"] == 1


def test_unfilled_order_waits(books):
    f = FillReconciler()
    f.track("AAPL", 10, 100.0, 5.0, {}, order_id="o1")
    f.reconcile_sync(_Client(by_id={"o1": _order("o1", "sell", 0, 0, status="new")}))
    assert f.apply() == 0 and f.pending == 1


def test_broker_close_is_matched_by_symbol_and_not_reused(books):
    f = FillReconciler()
    state.book_realized_pnl("MSFT", -2.0)
    state.book_realized_pnl("MSFT", -2.0)
    client = _Client(closed=[_order("b1", "sell", 5, 97.0), _order("x", "buy", 5, 100.0)])
    f.track("MSFT", 5, 100.0, -2.0, {}, since=0)
    f.reconcile_sync(client)
    f.track("MSFT", 5, 100.0, -2.0, {}, since=0)      # a second close: b1 is already used
    f.reconcile_sync(client)
    assert f.apply() == 1 and f.pending == 1
    assert state.realized_pnl_today == pytest.approx(-15.0 - 2.0)


def test_give_up_keeps_the_provisional_number(books, monkeypatch):
    import engine.fills as fm
    f = FillReconciler()
    f.track("AAPL", 10, 100.0, 5.0, {}, order_id="o1")
    f._checks[0]["at"] -= fm.GIVE_UP_SECONDS + 1
    f.reconcile_sync(_Client(by_id={"o1": _order("o1", "sell", 0, 0, status="new")}))
    assert f.pending == 0 and f.apply() == 0


# ---- the halt reads the broker account too ------------------------------------------

def test_halt_trips_on_broker_equity_even_when_bookings_look_fine(books, monkeypatch):
    from engine.risk_guard import risk_guard
    import core.market_hours as mh
    monkeypatch.setattr(state, "is_trading_active", True)
    monkeypatch.setattr(mh, "us_session", lambda *a, **k: mh.REGULAR)
    monkeypatch.setattr(mh, "minutes_to_close", lambda *a, **k: 240.0)
    monkeypatch.setitem(state.account_info, "last_equity", 100_000.0)
    monkeypatch.setitem(state.account_info, "equity", 99_500.0)     # -$500 = 5% of a $10k budget
    ok, why = risk_guard.can_open_position("AAPL")
    assert not ok and "broker account down 5.00%" in why
    monkeypatch.setattr(settings, "RISK_BROKER_EQUITY_GUARD", False)
    state.halt_reason = None
    ok, why = risk_guard.can_open_position("AAPL")
    assert "Daily loss" not in why


# ---- parity -------------------------------------------------------------------------

def test_quotes_move_bid_ask_but_not_history(monkeypatch):
    sym = "PARQ"
    state.price_history.pop(sym, None)
    state.update_price(sym, 100.0, record_history=True)
    state.update_quote(sym, 99.95, 100.05)
    assert len(state.price_history[sym]) == 1
    # a bar close without bid/ask keeps the fresh quote instead of bid=ask=close
    state.update_price(sym, 100.02, record_history=True)
    tick = state.latest_prices[sym]
    assert (tick.bid, tick.ask) == (99.95, 100.05)
    assert state.spread_estimate[sym] == pytest.approx(0.001, rel=1e-3)
    for d in (state.price_history, state.latest_prices, state.latest_quotes, state.spread_estimate):
        d.pop(sym, None)


def test_true_range_atr_uses_gaps_and_ranges():
    h = np.array([10.0, 10.5, 12.0, 11.0])
    l = np.array([9.5, 10.0, 11.5, 10.8])
    c = np.array([10.0, 10.2, 11.8, 10.9])
    # TR: max(0.5, .5, 0)=0.5 ; max(0.5, 1.8, 1.3)=1.8 ; max(0.2, 0.8, 1.0)=1.0
    assert QuantMatrix.true_range_atr(h, l, c, period=3) == pytest.approx((0.5 + 1.8 + 1.0) / 3)
    assert QuantMatrix.true_range_atr(h, l, c, period=4) is None


def test_backtest_atr_matches_the_live_formula():
    from backtest import sim
    rng = np.random.default_rng(0)
    c = 100 + np.cumsum(rng.normal(0, 0.1, 120))
    bars = [(29_000_000 + i, c[i], c[i] + 0.05, c[i] - 0.07, c[i], 100.0) for i in range(120)]
    tape = sim.Tape("AAPL", bars, sim.Costs())
    i = 100
    expect = QuantMatrix.true_range_atr(tape.h[i - 59:i + 1], tape.l[i - 59:i + 1], tape.c[i - 59:i + 1])
    assert tape.atr(i, tape.c[i]) == pytest.approx(expect)


def test_trading_day_is_the_new_york_date():
    # 02:30 UTC on Sep 29 is still Sep 28 in New York
    ts = datetime(2026, 9, 29, 2, 30, tzinfo=timezone.utc).timestamp()
    assert ny_date(ts) == "2026-09-28"
