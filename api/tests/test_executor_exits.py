"""
Executor exit orders against a fake Alpaca client.

Regressions:
  * regular hours: bracket legs held every share, so close_position failed with
    "qty must be > 0" and the sentinel retried it every second.
  * extended hours: each CLOSE sent a fresh sell of our cached quantity while the
    previous one was still working, overselling longs into shorts; the short was
    then "closed" with a negative-qty sell the broker rejected forever.
  * extended hours: a quiet pre-market gives no fresh price, so every re-price of
    an unfilled exit was cancelled and re-sent at the identical limit (REGN: eight
    cancels at $776.09) and never filled.
"""
import asyncio
import datetime as dt
import time
import types

import pytest

from core.state import state, TradeDecision
import engine.executor as ex_mod
from engine.executor import AlpacaExecutor, ORDER_ID_PREFIX


class FakeClient:
    def __init__(self, qty, orders=()):
        self.qty = qty
        self.orders = list(orders)
        self.submitted = []
        self.closed = []

    def get_open_position(self, symbol):
        if self.qty == 0:
            raise Exception('{"code":40410000,"message":"position does not exist"}')
        return types.SimpleNamespace(qty=str(self.qty))

    def get_orders(self, req):
        return list(self.orders)

    def cancel_order_by_id(self, oid):
        self.orders = [o for o in self.orders if o.id != oid]

    def submit_order(self, req):
        self.submitted.append(req)
        return types.SimpleNamespace(id="new")

    def close_position(self, symbol):
        if self.orders:
            raise Exception('{"code":40010001,"message":"qty must be > 0"}')
        self.closed.append(symbol)


def _order(side, coid, age_s=1.0, oid="o1"):
    return types.SimpleNamespace(
        id=oid, side=f"OrderSide.{side.upper()}", client_order_id=coid,
        submitted_at=dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=age_s))


@pytest.fixture
def ex(monkeypatch):
    e = AlpacaExecutor()
    state.active_positions["NEM"] = {"symbol": "NEM", "qty": 11.0, "avg_entry_price": 116.2,
                                     "current_price": 116.0, "mode": "ALPACA_PAPER"}
    yield e
    state.active_positions.pop("NEM", None)


def _session(monkeypatch, name):
    import core.market_hours as mh
    monkeypatch.setattr(mh, "us_session", lambda *a, **k: getattr(mh, name))


def _close(e):
    asyncio.run(e._execute_close(TradeDecision(symbol="NEM", action="CLOSE", close=True,
                                               reason="test")))


def test_regular_hours_cancels_bracket_legs_then_closes(ex, monkeypatch):
    _session(monkeypatch, "REGULAR")
    ex.trading_client = FakeClient(11, [_order("sell", "leg-tp", oid="tp"),
                                        _order("sell", "leg-sl", oid="sl")])
    _close(ex)
    assert ex.trading_client.closed == ["NEM"]
    assert "NEM" not in state.active_positions


def test_extended_exit_uses_broker_qty_and_keeps_position(ex, monkeypatch):
    _session(monkeypatch, "PRE")
    ex.trading_client = FakeClient(8)          # broker already sold 3 of our cached 11
    _close(ex)
    [req] = ex.trading_client.submitted
    assert req.qty == 8 and "SELL" in str(req.side).upper()
    assert "NEM" in state.active_positions, "a limit exit is not a fill"
    assert "NEM" in ex.pending_exits


def test_extended_exit_not_resent_while_one_is_working(ex, monkeypatch):
    _session(monkeypatch, "PRE")
    ex.trading_client = FakeClient(11, [_order("sell", ORDER_ID_PREFIX + "abc")])
    _close(ex)
    assert ex.trading_client.submitted == []


def test_repeated_close_signals_send_one_exit(ex, monkeypatch):
    _session(monkeypatch, "PRE")
    client = FakeClient(11)
    orig = client.submit_order

    def submit(req):  # the order stays open at the broker, as a limit would
        client.orders.append(_order("sell", req.client_order_id, oid="exit"))
        return orig(req)
    client.submit_order = submit
    ex.trading_client = client
    for _ in range(5):
        _close(ex)
        ex.close_retry_after.clear()           # even with the backoff bypassed
    assert len(client.submitted) == 1


def test_short_is_bought_back(ex, monkeypatch):
    _session(monkeypatch, "PRE")
    ex.trading_client = FakeClient(-9)
    _close(ex)
    [req] = ex.trading_client.submitted
    assert req.qty == 9 and "BUY" in str(req.side).upper()


def test_failed_close_backs_off(ex, monkeypatch):
    _session(monkeypatch, "REGULAR")
    client = FakeClient(11)
    client.close_position = lambda s: (_ for _ in ()).throw(Exception("boom"))
    ex.trading_client = client
    _close(ex)
    assert ex.close_failures["NEM"] == 1
    calls = []
    client.close_position = lambda s: calls.append(s)
    _close(ex)                                  # inside the backoff window
    assert calls == []


def _limit(client):
    return client.submitted[-1].limit_price


def test_unfilled_exit_reprices_further_through_the_touch(ex, monkeypatch):
    monkeypatch.setattr(ex_mod.settings, "PREMARKET_EXIT_OFFSET_PCT", 0.5)
    monkeypatch.setattr(ex_mod.settings, "EXTENDED_EXIT_STEP_PCT", 0.5)
    monkeypatch.setattr(ex_mod.settings, "EXTENDED_EXIT_MAX_OFFSET_PCT", 1.5)
    ex.trading_client = FakeClient(11)
    prices = []
    for _ in range(5):                          # same stale price every time
        ex._submit_extended_exit("NEM", 100.0, 100.0)
        prices.append(_limit(ex.trading_client))
    assert prices == [99.5, 99.0, 98.5, 98.5, 98.5]   # widens, then holds at the cap


def test_exit_offset_resets_for_a_new_exit(ex, monkeypatch):
    monkeypatch.setattr(ex_mod.settings, "PREMARKET_EXIT_OFFSET_PCT", 0.5)
    monkeypatch.setattr(ex_mod.settings, "EXTENDED_EXIT_STEP_PCT", 0.5)
    ex.trading_client = FakeClient(11)
    ex.exit_attempts["NEM"] = (4, time.time() - 10_000)   # an exit from long ago
    ex._submit_extended_exit("NEM", 100.0, 100.0)
    assert _limit(ex.trading_client) == 99.5


def test_short_cover_widens_upward(ex, monkeypatch):
    monkeypatch.setattr(ex_mod.settings, "PREMARKET_EXIT_OFFSET_PCT", 0.5)
    monkeypatch.setattr(ex_mod.settings, "EXTENDED_EXIT_STEP_PCT", 0.5)
    ex.trading_client = FakeClient(-9)
    ex._submit_extended_exit("NEM", 100.0, 100.0)
    ex._submit_extended_exit("NEM", 100.0, 100.0)
    assert [r.limit_price for r in ex.trading_client.submitted] == [100.5, 101.0]
