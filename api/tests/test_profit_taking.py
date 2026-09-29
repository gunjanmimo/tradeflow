"""
Profit-taking: sell into strength once a trade has earned it.

  * nothing happens before +1R (the old "sell half of any uptick" cut winners)
  * at +1R half is sold and the stop goes to breakeven, so the rest cannot lose
  * after that the stop trails the high by 1R and only ever moves up
  * the broker sells what the broker says is held, then protects the rest (OCO)
  * a stop at or above the entry after a scale-out is protection, not a broken bracket
"""
import asyncio
import time
import types

import pytest

from core.config import settings
from core.state import state, PriceTick
from engine import profit_manager as pm
from engine import brackets


def _pos(**kw):
    return {"symbol": "AMD", "avg_entry_price": 100.0, "stop_loss": 99.0, "initial_stop": 99.0,
            "take_profit": 102.0, "qty": 10.0, "mode": "SIMULATED", **kw}


def test_no_action_before_one_r():
    assert pm.plan(_pos(), 100.9, 100.9) is None
    assert pm.progress(_pos(), 100.5) == pytest.approx(0.5)


def test_scale_out_at_one_r_moves_the_stop_to_breakeven():
    act = pm.plan(_pos(), 101.0, 101.0)
    assert act["type"] == "scale_out" and act["fraction"] == settings.SCALE_OUT_FRACTION
    assert act["new_stop"] == pytest.approx(100.0 * (1 + settings.BREAKEVEN_BUFFER_PCT / 100))


def test_trailing_stop_only_moves_up():
    p = _pos(scaled_out=True, stop_loss=100.05)
    act = pm.plan(p, 101.8, 101.8)                      # high 101.8 - 1R = 100.8
    assert act["type"] == "raise_stop" and act["new_stop"] == pytest.approx(100.8)
    p["stop_loss"] = 100.8
    assert pm.plan(p, 101.2, 101.8) is None             # price dips: the stop stays
    assert pm.plan(p, 101.85, 101.85) is None           # under the 0.1R step: no churn


def test_disabled_means_no_profit_taking(monkeypatch):
    monkeypatch.setattr(settings, "PROFIT_TAKING_ENABLED", False)
    assert pm.plan(_pos(), 101.5, 101.5) is None


def test_raised_stop_is_a_valid_bracket():
    assert not brackets.is_valid(100.0, 100.05, 102.0)
    assert brackets.is_valid(100.0, 100.05, 102.0, raised=True)
    p = _pos(scaled_out=True, stop_loss=100.5)
    brackets.ensure(p, price=100.0, atr=1.0)
    assert p["stop_loss"] == 100.5                      # not "repaired" back below the entry


def test_simulated_scale_out(monkeypatch):
    from engine.executor import executor
    import core.market_hours as mh
    monkeypatch.setattr(mh, "us_session", lambda *a, **k: mh.REGULAR)
    monkeypatch.setattr(state, "active_positions", {"AMD": _pos()})
    monkeypatch.setattr(state, "latest_prices", {"AMD": PriceTick(symbol="AMD", price=101.0, bid=100.99,
                                                                  ask=101.01, volume=0, timestamp=time.time())})
    monkeypatch.setattr(state, "realized_pnl_today", 0.0)
    cash = state.account_info.get("cash", 0.0)
    asyncio.run(executor.manage_profit("AMD", pm.plan(state.active_positions["AMD"], 101.0, 101.0)))
    pos = state.active_positions["AMD"]
    assert pos["qty"] == 5 and pos["scaled_out"] and pos["stop_loss"] > 100.0
    assert state.realized_pnl_today == pytest.approx(5.0)          # 5 shares x $1
    assert state.account_info["cash"] == pytest.approx(cash + 505.0)
    assert state.closed_trades[-1]["qty"] == 5 and "Profit-taking" in state.closed_trades[-1]["exit_reason"]
    assert pm.meta.get("AMD")["scaled_out"] is True


def test_live_scale_out_sells_the_brokers_quantity_then_protects_the_rest(monkeypatch):
    from engine.executor import executor
    held = {"qty": "11"}
    sent, cancelled = [], []

    def submit(req):
        sent.append(req)
        if getattr(req, "order_class", None) is None:        # the market sell
            held["qty"] = str(11 - req.qty)
        return types.SimpleNamespace(id=f"o{len(sent)}")

    client = types.SimpleNamespace(
        submit_order=submit,
        get_open_position=lambda sym: types.SimpleNamespace(qty=held["qty"]),
        get_order_by_id=lambda oid: types.SimpleNamespace(status="filled"),
        get_orders=lambda req: [], cancel_order_by_id=lambda oid: cancelled.append(oid))
    monkeypatch.setattr(executor, "trading_client", client)
    sold, _ = executor._scale_out_sync("AMD", 0.5, 100.05, 102.0)
    from alpaca.trading.enums import OrderClass
    assert sold == 5                                            # floor(11 x 0.5), from the broker's 11
    market, oco = sent
    assert market.qty == 5 and str(market.side).lower().endswith("sell")
    assert oco.order_class == OrderClass.OCO and oco.qty == 6
    assert oco.stop_loss.stop_price == 100.05 and oco.take_profit.limit_price == 102.0


def test_a_rejected_oco_does_not_undo_the_scale_out(monkeypatch):
    """The sell went through; a failed OCO must not make the engine sell half again."""
    from engine.executor import executor
    held = {"qty": "10"}

    def submit(req):
        if getattr(req, "order_class", None) is not None:
            raise RuntimeError("OCO rejected")
        held["qty"] = "5"
        return types.SimpleNamespace(id="m1")

    monkeypatch.setattr(executor, "trading_client", types.SimpleNamespace(
        submit_order=submit, get_open_position=lambda s: types.SimpleNamespace(qty=held["qty"]),
        get_order_by_id=lambda oid: types.SimpleNamespace(status="filled"),
        get_orders=lambda req: [], cancel_order_by_id=lambda oid: None))
    sold, oid = executor._scale_out_sync("AMD", 0.5, 100.05, 102.0)
    assert sold == 5 and oid == "m1"
