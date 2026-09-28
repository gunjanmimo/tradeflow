"""
Loss recovery: confirmed stops, the one-time rescue add and the break-even lock.
The invariant under all of it: the stop never moves down, and the loss at the
stop after a rescue is at most RECOVERY_MAX_RISK_MULT x the original risk.
"""
import asyncio

import pytest

from core.capital_plan import capital_plan
from core.config import settings
from core.state import state
from engine import loss_recovery
from engine.trend import TrendRead, board

SYM = "RCVR"
ENTRY, STOP, QTY = 100.0, 95.0, 10.0   # 5/share, $50 original risk


def _pos(**kw):
    return {"symbol": SYM, "qty": QTY, "avg_entry_price": ENTRY, "stop_loss": STOP,
            "take_profit": 110.0, "mode": "SIMULATED", **kw}


@pytest.fixture
def trend(monkeypatch):
    """Sets the trend read loss_recovery sees for SYM."""
    def set_(**kw):
        r = TrendRead(symbol=SYM, ready=True, bars=60, **kw)
        monkeypatch.setattr(board, "read", lambda s, max_age_s=5.0: r)
    monkeypatch.setattr(state, "is_trading_active", True)
    set_(direction=-0.1, micro=0.2)
    return set_


def _rescue(pos, price, news_bearish=False, strategy_exiting=False, now=1e9):
    return loss_recovery.check_rescue(SYM, pos, price, float(pos["qty"]),
                                      float(pos["avg_entry_price"]), float(pos["stop_loss"]),
                                      news_bearish=news_bearish, strategy_exiting=strategy_exiting,
                                      now=now)


# ---- stop confirmation ------------------------------------------------------

def test_stop_waits_for_confirmation_then_closes():
    pos = _pos()
    assert loss_recovery.check_stop(pos, 94.9, ENTRY, STOP, now=1000.0) == (False, "")
    assert pos["stop_state"].startswith("confirming")
    close, why = loss_recovery.check_stop(pos, 94.9, ENTRY, STOP,
                                          now=1000.0 + settings.STOP_CONFIRM_SECONDS)
    assert close and "stayed" in why


def test_recovering_above_the_stop_resets_the_clock():
    pos = _pos()
    loss_recovery.check_stop(pos, 94.9, ENTRY, STOP, now=1000.0)
    loss_recovery.check_stop(pos, 95.5, ENTRY, STOP, now=1005.0)
    close, _ = loss_recovery.check_stop(pos, 94.9, ENTRY, STOP,
                                        now=1006.0 + settings.STOP_CONFIRM_SECONDS / 2)
    assert not close


def test_profit_locking_stop_fires_at_once():
    pos = _pos(stop_loss=101.0)
    close, _ = loss_recovery.check_stop(pos, 100.9, ENTRY, 101.0, now=1000.0)
    assert close


def test_broker_leg_sits_at_the_disaster_stop():
    assert loss_recovery.broker_stop(ENTRY, STOP) == pytest.approx(92.5)


# ---- rescue add ---------------------------------------------------------------

def test_rescue_caps_the_loss_at_the_stop(trend):
    pos = _pos()
    d = _rescue(pos, 97.5)                       # down 0.5R
    assert d is not None and d.rescue and d.action == "BUY"
    worst = (ENTRY - STOP) * QTY + (97.5 - STOP) * d.rescue_qty
    assert worst <= settings.RECOVERY_MAX_RISK_MULT * 50.0 + 1e-6
    assert d.rescue_qty <= QTY * settings.RECOVERY_ADD_FRACTION


@pytest.mark.parametrize("price", [99.0, 95.5])   # 0.2R: not a loser yet; 0.9R: too near the stop
def test_no_rescue_outside_the_window(trend, price):
    assert _rescue(_pos(), price) is None


def test_no_rescue_while_still_falling(trend):
    trend(direction=-0.2, micro=-0.3)
    assert _rescue(_pos(), 97.5) is None
    trend(direction=0.3, micro=-0.5, reversal_down=True)
    assert _rescue(_pos(), 97.5) is None


def test_no_rescue_on_bearish_news_or_a_strategy_exit(trend):
    assert _rescue(_pos(), 97.5, news_bearish=True) is None
    assert _rescue(_pos(), 97.5, strategy_exiting=True) is None


def test_no_rescue_on_crypto_by_default(trend, monkeypatch):
    pos = _pos(symbol="RCVR/USD")
    assert loss_recovery.check_rescue("RCVR/USD", pos, 97.5, QTY, ENTRY, STOP,
                                      news_bearish=False, strategy_exiting=False, now=1e9,
                                      trend=board.read(SYM)) is None


def test_rescue_is_once_and_throttled(trend):
    pos = _pos()
    assert _rescue(pos, 97.5, now=1e9) is not None
    assert _rescue(pos, 97.5, now=1e9 + 1) is None      # retry throttle
    pos["rescued"] = True
    assert _rescue(pos, 97.5, now=1e9 + 3600) is None   # once per position


def test_rescue_fills_through_the_executor(trend, monkeypatch):
    from engine.executor import AlpacaExecutor
    e = AlpacaExecutor()
    e.is_mock_mode = True
    monkeypatch.setattr(capital_plan, "_save", lambda: None)
    monkeypatch.setattr(state, "allocated_capital", 10000.0)
    monkeypatch.setattr(state, "account_info", {"cash": 100000.0})
    state.update_price(SYM, 97.5)
    state.active_positions[SYM] = _pos(invested_dollars=1000.0)
    try:
        d = _rescue(state.active_positions[SYM], 97.5)
        asyncio.run(e.execute_decision(d))
        pos = state.active_positions[SYM]
        assert pos["rescued"] and not pos.get("scaled_in")
        assert pos["qty"] == pytest.approx(QTY + d.rescue_qty, abs=1e-3)
        assert pos["avg_entry_price"] < ENTRY
        assert pos["stop_loss"] == STOP                  # the stop never moves down
    finally:
        state.active_positions.pop(SYM, None)
        state.latest_prices.pop(SYM, None)


# ---- break-even lock ----------------------------------------------------------

def test_breakeven_lock_after_a_recovered_loss():
    pos = _pos()
    assert loss_recovery.breakeven_lock(pos, 97.5, ENTRY, STOP) is None   # under water
    assert pos["recovery_mode"]
    assert loss_recovery.breakeven_lock(pos, 100.05, ENTRY, STOP) is None  # not clear of fees yet
    lock = loss_recovery.breakeven_lock(pos, 100.5, ENTRY, STOP)
    assert lock == pytest.approx(100.1)
    assert loss_recovery.breakeven_lock(pos, 101.0, ENTRY, lock) is None   # locks once


def test_no_lock_for_a_position_that_never_went_under_water():
    pos = _pos()
    loss_recovery.breakeven_lock(pos, 99.5, ENTRY, STOP)                 # 0.1R dip
    assert loss_recovery.breakeven_lock(pos, 100.5, ENTRY, STOP) is None
