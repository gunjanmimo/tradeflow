"""
Trend reading on 1-minute bars, the position manager's BUY / SELL / HOLD / CLOSE
decisions, and the partial-sell and add orders they send.
"""
import asyncio
import time

import numpy as np
import pytest

from core.config import settings
from core.state import state, TradeDecision
from core.minute_bars import minute_bars
from engine import trend
from engine.trend import TrendRead, drift, analyze
from engine.fleet import PositionManagerAgent


def _series(*legs, start=100.0):
    """Closes built from (bars, pct per bar) legs, with alternating 0.01% noise."""
    c, out = start, []
    for n, pct in legs:
        for _ in range(n):
            c *= 1 + pct / 100
            out.append(c * (1.0001 if len(out) % 2 else 0.9999))
    return np.array(out)


# ---- the math ---------------------------------------------------------------

def test_drift_reads_direction():
    up, _ = drift(_series((60, 0.1)))
    down, _ = drift(_series((60, -0.1)))
    flat, _ = drift(_series((60, 0.0)))
    assert up > 0.8 and down < -0.8 and abs(flat) < 0.3


def test_too_little_history_is_unknown_and_not_ready():
    r = analyze("X", _series((10, 0.2)), daily=None)
    assert r.label == "unknown" and not r.ready and "learning" in r.reasons[0]


def test_uptrend_and_downtrend_labels():
    assert analyze("X", _series((60, 0.1)), daily=None).label == "uptrend"
    assert analyze("X", _series((60, -0.1)), daily=None).label == "downtrend"


def test_daily_backdrop_moves_the_composite():
    s = _series((60, 0.02))
    bull = analyze("X", s, daily=1.0).direction
    bear = analyze("X", s, daily=-1.0).direction
    assert bull > bear


def test_reversal_down_is_seen_in_the_micro_trend_first():
    r = analyze("X", _series((50, 0.3), (12, -0.4)), daily=None)
    assert r.session > 0.2 and r.micro < -0.4
    assert r.reversal_down


def test_live_ticks_build_minute_bars():
    minute_bars.drop("TICK")
    t = 1_700_000_000.0 - (1_700_000_000.0 % 60)
    for i, p in enumerate((10.0, 10.5, 9.8, 10.2)):
        minute_bars.on_tick("TICK", p, 1.0, t + i)        # one minute
    minute_bars.on_tick("TICK", 11.0, 1.0, t + 61)        # the next
    assert list(minute_bars.closes("TICK")) == [10.2, 11.0]
    assert minute_bars._bars["TICK"][0][2:4] == [10.5, 9.8]    # high, low
    minute_bars.drop("TICK")


def test_history_replaces_a_provisional_tick_bar():
    """A bar built from trade prints is provisional: the API's real bar wins."""
    minute_bars.drop("HIST")
    minute_bars.on_tick("HIST", 5.0, 0.0, 600 * 60 + 1)
    minute_bars.merge_history("HIST", [(598, 1, 1, 1, 3.0, 0), (599, 1, 1, 1, 4.0, 0),
                                       (600, 1, 1, 1, 99.0, 0)])
    assert list(minute_bars.closes("HIST")) == [3.0, 4.0, 99.0]
    minute_bars.drop("HIST")


def test_a_streamed_bar_beats_history_and_keeps_its_own_minute():
    minute_bars.drop("HIST")
    minute_bars.on_bar("HIST", 600, 5.0, 5.5, 4.5, 5.0, 100.0)
    minute_bars.merge_history("HIST", [(599, 1, 1, 1, 4.0, 0), (600, 1, 1, 1, 99.0, 0)])
    assert [r[0] for r in minute_bars.rows("HIST")] == [599, 600]
    assert list(minute_bars.closes("HIST")) == [4.0, 5.0]
    # A late bar for an earlier minute is inserted in order, not appended.
    minute_bars.on_bar("HIST", 598, 3.0, 3.0, 3.0, 3.0, 1.0)
    assert [r[0] for r in minute_bars.rows("HIST")] == [598, 599, 600]
    minute_bars.drop("HIST")


def test_closed_rows_leave_out_the_forming_minute():
    minute_bars.drop("HIST")
    minute_bars.on_bar("HIST", 600, 5.0, 5.0, 5.0, 5.0, 1.0)
    minute_bars.on_tick("HIST", 6.0, 0.0, 601 * 60 + 5)
    assert [r[0] for r in minute_bars.closed_rows("HIST", now=601 * 60 + 30)] == [600]
    assert [r[0] for r in minute_bars.closed_rows("HIST", now=602 * 60)] == [600, 601]
    minute_bars.drop("HIST")


# ---- position manager decisions ----------------------------------------------

def _read(direction, conf=0.8, reversal_down=False, ready=True):
    return TrendRead(symbol="P", label="x", direction=direction, confidence=conf,
                     ready=ready, reversal_down=reversal_down, reasons=["r"])


def _pos(price, opened_min_ago=120, **kw):
    return {"qty": 10.0, "avg_entry_price": 100.0, "current_price": price,
            "stop_loss": 98.0, "opened_at": time.time() - opened_min_ago * 60, **kw}


pm = PositionManagerAgent()


def test_holds_while_the_trend_is_still_being_learned():
    assert pm.decide("AAPL", _pos(101), _read(-0.9, ready=False))[0] == "HOLD"


def test_closes_when_the_trend_turns_down():
    action, reason, _ = pm.decide("AAPL", _pos(99.5), _read(-0.6))
    assert action == "CLOSE" and "Trend turned down" in reason


def test_trend_close_waits_out_the_equity_minimum_hold():
    assert pm.decide("AAPL", _pos(99.5, opened_min_ago=5), _read(-0.6))[0] == "HOLD"




def test_trims_a_winner_when_the_trend_reverses():
    action, _, fraction = pm.decide("AAPL", _pos(102), _read(0.3, reversal_down=True))
    assert action == "SELL" and fraction == settings.TRIM_FRACTION


def test_trims_only_once():
    assert pm.decide("AAPL", _pos(102, trimmed=True), _read(-0.3))[0] == "HOLD"


def test_no_trim_on_a_loser():
    assert pm.decide("AAPL", _pos(99.8), _read(-0.3))[0] == "HOLD"


def test_adds_to_a_winner_in_a_strong_uptrend():
    # +2.5% with a $2/share stop distance = 1.25R
    action, _, fraction = pm.decide("AAPL", _pos(102.5), _read(0.7))
    assert action == "BUY" and fraction == settings.SCALE_IN_FRACTION


def test_no_add_below_one_r_or_twice():
    assert pm.decide("AAPL", _pos(101), _read(0.7))[0] == "HOLD"              # 0.5R
    assert pm.decide("AAPL", _pos(102.5, scaled_in=True), _read(0.7))[0] == "HOLD"


# ---- the orders ---------------------------------------------------------------

@pytest.fixture
def sim(monkeypatch):
    from engine.executor import AlpacaExecutor
    e = AlpacaExecutor()
    e.is_mock_mode = True
    booked = []
    monkeypatch.setattr(state, "book_realized_pnl", lambda s, p: booked.append((s, p)))
    saved = (state.allocated_capital, dict(state.account_info), state.halt_reason)
    state.allocated_capital = 10000.0
    state.account_info["cash"] = 100000.0
    state.halt_reason = None
    state.update_price("SIMX", 110.0)
    state.active_positions["SIMX"] = {"symbol": "SIMX", "qty": 10.0, "avg_entry_price": 100.0,
                                      "current_price": 110.0, "invested_dollars": 1000.0,
                                      "mode": "SIMULATED"}
    yield e, booked
    state.active_positions.pop("SIMX", None)
    state.latest_prices.pop("SIMX", None)
    minute_bars.drop("SIMX")
    state.allocated_capital, state.halt_reason = saved[0], saved[2]
    state.account_info.clear()
    state.account_info.update(saved[1])


def test_trim_sells_half_and_books_the_gain(sim):
    e, booked = sim
    asyncio.run(e._execute_trim(TradeDecision(symbol="SIMX", action="SELL", fraction=0.5)))
    pos = state.active_positions["SIMX"]
    assert pos["qty"] == 5.0 and pos["trimmed"]
    assert booked == [("SIMX", 50.0)]                     # 5 x ($110 - $100)


def test_trim_that_would_leave_nothing_closes(sim):
    e, booked = sim
    state.active_positions["SIMX"]["qty"] = 1.0
    asyncio.run(e._execute_trim(TradeDecision(symbol="SIMX", action="SELL", fraction=1.0)))
    assert "SIMX" not in state.active_positions


def test_one_share_is_not_split_and_not_asked_again(sim):
    e, booked = sim
    state.active_positions["SIMX"]["qty"] = 1.0
    asyncio.run(e._execute_trim(TradeDecision(symbol="SIMX", action="SELL", fraction=0.5)))
    pos = state.active_positions["SIMX"]
    assert pos["qty"] == 1.0 and pos["trimmed"] and booked == []


def test_add_is_capped_and_happens_once(sim):
    e, _ = sim
    asyncio.run(e._execute_add(TradeDecision(symbol="SIMX", action="BUY", fraction=0.5)))
    pos = state.active_positions["SIMX"]
    # half of $1,000 = $500 -> 4 shares at $110; the 15% cap ($1,500) leaves $400 room -> 3
    assert pos["qty"] == 13.0 and pos["scaled_in"]
    assert 100.0 < pos["avg_entry_price"] < 110.0
    asyncio.run(e._execute_add(TradeDecision(symbol="SIMX", action="BUY", fraction=0.5)))
    assert state.active_positions["SIMX"]["qty"] == 13.0
