"""
The backtester replays the live exit rules with honest fills:

  * a round trip's P&L is the price move minus exactly the costs it paid
  * a signal at a bar's close fills at the next bar's open
  * the stop wins when a bar touches both stop and target; gaps fill at the open
  * a strategy exit fills at the next bar's open; nothing is held past the day
"""
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from backtest import sim
from core.costs import CostModel
from engine.strategies.base import EntryDecision, ExitDecision

NY = ZoneInfo("America/New_York")
COSTS = CostModel(half_spread_bps=2.0, slippage_bps=1.0)


def _minute(h, m, day=28):
    return int(datetime(2026, 9, day, h, m, tzinfo=NY).timestamp() // 60)


def _tape(ohlc, start=(10, 0)):
    m0 = _minute(*start)
    bars = [(m0 + i, o, h, l, c, 1000.0) for i, (o, h, l, c) in enumerate(ohlc)]
    return sim.Tape("AAPL", bars, COSTS)


def _flat(n, p=100.0):
    return [(p, p, p, p)] * n


class Once:
    """Enters on bar `at`; exits at the close of bar `exit_at` (or never)."""
    name = "test_strategy"

    def __init__(self, at, exit_at=None):
        self.at, self.exit_at = at, exit_at

    def evaluate_entry(self, ctx):
        return EntryDecision(ctx._bar_index == self.at, 0.9, "t")

    def evaluate_exit(self, ctx):
        return ExitDecision(ctx._bar_index == self.exit_at, 0.0, 0.0, "thesis gone")


def test_round_trip_pnl_is_move_minus_costs():
    tape = _tape(_flat(80))
    b = sim.Book(tape, "x", 70, 1000.0)
    t = b.close(75, 101.0, "test")
    buy, sell = COSTS.buy_fill(100.0), COSTS.sell_fill(101.0)
    assert t.pnl == pytest.approx(b.qty * (sell - buy), abs=1e-6)
    assert t.costs == pytest.approx(b.qty * (buy - 100.0) + b.qty * (101.0 - sell), abs=1e-6)


def test_flat_market_loses_exactly_the_costs():
    tape = _tape(_flat(80))
    b = sim.Book(tape, "x", 70, 1000.0)
    t = b.close(75, 100.0, "test")
    assert t.pnl == pytest.approx(-t.costs, abs=1e-6)


def test_entry_fills_at_the_next_bar_open():
    bars = _flat(70) + [(101.0, 101.0, 101.0, 101.0)] * 20
    trades = sim.run(_tape(bars), Once(at=69, exit_at=75), 1000.0)
    assert len(trades) == 1
    assert trades[0].entry_price == pytest.approx(COSTS.buy_fill(101.0))   # bar 70's open


def test_stop_wins_when_a_bar_touches_both():
    bars = _flat(71) + [(100.0, 150.0, 50.0, 100.0)] + _flat(10)
    t = sim.run(_tape(bars), Once(at=69), 1000.0)[0]
    assert t.exit_reason == "stop" and t.pnl < 0


def test_a_gap_through_the_stop_fills_at_the_open():
    bars = _flat(71) + [(90.0, 90.0, 89.0, 89.5)] + _flat(10, 89.5)
    t = sim.run(_tape(bars), Once(at=69), 1000.0)[0]
    assert t.exit_reason == "stop (gap)"
    assert t.exit_price == pytest.approx(COSTS.sell_fill(90.0))


def test_target_is_a_limit_with_no_spread():
    bars = _flat(71) + [(100.0, 150.0, 100.0, 100.0)] + _flat(10)
    tape = _tape(bars)
    t = sim.run(tape, Once(at=69), 1000.0)[0]
    assert t.exit_reason == "target"
    b = sim.Book(tape, "x", 70, 1000.0)
    assert t.exit_price == pytest.approx(b.pos["take_profit"])


def test_strategy_exit_fills_at_the_next_open():
    bars = _flat(76) + [(102.0, 102.0, 102.0, 102.0)] * 5
    t = sim.run(_tape(bars), Once(at=69, exit_at=75), 1000.0)[0]
    assert t.exit_reason.startswith("strategy")
    assert t.exit_price == pytest.approx(COSTS.sell_fill(102.0))     # bar 76's open


def test_nothing_is_held_through_the_close():
    tape = _tape(_flat(120), start=(14, 0))                          # bar 119 = 15:59
    trades = sim.run(tape, Once(at=85), 1000.0)                      # signal 15:25
    assert len(trades) == 1 and trades[0].exit_reason == "end of day"
    # flattened inside the 10-minute window, well before the last bar
    assert tape.mins_to_close[list(tape.minute).index(trades[0].exit_minute)] < 10


def test_no_entries_in_the_last_half_hour():
    tape = _tape(_flat(120), start=(14, 0))
    assert sim.run(tape, Once(at=110), 1000.0) == []                 # 15:50: inside the cutoff


# --- profit-taking (engine/profit_manager.py), as the live executor runs it ---

W = 20                                   # warm-up bars before the entry


def _scale_settings(monkeypatch, at_r=1.0, fraction=0.5, trail_r=1.0):
    from core.config import settings
    monkeypatch.setattr(settings, "PROFIT_TAKING_ENABLED", True)
    monkeypatch.setattr(settings, "SCALE_OUT_AT_R", at_r)
    monkeypatch.setattr(settings, "SCALE_OUT_FRACTION", fraction)
    monkeypatch.setattr(settings, "TRAIL_DISTANCE_R", trail_r)
    return settings


def _book_with_stop(tape, i, stop, tp, notional=1000.0):
    b = sim.Book(tape, "x", i, notional)
    b.pos.update(stop_loss=stop, initial_stop=stop, take_profit=tp)
    return b


def test_scale_out_sells_half_at_plus_1r_then_the_rest_stops_at_breakeven(monkeypatch):
    s = _scale_settings(monkeypatch)
    from engine import profit_manager
    # bar 0 entry at 100; bar 1 reaches +1R (R = entry - 99); bar 2 falls through breakeven.
    tape = _tape(_flat(W) + [(100, 100, 100, 100), (100, 101.2, 100, 101), (100.5, 100.5, 99.5, 99.6), *_flat(5)])
    b = _book_with_stop(tape, W, 99.0, 110.0)
    entry = b.entry
    r = entry - 99.0
    qty = b.qty
    assert sim.step(tape, W + 1, b, Once(-1)) is None
    assert b.pos["scaled_out"] and b.qty == qty - qty // 2
    be = profit_manager.breakeven(entry)
    assert b.pos["stop_loss"] == pytest.approx(round(be, 4))
    t = sim.step(tape, W + 2, b, Once(-1))
    assert t is not None and t.exit_reason == "stop"
    sold = qty // 2
    level = entry + s.SCALE_OUT_AT_R * r
    expect = sold * (COSTS.sell_fill(level) - entry) + (qty - sold) * (COSTS.sell_fill(round(be, 4)) - entry)
    assert t.pnl == pytest.approx(expect, abs=1e-3)
    assert t.qty == qty                      # the trade reports the whole position


def test_a_bar_that_touches_the_stop_and_plus_1r_is_a_loss(monkeypatch):
    _scale_settings(monkeypatch)
    tape = _tape(_flat(W) + [(100, 100, 100, 100), (100, 101.5, 98.5, 100), *_flat(5)])
    b = _book_with_stop(tape, W, 99.0, 110.0)
    t = sim.step(tape, W + 1, b, Once(-1))
    assert t is not None and t.exit_reason == "stop" and not b.pos.get("scaled_out")


def test_trailing_stop_follows_the_high_after_the_scale_out(monkeypatch):
    _scale_settings(monkeypatch)
    tape = _tape(_flat(W) + [(100, 100, 100, 100), (100, 101.2, 100, 101), (101, 103, 101, 103), *_flat(5, 103)])
    b = _book_with_stop(tape, W, 99.0, 110.0)
    r = b.entry - 99.0
    sim.step(tape, W + 1, b, Once(-1))
    sim.step(tape, W + 2, b, Once(-1))
    assert b.pos["stop_loss"] == pytest.approx(round(103 - r, 4))   # 1R behind the new high


def test_no_scale_out_when_it_would_leave_no_share(monkeypatch):
    _scale_settings(monkeypatch)
    tape = _tape(_flat(W) + [(100, 100, 100, 100), (100, 101.2, 100, 101), *_flat(5, 101)])
    b = _book_with_stop(tape, W, 99.0, 110.0, notional=150.0)      # one share
    sim.step(tape, W + 1, b, Once(-1))
    assert b.qty == 1 and b.pos["scaled_out"] and b.realized == 0.0


def test_profit_taking_off_leaves_the_stop_alone(monkeypatch):
    s = _scale_settings(monkeypatch)
    monkeypatch.setattr(s, "PROFIT_TAKING_ENABLED", False)
    tape = _tape(_flat(W) + [(100, 100, 100, 100), (100, 101.2, 100, 101), *_flat(5, 101)])
    b = _book_with_stop(tape, W, 99.0, 110.0)
    sim.step(tape, W + 1, b, Once(-1))
    assert b.pos["stop_loss"] == 99.0 and not b.pos.get("scaled_out")
