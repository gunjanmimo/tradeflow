"""
Backtester bookkeeping: a round trip's P&L is the price move minus exactly the
spread it paid.
"""
import pytest

from backtest import sim


def _tape(symbol, closes, costs=sim.Costs()):
    bars = [(29_000_000 + i, c, c, c, c, 1000.0) for i, c in enumerate(closes)]
    return sim.Tape(symbol, bars, costs)


def test_round_trip_pnl_is_move_minus_costs():
    tape = _tape("AAPL", [100.0] * 80)
    b = sim.Book(tape, "x", 70, 100.0, 1000.0)
    qty = b.qty
    t = b.close(75, 101.0, "test")
    hs, fee = tape.hs, tape.fee
    buy, sell = 100.0 * (1 + hs), 101.0 * (1 - hs)
    assert t.pnl == pytest.approx(qty * (sell - buy) - qty * buy * fee - qty * sell * fee, abs=1e-3)
    assert t.costs == pytest.approx(qty * 100 * hs + qty * buy * fee + qty * 101 * hs + qty * sell * fee, abs=1e-3)


def test_flat_market_loses_exactly_the_costs():
    tape = _tape("NVDA", [200.0] * 80)
    b = sim.Book(tape, "x", 70, 200.0, 1000.0)
    t = b.close(75, 200.0, "test")
    assert t.pnl == pytest.approx(-t.costs, abs=1e-6)

