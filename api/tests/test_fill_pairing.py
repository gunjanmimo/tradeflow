"""Reloaded broker fills get their P&L by FIFO pairing, never by guessing."""
import pytest

from engine.executor import pair_fills


def f(t, sym, side, qty, price):
    return {"time": t, "symbol": sym, "side": side, "qty": qty, "price": price}


def test_round_trip_and_split_exit():
    rows = pair_fills([
        f(1, "IOVA", "BUY", 504, 14.71004), f(2, "IOVA", "SELL", 504, 14.87),          # one round trip
        f(3, "SPCX", "BUY", 31, 149.18),
        f(4, "SPCX", "SELL", 26, 147.9), f(5, "SPCX", "SELL", 4, 147.85), f(6, "SPCX", "SELL", 1, 147.86),
    ])
    iova = rows[1]
    assert iova["pnl"] == pytest.approx((14.87 - 14.71004) * 504, abs=0.01) and "partial" not in iova
    s1, s2, s3 = rows[3:]
    assert s1["pnl"] == pytest.approx(-1.28 * 26, abs=0.01) and s1["partial"] is True    # 5 shares still open
    assert s2["partial"] is True and "partial" not in s3                                # the last one closes it
    assert s1["entry_price"] == pytest.approx(149.18)


def test_fifo_across_two_buys_and_unknown_history():
    rows = pair_fills([f(1, "A", "BUY", 10, 100.0), f(2, "A", "BUY", 10, 110.0), f(3, "A", "SELL", 15, 120.0),
                       f(4, "B", "SELL", 5, 50.0)])                                       # B's buy predates the reload
    assert rows[2]["entry_price"] == pytest.approx((10 * 100 + 5 * 110) / 15)
    assert rows[2]["pnl"] == pytest.approx(15 * 120 - 1550)
    assert "pnl" not in rows[3]
