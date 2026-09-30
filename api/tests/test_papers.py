"""
The published-strategy replays (research/papers.py) cannot see the future and
fill where the rules say:

  * the opening-range inputs (relative volume, ATR, range high) use only earlier
    sessions and the first five minutes
  * a buy-stop fills at the range high (or the open on a gap), a live-style
    entry at the next bar's open, the placebo at 09:35; the stop and the close
    exit at the right prices
  * the noise-area trades of a session do not change when later sessions do
"""
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from core.costs import CostModel
from research import papers
from research.data import Bars

NY = ZoneInfo("America/New_York")


def _days(n, start=date(2026, 6, 1)):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _bars(symbol="TST", n_days=20, seed=0, price=100.0):
    rng = np.random.default_rng(seed)
    cols = [[] for _ in range(6)]
    for d in _days(n_days):
        m0 = int(datetime(d.year, d.month, d.day, 9, 30, tzinfo=NY).timestamp() // 60)
        c = price * np.exp(np.cumsum(rng.normal(0, 0.002, 390)))
        o = np.r_[price, c[:-1]]
        for k, x in enumerate((np.arange(m0, m0 + 390), o, np.maximum(o, c) * 1.001,
                               np.minimum(o, c) * 0.999, c, rng.integers(100, 5000, 390).astype(float))):
            cols[k].append(x)
        price = float(c[-1])
    return Bars(symbol, *[np.concatenate(c) for c in cols])


def _copy(b):
    return Bars(b.symbol, b.minute.copy(), b.o.copy(), b.h.copy(), b.l.copy(), b.c.copy(), b.v.copy())


def test_opening_range_inputs_use_only_the_past():
    b = _bars(seed=1)
    base = {c["day"]: c for c in papers.orb_candidates(b)}
    day = sorted(base)[2]
    b2 = _copy(b)
    days = b2.days()
    later_today = (days == day) & (np.arange(len(days)) >= np.flatnonzero(days == day)[0] + 5)
    b2.h[later_today] *= 1.5
    b2.v[later_today] *= 100
    b2.v[days > day] *= 100
    b2.h[days > day] *= 2
    got = {c["day"]: c for c in papers.orb_candidates(b2)}
    for k in ("rvol", "atr", "or_high", "up", "price"):
        assert got[day][k] == base[day][k]


def _cand(o, h, l, c, atr=10.0):
    grid = {k: np.full(papers.SESSION, np.nan) for k in "ohlcv"}
    for k, arr in zip("ohlc", (o, h, l, c)):
        grid[k][:len(arr)] = arr
    grid["v"] = np.ones(papers.SESSION)
    s = papers.Session(1, grid["o"], grid["h"], grid["l"], grid["c"], grid["v"], last=len(o) - 1)
    return {"symbol": "TST", "day": 1, "session": s, "atr": atr, "rvol": 3.0, "up": True,
            "or_high": float(np.max(h[:5])), "price": float(c[4])}


# Five opening bars (range high 101), a breakout, then a slide to the close.
O = [100, 100, 100, 100, 100, 100.5, 101.5, 101.2, 100.8]
H = [101, 100.5, 100.5, 100.5, 100.5, 101.6, 101.8, 101.3, 100.9]
L = [99.5, 99.8, 99.8, 99.8, 99.8, 100.4, 101.1, 100.7, 100.2]
C = [100.2, 100.2, 100.2, 100.2, 100.3, 101.5, 101.2, 100.8, 100.4]


def test_paper_entry_is_a_buy_stop_at_the_range_high_and_holds_to_the_close():
    t = papers.orb_trade(_cand(O, H, L, C), stop_atr=0.10)          # stop 1.00 below the trigger
    assert (t.entry, t.entry_min, t.risk) == (101.0, papers.OPEN_MIN + 5, pytest.approx(1.0))
    assert (t.exit, t.reason) == (100.4, "close")


def test_live_entry_buys_the_next_open_after_a_close_above_the_range():
    t = papers.orb_trade(_cand(O, H, L, C), stop_atr=0.10, entry_mode="close")
    assert (t.entry, t.entry_min) == (101.5, papers.OPEN_MIN + 6)
    assert (t.exit, t.reason) == (100.5, "stop")                     # 101.5 - 1.0, touched on bar 8


def test_placebo_buys_the_0935_open_without_a_breakout():
    t = papers.orb_trade(_cand(O, H, L, C), stop_atr=0.10, entry_mode="placebo")
    assert (t.entry, t.entry_min) == (100.5, papers.OPEN_MIN + 5)


def test_a_gap_through_the_stop_fills_at_the_open():
    o = O[:7] + [99.0, 99.0]
    h = H[:7] + [99.2, 99.2]
    l = L[:7] + [98.8, 98.8]
    c = C[:7] + [99.0, 99.0]
    t = papers.orb_trade(_cand(o, h, l, c), stop_atr=0.10)
    assert (t.exit, t.reason) == (99.0, "stop (gap)")


def test_noise_area_trades_do_not_depend_on_later_sessions():
    b = _bars(seed=5, n_days=24)
    full = papers.noise_area(b)
    cut_day = np.unique(b.days())[-4]
    keep = b.days() < cut_day
    short = Bars(b.symbol, *(x[keep] for x in (b.minute, b.o, b.h, b.l, b.c, b.v)))
    before = [(t.day, t.entry_min, t.exit_min, t.entry, t.exit) for t in full if t.day < cut_day]
    assert before == [(t.day, t.entry_min, t.exit_min, t.entry, t.exit) for t in papers.noise_area(short)]


def test_row_subtracts_the_round_trip_and_splits_the_holdout():
    trades = [papers.PaperTrade("A", d, 575, 600, 100.0, 100.0 * np.exp(x / 1e4))
              for d, x in [(1, 10.0), (2, 10.0), (3, 10.0), (4, 30.0), (5, 30.0), (6, 30.0)]]
    r = papers.row("x", trades, CostModel(half_spread_bps=2.0, slippage_bps=1.0), holdout_from=4)
    assert r["gross"] == pytest.approx(20.0) and r["net"] == pytest.approx(14.0)
    assert r["first"][0] == pytest.approx(4.0) and r["holdout"][0] == pytest.approx(24.0)
