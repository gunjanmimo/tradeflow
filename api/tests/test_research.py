"""
The research harness cannot be fooled by its own plumbing:

  * no signal looks ahead: changing bars after t never changes the signal at t
  * forward returns enter at the next open and never cross the close
  * events do not overlap, and t-stats are clustered by day
"""
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from core.costs import CostModel
from research import edge, signals
from research.data import Bars, ny_minute_of_day

NY = ZoneInfo("America/New_York")


def _session(day: int, n: int = 390, seed: int = 0, start_price: float = 100.0):
    m0 = int(datetime(2026, 9, day, 9, 30, tzinfo=NY).timestamp() // 60)
    rng = np.random.default_rng(seed)
    c = start_price * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    o = np.r_[start_price, c[:-1]]
    h = np.maximum(o, c) * 1.0003
    l = np.minimum(o, c) * 0.9997
    v = rng.integers(100, 5000, n).astype(float)
    return np.arange(m0, m0 + n), o, h, l, c, v


def _bars(symbol="TST", days=(21, 22, 23), seed=0):
    parts = [_session(d, seed=seed + i) for i, d in enumerate(days)]
    cols = [np.concatenate([p[k] for p in parts]) for k in range(6)]
    return Bars(symbol, *cols)


def test_minute_of_day_is_new_york_time():
    b = _bars()
    assert ny_minute_of_day(b.minute[:1])[0] == 570


@pytest.mark.parametrize("name", [n for n in signals.SIGNALS if n != "random"])
def test_no_signal_looks_ahead(name):
    b = _bars(seed=3)
    mkt = _bars("SPY", seed=9)
    fn = signals.SIGNALS[name]
    base = fn(b, {"market": mkt})
    cut = 450                                            # somewhere in day 2
    b2 = Bars(b.symbol, b.minute.copy(), b.o.copy(), b.h.copy(), b.l.copy(), b.c.copy(), b.v.copy())
    rng = np.random.default_rng(42)
    for arr in (b2.o, b2.h, b2.l, b2.c):
        arr[cut + 1:] *= np.exp(rng.normal(0, 0.02, len(arr) - cut - 1))
    b2.v[cut + 1:] = rng.integers(1, 10**6, len(b2.v) - cut - 1)
    after = fn(b2, {"market": mkt})
    assert np.array_equal(base[:cut + 1], after[:cut + 1]), f"{name} changed before t after editing the future"


def test_forward_returns_enter_next_open_and_stop_at_the_close():
    b = _bars(days=(21,))
    fr = edge.forward_returns(b, horizon=5)
    t = 100
    assert fr[t] == pytest.approx(np.log(b.o[t + 6] / b.o[t + 1]) * 1e4)
    near_end = 386                                       # t+1+5 falls past the last bar (389)
    assert fr[near_end] == pytest.approx(np.log(b.c[389] / b.o[387]) * 1e4)
    assert np.isnan(fr[389])                             # no next bar to enter on


def test_events_do_not_overlap():
    b = _bars(days=(21,))
    sig = np.zeros(len(b), dtype=bool)
    sig[100:110] = True                                  # ten signals in a row
    ev = edge.events_for(b, sig, horizon=5)
    assert len(ev) == 2                                  # t=100, then the next after 106


def test_clustered_t_is_smaller_when_events_share_a_day():
    rng = np.random.default_rng(0)
    day_effect = rng.normal(0, 10, 50)
    ret = np.repeat(day_effect, 20) + rng.normal(0, 1, 1000) + 0.5
    day = np.repeat(np.arange(50), 20)
    naive = ret.mean() / (ret.std(ddof=1) / np.sqrt(len(ret)))
    clustered = edge._clustered_t(ret, day)
    assert abs(clustered) < abs(naive) / 2


def test_study_runs_and_random_has_no_edge():
    bars = {"AAA": _bars("AAA", seed=1), "BBB": _bars("BBB", seed=2), "SPY": _bars("SPY", seed=3)}
    rows = edge.study(bars, ["random", "reversal_z"], [15], CostModel(2.0, 1.0))
    rnd = next(r for r in rows if r.signal == "random")
    assert rnd.events > 0
    assert abs(rnd.excess_bps) < 30
    assert rnd.net_bps == pytest.approx(rnd.gross_bps - 6.0)
