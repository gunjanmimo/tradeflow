"""
Trend reading on 1-minute bars, and the minute-bar store the trend reads.
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
