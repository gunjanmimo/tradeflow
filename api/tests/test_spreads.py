"""
The spread every gate reads: the real consolidated spread when known, else a
short-window median of IEX quotes, never one IEX snapshot, never a placeholder.
"""
import time

import pytest

from feeds.spreads import SpreadMonitor, SIP_MAX_AGE_SECONDS, IEX_WINDOW_SECONDS


def test_consolidated_spread_beats_a_thin_iex_book():
    m = SpreadMonitor()
    now = time.time()
    for i in range(10):
        m.on_quote("BE", 288.0, 310.15, now - i)             # IEX: 7.4% wide
    assert m.estimate("BE", now)[1] == "iex" and m.estimate("BE", now)[0] > 0.07
    m.sip["BE"] = (0.0007, now, 815)                          # the real market: 0.07%
    assert m.estimate("BE", now) == (0.0007, "sip")
    assert m.estimate("BE", now + SIP_MAX_AGE_SECONDS + 1)[1] != "sip"   # stale: falls back


def test_iex_median_ignores_spikes_and_needs_enough_quotes():
    m = SpreadMonitor()
    now = time.time()
    for i in range(4):
        m.on_quote("OKTA", 203.36, 203.69, now - i)
    assert m.estimate("OKTA", now) == (None, "unknown")        # too few quotes: unknown, not 1%
    m.on_quote("OKTA", 190.0, 209.0, now)                       # one 9% spike
    for i in range(6):
        m.on_quote("OKTA", 203.36, 203.69, now - 5 - i)
    spread, source = m.estimate("OKTA", now)
    assert source == "iex" and spread == pytest.approx(0.00162, abs=2e-4)
    assert m.estimate("OKTA", now + IEX_WINDOW_SECONDS + 30) == (None, "unknown")   # old quotes age out


def test_crossed_and_one_sided_quotes_are_ignored():
    m = SpreadMonitor()
    for b, a in ((0.0, 10.0), (10.0, 0.0), (10.1, 10.0)):
        m.on_quote("X", b, a)
    assert "X" not in m._iex


def test_quant_metrics_use_the_monitor_not_a_placeholder(monkeypatch):
    from core.state import state, PriceTick
    from engine.quant_matrix import quant_matrix
    import feeds.spreads as sp
    m = SpreadMonitor()
    m.sip["NEWP"] = (0.0005, time.time(), 100)
    monkeypatch.setattr(sp, "spreads", m)
    monkeypatch.setattr("engine.quant_matrix.spreads", m)
    state.latest_prices["NEWP"] = PriceTick(symbol="NEWP", price=50.0, bid=40.0, ask=60.0, volume=0,
                                            timestamp=time.time())
    try:
        q = quant_matrix.evaluate_symbol("NEWP")                # a brand-new symbol, no history yet
        assert q.spread == pytest.approx(0.0005)                # not the old 1% placeholder, not 40% IEX
    finally:
        state.latest_prices.pop("NEWP", None)
        state.quant_metrics.pop("NEWP", None)
