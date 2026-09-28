"""
Live pricing of open positions.

Regressions:
  * the pre-market poller fed a days-old, ~10%-wide closing quote in as the live
    price, so held positions froze on it (MU sat at $1,074.30 while the broker
    marked it below its stop);
  * held stocks were priced only from minute bars, so their stop and target
    were checked at most once a minute.
"""
import asyncio
import time
import types

import pytest

from core.config import settings
from core.state import state
from feeds.alpaca_stream import usable_quote, market_stream
from engine.executor import AlpacaExecutor

NOW = 1_800_000_000.0


def test_live_tight_quote_is_used():
    assert usable_quote(100.0, 100.1, NOW - 2, NOW)


def test_previous_session_quote_is_rejected():
    # MU on Monday pre-market: Friday's 20:00 closing quote.
    assert not usable_quote(1019.35, 1129.24, NOW - 3 * 86400, NOW)


def test_fresh_but_wide_quote_is_rejected():
    assert not usable_quote(1019.35, 1129.24, NOW - 1, NOW)


def test_one_sided_quote_is_rejected():
    assert not usable_quote(319.05, 0.0, NOW - 1, NOW)


@pytest.fixture
def held(monkeypatch):
    ticks = []

    class _Registry:
        async def dispatch_tick(self, symbol, price):
            ticks.append((symbol, price))

    import engine.sentinel_agent as sa
    monkeypatch.setattr(sa, "sentinel_registry", _Registry())
    state.active_positions["MU"] = {"symbol": "MU", "qty": 1.0, "avg_entry_price": 1064.74,
                                    "mode": "ALPACA_PAPER"}
    state.latest_prices.pop("MU", None)
    state.price_history.pop("MU", None)
    yield ticks
    state.active_positions.pop("MU", None)
    state.latest_prices.pop("MU", None)


def test_trade_print_reaches_sentinel_without_indicator_sample(held):
    asyncio.run(market_stream.on_position_trade("MU", 1063.0))
    assert held == [("MU", 1063.0)]
    assert state.latest_prices["MU"].price == 1063.0
    assert state.active_positions["MU"]["current_price"] == 1063.0
    assert len(state.price_history.get("MU", ())) == 0


def test_trade_print_for_unheld_symbol_is_ignored(held):
    asyncio.run(market_stream.on_position_trade("AAPL", 250.0))
    assert held == []


def test_stale_position_is_priced_from_broker_mark(held):
    ex = AlpacaExecutor()
    state.update_price("MU", 1074.295)
    state.latest_prices["MU"].timestamp -= settings.POSITION_MAX_PRICE_AGE_SECONDS + 1
    ex.broker_marks["MU"] = 1063.0
    asyncio.run(ex.remark_stale_positions())
    assert held == [("MU", 1063.0)]

    # Our own injection does not count as the feed being alive: the next
    # changed mark is applied straight away, an unchanged one is not resent.
    ex.broker_marks["MU"] = 1062.5
    asyncio.run(ex.remark_stale_positions())
    asyncio.run(ex.remark_stale_positions())
    assert held == [("MU", 1063.0), ("MU", 1062.5)]


def test_position_with_no_tick_at_all_is_priced_from_broker_mark(held):
    # After a restart, a stock with no prints has no tick whatsoever.
    ex = AlpacaExecutor()
    ex.broker_marks["MU"] = 1063.0
    asyncio.run(ex.remark_stale_positions())
    assert held == [("MU", 1063.0)]


def test_live_feed_is_not_overridden_by_broker_mark(held):
    ex = AlpacaExecutor()
    state.update_price("MU", 1070.0)
    ex.broker_marks["MU"] = 1063.0
    asyncio.run(ex.remark_stale_positions())
    assert held == []


def test_price_moved_at_ignores_repeated_prices():
    """A quiet market re-marks the same price every second; that is not movement."""
    sym = "QUIET"
    state.latest_prices.pop(sym, None)
    state.price_moved_at.pop(sym, None)
    state.update_price(sym, 10.0, record_history=False)
    first = state.price_moved_at[sym]
    time.sleep(0.02)
    state.update_price(sym, 10.0, record_history=False)
    assert state.price_moved_at[sym] == first
    time.sleep(0.02)
    state.update_price(sym, 10.01, record_history=False)
    assert state.price_moved_at[sym] > first
    state.latest_prices.pop(sym, None)
    state.price_moved_at.pop(sym, None)
