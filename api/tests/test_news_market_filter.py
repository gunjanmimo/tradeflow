"""
News scoring honours the market on/off switch.

Regression: switching a market off only gated entries. The news poller still
fetched its headlines and scored each one through Laya (~250ms apiece), so a
"paused" market kept producing [SENTIMENT] lines and burning scoring time.
"""
import asyncio
import types

import pytest

import feeds.news_feed as nf
from core.market_filter import market_filter
from core.state import state


def _item(nid, symbols):
    return types.SimpleNamespace(id=nid, headline="Apple beats estimates", summary="", source="t",
                                 created_at=None, symbols=symbols)


@pytest.fixture
def feed(monkeypatch):
    scored = []

    async def score(sym, text, persist=False):
        scored.append(sym)
        return types.SimpleNamespace(pos_prob=0.6, neg_prob=0.1)

    monkeypatch.setattr(nf.sentiment_service, "score_headline", score)
    import engine.discovery as disc
    monkeypatch.setattr(disc.discovery, "sentiment_symbols", lambda: [])
    monkeypatch.setattr(state, "watchlist", {"MSFT", "AAPL"}, raising=False)
    monkeypatch.setattr(market_filter, "disabled_markets", set())
    monkeypatch.setattr(market_filter, "disabled_symbols", set())
    monkeypatch.setattr(state, "seen_news_ids", set(), raising=False)
    state.active_positions.pop("MSFT", None)
    feed = nf.NewsFeedManager()
    feed.scored = scored
    yield feed
    state.active_positions.pop("MSFT", None)


def test_stock_scored_when_enabled(feed):
    asyncio.run(feed._ingest_item(_item("1", ["AAPL"])))
    assert feed.scored == ["AAPL"]


def test_not_scored_when_market_off(feed):
    market_filter.disabled_markets.add("stocks")
    assert feed._equity_symbols() == []
    asyncio.run(feed._ingest_item(_item("2", ["AAPL"])))
    assert feed.scored == []
    assert not state.is_news_seen("2"), "left unseen so it is scored if the market is switched back on"


def test_disabled_symbol_is_skipped_but_market_neighbours_are_not(feed):
    market_filter.disabled_symbols.add("MSFT")
    assert "MSFT" not in feed._tracked() and "AAPL" in feed._tracked()


def test_held_stock_still_scored_when_market_off(feed):
    market_filter.disabled_markets.add("stocks")
    state.active_positions["MSFT"] = {"symbol": "MSFT", "qty": 1.0}
    asyncio.run(feed._ingest_item(_item("3", ["MSFT"])))
    assert feed.scored == ["MSFT"]


def test_crypto_pairs_are_never_tracked(feed):
    """Crypto was removed: a pair mentioned by a news item is not scored."""
    asyncio.run(feed._ingest_item(_item("4", ["BTCUSD", "BTC/USD"])))
    assert feed.scored == []
