"""
News scoring honours the market on/off switch.

Regression: switching crypto off only gated entries. The news poller still
fetched BTC/USD headlines and scored each one through Laya (~250ms apiece), so a
"paused" market kept producing [SENTIMENT] lines and burning scoring time.
"""
import asyncio
import types

import pytest

import feeds.news_feed as nf
from core.market_filter import market_filter
from core.state import state


def _item(nid, symbols):
    return types.SimpleNamespace(id=nid, headline="Bitcoin surges", summary="", source="t",
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
    monkeypatch.setattr(state, "watchlist", {"BTC/USD", "AAPL"}, raising=False)
    monkeypatch.setattr(market_filter, "disabled_markets", set())
    monkeypatch.setattr(market_filter, "disabled_symbols", set())
    monkeypatch.setattr(state, "seen_news_ids", set(), raising=False)
    state.active_positions.pop("BTC/USD", None)
    feed = nf.NewsFeedManager()
    feed.scored = scored
    yield feed
    state.active_positions.pop("BTC/USD", None)


def test_crypto_scored_when_enabled(feed):
    asyncio.run(feed._ingest_item(_item("1", ["BTCUSD"])))
    assert feed.scored == ["BTC/USD"]


def test_crypto_not_scored_when_market_off(feed):
    market_filter.disabled_markets.add("crypto")
    assert feed._crypto_symbols() == []
    asyncio.run(feed._ingest_item(_item("2", ["BTCUSD"])))
    assert feed.scored == []
    assert not state.is_news_seen("2"), "left unseen so it is scored if the market is switched back on"


def test_disabled_symbol_is_skipped_but_market_neighbours_are_not(feed):
    market_filter.disabled_symbols.add("BTC/USD")
    assert "BTC/USD" not in feed._tracked() and "AAPL" in feed._tracked()


def test_held_crypto_still_scored_when_market_off(feed):
    market_filter.disabled_markets.add("crypto")
    state.active_positions["BTC/USD"] = {"symbol": "BTC/USD", "qty": 1.0}
    asyncio.run(feed._ingest_item(_item("3", ["BTCUSD"])))
    assert feed.scored == ["BTC/USD"]
