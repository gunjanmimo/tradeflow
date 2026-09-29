"""Every watched symbol ends up on the live stream, whenever it joined the watchlist."""
import asyncio
import types

from core.state import state
from feeds.alpaca_stream import market_stream


def test_sync_subscribes_symbols_added_before_the_stream_existed(monkeypatch):
    sent = []

    async def send():
        sent.append(1)

    stream = types.SimpleNamespace(_handlers={"bars": {"AAPL": object()}, "quotes": {}}, _running=True,
                                   _send_subscribe_msg=send)
    monkeypatch.setattr(market_stream, "_running", True)
    monkeypatch.setattr(market_stream, "_live", True)
    monkeypatch.setattr(market_stream, "_stock_stream", stream)
    monkeypatch.setattr(market_stream, "_stock_bar_handler", object())
    monkeypatch.setattr(market_stream, "_stock_quote_handler", object())
    monkeypatch.setattr(state, "watchlist", {"AAPL", "ASML", "TSM", "BTC/USD"})
    monkeypatch.setattr(state, "active_positions", {})
    added = asyncio.run(market_stream.sync_subscriptions())
    assert added == ["ASML", "SPY", "TSM"]                     # SPY is a context symbol; the pair is not a stock
    assert set(stream._handlers["bars"]) == {"AAPL", "ASML", "SPY", "TSM"}
    assert asyncio.run(market_stream.sync_subscriptions()) == []   # nothing left to add
