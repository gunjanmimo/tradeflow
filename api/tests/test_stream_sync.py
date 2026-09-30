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


def _fake_stream(bars=(), quotes=(), trades=()):
    sent = {"sub": 0, "unsub": []}

    async def sub():
        sent["sub"] += 1

    async def unsub(channel, symbols):
        sent["unsub"].append((channel, list(symbols)))

    handlers = {"bars": dict.fromkeys(bars, object()), "quotes": dict.fromkeys(quotes, object()),
                "trades": dict.fromkeys(trades, object())}
    return types.SimpleNamespace(_handlers=handlers, _running=True, _send_subscribe_msg=sub,
                                 _send_unsubscribe_msg=unsub), sent


def _live(monkeypatch, stream):
    monkeypatch.setattr(market_stream, "_running", True)
    monkeypatch.setattr(market_stream, "_live", True)
    monkeypatch.setattr(market_stream, "_stock_stream", stream)
    monkeypatch.setattr(market_stream, "_stock_bar_handler", object())
    monkeypatch.setattr(market_stream, "_stock_quote_handler", object())
    monkeypatch.setattr(market_stream, "_stock_trade_handler", object())
    monkeypatch.setattr(market_stream, "not_streamed", set())


def test_plan_keeps_held_then_context_then_watchlist_under_the_cap():
    from feeds.alpaca_stream import stream_plan
    plan = stream_plan(["ZZZ"], ("SPY",), ["AAA", "SPY", "BTC/USD", "BBB", "CCC"], 4)
    assert plan == ["ZZZ", "SPY", "AAA", "BBB"]           # no duplicate SPY, no pair, CCC over the cap


def test_stream_never_subscribes_past_the_data_plan_limit(monkeypatch):
    """33 symbols on a 30-symbol plan got the whole subscribe rejected: no symbol had a price."""
    from core.config import settings
    import scout.service as sc
    stream, _ = _fake_stream()
    _live(monkeypatch, stream)
    watch = {f"W{chr(65 + i // 26)}{chr(65 + i % 26)}" for i in range(40)}   # WAA..WBN
    monkeypatch.setattr(state, "watchlist", watch | {"HELD"})
    monkeypatch.setattr(state, "active_positions", {"HELD": {}})
    monkeypatch.setattr(settings, "STREAM_MAX_SYMBOLS", 30)
    # The scout ranks WBN first: it must beat the alphabetically earlier symbols.
    monkeypatch.setattr(sc.scout, "picks", {"WBN": {"rank": 1}, "WBM": {"rank": 2}})
    asyncio.run(market_stream.sync_subscriptions())
    bars = set(stream._handlers["bars"])
    assert len(bars) == 30 and set(stream._handlers["quotes"]) == bars
    assert {"HELD", "SPY", "WBN", "WBM"} <= bars
    assert set(stream._handlers["trades"]) == {"HELD"}      # trade prints for positions only
    assert len(market_stream.not_streamed) == 41 - 29 and "HELD" not in market_stream.not_streamed


def test_symbols_leaving_the_watchlist_are_unsubscribed(monkeypatch):
    """Nothing was ever unsubscribed, so a long session crept past the limit."""
    stream, sent = _fake_stream(bars=("AAPL", "OLD", "SPY"), quotes=("AAPL", "OLD", "SPY"),
                                trades=("SOLD",))
    _live(monkeypatch, stream)
    monkeypatch.setattr(state, "watchlist", {"AAPL", "NEW"})
    monkeypatch.setattr(state, "active_positions", {})
    added = asyncio.run(market_stream.sync_subscriptions())
    assert added == ["NEW"]
    assert set(stream._handlers["bars"]) == set(stream._handlers["quotes"]) == {"AAPL", "NEW", "SPY"}
    assert stream._handlers["trades"] == {}
    assert sorted(sent["unsub"]) == [("bars", ["OLD"]), ("quotes", ["OLD"]), ("trades", ["SOLD"])]
    assert sent["sub"] == 1


def test_manager_says_why_a_capped_symbol_has_no_price(monkeypatch):
    stream, _ = _fake_stream()
    _live(monkeypatch, stream)
    monkeypatch.setattr(market_stream, "not_streamed", {"CCC"})
    from engine.portfolio_manager import portfolio_manager
    import engine.portfolio_manager as pm
    from collections import Counter
    monkeypatch.setattr(state, "watchlist", {"CCC"})
    monkeypatch.setattr(state, "active_positions", {})
    monkeypatch.setattr(state, "latest_prices", {})
    monkeypatch.setattr("core.market_hours.us_session", lambda *a: "regular")
    monkeypatch.setattr("core.market_hours.minutes_to_close", lambda *a: 200.0)
    blocked = Counter()
    portfolio_manager._gather(0.0, blocked)
    assert portfolio_manager.watch[0]["status"] == "not streamed (data plan full)"
