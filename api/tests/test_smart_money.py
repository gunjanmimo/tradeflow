"""
Smart money: verdicts, tradability, routing and the day-trade strategy.

  * a sizeable insider purchase is a BUY for SMART_MONEY_LOOKBACK_DAYS; a small
    one is not; insiders selling is AVOID; one eToro holder of fifteen is HOLD
  * a BUY is traded only if it is a liquid, real US stock: not a crypto token,
    not a leveraged ETF, not a sub-$5 or thinly traded name
  * tradable BUYs route to the smart_money strategy; everything else keeps the
    default; a manual override still wins
  * the strategy enters once a day, not in the first half hour, and exits when
    the stock stops being a BUY
"""
import time
import types

import numpy as np
import pytest

from core.config import settings
from core.state import state, QuantMetrics, SentimentRecord
import engine.smart_money as sm


def _sig(source, symbol, value=0.0, breadth=0.0, details=None):
    return types.SimpleNamespace(source_name=source, symbol=symbol, value_usd=value, breadth=breadth,
                                 details=details or f"{source} {symbol} {value} {breadth}")


@pytest.fixture
def book(monkeypatch):
    b = sm.smart_money                     # conftest gives each test a fresh, isolated book
    from engine.executor import executor
    from feeds.daily_bars import daily_bars
    monkeypatch.setattr(executor, "is_connected", True)
    monkeypatch.setattr(executor, "is_mock_mode", False)
    monkeypatch.setattr(executor, "alpaca_tradable_symbols", {"AMD", "ACOG", "TECL", "PURR", "PENY", "THIN"})
    monkeypatch.setattr(executor, "alpaca_asset_names", {
        "AMD": "Advanced Micro Devices, Inc. Common Stock", "ACOG": "Alpha Cognition Inc. Common Stock",
        "TECL": "Direxion Daily Technology Bull 3x ETF", "PURR": "Hyperliquid Strategies Inc Common Stock",
        "PENY": "Penny Corp Common Stock", "THIN": "Thin Corp Common Stock"})
    series = {"AMD": (160.0, 5e9), "ACOG": (12.0, 2e6), "TECL": (90.0, 3e8), "PURR": (20.0, 4e7),
              "PENY": (2.0, 5e7), "THIN": (40.0, 1e6)}
    monkeypatch.setattr(daily_bars, "series", {s: (np.arange(30), np.full(30, p)) for s, (p, _) in series.items()})
    monkeypatch.setattr(daily_bars, "dollar_volume", {s: dv for s, (_, dv) in series.items()})
    return b


def test_verdicts(book):
    now = time.time()
    book.ingest([_sig("SEC Form 4", "AMD", 250_000), _sig("SEC Form 4", "ACOG", 24_000),
                 _sig("SEC Form 4", "CRWD", -5_162_355), _sig("eToro", "INTC", breadth=1 / 15),
                 _sig("eToro", "PURR", breadth=4 / 15)], now=now)
    assert book.verdict("AMD", now)["verdict"] == "BUY"
    assert book.verdict("ACOG", now)["verdict"] == "HOLD"          # $24k: too small to count
    assert book.verdict("CRWD", now)["verdict"] == "AVOID"
    assert book.verdict("INTC", now)["verdict"] == "HOLD"          # 1 of 15 investors
    assert book.verdict("PURR", now)["verdict"] == "BUY"           # 4 of 15
    later = now + (settings.SMART_MONEY_LOOKBACK_DAYS + 0.5) * 86400
    assert book.verdict("AMD", later)["verdict"] == "HOLD"         # the purchase has aged out


def test_selling_after_buying_turns_to_avoid_and_repeats_are_not_double_counted(book):
    t = time.time()
    book.ingest([_sig("SEC Form 4", "AMD", 300_000, details="buy")], now=t)
    book.ingest([_sig("SEC Form 4", "AMD", 300_000, details="buy")], now=t + 60)   # the feed repeats it
    assert book.verdict("AMD", t + 61)["insider_bought"] == 300_000
    book.ingest([_sig("SEC Form 4", "AMD", -2_000_000, details="sell")], now=t + 3600)
    assert book.verdict("AMD", t + 3601)["verdict"] == "AVOID"


def test_the_book_survives_a_restart(book):
    book.ingest([_sig("SEC Form 4", "AMD", 250_000)])
    fresh = sm.SmartMoneyBook()
    assert fresh.verdict("AMD")["verdict"] == "BUY"


def test_tradability_filter(book):
    assert book.tradable("AMD") is None
    assert "not a tradable US stock" in book.tradable("HYPE")       # a crypto token on eToro
    assert "leveraged" in book.tradable("TECL")
    assert "price" in book.tradable("PENY")
    assert "illiquid" in book.tradable("THIN")
    assert "illiquid" in book.tradable("ACOG")
    assert book.tradable("FRVIA.PA") == "not a US stock ticker"


def test_routing(book, monkeypatch):
    from engine.strategies import registry
    book.ingest([_sig("SEC Form 4", "AMD", 250_000), _sig("SEC Form 4", "THIN", 900_000)])
    defaults = {"equity": "news_catalyst"}
    assert registry.resolve("AMD", defaults, {}).name == "smart_money"
    assert registry.resolve("THIN", defaults, {}).name == "news_catalyst"       # BUY but illiquid
    assert registry.resolve("MSFT", defaults, {}).name == "news_catalyst"
    assert registry.resolve("AMD", defaults, {"AMD": "supertrend"}).name == "supertrend"
    monkeypatch.setattr(settings, "SMART_MONEY_TRADING", False)
    assert registry.resolve("AMD", defaults, {}).name == "news_catalyst"


def _ctx(symbol="AMD", position=None):
    from engine.strategies.base import StrategyContext
    q = QuantMetrics(symbol=symbol, rsi=55.0, ema_fast=101.0, ema_slow=100.0, atr=0.5, spread=0.0005)
    return StrategyContext(symbol=symbol, price=101.5, quant=q, sentiment=SentimentRecord(stock_id=symbol),
                           consensus=None, position=position)


def test_strategy_entry_rules(book, monkeypatch):
    import core.market_hours as mh
    from engine.strategies.smart_money import SmartMoneyStrategy
    s = SmartMoneyStrategy()
    book.ingest([_sig("SEC Form 4", "AMD", 250_000)])
    monkeypatch.setattr(mh, "us_session", lambda *a, **k: mh.REGULAR)
    monkeypatch.setattr(mh, "minutes_to_close", lambda *a, **k: 380.0)          # 10 min after the open
    assert s.evaluate_entry(_ctx()).blocked_by == "too_early"
    monkeypatch.setattr(mh, "minutes_to_close", lambda *a, **k: 300.0)
    assert s.evaluate_entry(_ctx()).should_enter
    state.closed_trades.append({"symbol": "AMD", "entry_strategy": "smart_money", "opened_at": time.time()})
    try:
        assert s.evaluate_entry(_ctx()).blocked_by == "once_per_day"
    finally:
        state.closed_trades.pop()
    assert s.evaluate_entry(_ctx("MSFT")).blocked_by == "sm_not_buy"


def test_strategy_exits_when_insiders_turn_sellers(book):
    from engine.strategies.smart_money import SmartMoneyStrategy
    s = SmartMoneyStrategy()
    t = time.time()
    book.ingest([_sig("SEC Form 4", "AMD", 250_000, details="buy")], now=t - 60)
    assert not s.evaluate_exit(_ctx(position={"qty": 10})).should_close
    book.ingest([_sig("SEC Form 4", "AMD", -3_000_000, details="sell")], now=t)
    ex = s.evaluate_exit(_ctx(position={"qty": 10}))
    assert ex.should_close and "AVOID" in ex.reason
