"""
Sentinel soft-exit behaviour on losing positions.

Regression: with no news, SentimentRecord's 0.5 neg_prob placeholder plus a small
dip and one down tick added up to the 0.65 "strong sell" bar, closing positions
well before their stop and ignoring the stock minimum hold.
"""
import asyncio
import time

import pytest

from core.config import settings
from core.state import state, QuantMetrics, SentimentRecord
from engine.sentinel_agent import PositionSentinelBot
from engine.strategies.base import ExitDecision

SYM = "TEST"
ENTRY = 100.0


@pytest.fixture
def closes(monkeypatch):
    """Positions/quant for one symbol; returns the list of CLOSE decisions sent."""
    sent = []

    class _Exec:
        async def execute_decision(self, decision):
            if decision.action == "CLOSE":  # profit-harvest SELLs are not exits
                sent.append(decision)

    class _NeverExit:
        name = "never_exit"

        def evaluate_exit(self, ctx):
            return ExitDecision(False, 0.0, 0.0, "holding")

    import engine.executor as executor_mod
    from engine.strategies import registry
    monkeypatch.setattr(executor_mod, "executor", _Exec())
    # Isolate the sentinel's own exit logic from the strategy's.
    monkeypatch.setattr(registry, "for_position", lambda *a, **k: _NeverExit())
    monkeypatch.setattr(state, "fresh_analysis", lambda s: None)
    monkeypatch.setitem(state.quant_metrics, SYM, QuantMetrics(
        symbol=SYM, rsi=45.0, ema_fast=99.0, ema_slow=100.0, atr=1.0))
    yield sent
    state.active_positions.pop(SYM, None)


def _bot(opened_minutes_ago: float) -> PositionSentinelBot:
    pos = {"symbol": SYM, "qty": 10.0, "avg_entry_price": ENTRY, "current_price": ENTRY,
           "stop_loss": 95.0, "take_profit": 110.0,
           "opened_at": time.time() - opened_minutes_ago * 60}
    state.active_positions[SYM] = pos
    return PositionSentinelBot(SYM, pos)


async def _dip(bot: PositionSentinelBot):
    """Run up to 100.6 (below the trail trigger), then fade to a -2% loss finishing on a sharp down tick."""
    for p in (100.6, 99.5, 98.7, 98.0):
        await bot._on_tick(p)
    await asyncio.sleep(0)


def _sentiment(monkeypatch, **kw):
    rec = SentimentRecord(stock_id=SYM, **kw)
    monkeypatch.setattr(state, "get_sentiment", lambda s: rec)


def test_newsless_dip_above_stop_is_held(closes, monkeypatch):
    _sentiment(monkeypatch, headline="No news yet")
    bot = _bot(opened_minutes_ago=120)
    asyncio.run(_dip(bot))
    assert closes == []
    assert bot.status == "WATCHING"


def test_stop_loss_still_closes(closes, monkeypatch):
    _sentiment(monkeypatch, headline="No news yet")
    bot = _bot(opened_minutes_ago=0)

    async def run():
        await bot._on_tick(94.9)
        await asyncio.sleep(0)
        # Still below the stop once the confirmation window has run out.
        state.active_positions[SYM]["stop_breached_at"] -= settings.STOP_CONFIRM_SECONDS
        await bot._on_tick(94.8)
        await asyncio.sleep(0)
    asyncio.run(run())
    assert len(closes) == 1
    assert "Stop-loss" in closes[0].reason


def test_one_print_through_the_stop_is_not_a_stop_out(closes, monkeypatch):
    _sentiment(monkeypatch, headline="No news yet")
    bot = _bot(opened_minutes_ago=0)

    async def run():
        await bot._on_tick(94.9)     # a wick through the 95 stop...
        await bot._on_tick(95.4)     # ...that recovers
        await asyncio.sleep(0)
    asyncio.run(run())
    assert closes == []
    assert bot.close_prob < 1.0
    assert "stop_breached_at" not in state.active_positions[SYM]


def test_disaster_stop_closes_at_once(closes, monkeypatch):
    _sentiment(monkeypatch, headline="No news yet")
    bot = _bot(opened_minutes_ago=0)

    async def run():
        await bot._on_tick(92.4)     # stop 95, 5 away: disaster stop 92.5
        await asyncio.sleep(0)
    asyncio.run(run())
    assert len(closes) == 1
    assert "disaster" in closes[0].reason


def test_bearish_news_alone_does_not_close_a_loser(closes, monkeypatch):
    from engine import loss_recovery
    monkeypatch.setattr(loss_recovery, "price_confirms_weakness", lambda s: False)
    _sentiment(monkeypatch, pos_prob=0.2, neg_prob=0.8, n_headlines=4,
               agreement=1.0, is_tradeable=True)
    bot = _bot(opened_minutes_ago=settings.STOCK_SCORE_MIN_HOLD_MINUTES + 5)

    async def run():
        for p in (99.5, 98.7, 98.0):   # a loser from the first tick
            await bot._on_tick(p)
        await asyncio.sleep(0)
    asyncio.run(run())
    assert closes == []


def test_bearish_news_closes_a_loser_when_the_trend_agrees(closes, monkeypatch):
    from engine import loss_recovery
    monkeypatch.setattr(loss_recovery, "price_confirms_weakness", lambda s: True)
    _sentiment(monkeypatch, pos_prob=0.2, neg_prob=0.8, n_headlines=4,
               agreement=1.0, is_tradeable=True)
    bot = _bot(opened_minutes_ago=settings.STOCK_SCORE_MIN_HOLD_MINUTES + 5)

    async def run():
        for p in (99.5, 98.7, 98.0):
            await bot._on_tick(p)
        await asyncio.sleep(0)
    asyncio.run(run())
    assert closes and "Bearish" in closes[0].reason


def test_bearish_news_closes_after_min_hold(closes, monkeypatch):
    _sentiment(monkeypatch, pos_prob=0.2, neg_prob=0.8, n_headlines=4,
               agreement=1.0, is_tradeable=True)
    bot = _bot(opened_minutes_ago=settings.STOCK_SCORE_MIN_HOLD_MINUTES + 5)
    asyncio.run(_dip(bot))
    assert closes, "tradeable bearish news should still force the exit"


def test_bearish_news_waits_out_min_hold(closes, monkeypatch):
    _sentiment(monkeypatch, pos_prob=0.2, neg_prob=0.8, n_headlines=4,
               agreement=1.0, is_tradeable=True)
    bot = _bot(opened_minutes_ago=1)
    asyncio.run(_dip(bot))
    assert not any("Bearish" in d.reason or "Model exit" in d.reason for d in closes)
