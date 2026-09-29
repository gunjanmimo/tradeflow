"""
Forced exits: a position with no new price for 2 minutes, and a day trade about to
meet its market's close.

  * stale price: closed at once when the loss is inside the 5% limit (or in profit),
    left to the stop when it is already down 5% or more, and never on a clock that
    started before the position did.
  * end of day: stocks are flattened FLATTEN_MINUTES_BEFORE_CLOSE ahead of the
    close, using the broker's close time; new entries stop earlier.
  * a forced exit that fails is retried within FORCED_EXIT_MAX_WAIT_SECONDS.
"""
import asyncio
import time
from datetime import datetime
import zoneinfo

import pytest

from core.config import settings
from core.state import state, QuantMetrics, SentimentRecord, TradeDecision
from engine.sentinel_agent import PositionSentinelBot
from engine.strategies.base import ExitDecision
import core.market_hours as mh

NY = zoneinfo.ZoneInfo("America/New_York")
SYM = "TEST"
ENTRY = 100.0


@pytest.fixture
def closes(monkeypatch):
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
    monkeypatch.setattr(registry, "for_position", lambda *a, **k: _NeverExit())
    monkeypatch.setattr(state, "fresh_analysis", lambda s: None)
    monkeypatch.setattr(state, "get_sentiment",
                        lambda s: SentimentRecord(stock_id=SYM, headline="No news yet"))
    monkeypatch.setitem(state.quant_metrics, SYM, QuantMetrics(
        symbol=SYM, rsi=50.0, ema_fast=100.0, ema_slow=100.0, atr=1.0))
    # Not the market close and not stale, unless a test says so.
    monkeypatch.setattr(mh, "us_session", lambda *a, **k: mh.REGULAR)
    monkeypatch.setattr(mh, "minutes_to_close", lambda *a, **k: 240.0)
    state.price_moved_at[SYM] = time.time()
    yield sent
    state.active_positions.pop(SYM, None)
    state.price_moved_at.pop(SYM, None)


def _bot(watched_for_s=600.0, opened_ago_s=600.0) -> PositionSentinelBot:
    pos = {"symbol": SYM, "qty": 10.0, "avg_entry_price": ENTRY, "current_price": ENTRY,
           "stop_loss": 90.0, "take_profit": 120.0, "opened_at": time.time() - opened_ago_s}
    state.active_positions[SYM] = pos
    bot = PositionSentinelBot(SYM, pos)
    bot.assigned_at = time.time() - watched_for_s
    return bot


def _tick(bot, price):
    async def run():
        await bot._on_tick(price)
        await asyncio.sleep(0)
    asyncio.run(run())


def _quiet_for(seconds):
    state.price_moved_at[SYM] = time.time() - seconds


# ---- stale price ------------------------------------------------------------

def test_stale_price_with_small_loss_closes_at_once(closes):
    bot = _bot()
    _quiet_for(130)
    _tick(bot, 98.0)                                   # -2%, no price change for 130s
    assert len(closes) == 1
    assert "No new price" in closes[0].reason
    assert closes[0].forced is True


def test_stale_price_in_profit_also_closes(closes):
    bot = _bot()
    _quiet_for(130)
    _tick(bot, 101.0)
    assert len(closes) == 1 and closes[0].forced


def test_stale_price_below_the_limit_waits(closes):
    bot = _bot()
    _quiet_for(100)                                    # under 120s
    _tick(bot, 98.0)
    assert closes == []


def test_stale_price_with_big_loss_is_left_to_the_stop(closes):
    bot = _bot()
    _quiet_for(300)
    _tick(bot, 94.0)                                   # -6%: at/over the 5% limit, above the 90 stop
    assert closes == []


def test_loss_just_inside_limit_still_closes(closes):
    bot = _bot()
    _quiet_for(300)
    _tick(bot, 95.5)                                   # -4.5%
    assert len(closes) == 1


def test_quiet_before_the_position_existed_does_not_count(closes):
    bot = _bot(watched_for_s=5, opened_ago_s=5)        # bought 5s ago into a quiet stock
    _quiet_for(600)
    _tick(bot, 99.0)
    assert closes == []


def test_moving_price_is_never_stale(closes):
    bot = _bot()
    _quiet_for(1)
    _tick(bot, 99.0)
    assert closes == []


# ---- end of day ---------------------------------------------------------------

def test_stock_is_flattened_inside_the_window(closes, monkeypatch):
    monkeypatch.setattr(mh, "minutes_to_close", lambda *a, **k: 8.0)
    bot = _bot()
    _tick(bot, 101.0)
    assert len(closes) == 1
    assert "market closes in 8.0 min" in closes[0].reason and closes[0].forced


def test_stock_outside_the_window_is_kept(closes, monkeypatch):
    monkeypatch.setattr(mh, "minutes_to_close", lambda *a, **k: 11.0)
    bot = _bot()
    _tick(bot, 101.0)
    assert closes == []


def test_flatten_overrides_the_equity_minimum_hold(closes, monkeypatch):
    monkeypatch.setattr(mh, "minutes_to_close", lambda *a, **k: 5.0)
    bot = _bot(opened_ago_s=60)                        # 1 minute old, min hold is 30
    _tick(bot, 100.5)
    assert len(closes) == 1


def test_missed_window_closes_in_after_hours(closes, monkeypatch):
    monkeypatch.setattr(mh, "us_session", lambda *a, **k: mh.POST)
    monkeypatch.setattr(mh, "minutes_to_close", lambda *a, **k: None)
    bot = _bot()
    _tick(bot, 100.5)
    assert len(closes) == 1 and "closed for the day" in closes[0].reason


def test_flatten_can_be_switched_off(closes, monkeypatch):
    monkeypatch.setattr(mh, "minutes_to_close", lambda *a, **k: 1.0)
    monkeypatch.setattr(settings, "DAY_TRADE_FLATTEN_ENABLED", False)
    bot = _bot()
    _tick(bot, 100.5)
    assert closes == []


# ---- the close clock itself ---------------------------------------------------

def _ny(h, m, day=28):                                  # 2026-09-28 is a Monday
    return datetime(2026, 9, day, h, m, tzinfo=NY)


def test_minutes_to_close_uses_the_16_00_ny_close():
    assert mh.minutes_to_close("AAPL", _ny(15, 45)) == pytest.approx(15.0)
    assert mh.minutes_to_close("AAPL", _ny(9, 30)) == pytest.approx(390.0)


def test_adr_closes_with_the_us_market():
    """A UK company's ADR trades on a US exchange: it follows the US close."""
    assert mh.minutes_to_close("AZN", _ny(15, 30)) == pytest.approx(30.0)




def test_no_close_outside_the_regular_session():
    assert mh.minutes_to_close("AAPL", _ny(16, 30)) is None      # after hours
    assert mh.minutes_to_close("AAPL", _ny(8, 0)) is None        # pre-market
    assert mh.minutes_to_close("AAPL", _ny(12, 0, day=27)) is None   # Sunday


def test_broker_close_time_wins_on_a_half_day(monkeypatch):
    import engine.market_scheduler as ms
    monkeypatch.setattr(ms.market_scheduler, "_clock_cache", (True, "2026-09-29T09:30:00-04:00"), raising=False)
    monkeypatch.setattr(ms.market_scheduler, "_clock_at", time.time(), raising=False)
    monkeypatch.setattr(ms.market_scheduler, "_next_close_iso",
                        datetime.now(NY).replace(microsecond=0).isoformat(), raising=False)
    # broker says the market closes right now -> ~0 minutes, not "16:00 today"
    assert mh._broker_close() is not None


# ---- entry cutoff ---------------------------------------------------------------

def test_no_new_stock_entries_near_the_close(monkeypatch):
    from engine.risk_guard import risk_guard
    monkeypatch.setattr(state, "is_trading_active", True)
    monkeypatch.setattr(mh, "us_session", lambda *a, **k: mh.REGULAR)
    monkeypatch.setattr(mh, "minutes_to_close", lambda *a, **k: 25.0)
    ok, why = risk_guard.can_open_position("AAPL")
    assert not ok and "market closes in 25 min" in why




# ---- retry within 50 seconds -------------------------------------------------------

def test_forced_close_retry_is_capped_at_50_seconds():
    from engine.executor import AlpacaExecutor
    forced = TradeDecision(symbol=SYM, action="CLOSE", close=True, forced=True)
    normal = TradeDecision(symbol=SYM, action="CLOSE", close=True)
    assert AlpacaExecutor._max_close_wait(forced) == settings.FORCED_EXIT_MAX_WAIT_SECONDS == 50.0
    assert AlpacaExecutor._max_close_wait(normal) == 120.0
    # The doubling backoff (5, 10, 20, 40, 80 ...) never passes the cap.
    waits = [min(5.0 * 2 ** (n - 1), 50.0) for n in range(1, 8)]
    assert max(waits) == 50.0
    assert settings.EXTENDED_EXIT_REPRICE_SECONDS <= 50.0
