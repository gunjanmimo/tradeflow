"""
The sentinel's whole exit policy: stop, target, forced exit, strategy exit.

  * the stop closes on the first print at or through it (no confirmation delay)
  * the target closes at or above it
  * a strategy exit closes; nothing else does (no harvest, trims, adds or
    "soft" reversal/news exits)
"""
import asyncio
import time

import pytest

from core.state import state, QuantMetrics
from engine.sentinel_agent import PositionSentinelBot
from engine.strategies.base import ExitDecision
import core.market_hours as mh

SYM = "SNTL"
ENTRY = 100.0


@pytest.fixture
def world(monkeypatch):
    sent = []
    strat = {"close": False}

    class _Exec:
        async def execute_decision(self, decision):
            sent.append(decision)

    class _Strat:
        name = "stub"

        def evaluate_exit(self, ctx):
            return ExitDecision(strat["close"], 0.0, 0.0, "policy says flat")

    import engine.executor as executor_mod
    from engine.strategies import registry
    monkeypatch.setattr(executor_mod, "executor", _Exec())
    monkeypatch.setattr(registry, "for_position", lambda *a, **k: _Strat())
    monkeypatch.setattr(mh, "us_session", lambda *a, **k: mh.REGULAR)
    monkeypatch.setattr(mh, "minutes_to_close", lambda *a, **k: 240.0)
    monkeypatch.setitem(state.quant_metrics, SYM, QuantMetrics(symbol=SYM, rsi=50.0, atr=1.0))
    state.price_moved_at[SYM] = time.time()
    pos = {"symbol": SYM, "qty": 10.0, "avg_entry_price": ENTRY, "current_price": ENTRY,
           "stop_loss": 99.0, "take_profit": 102.0, "opened_at": time.time() - 600,
           "entry_strategy": "stub"}
    state.active_positions[SYM] = pos
    bot = PositionSentinelBot(SYM, pos)
    yield bot, sent, strat
    state.active_positions.pop(SYM, None)
    state.price_moved_at.pop(SYM, None)


async def _tick(bot, p):
    state.price_moved_at[SYM] = time.time()
    await bot.on_tick(p)
    await asyncio.sleep(0)


def _run(bot, *prices):
    async def go():
        for p in prices:
            await _tick(bot, p)
    asyncio.run(go())


def test_holds_between_stop_and_target(world):
    bot, sent, _ = world
    _run(bot, 100.5, 99.5, 101.9, 99.1)
    assert sent == [] and bot.action == "HOLD"


def test_one_print_at_the_stop_closes_at_once(world):
    bot, sent, _ = world
    _run(bot, 99.0)
    assert len(sent) == 1 and sent[0].action == "CLOSE" and "Stop-loss" in sent[0].reason


def test_target_closes(world):
    bot, sent, _ = world
    _run(bot, 102.0)
    assert len(sent) == 1 and "Take-profit" in sent[0].reason


def test_strategy_exit_closes(world):
    bot, sent, strat = world
    strat["close"] = True
    _run(bot, 100.4)
    assert len(sent) == 1 and "policy says flat" in sent[0].reason


def test_no_partial_sells_or_adds_on_a_winner(world):
    """The old harvest sold half on any uptick; now a winner just runs to its exit."""
    bot, sent, _ = world
    _run(bot, 100.2, 100.8, 101.5)
    assert sent == []
    assert state.active_positions[SYM]["qty"] == 10.0


def test_dust_is_closed_at_once(world):
    """A leftover fraction (here 0.0079 shares, ~$0.80) is closed, not managed."""
    bot, sent, _ = world
    state.active_positions[SYM]["qty"] = 0.0079
    _run(bot, 100.1)
    assert len(sent) == 1 and "dust position" in sent[0].reason and sent[0].forced


def test_an_inherited_position_is_not_judged_by_a_policy_that_never_opened_it(world):
    bot, sent, strat = world
    strat["close"] = True                              # the policy would say "flat"
    state.active_positions[SYM].pop("entry_strategy")
    _run(bot, 100.4)
    assert sent == [] and state.active_positions[SYM]["inherited"]
    _run(bot, 98.9)                                     # the stop still protects it
    assert len(sent) == 1 and "Stop-loss" in sent[0].reason


def test_dust_waits_for_the_regular_session(world, monkeypatch):
    bot, sent, _ = world
    monkeypatch.setattr(mh, "us_session", lambda *a, **k: mh.PRE)
    monkeypatch.setattr(mh, "minutes_to_close", lambda *a, **k: None)
    state.active_positions[SYM]["qty"] = 0.0079
    _run(bot, 100.1)
    assert sent == []
