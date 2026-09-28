"""
Portfolio manager: ranks entries by conviction AND diversification fit, deploys
into idle budget, and says why budget stays idle.
"""
import asyncio
import time

import pytest

from core.config import settings
from core.state import state, TradeDecision, QuantMetrics
import engine.portfolio_manager as pm_mod
from engine.portfolio_manager import PortfolioManager


@pytest.fixture
def world(monkeypatch):
    """A quiet book with a $10k budget; returns (manager, executed symbols)."""
    from engine.executor import executor
    saved = (set(state.watchlist), dict(state.active_positions), state.is_trading_active,
             state.risk_factor, state.allocated_capital)
    executed = []

    async def fake_execute(decision):
        executed.append(decision.symbol)
        px = state.latest_prices[decision.symbol].price
        state.active_positions[decision.symbol] = {
            "symbol": decision.symbol, "qty": 5.0, "avg_entry_price": px, "current_price": px}

    monkeypatch.setattr(executor, "execute_decision", fake_execute)
    monkeypatch.setattr(pm_mod.risk_guard, "can_open_position", lambda s: (True, "ok"))
    monkeypatch.setattr(pm_mod.diversification, "_corr_to_holdings", {}, raising=False)
    import core.market_hours as mh
    monkeypatch.setattr(mh, "us_session", lambda *a, **k: mh.REGULAR)
    monkeypatch.setattr(mh, "minutes_to_close", lambda *a, **k: 240.0)

    state.watchlist.clear()
    state.active_positions.clear()
    state.is_trading_active = True
    state.risk_factor = 4
    state.allocated_capital = 10000.0
    state.last_gate_detail.clear()
    from engine.trend import board
    board.reads.clear()
    yield PortfolioManager(), executed
    board.reads.clear()
    from core.minute_bars import minute_bars
    for sym in ("AAPL", "MSFT", "JNJ", "NVDA", "KO"):
        minute_bars.drop(sym)
    state.watchlist.clear()
    state.watchlist.update(saved[0])
    state.active_positions.clear()
    state.active_positions.update(saved[1])
    state.is_trading_active, state.risk_factor, state.allocated_capital = saved[2:]
    for sym in ("AAPL", "MSFT", "JNJ", "NVDA", "KO"):
        state.latest_prices.pop(sym, None)
        state.price_moved_at.pop(sym, None)


def _bars(sym, px, step=0.0005):
    """60 one-minute bars trending into px (step > 0 up, < 0 down), with a little noise."""
    from core.minute_bars import minute_bars
    now_min = int(time.time() // 60)
    rows = []
    for i in range(60):
        c = px * (1 - step * (60 - i)) * (1.0001 if i % 2 else 0.9999)
        rows.append((now_min - 60 + i, c, c, c, c, 1.0))
    minute_bars.drop(sym)
    minute_bars.merge_history(sym, rows)


def _price(sym, px=100.0, step=0.0005):
    _bars(sym, px, step)
    state.update_price(sym, px)
    state.quant_metrics[sym] = QuantMetrics(symbol=sym, rsi=50.0, ema_fast=1.0, ema_slow=1.0, atr=1.0)
    state.watchlist.add(sym)


def _signals(monkeypatch, probs, blocked_by=None):
    def evaluate(sym, quant, sentiment):
        p = probs.get(sym)
        if p is None:
            state.last_gate_detail[sym] = {"blocked_by": blocked_by or "score"}
            return TradeDecision(symbol=sym, action="HOLD")
        return TradeDecision(symbol=sym, action="BUY", buy_prob=p)
    monkeypatch.setattr(pm_mod.decision_engine, "evaluate", evaluate)


def _hold(sym, dollars):
    state.update_price(sym, 100.0)
    state.active_positions[sym] = {"symbol": sym, "qty": dollars / 100.0,
                                   "avg_entry_price": 100.0, "current_price": 100.0}


def test_a_new_sector_outranks_a_slightly_stronger_signal_in_a_held_one(world, monkeypatch):
    mgr, executed = world
    _hold("MSFT", 1500)                                   # Information Technology already held
    _price("AAPL"); _price("JNJ")                         # AAPL is IT; JNJ is Health Care
    _signals(monkeypatch, {"AAPL": 0.80, "JNJ": 0.72})
    asyncio.run(mgr.run_cycle())
    assert executed[0] == "JNJ", "diversification should beat 0.08 of conviction"
    assert mgr.entries_total >= 1


def test_conviction_still_decides_between_equal_fits(world, monkeypatch):
    mgr, executed = world
    _price("AAPL"); _price("JNJ")                         # empty book: both fit equally
    _signals(monkeypatch, {"AAPL": 0.90, "JNJ": 0.70})
    asyncio.run(mgr.run_cycle())
    assert executed[0] == "AAPL"


def test_idle_budget_is_deployed_up_to_the_per_cycle_limit(world, monkeypatch):
    mgr, executed = world
    for s in ("AAPL", "JNJ", "KO", "NVDA"):
        _price(s)
    _signals(monkeypatch, {"AAPL": 0.8, "JNJ": 0.8, "KO": 0.8, "NVDA": 0.8})
    asyncio.run(mgr.run_cycle())
    assert len(executed) == settings.MANAGER_MAX_ENTRIES_PER_CYCLE
    asyncio.run(mgr.run_cycle())                          # the next cycle keeps deploying
    assert len(executed) > settings.MANAGER_MAX_ENTRIES_PER_CYCLE


def test_paused_trading_deploys_nothing_but_says_so(world, monkeypatch):
    mgr, executed = world
    state.is_trading_active = False
    _price("AAPL")
    _signals(monkeypatch, {"AAPL": 0.9})
    asyncio.run(mgr.run_cycle())
    assert executed == []
    assert mgr.status["state"] == "paused"
    assert "idle" in mgr.status["message"] and mgr.picks[0]["symbol"] == "AAPL"


def test_idle_budget_reports_the_blocking_gate(world, monkeypatch):
    mgr, executed = world
    _price("AAPL"); _price("JNJ")
    _signals(monkeypatch, {}, blocked_by="trend")
    asyncio.run(mgr.run_cycle())
    assert executed == []
    assert mgr.status["state"] == "watching"
    assert "Watching 2 watchlist symbols" in mgr.status["message"]
    assert "strategy: trend" in mgr.status["message"]
    assert mgr.blocked_by["strategy: trend"] == 2


def test_every_watchlist_symbol_is_watched_but_not_held_ones(world, monkeypatch):
    mgr, executed = world
    _hold("MSFT", 500)                                    # held: has its own sentinel
    _price("AAPL"); _price("JNJ")
    _signals(monkeypatch, {"AAPL": 0.9}, blocked_by="trend")
    state.watchlist.add("MSFT")
    asyncio.run(mgr.run_cycle())
    rows = {r["symbol"]: r for r in mgr.watch}
    assert set(rows) == {"AAPL", "JNJ"} or set(rows) == {"JNJ"}   # AAPL may have been bought
    assert rows["JNJ"]["status"] == "no entry" and rows["JNJ"]["verdict"] == "trend"
    assert rows["JNJ"]["price"] == 100.0
    assert mgr.snapshot()["watching"] == len(mgr.watch)


def test_price_changes_are_tracked_across_cycles(world, monkeypatch):
    mgr, _ = world
    _price("JNJ", 100.0)
    _signals(monkeypatch, {}, blocked_by="score")
    t0 = time.time()
    # Seed history as if the manager had been watching for 5 minutes: 100 -> 102,
    # every sample inside the 5-minute window the change is measured over.
    for i in range(160):
        mgr._sample("JNJ", 100.0 + 2.0 * i / 159, t0 - 298 + i * (296 / 159))
    state.update_price("JNJ", 102.0)
    asyncio.run(mgr.run_cycle())
    row = mgr.watch[0]
    assert row["chg_5m_pct"] == pytest.approx(2.0, abs=0.1)
    assert row["chg_1m_pct"] is not None and 0.0 < row["chg_1m_pct"] < 1.0


def test_no_change_figure_until_there_is_history(world, monkeypatch):
    mgr, _ = world
    _price("JNJ", 100.0)
    _signals(monkeypatch, {}, blocked_by="score")
    asyncio.run(mgr.run_cycle())
    assert mgr.watch[0]["chg_5m_pct"] is None             # first sample: nothing to compare


def test_full_slots_explain_why_budget_cannot_be_used(world, monkeypatch):
    mgr, executed = world
    for s in ("MSFT", "NVDA", "KO", "JNJ", "AAPL"):       # dial 4 allows 5 positions
        _hold(s, 500)
    _price("AAPL") if False else None
    state.watchlist.add("AAPL")
    _signals(monkeypatch, {"AAPL": 0.9})
    asyncio.run(mgr.run_cycle())
    assert executed == []
    assert mgr.status["state"] == "full" and "position slots" in mgr.status["message"]
    assert mgr.status["budget"]["dial_deployable_pct"] == 75.0    # 5 x 15%




def test_a_frozen_price_is_not_entered(world, monkeypatch):
    mgr, executed = world
    _price("AAPL")
    state.price_moved_at["AAPL"] = time.time() - settings.STALE_PRICE_EXIT_SECONDS
    _signals(monkeypatch, {"AAPL": 0.9})
    asyncio.run(mgr.run_cycle())
    assert executed == [] and mgr.blocked_by["price not moving"] == 1


def test_a_refused_entry_is_not_resent_every_cycle(world, monkeypatch):
    from engine.executor import executor
    mgr, executed = world
    tried = []

    async def refuse(decision):                           # e.g. a council veto: no order, no position
        tried.append(decision.symbol)
    monkeypatch.setattr(executor, "execute_decision", refuse)
    _price("AAPL")
    _signals(monkeypatch, {"AAPL": 0.9})
    for _ in range(3):
        asyncio.run(mgr.run_cycle())
    assert tried == ["AAPL"]
    assert mgr.blocked_by["entry recently refused"] == 1


def test_nothing_is_bought_before_the_trend_is_read(world, monkeypatch):
    mgr, executed = world
    from core.minute_bars import minute_bars
    _price("AAPL")
    minute_bars.drop("AAPL")                              # no history at all
    state.update_price("AAPL", 100.0)
    _signals(monkeypatch, {"AAPL": 0.95})
    asyncio.run(mgr.run_cycle())
    assert executed == []
    assert mgr.blocked_by["trend: not enough history"] == 1


def test_a_downtrend_is_not_bought_even_on_a_strong_signal(world, monkeypatch):
    mgr, executed = world
    _price("AAPL", step=-0.0005)                          # falling into the current price
    _signals(monkeypatch, {"AAPL": 0.95})
    asyncio.run(mgr.run_cycle())
    assert executed == []
    assert mgr.blocked_by["trend: downtrend"] == 1


def test_picks_carry_the_planned_dollars_and_trend(world, monkeypatch):
    mgr, executed = world
    state.is_trading_active = False                       # rank without buying
    _price("JNJ")
    _signals(monkeypatch, {"JNJ": 0.8})
    asyncio.run(mgr.run_cycle())
    [pick] = mgr.picks
    assert pick["plan"]["qty"] > 0 and pick["plan"]["dollars"] > 0 and pick["plan"]["risk"] > 0
    assert pick["trend"]["label"] == "uptrend"


def test_risk_guard_refusals_are_not_attempted(world, monkeypatch):
    mgr, executed = world
    monkeypatch.setattr(pm_mod.risk_guard, "can_open_position",
                        lambda s: (False, "Daily loss limit hit: -3.1% of budget"))
    _price("AAPL")
    _signals(monkeypatch, {"AAPL": 0.9})
    asyncio.run(mgr.run_cycle())
    assert executed == []
    assert "risk: Daily loss limit hit" in mgr.blocked_by


def test_tick_path_takes_entries_back_when_the_manager_stalls(world):
    mgr, _ = world
    assert mgr.owns_entries() is False                    # never started
    mgr._running = True
    mgr._started_at = time.time() - 120
    mgr.last_cycle_at = time.time()
    assert mgr.owns_entries() is True
    mgr.last_cycle_at = time.time() - 60
    assert mgr.owns_entries() is False                    # stalled: hand entries back
    assert mgr.snapshot()["state"] == "stalled"
    mgr.last_cycle_at = 0.0                               # wedged on its very first cycle
    assert mgr.snapshot()["state"] == "stalled"


def test_no_entries_near_the_close(world, monkeypatch):
    mgr, executed = world
    import core.market_hours as mh
    monkeypatch.setattr(mh, "minutes_to_close", lambda *a, **k: 20.0)
    _price("AAPL")
    _signals(monkeypatch, {"AAPL": 0.9})
    asyncio.run(mgr.run_cycle())
    assert executed == []
    assert mgr.blocked_by.get("too close to the market close") == 1
