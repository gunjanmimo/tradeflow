"""
Profit harvest: sell half whenever a position shows any profit at the bid, and
keep the gain as ring-fenced day income that is never traded or used to absorb a
loss. The remainder is harvested again only on new profit.
"""
import asyncio

import pytest

from core.capital_plan import capital_plan
from core.config import settings
from core.pnl_ledger import pnl_ledger
from core.state import state, TradeDecision
from engine import profit_harvest


@pytest.fixture
def sim(monkeypatch):
    from engine.executor import AlpacaExecutor
    e = AlpacaExecutor()
    e.is_mock_mode = True
    # Never touch the real ledgers on disk.
    monkeypatch.setattr(capital_plan, "_save", lambda: None)
    monkeypatch.setattr(pnl_ledger, "_save", lambda: None)
    monkeypatch.setattr(settings, "PROFIT_HARVEST_ENABLED", True)
    monkeypatch.setattr(settings, "PROFIT_HARVEST_USD", 0.0)
    monkeypatch.setattr(settings, "PROFIT_HARVEST_FRACTION", 0.5)
    monkeypatch.setattr(capital_plan.plan, "mode", "classic")
    monkeypatch.setattr(capital_plan.plan, "classic_realized", 0.0)
    monkeypatch.setattr(capital_plan.plan, "harvested_income", 0.0)
    monkeypatch.setattr(state, "realized_pnl_today", 0.0)
    monkeypatch.setattr(state, "harvested_today", 0.0)
    monkeypatch.setattr(state, "allocated_capital", 1000.0)
    monkeypatch.setattr(state, "account_info", {"cash": 100000.0})
    day = pnl_ledger.roll()
    saved_day = dict(day)
    state.update_price("HARV", 10.40)
    state.active_positions["HARV"] = {"symbol": "HARV", "qty": 10.0, "avg_entry_price": 10.0,
                                      "current_price": 10.40, "invested_dollars": 100.0,
                                      "mode": "SIMULATED"}
    yield e
    state.active_positions.pop("HARV", None)
    state.latest_prices.pop("HARV", None)
    day.clear()
    day.update(saved_day)


def _check(price, now=1e9):
    pos = state.active_positions["HARV"]
    return profit_harvest.check("HARV", pos, price, float(pos["qty"]),
                                float(pos["avg_entry_price"]), now=now)


def test_no_harvest_at_break_even_or_a_loss(sim):
    state.update_price("HARV", 10.00)
    assert _check(10.00) is None     # $0
    state.update_price("HARV", 9.00)
    assert _check(9.00) is None      # -$10


def test_any_profit_is_harvested_however_small(sim):
    state.update_price("HARV", 10.01)
    d = _check(10.01)                # +$0.10
    assert d is not None and d.harvest and d.fraction == 0.5
    asyncio.run(sim._execute_trim(d))
    assert state.harvested_today == pytest.approx(0.05)   # 5 x $0.01


def test_no_harvest_that_would_bank_under_a_cent(sim):
    # The live PLTR case: 1 share up $0.01 -> selling 0.5 banks $0.005, shown as $0.00.
    pos = state.active_positions["HARV"]
    pos.update(qty=1.0, avg_entry_price=188.80)
    state.update_price("HARV", 188.81)
    assert _check(188.81) is None
    state.update_price("HARV", 188.82)
    d = _check(188.82)                       # 0.5 x $0.02 = $0.01
    assert d is not None
    asyncio.run(sim._execute_trim(d))
    assert state.harvested_today == pytest.approx(0.01)


def test_executor_refuses_a_sub_cent_harvest_if_the_bid_slipped(sim):
    pos = state.active_positions["HARV"]
    pos.update(qty=1.0, avg_entry_price=188.80)
    state.update_price("HARV", 188.82)
    d = _check(188.82)
    state.update_price("HARV", 188.81)       # slipped before the order went out
    asyncio.run(sim._execute_trim(d))
    assert pos["qty"] == 1.0 and state.harvested_today == 0.0


def test_profit_is_measured_at_the_bid(sim):
    # Last trade +1c, but a sell fills at the bid, 1c under the entry: a loss.
    state.update_price("HARV", 10.01, bid=9.99, ask=10.02)
    assert _check(10.01) is None
    state.update_price("HARV", 10.03, bid=10.02, ask=10.04)
    asyncio.run(sim._execute_trim(_check(10.03)))
    assert state.harvested_today == pytest.approx(0.10)   # booked at the bid: 5 x $0.02


def test_a_minimum_can_still_be_set(sim, monkeypatch):
    monkeypatch.setattr(settings, "PROFIT_HARVEST_USD", 3.0)
    state.update_price("HARV", 10.29)
    assert _check(10.29) is None     # +$2.90
    state.update_price("HARV", 10.30)
    assert _check(10.30) is not None


def test_harvest_sells_half_and_books_day_income(sim):
    d = _check(10.40)                # +$4.00
    assert d is not None and d.harvest and d.fraction == 0.5
    asyncio.run(sim._execute_trim(d))
    pos = state.active_positions["HARV"]
    assert pos["qty"] == 5.0
    assert not pos.get("trimmed")    # the trend trim stays available
    assert state.harvested_today == 2.0 and state.realized_pnl_today == 2.0
    assert capital_plan.plan.harvested_income == 2.0
    assert pnl_ledger.harvested_today() >= 2.0


def test_income_never_raises_budget_or_offsets_a_loss(sim):
    cap_before = state.hard_cap
    asyncio.run(sim._execute_trim(_check(10.40)))
    assert state.hard_cap == cap_before              # not reinvested
    state.book_realized_pnl("OTHER", -2.0)           # a later $2 loss elsewhere
    assert state.hard_cap == cap_before - 2.0        # income did not cover it
    assert state.daily_loss_pct == pytest.approx(0.2)  # $2 of $1,000, income ignored


def test_one_share_sells_half_a_share(sim):
    # The TMO case: 1 share bought at $674.82, now $678.52 -> sell 0.5, bank ~$1.85.
    pos = state.active_positions["HARV"]
    pos.update(qty=1.0, avg_entry_price=674.82, invested_dollars=674.82)
    state.update_price("HARV", 678.52)
    d = _check(678.52)
    assert d is not None
    asyncio.run(sim._execute_trim(d))
    assert pos["qty"] == pytest.approx(0.5)
    assert state.harvested_today == pytest.approx(1.85)


def test_non_fractionable_stock_holds_one_share_whole(sim, monkeypatch):
    monkeypatch.setattr(state, "fractionable_symbols", set())
    pos = state.active_positions["HARV"]
    pos.update(qty=1.0, mode="ALPACA_PAPER")
    state.update_price("HARV", 14.0)
    assert _check(14.0) is None
    assert pos["harvest_unsplittable"]


def test_crypto_harvest_must_clear_the_fees(sim):
    sym = "HRV/USD"
    state.active_positions[sym] = {"symbol": sym, "qty": 10.0, "avg_entry_price": 100.0,
                                   "current_price": 100.2, "mode": "SIMULATED"}
    try:
        # +0.2%: under the 0.25% + 0.25% taker fees, so no "income" that is really a loss.
        state.update_price(sym, 100.2)
        assert profit_harvest.check(sym, state.active_positions[sym], 100.2, 10.0, 100.0, now=1e9) is None
        state.update_price(sym, 101.0)
        d = profit_harvest.check(sym, state.active_positions[sym], 101.0, 10.0, 100.0, now=1e9)
        assert d is not None
        asyncio.run(sim._execute_trim(d))
        # 5 sold: (101 * 0.9975 - 100 * 1.0025) * 5 = $2.49, net of both fees.
        assert state.harvested_today == pytest.approx(2.49, abs=0.01)
    finally:
        state.active_positions.pop(sym, None)
        state.latest_prices.pop(sym, None)


def test_harvest_repeats_only_on_new_profit_and_is_throttled(sim):
    asyncio.run(sim._execute_trim(_check(10.40)))
    assert _check(10.40, now=1e9 + 60) is None       # same price: no new profit
    state.update_price("HARV", 10.60)
    assert _check(10.60, now=1e9 + 1) is None        # new profit but inside the retry window
    d = _check(10.60, now=1e9 + 60)
    assert d is not None
    state.update_price("HARV", 10.60)
    asyncio.run(sim._execute_trim(d))
    # Fractional: half of 5 is 2.5 sold, 2.5 left running.
    assert state.active_positions["HARV"]["qty"] == 2.5
    assert state.harvested_today == pytest.approx(2.0 + 1.5)   # 2.5 x $0.60
