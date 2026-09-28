"""
Capital plan: "classic" vs "stair" money management.

Classic: the bots trade against a fixed budget (state.allocated_capital). The
budget is a HARD cap: realised losses shrink what the bots may deploy
(cap + realised PnL since the cap was set), and realised profit never raises it
above the cap -- profit is left as cash the engine does not trade.

Stair (a profit ratchet):
  1. Split the deposit: deploy_pct goes to trading, the rest is a RESERVE that
     the engine never trades.
  2. Each stage has a target: stage capital x target_multiple (default 2x).
  3. Progress is measured on REALISED PnL only. Unrealised gains can vanish,
     so they never count toward a target or get banked.
  4. On reaching the target, harvest_pct of the stage's profit is BANKED as
     income and taken out of the trading budget; the rest compounds as the
     next stage's capital.
  5. If trading capital falls below what can place a minimum order, new
     entries stop. The reserve and banked income are never used to top it up.

What stair mode deliberately does NOT do: it never raises risk to reach a
target. Strategies, stops and the risk dial are unchanged; only the budget the
bots may use changes. Chasing a target by sizing up after losses is how
accounts blow up, so "reach 2x" means "keep trading normally until 2x".

The engine cannot move money. Banked income and the reserve stay as cash in the
broker account; the engine simply never trades them. Withdrawing them is a
manual step at the broker.

State is persisted to data/capital_plan.json after every change, so banked
income and stage progress survive restarts.
"""
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger("tradeflow.capital")

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_PATH = os.path.join(_DATA_DIR, "capital_plan.json")

# Engine order minimums (engine/allocation_agent.py, engine/risk_guard.py).
MIN_ORDER_CRYPTO = 15.0
MIN_ORDER_EQUITY = 30.0


@dataclass
class Stage:
    ladder: int
    stage: int
    start_capital: float
    target: float
    reached_capital: float
    profit: float
    banked: float
    next_capital: float
    started_at: float
    completed_at: float


@dataclass
class CapitalPlan:
    mode: str = "classic"               # "classic" | "stair"
    deposit: float = 0.0
    deploy_pct: float = 0.5
    target_multiple: float = 2.0
    harvest_pct: float = 0.5
    reserve: float = 0.0                # never traded
    stage: int = 0
    stage_start_capital: float = 0.0
    stage_started_at: float = 0.0
    realized_in_stage: float = 0.0
    banked_income: float = 0.0          # never traded
    harvested_income: float = 0.0       # profit-harvest day income, cumulative; never traded
    halted: bool = False
    ladder: int = 0                     # increments each time a ladder is (re)started
    classic_budget: float = 10000.0     # the classic cap; also restored when stair is switched off
    classic_realized: float = 0.0       # classic: realised PnL since the cap was last set
    history: List[Dict[str, Any]] = field(default_factory=list)

    # ---- derived ----
    @property
    def trading_capital(self) -> float:
        return round(self.stage_start_capital + self.realized_in_stage, 2)

    @property
    def target(self) -> float:
        return round(self.stage_start_capital * self.target_multiple, 2)

    @property
    def progress(self) -> float:
        """0..1 of the way from stage start to target, on realised PnL."""
        span = self.target - self.stage_start_capital
        if span <= 0:
            return 0.0
        return round(min(max(self.realized_in_stage / span, 0.0), 1.0), 4)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.update(trading_capital=self.trading_capital, target=self.target,
                 progress=self.progress,
                 total_value=round(self.trading_capital + self.reserve + self.banked_income
                                   + self.harvested_income, 2))
        return d


def min_stage_capital(max_position_notional_pct: float) -> Dict[str, float]:
    """Smallest trading capital that can place one minimum-size order."""
    frac = max(max_position_notional_pct / 100.0, 1e-9)
    return {"crypto": round(MIN_ORDER_CRYPTO / frac, 2),
            "equity": round(MIN_ORDER_EQUITY / frac, 2)}


class CapitalPlanManager:
    def __init__(self):
        self.plan = CapitalPlan()

    # ---- persistence ----
    def load(self):
        try:
            with open(_PATH) as f:
                raw = json.load(f)
            known = {k: v for k, v in raw.items() if k in CapitalPlan.__dataclass_fields__}
            self.plan = CapitalPlan(**known)
            logger.info(f"Capital plan loaded: {self.plan.mode}, stage {self.plan.stage}")
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.error(f"Could not load capital plan ({e}); starting in classic mode")
            self.plan = CapitalPlan()
        self._apply_budget()
        from core.state import state
        state.sync_locked_equity()

    def _save(self):
        try:
            os.makedirs(_DATA_DIR, exist_ok=True)
            tmp = _PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump(asdict(self.plan), f, indent=2)
            os.replace(tmp, _PATH)       # atomic: never a half-written ledger
        except Exception as e:
            logger.error(f"Could not persist capital plan: {e}")

    # ---- budget link ----
    def _apply_budget(self):
        from core.state import state
        if self.plan.mode == "stair":
            state.allocated_capital = max(self.plan.trading_capital, 0.0)
        else:
            state.allocated_capital = self.plan.classic_budget

    @property
    def budget_realized(self) -> float:
        """Realised PnL not already folded into allocated_capital (stair folds its own)."""
        return self.plan.classic_realized if self.plan.mode == "classic" else 0.0

    def set_classic_budget(self, amount: float):
        """Sets a new classic cap. The loss/profit ledger restarts from the new cap."""
        self.plan.classic_budget = round(float(amount), 2)
        self.plan.classic_realized = 0.0
        self._apply_budget()
        from core.state import state
        state.reset_peak_equity()
        state.sync_locked_equity(self.plan.classic_budget)
        self._save()

    # ---- mode switches ----
    def enable_stair(self, deposit: float, deploy_pct: float = 0.5,
                     target_multiple: float = 2.0, harvest_pct: float = 0.5) -> Dict[str, Any]:
        """Starts a fresh ladder. Raises ValueError with an actionable message."""
        from core.state import state
        if not (0.05 <= deploy_pct <= 1.0):
            raise ValueError("deploy_pct must be between 0.05 and 1.0")
        if not (1.1 <= target_multiple <= 10.0):
            raise ValueError("target_multiple must be between 1.1 and 10")
        if not (0.0 <= harvest_pct <= 1.0):
            raise ValueError("harvest_pct must be between 0 and 1")
        stage_capital = round(deposit * deploy_pct, 2)
        mins = min_stage_capital(state.risk_profile.max_position_notional_pct)
        if stage_capital < mins["crypto"]:
            raise ValueError(
                f"Trading capital ${stage_capital:,.2f} is too small to place any order: at risk dial "
                f"{state.risk_profile.factor} one position is capped at "
                f"{state.risk_profile.max_position_notional_pct:.0f}% (${stage_capital * state.risk_profile.max_position_notional_pct / 100:,.2f}), "
                f"below the ${MIN_ORDER_CRYPTO:.0f} crypto / ${MIN_ORDER_EQUITY:.0f} stock minimum. "
                f"Need at least ${mins['crypto']:,.2f} trading capital for crypto "
                f"(${mins['equity']:,.2f} for stocks), i.e. a deposit of "
                f"${mins['crypto'] / deploy_pct:,.2f} at {deploy_pct:.0%} deployed.")
        cash = float(state.account_info.get("cash", 0.0))
        if deposit > cash + 1e-6:
            raise ValueError(f"Deposit ${deposit:,.2f} exceeds available broker cash ${cash:,.2f}")
        if stage_capital + 1e-6 < state.committed_capital:
            raise ValueError(
                f"Trading capital ${stage_capital:,.2f} is below the ${state.committed_capital:,.2f} "
                "already committed to open or pending bot positions. Close positions or use a larger deposit first.")

        classic = state.allocated_capital if self.plan.mode == "classic" else self.plan.classic_budget
        now = time.time()
        # Banked income is real cash at the broker: a new ladder must not forget it.
        prior_banked, prior_history, ladder = (self.plan.banked_income, list(self.plan.history),
                                               self.plan.ladder + 1)
        self.plan = CapitalPlan(
            banked_income=prior_banked, history=prior_history, ladder=ladder,
            mode="stair", deposit=round(deposit, 2), deploy_pct=deploy_pct,
            target_multiple=target_multiple, harvest_pct=harvest_pct,
            reserve=round(deposit - stage_capital, 2), stage=1,
            stage_start_capital=stage_capital, stage_started_at=now,
            classic_budget=classic,
        )
        self._apply_budget()
        state.reset_peak_equity()
        state.sync_locked_equity(self.plan.deposit)
        self._save()
        warn = ""
        if stage_capital < mins["equity"]:
            warn = f" Note: below ${mins['equity']:,.2f}, stock orders cannot be placed; only crypto will trade."
        state.log_event("CAPITAL_PLAN",
            f"Stair mode ON: ${deposit:,.2f} deposit -> ${stage_capital:,.2f} trading, "
            f"${self.plan.reserve:,.2f} reserve (untouched). Stage 1 target ${self.plan.target:,.2f} "
            f"({target_multiple:g}x); {harvest_pct:.0%} of each stage's profit is banked.{warn}")
        return self.plan.to_dict()

    def disable_stair(self) -> Dict[str, Any]:
        from core.state import state
        if self.plan.mode != "stair":
            return self.plan.to_dict()
        summary = (f"Stair mode OFF after stage {self.plan.stage}: banked income "
                   f"${self.plan.banked_income:,.2f}, trading capital ${self.plan.trading_capital:,.2f}, "
                   f"reserve ${self.plan.reserve:,.2f}. Budget restored to classic ${self.plan.classic_budget:,.2f}.")
        self.plan.mode = "classic"
        self.plan.classic_realized = 0.0
        self._apply_budget()
        state.reset_peak_equity()
        state.sync_locked_equity(self.plan.classic_budget)
        self._save()
        state.log_event("CAPITAL_PLAN", summary)
        return self.plan.to_dict()

    # ---- the ratchet ----
    def on_realized(self, symbol: str, pnl: float):
        """Called for every closed trade. O(1); runs on the close path, not the tick path."""
        p = self.plan
        if p.mode != "stair":
            p.classic_realized = round(p.classic_realized + float(pnl), 2)
            self._save()
            return
        from core.state import state
        p.realized_in_stage = round(p.realized_in_stage + float(pnl), 2)

        # Several targets can in principle be crossed by one large win.
        while p.trading_capital >= p.target and p.stage_start_capital > 0:
            reached = p.trading_capital
            profit = round(reached - p.stage_start_capital, 2)
            banked = round(profit * p.harvest_pct, 2)
            next_capital = round(reached - banked, 2)
            now = time.time()
            p.history.append(asdict(Stage(
                ladder=p.ladder, stage=p.stage, start_capital=p.stage_start_capital, target=p.target,
                reached_capital=reached, profit=profit, banked=banked,
                next_capital=next_capital, started_at=p.stage_started_at, completed_at=now)))
            p.banked_income = round(p.banked_income + banked, 2)
            state.log_event("STAIR_STEP",
                f"Stage {p.stage} complete: ${p.stage_start_capital:,.2f} -> ${reached:,.2f}. "
                f"Banked ${banked:,.2f} (total income ${p.banked_income:,.2f}); "
                f"stage {p.stage + 1} starts with ${next_capital:,.2f}, target "
                f"${next_capital * p.target_multiple:,.2f}.")
            p.stage += 1
            p.stage_start_capital = next_capital
            p.realized_in_stage = 0.0
            p.stage_started_at = now

        mins = min_stage_capital(state.risk_profile.max_position_notional_pct)
        was_halted = p.halted
        p.halted = p.trading_capital < mins["crypto"]
        if p.halted and not was_halted:
            state.log_event("STAIR_HALT",
                f"Trading capital ${p.trading_capital:,.2f} is below the ${mins['crypto']:,.2f} needed to "
                f"place an order. New entries stopped. Reserve ${p.reserve:,.2f} and banked income "
                f"${p.banked_income:,.2f} are untouched; restart the ladder to continue.")
        self._apply_budget()
        self._save()

    def on_harvest(self, symbol: str, pnl: float):
        """
        Profit-harvest income. Deliberately NOT added to classic_realized or
        realized_in_stage: it must neither raise the budget, compound into a
        stage, nor offset a later loss. It stays as untraded broker cash.
        """
        self.plan.harvested_income = round(self.plan.harvested_income + float(pnl), 2)
        self._save()

    def entry_block_reason(self) -> Optional[str]:
        p = self.plan
        if p.mode == "stair" and p.halted:
            return (f"Stair mode: trading capital ${p.trading_capital:,.2f} too small to place an order; "
                    f"reserve and banked income are not used to top it up.")
        return None


capital_plan = CapitalPlanManager()
