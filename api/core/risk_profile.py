"""
Portfolio-wide risk factor: a single 1-10 dial that drives every risk limit.

One number has to move a coherent *set* of limits, not just position size. Raising
size while leaving the position count, the daily-loss halt and the entry bar
unchanged does not make the portfolio 2x riskier -- it makes it lopsided. So each
level defines a complete, internally consistent profile.

Levels are an explicit table rather than a formula: the values are the policy, and
a table is auditable and tunable per level without re-deriving a curve.

Reading direction:
  1  = capital preservation. Small size, few positions, tight halts, high conviction bar.
  4  = default. Balanced.
  10 = aggressive. Large size, many positions, loose halts, low conviction bar.
"""
from dataclasses import dataclass, asdict
from typing import Dict, Any

DEFAULT_RISK_FACTOR = 4
MIN_RISK_FACTOR = 1
MAX_RISK_FACTOR = 10


@dataclass(frozen=True)
class RiskProfile:
    factor: int
    label: str
    # Sizing
    risk_per_trade_pct: float      # % of budget risked from entry to stop
    max_position_notional_pct: float  # % of budget in any single position
    # Portfolio shape
    max_concurrent_positions: int
    max_positions_per_asset_class: int
    # Circuit breakers
    max_daily_loss_pct: float
    max_drawdown_pct: float
    # Entry selectivity -- a lower risk appetite demands stronger evidence
    min_buy_prob: float
    min_sentiment_pos: float
    # Stop geometry: a defensive profile takes a wider stop with a smaller size,
    # which survives noise better at the cost of a worse reward:risk ratio.
    stop_atr_multiple: float
    take_profit_atr_multiple: float

    @property
    def reward_risk_ratio(self) -> float:
        return round(self.take_profit_atr_multiple / self.stop_atr_multiple, 2)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["reward_risk_ratio"] = self.reward_risk_ratio
        return d


# factor: (label, risk/trade%, max notional%, max pos, per-class, daily loss%,
#          drawdown%, min buy prob, min sentiment, stop ATR x, TP ATR x)
_TABLE = {
    1:  ("Capital Preservation", 0.20,  4.0,  2, 1, 1.0,  4.0, 0.80, 0.75, 2.50, 5.00),
    2:  ("Very Conservative",    0.35,  6.0,  3, 2, 1.5,  5.0, 0.76, 0.72, 2.20, 4.60),
    3:  ("Conservative",         0.60, 10.0,  4, 2, 2.0,  7.5, 0.70, 0.66, 1.80, 3.80),
    4:  ("Balanced (default)",   1.00, 15.0,  5, 3, 3.0, 10.0, 0.65, 0.60, 1.50, 3.00),
    5:  ("Balanced Plus",        1.30, 18.0,  6, 3, 3.5, 12.0, 0.62, 0.58, 1.40, 2.80),
    6:  ("Growth",               1.60, 21.0,  7, 4, 4.5, 14.0, 0.60, 0.56, 1.35, 2.60),
    7:  ("Assertive",            1.90, 24.0,  8, 4, 5.5, 16.0, 0.58, 0.54, 1.30, 2.40),
    8:  ("Aggressive",           2.20, 27.0,  9, 5, 6.5, 18.0, 0.56, 0.52, 1.25, 2.30),
    9:  ("Very Aggressive",      2.50, 30.0, 10, 6, 7.5, 20.0, 0.54, 0.50, 1.20, 2.20),
    10: ("Maximum Risk",         3.00, 35.0, 12, 7, 9.0, 25.0, 0.52, 0.48, 1.15, 2.10),
}


def clamp_factor(factor: Any) -> int:
    """Coerces any input to a valid integer level. Invalid input falls back to default."""
    try:
        f = int(round(float(factor)))
    except (TypeError, ValueError):
        return DEFAULT_RISK_FACTOR
    return max(MIN_RISK_FACTOR, min(MAX_RISK_FACTOR, f))


def get_profile(factor: Any = DEFAULT_RISK_FACTOR) -> RiskProfile:
    f = clamp_factor(factor)
    (label, rpt, notional, maxpos, perclass, daily, dd, minbuy, minsent,
     stop_x, tp_x) = _TABLE[f]
    return RiskProfile(
        factor=f,
        label=label,
        risk_per_trade_pct=rpt,
        max_position_notional_pct=notional,
        max_concurrent_positions=maxpos,
        max_positions_per_asset_class=perclass,
        max_daily_loss_pct=daily,
        max_drawdown_pct=dd,
        min_buy_prob=minbuy,
        min_sentiment_pos=minsent,
        stop_atr_multiple=stop_x,
        take_profit_atr_multiple=tp_x,
    )


def describe_change(old_factor: int, new_factor: int) -> str:
    """Human-readable summary of what moving the dial actually changes."""
    a, b = get_profile(old_factor), get_profile(new_factor)
    direction = "increased" if b.factor > a.factor else "decreased"
    return (
        f"Risk factor {direction} {a.factor} -> {b.factor} ({b.label}). "
        f"Risk/trade {a.risk_per_trade_pct}% -> {b.risk_per_trade_pct}%, "
        f"max position {a.max_position_notional_pct}% -> {b.max_position_notional_pct}%, "
        f"max positions {a.max_concurrent_positions} -> {b.max_concurrent_positions}, "
        f"daily-loss halt {a.max_daily_loss_pct}% -> {b.max_daily_loss_pct}%, "
        f"entry bar {a.min_buy_prob} -> {b.min_buy_prob}, "
        f"R:R {a.reward_risk_ratio} -> {b.reward_risk_ratio}."
    )


def all_levels() -> Dict[int, Dict[str, Any]]:
    """Full table, so the UI can show what each level means before committing."""
    return {f: get_profile(f).to_dict() for f in range(MIN_RISK_FACTOR, MAX_RISK_FACTOR + 1)}
