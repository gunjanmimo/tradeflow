"""
Strategy registry: which strategy runs on which symbol.

Resolution order, most specific first:
  1. explicit per-symbol override   (state.strategy_overrides["NVDA"])
  2. the default                    (state.strategy_class_defaults["equity"])

The platform trades US equities only, so there is one asset class. The
"equity" key is kept so the API and dashboard keep a stable shape.
"""
import logging
from typing import Dict, List, Optional

from engine.strategies.base import Strategy
from engine.strategies.news_catalyst import NewsCatalystStrategy
from engine.strategies.stock_score import StockScoreStrategy
from engine.strategies.mean_reversion import MeanReversionStrategy
from engine.strategies.library import LIBRARY
from engine.strategies.adaptive import AdaptiveStrategy
from engine.strategies.rl_ppo import RLPolicyStrategy

logger = logging.getLogger("tradeflow.strategies")

EQUITY = "equity"

# Instantiated once; strategies are stateless apart from their params.
_STRATEGIES: Dict[str, Strategy] = {
    s.name: s for s in (
        RLPolicyStrategy(),
        NewsCatalystStrategy(),
        StockScoreStrategy(),
        MeanReversionStrategy(),
        AdaptiveStrategy(),
        *(cls() for cls in LIBRARY),
    )
}

# The PPO policy decides by default. It places orders only once a trained policy
# has passed its promotion gate (settings.RL_MODE = "auto"); until then it runs
# in shadow mode and the platform stays flat.
DEFAULT_BY_CLASS: Dict[str, str] = {EQUITY: "rl_ppo"}


def asset_class(symbol: str) -> str:
    return EQUITY


def available() -> List[Strategy]:
    return list(_STRATEGIES.values())


def get(name: str) -> Optional[Strategy]:
    return _STRATEGIES.get(name)


def is_compatible(name: str, klass: str = EQUITY) -> bool:
    return klass == EQUITY and name in _STRATEGIES


def resolve(symbol: str, class_defaults: Dict[str, str],
            overrides: Dict[str, str]) -> Strategy:
    """The override for this symbol, else the configured default, else rl_ppo."""
    name = overrides.get(symbol)
    if name and name in _STRATEGIES:
        return _STRATEGIES[name]
    name = class_defaults.get(EQUITY) or DEFAULT_BY_CLASS[EQUITY]
    strat = _STRATEGIES.get(name)
    if strat is None:
        logger.warning("Unknown default strategy %s; falling back to %s", name, DEFAULT_BY_CLASS[EQUITY])
        strat = _STRATEGIES[DEFAULT_BY_CLASS[EQUITY]]
    return strat


def for_position(symbol: str, position: Optional[Dict], class_defaults: Dict[str, str],
                 overrides: Dict[str, str]) -> Strategy:
    """
    The strategy that manages an OPEN position's exits: the one that opened it.

    Resolving fresh each tick meant changing the default (or the adaptive
    selector changing its pick) silently swapped a live trade's exit rules.
    Falls back to normal resolution for positions with no recorded entry
    strategy, such as ones opened before this process started.
    """
    name = (position or {}).get("entry_strategy")
    if name and name in _STRATEGIES:
        return _STRATEGIES[name]
    return resolve(symbol, class_defaults, overrides)


def describe_all() -> Dict:
    return {
        "strategies": [s.describe() for s in _STRATEGIES.values()],
        "class_defaults": dict(DEFAULT_BY_CLASS),
        "compatibility": {s.name: [EQUITY] for s in _STRATEGIES.values()},
    }
