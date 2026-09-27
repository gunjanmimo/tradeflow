"""
Strategy registry: which strategy runs on which symbol.

Resolution order, most specific first:
  1. explicit per-symbol override   (state.strategy_overrides["BTC/USD"])
  2. asset-class default            (crypto -> momentum, equity -> news_catalyst)

Defaults are asset-class-aware because the two classes offer different
information. See engine/strategies/base.py for the reasoning; in short, equities
have dense real news and discrete catalysts, crypto has neither but trades
continuously.

A strategy is only assignable to a class it declares support for, so a
news-requiring strategy cannot be attached to alt-coins that have no news --
that configuration would silently never trade.
"""
import logging
from typing import Dict, List, Optional

from core.state import is_crypto_symbol
from engine.strategies.base import Strategy
from engine.strategies.momentum import MomentumBreakoutStrategy
from engine.strategies.news_catalyst import NewsCatalystStrategy
from engine.strategies.mean_reversion import MeanReversionStrategy

logger = logging.getLogger("tradeflow.strategies")

# Instantiated once; strategies are stateless apart from their params.
_STRATEGIES: Dict[str, Strategy] = {
    s.name: s for s in (
        MomentumBreakoutStrategy(),
        NewsCatalystStrategy(),
        MeanReversionStrategy(),
    )
}

# Asset-class defaults
DEFAULT_BY_CLASS: Dict[str, str] = {
    "crypto": "momentum_breakout",
    "equity": "news_catalyst",
}


def asset_class(symbol: str) -> str:
    return "crypto" if is_crypto_symbol(symbol) else "equity"


def available() -> List[Strategy]:
    return list(_STRATEGIES.values())


def get(name: str) -> Optional[Strategy]:
    return _STRATEGIES.get(name)


def is_compatible(name: str, klass: str) -> bool:
    strat = _STRATEGIES.get(name)
    if strat is None:
        return False
    return strat.applies_to in ("any", klass)


def resolve(symbol: str, class_defaults: Dict[str, str],
            overrides: Dict[str, str]) -> Strategy:
    """
    Returns the strategy for this symbol. Falls back to the asset-class default,
    and finally to momentum (which needs no external data and so can always run).
    """
    name = overrides.get(symbol)
    if name and name in _STRATEGIES:
        return _STRATEGIES[name]

    klass = asset_class(symbol)
    name = class_defaults.get(klass) or DEFAULT_BY_CLASS.get(klass)
    strat = _STRATEGIES.get(name)
    if strat and is_compatible(strat.name, klass):
        return strat

    if strat and not is_compatible(strat.name, klass):
        logger.warning(
            "Strategy %s is not valid for %s assets; falling back to momentum_breakout",
            name, klass,
        )
    return _STRATEGIES["momentum_breakout"]


def describe_all() -> Dict:
    return {
        "strategies": [s.describe() for s in _STRATEGIES.values()],
        "class_defaults": dict(DEFAULT_BY_CLASS),
        "compatibility": {
            s.name: [k for k in ("crypto", "equity") if is_compatible(s.name, k)]
            for s in _STRATEGIES.values()
        },
    }
