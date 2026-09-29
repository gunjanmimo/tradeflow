"""
One place that assembles a StrategyContext from live state.

The decision engine, the sentinel bots and the quant council all need the same
inputs. Building them in three places is how the sentinel previously ended up
with a context that lacked fields the engine's had.
"""
from typing import Any, Dict, Optional

import numpy as np

from core.state import state
from engine.strategies.base import StrategyContext
from engine.strategies.indicators import PriceSeries


def series_for(symbol: str) -> Optional[PriceSeries]:
    if not state.price_history.get(symbol):
        return None
    vols = state.volume_history.get(symbol) or ()
    return PriceSeries(state.history_array(symbol),
                       np.fromiter(vols, dtype=np.float64, count=len(vols)))


class LazySeries:
    """
    Stands in for a PriceSeries and builds it on first use.

    Most ticks go to strategies that read only QuantMetrics (momentum, news),
    so converting the whole price buffer to arrays for every context was O(n)
    work on the hot path for nothing.
    """
    __slots__ = ("_symbol", "_series")

    def __init__(self, symbol: str):
        self._symbol = symbol
        self._series = None

    def _get(self) -> PriceSeries:
        if self._series is None:
            self._series = series_for(self._symbol)
        return self._series

    def __len__(self):
        return len(state.price_history.get(self._symbol) or ())

    def __getattr__(self, name):
        return getattr(self._get(), name)


def build_context(symbol: str, price: Optional[float] = None,
                  position: Optional[Dict[str, Any]] = None,
                  highest_price: Optional[float] = None,
                  quant=None, sentiment=None) -> Optional[StrategyContext]:
    """Returns None when there is no price for the symbol yet."""
    if price is None:
        tick = state.latest_prices.get(symbol)
        if tick is None:
            return None
        price = tick.price

    from feeds.multi_source_aggregator import trend_aggregator
    ctx = StrategyContext(
        symbol=symbol,
        price=float(price),
        quant=quant if quant is not None else state.quant_metrics.get(symbol),
        sentiment=sentiment if sentiment is not None else state.get_sentiment(symbol),
        consensus=trend_aggregator.get_consensus(symbol),
        position=position,
        highest_price=highest_price,
        series=LazySeries(symbol) if state.price_history.get(symbol) else None,
    )
    # The risk dial's entry bar is enforced inside the strategy so its own
    # reason string can name it, rather than the engine silently overriding.
    ctx._min_buy_prob = state.risk_profile.min_buy_prob
    return ctx
