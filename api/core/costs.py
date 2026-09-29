"""
One cost model for every place that simulates a fill: the backtester, the
research harness and the RL environment.

A market order pays half the quoted spread plus slippage on each side. Alpaca
charges no commission on US equities; the SEC/FINRA fees on sells are a
fraction of a basis point and are folded into slippage.

Defaults are deliberately not optimistic for liquid large caps on the IEX
feed. Measure your own: the live engine keeps a smoothed spread per symbol
(state.spread_estimate), and CostModel.from_live() uses it.
"""
from dataclasses import dataclass, asdict
from typing import Dict, Optional


@dataclass(frozen=True)
class CostModel:
    half_spread_bps: float = 2.0     # half the bid/ask spread, paid on every market fill
    slippage_bps: float = 1.0        # price moving between decision and fill, per side

    @property
    def per_side(self) -> float:
        """Fraction of price lost on one market fill."""
        return (self.half_spread_bps + self.slippage_bps) / 1e4

    @property
    def round_trip_bps(self) -> float:
        return 2 * (self.half_spread_bps + self.slippage_bps)

    def buy_fill(self, mid: float) -> float:
        return mid * (1.0 + self.per_side)

    def sell_fill(self, mid: float) -> float:
        return mid * (1.0 - self.per_side)

    def to_dict(self) -> Dict[str, float]:
        return {**asdict(self), "round_trip_bps": self.round_trip_bps}

    @classmethod
    def from_live(cls, symbols=None, floor_half_spread_bps: float = 1.0,
                  slippage_bps: float = 1.0) -> "CostModel":
        """Median live half-spread across symbols, from the spread monitor (feeds/spreads.py)."""
        from core.state import state
        from feeds.spreads import spreads
        syms = [s for s in state.spread_estimate if symbols is None or s in symbols]
        vals = [v for v in (spreads.estimate(s)[0] for s in syms) if v is not None]
        if not vals:
            return cls(slippage_bps=slippage_bps)
        vals.sort()
        half = vals[len(vals) // 2] / 2 * 1e4
        return cls(half_spread_bps=max(half, floor_half_spread_bps), slippage_bps=slippage_bps)


DEFAULT_COSTS = CostModel()


def parse(half_spread_bps: Optional[float] = None, slippage_bps: Optional[float] = None) -> CostModel:
    return CostModel(half_spread_bps=DEFAULT_COSTS.half_spread_bps if half_spread_bps is None else half_spread_bps,
                     slippage_bps=DEFAULT_COSTS.slippage_bps if slippage_bps is None else slippage_bps)
