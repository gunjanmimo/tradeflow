"""
Strategy interface.

Why per-asset-class strategies
------------------------------
A single strategy was applied to everything, and it required Laya sentiment for
every entry. That works for equities, where Alpaca's news feed is dense, and fails
completely for alt-coins, where real coverage is sparse -- measured live, 1 of 26
watchlist symbols had a fresh headline, so a sentiment-gated engine produced zero
tradeable signals.

Different asset classes carry different information:

  * Equities trade around discrete catalysts (earnings, verdicts, guidance) during
    fixed hours, and real news is available. Sentiment is the primary edge.
  * Crypto trades continuously with no earnings calendar and thin per-asset news.
    Price structure and volume are the primary edge; sentiment is best used as a
    VETO (do not buy into clearly bad news) rather than as a requirement.

So a strategy declares what it needs, and the registry assigns strategies per
asset class. A strategy that cannot get its inputs abstains with a stated reason
instead of trading on a default.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class EntryDecision:
    """A strategy's verdict on opening a position."""
    should_enter: bool
    buy_prob: float                     # 0..1 conviction, comparable across strategies
    reason: str
    blocked_by: Optional[str] = None    # which gate refused, for diagnostics
    # Named gate results, surfaced to the UI so a "no trade" is explainable
    gates: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExitDecision:
    """A strategy's verdict on closing a position."""
    should_close: bool
    sell_prob: float
    close_prob: float
    reason: str
    # Fraction of the position to close (1.0 = all). Enables scaling out.
    close_fraction: float = 1.0


@dataclass
class StrategyContext:
    """Everything a strategy may read. Passed explicitly rather than reaching into globals."""
    symbol: str
    price: float
    quant: Any                       # QuantMetrics
    sentiment: Any                   # SentimentRecord
    consensus: Optional[float]       # None when no real source covered this symbol
    is_crypto: bool
    position: Optional[Dict[str, Any]] = None
    highest_price: Optional[float] = None
    # Price/volume history with memoised indicators (engine.strategies.indicators).
    # None only when a caller builds a context by hand; library strategies abstain.
    series: Any = None


class Strategy(ABC):
    """
    Base strategy.

    `name` is stable and used as the registry key and in persisted trade records,
    so renaming one breaks historical attribution -- treat it as an identifier.
    """
    name: str = "base"
    display_name: str = "Base"
    description: str = ""
    # Which asset classes this strategy is valid for: "crypto", "equity", or "any"
    applies_to: str = "any"
    # True when the strategy cannot function without fresh, agreeing sentiment.
    requires_sentiment: bool = False

    # Market regimes this strategy is built for (see engine/strategies/regime.py).
    # Empty means regime-agnostic. The adaptive selector only considers a strategy
    # when the live regime is in this set.
    regimes: tuple = ()
    # Where the method comes from, for the UI and for honest attribution.
    source: str = ""
    # True for the price-only library strategies that vote in the quant council.
    council_member: bool = False

    # Tunables, exposed so the UI can show and adjust them per strategy
    params: Dict[str, Any] = {}

    @abstractmethod
    def evaluate_entry(self, ctx: StrategyContext) -> EntryDecision:
        ...

    @abstractmethod
    def evaluate_exit(self, ctx: StrategyContext) -> ExitDecision:
        ...

    def bias(self, ctx: StrategyContext) -> float:
        """
        Directional read in -1..1 (+1 = this strategy's thesis is strongly long).

        Distinct from evaluate_entry: a strategy can be bullish without its entry
        trigger firing right now. Default derives it from entry conviction.
        """
        d = self.evaluate_entry(ctx)
        return round(min(max((d.buy_prob - 0.5) * 2.0, -1.0), 1.0), 4)

    # ---- shared helpers ----
    @staticmethod
    def _trend_score(quant, price: float) -> float:
        """0..1 where >0.5 is uptrend. Uses EMA stack plus price position."""
        if not quant or not quant.ema_fast or not quant.ema_slow:
            return 0.5
        if quant.ema_fast > quant.ema_slow and price > quant.ema_fast:
            return 0.85
        if quant.ema_fast > quant.ema_slow:
            return 0.65
        if quant.ema_fast < quant.ema_slow and price < quant.ema_fast:
            return 0.15
        return 0.35

    @staticmethod
    def _rsi_score(quant) -> float:
        """0..1 favourability of current RSI for a long entry."""
        if not quant or quant.rsi is None:
            return 0.5
        r = quant.rsi
        if r < 30:
            return 0.80      # deeply oversold: rebound candidate
        if r < 40:
            return 0.70
        if 45 <= r <= 62:
            return 0.72      # healthy trending range
        if r > 75:
            return 0.10      # overbought: poor entry
        if r > 68:
            return 0.25
        return 0.5

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "applies_to": self.applies_to,
            "requires_sentiment": self.requires_sentiment,
            "regimes": list(self.regimes),
            "source": self.source,
            "council_member": self.council_member,
            "params": dict(self.params),
        }
