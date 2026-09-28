"""
Quant council: every library strategy's read on one symbol, at one moment.

The engine's own entry decision comes from a single strategy per symbol. The
council is a second opinion made of the whole library, used in two places:

  * The manager (executor buy path) consults it before opening any trade and
    refuses an entry that the regime-suited strategies clearly oppose.
  * Each trade bot (sentinel) polls it while holding and exits if the suited
    strategies turn decisively bearish on its position.

Only price-driven library strategies vote. Sentiment and consensus already
reach the manager through Laya and the aggregator, so a price-only panel adds
independent evidence rather than counting the same news twice.

Votes are weighted: strategies suited to the live regime count fully, the rest
at UNSUITED_WEIGHT (a mean-reverter's view in a trend is weak evidence), and
each weight is scaled by the strategy's realised track record.
"""
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from core.config import settings
from engine.strategies import performance, regime as regime_mod
from engine.strategies.base import StrategyContext

UNSUITED_WEIGHT = 0.35


@dataclass
class Vote:
    strategy: str
    bias: float
    suited: bool
    weight: float
    would_enter: bool
    buy_prob: float
    blocked_by: Optional[str]
    reason: str


@dataclass
class CouncilReport:
    symbol: str
    regime: Dict[str, Any]
    consensus: float                 # weighted bias across all voters, -1..1
    suited_consensus: Optional[float]  # weighted bias of regime-suited voters only
    n_voters: int
    n_suited: int
    n_bullish: int
    n_bearish: int
    verdict: str                     # support | oppose | neutral | insufficient
    recommended: Optional[str]       # best suited strategy whose entry fires now
    summary: str
    votes: List[Vote] = field(default_factory=list)
    abstained: List[str] = field(default_factory=list)
    at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["consensus"] = round(self.consensus, 4)
        if self.suited_consensus is not None:
            d["suited_consensus"] = round(self.suited_consensus, 4)
        d["deciding_consensus"] = round(self.deciding_consensus, 4)
        return d

    @property
    def deciding_consensus(self) -> float:
        """Suited voters decide when there are enough of them; else everyone."""
        if self.suited_consensus is not None and self.n_suited >= settings.COUNCIL_MIN_VOTERS:
            return self.suited_consensus
        return self.consensus


def _members():
    from engine.strategies import registry
    return [s for s in registry.available() if s.council_member]


def convene(ctx: StrategyContext) -> CouncilReport:
    reg = regime_mod.detect(ctx.series)
    votes: List[Vote] = []
    abstained: List[str] = []

    for strat in _members():
        entry = strat.evaluate_entry(ctx)
        if entry.blocked_by in ("no_data", "no_volume"):
            abstained.append(strat.name)
            continue
        suited = (not strat.regimes) or reg.label in strat.regimes
        w = (1.0 if suited else UNSUITED_WEIGHT) * performance.weight(strat.name)
        votes.append(Vote(
            strategy=strat.name, bias=strat.bias(ctx), suited=suited, weight=round(w, 3),
            would_enter=entry.should_enter, buy_prob=entry.buy_prob,
            blocked_by=entry.blocked_by, reason=entry.reason,
        ))

    def wavg(vs):
        tw = sum(v.weight for v in vs)
        return sum(v.bias * v.weight for v in vs) / tw if tw > 0 else 0.0

    suited = [v for v in votes if v.suited]
    consensus = wavg(votes)
    suited_consensus = wavg(suited) if suited else None
    n_bull = sum(1 for v in votes if v.bias > 0.15)
    n_bear = sum(1 for v in votes if v.bias < -0.15)

    firing = sorted((v for v in suited if v.would_enter),
                    key=lambda v: v.buy_prob * v.weight, reverse=True)
    recommended = firing[0].strategy if firing else None

    report = CouncilReport(
        symbol=ctx.symbol, regime=reg.to_dict(), consensus=consensus,
        suited_consensus=suited_consensus, n_voters=len(votes), n_suited=len(suited),
        n_bullish=n_bull, n_bearish=n_bear, verdict="insufficient",
        recommended=recommended, summary="", votes=votes, abstained=abstained,
    )

    c = report.deciding_consensus
    if len(votes) < settings.COUNCIL_MIN_VOTERS:
        report.verdict = "insufficient"
    elif c >= settings.COUNCIL_SUPPORT_CONSENSUS:
        report.verdict = "support"
    elif c <= settings.COUNCIL_VETO_CONSENSUS:
        report.verdict = "oppose"
    else:
        report.verdict = "neutral"

    report.summary = (
        f"Council {report.verdict.upper()} on {ctx.symbol}: regime {reg.label}, "
        f"consensus {c:+.2f} ({n_bull} bull / {n_bear} bear of {len(votes)} voters, "
        f"{len(suited)} regime-suited)"
        + (f"; best fit now: {recommended}" if recommended else "")
        + (f"; {len(abstained)} abstained for lack of data" if abstained else "")
    )
    return report


