"""
Adaptive strategy: picks the right library strategy for the moment.

Assign "adaptive" to a symbol or an asset class and, on every evaluation, it:

  1. reads the live regime and a ranked shortlist of regime-suited strategies
     from the analysis worker process (no regime math on the tick path)
  2. re-checks the top ADAPTIVE_TOP_K of them against the live tick
  3. among those that fire, picks the highest buy_prob x track-record weight

The chosen strategy's name is returned in gates["selected_strategy"]. The
executor stamps it on the position as entry_strategy, and from then on the
position's exits are delegated to THAT strategy (registry.for_position). A trade
entered as a Turtle breakout is exited by the Turtle rules, even if the regime
has since changed -- switching exit logic mid-trade would mean exiting on a
thesis the trade was never opened on.

The engine is long-only, so in a trending_down regime every strategy stands
aside. That is the intended behaviour, not a gap.
"""
from engine.strategies import performance
from engine.strategies.base import (
    Strategy, StrategyContext, EntryDecision, ExitDecision,
)


class AdaptiveStrategy(Strategy):
    name = "adaptive"
    display_name = "Adaptive (Regime Selector)"
    description = (
        "Detects the market regime and routes to the best-suited library strategy "
        "at that moment. Exits are handled by whichever strategy opened the trade."
    )
    source = "Regime-switching meta-strategy over the quant library"
    applies_to = "any"
    requires_sentiment = False
    params = {"min_regime_confidence": 0.0}

    def _analysis(self, ctx: StrategyContext):
        """The worker's regime read and ranked candidates, or None if absent/stale."""
        from core.state import state
        a = state.fresh_analysis(ctx.symbol)
        if not a or not a.get("regime"):
            return None, []
        return a["regime"], a.get("candidates") or []

    def evaluate_entry(self, ctx: StrategyContext) -> EntryDecision:
        from core.config import settings
        from engine.strategies import registry
        reg, ranked = self._analysis(ctx)
        if reg is None:
            return EntryDecision(False, 0.0,
                "Regime analysis not available yet (analysis worker warming up or stale)",
                blocked_by="no_analysis", gates={"regime": None})
        label = reg["label"]
        gates = {"regime": label, "regime_detail": reg}

        if label == "unknown":
            return EntryDecision(False, 0.0, f"Regime unknown ({'; '.join(reg['reasons'])})",
                                 blocked_by="no_data", gates=gates)
        if label == "trending_down":
            return EntryDecision(False, 0.0,
                f"Regime trending_down ({'; '.join(reg['reasons'])}): long-only, standing aside",
                blocked_by="regime_downtrend", gates=gates)
        if reg["confidence"] < self.params["min_regime_confidence"]:
            return EntryDecision(False, 0.0,
                f"Regime {label} confidence {reg['confidence']:.2f} too low",
                blocked_by="regime_unclear", gates=gates)

        # The worker already ranked every suited strategy; only the top few are
        # re-checked here against the live tick, which keeps the per-tick cost at
        # a few strategies instead of the whole library.
        klass = "crypto" if ctx.is_crypto else "equity"
        shortlist = [registry.get(n) for n in ranked]
        shortlist = [st for st in shortlist
                     if st is not None and st.applies_to in ("any", klass)][:settings.ADAPTIVE_TOP_K]

        gates["candidates"] = {}
        best, best_score, best_dec = None, -1.0, None
        top_prob = 0.0
        for strat in shortlist:
            dec = strat.evaluate_entry(ctx)
            w = performance.weight(strat.name)
            gates["candidates"][strat.name] = {
                "enter": dec.should_enter, "buy_prob": dec.buy_prob,
                "blocked_by": dec.blocked_by, "weight": w,
            }
            top_prob = max(top_prob, dec.buy_prob)
            if dec.should_enter and dec.buy_prob * w > best_score:
                best, best_score, best_dec = strat, dec.buy_prob * w, dec

        if best is None:
            blocked = ", ".join(f"{n}:{c['blocked_by']}" for n, c in gates["candidates"].items())
            return EntryDecision(False, top_prob,
                f"Regime {label}: no suited strategy fires ({blocked or 'none suited'})",
                blocked_by="no_candidate", gates=gates)

        gates["selected_strategy"] = best.name
        gates["selected_gates"] = best_dec.gates
        return EntryDecision(True, best_dec.buy_prob,
            f"regime {label} -> {best.name}: {best_dec.reason}", gates=gates)

    def evaluate_exit(self, ctx: StrategyContext) -> ExitDecision:
        """
        Fallback only: reached when a position has no recorded entry strategy
        (e.g. opened before a restart). Polls the regime-suited strategies and
        closes when a majority of them would.
        """
        from engine.strategies import registry
        reg, ranked = self._analysis(ctx)
        label = reg["label"] if reg else "unknown"
        if label == "trending_down":
            return ExitDecision(True, 0.8, 1.0,
                f"Regime turned trending_down ({'; '.join(reg['reasons'])})")
        cands = [registry.get(n) for n in ranked if registry.get(n) is not None]
        if not cands:
            return ExitDecision(False, 0.3, 0.2, f"Holding: regime {label}, no suited strategy to consult")
        exits = [s.evaluate_exit(ctx) for s in cands]
        closing = [e for e in exits if e.should_close]
        sell = round(sum(e.sell_prob for e in exits) / len(exits), 4)
        if len(closing) * 2 > len(exits):
            return ExitDecision(True, sell, 1.0,
                f"{len(closing)}/{len(exits)} {label} strategies exit: {closing[0].reason}")
        return ExitDecision(False, sell, round(sell * 0.8, 4),
            f"Holding: {len(closing)}/{len(exits)} {label} strategies want out")
