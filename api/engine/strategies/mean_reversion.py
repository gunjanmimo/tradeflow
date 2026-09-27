"""
Mean-reversion strategy -- optional, for either asset class.

Buys oversold dislocations inside an intact longer-term uptrend. This is the
opposite premise to momentum, so the two should not run on the same symbol
simultaneously: the registry assigns one strategy per asset class.

The distinguishing gate is that the LONG-term trend must still be up (EMA slow
rising / price above it) while the SHORT term is stretched down. Buying oversold
in a genuine downtrend is how mean-reversion systems lose money, so that case is
explicitly refused rather than merely down-weighted.
"""
from engine.strategies.base import (
    Strategy, StrategyContext, EntryDecision, ExitDecision,
)


class MeanReversionStrategy(Strategy):
    name = "mean_reversion"
    display_name = "Mean Reversion"
    description = (
        "Buys oversold pullbacks inside an intact uptrend. Refuses oversold entries "
        "when the longer-term trend has actually broken down. Exits on reversion to "
        "the mean rather than on trend continuation."
    )
    applies_to = "any"
    requires_sentiment = False

    params = {
        "max_rsi_entry": 36.0,        # must be genuinely oversold
        "min_rsi_entry": 12.0,        # below this suggests a real breakdown, not noise
        "require_price_above_slow_ema": True,
        "max_spread_pct": 0.005,
        "veto_neg_prob": 0.65,
        "exit_rsi": 58.0,             # reversion achieved
    }

    def evaluate_entry(self, ctx: StrategyContext) -> EntryDecision:
        q = ctx.quant
        s = ctx.sentiment
        gates = {
            "rsi": q.rsi if q else None,
            "ema_fast": q.ema_fast if q else None,
            "ema_slow": q.ema_slow if q else None,
            "spread_pct": round((q.spread if q else 0) * 100, 4),
            "sentiment_neg": s.neg_prob,
            "sentiment_n": s.n_headlines,
        }

        if not q or q.rsi is None or not q.ema_slow:
            return EntryDecision(False, 0.0, "Insufficient indicator history",
                                 blocked_by="no_data", gates=gates)

        rsi = q.rsi
        # Deeper oversold = stronger signal, until it becomes a breakdown.
        depth = (self.params["max_rsi_entry"] - rsi) / self.params["max_rsi_entry"]
        buy_prob = round(min(max(0.50 + depth * 0.60, 0.0), 1.0), 4)
        if ctx.consensus is not None:
            buy_prob = round(min(max(buy_prob + (ctx.consensus - 0.5) * 0.20, 0.0), 1.0), 4)
            gates["consensus"] = ctx.consensus
        else:
            gates["consensus"] = None

        if rsi > self.params["max_rsi_entry"]:
            return EntryDecision(False, buy_prob,
                f"RSI {rsi:.1f} not oversold (need <= {self.params['max_rsi_entry']})",
                blocked_by="not_oversold", gates=gates)

        if rsi < self.params["min_rsi_entry"]:
            return EntryDecision(False, buy_prob,
                f"RSI {rsi:.1f} below {self.params['min_rsi_entry']}: this reads as a "
                f"breakdown, not a pullback",
                blocked_by="breakdown", gates=gates)

        # The trend filter that separates a pullback from a falling knife.
        if self.params["require_price_above_slow_ema"] and ctx.price < q.ema_slow:
            return EntryDecision(False, buy_prob,
                f"Price {ctx.price:.6g} below slow EMA {q.ema_slow:.6g}: longer-term "
                f"trend has broken, so this is not a pullback to buy",
                blocked_by="trend_broken", gates=gates)

        if q.spread > self.params["max_spread_pct"]:
            return EntryDecision(False, buy_prob,
                f"Spread {q.spread*100:.3f}% too wide", blocked_by="spread", gates=gates)

        if s.is_tradeable and s.neg_prob >= self.params["veto_neg_prob"]:
            return EntryDecision(False, buy_prob,
                f"Vetoed by bearish news: neg={s.neg_prob:.2f} across {s.n_headlines} headlines",
                blocked_by="sentiment_veto", gates=gates)

        profile_min = ctx.__dict__.get("_min_buy_prob")
        if profile_min is not None and buy_prob < profile_min:
            return EntryDecision(False, buy_prob,
                f"buy_prob {buy_prob:.2f} below risk-dial entry bar {profile_min:.2f}",
                blocked_by="risk_dial", gates=gates)

        return EntryDecision(True, buy_prob,
            f"Mean-reversion entry: RSI {rsi:.1f} oversold with price above slow EMA "
            f"{q.ema_slow:.6g} (trend intact)", gates=gates)

    def evaluate_exit(self, ctx: StrategyContext) -> ExitDecision:
        q = ctx.quant
        s = ctx.sentiment
        pos = ctx.position or {}
        entry = float(pos.get("avg_entry_price") or ctx.price)
        pnl_pct = (ctx.price - entry) / entry if entry > 0 else 0.0

        rsi = q.rsi if (q and q.rsi is not None) else 50.0
        sell_prob = round(min(max((rsi - 50.0) / 50.0 + 0.35 * s.neg_prob, 0.0), 1.0), 4)

        # Reversion achieved: the reason for the trade is satisfied, so take it off.
        if rsi >= self.params["exit_rsi"]:
            return ExitDecision(True, sell_prob, 1.0,
                f"Reversion complete: RSI recovered to {rsi:.1f} "
                f"(>= {self.params['exit_rsi']}), PnL {pnl_pct*100:+.2f}%")

        if q and q.ema_slow and ctx.price < q.ema_slow * 0.985:
            return ExitDecision(True, sell_prob, 1.0,
                f"Reversion thesis failed: price broke decisively below slow EMA "
                f"{q.ema_slow:.6g}, PnL {pnl_pct*100:+.2f}%")

        return ExitDecision(False, sell_prob, round(sell_prob * 0.8, 4),
                            f"Awaiting reversion: RSI {rsi:.1f}, PnL {pnl_pct*100:+.2f}%")
