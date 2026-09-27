"""
Momentum / trend-continuation strategy -- the crypto default.

Rationale: crypto trades 24/7 with no earnings calendar, and per-asset news
coverage is sparse (measured: 1 of 26 watchlist symbols had a fresh headline). A
sentiment-gated strategy therefore abstains permanently on alt-coins. Price
structure and participation are the available edge here.

Sentiment is used as a VETO only: we will not buy into clearly bad news, but we do
not require good news to act. That distinction is what makes the strategy usable
without turning "no data" into "bullish", which was the old system's failure mode.
"""
from core.config import settings
from engine.strategies.base import (
    Strategy, StrategyContext, EntryDecision, ExitDecision,
)


class MomentumBreakoutStrategy(Strategy):
    name = "momentum_breakout"
    display_name = "Momentum Breakout"
    description = (
        "Trend continuation on EMA stack with volume confirmation. Does not require "
        "news; uses sentiment only to veto entries into clearly bearish coverage. "
        "Built for 24/7 crypto where per-asset news is sparse."
    )
    applies_to = "crypto"
    requires_sentiment = False

    params = {
        "min_trend_score": 0.60,      # EMA stack must be constructive
        "max_rsi": 72.0,              # do not chase overbought
        "min_volume_ratio": 0.85,     # participation vs recent average
        "max_spread_pct": 0.005,      # 0.5% max bid/ask spread
        "veto_neg_prob": 0.60,        # abstain if news is this bearish
        "exit_trend_break": 0.30,     # close when trend decays past this
    }

    def evaluate_entry(self, ctx: StrategyContext) -> EntryDecision:
        q = ctx.quant
        gates = {}

        trend = self._trend_score(q, ctx.price)
        rsi_s = self._rsi_score(q)
        vol_ratio = q.volume_ratio if q else 1.0
        spread = q.spread if q else 1.0

        gates["trend_score"] = round(trend, 3)
        gates["rsi"] = q.rsi if q else None
        gates["volume_ratio"] = vol_ratio
        gates["spread_pct"] = round(spread * 100, 4)
        gates["sentiment_neg"] = ctx.sentiment.neg_prob
        gates["sentiment_n"] = ctx.sentiment.n_headlines

        # Conviction: mostly structure, with consensus as a mild tilt when real.
        base = 0.55 * trend + 0.30 * rsi_s
        vol_bonus = 0.15 if vol_ratio >= 1.2 else (0.08 if vol_ratio >= self.params["min_volume_ratio"] else 0.0)
        buy_prob = base + vol_bonus

        if ctx.consensus is not None:
            # Real conviction data nudges up or down by at most 0.10
            buy_prob += (ctx.consensus - 0.5) * 0.20
            gates["consensus"] = ctx.consensus
        else:
            gates["consensus"] = None

        buy_prob = round(min(max(buy_prob, 0.0), 1.0), 4)

        # --- hard gates ---
        if spread > self.params["max_spread_pct"]:
            return EntryDecision(False, buy_prob,
                f"Spread {spread*100:.3f}% exceeds {self.params['max_spread_pct']*100:.2f}% limit",
                blocked_by="spread", gates=gates)

        if trend < self.params["min_trend_score"]:
            return EntryDecision(False, buy_prob,
                f"Trend score {trend:.2f} below {self.params['min_trend_score']} (no constructive EMA stack)",
                blocked_by="trend", gates=gates)

        if q and q.rsi is not None and q.rsi > self.params["max_rsi"]:
            return EntryDecision(False, buy_prob,
                f"RSI {q.rsi:.1f} overbought (> {self.params['max_rsi']})",
                blocked_by="rsi", gates=gates)

        if vol_ratio < self.params["min_volume_ratio"]:
            return EntryDecision(False, buy_prob,
                f"Volume ratio {vol_ratio:.2f} below {self.params['min_volume_ratio']} (no participation)",
                blocked_by="volume", gates=gates)

        # Sentiment VETO, not requirement: only a fresh, agreeing bearish read blocks.
        s = ctx.sentiment
        if s.is_tradeable and s.neg_prob >= self.params["veto_neg_prob"]:
            return EntryDecision(False, buy_prob,
                f"Vetoed by bearish news: neg={s.neg_prob:.2f} across {s.n_headlines} "
                f"headlines (agreement {s.agreement:.2f})",
                blocked_by="sentiment_veto", gates=gates)

        profile_min = ctx.__dict__.get("_min_buy_prob")
        if profile_min is not None and buy_prob < profile_min:
            return EntryDecision(False, buy_prob,
                f"buy_prob {buy_prob:.2f} below risk-dial entry bar {profile_min:.2f}",
                blocked_by="risk_dial", gates=gates)

        news_note = (f", news neg={s.neg_prob:.2f}/n={s.n_headlines}"
                     if s.n_headlines else ", no news (not required)")
        return EntryDecision(
            True, buy_prob,
            f"Momentum entry: trend={trend:.2f}, RSI={q.rsi:.1f}, vol={vol_ratio:.2f}x{news_note}",
            gates=gates,
        )

    def evaluate_exit(self, ctx: StrategyContext) -> ExitDecision:
        q = ctx.quant
        pos = ctx.position or {}
        entry = float(pos.get("avg_entry_price") or ctx.price)
        trend = self._trend_score(q, ctx.price)
        s = ctx.sentiment

        pnl_pct = (ctx.price - entry) / entry if entry > 0 else 0.0

        sell_prob = 0.0
        sell_prob += 0.45 * (1.0 - trend)
        if q and q.rsi is not None and q.rsi > 78:
            sell_prob += 0.20
        if s.is_tradeable:
            sell_prob += 0.35 * s.neg_prob
        sell_prob = round(min(max(sell_prob, 0.0), 1.0), 4)

        # Trend break is the primary exit for a momentum trade: the thesis is gone.
        if trend <= self.params["exit_trend_break"]:
            return ExitDecision(True, sell_prob, 1.0,
                f"Momentum thesis invalidated: trend score {trend:.2f} "
                f"<= {self.params['exit_trend_break']} (PnL {pnl_pct*100:+.2f}%)")

        if s.is_tradeable and s.neg_prob >= settings.MAX_SELL_SENTIMENT_NEG:
            return ExitDecision(True, sell_prob, 1.0,
                f"Bearish news exit: neg={s.neg_prob:.2f} across {s.n_headlines} headlines")

        close_prob = round(min(sell_prob * 0.85, 1.0), 4)
        return ExitDecision(False, sell_prob, close_prob,
                            f"Holding: trend {trend:.2f} intact, PnL {pnl_pct*100:+.2f}%")
