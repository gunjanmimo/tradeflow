"""
News-catalyst strategy -- the equity default.

Equities move around discrete, datable events (earnings, guidance, verdicts,
analyst actions) during fixed hours, and Alpaca's news feed covers them densely.
That makes real sentiment the primary edge here, which is the opposite of the
crypto case.

Unlike the old engine, this strategy demands EVIDENCE rather than a single
probability: several fresh headlines that mostly agree. Laya's own model card
reports it ships over-confident (ECE 0.466 before temperature scaling), so one
0.95 reading is not trustworthy on its own -- cross-headline agreement is.
"""
from core.config import settings
from engine.strategies.base import (
    Strategy, StrategyContext, EntryDecision, ExitDecision,
)


class NewsCatalystStrategy(Strategy):
    name = "news_catalyst"
    display_name = "News Catalyst"
    description = (
        "Trades equities on fresh, corroborated news sentiment. Requires multiple "
        "agreeing headlines within the freshness window, and refuses to enter "
        "against the prevailing trend. Abstains entirely when news is absent."
    )
    applies_to = "equity"
    requires_sentiment = True

    params = {
        "min_pos_prob": 0.62,          # aggregated bullish probability
        "min_agreement": 0.66,         # fraction of headlines agreeing
        "min_headlines": 2,            # corroboration requirement
        "max_neg_prob": 0.30,          # reject mixed/negative coverage
        "min_trend_score": 0.35,       # do not buy into a clear downtrend
        "max_rsi": 74.0,
        "max_spread_pct": 0.004,
        "exit_neg_prob": 0.55,         # sentiment flip closes the trade
    }

    def evaluate_entry(self, ctx: StrategyContext) -> EntryDecision:
        q = ctx.quant
        s = ctx.sentiment
        gates = {
            "sentiment_pos": s.pos_prob,
            "sentiment_neg": s.neg_prob,
            "n_headlines": s.n_headlines,
            "agreement": s.agreement,
            "is_stale": s.is_stale,
            "sources": list(s.sources),
            "rsi": q.rsi if q else None,
            "spread_pct": round((q.spread if q else 0) * 100, 4),
        }

        trend = self._trend_score(q, ctx.price)
        rsi_s = self._rsi_score(q)
        gates["trend_score"] = round(trend, 3)

        # Conviction is sentiment-led, structure-confirmed.
        buy_prob = 0.55 * s.pos_prob + 0.25 * trend + 0.20 * rsi_s
        if ctx.consensus is not None:
            buy_prob += (ctx.consensus - 0.5) * 0.20
            gates["consensus"] = ctx.consensus
        else:
            gates["consensus"] = None
        buy_prob = round(min(max(buy_prob, 0.0), 1.0), 4)

        # --- evidence gates (this strategy's whole point) ---
        if s.is_stale or s.n_headlines == 0:
            return EntryDecision(False, buy_prob,
                "No fresh news: this strategy trades catalysts only and abstains without them",
                blocked_by="no_news", gates=gates)

        if s.n_headlines < self.params["min_headlines"]:
            return EntryDecision(False, buy_prob,
                f"Only {s.n_headlines} headline(s); need {self.params['min_headlines']} "
                f"for corroboration (a single headline is a rumour)",
                blocked_by="insufficient_corroboration", gates=gates)

        if s.agreement < self.params["min_agreement"]:
            return EntryDecision(False, buy_prob,
                f"Headlines disagree: agreement {s.agreement:.2f} < {self.params['min_agreement']} "
                f"(mixed coverage is not a signal)",
                blocked_by="disagreement", gates=gates)

        if s.pos_prob < self.params["min_pos_prob"]:
            return EntryDecision(False, buy_prob,
                f"Bullish sentiment {s.pos_prob:.2f} below {self.params['min_pos_prob']}",
                blocked_by="weak_sentiment", gates=gates)

        if s.neg_prob > self.params["max_neg_prob"]:
            return EntryDecision(False, buy_prob,
                f"Bearish component {s.neg_prob:.2f} too high (> {self.params['max_neg_prob']})",
                blocked_by="negative_sentiment", gates=gates)

        # --- structure gates ---
        if trend < self.params["min_trend_score"]:
            return EntryDecision(False, buy_prob,
                f"Trend score {trend:.2f} opposes the entry (< {self.params['min_trend_score']})",
                blocked_by="trend", gates=gates)

        if q and q.rsi is not None and q.rsi > self.params["max_rsi"]:
            return EntryDecision(False, buy_prob,
                f"RSI {q.rsi:.1f} overbought (> {self.params['max_rsi']})",
                blocked_by="rsi", gates=gates)

        if q and q.spread > self.params["max_spread_pct"]:
            return EntryDecision(False, buy_prob,
                f"Spread {q.spread*100:.3f}% too wide",
                blocked_by="spread", gates=gates)

        profile_min = ctx.__dict__.get("_min_buy_prob")
        if profile_min is not None and buy_prob < profile_min:
            return EntryDecision(False, buy_prob,
                f"buy_prob {buy_prob:.2f} below risk-dial entry bar {profile_min:.2f}",
                blocked_by="risk_dial", gates=gates)

        return EntryDecision(
            True, buy_prob,
            f"News catalyst: pos={s.pos_prob:.2f} across {s.n_headlines} headlines "
            f"(agreement {s.agreement:.2f}, sources: {', '.join(s.sources) or 'n/a'}), "
            f"trend={trend:.2f}",
            gates=gates,
        )

    def evaluate_exit(self, ctx: StrategyContext) -> ExitDecision:
        q = ctx.quant
        s = ctx.sentiment
        pos = ctx.position or {}
        entry = float(pos.get("avg_entry_price") or ctx.price)
        pnl_pct = (ctx.price - entry) / entry if entry > 0 else 0.0
        trend = self._trend_score(q, ctx.price)

        sell_prob = 0.50 * s.neg_prob + 0.30 * (1.0 - trend)
        if q and q.rsi is not None and q.rsi > 78:
            sell_prob += 0.20
        sell_prob = round(min(max(sell_prob, 0.0), 1.0), 4)

        # The catalyst reversing is the thesis breaking.
        if s.is_tradeable and s.neg_prob >= self.params["exit_neg_prob"]:
            return ExitDecision(True, sell_prob, 1.0,
                f"Catalyst reversed: neg={s.neg_prob:.2f} across {s.n_headlines} "
                f"headlines (agreement {s.agreement:.2f})")

        # Sentiment going stale on a news trade is itself a reason to reduce:
        # the edge was the catalyst, and the catalyst is no longer current.
        if s.is_stale and pnl_pct > 0:
            return ExitDecision(True, sell_prob, 0.85,
                f"Catalyst gone stale with profit locked ({pnl_pct*100:+.2f}%); "
                f"taking the trade off rather than holding without an edge",
                close_fraction=0.5)

        close_prob = round(min(sell_prob * 0.85, 1.0), 4)
        return ExitDecision(False, sell_prob, close_prob,
                            f"Holding: catalyst intact (neg={s.neg_prob:.2f}), PnL {pnl_pct*100:+.2f}%")
