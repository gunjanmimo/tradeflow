"""
Stock-score strategy -- the equity default.

Our own per-stock score, so stock trading does not hinge on any outside source:

  daily momentum     35%  12-1m momentum and 3m strength vs its sector ETF, ranked
                          across the whole discovery pool; below the 200-day
                          average is penalised (computed from Alpaca daily bars)
  intraday structure 25%  EMA stack + RSI on the live feed
  smart money        25%  SEC Form 4 / eToro consensus -- votes ONLY when a real
                          source covers the stock
  news               15%  Laya-scored headlines, shrunk toward neutral when they
                          are few or disagree -- votes ONLY when there is news

Daily and intraday always vote, so every stock with price history has a score.
External sources move it when they have something to say and are silent
otherwise, rather than blocking trades by their absence (the old News Catalyst
default needed two agreeing headlines and so almost never traded).

Vetoes: below the 200-day average, insiders clearly selling, clearly negative
news, intraday downtrend, overbought RSI, wide spread.

Exits are deliberately slow. Stop and target always apply, but a discretionary
exit needs the score to fall below STOCK_SCORE_EXIT_BELOW -- which takes the
daily picture weakening, not one bad minute -- or the news turning, and never
inside the minimum hold. Flipping in and out on intraday noise only pays costs.
"""
import time
from typing import Any, Dict, Optional

from core.config import settings
from engine.strategies.base import (
    Strategy, StrategyContext, EntryDecision, ExitDecision,
)

WEIGHTS = {"daily": 0.35, "intraday": 0.25, "smart_money": 0.25, "news": 0.15}


def score_stock(ctx: StrategyContext) -> Dict[str, Any]:
    """Composite 0..1 score plus the components that produced it."""
    from engine.discovery import discovery

    comps: Dict[str, float] = {}
    cand = discovery.candidates.get(ctx.symbol) or {}
    daily = (cand.get("components") or {}).get("momentum")
    above_200d = (cand.get("momentum") or {}).get("above_200d")
    if daily is not None:
        comps["daily"] = float(daily)

    trend = Strategy._trend_score(ctx.quant, ctx.price)
    comps["intraday"] = 0.6 * trend + 0.4 * Strategy._rsi_score(ctx.quant)

    smart: Optional[float] = ctx.consensus
    if smart is None:
        smart = (cand.get("components") or {}).get("smart_money")
    if smart is not None:
        comps["smart_money"] = float(smart)

    s = ctx.sentiment
    if s is not None and s.n_headlines > 0 and not s.is_stale:
        tone = 0.5 + (s.pos_prob - s.neg_prob) / 2
        trust = min(1.0, s.n_headlines / 4) * s.agreement
        comps["news"] = 0.5 + (tone - 0.5) * trust

    w = sum(WEIGHTS[k] for k in comps)
    score = sum(WEIGHTS[k] * v for k, v in comps.items()) / w
    return {
        "score": round(score, 4),
        "components": {k: round(v, 3) for k, v in comps.items()},
        "trend": trend,
        "above_200d": above_200d,
        "discovery_score": cand.get("score"),
    }


class StockScoreStrategy(Strategy):
    name = "stock_score"
    display_name = "TradeFlow Stock Score"
    description = (
        "Trades equities on our own composite score: daily momentum and intraday "
        "structure always vote; smart money (SEC insiders, eToro) and news vote when "
        "they have real data. Vetoes on bearish insiders, negative news, downtrends "
        "and overbought RSI. Exits slowly to avoid churn."
    )
    requires_sentiment = False
    source = "TradeFlow"

    params = {
        "max_rsi": 74.0,
        "min_trend_score": 0.35,
        "max_spread_pct": 0.004,
        "bearish_smart_money": 0.30,   # consensus at/below this = insiders selling
        "veto_neg_prob": 0.55,
    }

    def evaluate_entry(self, ctx: StrategyContext) -> EntryDecision:
        from core.market_hours import us_session, PRE
        r = score_stock(ctx)
        q = ctx.quant
        s = ctx.sentiment
        score = r["score"]
        gates = {
            "stock_score": score,
            "components": r["components"],
            "discovery_score": r["discovery_score"],
            "above_200d": r["above_200d"],
            "trend_score": round(r["trend"], 3),
            "rsi": q.rsi if q else None,
            "spread_pct": round((q.spread if q else 0) * 100, 4),
            "consensus": ctx.consensus,
        }
        parts = ", ".join(f"{k} {v:.2f}" for k, v in r["components"].items())

        if "daily" not in r["components"]:
            return EntryDecision(False, score,
                "No daily history scored yet for this stock (discovery computes it each minute)",
                blocked_by="no_daily", gates=gates)
        if r["above_200d"] is False:
            return EntryDecision(False, score, "Below its 200-day average: long-term trend is down",
                                 blocked_by="below_200d", gates=gates)
        if ctx.consensus is not None and ctx.consensus <= self.params["bearish_smart_money"]:
            return EntryDecision(False, score,
                f"Smart money is bearish (consensus {ctx.consensus:.2f}): insiders selling",
                blocked_by="smart_money_bearish", gates=gates)
        if s is not None and s.is_tradeable and s.neg_prob >= self.params["veto_neg_prob"]:
            return EntryDecision(False, score,
                f"Negative news: neg={s.neg_prob:.2f} across {s.n_headlines} headlines",
                blocked_by="negative_news", gates=gates)
        if r["trend"] < self.params["min_trend_score"]:
            return EntryDecision(False, score,
                f"Intraday trend {r['trend']:.2f} opposes the entry", blocked_by="trend", gates=gates)
        if q and q.rsi is not None and q.rsi > self.params["max_rsi"]:
            return EntryDecision(False, score, f"RSI {q.rsi:.1f} overbought", blocked_by="rsi", gates=gates)

        max_spread = (settings.PREMARKET_MAX_SPREAD_PCT / 100.0 if us_session() == PRE
                      else self.params["max_spread_pct"])
        if q and q.spread > max_spread:
            return EntryDecision(False, score, f"Spread {q.spread*100:.3f}% too wide",
                                 blocked_by="spread", gates=gates)

        bar = max(settings.STOCK_SCORE_MIN_ENTRY, ctx.__dict__.get("_min_buy_prob") or 0.0)
        if score < bar:
            return EntryDecision(False, score, f"Stock score {score:.2f} below entry bar {bar:.2f} ({parts})",
                                 blocked_by="score", gates=gates)

        return EntryDecision(True, score, f"Stock score {score:.2f} ({parts})", gates=gates)

    def evaluate_exit(self, ctx: StrategyContext) -> ExitDecision:
        r = score_stock(ctx)
        score = r["score"]
        s = ctx.sentiment
        pos = ctx.position or {}
        entry = float(pos.get("avg_entry_price") or ctx.price)
        pnl_pct = (ctx.price - entry) / entry if entry > 0 else 0.0
        # Capped at 0.5: this strategy's own read never trips the sentinel's
        # "strong sell" threshold on its own; it exits through should_close.
        sell_prob = round(min(0.5, 0.5 * (1.0 - score)), 4)
        close_prob = round(sell_prob * 0.8, 4)

        opened_at = pos.get("opened_at")
        held_min = (time.time() - float(opened_at)) / 60 if opened_at else float("inf")
        if held_min < settings.STOCK_SCORE_MIN_HOLD_MINUTES:
            return ExitDecision(False, sell_prob, close_prob,
                f"Holding: {held_min:.0f}/{settings.STOCK_SCORE_MIN_HOLD_MINUTES:.0f} min minimum hold, "
                f"score {score:.2f}, PnL {pnl_pct*100:+.2f}%")

        if s is not None and s.is_tradeable and s.neg_prob >= self.params["veto_neg_prob"]:
            return ExitDecision(True, sell_prob, 1.0,
                f"News turned negative: neg={s.neg_prob:.2f} across {s.n_headlines} headlines")
        if score < settings.STOCK_SCORE_EXIT_BELOW:
            return ExitDecision(True, sell_prob, 1.0,
                f"Stock score fell to {score:.2f} (< {settings.STOCK_SCORE_EXIT_BELOW}); thesis gone")
        return ExitDecision(False, sell_prob, close_prob,
                            f"Holding: score {score:.2f}, PnL {pnl_pct*100:+.2f}%")
