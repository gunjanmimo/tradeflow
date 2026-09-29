"""
Scout day trade: buy a scout pick once the watcher is confident enough.

Routed automatically to every scout pick (scout/service.py). The watcher agent
(scout/watcher.py) owns the judgement; this strategy turns it into entries and
exits the portfolio manager executes through the usual gates:

Entry: the watcher's read is fresh and READY (confidence at or above
SCOUT_ENTRY_CONFIDENCE, unblocked, for SCOUT_CONFIRM_SECONDS), and the symbol has
had fewer than SCOUT_MAX_ENTRIES_PER_DAY scout entries today.
Exit: the normal stop and target, flat by the close like every day trade, or --
after SCOUT_MIN_HOLD_MINUTES -- once the confidence falls below
SCOUT_EXIT_CONFIDENCE or the trend turns down with price under VWAP.

Trades are recorded with entry_strategy "scout", and every hourly ranking is
logged, so `python -m scout scorecard` can say whether any of this earns money.
"""
import time

from core.config import settings
from engine.strategies.base import Strategy, StrategyContext, EntryDecision, ExitDecision


def entries_today(symbol: str) -> int:
    """
    Buys of this symbol today, from the broker's fills (state.recent_trades is
    reloaded from Alpaca at every start-up) -- not from in-memory trade records,
    which a restart wiped: IOVA was bought twice on 2026-09-29 that way.
    """
    from core.state import state, ny_date
    today = ny_date()
    fills = sum(1 for t in list(state.recent_trades)
                if t.get("symbol") == symbol and t.get("side") == "BUY"
                and ny_date(float(t.get("time") or 0)) == today)
    pos = state.active_positions.get(symbol)
    held = 1 if pos and ny_date(float(pos.get("opened_at") or 0)) == today else 0
    return max(fills, held)


class ScoutStrategy(Strategy):
    name = "scout"
    display_name = "Scout picks (watcher-confirmed)"
    description = ("Day-trades the scout's hourly picks -- ranked on past performance, today's move, "
                   "news and public discussion -- once the watcher's live confidence has held above "
                   "the entry bar long enough. One entry per stock per day, flat by the close.")
    source = "Alpaca screener and news, Reddit (ApeWisdom), StockTwits"

    @property
    def params(self):
        return {"entry_confidence": settings.SCOUT_ENTRY_CONFIDENCE,
                "exit_confidence": settings.SCOUT_EXIT_CONFIDENCE,
                "confirm_seconds": settings.SCOUT_CONFIRM_SECONDS,
                "min_hold_minutes": settings.SCOUT_MIN_HOLD_MINUTES,
                "max_entries_per_day": settings.SCOUT_MAX_ENTRIES_PER_DAY}

    def evaluate_entry(self, ctx: StrategyContext) -> EntryDecision:
        from scout.watcher import watcher
        r = watcher.read(ctx.symbol)
        if r is None:
            return EntryDecision(False, 0.0, "Watcher has no fresh read on this pick", blocked_by="no_watch")
        blocks = r.get("blocks") or []
        gates = {"confidence": r.get("confidence"), "parts": r.get("parts"), "blocks": blocks,
                 "status": r.get("status"), "rank": r.get("rank"), "scout_score": r.get("scout_score"),
                 "confirmed_for_s": r.get("confirmed_for_s", 0.0)}
        conf = float(r.get("confidence") or 0.0)
        if not settings.SCOUT_TRADING:
            return EntryDecision(False, conf, "Scout trading is switched off (SCOUT_TRADING)",
                                 blocked_by="scout_off", gates=gates)
        if r.get("status") != "ready":
            why = blocks[0] if blocks else (
                f"confidence {conf:.2f} < {settings.SCOUT_ENTRY_CONFIDENCE:.2f}"
                if conf < settings.SCOUT_ENTRY_CONFIDENCE else r.get("status"))
            return EntryDecision(False, conf, f"Watcher: {why}",
                                 blocked_by="watcher_block" if blocks else "not_confident", gates=gates)
        if entries_today(ctx.symbol) >= settings.SCOUT_MAX_ENTRIES_PER_DAY:
            return EntryDecision(False, conf, "Already traded today", blocked_by="once_per_day", gates=gates)
        return EntryDecision(True, conf,
                             f"Scout pick #{r.get('rank')} confirmed: confidence {conf:.2f} for "
                             f"{r.get('confirmed_for_s', 0.0):.0f}s ({'; '.join(r.get('reasons') or [])})", gates=gates)

    def evaluate_exit(self, ctx: StrategyContext) -> ExitDecision:
        from scout.watcher import watcher
        pos = ctx.position or {}
        held_min = (time.time() - float(pos.get("opened_at") or time.time())) / 60.0
        r = watcher.read(ctx.symbol)
        if r is None:
            return ExitDecision(False, 0.0, 0.0, "No fresh watcher read; stop, target and the close govern")
        if r.get("status") == "no price":
            # A feed gap is not a verdict: the stop, target and stale-price exits cover it.
            return ExitDecision(False, 0.0, 0.0, "Watcher has no price this moment; stop, target and the close govern")
        conf = float(r.get("confidence") or 0.0)
        sell = round(max(0.0, 1.0 - conf), 4)
        if held_min < settings.SCOUT_MIN_HOLD_MINUTES:
            return ExitDecision(False, sell, 0.0, f"Held {held_min:.0f} min; confidence {conf:.2f}")
        if conf < settings.SCOUT_EXIT_CONFIDENCE:
            return ExitDecision(True, sell, 1.0, f"Watcher confidence fell to {conf:.2f} "
                                                 f"(< {settings.SCOUT_EXIT_CONFIDENCE:.2f}): "
                                                 + "; ".join(r.get("reasons") or []))
        vwap = r.get("vwap")
        if "in a downtrend" in (r.get("blocks") or []) and vwap and ctx.price < vwap:
            return ExitDecision(True, sell, 1.0, f"Trend turned down with price under VWAP {vwap:,.2f}")
        return ExitDecision(False, sell, round(sell * 0.5, 4), f"Pick intact: confidence {conf:.2f}")
