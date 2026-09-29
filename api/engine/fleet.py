"""
The agent fleet. Each agent has one job, its own loop and cadence, and its own
status card; they hand work on through shared state, never by calling each other,
so one agent failing leaves the rest running.

  Scout          scout/service.py          every hour  ranks the stocks worth watching today (past
                                                       performance, today's move, news, Reddit and
                                                       StockTwits) and puts the top picks on the watchlist
  Watcher        scout/watcher.py          every 5s    follows each pick live; a pick is READY once its
                                                       confidence holds above the entry bar
  Pool           engine/discovery.py       every 60s   scores smart money (SEC, eToro) and the curated
                                                       universe for manual picks and the risk report
  Curator        CuratorAgent              every 30s   puts smart-money buys on the watchlist (and the
                                                       pool's best when DISCOVERY_AUTO_PROMOTE is on)
  Trend analyst  TrendAnalystAgent         every 1s    reads 1-minute bars + daily bars for every
                                                       watched and held symbol (engine/trend.py);
                                                       logs trend flips and reversals
  Trader         engine/portfolio_manager  every 2s    the strategy's entry signals, ranked by
                                                       conviction and diversification, sized, bought
  Sentinels      engine/sentinel_agent.py  every tick  stop, target, strategy exit, stale price,
                                                       end of day
  Learner        engine/learner.py         after close retrains the RL policy on the day's bars;
                                                       deploys it only if it beats the current one

Every order still goes through the executor and its gates.
"""
import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from core.config import settings
from core.state import state, TradeDecision
from core.minute_bars import minute_bars
from engine.agent import Agent
from engine.trend import board as trend_board, analyze

logger = logging.getLogger("tradeflow.fleet")


# ---------------------------------------------------------------------------

class TrendAnalystAgent(Agent):
    name = "Trend analyst"
    role = "Reads 1-minute and daily time series for every watched and held symbol"

    def __init__(self):
        super().__init__()
        self._labels: Dict[str, str] = {}
        self._flip_logged: Dict[str, float] = {}
        self._backfill_at = 0.0

    @property
    def interval(self) -> float:
        return settings.TREND_INTERVAL_SECONDS

    def symbols(self) -> List[str]:
        return sorted(set(state.watchlist) | set(state.active_positions) | set(settings.CONTEXT_SYMBOLS))

    async def step(self):
        syms = self.symbols()
        now = time.time()
        # Backfill history for anything new, off the event loop, at most once a
        # minute; the reads below use whatever bars exist meanwhile.
        if now - self._backfill_at >= 60.0 or any(minute_bars.count(s) == 0 for s in syms):
            if now - self._backfill_at >= 10.0:
                self._backfill_at = now
                asyncio.create_task(minute_bars.ensure(syms))

        counts = {"uptrend": 0, "downtrend": 0, "range": 0, "unknown": 0}
        for i, sym in enumerate(syms):
            r = analyze(sym)
            trend_board.reads[sym] = r
            counts[r.label] += 1
            self._note_change(sym, r, now)
            if i % 10 == 9:
                await asyncio.sleep(0)       # yield: never hold the loop for a whole sweep
        for sym in [s for s in trend_board.reads if s not in syms]:
            trend_board.reads.pop(sym, None)
            self._labels.pop(sym, None)
        self.summary = (f"{len(syms)} symbols: {counts['uptrend']} up, {counts['downtrend']} down, "
                        f"{counts['range']} range, {counts['unknown']} learning")

    def _note_change(self, sym: str, r, now: float):
        prev = self._labels.get(sym)
        self._labels[sym] = r.label
        if not r.ready or prev is None or prev == "unknown":
            return
        flipped = prev != r.label and r.label != "range"
        if (flipped or r.reversal_down or r.reversal_up) and now - self._flip_logged.get(sym, 0.0) >= 60.0:
            self._flip_logged[sym] = now
            what = (f"{prev} -> {r.label}" if flipped
                    else "reversal down" if r.reversal_down else "reversal up")
            self.act(sym, what, r.reasons[-1])
            state.log_event("TREND", f"{sym}: {what} (direction {r.direction:+.2f}, "
                                     f"confidence {r.confidence:.2f}; {'; '.join(r.reasons)})")


# ---------------------------------------------------------------------------

class CuratorAgent(Agent):
    name = "Curator"
    role = "Puts smart-money buys (and the pool's best, if enabled) on the watchlist"

    @property
    def interval(self) -> float:
        return settings.CURATOR_INTERVAL_SECONDS

    def __init__(self):
        super().__init__()
        self.smart_added: set = set()      # symbols this agent added as smart-money buys

    async def _smart_money(self):
        """Tradable smart-money BUYs go on the watchlist; ones that stop qualifying come off."""
        if not settings.SMART_MONEY_TRADING:
            return
        from engine.smart_money import smart_money
        from feeds.daily_bars import daily_bars
        from feeds.alpaca_stream import market_stream
        syms = [s for s in smart_money.symbols() if smart_money.verdict(s)["verdict"] == "BUY"]
        if syms:
            await daily_bars.ensure(syms)          # price and liquidity for the tradable check
        buys = set(smart_money.buy_list())
        for sym in sorted(buys - set(state.watchlist)):
            state.watchlist.add(sym)
            self.smart_added.add(sym)
            await market_stream.ensure_stock_subscription(sym)
            state.log_event("SMART_MONEY", f"{sym} added to the watchlist: "
                                           + "; ".join(smart_money.verdict(sym)["reasons"]))
        for sym in sorted(self.smart_added - buys):
            self.smart_added.discard(sym)
            if sym in state.watchlist and sym not in state.active_positions:
                state.watchlist.discard(sym)
                state.log_event("SMART_MONEY", f"{sym} removed from the watchlist: no longer a tradable buy")

    async def step(self):
        from engine.discovery import discovery
        before = set(state.watchlist)
        await discovery._auto_select()
        await self._smart_money()
        added = sorted(set(state.watchlist) - before)
        dropped = sorted(before - set(state.watchlist))
        if added:
            # Learn the new symbols' trend before the trader may consider them.
            await minute_bars.ensure(added)
            self.act(", ".join(added), "added", "top-ranked discovery candidates")
        if dropped:
            self.act(", ".join(dropped), "dropped", "no longer a top pick")
        self.summary = (f"watchlist {len(state.watchlist)} ({len(discovery.auto_symbols)} auto-picked)"
                        + (f"; +{len(added)}" if added else "") + (f" -{len(dropped)}" if dropped else ""))


# ---------------------------------------------------------------------------

class Fleet:
    def __init__(self):
        from engine.learner import learner
        from scout.service import scout
        from scout.watcher import watcher
        self.trend = TrendAnalystAgent()
        self.curator = CuratorAgent()
        self.scout = scout
        self.watcher = watcher
        self.learner = learner

    async def start(self):
        from engine.portfolio_manager import portfolio_manager
        # Understand the market before anything acts: backfill minute bars (bounded,
        # so a slow data API cannot hold up start-up) and read every trend once.
        try:
            await asyncio.wait_for(minute_bars.ensure(self.trend.symbols()), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning("Minute-bar backfill still running after 30s; continuing")
        await self.trend.step()
        self.trend._backfill_at = time.time()
        await self.trend.start()
        await self.curator.start()
        await self.scout.start()
        await self.watcher.start()
        from desk.desk import desk
        await desk.start()
        from feeds.spreads import spreads
        await spreads.start()
        await portfolio_manager.start()
        await self.learner.start()

    async def stop(self):
        from engine.portfolio_manager import portfolio_manager
        for a in (self.learner, self.watcher, self.scout, self.curator, self.trend):
            await a.stop()
        from desk.desk import desk
        await desk.stop()
        from feeds.spreads import spreads
        await spreads.stop()
        await portfolio_manager.stop()

    def snapshot(self) -> Dict[str, Any]:
        from engine.discovery import discovery
        from engine.portfolio_manager import portfolio_manager
        from engine.sentinel_agent import sentinel_registry
        d = discovery.status()
        pool = {
            "name": "Pool", "role": "Scores smart money and the curated universe for manual picks",
            "state": ("running" if d["enabled"] else "disabled") if not d["last_error"] else "error",
            "summary": f"{d['candidates']} candidates, {d['scored']} scored",
            "cycles": None, "interval_s": settings.DISCOVERY_INTERVAL_SECONDS,
            "last_at": d["last_cycle_at"], "cycle_ms": d["last_cycle_ms"],
            "error": d["last_error"], "last_action": None,
        }
        m = portfolio_manager.snapshot()
        trader = {
            "name": "Trader", "role": "Buys trend-confirmed signals, ranked by conviction and diversification",
            "state": m["state"], "summary": m["message"], "cycles": m["cycles"],
            "interval_s": m["interval_s"], "last_at": m["last_cycle_at"], "cycle_ms": m["cycle_ms"],
            "error": m["error"],
            "last_action": ({"symbol": m["last_entry"]["symbol"], "action": "BUY",
                             "reason": f"score {m['last_entry']['score']:.2f}", "at": m["last_entry"]["at"]}
                            if m.get("last_entry") else None),
        }
        sentinels = {
            "name": "Sentinels", "role": "One per position: stop, target, stale price, end of day, every tick",
            "state": "running", "summary": f"{len(sentinel_registry.to_dict())} on duty",
            "cycles": None, "interval_s": 0, "last_at": None, "cycle_ms": None,
            "error": None, "last_action": None,
        }
        from desk.desk import desk
        desk_cards = [{"name": f"Desk {a['name'].lower()}",
                       "role": {"Observer": "Watches every buy signal before anyone decides",
                                "Analyst": f"Reasons over the case file ({a['model']})",
                                "Critic": f"Argues against the trade, may veto ({a['model']})",
                                "Decision": "Approves only when both agree and the odds beat breakeven"}[a["name"]],
                       "state": a["state"], "summary": a["summary"], "cycles": None, "interval_s": 1.0,
                       "last_at": None, "cycle_ms": None, "error": None, "last_action": None}
                      for a in desk.agents()] if desk.enabled else []
        return {
            "agents": [self.scout.card(), self.watcher.card(), pool, self.curator.card(),
                       self.trend.card(), trader, *desk_cards, sentinels, self.learner.card()],
            "trends": {s: r.brief() for s, r in trend_board.reads.items()},
            "bars": minute_bars.status(),
        }


fleet = Fleet()
