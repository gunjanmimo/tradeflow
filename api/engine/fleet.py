"""
The agent fleet. Each agent has one job, its own loop and cadence, and its own
status card; they hand work on through shared state, never by calling each other,
so one agent failing leaves the rest running.

  Scout          engine/discovery.py       every 60s   scans US stocks, UK/EU/Asia/India ADRs and
                                                       ETFs, and smart money (SEC, eToro, StockTwits)
                                                       into a scored candidate pool
  Curator        CuratorAgent              every 30s   moves the best candidates onto the watchlist,
                                                       drops ones that fell away; backfills their bars
  Trend analyst  TrendAnalystAgent         every 1s    reads 1-minute bars + daily bars for every
                                                       watched and held symbol (engine/trend.py);
                                                       logs trend flips and reversals
  Trader         engine/portfolio_manager  every 2s    trend-confirmed entry signals, ranked by
                                                       conviction and diversification, sized, bought
  Position mgr   PositionManagerAgent      every 2s    per held position, on the trend:
                                                       BUY more / SELL part / HOLD / CLOSE
  Sentinels      engine/sentinel_agent.py  every tick  stop, target, stale price, end of day

Every order still goes through the executor and its gates.
"""
import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from core.config import settings
from core.state import state, TradeDecision
from core.minute_bars import minute_bars
from engine.trend import board as trend_board, analyze

logger = logging.getLogger("tradeflow.fleet")


class Agent:
    name = "agent"
    role = ""

    def __init__(self):
        self.running = False
        self._task: Optional[asyncio.Task] = None
        self.cycles = 0
        self.last_at = 0.0
        self.last_ms = 0.0
        self.error: Optional[str] = None
        self.summary = "starting"
        self.last_action: Optional[Dict[str, Any]] = None

    @property
    def interval(self) -> float:
        return 5.0

    @property
    def enabled(self) -> bool:
        return True

    async def start(self):
        if not self.enabled:
            self.summary = "disabled"
            return
        self.running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()

    async def _loop(self):
        while self.running:
            t0 = time.perf_counter()
            try:
                await self.step()
                self.error = None
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
                logger.error(f"{self.name} failed: {e}", exc_info=True)
            self.cycles += 1
            self.last_at = time.time()
            self.last_ms = (time.perf_counter() - t0) * 1000
            try:
                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                break

    async def step(self):
        raise NotImplementedError

    def act(self, symbol: str, action: str, reason: str):
        self.last_action = {"symbol": symbol, "action": action, "reason": reason, "at": time.time()}

    def card(self) -> Dict[str, Any]:
        stalled = self.running and self.last_at and time.time() - self.last_at > max(15.0, 5 * self.interval)
        return {
            "name": self.name, "role": self.role,
            "state": "stalled" if stalled else ("running" if self.running else self.summary),
            "summary": self.summary, "cycles": self.cycles, "interval_s": self.interval,
            "last_at": self.last_at or None, "cycle_ms": round(self.last_ms, 1),
            "error": self.error, "last_action": self.last_action,
        }


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
        return sorted(set(state.watchlist) | set(state.active_positions))

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
    role = "Moves the scout's best candidates onto the watchlist and drops fading ones"

    @property
    def interval(self) -> float:
        return settings.CURATOR_INTERVAL_SECONDS

    async def step(self):
        from engine.discovery import discovery
        before = set(state.watchlist)
        await discovery._auto_select()
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

class PositionManagerAgent(Agent):
    """
    Trend-driven management of each open position, between the sentinel's
    tick-level exits:

      CLOSE  the trend has turned down with confidence
      SELL   a winner's trend is fading or reversing: sell TRIM_FRACTION, once
      BUY    a winner in a strong, confident uptrend, up SCALE_IN_MIN_R x its
             risk: add SCALE_IN_FRACTION of the original investment, once
      HOLD   otherwise

    The trend CLOSE waits out an equity's minimum hold, as the other discretionary
    exits do; the stop protects the position meanwhile. Trims and adds do not.
    """
    name = "Position manager"
    role = "Decides BUY more / SELL part / HOLD / CLOSE for each open position from its trend"

    def __init__(self):
        super().__init__()
        self._logged: Dict[tuple, float] = {}

    @property
    def interval(self) -> float:
        return settings.POSITION_MANAGER_INTERVAL_SECONDS

    @property
    def enabled(self) -> bool:
        return settings.POSITION_MANAGER_ENABLED

    def decide(self, sym: str, pos: Dict[str, Any], r) -> tuple:
        """(action, reason, fraction) for one position. Pure: no orders."""
        price = float(pos.get("current_price") or pos.get("avg_entry_price") or 0.0)
        entry = float(pos.get("avg_entry_price") or price)
        qty = float(pos.get("qty") or 0.0)
        if not r.ready or entry <= 0 or qty <= 0:
            return "HOLD", "learning the trend: " + (r.reasons[0] if r.reasons else ""), 0.0
        pnl_pct = (price - entry) / entry
        stop = pos.get("stop_loss")
        risk_ps = (entry - float(stop)) if stop is not None and float(stop) < entry else entry * 0.01
        r_mult = (price - entry) / risk_ps if risk_ps > 0 else 0.0

        in_min_hold = False
        if pos.get("opened_at"):
            in_min_hold = (time.time() - float(pos["opened_at"])) / 60 < settings.STOCK_SCORE_MIN_HOLD_MINUTES
        trend_txt = f"trend {r.label} {r.direction:+.2f} (conf {r.confidence:.2f})"

        if r.direction <= -settings.TREND_EXIT_DIRECTION and r.confidence >= 0.5 and not in_min_hold:
            return "CLOSE", f"Trend turned down: {trend_txt}, P&L {pnl_pct * 100:+.2f}%", 1.0
        if (pnl_pct > 0 and not pos.get("trimmed")
                and (r.reversal_down or r.direction <= -settings.TREND_TRIM_DIRECTION)):
            why = "micro trend reversing" if r.reversal_down else "trend fading"
            return ("SELL", f"Locking gains, {why}: {trend_txt}, P&L {pnl_pct * 100:+.2f}%",
                    settings.TRIM_FRACTION)
        if (settings.SCALE_IN_ENABLED and not pos.get("scaled_in")
                and r.direction >= settings.SCALE_IN_DIRECTION and r.confidence >= 0.6
                and not r.reversal_down and r_mult >= settings.SCALE_IN_MIN_R):
            return "BUY", f"Adding to a winner at {r_mult:.1f}R: {trend_txt}", settings.SCALE_IN_FRACTION
        if in_min_hold and r.direction <= -settings.TREND_EXIT_DIRECTION:
            return "HOLD", f"{trend_txt}, but inside the minimum hold; the stop protects it", 0.0
        return "HOLD", f"{trend_txt}, P&L {pnl_pct * 100:+.2f}%", 0.0

    async def step(self):
        from engine.executor import executor
        counts = {"HOLD": 0, "BUY": 0, "SELL": 0, "CLOSE": 0}
        for sym, pos in list(state.active_positions.items()):
            if sym in executor.closing_orders or sym in executor.pending_exits or sym in executor.pending_orders:
                continue
            r = trend_board.read(sym)
            action, reason, fraction = self.decide(sym, pos, r)
            counts[action] += 1
            pos["fleet_action"] = action
            pos["fleet_reason"] = reason
            pos["trend"] = r.brief()
            if action == "HOLD":
                continue
            if action == "BUY" and not state.is_trading_active:
                continue
            # The executor may refuse quietly (outside regular hours, at a cap) and
            # the same call comes back every cycle: log it once a minute, not each time.
            now = time.time()
            if now - self._logged.get((sym, action), 0.0) >= 60.0:
                self._logged[(sym, action)] = now
                self.act(sym, action, reason)
                state.log_event("POSITION_MGR", f"{action} {sym}: {reason}")
            await executor.execute_decision(TradeDecision(
                symbol=sym, action=action, close=action == "CLOSE", fraction=fraction,
                close_prob=1.0 if action == "CLOSE" else 0.0, reason=f"[Position manager] {reason}"))
        n = len(state.active_positions)
        self.summary = (f"{n} position{'s' if n != 1 else ''}: "
                        + ", ".join(f"{v} {k}" for k, v in counts.items() if v) if n else "no open positions")


# ---------------------------------------------------------------------------

class Fleet:
    def __init__(self):
        self.trend = TrendAnalystAgent()
        self.curator = CuratorAgent()
        self.positions = PositionManagerAgent()

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
        await portfolio_manager.start()
        await self.positions.start()

    async def stop(self):
        from engine.portfolio_manager import portfolio_manager
        for a in (self.positions, self.curator, self.trend):
            await a.stop()
        await portfolio_manager.stop()

    def snapshot(self) -> Dict[str, Any]:
        from engine.discovery import discovery
        from engine.portfolio_manager import portfolio_manager
        from engine.sentinel_agent import sentinel_registry
        d = discovery.status()
        scout = {
            "name": "Scout", "role": "Scans markets and smart money into a scored candidate pool",
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
        return {
            "agents": [scout, self.curator.card(), self.trend.card(), trader,
                       self.positions.card(), sentinels],
            "trends": {s: r.brief() for s, r in trend_board.reads.items()},
            "bars": minute_bars.status(),
        }


fleet = Fleet()
