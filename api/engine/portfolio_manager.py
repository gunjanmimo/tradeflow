"""
Portfolio manager: the one agent that decides what the bots buy.

Entries used to be first-come. Every price tick of every watchlist symbol ran its
own decision and fired a buy the instant its own gates passed, so nothing compared
candidates, nothing asked whether the book was already one sector, and nothing
noticed that budget was sitting idle -- or said why.

Now each cycle (MANAGER_INTERVAL_SECONDS) the manager:

  1. evaluates every watchlist symbol we do not hold with its strategy
  2. pre-checks the risk guard (halts, caps, session, close-of-day cutoff)
  3. ranks the survivors by conviction and by how much a position would spread
     the book RIGHT NOW (engine/diversification.py fit): a new sector, an
     under-target region, a defensive shortfall and low correlation rank higher
  4. deploys into the best, re-ranking after each entry because the book changed,
     up to the free position slots and the remaining budget
  5. records what it did, and when budget stays idle, exactly why

Sizing, stops and every final gate stay in the executor: the manager chooses
WHICH symbol and WHEN, never how much. If the manager stalls, the tick path takes
entries back (see owns_entries), so a manager fault can never leave the bots blind.
"""
import asyncio
import logging
import time
from collections import Counter, deque
from typing import Any, Dict, List, Optional

from core.config import settings
from core.state import state
from engine.decision_engine import decision_engine
from engine.risk_guard import risk_guard
from core.universe import universe
from engine.diversification import diversification
from engine.trend import board as trend_board

logger = logging.getLogger("tradeflow.manager")

MAX_PICKS_SHOWN = 8
WATCH_HISTORY_SECONDS = 330.0    # enough for the 5-minute change


class PortfolioManager:
    def __init__(self):
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self.cycles = 0
        self.last_cycle_at = 0.0
        self._started_at = 0.0
        self.last_cycle_ms = 0.0
        self.last_error: Optional[str] = None
        self.entries_total = 0
        self.last_entry: Optional[Dict[str, Any]] = None
        self._attempted: Dict[str, float] = {}
        self._logged: tuple = ()
        self._logged_at = 0.0
        self.status: Dict[str, Any] = {"state": "starting", "message": "Manager starting."}
        self.picks: List[Dict[str, Any]] = []
        self.blocked_by: Dict[str, int] = {}
        # One row per watchlist symbol we do not hold, refreshed every cycle, and
        # the price samples the 1m/5m change is measured from.
        self.watch: List[Dict[str, Any]] = []
        self._px: Dict[str, deque] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        if not settings.MANAGER_ENABLED:
            self.status = {"state": "disabled", "message": "MANAGER_ENABLED is off."}
            return
        self._running = True
        self._started_at = time.time()
        self._task = asyncio.create_task(self._loop())
        logger.info("Portfolio manager started.")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()

    def owns_entries(self) -> bool:
        """
        True while the manager is alive and cycling, so the tick path leaves entries
        to it. A manager that has stopped cycling hands them straight back.
        """
        if not (settings.MANAGER_ENABLED and self._running):
            return False
        return time.time() - self.last_cycle_at <= max(15.0, 5 * settings.MANAGER_INTERVAL_SECONDS)

    async def _loop(self):
        while self._running:
            try:
                await self.run_cycle()
                self.last_error = None
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
                logger.error(f"Manager cycle failed: {e}", exc_info=True)
            try:
                await asyncio.sleep(settings.MANAGER_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                break

    # ------------------------------------------------------------------
    # One cycle
    # ------------------------------------------------------------------

    def _sample(self, sym: str, price: float, now: float):
        """Records this cycle's price so the change over the last minutes can be shown."""
        buf = self._px.get(sym)
        if buf is None:
            buf = self._px[sym] = deque(maxlen=int(WATCH_HISTORY_SECONDS / max(settings.MANAGER_INTERVAL_SECONDS, 0.5)) + 5)
        buf.append((now, price))

    def _change_pct(self, sym: str, window_s: float, price: float, now: float) -> Optional[float]:
        """% change against the oldest sample inside the window; None until one exists."""
        buf = self._px.get(sym)
        if not buf:
            return None
        ref = next((p for t, p in buf if now - t <= window_s), None)
        if ref is None or ref <= 0 or buf[0][0] > now - window_s * 0.5:
            return None      # not enough history yet to speak for this window
        return round((price - ref) / ref * 100.0, 3)

    def _gather(self, now: float, blocked: Counter) -> List[Dict[str, Any]]:
        """
        Watches every watchlist symbol we do not hold (held ones have their own
        sentinel): samples its price, and returns the strategy-approved entries.
        Fills self.watch with one row per symbol -- price, 1m/5m change, and the
        verdict this cycle -- so the operator can see the manager looking at each.
        """
        from engine.executor import executor
        from core.market_hours import us_session, minutes_to_close, REGULAR, PRE

        session = us_session()
        cutoff = settings.NO_NEW_ENTRY_MINUTES_BEFORE_CLOSE if settings.DAY_TRADE_FLATTEN_ENABLED else 0.0
        out = []
        watch: List[Dict[str, Any]] = []
        for sym in sorted(state.watchlist):
            if sym in state.active_positions or sym in executor.pending_orders:
                continue
            tick = state.latest_prices.get(sym)
            row: Dict[str, Any] = {
                "symbol": sym, "price": None,
                "chg_1m_pct": None, "chg_5m_pct": None,
                "price_age_s": None, "tick_age_s": None, "status": "", "verdict": "",
            }
            tr = trend_board.read(sym)
            row["trend"] = tr.brief()
            if tick is not None:
                self._sample(sym, tick.price, now)
                row.update(
                    price=tick.price,
                    chg_1m_pct=self._change_pct(sym, 60.0, tick.price, now),
                    chg_5m_pct=self._change_pct(sym, 300.0, tick.price, now),
                    price_age_s=round(max(0.0, now - state.price_moved_at.get(sym, now)), 1),
                    tick_age_s=round(max(0.0, now - tick.timestamp), 1))
            watch.append(row)

            def skip(status: str, label: Optional[str] = None):
                row["status"] = status
                blocked[label or status] += 1

            if session not in (REGULAR, PRE):
                skip("market closed", "stock market closed")
                continue
            mins = minutes_to_close(sym)
            if mins is not None and mins <= cutoff:
                skip("near close", "too close to the market close")
                continue
            if tick is None:
                skip("no price", "no price yet")
                continue
            if row["tick_age_s"] > settings.MANAGER_MAX_TICK_AGE_SECONDS:
                skip("feed silent", "price feed silent")
                continue
            # A price that is not moving would be flagged stale and closed again
            # within minutes of entry: not worth the round trip.
            if row["price_age_s"] >= settings.STALE_PRICE_EXIT_SECONDS / 2:
                skip("price frozen", "price not moving")
                continue
            quant = state.quant_metrics.get(sym)
            if quant is None:
                skip("warming up", "indicators warming up")
                continue
            if now - self._attempted.get(sym, 0.0) < settings.MANAGER_RETRY_SECONDS:
                skip("retry wait", "entry recently refused")
                continue
            decision = decision_engine.evaluate(sym, quant, state.get_sentiment(sym))
            row["buy_prob"] = round(float(decision.buy_prob), 3)
            if decision.action != "BUY":
                gate = state.last_gate_detail.get(sym) or {}
                why = gate.get("blocked_by") or "no signal"
                row["status"] = "no entry"
                row["verdict"] = why
                blocked[f"strategy: {why}"] += 1
                continue
            row["status"] = "buy signal"
            out.append({"symbol": sym, "decision": decision, "trend": tr})
        # Held or in-flight symbols drop out; symbols removed from the watchlist too.
        for sym in [s for s in self._px if s not in state.watchlist or s in state.active_positions]:
            self._px.pop(sym, None)
        self.watch = watch
        return out

    def _rank(self, cands: List[Dict[str, Any]], blocked: Counter) -> List[Dict[str, Any]]:
        sleeves = diversification.sleeves()
        w = min(max(settings.MANAGER_FIT_WEIGHT, 0.0), 1.0)
        ranked = []
        for c in cands:
            fit, why = diversification.fit(c["symbol"], sleeves)
            if fit <= 0.0:
                blocked["diversification: no headroom"] += 1
                continue
            plan = self._plan(c["symbol"])
            if plan is None or plan["qty"] <= 0:
                blocked["sizing: nothing affordable"] += 1
                continue
            # The strategy's own conviction: the trend read is shown, not scored,
            # so the ranking adds nothing the strategy's backtest did not see.
            conviction = float(c["decision"].buy_prob)
            meta = universe.classify(c["symbol"])
            ranked.append({
                **c, "conviction": round(conviction, 3), "fit": round(fit, 3),
                "score": round((1 - w) * conviction + w * fit, 4),
                "sector": meta.sector, "region": meta.region, "why": why, "plan": plan,
            })
        ranked.sort(key=lambda r: r["score"], reverse=True)
        return ranked

    @staticmethod
    def _plan(sym: str) -> Optional[Dict[str, Any]]:
        """
        How much the entry would invest, with the same sizing the executor uses at
        order time (engine/allocation_agent.py), so the pick shows its dollars,
        quantity and risk before it is bought -- and an unaffordable one is not tried.
        """
        from engine.allocation_agent import allocation_manager
        from feeds.multi_source_aggregator import trend_aggregator
        tick = state.latest_prices.get(sym)
        quant = state.quant_metrics.get(sym)
        if not tick:
            return None
        try:
            a = allocation_manager.evaluate_allocation(
                symbol=sym, current_price=tick.price, atr=quant.atr if quant else 0.50,
                laya_pos_prob=state.get_sentiment(sym).pos_prob,
                consensus_score=trend_aggregator.get_consensus(sym))
        except Exception as e:
            logger.debug(f"Sizing preview failed for {sym}: {e}")
            return None
        return {"dollars": a["allocated_dollars"], "qty": a["qty"], "pct": a["allocated_pct"],
                "risk": a["risk_dollars"], "stop": a["stop_loss"], "target": a["take_profit"],
                "tier": a["conviction_tier"]}

    async def run_cycle(self):
        from engine.executor import executor
        t0 = time.perf_counter()
        now = time.time()
        profile = state.risk_profile
        active = state.is_trading_active
        blocked: Counter = Counter()

        cands = self._gather(now, blocked)
        await asyncio.sleep(0)

        # Only ask the risk guard once trading is on: paused, it would refuse every
        # candidate and hide which ones the manager would actually pick.
        if active:
            passed = []
            for c in cands:
                ok, why = risk_guard.can_open_position(c["symbol"])
                if ok:
                    passed.append(c)
                else:
                    blocked[f"risk: {why.split(':')[0][:60]}"] += 1
            cands = passed

        entered: List[Dict[str, Any]] = []
        ranked = self._rank(cands, blocked)

        if active:
            for _ in range(settings.MANAGER_MAX_ENTRIES_PER_CYCLE):
                if not ranked or self._slots(executor) <= 0:
                    break
                best = ranked[0]
                sym = best["symbol"]
                from engine.risk_guard import MIN_ORDER_DOLLARS
                if state.remaining_budget < MIN_ORDER_DOLLARS:
                    break
                self._attempted[sym] = time.time()
                state.log_event(
                    "MANAGER",
                    f"Deploying into {sym} ({best['sector']}, {best['region']}): rank 1 of {len(ranked)}, "
                    f"score {best['score']:.2f} = conviction {best['conviction']:.2f} + fit {best['fit']:.2f}, "
                    f"trend {best['trend'].label} {best['trend'].direction:+.2f}; "
                    f"plan ${best['plan']['dollars']:,.2f} ({best['plan']['qty']}x, risk ${best['plan']['risk']:,.2f})"
                    + (f" [{'; '.join(best['why'])}]" if best["why"] else "")
                    + f". ${state.remaining_budget:,.2f} of budget idle.")
                await executor.execute_decision(best["decision"])
                if sym in state.active_positions or sym in executor.awaiting_fill:
                    self.entries_total += 1
                    entered.append(best)
                    self.last_entry = {"symbol": sym, "at": time.time(), "score": best["score"],
                                       "sector": best["sector"], "region": best["region"]}
                # The book changed (or the entry was refused): re-rank the rest.
                cands = [c for c in cands if c["symbol"] != sym]
                ranked = self._rank(cands, Counter())

        self.picks = [self._pick_view(r) for r in ranked[:MAX_PICKS_SHOWN]]
        self.blocked_by = dict(blocked.most_common(6))
        self.status = self._describe(active, profile, entered, ranked, blocked, executor)
        self.cycles += 1
        self.last_cycle_at = time.time()
        self.last_cycle_ms = (time.perf_counter() - t0) * 1000
        self._log_transition()

    @staticmethod
    def _slots(executor) -> int:
        held = set(state.active_positions) | set(executor.pending_orders)
        return state.risk_profile.max_concurrent_positions - len(held)

    @staticmethod
    def _pick_view(r: Dict[str, Any]) -> Dict[str, Any]:
        return {"symbol": r["symbol"], "score": r["score"], "conviction": r["conviction"],
                "fit": r["fit"], "sector": r["sector"], "region": r["region"], "why": r["why"],
                "plan": r["plan"], "trend": r["trend"].brief()}

    def _describe(self, active, profile, entered, ranked, blocked, executor) -> Dict[str, Any]:
        cap = state.hard_cap
        idle = state.remaining_budget
        slots = self._slots(executor)
        # What the dial can ever deploy: N positions of at most X% each.
        dial_pct = min(100.0, profile.max_concurrent_positions * profile.max_position_notional_pct)
        top = ", ".join(f"{k} x{v}" for k, v in blocked.most_common(3))

        if not active:
            st, msg = "paused", (f"Trading is paused, so nothing is deployed. ${idle:,.2f} of budget is idle; "
                                 f"{len(ranked)} entries are ready when trading is switched on.")
        elif entered:
            st, msg = "active", ("Deployed into " + ", ".join(e["symbol"] for e in entered)
                                 + f". ${idle:,.2f} of budget still idle.")
        elif slots <= 0:
            st, msg = "full", (f"All {profile.max_concurrent_positions} position slots are used at risk dial "
                               f"{profile.factor}; ${idle:,.2f} idle cannot be deployed until one closes.")
        elif idle < 30.0:
            st, msg = "full", f"Budget fully committed (${idle:,.2f} left)."
        elif ranked:
            st, msg = "active", f"{len(ranked)} entries ranked; the best goes in next cycle."
        else:
            n = len(self.watch)
            st, msg = "watching", (
                f"Watching {n} watchlist symbol{'s' if n != 1 else ''} every "
                f"{settings.MANAGER_INTERVAL_SECONDS:g}s; none qualifies for entry right now, so "
                f"${idle:,.2f} of budget stays idle. "
                + (f"Blocked by: {top}." if top else "Nothing on the watchlist has a price yet."))
        return {
            "state": st, "message": msg,
            "slots_free": slots, "positions": len(state.active_positions),
            "max_positions": profile.max_concurrent_positions,
            "budget": {
                "hard_cap": cap, "idle": idle,
                "committed": state.committed_capital,
                "utilization_pct": round((cap - idle) / cap * 100, 1) if cap > 0 else 0.0,
                "dial_deployable_pct": round(dial_pct, 1),
            },
        }

    def _log_transition(self):
        """One MANAGER line when the manager's situation changes, not one per cycle."""
        key = (self.status["state"], self.status.get("slots_free"), tuple(self.blocked_by))
        now = time.time()
        if key != self._logged and now - self._logged_at >= 30.0 and self.status["state"] != "active":
            state.log_event("MANAGER", self.status["message"])
            self._logged, self._logged_at = key, now

    def snapshot(self) -> Dict[str, Any]:
        # Measured from the last cycle, or from start-up if none ever finished, so a
        # manager wedged on its first cycle is reported too.
        ref = self.last_cycle_at or self._started_at
        stalled = bool(self._running and ref
                       and time.time() - ref > max(15.0, 5 * settings.MANAGER_INTERVAL_SECONDS))
        return {
            **self.status,
            **({"state": "stalled", "message": "Manager stopped cycling; the tick path is entering instead."}
               if stalled else {}),
            "enabled": settings.MANAGER_ENABLED, "cycles": self.cycles,
            "last_cycle_at": self.last_cycle_at or None,
            "cycle_ms": round(self.last_cycle_ms, 1),
            "interval_s": settings.MANAGER_INTERVAL_SECONDS,
            "picks": self.picks, "blocked_by": self.blocked_by,
            "watching": len(self.watch), "watch": self.watch,
            "entries_total": self.entries_total, "last_entry": self.last_entry,
            "error": self.last_error,
        }


portfolio_manager = PortfolioManager()
