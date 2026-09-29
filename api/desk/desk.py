"""
The trade desk: no position opens until a committee of agents has watched the
stock and argued the trade.

A case opens when the portfolio manager has a ranked buy signal for a symbol:

  1. Observer   watches for DESK_OBSERVE_SECONDS: price, VWAP and whether the
                strategy keeps signalling. A signal missing for
                DESK_SIGNAL_GAP_SECONDS, or present in under DESK_MIN_PERSISTENCE
                of the checks, fades the case.
  2. Analyst    an LLM with reasoning (DESK_ANALYST_MODEL) reads the case file
                (desk/brief.py) and answers BUY/PASS with P(target before stop)
  3. Critic     a second LLM pass (DESK_CRITIC_MODEL) hunts for reasons the trade
                is a mistake and may veto, with its own probability
  4. Decision   approve only if the analyst says BUY, the critic approves, the
                signal is still there, and their mean probability clears
                max(DESK_MIN_PROB, breakeven + a margin set by the risk dial)

An approval clears the symbol for DESK_CLEARANCE_SECONDS while the price stays
within DESK_MAX_PRICE_DRIFT_PCT; the executor refuses any buy without one. A
rejection holds the symbol off for DESK_REJECT_COOLDOWN_SECONDS.

LLM probabilities are not calibrated. Every finished case, and the P&L of every
trade it cleared, is appended to data/desk/cases.jsonl so they can be checked.
"""
import asyncio
import json
import logging
import os
import time
from collections import Counter, deque
from typing import Any, Dict, List, Optional, Tuple

from core.config import settings
from core.state import state
from desk import brief as brief_mod
from desk import prompts
from desk.llm import ollama, LLMError

logger = logging.getLogger("tradeflow.desk")

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "desk")
_PATH = os.path.join(_DATA_DIR, "cases.jsonl")
ACTIVE = ("observing", "queued", "analyst", "critic")
LIVE_THINKING_CHARS = 1600
NOTE_EVERY_SECONDS = 30.0
ERROR_COOLDOWN_SECONDS = 120.0


def _clip01(x: Any) -> float:
    try:
        return float(min(1.0, max(0.0, float(x))))
    except (TypeError, ValueError):
        return 0.5


def edge_margin(dial: Optional[int] = None) -> float:
    """Margin over breakeven for the risk dial: cautious users need a clearer edge."""
    d = state.risk_profile.factor if dial is None else dial
    t = (min(10, max(1, int(d))) - 1) / 9.0
    return round(settings.DESK_EDGE_MARGIN_CAUTIOUS
                 + t * (settings.DESK_EDGE_MARGIN_AGGRESSIVE - settings.DESK_EDGE_MARGIN_CAUTIOUS), 4)


class TradeDesk:
    def __init__(self):
        self.cases: Dict[str, Dict[str, Any]] = {}      # symbol -> case under way
        self.cleared: Dict[str, Dict[str, Any]] = {}    # symbol -> approved case not yet traded
        self.cooldown: Dict[str, Tuple[float, str]] = {}
        self.history: deque = deque(maxlen=40)
        self.queue: deque = deque()
        self.counts: Counter = Counter()
        self.version = 0
        self.llm: Dict[str, Any] = {"ok": False, "detail": "not checked yet"}
        self.reviewing: Optional[str] = None
        self._task: Optional[asyncio.Task] = None
        self._worker: Optional[asyncio.Task] = None
        self._running = False
        self._health_at = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return settings.DESK_ENABLED

    async def start(self):
        if not self.enabled:
            return
        self._running = True
        self.llm = await ollama.health(0)
        self._task = asyncio.create_task(self._loop())
        state.log_event("DESK", "Trade desk open: observer, analyst "
                                f"({settings.DESK_ANALYST_MODEL}), critic ({settings.DESK_CRITIC_MODEL}). "
                        + ("LLM ready." if self.llm["ok"] else f"LLM NOT ready: {self.llm['detail']}"))

    async def stop(self):
        self._running = False
        for t in (self._task, self._worker):
            if t:
                t.cancel()

    async def _loop(self):
        while self._running:
            try:
                await self.step()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Desk step failed: %s", e, exc_info=True)
            try:
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                break

    # ------------------------------------------------------------------
    # The manager's side
    # ------------------------------------------------------------------

    def propose(self, symbol: str, buy_prob: float, reason: str, strategy: str,
                plan: Optional[Dict[str, Any]]) -> str:
        """
        The manager has a ranked buy signal. Returns "cleared" when the desk has
        approved it, otherwise where the review stands: observing, queued,
        analyst, critic, rejected, busy or offline.
        """
        if not self.enabled:
            return "cleared"
        now = time.time()
        tick = state.latest_prices.get(symbol)
        if self._valid(symbol, tick.price if tick else None, now):
            return "cleared"
        cd = self.cooldown.get(symbol)
        if cd and now < cd[0]:
            return "rejected"
        case = self.cases.get(symbol)
        if case:
            case["last_signal_at"] = now
            case["signal_prob"] = round(float(buy_prob), 3)
            if plan:
                case["plan"] = plan
            return case["stage"]
        if not self.llm.get("ok"):
            return "offline"
        if len(self.cases) >= settings.DESK_MAX_ACTIVE_CASES:
            return "busy"
        self._open(symbol, buy_prob, reason, strategy, plan, tick.price if tick else None)
        return "observing"

    def _open(self, symbol, buy_prob, reason, strategy, plan, price, manual=False, observe_s=None):
        now = time.time()
        case = {
            "id": f"{symbol}-{int(now)}", "symbol": symbol, "strategy": strategy, "reason": reason,
            "signal_prob": round(float(buy_prob), 3), "price_signal": price, "plan": plan or {},
            "opened_at": now, "stage": "observing", "stage_at": now, "last_signal_at": now,
            "observe_s": settings.DESK_OBSERVE_SECONDS if observe_s is None else observe_s,
            "checks": 0, "present": 0, "samples": [], "observation": {}, "manual": manual,
            "analyst": {"model": settings.DESK_ANALYST_MODEL, "status": "waiting", "thinking": ""},
            "critic": {"model": settings.DESK_CRITIC_MODEL, "status": "waiting", "thinking": ""},
            "decision": {}, "timeline": [], "_noted_at": now,
        }
        self.cases[symbol] = case
        self.counts["opened"] += 1
        self._note(case, "Observer",
                   f"Signal from {strategy} (conviction {buy_prob:.2f}) at {price or 0:,.2f}. "
                   f"Watching for {case['observe_s']:.0f}s before anyone decides.")
        return case

    def _valid(self, symbol: str, price: Optional[float], now: float) -> bool:
        c = self.cleared.get(symbol)
        if not c:
            return False
        drift = abs(price / c["clear_price"] - 1) * 100 if price and c.get("clear_price") else 0.0
        if now > c["cleared_until"] or drift > settings.DESK_MAX_PRICE_DRIFT_PCT:
            self.cleared.pop(symbol, None)
            c["stage"] = "expired"
            self._note(c, "Decision", "Approval lapsed before an order went out ("
                       + (f"price moved {drift:.2f}%" if drift > settings.DESK_MAX_PRICE_DRIFT_PCT
                          else "clearance window ended") + ").")
            self._finish(c, keep_stage=True)
            return False
        return True

    # ------------------------------------------------------------------
    # The executor's side
    # ------------------------------------------------------------------

    def clearance(self, symbol: str, price: Optional[float]) -> Tuple[bool, str]:
        """Whether a buy may go out now: (True, case id) or (False, why not)."""
        if not (self.enabled and settings.DESK_REQUIRED):
            return True, ""
        if self._valid(symbol, price, time.time()):
            return True, self.cleared[symbol]["id"]
        case = self.cases.get(symbol)
        if case:
            return False, f"trade desk still reviewing ({case['stage']})"
        cd = self.cooldown.get(symbol)
        if cd and time.time() < cd[0]:
            return False, f"trade desk rejected it: {cd[1]}"
        return False, "no trade-desk approval"

    def executed(self, symbol: str, order: Dict[str, Any]):
        c = self.cleared.pop(symbol, None)
        if not c:
            return
        c["stage"] = "executed"
        c["order"] = order
        self._note(c, "Decision", f"Order sent: {order.get('qty')} x {symbol} at ~{order.get('price', 0):,.2f}.")
        self.counts["executed"] += 1
        self._finish(c, keep_stage=True)

    def closed(self, record: Dict[str, Any]):
        """A trade the desk cleared has closed: log its outcome next to the case."""
        cid = record.get("desk_case")
        if cid:
            self._append({"type": "outcome", "case_id": cid, "symbol": record.get("symbol"),
                          "pnl": record.get("pnl"), "pnl_pct": record.get("pnl_pct"), "at": time.time()})

    # ------------------------------------------------------------------
    # One step: observe, hand over to the LLM agents, expire
    # ------------------------------------------------------------------

    async def step(self):
        now = time.time()
        if now - self._health_at > 30.0:
            self._health_at = now
            was = self.llm.get("ok")
            self.llm = await ollama.health(0)
            if was is not None and was != self.llm["ok"]:
                state.log_event("DESK", "LLM reachable again." if self.llm["ok"]
                                else f"LLM unavailable: {self.llm['detail']}. Entries are blocked.")
                self.version += 1
        for sym, case in list(self.cases.items()):
            if case["stage"] == "observing":
                self._observe(case, now)
        for sym in list(self.cleared):
            tick = state.latest_prices.get(sym)
            self._valid(sym, tick.price if tick else None, now)
        if self.queue and (self._worker is None or self._worker.done()):
            sym = self.queue.popleft()
            case = self.cases.get(sym)
            if case and case["stage"] == "queued":
                self._worker = asyncio.create_task(self._review(case))

    def _observe(self, case: Dict[str, Any], now: float):
        from core.minute_bars import minute_bars
        from scout.watcher import session_levels, session_open_minute
        sym = case["symbol"]
        tick = state.latest_prices.get(sym)
        if tick is None:
            return
        present = case["manual"] or now - case["last_signal_at"] <= max(3.0, 2 * settings.MANAGER_INTERVAL_SECONDS + 1)
        if not case["manual"] and now - case["last_signal_at"] > settings.DESK_SIGNAL_GAP_SECONDS:
            return self._reject(case, "faded", f"signal gone for {now - case['last_signal_at']:.0f}s while observing",
                                agent="Observer", cooldown=60.0)
        vwap = session_levels(minute_bars.closed_rows(sym, now), session_open_minute(now))["vwap"]
        case["checks"] += 1
        case["present"] += int(present)
        case["samples"].append((round(now, 1), tick.price, vwap))
        case["samples"] = case["samples"][-240:]
        obs = self._summarize(case, now)
        case["observation"] = obs
        self.version += 1
        if now - case["_noted_at"] >= NOTE_EVERY_SECONDS:
            case["_noted_at"] = now
            self._note(case, "Observer", f"{obs['seconds']:.0f}s: price {obs['move_pct']:+.2f}% since the signal, "
                                         + (f"above VWAP {obs['above_vwap_share']:.0%} of checks, "
                                            if obs["above_vwap_share"] is not None else "no VWAP yet, ")
                                         + f"signal present {obs['persistence']:.0%}.")
        if obs["seconds"] >= case["observe_s"]:
            if obs["persistence"] < settings.DESK_MIN_PERSISTENCE:
                return self._reject(case, "faded", f"signal present only {obs['persistence']:.0%} of the time",
                                    agent="Observer", cooldown=120.0)
            case["stage"], case["stage_at"] = "queued", now
            self.queue.append(sym)
            self._note(case, "Observer", f"Observation done ({obs['seconds']:.0f}s, move {obs['move_pct']:+.2f}%"
                                         + (f", above VWAP {obs['above_vwap_share']:.0%}"
                                            if obs["above_vwap_share"] is not None else "")
                                         + "). Handing to the analyst.")

    @staticmethod
    def _summarize(case: Dict[str, Any], now: float) -> Dict[str, Any]:
        s = case["samples"]
        p0 = case["price_signal"] or (s[0][1] if s else None)
        prices = [x[1] for x in s]
        with_vwap = [x for x in s if x[2]]
        pct = lambda v: round((v / p0 - 1) * 100, 3) if p0 else 0.0
        return {"seconds": round(now - case["opened_at"], 1), "checks": case["checks"],
                "persistence": round(case["present"] / case["checks"], 3) if case["checks"] else 0.0,
                "move_pct": pct(prices[-1]) if prices else 0.0,
                "high_pct": pct(max(prices)) if prices else 0.0, "low_pct": pct(min(prices)) if prices else 0.0,
                "above_vwap_share": round(sum(1 for x in with_vwap if x[1] > x[2]) / len(with_vwap), 3)
                if with_vwap else None,
                "path": [round(pct(x[1]), 3) for x in s[-60:]]}

    # ------------------------------------------------------------------
    # The LLM agents
    # ------------------------------------------------------------------

    async def _review(self, case: Dict[str, Any]):
        sym = case["symbol"]
        self.reviewing = sym
        try:
            case["stage"], case["stage_at"] = "analyst", time.time()
            b = brief_mod.build(sym, case)
            case["brief"] = {k: v for k, v in b.items() if k != "text"}
            case["brief_text"] = b["text"]
            case["breakeven_p"] = b["plan"]["breakeven_p"]
            self._note(case, "Analyst", f"Reading the case file; reasoning with {settings.DESK_ANALYST_MODEL}.")
            analyst = await self._ask(case, "analyst", settings.DESK_ANALYST_MODEL, settings.DESK_ANALYST_THINK,
                                      prompts.analyst_messages(b["text"]), prompts.ANALYST_SCHEMA)
            if analyst is None:
                return
            a = analyst["answer"]
            a["p_target_first"] = _clip01(a.get("p_target_first"))
            # The numbers decide, not the model's label: in replays it said PASS on
            # estimates above the hurdle. The analyst is never shown the hurdle.
            need = self._need(case)
            self._note(case, "Analyst", f"P(target first) {a['p_target_first']:.2f}, {str(a.get('lean')).lower()} "
                                        f"({a.get('confidence')} confidence): {a.get('thesis', '')[:220]}")
            if a["p_target_first"] < need:
                case["decision"] = {"p_analyst": a["p_target_first"], "breakeven": case.get("breakeven_p"), "need": need}
                return self._reject(case, "rejected", f"analyst's P {a['p_target_first']:.2f} is below the bar "
                                                      f"{need:.2f}", agent="Decision")

            case["stage"], case["stage_at"] = "critic", time.time()
            self._note(case, "Critic", f"Challenging the analyst with {settings.DESK_CRITIC_MODEL}.")
            critic = await self._ask(case, "critic", settings.DESK_CRITIC_MODEL, settings.DESK_CRITIC_THINK,
                                     prompts.critic_messages(b["text"], a), prompts.CRITIC_SCHEMA)
            if critic is None:
                return
            c = critic["answer"]
            c["p_target_first"] = _clip01(c.get("p_target_first"))
            objections = "; ".join(c.get("objections") or [])[:240]
            self._note(case, "Critic", f"{c.get('verdict')} with P(target first) {c['p_target_first']:.2f}. "
                                       f"{c.get('summary', '')[:160]}" + (f" Objections: {objections}" if objections else ""))
            self._decide(case, a, c)
        except Exception as e:
            logger.error("Desk review of %s failed: %s", sym, e, exc_info=True)
            self._reject(case, "error", f"{type(e).__name__}: {e}", agent="Decision", cooldown=ERROR_COOLDOWN_SECONDS)
        finally:
            self.reviewing = None
            self.version += 1

    async def _ask(self, case, role: str, model: str, think: bool, messages, schema) -> Optional[Dict[str, Any]]:
        slot = case[role]
        slot.update(status="thinking" if think else "answering", started_at=time.time(), thinking="", tokens=0)
        self.version += 1

        def on_thinking(chunk: str):
            slot["thinking"] += chunk
            slot["tokens"] += 1
            self.version += 1

        def on_content(chunk: str):
            slot["status"] = "answering"
            slot["tokens"] += 1
            self.version += 1

        # GPU placement per agent: the analyst wholly on the GPU, the critic split
        # between the VRAM left over and RAM, so neither evicts the other.
        layers = settings.DESK_ANALYST_NUM_GPU if role == "analyst" else settings.DESK_CRITIC_NUM_GPU
        options = {"num_gpu": layers} if layers is not None and layers >= 0 else None
        slot["num_gpu"] = layers
        try:
            budget = settings.DESK_THINK_BUDGET_TOKENS if think else settings.DESK_LLM_MAX_TOKENS
            out = await ollama.chat(model, messages, schema, think, on_thinking, on_content, max_tokens=budget,
                                    options=options)
            if out["truncated"]:
                self._note(case, role.capitalize(), f"Reasoning budget ({budget} tokens) used up after "
                                                    f"{out['seconds']:.0f}s; asking for the answer from its notes.")
                slot["status"] = "answering"
                final = await ollama.answer_from_notes(model, messages, schema, out["thinking"], on_content,
                                                       options=options)
                out = {**final, "thinking": out["thinking"], "tokens": out["tokens"] + final["tokens"],
                       "seconds": round(out["seconds"] + final["seconds"], 1)}
        except LLMError as e:
            slot.update(status="error", error=str(e))
            self._reject(case, "error", f"{role} failed: {e}", agent=role.capitalize(),
                         cooldown=ERROR_COOLDOWN_SECONDS)
            return None
        slot.update(status="done", answer=out["answer"], seconds=out["seconds"], tokens=out["tokens"])
        return out

    def _decide(self, case: Dict[str, Any], a: Dict[str, Any], c: Dict[str, Any]):
        now = time.time()
        p_final = round((a["p_target_first"] + c["p_target_first"]) / 2, 3)
        be = case.get("breakeven_p")
        need = self._need(case)
        still = case["manual"] or now - case["last_signal_at"] <= settings.DESK_SIGNAL_GAP_SECONDS
        case["decision"] = {"p_analyst": a["p_target_first"], "p_critic": c["p_target_first"], "p_final": p_final,
                            "breakeven": be, "need": need}
        if c.get("verdict") != "APPROVE":
            return self._reject(case, "rejected", f"critic veto (mean P {p_final:.2f})", agent="Decision")
        if p_final < need:
            return self._reject(case, "rejected", f"mean P {p_final:.2f} below the bar {need:.2f}"
                                + (f" (breakeven {be:.2f} + margin)" if be else ""), agent="Decision")
        if not still:
            return self._reject(case, "faded", "the signal disappeared during the review", agent="Decision",
                                cooldown=60.0)
        tick = state.latest_prices.get(case["symbol"])
        case.update(stage="approved", stage_at=now, clear_price=tick.price if tick else case["price_signal"],
                    cleared_until=now + settings.DESK_CLEARANCE_SECONDS)
        case["decision"]["verdict"] = "APPROVED"
        self.cases.pop(case["symbol"], None)
        self.cleared[case["symbol"]] = case
        self.counts["approved"] += 1
        self._note(case, "Decision", f"APPROVED: mean P(target first) {p_final:.2f} >= {need:.2f}. "
                                     f"Cleared to buy for {settings.DESK_CLEARANCE_SECONDS:.0f}s within "
                                     f"{settings.DESK_MAX_PRICE_DRIFT_PCT}% of {case['clear_price']:,.2f}.")
        state.log_event("DESK", f"{case['symbol']} approved: analyst {a['p_target_first']:.2f}, "
                                f"critic {c['p_target_first']:.2f}, bar {need:.2f}")
        self._persist(case)

    @staticmethod
    def _need(case: Dict[str, Any]) -> float:
        """The bar: the trade's breakeven P(target first) plus the dial's margin, never below DESK_MIN_PROB."""
        be = case.get("breakeven_p")
        return round(max(settings.DESK_MIN_PROB, (be or 0.0) + edge_margin()), 3)

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------

    def _reject(self, case: Dict[str, Any], stage: str, why: str, agent: str, cooldown: Optional[float] = None):
        now = time.time()
        case["stage"], case["stage_at"] = stage, now
        case["decision"] = {**case.get("decision", {}), "verdict": stage.upper(), "reason": why}
        self.cooldown[case["symbol"]] = (now + (settings.DESK_REJECT_COOLDOWN_SECONDS if cooldown is None
                                                else cooldown), why)
        self.counts[stage] += 1
        self._note(case, agent, f"{stage.upper()}: {why}.")
        state.log_event("DESK", f"{case['symbol']} {stage}: {why}")
        self._finish(case)

    def _finish(self, case: Dict[str, Any], keep_stage: bool = False):
        if self.cases.get(case["symbol"]) is case:
            self.cases.pop(case["symbol"])
        case["closed_at"] = time.time()
        if case not in self.history:
            self.history.appendleft(case)
        self._persist(case)
        self.version += 1

    def _note(self, case: Dict[str, Any], agent: str, text: str):
        case["timeline"].append({"t": round(time.time(), 1), "agent": agent, "text": text})
        self.version += 1

    def _persist(self, case: Dict[str, Any]):
        rec = {k: v for k, v in case.items() if not k.startswith("_") and k != "samples"}
        for role in ("analyst", "critic"):
            rec[role] = {**case[role], "thinking": case[role].get("thinking", "")[-4000:]}
        rec["type"] = "case"
        self._append(rec)

    @staticmethod
    def _append(rec: Dict[str, Any]):
        try:
            os.makedirs(_DATA_DIR, exist_ok=True)
            with open(_PATH, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        except OSError as e:
            logger.warning("Could not log a desk case: %s", e)

    # ------------------------------------------------------------------
    # On demand and views
    # ------------------------------------------------------------------

    def review_now(self, symbol: str, observe_seconds: float = 0.0) -> Dict[str, Any]:
        """Opens a case by hand (the dashboard's 'Ask the desk'). It can clear a later buy like any other."""
        from engine.portfolio_manager import portfolio_manager
        sym = symbol.upper().strip()
        if sym in self.cases:
            raise ValueError(f"{sym} is already under review")
        tick = state.latest_prices.get(sym)
        if tick is None:
            raise ValueError(f"no live price for {sym}; add it to the watchlist first")
        if not self.llm.get("ok"):
            raise ValueError(f"LLM unavailable: {self.llm.get('detail')}")
        self.cooldown.pop(sym, None)
        case = self._open(sym, 0.5, "manual review requested from the dashboard", "manual",
                          portfolio_manager._plan(sym), tick.price, manual=True, observe_s=observe_seconds)
        return self._view(case, live=True)

    def _view(self, c: Dict[str, Any], live: bool) -> Dict[str, Any]:
        n = LIVE_THINKING_CHARS if live else 500
        out = {k: v for k, v in c.items() if k not in ("samples", "brief_text", "analyst", "critic")
               and not k.startswith("_")}
        for role in ("analyst", "critic"):
            slot = c[role]
            out[role] = {**{k: v for k, v in slot.items() if k != "thinking"},
                         "thinking_tail": slot.get("thinking", "")[-n:],
                         "thinking_chars": len(slot.get("thinking", "")),
                         "elapsed_s": round(time.time() - slot["started_at"], 1)
                         if slot.get("status") in ("thinking", "answering") and slot.get("started_at") else None}
        if "brief" in c:
            b = c["brief"]
            out["brief"] = {k: b.get(k) for k in ("price", "chg_today_pct", "vs_vwap_pct", "since_open_pct",
                                                  "ret_15m_pct", "ret_60m_pct", "trend", "rsi", "plan",
                                                  "sentiment", "market")}
        return out

    def case(self, case_id: str) -> Optional[Dict[str, Any]]:
        for c in list(self.cases.values()) + list(self.cleared.values()) + list(self.history):
            if c["id"] == case_id:
                return {**self._view(c, live=True), "brief_text": c.get("brief_text"),
                        "analyst": {**c["analyst"]}, "critic": {**c["critic"]}}
        return None

    def agents(self) -> List[Dict[str, Any]]:
        observing = [c for c in self.cases.values() if c["stage"] == "observing"]
        by_stage = {c["stage"]: c for c in self.cases.values()}
        last = self.history[0] if self.history else None

        def llm_card(name: str, role: str, stage: str, model: str):
            c = by_stage.get(stage)
            if c:
                slot = c[role.lower()]
                return {"name": name, "state": slot.get("status", "working"), "symbol": c["symbol"],
                        "summary": f"{slot.get('status')} on {c['symbol']} for "
                                   f"{time.time() - (slot.get('started_at') or time.time()):.0f}s",
                        "model": model}
            return {"name": name, "state": "idle" if self.llm.get("ok") else "offline", "symbol": None,
                    "summary": "waiting for a case" if self.llm.get("ok") else self.llm.get("detail"), "model": model}

        return [
            {"name": "Observer", "state": "watching" if observing else "idle",
             "symbol": ", ".join(c["symbol"] for c in observing) or None,
             "summary": (f"watching {len(observing)} signal(s)" if observing else "waiting for a buy signal"),
             "model": None},
            llm_card("Analyst", "analyst", "analyst", settings.DESK_ANALYST_MODEL),
            llm_card("Critic", "critic", "critic", settings.DESK_CRITIC_MODEL),
            {"name": "Decision", "state": "idle", "symbol": last["symbol"] if last else None,
             "summary": (f"last: {last['symbol']} {last['decision'].get('verdict', last['stage']).lower()}"
                         if last else "no case decided yet"), "model": None},
        ]

    def snapshot(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled, "required": settings.DESK_REQUIRED, "version": self.version,
            "llm": {k: v for k, v in self.llm.items() if k != "models"},
            "agents": self.agents(),
            "active": [self._view(c, live=True) for c in self.cases.values()],
            "cleared": [self._view(c, live=False) for c in self.cleared.values()],
            "recent": [self._view(c, live=False) for c in list(self.history)[:12]],
            "queue": list(self.queue),
            "cooldown": {s: {"until": u, "why": w} for s, (u, w) in self.cooldown.items() if u > time.time()},
            "counts": dict(self.counts),
            "rules": {"observe_s": settings.DESK_OBSERVE_SECONDS, "min_persistence": settings.DESK_MIN_PERSISTENCE,
                      "min_prob": settings.DESK_MIN_PROB, "edge_margin": edge_margin(),
                      "risk_dial": state.risk_profile.factor,
                      "clearance_s": settings.DESK_CLEARANCE_SECONDS,
                      "max_drift_pct": settings.DESK_MAX_PRICE_DRIFT_PCT,
                      "analyst_model": settings.DESK_ANALYST_MODEL, "critic_model": settings.DESK_CRITIC_MODEL,
                      "analyst_think": settings.DESK_ANALYST_THINK, "critic_think": settings.DESK_CRITIC_THINK},
        }


desk = TradeDesk()
