"""
What the trading agents are thinking, per watched stock, for the dashboard.

Every portfolio-manager cycle (MANAGER_INTERVAL_SECONDS) each watchlist stock we
do not hold gets a view built from what the agents already computed -- nothing
here decides anything:

  prob       the chance the agents open a trade: the watcher's confidence for a
             scout pick, else the strategy's entry conviction (buy_prob)
  bar        the level prob must reach (watcher entry confidence, or the risk
             dial's min_buy_prob)
  drivers    what pushes the view up or down, each -1..+1 with its weight and
             source (news, scout, trend, vwap, momentum, market, smart money)
  lean       bullish / neutral / bearish: the weighted sum of the drivers
  status     where it stands: blocked and why, watching, confirming, ready, with
             the trade desk (observing / analyst / critic / approved / rejected)
  thought    one sentence tying it together

An event is recorded when a stock's status changes, its lean flips, its prob
crosses the bar, or the desk moves it along -- the running "thought stream".
"""
import time
from collections import deque
from typing import Any, Dict, List, Optional

from core.config import settings
from core.state import state

HISTORY_EVERY_SECONDS = 10.0
HISTORY_POINTS = 90                 # 15 minutes at one point per 10 s
EVENT_MIN_GAP_SECONDS = 5.0

# Watcher parts -> where the evidence comes from.
_SOURCE = {"scout": "scout", "trend": "trend", "vwap": "price vs VWAP", "drive": "move since open",
           "news": "news", "market": "market (SPY)"}


def _lean(drivers: List[Dict[str, Any]]) -> str:
    w = sum(d["weight"] for d in drivers) or 1.0
    net = sum(d["value"] * d["weight"] for d in drivers) / w
    return "bullish" if net > 0.1 else "bearish" if net < -0.1 else "neutral"


class Thoughts:
    def __init__(self):
        self.views: Dict[str, Dict[str, Any]] = {}
        self.events: deque = deque(maxlen=300)
        self.history: Dict[str, deque] = {}
        self._sampled: Dict[str, float] = {}
        self._evented: Dict[str, float] = {}
        self.version = 0

    # ------------------------------------------------------------------

    def update(self, rows: List[Dict[str, Any]], now: Optional[float] = None):
        """rows: the manager's watch rows (one per watchlist stock it does not hold)."""
        now = time.time() if now is None else now
        seen = set()
        for row in rows:
            sym = row["symbol"]
            seen.add(sym)
            view = self.view(sym, row, now)
            self._events(sym, self.views.get(sym), view, now)
            self.views[sym] = view
            if view["prob"] is not None and now - self._sampled.get(sym, 0.0) >= HISTORY_EVERY_SECONDS:
                self._sampled[sym] = now
                self.history.setdefault(sym, deque(maxlen=HISTORY_POINTS)).append(
                    (round(now), round(view["prob"], 3)))
        for sym in [s for s in self.views if s not in seen]:
            self.views.pop(sym, None)
            self.history.pop(sym, None)
        self.version += 1

    def view(self, sym: str, row: Dict[str, Any], now: float) -> Dict[str, Any]:
        from engine.strategies import registry
        from engine.trend import board as trend_board
        from scout.service import scout
        from scout.watcher import watcher, WEIGHTS

        gate = state.last_gate_detail.get(sym) or {}
        fresh_gate = now - float(gate.get("at") or 0.0) < 30.0
        strategy = (gate.get("strategy") if fresh_gate else None) or registry.resolve(
            sym, state.strategy_class_defaults, state.strategy_overrides).name
        pick = scout.picks.get(sym)
        drivers: List[Dict[str, Any]] = []
        prob, bar, status = None, None, row.get("status") or ""
        thought_bits: List[str] = []

        w = watcher.read(sym) if strategy == "scout" else None
        if w:
            for k, v in w["parts"].items():
                drivers.append({"name": k, "source": _SOURCE.get(k, k), "value": round((v - 0.5) * 2, 3),
                                "weight": WEIGHTS.get(k, 0.1)})
            prob, bar = w["confidence"], settings.SCOUT_ENTRY_CONFIDENCE
            status = w["status"] if status in ("", "no entry", "buy signal") else status
            thought_bits += w.get("reasons", [])[:3]
        else:
            sent = state.get_sentiment(sym)
            if sent.n_headlines and not sent.is_stale:
                trust = min(1.0, sent.n_headlines / 2) * max(0.5, sent.agreement)
                drivers.append({"name": "news", "source": "news", "weight": 0.35,
                                "value": round((sent.pos_prob - sent.neg_prob) * trust, 3)})
                thought_bits.append(f"news {sent.pos_prob:.0%} pos / {sent.neg_prob:.0%} neg "
                                    f"over {sent.n_headlines} headline(s)")
            tr = trend_board.read(sym)
            if tr.ready:
                drivers.append({"name": "trend", "source": "trend", "weight": 0.3, "value": round(tr.direction, 3)})
                thought_bits.append(f"trend {tr.label} ({tr.direction:+.2f})")
            if pick:
                drivers.append({"name": "scout", "source": "scout", "weight": 0.2,
                                "value": round(((pick.get("score") or 0.5) - 0.5) * 2, 3)})
            if strategy == "smart_money":
                drivers.append({"name": "smart money", "source": "smart money", "weight": 0.3, "value": 1.0})
            if fresh_gate and gate.get("buy_prob") is not None:
                prob = float(gate["buy_prob"])
                bar = float(state.risk_profile.min_buy_prob)
        if row.get("buy_prob") is not None and prob is None:
            prob = float(row["buy_prob"])

        desk = self._desk(sym)
        if desk:
            status = f"desk: {desk['stage']}"
        if fresh_gate and gate.get("reason") and not w:
            thought_bits.insert(0, str(gate["reason"])[:160])

        lean = _lean(drivers) if drivers else "neutral"
        pos = sorted((d for d in drivers if d["value"] > 0.05), key=lambda d: -d["value"] * d["weight"])[:2]
        neg = sorted((d for d in drivers if d["value"] < -0.05), key=lambda d: d["value"] * d["weight"])[:2]
        why = []
        if pos:
            why.append("for: " + ", ".join(d["source"] for d in pos))
        if neg:
            why.append("against: " + ", ".join(d["source"] for d in neg))
        return {
            "symbol": sym, "strategy": strategy, "prob": round(prob, 3) if prob is not None else None,
            "bar": bar, "lean": lean, "drivers": drivers, "status": status or row.get("verdict") or "",
            "desk": desk, "price": row.get("price"), "chg_5m_pct": row.get("chg_5m_pct"),
            "scout_rank": (pick or {}).get("rank"), "country": (pick or {}).get("country"),
            "thought": (f"{lean.capitalize()} ({'; '.join(why)})" if why else lean.capitalize())
                       + (f" -- {'; '.join(thought_bits)}" if thought_bits else ""),
            "at": now,
        }

    @staticmethod
    def _desk(sym: str) -> Optional[Dict[str, Any]]:
        from desk.desk import desk
        c = desk.cases.get(sym) or desk.cleared.get(sym)
        if not c:
            return None
        d = c.get("decision") or {}
        a = (c.get("analyst") or {}).get("answer") or {}
        return {"stage": c["stage"], "p_analyst": a.get("p_target_first"), "p_final": d.get("p_final"),
                "need": d.get("need")}

    # ------------------------------------------------------------------

    def _events(self, sym: str, old: Optional[Dict[str, Any]], new: Dict[str, Any], now: float):
        if old is None:
            return
        texts = []
        important = False
        if new["status"] != old["status"]:
            texts.append(f"{old['status'] or '—'} → {new['status']}")
            important = new["status"].startswith("desk") or new["status"] == "ready"
        if new["lean"] != old["lean"]:
            texts.append(f"turned {new['lean']}")
        if new["prob"] is not None and old["prob"] is not None and new["bar"]:
            if old["prob"] < new["bar"] <= new["prob"]:
                texts.append(f"probability {new['prob']:.2f} crossed above the bar {new['bar']:.2f}")
                important = True
            elif new["prob"] < new["bar"] <= old["prob"]:
                texts.append(f"probability {new['prob']:.2f} fell below the bar {new['bar']:.2f}")
        if not texts:
            return
        if not important and now - self._evented.get(sym, 0.0) < EVENT_MIN_GAP_SECONDS:
            return
        self._evented[sym] = now
        self.events.append({"at": now, "symbol": sym, "text": "; ".join(texts), "lean": new["lean"],
                            "prob": new["prob"], "status": new["status"]})

    def snapshot(self, events: int = 40) -> Dict[str, Any]:
        views = sorted(self.views.values(), key=lambda v: (v["prob"] is None, -(v["prob"] or 0.0)))
        return {"version": self.version,
                "views": [{**v, "history": [p for _, p in self.history.get(v["symbol"], [])][-30:]} for v in views],
                "events": list(reversed(self.events))[:events]}


thoughts = Thoughts()
