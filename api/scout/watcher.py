"""
The Watcher agent: keeps eyes on every scout pick and says when one is worth
trading.

Every SCOUT_WATCH_INTERVAL_SECONDS, for each pick (and each open scout position)
it reads the live state and computes a confidence in 0..1:

  scout score   30%  how well the stock ranked this hour
  trend         25%  the trend analyst's direction (engine/trend.py), -1..1 -> 0..1
  VWAP          15%  price above / below today's volume-weighted average, in
                     quarter-ATRs (a stock above VWAP has buyers in control)
  open drive    10%  move since today's open, in half-ATRs
  news          15%  fresh headline tone (the news feed tracks every pick),
                     shrunk to neutral with one headline; neutral without news
  market         5%  SPY's trend direction

Hard blocks, whatever the confidence: outside the regular session, inside the
first SCOUT_FIRST_ENTRY_MINUTES, trend still learning, a downtrend or a fresh
reversal down, RSI over 75, a spread over 0.4%, or fresh bad news.

A pick is READY once its confidence has stayed at or above SCOUT_ENTRY_CONFIDENCE,
with no block, for SCOUT_CONFIRM_SECONDS: one strong reading on one tick is
noise; holding the level across a couple of minute bars is not. The scout
strategy (engine/strategies/scout.py) buys only READY picks and exits when the
confidence falls below SCOUT_EXIT_CONFIDENCE.
"""
import time
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np

from core.config import settings
from core.state import state
from engine.agent import Agent

_NY = ZoneInfo("America/New_York")
WEIGHTS = {"scout": 0.30, "trend": 0.25, "vwap": 0.15, "drive": 0.10, "news": 0.15, "market": 0.05}
TRACK_EVERY_SECONDS = 60.0
TRACK_POINTS = 390


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(min(hi, max(lo, x)))


def session_open_minute(now: Optional[float] = None) -> int:
    """Epoch minute of today's 09:30 New York."""
    d = datetime.fromtimestamp(time.time() if now is None else now, tz=_NY)
    return int(d.replace(hour=9, minute=30, second=0, microsecond=0).timestamp() // 60)


def session_levels(rows: List[list], open_minute: int) -> Dict[str, Optional[float]]:
    """Today's session open and VWAP from closed one-minute bars [minute, o, h, l, c, v]."""
    today = [r for r in rows if r[0] >= open_minute]
    if not today:
        return {"open": None, "vwap": None, "bars": 0}
    a = np.asarray(today, dtype=float)
    typical, vol = (a[:, 2] + a[:, 3] + a[:, 4]) / 3.0, a[:, 5]
    vwap = float((typical * vol).sum() / vol.sum()) if vol.sum() > 0 else float(a[:, 4].mean())
    return {"open": float(a[0, 1]), "vwap": vwap, "bars": len(today)}


def score_live(*, scout_score: float, price: float, trend: Any, levels: Dict[str, Optional[float]],
               atr_pct: Optional[float], sentiment: Any, market_direction: Optional[float],
               rsi: Optional[float], spread: float, session_open: bool,
               minutes_since_open: Optional[float], spread_source: str = "") -> Dict[str, Any]:
    """Confidence, its parts and every block, from plain inputs. Pure."""
    atr = max((atr_pct or 2.0) / 100.0, 0.005)
    parts: Dict[str, float] = {"scout": _clip(scout_score, 0.0, 1.0)}
    reasons: List[str] = []
    blocks: List[str] = []

    if trend is not None and trend.ready:
        parts["trend"] = 0.5 + 0.5 * _clip(trend.direction)
        reasons.append(f"trend {trend.label} ({trend.direction:+.2f})")
    else:
        parts["trend"] = 0.5
        blocks.append("trend still learning")

    vwap, sess_open = levels.get("vwap"), levels.get("open")
    if vwap:
        dev = price / vwap - 1.0
        parts["vwap"] = 0.5 + 0.5 * _clip(dev / (0.25 * atr))
        reasons.append(f"{dev * 100:+.2f}% vs VWAP")
    else:
        parts["vwap"] = 0.5
    if sess_open:
        drive = price / sess_open - 1.0
        parts["drive"] = 0.5 + 0.5 * _clip(drive / (0.5 * atr))
        reasons.append(f"{drive * 100:+.2f}% since the open")
    else:
        parts["drive"] = 0.5

    s = sentiment
    if s is not None and not s.is_stale and s.n_headlines > 0:
        trust = min(1.0, s.n_headlines / 2) * max(0.5, s.agreement)
        parts["news"] = 0.5 + (s.pos_prob - s.neg_prob) / 2 * trust
        reasons.append(f"news {s.pos_prob:.0%} pos / {s.neg_prob:.0%} neg ({s.n_headlines})")
        if s.n_headlines >= 2 and s.neg_prob >= 0.55:
            blocks.append(f"bad news: {s.neg_prob:.0%} negative over {s.n_headlines} headlines")
    else:
        parts["news"] = 0.5
    parts["market"] = 0.5 + 0.5 * _clip(market_direction) if market_direction is not None else 0.5

    if not session_open:
        blocks.insert(0, "regular session closed")
    elif minutes_since_open is not None and minutes_since_open < settings.SCOUT_FIRST_ENTRY_MINUTES:
        blocks.insert(0, f"first {settings.SCOUT_FIRST_ENTRY_MINUTES:.0f} min of the session")
    if trend is not None and trend.ready:
        if trend.label == "downtrend":
            blocks.append("in a downtrend")
        elif trend.reversal_down:
            blocks.append("micro trend just turned down")
    if rsi is not None and rsi > 75:
        blocks.append(f"RSI {rsi:.0f} overbought")
    if spread > 0.004:
        blocks.append(f"spread {spread * 100:.2f}% too wide ({spread_source})")

    conf = sum(WEIGHTS[k] * v for k, v in parts.items())
    return {"confidence": round(conf, 4), "parts": {k: round(v, 3) for k, v in parts.items()},
            "blocks": blocks, "reasons": reasons}


class WatcherAgent(Agent):
    name = "Watcher"
    role = "Follows every scout pick live; says when one is confident enough to trade"

    def __init__(self):
        super().__init__()
        self.reads: Dict[str, Dict[str, Any]] = {}
        self.above_since: Dict[str, float] = {}
        self.track: Dict[str, deque] = {}
        self.extremes: Dict[str, List[float]] = {}     # [high, low] since picked
        self._tracked_at: Dict[str, float] = {}
        self._ready_logged: Dict[str, float] = {}

    @property
    def interval(self) -> float:
        return settings.SCOUT_WATCH_INTERVAL_SECONDS

    @property
    def enabled(self) -> bool:
        return settings.SCOUT_ENABLED

    def symbols(self) -> List[str]:
        from scout.service import scout
        held = [s for s, p in state.active_positions.items() if p.get("entry_strategy") == "scout"]
        return sorted(set(scout.picks) | set(held))

    async def step(self):
        now = time.time()
        syms = self.symbols()
        for sym in syms:
            self.reads[sym] = self.assess(sym, now)
        for sym in [s for s in self.reads if s not in syms]:
            for d in (self.reads, self.above_since, self.track, self.extremes, self._tracked_at):
                d.pop(sym, None)
        ready = [s for s, r in self.reads.items() if r["status"] == "ready"]
        best = max(self.reads.values(), key=lambda r: r["confidence"], default=None)
        self.summary = (f"{len(syms)} picks watched, {len(ready)} ready"
                        + (f"; best {best['symbol']} {best['confidence']:.2f}" if best else ""))

    def assess(self, sym: str, now: float) -> Dict[str, Any]:
        from core.market_hours import us_session, REGULAR
        from core.minute_bars import minute_bars
        from engine.trend import board as trend_board
        from scout.service import scout
        from feeds.spreads import spreads

        pick = scout.picks.get(sym) or {}
        row = pick.get("row") or {}
        tick = state.latest_prices.get(sym)
        read: Dict[str, Any] = {"symbol": sym, "at": now, "rank": pick.get("rank"),
                                "scout_score": pick.get("score"), "confidence": 0.0,
                                "parts": {}, "blocks": [], "reasons": [], "status": "no price",
                                "confirmed_for_s": 0.0, "vwap": None, "price": None}
        if tick is None or not tick.price:
            self.above_since.pop(sym, None)
            return read
        price = tick.price
        open_min = session_open_minute(now)
        levels = session_levels(minute_bars.closed_rows(sym, now), open_min)
        quant = state.quant_metrics.get(sym)
        spy = trend_board.read("SPY", max_age_s=30.0)
        session_open = us_session() == REGULAR
        out = score_live(
            scout_score=float(pick.get("score") or 0.5), price=price,
            trend=trend_board.read(sym), levels=levels,
            atr_pct=(row.get("features") or {}).get("atr_pct"),
            sentiment=state.get_sentiment(sym),
            market_direction=spy.direction if spy.ready else None,
            rsi=quant.rsi if quant else None, spread=spreads.estimate(sym)[0] or 0.0,
            spread_source=spreads.estimate(sym)[1],
            session_open=session_open,
            minutes_since_open=(now / 60 - open_min) if session_open else None)
        read.update(out, price=price, vwap=levels["vwap"], session_open_price=levels["open"])

        # Movement since the pick.
        pick_price = pick.get("pick_price")
        ext = self.extremes.setdefault(sym, [price, price])
        ext[0], ext[1] = max(ext[0], price), min(ext[1], price)
        read.update(pick_price=pick_price, high_since_pick=ext[0], low_since_pick=ext[1],
                    chg_since_pick_pct=round((price / pick_price - 1) * 100, 2) if pick_price else None)

        # Confirmation: the level must hold, unblocked, for SCOUT_CONFIRM_SECONDS.
        good = not out["blocks"] and out["confidence"] >= settings.SCOUT_ENTRY_CONFIDENCE
        if good:
            since = self.above_since.setdefault(sym, now)
            held_for = now - since
            if held_for >= settings.SCOUT_CONFIRM_SECONDS:
                read["status"] = "ready"
                if now - self._ready_logged.get(sym, 0.0) > 1800:
                    self._ready_logged[sym] = now
                    self.act(sym, "ready", f"confidence {out['confidence']:.2f}")
                    state.log_event("WATCHER", f"{sym} ready to trade: confidence {out['confidence']:.2f} "
                                               f"for {held_for:.0f}s ({'; '.join(out['reasons'])})")
            else:
                read["status"] = f"confirming {held_for:.0f}/{settings.SCOUT_CONFIRM_SECONDS:.0f}s"
        else:
            self.above_since.pop(sym, None)
            read["status"] = ("blocked: " + out["blocks"][0]) if out["blocks"] else "watching"
        read["confirmed_for_s"] = round(now - self.above_since[sym], 1) if sym in self.above_since else 0.0

        if now - self._tracked_at.get(sym, 0.0) >= TRACK_EVERY_SECONDS:
            self._tracked_at[sym] = now
            self.track.setdefault(sym, deque(maxlen=TRACK_POINTS)).append(
                (round(now), round(out["confidence"], 3), price))
        return read

    def read(self, symbol: str) -> Optional[Dict[str, Any]]:
        """The latest read, or None if there is none or it is too old to act on."""
        r = self.reads.get(symbol)
        if r is None or time.time() - r["at"] > max(30.0, 6 * self.interval):
            return None
        return r

    def snapshot(self) -> Dict[str, Any]:
        return {"reads": {s: {**r, "track": list(self.track.get(s, ()))[-60:]} for s, r in self.reads.items()},
                "rules": {"entry": settings.SCOUT_ENTRY_CONFIDENCE, "exit": settings.SCOUT_EXIT_CONFIDENCE,
                          "confirm_s": settings.SCOUT_CONFIRM_SECONDS,
                          "first_entry_min": settings.SCOUT_FIRST_ENTRY_MINUTES,
                          "min_hold_min": settings.SCOUT_MIN_HOLD_MINUTES, "weights": WEIGHTS}}


watcher = WatcherAgent()
