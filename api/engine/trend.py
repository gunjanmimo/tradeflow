"""
What the time series says the market is doing, read on three horizons before any
agent acts on a symbol.

  micro     last TREND_SHORT_BARS one-minute bars   -- a turn shows up here first
  session   last TREND_LONG_BARS one-minute bars    -- the intraday trend
  daily     200-day average and 3-month return      -- the backdrop (daily bars)

Per horizon, direction is the drift t-statistic of log price: the regression
slope over the window divided by the noise of one-minute returns across it. A
random walk scores about +-1, a clean trend 2 or more. tanh squashes it to -1..1
and sqrt(R^2) discounts a jagged line. On the session horizon EMA 9/21 of the
minute closes corroborates.

Composite direction = session 0.45, micro 0.35, daily 0.20, renormalised over the
horizons that have data. Confidence rises when the horizons agree.

  uptrend / downtrend   |direction| >= TREND_UP
  range                 in between
  unknown               fewer than TREND_MIN_BARS minute bars: no agent acts on it

reversal_down: the session trend is up but the micro trend has turned down hard
(reversal_up is the mirror). It is the early warning the position manager trims on.
"""
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.config import settings
from core.minute_bars import minute_bars

WEIGHTS = {"session": 0.45, "micro": 0.35, "daily": 0.20}


@dataclass
class TrendRead:
    symbol: str
    label: str = "unknown"                 # uptrend | downtrend | range | unknown
    direction: float = 0.0                 # -1..1
    confidence: float = 0.0                # 0..1
    ready: bool = False
    bars: int = 0
    micro: Optional[float] = None
    session: Optional[float] = None
    daily: Optional[float] = None
    r2: Optional[float] = None
    ret_15m_pct: Optional[float] = None
    ret_60m_pct: Optional[float] = None
    vol_1m_pct: Optional[float] = None     # std of one-minute returns
    reversal_down: bool = False
    reversal_up: bool = False
    reasons: List[str] = field(default_factory=list)
    at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k in ("direction", "confidence", "micro", "session", "daily", "r2"):
            if d[k] is not None:
                d[k] = round(d[k], 3)
        return d

    def brief(self) -> Dict[str, Any]:
        return {"label": self.label, "direction": round(self.direction, 3),
                "confidence": round(self.confidence, 3), "ready": self.ready,
                "bars": self.bars, "reversal_down": self.reversal_down,
                "reversal_up": self.reversal_up}


def drift(closes: np.ndarray) -> Tuple[float, float]:
    """(direction -1..1, R^2) of a window of closes."""
    n = len(closes)
    if n < 5 or np.any(closes <= 0):
        return 0.0, 0.0
    y = np.log(closes)
    x = np.arange(n, dtype=np.float64)
    xm, ym = x - x.mean(), y - y.mean()
    b = float((xm * ym).sum() / (xm * xm).sum())
    sst = float((ym * ym).sum())
    resid = ym - b * xm
    r2 = max(0.0, 1.0 - float((resid * resid).sum()) / sst) if sst > 0 else 0.0
    move = b * (n - 1)
    rets = np.diff(y)
    noise = float(rets.std(ddof=1) * np.sqrt(n - 1)) if len(rets) > 2 else 0.0
    if noise <= 1e-12:
        t = 0.0 if abs(move) < 1e-12 else float(np.sign(move)) * 3.0
    else:
        t = move / noise
    return float(np.tanh(t / 2.0) * np.sqrt(r2)), r2


def _ema(x: np.ndarray, span: int) -> float:
    a = 2.0 / (span + 1)
    e = float(x[0])
    for v in x[1:]:
        e = a * float(v) + (1 - a) * e
    return e


def daily_bias(symbol: str) -> Optional[float]:
    from feeds.daily_bars import daily_bars
    m = daily_bars.momentum(symbol)
    if not m:
        return None
    d = 0.5 * float(np.tanh(m["ret_3m"] / 0.10))
    if m.get("above_200d") is not None:
        d += 0.5 if m["above_200d"] else -0.5
    return float(np.clip(d, -1.0, 1.0))


def analyze(symbol: str, closes: Optional[np.ndarray] = None,
            daily: Optional[float] = "auto") -> TrendRead:
    c = minute_bars.closes(symbol) if closes is None else np.asarray(closes, dtype=np.float64)
    r = TrendRead(symbol=symbol, bars=len(c), at=time.time())
    if daily == "auto":
        daily = daily_bias(symbol)
    r.daily = daily
    if len(c) < settings.TREND_MIN_BARS:
        r.reasons.append(f"learning: {len(c)}/{settings.TREND_MIN_BARS} one-minute bars")
        return r
    r.ready = True

    long_w = c[-settings.TREND_LONG_BARS:]
    short_w = c[-settings.TREND_SHORT_BARS:]
    s_dir, r2 = drift(long_w)
    if len(long_w) >= 21:
        e9, e21 = _ema(long_w, 9), _ema(long_w, 21)
        ema = 1.0 if (e9 > e21 and long_w[-1] > e9) else -1.0 if (e9 < e21 and long_w[-1] < e9) else 0.0
        s_dir = 0.8 * s_dir + 0.2 * ema
    m_dir, _ = drift(short_w)
    r.session, r.micro, r.r2 = s_dir, m_dir, r2

    rets = np.diff(np.log(c[-61:]))
    r.vol_1m_pct = round(float(rets.std(ddof=1)) * 100, 4) if len(rets) > 2 else None
    if len(c) > 15:
        r.ret_15m_pct = round((c[-1] / c[-16] - 1) * 100, 3)
    if len(c) > 60:
        r.ret_60m_pct = round((c[-1] / c[-61] - 1) * 100, 3)

    parts = {"session": s_dir, "micro": m_dir}
    if daily is not None:
        parts["daily"] = daily
    w = sum(WEIGHTS[k] for k in parts)
    r.direction = float(sum(WEIGHTS[k] * v for k, v in parts.items()) / w)

    sign = np.sign(r.direction)
    agree = sum(WEIGHTS[k] for k, v in parts.items() if abs(v) > 0.1 and np.sign(v) == sign) / w
    r.confidence = float(min(1.0, abs(r.direction) / 0.6) * agree) if sign else 0.0

    r.label = ("uptrend" if r.direction >= settings.TREND_UP
               else "downtrend" if r.direction <= -settings.TREND_UP else "range")
    r.reversal_down = s_dir >= 0.2 and m_dir <= -0.4
    r.reversal_up = s_dir <= -0.2 and m_dir >= 0.4
    r.reasons.append(
        f"session {s_dir:+.2f} (R2 {r2:.2f}), micro {m_dir:+.2f}"
        + (f", daily {daily:+.2f}" if daily is not None else ", no daily history"))
    if r.reversal_down:
        r.reasons.append("micro trend turned down against the session trend")
    if r.reversal_up:
        r.reasons.append("micro trend turned up against the session trend")
    return r


class TrendBoard:
    """Latest read per symbol, kept fresh by the trend analyst agent."""

    def __init__(self):
        self.reads: Dict[str, TrendRead] = {}

    def read(self, symbol: str, max_age_s: float = 5.0) -> TrendRead:
        """The analyst's read, or a fresh one if it is missing or stale."""
        r = self.reads.get(symbol)
        if r is None or time.time() - r.at > max_age_s:
            r = analyze(symbol)
            self.reads[symbol] = r
        return r


board = TrendBoard()
