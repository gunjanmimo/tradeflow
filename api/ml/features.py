"""
Features for the trade scorer: what the last WINDOW one-minute bars looked like
at the moment a strategy wanted to enter.

One function, used by live scoring (core/minute_bars), by the backtester (its
bar tape) and by training, so the model is trained on exactly what it later
sees. Only bars that have closed at decision time go in: no lookahead.

Per bar (N_BAR_FEATURES):
  0 log return          1 high-low range / close    2 close-open body / close
  3 volume z-score      4 RSI(14) / 100 - 0.5        5 close vs EMA 9
  6 close vs EMA 21     7, 8 time of day (sin, cos, New York clock)

Per trade: which strategy fired (an index into STRATEGY_VOCAB, embedded by the
model) and whether the symbol is crypto.
"""
import math
from typing import Optional, Sequence

import numpy as np

WINDOW = 60
N_BAR_FEATURES = 9
_NY_OFFSET_MIN = -4 * 60      # EDT; the half-hour DST error does not matter to sin/cos

# Fixed, append-only: a saved model's embedding rows follow this order.
STRATEGY_VOCAB = (
    "unknown", "momentum_breakout", "news_catalyst", "stock_score", "mean_reversion", "adaptive",
    "bollinger_reversion", "connors_rsi2", "zscore_reversion", "vwap_reversion", "macd_trend",
    "ma_crossover", "donchian_turtle", "supertrend", "ts_momentum", "volatility_squeeze",
    "parabolic_sar", "awesome_saucer", "heikin_ashi", "candle_reversal", "pairs_reversion",
)
_STRAT_INDEX = {n: i for i, n in enumerate(STRATEGY_VOCAB)}


def strategy_index(name: Optional[str]) -> int:
    return _STRAT_INDEX.get(name or "", 0)


def _ema(x: np.ndarray, period: int) -> np.ndarray:
    a = 2.0 / (period + 1.0)
    out = np.empty_like(x)
    acc = x[0]
    for i, v in enumerate(x):
        acc = a * v + (1.0 - a) * acc
        out[i] = acc
    return out


def _rsi(c: np.ndarray, period: int = 14) -> np.ndarray:
    d = np.diff(c, prepend=c[0])
    g, l = np.clip(d, 0, None), np.clip(-d, 0, None)
    ag, al = _ema(g, 2 * period - 1), _ema(l, 2 * period - 1)   # Wilder's smoothing
    rs = np.divide(ag, al, out=np.full_like(ag, 1.0), where=al > 0)
    return 100.0 - 100.0 / (1.0 + rs)


def bar_features(minutes: Sequence[int], o, h, l, c, v) -> Optional[np.ndarray]:
    """
    (WINDOW, N_BAR_FEATURES) float32 for the bars given, oldest first, or None
    when there are too few. Pass more than WINDOW bars when available: the
    indicators warm up on the extra history and only the last WINDOW are kept.
    """
    c = np.asarray(c, dtype=np.float64)
    n = len(c)
    if n < WINDOW + 1 or np.any(c <= 0):
        return None
    o, h, l, v = (np.asarray(x, dtype=np.float64) for x in (o, h, l, v))
    m = np.asarray(minutes, dtype=np.int64)
    ret = np.diff(np.log(c), prepend=np.log(c[0]))
    lv = np.log1p(np.clip(v, 0, None))
    vz = (lv - lv[-WINDOW:].mean()) / (lv[-WINDOW:].std() + 1e-6)
    tod = ((m + _NY_OFFSET_MIN) % 1440) / 1440.0 * 2 * math.pi
    f = np.stack([
        ret,
        (h - l) / c,
        (c - o) / c,
        vz,
        _rsi(c) / 100.0 - 0.5,
        c / _ema(c, 9) - 1.0,
        c / _ema(c, 21) - 1.0,
        np.sin(tod),
        np.cos(tod),
    ], axis=1)[-WINDOW:]
    return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def live_features(symbol: str, now: Optional[float] = None) -> Optional[np.ndarray]:
    """From the live minute-bar store, closed bars only (the current minute is still forming)."""
    import time
    from core.minute_bars import minute_bars
    rows = minute_bars.rows(symbol)
    if not rows:
        return None
    cur = int((time.time() if now is None else now) // 60)
    if rows[-1][0] >= cur:
        rows = rows[:-1]
    if len(rows) < WINDOW + 1:
        return None
    a = np.asarray(rows[-(WINDOW + 40):], dtype=np.float64)
    return bar_features(a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5])
