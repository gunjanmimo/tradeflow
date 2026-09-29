"""
Candidate long-entry signals, vectorised over a symbol's one-minute bars.

Each signal returns a boolean array: True at bar t means "enter long at the open
of bar t+1". Only information available at the close of bar t is used, and
intraday statistics restart every session (nothing from yesterday's intraday
path leaks in, except where a signal uses the previous close on purpose).

They are hypotheses, each with a published or well-known rationale, not a
promise that any of them works -- research/edge.py decides that.

  reversal_z       short-term mean reversion: close far below its 20-bar mean
  vwap_dev         price stretched far below the session VWAP
  rsi2             Connors' RSI(2) oversold
  idio_reversal    a fall NOT explained by the market (stock minus SPY, 30 min)
  gap_fade         a large opening gap down, bought after the first 5 minutes
  momentum_30      30-minute continuation (the opposite bet of the reversals)
  orb              opening-range (first 30 min) breakout
  ema_cross        EMA 9 crossing above EMA 21 (the old platform's core signal)
  random           a coin flip per bar: the baseline every signal must beat
"""
from typing import Callable, Dict, Optional

import numpy as np

from research.data import Bars, ny_minute_of_day

OPEN_MIN = 570          # 09:30 NY
CLOSE_MIN = 960         # 16:00 NY


def _rolling_mean_std(x: np.ndarray, n: int):
    """Trailing mean/std over n samples (NaN until n are available)."""
    out_m = np.full(len(x), np.nan)
    out_s = np.full(len(x), np.nan)
    if len(x) < n:
        return out_m, out_s
    cs = np.cumsum(np.r_[0.0, x])
    cs2 = np.cumsum(np.r_[0.0, x * x])
    s = cs[n:] - cs[:-n]
    s2 = cs2[n:] - cs2[:-n]
    m = s / n
    var = np.maximum(s2 / n - m * m, 0.0) * n / max(n - 1, 1)
    out_m[n - 1:] = m
    out_s[n - 1:] = np.sqrt(var)
    return out_m, out_s


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    a = 2.0 / (span + 1.0)
    out = np.empty_like(x)
    acc = x[0]
    for i, v in enumerate(x):
        acc = a * v + (1 - a) * acc
        out[i] = acc
    return out


def _rsi(c: np.ndarray, n: int) -> np.ndarray:
    d = np.diff(c, prepend=c[0])
    g, l = np.clip(d, 0, None), np.clip(-d, 0, None)
    ag, al = _ema(g, 2 * n - 1), _ema(l, 2 * n - 1)
    rs = np.divide(ag, al, out=np.full_like(ag, np.inf), where=al > 0)
    return 100.0 - 100.0 / (1.0 + rs)


def _per_day(b: Bars, fn: Callable[[slice], np.ndarray]) -> np.ndarray:
    out = np.zeros(len(b), dtype=bool)
    for _, sl in b.day_slices():
        out[sl] = fn(sl)
    return out


def reversal_z(b: Bars, ctx=None, n: int = 20, k: float = 2.0) -> np.ndarray:
    def day(sl):
        m, s = _rolling_mean_std(b.c[sl], n)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (b.c[sl] - m) / s
        return np.nan_to_num(z, nan=0.0) <= -k
    return _per_day(b, day)


def vwap_dev(b: Bars, ctx=None, k: float = 2.5, n: int = 30) -> np.ndarray:
    def day(sl):
        c, v = b.c[sl], np.maximum(b.v[sl], 1.0)
        tp = (b.h[sl] + b.l[sl] + c) / 3.0
        vwap = np.cumsum(tp * v) / np.cumsum(v)
        dev = np.log(c / vwap)
        r = np.diff(np.log(c), prepend=np.log(c[0]))
        _, s = _rolling_mean_std(r, n)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = dev / (s * np.sqrt(n))
        return np.nan_to_num(z, nan=0.0) <= -k
    return _per_day(b, day)


def rsi2(b: Bars, ctx=None, th: float = 5.0) -> np.ndarray:
    def day(sl):
        c = b.c[sl]
        r = _rsi(c, 2)
        ok = np.arange(len(c)) >= 5
        return ok & (r <= th)
    return _per_day(b, day)


def idio_reversal(b: Bars, ctx=None, n: int = 30, k: float = 2.0) -> np.ndarray:
    """Stock return minus SPY return over n bars, z-scored by its own trailing spread."""
    mkt: Optional[Bars] = (ctx or {}).get("market")
    if mkt is None or b.symbol == mkt.symbol:
        return np.zeros(len(b), dtype=bool)
    idx = np.searchsorted(mkt.minute, b.minute)
    idx = np.clip(idx, 0, len(mkt) - 1)
    aligned = mkt.minute[idx] == b.minute
    mc = np.where(aligned, mkt.c[idx], np.nan)

    def day(sl):
        c, m = b.c[sl], mc[sl]
        # carry the market's last close over bars it has no print for
        good = np.isfinite(m)
        if good.sum() < n + 5:
            return np.zeros(len(c), dtype=bool)
        m = np.maximum.accumulate(np.where(good, np.arange(len(m)), 0))
        m = mc[sl][m]
        if not np.isfinite(m[0]):
            return np.zeros(len(c), dtype=bool)
        lr = np.log(c) - np.log(m)
        rel = np.full(len(c), np.nan)
        rel[n:] = lr[n:] - lr[:-n]
        diff1 = np.diff(lr, prepend=lr[0])
        _, s = _rolling_mean_std(diff1, n)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = rel / (s * np.sqrt(n))
        return np.nan_to_num(z, nan=0.0) <= -k
    return _per_day(b, day)


def gap_fade(b: Bars, ctx=None, gap: float = 0.01, at_bar: int = 5) -> np.ndarray:
    out = np.zeros(len(b), dtype=bool)
    prev_close = None
    for _, sl in b.day_slices():
        c = b.c[sl]
        if prev_close is not None and len(c) > at_bar:
            g = b.o[sl][0] / prev_close - 1.0
            if g <= -gap and c[at_bar] < prev_close:
                out[sl.start + at_bar] = True
        prev_close = c[-1]
    return out


def momentum_30(b: Bars, ctx=None, n: int = 30, k: float = 2.0) -> np.ndarray:
    def day(sl):
        c = b.c[sl]
        lr = np.log(c)
        rel = np.full(len(c), np.nan)
        rel[n:] = lr[n:] - lr[:-n]
        r = np.diff(lr, prepend=lr[0])
        _, s = _rolling_mean_std(r, n)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = rel / (s * np.sqrt(n))
        return np.nan_to_num(z, nan=0.0) >= k
    return _per_day(b, day)


def orb(b: Bars, ctx=None, minutes: int = 30) -> np.ndarray:
    mod = ny_minute_of_day(b.minute)

    def day(sl):
        c, h = b.c[sl], b.h[sl]
        m = mod[sl]
        in_range = m < OPEN_MIN + minutes
        if in_range.sum() < minutes // 2 or in_range.all():
            return np.zeros(len(c), dtype=bool)
        hi = h[in_range].max()
        after = ~in_range
        cross = after & (c > hi)
        first = np.zeros(len(c), dtype=bool)
        idx = np.flatnonzero(cross)
        if len(idx):
            first[idx[0]] = True
        return first
    return _per_day(b, day)


def ema_cross(b: Bars, ctx=None, fast: int = 9, slow: int = 21) -> np.ndarray:
    def day(sl):
        c = b.c[sl]
        f, s = _ema(c, fast), _ema(c, slow)
        up = f > s
        cross = np.zeros(len(c), dtype=bool)
        cross[1:] = up[1:] & ~up[:-1]
        cross[:slow] = False
        return cross
    return _per_day(b, day)


def random_entry(b: Bars, ctx=None, p: float = 0.01, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed + (hash(b.symbol) % 10_000))
    return rng.random(len(b)) < p


SIGNALS: Dict[str, Callable] = {
    "reversal_z": reversal_z,
    "vwap_dev": vwap_dev,
    "rsi2": rsi2,
    "idio_reversal": idio_reversal,
    "gap_fade": gap_fade,
    "momentum_30": momentum_30,
    "orb": orb,
    "ema_cross": ema_cross,
    "random": random_entry,
}
