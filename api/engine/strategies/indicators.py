"""
Close-only indicator kit for the quant strategy library.

Why close-only
--------------
The engine keeps a ring buffer of trade prices (and volumes) per symbol, not OHLC
bars. Every indicator here is therefore computed from closes; where the textbook
version needs highs/lows (ATR, Keltner, Supertrend, Donchian) the close-to-close
equivalent is used. All lookbacks are in SAMPLES of that buffer, not days.

Why a per-evaluation cache
--------------------------
The adaptive selector and the quant council run many strategies against the same
symbol at the same moment. Most of them want the same EMAs, RSI and ATR, so a
PriceSeries memoises each indicator for the lifetime of one evaluation. It is
rebuilt on the next tick, so nothing here can go stale.

These are clean-room implementations of public, textbook formulas (Wilder,
Bollinger, Appel, Kaufman, Hurst), not code copied from GPL projects such as
backtrader or freqtrade.
"""
from typing import Optional, Tuple

import numpy as np


def _memo(fn):
    """Memoise a PriceSeries method on its arguments for the object's lifetime."""
    def wrapper(self, *args):
        key = (fn.__name__,) + args
        if key not in self._cache:
            self._cache[key] = fn(self, *args)
        return self._cache[key]
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


def _make_filter():
    """
    First-order IIR filter y_t = alpha*x_t + (1-alpha)*y_{t-1}, as fast as available.

    scipy's public lfilter spends ~5us validating inputs around a ~2us C loop;
    on the tick path that overhead dominates. Use the C routine directly when it
    reproduces lfilter exactly on a probe input, else the public API, else a
    Python loop. The check runs once at import, so a scipy upgrade that changes
    the private API degrades speed, never correctness.
    """
    try:
        from scipy.signal import lfilter
    except ImportError:  # pragma: no cover - scipy is in requirements
        return None
    try:
        from scipy.signal._sigtools import _linear_filter
        probe = np.linspace(1.0, 2.0, 16)
        b, a, zi = np.array([0.3]), np.array([1.0, -0.7]), np.array([0.7])
        if np.array_equal(_linear_filter(b, a, probe, -1, zi)[0],
                          lfilter([0.3], [1.0, -0.7], probe, zi=[0.7])[0]):
            return lambda alpha, x, zi0: _linear_filter(
                np.array([alpha]), np.array([1.0, alpha - 1.0]), x, -1, np.array([zi0]))[0]
    except Exception:
        pass
    return lambda alpha, x, zi0: lfilter([alpha], [1.0, alpha - 1.0], x, zi=[zi0])[0]


iir = _make_filter()


def _ewm(x: np.ndarray, alpha: float, y0: float) -> np.ndarray:
    """y_t = alpha*x_t + (1-alpha)*y_{t-1} for t>=1, y_0 = y0."""
    if iir is not None:
        out = np.empty(len(x))
        out[0] = y0
        out[1:] = iir(alpha, np.ascontiguousarray(x[1:], dtype=np.float64), (1.0 - alpha) * y0)
        return out
    out = np.empty(len(x))
    out[0] = y0
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def ema_series(x: np.ndarray, n: int) -> np.ndarray:
    """EMA seeded with the SMA of the first n samples; earlier values are NaN."""
    out = np.full(len(x), np.nan)
    if len(x) < n or n < 1:
        return out
    out[n - 1:] = _ewm(x[n - 1:], 2.0 / (n + 1.0), x[:n].mean())
    return out


def sma_series(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) < n or n < 1:
        return out
    c = np.cumsum(x)
    out[n - 1] = c[n - 1] / n
    out[n:] = (c[n:] - c[:-n]) / n
    return out


def rolling_std(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) < n or n < 2:
        return out
    # O(n) via running sums of x and x^2. Centring on the mean first keeps the
    # E[x^2] - E[x]^2 subtraction well-conditioned for large prices (BTC ~1e5).
    xc = x - x.mean()
    c1 = np.cumsum(xc)
    c2 = np.cumsum(xc * xc)
    s1 = np.concatenate(([c1[n - 1]], c1[n:] - c1[:-n]))
    s2 = np.concatenate(([c2[n - 1]], c2[n:] - c2[:-n]))
    out[n - 1:] = np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 0.0))
    return out


class PriceSeries:
    """Price/volume history for one symbol, with memoised indicators."""

    def __init__(self, prices, volumes=None):
        self.close = np.asarray(prices, dtype=np.float64)
        v = np.asarray(volumes if volumes is not None else [], dtype=np.float64)
        # Volume is only appended on ticks that carried volume, so the two buffers
        # can differ in length. Align on the most recent common tail.
        n = min(len(self.close), len(v))
        self.aligned_close = self.close[-n:] if n else self.close[:0]
        self.volume = v[-n:] if n else v[:0]
        self._cache = {}

    def __len__(self) -> int:
        return len(self.close)

    @property
    def last(self) -> float:
        return float(self.close[-1])

    # ---- moving averages ----
    @_memo
    def ema(self, n: int) -> np.ndarray:
        return ema_series(self.close, n)

    @_memo
    def sma(self, n: int) -> np.ndarray:
        return sma_series(self.close, n)

    @_memo
    def std(self, n: int) -> np.ndarray:
        return rolling_std(self.close, n)

    # ---- oscillators ----
    @_memo
    def rsi(self, n: int) -> Optional[float]:
        """Wilder RSI of the latest sample."""
        x = self.close
        if len(x) <= n:
            return None
        d = np.diff(x)
        gains = np.where(d > 0, d, 0.0)
        losses = np.where(d < 0, -d, 0.0)
        g = _ewm(gains[n - 1:], 1.0 / n, gains[:n].mean())[-1]
        l = _ewm(losses[n - 1:], 1.0 / n, losses[:n].mean())[-1]
        if l == 0:
            return 100.0 if g > 0 else 50.0
        return float(100.0 - 100.0 / (1.0 + g / l))

    @_memo
    def macd(self, fast: int, slow: int, signal: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(macd line, signal line, histogram) -- Gerald Appel."""
        line = self.ema(fast) - self.ema(slow)
        valid = ~np.isnan(line)
        sig = np.full(len(line), np.nan)
        if valid.sum() >= signal:
            start = int(np.argmax(valid))
            sig[start:] = ema_series(line[start:], signal)
        return line, sig, line - sig

    @_memo
    def bollinger(self, n: int, k: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(middle, upper, lower) -- John Bollinger."""
        mid = self.sma(n)
        sd = self.std(n)
        return mid, mid + k * sd, mid - k * sd

    @_memo
    def zscore(self, n: int) -> Optional[float]:
        if len(self.close) < n:
            return None
        w = self.close[-n:]
        sd = w.std()
        if sd <= 0:
            return 0.0
        return float((w[-1] - w.mean()) / sd)

    # ---- volatility ----
    @_memo
    def atr_series(self, n: int) -> np.ndarray:
        """Close-to-close true range, Wilder-smoothed. Floored at 0.05% of price."""
        x = self.close
        out = np.full(len(x), np.nan)
        if len(x) <= n:
            return out
        tr = np.abs(np.diff(x))
        # Wilder smoothing (alpha = 1/n) seeded with the first n true ranges.
        out[n:] = _ewm(tr[n - 1:], 1.0 / n, tr[:n].mean())
        return np.maximum(out, x * 0.0005)

    def atr(self, n: int) -> Optional[float]:
        s = self.atr_series(n)
        return None if np.isnan(s[-1]) else float(s[-1])

    @_memo
    def realized_vol(self, n: int) -> Optional[float]:
        """Std-dev of simple returns over the last n samples."""
        if len(self.close) <= n:
            return None
        r = np.diff(self.close[-(n + 1):]) / self.close[-(n + 1):-1]
        return float(r.std())

    # ---- channels ----
    @_memo
    def donchian(self, n: int) -> Tuple[Optional[float], Optional[float]]:
        """(highest, lowest) of the n samples BEFORE the current one."""
        if len(self.close) <= n:
            return None, None
        w = self.close[-(n + 1):-1]
        return float(w.max()), float(w.min())

    # ---- regime statistics ----
    @_memo
    def efficiency_ratio(self, n: int) -> Optional[float]:
        """Kaufman ER: net move / path length. ~1 = clean trend, ~0 = chop."""
        if len(self.close) <= n:
            return None
        w = self.close[-(n + 1):]
        path = np.abs(np.diff(w)).sum()
        if path <= 0:
            return 0.0
        return float(abs(w[-1] - w[0]) / path)

    @_memo
    def hurst(self, max_lag: int) -> Optional[float]:
        """
        Hurst exponent from the scaling of lagged-difference dispersion.
        <0.5 mean-reverting, ~0.5 random walk, >0.5 trending. Noisy on short
        buffers, so callers treat it as corroboration, never as a sole gate.
        """
        x = np.log(self.close[self.close > 0])
        if len(x) < max_lag * 4:
            return None
        lags = np.arange(2, max_lag)
        tau = np.array([np.std(x[lag:] - x[:-lag]) for lag in lags])
        ok = tau > 0
        if ok.sum() < 3:
            return None
        slope = np.polyfit(np.log(lags[ok]), np.log(tau[ok]), 1)[0]
        return float(slope)

    # ---- volume ----
    @_memo
    def vwap(self, n: int) -> Tuple[Optional[float], Optional[float]]:
        """Rolling VWAP over n volume-bearing samples, and volume-weighted std."""
        if len(self.volume) < n:
            return None, None
        p = self.aligned_close[-n:]
        v = self.volume[-n:]
        vs = v.sum()
        if vs <= 0:
            return None, None
        vw = float((p * v).sum() / vs)
        sd = float(np.sqrt((v * (p - vw) ** 2).sum() / vs))
        return vw, sd

    # ---- micro-bars ----
    @_memo
    def bars(self, k: int):
        """
        OHLC bars built from consecutive groups of k samples, newest bar last and
        complete (a trailing partial group is dropped). Gives candle-based ideas
        (Heikin-Ashi, hammer/shooting star) real highs and lows to work with.
        Returns (open, high, low, close) arrays.
        """
        n = (len(self.close) // k) * k
        if n < k:
            e = np.empty(0)
            return e, e, e, e
        w = self.close[len(self.close) - n:].reshape(-1, k)
        return w[:, 0], w.max(axis=1), w.min(axis=1), w[:, -1]

    def ret(self, n: int) -> Optional[float]:
        if len(self.close) <= n or self.close[-(n + 1)] <= 0:
            return None
        return float(self.close[-1] / self.close[-(n + 1)] - 1.0)
