"""
The RL observation. One implementation, used by the training environment (over
whole historical sessions) and by the live engine (over the bars so far today).

Every feature at bar t uses only bars 0..t of the session, the previous
session's summary, and SPY up to t. So a feature row computed from a partial
session equals the row computed from the full session -- tests hold live and
offline to that.

A session is the regular US session on a one-minute grid: 390 bars, 09:30 to
15:59 New York. Minutes without an IEX print are filled with the last close and
zero volume (the historical store and the live stream both skip them).

Observation at decision bar t:
  window    the last WINDOW bars of BAR_FEATURES (zero-padded before the open)
  scalars   SCALAR_FEATURES at bar t
  position  POSITION_FEATURES from the environment / the live position
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

SESSION_BARS = 390
OPEN_MIN = 570                     # 09:30 NY in minutes after midnight
WINDOW = 30                        # default; a policy records its own in its metadata

BAR_FEATURES = ("r1", "range", "body", "vol_z", "vwap_dev", "ema9_dev", "ema21_dev", "rsi14", "spy_r1")
SCALAR_FEATURES = ("tod_sin", "tod_cos", "to_close", "ret_open", "gap", "prev_ret", "vol_ratio",
                   "spy_ret_open", "spy_r5", "spy_r15", "spy_r30", "spy_r60",
                   "rel_r5", "rel_r15", "rel_r30", "rel_r60", "atr_pct")
POSITION_FEATURES = ("in_pos", "unreal", "held", "to_stop", "to_target")
N_BAR, N_SCAL, N_POS = len(BAR_FEATURES), len(SCALAR_FEATURES), len(POSITION_FEATURES)
OBS_DIM = WINDOW * N_BAR + N_SCAL + N_POS


def obs_dim(window: int = WINDOW) -> int:
    return window * N_BAR + N_SCAL + N_POS

# Decision schedule (the environment and the live runtime both use it): decide
# at the close of bars 4, 9, ..., 374; an order fills at the next bar's open.
DECISION_EVERY = 5
FIRST_DECISION = 4                  # close of 09:34, fill at 09:35
LAST_ENTRY_DECISION = 358           # fill at 15:29; 15:30+ is inside the no-new-entry cutoff
FLATTEN_BAR = 380                   # 15:50 open: the flatten fill
DECISIONS = list(range(FIRST_DECISION, FLATTEN_BAR - DECISION_EVERY + 1, DECISION_EVERY))
N_STEPS = len(DECISIONS)            # 75

ATR_WINDOW = 60                    # bars, as engine/quant_matrix.py reads live
ATR_PERIOD = 14


@dataclass
class PrevDay:
    """What a session needs from the one before it."""
    close: float = float("nan")
    open: float = float("nan")
    r1_std: float = float("nan")       # std of one-minute log returns
    logv_mean: float = float("nan")    # mean of log1p(volume)

    @property
    def known(self) -> bool:
        return math.isfinite(self.close) and self.close > 0


# ---------------------------------------------------------------------------
# Grid and indicators
# ---------------------------------------------------------------------------

def to_grid(minute_of_session: np.ndarray, o, h, l, c, v) -> Tuple[np.ndarray, ...]:
    """
    Places one session's bars on the 390-minute grid. Bars outside 0..389 are
    dropped; missing minutes carry the last close forward with zero volume.
    Returns (o, h, l, c, v, real) with real marking minutes that had a print.
    """
    idx = np.asarray(minute_of_session, dtype=np.int64)
    keep = (idx >= 0) & (idx < SESSION_BARS)
    idx = idx[keep]
    go, gh, gl, gc, gv = (np.full(SESSION_BARS, np.nan) for _ in range(5))
    go[idx], gh[idx], gl[idx], gc[idx] = (np.asarray(x, dtype=np.float64)[keep] for x in (o, h, l, c))
    gv[:] = 0.0
    gv[idx] = np.asarray(v, dtype=np.float64)[keep]
    real = np.zeros(SESSION_BARS, dtype=bool)
    real[idx] = True
    if not real.any():
        return go, gh, gl, gc, gv, real
    first = int(np.flatnonzero(real)[0])
    pos = np.where(real, np.arange(SESSION_BARS), -1)
    pos = np.maximum.accumulate(pos)
    last_c = np.where(pos >= 0, gc[np.maximum(pos, 0)], go[first])
    fill = ~real
    go[fill] = gh[fill] = gl[fill] = gc[fill] = last_c[fill]
    return go, gh, gl, gc, gv, real


def ema_path(x: np.ndarray, span: int, seed: float) -> np.ndarray:
    a = 2.0 / (span + 1.0)
    out = np.empty(len(x))
    acc = seed
    for i, val in enumerate(x):
        acc = a * val + (1.0 - a) * acc
        out[i] = acc
    return out


def rsi_path(c: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder RSI along the session; 50 until `period` changes exist."""
    d = np.diff(c, prepend=c[0])
    g, l = np.clip(d, 0, None), np.clip(-d, 0, None)
    out = np.full(len(c), 50.0)
    if len(c) <= period:
        return out
    ag, al = g[1:period + 1].mean(), l[1:period + 1].mean()
    for i in range(period, len(c)):
        if i > period:
            ag = (ag * (period - 1) + g[i]) / period
            al = (al * (period - 1) + l[i]) / period
        out[i] = 100.0 if al == 0 and ag > 0 else (50.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al))
    return out


def rolling_atr(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    For every bar t, QuantMatrix.true_range_atr over bars t-59..t (Wilder, period
    14, seeded with the mean of the window's first 14 true ranges) -- vectorised.
    NaN before 60 bars exist.
    """
    h, l, c = (np.asarray(x, dtype=np.float64) for x in (h, l, c))
    n = len(c)
    out = np.full(n, np.nan)
    if n < ATR_WINDOW:
        return out
    prev = np.r_[np.nan, c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))   # tr[j] needs bar j-1
    k = (ATR_PERIOD - 1) / ATR_PERIOD
    steps = ATR_WINDOW - 1 - ATR_PERIOD                       # 45 recursion steps
    # seed: mean of tr[t-58 .. t-45]; then tr[t-44 .. t] with weights k^(44-i)/14
    cs = np.cumsum(np.r_[0.0, np.nan_to_num(tr)])
    t = np.arange(ATR_WINDOW - 1, n)
    seed = (cs[t - 44] - cs[t - 58]) / ATR_PERIOD
    w = (k ** np.arange(steps - 1, -1, -1)) / ATR_PERIOD      # oldest first
    tail = np.convolve(np.nan_to_num(tr), w[::-1], mode="full")[:n]   # sum_{i} tr[t-44+i]*w[i]
    out[t] = seed * k ** steps + tail[t]
    return out


# ---------------------------------------------------------------------------
# Session features
# ---------------------------------------------------------------------------

def session_features(o, h, l, c, v, prev: PrevDay, spy_c: Optional[np.ndarray] = None,
                     atr: Optional[np.ndarray] = None, upto: Optional[int] = None
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    (bar_feats [n, N_BAR], scalars [n, N_SCAL]) for grid bars 0..n-1, n = upto+1
    (default: the whole session). spy_c is SPY's grid close; atr is the ATR per
    grid bar (price units).
    """
    n = SESSION_BARS if upto is None else int(upto) + 1
    o, h, l, c, v = (np.asarray(x, dtype=np.float64)[:n] for x in (o, h, l, c, v))
    lc = np.log(c)
    r1 = np.empty(n)
    r1[0] = math.log(c[0] / o[0])
    r1[1:] = np.diff(lc)
    logv = np.log1p(np.maximum(v, 0.0))
    if math.isfinite(prev.logv_mean):
        vol_z = logv - prev.logv_mean
    else:
        vol_z = logv - np.cumsum(logv) / np.arange(1, n + 1)
    tp = (h + l + c) / 3.0
    cv = np.cumsum(v)
    vwap = np.where(cv > 0, np.cumsum(tp * v) / np.where(cv > 0, cv, 1.0), c)
    bar = np.empty((n, N_BAR))
    bar[:, 0] = r1 * 1e3
    bar[:, 1] = (h - l) / c * 1e3
    bar[:, 2] = (c - o) / c * 1e3
    bar[:, 3] = vol_z
    bar[:, 4] = np.log(c / vwap) * 1e3
    bar[:, 5] = np.log(c / ema_path(c, 9, o[0])) * 1e3
    bar[:, 6] = np.log(c / ema_path(c, 21, o[0])) * 1e3
    bar[:, 7] = rsi_path(c) / 100.0 - 0.5
    if spy_c is not None:
        sc = np.asarray(spy_c, dtype=np.float64)[:n]
        sr = np.diff(np.log(sc), prepend=math.log(sc[0]))
        bar[:, 8] = sr * 1e3
    else:
        sc = None
        bar[:, 8] = 0.0

    t = np.arange(n)
    sc_ = np.empty((n, N_SCAL))
    ang = 2 * math.pi * t / SESSION_BARS
    sc_[:, 0] = np.sin(ang)
    sc_[:, 1] = np.cos(ang)
    sc_[:, 2] = (SESSION_BARS - 1 - t) / SESSION_BARS
    sc_[:, 3] = (lc - math.log(o[0])) * 100
    sc_[:, 4] = math.log(o[0] / prev.close) * 100 if prev.known else 0.0
    sc_[:, 5] = (math.log(prev.close / prev.open) * 100
                 if prev.known and math.isfinite(prev.open) and prev.open > 0 else 0.0)
    # realised vol so far vs yesterday's
    csum, csum2 = np.cumsum(r1), np.cumsum(r1 * r1)
    cnt = np.arange(1, n + 1)
    var = np.maximum(csum2 / cnt - (csum / cnt) ** 2, 0.0)
    ref = prev.r1_std if (math.isfinite(prev.r1_std) and prev.r1_std > 0) else None
    sc_[:, 6] = np.log((np.sqrt(var) + 1e-6) / ref) if ref else 0.0
    sc_[:, 6] = np.where(cnt >= 5, sc_[:, 6], 0.0)

    def lag_ret(logp, k):
        out = np.zeros(n)
        if n > k:
            out[k:] = (logp[k:] - logp[:-k]) * 100
        out[1:min(k, n)] = (logp[1:min(k, n)] - logp[0]) * 100
        return out

    if sc is not None:
        ls = np.log(sc)
        sc_[:, 7] = (ls - ls[0]) * 100
        for j, k in enumerate((5, 15, 30, 60)):
            sr_k = lag_ret(ls, k)
            sc_[:, 8 + j] = sr_k
            sc_[:, 12 + j] = lag_ret(lc, k) - sr_k
    else:
        sc_[:, 7:12] = 0.0
        for j, k in enumerate((5, 15, 30, 60)):
            sc_[:, 12 + j] = lag_ret(lc, k)
    if atr is not None:
        a = np.asarray(atr, dtype=np.float64)[:n]
        sc_[:, 16] = np.where(np.isfinite(a), a / c * 100, 0.0)
    else:
        sc_[:, 16] = 0.0
    return bar, sc_


def prev_day_summary(o, c, v) -> PrevDay:
    """Summary of a (grid) session for the next day's features."""
    o, c, v = (np.asarray(x, dtype=np.float64) for x in (o, c, v))
    r1 = np.empty(len(c))
    r1[0] = math.log(c[0] / o[0])
    r1[1:] = np.diff(np.log(c))
    return PrevDay(close=float(c[-1]), open=float(o[0]), r1_std=float(r1.std()),
                   logv_mean=float(np.log1p(np.maximum(v, 0.0)).mean()))


def window(bar: np.ndarray, t: int, w: int = WINDOW) -> Tuple[np.ndarray, int]:
    """((w, N_BAR) ending at bar t, rows that are real bars). Rows before the open are padding."""
    out = np.zeros((w, bar.shape[1]), dtype=np.float32)
    lo = max(0, t - w + 1)
    seg = bar[lo:t + 1]
    out[w - len(seg):] = seg
    return out, len(seg)


def position_features(in_pos: bool, price: float, entry: float, stop: float, target: float,
                      bars_held: int) -> np.ndarray:
    """Position state, in % of price (log), and time held as a fraction of a session."""
    if not in_pos or entry <= 0:
        return np.zeros(N_POS, dtype=np.float32)
    return np.array([1.0,
                     math.log(price / entry) * 100,
                     bars_held / SESSION_BARS,
                     math.log(price / stop) * 100 if stop > 0 else 0.0,
                     math.log(target / price) * 100 if target > 0 else 0.0], dtype=np.float32)


def assemble(win: np.ndarray, n_real: int, scal: np.ndarray, pos: np.ndarray, norm: dict) -> np.ndarray:
    """One normalised observation vector (float32, OBS_DIM). Padding rows read as 0 (average)."""
    wz = (win - norm["bar_mean"]) / norm["bar_std"]
    wz[:win.shape[0] - n_real] = 0.0
    sz = (scal - norm["scal_mean"]) / norm["scal_std"]
    x = np.concatenate([wz.reshape(-1), sz, pos]).astype(np.float32)
    return np.clip(x, -8.0, 8.0)
