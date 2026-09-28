"""
Historical sessions -> the arrays the RL environment steps through.

One row per (symbol, trading day), each on the 390-minute grid:

  bar    [S, 390, N_BAR]   per-bar features (rl/features.py)
  scal   [S, 390, N_SCAL]  per-bar scalar context
  o h l c atr [S, 390]     prices for fills, stops and targets; ATR in price units
  sym, day [S]             symbol index and New York day number

Half-days and sessions with too few IEX prints are left out: the grid would be
mostly filled-in prices, which is not a market the policy should learn from.
"""
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from rl import features as F

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "rl")
FEATURE_VERSION = 3
MIN_REAL_BARS = 300              # of 390; half-days (210) and data holes drop out
NON_TRADED = ("SPY", "QQQ")


@dataclass
class SessionSet:
    bar: np.ndarray
    scal: np.ndarray
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    atr: np.ndarray
    sym: np.ndarray
    day: np.ndarray
    symbols: List[str]

    def __len__(self):
        return len(self.day)

    def subset(self, mask: np.ndarray) -> "SessionSet":
        m = np.asarray(mask)
        return SessionSet(self.bar[m], self.scal[m], self.o[m], self.h[m], self.l[m], self.c[m],
                          self.atr[m], self.sym[m], self.day[m], self.symbols)

    def days(self) -> np.ndarray:
        return np.unique(self.day)


def _grid_by_day(bars) -> Dict[int, tuple]:
    from research.data import ny_minute_of_day
    mos = ny_minute_of_day(bars.minute) - F.OPEN_MIN
    out = {}
    for day, sl in bars.day_slices():
        out[day] = (F.to_grid(mos[sl], bars.o[sl], bars.h[sl], bars.l[sl], bars.c[sl], bars.v[sl]),
                    sl)
    return out


def _build_symbol(args):
    symbol, spy_close_by_day = args
    from research import data
    b = data.load(symbol)
    if b is None or len(b) < 1000:
        return symbol, None
    # ATR as the live engine reads it: true range over the last 60 real bars
    # (yesterday's included), floored at 0.05% of price.
    atr_real = F.rolling_atr(b.h, b.l, b.c)
    atr_real = np.maximum(atr_real, b.c * 0.0005)
    from research.data import ny_minute_of_day
    mos = ny_minute_of_day(b.minute) - F.OPEN_MIN
    rows = {k: [] for k in ("bar", "scal", "o", "h", "l", "c", "atr", "day")}
    prev: Optional[F.PrevDay] = None
    for day, sl in b.day_slices():
        go, gh, gl, gc, gv, real = F.to_grid(mos[sl], b.o[sl], b.h[sl], b.l[sl], b.c[sl], b.v[sl])
        full_day = real.sum() >= MIN_REAL_BARS and real[F.SESSION_BARS - 15:].any()
        spy_c = spy_close_by_day.get(day)
        if full_day and prev is not None and spy_c is not None:
            idx = np.clip(mos[sl], 0, F.SESSION_BARS - 1)
            ga = np.full(F.SESSION_BARS, np.nan)
            ga[idx] = atr_real[sl]
            pos = np.maximum.accumulate(np.where(np.isfinite(ga), np.arange(F.SESSION_BARS), -1))
            ga = np.where(pos >= 0, ga[np.maximum(pos, 0)], np.nan)
            first_atr = atr_real[sl][np.isfinite(atr_real[sl])]
            ga = np.where(np.isfinite(ga), ga, first_atr[0] if len(first_atr) else gc * 0.001)
            bar, scal = F.session_features(go, gh, gl, gc, gv, prev, spy_c, ga)
            for k, x in (("bar", bar), ("scal", scal), ("o", go), ("h", gh), ("l", gl),
                         ("c", gc), ("atr", ga)):
                rows[k].append(x.astype(np.float32))
            rows["day"].append(day)
        if real.any():
            prev = F.prev_day_summary(go, gc, gv)
    if not rows["day"]:
        return symbol, None
    return symbol, {k: np.stack(v) if k != "day" else np.asarray(v, dtype=np.int64)
                    for k, v in rows.items()}


def build(symbols: Optional[Sequence[str]] = None, workers: int = 8, use_cache: bool = True,
          verbose: bool = True) -> SessionSet:
    from research import data
    symbols = [s for s in (symbols or data.available()) if s not in NON_TRADED]
    spy = data.load(data.MARKET)
    if spy is None:
        raise SystemExit("SPY bars are needed for market context: python -m research download")
    stamp = {s: [os.path.getmtime(data._path(s)), os.path.getsize(data._path(s))]
             for s in sorted(symbols + [data.MARKET])}
    key = hashlib.sha1(json.dumps({"v": FEATURE_VERSION, "s": stamp}, sort_keys=True).encode()).hexdigest()[:16]
    path = os.path.join(CACHE_DIR, f"sessions_{key}.npz")
    if use_cache and os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        if verbose:
            print(f"Loaded {len(z['day'])} sessions from cache {os.path.basename(path)}")
        return SessionSet(z["bar"], z["scal"], z["o"], z["h"], z["l"], z["c"], z["atr"],
                          z["sym"], z["day"], [str(x) for x in z["symbols"]])

    spy_grid = {day: g[3] for day, (g, _) in _grid_by_day(spy).items() if g[5].sum() >= MIN_REAL_BARS}
    parts, names = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for sym, res in pool.map(_build_symbol, [(s, spy_grid) for s in symbols]):
            if res is None:
                if verbose:
                    print(f"  {sym}: skipped (no usable sessions)")
                continue
            names.append(sym)
            res["sym"] = np.full(len(res["day"]), len(names) - 1, dtype=np.int32)
            parts.append(res)
            if verbose:
                print(f"  {sym}: {len(res['day'])} sessions")
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    order = np.lexsort((cat["sym"], cat["day"]))            # chronological, then by symbol
    cat = {k: v[order] for k, v in cat.items()}
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, symbols=np.asarray(names), **cat)
    os.replace(tmp, path)
    # One cache is enough: each new day of bars makes the old one stale, and
    # each is ~0.8 GB. Remove the rest.
    import glob
    for old in glob.glob(os.path.join(CACHE_DIR, "sessions_*.npz")):
        if os.path.abspath(old) != os.path.abspath(path):
            try:
                os.remove(old)
            except OSError:
                pass
    return SessionSet(cat["bar"], cat["scal"], cat["o"], cat["h"], cat["l"], cat["c"], cat["atr"],
                      cat["sym"], cat["day"], names)


def norm_stats(ss: SessionSet) -> Dict[str, np.ndarray]:
    """Per-feature mean/std over a (training) set, robust to the heavy tails of returns."""
    bar = ss.bar.reshape(-1, ss.bar.shape[-1]).astype(np.float64)
    scal = ss.scal.reshape(-1, ss.scal.shape[-1]).astype(np.float64)

    def robust(x):
        lo, hi = np.nanpercentile(x, [0.5, 99.5], axis=0)
        xc = np.clip(x, lo, hi)
        m, s = np.nanmean(xc, axis=0), np.nanstd(xc, axis=0)
        return m.astype(np.float32), np.where(s > 1e-8, s, 1.0).astype(np.float32)

    bm, bs = robust(bar)
    sm, sd = robust(scal)
    return {"bar_mean": bm, "bar_std": bs, "scal_mean": sm, "scal_std": sd}
