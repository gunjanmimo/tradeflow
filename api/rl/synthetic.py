"""
Synthetic markets for checking that the learner learns.

A real market may simply not contain an edge the features can see; then the
right policy is to stay flat, and a PPO run that ends up flat proves nothing
about the learner. These markets have a KNOWN answer:

  planted edge   a hidden regime z in {-1, +1} switches every 15-45 minutes and
                 pushes price by `edge_bps` per minute in its direction. z is
                 written into one observable feature. Optimal play: long while
                 z = +1, flat while z = -1. A working learner must find it.
  no edge        the same market with edge_bps = 0. Every trade loses the costs;
                 a working learner must learn to stop trading.
"""
import math

import numpy as np

from rl import features as F
from rl.dataset import SessionSet

SIGNAL_FEATURE = F.SCALAR_FEATURES.index("prev_ret")   # the column that carries z


def make(n_sessions: int = 400, edge_bps: float = 3.0, noise_bps: float = 6.0,
         n_symbols: int = 4, seed: int = 0) -> SessionSet:
    rng = np.random.default_rng(seed)
    n = F.SESSION_BARS
    bars, scals, os_, hs, ls, cs, atrs, syms, days = [], [], [], [], [], [], [], [], []
    for i in range(n_sessions):
        z = np.empty(n)
        t = 0
        cur = rng.choice([-1.0, 1.0])
        while t < n:
            span = int(rng.integers(15, 46))
            z[t:t + span] = cur
            cur = -cur
            t += span
        r = (edge_bps * z + rng.normal(0, noise_bps, n)) / 1e4
        c = 100.0 * np.exp(np.cumsum(r))
        o = np.r_[100.0, c[:-1]]
        h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, noise_bps / 2e4, n)))
        l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, noise_bps / 2e4, n)))
        v = rng.integers(1000, 5000, n).astype(float)
        prev = F.PrevDay(close=100.0, open=100.0, r1_std=noise_bps / 1e4, logv_mean=math.log1p(3000))
        atr = np.full(n, 100.0 * noise_bps / 1e4 * 1.5)
        bar, scal = F.session_features(o, h, l, c, v, prev, c, atr)
        scal[:, SIGNAL_FEATURE] = z            # the regime, observable at every bar
        bars.append(bar.astype(np.float32)); scals.append(scal.astype(np.float32))
        os_.append(o); hs.append(h); ls.append(l); cs.append(c); atrs.append(atr)
        syms.append(i % n_symbols); days.append(20000 + i // n_symbols)
    st = lambda x: np.stack(x).astype(np.float32)
    return SessionSet(st(bars), st(scals), st(os_), st(hs), st(ls), st(cs), st(atrs),
                      np.asarray(syms, dtype=np.int32), np.asarray(days, dtype=np.int64),
                      [f"SYN{k}" for k in range(n_symbols)])
