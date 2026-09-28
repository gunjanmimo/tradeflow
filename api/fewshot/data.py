"""
The forecasting dataset: one row per (session, decision bar).

  X  [S, 75, D]  the RL observation without position features (rl/features.py):
                 the last 30 bars of per-bar features + session/market scalars,
                 normalised with statistics from the TRAINING days only
  y  [S, 75]     the forward log return in bps from the next bar's open (the fill)
                 over `horizon` minutes, or to the 15:50 flatten if that is sooner

The target is what a long entry at that decision would earn before costs, so a
forecast above the round-trip cost is a trade worth taking.
"""
from dataclasses import dataclass
from typing import Dict

import numpy as np

from rl import features as F
from rl.dataset import SessionSet

DECISIONS = np.asarray(F.DECISIONS)
D_IN = F.WINDOW * F.N_BAR + F.N_SCAL


@dataclass
class DecisionSet:
    X: np.ndarray          # float32 [S, T, D]
    y: np.ndarray          # float32 [S, T] bps
    day: np.ndarray        # [S]
    sym: np.ndarray        # [S]
    horizon: int

    def __len__(self):
        return len(self.day)


def forward_returns(ss: SessionSet, horizon: int) -> np.ndarray:
    t = DECISIONS
    entry = ss.o[:, t + 1].astype(np.float64)
    exit_i = np.minimum(t + 1 + horizon, F.FLATTEN_BAR)
    ex = ss.o[:, exit_i].astype(np.float64)
    return (np.log(ex / entry) * 1e4).astype(np.float32)


def build(ss: SessionSet, norm: Dict[str, np.ndarray], horizon: int = 30,
          chunk: int = 2048) -> DecisionSet:
    S, T = len(ss), len(DECISIONS)
    X = np.empty((S, T, D_IN), dtype=np.float32)
    offs = np.arange(-F.WINDOW + 1, 1)
    for a in range(0, S, chunk):
        b = min(S, a + chunk)
        bar = (ss.bar[a:b] - norm["bar_mean"]) / norm["bar_std"]
        scal = (ss.scal[a:b] - norm["scal_mean"]) / norm["scal_std"]
        for k, t in enumerate(DECISIONS):
            idx = t + offs
            win = bar[:, np.clip(idx, 0, None)]
            win[:, idx < 0] = 0.0                       # padding before the open reads as average
            X[a:b, k, :F.WINDOW * F.N_BAR] = win.reshape(b - a, -1)
            X[a:b, k, F.WINDOW * F.N_BAR:] = scal[:, t]
    np.clip(X, -8, 8, out=X)
    return DecisionSet(X=X, y=forward_returns(ss, horizon), day=ss.day, sym=ss.sym, horizon=horizon)
