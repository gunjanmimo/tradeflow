"""
Walk-forward evaluation: every evaluation day is predicted by a model that has
seen only the days before it.

  static           the base as trained; never updated
  finetune         each day: the static base + a few SGD steps on the last K days
  fomaml           each day: the meta-learned base + the same few steps
  chronos_zero     Chronos-Bolt's forecast as it comes
  chronos_fewshot  each day: Chronos recalibrated on the last K days

Forecasts are scored two ways:
  IC       per day, the correlation between forecast and realised return across
           all stocks and decision bars; its mean and a t-stat over days. An IC
           indistinguishable from 0 means there is nothing to trade.
  trading  a fixed rule run through the RL environment (same fills, brackets,
           costs and close): enter when the forecast exceeds the round-trip
           cost, exit when it turns negative. Compared with flat, always-long
           and random at the same turnover.
"""
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Tuple

import numpy as np
import torch

from fewshot.meta import Rows, adapt
from fewshot import chronos as ch


@dataclass
class ICResult:
    mean_ic: float
    t_ic: float
    days: int
    positive_days: float

    def to_dict(self):
        return asdict(self)


def daily_ic(pred: np.ndarray, y: np.ndarray, day: np.ndarray) -> ICResult:
    ics = []
    for d in np.unique(day):
        m = day == d
        p, r = pred[m], y[m]
        if p.std() > 1e-12 and r.std() > 1e-12:
            ics.append(float(np.corrcoef(p, r)[0, 1]))
    ics = np.asarray(ics)
    if len(ics) < 2:
        return ICResult(float("nan"), float("nan"), len(ics), float("nan"))
    return ICResult(round(float(ics.mean()), 4), round(float(ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))), 2),
                    len(ics), round(float((ics > 0).mean()), 2))


@torch.no_grad()
def _predict(model, X: torch.Tensor, batch: int = 65536) -> np.ndarray:
    model.eval()
    return torch.cat([model(X[i:i + batch]) for i in range(0, len(X), batch)]).cpu().numpy()


def walk_forward(kind: str, R: Rows, days: Tuple[int, int], scale: float, k_days: int = 10,
                 inner_steps: int = 20, inner_lr: float = 0.05, base=None,
                 chronos_rows: np.ndarray = None, seed: int = 0) -> np.ndarray:
    """Forecasts (bps) for every row of days[d0:d1], each day from the past only."""
    gen = torch.Generator(device=R.X.device).manual_seed(seed)
    span = R.span(*days)
    if kind == "static":
        return _predict(base, R.X[span]) * scale
    if kind == "chronos_zero":
        return chronos_rows[span]
    out = []
    y_bps = None
    for d in range(days[0], days[1]):
        q = R.span(d, d + 1)
        s = R.span(max(0, d - k_days), d)
        if kind in ("finetune", "fomaml"):
            m = adapt(base, R.X[s], R.y[s], inner_steps, inner_lr, gen=gen)
            out.append(_predict(m, R.X[q]) * scale)
        elif kind == "chronos_fewshot":
            if y_bps is None:
                y_bps = R.y.cpu().numpy() * scale
            a, b = ch.calibrate(chronos_rows[s], y_bps[s])
            out.append(a + b * chronos_rows[q])
        else:
            raise ValueError(kind)
    return np.concatenate(out)


def trade(data, pred: np.ndarray, rules, enter_bps: float, exit_bps: float = 0.0,
          device: str = "cuda") -> Dict[str, np.ndarray]:
    """Plays every session with the threshold rule on `pred` [S, T]."""
    from rl.evaluate import run
    P = torch.as_tensor(pred, dtype=torch.float32, device=device)

    def policy(obs, mask, env):
        p = P[env.s, env.k]
        inpos = env.pos > 0
        want = torch.where(inpos, p > exit_bps, p > enter_bps)
        return (want & (mask[:, 1] | inpos)).long()

    return run(data, np.arange(len(pred)), rules, policy)
