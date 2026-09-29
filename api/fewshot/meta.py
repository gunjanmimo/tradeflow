"""
Training and fast adaptation for the tiny forecaster, on one flat GPU tensor.

All sessions live in one Rows object indexed by trading day, so "the K days
before day d" is a contiguous slice wherever d falls (training, validation or
test). Train / validation / test are just ranges of day indices.

  train_static   ordinary training on the training days, early-stopped on the
                 validation days' loss: the base that never changes
  adapt          few-shot: copy a base, take `steps` SGD steps on the support set
                 (the last K days), return the adapted copy
  fomaml         first-order MAML (Finn et al., 2017; Nichol et al., 2018): trains
                 the base so that `adapt` on K days improves the NEXT day. Each
                 meta-step samples a training day d, adapts on days d-K..d-1 and
                 moves the base along day d's loss gradient taken at the adapted
                 weights -- exactly the job it will do live.
"""
import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from fewshot.models import Forecaster, loss_fn


@dataclass
class Rows:
    X: torch.Tensor           # [N, D]
    y: torch.Tensor           # [N] target / scale
    days: np.ndarray          # sorted unique trading days
    start: np.ndarray         # first row of each day
    end: np.ndarray           # one past its last row

    def span(self, i0: int, i1: int) -> slice:
        """Rows of days[i0:i1]."""
        return slice(int(self.start[i0]), int(self.end[i1 - 1]))


def to_rows(ds, scale: float, device: str) -> Rows:
    """Sessions must be sorted by day (rl.dataset sorts them so)."""
    S, T, D = ds.X.shape
    X = torch.as_tensor(ds.X.reshape(S * T, D), device=device)
    y = torch.as_tensor(ds.y.reshape(-1) / scale, device=device)
    day_rows = np.repeat(ds.day, T)
    assert np.all(np.diff(day_rows) >= 0), "sessions must be sorted by day"
    days, start = np.unique(day_rows, return_index=True)
    end = np.r_[start[1:], len(day_rows)]
    return Rows(X, y, days, start, end)


def target_scale(y: np.ndarray) -> float:
    """Robust spread of the target (MAD x 1.4826), in bps."""
    y = y.reshape(-1)
    return float(np.median(np.abs(y - np.median(y))) * 1.4826) or 1.0


@torch.no_grad()
def eval_loss(model, X, y, batch: int = 65536) -> float:
    tot = 0.0
    for i in range(0, len(X), batch):
        tot += float(loss_fn(model(X[i:i + batch]), y[i:i + batch])) * len(X[i:i + batch])
    return tot / max(len(X), 1)


def adapt(base: Forecaster, X: torch.Tensor, y: torch.Tensor, steps: int, lr: float,
          batch: int = 4096, gen: Optional[torch.Generator] = None) -> Forecaster:
    m = copy.deepcopy(base)
    m.train()
    opt = torch.optim.SGD(m.parameters(), lr=lr)
    n = len(X)
    for _ in range(steps):
        idx = torch.randint(0, n, (min(batch, n),), device=X.device, generator=gen)
        loss = loss_fn(m(X[idx]), y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return m


def train_static(R: Rows, train: Tuple[int, int], val: Tuple[int, int], device: str, hidden=(64, 32),
                 lr: float = 1e-3, batch: int = 4096, steps_per_eval: int = 200, max_evals: int = 60,
                 patience: int = 6, seed: int = 0, wd: float = 1e-4,
                 verbose: bool = False) -> Tuple[Forecaster, List[dict]]:
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    model = Forecaster(R.X.shape[1], hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    tr, va = R.span(*train), R.span(*val)
    Xt, yt = R.X[tr], R.y[tr]
    best, best_state, stale, curve = float("inf"), copy.deepcopy(model.state_dict()), 0, []
    for e in range(max_evals):
        for _ in range(steps_per_eval):
            idx = torch.randint(0, len(Xt), (batch,), device=device, generator=gen)
            loss = loss_fn(model(Xt[idx]), yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        vl = eval_loss(model, R.X[va], R.y[va])
        curve.append({"step": (e + 1) * steps_per_eval, "val_loss": round(vl, 5)})
        if verbose:
            print(f"    static  step {(e + 1) * steps_per_eval:6d}  validation loss {vl:.5f}")
        if vl < best - 1e-6:
            best, best_state, stale = vl, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model, curve


def adapted_loss(model: Forecaster, R: Rows, day_idx: np.ndarray, k_days: int, steps: int,
                 lr: float, batch: int, gen) -> float:
    """Mean loss on each given day after adapting on the k_days before it."""
    out = []
    for d in day_idx:
        if d < k_days:
            continue
        s = R.span(d - k_days, d)
        a = adapt(model, R.X[s], R.y[s], steps, lr, batch, gen)
        q = R.span(d, d + 1)
        out.append(eval_loss(a, R.X[q], R.y[q]))
    return float(np.mean(out)) if out else float("nan")


def fomaml(R: Rows, train: Tuple[int, int], val: Tuple[int, int], device: str, k_days: int,
           inner_steps: int, inner_lr: float, hidden=(64, 32), meta_lr: float = 1e-3,
           meta_steps: int = 3000, tasks_per_step: int = 4, batch: int = 4096, eval_every: int = 250,
           patience: int = 6, seed: int = 0, init: Optional[Forecaster] = None,
           verbose: bool = False) -> Tuple[Forecaster, List[dict]]:
    """Meta-trains on training days; picks the checkpoint by the adapted loss on validation days."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    model = (copy.deepcopy(init) if init is not None else Forecaster(R.X.shape[1], hidden)).to(device)
    meta_opt = torch.optim.Adam(model.parameters(), lr=meta_lr)
    val_days = np.linspace(val[0], val[1] - 1, 16).astype(int)
    best, best_state, stale, curve = float("inf"), copy.deepcopy(model.state_dict()), 0, []
    for step in range(1, meta_steps + 1):
        grads = [torch.zeros_like(p) for p in model.parameters()]
        for _ in range(tasks_per_step):
            d = int(rng.integers(train[0] + k_days, train[1]))
            sup = R.span(d - k_days, d)
            a = adapt(model, R.X[sup], R.y[sup], inner_steps, inner_lr, batch, gen)
            q = R.span(d, d + 1)
            ql = loss_fn(a(R.X[q]), R.y[q])
            for acc, g in zip(grads, torch.autograd.grad(ql, list(a.parameters()))):
                acc += g / tasks_per_step
        meta_opt.zero_grad(set_to_none=True)
        for p, g in zip(model.parameters(), grads):
            p.grad = g
        meta_opt.step()
        if step % eval_every == 0:
            vl = adapted_loss(model, R, val_days, k_days, inner_steps, inner_lr, batch, gen)
            curve.append({"meta_step": step, "val_adapted_loss": round(vl, 5)})
            if verbose:
                print(f"    fomaml  step {step:5d}  validation loss after adapting {vl:.5f}")
            if vl < best - 1e-6:
                best, best_state, stale = vl, copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
                if stale >= patience:
                    break
    model.load_state_dict(best_state)
    return model, curve
