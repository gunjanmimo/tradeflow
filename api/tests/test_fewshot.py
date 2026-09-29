"""
Few-shot forecasting:

  * the target is the return a long entry at the next open would earn, capped
    at the 15:50 flatten
  * walk-forward never lets a day see itself: day d adapts on days d-K..d-1
  * the calibration shrinks a noise relation to ~0 and keeps a strong one
  * adaptation adapts: when the feature-return relation flips sign every few
    weeks, the static model cannot follow and the adapted ones can
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fewshot import data as fdata, meta, evaluate as fe
from fewshot.chronos import calibrate
from rl import features as F
from rl import synthetic


def test_target_is_next_open_to_horizon_and_capped_at_the_flatten():
    ss = synthetic.make(n_sessions=2, edge_bps=0.0, seed=3)
    y = fdata.forward_returns(ss, horizon=30)
    ss.o = ss.o.astype(np.float64)
    k, t = 3, F.DECISIONS[3]
    assert y[0, k] == pytest.approx(np.log(ss.o[0, t + 31] / ss.o[0, t + 1]) * 1e4, rel=1e-5)
    last = len(F.DECISIONS) - 1                  # decision at 15:44: 30 min would pass the flatten
    t = F.DECISIONS[last]
    assert y[0, last] == pytest.approx(np.log(ss.o[0, F.FLATTEN_BAR] / ss.o[0, t + 1]) * 1e4, rel=1e-5)


def test_calibration_shrinks_noise_and_keeps_signal():
    rng = np.random.default_rng(0)
    p = rng.normal(size=5000)
    a, b = calibrate(p, rng.normal(size=5000))
    assert abs(b) < 0.03
    a, b = calibrate(p, 2.0 * p + rng.normal(size=5000))
    assert b == pytest.approx(2.0, rel=0.05)


def _flipping_market(n_days=120, rows=300, d=8, flip_every=25, seed=0):
    """y = sign(regime) * x0 + noise; the regime flips every `flip_every` days."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_days * rows, d)).astype(np.float32)
    regime = np.repeat(np.where((np.arange(n_days) // flip_every) % 2 == 0, 1.0, -1.0), rows)
    y = (regime * X[:, 0] * 0.5 + rng.normal(size=len(X))).astype(np.float32)
    days = np.repeat(np.arange(n_days), rows)
    start = np.arange(n_days) * rows
    R = meta.Rows(torch.as_tensor(X), torch.as_tensor(y), np.arange(n_days), start, start + rows)
    return R, y, days


def test_walk_forward_adapts_only_on_past_days(monkeypatch):
    R, y, days = _flipping_market(n_days=20)
    seen = []
    real = meta.adapt

    def spy(base, X, yy, steps, lr, batch=4096, gen=None):
        seen.append(len(X))
        return real(base, X, yy, steps, lr, batch, gen)
    monkeypatch.setattr(fe, "adapt", spy)
    base = meta.Forecaster(8, (8,))
    fe.walk_forward("finetune", R, (15, 17), 1.0, k_days=5, inner_steps=1, inner_lr=0.01, base=base)
    assert seen == [5 * 300, 5 * 300]          # days 10-14 for day 15, days 11-15 for day 16


def test_fewshot_follows_a_regime_that_the_static_model_cannot():
    R, y, days = _flipping_market()
    base, _ = meta.train_static(R, (0, 60), (60, 80), "cpu", hidden=(16,), steps_per_eval=50,
                                max_evals=10, patience=3, batch=1024)
    ev = (80, 120)
    span = R.span(*ev)
    static = fe.walk_forward("static", R, ev, 1.0, base=base)
    tuned = fe.walk_forward("finetune", R, ev, 1.0, k_days=5, inner_steps=30, inner_lr=0.1, base=base)
    ic_static = fe.daily_ic(static, y[span], days[span]).mean_ic
    ic_tuned = fe.daily_ic(tuned, y[span], days[span]).mean_ic
    assert abs(ic_static) < 0.15                  # averaged over flips, nothing to learn statically
    assert ic_tuned > 0.2                         # few-shot tracks the current regime
