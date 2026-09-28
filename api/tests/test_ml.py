"""
Trade scorer: features see no future bars, the bot is unaffected without a
model, live experience records the whole net result, and training learns a
planted pattern and refuses to save a model that does not help.
"""
import json

import numpy as np
import pytest

from ml import experience as exp_mod
from ml.features import WINDOW, N_BAR_FEATURES, bar_features
from ml.model import Scorer


def _bars(n, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    return np.arange(n) + 29_000_000, c, c * 1.001, c * 0.999, c, rng.uniform(100, 200, n)


def test_features_shape_and_no_lookahead():
    m, o, h, l, c, v = _bars(150)
    f = bar_features(m[:120], o[:120], h[:120], l[:120], c[:120], v[:120])
    assert f.shape == (WINDOW, N_BAR_FEATURES) and f.dtype == np.float32
    # Changing bars after the decision point must not change the features.
    c2 = c.copy()
    c2[120:] *= 2
    g = bar_features(m[:120], o[:120], h[:120], l[:120], c2[:120], v[:120])
    assert np.array_equal(f, g)
    assert bar_features(m[:30], o[:30], h[:30], l[:30], c[:30], v[:30]) is None


def test_scorer_without_a_model_changes_nothing(tmp_path):
    s = Scorer(str(tmp_path / "missing.pt"))
    assert s.score([{"bars": np.zeros((WINDOW, N_BAR_FEATURES), np.float32),
                     "strategy": "x", "crypto": False}]) is None


def test_experience_records_the_whole_net_result(tmp_path, monkeypatch):
    monkeypatch.setattr(exp_mod, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(exp_mod, "EXPERIENCE_PATH", str(tmp_path / "e.jsonl"))
    monkeypatch.setattr(exp_mod, "PENDING_PATH", str(tmp_path / "p.json"))
    e = exp_mod.Experience()
    e.on_entry("NVDA", np.zeros((WINDOW, N_BAR_FEATURES), np.float32), "supertrend", False)
    assert json.load(open(tmp_path / "p.json"))["NVDA"]["strategy"] == "supertrend"
    pos = {"invested_dollars": 1000.0, "harvested_income": 1.5, "trimmed_pnl": 0.5}
    e.on_close("NVDA", pos, -1.0)                 # final sale -1, harvest +1.5, trim +0.5
    rec = json.loads(open(tmp_path / "e.jsonl").read())
    assert rec["net_pnl"] == pytest.approx(1.0) and rec["ret_pct"] == pytest.approx(0.1)
    e.on_close("NVDA", pos, 5.0)                  # no pending entry: nothing recorded
    assert len(open(tmp_path / "e.jsonl").read().splitlines()) == 1




def _planted(n=1200, seed=1):
    """Trades whose result follows the sign of the last bar's return: learnable."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        b = rng.normal(0, 1, (WINDOW, N_BAR_FEATURES)).astype(np.float32)
        rows.append({"bars": b, "strat": 13, "crypto": False, "t": float(i), "live": i % 5 == 0,
                     "ret": (0.5 if b[-1, 0] > 0 else -0.5) + rng.normal(0, 0.05)})
    return rows


def test_training_learns_a_planted_pattern():
    torch = pytest.importorskip("torch")
    from ml.train import train
    r = train(_planted(), epochs=15, device="cpu", verbose=False)["report"]
    assert r["test_auc"] > 0.8 and r["improves"]


def test_training_does_not_claim_a_pattern_in_noise():
    torch = pytest.importorskip("torch")
    from ml.train import train
    rows = _planted()
    rng = np.random.default_rng(7)
    for row in rows:
        row["ret"] = float(rng.normal(-0.1, 0.5))    # results unrelated to the bars
    r = train(rows, epochs=8, device="cpu", verbose=False)["report"]
    assert r["test_auc"] < 0.65
