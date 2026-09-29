"""
The RL layer:

  * the environment pays exactly what the account would: fills at the next
    open with costs, stops before targets, gaps at the open, flat at 15:50,
    no new entries after the cutoff
  * features computed live from a partial session equal the training features
  * the exported numpy policy matches the torch network
  * the learner learns: it finds a planted edge, and stays flat without one
  * the live runtime decides only at decision bars and trades only when allowed
"""
import math
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from engine.quant_matrix import QuantMatrix
from rl import features as F
from rl import env as E
from rl import synthetic
from rl.dataset import SessionSet

NY = ZoneInfo("America/New_York")
PS = 3e-4


def _flat_set(n=1, price=100.0):
    """Sessions with a perfectly flat price (the prices are then edited per test)."""
    z = lambda: np.full((n, F.SESSION_BARS), price, dtype=np.float32)
    return SessionSet(bar=np.zeros((n, F.SESSION_BARS, F.N_BAR), np.float32),
                      scal=np.zeros((n, F.SESSION_BARS, F.N_SCAL), np.float32),
                      o=z(), h=z(), l=z(), c=z(), atr=np.full((n, F.SESSION_BARS), 0.1, np.float32),
                      sym=np.zeros(n, np.int32), day=np.arange(n, dtype=np.int64), symbols=["T"])


NORM = {"bar_mean": np.zeros(F.N_BAR, np.float32), "bar_std": np.ones(F.N_BAR, np.float32),
        "scal_mean": np.zeros(F.N_SCAL, np.float32), "scal_std": np.ones(F.N_SCAL, np.float32)}


def _play(ss, actions_by_step, rules=None):
    """Plays one session with a scripted action per decision step. Returns (total bps, stats)."""
    data = E.SessionTensors(ss, NORM, "cpu")
    env = E.TradingEnv(data, rules or E.EnvRules(per_side=PS), 1, sessions=torch.tensor([0]))
    obs, mask = env.reset()
    total = 0.0
    for k in range(E.N_STEPS):
        a = torch.tensor([actions_by_step(k, mask)])
        obs, mask, r, done = env.step(a)
        total += float(r[0]) * E.REWARD_UNIT
    return total, {k: float(v[0]) for k, v in env.stats.items()}


def test_round_trip_pays_both_sides_and_the_move():
    ss = _flat_set()
    ss.o[0, 10:], ss.h[0, 10:], ss.l[0, 10:], ss.c[0, 10:] = 101.0, 101.0, 101.0, 101.0
    # long at decision 0 (fill at bar 5 = 100), flat at decision 2 (fill at bar 15 = 101)
    total, st = _play(ss, lambda k, m: 1 if k < 2 else 0)
    expect = (math.log(101 * (1 - PS)) - math.log(100 * (1 + PS))) * 1e4
    assert total == pytest.approx(expect, abs=1e-3)
    assert st["trades"] == 1 and st["policy_exits"] == 1


def test_flat_policy_earns_exactly_zero():
    ss = synthetic.make(n_sessions=1, edge_bps=3.0)
    total, st = _play(ss, lambda k, m: 0)
    assert total == 0.0 and st["trades"] == 0


def test_stop_beats_target_in_the_same_bar_and_pays_the_cost():
    ss = _flat_set()
    ss.h[0, 7], ss.l[0, 7] = 150.0, 50.0                       # both touched on bar 7
    total, st = _play(ss, lambda k, m: 1)
    fill = 100 * (1 + PS)
    stop = fill - fill * 0.008                                  # ATR tiny -> the 0.8% floor
    assert st["stops"] >= 1 and st["targets"] == 0
    first = (math.log(stop * (1 - PS)) - math.log(fill)) * 1e4
    assert total <= first + 1e-3                                # later re-entries only lose costs


def test_target_is_a_limit_without_cost():
    ss = _flat_set()
    ss.h[0, 8] = 110.0
    total, st = _play(ss, lambda k, m: 1 if k == 0 else 0)
    fill = 100 * (1 + PS)
    tp = fill + fill * 0.008 * 2.0
    assert st["targets"] == 1
    assert total == pytest.approx((math.log(tp) - math.log(fill)) * 1e4, abs=1e-3)


def test_gap_through_the_stop_fills_at_the_open():
    ss = _flat_set()
    ss.o[0, 12:], ss.h[0, 12:], ss.l[0, 12:], ss.c[0, 12:] = 90.0, 90.0, 90.0, 90.0
    # long from bar 5; the gap comes at bar 12, inside the second step
    total, st = _play(ss, lambda k, m: 1 if k < 2 else 0)
    assert st["stops"] == 1
    assert total == pytest.approx((math.log(90 * (1 - PS)) - math.log(100 * (1 + PS))) * 1e4, abs=1e-3)


def test_everything_is_flat_at_1550_and_no_late_entries():
    ss = _flat_set()
    seen_masks = []

    def act(k, mask):
        seen_masks.append(bool(mask[0, 1]))
        return 1 if E.DECISIONS[k] >= 300 else 0               # buy late in the day, hold
    total, st = _play(ss, act)
    assert st["eod_exits"] == 1 and st["trades"] == 1
    late = [m for k, m in zip(range(E.N_STEPS), seen_masks) if E.DECISIONS[k] > E.LAST_ENTRY_DECISION]
    assert late and all(late)                                   # held: may stay long
    ss2 = _flat_set()
    _, st2 = _play(ss2, lambda k, m: 1 if E.DECISIONS[k] > E.LAST_ENTRY_DECISION else 0)
    assert st2["trades"] == 0                                   # flat: may not open late


# ---- features ------------------------------------------------------------------

def _market_day(seed=0, n=F.SESSION_BARS, p0=100.0):
    rng = np.random.default_rng(seed)
    c = p0 * np.exp(np.cumsum(rng.normal(0, 8e-4, n)))
    o = np.r_[p0, c[:-1]]
    h = np.maximum(o, c) * 1.0004
    l = np.minimum(o, c) * 0.9996
    v = rng.integers(100, 9000, n).astype(float)
    return o, h, l, c, v


def test_live_partial_features_equal_the_training_features():
    o, h, l, c, v = _market_day(1)
    spy = _market_day(2, p0=500.0)[3]
    atr = np.full(F.SESSION_BARS, 0.08)
    prev = F.PrevDay(close=99.0, open=98.0, r1_std=7e-4, logv_mean=7.5)
    full_bar, full_scal = F.session_features(o, h, l, c, v, prev, spy, atr)
    for t in (0, 4, 37, 200, 389):
        bar, scal = F.session_features(o, h, l, c, v, prev, spy, atr, upto=t)
        np.testing.assert_allclose(bar[t], full_bar[t], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(scal[t], full_scal[t], rtol=1e-12, atol=1e-12)


def test_rolling_atr_matches_the_live_formula():
    rng = np.random.default_rng(3)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, 300)))
    h, l = c * 1.001, c * 0.999
    a = F.rolling_atr(h, l, c)
    for t in (59, 60, 150, 299):
        want = QuantMatrix.true_range_atr(h[t - 59:t + 1], l[t - 59:t + 1], c[t - 59:t + 1])
        assert a[t] == pytest.approx(want, rel=1e-9)
    assert np.isnan(a[58])


def test_grid_fills_missing_minutes_with_the_last_close():
    idx = np.array([0, 1, 3])
    o, h, l, c, v, real = F.to_grid(idx, [10, 11, 13], [10, 11, 13], [10, 11, 13], [10, 11, 13], [1, 1, 1])
    assert real[:4].tolist() == [True, True, False, True]
    assert c[2] == 11 and v[2] == 0 and c[4] == 13


# ---- numpy policy ---------------------------------------------------------------

def test_numpy_policy_matches_torch(tmp_path):
    from rl.ppo import ActorCritic
    from rl.policy import export, NumpyPolicy
    m = ActorCritic(window=10, hidden=(16, 8))
    x = torch.randn(5, F.obs_dim(10))
    mask = torch.tensor([[True, True], [True, False], [True, True], [True, True], [True, False]])
    path = str(tmp_path / "p.npz")
    export(m, NORM, {"obs": {"window": 10}, "gate": {"approved": True}}, path)
    p = NumpyPolicy(path)
    assert p.refresh() and p.approved
    np.testing.assert_allclose(p.probs(x.numpy(), mask.numpy()),
                               m.dist(x, mask).probs.detach().numpy(), atol=1e-5)
    np.testing.assert_allclose(p.value(x.numpy()), m.value(x).detach().numpy(), atol=1e-5)
    assert p.probs(x.numpy(), mask.numpy())[1, 1] == 0.0


def test_the_policy_starts_out_mostly_flat():
    from rl.ppo import ActorCritic
    m = ActorCritic(window=10, hidden=(16,))
    p_long = m.dist(torch.zeros(64, F.obs_dim(10)), torch.ones(64, 2, dtype=torch.bool)).probs[:, 1]
    assert float(p_long.mean()) < 0.2


# ---- the learner learns ------------------------------------------------------------

@pytest.mark.parametrize("edge", [2.0, 0.0])
def test_ppo_learns_a_planted_edge_and_stays_flat_without_one(edge):
    from rl.train import train_on
    ss = synthetic.make(n_sessions=800, edge_bps=edge, seed=5)
    r = train_on(ss, iterations=60, n_envs=512, device="cpu", hidden=(32, 16), eval_every=5,
                 patience=6, verbose=False, max_passes=40)
    ppo, rnd = r["test"]["ppo"], r["test"]["random"]
    if edge > 0:
        assert ppo.mean_daily_bps > 50 and ppo.mean_daily_bps > rnd.mean_daily_bps + 50
        assert ppo.trades_per_session > 2
    else:
        # the only way to earn in a random walk under costs is not to trade
        assert ppo.trades_per_session < 0.5
        assert ppo.mean_daily_bps > -2.0


# ---- live runtime ----------------------------------------------------------------

@pytest.fixture
def live_day(monkeypatch, tmp_path):
    """Two sessions of bars for XYZ and SPY in the live bar store, and a tiny exported policy."""
    from core.minute_bars import minute_bars
    from rl.ppo import ActorCritic
    from rl.policy import export
    import rl.live as live
    days = (datetime(2026, 9, 24, tzinfo=NY), datetime(2026, 9, 25, tzinfo=NY))
    for sym, p0 in (("XYZ", 100.0), ("SPY", 500.0)):
        minute_bars.drop(sym)
        for i, d in enumerate(days):
            o, h, l, c, v = _market_day(10 + i + (7 if sym == "SPY" else 0), p0=p0)
            m0 = int(d.replace(hour=9, minute=30).timestamp() // 60)
            for k in range(F.SESSION_BARS):
                minute_bars.on_bar(sym, m0 + k, o[k], h[k], l[k], c[k], v[k])
    path = str(tmp_path / "policy.npz")
    export(ActorCritic(window=30, hidden=(16,)), NORM,
           {"obs": {"window": 30}, "gate": {"approved": False}}, path)
    rt = live.RLRuntime(path)
    monkeypatch.setattr(live, "DECISION_LOG", str(tmp_path / "d.jsonl"))
    yield rt, days[1]
    for sym in ("XYZ", "SPY"):
        minute_bars.drop(sym)


def test_live_decisions_happen_at_decision_bars_and_are_cached(live_day):
    rt, day = live_day
    at = day.replace(hour=10, minute=0, second=20).timestamp()     # bar 29 just closed
    d1 = rt.decide("XYZ", None, now=at)
    assert d1 is not None and d1.bar == 29 and d1.action in (0, 1)
    d2 = rt.decide("XYZ", None, now=at + 100)                       # 10:01:40: still bar 29's decision
    assert d2 is d1
    assert rt.decide("XYZ", None, now=day.replace(hour=10, minute=5, second=5).timestamp()).bar == 34


def test_live_observation_equals_the_training_observation(live_day):
    """The live runtime's observation at bar t equals the dataset pipeline's for the same bars."""
    rt, day = live_day
    from core.minute_bars import minute_bars
    from rl import live
    now = day.replace(hour=11, minute=0, second=10).timestamp()
    obs, mask, t, price, today = rt.observation("XYZ", None, now=now)
    rows = minute_bars.closed_rows("XYZ", day.replace(hour=16).timestamp())
    sess = live.split_sessions(rows)
    o, h, l, c, v, _ = live.session_arrays(sess[today])
    po, _, _, pc, pv, _ = live.session_arrays(sess[sorted(sess)[0]])
    spy = live.session_arrays(live.split_sessions(minute_bars.closed_rows("SPY", now))[today])[3]
    allr = np.asarray(rows, dtype=np.float64)
    atr = F.rolling_atr(allr[:, 2], allr[:, 3], allr[:, 4])[F.SESSION_BARS:]
    atr = np.maximum(atr, allr[F.SESSION_BARS:, 4] * 0.0005)
    bar, scal = F.session_features(o, h, l, c, v, F.prev_day_summary(po, pc, pv), spy, atr)
    win, n = F.window(bar, t, 30)
    want = F.assemble(win, n, scal[t], F.position_features(False, 0, 0, 0, 0, 0), NORM)
    np.testing.assert_allclose(obs, want, rtol=1e-6, atol=1e-6)


def test_modes_gate_trading(live_day, monkeypatch):
    rt, _ = live_day
    assert rt.ready() and not rt.policy.approved
    rt.set_mode("auto")
    assert not rt.may_trade()                  # unapproved: shadow
    rt.set_mode("live")
    assert rt.may_trade()
    rt.set_mode("shadow")
    assert not rt.may_trade()
    with pytest.raises(ValueError):
        rt.set_mode("yolo")


def test_the_live_path_does_not_need_torch():
    """The Docker image has no torch: the runtime and the strategy must import without it."""
    import subprocess, sys
    code = ("import sys; sys.modules['torch'] = None\n"
            "import rl.live, engine.strategies.rl_ppo, engine.strategies.registry\n"
            "print('ok')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=__file__.rsplit("/tests/", 1)[0])
    assert out.stdout.strip() == "ok", out.stderr


def test_old_candidates_are_pruned(tmp_path):
    from rl.train import prune_candidates
    for i in range(14):
        for ext in (".npz", ".json", ".md"):
            (tmp_path / f"candidate-2026092{i:02d}{ext}").write_text("x")
    (tmp_path / "policy.npz").write_text("deployed")
    prune_candidates(str(tmp_path), keep=10)
    left = sorted(p.name for p in tmp_path.glob("candidate-*.npz"))
    assert len(left) == 10 and left[0] == "candidate-202609204.npz"
    assert (tmp_path / "policy.npz").exists()
    assert not (tmp_path / "candidate-202609200.json").exists()
