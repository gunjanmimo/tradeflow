"""
Evaluate a policy on sessions it was not trained on, against baselines run
through the very same environment (same fills, stops, costs and close):

  flat         never trades: 0 by construction. Anything worse lost money.
  always_long  long all day under the same brackets (re-entering after a stop
               or target): what the market itself paid intraday.
  random       flips its position at random, at the PPO policy's own trade rate:
               what trading that often costs with no information at all.

Results are aggregated to a daily portfolio: every symbol-session is one equal
slot, and a day's return is the mean over its slots. Everything is in bps of
the capital allotted per slot.
"""
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from rl.env import TradingEnv, SessionTensors, EnvRules, N_STEPS, DECISION_EVERY, REWARD_UNIT


@dataclass
class EvalResult:
    policy: str
    sessions: int
    days: int
    mean_daily_bps: float
    t_daily: float
    sharpe: float                 # annualised, of daily portfolio returns
    total_bps: float              # sum of daily portfolio returns
    max_drawdown_bps: float
    trades_per_session: float
    exposure: float               # share of the tradable bars spent long
    win_rate: float               # of sessions that traded, share with a positive result
    stops_per_session: float
    targets_per_session: float
    policy_exits_per_session: float
    eod_exits_per_session: float

    def to_dict(self):
        return asdict(self)


PolicyFn = Callable[[torch.Tensor, torch.Tensor, TradingEnv], torch.Tensor]


@torch.no_grad()
def run(data: SessionTensors, sessions: np.ndarray, rules: EnvRules, policy: PolicyFn,
        batch: int = 8192, window: int = 30) -> Dict[str, np.ndarray]:
    """Plays every session once. Returns per-session arrays (bps and counts)."""
    out: Dict[str, List[np.ndarray]] = {}
    for i in range(0, len(sessions), batch):
        idx = torch.as_tensor(sessions[i:i + batch], dtype=torch.long)
        env = TradingEnv(data, rules, len(idx), sessions=idx, window=window)
        obs, mask = env.reset()
        for _ in range(N_STEPS):
            a = policy(obs, mask, env)
            obs, mask, _, done = env.step(a)
        for k, v in env.stats.items():
            out.setdefault(k, []).append(v.cpu().numpy())
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["ret_bps"] = res.pop("ret") * REWARD_UNIT
    return res


def summarize(name: str, per: Dict[str, np.ndarray], days: np.ndarray) -> EvalResult:
    ret = per["ret_bps"]
    uniq, inv = np.unique(days, return_inverse=True)
    daily = np.bincount(inv, weights=ret) / np.bincount(inv)
    n = len(daily)
    mu, sd = float(daily.mean()), float(daily.std(ddof=1)) if n > 1 else 0.0
    eq = np.cumsum(daily)
    dd = float((np.maximum.accumulate(np.r_[0.0, eq])[1:] - eq).max()) if n else 0.0
    traded = per["trades"] > 0
    return EvalResult(
        policy=name, sessions=len(ret), days=n,
        mean_daily_bps=round(mu, 3),
        t_daily=round(mu / (sd / np.sqrt(n)), 2) if sd > 0 else 0.0,
        sharpe=round(mu / sd * np.sqrt(252), 2) if sd > 0 else 0.0,
        total_bps=round(float(daily.sum()), 1),
        max_drawdown_bps=round(dd, 1),
        trades_per_session=round(float(per["trades"].mean()), 3),
        exposure=round(float(per["bars_in"].mean() / (N_STEPS * DECISION_EVERY)), 3),
        win_rate=round(float((ret[traded] > 0).mean() * 100), 1) if traded.any() else 0.0,
        stops_per_session=round(float(per["stops"].mean()), 3),
        targets_per_session=round(float(per["targets"].mean()), 3),
        policy_exits_per_session=round(float(per["policy_exits"].mean()), 3),
        eod_exits_per_session=round(float(per["eod_exits"].mean()), 3),
    )


def ppo_policy(model, greedy: bool = True) -> PolicyFn:
    def f(obs, mask, env):
        d = model.dist(obs, mask)
        return d.probs.argmax(-1) if greedy else d.sample()
    return f


def flat_policy(obs, mask, env):
    return torch.zeros(obs.shape[0], dtype=torch.long, device=obs.device)


def long_policy(obs, mask, env):
    return mask[:, 1].long()


def random_policy(flip_prob: float, seed: int = 0) -> PolicyFn:
    g = torch.Generator(device="cpu").manual_seed(seed)

    def f(obs, mask, env):
        flip = (torch.rand(obs.shape[0], generator=g) < flip_prob).to(obs.device)
        cur = (env.pos > 0)
        want = torch.where(flip, ~cur, cur)
        return (want & mask[:, 1]).long()
    return f


def evaluate(model, data: SessionTensors, sessions: np.ndarray, days: np.ndarray,
             rules: EnvRules, baselines: bool = True, window: int = 30) -> Dict[str, EvalResult]:
    model.eval()
    w = getattr(model, "window", window)
    per = run(data, sessions, rules, ppo_policy(model), window=w)
    out = {"ppo": summarize("ppo", per, days)}
    if baselines:
        out["flat"] = summarize("flat", run(data, sessions, rules, flat_policy, window=w), days)
        out["always_long"] = summarize("always_long", run(data, sessions, rules, long_policy, window=w), days)
        # random at the policy's own turnover: each trade is two position flips
        flips = max(2 * out["ppo"].trades_per_session / N_STEPS, 1.0 / N_STEPS)
        out["random"] = summarize("random", run(data, sessions, rules, random_policy(min(flips, 1.0)),
                                                window=w), days)
    return out
