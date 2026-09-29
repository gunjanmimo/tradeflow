"""
Proximal Policy Optimisation (Schulman et al., 2017) for the trading environment.

  network    separate actor and critic MLPs (tanh). Small on purpose: the
             signal-to-noise ratio of minute returns is tiny, and a large network
             memorises the training days instead of learning anything general.
  rollout    N sessions x 75 decisions, complete episodes (no bootstrapping
             across the close: every episode ends flat at 15:50)
  advantage  GAE(gamma, lambda), normalised per minibatch
  update     clipped surrogate + clipped value loss + entropy bonus, several
             epochs of minibatches, gradient-norm clipping, linear LR decay
  masking    actions the rules forbid (opening after the entry cutoff) get zero
             probability, in sampling and in the loss
  flat prior the policy starts out mostly flat (FLAT_PRIOR on the flat logit):
             every trade costs the spread, so trading has to be earned by
             evidence that holds across many sessions. Started at 50/50, PPO
             traded from the first iteration and chased noise -- on a pure
             random walk it "learned" +25 bps/session on its training days.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn

from rl import features as F
from rl.env import TradingEnv, N_STEPS


@dataclass
class PPOConfig:
    hidden: tuple = (128, 64)
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    value_clip: float = 0.2
    entropy: float = 0.01
    entropy_final: float = 0.001
    value_coef: float = 0.5
    epochs: int = 4
    minibatch: int = 8192
    max_grad_norm: float = 0.5
    n_envs: int = 2048
    weight_decay: float = 1e-4
    # Gaussian noise (in normalised feature units) added to observations while
    # collecting training rollouts: the policy cannot key on exact values that
    # fingerprint a particular training day.
    obs_noise: float = 0.0
    window: int = F.WINDOW

    def to_dict(self):
        d = asdict(self)
        d["hidden"] = list(self.hidden)
        return d


def _mlp(inp: int, hidden, out: int, out_gain: float) -> nn.Sequential:
    layers, last = [], inp
    for h in hidden:
        lin = nn.Linear(last, h)
        nn.init.orthogonal_(lin.weight, gain=np.sqrt(2))
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.Tanh()]
        last = h
    head = nn.Linear(last, out)
    nn.init.orthogonal_(head.weight, gain=out_gain)
    nn.init.zeros_(head.bias)
    layers.append(head)
    return nn.Sequential(*layers)


FLAT_PRIOR = 2.0        # initial logit(flat) - logit(long): P(long) ~ 12% at start


class ActorCritic(nn.Module):
    def __init__(self, window: int = F.WINDOW, hidden=(128, 64), flat_prior: float = FLAT_PRIOR):
        super().__init__()
        obs_dim = F.obs_dim(window)
        self.window, self.obs_dim, self.hidden = int(window), obs_dim, tuple(hidden)
        self.actor = _mlp(obs_dim, hidden, 2, 0.01)      # near-constant policy at start...
        with torch.no_grad():
            self.actor[-1].bias[0] = flat_prior           # ...leaning flat
        self.critic = _mlp(obs_dim, hidden, 1, 1.0)

    @staticmethod
    def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return logits.masked_fill(~mask, -1e9)

    def dist(self, obs, mask):
        return torch.distributions.Categorical(logits=self.masked_logits(self.actor(obs), mask))

    def value(self, obs):
        return self.critic(obs).squeeze(-1)


@dataclass
class Rollout:
    obs: torch.Tensor
    mask: torch.Tensor
    act: torch.Tensor
    logp: torch.Tensor
    val: torch.Tensor
    adv: torch.Tensor
    ret: torch.Tensor
    ep_return: torch.Tensor           # per env, reward units
    stats: Dict[str, torch.Tensor] = field(default_factory=dict)


@torch.no_grad()
def collect(model: ActorCritic, env: TradingEnv, cfg: PPOConfig) -> Rollout:
    obs, mask = env.reset()
    T, n, dev = N_STEPS, env.n, env.dev
    O = torch.empty(T, n, obs.shape[1], device=dev)
    M = torch.empty(T, n, 2, dtype=torch.bool, device=dev)
    A = torch.empty(T, n, dtype=torch.long, device=dev)
    LP = torch.empty(T, n, device=dev)
    V = torch.empty(T, n, device=dev)
    R = torch.empty(T, n, device=dev)
    for k in range(T):
        if cfg.obs_noise > 0:
            obs = obs + cfg.obs_noise * torch.randn_like(obs)
        d = model.dist(obs, mask)
        a = d.sample()
        O[k], M[k], A[k], LP[k], V[k] = obs, mask, a, d.log_prob(a), model.value(obs)
        obs, mask, r, done = env.step(a)
        R[k] = r
    # GAE over complete episodes: the value after the last decision is 0 (flat, day over)
    adv = torch.zeros(T, n, device=dev)
    last = torch.zeros(n, device=dev)
    for k in reversed(range(T)):
        nv = V[k + 1] if k + 1 < T else torch.zeros(n, device=dev)
        delta = R[k] + cfg.gamma * nv - V[k]
        last = delta + cfg.gamma * cfg.lam * last
        adv[k] = last
    ret = adv + V
    flat = lambda x: x.reshape(T * n, *x.shape[2:])
    return Rollout(flat(O), flat(M), flat(A), flat(LP), flat(V), flat(adv), flat(ret),
                   ep_return=R.sum(0), stats={k: v.clone() for k, v in env.stats.items()})


def update(model: ActorCritic, opt: torch.optim.Optimizer, ro: Rollout, cfg: PPOConfig,
           entropy_coef: float) -> Dict[str, float]:
    N = ro.obs.shape[0]
    logs = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0}
    n_mb = 0
    for _ in range(cfg.epochs):
        perm = torch.randperm(N, device=ro.obs.device)
        for i in range(0, N, cfg.minibatch):
            j = perm[i:i + cfg.minibatch]
            d = model.dist(ro.obs[j], ro.mask[j])
            logp = d.log_prob(ro.act[j])
            ratio = torch.exp(logp - ro.logp[j])
            adv = ro.adv[j]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            pg = -torch.min(ratio * adv, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv).mean()
            v = model.value(ro.obs[j])
            v_clip = ro.val[j] + torch.clamp(v - ro.val[j], -cfg.value_clip, cfg.value_clip)
            vl = 0.5 * torch.max((v - ro.ret[j]) ** 2, (v_clip - ro.ret[j]) ** 2).mean()
            # entropy only where there is a real choice (a masked state has none)
            ent = d.entropy()
            free = ro.mask[j].all(dim=1).float()
            ent = (ent * free).sum() / free.sum().clamp(min=1)
            loss = pg + cfg.value_coef * vl - entropy_coef * ent
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
            with torch.no_grad():
                logs["policy_loss"] += pg.item()
                logs["value_loss"] += vl.item()
                logs["entropy"] += ent.item()
                logs["approx_kl"] += ((ratio - 1) - torch.log(ratio)).mean().item()
                logs["clip_frac"] += ((ratio - 1).abs() > cfg.clip).float().mean().item()
            n_mb += 1
    return {k: v / max(n_mb, 1) for k, v in logs.items()}


def explained_variance(pred: torch.Tensor, target: torch.Tensor) -> float:
    var = target.var()
    return float(1 - (target - pred).var() / var) if var > 0 else float("nan")
