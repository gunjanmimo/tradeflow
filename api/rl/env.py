"""
A vectorised intraday trading environment in torch: N sessions step together.

Each episode is one (symbol, day) session, flat at the open and flat by the
close. The rules are the live engine's, so what the policy learns is what can
actually be executed:

  decisions   at the close of bars 4, 9, ..., 374 (every DECISION_EVERY minutes,
              first at 09:34). An order fills at the NEXT bar's open.
  actions     0 = be flat, 1 = be long. Opening a position is masked once the
              fill would come after 15:29 (NO_NEW_ENTRY_MINUTES_BEFORE_CLOSE).
  brackets    on entry, stop and target from engine/brackets.py's rule (ATR x
              the risk dial's multiples, clamped to MIN/MAX_STOP_DISTANCE_PCT),
              checked bar by bar: stop first, a gap through it fills at the open;
              the target is a resting limit.
  close       anything still held is sold at 15:50's open (the flatten window).
  costs       core/costs.py per market fill (entry, policy exit, stop, flatten);
              the target limit pays none.
  reward      the step's net log return of a unit position, x 1e4 / REWARD_UNIT
              (1.0 = 10 bps). No shaping: the policy is paid exactly what the
              account would make.
"""
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

from rl import features as F

from rl.features import (  # noqa: F401  (re-exported for callers)
    DECISION_EVERY, FIRST_DECISION, LAST_ENTRY_DECISION, FLATTEN_BAR, DECISIONS, N_STEPS,
)
REWARD_UNIT = 10.0                      # bps per unit of reward


@dataclass
class EnvRules:
    per_side: float = 3e-4              # half-spread + slippage, fraction of price
    stop_atr: float = 1.5
    reward_risk: float = 2.0
    min_stop_pct: float = 0.008
    max_stop_pct: float = 0.06

    @classmethod
    def live(cls, costs=None) -> "EnvRules":
        """The rules the live engine runs with now (risk dial, settings, costs)."""
        from core.config import settings
        from core.costs import DEFAULT_COSTS
        from core.state import state
        p = state.risk_profile
        c = costs or DEFAULT_COSTS
        return cls(per_side=c.per_side, stop_atr=p.stop_atr_multiple, reward_risk=p.reward_risk_ratio,
                   min_stop_pct=settings.MIN_STOP_DISTANCE_PCT, max_stop_pct=settings.MAX_STOP_DISTANCE_PCT)

    def to_dict(self) -> Dict[str, float]:
        return dict(self.__dict__)


class SessionTensors:
    """A SessionSet moved to a torch device once."""

    def __init__(self, ss, norm: Dict[str, np.ndarray], device: str):
        t = lambda x, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(x), dtype=dt, device=device)
        self.n = len(ss)
        self.bar = (t(ss.bar) - t(norm["bar_mean"])) / t(norm["bar_std"])
        self.scal = torch.clamp((t(ss.scal) - t(norm["scal_mean"])) / t(norm["scal_std"]), -8, 8)
        self.bar = torch.clamp(self.bar, -8, 8)
        self.o, self.h, self.l, self.c = t(ss.o), t(ss.h), t(ss.l), t(ss.c)
        self.atr = t(ss.atr)
        self.device = device


class TradingEnv:
    def __init__(self, data: SessionTensors, rules: EnvRules, n_envs: int,
                 sessions: Optional[torch.Tensor] = None, pool: Optional[torch.Tensor] = None,
                 generator: Optional[torch.Generator] = None, window: int = F.WINDOW):
        """
        sessions: play exactly these (one env each), e.g. for evaluation.
        pool:     otherwise each reset draws n_envs sessions at random from here
                  (the training days); default all sessions.
        """
        self.d, self.rules, self.n = data, rules, n_envs
        self.dev = data.device
        self.gen = generator
        self.fixed = sessions
        self.pool = pool.to(data.device) if pool is not None else None
        self.window = int(window)
        self.offsets = torch.arange(-self.window + 1, 1, device=self.dev)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        n, dev = self.n, self.dev
        if self.fixed is not None:
            self.s = self.fixed.to(dev)
        elif self.pool is not None:
            pick = torch.randint(0, len(self.pool), (n,), device="cpu", generator=self.gen).to(dev)
            self.s = self.pool[pick]
        else:
            self.s = torch.randint(0, self.d.n, (n,), device="cpu", generator=self.gen).to(dev)
        self.k = 0                                  # decision index 0..N_STEPS-1
        self.pos = torch.zeros(n, device=dev)
        self.entry = torch.zeros(n, device=dev)
        self.stop = torch.zeros(n, device=dev)
        self.tp = torch.zeros(n, device=dev)
        self.held = torch.zeros(n, device=dev)
        self.mark = torch.zeros(n, device=dev)      # price the position is marked at
        self.stats = {k: torch.zeros(n, device=dev) for k in
                      ("ret", "trades", "stops", "targets", "policy_exits", "eod_exits", "bars_in")}
        return self.observe()

    @property
    def t(self) -> int:
        return DECISIONS[self.k]

    def entry_allowed(self) -> bool:
        return self.t <= LAST_ENTRY_DECISION

    def observe(self):
        """(obs [n, obs_dim(window)], mask [n, 2]) at the current decision bar."""
        t = self.t
        idx = t + self.offsets                                   # (W,)
        valid = (idx >= 0).float()
        idx_c = idx.clamp(min=0)
        win = self.d.bar[self.s][:, idx_c, :] * valid[None, :, None]         # (n, W, F)
        scal = self.d.scal[self.s, t]                                        # (n, S)
        price = self.d.c[self.s, t]
        inp = self.pos > 0
        safe_entry = torch.where(inp, self.entry, price)
        safe_stop = torch.where(inp, self.stop, price * 0.99)
        safe_tp = torch.where(inp, self.tp, price * 1.01)
        pos = torch.stack([
            self.pos,
            torch.where(inp, torch.log(price / safe_entry) * 100, torch.zeros_like(price)),
            torch.where(inp, self.held / F.SESSION_BARS, torch.zeros_like(price)),
            torch.where(inp, torch.log(price / safe_stop) * 100, torch.zeros_like(price)),
            torch.where(inp, torch.log(safe_tp / price) * 100, torch.zeros_like(price)),
        ], dim=1)
        obs = torch.cat([win.reshape(self.n, -1), scal, pos], dim=1)
        mask = torch.ones(self.n, 2, dtype=torch.bool, device=self.dev)
        if not self.entry_allowed():
            mask[:, 1] = inp                                     # may stay long, may not open
        return obs, mask

    # ------------------------------------------------------------------
    def step(self, action: torch.Tensor):
        """Applies actions at the current decision; returns (obs, mask, reward, done)."""
        r = self.rules
        ps = r.per_side
        t = self.t
        s = self.s
        o = self.d.o[s]                       # (n, 390)
        a = action.float()
        reward = torch.zeros(self.n, device=self.dev)
        nxt = o[:, t + 1]

        # 1. policy exit at the next open
        ex = (self.pos > 0) & (a < 0.5)
        reward += torch.where(ex, torch.log(nxt * (1 - ps) / self.mark), torch.zeros_like(reward))
        self.stats["policy_exits"] += ex.float()
        self.pos = torch.where(ex, torch.zeros_like(self.pos), self.pos)

        # 2. entry at the next open, with the bracket derived from the fill
        en = (self.pos == 0) & (a > 0.5) & self.entry_allowed()
        fill = nxt * (1 + ps)
        atr = self.d.atr[s, t]
        dist = torch.clamp(atr * r.stop_atr, min=fill * r.min_stop_pct, max=fill * r.max_stop_pct)
        self.entry = torch.where(en, fill, self.entry)
        self.stop = torch.where(en, fill - dist, self.stop)
        self.tp = torch.where(en, fill + dist * r.reward_risk, self.tp)
        self.held = torch.where(en, torch.zeros_like(self.held), self.held)
        self.mark = torch.where(en, nxt, self.mark)
        reward += torch.where(en, torch.full_like(reward, float(np.log(1 - ps / (1 + ps)))),
                              torch.zeros_like(reward))        # log(nxt / fill)
        self.pos = torch.where(en, torch.ones_like(self.pos), self.pos)
        self.stats["trades"] += en.float()

        # 3. bars t+1 .. t+k: stop first, then target
        h, l = self.d.h[s], self.d.l[s]
        for j in range(t + 1, t + DECISION_EVERY + 1):
            inp = self.pos > 0
            oj, hj, lj = o[:, j], h[:, j], l[:, j]
            # a position that entered at this very open cannot have gapped past it
            fresh = en if j == t + 1 else torch.zeros_like(en)
            gap_stop = inp & (oj <= self.stop) & ~fresh
            hit_stop = inp & ~gap_stop & (lj <= self.stop)
            stop_px = torch.where(gap_stop, oj, self.stop) * (1 - ps)
            stopped = gap_stop | hit_stop
            reward += torch.where(stopped, torch.log(stop_px / self.mark), torch.zeros_like(reward))
            self.stats["stops"] += stopped.float()
            inp = inp & ~stopped
            gap_tp = inp & (oj >= self.tp) & ~fresh
            hit_tp = inp & ~gap_tp & (hj >= self.tp)
            tp_px = torch.where(gap_tp, oj, self.tp)
            took = gap_tp | hit_tp
            reward += torch.where(took, torch.log(tp_px / self.mark), torch.zeros_like(reward))
            self.stats["targets"] += took.float()
            self.pos = torch.where(stopped | took, torch.zeros_like(self.pos), self.pos)
            self.held += (self.pos > 0).float()
            self.stats["bars_in"] += (self.pos > 0).float()

        # 4. mark what is still held to the next decision's fill price
        m_next = o[:, t + DECISION_EVERY + 1]
        still = self.pos > 0
        reward += torch.where(still, torch.log(m_next / self.mark), torch.zeros_like(reward))
        self.mark = torch.where(still, m_next, self.mark)

        self.k += 1
        done = self.k >= N_STEPS
        if done:
            # 5. the flatten fill: m_next is the 15:50 open
            still = self.pos > 0
            reward += torch.where(still, torch.full_like(reward, float(np.log(1 - ps))), torch.zeros_like(reward))
            self.stats["eod_exits"] += still.float()
            self.pos = torch.zeros_like(self.pos)
        reward = reward * (1e4 / REWARD_UNIT)
        self.stats["ret"] += reward
        if done:
            return None, None, reward, True
        obs, mask = self.observe()
        return obs, mask, reward, False
