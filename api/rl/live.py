"""
The deployed PPO policy inside the live engine.

Observations are built from the live one-minute bar store with the very same
feature code the policy was trained on (rl/features.py): today's regular-session
bars on the 390-minute grid, yesterday's summary, SPY for market context, and
the true-range ATR. The policy decides only at the bars it was trained to
decide at (the close of 09:34, 09:39, ...); between them the last decision
stands, exactly as in the environment.

Mode (settings.RL_MODE, switchable from the API):
  auto    orders only when the deployed policy passed its promotion gate
  shadow  decides and logs, never trades
  live    trades whatever is deployed
  off     not consulted

Every decision is appended to datasets/rl/live_decisions.jsonl, so live
behaviour can be compared with what the backtest said it would do.
"""
import json
import logging
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np

from rl import features as F
from rl.features import DECISIONS, LAST_ENTRY_DECISION
from rl.policy import NumpyPolicy, POLICY_PATH

logger = logging.getLogger("tradeflow.rl")

NY = ZoneInfo("America/New_York")
MARKET = "SPY"
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "rl")
DECISION_LOG = os.path.join(LOG_DIR, "live_decisions.jsonl")
MODES = ("auto", "shadow", "live", "off")
_DECISION_SET = set(DECISIONS)


@dataclass
class Decision:
    symbol: str
    day: str
    bar: int                  # decision bar index in the session (0 = 09:30)
    action: int               # 0 flat, 1 long
    p_long: float
    value: float
    in_position: bool
    price: float
    mode: str
    may_trade: bool
    reason: str
    at: float

    def to_dict(self):
        return asdict(self)


def _ny_parts(minute: int) -> Tuple[str, int]:
    t = datetime.fromtimestamp(int(minute) * 60, tz=NY)
    return t.strftime("%Y-%m-%d"), t.hour * 60 + t.minute


def split_sessions(rows: List[list]) -> Dict[str, List[list]]:
    """Regular-session bars per New York date."""
    out: Dict[str, List[list]] = {}
    for r in rows:
        day, mod = _ny_parts(r[0])
        if F.OPEN_MIN <= mod < F.OPEN_MIN + F.SESSION_BARS:
            out.setdefault(day, []).append(r)
    return out


def session_arrays(rows: List[list]):
    """Grid arrays for one session's rows: (o, h, l, c, v, real)."""
    a = np.asarray(rows, dtype=np.float64)
    mos = np.array([_ny_parts(m)[1] for m in a[:, 0]]) - F.OPEN_MIN
    return F.to_grid(mos, a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5])


class RLRuntime:
    def __init__(self, path: str = POLICY_PATH):
        self.policy = NumpyPolicy(path)
        self._checked_at = 0.0
        self._cache: Dict[Tuple[str, str, int, bool], Decision] = {}
        self._last: Dict[str, Decision] = {}
        self.recent: deque = deque(maxlen=200)
        self._lock = threading.Lock()
        self.mode_override: Optional[str] = None

    # ------------------------------------------------------------------
    @property
    def mode(self) -> str:
        from core.config import settings
        m = (self.mode_override or settings.RL_MODE or "auto").lower()
        return m if m in MODES else "auto"

    def set_mode(self, mode: str):
        mode = mode.lower().strip()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode_override = mode

    def ready(self) -> bool:
        now = time.time()
        if now - self._checked_at > 10.0 or not self.policy.loaded:
            self._checked_at = now
            self.policy.refresh()
        return self.policy.loaded

    def may_trade(self) -> bool:
        m = self.mode
        return self.ready() and (m == "live" or (m == "auto" and self.policy.approved))

    # ------------------------------------------------------------------
    def observation(self, symbol: str, position: Optional[Dict[str, Any]], now: Optional[float] = None
                    ) -> Optional[Tuple[np.ndarray, np.ndarray, int, float, str]]:
        """(obs, mask, decision bar, price, ny date) at the latest closed bar, or None."""
        from core.minute_bars import minute_bars
        if not self.ready():
            return None
        now = time.time() if now is None else now
        today, mod_now = _ny_parts(int(now // 60))
        t = mod_now - F.OPEN_MIN - 1                      # the bar that closed last
        if t < 0 or t >= F.SESSION_BARS:
            return None
        sessions = split_sessions(minute_bars.closed_rows(symbol, now))
        if today not in sessions:
            return None
        prev_days = sorted(d for d in sessions if d < today)
        o, h, l, c, v, real = session_arrays(sessions[today])
        if not real[:t + 1].any():
            return None
        if prev_days:
            po, _, _, pc, pv, _ = session_arrays(sessions[prev_days[-1]])
            prev = F.prev_day_summary(po, pc, pv)
        else:
            prev = F.PrevDay()
        spy_rows = split_sessions(minute_bars.closed_rows(MARKET, now)).get(today)
        spy_c = session_arrays(spy_rows)[3] if spy_rows else None
        # ATR as the dataset computes it: true range over the last 60 real bars
        # (yesterday included), mapped onto today's grid.
        real_rows = [r for d in (prev_days[-1:] + [today]) for r in sessions[d]]
        ra = np.asarray(real_rows, dtype=np.float64)
        atr_real = F.rolling_atr(ra[:, 2], ra[:, 3], ra[:, 4])
        atr_real = np.maximum(atr_real, ra[:, 4] * 0.0005)
        ga = np.full(F.SESSION_BARS, np.nan)
        n_prev = len(real_rows) - len(sessions[today])
        mos = np.array([_ny_parts(m)[1] for m in ra[n_prev:, 0]]) - F.OPEN_MIN
        ga[np.clip(mos, 0, F.SESSION_BARS - 1)] = atr_real[n_prev:]
        pos_i = np.maximum.accumulate(np.where(np.isfinite(ga), np.arange(F.SESSION_BARS), -1))
        ga = np.where(pos_i >= 0, ga[np.maximum(pos_i, 0)], np.nan)
        ga = np.where(np.isfinite(ga), ga, c * 0.001)

        bar, scal = F.session_features(o, h, l, c, v, prev, spy_c, ga, upto=t)
        window = int((self.policy.meta.get("obs") or {}).get("window", F.WINDOW))
        win, n_real = F.window(bar, t, window)
        price = float(c[t])
        in_pos = bool(position) and float(position.get("qty") or 0) > 0
        if in_pos:
            entry = float(position.get("avg_entry_price") or price)
            opened = float(position.get("opened_at") or now)
            held = max(0, int((now - opened) // 60))
            posf = F.position_features(True, price, entry, float(position.get("stop_loss") or 0),
                                       float(position.get("take_profit") or 0), held)
        else:
            posf = F.position_features(False, price, 0, 0, 0, 0)
        obs = F.assemble(win, n_real, scal[t], posf, self.policy.norm)
        mask = np.array([True, in_pos or t <= LAST_ENTRY_DECISION])
        return obs, mask, t, price, today

    def decide(self, symbol: str, position: Optional[Dict[str, Any]] = None,
               now: Optional[float] = None) -> Optional[Decision]:
        """The policy's action for this symbol now (the standing one between decision bars)."""
        if self.mode == "off" or not self.ready():
            return None
        in_pos = bool(position) and float(position.get("qty") or 0) > 0
        now = time.time() if now is None else now
        day, mod_now = _ny_parts(int(now // 60))
        t_closed = mod_now - F.OPEN_MIN - 1
        # the latest decision bar at or before the last closed bar
        t_dec = max((d for d in DECISIONS if d <= t_closed), default=None)
        if t_dec is None:
            return None
        key = (symbol, day, t_dec, in_pos)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        built = self.observation(symbol, position, now=(t_dec + F.OPEN_MIN + 1) * 60 + _day_start(day))
        if built is None:
            return None
        obs, mask, t, price, _ = built
        p = self.policy.probs(obs, mask)[0]
        val = float(self.policy.value(obs)[0])
        action = int(p.argmax())
        may = self.may_trade()
        why = ("policy says long" if action == 1 else "policy says flat")
        dec = Decision(symbol=symbol, day=day, bar=t, action=action, p_long=round(float(p[1]), 4),
                       value=round(val, 4), in_position=in_pos, price=price, mode=self.mode,
                       may_trade=may, reason=why, at=now)
        with self._lock:
            self._cache[key] = dec
            if len(self._cache) > 5000:
                for k in list(self._cache)[:2500]:
                    self._cache.pop(k, None)
            self._last[symbol] = dec
            self.recent.append(dec.to_dict())
        self._log(dec)
        return dec

    @staticmethod
    def _log(dec: Decision):
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            with open(DECISION_LOG, "a") as f:
                f.write(json.dumps(dec.to_dict()) + "\n")
        except OSError as e:
            logger.debug("Could not log RL decision: %s", e)

    # ------------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        self.ready()
        meta = self.policy.meta or {}
        curves = meta.get("curves") or []
        step = max(1, len(curves) // 60)
        return {
            "mode": self.mode,
            "may_trade": self.may_trade(),
            "loaded": self.policy.loaded,
            "error": self.policy.error,
            "path": self.policy.path,
            "trained_at": meta.get("trained_at"),
            "approved": self.policy.approved,
            "gate": meta.get("gate"),
            "deploy_reason": meta.get("deploy_reason"),
            "splits": meta.get("splits"),
            "n_symbols": meta.get("n_symbols"),
            "n_sessions": meta.get("n_sessions"),
            "rules": meta.get("rules"),
            "val": meta.get("val"),
            "test": meta.get("test"),
            "curves": curves[::step],
            "best_iteration": meta.get("best_iteration"),
            "recent_decisions": list(self.recent)[-40:],
        }


def _day_start(day: str) -> float:
    """Unix time of New York midnight for a YYYY-MM-DD date."""
    y, m, d = (int(x) for x in day.split("-"))
    return datetime(y, m, d, tzinfo=NY).timestamp()


runtime = RLRuntime()
