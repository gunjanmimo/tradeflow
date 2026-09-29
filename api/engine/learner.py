"""
Learner agent: keeps the RL policy learning from new market data.

Once per trading day, RL_RETRAIN_AFTER_CLOSE_MINUTES after the 16:00 close, it

  1. appends the day's one-minute bars to the research cache
     (python -m research download: only the missing days are fetched)
  2. retrains PPO warm-started from the deployed policy on the rolling window
     (python -m rl.train --warm-start)
  3. the trainer deploys the new policy only if it beats the deployed one on
     the same unseen test days (champion / challenger), and marks it approved
     only if it passes the promotion gate

Both run as a subprocess, so training never shares the trading event loop.
The live engine reloads rl/policy.npz when the file changes. Training needs
torch; where it is missing (the slim Docker image) the agent says so and the
same commands can be run on the host.
"""
import asyncio
import importlib.util
import logging
import os
import sys
import time
from collections import deque
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from core.config import settings
from core.state import state
from engine.agent import Agent

logger = logging.getLogger("tradeflow.learner")

NY = ZoneInfo("America/New_York")
API_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(API_DIR, "datasets", "rl", "learner.log")


class LearnerAgent(Agent):
    name = "Learner"
    role = "Retrains the RL policy on each day's new bars; deploys it only if it beats the current one"

    def __init__(self):
        super().__init__()
        self.done_for: Optional[str] = None
        self.proc: Optional[asyncio.subprocess.Process] = None
        self.started_at: Optional[float] = None
        self.last_result: Optional[str] = None
        self.tail: deque = deque(maxlen=40)

    @property
    def interval(self) -> float:
        return 60.0

    @property
    def enabled(self) -> bool:
        return settings.RL_RETRAIN_ENABLED

    @staticmethod
    def has_torch() -> bool:
        return importlib.util.find_spec("torch") is not None

    def due(self, now: Optional[datetime] = None) -> Optional[str]:
        """Today's NY date if a retrain is due now, else None."""
        t = (now or datetime.now(NY)).astimezone(NY)
        if t.weekday() >= 5:
            return None
        day = t.strftime("%Y-%m-%d")
        after = 16 * 60 + settings.RL_RETRAIN_AFTER_CLOSE_MINUTES
        if t.hour * 60 + t.minute < after or self.done_for == day:
            return None
        return day

    async def step(self):
        if self.proc is not None:
            self.summary = f"retraining since {time.strftime('%H:%M', time.localtime(self.started_at))}"
            return
        day = self.due()
        if day is None:
            self.summary = (f"last run {self.done_for}: {self.last_result}" if self.done_for
                            else "waiting for the close")
            return
        await self.retrain(day)

    async def retrain(self, day: Optional[str] = None) -> str:
        """Starts a retrain now (also used by POST /api/rl/retrain). Returns what happened."""
        day = day or datetime.now(NY).strftime("%Y-%m-%d")
        if self.proc is not None:
            return "a retrain is already running"
        if not self.has_torch():
            self.done_for = day
            self.last_result = ("torch is not installed here: run `python -m research download` and "
                                "`python -m rl.train --warm-start` on a machine that has it")
            self.summary = self.last_result
            return self.last_result
        cmd = (f"{sys.executable} -m research download --days 730 && "
               f"{sys.executable} -m rl.train --warm-start --iterations {settings.RL_RETRAIN_ITERATIONS}")
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        self.started_at = time.time()
        self.tail.clear()
        self.proc = await asyncio.create_subprocess_shell(
            cmd, cwd=API_DIR, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        state.log_event("LEARNER", f"Retraining the RL policy on data up to {day}")
        self.act("RL policy", "RETRAIN", f"started for {day}")
        asyncio.create_task(self._watch(day))
        return "started"

    async def _watch(self, day: str):
        proc = self.proc
        with open(LOG_PATH, "a") as log:
            log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} retrain for {day} ===\n")
            async for raw in proc.stdout:
                line = raw.decode(errors="replace").rstrip()
                self.tail.append(line)
                log.write(line + "\n")
        code = await proc.wait()
        self.proc = None
        self.done_for = day
        text = "\n".join(self.tail)
        if code != 0:
            self.last_result = f"failed (exit {code}); see {LOG_PATH}"
        elif "deployed to" in text:
            self.last_result = "new policy deployed (beat the previous one on unseen days)"
        else:
            self.last_result = "kept the deployed policy (the new one did not beat it)"
        approved = "APPROVED to trade" in text
        state.log_event("LEARNER", f"Retrain for {day}: {self.last_result}"
                                   + ("; policy approved to trade" if approved else "; not approved (shadow)"))
        self.summary = f"last run {day}: {self.last_result}"

    def card(self):
        c = super().card()
        c["retraining"] = self.proc is not None
        c["last_result"] = self.last_result
        c["log_tail"] = list(self.tail)[-8:]
        return c


learner = LearnerAgent()
