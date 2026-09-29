"""
The base every fleet agent shares: its own loop and cadence, a status card, and
a record of its last action. Kept apart from engine/fleet.py so agents that live
elsewhere (scout/, engine/learner.py) can subclass it without importing the fleet.
"""
import asyncio
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("tradeflow.fleet")


class Agent:
    name = "agent"
    role = ""

    def __init__(self):
        self.running = False
        self._task: Optional[asyncio.Task] = None
        self.cycles = 0
        self.last_at = 0.0
        self.last_ms = 0.0
        self.error: Optional[str] = None
        self.summary = "starting"
        self.last_action: Optional[Dict[str, Any]] = None

    @property
    def interval(self) -> float:
        return 5.0

    @property
    def enabled(self) -> bool:
        return True

    async def start(self):
        if not self.enabled:
            self.summary = "disabled"
            return
        self.running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()

    async def _loop(self):
        while self.running:
            t0 = time.perf_counter()
            try:
                await self.step()
                self.error = None
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
                logger.error(f"{self.name} failed: {e}", exc_info=True)
            self.cycles += 1
            self.last_at = time.time()
            self.last_ms = (time.perf_counter() - t0) * 1000
            try:
                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                break

    async def step(self):
        raise NotImplementedError

    def act(self, symbol: str, action: str, reason: str):
        self.last_action = {"symbol": symbol, "action": action, "reason": reason, "at": time.time()}

    def card(self) -> Dict[str, Any]:
        stalled = self.running and self.last_at and time.time() - self.last_at > max(15.0, 5 * self.interval)
        return {
            "name": self.name, "role": self.role,
            "state": "stalled" if stalled else ("running" if self.running else self.summary),
            "summary": self.summary, "cycles": self.cycles, "interval_s": self.interval,
            "last_at": self.last_at or None, "cycle_ms": round(self.last_ms, 1),
            "error": self.error, "last_action": self.last_action,
        }
