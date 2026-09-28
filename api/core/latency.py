"""
Latency tracking for the trading hot path and everything around it.

Recording is a perf_counter_ns() pair plus a deque append (~0.3 us), so tracking
never becomes a latency source itself. Percentiles are computed only when a
snapshot is requested (the telemetry broadcaster asks at most once a second),
never on the tick path.

Stages are grouped so the UI can separate what the operator can act on:

  hot    work done inside the tick path, between a price arriving and a decision
  broker network round-trips to the broker (order submit / close)
  feed   how old market data is when it reaches us (polling / bar delay)
  loop   asyncio event-loop lag: if background work ever blocked the loop, it
         shows up here first -- this is the "did we add latency" alarm
  worker off-process analytics (regime, council, risk): never on the tick path,
         shown so their cost is visible without being counted against it
"""
import asyncio
import time
from contextlib import contextmanager
from typing import Dict, Optional

import numpy as np

_WINDOW = 2048

STAGE_GROUPS: Dict[str, str] = {
    "tick_total": "hot",
    "quant_matrix": "hot",
    "decision": "hot",
    "sentinel_tick": "hot",
    "order_submit": "broker",
    "order_close": "broker",
    "feed_stock_bar_age": "feed",
    "event_loop_lag": "loop",
    "worker_cycle": "worker",
    "worker_snapshot": "worker",
    "telemetry_build": "loop",
}

# Budgets (microseconds) the UI colours against. Hot-path numbers are what the
# engine controls; broker/feed are mostly network and exchange.
BUDGET_US: Dict[str, float] = {
    "tick_total": 500.0,
    "quant_matrix": 200.0,
    "decision": 200.0,
    "sentinel_tick": 300.0,
    "order_submit": 250_000.0,
    "order_close": 250_000.0,
    "feed_stock_bar_age": 90_000_000.0,
    "event_loop_lag": 2_000.0,
    "worker_cycle": 2_000_000.0,
    "worker_snapshot": 50_000.0,
    "telemetry_build": 5_000.0,
}


class _Ring:
    """Fixed-size numpy ring buffer. Percentiles do not depend on order, so the
    raw buffer is summarised directly, with no deque -> array conversion."""
    __slots__ = ("buf", "n", "i", "count", "last")

    def __init__(self):
        self.buf = np.empty(_WINDOW, dtype=np.float64)
        self.n = 0
        self.i = 0
        self.count = 0
        self.last = 0.0


class LatencyTracker:
    def __init__(self):
        self._rings: Dict[str, _Ring] = {}
        self._started = time.time()
        self._snapshot: Optional[Dict] = None
        self._snapshot_at = 0.0

    def record_us(self, stage: str, us: float):
        r = self._rings.get(stage)
        if r is None:
            r = self._rings[stage] = _Ring()
        r.buf[r.i] = us
        r.i = (r.i + 1) % _WINDOW
        if r.n < _WINDOW:
            r.n += 1
        r.count += 1
        r.last = us

    def record_ns(self, stage: str, t0_ns: int):
        self.record_us(stage, (time.perf_counter_ns() - t0_ns) / 1000.0)

    @contextmanager
    def measure(self, stage: str):
        t0 = time.perf_counter_ns()
        try:
            yield
        finally:
            self.record_ns(stage, t0)

    @staticmethod
    def _summarize(stage: str, r: "_Ring") -> Dict:
        n = r.n
        a = r.buf[:n]
        ks = sorted({int(0.50 * (n - 1)), int(0.95 * (n - 1)), int(0.99 * (n - 1))})
        part = np.partition(a, ks)
        p50, p95, p99 = (float(part[int(q * (n - 1))]) for q in (0.50, 0.95, 0.99))
        budget = BUDGET_US.get(stage)
        return {
            "group": STAGE_GROUPS.get(stage, "other"),
            "last_us": round(r.last, 1),
            "p50_us": round(p50, 1),
            "p95_us": round(p95, 1),
            "p99_us": round(p99, 1),
            "max_us": round(float(a.max()), 1),
            "mean_us": round(float(a.mean()), 1),
            "count": r.count,
            "window": n,
            "budget_us": budget,
            "over_budget": bool(budget is not None and p95 > budget),
        }

    def _store(self, stages: Dict, now: float) -> Dict:
        self._snapshot = {"stages": stages, "at": now,
                          "uptime_s": round(now - self._started, 1)}
        self._snapshot_at = now
        return self._snapshot

    def snapshot(self, max_age_s: float = 1.0) -> Dict:
        """
        Percentile summary per stage, cached for max_age_s.

        One O(n) np.partition per stage instead of a sort: ~1.3ms -> ~0.35ms for
        a full snapshot. The telemetry loop uses refresh() instead, which also
        yields between stages; this synchronous form serves on-demand API calls.
        """
        now = time.time()
        if self._snapshot is not None and now - self._snapshot_at < max_age_s:
            return self._snapshot
        stages = {k: self._summarize(k, r) for k, r in list(self._rings.items()) if r.n}
        return self._store(stages, now)

    async def refresh(self, max_age_s: float = 1.0) -> Dict:
        """snapshot(), but yields to the event loop between stages so a tick never
        waits behind more than one stage's summary (~30us)."""
        now = time.time()
        if self._snapshot is not None and now - self._snapshot_at < max_age_s:
            return self._snapshot
        stages = {}
        for k, r in list(self._rings.items()):
            if r.n:
                stages[k] = self._summarize(k, r)
                await asyncio.sleep(0)
        return self._store(stages, now)


latency = LatencyTracker()


async def monitor_event_loop(interval_s: float = 0.1):
    """
    Measures how late the event loop wakes a sleeping task. Any synchronous work
    that hogs the loop (a slow computation, a blocking call) delays every tick
    handler by the same amount, and appears here as lag.
    """
    # perf_counter, not loop.time(): under uvicorn the loop is uvloop, whose clock
    # has millisecond resolution -- too coarse to see sub-millisecond lag.
    while True:
        try:
            t0 = time.perf_counter()
            await asyncio.sleep(interval_s)
            lag_s = time.perf_counter() - t0 - interval_s
            latency.record_us("event_loop_lag", max(lag_s, 0.0) * 1e6)
        except asyncio.CancelledError:
            break
