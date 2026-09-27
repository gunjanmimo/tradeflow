"""
Runs the analysis worker (engine/analysis/worker.py) in a separate process.

Why a process and not a thread or a task
----------------------------------------
The analytics are CPU-bound Python. On the event loop they would delay every
tick handler while they run. In a thread they would still contend for the GIL
with the tick path. A separate process has its own interpreter and GIL, so the
tick path pays only for copying a snapshot (numpy arrays, microseconds) and
storing the result.

The process is started with "spawn", not fork: the main process holds the Laya
model and several threads, and forking a multi-threaded process is unsafe.

If the pool cannot start, analysis is disabled with a warning. Consumers treat
missing analysis as "no opinion" -- they never compute it inline as a fallback,
because that would move the cost back onto the hot path.
"""
import asyncio
import logging
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, Optional

import numpy as np

from core.config import settings
from core.latency import latency
from core.state import state, is_crypto_symbol

logger = logging.getLogger("tradeflow.analysis")

# Symbols copied per event-loop slice while building a snapshot.
SNAPSHOT_CHUNK = 4


def _warmup() -> bool:
    """Imports the strategy stack in the child so the first real cycle is not slow."""
    import engine.analysis.worker  # noqa: F401
    import engine.strategies.registry  # noqa: F401
    return True


class AnalysisService:
    def __init__(self):
        self._pool: Optional[ProcessPoolExecutor] = None
        self._task: Optional[asyncio.Task] = None
        self.cycles = 0
        self.errors = 0
        self.last_error: Optional[str] = None
        self.last_compute_ms: Optional[float] = None
        self.enabled = False
        self.partial_errors: Dict[str, str] = {}
        self._logged_errors: Dict[str, str] = {}

    async def start(self):
        if not settings.ANALYSIS_ENABLED:
            logger.info("Analysis worker disabled by config")
            return
        # Import everything build_snapshot() and run_once() touch NOW, at startup.
        # Measured: a lazy first import inside the loop stalled it for ~70ms.
        import engine.analysis.worker  # noqa: F401
        import engine.brackets  # noqa: F401
        import engine.strategies.performance  # noqa: F401
        import feeds.multi_source_aggregator  # noqa: F401
        try:
            self._pool = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._pool, _warmup)
        except Exception as e:
            logger.warning(f"Analysis worker unavailable, running without it: {e}")
            state.log_event("WARNING", f"Analysis worker failed to start ({e}); council, adaptive "
                                       f"regime routing and Monte Carlo are inactive.")
            self._pool = None
            return
        self.enabled = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Analysis worker process started")

    async def stop(self):
        if self._task:
            self._task.cancel()
        if self._pool:
            self._pool.shutdown(wait=False, cancel_futures=True)

    async def build_snapshot(self) -> Dict[str, Any]:
        """
        Copies what the worker needs. Pure data copying, no analysis.

        Yields to the event loop after every few symbols, so a tick arriving
        mid-snapshot waits for at most one small chunk rather than the whole
        watchlist (measured: ~0.6ms in one block before chunking).
        """
        from engine import brackets
        from engine.strategies import performance
        from feeds.multi_source_aggregator import trend_aggregator

        symbols = {}
        for i, sym in enumerate(set(state.watchlist) | set(state.active_positions)):
            if i and i % SNAPSHOT_CHUNK == 0:
                await asyncio.sleep(0)
            prices = state.price_history.get(sym)
            tick = state.latest_prices.get(sym)
            if not prices or tick is None:
                continue
            q = state.quant_metrics.get(sym)
            # Brackets depend on the live risk dial, which only this process knows.
            sl, tp, _ = brackets.derive(tick.price, q.atr if q else None)
            pos = state.active_positions.get(sym)
            symbols[sym] = {
                "prices": np.fromiter(prices, dtype=np.float64, count=len(prices)),
                "volumes": np.fromiter(state.volume_history.get(sym) or (), dtype=np.float64),
                "times": np.fromiter(state.time_history.get(sym) or (), dtype=np.float64),
                "price": float(tick.price),
                "quant": q,
                "sentiment": state.get_sentiment(sym),
                "consensus": trend_aggregator.get_consensus(sym),
                "is_crypto": is_crypto_symbol(sym),
                "sl_dist": abs(tick.price - sl),
                "tp_dist": abs(tp - tick.price),
                "position": ({"qty": float(pos.get("qty") or 0.0),
                              "avg_entry_price": float(pos.get("avg_entry_price") or 0.0)}
                             if pos else None),
            }
        trades = [{"r": t.get("r_multiple"), "pnl": t.get("pnl")}
                  for t in list(state.closed_trades)]
        return {
            "at": time.time(),
            "symbols": symbols,
            "min_buy_prob": state.risk_profile.min_buy_prob,
            "perf": performance.stats(),
            "trades": trades,
            "mc_paths": settings.MC_PATHS,
        }

    async def run_once(self) -> Optional[Dict[str, Any]]:
        from engine.analysis.worker import run_cycle
        t0 = time.perf_counter_ns()
        snap = await self.build_snapshot()
        latency.record_ns("worker_snapshot", t0)
        if not snap["symbols"]:
            return None
        loop = asyncio.get_running_loop()
        t1 = time.perf_counter_ns()
        result = await loop.run_in_executor(self._pool, run_cycle, snap)
        latency.record_ns("worker_cycle", t1)

        # Store: one dict assignment per symbol.
        for i, (sym, res) in enumerate(result["symbols"].items()):
            if i and i % SNAPSHOT_CHUNK == 0:
                await asyncio.sleep(0)
            state.analysis[sym] = res
        for sym in list(state.analysis):
            if sym not in result["symbols"] and sym not in state.watchlist:
                state.analysis.pop(sym, None)
        state.portfolio_analytics = result["portfolio"]
        self.partial_errors = result.get("errors") or {}
        for key, msg in self.partial_errors.items():
            # Log each distinct failure once, not every second.
            if self._logged_errors.get(key) != msg:
                self._logged_errors[key] = msg
                logger.error(f"Analysis partial failure [{key}]: {msg}")
        state.analysis_at = snap["at"]
        self.last_compute_ms = result["compute_ms"]
        self.cycles += 1
        return result

    async def _loop(self):
        while True:
            t0 = time.time()
            try:
                await self.run_once()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.errors += 1
                self.last_error = str(e)
                # Full traceback (includes the worker process's remote traceback),
                # logged once per distinct message rather than every second.
                if self._logged_errors.get("_cycle") != str(e):
                    self._logged_errors["_cycle"] = str(e)
                    logger.exception(f"Analysis cycle failed: {e}")
            await asyncio.sleep(max(settings.ANALYSIS_INTERVAL_SECONDS - (time.time() - t0), 0.05))

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "cycles": self.cycles,
            "errors": self.errors,
            "last_error": self.last_error,
            "partial_errors": self.partial_errors,
            "last_compute_ms": self.last_compute_ms,
            "age_s": round(time.time() - state.analysis_at, 2) if state.analysis_at else None,
            "interval_s": settings.ANALYSIS_INTERVAL_SECONDS,
        }


analysis_service = AnalysisService()
