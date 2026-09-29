import asyncio
import logging
import random
import time
from typing import List, Dict, Any
from core.state import state
from sentiment.laya_service import laya_service

logger = logging.getLogger("tradeflow.experts")

# There is no expert/copy-trader data source wired in: the illustrative list of
# made-up portfolios that used to sit here is gone. Real copy-trading conviction
# comes from the eToro pillar in multi_source_aggregator.py (needs ETORO_API_KEY).
BENCHMARK_EXPERTS: List[Dict[str, Any]] = []


class ExpertTraderWatcher:
    """
    Monitors top expert trader portfolios and public holdings.
    Identifies high-conviction overlapping tickers and feeds them to the Watchlist & Laya.
    """
    def __init__(self):
        self._running = False
        self._task = None

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._sync_loop())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()

    async def sync_experts_now(self):
        """
        Expert-portfolio sync.

        DISABLED as a trading input. BENCHMARK_EXPERTS used to be a hardcoded
        illustrative list, not live data (now empty). It previously did two harmful things:
        it expanded the tradeable watchlist with unvetted symbols, and it fed
        invented thesis strings ("Aggressive accumulation in AMZN...") into Laya,
        which then reported bullish sentiment on fiction.

        The profiles are retained for the UI's reference panel only. Real
        copy-trading conviction now comes from the eToro pillar in
        multi_source_aggregator.py, which requires a genuine API key.
        """
        logger.info("Expert panel is reference-only; not used as a trading signal.")
        state.log_event(
            "EXPERTS",
            "Expert panel is illustrative reference data only -- it does not add "
            "symbols to the watchlist or produce sentiment. Real copy-trade "
            "conviction requires ETORO_API_KEY."
        )

    async def _sync_loop(self):
        # Initial sync on boot
        await self.sync_experts_now()
        while self._running:
            try:
                # Sync every 30 minutes in production, or periodically
                await asyncio.sleep(600)
                await self.sync_experts_now()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error syncing expert portfolios: {e}")
                await asyncio.sleep(30)

expert_watcher = ExpertTraderWatcher()
