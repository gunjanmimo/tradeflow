import asyncio
import logging
import random
import time
from typing import List, Dict, Any
from core.state import state
from sentiment.laya_service import laya_service

logger = logging.getLogger("tradeflow.experts")

# Sample expert profiles representing top algorithmic/copy traders and smart-money funds
BENCHMARK_EXPERTS = [
    {
        "trader_id": "quant_fund_alpha",
        "name": "Alpha Systematic (eToro Popular Investor)",
        "win_rate": 0.76,
        "portfolio": ["NVDA", "AAPL", "MSFT", "PLTR"],
        "recent_action": "Increased position in NVDA due to data center demand acceleration."
    },
    {
        "trader_id": "macro_whale_hedge",
        "name": "Smart Money Whale (13F Tracker)",
        "win_rate": 0.71,
        "portfolio": ["AMZN", "GOOGL", "META", "TSLA"],
        "recent_action": "Aggressive accumulation in AMZN cloud infrastructure."
    },
    {
        "trader_id": "momentum_breakout",
        "name": "Breakout Pro (Top Momentum)",
        "win_rate": 0.68,
        "portfolio": ["AMD", "AVGO", "ARM", "NVDA"],
        "recent_action": "Bullish breakout setup above 20 EMA in semiconductor sector."
    }
]

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

        DISABLED as a trading input. BENCHMARK_EXPERTS below is a hardcoded
        illustrative list, not live data. It previously did two harmful things:
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
