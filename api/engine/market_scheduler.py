import asyncio
import logging
from datetime import datetime, timezone
import zoneinfo
from typing import Dict, Any, Optional

from core.config import settings
from core.state import state
from engine.executor import executor
from feeds.expert_watcher import expert_watcher
from feeds.news_feed import news_feed

logger = logging.getLogger("tradeflow.scheduler")

class MarketSessionScheduler:
    """
    Automated Market-Session Cron Service.
    Monitors global market session clocks (US & Europe) and triggers:
    1. Pre-market Expert & Watchlist Sync (08:30 EST)
    2. Market Open Laya Sentiment Scoring & Quant Activation (09:30 EST)
    3. European Open Macro Check (09:00 CET)
    """
    def __init__(self):
        self._running = False
        self._task = None
        self.last_us_sync: Optional[datetime] = None
        self.last_eu_sync: Optional[datetime] = None
        
        # Timezones
        try:
            self.tz_ny = zoneinfo.ZoneInfo("America/New_York")
            self.tz_cet = zoneinfo.ZoneInfo("Europe/Berlin")
        except Exception:
            self.tz_ny = timezone.utc
            self.tz_cet = timezone.utc

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._clock_loop())
        logger.info("MarketSessionScheduler started (monitoring US & EU market sessions).")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()

    CLOCK_TTL_SECONDS = 30.0

    def _maybe_refresh_clock(self):
        """Starts a background clock fetch when the cache is stale. Never blocks."""
        import time as _time
        if not hasattr(self, "_clock_cache"):
            self._clock_cache = (False, None)
            self._clock_at = 0.0
            self._clock_inflight = False
        if (self._clock_inflight or not (executor.is_connected and executor.trading_client)
                or _time.time() - self._clock_at < self.CLOCK_TTL_SECONDS):
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._clock_inflight = True

        def fetch():
            clock = executor.trading_client.get_clock()
            next_close = getattr(clock, "next_close", None)
            return (clock.is_open, clock.next_open.isoformat(),
                    next_close.isoformat() if next_close else None)

        def done(fut):
            self._clock_inflight = False
            self._clock_at = _time.time()
            try:
                is_open, next_open, next_close = fut.result()
                self._clock_cache = (is_open, next_open)
                # While the market is open this is TODAY's close, half-days included.
                self._next_close_iso = next_close
            except Exception as e:
                logger.debug(f"Clock refresh failed, keeping last value: {e}")

        loop.run_in_executor(None, fetch).add_done_callback(done)

    def get_market_status(self) -> Dict[str, Any]:
        """Returns live market status, next open, and session info"""
        now_utc = datetime.now(timezone.utc)
        now_ny = datetime.now(self.tz_ny)
        now_cet = datetime.now(self.tz_cet)

        # Broker clock, served from a cache refreshed off the event loop.
        # This used to call trading_client.get_clock() -- a blocking HTTP request
        # -- on every dashboard frame: measured ~90ms, 4x/second, stalling the
        # tick loop ~37% of the time and spending 240 of Alpaca's 200 req/min.
        self._maybe_refresh_clock()
        is_us_open, next_us_open = self._clock_cache

        # European Market hours: 09:00 - 17:30 CET on weekdays
        is_weekday = now_cet.weekday() < 5
        is_eu_open = is_weekday and (9 <= now_cet.hour < 17 or (now_cet.hour == 17 and now_cet.minute <= 30))

        # US Market regular hours fallback: 09:30 - 16:00 EST on weekdays
        if not is_us_open and is_weekday:
            if (now_ny.hour == 9 and now_ny.minute >= 30) or (10 <= now_ny.hour < 16):
                is_us_open = True

        # The operator's clock (settings.DISPLAY_TIMEZONE): what the dashboard shows.
        from core.config import settings
        from core.market_hours import us_session
        try:
            local_tz = zoneinfo.ZoneInfo(settings.DISPLAY_TIMEZONE)
        except Exception:
            local_tz = self.tz_cet
        now_local = datetime.now(local_tz)

        def local(iso: Optional[str]) -> Optional[str]:
            if not iso:
                return None
            try:
                t = datetime.fromisoformat(iso).astimezone(local_tz)
            except ValueError:
                return None
            day = "today" if t.date() == now_local.date() else t.strftime("%a %d %b")
            return f"{day} {t.strftime('%H:%M %Z')}"

        return {
            "display_tz": str(local_tz),
            "current_time_local": now_local.strftime("%H:%M:%S %Z"),
            "us_session": us_session(),
            "next_us_open_local": local(next_us_open),
            "next_us_close_local": local(getattr(self, "_next_close_iso", None)) if is_us_open else None,
            "current_time_utc": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "current_time_ny": now_ny.strftime("%Y-%m-%d %I:%M:%S %p %Z"),
            "current_time_cet": now_cet.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "is_us_market_open": is_us_open,
            "is_eu_market_open": is_eu_open,
            "next_us_open": next_us_open or "Next weekday at 09:30 AM EST",
            "last_morning_sync": self.last_us_sync.isoformat() if self.last_us_sync else "None yet",
        }

    async def run_morning_routine(self, market: str = "US"):
        """
        The Morning Bell Ingestion Routine:
        1. Syncs expert trader picks & hedge fund 13Fs.
        2. Refreshes active watchlist.
        3. Pulls overnight & pre-market headlines.
        4. Runs Laya sentiment scoring.
        5. Syncs account balances and active positions from Alpaca.
        """
        logger.info(f"Executing scheduled Morning Routine for {market} Market...")
        state.log_event("SCHEDULER", f"Executing Morning Routine for {market} Market session...")

        # 1. Sync accounts and positions (blocking HTTP -> off the event loop)
        if executor.is_connected:
            await asyncio.get_running_loop().run_in_executor(None, executor.sync_account_and_positions)

        # 2. Sync Expert Portfolios & 4-Pillar Consensus Aggregator
        from feeds.multi_source_aggregator import trend_aggregator
        await trend_aggregator.aggregate_all_sources()
        await expert_watcher.sync_experts_now()

        # 3. Pull REAL overnight/pre-market headlines and score them.
        #
        # This previously synthesized a filler sentence per symbol ("Morning market
        # open preparation: assessing institutional volume...") and scored THAT.
        # Measured, that string reads 0.14 bullish / 0.79 neutral -- so every
        # morning it overwrote each symbol's genuine sentiment with noise, racing
        # the aggregator for whichever wrote last.
        ingested = await news_feed.poll_once()
        state.log_event("SCHEDULER", f"Morning news sweep ingested {ingested} real headlines.")

        if market == "US":
            self.last_us_sync = datetime.now(timezone.utc)
        else:
            self.last_eu_sync = datetime.now(timezone.utc)

        state.log_event("SCHEDULER", f"{market} Morning Routine complete! Watchlist and Laya sentiment prepared.")

    async def _clock_loop(self):
        """Continuously checks the clock every 30 seconds to trigger scheduled runs"""
        while self._running:
            try:
                now_ny = datetime.now(self.tz_ny)
                now_cet = datetime.now(self.tz_cet)
                is_weekday = now_ny.weekday() < 5

                if is_weekday:
                    # US Pre-Market trigger: 09:00 AM EST (30 mins before 09:30 open)
                    if now_ny.hour == 9 and now_ny.minute == 0:
                        if not self.last_us_sync or (datetime.now(timezone.utc) - self.last_us_sync).total_seconds() > 3600:
                            await self.run_morning_routine("US")

                    # EU Open trigger: 08:45 AM CET (15 mins before 09:00 open)
                    if now_cet.hour == 8 and now_cet.minute == 45:
                        if not self.last_eu_sync or (datetime.now(timezone.utc) - self.last_eu_sync).total_seconds() > 3600:
                            await self.run_morning_routine("EU")

                await asyncio.sleep(30)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in market scheduler clock loop: {e}")
                await asyncio.sleep(30)

market_scheduler = MarketSessionScheduler()
