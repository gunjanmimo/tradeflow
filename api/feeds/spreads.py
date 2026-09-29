"""
The bid/ask spread each stock really trades at: what every spread gate reads.

The live quote stream on the free data plan is IEX only -- IEX's own book, not
the national best bid and offer. For stocks where IEX holds little liquidity it
is persistently wide: measured 2026-09-29, BE quoted 4.6-7.4% on IEX against a
real (consolidated) 0.07%, KMX 0.8-5.3% against 0.15%, ASML 0.3-0.5% against
0.03%. A single IEX snapshot is also noisy (OKTA's median 0.15%, spikes to 9%).
Gating on it blocked good stocks for spreads they did not have; orders fill at
the national best price, not IEX's.

Sources, best first:
  sip   median consolidated (SIP) spread 16-17 minutes ago -- the free plan
        serves SIP only when it is 15+ minutes old; a liquid stock's spread does
        not change much in that time. Refreshed every SPREAD_REFRESH_SECONDS for
        watched and held stocks.
  iex   median IEX spread over the last IEX_WINDOW_SECONDS (5+ quotes): a
        fallback, still an overestimate where IEX is thin
  None  unknown: gates do not block on it (the scout only watches stocks that
        passed its liquidity filter)
"""
import asyncio
import logging
import statistics
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple

from core.config import settings

logger = logging.getLogger("tradeflow.spreads")

SPREAD_REFRESH_SECONDS = 300.0
SIP_DELAY_MINUTES = 16.0
SIP_WINDOW_MINUTES = 1.0
SIP_MAX_AGE_SECONDS = 1200.0
IEX_WINDOW_SECONDS = 120.0
IEX_MIN_QUOTES = 5


def relative(bid: float, ask: float) -> Optional[float]:
    if not bid or not ask or ask < bid:
        return None
    mid = (bid + ask) / 2.0
    return (ask - bid) / mid if mid > 0 else None


class SpreadMonitor:
    def __init__(self):
        self.sip: Dict[str, Tuple[float, float, int]] = {}      # symbol -> (spread, measured_at, n quotes)
        self._iex: Dict[str, deque] = {}                         # symbol -> (t, spread)
        self._client = None
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self.last_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def on_quote(self, symbol: str, bid: float, ask: float, ts: Optional[float] = None):
        rel = relative(bid, ask)
        if rel is None:
            return
        self._iex.setdefault(symbol, deque(maxlen=600)).append((time.time() if ts is None else ts, rel))

    def _sip_sync(self, symbols: List[str]) -> Dict[str, Tuple[float, int]]:
        from alpaca.data.enums import DataFeed
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockQuotesRequest
        if self._client is None:
            self._client = StockHistoricalDataClient(settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY)
        end = datetime.now(timezone.utc) - timedelta(minutes=SIP_DELAY_MINUTES)
        start = end - timedelta(minutes=SIP_WINDOW_MINUTES)
        out = {}
        for sym in symbols:            # one request each: a shared row limit starves the last symbols
            try:
                quotes = self._client.get_stock_quotes(StockQuotesRequest(
                    symbol_or_symbols=sym, start=start, end=end, feed=DataFeed.SIP, limit=2000)).data.get(sym, [])
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {str(e)[:120]}"
                continue
            rels = [r for r in (relative(q.bid_price, q.ask_price) for q in quotes) if r is not None]
            if len(rels) >= IEX_MIN_QUOTES:
                out[sym] = (statistics.median(rels), len(rels))
        return out

    async def refresh(self, symbols: Iterable[str]) -> int:
        key = settings.ALPACA_API_KEY
        if not key or key.startswith("PK_PLACEHOLDER"):
            return 0
        syms = sorted(s for s in set(symbols) if s and "/" not in s)
        if not syms:
            return 0
        got = await asyncio.get_running_loop().run_in_executor(None, self._sip_sync, syms)
        now = time.time()
        for sym, (spread, n) in got.items():
            self.sip[sym] = (spread, now, n)
        return len(got)

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()

    async def _loop(self):
        from core.state import state
        from core.market_hours import us_session, REGULAR
        while self._running:
            try:
                if us_session() == REGULAR:
                    await self.refresh(set(state.watchlist) | set(state.active_positions))
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
                logger.warning("Spread refresh failed: %s", e)
            try:
                await asyncio.sleep(SPREAD_REFRESH_SECONDS)
            except asyncio.CancelledError:
                break

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def iex_median(self, symbol: str, now: Optional[float] = None) -> Optional[float]:
        now = time.time() if now is None else now
        recent = [r for t, r in (self._iex.get(symbol) or ()) if now - t <= IEX_WINDOW_SECONDS]
        return statistics.median(recent) if len(recent) >= IEX_MIN_QUOTES else None

    def estimate(self, symbol: str, now: Optional[float] = None) -> Tuple[Optional[float], str]:
        """(spread as a fraction of price, source) -- source is "sip", "iex" or "unknown"."""
        now = time.time() if now is None else now
        s = self.sip.get(symbol)
        if s and now - s[1] <= SIP_MAX_AGE_SECONDS:
            return s[0], "sip"
        m = self.iex_median(symbol, now)
        if m is not None:
            return m, "iex"
        return None, "unknown"

    def status(self) -> Dict:
        now = time.time()
        return {"sip_symbols": sum(1 for v in self.sip.values() if now - v[1] <= SIP_MAX_AGE_SECONDS),
                "iex_symbols": len(self._iex), "last_error": self.last_error}


spreads = SpreadMonitor()
