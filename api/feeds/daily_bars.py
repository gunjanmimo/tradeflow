"""
Daily closing prices for correlation, volatility and momentum.

The tick buffers (250 samples) cover minutes, which is fine for entry timing but
meaningless for diversification: two stocks can look uncorrelated over five
minutes and move as one over a quarter. Portfolio risk is measured on daily
returns, so this keeps ~14 months of adjusted daily closes per symbol.

Fails closed: with placeholder keys or a failed request there is simply no
history, and every consumer treats "no history" as "unknown" -- correlation is
not assumed to be low, and momentum is not scored.
"""
import asyncio
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Iterable

import numpy as np

from core.config import settings

logger = logging.getLogger("tradeflow.daily_bars")

_US_TICKER = re.compile(r"^[A-Z]{1,5}(\.[A-Z])?$")


class DailyBarCache:
    def __init__(self):
        # symbol -> (day numbers since epoch, adjusted closes)
        self.series: Dict[str, tuple] = {}
        # symbol -> average daily dollar volume over the last 20 sessions
        self.dollar_volume: Dict[str, float] = {}
        # symbol -> (highs, lows, volumes), aligned with series[symbol]
        self.hlv: Dict[str, tuple] = {}
        self.fetched_at: Dict[str, float] = {}
        self.last_error: Optional[str] = None
        self.last_refresh_at: float = 0.0
        self._stock_client = None
        self._lock = asyncio.Lock()

    @property
    def available(self) -> bool:
        key = settings.ALPACA_API_KEY
        return bool(key) and not key.startswith("PK_PLACEHOLDER")

    def _clients(self):
        if self._stock_client is None:
            from alpaca.data.historical import StockHistoricalDataClient
            self._stock_client = StockHistoricalDataClient(
                settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY)
        return self._stock_client

    def closes(self, symbol: str) -> Optional[np.ndarray]:
        s = self.series.get(symbol)
        return s[1] if s else None

    def status(self) -> Dict:
        return {
            "available": self.available,
            "symbols": len(self.series),
            "last_refresh_at": self.last_refresh_at or None,
            "last_error": self.last_error,
        }

    async def ensure(self, symbols: Iterable[str], max_age_s: float = 6 * 3600) -> int:
        """Fetches history for symbols that are missing or older than max_age_s."""
        if not self.available:
            return 0
        now = time.time()
        todo = sorted({s for s in symbols if now - self.fetched_at.get(s, 0.0) > max_age_s})
        if not todo:
            return 0
        async with self._lock:
            loop = asyncio.get_running_loop()
            stocks = [s for s in todo if _US_TICKER.match(s)]
            got = 0
            for i in range(0, len(stocks), 100):
                got += await loop.run_in_executor(None, self._fetch_sync, stocks[i:i + 100])
            # Mark attempted symbols so an unknown ticker is not re-requested
            # every cycle; it is retried when max_age_s elapses.
            for s in todo:
                self.fetched_at[s] = now
            self.last_refresh_at = now
            return got

    def _fetch_sync(self, symbols: List[str]) -> int:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.enums import Adjustment, DataFeed

        stock_client = self._clients()
        # Free data plans may not query the most recent 15 minutes of SIP data.
        end = datetime.now(timezone.utc) - timedelta(minutes=20)
        start = end - timedelta(days=settings.DAILY_BARS_LOOKBACK_DAYS)
        try:
            try:
                resp = stock_client.get_stock_bars(StockBarsRequest(
                    symbol_or_symbols=symbols, timeframe=TimeFrame.Day, start=start,
                    end=end, adjustment=Adjustment.ALL))
            except Exception as e:
                if "subscription" not in str(e).lower():
                    raise
                resp = stock_client.get_stock_bars(StockBarsRequest(
                    symbol_or_symbols=symbols, timeframe=TimeFrame.Day, start=start,
                    end=end, adjustment=Adjustment.ALL, feed=DataFeed.IEX))
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {str(e)[:160]}"
            logger.warning(f"Daily bars fetch failed for {symbols[:3]}...: {self.last_error}")
            return 0

        got = 0
        for sym, bars in (getattr(resp, "data", None) or {}).items():
            if not bars:
                continue
            days = np.array([int(b.timestamp.timestamp() // 86400) for b in bars], dtype=np.int64)
            closes = np.array([float(b.close) for b in bars], dtype=float)
            highs = np.array([float(b.high) for b in bars], dtype=float)
            lows = np.array([float(b.low) for b in bars], dtype=float)
            vols = np.array([float(b.volume or 0.0) for b in bars], dtype=float)
            ok = closes > 0
            if ok.sum() < 20:
                continue
            self.series[sym] = (days[ok], closes[ok])
            self.hlv[sym] = (highs[ok], lows[ok], vols[ok])
            self.dollar_volume[sym] = float((closes[ok] * vols[ok])[-20:].mean())
            got += 1
        self.last_error = None
        return got

    # ------------------------------------------------------------------
    # Derived measures
    # ------------------------------------------------------------------

    def aligned_returns(self, symbols: List[str], window: int) -> Optional[tuple]:
        """
        Daily log returns over the last `window` common trading days, aligned
        on the days every series shares. Returns (symbols_kept, matrix[n_symbols, n_days]) or None.
        """
        kept = [s for s in symbols if s in self.series]
        if not kept:
            return None
        common = None
        for s in kept:
            d = self.series[s][0]
            common = set(d.tolist()) if common is None else common & set(d.tolist())
        if not common or len(common) < 21:
            return None
        grid = np.array(sorted(common))[-(window + 1):]
        rows = []
        for s in kept:
            days, closes = self.series[s]
            idx = np.searchsorted(days, grid)
            rows.append(np.diff(np.log(closes[idx])))
        return kept, np.array(rows)

    def momentum(self, symbol: str) -> Optional[Dict[str, float]]:
        """3-month return, 12-1 month momentum, and trend vs the 200-day average."""
        c = self.closes(symbol)
        if c is None or len(c) < 70:
            return None
        out = {"ret_3m": float(c[-1] / c[-64] - 1.0)}
        if len(c) >= 253:
            # Skip the latest month: short-term reversal contaminates momentum.
            out["mom_12_1"] = float(c[-22] / c[-253] - 1.0)
        if len(c) >= 200:
            out["above_200d"] = bool(c[-1] > c[-200:].mean())
        r = np.diff(np.log(c[-64:]))
        out["vol_3m_ann"] = float(r.std(ddof=1) * np.sqrt(252)) if len(r) > 2 else None
        return out


daily_bars = DailyBarCache()
