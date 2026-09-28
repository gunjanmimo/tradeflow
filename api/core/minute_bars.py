"""
One-minute OHLCV bars per symbol: the time series the trend analyst reads.

Bars are time-bucketed, so every symbol gets the same clock whatever its tick
rate.

  backfill   at start-up and for every symbol newly watched or held, the last
             sessions of 1-minute bars come from Alpaca's historical API, so the
             agents understand the trend before they act rather than waiting
             20+ minutes for live bars to accumulate
  live       a streamed minute bar (on_bar) is authoritative: it is stored
             under its own start minute with its real open/high/low/close/volume,
             exactly as the historical API and the backtester see it. Between
             bars, trade prints and quotes (on_tick) keep a provisional bar for
             the minute still forming; the streamed bar replaces it.

Decisions read closed_rows(): bars whose minute has ended. The forming minute is
never an input, so live features match the backtest and training bar for bar.

Nothing here makes a network call on the tick path: backfill runs in a thread
from the trend analyst's loop.
"""
import asyncio
import logging
import time
from collections import deque
from typing import Dict, Iterable, List, Optional

import numpy as np

logger = logging.getLogger("tradeflow.minute_bars")

MAX_BARS = 780                      # two regular US sessions
BACKFILL_RETRY_SECONDS = 900.0


class MinuteBars:
    def __init__(self):
        # symbol -> deque of [minute, open, high, low, close, volume]
        self._bars: Dict[str, deque] = {}
        # symbol -> minute of the newest bar that came from the stream or the
        # historical API (final), as opposed to one built from ticks.
        self._final_minute: Dict[str, int] = {}
        self.backfilled_at: Dict[str, float] = {}
        self.last_error: Optional[str] = None
        self._inflight = False

    # ------------------------------------------------------------------
    # Live
    # ------------------------------------------------------------------

    def on_tick(self, symbol: str, price: float, volume: float = 0.0,
                ts: Optional[float] = None):
        if not price or price <= 0:
            return
        minute = int((time.time() if ts is None else ts) // 60)
        dq = self._bars.get(symbol)
        if dq is None:
            dq = self._bars[symbol] = deque(maxlen=MAX_BARS)
        if dq and dq[-1][0] == minute:
            bar = dq[-1]
            bar[2] = max(bar[2], price)
            bar[3] = min(bar[3], price)
            bar[4] = price
            bar[5] += volume or 0.0
        elif dq and dq[-1][0] > minute:
            return                          # out of order: ignore
        else:
            dq.append([minute, price, price, price, price, volume or 0.0])

    def on_bar(self, symbol: str, minute: int, o: float, h: float, l: float, c: float,
               v: float = 0.0):
        """
        A completed bar from the stream, keyed by its START minute. Replaces a
        provisional bar built from ticks for that minute, or is inserted in order.
        """
        if not c or c <= 0:
            return
        minute = int(minute)
        row = [minute, float(o), float(h), float(l), float(c), float(v or 0.0)]
        dq = self._bars.get(symbol)
        if dq is None:
            dq = self._bars[symbol] = deque(maxlen=MAX_BARS)
        if not dq or dq[-1][0] < minute:
            dq.append(row)
        else:
            rows = [b for b in dq if b[0] != minute] + [row]
            rows.sort(key=lambda b: b[0])
            self._bars[symbol] = deque(rows[-MAX_BARS:], maxlen=MAX_BARS)
        self._final_minute[symbol] = max(minute, self._final_minute.get(symbol, minute))

    def merge_history(self, symbol: str, bars: Iterable[tuple]):
        """
        Merges historical bars by minute. A historical bar replaces a provisional
        one built from ticks for the same minute; a bar the stream delivered is
        already final and is kept.
        """
        final = self._final_minute.get(symbol)
        merged = {b[0]: list(b) for b in (self._bars.get(symbol) or ())}
        hist = [list(b) for b in bars]
        for b in hist:
            if final is not None and b[0] <= final and b[0] in merged:
                continue
            merged[b[0]] = b
        rows = [merged[m] for m in sorted(merged)][-MAX_BARS:]
        self._bars[symbol] = deque(rows, maxlen=MAX_BARS)
        if hist:
            newest = max(b[0] for b in hist)
            self._final_minute[symbol] = max(newest, final if final is not None else newest)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def closes(self, symbol: str, n: Optional[int] = None) -> np.ndarray:
        dq = self._bars.get(symbol)
        if not dq:
            return np.empty(0)
        rows = list(dq)[-n:] if n else list(dq)
        return np.fromiter((b[4] for b in rows), dtype=np.float64, count=len(rows))

    def rows(self, symbol: str) -> List[list]:
        """Copies of the bars held for a symbol: [minute, open, high, low, close, volume], oldest first."""
        return [list(b) for b in (self._bars.get(symbol) or ())]

    def closed_rows(self, symbol: str, now: Optional[float] = None) -> List[list]:
        """Bars whose minute has ended, oldest first. The forming minute is left out."""
        cur = int((time.time() if now is None else now) // 60)
        return [list(b) for b in (self._bars.get(symbol) or ()) if b[0] < cur]

    def count(self, symbol: str) -> int:
        return len(self._bars.get(symbol) or ())

    def drop(self, symbol: str):
        self._bars.pop(symbol, None)
        self._final_minute.pop(symbol, None)
        self.backfilled_at.pop(symbol, None)

    def status(self) -> Dict:
        return {"symbols": len(self._bars),
                "backfilled": len(self.backfilled_at),
                "last_error": self.last_error}

    # ------------------------------------------------------------------
    # Backfill
    # ------------------------------------------------------------------

    async def ensure(self, symbols: Iterable[str]) -> int:
        """Backfills symbols not backfilled recently. One run at a time; never raises."""
        from core.config import settings
        key = settings.ALPACA_API_KEY
        if not settings.MINUTE_BARS_BACKFILL or not key or key.startswith("PK_PLACEHOLDER"):
            return 0
        if self._inflight:
            return 0
        now = time.time()
        todo = sorted({s for s in symbols
                       if now - self.backfilled_at.get(s, 0.0) > BACKFILL_RETRY_SECONDS})
        if not todo:
            return 0
        self._inflight = True
        try:
            loop = asyncio.get_running_loop()
            got = 0
            for i in range(0, len(todo), 25):
                result = await loop.run_in_executor(None, self._fetch_sync, todo[i:i + 25])
                for sym, bars in result.items():
                    self.merge_history(sym, bars)
                    got += 1
            for s in todo:
                self.backfilled_at[s] = now
            if got:
                logger.info(f"Backfilled 1-minute bars for {got} symbols")
            return got
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {str(e)[:160]}"
            logger.warning(f"Minute-bar backfill failed: {self.last_error}")
            return 0
        finally:
            self._inflight = False

    def _fetch_sync(self, symbols: List[str]) -> Dict[str, list]:
        from datetime import datetime, timedelta, timezone
        from core.config import settings
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.enums import DataFeed

        end = datetime.now(timezone.utc)
        client = StockHistoricalDataClient(settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY)
        # IEX: the free plan may not query the last 15 minutes of SIP data,
        # which is exactly the part an intraday trend needs. Four calendar
        # days reach back across a weekend to the previous session.
        resp = client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=symbols, timeframe=TimeFrame.Minute,
            start=end - timedelta(days=4), end=end, feed=DataFeed.IEX))
        out: Dict[str, list] = {}
        for sym, bars in (getattr(resp, "data", None) or {}).items():
            rows = [(int(b.timestamp.timestamp() // 60), float(b.open), float(b.high),
                     float(b.low), float(b.close), float(b.volume or 0.0))
                    for b in bars if b.close and b.close > 0]
            if rows:
                out[sym] = rows[-MAX_BARS:]
        self.last_error = None
        return out


minute_bars = MinuteBars()
