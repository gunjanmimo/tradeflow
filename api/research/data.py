"""
Cached one-minute bars for research, backtests and RL training.

Bars come from Alpaca's IEX feed -- the same feed the live engine streams -- so
volume, gaps and prices match what the strategy will see when it trades.
Regular session (09:30-16:00 New York) only, split-adjusted.

One file per symbol, api/datasets/bars_1m/<SYMBOL>.npz, updated incrementally:
a download only fetches days not already held. Arrays:

    minute  int32   minutes since the Unix epoch, bar START
    o h l c float32
    v       float32 IEX volume
"""
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np

logger = logging.getLogger("tradeflow.research.data")

NY = ZoneInfo("America/New_York")
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
BAR_DIR = os.path.join(ROOT, "bars_1m")

# Liquid US large caps across sectors, plus SPY/QQQ for market context. Liquid
# names keep the IEX feed dense and the spread (the dominant cost) small.
DEFAULT_UNIVERSE = (
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "NFLX",
    "ORCL", "CRM", "ADBE", "QCOM", "INTC", "CSCO", "TXN", "PLTR",
    "JPM", "BAC", "V", "MA", "GS", "WFC",
    "UNH", "LLY", "JNJ", "MRK", "ABBV", "PFE",
    "XOM", "CVX", "COP",
    "WMT", "COST", "HD", "PG", "KO", "PEP", "MCD",
    "BA", "CAT", "GE",
    "SPY", "QQQ",
)
MARKET = "SPY"


@dataclass
class Bars:
    symbol: str
    minute: np.ndarray
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray

    def __len__(self):
        return len(self.minute)

    def days(self) -> np.ndarray:
        """New York day number (days since epoch in NY time) per bar."""
        return ny_day(self.minute)

    def day_slices(self) -> List[Tuple[int, slice]]:
        """(ny_day, slice) per session, oldest first."""
        d = self.days()
        if len(d) == 0:
            return []
        cut = np.flatnonzero(np.diff(d)) + 1
        starts = np.r_[0, cut]
        ends = np.r_[cut, len(d)]
        return [(int(d[s]), slice(int(s), int(e))) for s, e in zip(starts, ends)]

    def select_days(self, keep: Iterable[int]) -> "Bars":
        keep = np.asarray(sorted(set(keep)))
        m = np.isin(self.days(), keep)
        return Bars(self.symbol, self.minute[m], self.o[m], self.h[m], self.l[m], self.c[m], self.v[m])


def ny_offset_minutes(minute: np.ndarray) -> np.ndarray:
    """UTC offset of New York (in minutes) for each bar: -240 in summer, -300 in winter."""
    minute = np.asarray(minute, dtype=np.int64)
    out = np.empty(len(minute), dtype=np.int64)
    if len(minute) == 0:
        return out
    # DST changes at most twice a year: resolve per day, not per bar.
    days = minute // 1440
    uniq, inv = np.unique(days, return_inverse=True)
    offs = np.array([int(datetime.fromtimestamp(int(d) * 86400 + 12 * 3600, tz=NY)
                         .utcoffset().total_seconds() // 60) for d in uniq], dtype=np.int64)
    out[:] = offs[inv]
    return out


def ny_minute_of_day(minute: np.ndarray) -> np.ndarray:
    """Minutes after New York midnight for each bar start (570 = 09:30)."""
    local = np.asarray(minute, dtype=np.int64) + ny_offset_minutes(minute)
    return local % 1440


def ny_day(minute: np.ndarray) -> np.ndarray:
    local = np.asarray(minute, dtype=np.int64) + ny_offset_minutes(minute)
    return local // 1440


def _path(symbol: str) -> str:
    return os.path.join(BAR_DIR, f"{symbol.replace('/', '-').upper()}.npz")


def load(symbol: str) -> Optional[Bars]:
    try:
        z = np.load(_path(symbol))
    except (OSError, ValueError):
        return None
    return Bars(symbol.upper(), z["minute"].astype(np.int64), z["o"].astype(np.float64),
                z["h"].astype(np.float64), z["l"].astype(np.float64),
                z["c"].astype(np.float64), z["v"].astype(np.float64))


def load_many(symbols: Iterable[str]) -> Dict[str, Bars]:
    out = {}
    for s in symbols:
        b = load(s)
        if b is not None and len(b):
            out[s.upper()] = b
    return out


def available() -> List[str]:
    try:
        return sorted(f[:-4] for f in os.listdir(BAR_DIR) if f.endswith(".npz"))
    except OSError:
        return []


def _save(b: Bars):
    os.makedirs(BAR_DIR, exist_ok=True)
    tmp = _path(b.symbol) + ".tmp.npz"
    np.savez_compressed(tmp, minute=b.minute.astype(np.int32), o=b.o.astype(np.float32),
                        h=b.h.astype(np.float32), l=b.l.astype(np.float32),
                        c=b.c.astype(np.float32), v=b.v.astype(np.float32))
    os.replace(tmp, _path(b.symbol))


def _regular(minute: np.ndarray) -> np.ndarray:
    mod = ny_minute_of_day(minute)
    return (mod >= 570) & (mod < 960)


def _fetch(symbol: str, start: datetime, end: datetime, key: str, secret: str) -> np.ndarray:
    """Alpaca stock bars (IEX, split-adjusted), paged. Returns rows [minute, o, h, l, c, v]."""
    import requests
    url = "https://data.alpaca.markets/v2/stocks/bars"
    params = {"symbols": symbol, "timeframe": "1Min", "limit": 10000, "feed": "iex",
              "adjustment": "split", "sort": "asc",
              "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"), "end": end.strftime("%Y-%m-%dT%H:%M:%SZ")}
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    rows = []
    while True:
        for attempt in range(6):
            r = requests.get(url, params=params, headers=headers, timeout=60)
            if r.status_code == 429:
                time.sleep(2 + 3 * attempt)
                continue
            r.raise_for_status()
            break
        body = r.json()
        for b in (body.get("bars") or {}).get(symbol) or []:
            if b["c"] <= 0:
                continue
            t = datetime.fromisoformat(b["t"].replace("Z", "+00:00"))
            rows.append((int(t.timestamp() // 60), b["o"], b["h"], b["l"], b["c"], b.get("v") or 0.0))
        token = body.get("next_page_token")
        if not token:
            break
        params["page_token"] = token
    if not rows:
        return np.empty((0, 6))
    a = np.asarray(rows, dtype=np.float64)
    return a[_regular(a[:, 0].astype(np.int64))]


def update(symbols: Iterable[str], days: int, end: Optional[datetime] = None,
           workers: int = 4, verbose: bool = True) -> Dict[str, int]:
    """
    Makes sure each symbol holds bars for the last `days` calendar days. Only the
    missing span (before the oldest or after the newest bar held) is fetched.
    Returns {symbol: bars held}.
    """
    from core.config import settings
    key, secret = settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY
    if not key or key.startswith("PK_PLACEHOLDER"):
        raise SystemExit("Alpaca keys are needed to download bars (api/.env).")
    # The free plan may not query the last 15 minutes of SIP data; IEX is fine,
    # but stop a little short so a bar still forming is never stored.
    end = end or datetime.now(timezone.utc) - timedelta(minutes=16)
    start = end - timedelta(days=days)

    def one(sym: str) -> Tuple[str, int]:
        sym = sym.upper()
        have = load(sym)
        parts = []
        if have is None or not len(have):
            parts.append(_fetch(sym, start, end, key, secret))
        else:
            first = datetime.fromtimestamp(int(have.minute[0]) * 60, tz=timezone.utc)
            last = datetime.fromtimestamp(int(have.minute[-1]) * 60 + 60, tz=timezone.utc)
            if start < first - timedelta(minutes=1):
                parts.append(_fetch(sym, start, first, key, secret))
            if end > last:
                parts.append(_fetch(sym, last, end, key, secret))
            parts.append(np.column_stack([have.minute, have.o, have.h, have.l, have.c, have.v]))
        parts = [p for p in parts if len(p)]
        if not parts:
            return sym, 0
        a = np.concatenate(parts)
        _, idx = np.unique(a[:, 0].astype(np.int64), return_index=True)
        a = a[idx]
        b = Bars(sym, a[:, 0].astype(np.int64), a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5])
        _save(b)
        return sym, len(b)

    out = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for sym, n in pool.map(one, list(symbols)):
            out[sym] = n
            if verbose:
                print(f"  {sym}: {n:,} bars")
    return out
