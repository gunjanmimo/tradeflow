"""
Historical one-minute bars for the backtester, from Alpaca, cached on disk.

Stocks come from the IEX feed (the free plan), regular session only, since the
platform day-trades and flattens before the close. Each
symbol and day range is cached as CSV under backtest/output/bars/, so a rerun with other
settings replays the same data without another download.
"""
import csv
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple
from zoneinfo import ZoneInfo

logger = logging.getLogger("tradeflow.backtest")

Bar = Tuple[int, float, float, float, float, float]   # epoch minute, o, h, l, c, v
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "bars")
NY = ZoneInfo("America/New_York")


def _cache_path(symbol: str, start: datetime, end: datetime) -> str:
    name = f"{symbol.replace('/', '-')}_{start:%Y%m%d}_{end:%Y%m%d}.csv"
    return os.path.join(CACHE_DIR, name)


def in_regular_session(minute: int) -> bool:
    t = datetime.fromtimestamp(minute * 60, tz=NY)
    if t.weekday() >= 5:
        return False
    m = t.hour * 60 + t.minute
    return 9 * 60 + 30 <= m < 16 * 60


def _fetch(symbol: str, start: datetime, end: datetime) -> List[Bar]:
    """Alpaca market-data REST API, paged. Stocks from IEX (the free plan)."""
    import requests
    from core.config import settings
    url = "https://data.alpaca.markets/v2/stocks/bars"
    params = {"symbols": symbol, "timeframe": "1Min", "limit": 10000, "feed": "iex",
              "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"), "end": end.strftime("%Y-%m-%dT%H:%M:%SZ")}
    headers = {"APCA-API-KEY-ID": settings.ALPACA_API_KEY,
               "APCA-API-SECRET-KEY": settings.ALPACA_SECRET_KEY}
    bars: List[Bar] = []
    while True:
        r = requests.get(url, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        body = r.json()
        for b in (body.get("bars") or {}).get(symbol) or []:
            t = datetime.fromisoformat(b["t"].replace("Z", "+00:00"))
            if b["c"] > 0:
                bars.append((int(t.timestamp() // 60), float(b["o"]), float(b["h"]),
                             float(b["l"]), float(b["c"]), float(b.get("v") or 0.0)))
        token = body.get("next_page_token")
        if not token:
            break
        params["page_token"] = token
    bars = [b for b in bars if in_regular_session(b[0])]
    return sorted(set(bars))


def load(symbol: str, days: int, end: datetime = None) -> List[Bar]:
    """`days` calendar days of 1-minute bars ending at `end` (default: today 00:00 UTC)."""
    end = end or datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)
    path = _cache_path(symbol, start, end)
    if os.path.exists(path):
        with open(path) as f:
            return [(int(r[0]), *map(float, r[1:])) for r in csv.reader(f)]
    bars = _fetch(symbol, start, end)
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(bars)
    logger.info("Fetched %d bars for %s", len(bars), symbol)
    return bars


def load_many(symbols: List[str], days: int) -> Dict[str, List[Bar]]:
    out = {}
    for s in symbols:
        try:
            bars = load(s, days)
        except Exception as e:
            print(f"  ! {s}: could not load bars ({type(e).__name__}: {str(e)[:120]})")
            continue
        if bars:
            out[s] = bars
        else:
            print(f"  ! {s}: no bars in range")
    return out
