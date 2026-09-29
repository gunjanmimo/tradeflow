"""
Where the scout's candidates and evidence come from. Every source is queried by
us directly, needs no key beyond the Alpaca one already configured, and fails
closed: an unreachable source returns nothing and says so in health, and the
ranking runs on whatever answered.

  Alpaca screener   most active stocks by volume, top % gainers
  Alpaca news       every headline of the last day, all symbols (paginated)
  Alpaca snapshots  latest price, today's and the previous session's bar
  ApeWisdom         mentions and upvotes across the stock subreddits
                    (wallstreetbets, stocks, investing, ...), now and 24h ago
  StockTwits        the trending symbols list (Cloudflare blocks it at times)
"""
import asyncio
import json
import logging
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from core.config import settings

logger = logging.getLogger("tradeflow.scout.sources")

_NY = ZoneInfo("America/New_York")
# Plain urllib: StockTwits' Cloudflare front challenges aiohttp's client but not this.
_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) TradeFlow scout", "Accept": "application/json"}
ASSETS_MAX_AGE_SECONDS = 24 * 3600.0
APEWISDOM_URL = "https://apewisdom.io/api/v1.0/filter/all-stocks/page/{page}"
STOCKTWITS_TRENDING_URL = "https://api.stocktwits.com/api/2/trending/symbols.json"


class Sources:
    def __init__(self):
        self.health: Dict[str, Dict[str, Any]] = {}
        self._screener = None
        self._news = None
        self._data = None
        self.asset_names: Dict[str, str] = {}      # tradable US equities -> name
        self.asset_exchange: Dict[str, str] = {}   # -> NYSE, NASDAQ, ..., OTC
        self._assets_at = 0.0

    @property
    def alpaca_available(self) -> bool:
        key = settings.ALPACA_API_KEY
        return bool(key) and not key.startswith("PK_PLACEHOLDER")

    def _mark(self, name: str, ok: bool, detail: str = "", count: int = 0):
        self.health[name] = {"ok": ok, "detail": detail[:200], "count": count, "at": time.time()}

    def _clients(self):
        if self._data is None:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.historical.news import NewsClient
            from alpaca.data.historical.screener import ScreenerClient
            k, sk = settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY
            self._screener, self._news = ScreenerClient(k, sk), NewsClient(k, sk)
            self._data = StockHistoricalDataClient(k, sk)
        return self._screener, self._news, self._data

    async def _run(self, name: str, fn, *args):
        """Runs a blocking Alpaca call off the event loop; None (and health) on failure."""
        if not self.alpaca_available:
            self._mark(name, False, "Alpaca keys are placeholders")
            return None
        try:
            return await asyncio.get_running_loop().run_in_executor(None, fn, *args)
        except Exception as e:
            self._mark(name, False, f"{type(e).__name__}: {e}")
            logger.warning("Scout source %s failed: %s", name, e)
            return None

    # ------------------------------------------------------------------
    # Alpaca
    # ------------------------------------------------------------------

    def _assets_sync(self) -> Dict[str, tuple]:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.enums import AssetClass, AssetStatus
        from alpaca.trading.requests import GetAssetsRequest
        client = TradingClient(settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY, paper=True)
        assets = client.get_all_assets(GetAssetsRequest(status=AssetStatus.ACTIVE,
                                                        asset_class=AssetClass.US_EQUITY))
        return {a.symbol: (a.name or "", str(getattr(a.exchange, "value", a.exchange) or ""))
                for a in assets if a.tradable}

    async def assets(self) -> Dict[str, str]:
        """
        Alpaca's tradable US equities with their names (which tell an ETF or a
        warrant from a stock). The executor loads the same list at start-up; this
        copy lets the scout run without it (the CLI, mock mode). Refreshed daily.
        """
        if self.asset_names and time.time() - self._assets_at < ASSETS_MAX_AGE_SECONDS:
            return self.asset_names
        got = await self._run("Alpaca assets", self._assets_sync)
        if got:
            self.asset_names = {s: n for s, (n, _) in got.items()}
            self.asset_exchange = {s: e for s, (_, e) in got.items()}
            self._assets_at = time.time()
            self._mark("Alpaca assets", True, "tradable US-listed equities", len(got))
        return self.asset_names

    def _screener_sync(self) -> Dict[str, List[str]]:
        from alpaca.data.requests import MostActivesRequest, MarketMoversRequest
        screener, _, _ = self._clients()
        out: Dict[str, List[str]] = {}
        for m in screener.get_most_actives(MostActivesRequest(top=100)).most_actives:
            out.setdefault(m.symbol, []).append("Most active")
        for m in screener.get_market_movers(MarketMoversRequest(top=50)).gainers:
            out.setdefault(m.symbol, []).append("Top gainer")
        return out

    async def screener(self) -> Dict[str, List[str]]:
        got = await self._run("Alpaca screener", self._screener_sync) or {}
        if got:
            self._mark("Alpaca screener", True, "most active + top gainers", len(got))
        return got

    def _news_sync(self, hours: float, limit: int) -> List[Any]:
        from alpaca.data.requests import NewsRequest
        _, news, _ = self._clients()
        start = datetime.now(timezone.utc) - timedelta(hours=hours)
        return news.get_news(NewsRequest(start=start, limit=limit, sort="desc",
                                         exclude_contentless=False)).data.get("news", [])

    async def news(self) -> Dict[str, List[Dict[str, Any]]]:
        """symbol -> its headlines of the lookback window, newest first."""
        items = await self._run("Alpaca news", self._news_sync,
                                settings.SCOUT_NEWS_LOOKBACK_HOURS, settings.SCOUT_NEWS_MAX_ITEMS)
        if items is None:
            return {}
        out: Dict[str, List[Dict[str, Any]]] = {}
        for it in items:
            syms = [s.upper() for s in (getattr(it, "symbols", None) or [])]
            # A headline tagged with many tickers (a market wrap) is about none of them.
            if not syms or len(syms) > 5:
                continue
            created = getattr(it, "created_at", None)
            row = {"id": str(getattr(it, "id", "")), "headline": (it.headline or "").strip(),
                   "summary": (getattr(it, "summary", "") or "").strip(),
                   "source": (getattr(it, "source", "") or "").strip(), "symbols": syms,
                   "at": created.timestamp() if created else time.time()}
            for s in syms:
                out.setdefault(s, []).append(row)
        self._mark("Alpaca news", True, f"{len(items)} headlines, last {settings.SCOUT_NEWS_LOOKBACK_HOURS:.0f}h",
                   len(out))
        return out

    def _snapshots_sync(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        from alpaca.data.requests import StockSnapshotRequest
        _, _, data = self._clients()
        out: Dict[str, Dict[str, Any]] = {}
        for i in range(0, len(symbols), 100):
            try:
                snaps = data.get_stock_snapshot(StockSnapshotRequest(symbol_or_symbols=symbols[i:i + 100]))
            except Exception as e:
                logger.warning("Snapshot batch failed: %s", e)
                continue
            for sym, s in (snaps or {}).items():
                flat = flatten_snapshot(s)
                if flat:
                    out[sym] = flat
        return out

    async def snapshots(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        got = await self._run("Alpaca snapshots", self._snapshots_sync, sorted(set(symbols))) or {}
        if got:
            self._mark("Alpaca snapshots", True, "", len(got))
        return got

    # ------------------------------------------------------------------
    # Public discussion
    # ------------------------------------------------------------------

    @staticmethod
    def _get_json_sync(url: str) -> Any:
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.load(r)

    async def _get_json(self, url: str) -> Any:
        return await asyncio.get_running_loop().run_in_executor(None, self._get_json_sync, url)

    async def reddit(self) -> Dict[str, Dict[str, Any]]:
        """symbol -> {mentions, mentions_24h_ago, upvotes, rank} from ApeWisdom."""
        out: Dict[str, Dict[str, Any]] = {}
        try:
            for page in range(1, settings.SCOUT_REDDIT_PAGES + 1):
                data = await self._get_json(APEWISDOM_URL.format(page=page))
                for r in data.get("results", []):
                    sym = str(r.get("ticker", "")).upper()
                    if sym and sym not in out:
                        out[sym] = {"mentions": int(r.get("mentions") or 0),
                                    "mentions_24h_ago": int(r.get("mentions_24h_ago") or 0),
                                    "upvotes": int(r.get("upvotes") or 0),
                                    "rank": int(r.get("rank") or 0)}
                if page >= int(data.get("pages") or 1):
                    break
        except Exception as e:
            self._mark("Reddit (ApeWisdom)", False, f"{type(e).__name__}: {e}", len(out))
            return out
        self._mark("Reddit (ApeWisdom)", True, "mentions across stock subreddits", len(out))
        return out

    async def stocktwits(self) -> Dict[str, int]:
        """symbol -> 1-based place on StockTwits' trending list (crypto dropped)."""
        try:
            data = await self._get_json(STOCKTWITS_TRENDING_URL)
        except Exception as e:
            self._mark("StockTwits trending", False, f"{type(e).__name__}: {e}")
            return {}
        syms = [str(s.get("symbol", "")).upper() for s in data.get("symbols", [])]
        out = {s: i + 1 for i, s in enumerate(x for x in syms if x and "." not in x)}
        self._mark("StockTwits trending", True, "", len(out))
        return out


def flatten_snapshot(s: Any) -> Optional[Dict[str, Any]]:
    """The fields the scout reads from an Alpaca Snapshot, or None without a price."""
    trade, minute = getattr(s, "latest_trade", None), getattr(s, "minute_bar", None)
    price = float(trade.price) if trade and trade.price else (float(minute.close) if minute else None)
    day, prev = getattr(s, "daily_bar", None), getattr(s, "previous_daily_bar", None)
    if not price or day is None:
        return None
    ts = getattr(trade, "timestamp", None) if trade else getattr(minute, "timestamp", None)
    return {
        "price": price,
        "price_date": ts.astimezone(_NY).strftime("%Y-%m-%d") if ts else None,
        "day_date": day.timestamp.astimezone(_NY).strftime("%Y-%m-%d"),
        "day_open": float(day.open), "day_close": float(day.close), "day_volume": float(day.volume or 0),
        "prev_close": float(prev.close) if prev else None,
        "prev_volume": float(prev.volume or 0) if prev else None,
    }


sources = Sources()
