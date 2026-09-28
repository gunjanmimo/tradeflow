"""
Real financial news ingestion via the Alpaca News API.

Replaces the previous module, which replayed eight hardcoded headlines on a random
timer. That had two fatal properties: the "news" was fiction, and replaying the
same bullish string every 15-30s permanently pinned a symbol's sentiment high, so
the entry gate was effectively always open.

This feed uses the Alpaca keys already configured for trading -- no extra API key.
Every headline is real, deduplicated by Alpaca's news id, and scored by Laya once.
When the feed is unavailable, sentiment simply goes stale and entries stop: the
system fails CLOSED rather than inventing a bullish signal.
"""
import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set

from core.config import settings
from core.state import state, ScoredHeadline, is_crypto_symbol
from sentiment.router import sentiment_service

logger = logging.getLogger("tradeflow.news")

# Alpaca's news endpoint indexes equities. Crypto headlines are keyed to the base
# asset name, so map the pair to terms that appear in real coverage.
CRYPTO_NEWS_KEYS = {
    "BTC/USD": ("BTCUSD", "bitcoin"),
    "ETH/USD": ("ETHUSD", "ethereum"),
    "SOL/USD": ("SOLUSD", "solana"),
    "DOGE/USD": ("DOGEUSD", "dogecoin"),
    "XRP/USD": ("XRPUSD", "xrp"),
    "ADA/USD": ("ADAUSD", "cardano"),
    "AVAX/USD": ("AVAXUSD", "avalanche"),
    "LINK/USD": ("LINKUSD", "chainlink"),
    "LTC/USD": ("LTCUSD", "litecoin"),
    "BCH/USD": ("BCHUSD", "bitcoin cash"),
    "UNI/USD": ("UNIUSD", "uniswap"),
    "SHIB/USD": ("SHIBUSD", "shiba"),
    "TRX/USD": ("TRXUSD", "tron"),
    "XLM/USD": ("XLMUSD", "stellar"),
    "HBAR/USD": ("HBARUSD", "hedera"),
    "NEAR/USD": ("NEARUSD", "near protocol"),
    "SUI/USD": ("SUIUSD", "sui"),
    "BNB/USD": ("BNBUSD", "binance coin"),
}


class NewsFeedManager:
    """
    Polls Alpaca for real headlines on the active watchlist and scores them
    through Laya, feeding state.sentiment_history for aggregation.
    """

    def __init__(self):
        self._running = False
        self._task = None
        self.client = None
        self.is_live = False
        self.last_poll_at: Optional[float] = None
        self.last_error: Optional[str] = None
        self.headlines_ingested = 0
        self.polls_completed = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self):
        self._running = True
        self._init_client()
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()

    def _init_client(self):
        if settings.ALPACA_API_KEY.startswith("PK_PLACEHOLDER") or not settings.ALPACA_API_KEY:
            self.is_live = False
            self.last_error = "Alpaca keys are placeholders"
            logger.warning(
                "No Alpaca credentials: real news feed disabled. Sentiment will "
                "remain neutral and no entries will be authorised."
            )
            state.log_event(
                "NEWS_DISABLED",
                "Real news feed unavailable (no Alpaca keys). Sentiment stays neutral; "
                "entries are blocked rather than run on fabricated signals."
            )
            return
        try:
            from alpaca.data.historical.news import NewsClient
            self.client = NewsClient(settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY)
            self.is_live = True
            logger.info("Alpaca real news feed initialised.")
            state.log_event("NEWS", "Alpaca real news feed connected.")
        except Exception as e:
            self.is_live = False
            self.last_error = str(e)
            logger.error(f"Could not initialise Alpaca news client: {e}")

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def _equity_symbols(self) -> List[str]:
        return [s for s in self._tracked() if not is_crypto_symbol(s)]

    @staticmethod
    def _wanted(symbol: str) -> bool:
        """
        Whether this symbol's news is worth fetching and scoring. A market or
        symbol switched off in the UI is skipped, so a paused crypto market costs
        no Laya time. A held position is always kept: its sentinel still reads
        sentiment to manage the exit.
        """
        from core.market_filter import market_filter
        return symbol in state.active_positions or market_filter.entry_block_reason(symbol) is None

    @classmethod
    def _tracked(cls) -> Set[str]:
        """
        Watchlist plus the top discovery candidates. Candidates get their public
        news sentiment scored by Laya so they can be judged before promotion;
        they are never traded unless promoted onto the watchlist.
        """
        from engine.discovery import discovery
        return {s for s in set(state.watchlist) | set(discovery.sentiment_symbols())
                if cls._wanted(s)}

    def _crypto_symbols(self) -> List[str]:
        return [s for s in state.watchlist if is_crypto_symbol(s) and self._wanted(s)]

    def _fetch_sync(self, symbols_csv: str, start: datetime, limit: int = 50) -> List:
        """Blocking Alpaca call, run in the executor so the tick loop never waits."""
        from alpaca.data.requests import NewsRequest
        req = NewsRequest(
            symbols=symbols_csv,
            start=start,
            limit=limit,
            include_content=False,
            exclude_contentless=True,
            sort="desc",
        )
        return self.client.get_news(req).data.get("news", [])

    async def poll_once(self) -> int:
        """
        One ingestion pass. Returns the number of NEW headlines scored.
        Safe to call directly (the scheduler's morning routine does).
        """
        if not self.is_live or self.client is None:
            return 0

        loop = asyncio.get_running_loop()
        start = datetime.now(timezone.utc) - timedelta(hours=settings.NEWS_LOOKBACK_HOURS)
        new_count = 0

        # Equities: query in batches to stay inside request limits.
        equities = self._equity_symbols()
        batches: List[List[str]] = [
            equities[i:i + settings.NEWS_BATCH_SYMBOLS]
            for i in range(0, len(equities), settings.NEWS_BATCH_SYMBOLS)
        ]
        # Crypto: Alpaca indexes these under the concatenated pair.
        crypto = self._crypto_symbols()
        crypto_keys = [CRYPTO_NEWS_KEYS[s][0] for s in crypto if s in CRYPTO_NEWS_KEYS]
        for i in range(0, len(crypto_keys), settings.NEWS_BATCH_SYMBOLS):
            batches.append(crypto_keys[i:i + settings.NEWS_BATCH_SYMBOLS])

        for batch in batches:
            if not batch:
                continue
            try:
                items = await loop.run_in_executor(
                    None, self._fetch_sync, ",".join(batch), start, 50
                )
            except Exception as e:
                self.last_error = str(e)
                logger.warning(f"News fetch failed for {batch[:3]}...: {e}")
                continue

            for item in items:
                new_count += await self._ingest_item(item)

        self.polls_completed += 1
        self.last_poll_at = time.time()
        if new_count:
            self.headlines_ingested += new_count
            state.log_event("NEWS", f"Ingested {new_count} new real headlines from Alpaca.")
        return new_count

    async def _ingest_item(self, item) -> int:
        """Scores one news item against each tracked symbol it mentions."""
        news_id = str(getattr(item, "id", "") or "")
        if not news_id or state.is_news_seen(news_id):
            return 0

        headline = (getattr(item, "headline", "") or "").strip()
        if not headline:
            return 0
        summary = (getattr(item, "summary", "") or "").strip()
        source = (getattr(item, "source", "") or "").strip()
        created = getattr(item, "created_at", None)
        published_at = created.timestamp() if created else time.time()

        # Map Alpaca's symbols back onto our watchlist spelling
        raw_syms = list(getattr(item, "symbols", []) or [])
        targets: Set[str] = set()
        tracked = self._tracked()
        for rs in raw_syms:
            rs_u = rs.upper()
            if rs_u in tracked:
                targets.add(rs_u)
                continue
            for pair, (key, _name) in CRYPTO_NEWS_KEYS.items():
                if rs_u == key and pair in state.watchlist and self._wanted(pair):
                    targets.add(pair)

        if not targets:
            return 0

        state.mark_news_seen(news_id)

        # Laya reads the headline plus summary: more context inside its 512-token
        # budget yields a better-grounded reading than a headline alone.
        text = f"{headline}. {summary}" if summary else headline
        text = text[:1200]

        scored = 0
        for sym in targets:
            try:
                rec = await sentiment_service.score_headline(sym, text, persist=False)
            except Exception as e:
                logger.error(f"Sentiment scoring failed for {sym}: {e}")
                continue
            state.record_headline(ScoredHeadline(
                news_id=f"{news_id}:{sym}",
                symbol=sym,
                headline=headline,
                pos_prob=rec.pos_prob,
                neg_prob=rec.neg_prob,
                neutral_prob=max(0.0, 1.0 - rec.pos_prob - rec.neg_prob),
                source=source,
                published_at=published_at,
            ))
            scored += 1
        return 1 if scored else 0

    def status(self) -> Dict:
        return {
            "is_live": self.is_live,
            "last_poll_at": self.last_poll_at,
            "seconds_since_poll": round(time.time() - self.last_poll_at, 1) if self.last_poll_at else None,
            "polls_completed": self.polls_completed,
            "headlines_ingested": self.headlines_ingested,
            "symbols_with_news": len(state.sentiment_history),
            "last_error": self.last_error,
        }

    # ------------------------------------------------------------------
    async def _poll_loop(self):
        if self.is_live:
            try:
                n = await self.poll_once()
                logger.info(f"Initial news ingestion: {n} headlines.")
            except Exception as e:
                logger.error(f"Initial news poll failed: {e}")

        while self._running:
            try:
                await asyncio.sleep(settings.NEWS_POLL_SECONDS)
                await self.poll_once()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in news poll loop: {e}")
                await asyncio.sleep(10)


news_feed = NewsFeedManager()
