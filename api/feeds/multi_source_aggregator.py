"""
Multi-source conviction aggregator -- real sources only.

What was removed and why
------------------------
The previous version advertised four "pillars" of market intelligence. Two of them
(`fetch_etoro_popular_investors`, `fetch_dub_public_social_pies`) never made a
network call at all: they were bare `return [...]` statements of invented signals.
The other two fell back to hardcoded blocks whenever the network hiccuped. Every
single fabricated entry was BULLISH with conviction 0.72-0.94, so `consensus_score`
had a floor around 0.75 regardless of market conditions.

Worse, the synthesized thesis string was fed back into Laya for scoring. Laya read
the fabricated claim "CEO purchased $12M open-market shares" and correctly reported
bullish -- measured at 0.796 -- which cleared the entry gate. The engine was writing
a bullish story, reading it back, believing it, and buying.

Dub.app and Public.com are gone permanently: dub's social-investing app publishes no
API, and Public.com's is discretionary B2B licensing that exposes only your own
holdings, not creator portfolios. eToro's API is real and is wired below behind a key.

Design rule
-----------
Every pillar FAILS CLOSED. An unavailable source contributes zero weight and the
consensus is computed over whatever genuinely responded. If nothing responds,
consensus is None and the caller must treat the symbol as un-scored -- never as
neutral-but-tradeable, and never as bullish.
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import aiohttp

from core.config import settings
from core.state import state
from core.universe import universe

# eToro lists crypto bare; these are read as the USD pair even when not watched.
_CRYPTO_BASES = frozenset({
    "BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC", "BCH",
    "DOT", "SHIB", "UNI", "XLM", "TRX", "BNB",
})

logger = logging.getLogger("tradeflow.aggregator")


@dataclass
class SourceSignal:
    source_name: str
    symbol: str
    conviction_score: float      # 0.0 to 1.0
    sentiment: str               # "BULLISH", "BEARISH", "NEUTRAL"
    details: str
    is_real: bool = True         # provenance flag; only real data may size a trade
    observed_at: float = field(default_factory=time.time)


@dataclass
class AggregatedTrend:
    symbol: str
    consensus_score: Optional[float]   # None when no source responded
    sentiment_bias: str
    contributing_sources: tuple = ()
    total_weight: float = 0.0
    signals: Dict[str, SourceSignal] = field(default_factory=dict)
    overall_thesis: str = ""
    updated_at: float = field(default_factory=time.time)

    @property
    def is_tradeable(self) -> bool:
        """Consensus may only influence sizing when real sources actually backed it."""
        return self.consensus_score is not None and self.total_weight >= 0.30


# Source weights. These are normalised over whichever sources actually respond,
# so a missing source dilutes nobody -- it simply does not vote.
SOURCE_WEIGHTS = {
    "SEC Form 4": 0.45,     # hard regulatory filings: the highest-quality signal
    "eToro": 0.30,          # real copy-trading flows, when a key is configured
    "StockTwits": 0.25,     # retail chatter; noisy, lowest weight
}


class MultiSourceTrendAggregator:
    def __init__(self):
        self.aggregated_trends: Dict[str, AggregatedTrend] = {}
        self._running = False
        self._task = None
        self.source_health: Dict[str, Dict[str, Any]] = {}

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._continuous_aggregation_loop())
        logger.info("MultiSourceTrendAggregator started (real sources only).")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()

    def _mark_health(self, name: str, ok: bool, detail: str = "", count: int = 0):
        self.source_health[name] = {
            "available": ok,
            "detail": detail,
            "signals": count,
            "checked_at": time.time(),
        }

    # -----------------------------------------------------------------
    # Pillar 1: SEC Form 4 -- real insider open-market purchases
    # -----------------------------------------------------------------
    async def fetch_sec_form4_insiders(self) -> List[SourceSignal]:
        """
        Real Form 4 filings from SEC EDGAR.

        The previous implementation searched for `"(TICKER)"` inside the Atom entry
        title. Form 4 titles carry the issuer NAME, not the ticker, so that test
        essentially never matched and the code fell through to fabricated data.
        Here we read each filing's XML and take the issuer's tradingSymbol plus the
        transaction code, so only genuine open-market purchases (code "P") count.
        """
        signals: List[SourceSignal] = []
        headers = {"User-Agent": settings.SEC_USER_AGENT}
        index_url = (
            "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=4"
            "&dateb=&owner=only&count=60&output=atom"
        )
        try:
            timeout = aiohttp.ClientTimeout(total=12)
            async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
                async with session.get(index_url) as resp:
                    if resp.status != 200:
                        self._mark_health("SEC Form 4", False, f"index HTTP {resp.status}")
                        return []
                    atom = await resp.text()

                import xml.etree.ElementTree as ET
                ns = {"a": "http://www.w3.org/2005/Atom"}
                root = ET.fromstring(atom)
                links = []
                for entry in root.findall("a:entry", ns):
                    link_el = entry.find("a:link", ns)
                    href = link_el.get("href") if link_el is not None else None
                    # Each filing appears once per issuer and once per reporting
                    # owner, under different CIK paths; the accession number is shared.
                    if href and href.rsplit("/", 1)[-1] not in {l.rsplit("/", 1)[-1] for l in links}:
                        links.append(href)

                # Bounded fan-out: SEC asks for <=10 req/s; stay well under.
                sem = asyncio.Semaphore(4)

                async def load_filing(url: str):
                    async with sem:
                        try:
                            await asyncio.sleep(0.12)
                            return await self._parse_form4(session, url)
                        except Exception as e:
                            logger.debug(f"Form 4 parse skipped ({url}): {e}")
                            return None

                results = await asyncio.gather(
                    *[load_filing(u) for u in links[:settings.SEC_MAX_FILINGS]]
                )
                for r in results:
                    if r:
                        signals.append(r)

            self._mark_health("SEC Form 4", True, f"{len(links)} filings scanned", len(signals))
        except Exception as e:
            # Fail closed: no fabricated insider buys.
            self._mark_health("SEC Form 4", False, str(e)[:120])
            logger.warning(f"SEC EDGAR unavailable ({e}); pillar contributes nothing this cycle.")
            return []

        return signals

    async def _parse_form4(self, session, filing_url: str) -> Optional[SourceSignal]:
        """Extracts ticker + transaction intent from a single Form 4 XML document."""
        import re
        import xml.etree.ElementTree as ET

        async with session.get(filing_url) as resp:
            if resp.status != 200:
                return None
            page = await resp.text()

        # The index lists the filing twice: an XSL-rendered HTML view under
        # ".../xslF345X0N/..." and the raw XML. Only the raw one parses; taking
        # the first match (the rendered view) made this pillar silently return
        # nothing for every filing.
        raw = [h for h in re.findall(r'href="(/Archives/[^"]+\.xml)"', page)
               if "/xsl" not in h.lower()]
        if not raw:
            return None
        xml_url = "https://www.sec.gov" + raw[0]

        async with session.get(xml_url) as resp:
            if resp.status != 200:
                return None
            xml_text = await resp.text()

        root = ET.fromstring(xml_text)
        sym_el = root.find(".//issuerTradingSymbol")
        if sym_el is None or not (sym_el.text or "").strip():
            return None
        symbol = sym_el.text.strip().upper()

        # Every issuer is kept, not only watchlist symbols: an insider buying a
        # company we do not follow yet is exactly what discovery is for. Only
        # watchlist symbols ever size a trade; the rest feed the candidate pool.

        # Transaction codes: P = open-market purchase, S = sale, A = award/grant.
        # Awards are compensation, not conviction, so they are deliberately ignored.
        buy_value = 0.0
        sell_value = 0.0
        for txn in root.findall(".//nonDerivativeTransaction"):
            code_el = txn.find(".//transactionCode")
            code = (code_el.text or "").strip().upper() if code_el is not None else ""
            if code not in ("P", "S"):
                continue
            shares_el = txn.find(".//transactionShares/value")
            price_el = txn.find(".//transactionPricePerShare/value")
            try:
                shares = float(shares_el.text) if shares_el is not None and shares_el.text else 0.0
                price = float(price_el.text) if price_el is not None and price_el.text else 0.0
            except (TypeError, ValueError):
                continue
            value = shares * price
            if code == "P":
                buy_value += value
            else:
                sell_value += value

        if buy_value <= 0 and sell_value <= 0:
            return None

        owner_el = root.find(".//reportingOwnerId/rptOwnerName")
        owner = (owner_el.text or "insider").strip() if owner_el is not None else "insider"

        net = buy_value - sell_value
        if net > 0:
            # Scale conviction by transaction size; a $10M buy is not a $20k buy.
            conviction = min(0.95, 0.55 + min(net / 5_000_000.0, 1.0) * 0.40)
            return SourceSignal(
                "SEC Form 4", symbol, round(conviction, 3), "BULLISH",
                f"{owner} open-market PURCHASE of ${net:,.0f} (Form 4)",
            )
        conviction = min(0.95, 0.55 + min(-net / 5_000_000.0, 1.0) * 0.40)
        return SourceSignal(
            "SEC Form 4", symbol, round(conviction, 3), "BEARISH",
            f"{owner} net SALE of ${-net:,.0f} (Form 4)",
        )

    # -----------------------------------------------------------------
    # Pillar 2: eToro -- real copy-trading flows (requires an API key)
    # -----------------------------------------------------------------
    async def fetch_etoro_signals(self) -> List[SourceSignal]:
        """
        Real eToro Popular Investor conviction.

        Method: take the top-ranked Popular Investors by year-to-date gain, read
        each one's current asset allocation, and score a symbol by how many of them
        hold it and how heavily. That is genuine smart-money positioning, as opposed
        to the invented "held by 4 of top 5 investors" string this pillar used to
        return unconditionally without making any network call.

        Endpoints (both verified GRANTED on the configured credentials):
          GET /api/v2/portfolios/rankings                  -> the leaderboard
          GET /api/v2/portfolios/{username}/assets/history  -> that investor's holdings

        Note on holdings: the obvious endpoint for this,
        /api/v1/user-info/people/{u}/portfolio/live, returns 403 on these
        credentials. assets/history is granted and carries the same per-symbol
        allocation data, so it is used instead -- no extra permission needed.
        """
        if not (settings.ETORO_API_KEY and settings.ETORO_PRIVATE_KEY):
            self._mark_health("eToro", False, "ETORO_API_KEY / ETORO_PRIVATE_KEY not configured")
            return []

        signals: List[SourceSignal] = []
        try:
            timeout = aiohttp.ClientTimeout(total=20)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # 1. Leaderboard of Popular Investors
                rankings = await self._etoro_get(
                    session, "/api/v2/portfolios/rankings",
                    {
                        "period": settings.ETORO_RANKING_PERIOD,
                        "popularInvestor": "true",
                        "sort": "-gain",
                        "pageSize": settings.ETORO_TOP_INVESTORS,
                    },
                )
                if not rankings:
                    self._mark_health("eToro", False, "rankings request failed")
                    return []

                investors = rankings.get("results") or []
                if not investors:
                    self._mark_health("eToro", True, "rankings empty", 0)
                    return []

                # 2. Each investor's current allocation. Sequential with a small
                # delay: the quota is 60 req/60s and correctness matters more than
                # shaving a few seconds off a 15-minute cycle.
                holdings: Dict[str, float] = {}   # symbol -> summed invested weight
                holders: Dict[str, int] = {}      # symbol -> how many investors hold it
                counted = 0

                for inv in investors:
                    username = inv.get("username")
                    if not username:
                        continue
                    await asyncio.sleep(0.4)
                    data = await self._etoro_get(
                        session, f"/api/v2/portfolios/{username}/assets/history",
                        {"period": "CurrMonth", "count": 1},
                    )
                    if not data:
                        continue
                    results = data.get("results") or []
                    if not results:
                        continue
                    # Most recent dated snapshot == current allocation
                    latest = max(results, key=lambda r: r.get("date", ""))
                    counted += 1
                    for asset in latest.get("assets") or []:
                        sym = (asset.get("symbol") or "").strip().upper()
                        if not sym:
                            continue
                        mapped = self._map_etoro_symbol(sym)
                        if mapped is None:
                            continue
                        weight = float(asset.get("investedPct") or 0.0)
                        holdings[mapped] = holdings.get(mapped, 0.0) + weight
                        holders[mapped] = holders.get(mapped, 0) + 1

                if counted == 0:
                    self._mark_health("eToro", False, "no investor holdings readable")
                    return []

                # 3. Conviction = breadth (how many top investors hold it) blended
                # with depth (their average allocation to it).
                for sym, total_weight in holdings.items():
                    n = holders[sym]
                    breadth = n / counted
                    depth = total_weight / n
                    conviction = min(0.95, 0.40 + 0.40 * breadth + 0.30 * min(depth, 1.0))
                    signals.append(SourceSignal(
                        "eToro", sym, round(conviction, 3),
                        "BULLISH" if breadth >= 0.20 else "NEUTRAL",
                        f"Held by {n}/{counted} top Popular Investors "
                        f"(avg allocation {depth*100:.1f}%)",
                    ))

            self._mark_health("eToro", True,
                              f"{counted} investor portfolios read", len(signals))
        except Exception as e:
            self._mark_health("eToro", False, str(e)[:120])
            logger.warning(f"eToro API unavailable ({e}); pillar contributes nothing.")
            return []

        return signals

    async def _etoro_get(self, session, path: str,
                         params: Optional[Dict[str, Any]] = None) -> Optional[Dict]:
        """
        One authenticated eToro GET.

        X-Request-Id must be a FRESH uuid per request -- the API returns
        422 RequestIdRequired without it, and reusing one risks dedupe.
        """
        import uuid
        headers = {
            "x-api-key": settings.ETORO_API_KEY,
            "x-user-key": settings.ETORO_PRIVATE_KEY,
            "X-Request-Id": str(uuid.uuid4()),
            "Content-Type": "application/json",
        }
        url = f"{settings.ETORO_API_BASE}{path}"
        # aiohttp rejects non-string param values outright, so coerce everything.
        clean_params = {k: str(v) for k, v in (params or {}).items() if v is not None}
        try:
            async with session.get(url, headers=headers, params=clean_params) as resp:
                if resp.status == 403:
                    logger.debug(f"eToro 403 (insufficient read permission): {path}")
                    return None
                if resp.status != 200:
                    body = await resp.text()
                    logger.debug(f"eToro {resp.status} on {path}: {body[:160]}")
                    return None
                return await resp.json(content_type=None)
        except Exception as e:
            # Surfaced at warning level: a silent None here previously read as
            # "source unavailable" and hid genuine client-side bugs.
            logger.warning(f"eToro request error on {path}: {type(e).__name__}: {e}")
            return None

    @staticmethod
    def _map_etoro_symbol(etoro_sym: str) -> Optional[str]:
        """
        Maps an eToro instrument symbol onto the spelling we trade.

        Crypto is listed bare ("BTC"); we track pairs ("BTC/USD"). A foreign
        listing ("AZN.L") maps to its US ADR when one exists. Anything else is
        kept as-is: it is not tradable here, but it is still a discovery signal
        (a country ETF is offered as the proxy).
        """
        s = etoro_sym.upper().strip()
        if not s:
            return None
        pair = f"{s}/USD"
        if pair in state.watchlist or s in _CRYPTO_BASES:
            return pair
        if "." in s and s != "BRK.B":
            return universe.tradable_symbol(s) or s
        return s

    # -----------------------------------------------------------------
    # Pillar 3: StockTwits -- real sentiment ratio
    # -----------------------------------------------------------------
    async def fetch_stocktwits_sentiment(self) -> List[SourceSignal]:
        """
        Real StockTwits bull/bear message ratio. Cloudflare frequently blocks
        unauthenticated access; when it does, this contributes nothing rather than
        the old hardcoded "86% Bullish" block.
        """
        signals: List[SourceSignal] = []
        headers = {"User-Agent": "Mozilla/5.0 (compatible; TradeFlow/1.0)"}
        symbols = sorted(state.watchlist)[:settings.STOCKTWITS_MAX_SYMBOLS]
        # Public sentiment on the strongest discovery candidates too, so a
        # candidate is judged on crowd mood before anyone promotes it.
        from engine.discovery import discovery
        for sym in discovery.sentiment_symbols()[:settings.STOCKTWITS_MAX_CANDIDATES]:
            if sym not in symbols and "/" not in sym:
                symbols.append(sym)
        blocked = 0

        timeout = aiohttp.ClientTimeout(total=6)
        # One session for all requests, rather than one session per symbol.
        async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
            for sym in symbols:
                st_sym = sym.replace("/USD", ".X") if "/USD" in sym else sym
                url = f"https://api.stocktwits.com/api/2/streams/symbol/{st_sym}.json"
                try:
                    await asyncio.sleep(0.25)
                    async with session.get(url) as resp:
                        if resp.status != 200:
                            blocked += 1
                            continue
                        data = await resp.json()
                except Exception:
                    blocked += 1
                    continue

                messages = data.get("messages", [])
                bulls = sum(1 for m in messages
                            if (m.get("entities") or {}).get("sentiment", {}).get("basic") == "Bullish")
                bears = sum(1 for m in messages
                            if (m.get("entities") or {}).get("sentiment", {}).get("basic") == "Bearish")
                total = bulls + bears
                if total < settings.STOCKTWITS_MIN_MESSAGES:
                    continue
                ratio = bulls / total
                bias = "BULLISH" if ratio >= 0.65 else ("BEARISH" if ratio <= 0.35 else "NEUTRAL")
                signals.append(SourceSignal(
                    "StockTwits", sym, round(ratio, 3), bias,
                    f"{bulls} bullish vs {bears} bearish messages ({ratio*100:.0f}% bull)",
                ))

        ok = len(signals) > 0
        self._mark_health("StockTwits", ok,
                          f"{blocked}/{len(symbols)} requests blocked" if blocked else "ok",
                          len(signals))
        return signals

    # -----------------------------------------------------------------
    # Consensus
    # -----------------------------------------------------------------
    async def aggregate_all_sources(self):
        logger.info("Running conviction aggregator over real sources...")

        sec, etoro, stocktwits = await asyncio.gather(
            self.fetch_sec_form4_insiders(),
            self.fetch_etoro_signals(),
            self.fetch_stocktwits_sentiment(),
            return_exceptions=True,
        )
        collected: List[SourceSignal] = []
        for group in (sec, etoro, stocktwits):
            if isinstance(group, Exception):
                logger.warning(f"Source raised during aggregation: {group}")
                continue
            collected.extend(group)

        if not collected:
            state.log_event(
                "AGGREGATOR",
                "No real conviction sources responded this cycle. Consensus left "
                "un-scored; entries fall back to price/news evidence only."
            )
            return

        by_symbol: Dict[str, Dict[str, SourceSignal]] = {}
        for sig in collected:
            by_symbol.setdefault(sig.symbol, {})[sig.source_name] = sig

        for sym, sources in by_symbol.items():
            weighted = 0.0
            total_weight = 0.0
            for name, sig in sources.items():
                w = SOURCE_WEIGHTS.get(name, 0.0)
                if w <= 0:
                    continue
                # A bearish reading is inverted so the scale stays "bullishness".
                score = sig.conviction_score if sig.sentiment == "BULLISH" else (
                    1.0 - sig.conviction_score if sig.sentiment == "BEARISH" else 0.5
                )
                weighted += w * score
                total_weight += w

            if total_weight <= 0:
                continue
            consensus = round(weighted / total_weight, 3)

            if consensus >= 0.80:
                bias = "STRONG_BULL"
            elif consensus >= 0.65:
                bias = "BULL"
            elif consensus <= 0.20:
                bias = "STRONG_BEAR"
            elif consensus <= 0.35:
                bias = "BEAR"
            else:
                bias = "NEUTRAL"

            thesis = f"Consensus {consensus*100:.0f}% from {len(sources)} real source(s): " + " | ".join(
                f"{n} {s.sentiment} ({s.details[:60]})" for n, s in sources.items()
            )

            self.aggregated_trends[sym] = AggregatedTrend(
                symbol=sym,
                consensus_score=consensus,
                sentiment_bias=bias,
                contributing_sources=tuple(sorted(sources.keys())),
                total_weight=round(total_weight, 3),
                signals=dict(sources),
                overall_thesis=thesis,
            )

            # NOTE: the thesis string is deliberately NOT fed back into Laya.
            # Scoring our own generated summary and treating the result as market
            # sentiment was the circular-reasoning bug at the heart of the old
            # design. Laya scores real news text only; consensus is a separate,
            # independently-weighted input.

        available = [n for n, h in self.source_health.items() if h.get("available")]
        state.log_event(
            "AGGREGATOR",
            f"Consensus computed for {len(by_symbol)} symbols from real sources: "
            f"{', '.join(available) if available else 'none'}."
        )

    def get_consensus(self, symbol: str) -> Optional[float]:
        """
        Consensus for sizing, or None when no real source backed this symbol.
        Callers must handle None explicitly rather than defaulting to an optimistic
        value -- the old code defaulted to 0.70, which quietly inflated conviction.
        """
        trend = self.aggregated_trends.get(symbol)
        if trend is None or not trend.is_tradeable:
            return None
        if (time.time() - trend.updated_at) > settings.CONSENSUS_MAX_AGE_SECONDS:
            return None
        return trend.consensus_score

    def health(self) -> Dict[str, Any]:
        return {
            "sources": self.source_health,
            "symbols_scored": len(self.aggregated_trends),
            "real_sources_available": [
                n for n, h in self.source_health.items() if h.get("available")
            ],
        }

    async def _continuous_aggregation_loop(self):
        try:
            await self.aggregate_all_sources()
        except Exception as e:
            logger.error(f"Initial aggregation failed: {e}")
        while self._running:
            try:
                await asyncio.sleep(settings.AGGREGATOR_INTERVAL_SECONDS)
                await self.aggregate_all_sources()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in aggregation loop: {e}")
                await asyncio.sleep(60)


trend_aggregator = MultiSourceTrendAggregator()
