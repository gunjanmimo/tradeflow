"""
The Scout agent: runs the hourly ranking and hands its picks to the watchlist.

Picks and the watchlist
  - the top SCOUT_TOP_N scoring at least SCOUT_MIN_SCORE become picks and go
    on the watchlist (streamed, minute bars backfilled for the watcher)
  - a pick stays one while it ranks within SCOUT_KEEP_RANK, or while it is held
  - a dropped pick leaves the watchlist only if the scout put it there: stocks
    the user watches are never taken off
  - a pick the user takes off the watchlist is not picked again that day

Every ranking is appended to data/scout/rankings.jsonl with the pool's prices,
so `python -m scout scorecard` can tell whether the top of the list went on to
beat the rest.
"""
import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from core.config import settings
from core.state import state, ny_date
from engine.agent import Agent
from scout import ranker
from scout.features import daily_features, today_features
from scout.exchanges import exchanges, issuer_is_foreign
from scout.sources import sources

logger = logging.getLogger("tradeflow.scout")

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "scout")
_PATH = os.path.join(_DATA_DIR, "rankings.jsonl")
_NY = ZoneInfo("America/New_York")
_US_TICKER = re.compile(r"^[A-Z]{1,5}(\.[A-Z])?$")
LOGGED_ROWS = 25
TONE_CACHE_SECONDS = 2 * 86400.0


def session_fraction(now: Optional[datetime] = None) -> Optional[float]:
    """Share of the regular session elapsed (0..1), or None outside it."""
    from core.market_hours import us_session, REGULAR
    now_ny = (now or datetime.now(_NY)).astimezone(_NY)
    if us_session(now_ny) != REGULAR:
        return None
    return min(1.0, max(0.0, (now_ny.hour * 60 + now_ny.minute - 570) / 390.0))


class ScoutAgent(Agent):
    name = "Scout"
    role = "Ranks the stocks worth watching today: past performance, news, public discussion"

    def __init__(self):
        super().__init__()
        self.ranking: List[Dict[str, Any]] = []
        self.excluded: List[Dict[str, Any]] = []
        self.pool_size = 0
        self.ranked_at = 0.0
        # symbol -> {rank, score, picked_at, pick_price, row}
        self.picks: Dict[str, Dict[str, Any]] = {}
        self.added: set = set()                   # picks the scout itself put on the watchlist
        self._streamed: set = set()               # added picks already subscribed to the stream
        self.dismissed: Dict[str, str] = {}       # symbol -> NY date the user removed it
        self._tone: Dict[str, tuple] = {}         # "news_id:symbol" -> (pos, neg, scored_at)
        self._lock = asyncio.Lock()

    @property
    def interval(self) -> float:
        return settings.SCOUT_INTERVAL_SECONDS

    @property
    def enabled(self) -> bool:
        return settings.SCOUT_ENABLED

    def card(self) -> Dict[str, Any]:
        c = super().card()
        # An hourly agent is not stalled between runs.
        if c["state"] == "stalled" and time.time() - (self.last_at or 0) < self.interval * 1.5:
            c["state"] = "running"
        return c

    # ------------------------------------------------------------------
    # One ranking
    # ------------------------------------------------------------------

    async def step(self):
        async with self._lock:
            ranked = await self.rank_now()
            self._select(ranked)
            added = [s for s in self.picks if s in self.added and s not in self._streamed]
            if added:
                await self._start_watching(added)
            self.summary = (f"{len(self.ranking)} ranked of {self.pool_size}; picks: "
                            + (", ".join(sorted(self.picks, key=lambda s: self.picks[s]["rank"])) or "none"))

    async def rank_now(self, score_tone: bool = True, log: bool = True) -> List[Dict[str, Any]]:
        """Gathers, filters, scores. Returns the ranked rows (also kept on self)."""
        from feeds.daily_bars import daily_bars

        await sources.assets()                    # names and exchanges, refreshed daily
        screener, news, reddit, stocktwits, _ = await asyncio.gather(
            sources.screener(), sources.news(), sources.reddit(), sources.stocktwits(), exchanges.listen())
        home = await self._map_exchanges()
        origins = self._pool(screener, news, reddit, stocktwits, home)

        snaps = await sources.snapshots(list(origins))
        # Cheap price floor before paying for 14 months of daily bars per symbol.
        pool = [s for s in origins if s not in snaps or snaps[s]["price"] >= settings.SCOUT_MIN_PRICE]
        cheap = [{"symbol": s, "why": f"price ${snaps[s]['price']:,.2f} under ${settings.SCOUT_MIN_PRICE:,.0f}",
                  "origins": sorted(origins[s])} for s in origins if s not in pool]
        # Foreign names in the pool that no exchange board placed (ARM from the US
        # screener): their country from SEC EDGAR, remembered across restarts.
        from core.universe import universe, country_from_name
        todo = [s for s in pool if s not in home and universe.needs_lookup(s)
                and country_from_name(self._name(s))]
        if todo:
            await universe.enrich(todo, settings.SEC_USER_AGENT, limit=30)
        await daily_bars.ensure(pool)

        today, frac = ny_date(), session_fraction()
        cands = []
        for sym in pool:
            closes = daily_bars.closes(sym)
            hlv = daily_bars.hlv.get(sym)
            daily = daily_features(closes, *hlv) if closes is not None and hlv else (
                daily_features(closes) if closes is not None else None)
            snap = snaps.get(sym)
            t = today_features(snap, today, frac) if snap else None
            if t is not None:
                t["price"] = snap["price"]
            name = self._name(sym)
            h = home.get(sym)
            country, region = self._country(sym, name, h)
            cands.append({
                "symbol": sym, "name": name, "origins": sorted(origins[sym]), "home": h,
                "country": country, "region": region, "exchange": sources.asset_exchange.get(sym),
                "daily": daily, "dollar_volume": daily_bars.dollar_volume.get(sym, 0.0),
                "today": t, "listed": self._listed(sym),
                "news": self._news_evidence(sym, news.get(sym, [])),
                "reddit": reddit.get(sym), "stocktwits_rank": stocktwits.get(sym),
            })

        out = ranker.rank(cands, settings.SCOUT_MIN_PRICE, settings.SCOUT_MIN_DOLLAR_VOLUME)
        if score_tone:
            by_sym = {c["symbol"]: c for c in cands}
            shortlist = [r["symbol"] for r in out["ranked"][:settings.SCOUT_TONE_TOP_N]]
            await self._score_tone({s: news.get(s, []) for s in shortlist}, by_sym)
            out = ranker.rank(cands, settings.SCOUT_MIN_PRICE, settings.SCOUT_MIN_DOLLAR_VOLUME)

        self.ranking, self.excluded = out["ranked"], out["excluded"] + cheap
        self.pool_size, self.ranked_at = len(cands), time.time()
        if log:
            self._log()
        return self.ranking

    async def _map_exchanges(self) -> Dict[str, Dict[str, Any]]:
        """
        Maps each exchange's hot stocks to US lines. Name matches whose US issuer is
        not yet known to be foreign are checked against SEC EDGAR (remembered across
        restarts), then mapped again, so a same-named US company never slips in.
        """
        from core.universe import universe
        home = exchanges.map_boards(sources.asset_names, sources.asset_exchange, issuer_is_foreign)
        todo = [s for s in exchanges.unverified if universe.needs_lookup(s)]
        if todo and await universe.enrich(todo, settings.SEC_USER_AGENT, limit=60):
            home = exchanges.map_boards(sources.asset_names, sources.asset_exchange, issuer_is_foreign)
        return home

    @staticmethod
    def _country(sym: str, name: str, home: Optional[Dict[str, Any]] = None):
        """(country, region): the exchange it is hot on, else curated/SEC data, else the Alpaca name."""
        from core.universe import universe, country_from_name, region_of
        if home:
            return home["country"], region_of(home["country"])
        m = universe.classify(sym)
        if m.source != "unknown":
            return m.country, m.region
        hint = country_from_name(name)
        return (hint, region_of(hint)) if hint else ("US", "US")

    def _pool(self, screener, news, reddit, stocktwits, home=None) -> Dict[str, set]:
        home = home or {}
        origins: Dict[str, set] = {}

        def add(sym: str, origin: str):
            sym = sym.upper()
            if _US_TICKER.match(sym) and self._listed(sym) is not False:
                origins.setdefault(sym, set()).add(origin)

        for sym, tags in screener.items():
            for t in tags:
                add(sym, t)
        for sym in stocktwits:
            add(sym, "StockTwits trending")
        for sym in state.watchlist:
            add(sym, "Watchlist")
        for sym in sorted(reddit, key=lambda s: -reddit[s]["mentions"])[:200]:
            add(sym, "Reddit")
        for sym in sorted(news, key=lambda s: -len(news[s])):
            add(sym, "News")
        for sym, h in home.items():
            add(sym, f"Hot in {h['exchange_label']}")
        # Over the cap, the sources that say most about today go first.
        if len(origins) > settings.SCOUT_MAX_POOL:
            prio = lambda o: (0 if o == "Watchlist" else 1 if o.startswith("Hot in") else
                              ["Most active", "Top gainer", "StockTwits trending", "Reddit", "News"].index(o) + 2)
            keyed = sorted(origins, key=lambda s: min(prio(o) for o in origins[s]))
            origins = {s: origins[s] for s in keyed[:settings.SCOUT_MAX_POOL]}
        return origins

    @staticmethod
    def _listed(sym: str) -> Optional[bool]:
        from engine.executor import executor
        if executor.is_connected and not executor.is_mock_mode and executor.alpaca_tradable_symbols:
            return sym in executor.alpaca_tradable_symbols
        if sources.asset_names:
            return sym in sources.asset_names
        return None

    @staticmethod
    def _name(sym: str) -> str:
        from engine.executor import executor
        from core.universe import universe
        name = executor.alpaca_asset_names.get(sym) or sources.asset_names.get(sym)
        if name:
            return name
        m = universe.classify(sym)
        return m.name if m.source == "curated" else ""

    # ------------------------------------------------------------------
    # News tone
    # ------------------------------------------------------------------

    def _news_evidence(self, sym: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Headline count, and the tone of whichever of them are already scored."""
        if not items:
            return {"n": 0, "tone": None, "n_scored": 0, "headline": None}
        sym_tones = [0.5 + (t[0] - t[1]) / 2 for it in items[:settings.SCOUT_TONE_HEADLINES]
                     for t in [self._tone.get(f"{it['id']}:{sym}")] if t]
        return {"n": len(items), "tone": sum(sym_tones) / len(sym_tones) if sym_tones else None,
                "n_scored": len(sym_tones), "headline": items[0]["headline"][:160]}

    async def _score_tone(self, news_by_sym: Dict[str, List[Dict[str, Any]]], cands: Dict[str, Dict]):
        """Scores the newest headlines of the shortlist (cached), then refreshes their evidence."""
        from feeds.news_log import news_log
        from sentiment.router import sentiment_service
        now = time.time()
        self._tone = {k: v for k, v in self._tone.items() if now - v[2] < TONE_CACHE_SECONDS}
        for sym, items in news_by_sym.items():
            scored = []
            for it in items[:settings.SCOUT_TONE_HEADLINES]:
                key = f"{it['id']}:{sym}"
                if key not in self._tone:
                    text = (f"{it['headline']}. {it['summary']}" if it["summary"] else it["headline"])[:1200]
                    news_log.ingest(it["id"], it["headline"], it["summary"], it.get("source", ""),
                                    it.get("symbols") or [sym], it["at"], "scout", [sym])
                    t0 = time.perf_counter()
                    try:
                        rec = await sentiment_service.score_headline(sym, text, persist=False)
                    except Exception as e:
                        logger.warning("Tone scoring failed for %s: %s", sym, e)
                        news_log.failed(it["id"], "scout", sym, str(e))
                        continue
                    self._tone[key] = (float(rec.pos_prob), float(rec.neg_prob), now)
                    news_log.scored(it["id"], "scout", sym, rec.pos_prob, rec.neg_prob, sentiment_service.active,
                                    (time.perf_counter() - t0) * 1000)
                pos, neg, _ = self._tone[key]
                scored.append(0.5 + (pos - neg) / 2)
            if sym in cands and items:
                cands[sym]["news"] = {"n": len(items), "tone": sum(scored) / len(scored) if scored else None,
                                      "n_scored": len(scored), "headline": items[0]["headline"][:160]}

    # ------------------------------------------------------------------
    # Picks -> watchlist
    # ------------------------------------------------------------------

    def is_pick(self, symbol: str) -> bool:
        return symbol in self.picks

    def _select(self, ranked: List[Dict[str, Any]]):
        from core.market_filter import market_filter
        from engine.executor import executor
        now, today = time.time(), ny_date()
        self.dismissed = {s: d for s, d in self.dismissed.items() if d == today}
        by_sym = {r["symbol"]: r for r in ranked}

        ok = lambda r: r["symbol"] not in self.dismissed and not market_filter.entry_block_reason(r["symbol"])
        top = [r for r in ranked[:settings.SCOUT_TOP_N] if r["score"] >= settings.SCOUT_MIN_SCORE and ok(r)]
        keep = {r["symbol"] for r in ranked[:settings.SCOUT_KEEP_RANK]
                if r["score"] >= settings.SCOUT_MIN_SCORE - 0.05}

        # Reserved picks per exchange: the best tradable stock each exchange is hot on,
        # unless the overall top already has one from it; at most SCOUT_INTL_MAX_PICKS.
        by_exchange: Dict[str, List[Dict[str, Any]]] = {}
        for r in ranked:
            if r.get("home_exchange"):
                by_exchange.setdefault(r["home_exchange"], []).append(r)
        chosen = {r["symbol"] for r in top}
        best = []
        for label, rows in by_exchange.items():
            for i, r in enumerate(rows):
                r["exchange_rank"] = i + 1
            have = sum(1 for r in top if r.get("home_exchange") == label)
            for r in rows:
                if have >= settings.SCOUT_EXCHANGE_SLOTS:
                    break
                if r["symbol"] not in chosen and r["score"] >= settings.SCOUT_EXCHANGE_MIN_SCORE and ok(r):
                    best.append(r)
                    chosen.add(r["symbol"])
                    have += 1
            keep |= {r["symbol"] for r in rows[:2 * settings.SCOUT_EXCHANGE_SLOTS + 1]
                     if r["score"] >= settings.SCOUT_EXCHANGE_MIN_SCORE - 0.05}
        regional = sorted(best, key=lambda r: r["score"], reverse=True)[:settings.SCOUT_INTL_MAX_PICKS]

        for r in top + regional:
            sym = r["symbol"]
            slot = "overall" if r in top else f"{r['home_exchange']} slot"
            if sym not in self.picks:
                self.picks[sym] = {"picked_at": now, "pick_price": r["price"]}
                if sym not in state.watchlist:
                    state.watchlist.add(sym)
                    self.added.add(sym)
                where = f"{r.get('country')}, " if r.get("country") not in (None, "US") else ""
                state.log_event("SCOUT", f"Picked {sym} ({where}#{r['rank']}"
                                         + (f", best of {r['home_exchange']}" if slot != "overall" else "")
                                         + f", score {r['score']:.2f}): " + "; ".join(r["reasons"][:3]))
                self.act(sym, "picked", f"#{r['rank']} ({slot}), score {r['score']:.2f}")
            self.picks[sym]["slot"] = slot

        for sym in list(self.picks):
            row = by_sym.get(sym)
            if row:
                self.picks[sym].update(rank=row["rank"], score=row["score"], row=row,
                                       country=row.get("country"), region=row.get("region"),
                                       home_exchange=row.get("home_exchange"))
            held = sym in state.active_positions or sym in executor.pending_orders
            if sym in keep or held:
                if not row:
                    self.picks[sym].update(rank=None, row=self.picks[sym].get("row"))
                continue
            self._drop(sym, f"fell to #{row['rank']}" if row else "no longer ranked")

    def _drop(self, sym: str, why: str):
        self.picks.pop(sym, None)
        if sym in self.added:
            self.added.discard(sym)
            self._streamed.discard(sym)
            if sym in state.watchlist and sym not in state.active_positions:
                state.watchlist.discard(sym)
        state.log_event("SCOUT", f"Dropped pick {sym}: {why}.")
        self.act(sym, "dropped", why)

    async def _start_watching(self, symbols: List[str]):
        from core.minute_bars import minute_bars
        from feeds.alpaca_stream import market_stream
        from feeds.spreads import spreads
        for sym in symbols:
            await market_stream.ensure_stock_subscription(sym)
            self._streamed.add(sym)
        asyncio.create_task(minute_bars.ensure(symbols))
        asyncio.create_task(spreads.refresh(symbols))      # their real spread now, not in 5 minutes

    def on_unwatched(self, symbol: str):
        """The user took a symbol off the watchlist: it is not picked again today."""
        sym = symbol.upper().strip()
        if sym in self.picks:
            self.dismissed[sym] = ny_date()
            self.added.discard(sym)
            self._streamed.discard(sym)
            self.picks.pop(sym, None)

    # ------------------------------------------------------------------
    # Record and views
    # ------------------------------------------------------------------

    def _log(self):
        from core.market_hours import us_session
        slim = ("symbol", "rank", "score", "components", "price", "origins")
        rec = {
            "at": round(self.ranked_at, 1), "ny_date": ny_date(self.ranked_at), "session": us_session(),
            "top": [{k: r[k] for k in slim} for r in self.ranking[:LOGGED_ROWS]],
            "pool_prices": {r["symbol"]: r["price"] for r in self.ranking},
            "weights": ranker.WEIGHTS,
        }
        try:
            os.makedirs(_DATA_DIR, exist_ok=True)
            with open(_PATH, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError as e:
            logger.warning("Could not log the scout ranking: %s", e)

    def snapshot(self, limit: int = 30) -> Dict[str, Any]:
        reasons: Dict[str, int] = {}
        for e in self.excluded:
            key = e["why"].split(" (")[0].split(":")[0]
            reasons[key] = reasons.get(key, 0) + 1
        regions: Dict[str, Dict[str, Any]] = {}
        for r in self.ranking:
            g = regions.setdefault(r.get("region") or "US", {"ranked": 0, "picks": 0, "leaders": [], "countries": {}})
            g["ranked"] += 1
            g["countries"][r.get("country") or "US"] = g["countries"].get(r.get("country") or "US", 0) + 1
            if len(g["leaders"]) < 5:
                g["leaders"].append({"symbol": r["symbol"], "country": r.get("country"), "score": r["score"],
                                     "rank": r["rank"], "picked": r["symbol"] in self.picks})
        for p in self.picks.values():
            if p.get("region") in regions:
                regions[p["region"]]["picks"] += 1
        return {
            "regions": regions, "exchanges": exchanges.snapshot(self.picks, self.ranking, excluded=self.excluded),
            "enabled": settings.SCOUT_ENABLED, "trading": settings.SCOUT_TRADING,
            "ranked_at": self.ranked_at or None,
            "next_at": (self.last_at + self.interval) if self.last_at else None,
            "pool": self.pool_size, "eligible": len(self.ranking),
            "excluded_by": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
            "sources": {**sources.health, **{f"Exchange: {k}": v for k, v in exchanges.health.items()}},
            "weights": ranker.WEIGHTS,
            "picks": [{"symbol": s, **{k: v for k, v in p.items() if k != "row"},
                       "on_watchlist": s in state.watchlist, "held": s in state.active_positions}
                      for s, p in sorted(self.picks.items(), key=lambda kv: kv[1].get("rank") or 999)],
            "ranking": self.ranking[:limit],
            "error": self.error,
            "rules": {"top_n": settings.SCOUT_TOP_N, "keep_rank": settings.SCOUT_KEEP_RANK,
                      "min_score": settings.SCOUT_MIN_SCORE, "min_price": settings.SCOUT_MIN_PRICE,
                      "min_dollar_volume": settings.SCOUT_MIN_DOLLAR_VOLUME,
                      "interval_s": settings.SCOUT_INTERVAL_SECONDS,
                      "exchange_slots": settings.SCOUT_EXCHANGE_SLOTS, "intl_max_picks": settings.SCOUT_INTL_MAX_PICKS,
                      "exchange_min_score": settings.SCOUT_EXCHANGE_MIN_SCORE},
        }


scout = ScoutAgent()
