"""
Stock discovery: a scored candidate pool that is separate from the watchlist.

Where candidates come from
  - Smart money: SEC Form 4 insider trades and eToro top-investor holdings, for
    ANY symbol (the aggregator used to throw away everything off the watchlist,
    so it could re-score what we already followed but never discover anything).
  - Our own universe screen: GICS sector leaders, US-listed ADRs of UK, European,
    Chinese, Japanese and Indian companies, and sector/country ETFs.

How a candidate is scored (each component 0..1; missing components do not vote)
  smart money       30%  insider + copy-trader conviction (bearish counts against)
  public sentiment  20%  Laya-scored real news + StockTwits bull/bear ratio
  momentum          30%  12-1m momentum, 3m strength vs its sector ETF, 200d trend
  diversification   20%  headroom in its sleeves, under-target regions, and low
                         correlation with what is already held -- read from the
                         SAME caps the entry gate enforces, at the current dial
A candidate needs at least one evidence component (not just "fit") to be scored.

Nothing here trades. A candidate reaches the engine only when promoted onto the
watchlist -- by hand, or automatically when it ranks in the top
DISCOVERY_AUTO_TOP_N -- and even then every entry passes the strategy, the risk
guard and the diversification gate like any other symbol.
"""
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional

import numpy as np

from core.config import settings
from core.state import state, is_crypto_symbol
from core.universe import (
    universe, curated_symbols, benchmark_etfs, SECTOR_ETF, CRYPTO, DIVERSIFIED,
)
from engine.diversification import diversification, budget_base
from feeds.daily_bars import daily_bars

logger = logging.getLogger("tradeflow.discovery")

WEIGHTS = {"smart_money": 0.30, "public_sentiment": 0.20, "momentum": 0.30,
           "diversification": 0.20}
SMART_SOURCES = ("SEC Form 4", "eToro")


def _bullishness(sig) -> float:
    if sig.sentiment == "BULLISH":
        return sig.conviction_score
    if sig.sentiment == "BEARISH":
        return 1.0 - sig.conviction_score
    return 0.5


def _pct_rank(values: Dict[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    keys = list(values)
    arr = np.array([values[k] for k in keys])
    if len(arr) == 1:
        return {keys[0]: 0.5}
    order = arr.argsort().argsort()
    return {k: float(r / (len(arr) - 1)) for k, r in zip(keys, order)}


class DiscoveryService:
    def __init__(self):
        self.candidates: Dict[str, Dict[str, Any]] = {}
        self.sleeve_momentum: List[Dict[str, Any]] = []
        self._sentiment_symbols: List[str] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self.last_cycle_at: float = 0.0
        self.last_cycle_ms: float = 0.0
        self.last_error: Optional[str] = None
        self.risk_report: Dict[str, Any] = {}
        # Symbols this service put on the watchlist itself, as opposed to by hand.
        # Only these are ever taken off again automatically.
        self.auto_symbols: set[str] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self):
        if not settings.DISCOVERY_ENABLED:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Discovery service started.")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()

    async def _loop(self):
        while self._running:
            try:
                await self.run_cycle()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
                logger.error(f"Discovery cycle failed: {e}", exc_info=True)
            try:
                await asyncio.sleep(settings.DISCOVERY_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                break

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def sentiment_symbols(self) -> List[str]:
        """Top tradable candidates whose public sentiment is tracked."""
        return list(self._sentiment_symbols)

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": settings.DISCOVERY_ENABLED,
            "candidates": len(self.candidates),
            "scored": sum(1 for c in self.candidates.values() if c.get("score") is not None),
            "sentiment_tracked": len(self._sentiment_symbols),
            "last_cycle_at": self.last_cycle_at or None,
            "last_cycle_ms": round(self.last_cycle_ms, 1),
            "last_error": self.last_error,
            "auto_promote": settings.DISCOVERY_AUTO_PROMOTE,
            "auto_top_n": settings.DISCOVERY_AUTO_TOP_N,
            "auto_min_score": settings.DISCOVERY_AUTO_MIN_SCORE,
            "auto_symbols": sorted(self.auto_symbols),
            "daily_bars": daily_bars.status(),
        }

    # ------------------------------------------------------------------
    # One cycle
    # ------------------------------------------------------------------

    async def run_cycle(self):
        t0 = time.perf_counter()
        from feeds.multi_source_aggregator import trend_aggregator
        trends = dict(trend_aggregator.aggregated_trends)
        now = time.time()

        # 1. Pool: universe screen + every symbol any smart-money source mentioned.
        for sym in curated_symbols():
            self._touch(sym, "Universe screen", now)
        for sym in state.watchlist:
            self._touch(sym, "Watchlist", now)
        for sym, trend in trends.items():
            for src in trend.signals:
                self._touch(sym, src, now)

        # 2. Classify unknown US tickers (insider buys in names we never listed).
        unknown = [s for s in self.candidates if universe.needs_lookup(s)]
        if unknown:
            await universe.enrich(unknown, settings.SEC_USER_AGENT)

        # 3. Daily history for everything tradable + benchmarks + holdings.
        tradable = [s for s, c in self.candidates.items() if c["meta"]["tradable_on_alpaca"]]
        await daily_bars.ensure(set(tradable) | set(benchmark_etfs())
                                | set(state.active_positions))

        # 4. Correlations vs holdings, sleeve momentum, then scores.
        diversification.refresh_correlations(tradable)
        self._rank_sleeves()
        self._score(trends)
        # Watchlist selection is the Curator agent's job (engine/fleet.py), on its
        # own cadence; without the fleet running, discovery still does it here.
        from engine.fleet import fleet
        if not fleet.curator.running:
            await self._auto_select()
        self._prune(now)
        self.risk_report = diversification.portfolio_risk()

        self.last_cycle_at = time.time()
        self.last_cycle_ms = (time.perf_counter() - t0) * 1000
        self.last_error = None

    def _touch(self, sym: str, origin: str, now: float):
        c = self.candidates.get(sym)
        if c is None:
            if len(self.candidates) >= settings.DISCOVERY_MAX_CANDIDATES and origin != "Watchlist":
                return
            meta = universe.classify(sym)
            c = self.candidates[sym] = {
                "symbol": sym, "meta": meta.to_dict(), "origins": [],
                "first_seen": now, "last_seen": now, "status": "candidate",
            }
        if origin not in c["origins"]:
            c["origins"].append(origin)
        c["last_seen"] = now

    def _prune(self, now: float):
        for sym, c in list(self.candidates.items()):
            if sym in state.watchlist or sym in state.active_positions:
                continue
            if now - c["last_seen"] > settings.DISCOVERY_STALE_SECONDS:
                self.candidates.pop(sym, None)

    def _rank_sleeves(self):
        """Sector and country ETFs ranked by momentum: which sleeves lead right now."""
        rows = []
        for etf in benchmark_etfs():
            mom = daily_bars.momentum(etf)
            if not mom:
                continue
            m = universe.classify(etf)
            rows.append({
                "etf": etf,
                "sleeve": m.sector if m.sector != DIVERSIFIED else m.country,
                "kind": "sector" if m.sector != DIVERSIFIED else "country",
                "ret_3m_pct": round(mom["ret_3m"] * 100, 2),
                "mom_12_1_pct": round(mom["mom_12_1"] * 100, 2) if "mom_12_1" in mom else None,
                "above_200d": mom.get("above_200d"),
            })
        key = lambda r: r["mom_12_1_pct"] if r["mom_12_1_pct"] is not None else r["ret_3m_pct"]
        rows.sort(key=key, reverse=True)
        for i, r in enumerate(rows):
            r["rank"] = i + 1
        self.sleeve_momentum = rows

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score(self, trends):
        profile = state.risk_profile
        base = budget_base()
        std_position = base * profile.max_position_notional_pct / 100.0
        sleeves = diversification.sleeves()
        under_target_regions = {r["sleeve"] for r in sleeves["regions"] if r["under_target"]}
        defensive_short = (sleeves["invested_pct"] > 0
                           and sleeves["defensive_pct"] < profile.min_defensive_pct)
        held_sectors = {s["sleeve"] for s in sleeves["sectors"] if s["dollars"] > 0}

        # Momentum inputs, ranked cross-sectionally so stocks and ETFs compare fairly.
        mom_raw, rel_raw, trend_ok, momentum_detail = {}, {}, {}, {}
        for sym, c in self.candidates.items():
            if not c["meta"]["tradable_on_alpaca"]:
                continue
            m = daily_bars.momentum(sym)
            if not m:
                continue
            mom_raw[sym] = m.get("mom_12_1", m["ret_3m"])
            etf = SECTOR_ETF.get(c["meta"]["sector"])
            em = daily_bars.momentum(etf) if etf and etf != sym else None
            if em:
                rel_raw[sym] = m["ret_3m"] - em["ret_3m"]
            trend_ok[sym] = m.get("above_200d")
            momentum_detail[sym] = {
                "ret_3m_pct": round(m["ret_3m"] * 100, 2),
                "mom_12_1_pct": round(m["mom_12_1"] * 100, 2) if "mom_12_1" in m else None,
                "rel_vs_sector_3m_pct": round(rel_raw[sym] * 100, 2) if sym in rel_raw else None,
                "sector_etf": etf,
                "above_200d": m.get("above_200d"),
                "vol_3m_ann_pct": round(m["vol_3m_ann"] * 100, 1) if m.get("vol_3m_ann") else None,
            }
        mom_rank, rel_rank = _pct_rank(mom_raw), _pct_rank(rel_raw)

        for sym, c in self.candidates.items():
            meta = c["meta"]
            comps: Dict[str, float] = {}
            reasons: List[str] = []
            c["status"] = ("watching" if sym in state.watchlist
                           else "dismissed" if c["status"] == "dismissed" else "candidate")

            # --- Smart money ---
            trend = trends.get(sym)
            smart = {}
            if trend:
                for name in SMART_SOURCES:
                    sig = trend.signals.get(name)
                    if sig:
                        smart[name] = {"score": sig.conviction_score, "sentiment": sig.sentiment,
                                       "details": sig.details}
            if smart:
                vals = [(0.45 if n == "SEC Form 4" else 0.30,
                         _bullishness(trend.signals[n])) for n in smart]
                comps["smart_money"] = sum(w * v for w, v in vals) / sum(w for w, _ in vals)
                reasons += [f"{n}: {d['details']}" for n, d in smart.items()]
            c["smart_money"] = smart

            # --- Public sentiment: real news (Laya) + StockTwits ---
            public = {}
            rec = state.get_sentiment(sym)
            if rec.n_headlines > 0 and not rec.is_stale:
                public["news"] = {"pos": round(rec.pos_prob, 3), "neg": round(rec.neg_prob, 3),
                                  "n_headlines": rec.n_headlines,
                                  "agreement": round(rec.agreement, 2),
                                  "headline": rec.headline}
            st = trend.signals.get("StockTwits") if trend else None
            if st:
                public["stocktwits"] = {"bull_ratio": st.conviction_score, "details": st.details}
            parts = []
            if "news" in public:
                # Net tone, shrunk toward neutral when headlines disagree or are few.
                tone = 0.5 + (rec.pos_prob - rec.neg_prob) / 2
                trust = min(1.0, rec.n_headlines / 4) * rec.agreement
                parts.append(0.5 + (tone - 0.5) * trust)
            if st:
                parts.append(st.conviction_score)
            if parts:
                comps["public_sentiment"] = float(np.mean(parts))
                if "news" in public:
                    reasons.append(f"News tone {public['news']['pos']:.0%} pos / "
                                   f"{public['news']['neg']:.0%} neg over {rec.n_headlines} headlines")
                if st:
                    reasons.append(f"StockTwits {st.details}")
            c["public_sentiment"] = public

            # --- Momentum ---
            if sym in mom_rank:
                t = trend_ok.get(sym)
                score = 0.6 * mom_rank[sym] + 0.4 * rel_rank.get(sym, 0.5)
                if t is False:
                    score *= 0.6   # below its 200-day average: momentum is suspect
                comps["momentum"] = score
                d = momentum_detail[sym]
                reasons.append(
                    f"Momentum {d['mom_12_1_pct'] if d['mom_12_1_pct'] is not None else d['ret_3m_pct']}%"
                    + (f", {d['rel_vs_sector_3m_pct']:+}% vs {d['sector_etf']} (3m)"
                       if d["rel_vs_sector_3m_pct"] is not None else "")
                    + ("" if t is None else (", above 200d" if t else ", below 200d")))
            c["momentum"] = momentum_detail.get(sym)

            # --- Diversification fit (same caps as the entry gate) ---
            if meta["tradable_on_alpaca"]:
                a = diversification.assess(sym)
                min_useful = min(std_position, 30.0 if not is_crypto_symbol(sym) else 15.0)
                if a.max_dollars < min_useful:
                    fit = 0.0
                    reasons.append(f"Blocked by {a.binding}" if a.headroom_dollars < min_useful
                                   else a.notes[-1])
                else:
                    fit = 0.5 * min(1.0, a.max_dollars / std_position) if std_position else 0.5
                    if meta["region"] in under_target_regions:
                        fit += 0.25
                        reasons.append(f"Fills under-target region {meta['region']}")
                    if meta["sector"] not in held_sectors and meta["sector"] not in (CRYPTO, DIVERSIFIED):
                        fit += 0.15
                    if defensive_short and meta["is_defensive"]:
                        fit += 0.2
                        reasons.append("Adds defensive exposure the book is short of")
                    if a.max_corr is not None:
                        fit -= 0.4 * max(0.0, a.max_corr - 0.4) / 0.6
                        if a.max_corr >= profile.max_pair_correlation:
                            reasons.append(a.notes[-1])
                comps["diversification"] = max(0.0, min(1.0, fit))
                c["fit"] = {**a.to_dict(), "position_size": round(min(std_position, a.max_dollars), 2)}
            else:
                c["fit"] = None
                reasons.append(f"Not tradable on Alpaca; proxy {meta['proxy'] or 'none'}")

            evidence = [k for k in comps if k != "diversification"]
            if evidence:
                w = sum(WEIGHTS[k] for k in comps)
                c["score"] = round(sum(WEIGHTS[k] * v for k, v in comps.items()) / w, 4)
            else:
                c["score"] = None
            c["components"] = {k: round(v, 3) for k, v in comps.items()}
            c["evidence"] = len(evidence)
            c["reasons"] = reasons[:8]

        # Public sentiment is tracked for the strongest tradable, not-yet-watched stocks.
        ranked = sorted(
            (c for c in self.candidates.values()
             if c["meta"]["tradable_on_alpaca"] and c["status"] == "candidate"
             and not is_crypto_symbol(c["symbol"]) and c["meta"]["asset_type"] != "etf"),
            key=lambda c: (c.get("score") is not None, c.get("score") or 0.0, bool(c["smart_money"])),
            reverse=True)
        self._sentiment_symbols = [c["symbol"] for c in ranked[:settings.DISCOVERY_SENTIMENT_TOP_N]]

    async def _auto_select(self):
        """Keeps the best-scoring stocks on the watchlist without a manual Watch."""
        from core.market_filter import market_filter
        from engine.executor import executor
        from feeds.alpaca_stream import market_stream

        # A stock the user took off the watchlist stays off: treat it as hidden.
        for sym in list(self.auto_symbols):
            if sym not in state.watchlist:
                self.auto_symbols.discard(sym)
                if sym in self.candidates:
                    self.candidates[sym]["status"] = "dismissed"

        if not settings.DISCOVERY_AUTO_PROMOTE or "stocks" in market_filter.disabled_markets:
            return

        n = settings.DISCOVERY_AUTO_TOP_N
        ranked = sorted(
            (c for c in self.candidates.values()
             if c.get("score") is not None and c["status"] != "dismissed"
             and c["meta"]["tradable_on_alpaca"] and c["meta"]["asset_type"] != "etf"
             and not is_crypto_symbol(c["symbol"])
             and c["components"].get("diversification", 1.0) > 0
             and not market_filter.entry_block_reason(c["symbol"])),
            key=lambda c: c["score"], reverse=True)

        # Extra slots for what the smart-money sources are clearly bullish on
        # (insider buying, top eToro investors), as long as our own score does
        # not contradict them. Bearish or neutral mentions get no slot.
        from feeds.multi_source_aggregator import trend_aggregator
        smart = []
        for c in ranked:
            cons = trend_aggregator.get_consensus(c["symbol"])
            if (cons is not None and cons >= settings.DISCOVERY_AUTO_SMART_MONEY_MIN_CONSENSUS
                    and c["score"] >= settings.DISCOVERY_AUTO_SMART_MONEY_MIN_SCORE):
                smart.append(c)
        smart = smart[:settings.DISCOVERY_AUTO_SMART_MONEY_N]
        top = [c for c in ranked[:n] if c["score"] >= settings.DISCOVERY_AUTO_MIN_SCORE]

        for c in top + smart:
            sym = c["symbol"]
            if sym in state.watchlist:
                continue
            try:
                self.promote(sym, auto=True)
            except ValueError as e:
                c["reasons"] = [f"Auto-pick skipped: {e}"] + c["reasons"][:7]
                continue
            await market_stream.ensure_stock_subscription(sym)

        # Hysteresis: an auto pick is only dropped once it falls out of the top 2N
        # or clearly below the bar, so ranks shuffling by a place cause no churn.
        keep = {c["symbol"] for c in ranked[:2 * n]
                if c["score"] >= settings.DISCOVERY_AUTO_MIN_SCORE - 0.05}
        keep |= {c["symbol"] for c in smart}
        for sym in list(self.auto_symbols):
            if (sym in keep or sym in state.active_positions or sym in executor.pending_orders):
                continue
            state.watchlist.discard(sym)
            self.auto_symbols.discard(sym)
            if sym in self.candidates:
                self.candidates[sym]["status"] = "candidate"
            state.log_event("DISCOVERY", f"Auto-dropped {sym} from the watchlist: no longer a top pick.")

        for c in self.candidates.values():
            c["auto"] = c["symbol"] in self.auto_symbols

    # ------------------------------------------------------------------
    # Views and actions
    # ------------------------------------------------------------------

    def list(self, status: Optional[str] = None, sector: Optional[str] = None,
             region: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        rows = []
        for c in self.candidates.values():
            if status == "active":
                if c["status"] == "dismissed":
                    continue
            elif status and c["status"] != status:
                continue
            if sector and c["meta"]["sector"] != sector:
                continue
            if region and c["meta"]["region"] != region:
                continue
            rows.append(c)
        rows.sort(key=lambda c: (c.get("score") is not None, c.get("score") or 0.0), reverse=True)
        return rows[:limit]

    def promote(self, symbol: str, auto: bool = False) -> Dict[str, Any]:
        from engine.executor import executor
        sym = symbol.upper().strip()
        c = self.candidates.get(sym)
        meta = universe.classify(sym)
        if not meta.tradable_on_alpaca:
            raise ValueError(f"{sym} is not tradable on Alpaca"
                             + (f"; consider the proxy {meta.proxy}" if meta.proxy else ""))
        if executor.is_connected and not executor.is_mock_mode and not executor.is_symbol_tradable(sym):
            raise ValueError(f"{sym} is not in Alpaca's tradable asset list")
        state.watchlist.add(sym)
        # A manual Watch makes the symbol the user's: it is never auto-dropped.
        (self.auto_symbols.add if auto else self.auto_symbols.discard)(sym)
        if c:
            c["status"] = "watching"
            c["auto"] = auto
        a = diversification.assess(sym)
        score = f", score {c['score']:.2f}" if c and c.get("score") is not None else ""
        state.log_event("DISCOVERY",
                        f"{'Auto-picked' if auto else 'Promoted'} {sym} ({meta.sector}, {meta.region}{score}) "
                        f"to the watchlist. Diversification headroom ${a.max_dollars:,.2f}"
                        + (f" (bound by {a.binding})" if a.binding else "") + ".")
        return {"symbol": sym, "assessment": a.to_dict()}

    def on_unwatched(self, symbol: str):
        """The user took a symbol off the watchlist: an auto pick stays off (hidden)."""
        sym = symbol.upper().strip()
        c = self.candidates.get(sym)
        was_auto = sym in self.auto_symbols
        self.auto_symbols.discard(sym)
        if c:
            c["status"] = "dismissed" if was_auto else "candidate"
            c["auto"] = False

    def dismiss(self, symbol: str) -> bool:
        c = self.candidates.get(symbol.upper().strip())
        if not c or c["status"] == "watching":
            return False
        c["status"] = "dismissed"
        return True

    def restore(self, symbol: str) -> bool:
        c = self.candidates.get(symbol.upper().strip())
        if not c or c["status"] != "dismissed":
            return False
        c["status"] = "candidate"
        return True


discovery = DiscoveryService()
