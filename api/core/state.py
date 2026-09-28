import time
from collections import deque
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import asyncio
from core.risk_profile import (
    RiskProfile, get_profile, clamp_factor, DEFAULT_RISK_FACTOR,
)

@dataclass
class ScoredHeadline:
    """One real news item, scored once by Laya and retained for aggregation."""
    news_id: str
    symbol: str
    headline: str
    pos_prob: float
    neg_prob: float
    neutral_prob: float = 0.0
    source: str = ""
    published_at: float = field(default_factory=time.time)
    scored_at: float = field(default_factory=time.time)


@dataclass
class SentimentRecord:
    stock_id: str
    pos_prob: float = 0.5
    neg_prob: float = 0.5
    open: bool = False
    headline: str = ""
    updated_at: float = field(default_factory=time.time)
    # --- Aggregation metadata ---
    # How many real headlines back this score. A single headline is a rumour;
    # several agreeing headlines are a signal. Entries require a minimum count.
    n_headlines: int = 0
    # Fraction of contributing headlines that agree with the net direction (0.5-1.0).
    # Laya ships over-confident (per its model card, ECE 0.466 unscaled), so
    # cross-headline agreement is a more trustworthy confidence measure than any
    # single prediction's own probability.
    agreement: float = 0.0
    sources: tuple = ()
    is_stale: bool = False
    is_tradeable: bool = False

@dataclass
class PriceTick:
    symbol: str
    price: float
    bid: float
    ask: float
    volume: float
    timestamp: float = field(default_factory=time.time)

@dataclass
class QuantMetrics:
    symbol: str
    rsi: Optional[float] = None
    ema_fast: Optional[float] = None
    ema_slow: Optional[float] = None
    atr: Optional[float] = None
    spread: float = 0.0
    volume_ratio: float = 1.0
    updated_at: float = field(default_factory=time.time)

@dataclass
class TradeDecision:
    symbol: str
    action: str  # "BUY", "SELL", "CLOSE", "HOLD"
    buy_prob: float = 0.0
    hold_prob: float = 0.0
    sell_prob: float = 0.0
    close_prob: float = 0.0
    close: bool = False
    reason: str = ""
    timestamp: float = field(default_factory=time.time)
    # A protective exit (stale price, end of day) that must not wait out the
    # normal retry backoff: see settings.FORCED_EXIT_MAX_WAIT_SECONDS.
    forced: bool = False
    # SELL: share of the position to sell (0.5 = trim half). CLOSE is always all.
    fraction: float = 1.0
    # SELL from the profit harvest: the gain is booked as ring-fenced day income.
    harvest: bool = False
    # BUY from loss recovery: a one-time add of rescue_qty to a losing position.
    rescue: bool = False
    rescue_qty: float = 0.0

def ny_date(ts: Optional[float] = None) -> str:
    """The New York calendar date: the US trading day, whatever the host's timezone."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return datetime.fromtimestamp(time.time() if ts is None else ts,
                                  tz=ZoneInfo("America/New_York")).strftime("%Y-%m-%d")


def price_decimals(price: float) -> int:
    """
    Decimal places appropriate to an asset's price scale.

    Rounding every asset to 2dp silently destroys sub-cent assets: a SHIB stop
    distance of 1e-8 becomes 0.00, and a 0.00 take-profit reads as immediately hit.
    Any price-derived value (stop, target, ATR) must be rounded through this.
    """
    p = abs(float(price))
    if p == 0:
        return 8
    if p < 0.0001:
        return 10
    if p < 0.01:
        return 8
    if p < 1.0:
        return 6
    if p < 100.0:
        return 4
    return 2


def round_price(value: float, ref_price: float) -> float:
    """Rounds a price-derived value at the precision implied by ref_price."""
    return round(float(value), price_decimals(ref_price))


class InMemoryState:
    """
    In-memory state store for the trading engine. US equities only.
    """
    def __init__(self):
        # Global Kill Switch (Deactivated by default on boot for safety)
        self.is_trading_active: bool = False
        
        # Default watchlist: liquid US large caps (discovery adds more at runtime).
        self.watchlist: set[str] = {"NVDA", "AAPL", "MSFT", "PLTR"}
        
        # Real-time price ticks: symbol -> PriceTick
        self.latest_prices: Dict[str, PriceTick] = {}
        # Latest top of book from the quote stream: symbol -> (bid, ask, unix time),
        # and a smoothed relative spread per symbol (fraction of mid). The spread
        # estimate is what live trading actually pays; the cost model uses it.
        self.latest_quotes: Dict[str, tuple] = {}
        self.spread_estimate: Dict[str, float] = {}
        # symbol -> when its price last CHANGED. A tick that repeats the last price
        # (a re-marked quiet position, a heartbeat) does not count, so the age of
        # this stamp tells a quiet market apart from a dead feed.
        self.price_moved_at: Dict[str, float] = {}

        # Ring buffers of price history (up to 250 bars/ticks) for instant math
        self.price_history: Dict[str, deque] = {}
        self.volume_history: Dict[str, deque] = {}
        # Wall-clock time of each price sample, parallel to price_history. Lets
        # off-path analytics align symbols in time (pairs, correlation, VaR).
        self.time_history: Dict[str, deque] = {}

        # Results of the off-process analysis worker (engine/analysis): regime,
        # quant council, Monte Carlo, pair signals per symbol, plus portfolio
        # risk. The hot path only ever READS these; it never computes them.
        self.analysis: Dict[str, Dict[str, Any]] = {}
        self.portfolio_analytics: Dict[str, Any] = {}
        self.analysis_at: float = 0.0
        # symbol -> (tick object, price array); see history_array()
        self._history_arrays: Dict[str, tuple] = {}
        self.tick_count: Dict[str, int] = {}

        # Cached Laya sentiment scores: symbol -> SentimentRecord (derived view)
        self.sentiment_cache: Dict[str, SentimentRecord] = {}

        # Real scored headlines per symbol, newest last. Sentiment is AGGREGATED
        # from these rather than overwritten by whichever loop wrote last -- the
        # previous last-write-wins behaviour made the signal depend on scheduler
        # ordering rather than on the news.
        self.sentiment_history: Dict[str, deque] = {}
        # News ids already scored, so the same article is never counted twice.
        self.seen_news_ids: set = set()
        self._seen_news_order: deque = deque(maxlen=5000)

        # Pre-calculated Quant metrics (t1...tN): symbol -> QuantMetrics
        self.quant_metrics: Dict[str, QuantMetrics] = {}

        # Active open positions from Alpaca: symbol -> dict
        self.active_positions: Dict[str, Dict[str, Any]] = {}

        # Account Equity & Cash cache
        self.account_info: Dict[str, Any] = {
            "equity": 100000.0,
            "cash": 100000.0,
            "buying_power": 200000.0,
            "day_pnl": 0.0,
            "locked_equity": 90000.0,
            "main_broker_equity": 90000.0,
            "locked_cash": 90000.0,
            "is_broker_equity_locked": True,
        }

        # Main broker equity locking: ensures bot never touches broker capital outside budget
        self.broker_equity_baseline: float = 100000.0
        self.broker_cash_baseline: float = 100000.0
        self.locked_broker_equity: float = 90000.0
        self.locked_broker_cash: float = 90000.0
        self.is_broker_equity_locked: bool = True

        # Max trading capital budget allowed for bot execution (default $10,000.00 USD)
        self.allocated_capital: float = 10000.0

        # --- Strategy selection ---
        # The default strategy, and per-symbol overrides that win over it.
        # Changed live from the API; read on every tick.
        self.strategy_class_defaults: Dict[str, str] = {
            "equity": "stock_score",
        }
        self.strategy_overrides: Dict[str, str] = {}
        # Last entry-gate evaluation per symbol, so the UI can explain a no-trade.
        self.last_gate_detail: Dict[str, Dict[str, Any]] = {}

        # Portfolio-wide risk dial (1-10, default 4). Every risk limit is derived
        # from this, so changing it takes effect on the very next evaluation --
        # no restart, no recomputation step.
        self._risk_factor: int = DEFAULT_RISK_FACTOR

        # --- Portfolio-level risk tracking (drives the circuit breakers) ---
        # Realised PnL booked today, reset at the start of each trading day.
        self.realized_pnl_today: float = 0.0
        # Part of realized_pnl_today taken as ring-fenced day income by the
        # profit harvest. Excluded from the daily-loss breaker.
        self.harvested_today: float = 0.0
        # Stocks Alpaca trades in fractions (from the asset catalog at start-up):
        # a profit harvest sells half of a one-share winner as 0.5 share.
        self.fractionable_symbols: set = set()
        self.trading_day: str = ny_date()
        # High-water mark of equity, for drawdown measurement.
        self.peak_equity: float = 0.0
        # Set when a circuit breaker trips; blocks new entries but never blocks exits.
        self.halt_reason: Optional[str] = None

        # Latest trade decisions history (last 50 for UI)
        self.recent_decisions: deque[TradeDecision] = deque(maxlen=50)
        self.recent_trades: deque[Dict[str, Any]] = deque(maxlen=50)
        # Closed-trade outcome ledger: the learning substrate. Kept longer than the
        # UI feed because attribution and any bandit update need real sample counts.
        self.closed_trades: deque[Dict[str, Any]] = deque(maxlen=2000)
        self.logs: deque[Dict[str, Any]] = deque(maxlen=100)

        self._lock = asyncio.Lock()

    @property
    def risk_factor(self) -> int:
        return self._risk_factor

    @risk_factor.setter
    def risk_factor(self, value):
        self._risk_factor = clamp_factor(value)

    @property
    def risk_profile(self) -> RiskProfile:
        """
        Live risk limits derived from the current dial.

        Read fresh on every access rather than cached, so a slider change in the UI
        is reflected in the next sizing decision and the next risk check with no
        invalidation step to forget.
        """
        return get_profile(self._risk_factor)

    @property
    def total_position_exposure(self) -> float:
        """Calculates total dollar value currently locked in active positions"""
        total = 0.0
        for pos in self.active_positions.values():
            qty = float(pos.get("qty", 0.0))
            price = float(pos.get("current_price", pos.get("avg_entry_price", 0.0)))
            total += qty * price
        return round(total, 2)

    # ---- Hard budget cap ----
    #
    # The budget is a hard cap on the money the bots may put at risk, not a soft
    # target measured against market value. Three things used to let the bots
    # spend outside it: headroom was cap minus MARKET value (a falling position
    # reopened room to buy more, so cost basis crept past the cap); realised
    # losses never shrank the cap (after losing $500 of $10k the bots still
    # deployed $10k); and orders in flight were not counted at all.

    @property
    def bot_capital(self) -> float:
        """The bots' own money: the cap plus realised PnL booked against it."""
        from core.capital_plan import capital_plan
        return round(self.allocated_capital + capital_plan.budget_realized, 2)

    @property
    def hard_cap(self) -> float:
        """Most the bots may have committed at once: the cap, reduced by realised losses."""
        return max(0.0, round(min(self.allocated_capital, self.bot_capital), 2))

    @property
    def committed_capital(self) -> float:
        """Cost basis of open positions plus notional of buys still in flight."""
        total = 0.0
        for pos in self.active_positions.values():
            qty = float(pos.get("qty", 0.0))
            avg = float(pos.get("avg_entry_price") or pos.get("current_price") or 0.0)
            if qty > 0:
                total += qty * avg
        from engine.executor import executor
        total += executor.inflight_committed()
        return round(total, 2)

    @property
    def remaining_budget(self) -> float:
        """Dollars the bots may still commit to new trades under the hard cap."""
        return max(0.0, round(self.hard_cap - self.committed_capital, 2))

    @property
    def bot_cash(self) -> float:
        """Cash available strictly within the bot's budget under the hard cap. Never touches locked broker cash."""
        broker_cash = float(self.account_info.get("cash", 0.0))
        return max(0.0, round(min(broker_cash, self.remaining_budget), 2))

    @property
    def budget_base(self) -> float:
        """Capital that percentage-based sizing is measured against.
        Strictly capped by the bot's hard cap; never touches broker equity outside."""
        return max(0.0, min(self.hard_cap, self.bot_capital))

    @property
    def open_unrealized_pnl(self) -> float:
        return round(sum(float(p.get("unrealized_pl") or 0.0)
                         for p in self.active_positions.values()), 2)

    @property
    def bot_equity(self) -> float:
        """What the bot account/slice is worth now."""
        from core.capital_plan import capital_plan
        if capital_plan.plan.mode == "stair":
            return round(capital_plan.plan.trading_capital + capital_plan.plan.reserve + capital_plan.plan.banked_income
                         + capital_plan.plan.harvested_income + self.open_unrealized_pnl, 2)
        # Harvested income is counted so moving a gain from unrealised to income
        # is not read as a drawdown; it is still never part of bot_capital.
        return round(self.bot_capital + capital_plan.plan.harvested_income + self.open_unrealized_pnl, 2)

    def sync_locked_equity(self, assigned: Optional[float] = None):
        """
        Locks the main broker equity so the bot CANNOT touch or affect funds outside the budget cap.
        In classic mode: locked = max(0, total_broker_equity - allocated_capital).
        In stair mode: locked = max(0, total_broker_equity - deposit).
        """
        from core.capital_plan import capital_plan
        if assigned is None:
            assigned = capital_plan.plan.deposit if capital_plan.plan.mode == "stair" else self.allocated_capital

        current_equity = float(self.account_info.get("equity", 100000.0))
        if self.broker_equity_baseline <= 0 or current_equity > self.broker_equity_baseline:
            self.broker_equity_baseline = current_equity

        self.locked_broker_equity = max(0.0, round(self.broker_equity_baseline - float(assigned), 2))

        current_cash = float(self.account_info.get("cash", 100000.0))
        if self.broker_cash_baseline <= 0 or current_cash > self.broker_cash_baseline:
            self.broker_cash_baseline = current_cash
        self.locked_broker_cash = max(0.0, round(self.broker_cash_baseline - float(assigned), 2))

        self.is_broker_equity_locked = True
        self.account_info["locked_equity"] = self.locked_broker_equity
        self.account_info["main_broker_equity"] = self.locked_broker_equity
        self.account_info["locked_cash"] = self.locked_broker_cash
        self.account_info["is_broker_equity_locked"] = True
        self.account_info["bot_equity"] = self.bot_equity
        self.account_info["bot_capital"] = self.bot_capital
        self.account_info["hard_cap"] = self.hard_cap
        self.account_info["bot_cash"] = self.bot_cash

    @property
    def daily_pnl(self) -> Dict[str, Any]:
        from core.pnl_ledger import pnl_ledger
        return pnl_ledger.snapshot(
            self.open_unrealized_pnl,
            self.hard_cap,
            active_positions=self.active_positions,
            total_equity=float(self.account_info.get("equity", 100000.0)),
            locked_equity=self.locked_broker_equity,
        )

    def budget_snapshot(self) -> Dict[str, Any]:
        committed = self.committed_capital
        cap = self.hard_cap
        return {
            "allocated_capital": self.allocated_capital,
            "hard_cap": cap,
            "bot_capital": self.bot_capital,
            "bot_equity": self.bot_equity,
            "bot_cash": self.bot_cash,
            "committed_capital": committed,
            "total_position_exposure": self.total_position_exposure,
            "remaining_budget": max(0.0, round(cap - committed, 2)),
            "utilization_pct": round(committed / cap * 100, 1) if cap > 0 else 100.0,
            "locked_broker_equity": self.locked_broker_equity,
            "main_broker_equity": self.locked_broker_equity,
            "locked_broker_cash": self.locked_broker_cash,
            "is_broker_equity_locked": self.is_broker_equity_locked,
            "broker_equity_baseline": self.broker_equity_baseline,
        }

    def get_or_create_history(self, symbol: str, maxlen: int = 250) -> deque:
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=maxlen)
            self.volume_history[symbol] = deque(maxlen=maxlen)
            self.time_history[symbol] = deque(maxlen=maxlen)
        return self.price_history[symbol]

    QUOTE_MAX_AGE_S = 10.0

    def update_quote(self, symbol: str, bid: float, ask: float, ts: Optional[float] = None):
        """Records a live quote. Moves bid/ask and the spread estimate, never history."""
        if bid <= 0 or ask < bid:
            return
        now = time.time() if ts is None else ts
        self.latest_quotes[symbol] = (float(bid), float(ask), now)
        mid = (bid + ask) / 2.0
        rel = (ask - bid) / mid if mid > 0 else 0.0
        prev = self.spread_estimate.get(symbol)
        # Slow EWMA: one wide quote in a burst should not swing the estimate.
        self.spread_estimate[symbol] = rel if prev is None else 0.98 * prev + 0.02 * rel
        tick = self.latest_prices.get(symbol)
        if tick is not None:
            tick.bid, tick.ask = float(bid), float(ask)

    def fresh_quote(self, symbol: str, now: Optional[float] = None) -> Optional[tuple]:
        q = self.latest_quotes.get(symbol)
        if q is None:
            return None
        now = time.time() if now is None else now
        return q if now - q[2] <= self.QUOTE_MAX_AGE_S else None

    def update_price(self, symbol: str, price: float, bid: float = 0.0, ask: float = 0.0,
                     volume: float = 0.0, record_history: bool = True):
        """
        record_history=True only for a completed one-minute bar: that is the one
        cadence the indicators, the backtester and the RL policy are built on.
        Quotes and trade prints pass record_history=False and only move the
        live price. Without an explicit bid/ask, a fresh streamed quote is used.
        """
        if not (bid and ask):
            q = self.fresh_quote(symbol)
            if q is not None:
                bid, ask = q[0], q[1]
        tick = PriceTick(symbol=symbol, price=price, bid=bid or price, ask=ask or price, volume=volume)
        prev = self.latest_prices.get(symbol)
        if prev is None or prev.price != price or symbol not in self.price_moved_at:
            self.price_moved_at[symbol] = tick.timestamp
        self.latest_prices[symbol] = tick
        if record_history:
            buf = self.get_or_create_history(symbol)
            buf.append(price)
            # Monotonic per-symbol sample counter: lets incremental indicators tell
            # "exactly one new sample" apart from gaps they must recompute over.
            self.tick_count[symbol] = self.tick_count.get(symbol, 0) + 1
            self.time_history[symbol].append(tick.timestamp)
            if volume > 0:
                self.volume_history[symbol].append(volume)

        # Real-time Open PnL calculation on EVERY price tick (< 0.01 ms)
        if symbol in self.active_positions:
            pos = self.active_positions[symbol]
            qty = float(pos.get("qty", 0.0))
            avg = float(pos.get("avg_entry_price", price))
            if qty > 0 and avg > 0:
                precision = 8 if price < 0.0001 else (4 if price < 1.0 else 2)
                pos["current_price"] = round(price, precision)
                pos["unrealized_pl"] = round((price - avg) * qty, 2)
                pos["unrealized_plpc"] = round((price - avg) / avg, 5)

    def history_array(self, symbol: str):
        """
        The symbol's price history as a numpy array, converted at most once per
        tick: the quant matrix and the strategy context both need it, and the
        deque -> array copy is a measurable share of the tick path.
        """
        import numpy as np
        tick = self.latest_prices.get(symbol)
        cached = self._history_arrays.get(symbol)
        if cached is not None and cached[0] is tick:
            return cached[1]
        hist = self.price_history.get(symbol) or ()
        arr = np.fromiter(hist, dtype=np.float64, count=len(hist))
        self._history_arrays[symbol] = (tick, arr)
        return arr

    def fresh_analysis(self, symbol: str, max_age_s: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """This symbol's latest worker result, or None if absent or older than max_age_s."""
        from core.config import settings
        if max_age_s is None:
            max_age_s = settings.ANALYSIS_MAX_AGE_SECONDS
        a = self.analysis.get(symbol)
        if a is None or time.time() - a.get("at", 0.0) > max_age_s:
            return None
        return a

    def update_sentiment(self, record: SentimentRecord):
        """Legacy single-write path, retained for manual news injection."""
        self.sentiment_cache[record.stock_id] = record

    def is_news_seen(self, news_id: str) -> bool:
        return str(news_id) in self.seen_news_ids

    def mark_news_seen(self, news_id: str):
        nid = str(news_id)
        if nid in self.seen_news_ids:
            return
        self.seen_news_ids.add(nid)
        self._seen_news_order.append(nid)
        # Bound the set alongside the bounded deque it mirrors
        while len(self.seen_news_ids) > self._seen_news_order.maxlen:
            oldest = self._seen_news_order.popleft() if self._seen_news_order else None
            if oldest is None:
                break
            self.seen_news_ids.discard(oldest)

    def record_headline(self, scored: ScoredHeadline, maxlen: int = 40):
        """Appends a scored headline to a symbol's rolling sentiment history."""
        hist = self.sentiment_history.get(scored.symbol)
        if hist is None:
            hist = deque(maxlen=maxlen)
            self.sentiment_history[scored.symbol] = hist
        hist.append(scored)
        # Invalidate the derived view so the next read recomputes
        self.sentiment_cache.pop(scored.symbol, None)

    def get_sentiment(self, symbol: str, max_age_seconds: Optional[float] = None) -> SentimentRecord:
        """
        Recency-weighted consensus across the symbol's recent REAL headlines.

        Three properties this buys over the previous last-write-wins cache:

        1. Order independence. Two loops writing about the same symbol no longer
           race; both headlines contribute in proportion to their recency.
        2. Calibration by agreement. Laya ships over-confident, so one 0.95
           reading is not worth much. Agreement across independent headlines is a
           far better confidence estimate, and entries gate on it.
        3. Honest expiry. Once every headline ages out, the result is NEUTRAL --
           which fails the entry gates -- rather than an old score read as current.
        """
        from core.config import settings
        if max_age_seconds is None:
            max_age_seconds = settings.SENTIMENT_MAX_AGE_SECONDS

        now = time.time()
        cached = self.sentiment_cache.get(symbol)
        # Derived view is cheap to rebuild; reuse it only within the same second.
        if cached is not None and (now - cached.updated_at) < 1.0 and cached.n_headlines:
            return cached

        hist = self.sentiment_history.get(symbol)
        if not hist:
            # Fall back to any manually injected single record
            if cached is not None and cached.n_headlines == 0:
                age = now - cached.updated_at
                if age <= max_age_seconds:
                    return cached
            return SentimentRecord(stock_id=symbol, pos_prob=0.5, neg_prob=0.5,
                                   headline="No news yet", n_headlines=0)

        fresh = [h for h in hist if (now - h.published_at) <= max_age_seconds]
        if not fresh:
            newest = max(hist, key=lambda h: h.published_at)
            age_m = (now - newest.published_at) / 60.0
            return SentimentRecord(
                stock_id=symbol, pos_prob=0.5, neg_prob=0.5, open=False,
                headline=f"STALE ({age_m:.0f}m old): {newest.headline[:70]}",
                updated_at=newest.published_at, n_headlines=0,
                is_stale=True, is_tradeable=False,
            )

        # Exponential recency weighting: a headline at the half-life contributes half.
        half_life = max(60.0, max_age_seconds / 3.0)
        wsum = pos = neg = 0.0
        for h in fresh:
            w = 0.5 ** ((now - h.published_at) / half_life)
            wsum += w
            pos += w * h.pos_prob
            neg += w * h.neg_prob
        pos /= wsum
        neg /= wsum

        # Directional agreement among contributing headlines
        net_bullish = pos >= neg
        agreeing = sum(1 for h in fresh if (h.pos_prob >= h.neg_prob) == net_bullish)
        agreement = agreeing / len(fresh)

        newest = max(fresh, key=lambda h: h.published_at)
        tradeable = (
            len(fresh) >= settings.MIN_HEADLINES_FOR_ENTRY
            and agreement >= settings.MIN_HEADLINE_AGREEMENT
        )

        rec = SentimentRecord(
            stock_id=symbol,
            pos_prob=round(float(pos), 4),
            neg_prob=round(float(neg), 4),
            open=(pos >= 0.60 or neg >= 0.60),
            headline=newest.headline[:160],
            updated_at=now,
            n_headlines=len(fresh),
            agreement=round(agreement, 3),
            sources=tuple(sorted({h.source for h in fresh if h.source})),
            is_stale=False,
            is_tradeable=tradeable,
        )
        self.sentiment_cache[symbol] = rec
        return rec

    def sentiment_age(self, symbol: str) -> Optional[float]:
        """Seconds since the newest headline for this symbol, or None if none."""
        hist = self.sentiment_history.get(symbol)
        if hist:
            return round(time.time() - max(h.published_at for h in hist), 1)
        rec = self.sentiment_cache.get(symbol)
        return None if rec is None else round(time.time() - rec.updated_at, 1)

    def roll_trading_day_if_needed(self):
        """Resets the daily loss counter when the New York trading day turns over."""
        today = ny_date()
        if today != self.trading_day:
            self.log_event(
                "RISK_DAY_ROLL",
                f"Trading day rolled {self.trading_day} -> {today}. "
                f"Realised PnL for {self.trading_day}: ${self.realized_pnl_today:+,.2f}. Counters reset."
            )
            from core.pnl_ledger import pnl_ledger
            pnl_ledger.roll(today)
            self.trading_day = today
            self.realized_pnl_today = pnl_ledger.realized_today()
            self.harvested_today = pnl_ledger.harvested_today()
            # A daily-loss halt expires with the day; a drawdown halt does not.
            if self.halt_reason and "Daily loss" in self.halt_reason:
                self.halt_reason = None

    def book_realized_pnl(self, symbol: str, pnl: float):
        """Records realised PnL from a closed position against today's risk budget."""
        self.roll_trading_day_if_needed()
        self.realized_pnl_today = round(self.realized_pnl_today + float(pnl), 2)
        # Both capital modes shrink the hard cap on realised losses; stair also
        # ratchets its stage off realised PnL.
        from core.capital_plan import capital_plan
        capital_plan.on_realized(symbol, pnl)
        from core.pnl_ledger import pnl_ledger
        pnl_ledger.on_close(symbol, pnl)

    def revise_realized_pnl(self, symbol: str, old_pnl: float, new_pnl: float):
        """
        Corrects a booked close to the broker's actual fill. The budget, the
        stair ladder and the daily ledger all move by the difference, so the
        hard cap and the daily-loss halt see what the account really lost.
        """
        delta = round(float(new_pnl) - float(old_pnl), 2)
        if abs(delta) < 0.005:
            return
        self.roll_trading_day_if_needed()
        self.realized_pnl_today = round(self.realized_pnl_today + delta, 2)
        from core.capital_plan import capital_plan
        capital_plan.on_realized(symbol, delta)
        from core.pnl_ledger import pnl_ledger
        pnl_ledger.revise(symbol, old_pnl, new_pnl)

    @property
    def broker_day_loss_pct(self) -> float:
        """
        Today's loss of the whole broker account (equity vs. the previous close),
        as a percent of the bots' budget. It includes fees, slippage and open
        losses that booked P&L cannot see: the independent check behind the
        daily-loss halt.
        """
        last = float(self.account_info.get("last_equity") or 0.0)
        eq = float(self.account_info.get("equity") or 0.0)
        if last <= 0 or eq <= 0 or self.allocated_capital <= 0:
            return 0.0
        return max(0.0, round((last - eq) / self.allocated_capital * 100.0, 3))

    def book_harvested_income(self, symbol: str, pnl: float):
        """
        Books a profit-harvest gain as day income. It counts in today's P&L but
        never in the trading budget: not in classic's realised ledger, not in a
        stair stage's capital, and not as a cushion against the daily loss limit.
        """
        self.roll_trading_day_if_needed()
        value = round(float(pnl), 2)
        self.realized_pnl_today = round(self.realized_pnl_today + value, 2)
        self.harvested_today = round(self.harvested_today + value, 2)
        from core.capital_plan import capital_plan
        capital_plan.on_harvest(symbol, value)
        from core.pnl_ledger import pnl_ledger
        pnl_ledger.on_harvest(symbol, value)

    def update_peak_equity(self):
        # Measured on the bots' own equity. Against the whole broker account a
        # $10k budget in a $100k account diluted every drawdown tenfold, so the
        # breaker effectively never fired.
        eq = self.bot_equity
        if eq > self.peak_equity:
            self.peak_equity = eq

    def reset_peak_equity(self):
        """Called when the budget changes: the old high-water mark no longer applies."""
        self.peak_equity = self.bot_equity

    @property
    def drawdown_pct(self) -> float:
        """Percent of bot equity below its peak. 0.0 when at or above the high-water mark."""
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, round((self.peak_equity - self.bot_equity) / self.peak_equity * 100.0, 3))

    @property
    def daily_loss_pct(self) -> float:
        """Today's realised loss as a percent of the allocated budget (positive = loss)."""
        if self.allocated_capital <= 0:
            return 0.0
        # Harvested income is excluded: it is set aside, not a buffer for losses.
        trading_pnl = self.realized_pnl_today - self.harvested_today
        return max(0.0, round(-trading_pnl / self.allocated_capital * 100.0, 3))

    def log_event(self, level: str, message: str, meta: Optional[Dict[str, Any]] = None):
        self.logs.append({
            "timestamp": time.time(),
            "level": level,
            "message": message,
            "meta": meta or {}
        })

state = InMemoryState()
