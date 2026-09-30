import asyncio
import logging
import random
import re
import time
from typing import Dict, Optional
from core.config import settings
from core.state import state
from core.latency import latency
from engine.quant_matrix import quant_matrix
from engine.decision_engine import decision_engine
from engine.executor import executor
from engine.portfolio_manager import portfolio_manager

logger = logging.getLogger("tradeflow.stream")

class MarketStreamRunner:
    """
    The High-Speed Continuous Trading Loop.
    Evaluates ticks in sub-millisecond time.
    """
    def __init__(self):
        self._running = False
        self._task = None
        self._live = False
        self._stock_stream = None
        self._stock_bar_handler = None
        self._stock_trade_handler = None
        self._stock_quote_handler = None
        self._stock_task = None
        self._sync_lock = asyncio.Lock()
        # Watched symbols left off the live stream by STREAM_MAX_SYMBOLS.
        self.not_streamed: set = set()

    async def start(self):
        self._running = True
        # Check if we can connect to real Alpaca WebSocket stream
        if not settings.ALPACA_API_KEY.startswith("PK_PLACEHOLDER") and settings.ALPACA_API_KEY:
            self._live = True
            self._task = asyncio.create_task(self._run_alpaca_websocket())
            self._sync_task = asyncio.create_task(self._sync_loop())
        else:
            logger.info("Starting high-frequency simulated market stream (realistic ticks 24/7)...")
            self._task = asyncio.create_task(self._run_simulated_stream())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
        if getattr(self, "_sync_task", None):
            self._sync_task.cancel()

    def _held(self) -> set:
        return set(state.active_positions) | set(executor.pending_orders)

    def plan(self) -> list:
        """
        The symbols the live stream should carry, most important first, capped at
        settings.STREAM_MAX_SYMBOLS: held positions (their stops and targets run
        on it), then the context symbols, then the watchlist by scout rank
        (unranked ones after, alphabetically).
        """
        try:
            from scout.service import scout
            rank = {s: p.get("rank") for s, p in scout.picks.items()}
        except Exception:
            rank = {}
        watch = sorted(state.watchlist, key=lambda s: (rank.get(s) is None, rank.get(s) or 0, s))
        return stream_plan(sorted(self._held()), settings.CONTEXT_SYMBOLS, watch,
                           settings.STREAM_MAX_SYMBOLS)

    async def sync_subscriptions(self) -> list:
        """
        Makes the live stream carry exactly plan(): unsubscribes what fell out of
        it, then subscribes what is missing; returns the symbols whose bars were
        added. Symbols put on the watchlist before the stream existed (the scout's
        first ranking runs at start-up) were silently never subscribed and so
        never got a price. Symbols were also never unsubscribed, so a long
        session crept past the data plan's symbol limit -- and Alpaca rejects the
        whole subscribe then, leaving EVERY symbol without a price.
        """
        if not (self._running and self._live) or self._stock_stream is None:
            return []
        async with self._sync_lock:
            return await self._reconcile(self._stock_stream)

    async def _reconcile(self, stream) -> list:
        plan = self.plan()
        want = set(plan)
        held = self._held() & want
        wanted = {"bars": want, "quotes": want if self._stock_quote_handler else set(),
                  "trades": held if self._stock_trade_handler else set()}
        running = getattr(stream, "_running", False)
        # Unsubscribe first, so the stream never holds more than the limit.
        for channel, keep in wanted.items():
            handlers = stream._handlers.get(channel)
            gone = sorted(s for s in (handlers or {}) if s not in keep)
            if not gone:
                continue
            if running:
                try:
                    await stream._send_unsubscribe_msg(channel, gone)
                except Exception as e:
                    logger.warning(f"Live {channel} unsubscribe for {', '.join(gone)} failed: {e}")
            for s in gone:
                handlers.pop(s, None)

        handler_of = {"bars": self._stock_bar_handler, "quotes": self._stock_quote_handler,
                      "trades": self._stock_trade_handler}
        added = {}
        for channel, keep in wanted.items():
            handlers = stream._handlers.get(channel)
            if handlers is None:
                continue
            added[channel] = [s for s in plan if s in keep and s not in handlers]
            for s in added[channel]:
                handlers[s] = handler_of[channel]
        if running and any(added.values()):
            try:
                # Awaited directly: alpaca-py's public subscribe blocks on a
                # future scheduled onto this same loop, which would deadlock.
                await stream._send_subscribe_msg()
            except Exception as e:
                logger.warning(f"Live subscribe failed: {e}")
        self._note_left_out(plan)
        new_bars = sorted(added.get("bars", []))
        if new_bars:
            logger.info(f"Subscribed live bars for {len(new_bars)} symbol(s): {', '.join(new_bars)}")
        return new_bars

    def _note_left_out(self, plan: list):
        """Records (and logs once per change) the watched symbols the cap left off the stream."""
        left = set(state.watchlist) - set(plan)
        if left != self.not_streamed and left:
            state.log_event("STREAM", f"Live data plan full ({settings.STREAM_MAX_SYMBOLS} symbols): "
                                      f"not streaming {', '.join(sorted(left))} (lowest ranked). "
                                      "They get no price and cannot be entered.")
        self.not_streamed = left

    async def _sync_loop(self):
        while self._running:
            try:
                await asyncio.sleep(15.0)
                await self.sync_subscriptions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Subscription sync failed: {e}")

    async def ensure_stock_subscription(self, symbol: str):
        """
        Starts live bars for a stock added to the watchlist after startup (e.g. a
        promoted discovery candidate). The stream subscribes once at boot, so
        without this a promoted symbol would never receive a price until restart.
        The simulated stream reads the watchlist every loop and needs nothing.
        """
        if not (self._running and self._live):
            return
        if self._stock_stream is None:
            if self._stock_task is None or self._stock_task.done():
                self._stock_task = asyncio.create_task(self._run_stock_live_feed())
            return
        await self.sync_subscriptions()

    async def ensure_position_stream(self, symbol: str):
        """
        Streams every trade print for a held stock. Minute bars left an open
        position's stop and target checked at most once a minute, after the bar
        closed; each print now reaches its sentinel within milliseconds.
        """
        if not (self._running and self._live):
            return
        if self._stock_stream is None or self._stock_trade_handler is None:
            return  # the feed subscribes held positions itself when it starts
        await self.sync_subscriptions()

    async def on_position_trade(self, symbol: str, price: float, is_trade: bool = True):
        """
        Fast path for a trade print on a held position: move its live price and
        hand the tick to its sentinel, which re-evaluates BUY/HOLD/SELL/CLOSE and
        fires the exit on this tick. No indicator sample is added (see
        update_price), so the per-print cost stays in the sentinel.
        """
        if symbol not in state.active_positions or price <= 0:
            return
        t0 = time.perf_counter_ns()
        state.update_price(symbol, price, record_history=False)
        # A real trade print belongs to the forming minute's provisional bar
        # (replaced by the streamed bar when that minute closes). A broker mark
        # is not a trade and never makes a bar.
        if is_trade:
            from core.minute_bars import minute_bars
            minute_bars.on_tick(symbol, price)
        from engine.sentinel_agent import sentinel_registry
        await sentinel_registry.dispatch_tick(symbol, price)
        latency.record_ns("position_trade", t0)

    async def on_bar_received(self, symbol: str, minute: int, o: float, h: float, l: float,
                              c: float, volume: float):
        """
        A completed one-minute bar: the only thing that adds an indicator sample.
        Stored under its own start minute with its real OHLCV (core/minute_bars),
        so the live series is the same series the backtester and the RL trainer
        replay. Quotes and trade prints only move the live price between bars.
        """
        from core.minute_bars import minute_bars
        minute_bars.on_bar(symbol, minute, o, h, l, c, volume)
        await self.on_tick_received(symbol, c, 0.0, 0.0, volume, bar=True)

    def on_quote(self, symbol: str, bid: float, ask: float):
        """Live top of book: moves bid/ask (and the spread estimate), never history."""
        state.update_quote(symbol, bid, ask)

    async def on_tick_received(self, symbol: str, price: float, bid: float, ask: float, volume: float,
                               bar: bool = False):
        """
        Evaluates a new price. Only a completed bar (bar=True) records an
        indicator sample; anything else just moves the live price.
        """
        t0 = time.perf_counter_ns()

        # 1. Update In-Memory State
        state.update_price(symbol, price, bid, ask, volume, record_history=bar)
        if not bar:
            if symbol in state.active_positions:
                from engine.sentinel_agent import sentinel_registry
                asyncio.create_task(sentinel_registry.dispatch_tick(symbol, price))
            return

        # 2. Run Quant Matrix (t1...tN indicators)
        t_q = time.perf_counter_ns()
        quant = quant_matrix.evaluate_symbol(symbol)
        latency.record_ns("quant_matrix", t_q)

        # 3. Read Pre-computed Laya Sentiment from RAM
        sentiment = state.get_sentiment(symbol)

        # 4. If holding, dispatch tick to dedicated Sentinel Bot assigned to this trade
        if symbol in state.active_positions:
            from engine.sentinel_agent import sentinel_registry
            asyncio.create_task(sentinel_registry.dispatch_tick(symbol, price))
            latency.record_ns("tick_total", t0)
        elif symbol in executor.pending_orders:
            # Order currently transmitting to broker; skip redundant evaluate until fill confirms
            latency.record_ns("tick_total", t0)
        elif symbol not in state.watchlist:
            # Taken off the watchlist (by hand or auto-dropped) but the stream
            # subscription remains: keep the price fresh, never open a new trade.
            latency.record_ns("tick_total", t0)
        elif portfolio_manager.owns_entries():
            # The portfolio manager evaluates, ranks and enters (engine/portfolio_manager.py).
            # This path only keeps price and indicators fresh; if the manager stalls,
            # owns_entries() turns False and the branch below takes entries back.
            latency.record_ns("tick_total", t0)
        else:
            # Run Manager Decision Engine for new entries
            t_d = time.perf_counter_ns()
            decision = decision_engine.evaluate(symbol, quant, sentiment)
            latency.record_ns("decision", t_d)
            exec_time_us = (time.perf_counter_ns() - t0) / 1000.0
            latency.record_us("tick_total", exec_time_us)

            # 5. If Actionable and bot is active, dispatch entry order
            # A market or symbol switched off in the UI raises no signal at all,
            # rather than a BUY the risk guard then silently refuses.
            from core.market_filter import market_filter
            if (state.is_trading_active and decision.action == "BUY"
                    and not market_filter.entry_block_reason(symbol)):
                state.log_event(
                    "SIGNAL",
                    f"{decision.action} signal for {symbol} triggered in {exec_time_us:.0f}µs: {decision.reason}"
                )
                asyncio.create_task(executor.execute_decision(decision))

    async def _run_simulated_stream(self):
        """
        Generates simulated ticks for the watchlist when no Alpaca keys are set.
        """
        base_prices: Dict[str, float] = {}      # every simulated symbol starts at 150
        # One simulated "minute" bar per symbol every few seconds, so the demo
        # moves at a watchable pace. Bars carry consecutive synthetic minutes,
        # started far enough in the past that they never run ahead of the clock.
        minute = int(time.time() // 60) - 200_000

        def bar_for(sym: str) -> tuple:
            p = base_prices.get(sym, 150.0)
            path = [p]
            for _ in range(4):
                path.append(max(path[-1] * (1.0 + random.gauss(0.0, 0.0006)), 0.1))
            base_prices[sym] = path[-1]
            return (round(path[0], 2), round(max(path), 2), round(min(path), 2),
                    round(path[-1], 2), float(random.randint(500, 20000)))

        # Warm-up history so indicators have bars to read
        for _ in range(60):
            for sym in list(base_prices):
                o, h, l, c, v = bar_for(sym)
                await self.on_bar_received(sym, minute, o, h, l, c, v)
            minute += 1

        while self._running:
            try:
                for sym in sorted(set(state.watchlist) | set(settings.CONTEXT_SYMBOLS)):
                    o, h, l, c, v = bar_for(sym)
                    half = round(c * 0.0002, 2)
                    self.on_quote(sym, c - half, c + half)
                    await self.on_bar_received(sym, minute, o, h, l, c, v)
                minute += 1
                await asyncio.sleep(2.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in simulated market stream: {e}")
                await asyncio.sleep(1.0)

    async def _run_stock_live_feed(self):
        """Stock WebSocket Data Stream (runs during market hours, maintains baseline off-hours)"""
        from alpaca.data.live import StockDataStream

        stocks = sorted(set(state.watchlist) | set(settings.CONTEXT_SYMBOLS))
        if not stocks and not state.active_positions:
            return

        try:
            stock_stream = StockDataStream(settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY)

            async def handle_stock_bar(bar):
                # Bars are stamped with their START; the price is its close, a
                # minute later. Age = now - bar end, i.e. pure delivery delay.
                start = bar.timestamp.timestamp()
                latency.record_us("feed_stock_bar_age", max(time.time() - start - 60.0, 0.0) * 1e6)
                await self.on_bar_received(
                    bar.symbol, int(start // 60), float(bar.open), float(bar.high),
                    float(bar.low), float(bar.close), float(bar.volume or 0.0))

            async def handle_stock_quote(q):
                bid, ask = float(q.bid_price or 0.0), float(q.ask_price or 0.0)
                if bid > 0 and ask >= bid:
                    self.on_quote(q.symbol, bid, ask)

            async def handle_stock_trade(trade):
                try:
                    latency.record_us("feed_stock_trade_age",
                                      max(time.time() - trade.timestamp.timestamp(), 0.0) * 1e6)
                except Exception:
                    pass
                await self.on_position_trade(trade.symbol, float(trade.price))

            self._stock_bar_handler = handle_stock_bar
            self._stock_trade_handler = handle_stock_trade
            self._stock_quote_handler = handle_stock_quote
            # Not running yet: this only fills the handlers, which alpaca-py
            # sends as its first subscribe once connected.
            async with self._sync_lock:
                await self._reconcile(stock_stream)
            self._stock_stream = stock_stream

            await stock_stream._run_forever()
        except Exception as e:
            logger.debug(f"Stock WebSocket off-hours/closed ({e}). Stocks will resume at 09:30 AM EST.")

    async def _run_premarket_quote_poller(self):
        """
        Pre-market prices for watchlist stocks. The minute-bar stream only emits
        when a trade prints on the feed, which before 09:30 is rarely, so stocks
        would sit without a price (and without a spread) all pre-market. Polls the
        latest real bid/ask instead; the quote spread feeds the spread gate.
        """
        from core.market_hours import us_session, PRE
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest

        client = StockHistoricalDataClient(settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY)
        loop = asyncio.get_running_loop()
        while self._running:
            try:
                stocks = sorted(state.watchlist)
                if settings.PREMARKET_TRADING_ENABLED and stocks and us_session() == PRE:
                    quotes = await loop.run_in_executor(
                        None, client.get_stock_latest_quote,
                        StockLatestQuoteRequest(symbol_or_symbols=stocks))
                    now = time.time()
                    for sym, q in quotes.items():
                        bid, ask = float(q.bid_price or 0), float(q.ask_price or 0)
                        ts = q.timestamp.timestamp() if q.timestamp else None
                        if not usable_quote(bid, ask, ts, now):
                            continue
                        # A live price, not an indicator sample: the poll's 15s
                        # cadence must not leak into the one-minute bar series.
                        self.on_quote(sym, bid, ask)
                        await self.on_tick_received(symbol=sym, price=(bid + ask) / 2,
                                                    bid=bid, ask=ask, volume=0.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Pre-market quote poll failed: {e}")
            await asyncio.sleep(settings.PREMARKET_QUOTE_POLL_SECONDS)

    async def _run_alpaca_websocket(self):
        """Runs the stock bar stream and the pre-market quote poller concurrently."""
        await asyncio.gather(
            self._run_stock_live_feed(),
            self._run_premarket_quote_poller(),
        )

_STOCK_SYMBOL = re.compile(r"^[A-Z]{1,5}(\.[A-Z])?$")


def stream_plan(held, context, watch, limit: int) -> list:
    """First `limit` distinct stock symbols of held, then context, then watch, in that order."""
    out = []
    for s in (*held, *context, *watch):
        if s not in out and _STOCK_SYMBOL.match(s):
            out.append(s)
    return out[:max(int(limit), 0)]


def usable_quote(bid: float, ask: float, ts: Optional[float], now: float) -> bool:
    """
    Whether a polled quote is a live, tradeable price. One-sided and crossed
    quotes are not. Neither is the "latest" quote the API returns when nothing
    has quoted today: the previous session's close, days old and ~10% wide. Its
    midpoint used to be fed in as live on every poll, so prices froze and entries
    and exits fired on a number nobody could trade at.
    """
    if bid <= 0 or ask <= 0 or ask < bid:
        return False
    if ts is None or now - ts > settings.PREMARKET_MAX_QUOTE_AGE_SECONDS:
        return False
    return (ask - bid) / ((ask + bid) / 2) <= settings.PREMARKET_MAX_QUOTE_SPREAD_PCT / 100.0


market_stream = MarketStreamRunner()
