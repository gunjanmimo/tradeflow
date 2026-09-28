import asyncio
import logging
import random
import time
from typing import Optional
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
        self._stock_task = None

    async def start(self):
        self._running = True
        # Check if we can connect to real Alpaca WebSocket stream
        if not settings.ALPACA_API_KEY.startswith("PK_PLACEHOLDER") and settings.ALPACA_API_KEY:
            self._live = True
            self._task = asyncio.create_task(self._run_alpaca_websocket())
        else:
            logger.info("Starting high-frequency simulated market stream (realistic ticks 24/7)...")
            self._task = asyncio.create_task(self._run_simulated_stream())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()

    async def ensure_stock_subscription(self, symbol: str):
        """
        Starts live bars for a stock added to the watchlist after startup (e.g. a
        promoted discovery candidate). The stream subscribes once at boot, so
        without this a promoted symbol would never receive a price until restart.
        The simulated stream reads the watchlist every loop and needs nothing.
        """
        if not (self._running and self._live):
            return
        stream = self._stock_stream
        if stream is None:
            if self._stock_task is None or self._stock_task.done():
                self._stock_task = asyncio.create_task(self._run_stock_live_feed())
            return
        await self._subscribe(stream, "bars", symbol, self._stock_bar_handler)

    async def ensure_position_stream(self, symbol: str):
        """
        Streams every trade print for a held stock. Minute bars left an open
        position's stop and target checked at most once a minute, after the bar
        closed; each print now reaches its sentinel within milliseconds.
        """
        if not (self._running and self._live):
            return
        stream = self._stock_stream
        if stream is None or self._stock_trade_handler is None:
            return  # the feed subscribes held positions itself when it starts
        await self._subscribe(stream, "trades", symbol, self._stock_trade_handler)

    async def _subscribe(self, stream, channel: str, symbol: str, handler):
        handlers = stream._handlers.get(channel)
        if handlers is None or symbol in handlers:
            return
        handlers[symbol] = handler
        if getattr(stream, "_running", False):
            try:
                # Awaited directly: alpaca-py's public subscribe blocks on a
                # future scheduled onto this same loop, which would deadlock.
                await stream._send_subscribe_msg()
            except Exception as e:
                logger.warning(f"Live {channel} subscribe for {symbol} failed: {e}")

    async def on_position_trade(self, symbol: str, price: float):
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
        from engine.sentinel_agent import sentinel_registry
        await sentinel_registry.dispatch_tick(symbol, price)
        latency.record_ns("position_trade", t0)

    async def on_tick_received(self, symbol: str, price: float, bid: float, ask: float, volume: float):
        """
        THE SUB-SECOND CRITICAL PATH:
        Total processing time: < 0.15 milliseconds!
        """
        t0 = time.perf_counter_ns()

        # 1. Update In-Memory State
        state.update_price(symbol, price, bid, ask, volume)

        # 2. Run Quant Matrix (t1...tN indicators)
        t_q = time.perf_counter_ns()
        quant = quant_matrix.evaluate_symbol(symbol)
        latency.record_ns("quant_matrix", t_q)

        # 3. Read Pre-computed Laya Sentiment from RAM
        sentiment = state.get_sentiment(symbol)

        # 4. If holding, dispatch tick to dedicated Sentinel Bot assigned to this trade
        from engine.executor import executor
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
        base_prices = {
            "NVDA": 128.50,
            "AAPL": 224.30,
            "TSLA": 252.10,
            "MSFT": 428.80,
            "PLTR": 42.10,
        }

        # Initialize base price history so indicators have starting candles
        for sym, p in base_prices.items():
            for _ in range(30):
                drift = random.uniform(-0.002, 0.002)
                p = round(p * (1.0 + drift), 2)
                state.update_price(sym, p, p - 0.02, p + 0.02, random.randint(100, 5000))
            base_prices[sym] = p

        while self._running:
            try:
                symbols = list(state.watchlist)
                for sym in symbols:
                    current = base_prices.get(sym, 150.0)
                    # Gaussian random walk with micro-drift
                    change_pct = random.gauss(0.0001, 0.002)
                    new_price = round(max(current * (1.0 + change_pct), 0.1), 2)
                    base_prices[sym] = new_price
                    
                    spread = round(new_price * 0.0005, 2)
                    bid = round(new_price - (spread / 2), 2)
                    ask = round(new_price + (spread / 2), 2)
                    vol = random.randint(50, 2000)

                    await self.on_tick_received(sym, new_price, bid, ask, vol)

                # Loop interval: 250ms per batch of watchlist ticks
                await asyncio.sleep(0.25)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in simulated market stream: {e}")
                await asyncio.sleep(1.0)

    async def _run_stock_live_feed(self):
        """Stock WebSocket Data Stream (runs during market hours, maintains baseline off-hours)"""
        from alpaca.data.live import StockDataStream

        stocks = list(state.watchlist)
        if not stocks and not state.active_positions:
            return

        try:
            stock_stream = StockDataStream(settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY)

            async def handle_stock_bar(bar):
                # Bars are stamped with their START; the price is its close, a
                # minute later. Age = now - bar end, i.e. pure delivery delay.
                try:
                    bar_end = bar.timestamp.timestamp() + 60.0
                    latency.record_us("feed_stock_bar_age", max(time.time() - bar_end, 0.0) * 1e6)
                except Exception:
                    pass
                await self.on_tick_received(
                    symbol=bar.symbol,
                    price=float(bar.close),
                    bid=float(bar.close * 0.9998),
                    ask=float(bar.close * 1.0002),
                    volume=float(bar.volume)
                )

            async def handle_stock_trade(trade):
                try:
                    latency.record_us("feed_stock_trade_age",
                                      max(time.time() - trade.timestamp.timestamp(), 0.0) * 1e6)
                except Exception:
                    pass
                await self.on_position_trade(trade.symbol, float(trade.price))

            held = list(state.active_positions)
            for s in sorted(set(stocks) | set(held)):
                stock_stream.subscribe_bars(handle_stock_bar, s)
            for s in held:
                stock_stream.subscribe_trades(handle_stock_trade, s)
            self._stock_stream = stock_stream
            self._stock_bar_handler = handle_stock_bar
            self._stock_trade_handler = handle_stock_trade

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
