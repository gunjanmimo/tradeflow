import asyncio
import logging
import random
import time
from core.config import settings
from core.state import state
from core.latency import latency
from engine.quant_matrix import quant_matrix
from engine.decision_engine import decision_engine
from engine.executor import executor

logger = logging.getLogger("tradeflow.stream")

class MarketStreamRunner:
    """
    The High-Speed Continuous Trading Loop.
    Evaluates ticks in sub-millisecond time.
    """
    def __init__(self):
        self._running = False
        self._task = None

    async def start(self):
        self._running = True
        # Check if we can connect to real Alpaca WebSocket stream
        if not settings.ALPACA_API_KEY.startswith("PK_PLACEHOLDER") and settings.ALPACA_API_KEY:
            self._task = asyncio.create_task(self._run_alpaca_websocket())
        else:
            logger.info("Starting high-frequency simulated market stream (realistic ticks 24/7)...")
            self._task = asyncio.create_task(self._run_simulated_stream())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()

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
        else:
            # Run Manager Decision Engine for new entries
            t_d = time.perf_counter_ns()
            decision = decision_engine.evaluate(symbol, quant, sentiment)
            latency.record_ns("decision", t_d)
            exec_time_us = (time.perf_counter_ns() - t0) / 1000.0
            latency.record_us("tick_total", exec_time_us)

            # 5. If Actionable and bot is active, dispatch entry order
            if state.is_trading_active and decision.action == "BUY":
                state.log_event(
                    "SIGNAL",
                    f"{decision.action} signal for {symbol} triggered in {exec_time_us:.0f}µs: {decision.reason}"
                )
                asyncio.create_task(executor.execute_decision(decision))

    async def _run_simulated_stream(self):
        """
        Generates realistic high-frequency ticks for the active watchlist.
        Ensures continuous trading testing works anywhere anytime.
        """
        base_prices = {
            "NVDA": 128.50,
            "AAPL": 224.30,
            "TSLA": 252.10,
            "MSFT": 428.80,
            "PLTR": 42.10,
            "BTC/USD": 84950.00,
            "ETH/USD": 2715.00,
            "SOL/USD": 124.10,
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

    async def _run_crypto_live_feed(self):
        """24/7 Real-Time Crypto Data Feed from Alpaca + Binance Public Tickers"""
        import urllib.request
        import json
        from core.state import is_crypto_symbol
        from alpaca.data.historical.crypto import CryptoHistoricalDataClient
        from alpaca.data.requests import CryptoLatestQuoteRequest

        client = None
        try:
            client = CryptoHistoricalDataClient(settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY)
            logger.info("24/7 Alpaca Crypto Live Feed initialized.")
        except Exception as e:
            logger.warning(f"Alpaca crypto client setup warning: {e}")

        def fetch_binance_prices_sync(symbols_list):
            try:
                # Fast targeted query for active watchlist
                binance_symbols = [s.replace("/", "").replace("USD", "USDT") for s in symbols_list]
                query = json.dumps(binance_symbols).replace(" ", "")
                url = f"https://api.binance.com/api/v3/ticker/price?symbols={urllib.parse.quote(query)}"
                req = urllib.request.Request(url, headers={"User-Agent": "TradeFlow/1.0"})
                with urllib.request.urlopen(req, timeout=2) as resp:
                    data = json.loads(resp.read().decode())
                    return {item["symbol"]: float(item["price"]) for item in data}
            except Exception:
                try:
                    # Fallback to full endpoint
                    url = "https://api.binance.com/api/v3/ticker/price"
                    req = urllib.request.Request(url, headers={"User-Agent": "TradeFlow/1.0"})
                    with urllib.request.urlopen(req, timeout=3) as resp:
                        data = json.loads(resp.read().decode())
                        return {item["symbol"]: float(item["price"]) for item in data}
                except Exception as e:
                    logger.debug(f"Binance price poll note: {e}")
                    return {}

        while self._running:
            cryptos = [s for s in state.watchlist if is_crypto_symbol(s)]
            if cryptos:
                loop = asyncio.get_running_loop()
                t_poll = time.perf_counter_ns()
                binance_map = await loop.run_in_executor(None, fetch_binance_prices_sync, cryptos)
                # A polled price is at least one round-trip old when it lands.
                latency.record_ns("feed_crypto_poll", t_poll)
                
                for sym in cryptos:
                    binance_pair = sym.replace("/", "").replace("USD", "USDT")
                    price = binance_map.get(binance_pair)
                    if price and price > 0:
                        precision = 8 if price < 0.0001 else (4 if price < 1.0 else 2)
                        spread = price * 0.0004
                        await self.on_tick_received(
                            symbol=sym,
                            price=round(price, precision),
                            bid=round(price - (spread / 2), precision),
                            ask=round(price + (spread / 2), precision),
                            volume=1000.0
                        )

            await asyncio.sleep(0.5)

    async def _run_stock_live_feed(self):
        """Stock WebSocket Data Stream (runs during market hours, maintains baseline off-hours)"""
        from core.state import is_crypto_symbol
        from alpaca.data.live import StockDataStream

        stocks = [s for s in state.watchlist if not is_crypto_symbol(s)]
        if not stocks:
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

            for s in stocks:
                stock_stream.subscribe_bars(handle_stock_bar, s)

            await stock_stream._run_forever()
        except Exception as e:
            logger.debug(f"Stock WebSocket off-hours/closed ({e}). Stocks will resume at 09:30 AM EST.")

    async def _run_alpaca_websocket(self):
        """Runs 24/7 crypto and stock market streams concurrently without blocking"""
        await asyncio.gather(
            self._run_crypto_live_feed(),
            self._run_stock_live_feed()
        )

market_stream = MarketStreamRunner()
