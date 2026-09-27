import logging
import asyncio
import time
from typing import Optional, Dict, Any
from core.config import settings
from core.state import state, TradeDecision
from engine.risk_guard import risk_guard
from engine import brackets

logger = logging.getLogger("tradeflow.executor")

class AlpacaExecutor:
    """
    Sub-second order router and position manager for Alpaca Trading API.
    Uses bracket orders (Market entry + hard Stop-Loss + Take-Profit).
    """
    def __init__(self):
        self.trading_client = None
        self.is_connected = False
        self.is_mock_mode = False
        self.alpaca_tradable_symbols: set[str] = set()
        self.pending_orders: set[str] = set()
        # Symbols with a close/liquidation order currently in flight.
        # Guards against the sentinel firing a second close while the first
        # round-trip to the broker is still open (which would oversell into a short).
        self.closing_orders: set[str] = set()
        # symbol -> entry-context snapshot, merged into the position once the
        # broker sync reports the fill. Cleared when the position closes.
        self.pending_entry_context: Dict[str, Dict[str, Any]] = {}

    def is_symbol_tradable(self, symbol: str) -> bool:
        if self.is_mock_mode:
            return True
        sym = symbol.upper().strip()
        clean = sym.replace("/", "")
        return sym in self.alpaca_tradable_symbols or clean in self.alpaca_tradable_symbols

    async def initialize(self):
        """Initializes Alpaca TradingClient using official alpaca-py"""
        if settings.ALPACA_API_KEY.startswith("PK_PLACEHOLDER") or not settings.ALPACA_API_KEY:
            logger.warning("Alpaca API keys are placeholders. Operating in local simulated PAPER mode.")
            self.is_mock_mode = True
            state.log_event("WARNING", "Running in local simulated paper mode (add your Alpaca keys in .env for live Alpaca Paper connection)")
            return

        try:
            from alpaca.trading.client import TradingClient
            self.trading_client = TradingClient(
                api_key=settings.ALPACA_API_KEY,
                secret_key=settings.ALPACA_SECRET_KEY,
                paper=settings.IS_PAPER
            )
            # Sync initial account info & tradable asset catalog
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.sync_account_and_positions)
            from engine.sentinel_agent import sentinel_registry
            sentinel_registry.sync_with_positions(state.active_positions)
            
            def load_assets():
                try:
                    assets = self.trading_client.get_all_assets()
                    return {a.symbol for a in assets if a.tradable}
                except Exception as ex:
                    logger.debug(f"Could not load asset catalog: {ex}")
                    return set()

            def load_recent_orders():
                try:
                    from alpaca.trading.requests import GetOrdersRequest
                    from alpaca.trading.enums import QueryOrderStatus
                    req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=30)
                    orders = self.trading_client.get_orders(req)
                    recent = []
                    for o in reversed(orders):
                        if o.filled_at and o.filled_avg_price:
                            sym = o.symbol
                            if sym.endswith("USD") and "/" not in sym and len(sym) > 3:
                                sym = sym[:-3] + "/USD"
                            side_str = "BUY" if "BUY" in str(o.side).upper() else "SELL"
                            recent.append({
                                "time": o.filled_at.timestamp(),
                                "symbol": sym,
                                "side": side_str,
                                "qty": float(o.filled_qty or o.qty),
                                "price": float(o.filled_avg_price),
                                "mode": "ALPACA_PAPER"
                            })
                    return recent
                except Exception as ex:
                    logger.debug(f"Could not load recent orders: {ex}")
                    return []

            self.alpaca_tradable_symbols = await loop.run_in_executor(None, load_assets)
            past_orders = await loop.run_in_executor(None, load_recent_orders)
            for o in past_orders:
                state.recent_trades.append(o)

            self.is_connected = True
            self.is_mock_mode = False
            state.log_event("INFO", f"Connected to Alpaca Paper API ({len(self.alpaca_tradable_symbols)} tradable assets, {len(past_orders)} past fills loaded)")
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca API: {e}. Falling back to simulated paper mode.")
            self.is_mock_mode = True
            state.log_event("ERROR", f"Alpaca connection failed ({e}), using simulated paper mode.")

    def sync_account_and_positions(self):
        """Synchronizes account equity, cash, and active positions from Alpaca"""
        if self.is_mock_mode or not self.trading_client:
            return

        try:
            account = self.trading_client.get_account()
            state.account_info["equity"] = float(account.equity)
            state.account_info["cash"] = float(account.cash)
            state.account_info["buying_power"] = float(account.buying_power)
            state.update_peak_equity()
            state.roll_trading_day_if_needed()

            positions = self.trading_client.get_all_positions()
            new_positions = {}
            for pos in positions:
                sym = pos.symbol
                # Normalize crypto symbols: e.g. "BTCUSD" -> "BTC/USD"
                if sym.endswith("USD") and "/" not in sym and len(sym) > 3:
                    normalized_sym = sym[:-3] + "/USD"
                else:
                    normalized_sym = sym

                old_pos = state.active_positions.get(normalized_sym, {})
                qty = float(pos.qty)
                avg_entry = float(pos.avg_entry_price)
                current_price = float(pos.current_price)
                # Preserve the bracket we already hold; only derive one when the
                # position is new to us (e.g. opened before this process started).
                # Deriving from live ATR keeps it volatility-scaled instead of the
                # flat -2%/+4% this path used to stamp onto every position.
                sl = old_pos.get("stop_loss")
                tp = old_pos.get("take_profit")
                if not brackets.is_valid(avg_entry, sl, tp):
                    q = state.quant_metrics.get(normalized_sym)
                    sl, tp, _dist = brackets.derive(avg_entry, q.atr if q else None)
                invested = old_pos.get("invested_dollars") or round(qty * avg_entry, 2)
                alloc_pct = old_pos.get("allocated_pct") or round((invested / max(1.0, state.allocated_capital)) * 100, 1)
                risk = old_pos.get("dollar_risk") or round(abs(avg_entry - sl) * qty, 2)
                reward = old_pos.get("dollar_reward") or round(abs(tp - avg_entry) * qty, 2)

                ctx = self.pending_entry_context.pop(normalized_sym, None) or {}
                # Entry context already on the position wins over a pending snapshot
                merged_ctx = {**ctx, **{k: v for k, v in old_pos.items() if k in ctx}}

                new_positions[normalized_sym] = {
                    **merged_ctx,
                    "symbol": normalized_sym,
                    "qty": qty,
                    "avg_entry_price": avg_entry,
                    "current_price": current_price,
                    "unrealized_pl": float(pos.unrealized_pl),
                    "unrealized_plpc": float(pos.unrealized_plpc),
                    "stop_loss": sl,
                    "take_profit": tp,
                    "buy_prob": old_pos.get("buy_prob", 0.5),
                    "sell_prob": old_pos.get("sell_prob", 0.5),
                    "close_prob": old_pos.get("close_prob", 0.1),
                    "laya_pos": old_pos.get("laya_pos", 0.5),
                    "laya_neg": old_pos.get("laya_neg", 0.5),
                    "mode": old_pos.get("mode", "ALPACA_PAPER"),
                    "invested_dollars": invested,
                    "current_value": round(qty * current_price, 2),
                    "allocated_pct": alloc_pct,
                    "dollar_risk": risk,
                    "dollar_reward": reward,
                    "sell_plan": old_pos.get("sell_plan", f"SELL 100% ({qty}x) on SL/TP/Reversal"),
                    "sell_pct": old_pos.get("sell_pct", 100),
                    "sell_qty": old_pos.get("sell_qty", qty),
                    "bot_id": old_pos.get("bot_id", f"BOT-{normalized_sym.replace('/', '')}"),
                    "bot_thesis": old_pos.get("bot_thesis"),
                    "sentinel": old_pos.get("sentinel")
                }
            # Preserve simulated positions
            for sym, old in state.active_positions.items():
                if old.get("mode") in ("SIMULATED", "PAPER_SIMULATED") and sym not in new_positions:
                    new_positions[sym] = old
            state.active_positions = new_positions
        except Exception as e:
            logger.error(f"Error syncing account from Alpaca: {e}")

    async def execute_decision(self, decision: TradeDecision):
        """
        Executes a TradeDecision asynchronously.
        Takes ~30-60ms network roundtrip to Alpaca.
        """
        if not state.is_trading_active:
            return

        symbol = decision.symbol

        if decision.action == "BUY":
            await self._execute_buy(decision)
        elif decision.action == "CLOSE":
            await self._execute_close(decision)

    async def _execute_buy(self, decision: TradeDecision):
        symbol = decision.symbol
        if symbol in self.pending_orders or symbol in state.active_positions:
            return

        # 1. Risk check (evaluated before locking order in-flight)
        can_open, reason = risk_guard.can_open_position(symbol)
        if not can_open:
            state.log_event("RISK_REJECT", f"Cannot buy {symbol}: {reason}")
            return

        self.pending_orders.add(symbol)
        try:
            tick = state.latest_prices.get(symbol)
            quant = state.quant_metrics.get(symbol)
            if not tick:
                return

            # 2. Sizing calculation via Laya Allocation Manager
            atr = quant.atr if quant else 0.50
            sentiment = state.get_sentiment(symbol)
            from feeds.multi_source_aggregator import trend_aggregator
            # None when no real source backed this symbol. Defaulting to 0.70
            # (as this previously did) invented conviction out of nothing and
            # inflated position size for every unscored symbol.
            consensus_score = trend_aggregator.get_consensus(symbol)

            qty, stop_loss, take_profit, alloc = risk_guard.calculate_order_sizing(
                symbol=symbol,
                current_price=tick.price,
                atr=atr,
                laya_pos_prob=sentiment.pos_prob,
                consensus_score=consensus_score
            )
            if qty <= 0:
                state.log_event("RISK_REJECT", f"Order skipped for {symbol}: {alloc.get('rationale')}")
                return

            # --- Memory check: has this exact setup lost repeatedly before? ---
            # Consulted here rather than in the strategy because it is a
            # portfolio-level policy, not part of any single strategy's thesis.
            try:
                from memory.agent_memory import agent_memory
                if agent_memory.enabled:
                    strat_name = (state.last_gate_detail.get(symbol) or {}).get("strategy", "unknown")
                    from engine.strategies import registry as _reg
                    _strat = _reg.resolve(symbol, state.strategy_class_defaults,
                                          state.strategy_overrides)
                    verdict = await agent_memory.should_avoid(
                        strategy=_strat.name, symbol=symbol,
                        rsi=quant.rsi if quant else None,
                        trend_score=_strat._trend_score(quant, tick.price) if quant else None,
                        sent_pos=sentiment.pos_prob, sent_n=sentiment.n_headlines,
                        atr_pct=(quant.atr / tick.price * 100) if (quant and quant.atr) else None,
                    )
                    if verdict["avoid"]:
                        state.log_event("MEMORY_VETO", f"{symbol}: {verdict['reason']}")
                        return
            except Exception as e:
                # Memory must never block a trade by failing.
                logger.debug(f"Memory check skipped for {symbol}: {e}")

            state.log_event("ALLOCATION", alloc["rationale"])

            # Snapshot the conditions that justified this entry. Without it, a closed
            # trade tells you the PnL but not what the engine believed at the time,
            # so no attribution (or learning) is possible after the fact.
            entry_context = {
                "opened_at": time.time(),
                "laya_pos": sentiment.pos_prob,
                "laya_neg": sentiment.neg_prob,
                "sentiment_headline": sentiment.headline,
                "sentiment_age_s": state.sentiment_age(symbol),
                "consensus_score": consensus_score,
                "conviction_tier": alloc.get("conviction_tier"),
                "composite_conviction": alloc.get("composite_conviction"),
                "entry_rsi": quant.rsi if quant else None,
                "entry_spread": quant.spread if quant else None,
                "entry_atr_pct": round((quant.atr / tick.price) * 100, 4) if (quant and quant.atr and tick.price) else None,
                "stop_pct": round((abs(tick.price - stop_loss) / tick.price) * 100, 3) if tick.price else None,
                "dollar_risk": alloc.get("risk_dollars"),
                "dollar_reward": alloc.get("reward_dollars"),
                "buy_prob": decision.buy_prob,
                "entry_reason": decision.reason,
            }

            t0 = time.time()
            is_alpaca_tradable = (
                symbol in self.alpaca_tradable_symbols or 
                symbol.replace("/", "") in self.alpaca_tradable_symbols
            )

            if self.is_mock_mode or not is_alpaca_tradable:
                mode_tag = "SIMULATED" if self.is_mock_mode else "PAPER_SIMULATED"
                state.active_positions[symbol] = {
                    "symbol": symbol,
                    "qty": qty,
                    "avg_entry_price": tick.price,
                    "current_price": tick.price,
                    "unrealized_pl": 0.0,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "allocated_dollars": alloc["allocated_dollars"],
                    "allocated_pct": alloc["allocated_pct"],
                    "invested_dollars": alloc["allocated_dollars"],
                    "mode": mode_tag,
                    **entry_context,
                }
                order_cost = round(qty * tick.price, 2)
                state.account_info["cash"] -= order_cost
                state.recent_trades.append({
                    "time": time.time(),
                    "symbol": symbol,
                    "side": "BUY",
                    "qty": qty,
                    "price": tick.price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "mode": mode_tag
                })
                state.log_event("ORDER_FILLED", f"[{mode_tag}] BUY {qty}x {symbol} @ ${tick.price:.2f} (SL: ${stop_loss}, TP: ${take_profit})")
                return

            # Real Alpaca API execution
            from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
            from alpaca.trading.enums import OrderSide, TimeInForce
            from core.state import is_crypto_symbol

            if is_crypto_symbol(symbol):
                # Crypto uses direct GTC market order (stop-loss and take-profit managed in-engine)
                req = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC
                )
            else:
                # Equities use broker-native bracket order
                req = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC,
                    take_profit=TakeProfitRequest(limit_price=take_profit),
                    stop_loss=StopLossRequest(stop_price=stop_loss)
                )

            loop = asyncio.get_running_loop()
            order = await loop.run_in_executor(None, self.trading_client.submit_order, req)
            latency_ms = (time.time() - t0) * 1000

            # Hold the entry context until the broker sync materialises the position,
            # so the closed-trade record can still report what justified the entry.
            self.pending_entry_context[symbol] = entry_context

            # Deduct cash locally and trigger async broker sync
            order_cost = round(qty * tick.price, 2)
            state.account_info["cash"] = max(0.0, state.account_info["cash"] - order_cost)
            loop.run_in_executor(None, self.sync_account_and_positions)

            state.recent_trades.append({
                "time": time.time(),
                "symbol": symbol,
                "side": "BUY",
                "qty": qty,
                "price": tick.price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "order_id": str(order.id),
                "mode": "ALPACA_PAPER"
            })
            state.log_event("ORDER_SUBMITTED", f"Alpaca BUY {qty}x {symbol} submitted in {latency_ms:.1f}ms (ID: {order.id})")
        except Exception as e:
            err_str = str(e)
            logger.error(f"Alpaca order submission error: {err_str}")
            state.log_event("ORDER_ERROR", f"Failed to place BUY for {symbol}: {err_str}")
            if "insufficient balance" in err_str or "balance" in err_str:
                loop = asyncio.get_running_loop()
                loop.run_in_executor(None, self.sync_account_and_positions)
        finally:
            self.pending_orders.discard(symbol)

    async def _execute_close(self, decision: TradeDecision):
        symbol = decision.symbol
        if symbol not in state.active_positions:
            return

        # Claim the close before any await. The sentinel re-evaluates on every tick
        # and on a 1s heartbeat, so while a close is in flight it would otherwise
        # keep firing duplicate liquidations for the same position.
        if symbol in self.closing_orders:
            return
        self.closing_orders.add(symbol)
        try:
            await self._execute_close_locked(decision)
        finally:
            self.closing_orders.discard(symbol)

    async def _execute_close_locked(self, decision: TradeDecision):
        symbol = decision.symbol
        tick = state.latest_prices.get(symbol)
        pos = state.active_positions.get(symbol)
        if not pos:
            return
        t0 = time.time()

        is_simulated = self.is_mock_mode or pos.get("mode") in ("SIMULATED", "PAPER_SIMULATED")
        if is_simulated:
            price = tick.price if tick else pos["avg_entry_price"]
            pnl = round((price - pos["avg_entry_price"]) * pos["qty"], 2)
            state.account_info["cash"] += round(pos["qty"] * price, 2)
            state.account_info["equity"] += pnl
            state.book_realized_pnl(symbol, pnl)
            state.active_positions.pop(symbol, None)
            self._record_closed_trade(symbol, pos, price, pnl, decision)

            state.recent_trades.append({
                "time": time.time(),
                "symbol": symbol,
                "side": "SELL",
                "qty": pos["qty"],
                "price": price,
                "pnl": pnl,
                "mode": pos.get("mode", "SIMULATED")
            })
            state.log_event("ORDER_CLOSED", f"[{pos.get('mode', 'SIMULATED')}] CLOSED {symbol} @ ${price:.2f} (PnL: ${pnl:+.2f})")
            return

        try:
            from core.state import is_crypto_symbol
            # Alpaca API expects 'BTCUSD' instead of 'BTC/USD' for position closing route
            alpaca_sym = symbol.replace("/", "") if is_crypto_symbol(symbol) else symbol
            
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.trading_client.close_position, alpaca_sym)
            latency_ms = (time.time() - t0) * 1000
            # Book the position's last-known unrealised PnL as realised. The exact
            # fill price arrives on the next broker sync; this keeps the daily-loss
            # breaker responsive rather than blind until that sync lands.
            exit_price = tick.price if tick else float(pos.get("current_price") or pos["avg_entry_price"])
            pnl = round(float(pos.get("unrealized_pl") or
                              (exit_price - pos["avg_entry_price"]) * pos["qty"]), 2)
            state.book_realized_pnl(symbol, pnl)
            state.active_positions.pop(symbol, None)
            self._record_closed_trade(symbol, pos, exit_price, pnl, decision)
            state.log_event("ORDER_CLOSED",
                            f"Alpaca CLOSED {symbol} in {latency_ms:.1f}ms (PnL ${pnl:+,.2f}, "
                            f"day ${state.realized_pnl_today:+,.2f})")
            loop.run_in_executor(None, self.sync_account_and_positions)
        except Exception as e:
            err_str = str(e)
            logger.error(f"Failed to close position in {symbol}: {err_str}")
            # If position was already closed or not found on broker, remove from active positions to stop retry loop
            if "Not Found" in err_str or "not found" in err_str or "404" in err_str:
                state.active_positions.pop(symbol, None)
                state.log_event("ORDER_SYNC", f"Position {symbol} already liquidated on broker.")
                loop = asyncio.get_running_loop()
                loop.run_in_executor(None, self.sync_account_and_positions)
            else:
                state.log_event("ORDER_ERROR", f"Failed to close {symbol}: {err_str}")

    def _record_closed_trade(self, symbol: str, pos: Dict[str, Any], exit_price: float,
                             pnl: float, decision: Optional[TradeDecision] = None):
        """
        Writes a structured outcome record for a closed position.

        This is the substrate for attribution and for any learning loop: the reward
        signal is the R-MULTIPLE (pnl / initial risk), not raw dollars, so outcomes
        stay comparable across position sizes and asset classes. The setup context
        is captured at close so a later analysis can ask "which conditions actually
        preceded profit" rather than guessing.
        """
        entry = float(pos.get("avg_entry_price") or 0.0)
        qty = float(pos.get("qty") or 0.0)
        initial_risk = float(pos.get("dollar_risk") or 0.0)
        if initial_risk <= 0 and pos.get("stop_loss") and entry:
            initial_risk = abs(entry - float(pos["stop_loss"])) * qty

        r_multiple = round(pnl / initial_risk, 3) if initial_risk > 0 else None
        held_seconds = round(time.time() - float(pos.get("opened_at") or time.time()), 1)

        record = {
            "time": time.time(),
            "symbol": symbol,
            "side": "SELL",
            "qty": qty,
            "price": exit_price,
            "entry_price": entry,
            "pnl": pnl,
            "initial_risk": round(initial_risk, 2),
            "r_multiple": r_multiple,
            "held_seconds": held_seconds,
            "exit_reason": (decision.reason if decision else "") or pos.get("bot_thesis", ""),
            "mode": pos.get("mode", "ALPACA_PAPER"),
            # Setup context as it stood at entry, for later attribution
            "setup": {
                "laya_pos": pos.get("laya_pos"),
                "laya_neg": pos.get("laya_neg"),
                "consensus_score": pos.get("consensus_score"),
                "conviction_tier": pos.get("conviction_tier"),
                "entry_rsi": pos.get("entry_rsi"),
                "entry_spread": pos.get("entry_spread"),
                "entry_atr_pct": pos.get("entry_atr_pct"),
                "stop_pct": pos.get("stop_pct"),
            },
        }
        state.recent_trades.append(record)
        state.closed_trades.append(record)

        # Persist to agent memory WITHOUT awaiting: a slow graph write must never
        # delay a liquidation. Failures are logged inside remember_trade.
        record["strategy"] = pos.get("strategy")
        record["bot_id"] = pos.get("bot_id")
        record["opened_at"] = pos.get("opened_at")
        try:
            from memory.agent_memory import agent_memory
            if agent_memory.enabled:
                asyncio.create_task(agent_memory.remember_trade(record))
        except Exception as e:
            logger.debug(f"Could not schedule memory write: {e}")
        state.log_event(
            "TRADE_CLOSED",
            f"{symbol} closed: PnL ${pnl:+,.2f}"
            + (f" ({r_multiple:+.2f}R)" if r_multiple is not None else "")
            + f" after {held_seconds:.0f}s"
        )
        return record

    async def start_sync_loop(self):
        """Continuously keeps account cash and positions synchronized with Alpaca every 5 seconds"""
        while True:
            try:
                if not self.is_mock_mode and self.trading_client:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, self.sync_account_and_positions)
                    from engine.sentinel_agent import sentinel_registry
                    sentinel_registry.sync_with_positions(state.active_positions)
                await asyncio.sleep(5.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Account sync loop note: {e}")
                await asyncio.sleep(5.0)

    async def emergency_close_all(self):
        """Panic / Kill Switch: Instantly closes all active positions and pauses trading"""
        state.is_trading_active = False
        state.log_event("KILL_SWITCH", "EMERGENCY KILL SWITCH TRIGGERED: Halting trading and closing positions")

        if self.is_mock_mode:
            for sym in list(state.active_positions.keys()):
                await self._execute_close(TradeDecision(symbol=sym, action="CLOSE", close=True))
            return

        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.trading_client.close_all_positions, True)
            state.active_positions.clear()
            state.log_event("KILL_SWITCH", "All positions liquidated on Alpaca.")
            await loop.run_in_executor(None, self.sync_account_and_positions)
        except Exception as e:
            logger.error(f"Error executing emergency close all: {e}")

executor = AlpacaExecutor()
