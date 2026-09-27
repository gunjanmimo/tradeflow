import asyncio
import logging
import time
from typing import Dict, Any, Optional
from core.state import state, TradeDecision
from core.config import settings
from engine import brackets
from core.latency import latency

logger = logging.getLogger("tradeflow.sentinel")

class PositionSentinelBot:
    """
    Dedicated Autonomous Micro-Agent assigned exclusively to a single open trade.
    
    Responsibilities:
    - Continuously watches over ONLY this specific position in sub-millisecond time.
    - Monitors real-time tick-by-tick micro-price action for this symbol.
    - Manages dynamic trailing stop-loss (locks in profit as price reaches new highs).
    - Queries Laya sentiment specifically for this asset.
    - Computes real-time Buy / Sell / Close probabilities out of 100.
    - Autonomously executes the exit when risk limits or profit targets are hit.
    """
    def __init__(self, symbol: str, position_data: Dict[str, Any]):
        self.symbol = symbol
        self.bot_id = f"BOT-{symbol.replace('/', '').upper()}-{int(time.time()) % 10000:04d}"
        self.assigned_at = time.time()
        self.entry_price = float(position_data.get("avg_entry_price", position_data.get("current_price", 0.0)))
        self.highest_price = self.entry_price
        self.last_price = float(position_data.get("current_price") or self.entry_price)
        self.qty = float(position_data.get("qty", 0.0))
        # Initial Stop Loss & Take Profit brackets, via the shared derivation.
        # A private fallback here previously rounded to 2dp, which produced a 0.00
        # take-profit for sub-cent assets -- read as "already hit" on the first tick.
        q = state.quant_metrics.get(symbol)
        brackets.ensure(position_data, price=self.entry_price, atr=q.atr if q else None)
        self.stop_loss = float(position_data["stop_loss"])
        self.take_profit = float(position_data["take_profit"])

        # Capital Allocation & Sizing Intelligence (BUY SIZING)
        self.invested_dollars = float(position_data.get("allocated_dollars") or round(self.qty * self.entry_price, 2))
        budget = float(state.allocated_capital or 10000.0)
        self.allocated_pct = float(position_data.get("allocated_pct") or round((self.invested_dollars / budget) * 100, 2))
        self.dollar_risk = round(max(0.0, (self.entry_price - self.stop_loss) * self.qty), 2)
        self.dollar_reward = round(max(0.0, (self.take_profit - self.entry_price) * self.qty), 2)

        # Divestment Policy & Sizing Intelligence (SELL SIZING)
        self.sell_pct = 100
        self.sell_qty = self.qty
        self.sell_plan = f"SELL 100% ({self.qty}x = ${self.invested_dollars:,.2f}) on SL/TP/Reversal"

        # Real-time state
        self.status = "WATCHING"
        self.evaluations_count = 0
        self.buy_prob = 0.5
        self.sell_prob = 0.2
        self.close_prob = 0.1
        self.thesis = f"Assigned to {self.symbol}: ${self.invested_dollars:,.2f} invested ({self.allocated_pct}% of budget). Guarding capital."
        self.is_running = True
        self._task = None
        # Latest quant-council read on this position. Computed by the analysis
        # worker process; the bot only reads it, so it costs nothing per tick.
        self.council: Optional[Dict[str, Any]] = None
        self._council_seen_at = 0.0

    def start(self):
        try:
            loop = asyncio.get_running_loop()
            self._task = loop.create_task(self._sentinel_loop())
        except RuntimeError:
            self._task = None
        # Give this agent an identity in the memory graph so its trades have an owner.
        try:
            from memory.agent_memory import agent_memory
            from engine.strategies import registry as _reg
            if agent_memory.enabled:
                _strat = _reg.for_position(self.symbol, state.active_positions.get(self.symbol),
                                           state.strategy_class_defaults,
                                           state.strategy_overrides)
                loop = asyncio.get_running_loop()
                loop.create_task(agent_memory.remember_bot(
                    self.bot_id, self.symbol, _strat.name))
        except Exception:
            pass

        state.log_event(
            "AGENT_SPAWNED",
            f"Dedicated Sentinel Agent [{self.bot_id}] deployed to manage {self.symbol} "
            f"(Entry: ${self.entry_price:,.6g})"
        )

    def stop(self):
        self.is_running = False
        if self._task:
            self._task.cancel()
        state.log_event(
            "AGENT_RETIRED", 
            f"Sentinel Agent [{self.bot_id}] completed watch for {self.symbol}"
        )

    async def on_tick(self, price: float):
        """Called on every single price tick for this symbol."""
        if not self.is_running:
            return
        t0 = time.perf_counter_ns()
        try:
            await self._on_tick(price)
        finally:
            latency.record_ns("sentinel_tick", t0)

    async def _on_tick(self, price: float):

        price = float(price)
        last_price = self.last_price or price
        tick_delta_pct = (price - last_price) / last_price if last_price > 0 else 0.0
        self.last_price = price

        self.evaluations_count += 1
        pos = state.active_positions.get(self.symbol)
        if not pos:
            self.status = "CLOSED"
            self.stop()
            return

        pnl_pct = (price - self.entry_price) / self.entry_price if self.entry_price > 0 else 0.0

        # 1. Track High-Water Mark for dynamic trailing stop
        if price > self.highest_price:
            self.highest_price = price
            # Trail stop upward: lock in gains when in profit > 0.8%
            new_trailing_stop = brackets.trail(pos, self.highest_price, self.entry_price)
            if new_trailing_stop is not None and new_trailing_stop > self.stop_loss:
                self.stop_loss = new_trailing_stop
                pos["stop_loss"] = self.stop_loss
                self.thesis = (f"Trailing Stop raised to ${self.stop_loss:,.6g} "
                               f"(Locking gains at peak ${self.highest_price:,.6g})")

        drawdown_from_peak = (self.highest_price - price) / self.highest_price if self.highest_price > 0 else 0.0

        # 2. Get quant metrics & Laya semantic sentiment for this specific asset
        quant = state.quant_metrics.get(self.symbol)
        sentiment = state.get_sentiment(self.symbol)

        # 3. Dynamic Price-Action Responsive Modulations:
        # A) BUY Probability (0 - 100): Long Continuation & Momentum
        # Price surging above entry boosts buy probability; falling below entry reduces it
        pnl_gain_factor = min(max(pnl_pct * 12.0, -0.35), 0.25)
        vel_factor = min(max(tick_delta_pct * 80.0, -0.10), 0.10)
        drawdown_penalty = min(drawdown_from_peak * 8.0, 0.30)
        
        trend_score = 0.5
        if quant and quant.ema_fast and quant.ema_slow:
            trend_score = 0.80 if (quant.ema_fast > quant.ema_slow and price > quant.ema_fast) else 0.20
        rsi_score = 0.5
        if quant and quant.rsi is not None:
            if quant.rsi < 35:
                rsi_score = 0.75
            elif quant.rsi > 70:
                rsi_score = 0.20
            elif 45 <= quant.rsi <= 65:
                rsi_score = 0.70

        raw_buy = (
            0.45 * sentiment.pos_prob + 
            0.25 * (0.5 + pnl_gain_factor) + 
            0.15 * (0.5 + vel_factor) + 
            0.15 * (0.6 * trend_score + 0.4 * rsi_score) - 
            drawdown_penalty
        )
        self.buy_prob = round(float(min(max(raw_buy, 0.05), 0.98)), 4)

        # B) SELL Probability (0 - 100): Reversal & Fading Risk
        # Pullback from high or drop below entry accelerates sell probability
        giveback_factor = min(drawdown_from_peak * 18.0, 0.45)
        loss_factor = min(max(-pnl_pct * 18.0, 0.0), 0.40)
        tick_down = max(-tick_delta_pct * 60.0, 0.0)
        overbought = 0.15 if (quant and quant.rsi and quant.rsi > 70) else 0.0

        raw_sell = (
            0.40 * sentiment.neg_prob + 
            0.25 * giveback_factor + 
            0.20 * loss_factor + 
            0.15 * (1.0 - trend_score) + 
            tick_down + 
            overbought
        )
        self.sell_prob = round(float(min(max(raw_sell, 0.05), 0.95)), 4)

        # C) CLOSE Probability (0 - 100): Stop-Loss / Take-Profit Proximity & Urgent Exit
        hit_stop = price <= self.stop_loss
        hit_tp = price >= self.take_profit
        strong_sell_signal = self.sell_prob >= settings.MAX_SELL_SENTIMENT_NEG

        sl_urgency = 0.0
        if price < self.entry_price and self.stop_loss < self.entry_price:
            total_sl_dist = self.entry_price - self.stop_loss
            drop = self.entry_price - price
            sl_ratio = min(max(drop / total_sl_dist, 0.0), 1.0)
            sl_urgency = 0.15 + 0.85 * (sl_ratio ** 1.3)

        tp_urgency = 0.0
        if price > self.entry_price and self.take_profit > self.entry_price:
            total_tp_dist = self.take_profit - self.entry_price
            gain = price - self.entry_price
            tp_ratio = min(max(gain / total_tp_dist, 0.0), 1.0)
            tp_urgency = 0.10 + 0.90 * (tp_ratio ** 1.4)

        raw_close = max(sl_urgency, tp_urgency, self.sell_prob * 0.85)
        if hit_stop or hit_tp or strong_sell_signal:
            raw_close = 1.0
        self.close_prob = round(float(min(max(raw_close, 0.02), 1.0)), 4)

        # Dynamic SELL Sizing: Understand how much to divest based on current price action
        current_value = round(self.qty * price, 2)
        if hit_stop:
            self.sell_pct = 100
            self.sell_qty = self.qty
            self.sell_plan = f"SELL 100% ({self.qty}x = ${current_value:,.2f}) [STOP LOSS HIT]"
        elif hit_tp:
            self.sell_pct = 100
            self.sell_qty = self.qty
            self.sell_plan = f"SELL 100% ({self.qty}x = ${current_value:,.2f}) [TAKE PROFIT HIT]"
        elif strong_sell_signal:
            self.sell_pct = 100
            self.sell_qty = self.qty
            self.sell_plan = f"SELL 100% ({self.qty}x = ${current_value:,.2f}) [BEARISH REVERSAL]"
        elif self.sell_prob >= 0.40:
            self.sell_pct = 50
            self.sell_qty = round(self.qty * 0.50, 4)
            self.sell_plan = f"SELL 50% ({self.sell_qty}x = ${round(self.sell_qty * price, 2):,.2f}) [DE-RISKING TRIM]"
        elif pnl_pct >= 0.012:
            self.sell_pct = 50
            self.sell_qty = round(self.qty * 0.50, 4)
            self.sell_plan = f"SELL 50% ({self.sell_qty}x = ${round(self.sell_qty * price, 2):,.2f}) [PROFIT-TAKE TRIM]"
        else:
            self.sell_pct = 100
            self.sell_qty = self.qty
            self.sell_plan = f"Target Sell: 100% ({self.qty}x = ${current_value:,.2f}) | SL ${self.stop_loss:,.2f} | TP ${self.take_profit:,.2f}"

        pnl_dollars = round((price - self.entry_price) * self.qty, 2)
        self.thesis = f"Invested: ${self.invested_dollars:,.2f} ({self.allocated_pct}% cap) | PnL: {pnl_pct*100:+.2f}% (${pnl_dollars:+.2f}) | {self.sell_plan}"

        # Sync telemetry with position dict
        pos["bot_id"] = self.bot_id
        pos["current_price"] = price
        pos["unrealized_pl"] = pnl_dollars
        pos["unrealized_plpc"] = round(pnl_pct, 5)
        pos["buy_prob"] = self.buy_prob
        pos["sell_prob"] = self.sell_prob
        pos["close_prob"] = self.close_prob
        pos["stop_loss"] = self.stop_loss
        pos["take_profit"] = self.take_profit
        pos["invested_dollars"] = self.invested_dollars
        pos["current_value"] = current_value
        pos["allocated_pct"] = self.allocated_pct
        pos["dollar_risk"] = self.dollar_risk
        pos["dollar_reward"] = self.dollar_reward
        pos["sell_plan"] = self.sell_plan
        pos["sell_pct"] = self.sell_pct
        pos["sell_qty"] = self.sell_qty
        pos["laya_pos"] = sentiment.pos_prob
        pos["laya_neg"] = sentiment.neg_prob
        pos["bot_thesis"] = self.thesis

        # 3b. Delegate the discretionary exit to the position's own strategy, so a
        # momentum trade exits on trend break and a news trade on catalyst reversal,
        # instead of every position sharing one hardcoded sell formula.
        strategy_exit = False
        strategy_reason = ""
        council_exit = False
        council_reason = ""
        try:
            from engine.strategies import registry as _registry
            from engine.strategies.context import build_context
            # The strategy that OPENED this trade governs its exit, not whatever the
            # symbol's routing says now.
            _strat = _registry.for_position(self.symbol, pos, state.strategy_class_defaults,
                                            state.strategy_overrides)
            _ctx = build_context(self.symbol, price=price, position=pos,
                                 highest_price=self.highest_price,
                                 quant=quant, sentiment=sentiment)
            _res = _strat.evaluate_exit(_ctx)
            strategy_exit = _res.should_close
            strategy_reason = _res.reason
            self.sell_prob = _res.sell_prob
            self.close_prob = max(self.close_prob, _res.close_prob)
            pos["strategy"] = _strat.name

            council_exit, council_reason = self._council_check()
        except Exception as _e:
            logger.error(f"[{self.bot_id}] strategy exit evaluation failed: {_e}")

        # 4. Autonomous Exit Execution
        if hit_stop or hit_tp or strong_sell_signal or strategy_exit or council_exit:
            self.status = "TRIGGERING_EXIT"
            if hit_stop:
                reason = f"[{self.bot_id}] Stop-loss triggered: Price ${price:,.6g} <= SL ${self.stop_loss:,.6g}"
            elif hit_tp:
                reason = f"[{self.bot_id}] Take-profit triggered: Price ${price:,.6g} >= TP ${self.take_profit:,.6g}"
            elif strategy_exit:
                reason = f"[{self.bot_id}] Strategy exit: {strategy_reason}"
            elif council_exit:
                reason = f"[{self.bot_id}] Council exit: {council_reason}"
            else:
                reason = f"[{self.bot_id}] Model exit triggered: sell_prob={self.sell_prob:.2f} (Laya={sentiment.neg_prob:.2f})"

            self.thesis = reason
            state.log_event("SIGNAL", f"Sentinel [{self.bot_id}] CLOSE signal for {self.symbol}: {reason}")
            
            from engine.executor import executor
            asyncio.create_task(executor.execute_decision(TradeDecision(
                symbol=self.symbol,
                action="CLOSE",
                buy_prob=self.buy_prob,
                sell_prob=self.sell_prob,
                close_prob=1.0,
                close=True,
                reason=reason
            )))

    def _council_check(self):
        """
        Reads the worker's latest council report for this symbol. Returns
        (should_exit, reason).

        Exits only on a decisive bearish consensus among regime-suited strategies
        -- a stricter bar than the manager's entry veto, because closing a working
        trade on a marginal read churns fees for nothing. Each report is acted on
        once, when it first arrives.
        """
        from core.config import settings
        a = state.fresh_analysis(self.symbol)
        report = (a or {}).get("council")
        if not report or a["at"] <= self._council_seen_at:
            return False, ""
        self._council_seen_at = a["at"]
        c = report["deciding_consensus"]
        self.council = {
            "verdict": report["verdict"], "consensus": c,
            "regime": (a.get("regime") or {}).get("label"),
            "summary": report["summary"], "at": a["at"],
            "mc_p_tp_first": (a.get("mc") or {}).get("p_tp_first"),
        }
        pos = state.active_positions.get(self.symbol)
        if pos is not None:
            pos["council"] = self.council
        if (settings.COUNCIL_EXIT_CHECK
                and report["n_voters"] >= settings.COUNCIL_MIN_VOTERS
                and c <= settings.COUNCIL_EXIT_CONSENSUS):
            return True, report["summary"]
        return False, ""

    async def _sentinel_loop(self):
        """Heartbeat loop ensuring continuous watch even between ticks"""
        while self.is_running:
            try:
                await asyncio.sleep(1.0)
                tick = state.latest_prices.get(self.symbol)
                if tick:
                    await self.on_tick(tick.price)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in sentinel [{self.bot_id}] loop: {e}")
                await asyncio.sleep(1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bot_id": self.bot_id,
            "symbol": self.symbol,
            "assigned_at": self.assigned_at,
            "entry_price": self.entry_price,
            "highest_price": self.highest_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "invested_dollars": self.invested_dollars,
            "current_value": round(self.qty * self.last_price, 2),
            "allocated_pct": self.allocated_pct,
            "dollar_risk": self.dollar_risk,
            "dollar_reward": self.dollar_reward,
            "sell_plan": self.sell_plan,
            "sell_pct": self.sell_pct,
            "sell_qty": self.sell_qty,
            "status": self.status,
            "evaluations_count": self.evaluations_count,
            "buy_prob": self.buy_prob,
            "sell_prob": self.sell_prob,
            "close_prob": self.close_prob,
            "thesis": self.thesis,
            "council": self.council,
        }

class SentinelRegistry:
    """
    Registry & Orchestrator that spawns, tracks, and releases dedicated Sentinel Bots.
    Ensures 1-to-1 mapping: 1 Open Trade = 1 Dedicated Laya Sentinel Bot.
    """
    def __init__(self):
        self._sentinels: Dict[str, PositionSentinelBot] = {}

    def sync_with_positions(self, active_positions: Dict[str, Dict[str, Any]]):
        """Automatically assigns a dedicated bot to new positions and retires closed ones"""
        current_symbols = set(active_positions.keys())
        assigned_symbols = set(self._sentinels.keys())

        # 1. Spawn dedicated bot for any new position
        for sym in current_symbols - assigned_symbols:
            pos_data = active_positions[sym]
            bot = PositionSentinelBot(sym, pos_data)
            self._sentinels[sym] = bot
            bot.start()

        # 2. Retire bots for positions that have closed
        for sym in assigned_symbols - current_symbols:
            bot = self._sentinels.pop(sym, None)
            if bot:
                bot.stop()

    async def dispatch_tick(self, symbol: str, price: float):
        """Routes tick directly to the dedicated sentinel bot assigned to this symbol"""
        bot = self._sentinels.get(symbol)
        if not bot and symbol in state.active_positions:
            pos_data = state.active_positions[symbol]
            bot = PositionSentinelBot(symbol, pos_data)
            self._sentinels[symbol] = bot
            bot.start()
        if bot:
            await bot.on_tick(price)

    def get_sentinel(self, symbol: str) -> Optional[PositionSentinelBot]:
        return self._sentinels.get(symbol)

    def to_dict(self) -> Dict[str, Any]:
        return {sym: bot.to_dict() for sym, bot in self._sentinels.items()}

sentinel_registry = SentinelRegistry()
