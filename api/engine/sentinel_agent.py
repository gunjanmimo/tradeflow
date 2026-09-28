import asyncio
import logging
import time
from typing import Dict, Any, Optional
from core.state import state, TradeDecision
from core.config import settings
from engine import brackets, action_policy, forced_exits, profit_harvest, loss_recovery
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
    - Operates complete 4-action space: [BUY, HOLD, SELL, CLOSE].
    - Computes real-time Buy / Hold / Sell / Close probabilities out of 100.
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

        # Capital Allocation & Sizing Intelligence (BUY SIZING & HOLD POSITION SIZING)
        self.invested_dollars = float(position_data.get("allocated_dollars") or round(self.qty * self.entry_price, 2))
        budget = float(state.hard_cap or 1.0)
        self.allocated_pct = float(position_data.get("allocated_pct") or round((self.invested_dollars / budget) * 100, 2))
        self.dollar_risk = round(max(0.0, (self.entry_price - self.stop_loss) * self.qty), 2)
        self.dollar_reward = round(max(0.0, (self.take_profit - self.entry_price) * self.qty), 2)

        # Divestment Policy & Sizing Intelligence (SELL SIZING)
        self.sell_pct = 100
        self.sell_qty = self.qty
        self.sell_plan = f"SELL 100% ({self.qty}x = ${self.invested_dollars:,.2f}) on SL/TP/Reversal"

        # Action Space: Full 4-action space [BUY, HOLD, SELL, CLOSE]
        # BUY (momentum addition/scale-in conviction), HOLD (maintain position without new trade/share buy),
        # SELL (trim/de-risk sizing), CLOSE (full exit/liquidation on SL/TP/reversal)
        self.action_space = list(action_policy.ACTION_SPACE)
        self.action = "HOLD"

        # Real-time state (100% normalized probability distribution across 4 actions)
        self.status = "WATCHING"
        self.evaluations_count = 0
        self.buy_prob = 0.15
        self.hold_prob = 0.70
        self.sell_prob = 0.10
        self.close_prob = 0.05
        self.thesis = f"Assigned to {self.symbol}: ${self.invested_dollars:,.2f} invested ({self.allocated_pct}% of budget). Action: HOLD (Guarding capital)."
        self.is_running = True

        # Sync telemetry with position dict immediately upon deployment
        position_data["bot_id"] = self.bot_id
        position_data["action_space"] = list(self.action_space)
        position_data["action"] = self.action
        position_data["buy_prob"] = self.buy_prob
        position_data["hold_prob"] = self.hold_prob
        position_data["sell_prob"] = self.sell_prob
        position_data["close_prob"] = self.close_prob
        position_data["bot_thesis"] = self.thesis

        self._task = None
        # Latest quant-council read on this position. Computed by the analysis
        # worker process; the bot only reads it, so it costs nothing per tick.
        self.council: Optional[Dict[str, Any]] = None
        self._council_seen_at = 0.0
        self._close_logged_at = 0.0

    def start(self):
        try:
            loop = asyncio.get_running_loop()
            self._task = loop.create_task(self._sentinel_loop())
            # Every trade print for this symbol now drives this bot directly.
            from feeds.alpaca_stream import market_stream
            loop.create_task(market_stream.ensure_stock_subscription(self.symbol))
            loop.create_task(market_stream.ensure_position_stream(self.symbol))
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
        # The position manager may have sold part or added since the last tick:
        # quantity and average entry follow the position, not the spawn snapshot.
        qty_now = float(pos.get("qty") or 0.0)
        entry_now = float(pos.get("avg_entry_price") or 0.0)
        if qty_now > 0 and (qty_now != self.qty or (entry_now > 0 and entry_now != self.entry_price)):
            self.qty = qty_now
            if entry_now > 0:
                self.entry_price = entry_now
            self.invested_dollars = float(pos.get("invested_dollars") or round(self.qty * self.entry_price, 2))
            self.dollar_risk = round(max(0.0, (self.entry_price - self.stop_loss) * self.qty), 2)
            self.dollar_reward = round(max(0.0, (self.take_profit - self.entry_price) * self.qty), 2)

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

        # A position that went under water and has climbed back above break-even
        # gets its stop locked there: the recovered loss cannot come back.
        lock = loss_recovery.breakeven_lock(pos, price, self.entry_price, self.stop_loss)
        if lock is not None and lock > self.stop_loss:
            self.stop_loss = lock
            pos["stop_loss"] = lock
            state.log_event("LOSS_RECOVERED", f"{self.symbol}: back above break-even at ${price:,.6g}; "
                                              f"stop locked at ${lock:,.6g}")

        drawdown_from_peak = (self.highest_price - price) / self.highest_price if self.highest_price > 0 else 0.0

        # 2. Get quant metrics & Laya semantic sentiment for this specific asset
        quant = state.quant_metrics.get(self.symbol)
        sentiment = state.get_sentiment(self.symbol)

        trend_score = 0.5
        if quant and quant.ema_fast and quant.ema_slow:
            trend_score = 0.80 if (quant.ema_fast > quant.ema_slow and price > quant.ema_fast) else 0.20
        rsi_score = 0.5
        if quant and quant.rsi is not None:
            if quant.rsi < 35:
                rsi_score = 0.75
            elif quant.rsi > 70:
                rsi_score = 0.20
        # 3. Dynamic Price-Action Responsive Modulations across 4 Actions [BUY, HOLD, SELL, CLOSE]:
        # Compute raw conviction weights for each dimension:
        # A) BUY Weight: Long Continuation & Upward Momentum (Scale-In / Add Signal)
        pnl_gain_factor = min(max(pnl_pct * 12.0, -0.35), 0.25)
        vel_factor = min(max(tick_delta_pct * 80.0, -0.10), 0.10)
        drawdown_penalty = min(drawdown_from_peak * 8.0, 0.30)

        raw_buy = max(0.02, (
            0.45 * sentiment.pos_prob + 
            0.25 * (0.5 + pnl_gain_factor) + 
            0.15 * (0.5 + vel_factor) + 
            0.15 * (0.6 * trend_score + 0.4 * rsi_score) - 
            drawdown_penalty
        ))

        # B) HOLD Weight: Position Maintenance & Thesis Stability (Holding Without New Trade/Buy)
        # High when price is safely within SL/TP bounds, sentiment is stable/supportive, and drawdown is contained.
        sl_buffer = min(max((price - self.stop_loss) / (self.entry_price * 0.03), 0.0), 1.0) if self.entry_price > 0 else 0.5
        laya_hold_weight = max(0.0, min(1.0, sentiment.pos_prob - (0.4 * sentiment.neg_prob if sentiment.neg_prob > 0.50 else 0.0)))
        raw_hold = max(0.05, (
            0.40 * laya_hold_weight + 
            0.30 * max(0.0, 1.0 - drawdown_from_peak * 5.0) + 
            0.25 * sl_buffer + 
            0.15 * (0.8 if 40 <= (quant.rsi if quant and quant.rsi else 50) <= 65 else 0.4)
        ))

        # C) SELL Weight: Reversal & Fading Risk (Scale-out / Trim Sizing)
        giveback_factor = min(drawdown_from_peak * 18.0, 0.45)
        loss_factor = min(max(-pnl_pct * 18.0, 0.0), 0.40)
        tick_down = min(max(-tick_delta_pct * 40.0, 0.0), 0.15)
        overbought = 0.15 if (quant and quant.rsi and quant.rsi > 70) else 0.0

        # Without fresh headlines neg_prob is a 0.5 placeholder, not a bearish read.
        # Counting it put a newsless position 0.20 of the way to a forced exit.
        has_news = sentiment.n_headlines > 0 and not sentiment.is_stale
        news_neg = sentiment.neg_prob if has_news else 0.0

        # The reversal read drives the soft exit. It leaves out the loss itself:
        # the stop-loss already prices that, and counting it again closed small
        # dips well before the volatility-sized stop was reached.
        reversal_sell = (
            0.40 * news_neg +
            0.25 * giveback_factor +
            0.15 * (1.0 - trend_score) +
            tick_down +
            overbought
        )
        raw_sell = max(0.02, reversal_sell + 0.20 * loss_factor)

        # D) CLOSE Probability: Stop-Loss / Take-Profit Proximity & Urgent Exit (100% Liquidation).
        # A probability in its own right, not a weight: it takes its share first.
        # A loss stop must hold for a few seconds (or be broken through by a
        # margin) before it closes the trade; see engine/loss_recovery.py.
        hit_stop, stop_reason = loss_recovery.check_stop(pos, price, self.entry_price, self.stop_loss)
        hit_tp = price >= self.take_profit
        laya_bearish_reversal = (sentiment.is_tradeable and not sentiment.is_stale
                                 and sentiment.neg_prob >= settings.MAX_SELL_SENTIMENT_NEG)

        sl_urgency = 0.0
        if price < self.entry_price and self.stop_loss < self.entry_price:
            total_sl_dist = self.entry_price - self.stop_loss
            drop = self.entry_price - price
            sl_ratio = min(max(drop / total_sl_dist, 0.0), 1.0)
            sl_urgency = 0.15 + 0.85 * (sl_ratio ** 1.3)
            if not hit_stop:
                # At the stop but not yet confirmed: urgent, not yet certain.
                sl_urgency = min(sl_urgency, action_policy.CLOSE_CAP)

        tp_urgency = 0.0
        if price > self.entry_price and self.take_profit > self.entry_price:
            total_tp_dist = self.take_profit - self.entry_price
            gain = price - self.entry_price
            tp_ratio = min(max(gain / total_tp_dist, 0.0), 1.0)
            tp_urgency = 0.10 + 0.90 * (tp_ratio ** 1.4)

        raw_close = max(0.01, sl_urgency, tp_urgency, min(raw_sell, 0.95) * 0.85)

        # 3b. Delegate discretionary exit check to position's strategy
        strategy_exit = False
        strategy_reason = ""
        council_exit = False
        council_reason = ""
        try:
            from engine.strategies import registry as _registry
            from engine.strategies.context import build_context
            _strat = _registry.for_position(self.symbol, pos, state.strategy_class_defaults,
                                            state.strategy_overrides)
            _ctx = build_context(self.symbol, price=price, position=pos,
                                 highest_price=self.highest_price,
                                 quant=quant, sentiment=sentiment)
            _res = _strat.evaluate_exit(_ctx)
            strategy_exit = _res.should_close
            strategy_reason = _res.reason
            raw_sell = max(raw_sell, _res.sell_prob)
            reversal_sell = max(reversal_sell, _res.sell_prob)
            raw_close = max(raw_close, _res.close_prob)
            pos["strategy"] = _strat.name

            council_exit, council_reason = self._council_check()
        except Exception as _e:
            logger.error(f"[{self.bot_id}] strategy exit evaluation failed: {_e}")

        # Exits that are not a trading opinion: an unpriced position, or a day trade
        # about to run into its market's close. Never delayed by the minimum hold.
        forced_reason = None
        try:
            forced_reason = forced_exits.check(self.symbol, pos, pnl_pct, self.assigned_at)
        except Exception as _e:
            logger.error(f"[{self.bot_id}] forced-exit check failed: {_e}")

        # Check hard exit triggers. The soft ones (reversal read, bearish news) wait
        # out an equity's minimum hold, as the stock strategy's own exits do; the
        # stop, target, strategy and council exits are never delayed.
        # On a losing position the soft signals need the price trend to agree:
        # a bearish headline over a price that is holding is not a reason to
        # realise the loss -- the stop still protects it.
        strong_sell_signal = (((reversal_sell >= settings.MAX_SELL_SENTIMENT_NEG)
                               or laya_bearish_reversal)
                              and not self._in_min_hold(pos)
                              and loss_recovery.soft_exit_allowed(self.symbol, pnl_pct))
        # A close probability that has reached certainty is an exit, not a display value.
        close_certain = raw_close >= 1.0
        is_closing = (hit_stop or hit_tp or strong_sell_signal or strategy_exit
                      or council_exit or close_certain or forced_reason is not None)

        # 3c. 100% distribution across [BUY, HOLD, SELL, CLOSE]; CLOSE is 100% iff exiting.
        probs, self.action = action_policy.distribute(raw_buy, raw_hold, raw_sell,
                                                      raw_close, is_closing)
        self.buy_prob = probs["BUY"]
        self.hold_prob = probs["HOLD"]
        self.sell_prob = probs["SELL"]
        self.close_prob = probs["CLOSE"]

        # Dynamic Divestment / Trim Sizing Plan
        current_value = round(self.qty * price, 2)
        if hit_stop:
            self.sell_pct = 100
            self.sell_qty = self.qty
            self.sell_plan = f"SELL 100% ({self.qty}x = ${current_value:,.2f}) [STOP LOSS HIT]"
        elif str(pos.get("stop_state", "")).startswith("confirming"):
            self.sell_pct = 100
            self.sell_qty = self.qty
            self.sell_plan = (f"Stop ${self.stop_loss:,.6g} touched, {pos['stop_state']} before selling "
                              f"(disaster stop ${loss_recovery.disaster_stop(self.entry_price, self.stop_loss):,.6g})")
        elif hit_tp:
            self.sell_pct = 100
            self.sell_qty = self.qty
            self.sell_plan = f"SELL 100% ({self.qty}x = ${current_value:,.2f}) [TAKE PROFIT HIT]"
        elif forced_reason is not None:
            self.sell_pct = 100
            self.sell_qty = self.qty
            self.sell_plan = f"SELL 100% ({self.qty}x = ${current_value:,.2f}) [FORCED EXIT]"
        elif strong_sell_signal:
            self.sell_pct = 100
            self.sell_qty = self.qty
            self.sell_plan = f"SELL 100% ({self.qty}x = ${current_value:,.2f}) [BEARISH REVERSAL]"
        elif self.action == "SELL" or self.sell_prob >= 0.25:
            self.sell_pct = 50
            self.sell_qty = round(self.qty * 0.50, 4)
            self.sell_plan = f"SELL 50% ({self.sell_qty}x = ${round(self.sell_qty * price, 2):,.2f}) [DE-RISKING TRIM]"
        else:
            self.sell_pct = 100
            self.sell_qty = self.qty
            self.sell_plan = f"Target Sell: 100% ({self.qty}x = ${current_value:,.2f}) | SL ${self.stop_loss:,.2f} | TP ${self.take_profit:,.2f}"

        pnl_dollars = round((price - self.entry_price) * self.qty, 2)
        self.thesis = f"Action: {self.action} | Invested: ${self.invested_dollars:,.2f} ({self.allocated_pct}% cap) | PnL: {pnl_pct*100:+.2f}% (${pnl_dollars:+.2f}) | {self.sell_plan}"

        # Sync telemetry with position dict
        pos["bot_id"] = self.bot_id
        pos["action_space"] = list(self.action_space)
        pos["action"] = self.action
        pos["current_price"] = price
        # Seconds since the price last changed. The heartbeat refreshes this every
        # second, so a frozen number here means a quiet market, not a stalled bot.
        pos["price_age_s"] = round(max(0.0, time.time() - state.price_moved_at.get(self.symbol, time.time())), 1)
        pos["unrealized_pl"] = pnl_dollars
        pos["unrealized_plpc"] = round(pnl_pct, 5)
        pos["buy_prob"] = self.buy_prob
        pos["hold_prob"] = self.hold_prob
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
        pos.update(forced_exits.telemetry(self.symbol))

        # 4. Autonomous Exit Execution
        if is_closing:
            self.status = "TRIGGERING_EXIT"
            if hit_stop:
                reason = f"[{self.bot_id}] {stop_reason}"
            elif hit_tp:
                reason = f"[{self.bot_id}] Take-profit triggered: Price ${price:,.6g} >= TP ${self.take_profit:,.6g}"
            elif forced_reason is not None:
                reason = f"[{self.bot_id}] Forced exit: {forced_reason}"
            elif strategy_exit:
                reason = f"[{self.bot_id}] Strategy exit: {strategy_reason}"
            elif council_exit:
                reason = f"[{self.bot_id}] Council exit: {council_reason}"
            elif strong_sell_signal and laya_bearish_reversal:
                reason = f"[{self.bot_id}] Laya Bearish Reversal: neg_prob={sentiment.neg_prob:.2f} >= {settings.MAX_SELL_SENTIMENT_NEG:.2f}"
            else:
                reason = (f"[{self.bot_id}] Model exit triggered: close_prob={raw_close:.2f}, "
                          f"sell={raw_sell:.2f} (Laya={sentiment.neg_prob:.2f})")

            self.thesis = reason
            # The signal repeats on every tick until the exit fills; the executor
            # ignores the repeats, so log it once, then every 30s while it stands.
            now = time.time()
            if now - self._close_logged_at >= 30.0:
                self._close_logged_at = now
                state.log_event("SIGNAL", f"Sentinel [{self.bot_id}] CLOSE signal for {self.symbol}: {reason}")
            
            from engine.executor import executor
            asyncio.create_task(executor.execute_decision(TradeDecision(
                symbol=self.symbol,
                action="CLOSE",
                buy_prob=0.0,
                hold_prob=0.0,
                sell_prob=self.sell_prob,
                close_prob=1.0,
                close=True,
                reason=reason,
                # Stop and target keep the normal backoff; a stale or end-of-day
                # exit must be retried within FORCED_EXIT_MAX_WAIT_SECONDS.
                forced=forced_reason is not None,
            )))
        else:
            self.status = "WATCHING"
            # Bank part of a winner as day income; the remainder keeps running.
            harvest = profit_harvest.check(self.symbol, pos, price, self.qty, self.entry_price)
            # A loser whose fall has stalled: one small add to lower break-even.
            rescue = None if harvest is not None else loss_recovery.check_rescue(
                self.symbol, pos, price, self.qty, self.entry_price, self.stop_loss,
                news_bearish=has_news and sentiment.neg_prob >= settings.MAX_SELL_SENTIMENT_NEG,
                strategy_exiting=strategy_exit or council_exit)
            for order in (harvest, rescue):
                if order is not None:
                    from engine.executor import executor
                    if order.rescue:
                        state.log_event("LOSS_RECOVERY", f"{self.symbol}: {order.reason}")
                    asyncio.create_task(executor.execute_decision(order))

    def _in_min_hold(self, pos: Dict[str, Any]) -> bool:
        """True while an equity position is younger than the stock minimum hold."""
        opened_at = pos.get("opened_at")
        if not opened_at:
            return False
        held_min = (time.time() - float(opened_at)) / 60
        return held_min < settings.STOCK_SCORE_MIN_HOLD_MINUTES

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
            "action_space": list(self.action_space),
            "action": self.action,
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
            "hold_prob": self.hold_prob,
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
