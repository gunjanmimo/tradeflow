"""
One sentinel per open position: the exits, checked on every price.

A position leaves by exactly one of these, first match wins:

  stop        price at or below the stop. Immediate: no confirmation window.
              The same stop sits at the broker as the bracket leg, so it holds
              even when the engine is down.
  target      price at or above the take-profit (also a broker bracket leg).
  forced      no price for too long, or the day trade has reached its close
              window (engine/forced_exits.py).
  strategy    the strategy that opened the position says its thesis is gone
              (its evaluate_exit), e.g. the RL policy choosing to go flat.

That is the whole exit policy, and the backtester and the RL environment model
exactly these rules. The earlier stack -- profit harvest on any uptick, a
confirmation delay on stops, rescue adds into losers, trims, scale-ins, a
trailing stop and several "soft" reversal/news exits -- cut winners short and
let losers run, and none of it could be shown to help in a backtest.
"""
import asyncio
import logging
import time
from typing import Any, Dict, Optional

from core.state import state, TradeDecision
from engine import brackets, action_policy, forced_exits
from core.latency import latency

logger = logging.getLogger("tradeflow.sentinel")


class PositionSentinelBot:
    """Watches one open position and fires its exit."""

    def __init__(self, symbol: str, position_data: Dict[str, Any]):
        self.symbol = symbol
        self.bot_id = f"BOT-{symbol.upper()}-{int(time.time()) % 10000:04d}"
        self.assigned_at = time.time()
        self.entry_price = float(position_data.get("avg_entry_price", position_data.get("current_price", 0.0)))
        self.highest_price = self.entry_price
        self.last_price = float(position_data.get("current_price") or self.entry_price)
        self.qty = float(position_data.get("qty", 0.0))
        q = state.quant_metrics.get(symbol)
        brackets.ensure(position_data, price=self.entry_price, atr=q.atr if q else None)
        self.stop_loss = float(position_data["stop_loss"])
        self.take_profit = float(position_data["take_profit"])

        self.invested_dollars = float(position_data.get("allocated_dollars")
                                      or round(self.qty * self.entry_price, 2))
        budget = float(state.hard_cap or 1.0)
        self.allocated_pct = float(position_data.get("allocated_pct")
                                   or round((self.invested_dollars / budget) * 100, 2))
        self.dollar_risk = round(max(0.0, (self.entry_price - self.stop_loss) * self.qty), 2)
        self.dollar_reward = round(max(0.0, (self.take_profit - self.entry_price) * self.qty), 2)

        self.action_space = list(action_policy.ACTION_SPACE)
        self.action = "HOLD"
        self.status = "WATCHING"
        self.evaluations_count = 0
        self.buy_prob, self.hold_prob, self.sell_prob, self.close_prob = 0.0, 1.0, 0.0, 0.0
        self.sell_plan = f"Close 100% at stop ${self.stop_loss:,.2f}, target ${self.take_profit:,.2f} or strategy exit"
        self.thesis = (f"Assigned to {self.symbol}: ${self.invested_dollars:,.2f} invested "
                       f"({self.allocated_pct}% of budget).")
        self.is_running = True
        self.exit_reason: Optional[str] = None

        position_data.update(bot_id=self.bot_id, action_space=list(self.action_space),
                             action=self.action, buy_prob=self.buy_prob, hold_prob=self.hold_prob,
                             sell_prob=self.sell_prob, close_prob=self.close_prob,
                             bot_thesis=self.thesis, sell_plan=self.sell_plan)
        self._task = None
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
        state.log_event("AGENT_SPAWNED",
                        f"Sentinel [{self.bot_id}] watching {self.symbol} (entry ${self.entry_price:,.4g}, "
                        f"stop ${self.stop_loss:,.4g}, target ${self.take_profit:,.4g})")

    def stop(self):
        self.is_running = False
        if self._task:
            self._task.cancel()
        state.log_event("AGENT_RETIRED", f"Sentinel [{self.bot_id}] completed watch for {self.symbol}")

    async def on_tick(self, price: float):
        if not self.is_running:
            return
        t0 = time.perf_counter_ns()
        try:
            await self._on_tick(float(price))
        finally:
            latency.record_ns("sentinel_tick", t0)

    def _sync_from_position(self, pos: Dict[str, Any]):
        """Quantity, entry and bracket follow the position (a broker sync may update them)."""
        qty_now = float(pos.get("qty") or 0.0)
        entry_now = float(pos.get("avg_entry_price") or 0.0)
        if qty_now > 0:
            self.qty = qty_now
        if entry_now > 0:
            self.entry_price = entry_now
        try:
            self.stop_loss = float(pos.get("stop_loss", self.stop_loss))
            self.take_profit = float(pos.get("take_profit", self.take_profit))
        except (TypeError, ValueError):
            pass

    def _exit_reason(self, pos: Dict[str, Any], price: float, pnl_pct: float) -> Optional[str]:
        if price <= self.stop_loss:
            return f"Stop-loss: ${price:,.4g} <= stop ${self.stop_loss:,.4g}"
        if price >= self.take_profit:
            return f"Take-profit: ${price:,.4g} >= target ${self.take_profit:,.4g}"
        forced = forced_exits.check(self.symbol, pos, pnl_pct, self.assigned_at)
        if forced is not None:
            return f"Forced exit: {forced}"
        try:
            from engine.strategies import registry
            from engine.strategies.context import build_context
            strat = registry.for_position(self.symbol, pos, state.strategy_class_defaults,
                                          state.strategy_overrides)
            pos["strategy"] = strat.name
            ctx = build_context(self.symbol, price=price, position=pos,
                                highest_price=self.highest_price)
            res = strat.evaluate_exit(ctx)
            if res.should_close:
                return f"Strategy exit ({strat.name}): {res.reason}"
        except Exception as e:
            logger.error(f"[{self.bot_id}] strategy exit evaluation failed: {e}")
        return None

    async def _on_tick(self, price: float):
        self.last_price = price
        self.evaluations_count += 1
        pos = state.active_positions.get(self.symbol)
        if not pos:
            self.status = "CLOSED"
            self.stop()
            return
        self._sync_from_position(pos)
        self.highest_price = max(self.highest_price, price)
        pnl_pct = (price - self.entry_price) / self.entry_price if self.entry_price > 0 else 0.0

        reason = self._exit_reason(pos, price, pnl_pct)
        closing = reason is not None

        # Telemetry: how far the price has travelled towards the stop or the target.
        close_p = 0.0
        if price < self.entry_price and self.stop_loss < self.entry_price:
            close_p = (self.entry_price - price) / (self.entry_price - self.stop_loss)
        elif price > self.entry_price and self.take_profit > self.entry_price:
            close_p = (price - self.entry_price) / (self.take_profit - self.entry_price)
        close_p = min(max(close_p, 0.0), action_policy.CLOSE_CAP)
        probs, self.action = action_policy.distribute(0.0, 1.0 - close_p, 0.0, close_p, closing)
        self.buy_prob, self.hold_prob = probs["BUY"], probs["HOLD"]
        self.sell_prob, self.close_prob = probs["SELL"], probs["CLOSE"]

        current_value = round(self.qty * price, 2)
        pnl_dollars = round((price - self.entry_price) * self.qty, 2)
        self.dollar_risk = round(max(0.0, (self.entry_price - self.stop_loss) * self.qty), 2)
        self.dollar_reward = round(max(0.0, (self.take_profit - self.entry_price) * self.qty), 2)
        self.thesis = (reason if closing else
                       f"Holding: P&L {pnl_pct * 100:+.2f}% (${pnl_dollars:+.2f}); "
                       f"stop ${self.stop_loss:,.4g}, target ${self.take_profit:,.4g}")
        pos.update(
            bot_id=self.bot_id, action_space=list(self.action_space), action=self.action,
            current_price=price,
            price_age_s=round(max(0.0, time.time() - state.price_moved_at.get(self.symbol, time.time())), 1),
            unrealized_pl=pnl_dollars, unrealized_plpc=round(pnl_pct, 5),
            buy_prob=self.buy_prob, hold_prob=self.hold_prob, sell_prob=self.sell_prob,
            close_prob=self.close_prob, stop_loss=self.stop_loss, take_profit=self.take_profit,
            invested_dollars=self.invested_dollars, current_value=current_value,
            allocated_pct=self.allocated_pct, dollar_risk=self.dollar_risk,
            dollar_reward=self.dollar_reward, sell_plan=self.sell_plan, sell_pct=100,
            sell_qty=self.qty, bot_thesis=self.thesis,
        )
        pos.update(forced_exits.telemetry(self.symbol))

        if not closing:
            self.status = "WATCHING"
            return
        self.status = "TRIGGERING_EXIT"
        self.exit_reason = reason
        now = time.time()
        # The signal repeats on every tick until the exit fills; the executor
        # ignores the repeats, so log it once, then every 30s while it stands.
        if now - self._close_logged_at >= 30.0:
            self._close_logged_at = now
            state.log_event("SIGNAL", f"Sentinel [{self.bot_id}] CLOSE {self.symbol}: {reason}")
        from engine.executor import executor
        asyncio.create_task(executor.execute_decision(TradeDecision(
            symbol=self.symbol, action="CLOSE", close_prob=1.0, close=True,
            reason=f"[{self.bot_id}] {reason}",
            # Stop and target keep the normal backoff; a stale or end-of-day
            # exit must be retried within FORCED_EXIT_MAX_WAIT_SECONDS.
            forced=reason.startswith("Forced exit"),
        )))

    async def _sentinel_loop(self):
        """Heartbeat: re-checks the exits every second even when no price arrives."""
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
            "bot_id": self.bot_id, "symbol": self.symbol,
            "action_space": list(self.action_space), "action": self.action,
            "assigned_at": self.assigned_at, "entry_price": self.entry_price,
            "highest_price": self.highest_price, "stop_loss": self.stop_loss,
            "take_profit": self.take_profit, "invested_dollars": self.invested_dollars,
            "current_value": round(self.qty * self.last_price, 2),
            "allocated_pct": self.allocated_pct, "dollar_risk": self.dollar_risk,
            "dollar_reward": self.dollar_reward, "sell_plan": self.sell_plan,
            "sell_pct": 100, "sell_qty": self.qty, "status": self.status,
            "evaluations_count": self.evaluations_count,
            "buy_prob": self.buy_prob, "hold_prob": self.hold_prob,
            "sell_prob": self.sell_prob, "close_prob": self.close_prob,
            "thesis": self.thesis, "exit_reason": self.exit_reason,
        }


class SentinelRegistry:
    """Spawns one sentinel per open position and retires it when the position closes."""

    def __init__(self):
        self._sentinels: Dict[str, PositionSentinelBot] = {}

    def sync_with_positions(self, active_positions: Dict[str, Dict[str, Any]]):
        current, assigned = set(active_positions), set(self._sentinels)
        for sym in current - assigned:
            bot = PositionSentinelBot(sym, active_positions[sym])
            self._sentinels[sym] = bot
            bot.start()
        for sym in assigned - current:
            bot = self._sentinels.pop(sym, None)
            if bot:
                bot.stop()

    async def dispatch_tick(self, symbol: str, price: float):
        bot = self._sentinels.get(symbol)
        if not bot and symbol in state.active_positions:
            bot = PositionSentinelBot(symbol, state.active_positions[symbol])
            self._sentinels[symbol] = bot
            bot.start()
        if bot:
            await bot.on_tick(price)

    def get_sentinel(self, symbol: str) -> Optional[PositionSentinelBot]:
        return self._sentinels.get(symbol)

    def to_dict(self) -> Dict[str, Any]:
        return {sym: bot.to_dict() for sym, bot in self._sentinels.items()}


sentinel_registry = SentinelRegistry()
