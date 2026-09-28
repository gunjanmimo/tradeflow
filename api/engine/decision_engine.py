"""
Decision engine: resolves the strategy for a symbol and delegates to it.

This used to hold one hardcoded scoring formula applied to every asset, with the
entry gates inlined. The formula itself was reasonable; the problem was that it
demanded Laya sentiment for every entry, which is right for equities and
unworkable for alt-coins with no news coverage.

The engine now owns only orchestration:
  - pick the strategy (per-symbol override, else asset-class default)
  - assemble the context the strategy is allowed to read
  - apply the risk dial's entry bar
  - keep bracket and telemetry bookkeeping in one place

The trading logic itself lives in engine/strategies/.
"""
import time

from core.config import settings
from core.state import state, TradeDecision, QuantMetrics, SentimentRecord
from engine import brackets, action_policy
from engine.strategies import registry
from engine.strategies.context import build_context


class DecisionEngine:
    def evaluate(self, symbol: str, quant: QuantMetrics,
                 sentiment: SentimentRecord) -> TradeDecision:
        current_tick = state.latest_prices.get(symbol)
        if not current_tick:
            return TradeDecision(symbol=symbol, action="HOLD",
                                 reason="No price tick available")

        price = current_tick.price
        holding = symbol in state.active_positions

        position = state.active_positions.get(symbol)
        if holding:
            strategy = registry.for_position(symbol, position, state.strategy_class_defaults,
                                             state.strategy_overrides)
        else:
            strategy = registry.resolve(symbol, state.strategy_class_defaults,
                                        state.strategy_overrides)

        ctx = build_context(symbol, price=price, position=position,
                            quant=quant, sentiment=sentiment)

        if holding:
            decision = self._evaluate_holding(symbol, ctx, strategy, price, quant)
        else:
            decision = self._evaluate_entry(symbol, ctx, strategy, price)

        state.recent_decisions.append(decision)
        return decision

    # ------------------------------------------------------------------
    def _evaluate_entry(self, symbol, ctx, strategy, price) -> TradeDecision:
        # Broker tradability is an execution fact, not a strategy concern.
        from engine.executor import executor
        is_tradable = (
            executor.is_symbol_tradable(symbol)
            if (executor.is_connected and not executor.is_mock_mode) else True
        )

        entry = strategy.evaluate_entry(ctx)

        if not is_tradable:
            return TradeDecision(
                symbol=symbol, action="HOLD", buy_prob=entry.buy_prob,
                reason=f"[{strategy.name}] Monitoring only (not tradable on Alpaca)",
                timestamp=time.time(),
            )

        action = "BUY" if entry.should_enter else "HOLD"
        state.last_gate_detail[symbol] = {
            "strategy": strategy.name,
            # For the adaptive selector: the library strategy it actually chose.
            # The executor stamps this on the position so exits follow the thesis.
            "selected_strategy": entry.gates.get("selected_strategy") or strategy.name,
            "regime": entry.gates.get("regime"),
            "should_enter": entry.should_enter,
            "buy_prob": entry.buy_prob,
            "blocked_by": entry.blocked_by,
            "gates": entry.gates,
            "reason": entry.reason,
            "at": time.time(),
        }

        return TradeDecision(
            symbol=symbol, action=action,
            buy_prob=entry.buy_prob, sell_prob=0.0, close=False,
            reason=f"[{strategy.name}] {entry.reason}",
            timestamp=time.time(),
        )

    # ------------------------------------------------------------------
    def _evaluate_holding(self, symbol, ctx, strategy, price, quant) -> TradeDecision:
        pos = state.active_positions.get(symbol, {})
        avg_price = float(pos.get("avg_entry_price", price))

        # One shared bracket derivation; repairs in place if missing or degenerate.
        if not brackets.is_valid(avg_price, pos.get("stop_loss"), pos.get("take_profit")):
            brackets.ensure(pos, price=price, atr=quant.atr if quant else None)
        stop_loss = pos.get("stop_loss")
        take_profit = pos.get("take_profit")

        hit_stop = stop_loss is not None and price <= float(stop_loss)
        hit_tp = take_profit is not None and price >= float(take_profit)

        exit_decision = strategy.evaluate_exit(ctx)

        should_close = hit_stop or hit_tp or exit_decision.should_close

        # Close probability: bracket proximity, or the strategy's own read.
        close_prob = exit_decision.close_prob
        if stop_loss is not None and take_profit is not None and avg_price > 0:
            sl, tp = float(stop_loss), float(take_profit)
            if price < avg_price and sl < avg_price:
                close_prob = max(close_prob,
                                 min(max((avg_price - price) / (avg_price - sl), 0.0), 1.0))
            elif price > avg_price and tp > avg_price:
                close_prob = max(close_prob,
                                 min(max((price - avg_price) / (tp - avg_price), 0.0), 1.0))
        close_prob = float(min(max(close_prob, 0.0), 1.0))
        should_close = should_close or close_prob >= 1.0

        # Telemetry the UI reads off the position record (100% across 4 actions;
        # CLOSE is 100% only when the exit fires).
        sell_w = float(exit_decision.sell_prob)
        probs, action = action_policy.distribute(
            buy_w=float(ctx.sentiment.pos_prob) * 0.45,
            hold_w=max(0.2, 1.0 - max(close_prob, sell_w)),
            sell_w=sell_w, close_p=close_prob, closing=should_close)
        buy_prob, hold_prob = probs["BUY"], probs["HOLD"]
        sell_prob, close_prob = probs["SELL"], probs["CLOSE"]

        pos["action_space"] = list(action_policy.ACTION_SPACE)
        pos["action"] = action
        pos["buy_prob"] = buy_prob
        pos["hold_prob"] = hold_prob
        pos["sell_prob"] = sell_prob
        pos["close_prob"] = close_prob
        pos["laya_pos"] = ctx.sentiment.pos_prob
        pos["laya_neg"] = ctx.sentiment.neg_prob
        pos["strategy"] = strategy.name

        if hit_stop:
            reason = f"Hit Stop Loss: ${price:,.6g} <= SL ${float(stop_loss):,.6g}"
        elif hit_tp:
            reason = f"Hit Take Profit: ${price:,.6g} >= TP ${float(take_profit):,.6g}"
        else:
            reason = exit_decision.reason

        return TradeDecision(
            symbol=symbol,
            action=action,
            buy_prob=buy_prob,
            hold_prob=hold_prob,
            sell_prob=sell_prob,
            close_prob=close_prob,
            close=should_close,
            reason=f"[{strategy.name}] {reason}",
            timestamp=time.time(),
        )


decision_engine = DecisionEngine()
