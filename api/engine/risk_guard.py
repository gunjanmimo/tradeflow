import logging
from typing import Optional, Tuple, Dict, Any
from core.config import settings
from core.state import state

logger = logging.getLogger("tradeflow.risk")

# Smallest order worth placing: below this, a position is noise against costs.
MIN_ORDER_DOLLARS = 30.0

class RiskGuard:
    """
    Strict Capital and Risk Protection layer.
    Ensures no order is sent to the broker without passing sizing, exposure, and loss bounds.
    """
    def can_open_position(self, symbol: str) -> Tuple[bool, str]:
        # 1. Kill switch check
        if not state.is_trading_active:
            return False, "Trading engine is currently paused by kill-switch."

        # 1a. Markets / symbols switched off from the UI. Entries only.
        from core.market_filter import market_filter
        market_block = market_filter.entry_block_reason(symbol)
        if market_block:
            return False, market_block

        # 1a'. Session. A market order sent outside regular hours used to be
        # queued by the broker and filled at the open at whatever price printed.
        from core.market_hours import us_session, REGULAR, PRE
        session = us_session()
        if session == PRE and not settings.PREMARKET_TRADING_ENABLED:
            return False, "US pre-market: pre-market trading is switched off (PREMARKET_TRADING_ENABLED)."
        if session not in (REGULAR, PRE):
            return False, f"US stock market is {session}. Entries run 04:00-16:00 NY on trading days."
        # Day trading: nothing is held through the close, so a stock opened
        # shortly before it would only be flattened again straight away.
        if settings.DAY_TRADE_FLATTEN_ENABLED:
            from core.market_hours import minutes_to_close
            mins = minutes_to_close(symbol)
            if mins is not None and mins <= settings.NO_NEW_ENTRY_MINUTES_BEFORE_CLOSE:
                return False, (f"{symbol}: the US market closes in {max(mins, 0.0):.0f} min "
                               f"(no new day trades inside {settings.NO_NEW_ENTRY_MINUTES_BEFORE_CLOSE:.0f} min "
                               f"of the close).")

        # 1b. Portfolio circuit breakers.
        # These gate NEW ENTRIES ONLY -- exits must always remain possible, or a
        # tripped breaker would trap the very positions that tripped it.
        state.roll_trading_day_if_needed()

        profile = state.risk_profile

        from core.capital_plan import capital_plan
        stair_block = capital_plan.entry_block_reason()
        if stair_block:
            return False, stair_block

        if state.daily_loss_pct >= profile.max_daily_loss_pct:
            reason = (f"Daily loss limit hit: -{state.daily_loss_pct:.2f}% of budget "
                      f"(limit {profile.max_daily_loss_pct}% at risk dial {profile.factor}). No new entries until tomorrow.")
            if state.halt_reason != reason:
                state.halt_reason = reason
                state.log_event("RISK_HALT", reason)
            return False, reason

        if state.drawdown_pct >= profile.max_drawdown_pct:
            reason = (f"Max drawdown breached: -{state.drawdown_pct:.2f}% from peak bot equity "
                      f"${state.peak_equity:,.2f} (limit {profile.max_drawdown_pct}% at risk dial {profile.factor}). "
                      f"No new entries until manually reset.")
            if state.halt_reason != reason:
                state.halt_reason = reason
                state.log_event("RISK_HALT", reason)
            return False, reason

        # 1d. Diversification: sector / region / cyclical-share caps from
        # the same dial. The class count above cannot see that NVDA and MSFT are
        # one bet; this can.
        from engine.diversification import diversification
        div_block = diversification.entry_block_reason(symbol, MIN_ORDER_DOLLARS)
        if div_block:
            return False, div_block

        # 2. Duplicate or in-flight position check
        if symbol in state.active_positions:
            return False, f"Already holding an active position in {symbol}."

        from engine.executor import executor
        if symbol in executor.pending_orders:
            return False, f"An order is already in-flight for {symbol}."
        order_block = executor.entry_block_reason(symbol)
        if order_block:
            return False, order_block

        # 3. Check if broker supports trading this asset
        if executor.is_connected and not executor.is_mock_mode:
            if not executor.is_symbol_tradable(symbol):
                return False, f"{symbol} is in market monitoring mode (not listed for execution by Alpaca broker)."

        # 4. Max concurrent positions check
        if len(state.active_positions) >= profile.max_concurrent_positions:
            return False, f"Max concurrent positions limit ({profile.max_concurrent_positions} at risk dial {profile.factor}) reached."

        # 4. Hard budget cap. Committed = cost basis of open positions + buys in
        # flight; the cap shrinks with realised losses. Nothing outside it is used.
        min_order = MIN_ORDER_DOLLARS
        if state.hard_cap < min_order:
            return False, (f"Budget exhausted: realised losses have reduced the bots' capital to "
                           f"${state.bot_capital:,.2f} of the ${state.allocated_capital:,.2f} cap. "
                           f"No money outside the budget is used; raise or reset the budget to continue.")
        if state.remaining_budget < min_order:
            return False, (f"Hard budget cap reached: ${state.committed_capital:,.2f} committed of "
                           f"${state.hard_cap:,.2f} (${state.remaining_budget:,.2f} left).")

        # 5. Strict Cash & Budget Cap Check: bot can NEVER touch locked broker cash or funds outside budget
        bot_cash = state.bot_cash
        min_cash_required = min_order
        if bot_cash < min_cash_required:
            return False, (f"Insufficient bot cash under hard cap (${bot_cash:,.2f} available, "
                           f"minimum ${min_cash_required:,.2f} required). Main broker equity is locked outside the budget.")

        return True, "Passed risk filters."

    def calculate_order_sizing(
        self, 
        symbol: str, 
        current_price: float, 
        atr: float,
        laya_pos_prob: float = 0.70,
        consensus_score: Optional[float] = None
    ) -> Tuple[float, float, float, Dict[str, Any]]:
        """
        Uses Laya Allocation Manager to decide the exact dollar allocation,
        percentage of equity, quantity, and stop-loss/take-profit boundaries.
        """
        from engine.allocation_agent import allocation_manager

        alloc = allocation_manager.evaluate_allocation(
            symbol=symbol,
            current_price=current_price,
            atr=atr,
            laya_pos_prob=laya_pos_prob,
            consensus_score=consensus_score
        )

        return alloc["qty"], alloc["stop_loss"], alloc["take_profit"], alloc

risk_guard = RiskGuard()
