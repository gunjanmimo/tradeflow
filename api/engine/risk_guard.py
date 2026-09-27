import logging
from typing import Optional, Tuple, Dict, Any
from core.config import settings
from core.state import state

logger = logging.getLogger("tradeflow.risk")

class RiskGuard:
    """
    Strict Capital and Risk Protection layer.
    Ensures no order is sent to the broker without passing sizing, exposure, and loss bounds.
    """
    def can_open_position(self, symbol: str) -> Tuple[bool, str]:
        # 1. Kill switch check
        if not state.is_trading_active:
            return False, "Trading engine is currently paused by kill-switch."

        # 1b. Portfolio circuit breakers.
        # These gate NEW ENTRIES ONLY -- exits must always remain possible, or a
        # tripped breaker would trap the very positions that tripped it.
        state.roll_trading_day_if_needed()

        profile = state.risk_profile

        if state.daily_loss_pct >= profile.max_daily_loss_pct:
            reason = (f"Daily loss limit hit: -{state.daily_loss_pct:.2f}% of budget "
                      f"(limit {profile.max_daily_loss_pct}% at risk dial {profile.factor}). No new entries until tomorrow.")
            if state.halt_reason != reason:
                state.halt_reason = reason
                state.log_event("RISK_HALT", reason)
            return False, reason

        if state.drawdown_pct >= profile.max_drawdown_pct:
            reason = (f"Max drawdown breached: -{state.drawdown_pct:.2f}% from peak equity "
                      f"${state.peak_equity:,.2f} (limit {profile.max_drawdown_pct}% at risk dial {profile.factor}). "
                      f"No new entries until manually reset.")
            if state.halt_reason != reason:
                state.halt_reason = reason
                state.log_event("RISK_HALT", reason)
            return False, reason

        # 1c. Correlated-exposure cap. Five simultaneous L1 tokens is one macro bet,
        # not five independent positions, so cap concurrent holdings per asset class.
        from core.state import is_crypto_symbol
        same_class = sum(
            1 for s in state.active_positions
            if is_crypto_symbol(s) == is_crypto_symbol(symbol)
        )
        if same_class >= profile.max_positions_per_asset_class:
            klass = "crypto" if is_crypto_symbol(symbol) else "equity"
            return False, (f"Correlated exposure cap reached: already holding {same_class} "
                           f"{klass} positions (limit {profile.max_positions_per_asset_class} at risk dial {profile.factor}).")

        # 2. Duplicate or in-flight position check
        if symbol in state.active_positions:
            return False, f"Already holding an active position in {symbol}."

        from engine.executor import executor
        if symbol in executor.pending_orders:
            return False, f"An order is already in-flight for {symbol}."

        # 3. Check if broker supports trading this asset
        if executor.is_connected and not executor.is_mock_mode:
            if not executor.is_symbol_tradable(symbol):
                return False, f"{symbol} is in market monitoring mode (not listed for execution by Alpaca broker)."

        # 4. Max concurrent positions check
        if len(state.active_positions) >= profile.max_concurrent_positions:
            return False, f"Max concurrent positions limit ({profile.max_concurrent_positions} at risk dial {profile.factor}) reached."

        # 4. Allocated Capital Budget check (Do not allow using entire account equity)
        if state.total_position_exposure >= state.allocated_capital:
            return False, f"Allocated budget limit (${state.allocated_capital:,.2f}) reached. Current exposure: ${state.total_position_exposure:,.2f}."

        if state.remaining_budget < 30.0:
            return False, f"Remaining capital budget too low (${state.remaining_budget:,.2f} left of ${state.allocated_capital:,.2f})."

        # 5. Cash check
        from core.state import is_crypto_symbol
        cash = state.account_info.get("cash", 0.0)
        min_cash_required = 30.0 if is_crypto_symbol(symbol) else 100.0
        if cash < min_cash_required:
            return False, f"Insufficient available cash (${cash:,.2f} available, minimum ${min_cash_required:,.2f} required)."

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
