import logging
from typing import Dict, Any, Tuple, Optional
from core.state import state, is_crypto_symbol, round_price, qty_decimals
from core.config import settings
from engine import brackets

logger = logging.getLogger("tradeflow.allocation")

class LayaAllocationManager:
    """
    The Laya-driven AI Portfolio Capital Allocation Manager.
    Decides the exact dollar amount of money to allocate to each stock or cryptocurrency
    based on Laya's conviction probability, consensus alignment, and volatility risk budgeting.
    """
    def evaluate_allocation(
        self, 
        symbol: str, 
        current_price: float, 
        atr: float, 
        laya_pos_prob: float, 
        consensus_score: Optional[float]
    ) -> Dict[str, Any]:
        """
        Calculates:
        1. Exact dollar amount to allocate ($).
        2. Portfolio percentage (%).
        3. Fractional or whole share quantity.
        4. Clear AI rationale explaining why this exact amount was allocated.
        """
        # Strict Capital Budget: Do NOT use the entire 100k account equity!
        # Base calculations on user-configured allocated capital budget (default $10,000.00)
        # The hard cap already reflects realised losses, and remaining_budget is
        # measured on cost basis plus in-flight buys (see core/state.py).
        equity = state.budget_base
        bot_cash = state.bot_cash
        # Hold back a fill buffer so market-order slippage cannot breach the cap.
        remaining_budget = state.remaining_budget / (1 + settings.BUDGET_FILL_BUFFER_PCT / 100.0)

        # 1. Conviction Multiplier from Laya & Consensus.
        # An absent consensus (no real source covered this symbol) contributes a
        # NEUTRAL 0.5, which lowers composite conviction and therefore size. Absence
        # of evidence must never read as evidence of strength.
        has_consensus = consensus_score is not None
        consensus_component = float(consensus_score) if has_consensus else 0.5
        composite_conviction = (0.60 * laya_pos_prob) + (0.40 * consensus_component)

        if composite_conviction >= 0.85:
            conviction_tier = "AGGRESSIVE_HIGH_CONVICTION"
            risk_multiplier = 1.50
        elif composite_conviction >= 0.75:
            conviction_tier = "STRONG_CONVICTION"
            risk_multiplier = 1.00
        elif composite_conviction >= 0.65:
            conviction_tier = "STANDARD_CONVICTION"
            risk_multiplier = 0.60
        else:
            conviction_tier = "CONSERVATIVE_PROBING"
            risk_multiplier = 0.25

        # 2. Dynamic Stop Loss & Take Profit from volatility, via the one shared
        # derivation used by every other component (see engine/brackets.py).
        stop_loss_price, take_profit_price, stop_distance = brackets.derive(current_price, atr)

        # 3. RISK-BASED position sizing.
        #
        # Size is derived from the money we are willing to lose and how far away the
        # stop sits: qty = risk_dollars / stop_distance. This is the whole point of
        # RISK_PER_TRADE_PERCENT, which was previously computed and discarded while
        # sizing actually ran off flat notional tiers -- meaning a 1% stop and a 20%
        # stop received identical dollar allocations and wildly different real risk.
        profile = state.risk_profile
        risk_pct = profile.risk_per_trade_pct / 100.0
        risk_dollars = equity * risk_pct * risk_multiplier

        if stop_distance <= 0:
            return self._reject(
                symbol, stop_loss_price, take_profit_price, conviction_tier, composite_conviction,
                f"Laya Manager: Non-positive stop distance computed for {symbol}; refusing to size a position without a valid stop."
            )

        target_qty = risk_dollars / stop_distance
        uncapped_dollars = target_qty * current_price

        # Notional ceiling: risk-based sizing on a very tight stop implies a huge
        # notional (risking 1% on a 0.8% stop needs ~125% of budget), so the cap
        # will often bind. When it does, realised risk is BELOW target -- safe, but
        # the operator must be told rather than left believing the target was met.
        notional_cap = equity * (profile.max_position_notional_pct / 100.0)
        allocated_dollars = min(uncapped_dollars, notional_cap, remaining_budget)
        notional_capped = uncapped_dollars > min(notional_cap, remaining_budget) + 1e-9

        # Diversification headroom: the sector/region/crypto sleeve this symbol
        # belongs to may have less room than the single-position cap allows, and
        # a position highly correlated with a holding is halved.
        from engine.diversification import diversification
        div = diversification.assess(symbol)
        div_note = ""
        if div.max_dollars < allocated_dollars:
            allocated_dollars = div.max_dollars
            why = div.notes[-1] if div.size_scale < 1.0 else div.binding
            div_note = f" [DIVERSIFICATION-CAPPED to ${div.max_dollars:,.2f}: {why}]"

        # 4. Cash and per-position notional caps: strictly within bot's hard cap
        if is_crypto_symbol(symbol):
            # Crypto on Alpaca is non-marginable and settles in USD cash.
            # Keep a 5% cushion for fees and price drift between sizing and fill.
            safe_cash = max(0.0, min(remaining_budget, bot_cash * 0.95))
            min_order_value = 15.0
        else:
            safe_cash = max(0.0, min(remaining_budget, bot_cash - 5.0))
            min_order_value = 30.0

        allocated_dollars = min(allocated_dollars, safe_cash)

        if allocated_dollars < min_order_value:
            return self._reject(
                symbol, stop_loss_price, take_profit_price, conviction_tier, composite_conviction,
                f"Laya Manager: Insufficient bot budget/cash for {symbol} "
                f"(remaining ${remaining_budget:,.2f}, bot cash ${bot_cash:,.2f}, "
                f"need >= ${min_order_value:,.2f}). Main broker equity is locked."
            )

        # 5. Quantity at a precision appropriate to the asset's price scale.
        # A flat 2-decimal rounding silently zeroes the quantity of any sub-cent
        # asset (SHIB at $0.0000235 needs whole units, not hundredths).
        if is_crypto_symbol(symbol):
            decimals = qty_decimals(current_price)
            qty = round(float(allocated_dollars / current_price), decimals)
            step = 10 ** -decimals
            # Never let rounding push the order above available cash or budget
            while qty > 0 and (qty * current_price) > min(remaining_budget, safe_cash):
                qty = round(qty - step, decimals)
            if qty <= 0:
                return self._reject(
                    symbol, stop_loss_price, take_profit_price, conviction_tier, composite_conviction,
                    f"Laya Manager: Computed quantity rounds to zero for {symbol} at ${current_price}."
                )
        else:
            qty = int(allocated_dollars / current_price)
            if qty < 1:
                return self._reject(
                    symbol, stop_loss_price, take_profit_price, conviction_tier, composite_conviction,
                    f"Laya Manager: Cannot afford a single share of {symbol} at ${current_price:,.2f} "
                    f"within risk budget (${allocated_dollars:,.2f} allocatable)."
                )

        allocated_dollars = round(qty * current_price, 2)
        allocated_pct = round((allocated_dollars / equity) * 100, 2) if equity > 0 else 0.0

        # Actual risk carried, after all rounding and clamping. This is the number
        # the ledger and the reward function care about, not the intended risk.
        actual_risk_dollars = round(abs(current_price - stop_loss_price) * qty, 2)
        actual_reward_dollars = round(abs(take_profit_price - current_price) * qty, 2)
        actual_risk_pct = round((actual_risk_dollars / equity) * 100, 3) if equity > 0 else 0.0

        cap_note = ""
        if notional_capped:
            cap_note = (
                f" [NOTIONAL-CAPPED: target risk was ${risk_dollars:,.2f} "
                f"({profile.risk_per_trade_pct * risk_multiplier:.2f}% of budget) but the "
                f"{profile.max_position_notional_pct:.0f}% position cap limits actual risk to "
                f"${actual_risk_dollars:,.2f}]"
            )

        rationale = (
            f"Laya Manager: Allocated ${allocated_dollars:,.2f} ({allocated_pct}% of budget) into {symbol} "
            f"[{conviction_tier}, risk dial {profile.factor}/10 {profile.label}]. Conviction: {composite_conviction*100:.0f}% (Laya={laya_pos_prob*100:.0f}%, "
            f"Consensus={consensus_component*100:.0f}%{'' if has_consensus else ' [NO REAL SOURCE]'}). "
            f"Risk ${actual_risk_dollars:,.2f} ({actual_risk_pct}% of budget) "
            f"to make ${actual_reward_dollars:,.2f}. Stop {stop_distance/current_price*100:.2f}% away "
            f"(SL ${stop_loss_price} | TP ${take_profit_price}).{cap_note}{div_note}"
            f" Sleeve: {div.meta.sector} / {div.meta.region}."
        )

        return {
            "symbol": symbol,
            "allocated_dollars": allocated_dollars,
            "allocated_pct": allocated_pct,
            "qty": qty,
            "stop_loss": stop_loss_price,
            "take_profit": take_profit_price,
            "conviction_tier": conviction_tier,
            "composite_conviction": round(composite_conviction, 3),
            "risk_dollars": actual_risk_dollars,
            "reward_dollars": actual_reward_dollars,
            "risk_pct": actual_risk_pct,
            "stop_distance": round_price(stop_distance, current_price),
            "atr": atr,
            "rationale": rationale
        }

    @staticmethod
    def _reject(symbol, stop_loss, take_profit, tier, conviction, rationale) -> Dict[str, Any]:
        """A zero-quantity allocation. Callers treat qty <= 0 as 'do not trade'."""
        return {
            "symbol": symbol,
            "allocated_dollars": 0.0,
            "allocated_pct": 0.0,
            "qty": 0.0,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "conviction_tier": tier,
            "composite_conviction": round(conviction, 3),
            "risk_dollars": 0.0,
            "reward_dollars": 0.0,
            "risk_pct": 0.0,
            "rationale": rationale
        }

allocation_manager = LayaAllocationManager()
