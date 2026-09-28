"""Persistent daily profit-and-loss totals for bot-managed trades."""
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("tradeflow.pnl")

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_PATH = os.path.join(_DATA_DIR, "pnl_ledger.json")
_MAX_DAYS = 31


def _empty_day() -> Dict[str, Any]:
    return {
        "realized_pnl": 0.0,
        "gross_profit": 0.0,
        "gross_loss": 0.0,
        "closed_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "harvested_income": 0.0,  # profit-harvest gains set aside as day income
        "harvests": 0,
    }


class DailyPnLLedger:
    def __init__(self):
        self.days: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        try:
            with open(_PATH) as file:
                raw = json.load(file)
            days = raw.get("days", {})
            if isinstance(days, dict):
                self.days = {
                    str(day): {**_empty_day(), **values}
                    for day, values in days.items()
                    if isinstance(values, dict)
                }
        except FileNotFoundError:
            pass
        except Exception as error:
            logger.error("Could not load daily P&L ledger: %s", error)
            self.days = {}

    def _save(self):
        try:
            os.makedirs(_DATA_DIR, exist_ok=True)
            temporary_path = _PATH + ".tmp"
            with open(temporary_path, "w") as file:
                json.dump({"days": self.days}, file, indent=2, sort_keys=True)
            os.replace(temporary_path, _PATH)
        except Exception as error:
            logger.error("Could not persist daily P&L ledger: %s", error)

    @staticmethod
    def _today() -> str:
        from core.state import ny_date
        return ny_date()

    def roll(self, trading_day: str | None = None) -> Dict[str, Any]:
        day = trading_day or self._today()
        if day not in self.days:
            self.days[day] = _empty_day()
            if len(self.days) > _MAX_DAYS:
                for old_day in sorted(self.days)[:-_MAX_DAYS]:
                    self.days.pop(old_day, None)
            self._save()
        return self.days[day]

    def touch(self):
        self.roll()

    def realized_today(self) -> float:
        return round(float(self.roll().get("realized_pnl") or 0.0), 2)

    def on_close(self, symbol: str, pnl: float):
        day = self.roll()
        value = round(float(pnl), 2)
        day["realized_pnl"] = round(float(day["realized_pnl"]) + value, 2)
        day["closed_trades"] = int(day["closed_trades"]) + 1
        if value > 0:
            day["gross_profit"] = round(float(day["gross_profit"]) + value, 2)
            day["winning_trades"] = int(day["winning_trades"]) + 1
        elif value < 0:
            day["gross_loss"] = round(float(day["gross_loss"]) + abs(value), 2)
            day["losing_trades"] = int(day["losing_trades"]) + 1
        self._save()

    def revise(self, symbol: str, old_pnl: float, new_pnl: float):
        """
        Replaces a closed trade's provisional P&L (booked from the last mark when
        the close was sent) with the broker's fill. Re-classifies the trade as a
        win or a loss if the fill moved it across zero; the trade count is kept.
        """
        day = self.roll()
        old, new = round(float(old_pnl), 2), round(float(new_pnl), 2)
        day["realized_pnl"] = round(float(day["realized_pnl"]) - old + new, 2)
        for value, sign in ((old, -1), (new, 1)):
            if value > 0:
                day["gross_profit"] = round(float(day["gross_profit"]) + sign * value, 2)
                day["winning_trades"] = int(day["winning_trades"]) + sign
            elif value < 0:
                day["gross_loss"] = round(float(day["gross_loss"]) + sign * abs(value), 2)
                day["losing_trades"] = int(day["losing_trades"]) + sign
        self._save()

    def harvested_today(self) -> float:
        return round(float(self.roll().get("harvested_income") or 0.0), 2)

    def calculate_scenarios(self, active_positions: Dict[str, Any], realized_today: float) -> Dict[str, Any]:
        """Calculates potential PnL outcomes if all open positions hit TP vs SL."""
        best_case_gain = 0.0
        worst_case_loss = 0.0
        open_unrealized = 0.0
        positions_detail = []

        for sym, pos in active_positions.items():
            qty = float(pos.get("qty", 0.0))
            avg_entry = float(pos.get("avg_entry_price", 0.0))
            cur_price = float(pos.get("current_price", avg_entry))
            sl = float(pos.get("stop_loss") or 0.0)
            tp = float(pos.get("take_profit") or 0.0)
            unrealized = round(float(pos.get("unrealized_pl") or (cur_price - avg_entry) * qty), 2)
            open_unrealized += unrealized

            tp_pnl = round((tp - avg_entry) * qty, 2) if tp > 0 else unrealized
            sl_pnl = round((sl - avg_entry) * qty, 2) if sl > 0 else -round(avg_entry * qty, 2)

            best_case_gain += tp_pnl
            worst_case_loss += sl_pnl

            positions_detail.append({
                "symbol": sym,
                "qty": qty,
                "avg_entry_price": avg_entry,
                "current_price": cur_price,
                "unrealized_pl": unrealized,
                "stop_loss": sl,
                "take_profit": tp,
                "tp_pnl": tp_pnl,
                "sl_pnl": sl_pnl,
            })

        best_case_gain = round(best_case_gain, 2)
        worst_case_loss = round(worst_case_loss, 2)
        open_unrealized = round(open_unrealized, 2)

        return {
            "open_positions_count": len(active_positions),
            "open_unrealized": open_unrealized,
            "best_case_tp_gain": best_case_gain,
            "worst_case_sl_loss": worst_case_loss,
            "projected_best_net": round(realized_today + best_case_gain, 2),
            "projected_worst_net": round(realized_today + worst_case_loss, 2),
            "projected_current_net": round(realized_today + open_unrealized, 2),
            "positions": positions_detail,
        }

    def get_history(self, limit: int = 31) -> List[Dict[str, Any]]:
        """Returns sorted list of daily performance records (newest first)."""
        history = []
        for day in sorted(self.days.keys(), reverse=True)[:limit]:
            d = self.days[day]
            realized = round(float(d.get("realized_pnl", 0.0)), 2)
            gross_p = round(float(d.get("gross_profit", 0.0)), 2)
            gross_l = round(float(d.get("gross_loss", 0.0)), 2)
            closed = int(d.get("closed_trades", 0))
            winning = int(d.get("winning_trades", 0))
            losing = int(d.get("losing_trades", 0))
            win_rate = round((winning / closed * 100.0), 1) if closed > 0 else 0.0
            profit_factor = None if gross_l == 0 else round(gross_p / gross_l, 3)
            history.append({
                "date": day,
                "realized_pnl": realized,
                "gross_profit": gross_p,
                "gross_loss": gross_l,
                "closed_trades": closed,
                "winning_trades": winning,
                "losing_trades": losing,
                "win_rate_pct": win_rate,
                "profit_factor": profit_factor,
            })
        return history

    def calculate_projections(
        self,
        target_daily_profit: float = 100.0,
        max_daily_loss: float = 50.0,
        planned_trades: int = 5,
        win_rate_pct: float = 60.0,
        reward_risk_ratio: float = 2.0,
        risk_per_trade: float = 20.0,
        budget: float = 1000.0,
        current_net_pnl: float = 0.0,
    ) -> Dict[str, Any]:
        """Calculates mathematical expectancy, daily target progress, and projected return."""
        import math
        win_rate = max(0.0, min(100.0, float(win_rate_pct))) / 100.0
        rr = max(0.1, float(reward_risk_ratio))
        risk = max(1.0, float(risk_per_trade))
        trades = max(1, int(planned_trades))
        target = float(target_daily_profit)
        loss_limit = max(1.0, float(max_daily_loss))

        win_amount = round(risk * rr, 2)
        loss_amount = round(risk, 2)
        expected_value_per_trade = round((win_rate * win_amount) - ((1.0 - win_rate) * loss_amount), 2)
        projected_daily_pnl = round(expected_value_per_trade * trades, 2)
        projected_return_pct = round(projected_daily_pnl / budget * 100.0, 2) if budget > 0 else 0.0
        breakeven_win_rate_pct = round((1.0 / (1.0 + rr)) * 100.0, 1)

        # Progress toward daily goals
        target_gap = max(0.0, round(target - current_net_pnl, 2)) if target > 0 else 0.0
        target_progress_pct = round(min(100.0, max(0.0, (current_net_pnl / target) * 100.0)), 1) if target > 0 else 0.0
        current_drawdown_amount = abs(min(0.0, current_net_pnl))
        loss_limit_used_pct = round(min(100.0, max(0.0, (current_drawdown_amount / loss_limit) * 100.0)), 1)
        remaining_loss_cushion = max(0.0, round(loss_limit - current_drawdown_amount, 2))

        trades_to_target = math.ceil(target_gap / expected_value_per_trade) if (expected_value_per_trade > 0 and target_gap > 0) else 0
        trades_before_loss_limit = math.floor(remaining_loss_cushion / loss_amount) if loss_amount > 0 else 0

        status = "ON_TRACK"
        if current_net_pnl >= target and target > 0:
            status = "TARGET_REACHED"
        elif loss_limit_used_pct >= 100.0:
            status = "LOSS_LIMIT_REACHED"
        elif loss_limit_used_pct >= 80.0:
            status = "NEAR_LOSS_LIMIT"
        elif expected_value_per_trade < 0:
            status = "NEGATIVE_EXPECTANCY"

        return {
            "target_daily_profit": target,
            "max_daily_loss": loss_limit,
            "planned_trades": trades,
            "win_rate_pct": win_rate_pct,
            "reward_risk_ratio": rr,
            "risk_per_trade": risk,
            "budget": budget,
            "current_net_pnl": current_net_pnl,
            "expected_win_amount": win_amount,
            "expected_loss_amount": loss_amount,
            "expected_value_per_trade": expected_value_per_trade,
            "projected_daily_pnl": projected_daily_pnl,
            "projected_return_pct": projected_return_pct,
            "breakeven_win_rate_pct": breakeven_win_rate_pct,
            "target_progress_pct": target_progress_pct,
            "target_gap": target_gap,
            "loss_limit_used_pct": loss_limit_used_pct,
            "remaining_loss_cushion": remaining_loss_cushion,
            "trades_to_target": trades_to_target,
            "trades_before_loss_limit": trades_before_loss_limit,
            "status": status,
        }

    def snapshot(
        self,
        unrealized_pnl: float,
        budget: float,
        active_positions: Optional[Dict[str, Any]] = None,
        total_equity: float = 0.0,
        locked_equity: float = 0.0
    ) -> Dict[str, Any]:
        day = self.roll()
        realized = round(float(day["realized_pnl"]), 2)
        unrealized = round(float(unrealized_pnl), 2)
        gross_loss = round(float(day["gross_loss"]), 2)
        gross_profit = round(float(day["gross_profit"]), 2)
        profit_factor = None if gross_loss == 0 else round(gross_profit / gross_loss, 3)
        net = round(realized + unrealized, 2)
        closed = int(day["closed_trades"])
        winning = int(day["winning_trades"])
        losing = int(day["losing_trades"])
        win_rate = round((winning / closed * 100.0), 1) if closed > 0 else 0.0
        avg_win = round(gross_profit / winning, 2) if winning > 0 else 0.0
        avg_loss = round(gross_loss / losing, 2) if losing > 0 else 0.0

        scenarios = self.calculate_scenarios(active_positions or {}, realized)

        return {
            "trading_day": self._today(),
            "realized_pnl": realized,
            "unrealized_pnl": unrealized,
            "net_pnl": net,
            "return_pct": round(net / budget * 100.0, 3) if budget > 0 else 0.0,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "profit_factor": profit_factor,
            "closed_trades": closed,
            "winning_trades": winning,
            "losing_trades": losing,
            "win_rate_pct": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "total_broker_equity": total_equity,
            "locked_broker_equity": locked_equity,
            "return_on_total_equity_pct": round(net / total_equity * 100.0, 3) if total_equity > 0 else 0.0,
            "scenarios": scenarios,
            # Profit harvests: gains set aside as day income, never traded again.
            "harvested_income": round(float(day.get("harvested_income") or 0.0), 2),
            "harvests": int(day.get("harvests") or 0),
        }


pnl_ledger = DailyPnLLedger()
