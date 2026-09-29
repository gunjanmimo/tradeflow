"""
Smart-money day trade: buy what insiders and top investors are buying.

Routed automatically to every tradable BUY in the smart-money book
(engine/smart_money.py: a sizeable insider open-market purchase in the last few
days, or broad ownership among the top eToro investors; liquid US stocks only).

Entry, once per symbol per day:
  * from SMART_MONEY_FIRST_ENTRY_MINUTES after the open (the first half hour is
    the noisiest and widest-spread part of the day)
  * the intraday structure is not falling (EMA stack not bearish), RSI not
    overbought, spread tight
Exit: the normal stop and target, flat by 15:50 like every day trade, or at
once if the symbol stops being a BUY (e.g. insiders turn sellers).

Caveat, stated plainly: insider-purchase drift plays out over weeks. Holding it
for a day pays the round-trip cost daily for a small slice of that drift. The
trades are recorded (entry_strategy "smart_money") so the result can be judged.
"""
import time

from core.config import settings
from engine.strategies.base import Strategy, StrategyContext, EntryDecision, ExitDecision


def _entered_today(symbol: str) -> bool:
    from core.state import state, ny_date
    today = ny_date()
    for rec in list(state.closed_trades):
        if (rec.get("symbol") == symbol and rec.get("entry_strategy") == "smart_money"
                and ny_date(float(rec.get("opened_at") or rec.get("time") or 0)) == today):
            return True
    return False


class SmartMoneyStrategy(Strategy):
    name = "smart_money"
    display_name = "Smart money (insiders + top investors)"
    description = ("Day-trades stocks with a recent sizeable insider open-market purchase or broad "
                   "ownership among top eToro investors: liquid US stocks only, one entry per day, "
                   "flat by the close. Exits early if the stock stops being a buy.")
    source = "SEC Form 4, eToro"
    params = {"min_trend_score": 0.35, "max_rsi": 75.0, "max_spread_pct": 0.004}

    def evaluate_entry(self, ctx: StrategyContext) -> EntryDecision:
        from engine.smart_money import smart_money
        from core.market_hours import us_session, minutes_to_close, REGULAR
        v = smart_money.verdict(ctx.symbol)
        gates = {"verdict": v["verdict"], "reasons": v["reasons"]}
        conviction = min(0.9, 0.6 + 0.1 * (v["insider_bought"] >= 1e6) + 0.1 * bool(v["etoro_breadth"]))
        if v["verdict"] != "BUY":
            return EntryDecision(False, 0.0, f"Smart money: {v['verdict']}", blocked_by="sm_not_buy", gates=gates)
        why_not = smart_money.tradable(ctx.symbol)
        if why_not:
            return EntryDecision(False, conviction, f"Smart money BUY but not tradable: {why_not}",
                                 blocked_by="sm_not_tradable", gates=gates)
        if us_session() != REGULAR:
            return EntryDecision(False, conviction, "Regular session only", blocked_by="session", gates=gates)
        mins = minutes_to_close(ctx.symbol)
        if mins is not None and 390 - mins < settings.SMART_MONEY_FIRST_ENTRY_MINUTES:
            return EntryDecision(False, conviction,
                                 f"Waiting {settings.SMART_MONEY_FIRST_ENTRY_MINUTES:.0f} min after the open",
                                 blocked_by="too_early", gates=gates)
        if _entered_today(ctx.symbol):
            return EntryDecision(False, conviction, "Already traded today (one entry per day)",
                                 blocked_by="once_per_day", gates=gates)
        q = ctx.quant
        trend = self._trend_score(q, ctx.price)
        gates["trend_score"] = round(trend, 3)
        if trend < self.params["min_trend_score"]:
            return EntryDecision(False, conviction, f"Intraday trend {trend:.2f} is falling; waiting",
                                 blocked_by="trend", gates=gates)
        if q and q.rsi is not None and q.rsi > self.params["max_rsi"]:
            return EntryDecision(False, conviction, f"RSI {q.rsi:.1f} overbought", blocked_by="rsi", gates=gates)
        if q and q.spread > self.params["max_spread_pct"]:
            return EntryDecision(False, conviction, f"Spread {q.spread * 100:.2f}% too wide",
                                 blocked_by="spread", gates=gates)
        return EntryDecision(True, conviction, "Smart money BUY: " + "; ".join(v["reasons"]), gates=gates)

    def evaluate_exit(self, ctx: StrategyContext) -> ExitDecision:
        from engine.smart_money import smart_money
        v = smart_money.verdict(ctx.symbol)
        if v["verdict"] != "BUY":
            return ExitDecision(True, 1.0, 1.0, f"No longer a smart-money buy ({v['verdict']}: "
                                                f"{'; '.join(v['reasons']) or 'signal expired'})")
        return ExitDecision(False, 0.0, 0.0, "Smart-money buy intact; stop, target and the close govern")
