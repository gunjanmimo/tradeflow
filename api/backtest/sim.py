"""
Bar-by-bar replay of the live trading rules over historical one-minute bars.

Reused from the live engine, not re-implemented: every strategy's own
evaluate_entry / evaluate_exit, the indicator maths (engine/quant_matrix.py,
including the true-range ATR) and stop/target derivation (engine/brackets.py).

The exit rules are the live sentinel's (engine/sentinel_agent.py), in its order:

  stop       the bar's low reaches the stop: filled at the stop, or at the open
             if the bar gapped through it. Checked before the target, so a bar
             that touches both is a loss (conservative).
  target     the bar's high reaches the take-profit: a resting limit, filled at
             the target (or the open if the bar gapped past it), no spread paid.
  end of day the position is flat before the close: a close is sent inside the
             flatten window and fills at the next bar's open, or at the day's
             last bar.
  strategy   the strategy's evaluate_exit at a bar's close; fills at the next
             bar's open.
  scale-out  engine/profit_manager.py, as the live executor runs it: when the
             bar's high reaches +SCALE_OUT_AT_R, SCALE_OUT_FRACTION of the shares
             (whole shares, one left at least) are sold at that level as a market
             order, and the stop moves to breakeven. Checked after the stop, so a
             bar that touches both is a loss.
  trail      after the scale-out, profit_manager.plan at each bar's close raises
             the stop behind the high; the new stop holds from the next bar.

Entries: the strategy's evaluate_entry at a bar's close; the market order fills
at the NEXT bar's open, so no signal trades on its own bar. Every market fill
pays core/costs.py's half-spread + slippage. No news history exists, so
sentiment is neutral with no headlines: news-driven strategies cannot be tested.

Sizing is a fixed notional per entry, one position per symbol at a time.
"""
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np

from core.config import settings
from core.costs import CostModel
from core.state import state, QuantMetrics, SentimentRecord
from engine import brackets, profit_manager
from engine.quant_matrix import QuantMatrix
from engine.strategies.base import StrategyContext
from engine.strategies.indicators import PriceSeries

HISTORY = 250          # the live per-symbol price buffer
WARMUP = 60
NY = ZoneInfo("America/New_York")


@dataclass
class Trade:
    symbol: str
    strategy: str
    entry_minute: int
    exit_minute: int = 0
    entry_price: float = 0.0
    exit_price: float = 0.0
    qty: float = 0.0
    pnl: float = 0.0              # net of spread and slippage
    costs: float = 0.0            # spread + slippage paid
    exit_reason: str = ""


def round_qty(qty: float) -> float:
    """As the live allocator: whole shares."""
    return float(math.floor(qty))


class Tape:
    """One symbol's bars plus everything derived from them, computed once and shared."""

    def __init__(self, symbol: str, bars, costs: Optional[CostModel] = None):
        a = np.asarray(bars, dtype=np.float64)
        self.symbol = symbol
        self.costs = costs or CostModel()
        self.minute = a[:, 0].astype(np.int64)
        self.o, self.h, self.l, self.c, self.v = a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5]
        self.n = len(a)
        self._ctx: Dict[int, StrategyContext] = {}
        # Minutes to the 16:00 close at the end of each bar, and the last bar of
        # each day present in the data (a half-day or a data gap).
        self.mins_to_close = np.full(self.n, np.inf)
        self.day_last = np.zeros(self.n, dtype=bool)
        days = []
        for i, m in enumerate(self.minute):
            t = datetime.fromtimestamp(int(m) * 60, tz=NY)
            self.mins_to_close[i] = 16 * 60 - (t.hour * 60 + t.minute + 1)
            days.append(t.date())
        for i in range(self.n):
            self.day_last[i] = i == self.n - 1 or days[i + 1] != days[i]
        self.day = days

    def atr(self, i: int, price: float) -> float:
        """ATR at the close of bar i from the true range of the last 60 bars, as live."""
        lo = max(0, i - 59)
        a = QuantMatrix.true_range_atr(self.h[lo:i + 1], self.l[lo:i + 1], self.c[lo:i + 1])
        if a is None:
            n = i - lo + 1
            return QuantMatrix.calculate_atr(self.c[lo:i + 1], period=min(14, max(n - 1, 1)),
                                             ref_price=price)
        return max(a, price * 0.0005)

    def ctx(self, i: int) -> StrategyContext:
        """The StrategyContext a live strategy would see at the close of bar i."""
        hit = self._ctx.get(i)
        if hit is not None:
            return hit
        lo = max(0, i - HISTORY + 1)
        prices = self.c[lo:i + 1]
        vols = self.v[lo:i + 1]
        price = float(self.c[i])
        n = len(prices)
        rsi_p = min(14, n - 1)
        vr = self.v[max(0, i - 19):i + 1]
        quant = QuantMetrics(
            symbol=self.symbol,
            rsi=QuantMatrix.calculate_rsi(prices, period=rsi_p) or 50.0,
            ema_fast=QuantMatrix.calculate_ema(prices, period=min(9, n)),
            ema_slow=QuantMatrix.calculate_ema(prices, period=min(21, n)),
            atr=self.atr(i, price),
            spread=2 * self.costs.half_spread_bps / 1e4,
            volume_ratio=float(self.v[i] / vr.mean()) if vr.mean() > 0 else 1.0,
            updated_at=float(self.minute[i] * 60),
        )
        ctx = StrategyContext(
            symbol=self.symbol, price=price, quant=quant,
            sentiment=SentimentRecord(stock_id=self.symbol, headline="(backtest: no news history)"),
            consensus=None, series=PriceSeries(prices, vols))
        ctx._min_buy_prob = state.risk_profile.min_buy_prob
        ctx._bar_index = i
        ctx._tape = self
        self._ctx[i] = ctx
        return ctx

    def can_enter(self, i: int) -> bool:
        return (self.mins_to_close[i] > settings.NO_NEW_ENTRY_MINUTES_BEFORE_CLOSE
                and not self.day_last[i] and i < self.n - 1)


class Book:
    """One open position's fills and bookkeeping."""

    def __init__(self, tape: Tape, strat_name: str, i: int, notional: float):
        self.t = tape
        mid = float(tape.o[i])
        fill = tape.costs.buy_fill(mid)
        self.qty = round_qty(notional / fill)
        ctx = tape.ctx(max(i - 1, 0))
        stop, tp, _ = brackets.derive(fill, ctx.quant.atr)
        self.pos: Dict[str, Any] = {"symbol": tape.symbol, "qty": self.qty, "avg_entry_price": fill,
                                    "stop_loss": stop, "take_profit": tp, "initial_stop": stop,
                                    "opened_at": float(tape.minute[i] * 60), "entry_strategy": strat_name}
        self.highest = fill
        self.trade = Trade(tape.symbol, strat_name, int(tape.minute[i]), entry_price=fill, qty=self.qty)
        self.trade.costs += self.qty * (fill - mid)
        self.pending_exit: Optional[str] = None
        self.realized = 0.0            # P&L of shares already sold by a scale-out

    def sell_part(self, qty: float, price: float):
        fill = self.t.costs.sell_fill(price)
        self.trade.costs += qty * (price - fill)
        self.realized += (fill - self.entry) * qty
        self.qty -= qty
        self.pos["qty"] = self.qty

    @property
    def entry(self) -> float:
        return float(self.pos["avg_entry_price"])

    def close(self, i: int, price: float, reason: str, market: bool = True) -> Trade:
        fill = self.t.costs.sell_fill(price) if market else price
        self.trade.costs += self.qty * (price - fill)
        self.trade.exit_minute = int(self.t.minute[i])
        self.trade.exit_price = fill
        self.trade.pnl = round(self.realized + (fill - self.entry) * self.qty, 4)
        self.trade.exit_reason = reason
        return self.trade


def step(tape: Tape, i: int, b: Book, strat) -> Optional[Trade]:
    """Runs bar i against an open position. Returns the Trade if it closed."""
    o, h, l, c = (float(x[i]) for x in (tape.o, tape.h, tape.l, tape.c))

    # An exit decided at the previous bar's close fills at this bar's open.
    if b.pending_exit:
        return b.close(i, o, b.pending_exit)

    stop, tp = float(b.pos["stop_loss"]), float(b.pos["take_profit"])
    if o <= stop:
        return b.close(i, o, "stop (gap)")
    if l <= stop:
        return b.close(i, stop, "stop")
    if o >= tp:
        return b.close(i, o, "target (gap)", market=False)
    if h >= tp:
        return b.close(i, tp, "target", market=False)
    take_profit(b, o, h, c)
    b.highest = max(b.highest, h)

    if tape.day_last[i]:
        return b.close(i, c, "end of day")
    if settings.DAY_TRADE_FLATTEN_ENABLED and tape.mins_to_close[i] <= settings.FLATTEN_MINUTES_BEFORE_CLOSE:
        b.pending_exit = "end of day"
        return None

    ctx = tape.ctx(i)
    pctx = StrategyContext(symbol=ctx.symbol, price=c, quant=ctx.quant, sentiment=ctx.sentiment,
                           consensus=None, position=b.pos, highest_price=b.highest, series=ctx.series)
    pctx._min_buy_prob = ctx._min_buy_prob
    pctx._bar_index, pctx._tape = i, tape
    pctx._now = float((tape.minute[i] + 1) * 60)
    try:
        ex = strat.evaluate_exit(pctx)
    except Exception:
        ex = None
    if ex is not None and ex.should_close:
        b.pending_exit = f"strategy: {ex.reason[:60]}"
    return None


def take_profit(b: Book, o: float, h: float, c: float):
    """The live scale-out and trailing stop (engine/profit_manager.py) on one bar."""
    pos = b.pos
    r = profit_manager.risk_per_share(pos)
    if not settings.PROFIT_TAKING_ENABLED or r <= 0:
        return
    if not pos.get("scaled_out"):
        level = max(b.entry + settings.SCALE_OUT_AT_R * r, o)
        if h < level:
            return
        act = profit_manager.plan(pos, level, max(b.highest, level))
        if not act or act["type"] != "scale_out":
            return
        sold = math.floor(b.qty * act["fraction"])
        if sold >= 1 and b.qty - sold >= 1:        # as executor._scale_out
            b.sell_part(sold, level)
        pos["scaled_out"] = True
        pos["stop_loss"] = act["new_stop"]
        return
    act = profit_manager.plan(pos, c, max(b.highest, h))
    if act and act["type"] == "raise_stop":
        pos["stop_loss"] = act["new_stop"]


def run(tape: Tape, strat, notional: float) -> List[Trade]:
    """Replays one strategy on one symbol under the current settings."""
    state.is_trading_active = True
    trades: List[Trade] = []
    book: Optional[Book] = None
    enter_next = False
    for i in range(WARMUP, tape.n):
        if enter_next:
            enter_next = False
            if not tape.day_last[i - 1]:          # never carry a signal across the close
                book = Book(tape, strat.name, i, notional)
                if book.qty <= 0:
                    book = None
        if book is not None:
            t = step(tape, i, book, strat)
            if t is not None:
                trades.append(t)
                book = None
                continue
        if book is None and tape.can_enter(i):
            try:
                ctx = tape.ctx(i)
                ctx._now = float((tape.minute[i] + 1) * 60)
                enter_next = bool(strat.evaluate_entry(ctx).should_enter)
            except Exception:
                enter_next = False
    if book is not None:
        trades.append(book.close(tape.n - 1, float(tape.c[-1]), "end of data"))
    return trades
