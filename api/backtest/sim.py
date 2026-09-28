"""
Bar-by-bar replay of the live trading rules over historical one-minute bars.

Reused from the live engine, not re-implemented: every strategy's own
evaluate_entry / evaluate_exit, the trend analyst (engine/trend.py), the
indicator maths (engine/quant_matrix.py), stop/target derivation and the
trailing stop (engine/brackets.py), and loss recovery's disaster stop, rescue
check and break-even lock (engine/loss_recovery.py). Settings are read live, so
an exit variant is just a set of settings overrides.

What a one-minute bar cannot show, and how it is modelled (conservatively):

  * order inside a bar    stop is checked before harvest and target; if a bar
                          touches both the stop and the target, the stop wins
  * signals               computed on a bar's close; a new entry fills at the
                          NEXT bar's open, so no signal trades on its own bar
  * stop confirmation     the live stop must hold STOP_CONFIRM_SECONDS (20s);
                          here a bar that CLOSES below the stop confirms it, and
                          a wick below that closes back above does not
  * profit harvest        fires at the first price where the bid beats the
                          reference (entry, last harvest, or the USD minimum),
                          or the bar's open if it gapped past it
  * spread and fees       every market fill pays half the spread; crypto pays
                          Alpaca's taker fee on both sides. No news history
                          exists, so sentiment is neutral with no headlines:
                          news-driven strategies and exits cannot be tested.

Sizing is a fixed notional per entry (the live allocator sizes by conviction
and risk dial), one position per symbol at a time.
"""
import contextlib
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np

from core.config import settings
from core.state import state, QuantMetrics, SentimentRecord, is_crypto_symbol, qty_decimals
from engine import brackets, loss_recovery, profit_harvest
from engine.quant_matrix import QuantMatrix
from engine.strategies.base import StrategyContext
from engine.strategies.indicators import PriceSeries
from engine.trend import analyze

HISTORY = 250          # the live per-symbol price buffer
TREND_WINDOW = 390     # one session of minute bars, as core/minute_bars keeps
WARMUP = 60
NY = ZoneInfo("America/New_York")

# Exit rule sets compared by the backtest. "before" is the platform as it was
# at the start of this work; "current" is what is configured now.
VARIANTS: Dict[str, Dict[str, Any]] = {
    "no_harvest":  dict(PROFIT_HARVEST_ENABLED=False, STOP_CONFIRM_ENABLED=False, RECOVERY_ENABLED=False),
    "before":      dict(PROFIT_HARVEST_ENABLED=True, PROFIT_HARVEST_USD=3.0,
                        STOP_CONFIRM_ENABLED=False, RECOVERY_ENABLED=False),
    "harvest_any": dict(PROFIT_HARVEST_ENABLED=True, PROFIT_HARVEST_USD=0.0,
                        STOP_CONFIRM_ENABLED=False, RECOVERY_ENABLED=False),
    "recovery":    dict(PROFIT_HARVEST_ENABLED=True, PROFIT_HARVEST_USD=3.0,
                        STOP_CONFIRM_ENABLED=True, RECOVERY_ENABLED=True),
    "current":     dict(PROFIT_HARVEST_ENABLED=True, PROFIT_HARVEST_USD=0.0,
                        STOP_CONFIRM_ENABLED=True, RECOVERY_ENABLED=True),
    # current, one loss-recovery part at a time
    "current_no_rescue":  dict(PROFIT_HARVEST_ENABLED=True, PROFIT_HARVEST_USD=0.0,
                               STOP_CONFIRM_ENABLED=True, RECOVERY_ENABLED=True,
                               RECOVERY_ADD_ENABLED=False),
    "current_plain_stop": dict(PROFIT_HARVEST_ENABLED=True, PROFIT_HARVEST_USD=0.0,
                               STOP_CONFIRM_ENABLED=False, RECOVERY_ENABLED=True),
}


@dataclass
class Costs:
    stock_half_spread_bps: float = 2.0
    crypto_half_spread_bps: float = 5.0
    crypto_fee_bps: float = 25.0      # Alpaca crypto taker fee, lowest volume tier


@dataclass
class Trade:
    symbol: str
    strategy: str
    entry_minute: int
    exit_minute: int = 0
    entry_price: float = 0.0
    pnl: float = 0.0              # net of spread and fees, all fills of the position
    harvested: float = 0.0        # part of pnl booked as harvest income
    costs: float = 0.0            # spread + fees paid
    exit_reason: str = ""
    harvests: int = 0
    rescued: bool = False
    recovered: bool = False       # break-even lock engaged


@contextlib.contextmanager
def overrides(values: Dict[str, Any]):
    saved = {k: getattr(settings, k) for k in values}
    try:
        for k, v in values.items():
            setattr(settings, k, v)
        yield
    finally:
        for k, v in saved.items():
            setattr(settings, k, v)


def round_qty(symbol: str, qty: float, price: float) -> float:
    """As AlpacaExecutor._round_qty: whole shares, or crypto precision."""
    if is_crypto_symbol(symbol):
        d = qty_decimals(price)
        return math.floor(qty * 10 ** d) / 10 ** d
    return float(math.floor(qty))


class Tape:
    """One symbol's bars plus everything derived from them, computed once and shared."""

    def __init__(self, symbol: str, bars, costs: Costs):
        a = np.asarray(bars, dtype=np.float64)
        self.symbol = symbol
        self.minute = a[:, 0].astype(np.int64)
        self.o, self.h, self.l, self.c, self.v = a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5]
        self.n = len(a)
        self.crypto = is_crypto_symbol(symbol)
        self.hs = (costs.crypto_half_spread_bps if self.crypto else costs.stock_half_spread_bps) / 1e4
        self.fee = costs.crypto_fee_bps / 1e4 if self.crypto else 0.0
        self._ctx: Dict[int, tuple] = {}
        self._entry: Dict[tuple, bool] = {}
        # Stocks: minutes to the 16:00 close at the end of each bar, and the last
        # bar of each day present in the data (a half-day or a data gap).
        self.mins_to_close = np.full(self.n, np.inf)
        self.day_last = np.zeros(self.n, dtype=bool)
        if not self.crypto:
            days = []
            for i, m in enumerate(self.minute):
                t = datetime.fromtimestamp(int(m) * 60, tz=NY)
                self.mins_to_close[i] = 16 * 60 - (t.hour * 60 + t.minute + 1)
                days.append(t.date())
            for i in range(self.n):
                self.day_last[i] = i == self.n - 1 or days[i + 1] != days[i]
            self.day = days

    def ctx(self, i: int):
        """(StrategyContext, TrendRead) at the close of bar i."""
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
            atr=QuantMatrix.calculate_atr(prices, period=rsi_p, ref_price=price),
            spread=2 * self.hs,
            volume_ratio=float(self.v[i] / vr.mean()) if vr.mean() > 0 else 1.0,
            updated_at=float(self.minute[i] * 60),
        )
        ctx = StrategyContext(
            symbol=self.symbol, price=price, quant=quant,
            sentiment=SentimentRecord(stock_id=self.symbol, headline="(backtest: no news history)"),
            consensus=None, is_crypto=self.crypto,
            series=PriceSeries(prices, vols))
        ctx._min_buy_prob = state.risk_profile.min_buy_prob
        trend = analyze(self.symbol, closes=self.c[max(0, i - TREND_WINDOW + 1):i + 1], daily=None)
        self._ctx[i] = (ctx, trend)
        return ctx, trend

    def entry_signal(self, strat, i: int) -> bool:
        """The live entry path: the trend gate, then the strategy's own entry rule."""
        key = (strat.name, i)
        hit = self._entry.get(key)
        if hit is not None:
            return hit
        ok = False
        if self.crypto or self.mins_to_close[i] > settings.NO_NEW_ENTRY_MINUTES_BEFORE_CLOSE:
            ctx, tr = self.ctx(i)
            if tr.ready and not tr.reversal_down and tr.direction >= settings.TREND_ENTRY_MIN:
                try:
                    ok = bool(strat.evaluate_entry(ctx).should_enter)
                except Exception:
                    ok = False
        self._entry[key] = ok
        return ok


class Book:
    """One open position's fills and bookkeeping."""

    def __init__(self, tape: Tape, strat_name: str, i: int, price: float, notional: float):
        self.t = tape
        self.trade = Trade(tape.symbol, strat_name, int(tape.minute[i]))
        ctx, _ = tape.ctx(max(i - 1, 0))
        fill = price * (1 + tape.hs)
        self.qty = round_qty(tape.symbol, notional / fill, fill)
        stop, tp, _ = brackets.derive(fill, ctx.quant.atr)
        self.pos: Dict[str, Any] = {"symbol": tape.symbol, "qty": self.qty, "avg_entry_price": fill,
                                    "stop_loss": stop, "take_profit": tp, "opened_at": tape.minute[i] * 60,
                                    "entry_strategy": strat_name}
        self.highest = fill
        self.last_harvest_bid = 0.0
        self.cash = 0.0
        self.trade.entry_price = fill
        self._buy(self.qty, price)

    @property
    def entry(self) -> float:
        return float(self.pos["avg_entry_price"])

    @property
    def stop(self) -> float:
        return float(self.pos["stop_loss"])

    def _buy(self, q: float, mid: float):
        fill = mid * (1 + self.t.hs)
        fee = q * fill * self.t.fee
        self.cash -= q * fill + fee
        self.trade.costs += q * mid * self.t.hs + fee

    def _sell(self, q: float, mid: float, at_limit: bool = False) -> float:
        """Sells q; returns the net booked against the average entry."""
        fill = mid if at_limit else mid * (1 - self.t.hs)
        fee = q * fill * self.t.fee
        self.cash += q * fill - fee
        self.trade.costs += (0.0 if at_limit else q * mid * self.t.hs) + fee
        return (fill - self.entry) * q - fee

    def add(self, q: float, mid: float):
        fill = mid * (1 + self.t.hs)
        self._buy(q, mid)
        new_qty = self.qty + q
        self.pos["avg_entry_price"] = (self.entry * self.qty + fill * q) / new_qty
        self.qty = self.pos["qty"] = new_qty

    def reduce(self, q: float, mid: float, harvest: bool = False):
        booked = self._sell(q, mid)
        if harvest:
            self.trade.harvested += booked
            self.trade.harvests += 1
        self.qty = self.pos["qty"] = round(self.qty - q, 8)

    def close(self, i: int, mid: float, reason: str, at_limit: bool = False) -> Trade:
        self._sell(self.qty, mid, at_limit)
        self.qty = 0.0
        self.trade.exit_minute = int(self.t.minute[i])
        self.trade.pnl = round(self.cash, 4)
        self.trade.exit_reason = reason
        self.trade.recovered = bool(self.pos.get("recovered"))
        return self.trade


def _soft_reversal(tape: Tape, i: int, b: Book, ctx, tr) -> bool:
    """The sentinel's reversal read (engine/sentinel_agent.py) without news, at bar close."""
    q = ctx.quant
    price = float(tape.c[i])
    prev = float(tape.c[i - 1]) if i > 0 else price
    tick_delta = (price - prev) / prev if prev > 0 else 0.0
    dd = (b.highest - price) / b.highest if b.highest > 0 else 0.0
    trend_score = 0.5
    if q.ema_fast and q.ema_slow:
        trend_score = 0.80 if (q.ema_fast > q.ema_slow and price > q.ema_fast) else 0.20
    reversal = (0.25 * min(dd * 18.0, 0.45) + 0.15 * (1.0 - trend_score)
                + min(max(-tick_delta * 40.0, 0.0), 0.15)
                + (0.15 if q.rsi and q.rsi > 70 else 0.0))
    if reversal < settings.MAX_SELL_SENTIMENT_NEG:
        return False
    if _in_min_hold(tape, i, b):
        return False
    pnl = (price - b.entry) / b.entry
    if pnl >= 0 or not settings.RECOVERY_ENABLED:
        return True
    return tr.ready and (tr.reversal_down or tr.direction <= -settings.TREND_TRIM_DIRECTION)


def _in_min_hold(tape: Tape, i: int, b: Book) -> bool:
    if tape.crypto:
        return False
    return (tape.minute[i] * 60 - b.pos["opened_at"]) / 60 < settings.STOCK_SCORE_MIN_HOLD_MINUTES


def _bar(tape: Tape, i: int, b: Book, strat) -> Optional[Trade]:
    """Runs one bar against an open position. Returns the Trade if it closed."""
    o, h, l, c = (float(x[i]) for x in (tape.o, tape.h, tape.l, tape.c))
    hs = tape.hs

    # End of day: stocks are flat before the close.
    if not tape.crypto and settings.DAY_TRADE_FLATTEN_ENABLED and (
            tape.mins_to_close[i] <= settings.FLATTEN_MINUTES_BEFORE_CLOSE or tape.day_last[i]):
        return b.close(i, c, "end of day")

    # Stop. A loss stop under confirmation has a disaster floor that fires at once.
    stop, entry = b.stop, b.entry
    if settings.STOP_CONFIRM_ENABLED and stop < entry:
        risk_ps = loss_recovery._risk_per_share(b.pos, entry, stop)
        floor = loss_recovery.disaster_stop(stop + risk_ps, stop) if risk_ps > 0 else stop
        if o <= floor:
            return b.close(i, o, "stop (gap through disaster)")
        if l <= floor:
            return b.close(i, floor, "stop (disaster)")
        if c <= stop:
            return b.close(i, c, "stop (confirmed)")
    else:
        if o <= stop:
            return b.close(i, o, "stop (gap)")
        if l <= stop:
            return b.close(i, stop, "stop")

    # Profit harvest at the first price where the bid beats the reference: the
    # entry (plus both crypto fees, as engine/profit_harvest.py requires), the
    # last harvest, or the USD minimum. Stocks sell fractional shares.
    if settings.PROFIT_HARVEST_ENABLED and not b.pos.get("harvest_unsplittable"):
        fee = settings.CRYPTO_TAKER_FEE_BPS / 1e4 if tape.crypto else 0.0
        ref = max(entry * (1 + fee) / (1 - fee), b.last_harvest_bid)
        if settings.PROFIT_HARVEST_USD > 0:
            ref = max(ref, entry + settings.PROFIT_HARVEST_USD / b.qty)
        tick = 0.01 if not tape.crypto else ref * 1e-4
        if h * (1 - hs) > ref:
            bid = min(max(o * (1 - hs), ref + tick), h * (1 - hs))
            frac = min(max(settings.PROFIT_HARVEST_FRACTION, 0.0), 1.0)
            sell = profit_harvest.harvest_qty(tape.symbol, b.qty * frac, bid, simulated=True)
            if sell <= 0 or b.qty - sell <= 1e-9:
                b.pos["harvest_unsplittable"] = True
            elif profit_harvest.net_unit_profit(tape.symbol, bid, entry) * sell + 1e-9 \
                    < settings.PROFIT_HARVEST_MIN_INCOME:
                pass                     # would bank under a cent: no sale
            else:
                b.reduce(sell, bid / (1 - hs), harvest=True)
                b.last_harvest_bid = bid

    # Target: a limit at the take-profit (the bracket leg, or the sentinel for crypto).
    tp = float(b.pos["take_profit"])
    if h >= tp:
        return b.close(i, max(o, tp), "target", at_limit=not tape.crypto)

    # From here on, decisions made at the bar's close.
    ctx, tr = tape.ctx(i)
    if h > b.highest:
        b.highest = h
    new = brackets.trail(b.pos, b.highest, b.entry)
    if new is not None and new > b.stop:
        b.pos["stop_loss"] = new
    lock = loss_recovery.breakeven_lock(b.pos, c, b.entry, b.stop)
    if lock is not None and lock > b.stop:
        b.pos["stop_loss"] = lock

    pctx = StrategyContext(symbol=ctx.symbol, price=c, quant=ctx.quant, sentiment=ctx.sentiment,
                           consensus=None, is_crypto=ctx.is_crypto, position=b.pos,
                           highest_price=b.highest, series=ctx.series)
    pctx._min_buy_prob = ctx._min_buy_prob
    try:
        ex = strat.evaluate_exit(pctx)
    except Exception:
        ex = None
    if ex is not None and ex.should_close:
        return b.close(i, c, f"strategy: {ex.reason[:60]}")
    if _soft_reversal(tape, i, b, ctx, tr):
        return b.close(i, c, "reversal read")

    # Position manager (engine/fleet.py): trend CLOSE, and one trim of a winner.
    pnl = (c - b.entry) / b.entry
    if tr.ready:
        if tr.direction <= -settings.TREND_EXIT_DIRECTION and tr.confidence >= 0.5 \
                and not _in_min_hold(tape, i, b):
            return b.close(i, c, "trend turned down")
        if pnl > 0 and not b.pos.get("trimmed") and (
                tr.reversal_down or tr.direction <= -settings.TREND_TRIM_DIRECTION):
            sell = round_qty(tape.symbol, b.qty * settings.TRIM_FRACTION, c)
            if sell > 0 and round_qty(tape.symbol, b.qty - sell, c) > 0:
                b.reduce(sell, c)
            b.pos["trimmed"] = True

    # Loss recovery: one rescue add once the fall has stalled.
    d = loss_recovery.check_rescue(tape.symbol, b.pos, c, b.qty, b.entry, b.stop,
                                   news_bearish=False, strategy_exiting=False,
                                   now=float(tape.minute[i] * 60), trend=tr)
    if d is not None:
        add = round_qty(tape.symbol, d.rescue_qty, c)
        if add > 0:
            b.add(add, c)
            b.pos["rescued"] = True
            b.trade.rescued = True
    return None


def run(tape: Tape, strat, notional: float) -> List[Trade]:
    """Replays one strategy on one symbol under the current settings."""
    state.is_trading_active = True
    trades: List[Trade] = []
    book: Optional[Book] = None
    pending = False
    for i in range(WARMUP, tape.n):
        if pending:
            pending = False
            fresh_day = (not tape.crypto) and tape.day_last[i - 1]
            if not fresh_day:
                book = Book(tape, strat.name, i, float(tape.o[i]), notional)
                if book.qty <= 0:
                    book = None
        if book is not None:
            t = _bar(tape, i, book, strat)
            if t is not None:
                trades.append(t)
                book = None
                continue
        if book is None and i < tape.n - 1 and tape.entry_signal(strat, i):
            pending = True
    if book is not None:
        trades.append(book.close(tape.n - 1, float(tape.c[-1]), "end of data"))
    return trades
