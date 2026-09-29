import numpy as np
import time
from feeds.spreads import spreads
from collections import deque
from itertools import islice
from typing import Optional, Dict, Any
from core.state import state, QuantMetrics, price_decimals
from engine.strategies.indicators import iir as _iir

class QuantMatrix:
    """
    Sub-millisecond Quantitative Indicator Matrix (t1...tN).
    Calculates technical indicators from rolling in-memory deques.
    Each test is evaluated in < 0.05 milliseconds.
    """

    @staticmethod
    def calculate_ema(prices: np.ndarray, period: int) -> Optional[float]:
        """
        Exponential Moving Average seeded with the first price.

        Vectorised: the recursion ema_t = a*p_t + (1-a)*ema_{t-1} runs as a C-level
        IIR filter instead of a Python loop -- same numbers, ~20x less time on the
        tick path.
        """
        if len(prices) < period:
            return None
        alpha = 2.0 / (period + 1.0)
        if _iir is not None:
            return float(_iir(alpha, prices, (1.0 - alpha) * prices[0])[-1])
        ema = prices[0]
        for price in prices[1:]:
            ema = alpha * price + (1.0 - alpha) * ema
        return float(ema)

    @staticmethod
    def calculate_rsi(prices: np.ndarray, period: int = 14) -> Optional[float]:
        """
        Wilder RSI. The smoothing recursion has a closed form, so the final
        averages are two dot products rather than a Python loop.
        """
        if len(prices) <= period:
            return 50.0  # Neutral baseline until buffer matures
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        # avg_T = avg_seed*k^m + sum_i x_i/period * k^(T-i), k = 1 - 1/period
        k = 1.0 - 1.0 / period
        m = len(deltas) - period
        w = k ** np.arange(m - 1, -1, -1) / period if m > 0 else np.empty(0)
        decay = k ** m
        avg_gain = gains[:period].mean() * decay + float(gains[period:] @ w)
        avg_loss = losses[:period].mean() * decay + float(losses[period:] @ w)

        if avg_loss == 0:
            return 60.0 if avg_gain > 0 else 50.0
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return float(rsi)

    @staticmethod
    def calculate_atr(prices: np.ndarray, period: int = 14, ref_price: Optional[float] = None) -> float:
        """
        Proxy Average True Range from high-frequency ticks, in absolute price units.

        The floor MUST be proportional to price, not an absolute cent amount. A flat
        $0.05 floor is ~0.06% of BTC but 16% of SHIB, which previously drove stop
        distances of -23% on XLM and a negative (unreachable) stop on SHIB.
        """
        ref = float(ref_price) if ref_price else (float(prices[-1]) if len(prices) else 1.0)
        # 0.05% of price: small enough never to dominate a real reading, large enough
        # to keep a flat-quote asset from producing a zero-width stop.
        floor = max(ref * 0.0005, 1e-9)
        if len(prices) < 2:
            return max(ref * 0.005, floor)
        diffs = np.abs(np.diff(prices))
        recent_diffs = diffs[-period:] if len(diffs) >= period else diffs
        atr = float(np.mean(recent_diffs))
        return max(atr, floor)

    @staticmethod
    def true_range_atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14) -> Optional[float]:
        """
        Wilder ATR over one-minute bars from the true range (high, low and the
        previous close), in price units. None with fewer than period + 1 bars.

        The older calculate_atr averages close-to-close moves of whatever samples
        are in the buffer; on sub-second polls that was a few cents and pinned
        every stop to its minimum. Bars carry the real intrabar range.
        """
        h, l, c = (np.asarray(x, dtype=np.float64) for x in (h, l, c))
        n = len(c)
        if n < period + 1:
            return None
        prev = c[:-1]
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - prev), np.abs(l[1:] - prev)))
        atr = tr[:period].mean()
        for x in tr[period:]:
            atr = (atr * (period - 1) + x) / period
        return float(atr)

    # Once the buffer holds this many samples every period is fixed (EMA 21,
    # RSI 14), so from here on each tick can update state instead of recomputing.
    INCREMENTAL_FROM = 30

    def __init__(self):
        # symbol -> _Inc; the running indicator state for O(1) per-tick updates
        self._inc: Dict[str, "_Inc"] = {}

    @staticmethod
    def _wilder_state(prices: np.ndarray, period: int):
        """Final (avg_gain, avg_loss) of the Wilder recursion, as calculate_rsi uses."""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        k = 1.0 - 1.0 / period
        m = len(deltas) - period
        w = k ** np.arange(m - 1, -1, -1) / period if m > 0 else np.empty(0)
        decay = k ** m
        return (gains[:period].mean() * decay + float(gains[period:] @ w),
                losses[:period].mean() * decay + float(losses[period:] @ w))

    @staticmethod
    def _rsi_from(avg_gain: float, avg_loss: float) -> float:
        if avg_loss == 0:
            return 60.0 if avg_gain > 0 else 50.0
        return float(100.0 - 100.0 / (1.0 + avg_gain / avg_loss))

    def _indicators(self, symbol: str, prices_deque, tick_price: float):
        """
        (ema_fast, ema_slow, rsi, atr) for the latest tick.

        Full recompute while warming up, or whenever this symbol's tick count
        shows samples we did not see one by one (seeding, a missed evaluation).
        Otherwise an O(1) update, so the tick path no longer gets slower as the
        buffer fills -- measured 58 -> 78us from 20 to 250 samples before.

        Equivalence: while the buffer is filling the update is the same
        recursion from the same seed, so results match exactly. Once full, the
        old full recompute re-seeded from the oldest sample on every tick; the
        seed's weight after 249 steps is ~5e-11 (EMA 21) and ~3e-8 (RSI 14),
        which is far below the 2-decimal rounding of the published values.
        """
        n = len(prices_deque)
        count = state.tick_count.get(symbol, 0)
        inc = self._inc.get(symbol)
        if inc is not None and n >= self.INCREMENTAL_FROM and count == inc.count + 1:
            x = prices_deque[-1]
            a_f, a_s = 2.0 / 10.0, 2.0 / 22.0
            inc.ema_f = a_f * x + (1.0 - a_f) * inc.ema_f
            inc.ema_s = a_s * x + (1.0 - a_s) * inc.ema_s
            d = x - inc.last
            inc.g = (inc.g * 13.0 + (d if d > 0 else 0.0)) / 14.0
            inc.l = (inc.l * 13.0 + (-d if d < 0 else 0.0)) / 14.0
            inc.trs.append(abs(d))
            inc.last = x
            inc.count = count
            atr = max(sum(inc.trs) / len(inc.trs), max(tick_price * 0.0005, 1e-9))
            return inc.ema_f, inc.ema_s, self._rsi_from(inc.g, inc.l), atr

        prices = state.history_array(symbol)
        ema_fast = self.calculate_ema(prices, period=min(9, n))
        ema_slow = self.calculate_ema(prices, period=min(21, n))
        rsi_period = min(14, n - 1)
        rsi = self.calculate_rsi(prices, period=rsi_period) or 50.0
        atr = self.calculate_atr(prices, period=rsi_period, ref_price=tick_price)
        if n >= self.INCREMENTAL_FROM:
            g, l = self._wilder_state(prices, 14)
            trs = deque(np.abs(np.diff(prices[-15:])), maxlen=14)
            self._inc[symbol] = _Inc(count=count, ema_f=ema_fast, ema_s=ema_slow,
                                     g=g, l=l, last=float(prices[-1]), trs=trs)
        else:
            self._inc.pop(symbol, None)
        return ema_fast, ema_slow, rsi, atr

    def evaluate_symbol(self, symbol: str) -> QuantMetrics:
        """
        Runs the full (t1...tN) evaluation grid for a symbol.
        Returns QuantMetrics stored directly in RAM.
        """
        history = state.get_or_create_history(symbol)
        tick = state.latest_prices.get(symbol)

        if len(history) < 5 or tick is None:
            # Insufficient data yet, return neutral baseline
            metrics = QuantMetrics(
                symbol=symbol,
                rsi=50.0,
                ema_fast=tick.price if tick else 100.0,
                ema_slow=tick.price if tick else 100.0,
                atr=max((tick.price if tick else 100.0) * 0.005, 1e-9),
                # Not a placeholder 1%: that read as a real (too wide) spread and
                # blocked every stock for its first minutes on the watchlist.
                spread=spreads.estimate(symbol)[0] or 0.0,
                volume_ratio=1.0
            )
            state.quant_metrics[symbol] = metrics
            return metrics

        # t1-t3: EMA 9/21, RSI 14, ATR 14
        ema_fast, ema_slow, rsi, atr = self._indicators(symbol, history, tick.price)
        # ATR from the true range of closed one-minute bars when there are enough;
        # the backtester computes it the same way from its bar tape.
        from core.minute_bars import minute_bars
        rows = minute_bars.closed_rows(symbol)[-60:]
        if len(rows) >= 15:
            a = np.asarray(rows, dtype=np.float64)
            bar_atr = self.true_range_atr(a[:, 2], a[:, 3], a[:, 4])
            if bar_atr:
                atr = max(bar_atr, tick.price * 0.0005)

        # t4: Bid-Ask Spread: the real (consolidated) spread when known, else a
        # short-window median of IEX quotes -- never one IEX snapshot, which for
        # thinly-held names is many times the real spread (feeds/spreads.py).
        # Unknown reads as 0: the scout only watches liquid stocks.
        spread = spreads.estimate(symbol)[0] or 0.0

        # t5: Volume ratio over the last 20 volume samples (read from the deque's
        # tail only; converting the whole buffer made this O(n) too)
        vol_hist = state.volume_history.get(symbol)
        vol_ratio = 1.0
        if vol_hist and len(vol_hist) > 5:
            recent = list(islice(reversed(vol_hist), 20))
            avg_vol = sum(recent) / len(recent)
            if avg_vol > 0:
                vol_ratio = float(recent[0] / avg_vol)

        metrics = QuantMetrics(
            symbol=symbol,
            rsi=round(rsi, 2),
            ema_fast=round(ema_fast, price_decimals(tick.price)) if ema_fast else None,
            ema_slow=round(ema_slow, price_decimals(tick.price)) if ema_slow else None,
            atr=round(atr, price_decimals(tick.price) + 2),
            spread=round(spread, 5),
            volume_ratio=round(vol_ratio, 2),
            updated_at=time.time()
        )
        state.quant_metrics[symbol] = metrics
        return metrics


class _Inc:
    """Running indicator state for one symbol."""
    __slots__ = ("count", "ema_f", "ema_s", "g", "l", "last", "trs")

    def __init__(self, count, ema_f, ema_s, g, l, last, trs):
        self.count, self.ema_f, self.ema_s = count, ema_f, ema_s
        self.g, self.l, self.last, self.trs = g, l, last, trs


quant_matrix = QuantMatrix()
