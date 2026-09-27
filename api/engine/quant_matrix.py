import numpy as np
import time
from typing import Optional, Dict, Any
from core.state import state, QuantMetrics, price_decimals

class QuantMatrix:
    """
    Sub-millisecond Quantitative Indicator Matrix (t1...tN).
    Calculates technical indicators from rolling in-memory deques.
    Each test is evaluated in < 0.05 milliseconds.
    """

    @staticmethod
    def calculate_ema(prices: np.ndarray, period: int) -> Optional[float]:
        """Calculates Exponential Moving Average"""
        if len(prices) < period:
            return None
        alpha = 2.0 / (period + 1.0)
        ema = prices[0]
        for price in prices[1:]:
            ema = alpha * price + (1.0 - alpha) * ema
        return float(ema)

    @staticmethod
    def calculate_rsi(prices: np.ndarray, period: int = 14) -> Optional[float]:
        """Calculates Relative Strength Index in < 0.02ms"""
        if len(prices) <= period:
            return 50.0  # Neutral baseline until buffer matures
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

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

    def evaluate_symbol(self, symbol: str) -> QuantMetrics:
        """
        Runs the full (t1...tN) evaluation grid for a symbol.
        Returns QuantMetrics stored directly in RAM.
        """
        history = state.get_or_create_history(symbol)
        prices = np.array(history, dtype=np.float64)
        tick = state.latest_prices.get(symbol)

        if len(prices) < 5 or tick is None:
            # Insufficient data yet, return neutral baseline
            metrics = QuantMetrics(
                symbol=symbol,
                rsi=50.0,
                ema_fast=tick.price if tick else 100.0,
                ema_slow=tick.price if tick else 100.0,
                atr=max((tick.price if tick else 100.0) * 0.005, 1e-9),
                spread=0.01,
                volume_ratio=1.0
            )
            state.quant_metrics[symbol] = metrics
            return metrics

        # t1: EMA 9 & EMA 21
        ema_fast = self.calculate_ema(prices, period=min(9, len(prices)))
        ema_slow = self.calculate_ema(prices, period=min(21, len(prices)))

        # t2: RSI 14
        rsi = self.calculate_rsi(prices, period=min(14, len(prices) - 1)) or 50.0

        # t3: ATR 14
        atr = self.calculate_atr(prices, period=min(14, len(prices) - 1), ref_price=tick.price)

        # t4: Bid-Ask Spread check
        spread = 0.0
        if tick.ask > 0 and tick.bid > 0:
            spread = (tick.ask - tick.bid) / tick.price

        # t5: Volume ratio
        vol_hist = state.volume_history.get(symbol)
        vol_ratio = 1.0
        if vol_hist and len(vol_hist) > 5:
            vols = np.array(vol_hist, dtype=np.float64)
            avg_vol = np.mean(vols[-20:]) if len(vols) >= 20 else np.mean(vols)
            if avg_vol > 0:
                vol_ratio = float(vols[-1] / avg_vol)

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

quant_matrix = QuantMatrix()
