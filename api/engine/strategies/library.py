"""
Quant strategy library.

Ten well-known, price-driven strategies of the kind catalogued by awesome-quant
(https://github.com/wilsonfreitas/awesome-quant) and shipped as examples in its
backtesting frameworks (zipline, backtrader, backtesting.py, freqtrade, jesse).

They are re-implemented here from the published methods rather than copied:
backtrader and freqtrade are GPL-3, and pasting their code would put this
project under GPL. It also lets every strategy fit this engine's contract --
long-only, close-only tick buffer, explainable gates, and "abstain with a
reason" when inputs are missing.

Each strategy declares the regimes it is built for, so the adaptive selector
(engine/strategies/adaptive.py) can pick the right one for the moment, and each
exposes bias() so the quant council (engine/strategies/council.py) can poll all
of them for the manager and the trade bots.

Lookbacks are in samples of the engine's price buffer (up to 250), not days.
"""
from typing import Optional, Tuple

import numpy as np

from engine.strategies.base import (
    Strategy, StrategyContext, EntryDecision, ExitDecision,
)
from engine.strategies.indicators import PriceSeries


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(min(max(x, lo), hi))


def _crossed_up(a: np.ndarray, b: np.ndarray, within: int) -> bool:
    """True if series a crossed above b within the last `within` samples and is still above."""
    if len(a) < within + 1:
        return False
    d = a[-(within + 1):] - b[-(within + 1):]
    if np.isnan(d).any() or d[-1] <= 0:
        return False
    return bool((d[:-1] <= 0).any())


# Entry signal: (fires, buy_prob, reason, blocked_by)
Signal = Tuple[bool, float, str, Optional[str]]
# Exit signal: (close, sell_prob, reason, close_fraction)
ExitSignal = Tuple[bool, float, str, float]


class QuantStrategy(Strategy):
    """
    Template for library strategies.

    Subclasses implement _entry, _exit and bias. The template owns the parts
    every strategy must get identically right: the history requirement, the
    consensus tilt, the spread gate, the bearish-news veto and the risk dial.
    """
    requires_sentiment = False
    council_member = True
    min_history: int = 40

    # Shared gates; subclasses can override individual keys via their params.
    common_params = {
        "max_spread_pct": 0.005,
        "veto_neg_prob": 0.65,
    }

    def _p(self, key):
        return self.params.get(key, self.common_params.get(key))

    def _has_history(self, ctx: StrategyContext) -> bool:
        return ctx.series is not None and len(ctx.series) >= self.min_history

    def _missing(self, ctx: StrategyContext) -> str:
        n = 0 if ctx.series is None else len(ctx.series)
        return f"Need {self.min_history} price samples, have {n}"

    # ---- subclass hooks ----
    def _entry(self, ctx: StrategyContext, s: PriceSeries, gates: dict) -> Signal:
        raise NotImplementedError

    def _exit(self, ctx: StrategyContext, s: PriceSeries) -> ExitSignal:
        raise NotImplementedError

    def _bias(self, ctx: StrategyContext, s: PriceSeries) -> float:
        raise NotImplementedError

    # ---- template ----
    def evaluate_entry(self, ctx: StrategyContext) -> EntryDecision:
        gates = {"samples": 0 if ctx.series is None else len(ctx.series)}
        if not self._has_history(ctx):
            return EntryDecision(False, 0.0, self._missing(ctx),
                                 blocked_by="no_data", gates=gates)

        fires, buy_prob, reason, blocked = self._entry(ctx, ctx.series, gates)

        if ctx.consensus is not None:
            buy_prob += (ctx.consensus - 0.5) * 0.20
        gates["consensus"] = ctx.consensus
        buy_prob = round(_clip(buy_prob), 4)

        if not fires:
            return EntryDecision(False, buy_prob, reason, blocked_by=blocked, gates=gates)

        q, sent = ctx.quant, ctx.sentiment
        spread = q.spread if q else 0.0
        gates["spread_pct"] = round(spread * 100, 4)
        if spread > self._p("max_spread_pct"):
            return EntryDecision(False, buy_prob,
                f"Spread {spread*100:.3f}% exceeds {self._p('max_spread_pct')*100:.2f}%",
                blocked_by="spread", gates=gates)

        gates["sentiment_neg"] = sent.neg_prob
        gates["sentiment_n"] = sent.n_headlines
        if sent.is_tradeable and sent.neg_prob >= self._p("veto_neg_prob"):
            return EntryDecision(False, buy_prob,
                f"Vetoed by bearish news: neg={sent.neg_prob:.2f} across {sent.n_headlines} headlines",
                blocked_by="sentiment_veto", gates=gates)

        profile_min = ctx.__dict__.get("_min_buy_prob")
        if profile_min is not None and buy_prob < profile_min:
            return EntryDecision(False, buy_prob,
                f"buy_prob {buy_prob:.2f} below risk-dial entry bar {profile_min:.2f}",
                blocked_by="risk_dial", gates=gates)

        return EntryDecision(True, buy_prob, reason, gates=gates)

    def evaluate_exit(self, ctx: StrategyContext) -> ExitDecision:
        pos = ctx.position or {}
        entry = float(pos.get("avg_entry_price") or ctx.price)
        pnl_pct = (ctx.price - entry) / entry if entry > 0 else 0.0
        if not self._has_history(ctx):
            return ExitDecision(False, 0.3, 0.2,
                f"Holding: insufficient history for {self.name} exit read, PnL {pnl_pct*100:+.2f}%")

        close, sell_prob, reason, fraction = self._exit(ctx, ctx.series)
        sent = ctx.sentiment
        if sent.is_tradeable:
            sell_prob += 0.25 * sent.neg_prob
        sell_prob = round(_clip(sell_prob), 4)
        reason = f"{reason}, PnL {pnl_pct*100:+.2f}%"
        if close:
            return ExitDecision(True, sell_prob, 1.0, reason, close_fraction=fraction)
        return ExitDecision(False, sell_prob, round(sell_prob * 0.8, 4), reason)

    def bias(self, ctx: StrategyContext) -> float:
        if not self._has_history(ctx):
            return 0.0
        return round(_clip(self._bias(ctx, ctx.series), -1.0, 1.0), 4)

    def describe(self):
        d = super().describe()
        d["min_history"] = self.min_history
        return d


# =============================================================================
# Mean reversion family (ranging markets)
# =============================================================================

class BollingerReversionStrategy(QuantStrategy):
    name = "bollinger_reversion"
    display_name = "Bollinger Band Reversion"
    description = ("Buys a close back inside the lower Bollinger Band after a stretch "
                   "below it (the 'hook'), targets the middle band.")
    source = "John Bollinger, Bollinger on Bollinger Bands (2001); backtrader/freqtrade sample strategy"
    regimes = ("ranging", "mixed")
    min_history = 40
    params = {"period": 20, "k": 2.0, "max_pct_b": 0.15, "exit_pct_b": 0.5, "fail_pct_b": -0.5}

    def _pct_b(self, s: PriceSeries, i: int = -1) -> Optional[float]:
        mid, up, lo = s.bollinger(self.params["period"], self.params["k"])
        w = up[i] - lo[i]
        if np.isnan(w) or w <= 0:
            return None
        return float((s.close[i] - lo[i]) / w)

    def _entry(self, ctx, s, gates):
        b, prev = self._pct_b(s), self._pct_b(s, -2)
        gates.update(pct_b=b, prev_pct_b=prev)
        if b is None or prev is None:
            return False, 0.0, "Bands undefined (flat price)", "no_data"
        buy_prob = 0.5 + (self.params["max_pct_b"] - b) * 0.8
        if b > self.params["max_pct_b"]:
            return False, buy_prob, f"%B {b:.2f} not near lower band (need <= {self.params['max_pct_b']})", "not_stretched"
        if s.close[-1] <= s.close[-2]:
            return False, buy_prob, f"%B {b:.2f} stretched but still falling; waiting for the hook", "no_hook"
        return True, max(buy_prob, 0.6), f"Bollinger hook: %B {prev:.2f} -> {b:.2f}, targeting middle band", None

    def _exit(self, ctx, s):
        b = self._pct_b(s)
        if b is None:
            return False, 0.3, "Bands undefined", 1.0
        sell = _clip(b)
        if b >= self.params["exit_pct_b"]:
            return True, sell, f"Reverted to middle band (%B {b:.2f})", 1.0
        if b <= self.params["fail_pct_b"]:
            return True, 0.8, f"Band walk down (%B {b:.2f}): reversion thesis failed", 1.0
        return False, sell, f"Awaiting reversion, %B {b:.2f}", 1.0

    def _bias(self, ctx, s):
        b = self._pct_b(s)
        return 0.0 if b is None else (0.5 - b) * 1.5


class ConnorsRSI2Strategy(QuantStrategy):
    name = "connors_rsi2"
    display_name = "Connors RSI(2)"
    description = ("Buys extreme 2-period RSI dips only while price is above its long "
                   "moving average; exits on a close above the short average.")
    source = "Larry Connors & Cesar Alvarez, Short Term Trading Strategies That Work (2008)"
    regimes = ("ranging", "trending_up", "mixed")
    min_history = 105
    params = {"rsi_period": 2, "entry_rsi": 10.0, "trend_sma": 100, "exit_sma": 5, "exit_rsi": 70.0}

    def _entry(self, ctx, s, gates):
        r = s.rsi(self.params["rsi_period"])
        trend = s.sma(self.params["trend_sma"])[-1]
        gates.update(rsi2=r, trend_sma=trend)
        if r is None:
            return False, 0.0, "RSI(2) undefined", "no_data"
        buy_prob = 0.5 + (self.params["entry_rsi"] * 2 - r) / 100.0
        if s.last < trend:
            return False, buy_prob, f"Price below SMA{self.params['trend_sma']}: dips in downtrends are not bought", "trend_filter"
        if r > self.params["entry_rsi"]:
            return False, buy_prob, f"RSI(2) {r:.1f} not extreme (need <= {self.params['entry_rsi']})", "not_oversold"
        return True, max(buy_prob, 0.62), f"RSI(2) {r:.1f} washout above SMA{self.params['trend_sma']}", None

    def _exit(self, ctx, s):
        exit_ma = s.sma(self.params["exit_sma"])[-1]
        r = s.rsi(self.params["rsi_period"]) or 50.0
        sell = _clip(r / 100.0)
        if s.last > exit_ma or r >= self.params["exit_rsi"]:
            return True, sell, f"Bounce complete: close above SMA{self.params['exit_sma']} / RSI(2) {r:.0f}", 1.0
        if s.last < s.sma(self.params["trend_sma"])[-1] * 0.98:
            return True, 0.8, f"Broke 2% below SMA{self.params['trend_sma']}: trend filter lost", 1.0
        return False, sell, f"Awaiting bounce, RSI(2) {r:.0f}", 1.0

    def _bias(self, ctx, s):
        r = s.rsi(self.params["rsi_period"])
        if r is None:
            return 0.0
        above = s.last >= s.sma(self.params["trend_sma"])[-1]
        b = (50.0 - r) / 50.0
        return b if above else min(b, 0.0) - 0.3


class ZScoreReversionStrategy(QuantStrategy):
    name = "zscore_reversion"
    display_name = "Z-Score Statistical Reversion"
    description = ("Treats price as an Ornstein-Uhlenbeck process: buys a >=2 sigma "
                   "dislocation below the rolling mean, only when the Hurst exponent "
                   "confirms mean-reverting behaviour.")
    source = "Ernest Chan, Algorithmic Trading (2013), ch. 2; Ornstein-Uhlenbeck mean reversion"
    regimes = ("ranging",)
    min_history = 80
    params = {"lookback": 50, "entry_z": -2.0, "exit_z": 0.0, "stop_z": -3.5, "max_hurst": 0.5}

    def _entry(self, ctx, s, gates):
        z = s.zscore(self.params["lookback"])
        h = s.hurst(20)
        gates.update(zscore=z, hurst=h)
        if z is None:
            return False, 0.0, "Z-score undefined", "no_data"
        buy_prob = 0.5 + (-z - 1.0) * 0.15
        if z > self.params["entry_z"]:
            return False, buy_prob, f"z {z:.2f} not dislocated (need <= {self.params['entry_z']})", "not_stretched"
        if z < self.params["stop_z"]:
            return False, buy_prob, f"z {z:.2f} beyond {self.params['stop_z']}: regime break, not noise", "breakdown"
        if h is not None and h > self.params["max_hurst"]:
            return False, buy_prob, f"Hurst {h:.2f} > {self.params['max_hurst']}: series is trending, not reverting", "not_mean_reverting"
        return True, max(buy_prob, 0.6), f"z-score {z:.2f} dislocation, Hurst {h if h is not None else float('nan'):.2f}", None

    def _exit(self, ctx, s):
        z = s.zscore(self.params["lookback"]) or 0.0
        sell = _clip(0.5 + z * 0.25)
        if z >= self.params["exit_z"]:
            return True, sell, f"Reverted to mean (z {z:.2f})", 1.0
        if z <= self.params["stop_z"]:
            return True, 0.85, f"z {z:.2f} broke {self.params['stop_z']}: dislocation widening", 1.0
        return False, sell, f"Awaiting reversion, z {z:.2f}", 1.0

    def _bias(self, ctx, s):
        z = s.zscore(self.params["lookback"])
        return 0.0 if z is None else -z / 2.5


class VWAPReversionStrategy(QuantStrategy):
    name = "vwap_reversion"
    display_name = "VWAP Reversion"
    description = ("Buys when price is stretched well below its rolling volume-weighted "
                   "average price and turning up; exits back at VWAP. Needs volume data.")
    source = "Institutional VWAP execution benchmark (Berkowitz, Logue & Noser 1988); intraday reversion"
    regimes = ("ranging", "mixed")
    min_history = 60
    params = {"lookback": 50, "entry_dev": -2.0, "exit_dev": 0.0, "stop_dev": -4.0}

    def _dev(self, s: PriceSeries) -> Optional[float]:
        vw, sd = s.vwap(self.params["lookback"])
        if vw is None or not sd:
            return None
        return (s.last - vw) / sd

    def _entry(self, ctx, s, gates):
        d = self._dev(s)
        gates.update(vwap_dev=d, volume_samples=len(s.volume))
        if d is None:
            return False, 0.0, f"Need {self.params['lookback']} volume samples for VWAP, have {len(s.volume)}", "no_volume"
        buy_prob = 0.5 + (-d - 1.0) * 0.15
        if d > self.params["entry_dev"]:
            return False, buy_prob, f"{d:.2f} sd from VWAP, need <= {self.params['entry_dev']}", "not_stretched"
        if d < self.params["stop_dev"]:
            return False, buy_prob, f"{d:.2f} sd below VWAP: capitulation, not a stretch", "breakdown"
        if s.close[-1] <= s.close[-2]:
            return False, buy_prob, f"{d:.2f} sd below VWAP but still falling", "no_hook"
        return True, max(buy_prob, 0.6), f"{d:.2f} sd below VWAP and turning up", None

    def _exit(self, ctx, s):
        d = self._dev(s)
        if d is None:
            return False, 0.3, "VWAP undefined", 1.0
        sell = _clip(0.5 + d * 0.25)
        if d >= self.params["exit_dev"]:
            return True, sell, f"Back at VWAP ({d:+.2f} sd)", 1.0
        if d <= self.params["stop_dev"]:
            return True, 0.85, f"{d:.2f} sd below VWAP: selling pressure persisting", 1.0
        return False, sell, f"Awaiting VWAP reclaim ({d:+.2f} sd)", 1.0

    def _bias(self, ctx, s):
        d = self._dev(s)
        return 0.0 if d is None else -d / 3.0


# =============================================================================
# Trend following family (trending markets)
# =============================================================================

class MACDTrendStrategy(QuantStrategy):
    name = "macd_trend"
    display_name = "MACD Signal Cross"
    description = ("Enters on a fresh MACD bullish signal-line cross with a rising "
                   "histogram, above the zero line; exits on the bearish cross.")
    source = "Gerald Appel (1970s); canonical example in backtrader, backtesting.py and TA-Lib"
    regimes = ("trending_up", "mixed")
    min_history = 50
    params = {"fast": 12, "slow": 26, "signal": 9, "cross_within": 3, "require_above_zero": True}

    def _entry(self, ctx, s, gates):
        line, sig, hist = s.macd(self.params["fast"], self.params["slow"], self.params["signal"])
        atr = s.atr(14) or (s.last * 0.005)
        gates.update(macd=float(line[-1]), signal=float(sig[-1]), hist=float(hist[-1]))
        if np.isnan(hist[-1]):
            return False, 0.0, "MACD not yet defined", "no_data"
        strength = hist[-1] / atr
        buy_prob = 0.55 + _clip(strength, -1, 1) * 0.25
        if not _crossed_up(line, sig, self.params["cross_within"]):
            return False, buy_prob, "No fresh bullish MACD cross", "no_cross"
        if self.params["require_above_zero"] and line[-1] <= 0:
            return False, buy_prob, f"Cross below zero line (MACD {line[-1]:.4g}): counter-trend", "below_zero"
        if hist[-1] <= hist[-2]:
            return False, buy_prob, "Histogram not expanding", "weak_momentum"
        return True, max(buy_prob, 0.6), f"MACD bullish cross, histogram {hist[-1]:.4g} expanding", None

    def _exit(self, ctx, s):
        line, sig, hist = s.macd(self.params["fast"], self.params["slow"], self.params["signal"])
        atr = s.atr(14) or (s.last * 0.005)
        sell = _clip(0.5 - hist[-1] / atr * 0.5)
        if line[-1] < sig[-1]:
            return True, sell, "MACD bearish cross: trend momentum lost", 1.0
        return False, sell, f"MACD above signal (hist {hist[-1]:.4g})", 1.0

    def _bias(self, ctx, s):
        _, _, hist = s.macd(self.params["fast"], self.params["slow"], self.params["signal"])
        atr = s.atr(14) or (s.last * 0.005)
        return 0.0 if np.isnan(hist[-1]) else float(np.tanh(hist[-1] / atr))


class MACrossoverStrategy(QuantStrategy):
    name = "ma_crossover"
    display_name = "Dual Moving Average Crossover"
    description = ("The classic golden cross: enters when the fast SMA crosses above "
                   "the slow SMA with price above both; exits on the death cross.")
    source = "Dual moving average crossover; the reference example in zipline, backtrader and backtesting.py"
    regimes = ("trending_up",)
    min_history = 60
    params = {"fast": 10, "slow": 40, "cross_within": 5}

    def _entry(self, ctx, s, gates):
        f, sl = s.sma(self.params["fast"]), s.sma(self.params["slow"])
        atr = s.atr(14) or (s.last * 0.005)
        gates.update(sma_fast=float(f[-1]), sma_slow=float(sl[-1]))
        gap = (f[-1] - sl[-1]) / atr
        buy_prob = 0.55 + _clip(gap / 3.0, -1, 1) * 0.25
        if not _crossed_up(f, sl, self.params["cross_within"]):
            return False, buy_prob, "No recent golden cross", "no_cross"
        if s.last < f[-1]:
            return False, buy_prob, "Price back below fast SMA after the cross", "failed_cross"
        return True, max(buy_prob, 0.6), f"Golden cross SMA{self.params['fast']}/{self.params['slow']}", None

    def _exit(self, ctx, s):
        f, sl = s.sma(self.params["fast"]), s.sma(self.params["slow"])
        atr = s.atr(14) or (s.last * 0.005)
        sell = _clip(0.5 - (f[-1] - sl[-1]) / atr * 0.2)
        if f[-1] < sl[-1]:
            return True, sell, "Death cross: fast SMA below slow", 1.0
        return False, sell, "Fast SMA above slow", 1.0

    def _bias(self, ctx, s):
        f, sl = s.sma(self.params["fast"])[-1], s.sma(self.params["slow"])[-1]
        atr = s.atr(14) or (s.last * 0.005)
        return float(np.tanh((f - sl) / atr / 2.0))


class DonchianBreakoutStrategy(QuantStrategy):
    name = "donchian_turtle"
    display_name = "Donchian Breakout (Turtle)"
    description = ("Turtle System 1: buys a close above the prior 20-sample high; "
                   "exits on a close below the prior 10-sample low.")
    source = "Richard Dennis & William Eckhardt Turtle Trading rules (1983), per Curtis Faith, Way of the Turtle"
    regimes = ("trending_up", "volatile")
    min_history = 30
    params = {"entry_period": 20, "exit_period": 10, "min_volume_ratio": 0.8}

    def _entry(self, ctx, s, gates):
        hi, lo = s.donchian(self.params["entry_period"])
        vol_ratio = ctx.quant.volume_ratio if ctx.quant else 1.0
        gates.update(channel_high=hi, channel_low=lo, volume_ratio=vol_ratio)
        pos = (s.last - lo) / (hi - lo) if hi and lo is not None and hi > lo else 0.5
        buy_prob = 0.5 + _clip(pos - 0.5, -0.5, 0.5) * 0.4
        if s.last <= hi:
            return False, buy_prob, f"No breakout: {s.last:.6g} <= {self.params['entry_period']}-high {hi:.6g}", "no_breakout"
        if vol_ratio < self.params["min_volume_ratio"]:
            return False, buy_prob, f"Breakout on thin volume ({vol_ratio:.2f}x)", "volume"
        return True, max(buy_prob, 0.65), f"Turtle breakout above {self.params['entry_period']}-sample high {hi:.6g}", None

    def _exit(self, ctx, s):
        hi, lo = s.donchian(self.params["exit_period"])
        pos = (s.last - lo) / (hi - lo) if hi > lo else 0.5
        sell = _clip(1.0 - pos)
        if s.last < lo:
            return True, sell, f"Close below {self.params['exit_period']}-sample low {lo:.6g}", 1.0
        return False, sell, f"Above {self.params['exit_period']}-low {lo:.6g}", 1.0

    def _bias(self, ctx, s):
        hi, lo = s.donchian(self.params["entry_period"])
        if not hi or hi <= lo:
            return 0.0
        return ((s.last - lo) / (hi - lo) - 0.5) * 2.0


class SupertrendStrategy(QuantStrategy):
    name = "supertrend"
    display_name = "Supertrend"
    description = ("ATR trailing-band trend filter: enters when the Supertrend flips "
                   "up, exits when it flips down. The band doubles as a trailing stop.")
    source = "Olivier Seban, Supertrend indicator; widely used in freqtrade and jesse community strategies"
    regimes = ("trending_up", "volatile")
    min_history = 40
    params = {"atr_period": 10, "multiplier": 3.0, "flip_within": 3}

    def _line(self, s: PriceSeries):
        """(band, direction) series; direction +1 up / -1 down."""
        key = ("supertrend", self.params["atr_period"], self.params["multiplier"])
        if key in s._cache:
            return s._cache[key]
        x = s.close
        atr = s.atr_series(self.params["atr_period"])
        m = self.params["multiplier"]
        n = len(x)
        band = np.full(n, np.nan)
        direction = np.zeros(n)
        start = self.params["atr_period"] + 1
        if n <= start:
            s._cache[key] = (band, direction)
            return band, direction
        upper, lower = x[start] + m * atr[start], x[start] - m * atr[start]
        d = 1
        for i in range(start, n):
            up_b, lo_b = x[i] + m * atr[i], x[i] - m * atr[i]
            # Bands only ratchet in the trend's favour.
            lower = max(lo_b, lower) if x[i - 1] > lower else lo_b
            upper = min(up_b, upper) if x[i - 1] < upper else up_b
            if d == 1 and x[i] < lower:
                d = -1
            elif d == -1 and x[i] > upper:
                d = 1
            direction[i] = d
            band[i] = lower if d == 1 else upper
        s._cache[key] = (band, direction)
        return band, direction

    def _entry(self, ctx, s, gates):
        band, d = self._line(s)
        gates.update(supertrend=float(band[-1]), direction=int(d[-1]))
        w = self.params["flip_within"]
        buy_prob = 0.5 + 0.2 * d[-1]
        if d[-1] != 1:
            return False, buy_prob, f"Supertrend down (band {band[-1]:.6g})", "downtrend"
        if not (d[-(w + 1):-1] == -1).any():
            return False, buy_prob, f"Supertrend already up for > {w} samples; waiting for a fresh flip", "stale_signal"
        return True, max(buy_prob, 0.65), f"Supertrend flipped up, trailing band {band[-1]:.6g}", None

    def _exit(self, ctx, s):
        band, d = self._line(s)
        if d[-1] == -1:
            return True, 0.8, f"Supertrend flipped down (band {band[-1]:.6g})", 1.0
        dist = (s.last - band[-1]) / s.last if s.last else 0.0
        return False, _clip(0.5 - dist * 20), f"Supertrend up, band {band[-1]:.6g}", 1.0

    def _bias(self, ctx, s):
        band, d = self._line(s)
        atr = s.atr(self.params["atr_period"]) or (s.last * 0.005)
        if np.isnan(band[-1]):
            return 0.0
        return float(d[-1] * min(0.4 + abs(s.last - band[-1]) / atr * 0.15, 1.0))


class TimeSeriesMomentumStrategy(QuantStrategy):
    name = "ts_momentum"
    display_name = "Time-Series Momentum"
    description = ("Volatility-scaled trailing return: enters when the lookback return "
                   "is at least one volatility unit positive and the path is efficient; "
                   "exits when it turns negative.")
    source = "Moskowitz, Ooi & Pedersen, Time Series Momentum, JFE (2012)"
    regimes = ("trending_up",)
    min_history = 70
    params = {"lookback": 60, "entry_t": 1.0, "exit_t": 0.0, "min_efficiency": 0.25}

    def _t(self, s: PriceSeries) -> Optional[float]:
        L = self.params["lookback"]
        r = s.ret(L)
        v = s.realized_vol(L)
        if r is None or not v:
            return None
        return r / (v * np.sqrt(L))

    def _entry(self, ctx, s, gates):
        t = self._t(s)
        er = s.efficiency_ratio(self.params["lookback"])
        gates.update(tsmom_t=t, efficiency_ratio=er)
        if t is None:
            return False, 0.0, "Momentum undefined", "no_data"
        buy_prob = 0.5 + float(np.tanh(t / 2.0)) * 0.3
        if t < self.params["entry_t"]:
            return False, buy_prob, f"Vol-scaled momentum {t:.2f} < {self.params['entry_t']}", "weak_momentum"
        if er is not None and er < self.params["min_efficiency"]:
            return False, buy_prob, f"Efficiency {er:.2f}: return came from noise, not trend", "choppy"
        return True, max(buy_prob, 0.62), f"TSMOM {t:.2f} vol-units over {self.params['lookback']} samples, ER {er:.2f}", None

    def _exit(self, ctx, s):
        t = self._t(s) or 0.0
        sell = _clip(0.5 - float(np.tanh(t / 2.0)) * 0.5)
        if t <= self.params["exit_t"]:
            return True, sell, f"Momentum turned negative ({t:.2f})", 1.0
        return False, sell, f"Momentum {t:.2f} intact", 1.0

    def _bias(self, ctx, s):
        t = self._t(s)
        return 0.0 if t is None else float(np.tanh(t / 2.0))


# =============================================================================
# Volatility family
# =============================================================================

class VolatilitySqueezeStrategy(QuantStrategy):
    name = "volatility_squeeze"
    display_name = "Volatility Squeeze Breakout"
    description = ("Bollinger's 'Squeeze': waits for band width to compress into the "
                   "lowest fifth of its recent range, then buys the upside release. "
                   "Exits when price loses the 20-sample EMA.")
    source = "John Bollinger, Bollinger on Bollinger Bands (2001), 'The Squeeze'; John Carter, TTM Squeeze"
    regimes = ("volatile", "mixed", "ranging")
    min_history = 130
    params = {"period": 20, "bb_k": 2.0, "bandwidth_lookback": 100, "squeeze_quantile": 0.2,
              "squeeze_lookback": 10, "min_squeeze_samples": 5}

    def _squeeze_on(self, s: PriceSeries) -> np.ndarray:
        """
        Squeeze flags for the last squeeze_lookback+1 samples.

        The TTM form (Bollinger inside Keltner) needs a true-range ATR; with this
        engine's close-only data the close-to-close ATR is too narrow and the
        bands almost never fit inside, so the strategy never fired. Bollinger's
        own definition -- bandwidth at a relative low -- needs no ATR at all.
        """
        mid, up, lo = s.bollinger(self.params["period"], self.params["bb_k"])
        with np.errstate(invalid="ignore", divide="ignore"):
            bw = (up - lo) / mid
        hist = bw[-self.params["bandwidth_lookback"]:]
        hist = hist[np.isfinite(hist)]
        n = self.params["squeeze_lookback"] + 1
        if len(hist) < 20:
            return np.zeros(n, dtype=bool)
        threshold = np.quantile(hist, self.params["squeeze_quantile"])
        return bw[-n:] <= threshold

    def _entry(self, ctx, s, gates):
        sq = self._squeeze_on(s)
        L = self.params["squeeze_lookback"]
        recent = int(sq[-(L + 1):-1].sum())
        mom = s.last - s.sma(self.params["period"])[-1]
        gates.update(squeeze_now=bool(sq[-1]), squeeze_samples=recent, momentum=float(mom))
        buy_prob = 0.5 + (0.15 if mom > 0 else -0.15)
        if sq[-1]:
            return False, buy_prob, f"Squeeze still on ({recent}/{L} samples); waiting for release", "squeeze_on"
        if recent < self.params["min_squeeze_samples"]:
            return False, buy_prob, f"No prior squeeze ({recent}/{L} samples compressed)", "no_squeeze"
        if mom <= 0 or s.close[-1] <= s.close[-2]:
            return False, buy_prob, "Squeeze released to the downside", "wrong_direction"
        return True, max(buy_prob, 0.65), f"Squeeze released up after {recent} compressed samples", None

    def _exit(self, ctx, s):
        ema = s.ema(self.params["period"])[-1]
        dist = (s.last - ema) / s.last if s.last else 0.0
        if s.last < ema:
            return True, 0.75, f"Lost EMA{self.params['period']}: expansion move over", 1.0
        return False, _clip(0.5 - dist * 30), f"Holding above EMA{self.params['period']}", 1.0

    def _bias(self, ctx, s):
        sq = self._squeeze_on(s)
        mom = s.last - s.sma(self.params["period"])[-1]
        atr = s.atr(self.params["period"]) or (s.last * 0.005)
        # Direction is unresolved while compressed, so abstain from a view.
        return 0.0 if sq[-1] else float(np.tanh(mom / atr / 2.0))


# =============================================================================
# Ideas from Krexibd/quant-trading (MIT), re-implemented for this engine
# =============================================================================

class ParabolicSARStrategy(QuantStrategy):
    name = "parabolic_sar"
    display_name = "Parabolic SAR"
    description = ("Wilder's stop-and-reverse: enters when the SAR flips below price, "
                   "exits when it flips back above. The SAR itself is an accelerating "
                   "trailing stop.")
    source = "J. Welles Wilder, New Concepts in Technical Trading Systems (1978); Krexibd/quant-trading"
    regimes = ("trending_up", "volatile")
    min_history = 40
    params = {"af_start": 0.02, "af_step": 0.02, "af_max": 0.2, "flip_within": 3}

    def _sar(self, s: PriceSeries):
        """(sar, direction) series on closes; direction +1 = SAR below price (long)."""
        key = ("psar", self.params["af_start"], self.params["af_step"], self.params["af_max"])
        if key in s._cache:
            return s._cache[key]
        x = s.close
        n = len(x)
        sar = np.full(n, np.nan)
        d = np.zeros(n)
        if n < 3:
            s._cache[key] = (sar, d)
            return sar, d
        up = x[1] >= x[0]
        af = self.params["af_start"]
        ep = max(x[0], x[1]) if up else min(x[0], x[1])
        cur = min(x[0], x[1]) if up else max(x[0], x[1])
        for i in range(2, n):
            cur = cur + af * (ep - cur)
            if up:
                cur = min(cur, x[i - 1], x[i - 2])
                if x[i] < cur:
                    up, cur, ep, af = False, ep, x[i], self.params["af_start"]
                elif x[i] > ep:
                    ep, af = x[i], min(af + self.params["af_step"], self.params["af_max"])
            else:
                cur = max(cur, x[i - 1], x[i - 2])
                if x[i] > cur:
                    up, cur, ep, af = True, ep, x[i], self.params["af_start"]
                elif x[i] < ep:
                    ep, af = x[i], min(af + self.params["af_step"], self.params["af_max"])
            sar[i] = cur
            d[i] = 1 if up else -1
        s._cache[key] = (sar, d)
        return sar, d

    def _entry(self, ctx, s, gates):
        sar, d = self._sar(s)
        gates.update(sar=float(sar[-1]), direction=int(d[-1]))
        w = self.params["flip_within"]
        buy_prob = 0.5 + 0.2 * d[-1]
        if d[-1] != 1:
            return False, buy_prob, f"SAR above price ({sar[-1]:.6g}): downtrend", "downtrend"
        if not (d[-(w + 1):-1] == -1).any():
            return False, buy_prob, f"SAR already below price for > {w} samples; waiting for a fresh flip", "stale_signal"
        return True, max(buy_prob, 0.65), f"Parabolic SAR flipped below price ({sar[-1]:.6g})", None

    def _exit(self, ctx, s):
        sar, d = self._sar(s)
        if d[-1] == -1:
            return True, 0.8, f"Parabolic SAR flipped above price ({sar[-1]:.6g})", 1.0
        dist = (s.last - sar[-1]) / s.last if s.last else 0.0
        return False, _clip(0.5 - dist * 20), f"SAR trailing at {sar[-1]:.6g}", 1.0

    def _bias(self, ctx, s):
        sar, d = self._sar(s)
        atr = s.atr(14) or (s.last * 0.005)
        if np.isnan(sar[-1]):
            return 0.0
        return float(d[-1] * min(0.4 + abs(s.last - sar[-1]) / atr * 0.15, 1.0))


class AwesomeOscillatorStrategy(QuantStrategy):
    name = "awesome_saucer"
    display_name = "Awesome Oscillator Saucer"
    description = ("Bill Williams' Awesome Oscillator (SMA5 - SMA34): enters on a bullish "
                   "saucer above zero (two falling bars then a rising one) or a zero-line "
                   "cross up; exits when AO drops below zero.")
    source = "Bill Williams, Trading Chaos (1995); Krexibd/quant-trading"
    regimes = ("trending_up", "mixed")
    min_history = 45
    params = {"fast": 5, "slow": 34}

    def _ao(self, s: PriceSeries) -> np.ndarray:
        # Close-only: the midpoint (H+L)/2 is not available per sample.
        return s.sma(self.params["fast"]) - s.sma(self.params["slow"])

    def _entry(self, ctx, s, gates):
        ao = self._ao(s)
        atr = s.atr(14) or (s.last * 0.005)
        gates.update(ao=float(ao[-1]))
        if np.isnan(ao[-4:]).any():
            return False, 0.0, "AO not yet defined", "no_data"
        buy_prob = 0.5 + _clip(ao[-1] / atr / 3.0, -1, 1) * 0.25
        a3, a2, a1 = ao[-3], ao[-2], ao[-1]
        saucer = a3 > 0 and a2 > 0 and a1 > 0 and a3 > a2 and a1 > a2
        zero_cross = ao[-2] <= 0 < a1
        if not (saucer or zero_cross):
            return False, buy_prob, f"No saucer or zero-line cross (AO {a1:.4g})", "no_signal"
        kind = "saucer" if saucer else "zero-line cross"
        return True, max(buy_prob, 0.6), f"AO bullish {kind} (AO {a1:.4g})", None

    def _exit(self, ctx, s):
        ao = self._ao(s)
        atr = s.atr(14) or (s.last * 0.005)
        sell = _clip(0.5 - ao[-1] / atr * 0.2)
        if ao[-1] < 0:
            return True, sell, f"AO below zero ({ao[-1]:.4g})", 1.0
        return False, sell, f"AO {ao[-1]:.4g} above zero", 1.0

    def _bias(self, ctx, s):
        ao = self._ao(s)
        atr = s.atr(14) or (s.last * 0.005)
        return 0.0 if np.isnan(ao[-1]) else float(np.tanh(ao[-1] / atr / 2.0))


class HeikinAshiStrategy(QuantStrategy):
    name = "heikin_ashi"
    display_name = "Heikin-Ashi Reversal"
    description = ("Builds Heikin-Ashi candles from micro-bars. Enters on a strong bullish "
                   "HA candle (no lower shadow) after bearish ones; exits on a strong "
                   "bearish HA candle.")
    source = "Heikin-Ashi averaging (Munehisa Homma tradition; Dan Valcu, 2004); Krexibd/quant-trading"
    regimes = ("trending_up", "mixed", "volatile")
    params = {"bar_samples": 5, "min_bars": 12, "min_bearish_before": 2}
    min_history = 60

    def _ha(self, s: PriceSeries):
        key = ("heikin", self.params["bar_samples"])
        if key in s._cache:
            return s._cache[key]
        o, h, l, c = s.bars(self.params["bar_samples"])
        n = len(c)
        hc = (o + h + l + c) / 4.0
        ho = np.empty(n)
        if n:
            ho[0] = (o[0] + c[0]) / 2.0
            for i in range(1, n):
                ho[i] = (ho[i - 1] + hc[i - 1]) / 2.0
        hh = np.maximum.reduce([h, ho, hc]) if n else h
        hl = np.minimum.reduce([l, ho, hc]) if n else l
        s._cache[key] = (ho, hh, hl, hc)
        return ho, hh, hl, hc

    def _entry(self, ctx, s, gates):
        ho, hh, hl, hc = self._ha(s)
        if len(hc) < self.params["min_bars"]:
            return False, 0.0, f"Need {self.params['min_bars']} HA bars, have {len(hc)}", "no_data"
        body = hc[-1] - ho[-1]
        tol = abs(body) * 0.05
        strong_bull = body > 0 and (min(ho[-1], hc[-1]) - hl[-1]) <= tol
        k = self.params["min_bearish_before"]
        prior_bear = bool((hc[-(k + 1):-1] < ho[-(k + 1):-1]).all())
        gates.update(ha_body=float(body), strong_bull=bool(strong_bull), prior_bearish=prior_bear)
        buy_prob = 0.5 + (0.2 if body > 0 else -0.2)
        if not strong_bull:
            return False, buy_prob, "Latest Heikin-Ashi candle is not a strong bullish bar", "no_signal"
        if not prior_bear:
            return False, buy_prob, f"No bearish run in the prior {k} HA bars to reverse", "no_reversal"
        return True, max(buy_prob, 0.65), f"Heikin-Ashi reversal: strong bull bar after {k} bearish", None

    def _exit(self, ctx, s):
        ho, hh, hl, hc = self._ha(s)
        if len(hc) < 2:
            return False, 0.3, "Too few HA bars", 1.0
        body = hc[-1] - ho[-1]
        tol = abs(body) * 0.05
        strong_bear = body < 0 and (hh[-1] - max(ho[-1], hc[-1])) <= tol
        two_bear = bool((hc[-2:] < ho[-2:]).all())
        sell = 0.7 if body < 0 else 0.3
        if strong_bear or two_bear:
            return True, sell, "Heikin-Ashi turned bearish" + (" (no upper shadow)" if strong_bear else " (2 bars)"), 1.0
        return False, sell, "Heikin-Ashi still bullish", 1.0

    def _bias(self, ctx, s):
        ho, hh, hl, hc = self._ha(s)
        if len(hc) < 3:
            return 0.0
        rng = (hh[-3:] - hl[-3:]).mean()
        return 0.0 if rng <= 0 else float(np.tanh(((hc[-3:] - ho[-3:]).mean()) / rng * 2))


class CandleReversalStrategy(QuantStrategy):
    name = "candle_reversal"
    display_name = "Hammer / Shooting Star"
    description = ("Candlestick reversal on micro-bars: buys a hammer (long lower shadow) "
                   "after a decline, exits on a shooting star (long upper shadow) or "
                   "a close below the recent low.")
    source = "Steve Nison, Japanese Candlestick Charting Techniques (1991); Krexibd/quant-trading shooting star"
    regimes = ("ranging", "mixed")
    min_history = 60
    params = {"bar_samples": 5, "shadow_ratio": 2.0, "max_other_shadow": 0.35, "trend_bars": 6}

    def _shape(self, o, h, l, c, i):
        body = abs(c[i] - o[i])
        rng = h[i] - l[i]
        if rng <= 0:
            return 0.0, 0.0, 0.0
        body = max(body, rng * 0.02)
        return body, min(o[i], c[i]) - l[i], h[i] - max(o[i], c[i])

    def _hammer(self, o, h, l, c, i) -> bool:
        body, lower, upper = self._shape(o, h, l, c, i)
        return body > 0 and lower >= self.params["shadow_ratio"] * body and upper <= self.params["max_other_shadow"] * (h[i] - l[i])

    def _star(self, o, h, l, c, i) -> bool:
        body, lower, upper = self._shape(o, h, l, c, i)
        return body > 0 and upper >= self.params["shadow_ratio"] * body and lower <= self.params["max_other_shadow"] * (h[i] - l[i])

    def _entry(self, ctx, s, gates):
        o, h, l, c = s.bars(self.params["bar_samples"])
        t = self.params["trend_bars"]
        if len(c) < t + 1:
            return False, 0.0, "Too few bars", "no_data"
        declined = c[-2] < c[-(t + 1):-1].mean()
        hammer = self._hammer(o, h, l, c, -1)
        gates.update(hammer=bool(hammer), prior_decline=bool(declined))
        buy_prob = 0.55 if hammer else 0.45
        if not hammer:
            return False, buy_prob, "No hammer on the latest bar", "no_signal"
        if not declined:
            return False, buy_prob, "Hammer without a prior decline is not a reversal", "no_decline"
        return True, 0.62, "Hammer after a decline", None

    def _exit(self, ctx, s):
        o, h, l, c = s.bars(self.params["bar_samples"])
        if len(c) < 4:
            return False, 0.3, "Too few bars", 1.0
        if self._star(o, h, l, c, -1):
            return True, 0.75, "Shooting star: bearish reversal bar", 1.0
        if s.last < l[-4:-1].min():
            return True, 0.75, f"Closed below the 3-bar low {l[-4:-1].min():.6g}", 1.0
        return False, 0.35, "No bearish reversal bar", 1.0

    def _bias(self, ctx, s):
        o, h, l, c = s.bars(self.params["bar_samples"])
        if len(c) < 2:
            return 0.0
        if self._hammer(o, h, l, c, -1):
            return 0.5
        if self._star(o, h, l, c, -1):
            return -0.5
        return 0.0


class PairsReversionStrategy(QuantStrategy):
    name = "pairs_reversion"
    display_name = "Pairs Cointegration Reversion"
    description = ("Engle-Granger pairs trade, long leg only: buys a symbol when its "
                   "spread against a cointegrated partner is >= 2 sigma cheap, exits when "
                   "the spread closes. Partner, hedge ratio and spread stats come from the "
                   "analysis worker; only the live z-score is computed on the tick path.")
    source = "Engle & Granger (1987); Gatev, Goetzmann & Rouwenhorst (2006); Krexibd/quant-trading pair trading"
    regimes = ("ranging", "mixed", "trending_up", "volatile")
    min_history = 100
    params = {"entry_z": -2.0, "exit_z": 0.0, "stop_z": -4.0}

    def _live_z(self, ctx):
        from core.state import state
        a = state.fresh_analysis(ctx.symbol)
        pair = (a or {}).get("pair")
        if not pair:
            return None, None
        tick = state.latest_prices.get(pair["partner"])
        if tick is None or tick.price <= 0 or ctx.price <= 0:
            return None, pair
        spread = np.log(ctx.price) - pair["alpha"] - pair["beta"] * np.log(tick.price)
        return float((spread - pair["mean"]) / pair["std"]), pair

    def _has_history(self, ctx):
        return super()._has_history(ctx) and self._live_z(ctx)[0] is not None

    def _missing(self, ctx):
        if not super()._has_history(ctx):
            return super()._missing(ctx)
        return "No cointegrated partner from the analysis worker (or partner has no live price)"

    def _entry(self, ctx, s, gates):
        z, pair = self._live_z(ctx)
        gates.update(pair_z=z, partner=pair["partner"], adf_t=pair["adf_t"], half_life=pair["half_life"])
        buy_prob = 0.5 + (-z - 1.0) * 0.15
        if z > self.params["entry_z"]:
            return False, buy_prob, f"Spread vs {pair['partner']} z {z:.2f} not cheap (need <= {self.params['entry_z']})", "not_stretched"
        if z < self.params["stop_z"]:
            return False, buy_prob, f"Spread z {z:.2f}: relationship may have broken", "breakdown"
        return True, max(buy_prob, 0.6), (f"Cheap vs cointegrated {pair['partner']}: z {z:.2f}, "
                                          f"beta {pair['beta']:.2f}, ADF t {pair['adf_t']:.2f}, half-life {pair['half_life']:.0f}"), None

    def _exit(self, ctx, s):
        z, pair = self._live_z(ctx)
        sell = _clip(0.5 + z * 0.25)
        if z >= self.params["exit_z"]:
            return True, sell, f"Spread vs {pair['partner']} closed (z {z:.2f})", 1.0
        if z <= self.params["stop_z"]:
            return True, 0.85, f"Spread vs {pair['partner']} blew out (z {z:.2f})", 1.0
        return False, sell, f"Spread z {z:.2f} vs {pair['partner']}", 1.0

    def _bias(self, ctx, s):
        z, _ = self._live_z(ctx)
        return -z / 2.5


LIBRARY = (
    BollingerReversionStrategy,
    ConnorsRSI2Strategy,
    ZScoreReversionStrategy,
    VWAPReversionStrategy,
    MACDTrendStrategy,
    MACrossoverStrategy,
    DonchianBreakoutStrategy,
    SupertrendStrategy,
    TimeSeriesMomentumStrategy,
    VolatilitySqueezeStrategy,
    ParabolicSARStrategy,
    AwesomeOscillatorStrategy,
    HeikinAshiStrategy,
    CandleReversalStrategy,
    PairsReversionStrategy,
)
