"""
Market regime detection.

Strategies are not universally good or bad; they are good in a regime. Trend
followers bleed in chop and mean-reverters get run over in trends. So before
choosing a strategy, classify what the market is doing right now:

  trending_up    clean directional move up        -> trend / breakout strategies
  trending_down  clean directional move down      -> nothing (the engine is long-only)
  ranging        price oscillating around a mean  -> mean-reversion strategies
  volatile       short-term vol well above normal -> breakout / squeeze strategies
  mixed          no clear read                    -> only regime-agnostic strategies
  unknown        not enough history               -> abstain

Inputs, all close-only:
  * Kaufman efficiency ratio  -- directional cleanliness of recent path
  * EMA 10/30 slope + return   -- which way
  * short/long realized vol    -- expansion vs normal
  * Hurst exponent             -- corroborates trending vs mean-reverting
"""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np

from engine.strategies.indicators import PriceSeries

REGIMES = ("trending_up", "trending_down", "ranging", "volatile", "mixed", "unknown")

MIN_SAMPLES = 60

# Thresholds, tuned conservatively: "mixed" is the honest default.
ER_TREND = 0.35
ER_TREND_WITH_HURST = 0.25
ER_RANGE = 0.18
HURST_TREND = 0.55
HURST_REVERT = 0.45
VOL_EXPANSION = 1.8


@dataclass
class RegimeRead:
    label: str
    efficiency_ratio: Optional[float] = None
    hurst: Optional[float] = None
    vol_ratio: Optional[float] = None
    slope: Optional[float] = None          # (EMA10 - EMA30) / price
    confidence: float = 0.0                # 0..1, how clearly the label is met
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k in ("efficiency_ratio", "hurst", "vol_ratio", "slope"):
            if d[k] is not None:
                d[k] = round(d[k], 4)
        d["confidence"] = round(d["confidence"], 3)
        return d


def detect(series: Optional[PriceSeries]) -> RegimeRead:
    if series is None or len(series) < MIN_SAMPLES:
        n = 0 if series is None else len(series)
        return RegimeRead("unknown", reasons=[f"{n}/{MIN_SAMPLES} samples"])

    er = series.efficiency_ratio(30) or 0.0
    hurst = series.hurst(20)
    fast, slow = series.ema(10)[-1], series.ema(30)[-1]
    price = series.last
    slope = float((fast - slow) / price) if price > 0 and not np.isnan(slow) else 0.0
    ret30 = series.ret(30) or 0.0
    short_vol = series.realized_vol(10)
    long_vol = series.realized_vol(min(60, len(series) - 1))
    vol_ratio = (short_vol / long_vol) if (short_vol and long_vol) else None

    read = RegimeRead("mixed", efficiency_ratio=er, hurst=hurst,
                      vol_ratio=vol_ratio, slope=slope)
    up = slope > 0 and ret30 > 0
    down = slope < 0 and ret30 < 0

    if vol_ratio is not None and vol_ratio >= VOL_EXPANSION:
        read.label = "volatile"
        read.confidence = min((vol_ratio - VOL_EXPANSION) / VOL_EXPANSION + 0.5, 1.0)
        read.reasons.append(f"short/long vol {vol_ratio:.2f}x >= {VOL_EXPANSION}")
        return read

    trending = er >= ER_TREND or (
        hurst is not None and hurst >= HURST_TREND and er >= ER_TREND_WITH_HURST)
    if trending and (up or down):
        read.label = "trending_up" if up else "trending_down"
        read.confidence = min(er / 0.7, 1.0)
        read.reasons.append(f"ER {er:.2f}, EMA10/30 slope {slope*100:+.3f}%, 30-sample return {ret30*100:+.2f}%")
        if hurst is not None:
            read.reasons.append(f"Hurst {hurst:.2f}")
        return read

    if er <= ER_RANGE or (hurst is not None and hurst <= HURST_REVERT):
        read.label = "ranging"
        read.confidence = min((ER_RANGE + 0.1 - er) / (ER_RANGE + 0.1), 1.0) if er <= ER_RANGE else 0.4
        read.reasons.append(f"ER {er:.2f} (choppy)"
                            + (f", Hurst {hurst:.2f} (mean-reverting)" if hurst is not None else ""))
        return read

    read.confidence = 0.2
    read.reasons.append(f"ER {er:.2f} between range ({ER_RANGE}) and trend ({ER_TREND}) thresholds")
    return read
