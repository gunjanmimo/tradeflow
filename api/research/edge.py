"""
Does a signal have an edge after costs? An event study that is hard to fool.

For every signal event (bar t): buy at the open of bar t+1, sell H bars later at
that bar's open -- or at the session's last close, since nothing is held
overnight. Events are taken one at a time per symbol (no overlapping trades),
as a single position would, and only between 09:35 and the no-new-entry cutoff.

Reported per signal and horizon, all in basis points per trade:

  gross      mean forward return of the events
  excess     gross minus the mean forward return of EVERY eligible bar over the
             same horizon (what a random entry earned): the signal's own edge
  t          t-statistic of gross, with standard errors clustered by DAY. Events
             on the same day share market moves; treating them as independent
             inflates t several-fold.
  net        gross minus the round-trip cost (core/costs.py)
  holdout    the same numbers on the newest 40% of days only. The signal was not
             tuned on them, so they are the honest estimate.
  months+    share of calendar months with positive net P&L

A signal worth trading needs net > 0 and a holdout t well above 2 -- not a
single good split.
"""
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from core.costs import CostModel
from research.data import Bars, ny_minute_of_day, MARKET
from research import signals as sigmod

FIRST_ENTRY_MIN = 575                # 09:35 NY: skip the opening auction's noise
LAST_ENTRY_MIN = 930                 # 15:30 NY: the platform's no-new-entry cutoff
HOLDOUT_FRACTION = 0.40


@dataclass
class EdgeRow:
    signal: str
    horizon: int
    events: int
    events_per_day: float
    gross_bps: float
    excess_bps: float
    t_gross: float
    net_bps: float
    win_rate: float
    holdout_events: int
    holdout_net_bps: float
    holdout_t: float
    months_positive: float

    def to_dict(self):
        return asdict(self)


@dataclass
class Events:
    """Forward returns (log, as bps) of one signal's events, with day and month labels."""
    ret: np.ndarray
    day: np.ndarray
    base: float                      # mean forward return of every eligible bar

    def __len__(self):
        return len(self.ret)


def _eligible(b: Bars) -> np.ndarray:
    mod = ny_minute_of_day(b.minute)
    return (mod >= FIRST_ENTRY_MIN) & (mod < LAST_ENTRY_MIN)


def forward_returns(b: Bars, horizon: int) -> np.ndarray:
    """log(exit / entry) in bps for an entry at open[t+1], exit open[t+1+H] or the day's last close."""
    n = len(b)
    out = np.full(n, np.nan)
    days = b.days()
    for _, sl in b.day_slices():
        s, e = sl.start, sl.stop           # bars s..e-1 are one session
        t = np.arange(s, e - 1)             # need an entry bar t+1 inside the session
        entry = b.o[t + 1]
        x = t + 1 + horizon
        inside = x < e
        exit_px = np.where(inside, b.o[np.minimum(x, e - 1)], b.c[e - 1])
        out[t] = np.log(exit_px / entry) * 1e4
    return out


def events_for(b: Bars, sig: np.ndarray, horizon: int) -> Events:
    fr = forward_returns(b, horizon)
    ok = _eligible(b) & np.isfinite(fr)
    base = float(np.nanmean(fr[ok])) if ok.any() else 0.0
    idx = np.flatnonzero(sig & ok)
    days = b.days()
    keep, busy_until = [], -1
    for t in idx:                       # one position at a time
        if t <= busy_until:
            continue
        keep.append(t)
        busy_until = t + 1 + horizon
    keep = np.asarray(keep, dtype=np.int64)
    return Events(ret=fr[keep], day=days[keep], base=base)


def _clustered_t(ret: np.ndarray, day: np.ndarray) -> float:
    """t-stat of the mean with day-clustered standard errors."""
    if len(ret) < 3:
        return float("nan")
    uniq, inv = np.unique(day, return_inverse=True)
    if len(uniq) < 3:
        return float("nan")
    sums = np.bincount(inv, weights=ret - ret.mean())
    var = (sums ** 2).sum() / len(ret) ** 2 * len(uniq) / (len(uniq) - 1)
    return float(ret.mean() / np.sqrt(var)) if var > 0 else float("nan")


def _month(day: np.ndarray) -> np.ndarray:
    return (day.astype("datetime64[D]").astype("datetime64[M]")).astype(np.int64)


def summarize(name: str, horizon: int, evs: List[Events], costs: CostModel,
              holdout_from_day: int) -> EdgeRow:
    ret = np.concatenate([e.ret for e in evs]) if evs else np.empty(0)
    day = np.concatenate([e.day for e in evs]) if evs else np.empty(0, dtype=np.int64)
    n = len(ret)
    rt = costs.round_trip_bps
    if n == 0:
        return EdgeRow(name, horizon, 0, 0.0, *([float("nan")] * 3), float("nan"), float("nan"),
                       0, float("nan"), float("nan"), float("nan"))
    base = float(np.average([e.base for e in evs], weights=[max(len(e), 1) for e in evs]))
    ho = day >= holdout_from_day
    net = ret - rt
    months = _month(day)
    per_month = defaultdict(float)
    for m, x in zip(months, net):
        per_month[m] += x
    n_days = len(np.unique(day))
    return EdgeRow(
        signal=name, horizon=horizon, events=n,
        events_per_day=round(n / max(n_days, 1), 2),
        gross_bps=round(float(ret.mean()), 2),
        excess_bps=round(float(ret.mean() - base), 2),
        t_gross=round(_clustered_t(ret, day), 2),
        net_bps=round(float(net.mean()), 2),
        win_rate=round(float((net > 0).mean() * 100), 1),
        holdout_events=int(ho.sum()),
        holdout_net_bps=round(float(net[ho].mean()), 2) if ho.any() else float("nan"),
        holdout_t=round(_clustered_t(ret[ho], day[ho]), 2) if ho.sum() >= 3 else float("nan"),
        months_positive=round(sum(1 for v in per_month.values() if v > 0) / max(len(per_month), 1), 2),
    )


def study(bars: Dict[str, Bars], signal_names: Sequence[str], horizons: Sequence[int],
          costs: CostModel, params: Optional[Dict[str, dict]] = None) -> List[EdgeRow]:
    params = params or {}
    ctx = {"market": bars.get(MARKET)}
    all_days = np.unique(np.concatenate([b.days() for b in bars.values()]))
    holdout_from = int(all_days[int(len(all_days) * (1 - HOLDOUT_FRACTION))]) if len(all_days) else 0
    trade_syms = [s for s in bars if s not in ("SPY", "QQQ")]
    rows = []
    for name in signal_names:
        fn = sigmod.SIGNALS[name]
        sigs = {s: fn(bars[s], ctx, **params.get(name, {})) for s in trade_syms}
        for h in horizons:
            evs = [events_for(bars[s], sigs[s], h) for s in trade_syms]
            rows.append(summarize(name, h, evs, costs, holdout_from))
    return rows


def render(rows: Iterable[EdgeRow], costs: CostModel, n_symbols: int, n_days: int) -> str:
    lines = [
        f"# Signal edge: {n_symbols} symbols, {n_days} sessions",
        "",
        f"Costs {costs.round_trip_bps:g} bps per round trip ({costs.half_spread_bps:g} bps half-spread + "
        f"{costs.slippage_bps:g} bps slippage per side). Entry at the next bar's open, exit H bars later; "
        "t-stats clustered by day. Holdout = newest 40% of sessions.",
        "",
        "| signal | H (min) | events | /day | gross bps | excess bps | t | net bps | win% "
        "| holdout n | holdout net | holdout t | months+ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        flag = " **" if (r.holdout_net_bps == r.holdout_net_bps and r.holdout_net_bps > 0
                         and r.holdout_t == r.holdout_t and r.holdout_t > 2 and r.net_bps > 0) else ""
        lines.append(
            f"| {r.signal}{flag} | {r.horizon} | {r.events} | {r.events_per_day} | {r.gross_bps:+.2f} "
            f"| {r.excess_bps:+.2f} | {r.t_gross:+.2f} | {r.net_bps:+.2f} | {r.win_rate:.0f} "
            f"| {r.holdout_events} | {r.holdout_net_bps:+.2f} | {r.holdout_t:+.2f} | {r.months_positive:.0%} |")
    lines += ["", "`**` = positive net overall AND on the holdout with holdout t > 2: a candidate worth "
              "paper-trading, not a guarantee."]
    return "\n".join(lines) + "\n"
