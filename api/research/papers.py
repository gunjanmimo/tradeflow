"""
Published intraday strategies, replayed trade by trade on our one-minute bars.

research/edge.py asks whether a signal's next H minutes beat a random entry.
The two strategies here are defined by their exits (a stop, a trailing band,
the close) as much as by their entries, so they are simulated path by path.
Both are LONG-ONLY here, as the engine is; the papers also trade the short side.

  orb_sip      Zarattini, Barbon & Aziz (2024), "A Profitable Day Trading
               Strategy for the U.S. Equity Market" (SSRN 4729284): 5-minute
               opening-range breakout on "stocks in play".
                 - in play: first-5-minute volume / its mean over the previous 14
                   sessions (relative volume) >= rvol_min; price > $5; daily
                   ATR(14) > $0.50; optionally only the top_k by relative volume
                 - long only if the first 5-minute candle closed up
                 - buy-stop at the opening-range high, from 09:35
                 - stop: stop_atr x daily ATR(14) below the entry trigger
                 - exit at the stop or the session's last close
               Replicated on QuantConnect (Sharpe 2.4 on 2016). An independent
               replication of the index version found the gross edge real and
               the net edge ~zero after costs: the stop is so tight that costs
               are a large share of the risk.

  noise_area   Zarattini, Aziz & Barbon (2024), "Beat the Market: An Effective
               Intraday Momentum Strategy for S&P500 ETF (SPY)" (SSRN 4824172).
                 - sigma(m): mean over the previous `lookback` sessions of
                   |close at minute m / that day's open - 1|
                 - upper band UB(m) = max(open, previous close) x (1 + band x sigma(m))
                 - checked every `every` minutes (10:00, 10:30, ...): buy when the
                   close is above UB and above the session VWAP; sell when it falls
                   below max(UB, VWAP), or at the session's last close
               Orders fill at the next bar's open.

Returns are gross log returns in bps from mid prices; summaries subtract the
round-trip cost (core/costs.py) exactly as research/edge.py does, with t-stats
clustered by day and the newest 40% of sessions as the holdout.

Everything a decision reads is known at that moment: relative volume and ATR
come from earlier sessions only, sigma(m) from earlier sessions only, the
opening range from bars that have closed.
"""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from core.costs import CostModel
from research.data import Bars, ny_minute_of_day
from research.edge import Events, EdgeRow, summarize, HOLDOUT_FRACTION

OPEN_MIN = 570                      # 09:30 NY
SESSION = 390                       # regular-session minutes


@dataclass
class PaperTrade:
    symbol: str
    day: int
    entry_min: int                  # NY minute of day of the entry fill
    exit_min: int
    entry: float                    # mid price of the entry (trigger or open)
    exit: float
    risk: float = 0.0               # entry - stop (0 when the strategy has no fixed stop)
    reason: str = ""

    @property
    def ret_bps(self) -> float:
        return float(np.log(self.exit / self.entry) * 1e4)


@dataclass
class Session:
    """One symbol's regular session laid on a 390-minute grid (NaN where no bar printed)."""
    day: int
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray
    last: int = field(default=-1)   # index of the last bar that printed

    @property
    def open(self) -> float:
        return float(self.o[0])


def sessions(b: Bars) -> List[Session]:
    """Regular sessions whose 09:30 bar printed, oldest first."""
    mod = ny_minute_of_day(b.minute)
    out = []
    for day, sl in b.day_slices():
        k = mod[sl] - OPEN_MIN
        ok = (k >= 0) & (k < SESSION)
        if not ok.any():
            continue
        grid = {name: np.full(SESSION, np.nan) for name in "ohlcv"}
        for name, arr in zip("ohlcv", (b.o, b.h, b.l, b.c, b.v)):
            grid[name][k[ok]] = arr[sl][ok]
        if not np.isfinite(grid["o"][0]):
            continue
        s = Session(day, grid["o"], grid["h"], grid["l"], grid["c"], np.nan_to_num(grid["v"]))
        s.last = int(np.flatnonzero(np.isfinite(s.c))[-1])
        out.append(s)
    return out


def _daily_atr(ss: List[Session], n: int = 14) -> np.ndarray:
    """ATR(n) of the daily bars BEFORE each session (NaN until n days exist)."""
    hi = np.array([np.nanmax(s.h) for s in ss])
    lo = np.array([np.nanmin(s.l) for s in ss])
    cl = np.array([s.c[s.last] for s in ss])
    prev = np.r_[np.nan, cl[:-1]]
    tr = np.nanmax(np.vstack([hi - lo, np.abs(hi - prev), np.abs(lo - prev)]), axis=0)
    out = np.full(len(ss), np.nan)
    for i in range(n, len(ss)):
        out[i] = tr[i - n:i].mean()
    return out


# ---------------------------------------------------------------------------
# Opening-range breakout on stocks in play
# ---------------------------------------------------------------------------

def orb_candidates(b: Bars, or_min: int = 5, lookback: int = 14) -> List[dict]:
    """Per session: the long setup's inputs, all known at 09:35."""
    ss = sessions(b)
    atr = _daily_atr(ss, lookback)
    or_vol = np.array([s.v[:or_min].sum() for s in ss])
    out = []
    for i, s in enumerate(ss):
        if i < lookback or not np.isfinite(atr[i]):
            continue
        if not np.isfinite(s.c[or_min - 1]) or s.last <= or_min:
            continue
        base = or_vol[i - lookback:i].mean()
        out.append({"symbol": b.symbol, "day": s.day, "session": s, "atr": float(atr[i]),
                    "rvol": float(or_vol[i] / base) if base > 0 else 0.0,
                    "up": bool(s.c[or_min - 1] > s.o[0]),
                    "or_high": float(np.nanmax(s.h[:or_min])), "price": float(s.c[or_min - 1])})
    return out


def _entry_bar(s: "Session", k: int) -> Optional[int]:
    """The first bar at or after k that printed."""
    while k <= s.last and not np.isfinite(s.o[k]):
        k += 1
    return k if k <= s.last else None


def orb_trade(cand: dict, stop_atr: float = 0.10, or_min: int = 5, entry_mode: str = "stop") -> Optional[PaperTrade]:
    """
    entry_mode
      stop     the paper: a resting buy-stop at the range high (fills at the high, or the open on a gap)
      close    the live engine: a bar CLOSES above the range high, market buy at the next bar's open
      placebo  no breakout needed: market buy at the 09:35 open on the same days (the control)
    The stop sits stop_atr x ATR below the entry price.
    """
    s: Session = cand["session"]
    trigger = cand["or_high"]
    k = None
    if entry_mode == "placebo":
        k = _entry_bar(s, or_min)
        entry = float(s.o[k]) if k is not None else None
    elif entry_mode == "close":
        for j in range(or_min, s.last):
            if np.isfinite(s.c[j]) and s.c[j] > trigger:
                k = _entry_bar(s, j + 1)
                break
        entry = float(s.o[k]) if k is not None else None
    else:
        for j in range(or_min, s.last + 1):
            if np.isfinite(s.h[j]) and s.h[j] > trigger:
                k = j
                break
        entry = max(trigger, float(s.o[k])) if k is not None else None   # a gap fills at the open
    if k is None:
        return None
    stop = entry - stop_atr * cand["atr"]
    risk = entry - stop
    # Same bar: the order of its high and low is unknown; if it also reached
    # the stop, count the stop (conservative).
    if s.l[k] <= stop:
        return PaperTrade(cand["symbol"], s.day, OPEN_MIN + k, OPEN_MIN + k, entry, stop, risk, "stop")
    for j in range(k + 1, s.last + 1):
        if not np.isfinite(s.o[j]):
            continue
        if s.o[j] <= stop:
            return PaperTrade(cand["symbol"], s.day, OPEN_MIN + k, OPEN_MIN + j, entry, float(s.o[j]),
                              risk, "stop (gap)")
        if s.l[j] <= stop:
            return PaperTrade(cand["symbol"], s.day, OPEN_MIN + k, OPEN_MIN + j, entry, stop, risk, "stop")
    return PaperTrade(cand["symbol"], s.day, OPEN_MIN + k, OPEN_MIN + s.last, entry,
                      float(s.c[s.last]), risk, "close")


def orb_sip(bars: Dict[str, Bars], rvol_min: float = 1.0, top_k: Optional[int] = None,
            stop_atr: float = 0.10, min_price: float = 5.0, min_atr: float = 0.50,
            entry_mode: str = "stop") -> List[PaperTrade]:
    by_day = defaultdict(list)
    for sym, b in bars.items():
        for c in orb_candidates(b):
            if c["up"] and c["rvol"] >= rvol_min and c["price"] > min_price and c["atr"] > min_atr:
                by_day[c["day"]].append(c)
    trades = []
    for day in sorted(by_day):
        cands = sorted(by_day[day], key=lambda c: c["rvol"], reverse=True)
        for c in cands[:top_k] if top_k else cands:
            t = orb_trade(c, stop_atr, entry_mode=entry_mode)
            if t is not None:
                trades.append(t)
    return trades


# ---------------------------------------------------------------------------
# Noise-area intraday momentum
# ---------------------------------------------------------------------------

def noise_area(b: Bars, lookback: int = 14, band: float = 1.0, every: int = 30) -> List[PaperTrade]:
    ss = sessions(b)
    if len(ss) <= lookback:
        return []
    # |close / open - 1| per session and minute, carried forward over missing bars.
    move = np.full((len(ss), SESSION), np.nan)
    for i, s in enumerate(ss):
        c = s.c.copy()
        for k in range(1, SESSION):
            if not np.isfinite(c[k]):
                c[k] = c[k - 1]
        move[i] = np.abs(c / s.open - 1.0)
    trades = []
    checks = [k for k in range(every - 1, SESSION, every)]   # bar ending at 10:00, 10:30, ...
    for i in range(lookback, len(ss)):
        s, prev = ss[i], ss[i - 1]
        sigma = np.nanmean(move[i - lookback:i], axis=0)
        anchor_up = max(s.open, float(prev.c[prev.last]))
        ub = anchor_up * (1.0 + band * sigma)
        tp = (s.h + s.l + s.c) / 3.0
        pv = np.nancumsum(np.where(np.isfinite(tp), tp * s.v, 0.0))
        vv = np.cumsum(s.v)
        vwap = np.where(vv > 0, pv / np.maximum(vv, 1e-12), np.nan)
        entry_k = None
        for k in checks:
            if k >= s.last or not np.isfinite(s.c[k]) or not np.isfinite(vwap[k]):
                continue
            nxt = k + 1
            while nxt <= s.last and not np.isfinite(s.o[nxt]):
                nxt += 1
            if nxt > s.last:
                break
            if entry_k is None:
                if s.c[k] > ub[k] and s.c[k] > vwap[k]:
                    entry_k = nxt
            elif s.c[k] < max(ub[k], vwap[k]):
                trades.append(PaperTrade(b.symbol, s.day, OPEN_MIN + entry_k, OPEN_MIN + nxt,
                                         float(s.o[entry_k]), float(s.o[nxt]), reason="band/VWAP"))
                entry_k = None
        if entry_k is not None:
            trades.append(PaperTrade(b.symbol, s.day, OPEN_MIN + entry_k, OPEN_MIN + s.last,
                                     float(s.o[entry_k]), float(s.c[s.last]), reason="close"))
    return trades


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def holdout_start(bars: Dict[str, Bars]) -> int:
    days = np.unique(np.concatenate([b.days() for b in bars.values()]))
    return int(days[int(len(days) * (1 - HOLDOUT_FRACTION))]) if len(days) else 0


def summarize_trades(name: str, trades: Sequence[PaperTrade], costs: CostModel,
                     holdout_from: int) -> EdgeRow:
    """research/edge.py's row for a list of trades (horizon 0 = held to its own exit; excess = gross)."""
    ev = Events(ret=np.array([t.ret_bps for t in trades]), day=np.array([t.day for t in trades], dtype=np.int64),
                base=0.0)
    return summarize(name, 0, [ev], costs, holdout_from)


def r_stats(trades: Sequence[PaperTrade], costs: CostModel) -> dict:
    """For fixed-stop trades: mean result in R net of costs, and costs as a share of R."""
    rs = [t for t in trades if t.risk > 0]
    if not rs:
        return {}
    net_r = [((t.exit - t.entry) - costs.per_side * (t.entry + t.exit)) / t.risk for t in rs]
    cost_r = [costs.per_side * (t.entry + t.exit) / t.risk for t in rs]
    stop_bps = [t.risk / t.entry * 1e4 for t in rs]
    return {"net_R": round(float(np.mean(net_r)), 3), "cost_R": round(float(np.mean(cost_r)), 3),
            "stop_bps": round(float(np.median(stop_bps)), 1),
            "hit_rate": round(float(np.mean([t.exit > t.entry for t in rs]) * 100), 1)}


# ---------------------------------------------------------------------------
# The study: python -m research papers
# ---------------------------------------------------------------------------

def _market_ret(t: PaperTrade, market: Dict[int, Session]) -> float:
    """The market's (SPY's) log return in bps over the trade's own minutes."""
    s = market.get(t.day)
    if s is None:
        return float("nan")
    a, b = t.entry_min - OPEN_MIN, min(t.exit_min - OPEN_MIN, s.last)
    pa = s.o[a]
    pb = s.c[s.last] if t.reason == "close" else s.o[b] if np.isfinite(s.o[b]) else s.c[b]
    return float(np.log(pb / pa) * 1e4) if np.isfinite(pa) and np.isfinite(pb) else float("nan")


def row(name: str, trades: Sequence[PaperTrade], costs: CostModel, holdout_from: int,
        market: Optional[Dict[int, Session]] = None) -> dict:
    from research.edge import _clustered_t
    rt = costs.round_trip_bps
    if not trades:
        return {"name": name, "trades": 0}
    r = np.array([t.ret_bps for t in trades])
    d = np.array([t.day for t in trades], dtype=np.int64)
    first, ho = d < holdout_from, d >= holdout_from

    def part(x, dd):
        return (round(float(x.mean() - rt), 2), round(_clustered_t(x, dd), 2)) if len(x) >= 3 else (float("nan"),) * 2
    out = {"name": name, "trades": len(r), "per_day": round(len(r) / max(len(np.unique(d)), 1), 2),
           "gross": round(float(r.mean()), 2), "t": round(_clustered_t(r, d), 2),
           "net": round(float(r.mean() - rt), 2), "first": part(r[first], d[first]),
           "holdout": part(r[ho], d[ho]), **r_stats(trades, costs)}
    if market:
        m = np.array([_market_ret(t, market) for t in trades])
        ok = np.isfinite(m) & ho
        out["holdout_hedged"] = part((r - m)[ok], d[ok])
    return out


def study(bars: Dict[str, Bars], costs: CostModel) -> List[dict]:
    stocks = {s: b for s, b in bars.items() if s not in ("SPY", "QQQ")}
    ho = holdout_start(bars)
    market = {s.day: s for s in sessions(bars["SPY"])} if "SPY" in bars else None
    rows = []
    for rv in (1.0, 2.0):
        for sa in (0.10, 0.50):
            for mode in ("stop", "close", "placebo"):
                label = {"stop": "paper entry (buy-stop)", "close": "live entry (next open)",
                         "placebo": "PLACEBO: buy 09:35, no breakout"}[mode]
                rows.append(row(f"orb_sip rvol>={rv:g} stop {sa:g} ATR, {label}",
                                orb_sip(stocks, rvol_min=rv, stop_atr=sa, entry_mode=mode), costs, ho, market))
    for sym in ("SPY", "QQQ"):
        if sym in bars:
            rows.append(row(f"noise_area {sym} (long only)", noise_area(bars[sym]), costs, ho))
    rows.append(row(f"noise_area {len(stocks)} stocks (long only)",
                    [t for b in stocks.values() for t in noise_area(b)], costs, ho, market))
    return rows


def render(rows: List[dict], costs: CostModel, n_symbols: int, n_days: int) -> str:
    f = lambda p: f"{p[0]:+.2f} ({p[1]:+.2f})" if p and p[0] == p[0] else "-"
    lines = [f"# Published intraday strategies: {n_symbols} symbols, {n_days} sessions", "",
             f"Long only. Costs {costs.round_trip_bps:g} bps per round trip. bps per trade, t clustered by day. "
             "`first 60%` = oldest sessions, `holdout` = newest 40%; `hedged` subtracts SPY's return over "
             "the same minutes. See research/papers.py for the rules and sources.", "",
             "| strategy | trades | /day | gross | t | net | first 60% net (t) | holdout net (t) "
             "| holdout hedged (t) | net R | costs R | stop bps | hit% |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if not r.get("trades"):
            lines.append(f"| {r['name']} | 0 |" + " - |" * 11)
            continue
        lines.append(f"| {r['name']} | {r['trades']} | {r['per_day']} | {r['gross']:+.2f} | {r['t']:+.2f} "
                     f"| {r['net']:+.2f} | {f(r['first'])} | {f(r['holdout'])} | {f(r.get('holdout_hedged'))} "
                     f"| {r.get('net_R', '-')} | {r.get('cost_R', '-')} | {r.get('stop_bps', '-')} "
                     f"| {r.get('hit_rate', '-')} |")
    lines += ["", "A strategy worth paper-trading needs positive net on BOTH halves, and must beat its "
              "placebo: the same days and stocks without the entry signal."]
    return "\n".join(lines) + "\n"
