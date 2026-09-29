"""
The case file the desk's LLM agents read: what the observer saw, in numbers.

Built only from data the engine already holds -- minute bars, the trend read,
daily bars, the news log and sentiment, the scout's ranking, SPY -- so the
models reason over the same facts the strategy saw, never over anything they
could invent. Returned both as a dict (the dashboard) and as compact text
(the prompt).
"""
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np

from core.state import state

_NY = ZoneInfo("America/New_York")


def _pct(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return round((a / b - 1.0) * 100.0, 2) if a and b else None


def breakeven(price: float, stop: Optional[float], target: Optional[float]) -> Optional[float]:
    """P(target before stop) at which the trade pays zero before costs."""
    if not (price and stop and target) or not (stop < price < target):
        return None
    risk, reward = price - stop, target - price
    return round(risk / (risk + reward), 3)


def five_minute_candles(rows: List[list], n: int = 12) -> List[Dict[str, float]]:
    """The last n five-minute candles from closed one-minute bars [minute, o, h, l, c, v]."""
    if not rows:
        return []
    buckets: Dict[int, List[list]] = {}
    for r in rows:
        buckets.setdefault(int(r[0]) // 5, []).append(r)
    out = []
    for k in sorted(buckets)[-n:]:
        b = buckets[k]
        out.append({"t": datetime.fromtimestamp(k * 300, _NY).strftime("%H:%M"), "o": b[0][1],
                    "h": max(x[2] for x in b), "l": min(x[3] for x in b), "c": b[-1][4],
                    "v": sum(x[5] for x in b)})
    return out


def build(symbol: str, case: Dict[str, Any]) -> Dict[str, Any]:
    from core.minute_bars import minute_bars
    from core.universe import universe
    from engine.trend import board as trend_board
    from feeds.daily_bars import daily_bars
    from feeds.news_log import news_log
    from scout.features import daily_features
    from scout.watcher import session_levels, session_open_minute

    now = time.time()
    tick = state.latest_prices.get(symbol)
    price = tick.price if tick else case.get("price_signal")
    meta = universe.classify(symbol)
    rows = minute_bars.closed_rows(symbol, now)
    closes = np.array([r[4] for r in rows]) if rows else np.empty(0)
    levels = session_levels(rows, session_open_minute(now))
    tr = trend_board.read(symbol)
    q = state.quant_metrics.get(symbol)

    def ret(n):
        return _pct(closes[-1], closes[-1 - n]) if len(closes) > n else None

    dc = daily_bars.closes(symbol)
    hlv = daily_bars.hlv.get(symbol)
    today_day = int(datetime.now(_NY).replace(hour=0, minute=0, second=0, microsecond=0).timestamp() // 86400)
    days = daily_bars.series.get(symbol, (np.empty(0), None))[0]
    prev_close = float(dc[days < today_day][-1]) if dc is not None and (days < today_day).any() else None
    daily = daily_features(dc, *hlv) if dc is not None and hlv else None

    plan = case.get("plan") or {}
    stop, target = plan.get("stop"), plan.get("target")
    obs = case.get("observation") or {}

    sent = state.get_sentiment(symbol)
    news = [{"age_min": round((now - (r["published_at"] or r["at"])) / 60), "headline": r["headline"],
             "events": [e["event"] for e in r["events"]], "lean": r["lean"],
             "score": r["scores"].get(symbol)} for r in news_log.for_symbol(symbol, 6)]
    if not news:
        news = [{"age_min": round((now - h.published_at) / 60), "headline": h.headline, "events": [],
                 "lean": None, "score": {"pos": round(h.pos_prob, 2), "neg": round(h.neg_prob, 2)}}
                for h in list(state.sentiment_history.get(symbol) or [])[-6:]][::-1]

    spy = trend_board.read("SPY", max_age_s=30.0)
    spy_rows = minute_bars.closed_rows("SPY", now)
    spy_levels = session_levels(spy_rows, session_open_minute(now))
    spy_last = spy_rows[-1][4] if spy_rows else None

    from scout.service import scout
    pick = scout.picks.get(symbol) or {}
    row = pick.get("row") or {}

    brief = {
        "symbol": symbol, "name": meta.name, "sector": meta.sector, "at": now,
        "signal": {"strategy": case.get("strategy"), "buy_prob": case.get("signal_prob"),
                   "reason": (case.get("reason") or "")[:300]},
        "price": price, "prev_close": prev_close, "chg_today_pct": _pct(price, prev_close),
        "session_open": levels["open"], "vwap": levels["vwap"], "vs_vwap_pct": _pct(price, levels["vwap"]),
        "since_open_pct": _pct(price, levels["open"]),
        "ret_5m_pct": ret(5), "ret_15m_pct": ret(15), "ret_60m_pct": ret(60),
        "candles_5m": five_minute_candles(rows[-60:]),
        "trend": {"label": tr.label, "direction": round(tr.direction, 2), "confidence": round(tr.confidence, 2),
                  "reversal_down": tr.reversal_down, "reversal_up": tr.reversal_up},
        "rsi": round(q.rsi, 1) if q and q.rsi is not None else None,
        "spread_pct": round(q.spread * 100, 3) if q else None,
        "daily": {k: round(v * 100, 2) if k.startswith("ret_") else round(v, 3)
                  for k, v in (daily or {}).items() if k != "price"},
        "plan": {"stop": stop, "target": target, "stop_pct": _pct(stop, price), "target_pct": _pct(target, price),
                 "dollars": plan.get("dollars"), "qty": plan.get("qty"),
                 "breakeven_p": breakeven(price, stop, target)},
        "observation": obs,
        "sentiment": {"pos": round(sent.pos_prob, 2), "neg": round(sent.neg_prob, 2), "n": sent.n_headlines,
                      "agreement": round(sent.agreement, 2), "stale": sent.is_stale},
        "news": news,
        "market": {"spy_trend": spy.label if spy.ready else "unknown",
                   "spy_direction": round(spy.direction, 2) if spy.ready else None,
                   "spy_since_open_pct": _pct(spy_last, spy_levels["open"])},
        "scout": {"rank": pick.get("rank"), "score": pick.get("score"), "components": row.get("components"),
                  "reasons": row.get("reasons")} if pick else None,
    }
    brief["risk"] = risk_appetite(brief["plan"], price, stop, target)
    brief["text"] = render(brief)
    return brief


def risk_appetite(plan: Dict[str, Any], price, stop, target) -> Dict[str, Any]:
    """How much risk the user wants to take (the risk dial) and how much is left today."""
    p = state.risk_profile
    qty = plan.get("qty") or 0
    return {
        "dial": p.factor, "label": p.label,
        "stance": ("wants only clear, high-quality setups and little volatility" if p.factor <= 3 else
                   "balanced: a real edge, normal volatility" if p.factor <= 6 else
                   "accepts thinner edges and more volatility in exchange for more opportunities"),
        "risk_per_trade_pct": p.risk_per_trade_pct, "max_position_pct": p.max_position_notional_pct,
        "max_daily_loss_pct": p.max_daily_loss_pct, "loss_today_pct": round(float(state.daily_loss_pct or 0.0), 2),
        "positions": len(state.active_positions), "max_positions": p.max_concurrent_positions,
        "budget": round(float(state.hard_cap or 0.0), 2),
        "trade_risk_usd": round((price - stop) * qty, 2) if price and stop and qty else None,
        "trade_reward_usd": round((target - price) * qty, 2) if price and target and qty else None,
    }


def _f(v, suffix="", digits=2):
    return "n/a" if v is None else f"{v:+.{digits}f}{suffix}" if suffix == "%" else f"{v:.{digits}f}{suffix}"


def render(b: Dict[str, Any]) -> str:
    p = b["plan"]
    lines = [
        "UNITS: every % below is a percentage of price with two decimals (-0.47% is less than half of one "
        "percent). 'n/a' means the data does not exist yet (e.g. before the open) -- it is not a bad sign.",
        f"STOCK {b['symbol']} ({b['name'] or 'n/a'}, {b['sector']}). Long-only day trade; flat by the close.",
        f"SIGNAL from strategy '{b['signal']['strategy']}' (its conviction {_f(b['signal']['buy_prob'])}): "
        f"{b['signal']['reason']}",
        f"PRICE {_f(b['price'])}; today {_f(b['chg_today_pct'], '%')} vs prior close; "
        f"{_f(b['since_open_pct'], '%')} since the open; {_f(b['vs_vwap_pct'], '%')} vs VWAP {_f(b['vwap'])}.",
        f"MOMENTUM last 5m {_f(b['ret_5m_pct'], '%')}, 15m {_f(b['ret_15m_pct'], '%')}, "
        f"60m {_f(b['ret_60m_pct'], '%')}. Trend {b['trend']['label']} (direction {b['trend']['direction']:+.2f}, "
        f"confidence {b['trend']['confidence']:.2f}"
        + (", micro trend turning DOWN" if b["trend"]["reversal_down"] else "")
        + (", micro trend turning UP" if b["trend"]["reversal_up"] else "")
        + f"). RSI {_f(b['rsi'], '', 1)}; spread {_f(b['spread_pct'], '%', 3)}.",
    ]
    if b["candles_5m"]:
        lines.append("5-MIN CANDLES (time open/high/low/close volume): " + "; ".join(
            f"{c['t']} {c['o']:.2f}/{c['h']:.2f}/{c['l']:.2f}/{c['c']:.2f} {int(c['v'])}" for c in b["candles_5m"]))
    o = b["observation"]
    if o:
        above = o.get("above_vwap_share")
        lines.append(
            f"OBSERVED for {o.get('seconds', 0):.0f}s after the signal: price {_f(o.get('move_pct'), '%')} "
            f"(high {_f(o.get('high_pct'), '%')}, low {_f(o.get('low_pct'), '%')}), "
            + (f"above VWAP in {above:.0%} of checks" if above is not None else "no VWAP yet")
            + f", signal present in {o.get('persistence', 0):.0%} of checks.")
    d = b["daily"]
    if d:
        lines.append(f"DAILY 5d {_f(d.get('ret_5d'), '%')}, 20d {_f(d.get('ret_20d'), '%')}, "
                     f"60d {_f(d.get('ret_60d'), '%')}; ATR {_f(d.get('atr_pct'), '%')} a day; "
                     f"close/60d-high {_f(d.get('near_high'), '', 3)}; volume 5d vs norm {_f(d.get('rvol_5d'))}x.")
    # The breakeven stays out of the text: shown it, the model anchored its estimate
    # on it (5 of 8 replayed cases came back exactly at breakeven). The desk's
    # Decision step applies it.
    lines.append(f"PLAN entry ~{_f(b['price'])}, stop {_f(p['stop'])} ({_f(p['stop_pct'], '%')}), "
                 f"target {_f(p['target'])} ({_f(p['target_pct'], '%')}).")
    s = b["sentiment"]
    lines.append(f"NEWS SENTIMENT {s['n']} fresh headline(s), {s['pos']:.0%} positive / {s['neg']:.0%} negative, "
                 f"agreement {s['agreement']:.0%}" + (" (stale)" if s["stale"] else "") + ".")
    for n in b["news"][:5]:
        ev = f" [{', '.join(n['events'])}]" if n["events"] else ""
        lines.append(f"  - {n['age_min']} min ago: {n['headline']}{ev}")
    r = b.get("risk") or {}
    if r:
        lines.append(
            f"RISK APPETITE: the user set the risk dial to {r['dial']}/10 ({r['label']}): {r['stance']}. They accept "
            f"losing {r['risk_per_trade_pct']}% of the ${r['budget']:,.0f} budget per trade and {r['max_daily_loss_pct']}% "
            f"in a day (lost {r['loss_today_pct']}% so far today); {r['positions']} of {r['max_positions']} positions open."
            + (f" This trade risks ${r['trade_risk_usd']:,.2f} to make ${r['trade_reward_usd']:,.2f}."
               if r.get("trade_risk_usd") is not None and r.get("trade_reward_usd") is not None else ""))
    m = b["market"]
    lines.append(f"MARKET SPY {m['spy_trend']} (direction {_f(m['spy_direction'])}), "
                 f"{_f(m['spy_since_open_pct'], '%')} since the open.")
    if b["scout"]:
        sc = b["scout"]
        lines.append(f"SCOUT ranked it #{sc['rank']} today (score {_f(sc['score'])}): "
                     + "; ".join((sc.get("reasons") or [])[:3]))
    return "\n".join(lines)
