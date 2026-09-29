"""
Eligibility and scoring. Pure: takes gathered candidates, returns ranked rows.

A candidate is a dict:
  symbol, name, origins            what it is and which sources surfaced it
  daily                            scout.features.daily_features(...) or None
  dollar_volume                    average daily $ traded (20 sessions)
  today                            scout.features.today_features(...) or None
  news        {"n", "tone", "n_scored", "headline"}   last SCOUT_NEWS_LOOKBACK_HOURS
  reddit      {"mentions", "mentions_24h_ago", "upvotes", "rank"} or None
  home        the exchange board evidence (scout/exchanges.py) or None for US stocks
  stocktwits_rank                  1-based place on the trending list, or None
  listed                           True/False on Alpaca's tradable list, None unknown

Components (each 0..1; a missing one does not vote and the rest are renormalized):
  performance  30%  cross-sectional ranks of risk-adjusted 20-day momentum (35%),
                    5-day (15%) and 60-day (15%) return and 5-day volume surge (15%),
                    plus trend structure (20%): above the 20-day average, 20 above
                    50, close to the 60-day high
  today        15%  rank of today's % change (pre-market before the open) and the
                    volume pace against the previous session; absent on days with
                    no session, and then it does not vote
  home         15%  foreign stocks only: how it ranks on its own exchange today --
                    its % move there (60%) and how heavily it traded (40%) --
                    from the exchange boards (scout/exchanges.py). Asia closes
                    before New York opens, so this is fresh news for the US line
  news         20%  headline count (buzz, 30%) and tone (70%); tone is shrunk to
                    neutral when few headlines were scored, and no news is 0.35
  discussion   20%  Reddit attention (50%) and its growth over 24h (25%), and
                    StockTwits trending place (25%); undiscussed is 0.15

Attention without direction (a stock everyone is talking about because it is
collapsing) is capped by the other three: performance, today and tone all carry
the sign.
"""
import math
import re
from typing import Any, Dict, List, Optional

import numpy as np

from scout.features import MIN_HISTORY

WEIGHTS = {"performance": 0.30, "today": 0.15, "home": 0.15, "news": 0.20, "discussion": 0.20}

_US_TICKER = re.compile(r"^[A-Z]{1,5}(\.[A-Z])?$")
_NOT_COMMON = re.compile(r"\b(ETFs?|ETNs?|ETP|Fund|Warrants?|Rights?|Units?|Notes?|Depositary Shares? "
                         r"Representing .* Preferred|Preferred|Acquisition Corp)\b", re.IGNORECASE)
# Issuers that only sell funds: their name alone says "not a stock" ("ProShares
# UltraPro QQQ" has no "ETF" in it).
_FUND_ONLY = re.compile(r"\b(ProShares|Direxion|GraniteShares|iShares|SPDR|Vanguard|Tradr|Defiance|"
                        r"T-Rex|YieldMax|Roundhill|Leverage Shares|Global X|VanEck|MicroSectors|"
                        r"Xtrackers|First Trust|Amplify|Simplify|Volatility Shares|Tuttle|AXS)\b",
                        re.IGNORECASE)
# Issuers that are also listed companies (Invesco Ltd, Charles Schwab, WisdomTree
# Inc, State Street Corp, Grayscale): a fund only when the name reads like one.
_FUND_FAMILY = re.compile(r"\b(Invesco|Schwab|WisdomTree|State Street|Grayscale|Franklin|Fidelity|"
                          r"JPMorgan|Goldman Sachs|Select Sector)\b", re.IGNORECASE)
_FUNDISH = re.compile(r"\b(Trust|Series|Shares|Portfolio|Index)\b", re.IGNORECASE)
_LEVERAGED = re.compile(r"\b(\d(\.\d)?x|ultra(pro)?|leveraged|inverse|bull|bear|daily)\b", re.IGNORECASE)


def eligibility(symbol: str, name: str, daily: Optional[Dict[str, float]],
                dollar_volume: float, listed: Optional[bool],
                min_price: float, min_dollar_volume: float, exchange: Optional[str] = None) -> Optional[str]:
    """None when this is a liquid, exchange-listed common stock or ADR the bots may trade, else why not."""
    if not _US_TICKER.match(symbol):
        return "not a US stock ticker"
    if listed is False:
        return "not tradable on Alpaca"
    if exchange == "OTC":
        return "OTC-only: no prices on this market-data plan"
    if name:
        if (_NOT_COMMON.search(name) or _FUND_ONLY.search(name)
                or (_FUND_FAMILY.search(name) and _FUNDISH.search(name))):
            return f"not a common stock ({name})"
        if _LEVERAGED.search(name) and re.search(r"\bshares\b|\btrust\b", name, re.IGNORECASE):
            return f"leveraged / inverse product ({name})"
    if daily is None:
        return f"under {MIN_HISTORY} sessions of daily history"
    if daily["price"] < min_price:
        return f"price ${daily['price']:,.2f} under ${min_price:,.0f}"
    if dollar_volume < min_dollar_volume:
        return f"illiquid: ${dollar_volume / 1e6:,.1f}M a day (minimum ${min_dollar_volume / 1e6:,.0f}M)"
    return None


def pct_rank(values: Dict[str, float]) -> Dict[str, float]:
    """0..1 cross-sectional rank; ties share their average rank."""
    if not values:
        return {}
    keys = list(values)
    if len(keys) == 1:
        return {keys[0]: 0.5}
    arr = np.array([values[k] for k in keys], dtype=float)
    order = arr.argsort(kind="mergesort")
    ranks = np.empty(len(arr))
    ranks[order] = np.arange(len(arr))
    for v in np.unique(arr):                      # average ties
        m = arr == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return {k: float(r / (len(arr) - 1)) for k, r in zip(keys, ranks)}


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def news_score(news: Optional[Dict[str, Any]]) -> float:
    n = int((news or {}).get("n") or 0)
    buzz = _clip01(math.log1p(n) / math.log1p(8))
    tone = (news or {}).get("tone")
    if tone is None:
        tone_term = 0.5
    else:
        trust = min(1.0, int(news.get("n_scored") or 0) / 3)
        tone_term = 0.5 + (float(tone) - 0.5) * trust
    return 0.3 * buzz + 0.7 * tone_term


def discussion_score(reddit: Optional[Dict[str, Any]], st_rank: Optional[int]) -> float:
    if not reddit and not st_rank:
        return 0.15
    st = _clip01(1.0 - (st_rank - 1) / 30.0) if st_rank else 0.0
    if not reddit:
        return 0.125 + 0.25 * st
    m = float(reddit.get("mentions") or 0)
    m24 = float(reddit.get("mentions_24h_ago") or 0)
    attention = _clip01(math.log1p(m) / math.log1p(150))
    growth = max(-1.0, min(1.0, (m - m24) / (m24 + 10.0)))
    return 0.5 * attention + 0.25 * (0.5 + 0.5 * growth) + 0.25 * st


def performance_scores(daily: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Cross-sectional performance score for every stock with daily features."""
    ranks = {k: pct_rank({s: d[k] for s, d in daily.items() if d.get(k) is not None})
             for k in ("mom_adj_20d", "ret_5d", "ret_60d", "rvol_5d")}
    out = {}
    for s, d in daily.items():
        structure = (d["above_sma20"] + d["sma20_gt_50"] + _clip01((d["near_high"] - 0.85) / 0.15)) / 3
        out[s] = (0.35 * ranks["mom_adj_20d"].get(s, 0.5) + 0.15 * ranks["ret_5d"].get(s, 0.5)
                  + 0.15 * ranks["ret_60d"].get(s, 0.5) + 0.15 * ranks["rvol_5d"].get(s, 0.5)
                  + 0.20 * structure)
    return out


def today_scores(today: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    chg = pct_rank({s: t["chg_pct"] for s, t in today.items() if t.get("chg_pct") is not None})
    out = {}
    for s, t in today.items():
        if s not in chg:
            continue
        rv = t.get("rvol_today")
        out[s] = chg[s] if rv is None else 0.6 * chg[s] + 0.4 * _clip01(rv / 3.0)
    return out


def rank(candidates: List[Dict[str, Any]], min_price: float = 5.0,
         min_dollar_volume: float = 20e6) -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns {"ranked": eligible rows best first, "excluded": [{symbol, why}]}.
    Each ranked row carries its score, components, key features and reasons.
    """
    eligible, excluded = [], []
    for c in candidates:
        why = eligibility(c["symbol"], c.get("name") or "", c.get("daily"),
                          float(c.get("dollar_volume") or 0.0), c.get("listed"),
                          min_price, min_dollar_volume, c.get("exchange"))
        if why:
            excluded.append({"symbol": c["symbol"], "why": why, "origins": c.get("origins", [])})
        else:
            eligible.append(c)

    perf = performance_scores({c["symbol"]: c["daily"] for c in eligible})
    today = today_scores({c["symbol"]: c["today"] for c in eligible if c.get("today")})

    rows = []
    for c in eligible:
        s = c["symbol"]
        comps = {"performance": perf[s], "news": news_score(c.get("news")),
                 "discussion": discussion_score(c.get("reddit"), c.get("stocktwits_rank"))}
        if s in today:
            comps["today"] = today[s]
        if c.get("home"):
            comps["home"] = float(c["home"]["home"])
        w = sum(WEIGHTS[k] for k in comps)
        score = sum(WEIGHTS[k] * v for k, v in comps.items()) / w
        rows.append({
            "symbol": s, "name": c.get("name") or "", "score": round(score, 4),
            "country": c.get("country") or "US", "region": c.get("region") or "US",
            "home_exchange": (c.get("home") or {}).get("exchange_label"),
            "components": {k: round(v, 3) for k, v in comps.items()},
            "origins": sorted(c.get("origins", [])),
            "price": (c.get("today") or {}).get("price") or c["daily"]["price"],
            "features": _brief(c),
            "reasons": _reasons(c, comps),
        })
    rows.sort(key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    return {"ranked": rows, "excluded": excluded}


def _brief(c: Dict[str, Any]) -> Dict[str, Any]:
    d, t, n, rd = c["daily"], c.get("today") or {}, c.get("news") or {}, c.get("reddit") or {}
    pct = lambda v: None if v is None else round(v * 100, 2)
    return {
        "ret_5d_pct": pct(d.get("ret_5d")), "ret_20d_pct": pct(d.get("ret_20d")),
        "ret_60d_pct": pct(d.get("ret_60d")), "mom_adj_20d": round(d["mom_adj_20d"], 2),
        "rvol_5d": round(d["rvol_5d"], 2) if d.get("rvol_5d") is not None else None,
        "atr_pct": round(d["atr_pct"], 2), "near_high": round(d["near_high"], 3),
        "above_sma20": bool(d["above_sma20"]),
        "chg_today_pct": round(t["chg_pct"], 2) if t.get("chg_pct") is not None else None,
        "rvol_today": round(t["rvol_today"], 2) if t.get("rvol_today") is not None else None,
        "headlines": int(n.get("n") or 0),
        "tone": round(n["tone"], 3) if n.get("tone") is not None else None,
        "headline": n.get("headline"),
        "reddit_mentions": rd.get("mentions"), "reddit_mentions_24h_ago": rd.get("mentions_24h_ago"),
        "stocktwits_rank": c.get("stocktwits_rank"),
        "dollar_volume_m": round(float(c.get("dollar_volume") or 0) / 1e6, 1),
    }


def _reasons(c: Dict[str, Any], comps: Dict[str, float]) -> List[str]:
    d, t, n, rd = c["daily"], c.get("today") or {}, c.get("news") or {}, c.get("reddit")
    out = [f"20d {d['ret_20d'] * 100:+.1f}% (risk-adj {d['mom_adj_20d']:+.1f}), 5d {d['ret_5d'] * 100:+.1f}%"
           + (", above 20d avg" if d["above_sma20"] else ", below 20d avg")]
    if d.get("rvol_5d") and d["rvol_5d"] >= 1.5:
        out.append(f"volume {d['rvol_5d']:.1f}x its 3-month norm this week")
    if t.get("chg_pct") is not None:
        out.append(("today " if t.get("live_session") else "pre-market ") + f"{t['chg_pct']:+.1f}%"
                   + (f" on {t['rvol_today']:.1f}x volume pace" if t.get("rvol_today") else ""))
    h = c.get("home")
    if h:
        out.append(f"hot in {h['exchange_label']} as {h['home_symbol']}: #{h['activity_rank']} of {h['of']} "
                   f"by value traded, {h['change_pct']:+.1f}% there")
    if n.get("n"):
        tone = n.get("tone")
        out.append(f"{n['n']} headline(s)" + (f", tone {'+' if tone >= 0.5 else ''}{(tone - 0.5) * 200:.0f}"
                                               if tone is not None else ""))
    if rd:
        out.append(f"Reddit {rd.get('mentions', 0)} mentions (24h ago {rd.get('mentions_24h_ago', 0)})")
    if c.get("stocktwits_rank"):
        out.append(f"StockTwits trending #{c['stocktwits_rank']}")
    return out
