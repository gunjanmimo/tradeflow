"""
Scout: our own discovery, the watcher, and the scout strategy.

  * features read past-window performance from daily bars and today's move from
    a snapshot; a stock with no trade today has no "today" component
  * eligibility keeps liquid US common stocks: no ETFs (whatever the issuer
    calls them), warrants, leveraged products, penny or thinly traded names
  * the ranking rewards performance, today's move, bullish news and discussion;
    bad news pulls a stock down
  * picks go on the watchlist; ranks shuffling cause no churn; a dropped pick
    leaves the watchlist only if the scout put it there; a held one stays; one
    the user removed is not picked again that day; every ranking is logged
  * the watcher blocks outside the session, early in it, in a downtrend or on
    bad news, and is READY only once confidence has held for the confirm window
  * the scout strategy trades only READY picks, once a day, and exits when the
    confidence collapses after the minimum hold
  * the scorecard and backtest measure the top of the list against the pool
"""
import asyncio
import json
import time
import types

import numpy as np
import pytest

from core.config import settings
from core.state import state, QuantMetrics, SentimentRecord, PriceTick
from scout import features, ranker, evaluate
import scout.service as svc
import scout.watcher as wmod


def _series(n=120, drift=0.0, start=100.0, vol=1e6, seed=0):
    rng = np.random.default_rng(seed)
    c = start * np.exp(np.cumsum(drift + rng.normal(0, 0.01, n)))
    return c, c * 1.01, c * 0.99, np.full(n, vol)


def _cand(sym, drift=0.0, news=None, reddit=None, st=None, today=None, name=None, dv=5e8, price=None):
    c, h, lo, v = _series(drift=drift, seed=sum(map(ord, sym)))
    if price is not None:
        c, h, lo = c / c[-1] * price, h / c[-1] * price, lo / c[-1] * price
    return {"symbol": sym, "name": name or f"{sym} Inc. Common Stock", "origins": ["test"],
            "daily": features.daily_features(c, h, lo, v), "dollar_volume": dv, "today": today,
            "listed": True, "news": news or {"n": 0}, "reddit": reddit, "stocktwits_rank": st}


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

def test_daily_features():
    c, h, lo, v = _series(drift=0.004)
    f = features.daily_features(c, h, lo, v)
    assert f["ret_20d"] > 0 and f["mom_adj_20d"] > 0 and f["sma20_gt_50"] == 1.0
    assert 0.5 < f["atr_pct"] < 5 and abs(f["rvol_5d"] - 1.0) < 1e-9
    v2 = v.copy()
    v2[-5:] *= 3
    assert features.daily_features(c, h, lo, v2)["rvol_5d"] == pytest.approx(3.0)
    assert features.daily_features(c[:40]) is None                     # too little history


def test_today_features():
    live = {"price": 110.0, "price_date": "2026-09-29", "day_date": "2026-09-29", "day_close": 110.0,
            "day_volume": 500.0, "prev_close": 100.0, "prev_volume": 1000.0}
    t = features.today_features(live, "2026-09-29", 0.5)
    assert t["chg_pct"] == pytest.approx(10.0) and t["rvol_today"] == pytest.approx(1.0) and t["live_session"]
    pre = {**live, "day_date": "2026-09-28", "day_close": 100.0, "price": 102.0}
    assert features.today_features(pre, "2026-09-29", None)["chg_pct"] == pytest.approx(2.0)
    stale = {**pre, "price_date": "2026-09-28"}                          # no print today yet
    assert features.today_features(stale, "2026-09-29", None) is None


# ---------------------------------------------------------------------------
# eligibility and ranking
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "ProShares UltraPro QQQ", "ProShares Ultra Bloomberg Crude Oil", "iShares Bitcoin Trust ETF Shares",
    "Invesco QQQ Trust, Series 1", "Direxion Daily Semiconductor Bull 3X ETF", "ARK Innovation ETF",
    "Acme Corp Warrants", "SPDR Gold Trust, SPDR Gold Shares"])
def test_funds_and_warrants_are_not_stocks(name):
    assert "not a common stock" in ranker.eligibility("XYZ", name, {"price": 50}, 1e9, True, 5, 2e7)


def test_eligibility():
    ok = {"price": 50.0}
    for name in ("The Charles Schwab Corporation", "Invesco LTD", "Kodiak Sciences Inc Common Stock"):
        assert ranker.eligibility("XYZ", name, ok, 1e9, True, 5, 2e7) is None
    assert ranker.eligibility("BTCUSD", "", ok, 1e9, None, 5, 2e7) == "not a US stock ticker"
    assert ranker.eligibility("XYZ", "", ok, 1e9, False, 5, 2e7) == "not tradable on Alpaca"
    assert "price" in ranker.eligibility("XYZ", "", {"price": 2.0}, 1e9, True, 5, 2e7)
    assert "illiquid" in ranker.eligibility("XYZ", "", ok, 1e6, True, 5, 2e7)
    assert "history" in ranker.eligibility("XYZ", "", None, 1e9, True, 5, 2e7)


def test_ranking_rewards_performance_news_and_discussion():
    cands = [_cand("F" + "ABCDEFGH"[i], drift=0.0) for i in range(8)]
    cands += [_cand("HOT", drift=0.006, news={"n": 6, "tone": 0.85, "n_scored": 3, "headline": "beats"},
                    reddit={"mentions": 120, "mentions_24h_ago": 20}, st=2),
              _cand("BAD", drift=0.006, news={"n": 6, "tone": 0.15, "n_scored": 3, "headline": "probe"}),
              _cand("ETF", drift=0.01, name="ProShares UltraPro QQQ"),
              _cand("PENY", drift=0.01, price=2.0)]
    out = ranker.rank(cands, 5.0, 2e7)
    order = [r["symbol"] for r in out["ranked"]]
    assert order[0] == "HOT"
    assert order.index("HOT") < order.index("BAD")
    assert {e["symbol"] for e in out["excluded"]} == {"ETF", "PENY"}
    hot = out["ranked"][0]
    assert "today" not in hot["components"]                      # no trade today: does not vote
    assert any("Reddit" in r for r in hot["reasons"]) and hot["rank"] == 1


def test_today_votes_when_present():
    flat = [_cand("F" + "ABCDEFGH"[i], today={"chg_pct": 0.0, "rvol_today": 1.0, "price": 100.0}) for i in range(6)]
    mover = _cand("MOVE", today={"chg_pct": 6.0, "rvol_today": 3.0, "price": 100.0})
    out = ranker.rank(flat + [mover], 5.0, 2e7)
    today = {r["symbol"]: r["components"]["today"] for r in out["ranked"]}
    assert today["MOVE"] == 1.0 and max(v for k, v in today.items() if k != "MOVE") < 0.6


# ---------------------------------------------------------------------------
# the Scout agent: picks and the watchlist
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh(monkeypatch):
    sc, w = svc.ScoutAgent(), wmod.WatcherAgent()
    monkeypatch.setattr(svc, "scout", sc)
    monkeypatch.setattr(wmod, "watcher", w)
    monkeypatch.setattr(state, "watchlist", {"AAPL"})
    monkeypatch.setattr(state, "active_positions", {})
    from core.market_filter import market_filter
    monkeypatch.setattr(market_filter, "entry_block_reason", lambda s: None)
    return sc, w


def _rows(symbols, score=0.7):
    return [{"symbol": s, "rank": i + 1, "score": score - i * 0.001, "price": 100.0, "reasons": ["r"],
             "components": {}, "origins": [], "features": {"atr_pct": 2.0}} for i, s in enumerate(symbols)]


def test_picks_go_on_the_watchlist_and_churn_is_damped(fresh, monkeypatch):
    sc, _ = fresh
    monkeypatch.setattr(settings, "SCOUT_TOP_N", 3)
    monkeypatch.setattr(settings, "SCOUT_KEEP_RANK", 5)
    sc._select(_rows(["AAPL", "NVDA", "MU", "X1", "X2", "X3"]))
    assert set(sc.picks) == {"AAPL", "NVDA", "MU"}
    assert {"NVDA", "MU"} <= state.watchlist and sc.added == {"NVDA", "MU"}
    # NVDA slips to #4: inside the keep band, stays a pick.
    sc._select(_rows(["AAPL", "MU", "X1", "NVDA", "X2", "X3"]))
    assert "NVDA" in sc.picks and "X1" in sc.picks
    # Everyone falls out; MU is held, AAPL was the user's.
    state.active_positions["MU"] = {"qty": 1}
    sc._select(_rows(["Y1", "Y2", "Y3", "Y4", "Y5", "Y6", "AAPL", "MU", "NVDA", "X1"]))
    assert "NVDA" not in sc.picks and "NVDA" not in state.watchlist          # scout-added: removed
    assert "AAPL" not in sc.picks and "AAPL" in state.watchlist               # user's: stays watched
    assert "MU" in sc.picks and "MU" in state.watchlist                       # held: stays


def test_a_removed_pick_is_not_picked_again_today(fresh):
    sc, _ = fresh
    sc._select(_rows(["NVDA"]))
    state.watchlist.discard("NVDA")
    sc.on_unwatched("NVDA")
    sc._select(_rows(["NVDA"]))
    assert "NVDA" not in sc.picks and "NVDA" not in state.watchlist


def test_low_scores_are_not_picked(fresh):
    sc, _ = fresh
    sc._select(_rows(["NVDA"], score=settings.SCOUT_MIN_SCORE - 0.01))
    assert not sc.picks


def test_rank_now_end_to_end(fresh, monkeypatch):
    from feeds.daily_bars import daily_bars
    from scout.sources import sources
    sc, _ = fresh

    async def screener():
        return {"NVDA": ["Most active"], "TQQQ": ["Most active"], "KOD": ["Top gainer"]}

    async def news():
        return {"NVDA": [{"id": "1", "headline": "Nvidia beats", "summary": "", "at": time.time()}]}

    async def reddit():
        return {"NVDA": {"mentions": 150, "mentions_24h_ago": 30, "upvotes": 10, "rank": 1}}

    async def stocktwits():
        return {"KOD": 1}

    async def snapshots(symbols):
        return {s: {"price": 100.0, "price_date": None, "day_date": "2000-01-01", "day_close": 100.0,
                    "day_volume": 0.0, "prev_close": 99.0, "prev_volume": 1.0} for s in symbols}

    async def ensure(symbols, max_age_s=0):
        return 0

    for name, fn in (("screener", screener), ("news", news), ("reddit", reddit),
                     ("stocktwits", stocktwits), ("snapshots", snapshots)):
        monkeypatch.setattr(sources, name, fn)
    monkeypatch.setattr(sources, "asset_names", {"NVDA": "NVIDIA Corporation Common Stock",
                                                 "TQQQ": "ProShares UltraPro QQQ",
                                                 "KOD": "Kodiak Sciences Inc Common Stock",
                                                 "AAPL": "Apple Inc. Common Stock",
                                                 "BABA": "Alibaba Group Holding Limited American Depositary Shares",
                                                 "TCEHY": "Tencent Holdings Limited Unsponsored ADR"})
    monkeypatch.setattr(daily_bars, "ensure", ensure)
    series, hlv, dv = {}, {}, {}
    for s, d in (("NVDA", 0.004), ("TQQQ", 0.006), ("KOD", 0.002), ("AAPL", 0.0), ("BABA", 0.001)):
        c, h, lo, v = _series(drift=d, seed=len(s))
        series[s], hlv[s], dv[s] = (np.arange(len(c)), c), (h, lo, v), 5e8
    monkeypatch.setattr(daily_bars, "series", series)
    monkeypatch.setattr(daily_bars, "hlv", hlv)
    monkeypatch.setattr(daily_bars, "dollar_volume", dv)
    monkeypatch.setattr(svc.ScoutAgent, "_score_tone", lambda self, *a: asyncio.sleep(0))
    import scout.exchanges as ex

    async def listen():
        ex.exchanges.boards = {"hongkong": {"market": "hongkong", "label": "Hong Kong", "country": "Hong Kong",
                                            "at": time.time(), "rows": [
            {"name": "9988", "description": "Alibaba Group Holding Limited", "change": 3.0, "Value.Traded": 9e9},
            {"name": "700", "description": "Tencent Holdings Ltd", "change": -1.0, "Value.Traded": 8e9}]}}
        return ex.exchanges.boards

    async def assets():
        return sources.asset_names

    monkeypatch.setattr(ex.exchanges, "listen", listen)
    monkeypatch.setattr(sources, "assets", assets)
    monkeypatch.setattr(sources, "asset_exchange", {"NVDA": "NASDAQ", "TQQQ": "NASDAQ", "KOD": "NASDAQ",
                                                     "AAPL": "NASDAQ", "BABA": "NYSE", "TCEHY": "OTC"})

    rows = asyncio.run(sc.rank_now())
    syms = [r["symbol"] for r in rows]
    assert syms[0] == "NVDA" and "TQQQ" not in syms
    assert any(e["symbol"] == "TQQQ" for e in sc.excluded)
    baba = next(r for r in rows if r["symbol"] == "BABA")                  # heard on Hong Kong's board
    assert baba["country"] == "Hong Kong" and baba["region"] == "Asia" and baba["home_exchange"] == "Hong Kong"
    assert "home" in baba["components"] and any("hot in Hong Kong" in x for x in baba["reasons"])
    board = ex.exchanges.snapshot(sc.picks, rows)
    hk = next(b for b in board if b["label"] == "Hong Kong")
    assert [r["us_symbol"] for r in hk["rows"]] == ["BABA", None]
    assert "OTC" in hk["rows"][1]["why_not"]                                   # Tencent: OTC-only ADR
    logged = [json.loads(l) for l in open(svc._PATH)]
    assert logged[-1]["top"][0]["symbol"] == "NVDA" and "NVDA" in logged[-1]["pool_prices"]


# ---------------------------------------------------------------------------
# the watcher
# ---------------------------------------------------------------------------

def _trend(direction=0.6, label="uptrend", ready=True, reversal_down=False):
    return types.SimpleNamespace(direction=direction, label=label, ready=ready, reversal_down=reversal_down)


def _live(**kw):
    base = dict(scout_score=0.75, price=102.0, trend=_trend(), levels={"vwap": 101.0, "open": 100.0},
                atr_pct=2.0, sentiment=None, market_direction=0.3, rsi=60.0, spread=0.0005,
                session_open=True, minutes_since_open=60.0)
    base.update(kw)
    return wmod.score_live(**base)


def test_score_live_confidence_and_blocks():
    good = _live()
    assert good["confidence"] >= settings.SCOUT_ENTRY_CONFIDENCE and not good["blocks"]
    weak = _live(trend=_trend(-0.1, "range"), levels={"vwap": 103.0, "open": 104.0})
    assert weak["confidence"] < good["confidence"] - 0.15
    assert "regular session closed" in _live(session_open=False)["blocks"]
    assert any("first" in b for b in _live(minutes_since_open=5.0)["blocks"])
    assert "in a downtrend" in _live(trend=_trend(-0.5, "downtrend"))["blocks"]
    assert "trend still learning" in _live(trend=_trend(ready=False))["blocks"]
    assert any("RSI" in b for b in _live(rsi=80.0)["blocks"])
    bad = SentimentRecord(stock_id="X", pos_prob=0.1, neg_prob=0.8, n_headlines=3, agreement=0.9)
    out = _live(sentiment=bad)
    assert any("bad news" in b for b in out["blocks"]) and out["parts"]["news"] < 0.3


def _watch(monkeypatch, sc, w, sym="NVDA", conf_inputs=None):
    import core.market_hours as mh
    sc.picks[sym] = {"rank": 1, "score": 0.75, "pick_price": 100.0, "picked_at": time.time(),
                     "row": {"features": {"atr_pct": 2.0}}}
    state.latest_prices[sym] = PriceTick(symbol=sym, price=102.0, bid=101.99, ask=102.01, volume=0,
                                         timestamp=time.time())
    monkeypatch.setattr(mh, "us_session", lambda *a, **k: mh.REGULAR)
    monkeypatch.setattr(wmod, "score_live", lambda **kw: conf_inputs["out"])


def test_watcher_is_ready_only_after_the_confirm_window(fresh, monkeypatch):
    sc, w = fresh
    box = {"out": {"confidence": 0.7, "parts": {}, "blocks": [], "reasons": []}}
    _watch(monkeypatch, sc, w, conf_inputs=box)
    t0 = time.time()
    assert w.assess("NVDA", t0)["status"].startswith("confirming")
    assert w.assess("NVDA", t0 + settings.SCOUT_CONFIRM_SECONDS / 2)["status"].startswith("confirming")
    box["out"] = {"confidence": 0.6, "parts": {}, "blocks": [], "reasons": []}          # dips: resets
    assert w.assess("NVDA", t0 + settings.SCOUT_CONFIRM_SECONDS - 1)["status"] == "watching"
    box["out"] = {"confidence": 0.7, "parts": {}, "blocks": [], "reasons": []}
    t1 = t0 + settings.SCOUT_CONFIRM_SECONDS
    w.assess("NVDA", t1)
    r = w.assess("NVDA", t1 + settings.SCOUT_CONFIRM_SECONDS)
    assert r["status"] == "ready" and r["chg_since_pick_pct"] == pytest.approx(2.0)
    box["out"] = {"confidence": 0.9, "parts": {}, "blocks": ["in a downtrend"], "reasons": []}
    assert w.assess("NVDA", t1 + 2 * settings.SCOUT_CONFIRM_SECONDS)["status"] == "blocked: in a downtrend"
    state.latest_prices.pop("NVDA", None)


# ---------------------------------------------------------------------------
# routing and the scout strategy
# ---------------------------------------------------------------------------

def _ctx(symbol="NVDA", position=None, price=102.0):
    from engine.strategies.base import StrategyContext
    q = QuantMetrics(symbol=symbol, rsi=55.0, ema_fast=101.0, ema_slow=100.0, atr=0.5, spread=0.0005)
    return StrategyContext(symbol=symbol, price=price, quant=q, sentiment=SentimentRecord(stock_id=symbol),
                           consensus=None, position=position)


def test_routing(fresh, monkeypatch):
    from engine.strategies import registry
    sc, _ = fresh
    sc.picks["NVDA"] = {"rank": 1, "score": 0.7}
    defaults = {"equity": "news_catalyst"}
    assert registry.resolve("NVDA", defaults, {}).name == "scout"
    assert registry.resolve("MSFT", defaults, {}).name == "news_catalyst"
    assert registry.resolve("NVDA", defaults, {"NVDA": "supertrend"}).name == "supertrend"
    monkeypatch.setattr(settings, "SCOUT_ENABLED", False)
    assert registry.resolve("NVDA", defaults, {}).name == "news_catalyst"


def _read(w, sym, status, conf, blocks=(), vwap=101.0):
    w.reads[sym] = {"symbol": sym, "at": time.time(), "rank": 1, "scout_score": 0.7, "confidence": conf,
                    "parts": {}, "blocks": list(blocks), "reasons": ["r"], "status": status,
                    "confirmed_for_s": 100.0, "vwap": vwap}


def test_strategy_enters_only_when_ready_and_once_a_day(fresh, monkeypatch):
    from engine.strategies.scout import ScoutStrategy
    _, w = fresh
    s = ScoutStrategy()
    assert s.evaluate_entry(_ctx()).blocked_by == "no_watch"
    _read(w, "NVDA", "confirming 30/90s", 0.7)
    assert not s.evaluate_entry(_ctx()).should_enter
    _read(w, "NVDA", "watching", 0.5)
    assert s.evaluate_entry(_ctx()).blocked_by == "not_confident"
    _read(w, "NVDA", "blocked: in a downtrend", 0.8, blocks=["in a downtrend"])
    assert s.evaluate_entry(_ctx()).blocked_by == "watcher_block"
    _read(w, "NVDA", "ready", 0.72)
    d = s.evaluate_entry(_ctx())
    assert d.should_enter and d.buy_prob == pytest.approx(0.72)
    # A buy fill today -- as reloaded from the broker after a restart -- counts.
    state.recent_trades.append({"symbol": "NVDA", "side": "BUY", "qty": 5, "price": 100.0, "time": time.time()})
    try:
        assert s.evaluate_entry(_ctx()).blocked_by == "once_per_day"
    finally:
        state.recent_trades.pop()
    monkeypatch.setattr(settings, "SCOUT_TRADING", False)
    assert s.evaluate_entry(_ctx()).blocked_by == "scout_off"
    _read(w, "NVDA", "ready", 0.72)
    w.reads["NVDA"]["at"] = time.time() - 600                                   # stale read
    monkeypatch.setattr(settings, "SCOUT_TRADING", True)
    assert s.evaluate_entry(_ctx()).blocked_by == "no_watch"


def test_strategy_exits_when_confidence_collapses(fresh):
    from engine.strategies.scout import ScoutStrategy
    _, w = fresh
    s = ScoutStrategy()
    young = {"qty": 1, "opened_at": time.time() - 60, "entry_strategy": "scout"}
    old = {**young, "opened_at": time.time() - 3600}
    _read(w, "NVDA", "watching", 0.3)
    assert not s.evaluate_exit(_ctx(position=young)).should_close           # minimum hold
    assert s.evaluate_exit(_ctx(position=old)).should_close
    _read(w, "NVDA", "blocked: in a downtrend", 0.5, blocks=["in a downtrend"], vwap=103.0)
    assert s.evaluate_exit(_ctx(position=old, price=102.0)).should_close     # downtrend under VWAP
    _read(w, "NVDA", "watching", 0.6)
    assert not s.evaluate_exit(_ctx(position=old)).should_close


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def test_scorecard_measures_picks_against_the_pool():
    day = evaluate._day_number("2026-09-28")
    recs = [{"ny_date": "2026-09-28", "top": [{"symbol": f"W{i}"} for i in range(3)],
             "pool_prices": {**{f"W{i}": 100.0 for i in range(3)}, **{f"L{i}": 100.0 for i in range(6)}}}]
    closes = {**{f"W{i}": {day: 102.0, day + 1: 103.0} for i in range(3)},
              **{f"L{i}": {day: 99.0, day + 1: 98.0} for i in range(6)}}
    out = evaluate.scorecard(recs, closes, top_n=3)
    assert out["to_close"]["days"] == 1
    assert out["to_close"]["top_bps"] == pytest.approx(200.0)
    assert out["to_close"]["excess_bps"] == pytest.approx(200.0 - (3 * 200 - 6 * 100) / 9)
    assert out["next_close"]["top_bps"] == pytest.approx(300.0)


def test_backtest_finds_a_planted_edge():
    rng = np.random.default_rng(1)
    n, bars = 160, {}
    for k in range(40):
        drift = 0.003 if k < 10 else 0.0            # the strong-trend names keep rising
        c = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.01, n)))
        o = np.r_[c[0], c[:-1]] * (1 + rng.normal(0, 0.001, n))
        bars[f"S{k}"] = {"day": np.arange(n), "open": o, "high": np.maximum(o, c) * 1.005,
                         "low": np.minimum(o, c) * 0.995, "close": c, "volume": np.full(n, 1e6)}
    out = evaluate.backtest(bars, top_n=5)
    cc = out["hold_close_to_close"]
    assert cc["days"] > 50 and cc["excess_bps"] > 0


# ---------------------------------------------------------------------------
# the world's exchanges
# ---------------------------------------------------------------------------

def test_name_mapping_to_us_lines():
    from scout.exchanges import NameIndex, normalize_name
    names = {"HSBC": "HSBC Holdings PLC", "SAP": "SAP SE", "BTI": "British American Tobacco Industries p.l.c. ADS",
             "MRK": "Merck & Co., Inc.", "MUFG": "Mitsubishi UFJ Financial Group, Inc.", "SIEGY": "SIEMENS AG SPONSORED ADR (Germany)",
             "BSEX": "BSE Holdings Inc. Common Stock"}
    exch = {s: "NYSE" for s in names} | {"SIEGY": "OTC"}
    known = {"MRK": False, "MUFG": True}
    idx = NameIndex(names, exch, lambda s: known.get(s))
    assert normalize_name("HSBC Holdings Plc") == ("hsbc",)
    assert idx.match("HSBC Holdings Plc")[0] == "HSBC"
    assert idx.match("SAP SE")[0] == "SAP"
    assert idx.match("British American Tobacco p.l.c.")[0] == "BTI"             # words contained
    assert idx.match("Mitsubishi UFJ Financial Group, Inc.")[0] == "MUFG"        # SEC: Japanese issuer
    sym, why = idx.match("Merck KGaA")
    assert sym is None and "US company" in why                                   # not Merck & Co
    sym, why = idx.match("Siemens AG")
    assert sym is None and "OTC" in why
    assert idx.match("BSE Ltd.")[0] is None and idx.match("Reliance Industries Limited") == (None, "no US listing")


def test_one_pick_per_exchange(fresh, monkeypatch):
    sc, _ = fresh
    monkeypatch.setattr(settings, "SCOUT_TOP_N", 2)
    monkeypatch.setattr(settings, "SCOUT_EXCHANGE_SLOTS", 1)
    monkeypatch.setattr(settings, "SCOUT_INTL_MAX_PICKS", 2)
    rows = _rows(["US1", "US2", "US3", "HK1", "HK2", "LN1", "IN1"], score=0.7)
    for r, ex in zip(rows[3:], ["Hong Kong", "Hong Kong", "London", "India NSE"]):
        r["home_exchange"], r["region"] = ex, "Asia" if ex != "London" else "Europe/UK"
    rows[6]["score"] = 0.51                                                       # India's best is weakest
    sc._select(rows)
    assert set(sc.picks) == {"US1", "US2", "HK1", "LN1"}                          # at most 2 foreign
    assert sc.picks["HK1"]["slot"] == "Hong Kong slot"


def test_a_pick_without_a_price_does_not_crash_the_strategy(fresh, monkeypatch):
    """The watcher's no-price read once lacked fields and the KeyError aborted every manager cycle."""
    import core.market_hours as mh
    from engine.strategies.scout import ScoutStrategy
    sc, w = fresh
    sc.picks["NVDA"] = {"rank": 1, "score": 0.7, "pick_price": 100.0}
    state.latest_prices.pop("NVDA", None)
    monkeypatch.setattr(mh, "us_session", lambda *a, **k: mh.REGULAR)
    w.reads["NVDA"] = w.assess("NVDA", time.time())
    assert w.reads["NVDA"]["status"] == "no price"
    d = ScoutStrategy().evaluate_entry(_ctx())
    assert not d.should_enter and d.blocked_by == "not_confident"
    assert not ScoutStrategy().evaluate_exit(_ctx(position={"opened_at": time.time() - 3600})).should_close
