"""
The thought stream: what the dashboard shows the agents thinking per stock.

  * a scout pick's view comes from the watcher: its confidence is the trade
    probability, its parts are the drivers, and their weighted sign is the lean
  * a news-driven stock's view comes from its headlines and the trend
  * events fire when the status changes, the lean flips, or the probability
    crosses the bar -- not on every cycle
"""
import time

import pytest

from core.config import settings
from core.state import state, SentimentRecord
from engine.thoughts import Thoughts
import scout.service as svc
import scout.watcher as wmod


@pytest.fixture
def setup(monkeypatch):
    sc, w = svc.ScoutAgent(), wmod.WatcherAgent()
    monkeypatch.setattr(svc, "scout", sc)
    monkeypatch.setattr(wmod, "watcher", w)
    monkeypatch.setattr(state, "last_gate_detail", {})
    sc.picks["NVDA"] = {"rank": 2, "score": 0.8, "country": "US"}
    return sc, w


def _read(w, conf, status="watching", news=0.5, trend=0.8):
    w.reads["NVDA"] = {"symbol": "NVDA", "at": time.time(), "confidence": conf, "status": status, "blocks": [],
                       "reasons": ["trend uptrend (+0.60)"],
                       "parts": {"scout": 0.8, "trend": trend, "vwap": 0.7, "drive": 0.6, "news": news, "market": 0.5}}


def test_scout_pick_view_and_events(setup):
    _, w = setup
    t = Thoughts()
    _read(w, 0.60)
    t.update([{"symbol": "NVDA", "status": ""}])
    v = t.views["NVDA"]
    assert v["strategy"] == "scout" and v["prob"] == 0.6 and v["bar"] == settings.SCOUT_ENTRY_CONFIDENCE
    assert v["lean"] == "bullish" and "for: scout" in v["thought"]
    assert {d["name"] for d in v["drivers"]} == {"scout", "trend", "vwap", "drive", "news", "market"}
    assert not t.events                                              # first sight is not an event

    _read(w, 0.70, status="confirming 10/90s")
    t.update([{"symbol": "NVDA", "status": ""}])
    ev = t.events[-1]["text"]
    assert "confirming" in ev and "crossed above the bar" in ev

    _read(w, 0.30, status="watching", news=0.0, trend=0.0)            # bad news, trend rolls over
    w.reads["NVDA"]["parts"].update(scout=0.5, vwap=0.2, drive=0.2)
    t._evented.clear()                                                # past the 5 s per-stock event limit
    t.update([{"symbol": "NVDA", "status": ""}])
    assert t.views["NVDA"]["lean"] == "bearish" and "against: trend" in t.views["NVDA"]["thought"]
    assert "turned bearish" in t.events[-1]["text"]
    assert t.snapshot()["views"][0]["history"]


def test_news_driven_view(setup, monkeypatch):
    t = Thoughts()
    rec = SentimentRecord(stock_id="MSFT", pos_prob=0.8, neg_prob=0.1, n_headlines=3, agreement=0.9)
    monkeypatch.setattr(state, "get_sentiment", lambda sym, *a, **k: rec)
    state.last_gate_detail["MSFT"] = {"strategy": "news_catalyst", "buy_prob": 0.61, "at": time.time(),
                                      "reason": "News catalyst: pos=0.80 across 3 headlines"}
    t.update([{"symbol": "MSFT", "status": "no entry"}])
    v = t.views["MSFT"]
    assert v["strategy"] == "news_catalyst" and v["prob"] == 0.61
    assert v["bar"] == state.risk_profile.min_buy_prob
    news = next(d for d in v["drivers"] if d["name"] == "news")
    assert news["value"] > 0.5 and v["lean"] == "bullish" and "News catalyst" in v["thought"]


def test_dropped_symbols_leave_the_view(setup):
    _, w = setup
    t = Thoughts()
    _read(w, 0.6)
    t.update([{"symbol": "NVDA", "status": ""}])
    t.update([])
    assert not t.views and not t.snapshot()["views"]
