"""Named entities and events in headlines, and the news log the dashboard shows."""
import pytest

from feeds.news_entities import extract, short_name


def test_company_names_are_shortened():
    assert short_name("NVIDIA Corporation Common Stock") == "NVIDIA"
    assert short_name("Wells Fargo & Company") == "Wells Fargo"
    assert short_name("TRONOX LIMITED CL A") == "TRONOX"


def test_entities():
    r = extract("Nvidia CEO Jensen Huang Over His Stance; FDA and China weigh in on $4.2 billion deal, shares +12%",
                ["NVDA"], {"NVDA": "NVIDIA Corporation Common Stock"})
    e = r["entities"]
    assert e["ticker"] == ["NVDA"] and e["company"] == ["NVIDIA"]
    assert e["person"] == ["Jensen Huang"]
    assert "FDA" in e["org"] and "China" in e["place"]
    assert e["money"] == ["$4.2 billion"] and e["percent"] == ["+12%"]
    assert extract("Amcor Elects Tom Long As Board Chairman", [])["entities"]["person"] == ["Tom Long"]
    assert extract("Buy $TSLA now (NASDAQ: AAPL)", [])["entities"]["ticker"] == ["TSLA", "AAPL"]


@pytest.mark.parametrize("text,event,lean", [
    ("CarMax Q2 EPS $1.16 Beats $0.73 Estimate", "earnings beat", "bullish"),
    ("Acme misses revenue estimates", "earnings miss", "bearish"),
    ("Baird Downgrades Ryder System to Neutral", "downgrade", "bearish"),
    ("BTIG Maintains Buy on Robinhood, Raises Price Target to $135", "price target raised", "bullish"),
    ("Acme announces $500M share offering", "share offering", "bearish"),
    ("FDA approves Acme's drug", "regulatory approval", "bullish"),
])
def test_events(text, event, lean):
    r = extract(text, [])
    assert event in [e["event"] for e in r["events"]] and r["lean"] == lean


def test_earnings_beat_supersedes_plain_earnings():
    events = [e["event"] for e in extract("Acme earnings beat estimates", [])["events"]]
    assert "earnings beat" in events and "earnings" not in events


def test_news_log_records_once_and_tracks_scores():
    from feeds.news_log import NewsLog
    log = NewsLog()
    row = log.ingest("1", "Acme beats estimates", "", "benzinga", ["ACME"], 0.0, "feed", ["ACME"])
    assert log.ingest("1", "Acme beats estimates", "", "benzinga", ["ACME"], 0.0, "feed", ["ACME"]) is row
    log.scored("1", "feed", "ACME", 0.8, 0.1, "laya", 12.3)
    assert row["status"] == "scored" and row["scores"]["ACME"]["backend"] == "laya"
    assert log.for_symbol("ACME")[0] is row and log.counts["ingested"] == 1
