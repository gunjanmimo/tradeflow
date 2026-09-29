"""
What the news pipeline did, item by item, for the dashboard's News ingestion
box: every headline ingested, the entities and events found in it, which
symbols it was scored for, by which backend, and the score.

Two producers:
  news feed   headlines for the watchlist and tracked candidates, polled every
              NEWS_POLL_SECONDS and scored per symbol (these drive trading)
  scout       the hourly market-wide pull; tone is scored for the shortlist only
"""
import time
from collections import Counter, deque
from typing import Any, Dict, Iterable, List, Optional

from feeds.news_entities import extract

MAX_ITEMS = 300


class NewsLog:
    def __init__(self):
        self.items: deque = deque(maxlen=MAX_ITEMS)
        self._by_id: Dict[str, Dict[str, Any]] = {}
        self.counts: Counter = Counter()
        self.version = 0

    def _names(self) -> Dict[str, str]:
        from engine.executor import executor
        from scout.sources import sources
        return executor.alpaca_asset_names or sources.asset_names

    def ingest(self, news_id: str, headline: str, summary: str, source: str, symbols: Iterable[str],
               published_at: float, origin: str, tracked: Iterable[str] = ()) -> Dict[str, Any]:
        """Records a headline (once per id and origin) with its entities; returns the row."""
        key = f"{origin}:{news_id}"
        row = self._by_id.get(key)
        if row is not None:
            return row
        syms = [s.upper() for s in symbols]
        ner = extract(f"{headline}. {summary}" if summary else headline, syms, self._names())
        row = {"id": news_id, "origin": origin, "at": time.time(), "published_at": published_at,
               "source": source, "headline": headline[:240], "symbols": syms, "tracked": sorted(tracked),
               "entities": ner["entities"], "events": ner["events"], "lean": ner["lean"],
               "scores": {}, "status": "extracted"}
        if len(self.items) == self.items.maxlen:
            old = self.items[0]
            self._by_id.pop(f"{old['origin']}:{old['id']}", None)
        self.items.append(row)
        self._by_id[key] = row
        self.counts["ingested"] += 1
        for e in ner["events"]:
            self.counts[f"event:{e['event']}"] += 1
        self.version += 1
        return row

    def scored(self, news_id: str, origin: str, symbol: str, pos: float, neg: float, backend: str,
               ms: Optional[float] = None):
        row = self._by_id.get(f"{origin}:{news_id}")
        if row is None:
            return
        row["scores"][symbol] = {"pos": round(pos, 3), "neg": round(neg, 3), "backend": backend,
                                 "ms": round(ms, 1) if ms is not None else None}
        row["status"] = "scored"
        self.counts["scored"] += 1
        self.version += 1

    def failed(self, news_id: str, origin: str, symbol: str, error: str):
        row = self._by_id.get(f"{origin}:{news_id}")
        if row is not None:
            row["scores"][symbol] = {"error": error[:120]}
            row["status"] = "error"
            self.counts["errors"] += 1
            self.version += 1

    def for_symbol(self, symbol: str, limit: int = 6) -> List[Dict[str, Any]]:
        """The newest rows mentioning a symbol (the trade desk's news brief)."""
        return [r for r in reversed(self.items) if symbol in r["symbols"]][:limit]

    def snapshot(self, limit: int = 40) -> Dict[str, Any]:
        from feeds.news_feed import news_feed
        from sentiment.router import sentiment_service
        now = time.time()
        recent = [r for r in self.items if now - r["at"] < 3600]
        ents = Counter(v for r in recent for kind in ("company", "org", "broker", "person", "place")
                       for v in r["entities"].get(kind, []))
        events = Counter(e["event"] for r in recent for e in r["events"])
        return {
            "version": self.version,
            "items": list(reversed(self.items))[:limit],
            "counts": dict(self.counts),
            "last_hour": {"ingested": len(recent),
                          "scored": sum(1 for r in recent if r["status"] == "scored"),
                          "top_entities": ents.most_common(10), "top_events": events.most_common(8)},
            "backend": sentiment_service.status(),
            "feed": news_feed.status(),
        }


news_log = NewsLog()
