"""
Listens to the world's stock exchanges: what each one is trading most today,
and which of those stocks the bots can actually trade.

  listen   For every market in SCOUT_EXCHANGES, the SCOUT_EXCHANGE_TOP stocks
           with the highest value traded in their latest session, with the
           session's % change (TradingView's public screener: an unofficial
           endpoint that fails closed -- a market that does not answer is
           skipped and shown as down).
  map      Alpaca executes US listings only, so each hot stock is matched by
           company name to a US-listed line in Alpaca's asset list: an ADR
           (HSBC Holdings -> HSBC), an ordinary-share listing (SAP SE -> SAP) or a
           New York registry share (ASML). A match that trades only over the
           counter has no prices on this data plan; no match means no US line.
           Both stay listed, with the reason, as "hot there, not tradable here".

Nothing is hard-coded: which stocks come through, and what they map to, is
whatever the exchanges traded and whatever Alpaca lists that day.
"""
import asyncio
import json
import logging
import re
import time
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.config import settings

logger = logging.getLogger("tradeflow.scout.exchanges")

SCAN_URL = "https://scanner.tradingview.com/{market}/scan"
_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) TradeFlow scout", "Content-Type": "application/json",
       "Accept": "application/json"}
COLUMNS = ["name", "description", "close", "change", "volume", "Value.Traded", "market_cap_basic",
           "sector", "currency", "exchange"]

# TradingView market -> our country name (core/universe.py regions).
MARKET_COUNTRY = {
    "uk": "UK", "germany": "Germany", "france": "France", "netherlands": "Netherlands",
    "switzerland": "Switzerland", "spain": "Spain", "italy": "Italy", "sweden": "Sweden",
    "denmark": "Denmark", "norway": "Norway", "finland": "Finland", "belgium": "Belgium",
    "hongkong": "Hong Kong", "china": "China", "india": "India", "japan": "Japan", "korea": "South Korea",
    "taiwan": "Taiwan", "singapore": "Singapore",
}

# Words that describe the share line or the legal form, not the company.
_DROP_AFTER = re.compile(r"\b(american depositary|depositary|sponsored|unsponsored|new york registry|"
                         r"ordinary shares?|common stock|class [a-c]\b|each representing|\bads\b|\badr\b|"
                         r"registered shares?|preferred|pref\b)", re.IGNORECASE)
_LEGAL = {"plc", "ag", "se", "sa", "nv", "ltd", "limited", "inc", "incorporated", "corp", "corporation",
          "co", "company", "spa", "asa", "ab", "oyj", "as", "kk", "kgaa", "the", "group", "holding",
          "holdings", "hldgs", "hldg", "bhd", "tbk", "pcl", "and", "of", "de", "cayman"}


def normalize_name(name: str) -> Tuple[str, ...]:
    """'HSBC Holdings Plc' and 'HSBC Holdings PLC American Depositary Shares' -> ('hsbc',)."""
    n = _DROP_AFTER.split(name or "", maxsplit=1)[0].lower()
    n = n.replace("&", " and ").replace("p.l.c", "plc").replace("n.v.", "nv").replace("s.a.", "sa")
    tokens = re.findall(r"[a-z0-9]+", n)
    return tuple(t for t in tokens if t not in _LEGAL)


# Case-sensitive for the short forms (SE, AG, SA, AB are words in lower case);
# the long ones in any case ("HSBC Holdings PLC", "Nokia Corporation Limited").
_FOREIGN_FORM = re.compile(r"\b(N\.V\.|NV|SE|AG|S\.A\.|SA|S\.p\.A\.|A/S|ASA|AB|Oyj|ADR|ADS)\b|"
                           r"(?i:\b(plc|p\.l\.c|limited|ltd\.?|new york registry|ordinary shares|depositary)\b)")


def _looks_foreign(us_name: str) -> bool:
    return bool(_FOREIGN_FORM.search(us_name or "")) and "Common Stock" not in (us_name or "")


class NameIndex:
    """Alpaca's US-listed assets by normalized company name."""

    def __init__(self, names: Dict[str, str], exchanges: Dict[str, str],
                 foreign: Optional[Callable[[str], Optional[bool]]] = None):
        self.names = names
        self.exchanges = exchanges
        # Is this US line's issuer foreign? True / False from SEC or curated data, None unknown.
        self.foreign = foreign or (lambda sym: None)
        self.unverified: set = set()
        self.by_key: Dict[Tuple[str, ...], List[str]] = {}
        self.by_first: Dict[str, List[Tuple[Tuple[str, ...], str]]] = {}
        for sym, name in names.items():
            key = normalize_name(name)
            if not key:
                continue
            self.by_key.setdefault(key, []).append(sym)
            self.by_first.setdefault(key[0], []).append((key, sym))

    def match(self, name: str) -> Tuple[Optional[str], str]:
        """(US symbol, how) for a foreign company name, or (None, why not)."""
        key = normalize_name(name)
        if not key:
            return None, "no usable name"
        found = list(self.by_key.get(key, []))
        how = "same name"
        # The same name is not the same company: Merck KGaA (Xetra) is not Merck & Co
        # (MRK). A match stands only if the US line reads as a foreign issuer's (ADR,
        # plc, N.V., Ordinary Shares...) or SEC data says its issuer is foreign.
        # Unknown ones are held back and reported as unverified until SEC answers.
        checked = []
        for s in found:
            if _looks_foreign(self.names.get(s, "")):
                checked.append(s)
                continue
            f = self.foreign(s)
            if f:
                checked.append(s)
            elif f is None and self.exchanges.get(s) != "OTC":
                self.unverified.add(s)
        if found and not checked:
            return None, ("same name as a US company" if not any(self.foreign(s) is None for s in found)
                          else "name matches a US line whose country is not verified yet")
        found = checked
        if not found:
            # Truncated or longer names ("British American Tobacco p.l" vs "British American
            # Tobacco Industries"): one name's words must all appear in the other's, a
            # one-word name never matches this way, and the US line must read as a foreign
            # company -- a "... Common Stock" is a US company that happens to share words.
            for k, sym in self.by_first.get(key[0], []):
                short, long_ = (key, k) if len(key) <= len(k) else (k, key)
                if len(short) >= 2 and set(short) <= set(long_) and _looks_foreign(self.names.get(sym, "")):
                    found.append(sym)
            how = "name words"
        if not found:
            return None, "no US listing"
        listed = [s for s in found if self.exchanges.get(s) and self.exchanges[s] != "OTC"]
        if not listed:
            return None, f"US line {found[0]} is OTC-only (no prices on this data plan)"
        if len(set(listed)) > 1:
            # Several lines: prefer the one whose normalized name is shortest (the primary line).
            listed.sort(key=lambda s: (len(s), s))
        return listed[0], how


class ExchangeListener:
    def __init__(self):
        self.health: Dict[str, Dict[str, Any]] = {}
        self.boards: Dict[str, Dict[str, Any]] = {}     # market -> {label, country, rows, at}
        self.unverified: set = set()

    @staticmethod
    def markets() -> List[Tuple[str, str]]:
        """SCOUT_EXCHANGES ("uk:London,hongkong:Hong Kong,...") as [(market, label)]."""
        out = []
        for part in (settings.SCOUT_EXCHANGES or "").split(","):
            market, _, label = part.partition(":")
            if market.strip() in MARKET_COUNTRY:
                out.append((market.strip(), label.strip() or market.strip()))
        return out

    @staticmethod
    def _scan_sync(market: str, top: int) -> List[Dict[str, Any]]:
        body = {"columns": COLUMNS,
                # Secondary listings count: Alibaba's primary line is New York, but
                # Hong Kong trading it heavily is exactly the signal wanted.
                "filter": [{"left": "type", "operation": "equal", "right": "stock"}],
                "sort": {"sortBy": "Value.Traded", "sortOrder": "desc"}, "range": [0, top]}
        req = urllib.request.Request(SCAN_URL.format(market=market), data=json.dumps(body).encode(), headers=_UA)
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.load(r)
        return [dict(zip(COLUMNS, x["d"])) for x in data.get("data", [])]

    async def listen(self) -> Dict[str, Dict[str, Any]]:
        """Scans every configured exchange concurrently; returns the boards that answered."""
        loop = asyncio.get_running_loop()
        markets = self.markets()

        async def one(market: str, label: str):
            try:
                rows = await loop.run_in_executor(None, self._scan_sync, market, settings.SCOUT_EXCHANGE_TOP)
            except Exception as e:
                self.health[label] = {"ok": False, "detail": f"{type(e).__name__}: {str(e)[:120]}", "count": 0,
                                      "at": time.time()}
                logger.warning("Exchange %s did not answer: %s", label, e)
                return
            self.health[label] = {"ok": True, "detail": f"top {len(rows)} by value traded", "count": len(rows),
                                  "at": time.time()}
            self.boards[market] = {"market": market, "label": label, "country": MARKET_COUNTRY[market],
                                   "rows": rows, "at": time.time()}

        await asyncio.gather(*(one(m, l) for m, l in markets))
        return {m: b for m, b in self.boards.items() if any(m == x for x, _ in markets)}

    def map_boards(self, names: Dict[str, str], exchanges: Dict[str, str],
                   foreign: Optional[Callable[[str], Optional[bool]]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Every board row gets `us_symbol` (or `why_not`), its activity rank on the
        exchange and a 0..1 `home` score. Returns {us_symbol: home evidence} for the
        tradable ones, keeping each US line's best-ranked home listing. US lines
        whose issuer country still needs checking are left in self.unverified.
        """
        index = NameIndex(names, exchanges, foreign)
        tradable: Dict[str, Dict[str, Any]] = {}
        for market, board in self.boards.items():
            rows = board["rows"]
            n = len(rows)
            changes = sorted(r.get("change") or 0.0 for r in rows)
            for i, r in enumerate(rows):
                sym, how = index.match(r.get("description") or r.get("name") or "")
                r["activity_rank"] = i + 1
                r["us_symbol"], r["match"] = (sym, how) if sym else (None, None)
                r["why_not"] = None if sym else how
                chg = r.get("change") or 0.0
                chg_rank = (changes.index(chg) / (n - 1)) if n > 1 else 0.5
                activity = 1.0 - i / max(n - 1, 1)
                # Where the move ranks on its own exchange today (direction) and how
                # heavily it traded there (attention).
                r["home"] = round(0.6 * chg_rank + 0.4 * activity, 3)
                if sym and (sym not in tradable or tradable[sym]["activity_rank"] > i + 1):
                    tradable[sym] = {"market": market, "exchange_label": board["label"],
                                     "country": board["country"], "home_symbol": r.get("name"),
                                     "home_name": r.get("description"), "exchange": r.get("exchange"),
                                     "change_pct": round(chg, 2), "activity_rank": i + 1, "of": n,
                                     "value_traded": r.get("Value.Traded"), "currency": r.get("currency"),
                                     "home": r["home"], "match": how}
        self.unverified = index.unverified
        return tradable

    def snapshot(self, picks: Dict[str, Any], ranking: List[Dict[str, Any]], per_board: int = 12,
                 excluded: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Each exchange: its hottest stocks, what each maps to, and which one the scout picked."""
        ranked = {r["symbol"]: r for r in ranking}
        why_out = {e["symbol"]: e["why"] for e in (excluded or [])}
        out = []
        for market, label in self.markets():
            b = self.boards.get(market)
            h = self.health.get(label, {})
            rows = []
            for r in (b["rows"][:per_board] if b else []):
                us = r.get("us_symbol")
                rr = ranked.get(us) if us else None
                rows.append({"symbol": r.get("name"), "name": r.get("description"), "change_pct": r.get("change"),
                             "value_traded": r.get("Value.Traded"), "currency": r.get("currency"),
                             "sector": r.get("sector"), "activity_rank": r.get("activity_rank"),
                             "us_symbol": us,
                             "why_not": r.get("why_not") or (why_out.get(us, "not in this hour's pool")
                                                             if us and not rr else None),
                             "score": rr["score"] if rr else None, "rank": rr["rank"] if rr else None,
                             "eligible": rr is not None, "picked": bool(us and us in picks)})
            out.append({"market": market, "label": label, "country": MARKET_COUNTRY[market],
                        "ok": h.get("ok", False), "detail": h.get("detail"), "at": b["at"] if b else None,
                        "tradable": sum(1 for r in (b["rows"] if b else []) if r.get("us_symbol")),
                        "rows": rows})
        return out


exchanges = ExchangeListener()


def issuer_is_foreign(sym: str) -> Optional[bool]:
    """From curated or SEC data (core/universe.py); None when neither knows the symbol."""
    from core.universe import universe
    m = universe.classify(sym)
    if m.source in ("curated", "sec_sic"):
        return m.country != "US"
    return None
