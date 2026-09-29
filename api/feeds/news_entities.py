"""
Named entities and event keywords in a financial headline. Rule-based, so it
runs on every ingested headline in microseconds without touching the GPU the
trade desk's LLM uses, and says exactly why it tagged what it tagged.

  ticker     symbols Alpaca tagged, $CASHTAGS, "(NASDAQ: XYZ)"
  company    the tagged symbols' names (from Alpaca's asset list), as written
  org        regulators, agencies, central banks, governments
  broker     analyst firms (upgrades, downgrades, price targets)
  person     a name next to a title: "CEO Jensen Huang", "Lisa Su, chief executive"
  place      countries and regions
  money      "$4.2 billion", "$120M"
  percent    "+12%", "3.5 percent"
  event      what happened -- earnings beat, downgrade, FDA approval, offering... --
             each with the direction it usually pushes the stock (+1 / -1 / 0)
"""
import re
from typing import Any, Dict, Iterable, List, Optional

_ORGS = ("FDA", "SEC", "FTC", "DOJ", "Justice Department", "EPA", "FAA", "NHTSA", "FCC", "CMS", "IRS",
         "Federal Reserve", "Fed", "ECB", "Bank of England", "Bank of Japan", "PBOC", "Treasury",
         "White House", "Congress", "Senate", "House", "Supreme Court", "Pentagon", "Department of Defense",
         "European Commission", "EU", "OPEC", "IMF", "World Bank", "NATO", "CFIUS", "Commerce Department",
         "Nasdaq", "NYSE", "S&P", "Dow")
_BROKERS = ("Goldman Sachs", "Morgan Stanley", "JPMorgan", "J.P. Morgan", "Bank of America", "BofA", "Citi",
            "Citigroup", "Wells Fargo", "Barclays", "UBS", "Deutsche Bank", "Jefferies", "Baird", "Piper Sandler",
            "Wedbush", "Needham", "Oppenheimer", "Raymond James", "KeyBanc", "Mizuho", "Bernstein", "Evercore",
            "RBC", "TD Cowen", "Truist", "Stifel", "BMO", "Benchmark", "Loop Capital", "Rosenblatt", "HSBC",
            "Macquarie", "Cantor Fitzgerald", "BTIG", "HC Wainwright", "Roth Capital", "Roth MKM", "Northland",
            "DA Davidson", "D.A. Davidson", "JMP", "Craig-Hallum", "Telsey", "Keefe Bruyette", "KBW", "Guggenheim", "Wolfe Research", "Susquehanna", "Argus",
            "Morningstar", "Citizens JMP", "B. Riley", "H.C. Wainwright", "Canaccord", "Roth", "Ladenburg")
_PLACES = ("United States", "U.S.", "US", "China", "Chinese", "Japan", "India", "Taiwan", "Korea", "Europe",
           "European", "Germany", "France", "UK", "Britain", "Canada", "Mexico", "Brazil", "Russia", "Ukraine",
           "Israel", "Iran", "Saudi Arabia", "Middle East", "Asia", "Australia", "Singapore", "Hong Kong",
           "Vietnam", "Switzerland", "Netherlands")

# (event, direction, pattern). Order matters only for display.
_EVENTS = [
    ("earnings beat", +1, r"\b(beats?|tops?|surpass\w*|exceeds?)\b.{0,40}\b(estimates?|expectations|consensus|eps|revenue|forecasts?)\b|\bbetter[- ]than[- ]expected\b"),
    ("earnings miss", -1, r"\b(miss(es|ed)?|falls? short|below)\b.{0,40}\b(estimates?|expectations|consensus|eps|revenue|forecasts?)\b|\bworse[- ]than[- ]expected\b"),
    ("earnings", 0, r"\b(earnings|quarterly results|q[1-4] results|eps|fiscal (first|second|third|fourth)[- ]quarter)\b"),
    ("guidance raised", +1, r"\b(raises?|lifts?|boosts?|hikes?|ups)\b.{0,30}\b(guidance|outlook|forecast)\b"),
    ("guidance cut", -1, r"\b(cuts?|lowers?|slashes?|reduces?|withdraws?)\b.{0,30}\b(guidance|outlook|forecast)\b"),
    ("upgrade", +1, r"\bupgrade[sd]?\b"),
    ("downgrade", -1, r"\bdowngrade[sd]?\b"),
    ("price target raised", +1, r"\b(raises?|lifts?|boosts?|ups)\b.{0,30}\b(price target|pt)\b|\bprice target (raised|increased)\b"),
    ("price target cut", -1, r"\b(cuts?|lowers?|reduces?|trims?)\b.{0,30}\b(price target|pt)\b|\bprice target (cut|lowered)\b"),
    ("rating reiterated", 0, r"\b(reiterates?|maintains?)\b.{0,20}\b(buy|outperform|overweight|neutral|hold|underweight|sell)\b"),
    ("coverage initiated", 0, r"\binitiat\w+ coverage\b|\bstarts? coverage\b"),
    ("acquisition", +1, r"\b(acquir\w+|acquisition|buyout|to buy|takeover|merger|merge)\b"),
    ("regulatory approval", +1, r"\b(fda|ema)\b.{0,40}\b(approv\w+|clear\w+|authoriz\w+)\b|\bapproval\b"),
    ("regulatory setback", -1, r"\b(complete response letter|crl|rejects?|clinical hold|warning letter)\b"),
    ("trial results", 0, r"\b(phase [123i]+|trial|study) (data|results|readout|met|succeeded|failed)\b"),
    ("lawsuit / probe", -1, r"\b(lawsuit|sues?|sued|probe|investigation|subpoena|antitrust|class action|fraud|charged)\b"),
    ("buyback", +1, r"\b(buyback|repurchase)\b"),
    ("dividend", +1, r"\b(dividend)\b"),
    ("share offering", -1, r"\b(offering|dilut\w+|priced .{0,20}shares|secondary sale|at-the-market)\b"),
    ("contract / order", +1, r"\b(contract|award(ed|s)?|order(s)? (worth|for|valued)|deal worth)\b"),
    ("partnership", +1, r"\b(partner(s|ship)?|collaborat\w+|teams up|alliance)\b"),
    ("layoffs", 0, r"\b(layoffs?|job cuts|cut .{0,15}jobs|restructur\w+)\b"),
    ("recall / outage", -1, r"\b(recall\w*|outage|breach|hack\w*|cyberattack)\b"),
    ("bankruptcy", -1, r"\b(bankrupt\w*|chapter 11|insolven\w+|going concern)\b"),
    ("insider buying", +1, r"\b(insider|director|ceo|cfo)\b.{0,30}\b(buys?|bought|purchase[sd]?)\b"),
    ("insider selling", -1, r"\b(insider|director|ceo|cfo)\b.{0,30}\b(sells?|sold|dump\w*)\b"),
    ("short report", -1, r"\bshort[- ]seller\b|\bshort report\b"),
    ("stock split", +1, r"\bstock split\b"),
    ("tariffs / trade", -1, r"\b(tariffs?|export (ban|curbs?|controls?|restrictions?)|sanctions?)\b"),
    ("rally", +1, r"\b(soars?|surges?|jumps?|rall(y|ies)|skyrockets?|record high|all-time high)\b"),
    ("selloff", -1, r"\b(plunges?|tumbles?|sinks?|slumps?|crash\w*|plummets?|tanks?)\b"),
]
_EVENT_RX = [(name, d, re.compile(p, re.IGNORECASE)) for name, d, p in _EVENTS]

_MONEY = re.compile(r"(?:US)?\$\s?\d[\d,]*(?:\.\d+)?\s?(?:trillion|billion|million|thousand|bn|mn|tn|[BMKT]\b)?",
                    re.IGNORECASE)
_PERCENT = re.compile(r"[+-]?\d+(?:\.\d+)?\s?(?:%|percent\b)", re.IGNORECASE)
_CASHTAG = re.compile(r"\$([A-Z]{1,5})\b")
_EXCHANGE = re.compile(r"\((?:NASDAQ|NYSE|NYSE American|AMEX|OTC)\s*:\s*([A-Z.]{1,6})\)", re.IGNORECASE)
_TITLE = r"(?:CEO|CFO|COO|CTO|Chairman|Chairwoman|Chair|President|founder|co-founder|Chief Executive|Secretary|Governor|Senator|analyst)"
_NAME = r"([A-Z][a-z]+(?:\s(?:[A-Z]\.\s)?[A-Z][a-zA-Z'-]+){1,2})"
_PERSON_AFTER = re.compile(_TITLE + r"\s" + _NAME)
_PERSON_BEFORE = re.compile(_NAME + r",\s(?:the\s)?(?:company's\s)?" + _TITLE)
_PERSON_NAMED = re.compile(r"\b(?:Elects|Appoints|Names|Hires|Taps|Promotes)\s" + _NAME + r"\s(?:As|as)\b")
# Headlines are Title Case: "Jensen Huang Over His..." -- a name ends at the first of these.
_NAME_STOP = {"Over", "As", "To", "On", "In", "For", "Of", "And", "With", "After", "Says", "Said", "Slams",
              "Says", "Sees", "Warns", "Plans", "Buys", "Sells", "Is", "Was", "At", "By", "From", "Amid",
              "Ahead", "Into", "Backs", "Calls", "Joins", "Steps", "Will", "Could", "The"}
_NOT_PEOPLE = {"Wall Street", "Street Journal", "White House", "Supreme Court", "Federal Reserve"}

_SUFFIX = re.compile(r"\s*(,?\s(Inc\.?|Incorporated|Corp\.?|Corporation|Company|Co\.?|Ltd\.?|Limited|PLC|plc|"
                     r"Holdings?|Group|N\.V\.|S\.A\.|AG|SE|Class [A-C]|Common Stock|Ordinary Shares|"
                     r"American Depositary Shares?|ADS|CL [A-C]|Technologies|Technology|&|& Co)(?:\b|(?<=&))\.?)+$", re.IGNORECASE)


def short_name(name: str) -> str:
    """'NVIDIA Corporation Common Stock' -> 'NVIDIA'."""
    n = (name or "").strip()
    for _ in range(4):
        m = _SUFFIX.sub("", n).strip(" ,.")
        if m == n:
            break
        n = m
    return n


def _word(term: str) -> re.Pattern:
    return re.compile(r"(?<![A-Za-z])" + re.escape(term) + r"(?![A-Za-z])")


_ORG_RX = [(o, _word(o)) for o in _ORGS]
_BROKER_RX = [(b, _word(b)) for b in _BROKERS]
_PLACE_RX = [(p, _word(p)) for p in _PLACES]


def extract(text: str, symbols: Iterable[str] = (),
            names: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Entities and events in `text`; `symbols` are the tickers the feed tagged."""
    text = text or ""
    names = names or {}
    ents: Dict[str, List[str]] = {k: [] for k in
                                  ("ticker", "company", "org", "broker", "person", "place", "money", "percent")}

    def add(kind: str, value: str):
        value = value.strip()
        if value and value not in ents[kind]:
            ents[kind].append(value)

    for s in symbols:
        add("ticker", s.upper())
    for m in _CASHTAG.finditer(text):
        add("ticker", m.group(1))
    for m in _EXCHANGE.finditer(text):
        add("ticker", m.group(1).upper())
    for s in ents["ticker"]:
        n = short_name(names.get(s, ""))
        if len(n) >= 3 and re.search(r"(?<![A-Za-z])" + re.escape(n.split()[0]) + r"(?![A-Za-z])", text,
                                     re.IGNORECASE):
            add("company", n)
    for name, rx in _BROKER_RX:
        if rx.search(text):
            add("broker", name)
    for name, rx in _ORG_RX:
        # "Fed"/"EU"/"House" are common words or substrings elsewhere: require the exact case.
        if rx.search(text) and name not in ents["broker"]:
            add("org", name)
    for rx in (_PERSON_AFTER, _PERSON_BEFORE, _PERSON_NAMED):
        for m in rx.finditer(text):
            words = []
            for w in m.group(1).split():
                if w in _NAME_STOP:
                    break
                words.append(w)
            name = " ".join(words)
            if (len(words) >= 2 and name not in _NOT_PEOPLE
                    and not any(name in c or c in name for c in ents["company"])):
                add("person", name)
    for name, rx in _PLACE_RX:
        if rx.search(text):
            add("place", {"U.S.": "US", "United States": "US", "Chinese": "China", "European": "Europe",
                          "Britain": "UK"}.get(name, name))
    for m in _MONEY.finditer(text):
        add("money", m.group(0))
    for m in _PERCENT.finditer(text):
        add("percent", m.group(0).replace(" ", ""))

    events = [{"event": name, "direction": d} for name, d, rx in _EVENT_RX if rx.search(text)]
    # A specific event supersedes its generic parent.
    have = {e["event"] for e in events}
    if have & {"earnings beat", "earnings miss"}:
        events = [e for e in events if e["event"] != "earnings"]
    lean = sum(e["direction"] for e in events)
    return {"entities": {k: v for k, v in ents.items() if v}, "events": events,
            "lean": "bullish" if lean > 0 else "bearish" if lean < 0 else "neutral"}
