"""
What every symbol IS: sector, theme, country, region, and how it can be traded.

Diversification is impossible without this. NVDA and MSFT are one bet on the same
capex cycle; the engine could not see that because nothing knew what a symbol was.

Classification order (first hit wins):
  1. The curated table below: GICS sector leaders, defence/biotech/energy themes,
     US-listed ADRs of UK/European/Asian companies, and sector/country ETFs.
  2. Crypto pairs: one "Crypto" sleeve.
  3. SEC EDGAR SIC code, fetched lazily for US tickers we have never seen
     (e.g. an insider buy discovered via Form 4). SIC is not GICS, so the mapping
     is approximate and marked source="sec_sic".
  4. Foreign listing suffix (".L", ".NS", ".HK"...): country is known, and a
     US-listed ADR is substituted where one exists. Without one the symbol is
     visible for discovery but not tradable on Alpaca; a country ETF is offered
     as the tradable proxy.
  5. Otherwise "Unclassified" -- still capped like any sector, never exempt.
"""
import asyncio
import logging
import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, List, Any

logger = logging.getLogger("tradeflow.universe")

# The 11 GICS sectors, plus sleeves that are not GICS sectors.
GICS_SECTORS = (
    "Information Technology", "Financials", "Communication Services",
    "Consumer Discretionary", "Health Care", "Industrials", "Consumer Staples",
    "Energy", "Utilities", "Materials", "Real Estate",
)
CRYPTO = "Crypto"
DIVERSIFIED = "Diversified"      # broad/country ETFs: diversified by construction
COMMODITIES = "Commodities"
UNCLASSIFIED = "Unclassified"

DEFENSIVE_SECTORS = frozenset({"Health Care", "Consumer Staples", "Utilities"})

# Benchmark ETF per sector: relative strength is measured against these.
SECTOR_ETF = {
    "Information Technology": "XLK", "Financials": "XLF",
    "Communication Services": "XLC", "Consumer Discretionary": "XLY",
    "Health Care": "XLV", "Industrials": "XLI", "Consumer Staples": "XLP",
    "Energy": "XLE", "Utilities": "XLU", "Materials": "XLB", "Real Estate": "XLRE",
}

US, EUROPE, ASIA, REGION_CRYPTO, GLOBAL, OTHER = (
    "US", "Europe/UK", "Asia", "Crypto", "Global", "Other")

_COUNTRY_REGION = {
    "US": US,
    "UK": EUROPE, "Europe": EUROPE, "Eurozone": EUROPE, "Germany": EUROPE,
    "France": EUROPE, "Netherlands": EUROPE, "Denmark": EUROPE,
    "Switzerland": EUROPE, "Sweden": EUROPE, "Norway": EUROPE, "Italy": EUROPE,
    "Spain": EUROPE, "Finland": EUROPE, "Belgium": EUROPE, "Portugal": EUROPE,
    "China": ASIA, "Hong Kong": ASIA, "India": ASIA, "Japan": ASIA,
    "Taiwan": ASIA, "South Korea": ASIA, "Singapore": ASIA,
    "Global": GLOBAL,
}

_COUNTRY_CURRENCY = {
    "US": "USD", "UK": "GBP", "Europe": "EUR", "Eurozone": "EUR", "Germany": "EUR",
    "France": "EUR", "Netherlands": "EUR", "Italy": "EUR", "Spain": "EUR",
    "Finland": "EUR", "Belgium": "EUR", "Portugal": "EUR", "Denmark": "DKK",
    "Switzerland": "CHF", "Sweden": "SEK", "Norway": "NOK", "China": "CNY",
    "Hong Kong": "HKD", "India": "INR", "Japan": "JPY", "Taiwan": "TWD",
    "South Korea": "KRW", "Singapore": "SGD",
}

# Tradable proxy for a country whose shares Alpaca cannot buy directly.
COUNTRY_ETF = {
    "US": "SPY", "UK": "EWU", "Germany": "EWG", "France": "EWQ", "Japan": "EWJ",
    "Hong Kong": "EWH", "China": "MCHI", "India": "INDA", "Taiwan": "EWT",
    "South Korea": "EWY", "Europe": "VGK", "Eurozone": "EZU",
}
_EUROPE_FALLBACK_ETF = "VGK"


@dataclass(frozen=True)
class SymbolMeta:
    symbol: str
    name: str
    sector: str
    theme: str
    country: str
    region: str
    currency: str
    asset_type: str           # "stock" | "adr" | "etf" | "crypto" | "foreign"
    source: str               # "curated" | "rule" | "sec_sic" | "suffix" | "unknown"
    tradable_on_alpaca: bool = True
    proxy: Optional[str] = None   # tradable substitute when not directly tradable

    @property
    def is_defensive(self) -> bool:
        return self.sector in DEFENSIVE_SECTORS

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["is_defensive"] = self.is_defensive
        return d


# (symbol, name, sector, theme, country, asset_type)
_IT, _FIN, _COM, _CD, _HC, _IND, _CS, _EN, _UT, _MAT, _RE = GICS_SECTORS
_CURATED: List[Tuple[str, str, str, str, str, str]] = [
    # --- US: Information Technology ---
    ("AAPL", "Apple", _IT, "Hardware", "US", "stock"),
    ("MSFT", "Microsoft", _IT, "Software & cloud", "US", "stock"),
    ("NVDA", "Nvidia", _IT, "Semiconductors / AI", "US", "stock"),
    ("AVGO", "Broadcom", _IT, "Semiconductors / AI", "US", "stock"),
    ("AMD", "Advanced Micro Devices", _IT, "Semiconductors / AI", "US", "stock"),
    ("ORCL", "Oracle", _IT, "Software & cloud", "US", "stock"),
    ("CRM", "Salesforce", _IT, "Software & cloud", "US", "stock"),
    ("PLTR", "Palantir", _IT, "Software / AI", "US", "stock"),
    ("FSLR", "First Solar", _IT, "Solar", "US", "stock"),
    # --- US: Financials ---
    ("JPM", "JPMorgan Chase", _FIN, "Banks", "US", "stock"),
    ("BAC", "Bank of America", _FIN, "Banks", "US", "stock"),
    ("GS", "Goldman Sachs", _FIN, "Capital markets", "US", "stock"),
    ("BRK.B", "Berkshire Hathaway", _FIN, "Insurance / conglomerate", "US", "stock"),
    ("V", "Visa", _FIN, "Payments", "US", "stock"),
    ("MA", "Mastercard", _FIN, "Payments", "US", "stock"),
    # --- US: Communication Services ---
    ("GOOGL", "Alphabet", _COM, "Search & ads", "US", "stock"),
    ("META", "Meta Platforms", _COM, "Social & ads", "US", "stock"),
    ("NFLX", "Netflix", _COM, "Streaming", "US", "stock"),
    ("TMUS", "T-Mobile US", _COM, "Telecom", "US", "stock"),
    ("DIS", "Walt Disney", _COM, "Media", "US", "stock"),
    # --- US: Consumer Discretionary ---
    ("AMZN", "Amazon", _CD, "E-commerce & cloud", "US", "stock"),
    ("TSLA", "Tesla", _CD, "Electric vehicles", "US", "stock"),
    ("HD", "Home Depot", _CD, "Home improvement", "US", "stock"),
    ("MCD", "McDonald's", _CD, "Restaurants", "US", "stock"),
    ("NKE", "Nike", _CD, "Apparel", "US", "stock"),
    # --- US: Health Care ---
    ("LLY", "Eli Lilly", _HC, "Pharma", "US", "stock"),
    ("JNJ", "Johnson & Johnson", _HC, "Pharma", "US", "stock"),
    ("UNH", "UnitedHealth", _HC, "Managed care", "US", "stock"),
    ("ABBV", "AbbVie", _HC, "Pharma", "US", "stock"),
    ("MRK", "Merck", _HC, "Pharma", "US", "stock"),
    ("PFE", "Pfizer", _HC, "Pharma", "US", "stock"),
    ("TMO", "Thermo Fisher", _HC, "Life-science tools", "US", "stock"),
    ("ISRG", "Intuitive Surgical", _HC, "Medical devices", "US", "stock"),
    ("VRTX", "Vertex Pharmaceuticals", _HC, "Biotech", "US", "stock"),
    ("REGN", "Regeneron", _HC, "Biotech", "US", "stock"),
    ("AMGN", "Amgen", _HC, "Biotech", "US", "stock"),
    ("GILD", "Gilead Sciences", _HC, "Biotech", "US", "stock"),
    # --- US: Industrials (incl. defence) ---
    ("CAT", "Caterpillar", _IND, "Machinery", "US", "stock"),
    ("UNP", "Union Pacific", _IND, "Rail", "US", "stock"),
    ("GE", "GE Aerospace", _IND, "Aerospace", "US", "stock"),
    ("HON", "Honeywell", _IND, "Conglomerate", "US", "stock"),
    ("DE", "Deere", _IND, "Machinery", "US", "stock"),
    ("RTX", "RTX", _IND, "Aerospace & defence", "US", "stock"),
    ("LMT", "Lockheed Martin", _IND, "Aerospace & defence", "US", "stock"),
    ("NOC", "Northrop Grumman", _IND, "Aerospace & defence", "US", "stock"),
    ("GD", "General Dynamics", _IND, "Aerospace & defence", "US", "stock"),
    # --- US: Consumer Staples ---
    ("PG", "Procter & Gamble", _CS, "Household products", "US", "stock"),
    ("KO", "Coca-Cola", _CS, "Beverages", "US", "stock"),
    ("PEP", "PepsiCo", _CS, "Beverages", "US", "stock"),
    ("WMT", "Walmart", _CS, "Retail", "US", "stock"),
    ("COST", "Costco", _CS, "Retail", "US", "stock"),
    # --- US: Energy ---
    ("XOM", "ExxonMobil", _EN, "Oil & gas", "US", "stock"),
    ("CVX", "Chevron", _EN, "Oil & gas", "US", "stock"),
    ("COP", "ConocoPhillips", _EN, "Oil & gas E&P", "US", "stock"),
    ("EOG", "EOG Resources", _EN, "Oil & gas E&P", "US", "stock"),
    ("OXY", "Occidental Petroleum", _EN, "Oil & gas E&P", "US", "stock"),
    ("SLB", "SLB", _EN, "Oilfield services", "US", "stock"),
    # --- US: Utilities ---
    ("NEE", "NextEra Energy", _UT, "Electric & renewables", "US", "stock"),
    ("DUK", "Duke Energy", _UT, "Electric utility", "US", "stock"),
    ("SO", "Southern Company", _UT, "Electric utility", "US", "stock"),
    # --- US: Materials ---
    ("LIN", "Linde", _MAT, "Industrial gases", "US", "stock"),
    ("SHW", "Sherwin-Williams", _MAT, "Chemicals", "US", "stock"),
    ("NEM", "Newmont", _MAT, "Gold mining", "US", "stock"),
    ("FCX", "Freeport-McMoRan", _MAT, "Copper mining", "US", "stock"),
    # --- US: Real Estate ---
    ("AMT", "American Tower", _RE, "Tower REIT", "US", "stock"),
    ("PLD", "Prologis", _RE, "Logistics REIT", "US", "stock"),
    ("SPG", "Simon Property Group", _RE, "Retail REIT", "US", "stock"),
    ("O", "Realty Income", _RE, "Net-lease REIT", "US", "stock"),

    # --- UK ADRs ---
    ("AZN", "AstraZeneca", _HC, "Pharma", "UK", "adr"),
    ("GSK", "GSK", _HC, "Pharma", "UK", "adr"),
    ("SHEL", "Shell", _EN, "Oil & gas", "UK", "adr"),
    ("BP", "BP", _EN, "Oil & gas", "UK", "adr"),
    ("HSBC", "HSBC", _FIN, "Banks", "UK", "adr"),
    ("UL", "Unilever", _CS, "Household products", "UK", "adr"),
    ("DEO", "Diageo", _CS, "Beverages", "UK", "adr"),
    ("BTI", "British American Tobacco", _CS, "Tobacco", "UK", "adr"),
    ("RIO", "Rio Tinto", _MAT, "Diversified mining", "UK", "adr"),
    # --- European ADRs ---
    ("ASML", "ASML", _IT, "Semiconductor equipment", "Netherlands", "adr"),
    ("SAP", "SAP", _IT, "Software", "Germany", "adr"),
    ("NVO", "Novo Nordisk", _HC, "Pharma", "Denmark", "adr"),
    ("NVS", "Novartis", _HC, "Pharma", "Switzerland", "adr"),
    ("SNY", "Sanofi", _HC, "Pharma", "France", "adr"),
    ("TTE", "TotalEnergies", _EN, "Oil & gas", "France", "adr"),
    ("EQNR", "Equinor", _EN, "Oil & gas", "Norway", "adr"),
    ("ERIC", "Ericsson", _IT, "Telecom equipment", "Sweden", "adr"),
    # --- Asian ADRs ---
    ("TSM", "Taiwan Semiconductor", _IT, "Semiconductors / AI", "Taiwan", "adr"),
    ("SONY", "Sony", _CD, "Consumer electronics", "Japan", "adr"),
    ("TM", "Toyota", _CD, "Autos", "Japan", "adr"),
    ("BABA", "Alibaba", _CD, "E-commerce & cloud", "China", "adr"),
    ("JD", "JD.com", _CD, "E-commerce", "China", "adr"),
    ("PDD", "PDD Holdings", _CD, "E-commerce", "China", "adr"),
    ("BIDU", "Baidu", _COM, "Search / AI", "China", "adr"),
    ("NTES", "NetEase", _COM, "Gaming", "China", "adr"),
    # --- Indian ADRs ---
    ("INFY", "Infosys", _IT, "IT services", "India", "adr"),
    ("WIT", "Wipro", _IT, "IT services", "India", "adr"),
    ("HDB", "HDFC Bank", _FIN, "Banks", "India", "adr"),
    ("IBN", "ICICI Bank", _FIN, "Banks", "India", "adr"),
    ("RDY", "Dr. Reddy's Laboratories", _HC, "Pharma / generics", "India", "adr"),

    # --- Sector ETFs ---
    ("XLK", "Technology Select Sector SPDR", _IT, "Sector ETF", "US", "etf"),
    ("XLF", "Financial Select Sector SPDR", _FIN, "Sector ETF", "US", "etf"),
    ("XLC", "Communication Services Select Sector SPDR", _COM, "Sector ETF", "US", "etf"),
    ("XLY", "Consumer Discretionary Select Sector SPDR", _CD, "Sector ETF", "US", "etf"),
    ("XLV", "Health Care Select Sector SPDR", _HC, "Sector ETF", "US", "etf"),
    ("XLI", "Industrial Select Sector SPDR", _IND, "Sector ETF", "US", "etf"),
    ("XLP", "Consumer Staples Select Sector SPDR", _CS, "Sector ETF", "US", "etf"),
    ("XLE", "Energy Select Sector SPDR", _EN, "Sector ETF", "US", "etf"),
    ("XLU", "Utilities Select Sector SPDR", _UT, "Sector ETF", "US", "etf"),
    ("XLB", "Materials Select Sector SPDR", _MAT, "Sector ETF", "US", "etf"),
    ("XLRE", "Real Estate Select Sector SPDR", _RE, "Sector ETF", "US", "etf"),
    # --- Thematic ETFs ---
    ("SMH", "VanEck Semiconductor ETF", _IT, "Semiconductors", "US", "etf"),
    ("XBI", "SPDR S&P Biotech ETF", _HC, "Biotech", "US", "etf"),
    ("ITA", "iShares US Aerospace & Defense ETF", _IND, "Aerospace & defence", "US", "etf"),
    ("XOP", "SPDR S&P Oil & Gas E&P ETF", _EN, "Oil & gas E&P", "US", "etf"),
    ("ICLN", "iShares Global Clean Energy ETF", DIVERSIFIED, "Clean energy", "Global", "etf"),
    # --- Broad and country ETFs ---
    ("SPY", "SPDR S&P 500 ETF", DIVERSIFIED, "US large cap", "US", "etf"),
    ("QQQ", "Invesco QQQ", DIVERSIFIED, "US growth", "US", "etf"),
    ("EWU", "iShares MSCI United Kingdom ETF", DIVERSIFIED, "Country ETF", "UK", "etf"),
    ("VGK", "Vanguard FTSE Europe ETF", DIVERSIFIED, "Country ETF", "Europe", "etf"),
    ("EZU", "iShares MSCI Eurozone ETF", DIVERSIFIED, "Country ETF", "Eurozone", "etf"),
    ("EWG", "iShares MSCI Germany ETF", DIVERSIFIED, "Country ETF", "Germany", "etf"),
    ("EWQ", "iShares MSCI France ETF", DIVERSIFIED, "Country ETF", "France", "etf"),
    ("EWJ", "iShares MSCI Japan ETF", DIVERSIFIED, "Country ETF", "Japan", "etf"),
    ("EWH", "iShares MSCI Hong Kong ETF", DIVERSIFIED, "Country ETF", "Hong Kong", "etf"),
    ("MCHI", "iShares MSCI China ETF", DIVERSIFIED, "Country ETF", "China", "etf"),
    ("FXI", "iShares China Large-Cap ETF", DIVERSIFIED, "Country ETF", "China", "etf"),
    ("KWEB", "KraneShares CSI China Internet ETF", DIVERSIFIED, "China internet", "China", "etf"),
    ("INDA", "iShares MSCI India ETF", DIVERSIFIED, "Country ETF", "India", "etf"),
    ("EPI", "WisdomTree India Earnings Fund", DIVERSIFIED, "Country ETF", "India", "etf"),
    ("SMIN", "iShares MSCI India Small-Cap ETF", DIVERSIFIED, "Country ETF", "India", "etf"),
    ("EWT", "iShares MSCI Taiwan ETF", DIVERSIFIED, "Country ETF", "Taiwan", "etf"),
    ("EWY", "iShares MSCI South Korea ETF", DIVERSIFIED, "Country ETF", "South Korea", "etf"),
    # --- Commodities ---
    ("GLD", "SPDR Gold Shares", COMMODITIES, "Gold", "Global", "etf"),
    ("USO", "United States Oil Fund", COMMODITIES, "Crude oil", "Global", "etf"),
]

# Foreign listings that are NOT tradable on Alpaca but worth knowing about when a
# copy trader holds them: Indian energy/defence has no US listing at all.
# (code, name, sector, theme)
_FOREIGN_KNOWN: Dict[str, Tuple[str, str, str]] = {
    "HAL.NS": ("Hindustan Aeronautics", _IND, "Aerospace & defence"),
    "BEL.NS": ("Bharat Electronics", _IND, "Aerospace & defence"),
    "RELIANCE.NS": ("Reliance Industries", _EN, "Oil & gas / conglomerate"),
    "ONGC.NS": ("Oil & Natural Gas Corp", _EN, "Oil & gas"),
    "NTPC.NS": ("NTPC", _UT, "Power generation"),
    "TCS.NS": ("Tata Consultancy Services", _IT, "IT services"),
    "RHM.DE": ("Rheinmetall", _IND, "Aerospace & defence"),
    "BA.L": ("BAE Systems", _IND, "Aerospace & defence"),
    "0700.HK": ("Tencent", _COM, "Social & gaming"),
}

# Foreign listing -> US-listed ADR of the same company.
_ADR_OF: Dict[str, str] = {
    "AZN.L": "AZN", "GSK.L": "GSK", "SHEL.L": "SHEL", "BP.L": "BP",
    "HSBA.L": "HSBC", "ULVR.L": "UL", "DGE.L": "DEO", "BATS.L": "BTI",
    "RIO.L": "RIO", "ASML.AS": "ASML", "SAP.DE": "SAP", "NOVOB.CO": "NVO",
    "NOVN.SW": "NVS", "SAN.PA": "SNY", "TTE.PA": "TTE", "EQNR.OL": "EQNR",
    "ERICB.ST": "ERIC", "2330.TW": "TSM", "6758.T": "SONY", "7203.T": "TM",
    "9988.HK": "BABA", "9618.HK": "JD", "9888.HK": "BIDU", "9999.HK": "NTES",
    "INFY.NS": "INFY", "WIPRO.NS": "WIT", "HDFCBANK.NS": "HDB",
    "ICICIBANK.NS": "IBN", "DRREDDY.NS": "RDY",
}

_SUFFIX_COUNTRY = {
    "L": "UK", "DE": "Germany", "F": "Germany", "PA": "France", "AS": "Netherlands",
    "MI": "Italy", "MC": "Spain", "SW": "Switzerland", "ZU": "Switzerland",
    "CO": "Denmark", "ST": "Sweden", "OL": "Norway", "HE": "Finland",
    "BR": "Belgium", "LS": "Portugal", "HK": "Hong Kong", "NS": "India",
    "BO": "India", "T": "Japan", "KS": "South Korea", "TW": "Taiwan",
    "SS": "China", "SZ": "China", "SI": "Singapore",
}

_US_STATES = frozenset(
    "AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS "
    "MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY "
    "PR".split()
)


def _norm_foreign(code: str) -> str:
    """'NOVO-B.CO' and 'NOVOB.CO' are the same listing."""
    base, _, suffix = code.upper().partition(".")
    return f"{base.replace('-', '').replace('_', '')}.{suffix}"


def region_of(country: str) -> str:
    return _COUNTRY_REGION.get(country, OTHER)


def _meta(symbol, name, sector, theme, country, asset_type, source,
          tradable=True, proxy=None) -> SymbolMeta:
    region = REGION_CRYPTO if asset_type == "crypto" else region_of(country)
    return SymbolMeta(
        symbol=symbol, name=name, sector=sector, theme=theme, country=country,
        region=region, currency="USD" if asset_type in ("crypto", "adr", "etf", "stock")
        else _COUNTRY_CURRENCY.get(country, "?"),
        asset_type=asset_type, source=source, tradable_on_alpaca=tradable, proxy=proxy,
    )


_CURATED_META: Dict[str, SymbolMeta] = {
    s: _meta(s, n, sec, th, c, t, "curated") for (s, n, sec, th, c, t) in _CURATED
}
# ADRs trade in USD but carry the home currency's risk; record which one.
_CURATED_META.update({
    s: SymbolMeta(**{**asdict(m), "currency": _COUNTRY_CURRENCY.get(m.country, "USD")})
    for s, m in _CURATED_META.items() if m.asset_type == "adr"
})


def curated_symbols(asset_types: Tuple[str, ...] = ("stock", "adr", "etf")) -> List[str]:
    return [s for s, m in _CURATED_META.items() if m.asset_type in asset_types]


def benchmark_etfs() -> List[str]:
    """Sector and country ETFs: the sleeves whose momentum is ranked."""
    return sorted(set(SECTOR_ETF.values()) | {
        s for s, m in _CURATED_META.items()
        if m.asset_type == "etf" and m.theme == "Country ETF"
    } | {"SPY"})


# ----------------------------------------------------------------------------
# SEC SIC -> approximate GICS sector
# ----------------------------------------------------------------------------

def sic_to_sector(sic: int) -> str:
    """
    SIC codes predate GICS and cut industries differently, so this is a coarse
    translation. It exists so that a never-seen ticker lands in a capped sleeve
    rather than escaping diversification limits entirely.
    """
    s = int(sic)
    if s in (1311, 1381, 1382, 1389) or 1200 <= s <= 1299 or 2900 <= s <= 2999:
        return _EN
    if 1000 <= s <= 1499:
        return _MAT
    if 100 <= s <= 999 or 2000 <= s <= 2199 or 2840 <= s <= 2844:
        return _CS
    if 2833 <= s <= 2836 or 3841 <= s <= 3851 or 8000 <= s <= 8099 or s in (5122, 8731):
        return _HC
    if 2200 <= s <= 2399 or 2500 <= s <= 2599 or 3630 <= s <= 3639 \
            or 3711 <= s <= 3716 or 3900 <= s <= 3999 or 7000 <= s <= 7099 \
            or 8200 <= s <= 8299 or s == 7990:
        return _CD
    if 2400 <= s <= 2499 or 2600 <= s <= 2699 or 2800 <= s <= 2899 or 3000 <= s <= 3399:
        return _MAT
    if 2700 <= s <= 2799 or 4800 <= s <= 4899 or 7800 <= s <= 7999:
        return _COM
    if 3570 <= s <= 3579 or 3660 <= s <= 3699 or 3820 <= s <= 3829 or 7370 <= s <= 7379:
        return _IT
    if 4950 <= s <= 4959:
        return _IND
    if 4900 <= s <= 4999:
        return _UT
    if 5400 <= s <= 5499 or s == 5912:
        return _CS
    if 5200 <= s <= 5999:
        return _CD
    if 6500 <= s <= 6553 or s == 6798:
        return _RE
    if 6000 <= s <= 6799:
        return _FIN
    if 1500 <= s <= 1799 or 3400 <= s <= 3569 or 3580 <= s <= 3629 \
            or 3640 <= s <= 3659 or 3700 <= s <= 3899 or 4000 <= s <= 4799 \
            or 5000 <= s <= 5199 or 7000 <= s <= 8999:
        return _IND
    return UNCLASSIFIED


class Universe:
    """Symbol classification with a lazily-filled SEC cache for unknown US tickers."""

    def __init__(self):
        self._dynamic: Dict[str, SymbolMeta] = {}
        self._ticker_cik: Dict[str, int] = {}
        self._ticker_map_at: float = 0.0
        self._sec_failed: Dict[str, float] = {}
        self._lock = asyncio.Lock()

    def classify(self, symbol: str) -> SymbolMeta:
        sym = (symbol or "").upper().strip()
        m = _CURATED_META.get(sym)
        if m:
            return m
        if "/" in sym:
            base = sym.split("/")[0]
            return _meta(sym, base, CRYPTO, "Crypto", "Global", "crypto", "rule")
        m = self._dynamic.get(sym)
        if m:
            return m
        if "." in sym and sym.rsplit(".", 1)[1] in _SUFFIX_COUNTRY and sym != "BRK.B":
            return self._classify_foreign(sym)
        return _meta(sym, sym, UNCLASSIFIED, "", "US", "stock", "unknown")

    def _classify_foreign(self, sym: str) -> SymbolMeta:
        norm = _norm_foreign(sym)
        adr = _ADR_OF.get(norm)
        if adr:
            base = _CURATED_META[adr]
            return SymbolMeta(**{**asdict(base), "source": "suffix", "proxy": adr})
        country = _SUFFIX_COUNTRY[sym.rsplit(".", 1)[1]]
        known = _FOREIGN_KNOWN.get(norm) or _FOREIGN_KNOWN.get(sym)
        name, sector, theme = known if known else (sym, UNCLASSIFIED, "")
        proxy = COUNTRY_ETF.get(country) or (
            _EUROPE_FALLBACK_ETF if region_of(country) == EUROPE else None)
        return SymbolMeta(
            symbol=sym, name=name, sector=sector, theme=theme, country=country,
            region=region_of(country), currency=_COUNTRY_CURRENCY.get(country, "?"),
            asset_type="foreign", source="suffix", tradable_on_alpaca=False, proxy=proxy,
        )

    def tradable_symbol(self, symbol: str) -> Optional[str]:
        """The symbol we would actually buy for this listing (its ADR), or None."""
        m = self.classify(symbol)
        if m.asset_type == "foreign":
            return None
        if m.source == "suffix" and m.proxy:
            return m.proxy
        return m.symbol

    def needs_lookup(self, symbol: str) -> bool:
        m = self.classify(symbol)
        return m.source == "unknown" and "." not in symbol.replace("BRK.B", "")

    async def enrich(self, symbols: List[str], user_agent: str, limit: int = 20) -> int:
        """
        Classifies unknown US tickers from SEC EDGAR (SIC code + business address).
        Bounded per call and remembers failures for a day so a bad ticker is not
        re-requested every cycle. Returns how many were newly classified.
        """
        todo = [s for s in symbols if self.needs_lookup(s)
                and time.time() - self._sec_failed.get(s, 0.0) > 86400][:limit]
        if not todo:
            return 0
        import aiohttp
        added = 0
        async with self._lock:
            try:
                timeout = aiohttp.ClientTimeout(total=15)
                headers = {"User-Agent": user_agent}
                async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
                    if time.time() - self._ticker_map_at > 86400 or not self._ticker_cik:
                        async with session.get("https://www.sec.gov/files/company_tickers.json") as r:
                            if r.status == 200:
                                data = await r.json(content_type=None)
                                self._ticker_cik = {
                                    str(v["ticker"]).upper(): int(v["cik_str"])
                                    for v in data.values()
                                }
                                self._ticker_map_at = time.time()
                    for sym in todo:
                        cik = self._ticker_cik.get(sym) or self._ticker_cik.get(sym.replace(".", "-"))
                        if not cik:
                            self._sec_failed[sym] = time.time()
                            continue
                        await asyncio.sleep(0.15)   # SEC asks for <= 10 req/s
                        url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
                        async with session.get(url) as r:
                            if r.status != 200:
                                self._sec_failed[sym] = time.time()
                                continue
                            sub = await r.json(content_type=None)
                        meta = self._from_submission(sym, sub)
                        if meta:
                            self._dynamic[sym] = meta
                            added += 1
                        else:
                            self._sec_failed[sym] = time.time()
            except Exception as e:
                logger.warning(f"SEC classification unavailable: {e}")
        return added

    @staticmethod
    def _from_submission(sym: str, sub: Dict[str, Any]) -> Optional[SymbolMeta]:
        try:
            sic = int(sub.get("sic") or 0)
        except (TypeError, ValueError):
            sic = 0
        if not sic:
            return None
        where = ((sub.get("addresses") or {}).get("business") or {}).get("stateOrCountry") or ""
        country = "US" if where.upper() in _US_STATES else (f"Non-US ({where})" if where else "US")
        return SymbolMeta(
            symbol=sym, name=(sub.get("name") or sym).title(), sector=sic_to_sector(sic),
            theme=(sub.get("sicDescription") or "").capitalize(), country=country,
            region=region_of(country), currency="USD", asset_type="stock", source="sec_sic",
        )


universe = Universe()
