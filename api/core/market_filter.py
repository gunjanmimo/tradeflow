"""
Which markets and symbols the bots may open NEW positions in.

Turning a market or symbol off only gates entries. Positions already open keep
their stops, targets and strategy exits, so switching a market off never strands
a position the bots can no longer sell. US stocks are the only market.
"""
import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("tradeflow.markets")

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_PATH = os.path.join(_DATA_DIR, "market_filter.json")

MARKETS = ("stocks",)


def market_of(symbol: str) -> str:
    return "stocks"


class MarketFilter:
    def __init__(self):
        self.disabled_markets: set[str] = set()
        self.disabled_symbols: set[str] = set()
        self._load()

    def _load(self):
        try:
            with open(_PATH) as file:
                raw = json.load(file)
            self.disabled_markets = {m for m in raw.get("disabled_markets", []) if m in MARKETS}
            self.disabled_symbols = {str(s).upper() for s in raw.get("disabled_symbols", [])}
        except FileNotFoundError:
            pass
        except Exception as error:
            logger.error("Could not load market filter: %s", error)

    def _save(self):
        try:
            os.makedirs(_DATA_DIR, exist_ok=True)
            temporary_path = _PATH + ".tmp"
            with open(temporary_path, "w") as file:
                json.dump({"disabled_markets": sorted(self.disabled_markets),
                           "disabled_symbols": sorted(self.disabled_symbols)}, file, indent=2)
            os.replace(temporary_path, _PATH)
        except Exception as error:
            logger.error("Could not persist market filter: %s", error)

    def set_market(self, market: str, enabled: bool):
        if market not in MARKETS:
            raise ValueError(f"Unknown market '{market}'. Expected one of {', '.join(MARKETS)}.")
        (self.disabled_markets.discard if enabled else self.disabled_markets.add)(market)
        self._save()

    def set_symbol(self, symbol: str, enabled: bool):
        sym = symbol.upper().strip()
        (self.disabled_symbols.discard if enabled else self.disabled_symbols.add)(sym)
        self._save()

    def entry_block_reason(self, symbol: str) -> Optional[str]:
        market = market_of(symbol)
        if market in self.disabled_markets:
            return f"{market.capitalize()} trading is switched off. No new {market} entries."
        if symbol.upper() in self.disabled_symbols:
            return f"{symbol} is switched off. No new entries in it."
        return None

    def snapshot(self, watchlist: Optional[List[str]] = None) -> Dict[str, Any]:
        symbols = sorted(set(watchlist or []) | self.disabled_symbols)
        return {
            "markets": {m: m not in self.disabled_markets for m in MARKETS},
            "symbols": [{"symbol": s, "market": market_of(s), "enabled": s not in self.disabled_symbols}
                        for s in symbols],
        }


market_filter = MarketFilter()
