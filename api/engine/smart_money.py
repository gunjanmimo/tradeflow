"""
Smart money: turn what insiders and top investors did into BUY / HOLD / AVOID,
and decide which BUYs the bots may actually trade.

The aggregator (feeds/multi_source_aggregator.py) scores whatever the sources
mention, bullish or bearish, and its consensus expires after an hour. An
insider purchase is a multi-day signal, so this book keeps its own dated record
of every Form 4 transaction and the latest eToro holdings (api/data/
smart_money.json, survives restarts), and reads them as:

  BUY    an insider open-market purchase of at least SMART_MONEY_MIN_BUY_USD in
         the last SMART_MONEY_LOOKBACK_DAYS, with no larger insider selling
         since; or at least SMART_MONEY_MIN_BREADTH of the top eToro investors
         holding it
  AVOID  insiders net SELLING in the lookback (and no newer purchase): a
         buy-side strategy should not be buying what insiders are selling
  HOLD   anything else: mentioned by a source, but not a buy signal (e.g. held
         by one of fifteen eToro investors)

A BUY is TRADABLE only if it is a real, liquid US stock:
  * a US ticker that Alpaca lists as a tradable equity (a crypto token eToro
    lists under a bare symbol is not)
  * not a leveraged, inverse or volatility product (by Alpaca's asset name)
  * last close >= SMART_MONEY_MIN_PRICE and average daily dollar volume
    >= SMART_MONEY_MIN_DOLLAR_VOLUME: micro-caps with thin books fill far
    worse live than on paper
Every non-tradable BUY is still listed, with the reason.
"""
import json
import logging
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional

from core.config import settings

logger = logging.getLogger("tradeflow.smart_money")

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_PATH = os.path.join(_DATA_DIR, "smart_money.json")

_US_TICKER = re.compile(r"^[A-Z]{1,5}(\.[A-Z])?$")
# Alpaca asset names of leveraged / inverse / volatility products, e.g.
# "Direxion Daily Technology Bull 3x ETF", "ProShares UltraPro Short QQQ".
_LEVERAGED = re.compile(r"\b(\d(\.\d)?x|ultra(pro)?|leveraged|inverse|short|bull|bear|daily|vix|volatility)\b",
                        re.IGNORECASE)
_FUND = re.compile(r"\b(etf|etn|fund|shares|trust|proshares|direxion)\b", re.IGNORECASE)


class SmartMoneyBook:
    def __init__(self):
        self.events: Dict[str, List[Dict[str, Any]]] = {}     # symbol -> Form 4 events, newest last
        self.etoro: Dict[str, Dict[str, Any]] = {}            # symbol -> latest holding snapshot
        self.etoro_at = 0.0
        self._loaded = False

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def _load(self):
        if self._loaded:
            return
        self._loaded = True
        try:
            with open(_PATH) as f:
                raw = json.load(f)
            self.events = raw.get("events", {})
            self.etoro = raw.get("etoro", {})
            self.etoro_at = float(raw.get("etoro_at", 0.0))
        except (OSError, ValueError):
            pass

    def _save(self):
        try:
            os.makedirs(_DATA_DIR, exist_ok=True)
            tmp = _PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"events": self.events, "etoro": self.etoro, "etoro_at": self.etoro_at}, f)
            os.replace(tmp, _PATH)
        except OSError as e:
            logger.warning("Could not save the smart-money book: %s", e)

    # ------------------------------------------------------------------
    # input
    # ------------------------------------------------------------------
    def ingest(self, signals: Iterable[Any], now: Optional[float] = None):
        """Records one aggregator cycle's signals (SourceSignal objects)."""
        self._load()
        now = time.time() if now is None else now
        etoro_seen: Dict[str, Dict[str, Any]] = {}
        for s in signals:
            if s.source_name == "SEC Form 4":
                evs = self.events.setdefault(s.symbol, [])
                if not any(e["details"] == s.details for e in evs):     # the feed repeats filings
                    evs.append({"at": now, "value_usd": float(getattr(s, "value_usd", 0.0) or 0.0),
                                "details": s.details})
            elif s.source_name == "eToro":
                etoro_seen[s.symbol] = {"breadth": float(getattr(s, "breadth", 0.0) or 0.0),
                                        "details": s.details}
        if etoro_seen:
            self.etoro, self.etoro_at = etoro_seen, now
        horizon = now - settings.SMART_MONEY_LOOKBACK_DAYS * 86400 * 2
        for sym in list(self.events):
            self.events[sym] = [e for e in self.events[sym] if e["at"] >= horizon]
            if not self.events[sym]:
                del self.events[sym]
        self._save()

    # ------------------------------------------------------------------
    # reading
    # ------------------------------------------------------------------
    def verdict(self, symbol: str, now: Optional[float] = None) -> Dict[str, Any]:
        self._load()
        now = time.time() if now is None else now
        since = now - settings.SMART_MONEY_LOOKBACK_DAYS * 86400
        evs = [e for e in self.events.get(symbol, []) if e["at"] >= since]
        buys = [e for e in evs if e["value_usd"] >= settings.SMART_MONEY_MIN_BUY_USD]
        sells = [e for e in evs if e["value_usd"] < 0]
        et = self.etoro.get(symbol)
        reasons: List[str] = []
        verdict = "HOLD"
        last_buy = max((e["at"] for e in buys), default=None)
        last_sell = max((e["at"] for e in sells), default=None)
        bought = sum(e["value_usd"] for e in buys)
        sold = -sum(e["value_usd"] for e in sells)
        if buys and (last_sell is None or last_buy >= last_sell or bought >= sold):
            verdict = "BUY"
            reasons.append(f"insider buying ${bought:,.0f} in {len(buys)} filing(s)")
        elif sells and (last_buy is None or last_sell > last_buy):
            verdict = "AVOID"
            reasons.append(f"insider selling ${sold:,.0f} in {len(sells)} filing(s)")
        small = [e for e in evs if 0 < e["value_usd"] < settings.SMART_MONEY_MIN_BUY_USD]
        if small and verdict != "BUY":
            reasons.append(f"insider buy under ${settings.SMART_MONEY_MIN_BUY_USD:,.0f} (too small to count)")
        if et:
            if et["breadth"] >= settings.SMART_MONEY_MIN_BREADTH and verdict != "AVOID":
                verdict = "BUY"
            reasons.append(et["details"])
        return {"symbol": symbol, "verdict": verdict, "reasons": reasons,
                "insider_bought": round(bought, 2), "insider_sold": round(sold, 2),
                "etoro_breadth": et["breadth"] if et else None}

    def tradable(self, symbol: str) -> Optional[str]:
        """None when the bots may trade this symbol, else why not."""
        from engine.executor import executor
        from feeds.daily_bars import daily_bars
        if not _US_TICKER.match(symbol):
            return "not a US stock ticker"
        if executor.is_connected and not executor.is_mock_mode:
            if symbol not in executor.alpaca_tradable_symbols:
                return "not a tradable US stock on Alpaca (e.g. a crypto token)"
            name = executor.alpaca_asset_names.get(symbol, "")
            if _LEVERAGED.search(name) and _FUND.search(name):
                return f"leveraged / inverse product ({name})"
        closes = daily_bars.closes(symbol)
        if closes is None or not len(closes):
            return "no daily history yet (price and liquidity unknown)"
        if closes[-1] < settings.SMART_MONEY_MIN_PRICE:
            return f"price ${closes[-1]:,.2f} under ${settings.SMART_MONEY_MIN_PRICE:,.0f}"
        dv = daily_bars.dollar_volume.get(symbol, 0.0)
        if dv < settings.SMART_MONEY_MIN_DOLLAR_VOLUME:
            return (f"illiquid: ${dv / 1e6:,.1f}M a day traded "
                    f"(minimum ${settings.SMART_MONEY_MIN_DOLLAR_VOLUME / 1e6:,.0f}M)")
        return None

    def symbols(self) -> List[str]:
        self._load()
        return sorted(set(self.events) | set(self.etoro))

    def buy_list(self) -> List[str]:
        """Tradable BUY symbols: what the smart_money strategy trades."""
        return [s for s in self.symbols()
                if self.verdict(s)["verdict"] == "BUY" and self.tradable(s) is None]

    def snapshot(self) -> Dict[str, Any]:
        rows = []
        for s in self.symbols():
            v = self.verdict(s)
            why_not = self.tradable(s) if v["verdict"] == "BUY" else None
            v["tradable"] = v["verdict"] == "BUY" and why_not is None
            v["why_not"] = why_not
            rows.append(v)
        order = {"BUY": 0, "HOLD": 1, "AVOID": 2}
        rows.sort(key=lambda r: (order[r["verdict"]], not r["tradable"], -r["insider_bought"]))
        return {"rows": rows, "trading_enabled": settings.SMART_MONEY_TRADING,
                "rules": {"lookback_days": settings.SMART_MONEY_LOOKBACK_DAYS,
                          "min_buy_usd": settings.SMART_MONEY_MIN_BUY_USD,
                          "min_breadth": settings.SMART_MONEY_MIN_BREADTH,
                          "min_price": settings.SMART_MONEY_MIN_PRICE,
                          "min_dollar_volume": settings.SMART_MONEY_MIN_DOLLAR_VOLUME}}


smart_money = SmartMoneyBook()
