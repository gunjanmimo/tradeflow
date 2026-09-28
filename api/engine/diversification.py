"""
Portfolio diversification: sleeve caps at entry, and daily-bar risk analysis.

Two jobs, both driven by the risk dial (core/risk_profile.py):

1. GATE. Before any entry, how many dollars may this symbol take without pushing
   its sector, region, crypto sleeve or the cyclical share of the book past the
   dial's caps? Sizing is clamped to that headroom, and an entry with no
   headroom is refused. A symbol highly correlated with something already held
   gets half size: two 0.9-correlated positions are one position at double risk.

2. MEASURE. Portfolio volatility, VaR/CVaR, diversification ratio and each
   sleeve's share of total risk, from ~90 days of daily returns -- plus a
   what-if for any candidate: what adding a standard-size position would do.

Percentages are of the trading budget (the same base position sizing uses), so
"30% sector cap" means 30% of the capital the bot is allowed to deploy.

The gate reads only in-memory state and a correlation cache refreshed off the
order path; nothing here makes a network call when an order is being placed.
"""
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from core.config import settings
from core.state import state
from core.universe import (
    universe, SymbolMeta, CRYPTO, DIVERSIFIED, GICS_SECTORS, COMMODITIES,
    UNCLASSIFIED, US, EUROPE, ASIA, OTHER, GLOBAL, REGION_CRYPTO,
)
from feeds.daily_bars import daily_bars

logger = logging.getLogger("tradeflow.diversification")

_Z95 = 1.645
_PHI95 = math.exp(-_Z95 ** 2 / 2) / math.sqrt(2 * math.pi)


def budget_base() -> float:
    """The capital percentages are measured against (matches allocation sizing)."""
    return state.budget_base


@dataclass
class Assessment:
    symbol: str
    meta: SymbolMeta
    headroom_dollars: float                  # before the correlation scale
    binding: Optional[str]                   # which cap is tightest
    size_scale: float = 1.0
    max_corr: Optional[float] = None
    corr_partner: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    @property
    def max_dollars(self) -> float:
        return max(0.0, self.headroom_dollars * self.size_scale)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "sector": self.meta.sector, "region": self.meta.region,
            "theme": self.meta.theme, "country": self.meta.country,
            "headroom_dollars": round(self.headroom_dollars, 2),
            "max_dollars": round(self.max_dollars, 2),
            "binding": self.binding,
            "size_scale": self.size_scale,
            "max_corr": None if self.max_corr is None else round(self.max_corr, 3),
            "corr_partner": self.corr_partner,
            "notes": list(self.notes),
        }


class DiversificationManager:
    def __init__(self):
        # symbol -> (dollars, reserved_at): sized entries not yet visible as positions
        self._reserved: Dict[str, Tuple[float, float]] = {}
        # symbol -> (max |corr| with any holding, which holding); refreshed off-path
        self._corr_to_holdings: Dict[str, Tuple[float, str]] = {}
        self._corr_at: float = 0.0
        self._risk_cache: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Exposure
    # ------------------------------------------------------------------

    def reserve(self, symbol: str, dollars: float):
        self._reserved[symbol] = (float(dollars), time.time())

    def _live_reservations(self) -> Dict[str, float]:
        """Reservations for buys still in flight. Anything else has expired or filled."""
        from engine.executor import executor
        now = time.time()
        out = {}
        for sym, (dollars, at) in list(self._reserved.items()):
            in_flight = (sym in executor.pending_orders or sym in executor.awaiting_fill
                         or any(o["side"] == "buy" for o in executor.open_orders.get(sym, ())))
            if (sym in state.active_positions or not in_flight
                    or now - at > settings.DIVERSIFICATION_RESERVATION_SECONDS):
                self._reserved.pop(sym, None)
                continue
            out[sym] = dollars
        return out

    def holdings(self, include_reserved: bool = True) -> Dict[str, float]:
        """symbol -> dollar notional currently committed."""
        out: Dict[str, float] = {}
        for sym, pos in state.active_positions.items():
            qty = float(pos.get("qty") or 0.0)
            px = float(pos.get("current_price") or pos.get("avg_entry_price") or 0.0)
            if qty > 0 and px > 0:
                out[sym] = qty * px
        if include_reserved:
            for sym, dollars in self._live_reservations().items():
                out[sym] = out.get(sym, 0.0) + dollars
        return out

    def exposures(self, holdings: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        h = self.holdings() if holdings is None else holdings
        by_sector: Dict[str, float] = {}
        by_region: Dict[str, float] = {}
        by_theme: Dict[str, float] = {}
        defensive = 0.0
        for sym, dollars in h.items():
            m = universe.classify(sym)
            by_sector[m.sector] = by_sector.get(m.sector, 0.0) + dollars
            by_region[m.region] = by_region.get(m.region, 0.0) + dollars
            if m.theme:
                by_theme[m.theme] = by_theme.get(m.theme, 0.0) + dollars
            if m.is_defensive:
                defensive += dollars
        total = sum(h.values())
        return {
            "total": total, "by_sector": by_sector, "by_region": by_region,
            "by_theme": by_theme, "defensive": defensive,
            "non_defensive": total - defensive, "crypto": by_sector.get(CRYPTO, 0.0),
        }

    # ------------------------------------------------------------------
    # The entry gate
    # ------------------------------------------------------------------

    def _region_cap_pct(self, region: str, profile) -> Optional[float]:
        if region == US:
            return profile.max_us_pct
        if region in (EUROPE, ASIA, OTHER):
            return profile.max_intl_region_pct
        return None   # Crypto has its own sleeve cap; Global ETFs span regions

    def assess(self, symbol: str, holdings: Optional[Dict[str, float]] = None) -> Assessment:
        """Dollar headroom for a NEW position in `symbol` under the current dial."""
        profile = state.risk_profile
        base = budget_base()
        m = universe.classify(symbol)
        exp = self.exposures(holdings)
        pct = lambda p: base * p / 100.0

        limits: List[Tuple[str, float]] = []
        if m.sector == CRYPTO:
            limits.append((f"crypto sleeve cap {profile.max_crypto_pct:.0f}%",
                           pct(profile.max_crypto_pct) - exp["crypto"]))
        elif m.sector != DIVERSIFIED:
            limits.append((f"{m.sector} sector cap {profile.max_sector_pct:.0f}%",
                           pct(profile.max_sector_pct) - exp["by_sector"].get(m.sector, 0.0)))
        region_cap = self._region_cap_pct(m.region, profile)
        if region_cap is not None:
            limits.append((f"{m.region} region cap {region_cap:.0f}%",
                           pct(region_cap) - exp["by_region"].get(m.region, 0.0)))
        if not m.is_defensive and profile.min_defensive_pct > 0:
            limits.append((f"cyclical share cap {100 - profile.min_defensive_pct:.0f}% "
                           f"(keeps {profile.min_defensive_pct:.0f}% for defensive sectors)",
                           pct(100 - profile.min_defensive_pct) - exp["non_defensive"]))

        binding, headroom = None, base
        for name, room in limits:
            if room < headroom:
                binding, headroom = name, room
        a = Assessment(symbol=symbol, meta=m, headroom_dollars=max(0.0, headroom),
                       binding=binding)

        if m.sector == UNCLASSIFIED:
            a.notes.append("unclassified: capped as its own sector until SEC lookup classifies it")

        corr = self._corr_to_holdings.get(symbol)
        if corr and symbol not in state.active_positions:
            a.max_corr, a.corr_partner = corr
            if corr[0] >= profile.max_pair_correlation:
                a.size_scale = 0.5
                a.notes.append(
                    f"{corr[0]:.2f} correlated with held {corr[1]} "
                    f"(limit {profile.max_pair_correlation:.2f}): size halved")
        return a

    def fit(self, symbol: str, sleeves: Optional[Dict[str, Any]] = None) -> Tuple[float, List[str]]:
        """
        0..1: how much a new position in `symbol` spreads the book as it stands
        NOW (the manager calls this after every entry). Room under the caps, a
        sector or asset class nothing is held in yet, an under-target region and a
        defensive shortfall raise it; correlation with a holding lowers it. Zero
        means the entry gate would refuse it. Same rules as discovery's fit.
        """
        profile = state.risk_profile
        base = budget_base()
        std = base * profile.max_position_notional_pct / 100.0
        a = self.assess(symbol)
        m = a.meta
        crypto = m.sector == CRYPTO
        min_useful = min(std, 15.0 if crypto else 30.0) if std > 0 else 0.0
        if a.max_dollars < min_useful or a.max_dollars <= 0:
            return 0.0, [f"blocked by {a.binding}" if a.headroom_dollars < min_useful
                         else (a.notes[-1] if a.notes else "no headroom")]

        sl = sleeves if sleeves is not None else self.sleeves()
        held = self.holdings()
        reasons: List[str] = []
        fit = 0.5 * min(1.0, a.max_dollars / std) if std > 0 else 0.5

        if crypto:
            if not any(universe.classify(s).sector == CRYPTO for s in held) and held:
                fit += 0.15
                reasons.append("first crypto exposure")
        elif m.sector not in (DIVERSIFIED,) and not any(
                universe.classify(s).sector == m.sector for s in held):
            fit += 0.15
            reasons.append(f"new sector {m.sector}")

        under = {r["sleeve"] for r in sl["regions"] if r["under_target"]}
        if m.region in under:
            fit += 0.25
            reasons.append(f"fills under-target region {m.region}")
        if (sl["invested_pct"] > 0 and sl["defensive_pct"] < profile.min_defensive_pct
                and m.is_defensive):
            fit += 0.2
            reasons.append("adds defensive exposure")
        if a.max_corr is not None:
            fit -= 0.4 * max(0.0, a.max_corr - 0.4) / 0.6
            if a.max_corr >= profile.max_pair_correlation:
                reasons.append(f"{a.max_corr:.2f} correlated with {a.corr_partner}")
        return max(0.0, min(1.0, fit)), reasons

    def entry_block_reason(self, symbol: str, min_order: float) -> Optional[str]:
        a = self.assess(symbol)
        if a.max_dollars >= min_order:
            return None
        base = budget_base()
        if a.headroom_dollars < min_order:
            return (f"Diversification: {symbol} ({a.meta.sector}, {a.meta.region}) has "
                    f"${a.headroom_dollars:,.2f} headroom under the {a.binding} "
                    f"of ${base:,.0f} budget at risk dial {state.risk_factor}.")
        return (f"Diversification: {symbol} headroom ${a.headroom_dollars:,.2f} halved to "
                f"${a.max_dollars:,.2f} by correlation with {a.corr_partner}; below the "
                f"${min_order:,.2f} minimum order.")

    # ------------------------------------------------------------------
    # Correlation cache (refreshed off the order path)
    # ------------------------------------------------------------------

    def refresh_correlations(self, symbols: List[str]):
        held = list(self.holdings(include_reserved=False).keys())
        cache: Dict[str, Tuple[float, str]] = {}
        window = settings.CORRELATION_WINDOW_DAYS
        if held:
            for sym in set(symbols):
                others = [h for h in held if h != sym]
                if not others:
                    continue
                res = daily_bars.aligned_returns([sym] + others, window)
                if not res or res[0][0] != sym or len(res[0]) < 2:
                    continue
                names, R = res
                if R.shape[1] < 20:
                    continue
                with np.errstate(invalid="ignore", divide="ignore"):
                    c = np.corrcoef(R)[0, 1:]
                c = np.where(np.isfinite(c), c, 0.0)
                j = int(np.argmax(c))
                cache[sym] = (float(c[j]), names[1 + j])
        self._corr_to_holdings = cache
        self._corr_at = time.time()

    # ------------------------------------------------------------------
    # Risk analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _risk_stats(weights: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Var-covar and historical risk of a dollar-weighted book on daily returns."""
        if not weights:
            return None
        res = daily_bars.aligned_returns(list(weights), settings.CORRELATION_WINDOW_DAYS)
        if not res:
            return None
        names, R = res
        if R.shape[1] < 20:
            return None
        w = np.array([weights[s] for s in names])
        cov = np.atleast_2d(np.cov(R))
        var = float(w @ cov @ w)
        sigma = math.sqrt(max(var, 0.0))
        sig_i = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        pnl = w @ R
        gross = float(w.sum())
        rc = (w * (cov @ w) / var) if var > 0 else np.zeros_like(w)
        corr = np.eye(len(names))
        if len(names) > 1:
            with np.errstate(invalid="ignore", divide="ignore"):
                cc = np.corrcoef(R)
            corr = np.where(np.isfinite(cc), cc, 0.0)
            np.fill_diagonal(corr, 1.0)
        return {
            "names": names,
            "covered_notional": gross,
            "days": int(R.shape[1]),
            "vol_1d_dollars": sigma,
            "vol_ann_pct": (sigma * math.sqrt(252) / gross * 100) if gross > 0 else None,
            "var95_1d": _Z95 * sigma,
            "cvar95_1d": sigma * _PHI95 / 0.05,
            "hist_var95_1d": float(-np.percentile(pnl, 5)),
            "worst_day": float(pnl.min()),
            "diversification_ratio": float((w * sig_i).sum() / sigma) if sigma > 0 else None,
            "risk_contrib": {n: float(r) for n, r in zip(names, rc)},
            "effective_bets": float(1.0 / np.sum(rc ** 2)) if np.sum(rc ** 2) > 0 else None,
            "corr": corr,
        }

    def sleeves(self) -> Dict[str, Any]:
        """Exposure per sleeve against the dial's caps and targets."""
        profile = state.risk_profile
        base = budget_base()
        exp = self.exposures()
        pct = lambda d: round(d / base * 100, 2) if base > 0 else 0.0
        room = lambda cap, used: round(max(0.0, base * cap / 100 - used), 2)

        sectors = []
        for name in list(GICS_SECTORS) + [CRYPTO, COMMODITIES, UNCLASSIFIED, DIVERSIFIED]:
            used = exp["by_sector"].get(name, 0.0)
            if name in (UNCLASSIFIED, COMMODITIES, DIVERSIFIED) and used <= 0:
                continue
            cap = (profile.max_crypto_pct if name == CRYPTO
                   else None if name == DIVERSIFIED else profile.max_sector_pct)
            sectors.append({
                "sleeve": name, "dollars": round(used, 2), "pct": pct(used), "cap_pct": cap,
                "headroom": room(cap, used) if cap is not None else None,
                "defensive": name in ("Health Care", "Consumer Staples", "Utilities"),
            })
        regions = []
        for name in (US, EUROPE, ASIA, REGION_CRYPTO, GLOBAL, OTHER):
            used = exp["by_region"].get(name, 0.0)
            if name in (GLOBAL, OTHER) and used <= 0:
                continue
            cap = (profile.max_crypto_pct if name == REGION_CRYPTO
                   else self._region_cap_pct(name, profile))
            target = profile.intl_region_target_pct if name in (EUROPE, ASIA) else None
            regions.append({
                "sleeve": name, "dollars": round(used, 2), "pct": pct(used), "cap_pct": cap,
                "target_pct": target,
                "headroom": room(cap, used) if cap is not None else None,
                "under_target": bool(target and pct(used) < target),
            })
        themes = sorted(({"theme": t, "dollars": round(d, 2), "pct": pct(d)}
                         for t, d in exp["by_theme"].items()), key=lambda x: -x["dollars"])
        return {
            "base": round(base, 2),
            "invested": round(exp["total"], 2),
            "invested_pct": pct(exp["total"]),
            "defensive_pct": pct(exp["defensive"]),
            "min_defensive_pct": profile.min_defensive_pct,
            "cyclical_headroom": room(100 - profile.min_defensive_pct, exp["non_defensive"]),
            "sectors": sectors,
            "regions": regions,
            "themes": themes[:12],
        }

    def portfolio_risk(self) -> Dict[str, Any]:
        profile = state.risk_profile
        held = self.holdings(include_reserved=False)
        sleeves = self.sleeves()
        out: Dict[str, Any] = {
            "risk_factor": profile.factor,
            "risk_label": profile.label,
            "limits": {
                "max_sector_pct": profile.max_sector_pct,
                "max_crypto_pct": profile.max_crypto_pct,
                "max_us_pct": profile.max_us_pct,
                "max_intl_region_pct": profile.max_intl_region_pct,
                "intl_region_target_pct": profile.intl_region_target_pct,
                "min_defensive_pct": profile.min_defensive_pct,
                "max_pair_correlation": profile.max_pair_correlation,
                "max_position_pct": profile.max_position_notional_pct,
            },
            "sleeves": sleeves,
            "positions": len(held),
            "data": daily_bars.status(),
            "method": (f"daily log returns over the last {settings.CORRELATION_WINDOW_DAYS} "
                       f"shared trading days; parametric and historical 1-day VaR"),
        }
        stats = self._risk_stats(held)
        warnings: List[str] = []
        if stats:
            names = stats["names"]
            total = sum(held.values())
            by_sector_rc: Dict[str, float] = {}
            by_region_rc: Dict[str, float] = {}
            for n, rc in stats["risk_contrib"].items():
                m = universe.classify(n)
                by_sector_rc[m.sector] = by_sector_rc.get(m.sector, 0.0) + rc
                by_region_rc[m.region] = by_region_rc.get(m.region, 0.0) + rc
            out["risk"] = {
                "coverage_pct": round(stats["covered_notional"] / total * 100, 1) if total else 0.0,
                "days": stats["days"],
                "vol_ann_pct": _r(stats["vol_ann_pct"], 2),
                "vol_1d_dollars": _r(stats["vol_1d_dollars"], 2),
                "var95_1d": _r(stats["var95_1d"], 2),
                "cvar95_1d": _r(stats["cvar95_1d"], 2),
                "hist_var95_1d": _r(stats["hist_var95_1d"], 2),
                "worst_day": _r(stats["worst_day"], 2),
                "var95_pct_of_budget": _r(stats["var95_1d"] / sleeves["base"] * 100, 3)
                if sleeves["base"] else None,
                "diversification_ratio": _r(stats["diversification_ratio"], 3),
                "effective_bets": _r(stats["effective_bets"], 2),
                "risk_contrib_by_position": {n: round(v * 100, 1) for n, v in stats["risk_contrib"].items()},
                "risk_contrib_by_sector": {k: round(v * 100, 1) for k, v in by_sector_rc.items()},
                "risk_contrib_by_region": {k: round(v * 100, 1) for k, v in by_region_rc.items()},
                "correlation": {"symbols": names, "matrix": np.round(stats["corr"], 3).tolist()},
            }
            for k, v in by_sector_rc.items():
                if v > 0.5 and len(names) > 1:
                    warnings.append(f"{k} carries {v * 100:.0f}% of portfolio risk.")
            corr = stats["corr"]
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    if corr[i, j] >= profile.max_pair_correlation:
                        warnings.append(f"{names[i]} and {names[j]} are {corr[i, j]:.2f} "
                                        f"correlated: close to one bet.")
            if out["risk"]["var95_pct_of_budget"] and \
                    out["risk"]["var95_pct_of_budget"] > profile.max_daily_loss_pct:
                warnings.append(
                    f"1-day 95% VaR is {out['risk']['var95_pct_of_budget']:.2f}% of budget, above "
                    f"the {profile.max_daily_loss_pct}% daily-loss halt: a normal bad day would trip it.")
        elif held:
            out["risk"] = None
            warnings.append("No daily history for current holdings yet: correlation and VaR "
                            "unavailable (needs Alpaca data keys).")

        for s in sleeves["sectors"] + sleeves["regions"]:
            if s["cap_pct"] is not None and s["pct"] > s["cap_pct"] + 0.01:
                warnings.append(f"{s['sleeve']} at {s['pct']:.1f}% is over its {s['cap_pct']:.0f}% "
                                f"cap (price drift or a lowered dial); no new entries there.")
        if sleeves["invested_pct"] >= 50 and sleeves["defensive_pct"] < profile.min_defensive_pct:
            warnings.append(f"Defensive sectors are {sleeves['defensive_pct']:.1f}% of budget, below "
                            f"the {profile.min_defensive_pct:.0f}% target.")
        out["warnings"] = warnings
        return out

    def what_if(self, symbol: str) -> Dict[str, Any]:
        """Risk before/after adding a standard-size position in `symbol`."""
        profile = state.risk_profile
        base = budget_base()
        a = self.assess(symbol)
        std = base * profile.max_position_notional_pct / 100.0
        size = min(std, a.max_dollars, state.remaining_budget)
        limited_by = (f"single-position cap {profile.max_position_notional_pct:.0f}%" if size >= std
                      else a.binding if a.headroom_dollars < std
                      else "correlation halving" if a.max_dollars < std
                      else "remaining budget")
        held = self.holdings(include_reserved=False)
        before = self._risk_stats(held)
        after_book = dict(held)
        after_book[symbol] = after_book.get(symbol, 0.0) + size
        after = self._risk_stats(after_book) if size > 0 else None

        def brief(s):
            if not s:
                return None
            return {
                "vol_ann_pct": _r(s["vol_ann_pct"], 2),
                "var95_1d": _r(s["var95_1d"], 2),
                "cvar95_1d": _r(s["cvar95_1d"], 2),
                "diversification_ratio": _r(s["diversification_ratio"], 3),
                "effective_bets": _r(s["effective_bets"], 2),
                "symbol_risk_share_pct": _r(s["risk_contrib"].get(symbol, 0.0) * 100, 1)
                if symbol in s["risk_contrib"] else None,
            }

        sym_stats = self._risk_stats({symbol: 1.0})
        return {
            "symbol": symbol,
            "meta": a.meta.to_dict(),
            "assessment": a.to_dict(),
            "position_size": round(size, 2),
            "position_pct": round(size / base * 100, 2) if base else 0.0,
            "limited_by": limited_by,
            "symbol_vol_ann_pct": _r(sym_stats["vol_ann_pct"], 2) if sym_stats else None,
            "before": brief(before),
            "after": brief(after),
            "verdict": _verdict(a, before, after, size),
        }


def _r(v, n):
    return None if v is None else round(float(v), n)


def _verdict(a: Assessment, before, after, size) -> str:
    if size <= 0:
        return f"Blocked: no headroom under the {a.binding}."
    if not after:
        return "No daily history to measure the risk impact yet."
    if not before:
        return "First position: nothing to diversify against yet."
    b, c = before.get("diversification_ratio") or 0, after.get("diversification_ratio") or 0
    if c > b + 0.02:
        return f"Diversifies: diversification ratio {b:.2f} -> {c:.2f}."
    if c < b - 0.02:
        return f"Concentrates: diversification ratio {b:.2f} -> {c:.2f}."
    return f"Neutral: diversification ratio {b:.2f} -> {c:.2f}."


diversification = DiversificationManager()
