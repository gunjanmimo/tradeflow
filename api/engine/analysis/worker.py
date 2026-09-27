"""
Analysis worker: everything too heavy for the tick path, run in a separate process.

run_cycle() receives a snapshot of prices and context from the main process and
returns plain dicts. It never touches the main process's state, so it adds no
work to the event loop that handles ticks; the only main-process costs are
copying the snapshot and storing the result.

Per symbol:
  regime        engine/strategies/regime.py
  council       every library strategy's vote (engine/strategies/council.py)
  candidates    regime-suited strategies ranked for the adaptive selector
  mc            Monte Carlo P(take-profit before stop) for a trade opened now
  pair          best Engle-Granger cointegrated partner and spread statistics

Portfolio (ideas from FinceptTerminal's risk tooling; no code taken, it is AGPL):
  ledger        Sharpe / Sortino / profit factor / expectancy / max drawdown
  open_risk     variance-covariance VaR and CVaR of open positions, correlations
"""
import math
import time
from typing import Any, Dict, List, Optional

import numpy as np

# Engle-Granger 5% critical value for two series (MacKinnon 2010, large-sample).
EG_CRITICAL_5PCT = -3.34
MIN_PAIR_SAMPLES = 80
MIN_RETURN_CORR = 0.5
MC_HORIZON = 120
MC_BLOCK = 5


# -----------------------------------------------------------------------------
# Pairs
# -----------------------------------------------------------------------------

def _align(t_a, p_a, t_b, p_b):
    """B's last-known price at each of A's sample times (forward fill)."""
    idx = np.searchsorted(t_b, t_a, side="right") - 1
    ok = idx >= 0
    return p_a[ok], p_b[idx[ok]]


def engle_granger(pa: np.ndarray, pb: np.ndarray) -> Optional[Dict[str, float]]:
    """Regress log A on log B, then Dickey-Fuller on the residual spread."""
    if len(pa) < MIN_PAIR_SAMPLES or (pa <= 0).any() or (pb <= 0).any():
        return None
    y, x = np.log(pa), np.log(pb)
    vx = x.var()
    if vx <= 0 or y.var() <= 0:
        return None
    beta = float(((x - x.mean()) * (y - y.mean())).mean() / vx)
    alpha = float(y.mean() - beta * x.mean())
    e = y - alpha - beta * x
    lag, de = e[:-1], np.diff(e)
    ss = float(lag @ lag)
    if ss <= 0:
        return None
    gamma = float(lag @ de) / ss
    resid = de - gamma * lag
    se = math.sqrt(float(resid @ resid) / (len(de) - 1) / ss)
    if se <= 0 or gamma >= 0:
        return None
    t = gamma / se
    # gamma <= -1 means the spread fully reverts within one sample (white-noise
    # residual) -- the strongest cointegration, not an invalid one.
    if gamma <= -1:
        half_life = 0.5
    else:
        decay = math.log1p(gamma)
        # A gamma so small that the spread effectively never reverts (log1p
        # rounds to 0, e.g. flat or repeated prices) is not cointegration.
        # Measured live: this divided by zero and failed every analysis cycle.
        if decay >= 0:
            return None
        half_life = -math.log(2) / decay
    sd = float(e.std())
    if sd <= 0:
        return None
    rc = np.corrcoef(np.diff(y), np.diff(x))[0, 1]
    return {"alpha": alpha, "beta": beta, "mean": float(e.mean()), "std": sd,
            "return_corr": round(float(rc), 3) if np.isfinite(rc) else 0.0,
            "adf_t": round(t, 3), "half_life": round(half_life, 1),
            "z": round(float((e[-1] - e.mean()) / sd), 3), "n": len(e)}


def scan_pairs(symbols: Dict[str, Dict]) -> Dict[str, Dict]:
    """Best cointegrated partner per symbol, within the same asset class."""
    best: Dict[str, Dict] = {}
    names = [s for s, d in symbols.items() if len(d["prices"]) >= MIN_PAIR_SAMPLES]
    for a in names:
        da = symbols[a]
        for b in names:
            if a == b or symbols[b]["is_crypto"] != da["is_crypto"]:
                continue
            db = symbols[b]
            pa, pb = _align(da["times"], da["prices"], db["times"], db["prices"])
            res = engle_granger(pa, pb)
            if res is None or res["adf_t"] > EG_CRITICAL_5PCT:
                continue
            # Pre-filter used in the pairs literature: the legs must actually move
            # together. Without it, short windows produce spurious "cointegration"
            # between unrelated series.
            if res["return_corr"] < MIN_RETURN_CORR:
                continue
            if res["half_life"] > res["n"] / 4:
                continue
            if a not in best or res["adf_t"] < best[a]["adf_t"]:
                best[a] = {"partner": b, **res}
    return best


# -----------------------------------------------------------------------------
# Monte Carlo
# -----------------------------------------------------------------------------

def monte_carlo(prices: np.ndarray, price: float, sl_dist: float, tp_dist: float,
                paths: int, rng: np.random.Generator) -> Optional[Dict[str, Any]]:
    """
    Block-bootstrap recent returns into future paths and count which bracket is
    hit first. Blocks keep short-range autocorrelation that i.i.d. resampling
    would destroy. Reported twice: with the recent drift, and with drift removed
    -- the difference is how much the estimate leans on recent direction.
    """
    if len(prices) < 60 or price <= 0 or sl_dist <= 0 or tp_dist <= 0 or sl_dist >= price:
        return None
    r = np.diff(np.log(prices[-201:]))
    if len(r) <= MC_BLOCK * 2 or r.std() <= 0:
        return None
    up, dn = math.log(1 + tp_dist / price), math.log(1 - sl_dist / price)
    n_blocks = MC_HORIZON // MC_BLOCK
    starts = rng.integers(0, len(r) - MC_BLOCK + 1, size=(paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(MC_BLOCK)).reshape(paths, -1)

    def first_hits(rets):
        path = np.cumsum(rets[idx], axis=1)
        hit_up, hit_dn = path >= up, path <= dn
        t_up = np.where(hit_up.any(1), hit_up.argmax(1), np.inf)
        t_dn = np.where(hit_dn.any(1), hit_dn.argmax(1), np.inf)
        tp_first = float(np.mean(t_up < t_dn))
        sl_first = float(np.mean(t_dn < t_up))
        return tp_first, sl_first

    tp, sl = first_hits(r)
    tp0, sl0 = first_hits(r - r.mean())
    return {
        "p_tp_first": round(tp, 3), "p_sl_first": round(sl, 3),
        "p_neither": round(1 - tp - sl, 3),
        "p_tp_first_driftless": round(tp0, 3),
        "drift_edge": round(tp - tp0, 3),
        "horizon_samples": MC_HORIZON, "paths": paths,
        "rr": round(tp_dist / sl_dist, 2),
    }


# -----------------------------------------------------------------------------
# Portfolio analytics
# -----------------------------------------------------------------------------

def ledger_stats(trades: List[Dict]) -> Dict[str, Any]:
    rs = np.array([t["r"] for t in trades if t.get("r") is not None], dtype=float)
    pnl = np.array([t.get("pnl") or 0.0 for t in trades], dtype=float)
    out: Dict[str, Any] = {"n_trades": len(trades), "n_with_r": int(len(rs))}
    if len(pnl):
        eq = np.cumsum(pnl)
        peak = np.maximum.accumulate(np.concatenate(([0.0], eq)))[1:]
        gains, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
        out.update({
            "total_pnl": round(float(eq[-1]), 2),
            "win_rate": round(float((pnl > 0).mean()), 3),
            "profit_factor": round(float(gains / losses), 3) if losses > 0 else None,
            "max_drawdown": round(float((peak - eq).max()), 2),
        })
    if len(rs) >= 2:
        sd = rs.std(ddof=1)
        downside = rs[rs < 0]
        dsd = math.sqrt(float((downside ** 2).mean())) if len(downside) else 0.0
        out.update({
            "expectancy_r": round(float(rs.mean()), 3),
            # Per-trade ratios on R-multiples: comparable across sizes, not annualised.
            "sharpe_per_trade": round(float(rs.mean() / sd), 3) if sd > 0 else None,
            "sortino_per_trade": round(float(rs.mean() / dsd), 3) if dsd > 0 else None,
        })
    return out


def open_risk(symbols: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Variance-covariance VaR of open positions. Per-second volatility comes from
    tick returns divided by their time gaps, correlations from returns aligned on
    a common time grid, and horizons are sqrt-time scaled. It is an estimate from
    a short buffer; the UI labels it as such.
    """
    held = {s: d for s, d in symbols.items() if d.get("position") and len(d["prices"]) >= 30}
    if not held:
        return {"positions": 0}
    names = list(held)
    notional = np.array([held[s]["position"]["qty"] * held[s]["price"] for s in names])
    sig = []
    for s in names:
        p, t = held[s]["prices"], held[s]["times"]
        r = np.diff(np.log(p))
        dt = np.maximum(np.diff(t), 1e-3)
        sig.append(math.sqrt(float(np.mean(r ** 2 / dt))))
    sig = np.array(sig)

    corr = np.eye(len(names))
    if len(names) > 1:
        t0 = max(float(held[s]["times"][0]) for s in names)
        t1 = min(float(held[s]["times"][-1]) for s in names)
        step = max(float(np.median(np.diff(held[names[0]]["times"]))), 0.25)
        grid = np.arange(t0, t1, step)
        if len(grid) >= 20:
            cols = []
            for s in names:
                i = np.searchsorted(held[s]["times"], grid, side="right") - 1
                cols.append(np.diff(np.log(held[s]["prices"][np.maximum(i, 0)])))
            m = np.array(cols)
            with np.errstate(invalid="ignore", divide="ignore"):
                c = np.corrcoef(m)
            corr = np.where(np.isfinite(c), c, 0.0)
            np.fill_diagonal(corr, 1.0)

    cov = corr * np.outer(sig, sig)
    port_sig_1s = math.sqrt(max(float(notional @ cov @ notional), 0.0))
    phi = lambda z: math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
    out: Dict[str, Any] = {
        "positions": len(names),
        "gross_notional": round(float(notional.sum()), 2),
        "largest_position_pct": round(float(notional.max() / notional.sum() * 100), 1) if notional.sum() > 0 else 0.0,
        "method": "parametric var-covar, sqrt-time scaled from tick returns",
        "correlation": {"symbols": names, "matrix": np.round(corr, 3).tolist()},
    }
    for label, secs in (("1m", 60), ("1h", 3600)):
        s = port_sig_1s * math.sqrt(secs)
        out[f"var95_{label}"] = round(1.645 * s, 2)
        out[f"var99_{label}"] = round(2.326 * s, 2)
        out[f"cvar95_{label}"] = round(s * phi(1.645) / 0.05, 2)
    # Diversification benefit: how much less risk than if all were perfectly correlated
    undiversified = float((notional * sig).sum())
    out["diversification_ratio"] = round(undiversified / port_sig_1s, 3) if port_sig_1s > 0 else None
    return out


# -----------------------------------------------------------------------------
# Entry point (runs in the child process)
# -----------------------------------------------------------------------------

def run_cycle(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    t_start = time.perf_counter()
    from core.state import state, PriceTick
    from engine.strategies import performance, regime as regime_mod
    from engine.strategies.base import StrategyContext
    from engine.strategies.council import convene
    from engine.strategies.indicators import PriceSeries

    symbols: Dict[str, Dict] = snapshot["symbols"]
    at = snapshot["at"]
    rng = np.random.default_rng()

    # This process's `state` is a private scratch copy. Load what strategies read.
    performance.install(snapshot.get("perf", {}))
    state.latest_prices = {
        s: PriceTick(symbol=s, price=d["price"], bid=d["price"], ask=d["price"], volume=0.0)
        for s, d in symbols.items()
    }

    errors: Dict[str, str] = {}
    try:
        pairs = scan_pairs(symbols)
    except Exception as e:
        # Pairs are one input among several; never let them take the cycle down.
        errors["_pairs"] = f"{type(e).__name__}: {e}"
        pairs = {}
    state.analysis = {s: {"pair": pairs.get(s), "at": at} for s in symbols}

    results: Dict[str, Dict] = {}
    for sym, d in symbols.items():
      try:
        series = PriceSeries(d["prices"], d["volumes"])
        ctx = StrategyContext(
            symbol=sym, price=d["price"], quant=d["quant"], sentiment=d["sentiment"],
            consensus=d["consensus"], is_crypto=d["is_crypto"], series=series,
        )
        ctx._min_buy_prob = snapshot["min_buy_prob"]
        report = convene(ctx)
        reg = report.regime
        suited = [v for v in report.votes if v.suited]
        suited.sort(key=lambda v: (v.would_enter, v.bias * v.weight), reverse=True)
        results[sym] = {
            "at": at,
            "regime": reg,
            "council": report.to_dict(),
            "candidates": [v.strategy for v in suited],
            "mc": monte_carlo(d["prices"], d["price"], d["sl_dist"], d["tp_dist"],
                              snapshot["mc_paths"], rng),
            "pair": pairs.get(sym),
        }
      except Exception as e:
        # One malformed symbol must not blank out analysis for every other one.
        errors[sym] = f"{type(e).__name__}: {e} @ {_where(e)}"

    portfolio = {"at": at}
    for key, fn, arg in (("ledger", ledger_stats, snapshot.get("trades", [])),
                         ("open_risk", open_risk, symbols)):
        try:
            portfolio[key] = fn(arg)
        except Exception as e:
            errors[f"_{key}"] = f"{type(e).__name__}: {e} @ {_where(e)}"
            portfolio[key] = {}
    return {"symbols": results, "portfolio": portfolio, "errors": errors,
            "compute_ms": round((time.perf_counter() - t_start) * 1000, 1)}


def _where(e: BaseException) -> str:
    """file:line of the innermost frame, so a logged error is actionable."""
    import traceback
    tb = traceback.extract_tb(e.__traceback__)
    return f"{tb[-1].filename.rsplit('/', 1)[-1]}:{tb[-1].lineno}" if tb else "?"
