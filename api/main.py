import asyncio
import re
import time
import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.config import settings
from core.version import __version__
from core.state import state
from sentiment.router import sentiment_service
from engine.executor import executor
from feeds.alpaca_stream import market_stream
from feeds.expert_watcher import expert_watcher, BENCHMARK_EXPERTS
from feeds.news_feed import news_feed
from feeds.multi_source_aggregator import trend_aggregator
from engine.market_scheduler import market_scheduler
from engine.sentinel_agent import sentinel_registry
from engine import brackets
from core import risk_profile as risk_profile_mod
from engine.strategies import registry as strategy_registry
from memory.agent_memory import agent_memory
from core.latency import latency, monitor_event_loop
from engine.analysis.service import analysis_service
from core.capital_plan import capital_plan
from core.market_filter import market_filter
from engine.discovery import discovery, WEIGHTS as DISCOVERY_WEIGHTS
from engine.diversification import diversification
from engine.portfolio_manager import portfolio_manager
from engine.fleet import fleet
from engine.trend import board as trend_board
from feeds.daily_bars import daily_bars

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("tradeflow.api")

# Active WebSocket UI clients
active_connections: List[WebSocket] = []

async def broadcast_telemetry():
    """Streams live engine telemetry to all connected frontend dashboards at 4Hz"""
    while True:
        try:
            if active_connections:
                # Refreshed at most once a second, yielding between stages.
                latency_snapshot = await latency.refresh()
                t_build = time.perf_counter_ns()
                payload = {
                    "is_trading_active": state.is_trading_active,
                    "account": state.account_info,
                    "positions": state.active_positions,
                    "watchlist": list(state.watchlist),
                    "latest_prices": {k: {"price": v.price, "bid": v.bid, "ask": v.ask, "vol": v.volume} for k, v in state.latest_prices.items()},
                    "prices": {k: {"price": v.price, "bid": v.bid, "ask": v.ask, "vol": v.volume} for k, v in state.latest_prices.items()},
                    "quant_metrics": {k: {"rsi": v.rsi, "ema_fast": v.ema_fast, "ema_slow": v.ema_slow, "atr": v.atr, "spread": v.spread} for k, v in state.quant_metrics.items()},
                    "sentiment": {
                        sym: {
                            "pos_prob": r.pos_prob, "neg_prob": r.neg_prob,
                            "open": r.open, "headline": r.headline,
                            "n_headlines": r.n_headlines, "agreement": r.agreement,
                            "sources": list(r.sources), "is_stale": r.is_stale,
                            "is_tradeable": r.is_tradeable,
                            "age_s": state.sentiment_age(sym),
                        }
                        for sym in list(state.watchlist)
                        for r in [state.get_sentiment(sym)]
                    },
                    "recent_decisions": [
                        {"symbol": d.symbol, "action": d.action, "buy_prob": d.buy_prob, "hold_prob": getattr(d, "hold_prob", 0.0), "sell_prob": d.sell_prob, "close": d.close, "reason": d.reason, "time": d.timestamp}
                        for d in list(state.recent_decisions)[-10:]
                    ],
                    "aggregated_trends": {
                        k: {
                            "symbol": v.symbol,
                            "consensus_score": v.consensus_score,
                            "sentiment_bias": v.sentiment_bias,
                            "thesis": v.overall_thesis,
                            "contributing_sources": list(v.contributing_sources),
                            "total_weight": v.total_weight,
                            "is_tradeable": v.is_tradeable,
                            "updated_at": v.updated_at,
                            "signals": {
                                name: {
                                    "score": sig.conviction_score,
                                    "sentiment": sig.sentiment,
                                    "details": sig.details,
                                }
                                for name, sig in v.signals.items()
                            },
                        }
                        for k, v in trend_aggregator.aggregated_trends.items()
                    },
                    "source_health": trend_aggregator.health(),
                    "strategy_routing": {
                        sym: strategy_registry.resolve(
                            sym, state.strategy_class_defaults, state.strategy_overrides
                        ).name
                        for sym in list(state.watchlist)
                    },
                    "strategy_class_defaults": dict(state.strategy_class_defaults),
                    "gate_detail": dict(state.last_gate_detail),
                    "news_status": news_feed.status(),
                    "market_clock": market_scheduler.get_market_status(),
                    "budget": state.budget_snapshot(),
                    "markets": market_filter.snapshot(list(state.watchlist)),
                    "daily_pnl": state.daily_pnl,
                    "risk": _portfolio_risk_snapshot(),
                    "sentinel_bots": sentinel_registry.to_dict(),
                    "recent_trades": list(state.recent_trades)[-50:],
                    "logs": list(state.logs)[-25:],
                    "analysis": _analysis_summary(),
                    "portfolio_analytics": state.portfolio_analytics,
                    "analysis_status": analysis_service.status(),
                    "capital_plan": capital_plan.plan.to_dict(),
                    "diversification": _diversification_brief(),
                    "manager": portfolio_manager.snapshot(),
                    "fleet": fleet.snapshot(),
                    "latency": latency_snapshot,
                    "server_time": time.time(),
                }
                latency.record_ns("telemetry_build", t_build)
                dead_connections = []
                for ws in active_connections:
                    try:
                        await ws.send_json(payload)
                    except Exception:
                        dead_connections.append(ws)
                for dead in dead_connections:
                    if dead in active_connections:
                        active_connections.remove(dead)
            await asyncio.sleep(settings.TELEMETRY_BROADCAST_MS / 1000.0)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in telemetry broadcast: {e}")
            await asyncio.sleep(1.0)

_summary_cache: Dict[str, Any] = {"at": None, "data": {}}


def _analysis_summary() -> Dict[str, Any]:
    """
    Compact per-symbol view of the worker's results, for the dashboard.
    Rebuilt only when a new worker cycle lands (1Hz), not on every 4Hz frame.
    """
    if _summary_cache["at"] == state.analysis_at:
        return _summary_cache["data"]
    out = {}
    for sym, a in list(state.analysis.items()):
        c = a.get("council") or {}
        reg = a.get("regime") or {}
        mc = a.get("mc") or {}
        pair = a.get("pair") or {}
        out[sym] = {
            "regime": reg.get("label"),
            "regime_confidence": reg.get("confidence"),
            "verdict": c.get("verdict"),
            "consensus": c.get("deciding_consensus"),
            "n_bullish": c.get("n_bullish"),
            "n_bearish": c.get("n_bearish"),
            "n_voters": c.get("n_voters"),
            "recommended": c.get("recommended"),
            "top_candidates": (a.get("candidates") or [])[:3],
            "mc_p_tp_first": mc.get("p_tp_first"),
            "mc_drift_edge": mc.get("drift_edge"),
            "pair_partner": pair.get("partner"),
            "pair_z": pair.get("z"),
            "at": a.get("at"),
        }
    _summary_cache["at"], _summary_cache["data"] = state.analysis_at, out
    return out


def _diversification_brief() -> Dict[str, Any]:
    """Headline numbers from the last discovery cycle's risk report (refreshed ~1/min)."""
    r = discovery.risk_report or {}
    risk = r.get("risk") or {}
    sl = r.get("sleeves") or {}
    return {
        "warnings": r.get("warnings", []),
        "invested_pct": sl.get("invested_pct"),
        "defensive_pct": sl.get("defensive_pct"),
        "var95_pct_of_budget": risk.get("var95_pct_of_budget"),
        "diversification_ratio": risk.get("diversification_ratio"),
        "effective_bets": risk.get("effective_bets"),
        "candidates": len(discovery.candidates),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Tradeflow Continuous Trading Engine...")
    
    # 0. Agent memory (optional; failure here must not stop the engine)
    if settings.MEMORY_ENABLED:
        try:
            await agent_memory.initialize()
        except Exception as e:
            logger.warning(f"Agent memory unavailable, continuing without it: {e}")

    # 0b. Capital plan (classic / stair). Loaded before trading can start so the
    # budget is right from the first tick and banked income survives restarts.
    capital_plan.load()
    from core.pnl_ledger import pnl_ledger
    pnl_ledger.roll(state.trading_day)
    state.realized_pnl_today = pnl_ledger.realized_today()
    state.harvested_today = pnl_ledger.harvested_today()

    # 1. Initialize sentiment: Jev (TypeSafe) first, Laya loaded as fallback
    await sentiment_service.initialize()

    # 2. Connect to Alpaca Trading API / Paper Sandbox
    await executor.initialize()

    # 3. Start Expert Watcher (Universe generation)
    await expert_watcher.start()

    # 4. Start News Ingestion
    await news_feed.start()

    # 5. Start Sub-Second Market Stream
    await market_stream.start()

    # 6. Start Market Session Cron Scheduler
    await market_scheduler.start()

    # 7. Start 4-Pillar Trend Aggregator (SEC + eToro + Dub/Public + StockTwits)
    await trend_aggregator.start()

    # 8. Start Continuous Broker Account & Position Sync
    sync_task = asyncio.create_task(executor.start_sync_loop())

    # 9. Start Telemetry Broadcaster
    telemetry_task = asyncio.create_task(broadcast_telemetry())

    # 10. Off-process analytics (council, regime, Monte Carlo, pairs, risk) and
    # the event-loop lag monitor that proves they are not slowing the tick path.
    await analysis_service.start()

    # 11. Discovery pool + diversification risk analysis (off the order path).
    await discovery.start()
    loop_monitor_task = asyncio.create_task(monitor_event_loop())

    # Trade scorer: import torch and load the model in a thread now (~0.8s), so
    # the manager's first cycle does not stall the event loop doing it.
    if settings.ML_MODE != "off":
        from ml.model import scorer
        await asyncio.get_running_loop().run_in_executor(None, lambda: scorer.ready)

    # 12. Agent fleet: trend analyst (reads the time series first), curator,
    # trader (the portfolio manager) and position manager.
    await fleet.start()

    logger.info("Tradeflow sub-second trading engine is LIVE!")
    yield

    logger.info("Shutting down Tradeflow engine...")
    telemetry_task.cancel()
    sync_task.cancel()
    loop_monitor_task.cancel()
    await fleet.stop()
    await analysis_service.stop()
    await discovery.stop()
    await trend_aggregator.stop()
    await market_scheduler.stop()
    await market_stream.stop()
    await news_feed.stop()
    await expert_watcher.stop()
    await sentiment_service.close()

app = FastAPI(
    title="Tradeflow Quant API",
    description="Sub-second Continuous Quant Trading Engine with Embedded Laya Decision Model",
    version=__version__,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Models
class WatchlistAddRequest(BaseModel):
    symbol: str

class MarketToggleRequest(BaseModel):
    market: str
    enabled: bool

class SymbolToggleRequest(BaseModel):
    symbol: str
    enabled: bool

class BudgetUpdateRequest(BaseModel):
    allocated_capital: float

class RiskFactorRequest(BaseModel):
    risk_factor: int

class CapitalPlanRequest(BaseModel):
    mode: str                          # "classic" | "stair"
    deposit: Optional[float] = None    # required for stair
    deploy_pct: float = 0.5
    target_multiple: float = 2.0
    harvest_pct: float = 0.5

class StrategyClassRequest(BaseModel):
    asset_class: str = "equity"   # the only asset class
    strategy: str

class SymbolRequest(BaseModel):
    symbol: str

class StrategyOverrideRequest(BaseModel):
    symbol: str
    strategy: Optional[str] = None   # None clears the override

class PnLCalculatorRequest(BaseModel):
    target_daily_profit: float = 100.0
    max_daily_loss: float = 50.0
    planned_trades: int = 5
    win_rate_pct: float = 60.0
    reward_risk_ratio: float = 2.0
    risk_per_trade: float = 20.0
    budget: Optional[float] = None

@app.get("/api/status")
async def get_status():
    return {
        "status": "online",
        "version": __version__,
        "is_trading_active": state.is_trading_active,
        "is_connected_to_alpaca": executor.is_connected,
        "mode": "ALPACA_PAPER" if not executor.is_mock_mode else "SIMULATED_PAPER",
        "account": state.account_info,
        "budget": state.budget_snapshot(),
        "daily_pnl": state.daily_pnl,
        "active_positions_count": len(state.active_positions),
        "watchlist_count": len(state.watchlist),
        "sentiment_backend": sentiment_service.status(),
    }

def _portfolio_risk_snapshot() -> Dict[str, Any]:
    """
    Whole-portfolio risk as it stands right now, plus what the current dial would
    do on the NEXT entry.

    The forward-looking part is what makes the slider feel immediate: the operator
    sees the size and stop the next trade will use the moment they move it, rather
    than having to wait for a fill to find out.
    """
    profile = state.risk_profile
    budget = state.hard_cap or 1.0

    # Risk actually at stake across open positions: distance to each stop.
    open_risk = 0.0
    per_position = []
    for sym, pos in state.active_positions.items():
        qty = float(pos.get("qty") or 0.0)
        entry = float(pos.get("avg_entry_price") or 0.0)
        cur = float(pos.get("current_price") or entry)
        sl = pos.get("stop_loss")
        try:
            sl = float(sl) if sl is not None else None
        except (TypeError, ValueError):
            sl = None
        # Risk from here to the stop, not from entry: capital already lost is sunk,
        # and a trailed stop may have locked in a profit (negative remaining risk).
        risk = round((cur - sl) * qty, 2) if (sl is not None and qty > 0) else 0.0
        open_risk += max(0.0, risk)
        per_position.append({
            "symbol": sym,
            "qty": qty,
            "notional": round(qty * cur, 2),
            "risk_to_stop": risk,
            "risk_pct_of_budget": round(risk / budget * 100, 3),
            "unrealized_pl": pos.get("unrealized_pl", 0.0),
            "stop_loss": sl,
            "take_profit": pos.get("take_profit"),
        })

    # What the next entry looks like under the current dial, using a mid-priced
    # watchlist symbol we actually have a quote and volatility reading for.
    next_trade = None
    for sym in sorted(state.watchlist):
        tick = state.latest_prices.get(sym)
        q = state.quant_metrics.get(sym)
        if tick and q and tick.price > 0:
            sl, tp, dist = brackets.derive(tick.price, q.atr)
            equity = min(float(state.account_info.get("equity", budget)), budget)
            risk_dollars = equity * (profile.risk_per_trade_pct / 100.0)
            notional = min(
                (risk_dollars / dist) * tick.price if dist > 0 else 0.0,
                equity * (profile.max_position_notional_pct / 100.0),
                state.remaining_budget,
            )
            next_trade = {
                "example_symbol": sym,
                "price": tick.price,
                "stop_loss": sl,
                "take_profit": tp,
                "stop_pct": round(dist / tick.price * 100, 3),
                "target_risk_dollars": round(risk_dollars, 2),
                "position_notional": round(notional, 2),
                "actual_risk_dollars": round(min(risk_dollars, notional * dist / tick.price), 2),
                "reward_risk_ratio": profile.reward_risk_ratio,
            }
            break

    return {
        "risk_factor": profile.factor,
        "risk_label": profile.label,
        "open_positions": len(state.active_positions),
        "max_positions": profile.max_concurrent_positions,
        "total_notional": state.total_position_exposure,
        "total_risk_to_stops": round(open_risk, 2),
        "total_risk_pct_of_budget": round(open_risk / budget * 100, 3),
        "realized_pnl_today": state.realized_pnl_today,
        "harvested_income_today": state.harvested_today,
        "daily_loss_pct": state.daily_loss_pct,
        "daily_loss_limit_pct": profile.max_daily_loss_pct,
        "drawdown_pct": state.drawdown_pct,
        "drawdown_limit_pct": profile.max_drawdown_pct,
        "peak_equity": state.peak_equity,
        "halt_reason": state.halt_reason,
        "positions": per_position,
        "next_trade_preview": next_trade,
    }


@app.get("/api/memory")
async def memory_status():
    """Memory layer health plus per-strategy outcome history."""
    from memory.hydra_client import hydra
    strategies = {}
    if agent_memory.enabled:
        for s in strategy_registry.available():
            strategies[s.name] = (await agent_memory.recall_strategy(s.name)).to_dict()
    return {
        "memory": agent_memory.status(),
        "hydra": await hydra.health() if settings.MEMORY_ENABLED else {"available": False, "disabled": True},
        "by_strategy": strategies,
    }


@app.get("/api/memory/symbol/{symbol:path}")
async def memory_symbol(symbol: str):
    """What memory knows about one symbol, including the live setup bucket."""
    sym = symbol.upper()
    q = state.quant_metrics.get(sym)
    sent = state.get_sentiment(sym)
    strat = strategy_registry.resolve(sym, state.strategy_class_defaults,
                                     state.strategy_overrides)
    tick = state.latest_prices.get(sym)
    atr_pct = (q.atr / tick.price * 100) if (q and q.atr and tick and tick.price) else None
    ctx = await agent_memory.context_for(
        symbol=sym, strategy=strat.name,
        rsi=q.rsi if q else None,
        trend_score=strat._trend_score(q, tick.price) if (q and tick) else None,
        sent_pos=sent.pos_prob, sent_n=sent.n_headlines, atr_pct=atr_pct,
    )
    return {"symbol": sym, "strategy": strat.name, "context": ctx}


@app.get("/api/memory/bot/{bot_id}")
async def memory_bot(bot_id: str):
    """One agent's own trade history -- its personal memory."""
    return {"bot_id": bot_id, "trades": await agent_memory.recall_bot(bot_id)}


@app.get("/api/strategies")
async def list_strategies():
    """Available strategies, current per-class assignment, and live per-symbol routing."""
    routing = {}
    for sym in sorted(state.watchlist):
        strat = strategy_registry.resolve(sym, state.strategy_class_defaults,
                                         state.strategy_overrides)
        routing[sym] = {
            "strategy": strat.name,
            "asset_class": strategy_registry.asset_class(sym),
            "is_override": sym in state.strategy_overrides,
            "requires_sentiment": strat.requires_sentiment,
        }
    return {
        **strategy_registry.describe_all(),
        "active_class_defaults": dict(state.strategy_class_defaults),
        "overrides": dict(state.strategy_overrides),
        "routing": routing,
    }


@app.post("/api/strategies/class")
async def set_class_strategy(req: StrategyClassRequest):
    """Assigns the default strategy for US equities (the only asset class)."""
    klass = req.asset_class.lower().strip()
    if klass != "equity":
        raise HTTPException(status_code=400, detail="asset_class must be 'equity' (US stocks only)")
    if strategy_registry.get(req.strategy) is None:
        raise HTTPException(status_code=400,
            detail=f"Unknown strategy '{req.strategy}'. Available: "
                   f"{[s.name for s in strategy_registry.available()]}")

    old = state.strategy_class_defaults.get(klass)
    state.strategy_class_defaults[klass] = req.strategy
    msg = f"{klass.capitalize()} strategy changed: {old} -> {req.strategy}"
    state.log_event("STRATEGY", msg)
    logger.info(msg)
    return {"status": "success", "asset_class": klass, "strategy": req.strategy,
            "previous": old, "class_defaults": dict(state.strategy_class_defaults)}


@app.post("/api/strategies/override")
async def set_symbol_strategy(req: StrategyOverrideRequest):
    """Per-symbol strategy override, or clears it when strategy is null."""
    sym = req.symbol.upper().strip()
    if req.strategy is None:
        removed = state.strategy_overrides.pop(sym, None)
        state.log_event("STRATEGY", f"Cleared strategy override for {sym} (was {removed})")
        return {"status": "success", "symbol": sym, "strategy": None, "cleared": removed}

    if strategy_registry.get(req.strategy) is None:
        raise HTTPException(status_code=400, detail=f"Unknown strategy '{req.strategy}'")
    klass = strategy_registry.asset_class(sym)
    if not strategy_registry.is_compatible(req.strategy, klass):
        raise HTTPException(status_code=400,
            detail=f"Strategy '{req.strategy}' does not support {klass} assets")

    state.strategy_overrides[sym] = req.strategy
    state.log_event("STRATEGY", f"{sym} strategy override set to {req.strategy}")
    return {"status": "success", "symbol": sym, "strategy": req.strategy}


@app.get("/api/quant/library")
async def quant_library():
    """The quant strategy library, each strategy's regimes/source, and realised track record."""
    from engine.strategies import performance
    return {
        "strategies": [s.describe() for s in strategy_registry.available() if s.council_member],
        "track_record": performance.stats(),
        "council_settings": {
            "analysis_interval_s": settings.ANALYSIS_INTERVAL_SECONDS,
            "adaptive_top_k": settings.ADAPTIVE_TOP_K,
            "mc_min_tp_first_prob": settings.MC_MIN_TP_FIRST_PROB,
            "entry_check": settings.COUNCIL_ENTRY_CHECK,
            "exit_check": settings.COUNCIL_EXIT_CHECK,
            "min_voters": settings.COUNCIL_MIN_VOTERS,
            "veto_consensus": settings.COUNCIL_VETO_CONSENSUS,
            "exit_consensus": settings.COUNCIL_EXIT_CONSENSUS,
        },
    }


@app.get("/api/quant/regimes")
async def quant_regimes():
    """Live market regime per symbol and the ranked suited strategies (from the worker)."""
    return {
        "regimes": {
            sym: {**(a.get("regime") or {}), "suited_strategies": a.get("candidates") or [],
                  "age_s": round(time.time() - a.get("at", 0.0), 1)}
            for sym, a in sorted(state.analysis.items())
        },
        "worker": analysis_service.status(),
    }


@app.get("/api/quant/analyze/{symbol:path}")
async def quant_analyze(symbol: str):
    """Full worker report for one symbol: council votes, regime, Monte Carlo, pair."""
    sym = symbol.upper().strip()
    a = state.analysis.get(sym)
    if a is None:
        raise HTTPException(status_code=404,
            detail=f"No analysis for {sym} yet (worker runs every {settings.ANALYSIS_INTERVAL_SECONDS}s)")
    return {"symbol": sym, "age_s": round(time.time() - a.get("at", 0.0), 2), **a}


@app.get("/api/quant/portfolio")
async def quant_portfolio():
    """Portfolio analytics: ledger performance ratios and open-position VaR/CVaR."""
    return state.portfolio_analytics or {"status": "warming up"}


@app.get("/api/ml")
async def get_ml():
    """Trade scorer: mode, whether a model is loaded, its holdout results, and experience recorded."""
    from ml.model import scorer
    from ml.experience import experience
    return {"mode": settings.ML_MODE, "model": scorer.status(), "experience": experience.status()}


@app.get("/api/latency")
async def get_latency():
    """Latency percentiles per stage (hot path, broker, feed, event loop, worker)."""
    return {**latency.snapshot(max_age_s=0.0), "worker": analysis_service.status()}


@app.get("/api/manager")
async def get_manager():
    """What the portfolio manager is doing: state, idle budget and why, ranked picks."""
    return portfolio_manager.snapshot()


@app.get("/api/fleet")
async def get_fleet():
    """Every agent's status card, and the trend read for each watched or held symbol."""
    return fleet.snapshot()


@app.get("/api/trend/{symbol:path}")
async def get_trend(symbol: str):
    """Full multi-horizon trend read for one symbol."""
    return trend_board.read(symbol.upper().strip()).to_dict()


@app.get("/api/gates")
async def get_gates():
    """
    Why each symbol is or is not being traded right now.

    Diagnostic surface for 'the bot is not trading': names the exact gate that
    refused, per symbol, instead of leaving it to be inferred from logs.
    """
    blocked_counts = {}
    for detail in state.last_gate_detail.values():
        if detail.get("blocked_by"):
            blocked_counts[detail["blocked_by"]] = blocked_counts.get(detail["blocked_by"], 0) + 1
    return {
        "per_symbol": state.last_gate_detail,
        "blocked_by_counts": blocked_counts,
        "entry_bar": state.risk_profile.min_buy_prob,
        "risk_factor": state.risk_factor,
    }


@app.get("/api/risk-factor")
async def get_risk_factor():
    """Current risk dial, the limits it implies, and the full level table for the UI."""
    return {
        "risk_factor": state.risk_factor,
        "profile": state.risk_profile.to_dict(),
        "levels": risk_profile_mod.all_levels(),
        "portfolio": _portfolio_risk_snapshot(),
    }


@app.post("/api/risk-factor")
async def set_risk_factor(req: RiskFactorRequest):
    """
    Sets the portfolio-wide risk dial (1-10).

    Takes effect immediately: every limit is read live from state.risk_profile on
    each evaluation, so the next tick already sizes and gates with the new value.
    Existing positions keep the brackets they were opened with -- moving the dial
    never retroactively widens a stop on capital already at risk.
    """
    new_factor = risk_profile_mod.clamp_factor(req.risk_factor)
    if new_factor != int(req.risk_factor):
        raise HTTPException(
            status_code=400,
            detail=f"risk_factor must be an integer between "
                   f"{risk_profile_mod.MIN_RISK_FACTOR} and {risk_profile_mod.MAX_RISK_FACTOR}"
        )

    old_factor = state.risk_factor
    if new_factor == old_factor:
        return {
            "status": "unchanged",
            "risk_factor": state.risk_factor,
            "profile": state.risk_profile.to_dict(),
            "portfolio": _portfolio_risk_snapshot(),
        }

    state.risk_factor = new_factor
    summary = risk_profile_mod.describe_change(old_factor, new_factor)
    state.log_event("RISK_FACTOR", summary)
    logger.info(summary)

    return {
        "status": "success",
        "risk_factor": state.risk_factor,
        "previous_risk_factor": old_factor,
        "profile": state.risk_profile.to_dict(),
        "summary": summary,
        "portfolio": _portfolio_risk_snapshot(),
    }


@app.get("/api/capital-plan")
async def get_capital_plan():
    """Current capital mode, stair ladder progress, and stage history."""
    from core.capital_plan import min_stage_capital
    return {**capital_plan.plan.to_dict(),
            "min_trading_capital": min_stage_capital(state.risk_profile.max_position_notional_pct)}


@app.post("/api/capital-plan")
async def set_capital_plan(req: CapitalPlanRequest):
    """
    Switches between classic and stair capital management. Takes effect on the
    next evaluation. Switching stair ON starts a fresh ladder from `deposit`.
    """
    mode = req.mode.lower().strip()
    if mode == "classic":
        return {"status": "success", "plan": capital_plan.disable_stair()}
    if mode != "stair":
        raise HTTPException(status_code=400, detail="mode must be 'classic' or 'stair'")
    if req.deposit is None or req.deposit <= 0:
        raise HTTPException(status_code=400, detail="deposit is required for stair mode")
    try:
        plan = capital_plan.enable_stair(req.deposit, req.deploy_pct,
                                         req.target_multiple, req.harvest_pct)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "success", "plan": plan}


@app.post("/api/budget")
async def update_budget(req: BudgetUpdateRequest):
    if capital_plan.plan.mode == "stair":
        raise HTTPException(status_code=409,
            detail="Stair mode manages the trading budget from the ladder. Switch to classic to set it manually.")
    if req.allocated_capital < 100.0:
        raise HTTPException(status_code=400, detail="Minimum trading budget is $100.00")
    if req.allocated_capital > 1000000.0:
        raise HTTPException(status_code=400, detail="Trading budget cannot exceed account limits")
    if req.allocated_capital + 1e-6 < state.committed_capital:
        raise HTTPException(
            status_code=409,
            detail=(f"Cannot set a ${req.allocated_capital:,.2f} cap while ${state.committed_capital:,.2f} "
                    "is committed to open or pending bot positions. Close positions first."),
        )

    old_val = state.allocated_capital
    capital_plan.set_classic_budget(req.allocated_capital)
    state.log_event("BUDGET", f"Trading capital budget updated: ${old_val:,.2f} → ${state.allocated_capital:,.2f}")
    return {
        "status": "success",
        "budget": state.budget_snapshot(),
    }


@app.get("/api/daily-pnl")
async def get_daily_pnl():
    """Bot-only daily P&L: closed-trade results, live open-position P&L, scenarios, and history."""
    from core.pnl_ledger import pnl_ledger
    return {
        "summary": state.daily_pnl,
        "history": pnl_ledger.get_history(31),
        "scenarios": pnl_ledger.calculate_scenarios(state.active_positions, state.daily_pnl.get("realized_pnl", 0.0)),
        "budget": state.budget_snapshot(),
    }


@app.post("/api/daily-pnl/calculator")
async def calculate_daily_pnl_projection(req: PnLCalculatorRequest):
    """Calculates risk-reward expectancy, daily target progress, and trade projections."""
    from core.pnl_ledger import pnl_ledger
    budget = req.budget if (req.budget and req.budget > 0) else state.hard_cap
    current_net = state.daily_pnl.get("net_pnl", 0.0)
    projections = pnl_ledger.calculate_projections(
        target_daily_profit=req.target_daily_profit,
        max_daily_loss=req.max_daily_loss,
        planned_trades=req.planned_trades,
        win_rate_pct=req.win_rate_pct,
        reward_risk_ratio=req.reward_risk_ratio,
        risk_per_trade=req.risk_per_trade,
        budget=budget,
        current_net_pnl=current_net,
    )
    return {
        "status": "success",
        "projections": projections,
        "summary": state.daily_pnl,
        "scenarios": pnl_ledger.calculate_scenarios(state.active_positions, state.daily_pnl.get("realized_pnl", 0.0)),
    }

@app.get("/api/watchlist")
async def get_watchlist():
    items = []
    for sym in state.watchlist:
        tick = state.latest_prices.get(sym)
        quant = state.quant_metrics.get(sym)
        sentiment = state.get_sentiment(sym)
        items.append({
            "symbol": sym,
            "price": tick.price if tick else None,
            "rsi": quant.rsi if quant else None,
            "ema_fast": quant.ema_fast if quant else None,
            "ema_slow": quant.ema_slow if quant else None,
            "atr": quant.atr if quant else None,
            "laya_pos": sentiment.pos_prob,
            "laya_neg": sentiment.neg_prob,
            "laya_open": sentiment.open,
            "latest_headline": sentiment.headline
        })
    return {"watchlist": items}

@app.post("/api/watchlist")
async def add_to_watchlist(req: WatchlistAddRequest):
    sym = req.symbol.upper().strip()
    # US equity tickers only (letters, optionally one class suffix like BRK.B).
    # Pairs such as BTC/USD are refused: the platform does not trade crypto.
    if not re.fullmatch(r"[A-Z]{1,5}(\.[A-Z])?", sym):
        raise HTTPException(status_code=400,
                            detail=f"'{sym}' is not a US stock ticker. Only US equities are traded.")
    state.watchlist.add(sym)
    await market_stream.ensure_stock_subscription(sym)
    state.log_event("WATCHLIST", f"Added {sym} to active watchlist")
    return {"message": f"{sym} added to watchlist", "watchlist": list(state.watchlist)}

@app.delete("/api/watchlist/{symbol}")
async def remove_from_watchlist(symbol: str):
    sym = symbol.upper().strip()
    if sym in state.watchlist:
        state.watchlist.remove(sym)
        state.log_event("WATCHLIST", f"Removed {sym} from active watchlist")
        discovery.on_unwatched(sym)
    return {"message": f"{sym} removed from watchlist", "watchlist": list(state.watchlist)}

@app.get("/api/markets")
async def get_markets():
    return market_filter.snapshot(list(state.watchlist))

@app.post("/api/markets")
async def toggle_market(req: MarketToggleRequest):
    try:
        market_filter.set_market(req.market, req.enabled)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    state.log_event("MARKETS", f"{req.market.capitalize()} trading switched {'ON' if req.enabled else 'OFF'} (new entries only; open positions still exit normally)")
    return market_filter.snapshot(list(state.watchlist))

@app.post("/api/markets/symbol")
async def toggle_symbol(req: SymbolToggleRequest):
    market_filter.set_symbol(req.symbol, req.enabled)
    state.log_event("MARKETS", f"{req.symbol.upper()} switched {'ON' if req.enabled else 'OFF'} (new entries only)")
    return market_filter.snapshot(list(state.watchlist))

@app.get("/api/positions")
async def get_positions():
    return {"positions": state.active_positions}

@app.get("/api/trades")
async def get_trades():
    return {"trades": list(state.recent_trades)}

@app.get("/api/experts")
async def get_experts():
    return {"experts": BENCHMARK_EXPERTS}

@app.post("/api/kill-switch")
async def trigger_kill_switch():
    """Emergency Stop: halts trading and closes all positions"""
    await executor.emergency_close_all()
    return {"status": "EMERGENCY_STOP_ACTIVATED", "is_trading_active": state.is_trading_active}

@app.post("/api/toggle-trading")
async def toggle_trading():
    state.is_trading_active = not state.is_trading_active
    status_str = "ACTIVE" if state.is_trading_active else "PAUSED"
    state.log_event("SYSTEM", f"Trading engine status toggled to {status_str}")
@app.get("/api/market-clock")
async def get_market_clock():
    return market_scheduler.get_market_status()

@app.post("/api/schedule/run-now")
async def trigger_morning_routine(market: str = "US"):
    """Manually triggers the Morning Bell routine on demand"""
    await market_scheduler.run_morning_routine(market.upper())
@app.get("/api/trends")
async def get_market_trends():
    """Returns the 4-pillar consensus trends (SEC + eToro + Dub/Public + StockTwits)"""
    return {"trends": trend_aggregator.aggregated_trends}

@app.post("/api/trends/sync")
async def trigger_trend_sync():
    """Manually re-aggregates all 4 intelligence feeds on demand"""
    await trend_aggregator.aggregate_all_sources()
    return {"status": "SUCCESS", "count": len(trend_aggregator.aggregated_trends)}

# ---------------------------------------------------------------------------
# Discovery & diversification
# ---------------------------------------------------------------------------

@app.get("/api/discovery")
async def get_discovery(status: Optional[str] = None, sector: Optional[str] = None,
                        region: Optional[str] = None, limit: int = 100):
    """Scored candidate pool, sleeve momentum ranking, and service status."""
    return {
        "status": discovery.status(),
        "candidates": discovery.list(status, sector, region, max(1, min(limit, 300))),
        "sleeve_momentum": discovery.sleeve_momentum,
        "sentiment_tracked": discovery.sentiment_symbols(),
        "weights": DISCOVERY_WEIGHTS,
    }


@app.post("/api/discovery/refresh")
async def refresh_discovery():
    await discovery.run_cycle()
    return {"status": "success", **discovery.status()}


@app.post("/api/discovery/promote")
async def promote_candidate(req: SymbolRequest):
    """Moves a candidate onto the watchlist. Entries still pass every risk gate."""
    try:
        result = discovery.promote(req.symbol)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    await market_stream.ensure_stock_subscription(result["symbol"])
    return {"status": "success", **result}


@app.post("/api/discovery/dismiss")
async def dismiss_candidate(req: SymbolRequest):
    if not discovery.dismiss(req.symbol):
        raise HTTPException(status_code=404, detail="No such candidate, or it is on the watchlist")
    return {"status": "success"}


@app.post("/api/discovery/restore")
async def restore_candidate(req: SymbolRequest):
    if not discovery.restore(req.symbol):
        raise HTTPException(status_code=404, detail="No dismissed candidate with that symbol")
    return {"status": "success"}


@app.get("/api/discovery/what-if/{symbol:path}")
async def candidate_what_if(symbol: str):
    """Portfolio risk before and after a standard-size position in this symbol."""
    sym = symbol.upper().strip()
    await daily_bars.ensure([sym] + list(state.active_positions))
    return diversification.what_if(sym)


@app.get("/api/diversification")
async def get_diversification():
    """Sleeve exposure vs the dial's caps, daily-bar portfolio risk, and warnings."""
    await daily_bars.ensure(list(state.active_positions))
    return diversification.portfolio_risk()


@app.websocket("/ws")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            # Keep socket alive and receive client commands if any
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)
    except Exception:
        if websocket in active_connections:
            active_connections.remove(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=False)
