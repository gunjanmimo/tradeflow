"""
Agent memory: what each bot has done, and what it should conclude from that.

Graph model
-----------
    (:Bot     {key, bot_id, symbol, strategy})
    (:Trade   {key, trade_id, symbol, strategy, pnl, r_multiple, held_s, ...})
    (:Setup   {key, strategy, asset_class, rsi_bucket, trend_bucket, sent_bucket, session})
    (:Symbol  {key, ticker, asset_class})

    (Bot)   -[:EXECUTED]->   (Trade)
    (Trade) -[:ON]->         (Symbol)
    (Trade) -[:FROM_SETUP]-> (Setup)

Why a graph rather than a flat table
------------------------------------
The question that matters is not "list my trades" but "under conditions like the
ones in front of me right now, what has actually happened before?". That is a
traversal: Setup -> its Trades -> their outcomes. The Setup node is the join point
that makes the same question answerable from either direction -- by condition, by
symbol, or by bot.

Why R-multiple is the unit
--------------------------
Outcomes are recorded as R-multiples (pnl / initial risk), not dollars. A $50 win
on a $10 risk and a $50 win on a $500 risk are completely different results, and
dollar PnL cannot tell them apart. R-multiple is also invariant to the risk dial,
so memory stays comparable after the operator changes risk settings.

Discipline
----------
Every method degrades to a neutral answer when HydraDB is unavailable. A missing
memory must never block a trade, and `n_trades == 0` must never read as
"this setup is bad" -- absence of data is not evidence.
"""
import logging
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from core.config import settings
from core.state import state
from memory.hydra_client import hydra, stable_id

logger = logging.getLogger("tradeflow.memory")


# ----------------------------------------------------------------------
# Setup discretisation
# ----------------------------------------------------------------------
def _rsi_bucket(rsi: Optional[float]) -> str:
    if rsi is None:
        return "unknown"
    if rsi < 30:
        return "rsi_lt30"
    if rsi < 45:
        return "rsi_30_45"
    if rsi < 60:
        return "rsi_45_60"
    if rsi < 72:
        return "rsi_60_72"
    return "rsi_gt72"


def _trend_bucket(score: Optional[float]) -> str:
    if score is None:
        return "unknown"
    if score >= 0.80:
        return "strong_up"
    if score >= 0.60:
        return "up"
    if score >= 0.40:
        return "flat"
    if score >= 0.20:
        return "down"
    return "strong_down"


def _sentiment_bucket(pos: Optional[float], n: int) -> str:
    if not n:
        return "no_news"
    if pos is None:
        return "unknown"
    if pos >= 0.70:
        return "bullish"
    if pos >= 0.55:
        return "mild_bull"
    if pos >= 0.45:
        return "neutral"
    return "bearish"


def _vol_bucket(atr_pct: Optional[float]) -> str:
    if atr_pct is None:
        return "unknown"
    if atr_pct < 0.05:
        return "vol_low"
    if atr_pct < 0.20:
        return "vol_mid"
    if atr_pct < 0.60:
        return "vol_high"
    return "vol_extreme"


def _session_bucket(ts: Optional[float] = None) -> str:
    """UTC hour band."""
    t = time.gmtime(ts or time.time())
    h = t.tm_hour
    if 13 <= h < 21:
        return "us_hours"
    if 7 <= h < 13:
        return "eu_hours"
    if 0 <= h < 7:
        return "asia_hours"
    return "off_hours"


def setup_key(strategy: str, symbol: str, rsi: Optional[float],
              trend_score: Optional[float], sent_pos: Optional[float],
              sent_n: int, atr_pct: Optional[float],
              ts: Optional[float] = None) -> Dict[str, str]:
    """
    Builds the discretised setup bucket.

    Deliberately coarse. A bucket must accumulate enough samples to mean anything,
    and a finer grid would spread a handful of trades across hundreds of cells
    where every one looks like a 100% or 0% win rate on n=1.
    """
    klass = "equity"
    parts = {
        "strategy": strategy,
        "asset_class": klass,
        "rsi_bucket": _rsi_bucket(rsi),
        "trend_bucket": _trend_bucket(trend_score),
        "sent_bucket": _sentiment_bucket(sent_pos, sent_n),
        "vol_bucket": _vol_bucket(atr_pct),
        "session": _session_bucket(ts),
    }
    parts["key"] = "|".join([
        parts["strategy"], parts["asset_class"], parts["rsi_bucket"],
        parts["trend_bucket"], parts["sent_bucket"], parts["vol_bucket"],
        parts["session"],
    ])
    return parts


# ----------------------------------------------------------------------
@dataclass
class SetupStats:
    """Aggregate outcome of one setup bucket. Computed in Python: engine aggregates are unreliable."""
    setup: str
    n_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: Optional[float] = None
    avg_r: Optional[float] = None
    total_r: float = 0.0
    best_r: Optional[float] = None
    worst_r: Optional[float] = None
    expectancy: Optional[float] = None     # avg R per trade; the number that matters
    has_enough_samples: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AgentMemory:
    """
    Per-bot and per-setup memory over HydraDB.

    Every write is fire-and-forget from the trading path's perspective; every read
    returns a neutral result when memory is down.
    """

    def __init__(self):
        self.enabled = False
        self.trades_written = 0
        self.write_failures = 0
        # Small in-process cache so repeated recalls on the tick path never wait
        # on the network. Refreshed on write and on a short TTL.
        self._stats_cache: Dict[str, tuple] = {}
        self._cache_ttl = 60.0

    async def initialize(self) -> bool:
        self.enabled = await hydra.initialize()
        if self.enabled:
            state.log_event("MEMORY", "HydraDB agent memory connected.")
        else:
            state.log_event(
                "MEMORY_DEGRADED",
                "HydraDB unavailable: agents run without persistent memory. "
                "Trading is unaffected; recall returns neutral."
            )
        return self.enabled

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    async def remember_bot(self, bot_id: str, symbol: str, strategy: str):
        """Registers a sentinel agent so its trades have an owner to hang from."""
        if not self.enabled:
            return
        try:
            await hydra.upsert_node("Bot", f"bot:{bot_id}", {
                "bot_id": bot_id,
                "symbol": symbol,
                "strategy": strategy,
                "created_at": time.time(),
            })
        except Exception as e:
            logger.debug(f"remember_bot failed: {e}")

    async def remember_trade(self, record: Dict[str, Any]) -> bool:
        """
        Persists one closed trade and its edges.

        Called from the close path but never awaited by it -- the caller schedules
        this as a task so a slow graph write cannot delay a liquidation.
        """
        if not self.enabled:
            return False

        try:
            symbol = record.get("symbol", "?")
            setup = record.get("setup") or {}
            strategy = record.get("strategy") or setup.get("strategy") or "unknown"
            trade_id = record.get("trade_id") or f"{symbol}:{int(record.get('time', time.time())*1000)}"
            bot_id = record.get("bot_id") or f"BOT-{symbol.replace('/', '')}"

            sk = setup_key(
                strategy=strategy,
                symbol=symbol,
                rsi=setup.get("entry_rsi"),
                trend_score=setup.get("entry_trend_score"),
                sent_pos=setup.get("laya_pos"),
                sent_n=int(setup.get("sentiment_n") or 0),
                atr_pct=setup.get("entry_atr_pct"),
                ts=record.get("opened_at") or record.get("time"),
            )

            r = record.get("r_multiple")
            trade_props = {
                "trade_id": trade_id,
                "symbol": symbol,
                "strategy": strategy,
                "pnl": float(record.get("pnl") or 0.0),
                "r_multiple": float(r) if r is not None else 0.0,
                "has_r": r is not None,
                "initial_risk": float(record.get("initial_risk") or 0.0),
                "entry_price": float(record.get("entry_price") or 0.0),
                "exit_price": float(record.get("price") or 0.0),
                "held_seconds": float(record.get("held_seconds") or 0.0),
                "exit_reason": str(record.get("exit_reason") or "")[:300],
                "setup_key": sk["key"],
                "closed_at": float(record.get("time") or time.time()),
            }

            # Bot -> Trade
            ok = await hydra.link(
                "Bot", f"bot:{bot_id}", "EXECUTED", "Trade", f"trade:{trade_id}",
                from_props={"bot_id": bot_id, "symbol": symbol, "strategy": strategy},
                to_props=trade_props,
            )
            # Trade -> Symbol
            ok &= await hydra.link(
                "Trade", f"trade:{trade_id}", "ON", "Symbol", f"symbol:{symbol}",
                to_props={"ticker": symbol,
                          "asset_class": "equity"},
            )
            # Trade -> Setup  (the join point that makes condition-based recall work)
            ok &= await hydra.link(
                "Trade", f"trade:{trade_id}", "FROM_SETUP", "Setup", f"setup:{sk['key']}",
                to_props={k: v for k, v in sk.items()},
            )
            if not ok:
                self.write_failures += 1
                logger.warning(f"Memory: HydraDB rejected the write for {symbol} "
                               f"({hydra.last_error}); outcome not recorded")
                return False

            self.trades_written += 1
            self._stats_cache.pop(sk["key"], None)
            logger.info(
                f"Memory: recorded {symbol} {('%+.2fR' % r) if r is not None else 'n/a R'} "
                f"under setup {sk['key']}"
            )
            return True
        except Exception as e:
            self.write_failures += 1
            logger.warning(f"remember_trade failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    @staticmethod
    def _summarise(setup: str, rows: List[Dict[str, Any]]) -> SetupStats:
        rs = [float(r["r"]) for r in rows if r.get("r") is not None and r.get("has_r")]
        st = SetupStats(setup=setup, n_trades=len(rs))
        if not rs:
            return st
        st.wins = sum(1 for x in rs if x > 0)
        st.losses = sum(1 for x in rs if x <= 0)
        st.win_rate = round(st.wins / len(rs), 3)
        st.total_r = round(sum(rs), 3)
        st.avg_r = round(sum(rs) / len(rs), 3)
        st.expectancy = st.avg_r
        st.best_r = round(max(rs), 3)
        st.worst_r = round(min(rs), 3)
        st.has_enough_samples = len(rs) >= settings.MEMORY_MIN_SAMPLES
        return st

    async def recall_setup(self, setup: str, use_cache: bool = True) -> SetupStats:
        """What has happened before under this exact setup bucket."""
        if not self.enabled:
            return SetupStats(setup=setup)

        now = time.time()
        if use_cache:
            hit = self._stats_cache.get(setup)
            if hit and (now - hit[0]) < self._cache_ttl:
                return hit[1]

        # Node keys are namespaced ("setup:<bucket>"), which is what upsert stores in
        # the `key` property. Querying the bare bucket silently matched nothing.
        rows = await hydra.query(
            "MATCH (t:Trade)-[:FROM_SETUP]->(s:Setup {key: $key}) "
            "RETURN t.r_multiple AS r, t.has_r AS has_r, t.pnl AS pnl",
            {"key": setup if setup.startswith("setup:") else f"setup:{setup}"},
        )
        stats = self._summarise(setup, rows)
        self._stats_cache[setup] = (now, stats)
        return stats

    async def recall_symbol(self, symbol: str) -> SetupStats:
        """Aggregate outcome history for one symbol, across all setups."""
        if not self.enabled:
            return SetupStats(setup=f"symbol:{symbol}")
        rows = await hydra.query(
            "MATCH (t:Trade)-[:ON]->(s:Symbol {key: $key}) "
            "RETURN t.r_multiple AS r, t.has_r AS has_r, t.pnl AS pnl",
            {"key": f"symbol:{symbol}"},
        )
        return self._summarise(f"symbol:{symbol}", rows)

    async def recall_strategy(self, strategy: str) -> SetupStats:
        """Aggregate outcome history for one strategy, for comparing strategies."""
        if not self.enabled:
            return SetupStats(setup=f"strategy:{strategy}")
        rows = await hydra.query(
            "MATCH (t:Trade) WHERE t.strategy = $s "
            "RETURN t.r_multiple AS r, t.has_r AS has_r, t.pnl AS pnl",
            {"s": strategy},
        )
        return self._summarise(f"strategy:{strategy}", rows)

    async def recall_bot(self, bot_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """A single agent's own trade history -- its personal memory."""
        if not self.enabled:
            return []
        return await hydra.query(
            "MATCH (b:Bot {key: $key})-[:EXECUTED]->(t:Trade) "
            "RETURN t.trade_id AS trade_id, t.symbol AS symbol, t.pnl AS pnl, "
            "t.r_multiple AS r_multiple, t.exit_reason AS exit_reason, "
            "t.closed_at AS closed_at, t.setup_key AS setup_key "
            f"ORDER BY t.closed_at DESC LIMIT {int(limit)}",
            {"key": f"bot:{bot_id}"},
        )

    # ------------------------------------------------------------------
    # The "don't repeat the same mistake" rule
    # ------------------------------------------------------------------
    async def should_avoid(self, strategy: str, symbol: str, rsi: Optional[float],
                           trend_score: Optional[float], sent_pos: Optional[float],
                           sent_n: int, atr_pct: Optional[float]) -> Dict[str, Any]:
        """
        Should this setup be skipped because it has repeatedly lost?

        A deliberate HARD RULE, not a learned policy. It works at the sample sizes
        actually available (a handful of trades per bucket), whereas a bandit or
        Q-learner needs tens of samples per cell before its estimates mean anything.
        The rule also fails safe in the right direction: with no data it permits the
        trade, so it can only ever veto a pattern that has demonstrably lost money.
        """
        result = {"avoid": False, "reason": "", "stats": None, "setup": None}
        if not self.enabled:
            return result

        sk = setup_key(strategy, symbol, rsi, trend_score, sent_pos, sent_n, atr_pct)
        result["setup"] = sk["key"]
        stats = await self.recall_setup(sk["key"])
        result["stats"] = stats.to_dict()

        # Absence of evidence is not evidence: too few samples means "go ahead".
        if not stats.has_enough_samples:
            result["reason"] = (
                f"Only {stats.n_trades} prior trade(s) in this setup "
                f"(need {settings.MEMORY_MIN_SAMPLES}); not enough to judge."
            )
            return result

        if stats.expectancy is not None and stats.expectancy <= settings.MEMORY_AVOID_EXPECTANCY:
            result["avoid"] = True
            result["reason"] = (
                f"Setup has lost money repeatedly: {stats.n_trades} trades, "
                f"expectancy {stats.expectancy:+.2f}R, win rate {stats.win_rate:.0%} "
                f"(worst {stats.worst_r:+.2f}R). Skipping to avoid repeating it."
            )
            return result

        result["reason"] = (
            f"Setup is acceptable: {stats.n_trades} trades, expectancy "
            f"{stats.expectancy:+.2f}R, win rate {stats.win_rate:.0%}."
        )
        return result

    async def context_for(self, symbol: str, strategy: str, rsi=None, trend_score=None,
                          sent_pos=None, sent_n=0, atr_pct=None) -> Dict[str, Any]:
        """
        Everything memory knows that is relevant to a decision on this symbol now.
        This is the 'give the agent context' surface.
        """
        if not self.enabled:
            return {"enabled": False}
        sk = setup_key(strategy, symbol, rsi, trend_score, sent_pos, sent_n, atr_pct)
        setup_stats = await self.recall_setup(sk["key"])
        symbol_stats = await self.recall_symbol(symbol)
        strat_stats = await self.recall_strategy(strategy)
        return {
            "enabled": True,
            "setup": sk,
            "setup_stats": setup_stats.to_dict(),
            "symbol_stats": symbol_stats.to_dict(),
            "strategy_stats": strat_stats.to_dict(),
        }

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "trades_written": self.trades_written,
            "write_failures": self.write_failures,
            "cached_setups": len(self._stats_cache),
            "min_samples_to_judge": settings.MEMORY_MIN_SAMPLES,
            "avoid_expectancy_threshold": settings.MEMORY_AVOID_EXPECTANCY,
        }


agent_memory = AgentMemory()
