import logging
import asyncio
import math
import time
import uuid
from typing import Optional, Dict, Any
from core.config import settings
from core.state import state, TradeDecision
from engine.risk_guard import risk_guard
from core.latency import latency
from engine import brackets
from engine.fills import fills

logger = logging.getLogger("tradeflow.executor")

ORDER_ID_PREFIX = "tf-"

# Fields snapshotted at entry (see _execute_buy) that must survive broker syncs.
ENTRY_CONTEXT_KEYS = frozenset({
    "opened_at", "laya_pos", "laya_neg", "sentiment_headline", "sentiment_age_s",
    "consensus_score", "conviction_tier", "composite_conviction", "entry_rsi",
    "entry_spread", "entry_atr_pct", "stop_pct", "buy_prob", "entry_reason",
    "entry_strategy", "entry_regime", "extended_hours", "brackets_in_engine",
    # RL policy bookkeeping (rl/live.py): the observation and action at entry.
    "rl_entry",
})

class AlpacaExecutor:
    """
    Order router and position sync for the Alpaca Trading API.
    Entries are bracket orders (market entry + stop-loss + take-profit legs);
    exits are full closes. There are no partial sells or adds.
    """
    def __init__(self):
        self.trading_client = None
        self.is_connected = False
        self.is_mock_mode = False
        self.alpaca_tradable_symbols: set[str] = set()
        self.pending_orders: set[str] = set()
        # Symbols with a close/liquidation order currently in flight.
        # Guards against the sentinel firing a second close while the first
        # round-trip to the broker is still open (which would oversell into a short).
        self.closing_orders: set[str] = set()
        # symbol -> unix time before which a failed close is not retried, and the
        # consecutive failure count that sets the backoff.
        self.close_retry_after: Dict[str, float] = {}
        self.close_failures: Dict[str, int] = {}
        # symbol -> the CLOSE decision behind an extended-hours exit order that is
        # working at the broker. The position stays listed until it fills.
        self.pending_exits: Dict[str, TradeDecision] = {}
        # symbol -> (exit orders sent so far, when the last one went out); widens
        # each re-price of an unfilled extended-hours exit.
        self.exit_attempts: Dict[str, tuple] = {}
        # symbol -> the broker's own mark for a held position, from the last sync.
        self.broker_marks: Dict[str, float] = {}
        self._remark_logged: set[str] = set()
        # Non-equity symbols already reported as ignored by the position sync.
        self._foreign_logged: set[str] = set()
        # symbol -> the PriceTick object remark_stale_positions last injected
        self._remark_tick: Dict[str, Any] = {}
        self._mark_task: Optional[asyncio.Task] = None
        # symbol -> entry-context snapshot, merged into the position once the
        # broker sync reports the fill. Cleared when the position closes.
        self.pending_entry_context: Dict[str, Dict[str, Any]] = {}
        # symbol -> dollars of a submitted BUY not yet visible as a position.
        # Counted against the hard budget cap (see inflight_committed).
        self.inflight_notional: Dict[str, float] = {}
        # Positions that disappeared from the broker without our close path, and
        # when we last closed each symbol ourselves (so a sync racing our own
        # close does not book it twice).
        self.vanished_positions: list = []
        self.recently_closed: Dict[str, float] = {}
        # client_order_ids of pre-market limit buys: cancelled if unfilled after
        # PREMARKET_STALE_ORDER_SECONDS rather than left to fill at the open.
        self.extended_order_ids: set[str] = set()

        # --- Unfilled-order tracking ---
        # A submitted order is not a position until it fills. Previously nothing
        # tracked the gap: an order that never filled left no position, so the next signal
        # submitted another one -- 27 stacked orders held ~all the cash.
        # symbol -> [{id, side, submitted_at, client_order_id}], refreshed each sync
        self.open_orders: Dict[str, list] = {}
        # symbol -> submit time; set on submit, cleared once the sync shows a
        # position or no open order. Covers the gap before the next sync.
        self.awaiting_fill: Dict[str, float] = {}
        # symbol -> unix time until which new buys are refused (after a stale cancel)
        self.cooldown_until: Dict[str, float] = {}
        self._reject_logged: Dict[tuple, float] = {}

    def inflight_committed(self) -> float:
        """
        Notional of this engine's buys that have not yet shown up as a position.
        Counted against the hard budget cap for as long as the order is live, with
        no timeout: an after-hours stock order can legitimately sit for hours.
        """
        total = 0.0
        for sym, dollars in list(self.inflight_notional.items()):
            live = (sym in self.pending_orders or sym in self.awaiting_fill
                    or any(o["side"] == "buy" for o in self.open_orders.get(sym, ())))
            if sym in state.active_positions or not live:
                self.inflight_notional.pop(sym, None)
                continue
            total += dollars
        return total

    def _queue_vanished(self, sym: str, pos: Dict[str, Any]):
        """A position that closed at the broker without passing through _execute_close."""
        if sym in self.closing_orders or time.time() - self.recently_closed.get(sym, 0.0) < 30.0:
            return
        self.vanished_positions.append((sym, pos))

    def book_vanished_positions(self):
        """
        Books PnL for positions the broker closed on its own (equity bracket stop or
        target, manual close, kill switch). Without this those results never
        reached the budget ledger, so a stop-out at the broker left the hard cap
        unchanged and the bots kept trading money they had already lost.
        Runs on the event loop, never in the sync thread.
        """
        while self.vanished_positions:
            sym, pos = self.vanished_positions.pop(0)
            if sym in state.active_positions:
                continue
            exit_price = float(pos.get("current_price") or pos.get("avg_entry_price") or 0.0)
            pnl = round(float(pos.get("unrealized_pl") or 0.0), 2)
            self.recently_closed[sym] = time.time()
            state.book_realized_pnl(sym, pnl)
            decision = self.pending_exits.pop(sym, None) or TradeDecision(
                symbol=sym, action="CLOSE", close=True,
                reason="Closed at broker (bracket stop/target or manual)")
            self.close_retry_after.pop(sym, None)
            self.exit_attempts.pop(sym, None)
            record = self._record_closed_trade(sym, pos, exit_price, pnl, decision)
            if not self.is_mock_mode and pos.get("mode") == "ALPACA_PAPER":
                # Re-booked at the broker's fill once it is found (engine/fills.py).
                fills.track(sym, float(pos.get("qty") or 0.0), float(pos.get("avg_entry_price") or 0.0),
                            pnl, record, since=float(pos.get("opened_at") or time.time() - 86400))
            state.log_event("ORDER_SYNC",
                f"{sym} closed at the broker; booked last-marked PnL ${pnl:+,.2f} against the budget "
                f"(corrected to the fill once the broker reports it)")

    def entry_block_reason(self, symbol: str) -> Optional[str]:
        """Why a new BUY for this symbol must not be sent, or None."""
        now = time.time()
        until = self.cooldown_until.get(symbol)
        if until and now < until:
            return (f"{symbol} is on cooldown for {(until - now) / 60:.0f} more min: a previous "
                    f"order sat unfilled and was cancelled")
        if any(o["side"] == "buy" for o in self.open_orders.get(symbol, ())):
            return f"{symbol} already has an unfilled BUY order at the broker"
        if symbol in self.awaiting_fill:
            return f"{symbol} order submitted {now - self.awaiting_fill[symbol]:.0f}s ago, awaiting fill"
        return None

    def _sync_open_orders(self, filled_symbols: set):
        """
        Refreshes open broker orders, clears awaiting-fill markers, and cancels
        this engine's own pre-market limit buys that have sat unfilled too long.
        Runs inside the sync thread, never on the event loop.
        """
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        orders = self.trading_client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500))
        now = time.time()
        by_symbol: Dict[str, list] = {}
        for o in orders:
            sym = o.symbol
            side = "buy" if "BUY" in str(o.side).upper() else "sell"
            submitted = o.submitted_at.timestamp() if o.submitted_at else now
            coid = o.client_order_id or ""
            if (side == "buy" and coid in self.extended_order_ids
                    and now - submitted > settings.PREMARKET_STALE_ORDER_SECONDS):
                try:
                    self.trading_client.cancel_order_by_id(o.id)
                    self.extended_order_ids.discard(coid)
                    self.cooldown_until[sym] = now + settings.PREMARKET_UNFILLED_COOLDOWN_SECONDS
                    state.log_event("ORDER_STALE",
                        f"Cancelled unfilled pre-market limit BUY {sym} after {now - submitted:.0f}s; "
                        f"no new {sym} buys for {settings.PREMARKET_UNFILLED_COOLDOWN_SECONDS / 60:.0f} min")
                    continue
                except Exception as e:
                    logger.warning(f"Could not cancel stale pre-market order {o.id} for {sym}: {e}")
            by_symbol.setdefault(sym, []).append(
                {"id": str(o.id), "side": side, "submitted_at": submitted, "client_order_id": coid})
        self.open_orders = by_symbol
        for sym in list(self.awaiting_fill):
            if sym in filled_symbols or sym not in by_symbol:
                self.awaiting_fill.pop(sym, None)

    def is_symbol_tradable(self, symbol: str) -> bool:
        if self.is_mock_mode:
            return True
        return symbol.upper().strip() in self.alpaca_tradable_symbols

    async def initialize(self):
        """Initializes Alpaca TradingClient using official alpaca-py"""
        if settings.ALPACA_API_KEY.startswith("PK_PLACEHOLDER") or not settings.ALPACA_API_KEY:
            logger.warning("Alpaca API keys are placeholders. Operating in local simulated PAPER mode.")
            self.is_mock_mode = True
            state.log_event("WARNING", "Running in local simulated paper mode (add your Alpaca keys in .env for live Alpaca Paper connection)")
            return

        try:
            from alpaca.trading.client import TradingClient
            self.trading_client = TradingClient(
                api_key=settings.ALPACA_API_KEY,
                secret_key=settings.ALPACA_SECRET_KEY,
                paper=settings.IS_PAPER
            )
            # Sync initial account info & tradable asset catalog
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.sync_account_and_positions)
            from engine.sentinel_agent import sentinel_registry
            sentinel_registry.sync_with_positions(state.active_positions)
            
            def load_assets():
                try:
                    from alpaca.trading.requests import GetAssetsRequest
                    from alpaca.trading.enums import AssetClass
                    assets = self.trading_client.get_all_assets(
                        GetAssetsRequest(asset_class=AssetClass.US_EQUITY))
                    return {a.symbol for a in assets if a.tradable}
                except Exception as ex:
                    logger.debug(f"Could not load asset catalog: {ex}")
                    return set()

            def load_recent_orders():
                try:
                    from alpaca.trading.requests import GetOrdersRequest
                    from alpaca.trading.enums import QueryOrderStatus
                    req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=30)
                    orders = self.trading_client.get_orders(req)
                    recent = []
                    for o in reversed(orders):
                        if o.filled_at and o.filled_avg_price:
                            sym = o.symbol
                            side_str = "BUY" if "BUY" in str(o.side).upper() else "SELL"
                            recent.append({
                                "time": o.filled_at.timestamp(),
                                "symbol": sym,
                                "side": side_str,
                                "qty": float(o.filled_qty or o.qty),
                                "price": float(o.filled_avg_price),
                                "mode": "ALPACA_PAPER"
                            })
                    return recent
                except Exception as ex:
                    logger.debug(f"Could not load recent orders: {ex}")
                    return []

            self.alpaca_tradable_symbols = await loop.run_in_executor(None, load_assets)
            past_orders = await loop.run_in_executor(None, load_recent_orders)
            for o in past_orders:
                state.recent_trades.append(o)

            self.is_connected = True
            self.is_mock_mode = False
            state.log_event("INFO", f"Connected to Alpaca Paper API ({len(self.alpaca_tradable_symbols)} tradable assets, {len(past_orders)} past fills loaded)")
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca API: {e}. Falling back to simulated paper mode.")
            self.is_mock_mode = True
            state.log_event("ERROR", f"Alpaca connection failed ({e}), using simulated paper mode.")

    def sync_account_and_positions(self):
        """Synchronizes account equity, cash, and active positions from Alpaca"""
        if self.is_mock_mode or not self.trading_client:
            return

        try:
            account = self.trading_client.get_account()
            state.account_info["equity"] = float(account.equity)
            state.account_info["cash"] = float(account.cash)
            state.account_info["buying_power"] = float(account.buying_power)
            if getattr(account, "last_equity", None) is not None:
                state.account_info["last_equity"] = float(account.last_equity)
            state.sync_locked_equity()
            state.roll_trading_day_if_needed()

            positions = self.trading_client.get_all_positions()
            new_positions = {}
            for pos in positions:
                # US equities only. Anything else in the account (e.g. crypto left
                # over from before crypto was removed) is not the engine's to manage.
                if "EQUITY" not in str(getattr(pos, "asset_class", "us_equity")).upper():
                    if pos.symbol not in self._foreign_logged:
                        self._foreign_logged.add(pos.symbol)
                        state.log_event("ORDER_SYNC", f"Ignoring non-equity position {pos.symbol} "
                                                      f"({pos.asset_class}); close it at the broker.")
                    continue
                normalized_sym = pos.symbol

                old_pos = state.active_positions.get(normalized_sym, {})
                qty = float(pos.qty)
                avg_entry = float(pos.avg_entry_price)
                current_price = float(pos.current_price)
                self.broker_marks[normalized_sym] = current_price
                # Preserve the bracket we already hold; only derive one when the
                # position is new to us (e.g. opened before this process started).
                # Deriving from live ATR keeps it volatility-scaled instead of the
                # flat -2%/+4% this path used to stamp onto every position.
                sl = old_pos.get("stop_loss")
                tp = old_pos.get("take_profit")
                if not brackets.is_valid(avg_entry, sl, tp):
                    q = state.quant_metrics.get(normalized_sym)
                    sl, tp, _dist = brackets.derive(avg_entry, q.atr if q else None)
                invested = old_pos.get("invested_dollars") or round(qty * avg_entry, 2)
                alloc_pct = old_pos.get("allocated_pct") or round((invested / max(1.0, state.hard_cap)) * 100, 1)
                risk = old_pos.get("dollar_risk") or round(abs(avg_entry - sl) * qty, 2)
                reward = old_pos.get("dollar_reward") or round(abs(tp - avg_entry) * qty, 2)

                ctx = self.pending_entry_context.pop(normalized_sym, None) or {}
                # Carry entry context forward on every sync, not just the first: the
                # pending snapshot is consumed once, so keying only on it dropped the
                # context (and the position's entry strategy) on the next sync.
                carried = {k: v for k, v in old_pos.items() if k in ENTRY_CONTEXT_KEYS}
                merged_ctx = {**ctx, **carried}

                new_positions[normalized_sym] = {
                    **merged_ctx,
                    "symbol": normalized_sym,
                    "qty": qty,
                    "avg_entry_price": avg_entry,
                    "current_price": current_price,
                    "unrealized_pl": float(pos.unrealized_pl),
                    "unrealized_plpc": float(pos.unrealized_plpc),
                    "stop_loss": sl,
                    "take_profit": tp,
                    "action_space": old_pos.get("action_space", ["BUY", "HOLD", "SELL", "CLOSE"]),
                    "action": old_pos.get("action", "HOLD"),
                    "buy_prob": old_pos.get("buy_prob", 0.15),
                    "hold_prob": old_pos.get("hold_prob", 0.70),
                    "sell_prob": old_pos.get("sell_prob", 0.10),
                    "close_prob": old_pos.get("close_prob", 0.05),
                    "laya_pos": old_pos.get("laya_pos", 0.5),
                    "laya_neg": old_pos.get("laya_neg", 0.5),
                    "mode": old_pos.get("mode", "ALPACA_PAPER"),
                    "invested_dollars": invested,
                    "current_value": round(qty * current_price, 2),
                    "allocated_pct": alloc_pct,
                    "dollar_risk": risk,
                    "dollar_reward": reward,
                    "sell_plan": old_pos.get("sell_plan", f"SELL 100% ({qty}x) on SL/TP/Reversal"),
                    "sell_pct": old_pos.get("sell_pct", 100),
                    "sell_qty": old_pos.get("sell_qty", qty),
                    "bot_id": old_pos.get("bot_id", f"BOT-{normalized_sym.replace('/', '')}"),
                    "bot_thesis": old_pos.get("bot_thesis"),
                    "sentinel": old_pos.get("sentinel")
                }
            # Preserve simulated positions
            for sym, old in state.active_positions.items():
                if old.get("mode") in ("SIMULATED", "PAPER_SIMULATED") and sym not in new_positions:
                    new_positions[sym] = old
                elif sym not in new_positions:
                    self._queue_vanished(sym, old)
            state.active_positions = new_positions
            state.update_peak_equity()
            try:
                self._sync_open_orders(set(new_positions))
            except Exception as e:
                logger.error(f"Open-order sync failed: {e}")
            if fills.pending:
                fills.reconcile_sync(self.trading_client)
        except Exception as e:
            logger.error(f"Error syncing account from Alpaca: {e}")

    async def execute_decision(self, decision: TradeDecision):
        """
        Executes a TradeDecision asynchronously.
        Takes ~30-60ms network roundtrip to Alpaca.
        """
        # Pausing stops new entries only. A protective exit must still go out,
        # or a paused bot sits on a CLOSE it never acts on.
        if decision.action in ("CLOSE", "SELL"):
            await self._execute_close(decision)
            return
        if not state.is_trading_active:
            return
        # One entry per symbol; a BUY for a held symbol is ignored (no adds).
        if decision.action == "BUY" and decision.symbol not in state.active_positions:
            await self._execute_buy(decision)

    async def _execute_buy(self, decision: TradeDecision):
        symbol = decision.symbol
        if symbol in self.pending_orders or symbol in state.active_positions:
            return

        # 1. Risk check (evaluated before locking order in-flight)
        can_open, reason = risk_guard.can_open_position(symbol)
        if not can_open:
            # A standing block (cooldown, open order, halt) re-fires on every
            # tick; log each distinct reason once a minute, not twice a second.
            key = (symbol, " ".join(reason.split()[:3]))
            now = time.time()
            if now - self._reject_logged.get(key, 0.0) >= 60.0:
                self._reject_logged[key] = now
                state.log_event("RISK_REJECT", f"Cannot buy {symbol}: {reason}")
            return

        self.pending_orders.add(symbol)
        try:
            tick = state.latest_prices.get(symbol)
            quant = state.quant_metrics.get(symbol)
            if not tick:
                return

            gate = state.last_gate_detail.get(symbol) or {}
            entry_strategy = gate.get("selected_strategy") or gate.get("strategy")

            # 2. Sizing calculation via Laya Allocation Manager
            atr = quant.atr if quant else 0.50
            sentiment = state.get_sentiment(symbol)
            from feeds.multi_source_aggregator import trend_aggregator
            # None when no real source backed this symbol. Defaulting to 0.70
            # (as this previously did) invented conviction out of nothing and
            # inflated position size for every unscored symbol.
            consensus_score = trend_aggregator.get_consensus(symbol)

            qty, stop_loss, take_profit, alloc = risk_guard.calculate_order_sizing(
                symbol=symbol,
                current_price=tick.price,
                atr=atr,
                laya_pos_prob=sentiment.pos_prob,
                consensus_score=consensus_score
            )
            if qty <= 0:
                state.log_event("RISK_REJECT", f"Order skipped for {symbol}: {alloc.get('rationale')}")
                return
            # Hold this notional against the sleeve caps until the fill shows up
            # as a position, so a concurrent entry cannot spend the same headroom.
            from engine.diversification import diversification
            diversification.reserve(symbol, alloc.get("allocated_dollars") or qty * tick.price)

            # Last line of defence for the hard cap.
            order_cost = qty * tick.price * (1 + settings.BUDGET_FILL_BUFFER_PCT / 100.0)
            if order_cost > state.remaining_budget + 0.01:
                state.log_event("RISK_REJECT",
                    f"Order skipped for {symbol}: ${order_cost:,.2f} (incl. fill buffer) would exceed the "
                    f"hard budget cap (${state.remaining_budget:,.2f} left of ${state.hard_cap:,.2f})")
                return

            state.log_event("ALLOCATION", alloc["rationale"])

            # Snapshot the conditions that justified this entry. Without it, a closed
            # trade tells you the PnL but not what the engine believed at the time,
            # so no attribution (or learning) is possible after the fact.
            entry_context = {
                "opened_at": time.time(),
                "laya_pos": sentiment.pos_prob,
                "laya_neg": sentiment.neg_prob,
                "sentiment_headline": sentiment.headline,
                "sentiment_age_s": state.sentiment_age(symbol),
                "consensus_score": consensus_score,
                "conviction_tier": alloc.get("conviction_tier"),
                "composite_conviction": alloc.get("composite_conviction"),
                "entry_rsi": quant.rsi if quant else None,
                "entry_spread": quant.spread if quant else None,
                "entry_atr_pct": round((quant.atr / tick.price) * 100, 4) if (quant and quant.atr and tick.price) else None,
                "stop_pct": round((abs(tick.price - stop_loss) / tick.price) * 100, 3) if tick.price else None,
                "dollar_risk": alloc.get("risk_dollars"),
                "dollar_reward": alloc.get("reward_dollars"),
                "buy_prob": decision.buy_prob,
                "entry_reason": decision.reason,
                # The strategy whose thesis this trade is; its exit rules govern it.
                "entry_strategy": entry_strategy,
                "entry_regime": gate.get("regime"),
            }
            if gate.get("rl_entry"):
                entry_context["rl_entry"] = gate["rl_entry"]

            t0 = time.time()
            is_alpaca_tradable = symbol in self.alpaca_tradable_symbols

            if self.is_mock_mode or not is_alpaca_tradable:
                mode_tag = "SIMULATED" if self.is_mock_mode else "PAPER_SIMULATED"
                state.active_positions[symbol] = {
                    "symbol": symbol,
                    "qty": qty,
                    "avg_entry_price": tick.price,
                    "current_price": tick.price,
                    "unrealized_pl": 0.0,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "allocated_dollars": alloc["allocated_dollars"],
                    "allocated_pct": alloc["allocated_pct"],
                    "invested_dollars": alloc["allocated_dollars"],
                    "mode": mode_tag,
                    "action_space": ["BUY", "HOLD", "SELL", "CLOSE"],
                    "action": "HOLD",
                    "buy_prob": 0.15,
                    "hold_prob": 0.70,
                    "sell_prob": 0.10,
                    "close_prob": 0.05,
                    **entry_context,
                }
                order_cost = round(qty * tick.price, 2)
                state.account_info["cash"] -= order_cost
                state.recent_trades.append({
                    "time": time.time(),
                    "symbol": symbol,
                    "side": "BUY",
                    "qty": qty,
                    "price": tick.price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "mode": mode_tag
                })
                state.log_event("ORDER_FILLED", f"[{mode_tag}] BUY {qty}x {symbol} @ ${tick.price:.2f} (SL: ${stop_loss}, TP: ${take_profit})")
                return

            # Real Alpaca API execution
            from alpaca.trading.requests import (
                MarketOrderRequest, LimitOrderRequest, TakeProfitRequest, StopLossRequest,
            )
            from alpaca.trading.enums import OrderSide, TimeInForce
            from core.market_hours import us_session, PRE

            # Tagged so the engine can tell its own orders from manual ones.
            client_order_id = f"{ORDER_ID_PREFIX}{uuid.uuid4().hex[:20]}"
            if us_session() == PRE:
                # Pre-market: Alpaca takes only DAY limit orders flagged
                # extended_hours, with no bracket legs. The sentinel enforces the
                # stop and target in-engine. Half size and
                # whole shares: the book is thin and fractional extended-hours
                # orders are not guaranteed.
                whole = math.floor(qty * settings.PREMARKET_SIZE_FACTOR)
                if whole < 1:
                    state.log_event("RISK_REJECT",
                        f"Pre-market order skipped for {symbol}: under one whole share at "
                        f"{settings.PREMARKET_SIZE_FACTOR:.0%} size (${tick.price:,.2f}/share)")
                    return
                qty = whole
                ask = tick.ask if tick.ask and tick.ask > 0 else tick.price
                req = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                    limit_price=round(ask * (1 + settings.PREMARKET_LIMIT_OFFSET_PCT / 100.0), 2),
                    extended_hours=True,
                    client_order_id=client_order_id,
                )
                self.extended_order_ids.add(client_order_id)
                entry_context["extended_hours"] = True
            else:
                # Equities use broker-native bracket order
                req = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC,
                    client_order_id=client_order_id,
                    take_profit=TakeProfitRequest(limit_price=round(take_profit, 2)),
                    # The broker holds the same stop the sentinel enforces, so
                    # the position is protected even when the engine is down.
                    stop_loss=StopLossRequest(stop_price=round(stop_loss, 2))
                )

            loop = asyncio.get_running_loop()
            self.inflight_notional[symbol] = round(
                qty * tick.price * (1 + settings.BUDGET_FILL_BUFFER_PCT / 100.0), 2)
            t_sub = time.perf_counter_ns()
            order = await loop.run_in_executor(None, self.trading_client.submit_order, req)
            latency.record_ns("order_submit", t_sub)
            latency_ms = (time.time() - t0) * 1000

            # Hold the entry context until the broker sync materialises the position,
            # so the closed-trade record can still report what justified the entry.
            self.pending_entry_context[symbol] = entry_context
            self.awaiting_fill[symbol] = time.time()

            # Deduct cash locally and trigger async broker sync
            order_cost = round(qty * tick.price, 2)
            state.account_info["cash"] = max(0.0, state.account_info["cash"] - order_cost)
            loop.run_in_executor(None, self.sync_account_and_positions)

            state.recent_trades.append({
                "time": time.time(),
                "symbol": symbol,
                "side": "BUY",
                "qty": qty,
                "price": tick.price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "order_id": str(order.id),
                "mode": "ALPACA_PAPER"
            })
            state.log_event("ORDER_SUBMITTED", f"Alpaca BUY {qty}x {symbol} submitted in {latency_ms:.1f}ms (ID: {order.id})")
        except Exception as e:
            err_str = str(e)
            self.inflight_notional.pop(symbol, None)
            logger.error(f"Alpaca order submission error: {err_str}")
            state.log_event("ORDER_ERROR", f"Failed to place BUY for {symbol}: {err_str}")
            if "insufficient balance" in err_str or "balance" in err_str:
                loop = asyncio.get_running_loop()
                loop.run_in_executor(None, self.sync_account_and_positions)
        finally:
            self.pending_orders.discard(symbol)

    async def _execute_close(self, decision: TradeDecision):
        symbol = decision.symbol
        if symbol not in state.active_positions:
            return

        # Claim the close before any await. The sentinel re-evaluates on every tick
        # and on a 1s heartbeat, so while a close is in flight it would otherwise
        # keep firing duplicate liquidations for the same position.
        if symbol in self.closing_orders:
            return
        if time.time() < self.close_retry_after.get(symbol, 0.0):
            return
        self.closing_orders.add(symbol)
        try:
            await self._execute_close_locked(decision)
        finally:
            self.closing_orders.discard(symbol)

    async def _execute_close_locked(self, decision: TradeDecision):
        symbol = decision.symbol
        tick = state.latest_prices.get(symbol)
        pos = state.active_positions.get(symbol)
        if not pos:
            return
        t0 = time.time()

        is_simulated = self.is_mock_mode or pos.get("mode") in ("SIMULATED", "PAPER_SIMULATED")
        if is_simulated:
            price = tick.price if tick else pos["avg_entry_price"]
            pnl = round((price - pos["avg_entry_price"]) * pos["qty"], 2)
            state.account_info["cash"] += round(pos["qty"] * price, 2)
            state.book_realized_pnl(symbol, pnl)
            # Main broker equity remains locked; total equity is locked broker equity + bot equity
            state.account_info["equity"] = round(state.locked_broker_equity + state.bot_equity, 2)
            state.account_info["locked_equity"] = state.locked_broker_equity
            state.account_info["main_broker_equity"] = state.locked_broker_equity
            state.active_positions.pop(symbol, None)
            self._record_closed_trade(symbol, pos, price, pnl, decision)

            state.recent_trades.append({
                "time": time.time(),
                "symbol": symbol,
                "side": "SELL",
                "qty": pos["qty"],
                "price": price,
                "pnl": pnl,
                "mode": pos.get("mode", "SIMULATED")
            })
            state.log_event("ORDER_CLOSED", f"[{pos.get('mode', 'SIMULATED')}] CLOSED {symbol} @ ${price:.2f} (PnL: ${pnl:+.2f})")
            return

        try:
            loop = asyncio.get_running_loop()
            t_cl = time.perf_counter_ns()
            from core.market_hours import us_session, PRE, POST
            if us_session() in (PRE, POST):
                # close_position sends a market order, which Alpaca rejects outside
                # regular hours. Exit with a marketable extended-hours limit instead.
                ref = float(pos.get("current_price") or pos["avg_entry_price"])
                bid = tick.bid if (tick and tick.bid) else ref
                ask = tick.ask if (tick and tick.ask) else ref
                result = await loop.run_in_executor(None, self._submit_extended_exit,
                                                    symbol, bid, ask)
                latency.record_ns("order_close", t_cl)
                self.close_failures.pop(symbol, None)
                # A limit exit is not a fill. Keep the position until the broker
                # stops reporting it (booked then by book_vanished_positions), and
                # hold off re-sending: dropping it here let the next sync re-add it
                # and the sentinel sell it again, overselling into a short.
                self.pending_exits[symbol] = decision
                self.close_retry_after[symbol] = time.time() + min(
                    settings.EXTENDED_EXIT_REPRICE_SECONDS, self._max_close_wait(decision))
                if result == "submitted":
                    state.log_event("ORDER_SUBMITTED",
                                    f"Extended-hours exit order placed for {symbol}; closes when it fills")
                return
            # An equity bought with a bracket still has its stop and target legs
            # open, and they hold every share: close_position then fails with
            # "qty must be > 0". Clear the symbol's orders first.
            await loop.run_in_executor(None, self._cancel_open_orders, symbol)
            close_order = await loop.run_in_executor(None, self.trading_client.close_position, symbol)
            latency.record_ns("order_close", t_cl)
            self.close_retry_after.pop(symbol, None)
            self.close_failures.pop(symbol, None)
            self.pending_exits.pop(symbol, None)
            self.exit_attempts.pop(symbol, None)
            latency_ms = (time.time() - t0) * 1000
            # Book the position's last-known unrealised PnL as realised. The exact
            # fill price arrives on the next broker sync; this keeps the daily-loss
            # breaker responsive rather than blind until that sync lands.
            exit_price = tick.price if tick else float(pos.get("current_price") or pos["avg_entry_price"])
            pnl = round(float(pos.get("unrealized_pl") or
                              (exit_price - pos["avg_entry_price"]) * pos["qty"]), 2)
            self.recently_closed[symbol] = time.time()
            state.book_realized_pnl(symbol, pnl)
            state.active_positions.pop(symbol, None)
            record = self._record_closed_trade(symbol, pos, exit_price, pnl, decision)
            fills.track(symbol, float(pos["qty"]), float(pos["avg_entry_price"]), pnl, record,
                        order_id=getattr(close_order, "id", None),
                        since=float(pos.get("opened_at") or time.time() - 86400))
            state.log_event("ORDER_CLOSED",
                            f"Alpaca CLOSED {symbol} in {latency_ms:.1f}ms (PnL ${pnl:+,.2f}, "
                            f"day ${state.realized_pnl_today:+,.2f})")
            loop.run_in_executor(None, self.sync_account_and_positions)
        except Exception as e:
            err_str = str(e)
            logger.error(f"Failed to close position in {symbol}: {err_str}")
            # If position was already closed or not found on broker, remove from active positions to stop retry loop
            if ("Not Found" in err_str or "not found" in err_str or "404" in err_str
                    or "does not exist" in err_str):
                # Already closed at the broker: still book it against the budget.
                gone = state.active_positions.pop(symbol, None)
                if gone:
                    self.vanished_positions.append((symbol, gone))
                    self.book_vanished_positions()
                state.log_event("ORDER_SYNC", f"Position {symbol} already liquidated on broker.")
                loop = asyncio.get_running_loop()
                loop.run_in_executor(None, self.sync_account_and_positions)
            else:
                # Back off instead of retrying on the sentinel's next tick: a close
                # the broker keeps rejecting was otherwise resent every second.
                n = self.close_failures.get(symbol, 0) + 1
                self.close_failures[symbol] = n
                wait = min(5.0 * 2 ** (n - 1), self._max_close_wait(decision))
                self.close_retry_after[symbol] = time.time() + wait
                state.log_event("ORDER_ERROR",
                                f"Failed to close {symbol} (attempt {n}), retrying in {wait:.0f}s: {err_str}")

    @staticmethod
    def _max_close_wait(decision: TradeDecision) -> float:
        """Longest a failed or unfilled close may sit before it is tried again."""
        return settings.FORCED_EXIT_MAX_WAIT_SECONDS if decision.forced else 120.0

    def _cancel_open_orders(self, symbol: str, wait_s: float = 3.0):
        """
        Cancels every open order for `symbol` and waits until the broker shows
        none, so the shares they held are free to sell. Alpaca acknowledges a
        cancel before it takes effect; selling in that window is rejected.
        Runs in the sync thread.
        """
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
        orders = self.trading_client.get_orders(req)
        if not orders:
            return
        for o in orders:
            try:
                self.trading_client.cancel_order_by_id(o.id)
            except Exception as e:
                logger.warning(f"Could not cancel order {o.id} for {symbol}: {e}")
        deadline = time.time() + wait_s
        while time.time() < deadline:
            time.sleep(0.25)
            if not self.trading_client.get_orders(req):
                return
        logger.warning(f"{symbol}: open orders still pending cancel after {wait_s:.0f}s")

    def _submit_extended_exit(self, symbol: str, bid: float, ask: float) -> str:
        """
        Extended-hours exit: a DAY limit order priced PREMARKET_EXIT_OFFSET_PCT
        through the touch so it fills like a market order would. Runs in the
        sync thread. Returns "pending" when an exit is already working, else
        "submitted".

        Size and direction come from the broker's live position, never from our
        copy: that copy can lag a fill, and selling its stale quantity again is
        how a long became a short. A short (from such an oversell) is bought back.
        An exit already working is left alone until it is older than
        EXTENDED_EXIT_REPRICE_SECONDS, then cancelled and re-placed at a fresh price.
        """
        from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest
        from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
        qty = float(self.trading_client.get_open_position(symbol).qty)
        side = OrderSide.SELL if qty > 0 else OrderSide.BUY
        side_name = "sell" if qty > 0 else "buy"
        now = time.time()
        for o in self.trading_client.get_orders(
                GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])):
            ours = (o.client_order_id or "").startswith(ORDER_ID_PREFIX)
            if (ours and side_name in str(o.side).lower() and o.submitted_at
                    and now - o.submitted_at.timestamp() < settings.EXTENDED_EXIT_REPRICE_SECONDS):
                return "pending"
        # Frees the shares: bracket legs, or our own stale exit being repriced.
        self._cancel_open_orders(symbol)
        # Re-read after the cancel: a stale exit may have filled in the meantime.
        qty = float(self.trading_client.get_open_position(symbol).qty)
        if (qty > 0) != (side == OrderSide.SELL):
            return "pending"
        n, last_at = self.exit_attempts.get(symbol, (0, 0.0))
        if now - last_at > 4 * settings.EXTENDED_EXIT_REPRICE_SECONDS:
            n = 0   # a fresh exit, not a re-price of one that has been going on
        off = min(settings.PREMARKET_EXIT_OFFSET_PCT + n * settings.EXTENDED_EXIT_STEP_PCT,
                  max(settings.EXTENDED_EXIT_MAX_OFFSET_PCT,
                      settings.PREMARKET_EXIT_OFFSET_PCT)) / 100.0
        price = bid * (1 - off) if side == OrderSide.SELL else ask * (1 + off)
        self.trading_client.submit_order(LimitOrderRequest(
            symbol=symbol,
            qty=abs(qty),
            side=side,
            time_in_force=TimeInForce.DAY,
            limit_price=round(price, 2),
            extended_hours=True,
            client_order_id=f"{ORDER_ID_PREFIX}{uuid.uuid4().hex[:20]}",
        ))
        self.exit_attempts[symbol] = (n + 1, now)   # only an order that went out counts
        return "submitted"

    def _record_closed_trade(self, symbol: str, pos: Dict[str, Any], exit_price: float,
                             pnl: float, decision: Optional[TradeDecision] = None):
        """
        Writes a structured outcome record for a closed position.

        This is the substrate for attribution and for any learning loop: the reward
        signal is the R-MULTIPLE (pnl / initial risk), not raw dollars, so outcomes
        stay comparable across position sizes and asset classes. The setup context
        is captured at close so a later analysis can ask "which conditions actually
        preceded profit" rather than guessing.
        """
        entry = float(pos.get("avg_entry_price") or 0.0)
        qty = float(pos.get("qty") or 0.0)
        initial_risk = float(pos.get("dollar_risk") or 0.0)
        if initial_risk <= 0 and pos.get("stop_loss") and entry:
            initial_risk = abs(entry - float(pos["stop_loss"])) * qty

        r_multiple = round(pnl / initial_risk, 3) if initial_risk > 0 else None
        held_seconds = round(time.time() - float(pos.get("opened_at") or time.time()), 1)

        record = {
            "time": time.time(),
            "symbol": symbol,
            "side": "SELL",
            "qty": qty,
            "price": exit_price,
            "entry_price": entry,
            "pnl": pnl,
            "initial_risk": round(initial_risk, 2),
            "r_multiple": r_multiple,
            "held_seconds": held_seconds,
            "exit_reason": (decision.reason if decision else "") or pos.get("bot_thesis", ""),
            "mode": pos.get("mode", "ALPACA_PAPER"),
            # Setup context as it stood at entry, for later attribution
            "setup": {
                "laya_pos": pos.get("laya_pos"),
                "laya_neg": pos.get("laya_neg"),
                "consensus_score": pos.get("consensus_score"),
                "conviction_tier": pos.get("conviction_tier"),
                "entry_rsi": pos.get("entry_rsi"),
                "entry_spread": pos.get("entry_spread"),
                "entry_atr_pct": pos.get("entry_atr_pct"),
                "stop_pct": pos.get("stop_pct"),
            },
        }
        state.recent_trades.append(record)
        state.closed_trades.append(record)

        # Persist to agent memory WITHOUT awaiting: a slow graph write must never
        # delay a liquidation. Failures are logged inside remember_trade.
        record["strategy"] = pos.get("strategy")
        record["entry_strategy"] = pos.get("entry_strategy")
        record["entry_regime"] = pos.get("entry_regime")
        record["bot_id"] = pos.get("bot_id")
        record["opened_at"] = pos.get("opened_at")
        try:
            from memory.agent_memory import agent_memory
            if agent_memory.enabled:
                asyncio.create_task(agent_memory.remember_trade(record))
        except Exception as e:
            logger.debug(f"Could not schedule memory write: {e}")
        state.log_event(
            "TRADE_CLOSED",
            f"{symbol} closed: PnL ${pnl:+,.2f}"
            + (f" ({r_multiple:+.2f}R)" if r_multiple is not None else "")
            + f" after {held_seconds:.0f}s"
        )
        return record

    async def start_sync_loop(self):
        """Continuously keeps account cash and positions synchronized with Alpaca every 5 seconds"""
        while True:
            try:
                if not self.is_mock_mode and self.trading_client:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, self.sync_account_and_positions)
                    from engine.sentinel_agent import sentinel_registry
                    sentinel_registry.sync_with_positions(state.active_positions)
                    if self._mark_task is None or self._mark_task.done():
                        self._mark_task = asyncio.create_task(self._run_mark_loop())
                self.book_vanished_positions()
                fills.apply()
                state.roll_trading_day_if_needed()
                state.update_peak_equity()
                from core.pnl_ledger import pnl_ledger
                pnl_ledger.touch()
                await asyncio.sleep(5.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Account sync loop note: {e}")
                await asyncio.sleep(5.0)

    def _feed_is_stale(self, sym: str, now: float) -> bool:
        tick = state.latest_prices.get(sym)
        if tick is None or self._remark_tick.get(sym) is tick:
            return True
        return now - tick.timestamp > settings.POSITION_MAX_PRICE_AGE_SECONDS

    async def _run_mark_loop(self):
        """
        Once a second, while any held position's feed is quiet, refreshes the
        broker marks and prices those positions from them. Costs nothing while
        every position is streaming trade prints.
        """
        loop = asyncio.get_running_loop()
        while True:
            try:
                now = time.time()
                live = [s for s, p in state.active_positions.items()
                        if p.get("mode") == "ALPACA_PAPER"]
                if self.trading_client and any(self._feed_is_stale(s, now) for s in live):
                    positions = await loop.run_in_executor(None, self.trading_client.get_all_positions)
                    for p in positions:
                        self.broker_marks[p.symbol] = float(p.current_price)
                    await self.remark_stale_positions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Not debug: a silent failure here leaves positions unpriced.
                logger.warning(f"Broker mark refresh failed: {e}")
            await asyncio.sleep(settings.POSITION_MARK_POLL_SECONDS)

    async def remark_stale_positions(self):
        """
        Prices a held position from the broker's mark when the data feed has gone
        quiet for it. The feed can stall (no pre-market prints on IEX, a symbol
        missing from the stream); the position then sat on its last price with
        its stop and target never re-checked, while the broker's mark moved on.
        The mark goes through the same path as a streamed trade print, so the
        sentinel treats it identically and indicators are left untouched.
        """
        from feeds.alpaca_stream import market_stream
        now = time.time()
        for sym, pos in list(state.active_positions.items()):
            mark = self.broker_marks.get(sym)
            if not mark or mark <= 0 or pos.get("mode") != "ALPACA_PAPER":
                continue
            tick = state.latest_prices.get(sym)
            age = now - tick.timestamp if tick else float("inf")
            if not self._feed_is_stale(sym, now):
                self._remark_logged.discard(sym)
                continue
            # Our own last injection at the same price: nothing new to evaluate.
            if tick is not None and self._remark_tick.get(sym) is tick and mark == tick.price:
                continue
            if sym not in self._remark_logged:
                self._remark_logged.add(sym)
                state.log_event("PRICE_STALE",
                    f"{sym}: no feed price for {age:.0f}s; tracking the broker mark "
                    f"${mark:,.6g} so its stop and target stay live")
            await market_stream.on_position_trade(sym, mark, is_trade=False)
            self._remark_tick[sym] = state.latest_prices.get(sym)

    async def emergency_close_all(self):
        """Panic / Kill Switch: Instantly closes all active positions and pauses trading"""
        state.is_trading_active = False
        state.log_event("KILL_SWITCH", "EMERGENCY KILL SWITCH TRIGGERED: Halting trading and closing positions")

        if self.is_mock_mode:
            for sym in list(state.active_positions.keys()):
                await self._execute_close(TradeDecision(symbol=sym, action="CLOSE", close=True))
            return

        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.trading_client.close_all_positions, True)
            for sym, pos in list(state.active_positions.items()):
                self.vanished_positions.append((sym, pos))
            state.active_positions.clear()
            self.book_vanished_positions()
            state.log_event("KILL_SWITCH", "All positions liquidated on Alpaca.")
            await loop.run_in_executor(None, self.sync_account_and_positions)
        except Exception as e:
            logger.error(f"Error executing emergency close all: {e}")

executor = AlpacaExecutor()
