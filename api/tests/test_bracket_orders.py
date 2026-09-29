"""Every regular-session entry is a real bracket: its stop and target live at the broker."""
import asyncio
import time
import types

from core.state import state, PriceTick, QuantMetrics, TradeDecision


def test_entry_is_a_broker_bracket(monkeypatch):
    from alpaca.trading.enums import OrderClass
    import core.market_hours as mh
    from engine.executor import executor
    from engine.risk_guard import risk_guard
    sent = []
    client = types.SimpleNamespace(submit_order=lambda req: sent.append(req) or types.SimpleNamespace(id="o1"))
    monkeypatch.setattr(executor, "trading_client", client)
    monkeypatch.setattr(executor, "is_mock_mode", False)
    monkeypatch.setattr(executor, "alpaca_tradable_symbols", {"AMD"})
    monkeypatch.setattr(executor, "sync_account_and_positions", lambda: None)
    monkeypatch.setattr(mh, "us_session", lambda *a, **k: mh.REGULAR)
    monkeypatch.setattr(state, "active_positions", {})
    monkeypatch.setattr(state, "is_trading_active", True)
    monkeypatch.setattr(state, "latest_prices", {"AMD": PriceTick(symbol="AMD", price=100.0, bid=99.99, ask=100.01,
                                                                  volume=0, timestamp=time.time())})
    monkeypatch.setattr(state, "quant_metrics", {"AMD": QuantMetrics(symbol="AMD", atr=1.0)})
    from engine.diversification import diversification
    monkeypatch.setattr(diversification, "reserve", lambda *a, **k: None)      # no leak into later tests
    for name in ("pending_entry_context", "awaiting_fill", "inflight_notional"):
        monkeypatch.setattr(executor, name, {})
    monkeypatch.setattr(state, "recent_trades", type(state.recent_trades)(maxlen=state.recent_trades.maxlen))
    monkeypatch.setattr(risk_guard, "can_open_position", lambda sym: (True, ""))
    monkeypatch.setattr(risk_guard, "calculate_order_sizing", lambda **kw: (
        12, 99.0, 102.0, {"allocated_dollars": 1200.0, "allocated_pct": 5.0, "rationale": "test"}))
    asyncio.run(executor._execute_buy(TradeDecision(symbol="AMD", action="BUY", buy_prob=0.7, reason="test")))
    assert len(sent) == 1
    req = sent[0]
    assert req.order_class == OrderClass.BRACKET and req.qty == 12
    assert req.stop_loss.stop_price == 99.0 and req.take_profit.limit_price == 102.0
