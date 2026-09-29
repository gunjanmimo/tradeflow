"""
The trade desk: observe, analyst, critic, decision -- and the executor's gate.

  * a signal is observed first; one that disappears fades the case
  * the analyst and critic must both agree and their mean probability must
    clear max(DESK_MIN_PROB, breakeven + the risk dial's margin)
  * a veto or a pass holds the symbol off (cooldown); an approval clears one buy
    for a while and lapses if the price drifts
  * a reasoning budget that runs out gets an answer from the model's notes
  * an unreachable LLM opens no cases, and the executor refuses unapproved buys
"""
import asyncio
import time

import pytest

from core.config import settings
from core.state import state, PriceTick
import desk.desk as dk
from desk.llm import LLMError


class FakeLLM:
    def __init__(self, analyst=None, critic=None, ok=True, truncate=False, fail=None):
        self.analyst = analyst or {"lean": "BULLISH", "p_target_first": 0.66, "confidence": "medium",
                                   "thesis": "up", "key_evidence": [], "risks": []}
        self.critic = critic or {"verdict": "APPROVE", "p_target_first": 0.62, "objections": [], "summary": "ok"}
        self.ok, self.truncate, self.fail = ok, truncate, fail
        self.calls = []
        self.models = []            # (role, model, num_gpu) per call
        self.critic_saw = ""

    async def health(self, max_age_s=0):
        return {"ok": self.ok, "detail": "fake", "models": [], "at": time.time()}

    async def chat(self, model, messages, schema, think, on_thinking=None, on_content=None, max_tokens=None,
                   options=None):
        role = "critic" if "verdict" in schema["properties"] else "analyst"
        self.calls.append(role)
        self.models.append((role, model, (options or {}).get("num_gpu")))
        if role == "critic":
            self.critic_saw = messages[-1]["content"]
        if self.fail == role:
            raise LLMError("boom")
        if think and on_thinking:
            on_thinking("thinking about it... ")
        if role == "analyst" and self.truncate:
            return {"answer": None, "thinking": "notes", "tokens": 10, "seconds": 1.0, "truncated": True}
        return {"answer": dict(self.analyst if role == "analyst" else self.critic), "thinking": "t",
                "tokens": 5, "seconds": 0.1, "truncated": False}

    async def answer_from_notes(self, model, messages, schema, notes, on_content=None, options=None):
        self.calls.append("notes")
        return {"answer": dict(self.analyst), "thinking": "", "tokens": 3, "seconds": 0.1, "truncated": False}


@pytest.fixture
def desk(monkeypatch):
    monkeypatch.setattr(settings, "DESK_ENABLED", True)
    monkeypatch.setattr(settings, "DESK_REQUIRED", True)
    d = dk.TradeDesk()
    monkeypatch.setattr(dk, "desk", d)
    fake = FakeLLM()
    monkeypatch.setattr(dk, "ollama", fake)
    d.llm = {"ok": True}
    d._health_at = time.time() + 3600          # no health polling during the test
    monkeypatch.setattr(dk.brief_mod, "build", lambda sym, case: {
        "text": "CASE FILE", "plan": {"breakeven_p": 0.33}, "price": 100.0})
    monkeypatch.setattr(state, "latest_prices", {"AMD": PriceTick(symbol="AMD", price=100.0, bid=99.99, ask=100.01,
                                                                  volume=0, timestamp=time.time())})
    import scout.watcher as w
    monkeypatch.setattr(w, "session_levels", lambda rows, m: {"open": 99.0, "vwap": 99.5, "bars": 10})
    return d, fake


def _run(d, sym="AMD", observe=None):
    """Drives one case to a verdict: keeps the signal alive, lets the observation end, runs the review."""
    plan = {"stop": 99.0, "target": 102.0}
    assert d.propose(sym, 0.7, "signal", "scout", plan) == "observing"
    case = d.cases[sym]
    case["opened_at"] -= (observe if observe is not None else settings.DESK_OBSERVE_SECONDS) + 1
    asyncio.run(_drive(d, sym))
    return case


async def _drive(d, sym):
    for _ in range(5):
        d.propose(sym, 0.7, "signal", "scout", None)
        await d.step()
        if d._worker:
            await d._worker
        if sym not in d.cases:
            return


def test_approved_case_clears_one_buy(desk):
    d, fake = desk
    case = _run(d)
    assert case["stage"] == "approved" and fake.calls == ["analyst", "critic"]
    assert case["decision"]["p_final"] == pytest.approx(0.64)
    assert case["decision"]["need"] == pytest.approx(max(settings.DESK_MIN_PROB, 0.33 + dk.edge_margin()))
    assert d.propose("AMD", 0.7, "signal", "scout", None) == "cleared"
    ok, cid = d.clearance("AMD", 100.1)
    assert ok and cid == case["id"]
    d.executed("AMD", {"qty": 1, "price": 100.1})
    assert case["stage"] == "executed" and d.clearance("AMD", 100.1)[0] is False
    assert [e["agent"] for e in case["timeline"]][:2] == ["Observer", "Observer"]


def test_critic_veto_rejects_and_cools_down(desk):
    d, fake = desk
    fake.critic = {"verdict": "VETO", "p_target_first": 0.4, "objections": ["chasing"], "summary": "no"}
    case = _run(d)
    assert case["stage"] == "rejected" and "veto" in case["decision"]["reason"]
    assert d.propose("AMD", 0.7, "signal", "scout", None) == "rejected"
    ok, why = d.clearance("AMD", 100.0)
    assert not ok and "rejected" in why


def test_analyst_pass_skips_the_critic(desk):
    d, fake = desk
    fake.analyst = {**fake.analyst, "lean": "BEARISH", "p_target_first": 0.3}
    case = _run(d)
    assert case["stage"] == "rejected" and fake.calls == ["analyst"]


def test_probability_must_clear_breakeven_plus_margin(desk, monkeypatch):
    d, fake = desk
    monkeypatch.setattr(settings, "DESK_MIN_PROB", 0.0)
    monkeypatch.setattr(dk.brief_mod, "build", lambda sym, case: {
        "text": "x", "plan": {"breakeven_p": 0.6}, "price": 100.0})          # a far target, a near stop
    case = _run(d)
    assert case["stage"] == "rejected" and case["decision"]["need"] == pytest.approx(0.6 + dk.edge_margin())


def test_a_signal_that_disappears_fades(desk):
    d, _ = desk
    d.propose("AMD", 0.7, "signal", "scout", None)
    d.cases["AMD"]["last_signal_at"] -= settings.DESK_SIGNAL_GAP_SECONDS + 1
    asyncio.run(d.step())
    assert "AMD" not in d.cases and d.history[0]["stage"] == "faded"


def test_a_spent_reasoning_budget_is_answered_from_notes(desk):
    d, fake = desk
    fake.truncate = True
    case = _run(d)
    assert fake.calls == ["analyst", "notes", "critic"] and case["stage"] == "approved"


def test_llm_errors_and_outages_block(desk):
    d, fake = desk
    fake.fail = "critic"
    case = _run(d)
    assert case["stage"] == "error" and d.clearance("AMD", 100.0)[0] is False
    d2 = dk.TradeDesk()
    d2.llm = {"ok": False, "detail": "down"}
    assert d2.propose("AMD", 0.7, "signal", "scout", None) == "offline" and not d2.cases


def test_approval_lapses_when_the_price_drifts(desk):
    d, _ = desk
    _run(d)
    assert d.clearance("AMD", 100.0 * (1 + settings.DESK_MAX_PRICE_DRIFT_PCT / 100 + 0.002))[0] is False
    assert "AMD" not in d.cleared and d.history[0]["stage"] == "expired"


def test_the_executor_refuses_unapproved_buys(desk, monkeypatch):
    from core.state import TradeDecision, QuantMetrics
    from engine.executor import executor
    from engine.risk_guard import risk_guard
    d, _ = desk
    monkeypatch.setattr(executor, "is_mock_mode", True)
    monkeypatch.setattr(state, "active_positions", {})
    monkeypatch.setattr(state, "quant_metrics", {"AMD": QuantMetrics(symbol="AMD", atr=1.0)})
    monkeypatch.setattr(risk_guard, "can_open_position", lambda sym: (True, ""))
    monkeypatch.setattr(risk_guard, "calculate_order_sizing", lambda **kw: (
        1, 99.0, 102.0, {"allocated_dollars": 100.0, "allocated_pct": 1.0, "rationale": "test"}))
    buy = TradeDecision(symbol="AMD", action="BUY", buy_prob=0.7, reason="test")
    asyncio.run(executor._execute_buy(buy))
    assert "AMD" not in state.active_positions
    assert any(l["level"] == "DESK_BLOCK" and "AMD" in l["message"] for l in list(state.logs)[-5:])
    case = _run(d)
    asyncio.run(executor._execute_buy(buy))
    pos = state.active_positions.get("AMD")
    assert pos and pos["desk_case"] == case["id"] and pos["desk_p"] == pytest.approx(0.64)
    assert case["stage"] == "executed"


def test_numbers_decide_not_the_label(desk):
    """In replays the model said PASS on estimates above the hurdle: its label is ignored."""
    d, fake = desk
    fake.analyst = {**fake.analyst, "lean": "NEUTRAL", "p_target_first": 0.62}
    case = _run(d)
    assert fake.calls == ["analyst", "critic"] and case["stage"] == "approved"


def test_analyst_and_critic_are_different_models_placed_on_the_gpu(desk):
    """Two models for two opinions; the critic never sees the analyst's number (small models echoed it)."""
    d, fake = desk
    _run(d)
    roles = {role: (model, gpu) for role, model, gpu in fake.models}
    assert roles["analyst"][0] == settings.DESK_ANALYST_MODEL
    assert roles["critic"][0] == settings.DESK_CRITIC_MODEL != settings.DESK_ANALYST_MODEL
    assert roles["analyst"][1] is None                                  # all layers on the GPU
    assert roles["critic"][1] == settings.DESK_CRITIC_NUM_GPU           # split GPU + RAM
    assert "0.66" not in fake.critic_saw and "thesis" in fake.critic_saw


def test_margin_follows_the_risk_dial():
    assert dk.edge_margin(1) == pytest.approx(settings.DESK_EDGE_MARGIN_CAUTIOUS)
    assert dk.edge_margin(10) == pytest.approx(settings.DESK_EDGE_MARGIN_AGGRESSIVE)
    assert dk.edge_margin(1) > dk.edge_margin(5) > dk.edge_margin(8) > dk.edge_margin(10)


def test_the_case_file_carries_the_users_risk_appetite(monkeypatch):
    from desk import brief as b
    monkeypatch.setattr(state, "_risk_factor", 8)
    r = b.risk_appetite({"qty": 10}, 100.0, 99.0, 102.0)
    assert r["dial"] == 8 and "thinner edges" in r["stance"]
    assert r["trade_risk_usd"] == pytest.approx(10.0) and r["trade_reward_usd"] == pytest.approx(20.0)
    text = b.render({**_BRIEF, "risk": r})
    assert "RISK APPETITE: the user set the risk dial to 8/10" in text


_BRIEF = {"symbol": "AMD", "name": "AMD", "sector": "IT", "signal": {"strategy": "scout", "buy_prob": 0.7, "reason": "x"},
          "price": 100.0, "chg_today_pct": 1.0, "since_open_pct": 0.5, "vs_vwap_pct": 0.2, "vwap": 99.8,
          "ret_5m_pct": 0.1, "ret_15m_pct": 0.2, "ret_60m_pct": 0.3,
          "trend": {"label": "uptrend", "direction": 0.5, "confidence": 0.6, "reversal_down": False, "reversal_up": False},
          "rsi": 60.0, "spread_pct": 0.02, "candles_5m": [], "observation": {}, "daily": {},
          "plan": {"stop": 99.0, "target": 102.0, "stop_pct": -1.0, "target_pct": 2.0, "breakeven_p": 0.33},
          "sentiment": {"pos": 0.5, "neg": 0.5, "n": 0, "agreement": 0.0, "stale": True}, "news": [],
          "market": {"spy_trend": "range", "spy_direction": 0.0, "spy_since_open_pct": 0.1}, "scout": None}
