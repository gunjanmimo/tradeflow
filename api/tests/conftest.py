"""
Test isolation: nothing a test books may reach the real files under api/data/.

The ledger, the capital plan and the market filter persist on every change. A
test that closes a position would otherwise append to the operator's live
daily P&L ledger and budget.
"""
import pytest


@pytest.fixture(autouse=True)
def _isolated_state_files(tmp_path, monkeypatch):
    import core.pnl_ledger as pl
    import core.capital_plan as cp
    import core.market_filter as mf
    import engine.smart_money as sm
    import scout.service as sc
    import desk.desk as dk
    import core.universe as cu
    import core.state as cs
    from core.config import settings
    for mod, name in ((pl, "pnl_ledger.json"), (cp, "capital_plan.json"), (mf, "market_filter.json"),
                      (sm, "smart_money.json")):
        monkeypatch.setattr(mod, "_DATA_DIR", str(tmp_path))
        monkeypatch.setattr(mod, "_PATH", str(tmp_path / name))
    # The ledger singleton was loaded from the real file at import: start empty.
    monkeypatch.setattr(pl.pnl_ledger, "days", {})
    monkeypatch.setattr(sm, "smart_money", sm.SmartMoneyBook())
    monkeypatch.setattr(sc, "_DATA_DIR", str(tmp_path / "scout"))
    monkeypatch.setattr(sc, "_PATH", str(tmp_path / "scout" / "rankings.jsonl"))
    monkeypatch.setattr(dk, "_DATA_DIR", str(tmp_path / "desk"))
    monkeypatch.setattr(cu, "_PATH", str(tmp_path / "universe_sec.json"))
    monkeypatch.setattr(cs, "_RISK_PATH", str(tmp_path / "risk_dial.json"))
    import engine.profit_manager as pm
    monkeypatch.setattr(pm, "_PATH", str(tmp_path / "position_meta.json"))
    monkeypatch.setattr(pm, "meta", pm.PositionMeta())
    # The state singleton read the operator's saved dial at import: tests start at the default.
    from core.risk_profile import DEFAULT_RISK_FACTOR
    monkeypatch.setattr(cs.state, "_risk_factor", DEFAULT_RISK_FACTOR)
    monkeypatch.setattr(dk, "_PATH", str(tmp_path / "desk" / "cases.jsonl"))
    # The desk needs a local LLM; tests of everything else trade without it.
    # tests/test_desk.py switches it back on.
    monkeypatch.setattr(settings, "DESK_ENABLED", False)
    yield
