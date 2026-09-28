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
    for mod, name in ((pl, "pnl_ledger.json"), (cp, "capital_plan.json"), (mf, "market_filter.json")):
        monkeypatch.setattr(mod, "_DATA_DIR", str(tmp_path))
        monkeypatch.setattr(mod, "_PATH", str(tmp_path / name))
    # The ledger singleton was loaded from the real file at import: start empty.
    monkeypatch.setattr(pl.pnl_ledger, "days", {})
    yield
