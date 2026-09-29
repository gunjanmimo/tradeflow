"""The risk dial the user sets on the dashboard survives a restart; 4 only until it is first set."""
import core.state as cs
from core.risk_profile import DEFAULT_RISK_FACTOR


def test_the_dashboard_dial_survives_a_restart():
    assert DEFAULT_RISK_FACTOR == 4
    assert cs._load_risk_factor() == 4                      # never set: the default
    s = cs.InMemoryState()
    s.risk_factor = 7                                       # what POST /api/risk-factor does
    assert cs._load_risk_factor() == 7
    assert cs.InMemoryState().risk_factor == 7              # a fresh process reads it back
    s.risk_factor = 99                                      # clamped, and saved clamped
    assert cs._load_risk_factor() == 10
