"""
Sentiment router: Jev first, Laya once Jev is out of quota or keeps failing.
"""
import asyncio
import types

import pytest

import sentiment.router as r
from sentiment.jev_service import JevUnavailable


def _rec(src):
    return types.SimpleNamespace(pos_prob=0.7, neg_prob=0.1, src=src)


@pytest.fixture
def router(monkeypatch):
    calls = {"jev": 0, "laya": 0, "jev_error": None}

    async def jev(sym, text, persist=True):
        calls["jev"] += 1
        if calls["jev_error"]:
            raise calls["jev_error"]
        return _rec("jev")

    async def laya(sym, text, persist=True):
        calls["laya"] += 1
        return _rec("laya")

    monkeypatch.setattr(r.jev_service, "score_headline", jev)
    monkeypatch.setattr(r.laya_service, "score_headline", laya)
    monkeypatch.setattr(r.settings, "TYPESAFE_MAX_CONSECUTIVE_FAILURES", 3)
    router = r.SentimentRouter()
    router.active = "jev"
    router.calls = calls
    return router


def _score(router):
    return asyncio.run(router.score_headline("AAPL", "Apple beats"))


def test_jev_is_used_first(router):
    assert _score(router).src == "jev"
    assert router.calls["laya"] == 0


def test_quota_error_switches_to_laya_for_good(router):
    router.calls["jev_error"] = JevUnavailable("HTTP 429", fatal=True, status=429)
    assert _score(router).src == "laya"  # failed headline re-scored by Laya
    assert router.active == "laya"
    router.calls["jev_error"] = None
    assert _score(router).src == "laya"
    assert router.calls["jev"] == 1  # Jev never called again


def test_transient_errors_fall_back_after_threshold(router):
    router.calls["jev_error"] = JevUnavailable("timeout")
    for _ in range(2):
        assert _score(router).src == "laya"
        assert router.active == "jev"
    _score(router)
    assert router.active == "laya"


def test_success_resets_failure_count(router):
    router.calls["jev_error"] = JevUnavailable("timeout")
    _score(router); _score(router)
    router.calls["jev_error"] = None
    _score(router)
    router.calls["jev_error"] = JevUnavailable("timeout")
    _score(router); _score(router)
    assert router.active == "jev"
