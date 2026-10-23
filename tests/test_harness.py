"""Offline tests for deai.harness against a mocked SlopTotal (no server needed)."""

import asyncio

import httpx
import pytest

from deai import harness
from deai.harness import HarnessError, SlopTotalClient, TextRejected

TEXT = "The quick brown fox jumps over the lazy dog, again and again, all afternoon long."

ENGINES = [
    {"key": "classifier_fakespot", "name": "Fakespot", "engine_type": "neural"},
    {"key": "gltr", "name": "GLTR", "engine_type": "statistical"},
]
QUICK = {"score": 61.0, "verdict": "slop", "engines": [
    {"engine": "classifier_fakespot", "score": 80.0, "verdict": "slop"},
    {"engine": "linguistic", "score": 0.0, "verdict": "clean"},
]}
ANALYZE = {"id": "abc123", "overall_score": 72.5, "overall_verdict": "Slop detected",
           "created_at": "2026-10-23T14:00:00", "engine_results": [
               {"engine_name": "Fakespot", "score": 0.9},
               {"engine_name": "GLTR", "score": 0.2},
           ]}


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    async def instant(attempt):
        return None
    monkeypatch.setattr(SlopTotalClient, "_backoff", staticmethod(instant))


def run(handler, coro_fn, **kw):
    async def go():
        async with SlopTotalClient(transport=httpx.MockTransport(handler), **kw) as st:
            return await coro_fn(st)
    return asyncio.run(go())


def routes(**extra):
    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if path in extra:
            return extra[path](req)
        if path == "/api/engines":
            return httpx.Response(200, json=ENGINES)
        if path == "/api/quick-score":
            return httpx.Response(200, json=QUICK)
        if path == "/api/analyze":
            return httpx.Response(200, json=ANALYZE)
        return httpx.Response(404)
    return handler


def test_quick_score_normalised_to_unit_scale():
    s = run(routes(), lambda st: st.quick_score(TEXT))
    assert s.overall == 61.0
    assert s.engines == {"classifier_fakespot": 0.8, "linguistic": 0.0}
    assert s.endpoint == "quick" and s.attempts == 1


def test_analyze_maps_display_names_to_keys():
    s = run(routes(), lambda st: st.analyze(TEXT))
    assert s.overall == 72.5
    assert s.engines == {"classifier_fakespot": 0.9, "gltr": 0.2}
    assert s.report_id == "abc123"


def test_queued_request_is_polled_until_done():
    polls = {"n": 0}

    def ticket(req):
        polls["n"] += 1
        if polls["n"] < 3:
            return httpx.Response(202, json={"status": "queued", "position": 1, "estimated_wait_ms": 1})
        return httpx.Response(200, json=QUICK)

    handler = routes(**{
        "/api/quick-score": lambda req: httpx.Response(202, json={"status": "queued", "ticket_id": "t1"}),
        "/api/queue/ticket/t1": ticket,
    })
    s = run(handler, lambda st: st.quick_score(TEXT))
    assert s.overall == 61.0
    assert polls["n"] == 3


def test_expired_ticket_resubmits():
    calls = {"n": 0}

    def quick(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(202, json={"status": "queued", "ticket_id": "gone"})
        return httpx.Response(200, json=QUICK)

    handler = routes(**{
        "/api/quick-score": quick,
        "/api/queue/ticket/gone": lambda req: httpx.Response(404, json={"detail": "expired"}),
    })
    s = run(handler, lambda st: st.quick_score(TEXT))
    assert s.attempts == 2


def test_429_then_success():
    calls = {"n": 0}

    def quick(req):
        calls["n"] += 1
        if calls["n"] <= 2:
            return httpx.Response(429, json={"error": "queue full", "retry_after": 0})
        return httpx.Response(200, json=QUICK)

    s = run(routes(**{"/api/quick-score": quick}), lambda st: st.quick_score(TEXT))
    assert s.attempts == 3


def test_5xx_exhausts_retries():
    handler = routes(**{"/api/analyze": lambda req: httpx.Response(500, json={"error": "Analysis failed"})})
    with pytest.raises(HarnessError, match="Analysis failed"):
        run(handler, lambda st: st.analyze(TEXT), max_retries=3)


def test_400_is_not_retried():
    calls = {"n": 0}

    def quick(req):
        calls["n"] += 1
        return httpx.Response(400, json={"error": "Text must be at least 50 characters."})

    with pytest.raises(TextRejected):
        run(routes(**{"/api/quick-score": quick}), lambda st: st.quick_score(TEXT))
    assert calls["n"] == 1


def test_short_text_rejected_client_side():
    with pytest.raises(TextRejected):
        run(routes(), lambda st: st.quick_score("   too short   "))


def test_concurrency_is_bounded():
    state = {"active": 0, "peak": 0}

    class SlowTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, req):
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            await asyncio.sleep(0.02)
            state["active"] -= 1
            return httpx.Response(200, json=QUICK)

    async def go():
        async with SlopTotalClient(transport=SlowTransport(), concurrency={"quick": 3}) as st:
            return await st.score_many([TEXT] * 12, "quick")

    results = asyncio.run(go())
    assert all(isinstance(r, harness.Score) for r in results)
    assert state["peak"] == 3
