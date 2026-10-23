"""Async client for the SlopTotal detector server.

Both endpoints come back in one shape, ``Score``: the overall score (0-100) and
a ``{engine_key: score}`` dict with every engine score on 0-1.

The server reports them differently, which this module hides:

- ``POST /api/quick-score`` -> ``{"score": 0-100, "engines": [{"engine": key, "score": 0-100}]}``
- ``POST /api/analyze``     -> ``{"overall_score": 0-100, "engine_results": [{"engine_name": display name, "score": 0-1}]}``

Display names are mapped to keys via ``GET /api/engines``, which also gives each
engine's family (``engine_type``: neural, statistical, linguistic, ...).

When every slot is busy the server answers 202 with a ticket, polled at
``GET /api/queue/ticket/{id}``; when its queue is full it answers 429 with
``retry_after``. Both are handled here.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Literal

import httpx

Endpoint = Literal["quick", "analyze"]

_PATHS: dict[str, str] = {"quick": "/api/quick-score", "analyze": "/api/analyze"}

# Matches the server's slots per lane (GET /api/queue/status): 4 quick, 2 full.
# Going higher only moves the waiting into the server's queue.
DEFAULT_CONCURRENCY: dict[str, int] = {"quick": 4, "analyze": 2}

MIN_TEXT_LENGTH = 50  # server-side MIN_TEXT_LENGTH, after strip()


class HarnessError(RuntimeError):
    """The server could not score a text after all retries."""


class TextRejected(ValueError):
    """The server refused the text itself (HTTP 400, e.g. too short). Not retried."""


@dataclass
class Score:
    overall: float  # 0-100, higher = more likely AI
    engines: dict[str, float]  # engine key -> 0-1
    verdict: str
    endpoint: Endpoint
    latency_s: float  # wall time including queueing, polling and retries
    report_id: str | None = None  # analyze only
    created_at: str | None = None  # analyze only; older than the request = cache hit
    attempts: int = 1
    raw: dict = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class EngineInfo:
    key: str
    name: str
    engine_type: str


class SlopTotalClient:
    """Use as ``async with SlopTotalClient() as st: await st.quick_score(text)``."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        concurrency: dict[str, int] | None = None,
        max_retries: int = 6,
        request_timeout: float = 300.0,
        ticket_timeout: float = 900.0,
        transport: httpx.AsyncBaseTransport | None = None,  # for tests
    ):
        self.base_url = base_url.rstrip("/")
        self._transport = transport
        self.concurrency = {**DEFAULT_CONCURRENCY, **(concurrency or {})}
        self._sems = {ep: asyncio.Semaphore(n) for ep, n in self.concurrency.items()}
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.ticket_timeout = ticket_timeout
        self._http: httpx.AsyncClient | None = None
        self._engines: dict[str, EngineInfo] | None = None
        self._name_to_key: dict[str, str] = {}

    async def __aenter__(self) -> SlopTotalClient:
        limits = httpx.Limits(max_connections=sum(self.concurrency.values()) + 4)
        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.request_timeout,
            limits=limits,
            transport=self._transport,
        )
        return self

    async def __aexit__(self, *exc) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def health(self) -> bool:
        try:
            r = await self._client.get("/health", timeout=5)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def engines(self) -> dict[str, EngineInfo]:
        """Engine key -> info, fetched once."""
        if self._engines is None:
            r = await self._client.get("/api/engines")
            r.raise_for_status()
            self._engines = {
                e["key"]: EngineInfo(e["key"], e["name"], e.get("engine_type", ""))
                for e in r.json()
            }
            self._name_to_key = {e.name: e.key for e in self._engines.values()}
        return self._engines

    async def quick_score(self, text: str) -> Score:
        return await self.score(text, "quick")

    async def analyze(self, text: str) -> Score:
        return await self.score(text, "analyze")

    async def score(self, text: str, endpoint: Endpoint) -> Score:
        text = text.strip()
        if len(text) < MIN_TEXT_LENGTH:
            raise TextRejected(f"text is {len(text)} chars; the server needs {MIN_TEXT_LENGTH}")
        if endpoint == "analyze":
            await self.engines()  # for name -> key
        async with self._sems[endpoint]:
            t0 = time.perf_counter()
            body, attempts = await self._post_with_retries(_PATHS[endpoint], {"text": text})
            latency = time.perf_counter() - t0
        return self._parse(body, endpoint, latency, attempts)

    async def score_many(
        self,
        texts: list[str],
        endpoint: Endpoint,
        return_exceptions: bool = True,
    ) -> list[Score | BaseException]:
        """Score all texts concurrently (bounded by the endpoint's concurrency).

        Results are in input order. With ``return_exceptions`` a failed text
        yields its exception instead of cancelling the batch.
        """
        return await asyncio.gather(
            *(self.score(t, endpoint) for t in texts), return_exceptions=return_exceptions
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @property
    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            raise RuntimeError("use SlopTotalClient as an async context manager")
        return self._http

    async def _post_with_retries(self, path: str, payload: dict) -> tuple[dict, int]:
        last_err: str = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                r = await self._client.post(path, json=payload)
            except httpx.TransportError as e:  # connection refused/reset, timeouts
                last_err = f"{type(e).__name__}: {e}"
                await self._backoff(attempt)
                continue

            if r.status_code == 200:
                body = r.json()
                if "error" in body:
                    last_err = f"200 with error: {body['error']}"
                    await self._backoff(attempt)
                    continue
                return body, attempt

            if r.status_code == 202:
                ticket = r.json().get("ticket_id")
                if not ticket:
                    last_err = f"202 without ticket_id: {r.text[:200]}"
                    await self._backoff(attempt)
                    continue
                body = await self._poll_ticket(ticket)
                if body is not None and "error" not in body:
                    return body, attempt
                last_err = "ticket expired" if body is None else f"ticket error: {body['error']}"
                continue  # resubmit straight away

            if r.status_code == 400:
                raise TextRejected(_error_text(r))

            if r.status_code == 429:
                last_err = f"429: {_error_text(r)}"
                retry_after = _json_or_empty(r).get("retry_after", 2)
                await asyncio.sleep(float(retry_after) + random.uniform(0, 1))
                continue

            # 5xx and anything unexpected
            last_err = f"{r.status_code}: {_error_text(r)}"
            await self._backoff(attempt)

        raise HarnessError(f"{path} failed after {self.max_retries} attempts; last: {last_err}")

    async def _poll_ticket(self, ticket_id: str) -> dict | None:
        """Wait for a queued request. Returns the result, or None if the ticket expired."""
        deadline = time.monotonic() + self.ticket_timeout
        delay = 0.25
        while time.monotonic() < deadline:
            try:
                r = await self._client.get(f"/api/queue/ticket/{ticket_id}")
            except httpx.TransportError:
                await asyncio.sleep(1.0)
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            if r.status_code == 202:
                hint = _json_or_empty(r).get("estimated_wait_ms")
                wait = delay if not hint else min(max(hint / 1000 / 2, 0.25), 5.0)
                await asyncio.sleep(wait)
                delay = min(delay * 1.5, 5.0)
                continue
            await asyncio.sleep(1.0)
        raise HarnessError(f"ticket {ticket_id} still pending after {self.ticket_timeout:.0f}s")

    @staticmethod
    async def _backoff(attempt: int) -> None:
        await asyncio.sleep(min(2 ** (attempt - 1), 30) * random.uniform(0.5, 1.0))

    def _parse(self, body: dict, endpoint: Endpoint, latency: float, attempts: int) -> Score:
        if endpoint == "quick":
            return Score(
                overall=float(body["score"]),
                engines={e["engine"]: float(e["score"]) / 100 for e in body["engines"]},
                verdict=body.get("verdict", ""),
                endpoint=endpoint,
                latency_s=latency,
                attempts=attempts,
                raw=body,
            )
        return Score(
            overall=float(body["overall_score"]),
            engines={
                self._name_to_key.get(e["engine_name"], e["engine_name"]): float(e["score"])
                for e in body["engine_results"]
            },
            verdict=body.get("overall_verdict", ""),
            endpoint=endpoint,
            latency_s=latency,
            report_id=body.get("id"),
            created_at=body.get("created_at"),
            attempts=attempts,
            raw=body,
        )


def _json_or_empty(r: httpx.Response) -> dict:
    try:
        body = r.json()
        return body if isinstance(body, dict) else {}
    except ValueError:
        return {}


def _error_text(r: httpx.Response) -> str:
    body = _json_or_empty(r)
    return str(body.get("error") or body.get("detail") or r.text[:200])
