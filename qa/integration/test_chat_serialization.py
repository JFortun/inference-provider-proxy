"""Integration tests: overlapping chat requests serialize instead of colliding (US1 / SC-001)."""

from __future__ import annotations

import asyncio
import json

import httpx

from inference_provider_proxy.domain.errors import ErrorKind
from qa.integration.conftest import build_harness


def _chat_handler(delay_s: float, *, fail_once: bool = False):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat/completions"):
            calls["n"] += 1
            body = json.loads(request.content)
            if fail_once and calls["n"] == 1:
                return httpx.Response(
                    429,
                    headers={"retry-after": "0"},
                    json={
                        "error": {
                            "message": "concurrency",
                            "type": "throttling_error",
                            "code": "429",
                            "param": None,
                        }
                    },
                )
            if body.get("stream"):
                chunks = [
                    'data: {"id":"c1","choices":[{"delta":{"content":"He"}}]}\n\n',
                    'data: {"id":"c1","choices":[{"delta":{"content":"llo"}}]}\n\n',
                ]
                if body.get("stream_options", {}).get("include_usage"):
                    chunks.append(
                        'data: {"id":"c1","choices":[],"usage":'
                        '{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n\n'
                    )
                chunks.append("data: [DONE]\n\n")
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content="".join(chunks).encode(),
                )
            return httpx.Response(
                200,
                json={
                    "id": "c1",
                    "object": "chat.completion",
                    "model": body.get("model"),
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"}}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                },
            )
        return httpx.Response(404, json={"error": "not found"})

    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(delay_s)
        return handler(request)

    return slow


async def test_five_concurrent_requests_all_succeed(asgi_client_factory) -> None:
    harness = build_harness(_chat_handler(0.05), provider_concurrency=1, max_queue_depth=10)
    client = asgi_client_factory(harness.app)

    async def one(i: int) -> httpx.Response:
        return await client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "hi"}]},
            headers={"user-agent": f"client-{i}"},
        )

    responses = await asyncio.gather(*(one(i) for i in range(5)))

    assert [r.status_code for r in responses] == [200] * 5
    for response in responses:
        assert response.json()["choices"][0]["message"]["content"] == "Hello"
        assert response.headers["X-Proxy-Request-Id"]
        assert "X-Queue-Wait-Ms" in response.headers

    # The gate never exceeded capacity and nothing was rejected.
    snapshot = harness.metrics.errors()
    assert snapshot.by_kind[ErrorKind.PROVIDER_BUSY.value] == 0


async def test_streaming_preserves_usage_chunk(asgi_client_factory) -> None:
    harness = build_harness(_chat_handler(0.01))
    client = asgi_client_factory(harness.app)

    async with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "deepseek-v4.1-flash",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ) as response:
        body = "".join([chunk async for chunk in response.aiter_text()])

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"usage"' in body
    assert "data: [DONE]" in body
    assert body.index('"usage"') < body.index("data: [DONE]")


async def test_queue_wait_header_grows_under_contention(asgi_client_factory) -> None:
    harness = build_harness(_chat_handler(0.05), provider_concurrency=1, max_queue_depth=10)
    client = asgi_client_factory(harness.app)

    async def one() -> httpx.Response:
        return await client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )

    responses = await asyncio.gather(*(one() for _ in range(3)))
    waits = [int(r.headers["X-Queue-Wait-Ms"]) for r in responses]
    assert waits[0] == 0
    assert max(waits) > 0


async def test_wait_bound_rejection_is_surfaced_in_the_taxonomy(asgi_client_factory) -> None:
    # A 50 ms wait bound against a 300 ms handler guarantees a timeout rejection.
    harness = build_harness(
        _chat_handler(0.3),
        provider_concurrency=1,
        max_queue_depth=10,
        queue_timeout=0.05,
    )
    client = asgi_client_factory(harness.app)

    async def one() -> httpx.Response:
        return await client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )

    responses = await asyncio.gather(*(one() for _ in range(3)))
    rejected = [r for r in responses if r.status_code == 429]
    assert rejected, "expected at least one bounded-wait rejection"
    assert rejected[0].json()["error"]["type"] == "queue_timeout"
    assert rejected[0].headers["retry-after"]

    document = (await client.get("/dashboard")).json()
    assert document["errors"]["by_kind"]["queue_timeout"] >= 1
    assert document["errors"]["total"] >= 1
    assert document["throughput"]["requests_failed"] >= 1
    assert any(e["kind"] == "queue_timeout" for e in document["errors"]["recent"])


async def test_queue_full_rejection_is_surfaced_in_the_taxonomy(asgi_client_factory) -> None:
    harness = build_harness(
        _chat_handler(0.2),
        provider_concurrency=1,
        max_queue_depth=0,
    )
    client = asgi_client_factory(harness.app)

    async def one() -> httpx.Response:
        return await client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )

    responses = await asyncio.gather(*(one() for _ in range(2)))
    rejected = [r for r in responses if r.status_code == 429]
    assert rejected, "expected a queue-full rejection"
    assert rejected[0].json()["error"]["type"] == "queue_full"

    document = (await client.get("/dashboard")).json()
    assert document["errors"]["by_kind"]["queue_full"] >= 1
    # The rejection consumed no capacity, so the gate must still be clean.
    assert document["gate"]["in_flight"] == 0
    assert document["gate"]["queued"] == 0


async def test_transient_provider_failure_is_retried_and_succeeds(asgi_client_factory) -> None:
    """A retryable failure is retried within the budget rather than failing (FR-012)."""
    harness = build_harness(_chat_handler(0.0, fail_once=True), max_retries=2)
    client = asgi_client_factory(harness.app)

    response = await client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Hello"
    assert harness.metrics.upstream().retries_total == 1
    # The retried request succeeded, so no failure is surfaced anywhere.
    assert harness.metrics.errors().total == 0


async def test_transient_provider_failure_is_retried_on_the_streaming_path(
    asgi_client_factory,
) -> None:
    """The retry budget applies to streams too, not only to buffered calls (FR-012)."""
    harness = build_harness(_chat_handler(0.0, fail_once=True), max_retries=2)
    client = asgi_client_factory(harness.app)

    async with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as response:
        body = "".join([chunk async for chunk in response.aiter_text()])

    assert response.status_code == 200
    assert "data: [DONE]" in body
    assert harness.metrics.upstream().retries_total == 1


async def test_retry_budget_exhaustion_fails_fast(asgi_client_factory) -> None:
    """Once the budget is spent the classified error is surfaced (FR-012)."""

    def always_429(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"retry-after": "0"},
            json={
                "error": {
                    "message": "concurrency",
                    "type": "throttling_error",
                    "code": "429",
                    "param": None,
                }
            },
        )

    harness = build_harness(always_429, max_retries=2)
    client = asgi_client_factory(harness.app)

    response = await client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 429
    assert response.json()["error"]["type"] == ErrorKind.PROVIDER_BUSY.value
    assert harness.metrics.upstream().retries_total == 2  # exactly the budget
    assert harness.metrics.errors().by_kind[ErrorKind.PROVIDER_BUSY.value] == 1
    # Recorded while this request itself held the only slot: within capacity, so
    # the episode documents that the gate never over-admitted (F1 / SC-002).
    episode = harness.metrics.errors().recent[0]
    assert episode.kind is ErrorKind.PROVIDER_BUSY
    assert (episode.gate_in_flight, episode.gate_capacity) == (1, 1)


async def test_error_responses_carry_transparency_headers(asgi_client_factory) -> None:
    """Transparency headers are documented for all responses, errors included (FR-013)."""
    harness = build_harness(
        _chat_handler(0.2), provider_concurrency=1, max_queue_depth=0, max_retries=0
    )
    client = asgi_client_factory(harness.app)

    async def one() -> httpx.Response:
        return await client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )

    responses = await asyncio.gather(*(one() for _ in range(2)))
    rejected = [r for r in responses if r.status_code == 429]
    assert rejected, "expected a queue-full rejection"
    assert rejected[0].json()["error"]["type"] == "queue_full"
    # Rejected before it could wait, so the wait is a real zero, not missing.
    assert rejected[0].headers["X-Queue-Wait-Ms"] == "0"
    assert rejected[0].headers["X-Proxy-Request-Id"]
    assert rejected[0].headers["retry-after"]


async def test_upstream_failure_error_carries_queue_transparency(asgi_client_factory) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "boom", "type": "internal"}})

    harness = build_harness(handler, max_retries=0)
    client = asgi_client_factory(harness.app)

    response = await client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 500
    assert response.json()["error"]["type"] == ErrorKind.UPSTREAM_5XX.value
    assert response.headers["X-Queue-Wait-Ms"] == "0"
    assert response.headers["retry-after"]


async def test_queue_position_is_recoverable_from_the_dashboard(asgi_client_factory) -> None:
    """SC-008: a served request's wait position is recoverable from the dashboard."""
    harness = build_harness(_chat_handler(0.05), provider_concurrency=1, max_queue_depth=10)
    client = asgi_client_factory(harness.app)

    async def one() -> httpx.Response:
        return await client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )

    responses = await asyncio.gather(*(one() for _ in range(3)))
    assert [r.status_code for r in responses] == [200] * 3

    # One is admitted immediately (no position); the other two queued at 1 and 2.
    header_positions = sorted(
        int(p) for p in (r.headers.get("X-Queue-Position") for r in responses) if p is not None
    )
    assert header_positions == [1, 2]

    # The same positions are recoverable from the dashboard, not only the headers.
    served = (await client.get("/dashboard")).json()["recent_requests"]
    assert len(served) == 3
    dashboard_positions = [
        item["queue_position"] for item in served if item["queue_position"] is not None
    ]
    assert sorted(dashboard_positions) == [1, 2]
    assert sum(1 for item in served if item["queue_position"] is None) == 1
