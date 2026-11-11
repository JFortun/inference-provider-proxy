"""Integration tests: overlapping chat requests serialize instead of colliding."""

from __future__ import annotations

import asyncio
import json

import httpx

from inference_provider_proxy.domain.errors import ErrorKind
from qa.integration.conftest import build_harness

_PAYLOAD = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
_MODEL_PAYLOAD = {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "hi"}]}


def _busy_429(request: httpx.Request) -> httpx.Response:
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


def _upstream_500(request: httpx.Request) -> httpx.Response:
    return httpx.Response(500, json={"error": {"message": "boom", "type": "internal"}})


def _chat_handler(delay_s: float, *, fail_once: bool = False):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat/completions"):
            calls["n"] += 1
            body = json.loads(request.content)
            if fail_once and calls["n"] == 1:
                return _busy_429(request)
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


async def _stream_post(client, payload) -> tuple[httpx.Response, str]:
    async with client.stream("POST", "/v1/chat/completions", json=payload) as r:
        body = "".join([chunk async for chunk in r.aiter_text()])
        return r, body


async def _gather_posts(client, n: int, payload) -> list[httpx.Response]:
    async def one(i: int) -> httpx.Response:
        return await client.post("/v1/chat/completions", json=payload)

    return await asyncio.gather(*(one(i) for i in range(n)))


async def test_five_concurrent_requests_when_serialized_then_all_succeed_without_rejection(
    asgi_client_factory,
) -> None:
    harness = build_harness(_chat_handler(0.05), provider_concurrency=1, max_queue_depth=10)
    client = asgi_client_factory(harness.app)

    responses = await _gather_posts(client, 5, _MODEL_PAYLOAD)

    assert [r.status_code for r in responses] == [200] * 5
    for response in responses:
        assert response.json()["choices"][0]["message"]["content"] == "Hello"
        assert response.headers["X-Proxy-Request-Id"]
        assert "X-Queue-Wait-Ms" in response.headers
    assert harness.metrics.errors().by_kind[ErrorKind.PROVIDER_BUSY.value] == 0


async def test_streamed_response_when_usage_included_then_preserves_the_usage_chunk(
    asgi_client_factory,
) -> None:
    harness = build_harness(_chat_handler(0.01))
    client = asgi_client_factory(harness.app)
    payload = {**_MODEL_PAYLOAD, "stream": True, "stream_options": {"include_usage": True}}

    async with client.stream("POST", "/v1/chat/completions", json=payload) as response:
        body = "".join([chunk async for chunk in response.aiter_text()])

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"usage"' in body
    assert "data: [DONE]" in body
    assert body.index('"usage"') < body.index("data: [DONE]")


async def test_queue_wait_header_when_requests_contend_then_grows_beyond_zero(
    asgi_client_factory,
) -> None:
    harness = build_harness(_chat_handler(0.05), provider_concurrency=1, max_queue_depth=10)
    client = asgi_client_factory(harness.app)

    responses = await _gather_posts(client, 3, _PAYLOAD)
    waits = [int(r.headers["X-Queue-Wait-Ms"]) for r in responses]

    assert waits[0] == 0
    assert max(waits) > 0


async def test_wait_bound_when_exceeded_then_surfaces_queue_timeout_in_the_taxonomy(
    asgi_client_factory,
) -> None:
    harness = build_harness(
        _chat_handler(0.3), provider_concurrency=1, max_queue_depth=10, queue_timeout=0.05
    )
    client = asgi_client_factory(harness.app)

    responses = await _gather_posts(client, 3, _PAYLOAD)
    rejected = [r for r in responses if r.status_code == 429]
    document = (await client.get("/dashboard")).json()

    assert rejected
    assert rejected[0].json()["error"]["type"] == "queue_timeout"
    assert rejected[0].headers["retry-after"]
    assert document["errors"]["by_kind"]["queue_timeout"] >= 1
    assert document["errors"]["total"] >= 1
    assert document["throughput"]["requests_failed"] >= 1
    assert any(e["kind"] == "queue_timeout" for e in document["errors"]["recent"])


async def test_queue_when_full_then_rejects_without_consuming_capacity(
    asgi_client_factory,
) -> None:
    harness = build_harness(_chat_handler(0.2), provider_concurrency=1, max_queue_depth=0)
    client = asgi_client_factory(harness.app)

    responses = await _gather_posts(client, 2, _PAYLOAD)
    rejected = [r for r in responses if r.status_code == 429]
    document = (await client.get("/dashboard")).json()

    assert rejected
    assert rejected[0].json()["error"]["type"] == "queue_full"
    assert document["errors"]["by_kind"]["queue_full"] >= 1
    assert document["gate"]["in_flight"] == 0
    assert document["gate"]["queued"] == 0


async def test_transient_provider_failure_when_retryable_then_is_retried_and_succeeds(
    asgi_client_factory,
) -> None:
    harness = build_harness(_chat_handler(0.0, fail_once=True), max_retries=2)
    client = asgi_client_factory(harness.app)

    response = await client.post("/v1/chat/completions", json=_PAYLOAD)

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "Hello"
    assert harness.metrics.upstream().retries_total == 1
    assert harness.metrics.errors().total == 0


async def test_transient_provider_failure_when_streaming_then_is_also_retried(
    asgi_client_factory,
) -> None:
    harness = build_harness(_chat_handler(0.0, fail_once=True), max_retries=2)
    client = asgi_client_factory(harness.app)
    payload = {**_PAYLOAD, "stream": True}

    _, body = await _stream_post(client, payload)

    assert "data: [DONE]" in body
    assert harness.metrics.upstream().retries_total == 1


async def test_retry_budget_when_exhausted_then_fails_fast_with_the_classified_error(
    asgi_client_factory,
) -> None:
    harness = build_harness(_busy_429, max_retries=2)
    client = asgi_client_factory(harness.app)

    response = await client.post("/v1/chat/completions", json=_PAYLOAD)

    assert response.status_code == 429
    assert response.json()["error"]["type"] == ErrorKind.PROVIDER_BUSY.value
    assert harness.metrics.upstream().retries_total == 2
    assert harness.metrics.errors().by_kind[ErrorKind.PROVIDER_BUSY.value] == 1
    episode = harness.metrics.errors().recent[0]
    assert episode.kind is ErrorKind.PROVIDER_BUSY
    assert (episode.gate_in_flight, episode.gate_capacity) == (1, 1)


async def test_error_responses_when_rejected_then_carry_transparency_headers(
    asgi_client_factory,
) -> None:
    harness = build_harness(
        _chat_handler(0.2), provider_concurrency=1, max_queue_depth=0, max_retries=0
    )
    client = asgi_client_factory(harness.app)

    responses = await _gather_posts(client, 2, _PAYLOAD)
    rejected = [r for r in responses if r.status_code == 429]

    assert rejected
    assert rejected[0].json()["error"]["type"] == "queue_full"
    assert rejected[0].headers["X-Queue-Wait-Ms"] == "0"
    assert rejected[0].headers["X-Proxy-Request-Id"]
    assert rejected[0].headers["retry-after"]


async def test_upstream_failure_when_returned_then_carries_queue_transparency(
    asgi_client_factory,
) -> None:
    harness = build_harness(_upstream_500, max_retries=0)
    client = asgi_client_factory(harness.app)

    response = await client.post("/v1/chat/completions", json=_PAYLOAD)

    assert response.status_code == 500
    assert response.json()["error"]["type"] == ErrorKind.UPSTREAM_5XX.value
    assert response.headers["X-Queue-Wait-Ms"] == "0"
    assert response.headers["retry-after"]


async def test_queue_position_when_served_then_is_recoverable_from_headers_and_dashboard(
    asgi_client_factory,
) -> None:
    harness = build_harness(_chat_handler(0.05), provider_concurrency=1, max_queue_depth=10)
    client = asgi_client_factory(harness.app)

    responses = await _gather_posts(client, 3, _PAYLOAD)
    header_positions = sorted(
        int(p) for p in (r.headers.get("X-Queue-Position") for r in responses) if p is not None
    )
    served = (await client.get("/dashboard")).json()["recent_requests"]
    dashboard_positions = [
        item["queue_position"] for item in served if item["queue_position"] is not None
    ]

    assert [r.status_code for r in responses] == [200] * 3
    assert header_positions == [1, 2]
    assert len(served) == 3
    assert sorted(dashboard_positions) == [1, 2]
    assert sum(1 for item in served if item["queue_position"] is None) == 1
