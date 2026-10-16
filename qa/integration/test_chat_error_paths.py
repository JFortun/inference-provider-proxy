"""Chat error paths that only surface when the upstream misbehaves (FR-012, FR-024)."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from inference_provider_proxy.domain.errors import ErrorKind, ProxyError
from inference_provider_proxy.domain.models import ChatRequest, ClientIdentity
from qa.integration.conftest import build_harness

_BODY = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


def _request(request_id: str, *, stream: bool) -> ChatRequest:
    return ChatRequest(
        request_id=request_id,
        client=ClientIdentity(address="10.0.0.7", user_agent="pytest"),
        model="m",
        stream=stream,
        body={**_BODY, "stream": stream},
    )


async def test_unexpected_provider_failure_is_classified(asgi_client_factory) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>not json</html>")

    harness = build_harness(handler, max_retries=0)
    client = asgi_client_factory(harness.app)

    response = await client.post("/v1/chat/completions", json=_BODY)

    assert response.status_code == 502
    assert response.json()["error"]["type"] == ErrorKind.UNKNOWN.value
    assert harness.metrics.errors().by_kind[ErrorKind.UNKNOWN.value] == 1


async def test_streaming_gate_rejection_is_recorded(asgi_client_factory) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "c1", "choices": []})

    harness = build_harness(handler, provider_concurrency=1, max_queue_depth=0)
    client = asgi_client_factory(harness.app)

    slot_cm = harness.gate.slot(
        request_id="holder",
        client=ClientIdentity(address="10.0.0.8"),
        model="m",
        streaming=False,
        max_wait_s=5.0,
    )
    await slot_cm.__aenter__()
    try:
        response = await client.post("/v1/chat/completions", json={**_BODY, "stream": True})
    finally:
        await slot_cm.__aexit__(None, None, None)

    assert response.status_code == 429
    assert response.json()["error"]["type"] == ErrorKind.QUEUE_FULL.value
    assert harness.metrics.errors().by_kind[ErrorKind.QUEUE_FULL.value] == 1


async def test_stream_open_classified_failure_is_recorded(asgi_client_factory) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"retry-after": "0"},
            json={"error": {"message": "c", "type": "throttling_error", "code": "429"}},
        )

    harness = build_harness(handler, max_retries=0)
    client = asgi_client_factory(harness.app)

    response = await client.post("/v1/chat/completions", json={**_BODY, "stream": True})

    assert response.status_code == 429
    assert response.json()["error"]["type"] == ErrorKind.PROVIDER_BUSY.value


async def test_stream_open_unexpected_failure_is_classified(asgi_client_factory) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("boom")

    harness = build_harness(handler, max_retries=0)
    client = asgi_client_factory(harness.app)

    response = await client.post("/v1/chat/completions", json={**_BODY, "stream": True})

    assert response.status_code == 502
    assert response.json()["error"]["type"] == ErrorKind.UNKNOWN.value


class _StallingBody(httpx.AsyncByteStream):
    """Yields one chunk then stalls past the idle timeout."""

    async def __aiter__(self):
        yield b'data: {"id":"c1","choices":[{"delta":{"content":"x"}}]}\n\n'
        await asyncio.sleep(30)
        yield b"never"


class _ExplodingBody(httpx.AsyncByteStream):
    """Yields one chunk then fails with an unclassified error."""

    async def __aiter__(self):
        yield b'data: {"id":"c1","choices":[{"delta":{"content":"x"}}]}\n\n'
        raise RuntimeError("mid-stream boom")


def _streaming_handler(body: httpx.AsyncByteStream):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body)

    return handler


async def test_midstream_proxy_error_is_classified_and_releases_the_slot() -> None:
    harness = build_harness(_streaming_handler(_StallingBody()), stream_idle_timeout=0.05)
    result = await harness.chat.stream(_request("r1", stream=True))

    with pytest.raises(ProxyError) as excinfo:
        async for _ in result.chunks:
            pass

    assert excinfo.value.kind is ErrorKind.UPSTREAM_UNREACHABLE
    assert harness.gate.snapshot().in_flight == 0


async def test_midstream_unexpected_failure_is_classified_and_releases_the_slot() -> None:
    harness = build_harness(_streaming_handler(_ExplodingBody()))
    result = await harness.chat.stream(_request("r1", stream=True))

    with pytest.raises(ProxyError) as excinfo:
        async for _ in result.chunks:
            pass

    assert excinfo.value.kind is ErrorKind.UNKNOWN
    assert harness.gate.snapshot().in_flight == 0


async def test_retry_honours_a_retry_after_hint(asgi_client_factory) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                429,
                headers={"retry-after": "0.01"},
                json={"error": {"message": "c", "type": "throttling_error", "code": "429"}},
            )
        return httpx.Response(
            200,
            json={
                "id": "c1",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
            },
        )

    harness = build_harness(handler, max_retries=2)
    client = asgi_client_factory(harness.app)

    response = await client.post("/v1/chat/completions", json=_BODY)

    assert response.status_code == 200
    assert harness.metrics.upstream().retries_total == 1
