"""Chat error paths that only surface when the upstream misbehaves."""

from __future__ import annotations

import asyncio
import logging
from functools import partial

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


def _not_json(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=b"<html>not json</html>")


def _empty_choices(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"id": "c1", "choices": []})


def _busy_stream(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        429,
        headers={"retry-after": "0"},
        json={"error": {"message": "c", "type": "throttling_error", "code": "429"}},
    )


def _boom(request: httpx.Request) -> httpx.Response:
    raise RuntimeError("boom")


def _upstream_500(request: httpx.Request) -> httpx.Response:
    return httpx.Response(500, json={"error": {"message": "boom", "type": "internal"}})


async def test_provider_when_returning_html_then_is_classified_as_unknown(
    asgi_client_factory,
) -> None:
    harness = build_harness(_not_json, max_retries=0)
    client = asgi_client_factory(harness.app)

    response = await client.post("/v1/chat/completions", json=_BODY)

    assert response.status_code == 502
    assert response.json()["error"]["type"] == ErrorKind.UNKNOWN.value
    assert harness.metrics.errors().by_kind[ErrorKind.UNKNOWN.value] == 1


async def test_streaming_request_when_queue_is_full_then_records_the_gate_rejection(
    asgi_client_factory,
) -> None:
    harness = build_harness(_empty_choices, provider_concurrency=1, max_queue_depth=0)
    client = asgi_client_factory(harness.app)
    slot_cm = harness.gate.slot(
        request_id="holder",
        client=ClientIdentity(address="10.0.0.8"),
        model="m",
        streaming=False,
        max_wait_s=5.0,
    )
    await slot_cm.__aenter__()

    response = await client.post("/v1/chat/completions", json={**_BODY, "stream": True})
    await slot_cm.__aexit__(None, None, None)

    assert response.status_code == 429
    assert response.json()["error"]["type"] == ErrorKind.QUEUE_FULL.value
    assert harness.metrics.errors().by_kind[ErrorKind.QUEUE_FULL.value] == 1


async def test_stream_open_when_provider_throttles_then_is_classified_as_busy(
    asgi_client_factory,
) -> None:
    harness = build_harness(_busy_stream, max_retries=0)
    client = asgi_client_factory(harness.app)

    response = await client.post("/v1/chat/completions", json={**_BODY, "stream": True})

    assert response.status_code == 429
    assert response.json()["error"]["type"] == ErrorKind.PROVIDER_BUSY.value


async def test_stream_open_when_provider_raises_then_is_classified_as_unknown(
    asgi_client_factory,
) -> None:
    harness = build_harness(_boom, max_retries=0)
    client = asgi_client_factory(harness.app)

    response = await client.post("/v1/chat/completions", json={**_BODY, "stream": True})

    assert response.status_code == 502
    assert response.json()["error"]["type"] == ErrorKind.UNKNOWN.value


class _StallingBody(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'data: {"id":"c1","choices":[{"delta":{"content":"x"}}]}\n\n'
        await asyncio.sleep(30)
        yield b"never"


class _ExplodingBody(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'data: {"id":"c1","choices":[{"delta":{"content":"x"}}]}\n\n'
        raise RuntimeError("mid-stream boom")


def _streaming_handler(body: httpx.AsyncByteStream):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body)

    return handler


async def test_midstream_timeout_when_hit_then_classifies_error_and_releases_the_slot() -> None:
    harness = build_harness(_streaming_handler(_StallingBody()), stream_idle_timeout=0.05)
    result = await harness.chat.stream(_request("r1", stream=True))

    with pytest.raises(ProxyError) as excinfo:
        async for _ in result.chunks:
            pass

    assert excinfo.value.kind is ErrorKind.UPSTREAM_UNREACHABLE
    assert harness.gate.snapshot().in_flight == 0


async def test_midstream_crash_when_hit_then_classifies_error_and_releases_the_slot() -> None:
    harness = build_harness(_streaming_handler(_ExplodingBody()))
    result = await harness.chat.stream(_request("r1", stream=True))

    with pytest.raises(ProxyError) as excinfo:
        async for _ in result.chunks:
            pass

    assert excinfo.value.kind is ErrorKind.UNKNOWN
    assert harness.gate.snapshot().in_flight == 0


async def test_retry_budget_when_exhausted_then_is_logged_as_an_error(
    asgi_client_factory, caplog
) -> None:
    harness = build_harness(_upstream_500, max_retries=2)
    client = asgi_client_factory(harness.app)

    with caplog.at_level(logging.DEBUG, logger="ipp.chat"):
        response = await client.post("/v1/chat/completions", json=_BODY)

    chat = [record for record in caplog.records if record.name == "ipp.chat"]
    retries = [record for record in chat if record.levelno == logging.WARNING]
    exhausted = [record for record in chat if record.levelno == logging.ERROR]
    message = exhausted[0].getMessage()
    assert response.status_code == 500
    assert harness.metrics.upstream().retries_total == 2
    assert [record.getMessage().split(" request_id")[0] for record in retries] == [
        "retry 1/2",
        "retry 2/2",
    ]
    assert len(exhausted) == 1
    assert "retry budget exhausted after 2 attempts" in message
    assert "kind=upstream_5xx" in message
    assert "request_id=" in message


def _retry_after_then_ok(request: httpx.Request, calls: dict) -> httpx.Response:
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


async def test_a_retry_after_hint_when_provided_then_is_honoured_by_the_retry(
    asgi_client_factory,
) -> None:
    calls = {"n": 0}
    handler = partial(_retry_after_then_ok, calls=calls)

    harness = build_harness(handler, max_retries=2)
    client = asgi_client_factory(harness.app)
    response = await client.post("/v1/chat/completions", json=_BODY)

    assert response.status_code == 200
    assert harness.metrics.upstream().retries_total == 1
