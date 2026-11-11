"""Mid-stream abort and waiter cancellation free the key."""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest

from inference_provider_proxy.adapters.outbound.gate.asyncio_gate import AsyncioGate
from inference_provider_proxy.adapters.outbound.metrics.memory import MemoryMetrics
from inference_provider_proxy.application.ports.clock import SystemClock
from inference_provider_proxy.application.services.chat import ChatService
from inference_provider_proxy.config import Settings
from inference_provider_proxy.domain.errors import ErrorKind, ProxyError
from inference_provider_proxy.domain.models import ChatRequest, ClientIdentity, SubscriptionInfo
from qa.integration.conftest import build_harness


class _SlowStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes], delay_s: float) -> None:
        self._chunks = chunks
        self._delay_s = delay_s

    async def __aiter__(self):
        for chunk in self._chunks:
            await asyncio.sleep(self._delay_s)
            yield chunk


def _slow_stream_handler(delay_s: float = 0.05, count: int = 6):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/chat/completions"):
            body = json.loads(request.content)
            if not body.get("stream"):
                return httpx.Response(
                    200,
                    json={
                        "id": "c1",
                        "object": "chat.completion",
                        "model": "m",
                        "choices": [
                            {"index": 0, "message": {"role": "assistant", "content": "ok"}}
                        ],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    },
                )
            chunks = [
                f'data: {{"id":"c1","choices":[{{"delta":{{"content":"{i}"}}}}]}}\n\n'.encode()
                for i in range(count)
            ]
            chunks.append(b"data: [DONE]\n\n")
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=_SlowStream(chunks, delay_s),
            )
        return httpx.Response(404, json={"error": "not found"})

    return handler


def _refuse(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("refused")


def _request(harness, request_id: str, *, stream: bool) -> ChatRequest:
    return ChatRequest(
        request_id=request_id,
        client=ClientIdentity(address="10.0.0.7", user_agent="pytest"),
        model="m",
        stream=stream,
        body={"model": "m", "stream": stream, "messages": [{"role": "user", "content": "hi"}]},
    )


async def _drain(chunks) -> None:
    async for _ in chunks:
        pass


async def test_midstream_abort_when_client_disconnects_then_releases_the_slot() -> None:
    harness = build_harness(_slow_stream_handler())
    result = await harness.chat.stream(_request(harness, "r1", stream=True))
    chunks = result.chunks
    first = await anext(chunks)
    held = harness.gate.snapshot().in_flight

    await chunks.aclose()
    released = harness.gate.snapshot()
    follow_up = await harness.chat.complete(_request(harness, "r2", stream=False))

    assert first
    assert held == 1
    assert released.in_flight == 0
    assert harness.metrics.errors().by_kind[ErrorKind.CLIENT_ABORTED.value] == 1
    assert follow_up.status_code == 200


async def test_waiter_when_cancelled_before_admission_then_never_consumes_capacity() -> None:
    harness = build_harness(_slow_stream_handler(delay_s=0.1, count=8))
    slot_cm = harness.gate.slot(
        request_id="hold",
        client=ClientIdentity(address="10.0.0.1"),
        model="m",
        streaming=True,
        max_wait_s=5.0,
    )
    await slot_cm.__aenter__()
    held = harness.gate.snapshot().in_flight

    task = asyncio.create_task(harness.chat.complete(_request(harness, "waiter", stream=False)))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    snapshot = harness.gate.snapshot()
    await slot_cm.__aexit__(None, None, None)
    released = harness.gate.snapshot()

    assert held == 1
    assert snapshot.cancelled_while_waiting_total == 1
    assert snapshot.queued == 0
    assert snapshot.in_flight == 1
    assert released.in_flight == 0


async def test_failure_before_the_provider_when_raised_then_releases_capacity() -> None:
    harness = build_harness(_refuse, max_retries=0)

    with pytest.raises(ProxyError) as excinfo:
        await harness.chat.complete(_request(harness, "r1", stream=False))

    assert excinfo.value.kind is ErrorKind.UPSTREAM_UNREACHABLE
    assert harness.gate.snapshot().in_flight == 0
    assert harness.metrics.errors().by_kind[ErrorKind.UPSTREAM_UNREACHABLE.value] == 1


class _StalledStream:
    status_code = 200
    headers: dict[str, str] = {}
    provider_id: str | None = None
    provider_correlation_id: str | None = None
    usage = None

    def __init__(self, stall_s: float) -> None:
        self._stall_s = stall_s

    async def aiter_bytes(self):
        yield b'data: {"id":"c1","choices":[{"delta":{"content":"x"}}]}\n\n'

    async def __aenter__(self) -> _StalledStream:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        await asyncio.sleep(self._stall_s)
        return False


class _StallingProvider:
    def __init__(self, stall_s: float) -> None:
        self._stall_s = stall_s

    async def complete(self, body):  # pragma: no cover - not exercised here
        raise AssertionError("complete() is not used by the streaming test")

    def open_stream(self, body):
        return _StalledStream(self._stall_s)

    async def list_models(self):
        return []

    async def fetch_usage(self):
        return SubscriptionInfo()

    async def aclose(self) -> None:
        return None


def _stream_request(clock, request_id: str) -> ChatRequest:
    return ChatRequest(
        request_id=request_id,
        client=ClientIdentity(address="10.0.0.9", user_agent="pytest"),
        model="m",
        stream=True,
        body={"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
    )


async def test_stalled_upstream_teardown_when_hit_then_releases_within_the_bound() -> None:
    settings = Settings(_env_file=None, upstream_api_key="k", disconnect_release=0.05)
    clock = SystemClock()
    gate = AsyncioGate(capacity=1, max_depth=8, clock=clock)
    metrics = MemoryMetrics(base_url="https://provider.test", retention=100, clock=clock)
    chat = ChatService(
        gate=gate,
        provider=_StallingProvider(stall_s=30.0),
        metrics=metrics,
        clock=clock,
        settings=settings,
    )
    result = await chat.stream(_stream_request(clock, "r1"))
    first = await anext(result.chunks)
    held = gate.snapshot().in_flight

    started = time.monotonic()
    await result.chunks.aclose()
    elapsed = time.monotonic() - started
    released = gate.snapshot()

    assert first
    assert held == 1
    assert released.in_flight == 0
    assert elapsed < 1.0
    assert metrics.errors().by_kind[ErrorKind.CLIENT_ABORTED.value] == 1


async def test_cancelled_stream_consumer_when_disconnects_then_releases_the_slot() -> None:
    harness = build_harness(_slow_stream_handler(delay_s=0.05, count=8))
    result = await harness.chat.stream(_request(harness, "r1", stream=True))
    first = await anext(result.chunks)
    held = harness.gate.snapshot().in_flight

    task = asyncio.create_task(_drain(result.chunks))
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    released = harness.gate.snapshot()

    assert first
    assert held == 1
    assert released.in_flight == 0
    assert harness.metrics.errors().by_kind[ErrorKind.CLIENT_ABORTED.value] == 1
