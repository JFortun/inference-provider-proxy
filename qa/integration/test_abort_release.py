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
from inference_provider_proxy.domain.errors import ErrorKind
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


def _request(harness, request_id: str, *, stream: bool) -> ChatRequest:
    return ChatRequest(
        request_id=request_id,
        client=ClientIdentity(address="10.0.0.7", user_agent="pytest"),
        model="m",
        stream=stream,
        body={"model": "m", "stream": stream, "messages": [{"role": "user", "content": "hi"}]},
    )


async def test_midstream_abort_releases_the_slot() -> None:
    harness = build_harness(_slow_stream_handler())
    result = await harness.chat.stream(_request(harness, "r1", stream=True))
    chunks = result.chunks

    first = await anext(chunks)
    assert first
    assert harness.gate.snapshot().in_flight == 1

    await chunks.aclose()

    # Released promptly, well inside the configured disconnect-to-release bound.
    assert harness.gate.snapshot().in_flight == 0
    assert harness.metrics.errors().by_kind[ErrorKind.CLIENT_ABORTED.value] == 1

    # A subsequent request is served.
    follow_up = await harness.chat.complete(_request(harness, "r2", stream=False))
    assert follow_up.status_code == 200


async def test_waiter_cancelled_before_admission_never_consumes_capacity() -> None:
    harness = build_harness(_slow_stream_handler(delay_s=0.1, count=8))

    slot_cm = harness.gate.slot(
        request_id="hold",
        client=ClientIdentity(address="10.0.0.1"),
        model="m",
        streaming=True,
        max_wait_s=5.0,
    )
    await slot_cm.__aenter__()
    assert harness.gate.snapshot().in_flight == 1

    task = asyncio.create_task(harness.chat.complete(_request(harness, "waiter", stream=False)))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    snapshot = harness.gate.snapshot()
    assert snapshot.cancelled_while_waiting_total == 1
    assert snapshot.queued == 0
    assert snapshot.in_flight == 1  # still only the holder

    await slot_cm.__aexit__(None, None, None)
    assert harness.gate.snapshot().in_flight == 0


async def test_failure_that_never_reached_the_provider_releases_capacity() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    harness = build_harness(handler, max_retries=0)

    from inference_provider_proxy.domain.errors import ProxyError

    with pytest.raises(ProxyError) as excinfo:
        await harness.chat.complete(_request(harness, "r1", stream=False))

    assert excinfo.value.kind is ErrorKind.UPSTREAM_UNREACHABLE
    assert harness.gate.snapshot().in_flight == 0
    assert harness.metrics.errors().by_kind[ErrorKind.UPSTREAM_UNREACHABLE.value] == 1


class _StalledStream:
    """A provider stream whose teardown stalls, to prove the release bound."""

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
    """Minimal ``LLMProviderPort`` double whose stream teardown stalls."""

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


async def test_stalled_upstream_teardown_still_releases_within_the_bound() -> None:
    """A stalled upstream teardown must not hold the key past the bound."""
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
    assert await anext(result.chunks)
    assert gate.snapshot().in_flight == 1

    started = time.monotonic()
    await result.chunks.aclose()
    elapsed = time.monotonic() - started

    assert gate.snapshot().in_flight == 0
    assert elapsed < 1.0
    assert metrics.errors().by_kind[ErrorKind.CLIENT_ABORTED.value] == 1


async def test_cancelled_stream_consumer_releases_the_slot() -> None:
    """Cancelling the consumer (a disconnect) frees the key promptly."""
    harness = build_harness(_slow_stream_handler(delay_s=0.05, count=8))
    result = await harness.chat.stream(_request(harness, "r1", stream=True))
    assert await anext(result.chunks)
    assert harness.gate.snapshot().in_flight == 1

    async def drain() -> None:
        async for _ in result.chunks:
            pass

    task = asyncio.create_task(drain())
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert harness.gate.snapshot().in_flight == 0
    assert harness.metrics.errors().by_kind[ErrorKind.CLIENT_ABORTED.value] == 1
