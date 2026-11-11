"""A client that leaves before the response starts is observed, not served.

These drive the ASGI app by hand so ``receive()`` can deliver a real
``http.disconnect``, instead of simulating it with ``task.cancel()``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import httpx
import pytest
from fastapi import Request

from inference_provider_proxy.adapters.inbound.http.openai_routes import _await_pre_response
from inference_provider_proxy.application.ports.clock import SystemClock
from inference_provider_proxy.application.services.chat import ChatService
from inference_provider_proxy.domain.errors import ErrorKind
from inference_provider_proxy.domain.models import ChatRequest, ClientIdentity, SubscriptionInfo
from qa.integration.conftest import build_harness

_BODY = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


async def _wait_until(predicate: Any, budget: float = 5.0) -> None:
    deadline = time.monotonic() + budget
    while not predicate():
        assert time.monotonic() < deadline, "condition never became true"
        await asyncio.sleep(0.02)


def _payload(stream: bool) -> bytes:
    return json.dumps({**_BODY, "stream": stream}).encode()


async def _start(
    harness: Any, *, stream: bool
) -> tuple[asyncio.Task[Any], asyncio.Event, list[dict[str, Any]]]:
    sent: list[dict[str, Any]] = []
    body_sent = False
    disconnect = asyncio.Event()

    async def receive() -> dict[str, Any]:
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {"type": "http.request", "body": _payload(stream), "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "scheme": "http",
        "headers": [
            (b"host", b"test"),
            (b"content-type", b"application/json"),
            (b"user-agent", b"pytest"),
        ],
        "server": ("test", 80),
        "client": ("10.0.0.5", 12345),
        "root_path": "",
    }
    return asyncio.create_task(harness.app(scope, receive, send)), disconnect, sent


def _chat_request(request_id: str, *, stream: bool) -> ChatRequest:
    return ChatRequest(
        request_id=request_id,
        client=ClientIdentity(address="10.0.0.7", user_agent="pytest"),
        model="m",
        stream=stream,
        body={**_BODY, "stream": stream},
    )


def _counting_handler(calls: dict[str, int]):
    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"id": "c1", "choices": []})

    return handler


class _SlowBody(httpx.AsyncByteStream):
    async def __aiter__(self):
        await asyncio.sleep(30)
        yield b'{"id":"c1","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}'


def _slow_then_ok_handler(state: dict[str, bool]):
    def handler(request: httpx.Request) -> httpx.Response:
        if state["slow"]:
            return httpx.Response(
                200, headers={"content-type": "application/json"}, stream=_SlowBody()
            )
        return httpx.Response(200, json={"id": "c1", "choices": []})

    return handler


class _BlockingStream:
    status_code = 200
    headers: dict[str, str] = {}
    provider_id: str | None = None
    provider_correlation_id: str | None = None
    usage = None

    def __init__(self) -> None:
        self.opened = asyncio.Event()

    async def __aenter__(self) -> _BlockingStream:
        self.opened.set()
        await asyncio.sleep(30)
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def aiter_bytes(self):  # pragma: no cover - never iterated
        yield b""


class _BlockingProvider:
    def __init__(self) -> None:
        self.stream = _BlockingStream()

    async def complete(self, body: Any) -> Any:  # pragma: no cover - unused
        raise AssertionError("complete() is not used here")

    def open_stream(self, body: Any) -> _BlockingStream:
        return self.stream

    async def list_models(self) -> list[Any]:
        return []

    async def fetch_usage(self) -> SubscriptionInfo:
        return SubscriptionInfo()

    async def aclose(self) -> None:
        return None


async def test_queued_client_when_it_disconnects_then_frees_the_queue_and_is_recorded(
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls = {"n": 0}
    harness = build_harness(_counting_handler(calls))
    holder = harness.gate.slot(
        request_id="hold",
        client=ClientIdentity(address="10.0.0.1"),
        model="m",
        streaming=False,
        max_wait_s=5.0,
    )
    await holder.__aenter__()

    with caplog.at_level(logging.WARNING, logger="ipp.chat"):
        task, disconnect, sent = await _start(harness, stream=True)
        await _wait_until(lambda: harness.gate.snapshot().queued == 1)
        disconnect.set()
        await asyncio.wait_for(task, 5)
    snapshot = harness.gate.snapshot()
    await holder.__aexit__(None, None, None)

    assert snapshot.cancelled_while_waiting_total == 1
    assert snapshot.queued == 0
    assert snapshot.in_flight == 1
    assert calls["n"] == 0
    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 499
    metrics = harness.metrics
    assert metrics.errors().by_kind[ErrorKind.CLIENT_ABORTED.value] == 1
    throughput = metrics.throughput()
    assert throughput.requests_aborted == 1
    assert throughput.requests_completed == 0
    rows = metrics.clients()
    assert sum(row.failed for row in rows) == throughput.requests_failed
    assert sum(row.aborted for row in rows) == throughput.requests_aborted
    record = metrics.recent_requests()[0]
    assert (record.status, record.aborted, record.streaming) == (499, True, True)
    assert "client disconnected while queued" in caplog.text


async def test_disconnect_during_the_provider_call_when_it_frees_the_slot_then_records_abort(
    caplog: pytest.LogCaptureFixture,
) -> None:
    state = {"slow": True}
    harness = build_harness(_slow_then_ok_handler(state))

    with caplog.at_level(logging.WARNING, logger="ipp.chat"):
        task, disconnect, sent = await _start(harness, stream=False)
        await _wait_until(lambda: harness.gate.snapshot().in_flight == 1)
        await asyncio.sleep(0.35)
        started = time.monotonic()
        disconnect.set()
        await asyncio.wait_for(task, 5)
        elapsed = time.monotonic() - started
    snapshot = harness.gate.snapshot()
    metrics = harness.metrics
    record = metrics.recent_requests()[0]

    assert elapsed < 3.0
    assert snapshot.in_flight == 0
    assert snapshot.queued == 0
    assert sent[0]["status"] == 499
    assert (record.status, record.aborted) == (499, True)
    assert metrics.errors().by_kind[ErrorKind.CLIENT_ABORTED.value] == 1
    assert metrics.throughput().requests_aborted == 1
    assert metrics.throughput().requests_completed == 0
    assert "client disconnected while in flight" in caplog.text


async def test_slot_when_freed_by_a_disconnect_then_serves_the_next_client() -> None:
    state = {"slow": True}
    harness = build_harness(_slow_then_ok_handler(state))
    task, disconnect, _ = await _start(harness, stream=False)
    await _wait_until(lambda: harness.gate.snapshot().in_flight == 1)
    disconnect.set()
    await asyncio.wait_for(task, 5)
    state["slow"] = False

    follow_up = await harness.chat.complete(_chat_request("next", stream=False))

    assert harness.gate.snapshot().in_flight == 0
    assert follow_up.status_code == 200


async def _stalled_call(finished: asyncio.Event) -> None:
    try:
        await asyncio.sleep(30)
    finally:
        finished.set()


async def _late_disconnect_receive() -> dict[str, Any]:
    await asyncio.sleep(30)
    return {"type": "http.disconnect"}


async def test_cancellation_from_outside_when_raised_then_is_not_turned_into_an_abort() -> None:
    finished = asyncio.Event()
    scope = {"type": "http", "headers": [], "method": "POST", "path": "/", "query_string": b""}

    task = asyncio.create_task(
        _await_pre_response(Request(scope, _late_disconnect_receive), _stalled_call(finished))
    )
    await asyncio.sleep(0.05)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()


async def test_stream_open_when_cancelled_then_records_the_abort_and_frees_the_slot(
    caplog: pytest.LogCaptureFixture,
) -> None:
    harness = build_harness(lambda request: httpx.Response(200, json={"data": []}))
    provider = _BlockingProvider()
    chat = ChatService(
        gate=harness.gate,
        provider=provider,
        metrics=harness.metrics,
        clock=SystemClock(),
        settings=harness.settings,
    )

    with caplog.at_level(logging.WARNING, logger="ipp.chat"):
        task = asyncio.create_task(chat.stream(_chat_request("r1", stream=True)))
        await asyncio.wait_for(provider.stream.opened.wait(), 5)
        held = harness.gate.snapshot().in_flight
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
    released = harness.gate.snapshot()

    assert held == 1
    assert released.in_flight == 0
    metrics = harness.metrics
    assert metrics.errors().by_kind[ErrorKind.CLIENT_ABORTED.value] == 1
    throughput = metrics.throughput()
    assert (throughput.requests_aborted, throughput.requests_completed) == (1, 0)
    assert "client disconnected while in flight" in caplog.text
