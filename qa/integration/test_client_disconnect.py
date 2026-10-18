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


async def _start(
    harness: Any, *, stream: bool
) -> tuple[asyncio.Task[Any], asyncio.Event, list[dict[str, Any]]]:
    """Run the ASGI app with a receive channel the test controls."""
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


def _payload(stream: bool) -> bytes:
    return json.dumps({**_BODY, "stream": stream}).encode()


def _chat_request(request_id: str, *, stream: bool) -> ChatRequest:
    return ChatRequest(
        request_id=request_id,
        client=ClientIdentity(address="10.0.0.7", user_agent="pytest"),
        model="m",
        stream=stream,
        body={**_BODY, "stream": stream},
    )


async def test_queued_client_disconnect_frees_the_queue_and_is_recorded(
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"id": "c1", "choices": []})

    harness = build_harness(handler)
    holder = harness.gate.slot(
        request_id="hold",
        client=ClientIdentity(address="10.0.0.1"),
        model="m",
        streaming=False,
        max_wait_s=5.0,
    )
    await holder.__aenter__()
    try:
        with caplog.at_level(logging.WARNING, logger="ipp.chat"):
            task, disconnect, sent = await _start(harness, stream=True)
            await _wait_until(lambda: harness.gate.snapshot().queued == 1)
            disconnect.set()
            await asyncio.wait_for(task, 5)

        snapshot = harness.gate.snapshot()
        assert snapshot.cancelled_while_waiting_total == 1  # observed over real ASGI
        assert snapshot.queued == 0  # the place is given back at once
        assert snapshot.in_flight == 1  # only the holder remains
        assert calls["n"] == 0  # a dead client never reaches the provider

        assert sent[0]["type"] == "http.response.start"
        assert sent[0]["status"] == 499  # answered in the dialect
        metrics = harness.metrics
        assert metrics.errors().by_kind[ErrorKind.CLIENT_ABORTED.value] == 1
        throughput = metrics.throughput()
        assert throughput.requests_aborted == 1
        assert throughput.requests_completed == 0
        rows = metrics.clients()  # the two sections label the abort alike
        assert sum(row.failed for row in rows) == throughput.requests_failed
        assert sum(row.aborted for row in rows) == throughput.requests_aborted
        record = metrics.recent_requests()[0]
        assert (record.status, record.aborted, record.streaming) == (499, True, True)
        assert "client disconnected while queued" in caplog.text
    finally:
        await holder.__aexit__(None, None, None)


async def test_disconnect_during_the_provider_call_aborts_it_and_frees_the_slot(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _SlowBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.sleep(30)
            yield b'{"id":"c1","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}'

    state = {"slow": True}

    def handler(request: httpx.Request) -> httpx.Response:
        if state["slow"]:
            return httpx.Response(
                200, headers={"content-type": "application/json"}, stream=_SlowBody()
            )
        return httpx.Response(200, json={"id": "c1", "choices": []})

    harness = build_harness(handler)
    with caplog.at_level(logging.WARNING, logger="ipp.chat"):
        task, disconnect, sent = await _start(harness, stream=False)
        await _wait_until(lambda: harness.gate.snapshot().in_flight == 1)
        await asyncio.sleep(0.35)  # at least one watcher poll while in flight

        started = time.monotonic()
        disconnect.set()
        await asyncio.wait_for(task, 5)
        elapsed = time.monotonic() - started

    assert elapsed < 3.0  # did not wait out the 30s provider call
    snapshot = harness.gate.snapshot()
    assert snapshot.in_flight == 0  # slot released without finishing the call
    assert snapshot.queued == 0

    assert sent[0]["status"] == 499
    metrics = harness.metrics
    assert metrics.errors().by_kind[ErrorKind.CLIENT_ABORTED.value] == 1
    assert metrics.throughput().requests_aborted == 1
    assert metrics.throughput().requests_completed == 0  # never a ghost 200
    record = metrics.recent_requests()[0]
    assert (record.status, record.aborted) == (499, True)
    assert "client disconnected while in flight" in caplog.text

    state["slow"] = False
    follow_up = await harness.chat.complete(
        ChatRequest(
            request_id="next",
            client=ClientIdentity(address="10.0.0.7"),
            model="m",
            stream=False,
            body=_BODY,
        )
    )
    assert follow_up.status_code == 200  # the freed slot serves the next client


async def test_cancellation_from_outside_is_not_turned_into_an_abort() -> None:
    """Only a gone client becomes 499; a shutdown cancel must pass through."""

    finished = asyncio.Event()

    async def slow() -> None:
        try:
            await asyncio.sleep(30)
        finally:
            finished.set()

    async def receive() -> dict[str, Any]:
        await asyncio.sleep(30)
        return {"type": "http.disconnect"}

    scope = {"type": "http", "headers": [], "method": "POST", "path": "/", "query_string": b""}
    task = asyncio.create_task(_await_pre_response(Request(scope, receive), slow()))
    await asyncio.sleep(0.05)  # let the watcher poll at least once
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()  # the inner call was stopped, not left running


async def test_cancelling_the_stream_open_records_the_abort_and_frees_the_slot(
    caplog: pytest.LogCaptureFixture,
) -> None:
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
        assert harness.gate.snapshot().in_flight == 1  # admitted, now opening
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)

    assert harness.gate.snapshot().in_flight == 0
    metrics = harness.metrics
    assert metrics.errors().by_kind[ErrorKind.CLIENT_ABORTED.value] == 1
    throughput = metrics.throughput()
    assert (throughput.requests_aborted, throughput.requests_completed) == (1, 0)
    assert "client disconnected while in flight" in caplog.text
