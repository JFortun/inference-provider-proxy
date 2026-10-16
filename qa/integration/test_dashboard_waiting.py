"""A queued request is visible in the dashboard while it waits (FR-019, SC-008)."""

from __future__ import annotations

import asyncio

import httpx

from qa.integration.conftest import build_harness

_BODY = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


def _slow_handler(delay_s: float = 0.3):
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(delay_s)
        return httpx.Response(
            200,
            json={
                "id": "c1",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    return handler


async def _wait_for(predicate, *, timeout_s: float = 3.0) -> None:
    for _ in range(int(timeout_s / 0.01)):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition was not met in time")


async def test_a_waiter_is_visible_in_the_dashboard(asgi_client_factory) -> None:
    harness = build_harness(_slow_handler(), provider_concurrency=1, max_queue_depth=4)
    client = asgi_client_factory(harness.app)

    first = asyncio.create_task(client.post("/v1/chat/completions", json=_BODY))
    await _wait_for(lambda: harness.gate.snapshot().in_flight == 1)

    second = asyncio.create_task(client.post("/v1/chat/completions", json=_BODY))
    await _wait_for(lambda: harness.gate.snapshot().queued == 1)

    document = (await client.get("/dashboard")).json()
    waiting = document["gate"]["waiting"]
    assert len(waiting) == 1
    assert waiting[0]["position"] == 1
    assert waiting[0]["model"] == "m"
    assert waiting[0]["waited_s"] >= 0

    responses = await asyncio.gather(first, second)
    assert [r.status_code for r in responses] == [200, 200]
