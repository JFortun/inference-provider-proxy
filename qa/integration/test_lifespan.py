"""The application lifespan owns the background poller (FR-015, FR-032).

Driving a real ASGI ``lifespan`` scope through the app also proves the
request-context middleware passes non-HTTP scopes straight through.
"""

from __future__ import annotations

import asyncio

import httpx

from inference_provider_proxy import main
from qa.integration.conftest import build_harness


class _FakePoller:
    def __init__(self) -> None:
        self.events: list[str] = []

    async def start(self) -> None:
        self.events.append("start")

    async def stop(self) -> None:
        self.events.append("stop")


def _unused_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"object": "list", "data": []})


async def _drive_lifespan(app) -> list[dict]:
    """Send a startup then a shutdown over the ASGI lifespan protocol."""
    inbox: asyncio.Queue[dict] = asyncio.Queue()
    inbox.put_nowait({"type": "lifespan.startup"})
    inbox.put_nowait({"type": "lifespan.shutdown"})
    sent: list[dict] = []

    async def receive() -> dict:
        return await inbox.get()

    async def send(message: dict) -> None:
        sent.append(message)

    await app({"type": "lifespan"}, receive, send)
    return sent


async def test_lifespan_starts_then_stops_the_poller(monkeypatch) -> None:
    harness = build_harness(_unused_handler)
    services = harness.app.state.services
    poller = _FakePoller()

    closes = {"n": 0}
    real_aclose = services.provider.aclose

    async def _record_aclose() -> None:
        closes["n"] += 1
        await real_aclose()

    monkeypatch.setattr(services.provider, "aclose", _record_aclose)
    monkeypatch.setattr(main, "build_services", lambda settings: (services, poller))

    app = main.build_app()
    sent = await _drive_lifespan(app)

    assert [message["type"] for message in sent] == [
        "lifespan.startup.complete",
        "lifespan.shutdown.complete",
    ]
    assert poller.events == ["start", "stop"]
    assert closes["n"] == 1
