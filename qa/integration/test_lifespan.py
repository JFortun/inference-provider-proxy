"""The application lifespan owns the background poller.

Driving a real ASGI ``lifespan`` scope through the app also proves the
request-context middleware passes non-HTTP scopes straight through.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from inference_provider_proxy import main
from inference_provider_proxy.config import Settings
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


class _counting_aclose:
    """Wrap a provider's ``aclose`` and count how often it is called."""

    def __init__(self, provider) -> None:
        self._provider = provider
        self._real = provider.aclose
        self.calls = 0

    async def __call__(self) -> None:
        self.calls += 1
        await self._real()


async def _drive_lifespan(app) -> list[dict]:
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


async def test_lifespan_when_driven_then_starts_then_stops_the_poller(monkeypatch) -> None:
    harness = build_harness(_unused_handler)
    services = harness.app.state.services
    poller = _FakePoller()
    aclose_calls = _counting_aclose(services.provider)

    monkeypatch.setattr(services.provider, "aclose", aclose_calls)
    monkeypatch.setattr(main, "build_services", lambda settings: (services, poller))
    app = main.build_app()
    sent = await _drive_lifespan(app)

    assert [message["type"] for message in sent] == [
        "lifespan.startup.complete",
        "lifespan.shutdown.complete",
    ]
    assert poller.events == ["start", "stop"]
    assert aclose_calls.calls == 1


async def test_startup_and_shutdown_when_logged_then_never_expose_the_key(
    monkeypatch, caplog
) -> None:
    harness = build_harness(_unused_handler)
    services = harness.app.state.services
    settings = Settings(
        _env_file=None,
        upstream_api_key="sekret-provider-key-abc123",
        bind_port=4242,
    )

    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "build_services", lambda _settings: (services, _FakePoller()))
    app = main.build_app()
    with caplog.at_level(logging.INFO, logger="ipp.app"):
        await _drive_lifespan(app)

    lines = [record.getMessage() for record in caplog.records if record.name == "ipp.app"]
    assert any(line.startswith("starting proxy on 0.0.0.0:4242") for line in lines)
    assert any("key=set" in line for line in lines)
    assert any(line.startswith("limits capacity=1") for line in lines)
    assert "proxy stopped" in lines
    assert "sekret-provider-key-abc123" not in caplog.text
    assert "Bearer" not in caplog.text
