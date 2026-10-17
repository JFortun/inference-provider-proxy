"""What an operator sees in the log: one aligned line per event, at the right level.

Covers the three things that only surface end to end: the access line the
middleware emits, the level each classified failure is reported at, and the
catch-all that answers (and logs) what no one classified.
"""

from __future__ import annotations

import logging
import re

import httpx

from inference_provider_proxy.domain.models import ClientIdentity
from inference_provider_proxy.logging_setup import configure_logging
from qa.integration.conftest import build_harness

_BODY = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


def _ok_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"id": "c1", "object": "list", "data": []})


# -- the access line ---------------------------------------------------------


async def test_api_requests_are_logged_and_browser_pages_stay_quiet(
    capsys, asgi_client_factory
) -> None:
    configure_logging("info", color="never")
    harness = build_harness(_ok_handler)
    client = asgi_client_factory(harness.app)

    await client.get("/v1/models")
    await client.get("/healthz")

    lines = [text for text in capsys.readouterr().err.splitlines() if "[access" in text]
    assert len(lines) == 1  # /healthz is polled: its line drops to DEBUG
    assert re.fullmatch(
        r"\d{2}:\d{2}:\d{2}\.\d{3} INFO  \[access\s+\] GET /v1/models 200 \d+ms "
        r"client=127\.0\.0\.1 request_id=[0-9a-f]{32}",
        lines[0],
    )


# -- levels per failure class ------------------------------------------------


async def test_provider_trouble_is_reported_at_error(caplog, asgi_client_factory) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "boom", "type": "internal"}})

    harness = build_harness(handler, max_retries=0)
    client = asgi_client_factory(harness.app)

    with caplog.at_level(logging.DEBUG, logger="ipp.http"):
        response = await client.post("/v1/chat/completions", json=_BODY)

    assert response.status_code == 500
    record = next(record for record in caplog.records if record.name == "ipp.http")
    assert record.levelno == logging.ERROR
    assert "500 upstream_5xx" in record.getMessage()
    assert "request_id=" in record.getMessage()


async def test_a_client_mistake_stays_at_info(caplog, asgi_client_factory) -> None:
    harness = build_harness(_ok_handler)
    client = asgi_client_factory(harness.app)

    with caplog.at_level(logging.DEBUG, logger="ipp.http"):
        response = await client.post(
            "/v1/chat/completions",
            content=b"{not json",
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 400
    record = next(record for record in caplog.records if record.name == "ipp.http")
    assert record.levelno == logging.INFO
    assert "400 unknown" in record.getMessage()


async def test_shedding_load_is_reported_at_warning(caplog, asgi_client_factory) -> None:
    harness = build_harness(_ok_handler, provider_concurrency=1, max_queue_depth=0)
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
        with caplog.at_level(logging.DEBUG, logger="ipp.http"):
            response = await client.post("/v1/chat/completions", json={**_BODY, "stream": True})
    finally:
        await slot_cm.__aexit__(None, None, None)

    assert response.status_code == 429
    record = next(record for record in caplog.records if record.name == "ipp.http")
    assert record.levelno == logging.WARNING
    assert "429 queue_full" in record.getMessage()


# -- the catch-all -----------------------------------------------------------


async def test_an_unclassified_crash_is_logged_with_its_traceback(caplog, monkeypatch) -> None:
    harness = build_harness(_ok_handler)

    async def explode() -> dict:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(harness.catalog, "models", explode)
    transport = httpx.ASGITransport(app=harness.app, raise_app_exceptions=False)

    with caplog.at_level(logging.ERROR, logger="ipp.http"):
        async with httpx.AsyncClient(transport=transport, base_url="http://proxy.local") as client:
            response = await client.get("/v1/models")

    assert response.status_code == 500
    assert response.json()["error"] == {
        "message": "Unexpected proxy failure: RuntimeError.",
        "type": "server_error",
        "code": "server_error",
        "param": None,
    }
    record = next(record for record in caplog.records if record.name == "ipp.http")
    assert record.levelno == logging.ERROR
    assert "RuntimeError" in record.getMessage()
    assert record.exc_info is not None
    assert "Traceback" in caplog.text
    # The traceback carries the id that ties it back to the access log.
    assert "request_id=" in record.getMessage()
