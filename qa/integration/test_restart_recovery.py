"""Restart recovery.

All state is in memory, so an unexpected exit leaves nothing to clean up: a
fresh process must resume serving immediately, and a request interrupted by the
exit must never be reported as successful.
"""

from __future__ import annotations

import time

import httpx
import pytest

from inference_provider_proxy.config import get_settings
from inference_provider_proxy.main import build_app


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("IPP_UPSTREAM_API_KEY", "test-key")
    monkeypatch.setenv("IPP_BIND_PORT", "13306")
    yield


async def test_service_when_restarted_without_operator_action_then_resumes_serving(
    clean_env,
) -> None:
    get_settings.cache_clear()
    first = build_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=first), base_url="http://proxy.local"
    ) as client:
        assert (await client.get("/healthz")).status_code == 200

    started = time.monotonic()
    second = build_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=second), base_url="http://proxy.local"
    ) as client:
        assert (await client.get("/healthz")).status_code == 200
        document = (await client.get("/dashboard")).json()
    elapsed = time.monotonic() - started
    get_settings.cache_clear()

    assert elapsed < 10.0
    assert document["throughput"]["requests_total"] == 0
    assert document["gate"]["in_flight"] == 0
    assert document["gate"]["queued"] == 0
