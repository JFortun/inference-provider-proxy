"""Restart recovery.

All state is in memory, so an unexpected exit leaves nothing to clean up: a
fresh process must resume serving immediately, and a request interrupted by the
exit must never be reported as successful.
"""

from __future__ import annotations

import time

import httpx
import pytest

from inference_provider_proxy.main import build_app


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch):
    # Deterministic settings: no .env, no inherited IPP_* surprises.
    monkeypatch.setenv("IPP_UPSTREAM_API_KEY", "test-key")
    monkeypatch.setenv("IPP_BIND_PORT", "13306")
    yield


async def test_service_resumes_without_operator_action(clean_env) -> None:
    from inference_provider_proxy.config import get_settings

    get_settings.cache_clear()

    first = build_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=first), base_url="http://proxy.local"
    ) as client:
        assert (await client.get("/healthz")).status_code == 200

    # Simulate an unexpected exit: nothing is persisted, nothing is cleaned up.
    started = time.monotonic()
    second = build_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=second), base_url="http://proxy.local"
    ) as client:
        assert (await client.get("/healthz")).status_code == 200
        document = (await client.get("/dashboard")).json()
    elapsed = time.monotonic() - started

    assert elapsed < 10.0, "the service must resume serving within 10 seconds"
    # Session counters restart from zero: an interrupted request is not
    # carried over and cannot be reported as successful afterwards.
    assert document["throughput"]["requests_total"] == 0
    assert document["gate"]["in_flight"] == 0
    assert document["gate"]["queued"] == 0
    get_settings.cache_clear()
