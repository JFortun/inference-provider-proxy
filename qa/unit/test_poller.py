"""The background subscription poller refreshes without touching the gate (FR-015)."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import httpx

from inference_provider_proxy.adapters.outbound.metrics.memory import MemoryMetrics
from inference_provider_proxy.adapters.outbound.provider.cheapestinference import (
    CheapestInferenceProvider,
)
from inference_provider_proxy.adapters.outbound.subscription.poller import SubscriptionPoller
from qa.fakes import FakeClock

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _provider(handler) -> CheapestInferenceProvider:
    return CheapestInferenceProvider(
        base_url="https://provider.test",
        api_key="k",
        user_agent="ipp/0.1",
        timeout_s=10,
        stream_idle_timeout_s=10,
        transport=httpx.MockTransport(handler),
    )


def _poller(provider: CheapestInferenceProvider, metrics: MemoryMetrics) -> SubscriptionPoller:
    return SubscriptionPoller(
        provider=provider, subscription=metrics, interval_s=30.0, clock=FakeClock()
    )


def _metrics() -> MemoryMetrics:
    return MemoryMetrics(base_url="https://provider.test", retention=10, clock=FakeClock())


async def test_refresh_once_updates_the_subscription_on_success() -> None:
    body = (FIXTURES / "usage_200.json").read_bytes()
    provider = _provider(lambda request: httpx.Response(200, content=body))
    metrics = _metrics()
    try:
        await _poller(provider, metrics).refresh_once()
    finally:
        await provider.aclose()

    assert metrics.current().pool == "core"
    assert metrics.current().stale is False


async def test_refresh_once_marks_the_view_stale_on_failure(caplog) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "boom", "type": "internal"}})

    provider = _provider(handler)
    metrics = _metrics()
    try:
        with caplog.at_level(logging.WARNING, logger="ipp.subscription"):
            await _poller(provider, metrics).refresh_once()
    finally:
        await provider.aclose()

    assert metrics.current().stale is True
    assert "subscription refresh failed" in caplog.text


async def test_repeated_failures_are_damped_and_recovery_is_announced(caplog) -> None:
    calls = {"n": 0}
    body = (FIXTURES / "usage_200.json").read_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= 2:
            return httpx.Response(500, json={"error": {"message": "boom", "type": "internal"}})
        return httpx.Response(200, content=body)

    provider = _provider(handler)
    metrics = _metrics()
    try:
        with caplog.at_level(logging.DEBUG, logger="ipp.subscription"):
            poller = _poller(provider, metrics)
            await poller.refresh_once()  # first failure: a WARNING
            await poller.refresh_once()  # second: damped to DEBUG
            await poller.refresh_once()  # healthy again: an INFO announcement
    finally:
        await provider.aclose()

    records = [record for record in caplog.records if record.name == "ipp.subscription"]
    warnings = [record for record in records if record.levelno == logging.WARNING]
    debugs = [record for record in records if record.levelno == logging.DEBUG]
    infos = [record for record in records if record.levelno == logging.INFO]

    assert len(warnings) == 1
    assert "subscription refresh failed" in warnings[0].getMessage()
    assert any("attempt 2" in record.getMessage() for record in debugs)
    assert len(infos) == 1
    assert "recovered after 2 failed attempts" in infos[0].getMessage()


async def test_start_is_idempotent_and_stop_cancels_the_loop() -> None:
    calls = {"n": 0}
    body = (FIXTURES / "usage_200.json").read_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, content=body)

    provider = _provider(handler)
    metrics = _metrics()
    poller = _poller(provider, metrics)
    try:
        await poller.start()
        await poller.start()  # a second start must not spawn a second refresh task
        for _ in range(50):
            if calls["n"]:
                break
            await asyncio.sleep(0)
        await poller.stop()
    finally:
        await provider.aclose()

    assert calls["n"] == 1
    assert metrics.current().pool == "core"


async def test_stop_without_start_is_a_no_op() -> None:
    provider = _provider(lambda request: httpx.Response(200, content=b"{}"))
    try:
        await _poller(provider, _metrics()).stop()
    finally:
        await provider.aclose()
