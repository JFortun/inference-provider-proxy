"""The background subscription poller refreshes without touching the gate."""

from __future__ import annotations

import asyncio
import logging
from functools import partial
from pathlib import Path

import httpx

from inference_provider_proxy.adapters.outbound.metrics.memory import MemoryMetrics
from inference_provider_proxy.adapters.outbound.provider.cheapestinference import (
    CheapestInferenceProvider,
)
from inference_provider_proxy.adapters.outbound.subscription.poller import SubscriptionPoller
from qa.fakes import FakeClock

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
USAGE_BODY = (FIXTURES / "usage_200.json").read_bytes()


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


def _usage_ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=USAGE_BODY)


def _usage_500(request: httpx.Request) -> httpx.Response:
    return httpx.Response(500, json={"error": {"message": "boom", "type": "internal"}})


def _fail_twice_then_ok(request: httpx.Request, calls: dict) -> httpx.Response:
    calls["n"] += 1
    if calls["n"] <= 2:
        return httpx.Response(500, json={"error": {"message": "boom", "type": "internal"}})
    return httpx.Response(200, content=USAGE_BODY)


def _count_ok(request: httpx.Request, calls: dict) -> httpx.Response:
    calls["n"] += 1
    return httpx.Response(200, content=USAGE_BODY)


def _model_payload() -> dict:
    return {"data": [{"id": "m", "object": "model", "context_length": 1}]}


def _catalogue_then_usage(request: httpx.Request, successes: dict) -> httpx.Response:
    if request.url.path.endswith("/v1/models"):
        return httpx.Response(200, json=_model_payload())
    if successes["usage"] < 1:
        successes["usage"] += 1
        return httpx.Response(200, content=USAGE_BODY)
    return httpx.Response(500, json={"error": {"message": "boom", "type": "internal"}})


def _models_then_usage(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/v1/models"):
        return httpx.Response(200, json=_model_payload())
    return httpx.Response(200, content=USAGE_BODY)


async def _refresh_once(provider, metrics, caplog=None):
    try:
        if caplog is None:
            await _poller(provider, metrics).refresh_once()
        else:
            with caplog.at_level(logging.WARNING, logger="ipp.subscription"):
                await _poller(provider, metrics).refresh_once()
    finally:
        await provider.aclose()


async def test_refresh_once_when_provider_succeeds_then_updates_the_subscription() -> None:
    provider = _provider(_usage_ok)
    metrics = _metrics()

    await _refresh_once(provider, metrics)

    assert metrics.current().pool == "core"
    assert metrics.current().stale is False


async def test_refresh_once_when_provider_fails_then_marks_the_view_stale(caplog) -> None:
    provider = _provider(_usage_500)
    metrics = _metrics()

    await _refresh_once(provider, metrics, caplog)

    assert metrics.current().stale is True
    assert "subscription refresh failed" in caplog.text


async def test_repeated_failures_when_damped_then_recovery_is_announced(caplog) -> None:
    calls = {"n": 0}
    provider = _provider(partial(_fail_twice_then_ok, calls=calls))
    metrics = _metrics()
    poller = _poller(provider, metrics)

    with caplog.at_level(logging.DEBUG, logger="ipp.subscription"):
        await poller.refresh_once()
        await poller.refresh_once()
        await poller.refresh_once()
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


async def test_start_when_called_twice_then_is_idempotent_and_stop_cancels_the_loop() -> None:
    calls = {"n": 0}
    provider = _provider(partial(_count_ok, calls=calls))
    metrics = _metrics()
    poller = _poller(provider, metrics)

    try:
        await poller.start()
        await poller.start()
        for _ in range(200):
            if metrics.current().pool == "core":
                break
            await asyncio.sleep(0)
        await poller.stop()
    finally:
        await provider.aclose()

    assert calls["n"] == 2
    assert metrics.current().pool == "core"


async def test_stop_when_never_started_then_is_a_no_op() -> None:
    provider = _provider(lambda request: httpx.Response(200, content=b"{}"))
    poller = _poller(provider, _metrics())

    await poller.stop()
    await provider.aclose()

    assert poller._task is None


async def test_refresh_once_when_models_are_returned_then_populates_the_catalogue() -> None:
    provider = _provider(_models_then_usage)
    metrics = _metrics()

    await _refresh_once(provider, metrics)

    upstream = metrics.upstream()
    assert [model.id for model in upstream.models] == ["m"]
    assert upstream.models_checked_at is not None
    assert metrics.current().pool == "core"


async def test_refresh_when_usage_fails_after_a_success_then_keeps_the_last_catalogue() -> None:
    successes = {"usage": 0}
    provider = _provider(partial(_catalogue_then_usage, successes=successes))
    metrics = _metrics()
    poller = _poller(provider, metrics)

    try:
        await poller.refresh_once()
        checked = metrics.upstream().models_checked_at
        await poller.refresh_once()
    finally:
        await provider.aclose()

    upstream = metrics.upstream()
    assert [model.id for model in upstream.models] == ["m"]
    assert upstream.models_checked_at == checked
    assert metrics.current().stale is True
