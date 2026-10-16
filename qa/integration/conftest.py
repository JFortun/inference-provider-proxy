"""Test harness: build the real app around a fake provider transport."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from inference_provider_proxy.adapters.inbound.http.app import create_app
from inference_provider_proxy.adapters.inbound.http.deps import Services
from inference_provider_proxy.adapters.outbound.gate.asyncio_gate import AsyncioGate
from inference_provider_proxy.adapters.outbound.metrics.memory import MemoryMetrics
from inference_provider_proxy.adapters.outbound.provider.cheapestinference import (
    CheapestInferenceProvider,
)
from inference_provider_proxy.application.ports.clock import SystemClock
from inference_provider_proxy.application.services.catalog import CatalogService
from inference_provider_proxy.application.services.chat import ChatService
from inference_provider_proxy.application.services.dashboard import DashboardService
from inference_provider_proxy.config import Settings
from qa.fakes import FakeClock

BASE_URL = "https://provider.test"


@dataclass
class Harness:
    app: Any
    provider: CheapestInferenceProvider
    metrics: MemoryMetrics
    settings: Settings
    gate: AsyncioGate
    chat: ChatService
    catalog: CatalogService


def build_harness(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    clock: object | None = None,
    **settings_overrides: Any,
) -> Harness:
    base: dict[str, Any] = {
        "_env_file": None,
        "upstream_base_url": BASE_URL,
        "upstream_api_key": "test-key",
    }
    base.update(settings_overrides)
    settings = Settings(**base)
    real_clock = clock or SystemClock()
    metrics = MemoryMetrics(
        base_url=settings.upstream_base_url,
        retention=settings.metrics_retention,
        clock=real_clock,
    )
    gate = AsyncioGate(
        capacity=settings.provider_concurrency,
        max_depth=settings.max_queue_depth,
        clock=real_clock,
    )
    provider = CheapestInferenceProvider(
        base_url=settings.upstream_base_url,
        api_key=settings.provider_key,
        user_agent=settings.user_agent,
        timeout_s=settings.upstream_timeout,
        stream_idle_timeout_s=settings.stream_idle_timeout,
        transport=httpx.MockTransport(handler),
    )
    chat = ChatService(
        gate=gate,
        provider=provider,
        metrics=metrics,
        clock=real_clock,
        settings=settings,
    )
    catalog = CatalogService(
        provider=provider,
        metrics=metrics,
        settings=settings,
        clock=real_clock,
        subscription=metrics,
    )
    dashboard = DashboardService(
        gate=gate,
        metrics=metrics,
        subscription=metrics,
        settings=settings,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        clock=real_clock,
    )
    services = Services(
        settings=settings,
        provider=provider,
        chat=chat,
        catalog=catalog,
        dashboard=dashboard,
    )
    return Harness(
        app=create_app(services),
        provider=provider,
        metrics=metrics,
        settings=settings,
        gate=gate,
        chat=chat,
        catalog=catalog,
    )


@pytest.fixture
def asgi_client_factory():
    def factory(app: Any) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.local"
        )

    return factory


__all__ = ["BASE_URL", "FakeClock", "Harness", "build_harness"]
