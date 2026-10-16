"""Composition root: build the adapters and wire the application (T021).

This is the only place that knows about both ports and concrete adapters. All
state is in memory, so an unexpected exit leaves nothing to clean up and the
service resumes serving on restart (FR-032, SC-013).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from inference_provider_proxy.adapters.inbound.http.app import create_app
from inference_provider_proxy.adapters.inbound.http.deps import Services
from inference_provider_proxy.adapters.outbound.gate.asyncio_gate import AsyncioGate
from inference_provider_proxy.adapters.outbound.metrics.memory import MemoryMetrics
from inference_provider_proxy.adapters.outbound.provider.cheapestinference import (
    CheapestInferenceProvider,
)
from inference_provider_proxy.adapters.outbound.subscription.poller import SubscriptionPoller
from inference_provider_proxy.application.ports.clock import SystemClock
from inference_provider_proxy.application.services.catalog import CatalogService
from inference_provider_proxy.application.services.chat import ChatService
from inference_provider_proxy.application.services.dashboard import DashboardService
from inference_provider_proxy.config import Settings, get_settings


def build_services(settings: Settings) -> tuple[Services, SubscriptionPoller]:
    """Construct every adapter and service exactly once."""
    clock = SystemClock()
    started_at = clock.now()

    metrics = MemoryMetrics(
        base_url=settings.upstream_base_url,
        retention=settings.metrics_retention,
        clock=clock,
    )
    gate = AsyncioGate(
        capacity=settings.provider_concurrency,
        max_depth=settings.max_queue_depth,
        clock=clock,
    )
    provider = CheapestInferenceProvider(
        base_url=settings.upstream_base_url,
        api_key=settings.provider_key,
        user_agent=settings.user_agent,
        timeout_s=settings.upstream_timeout,
        stream_idle_timeout_s=settings.stream_idle_timeout,
    )
    chat = ChatService(
        gate=gate,
        provider=provider,
        metrics=metrics,
        clock=clock,
        settings=settings,
    )
    catalog = CatalogService(
        provider=provider,
        metrics=metrics,
        settings=settings,
        clock=clock,
        subscription=metrics,
    )
    dashboard = DashboardService(
        gate=gate,
        metrics=metrics,
        subscription=metrics,
        settings=settings,
        started_at=started_at,
        clock=clock,
    )
    services = Services(
        settings=settings,
        provider=provider,
        chat=chat,
        catalog=catalog,
        dashboard=dashboard,
    )
    poller = SubscriptionPoller(
        provider=provider,
        subscription=metrics,
        interval_s=settings.metadata_cache_ttl,
        clock=clock,
    )
    return services, poller


def build_app() -> FastAPI:
    """Entry point for ``uvicorn --factory``."""
    settings = get_settings()
    services, poller = build_services(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        await poller.start()
        try:
            yield
        finally:
            await poller.stop()
            await services.provider.aclose()

    return create_app(services, lifespan=lifespan)
