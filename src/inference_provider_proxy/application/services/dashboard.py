"""Assembles the single dashboard document.

Every read is from in-memory state and synchronous, so the document is
retrievable while the key is busy and is never blocked by the gate.
"""

from __future__ import annotations

import platform
from typing import TYPE_CHECKING

from inference_provider_proxy import __version__
from inference_provider_proxy.domain.metrics import (
    BindAddress,
    DashboardDocument,
    SessionInfo,
)

if TYPE_CHECKING:
    from datetime import datetime

    from inference_provider_proxy.application.ports.clock import ClockPort
    from inference_provider_proxy.application.ports.gate import ConcurrencyGatePort
    from inference_provider_proxy.application.ports.metrics import MetricsPort, SubscriptionPort
    from inference_provider_proxy.config import Settings


class DashboardService:
    """Implements ``DashboardUseCase``."""

    def __init__(
        self,
        *,
        gate: ConcurrencyGatePort,
        metrics: MetricsPort,
        subscription: SubscriptionPort,
        settings: Settings,
        started_at: datetime,
        clock: ClockPort,
    ) -> None:
        self._gate = gate
        self._metrics = metrics
        self._subscription = subscription
        self._settings = settings
        self._started_at = started_at
        self._clock = clock

    def document(self) -> DashboardDocument:
        now = self._clock.now()
        return DashboardDocument(
            generated_at=now,
            session=SessionInfo(
                started_at=self._started_at,
                uptime_s=round((now - self._started_at).total_seconds(), 3),
                package_version=__version__,
                python_version=platform.python_version(),
                bind=BindAddress(host=self._settings.bind_host, port=self._settings.bind_port),
            ),
            subscription=self._subscription.current(),
            gate=self._gate.snapshot(),
            throughput=self._metrics.throughput(),
            latency=self._metrics.latency(),
            errors=self._metrics.errors(),
            upstream=self._metrics.upstream(),
            clients=tuple(self._metrics.clients()),
            recent_requests=tuple(self._metrics.recent_requests()),
        )
