"""Driven ports for metrics aggregation and subscription state."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from datetime import datetime

    from inference_provider_proxy.domain.metrics import (
        ClientSnapshot,
        ErrorEpisode,
        ErrorSnapshot,
        LatencySnapshot,
        RequestRecord,
        ThroughputSnapshot,
        UpstreamSnapshot,
    )
    from inference_provider_proxy.domain.models import ClientIdentity, ModelInfo, SubscriptionInfo


class MetricsPort(Protocol):
    """In-memory aggregation over the process session."""

    def note_client(self, client: ClientIdentity) -> None: ...

    def record_request(self, record: RequestRecord) -> None: ...

    def record_error(self, episode: ErrorEpisode) -> None: ...

    def record_retry(self) -> None: ...

    def set_models(self, models: list[ModelInfo], *, checked_at: datetime) -> None: ...

    def set_upstream_error(self, message: str | None) -> None: ...

    def throughput(self) -> ThroughputSnapshot: ...

    def latency(self) -> LatencySnapshot: ...

    def errors(self) -> ErrorSnapshot: ...

    def clients(self) -> list[ClientSnapshot]: ...

    def recent_requests(self) -> list[RequestRecord]: ...

    def upstream(self) -> UpstreamSnapshot: ...


class SubscriptionPort(Protocol):
    """Holds the latest upstream view: entitlement plus the model catalogue."""

    def current(self) -> SubscriptionInfo: ...

    def update(self, info: SubscriptionInfo) -> None: ...

    def mark_stale(self, stale: bool = True) -> None: ...

    def set_models(self, models: list[ModelInfo], *, checked_at: datetime) -> None: ...
