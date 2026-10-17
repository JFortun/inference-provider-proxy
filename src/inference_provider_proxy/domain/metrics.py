"""Metrics value objects (FR-019, FR-031).

Field names are canonical and pinned by the dashboard schema test (FR-023);
see ``inference_provider_proxy/contracts/dashboard.schema.json``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from inference_provider_proxy.domain.errors import ErrorKind
from inference_provider_proxy.domain.models import ModelInfo, SubscriptionInfo


@dataclass(frozen=True, slots=True)
class RequestRecord:
    """The outcome of one request (``recent_requests``)."""

    request_id: str
    at: datetime
    client: str
    model: str
    streaming: bool
    status: int
    queue_wait_ms: int
    queue_position: int | None
    duration_ms: int
    aborted: bool = False
    ttfb_ms: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    provider_id: str | None = None
    provider_correlation_id: str | None = None
    error_kind: ErrorKind | None = None


@dataclass(frozen=True, slots=True)
class InFlightRequest:
    """A request currently holding a slot (``gate.current[]``)."""

    request_id: str
    client: str
    model: str
    streaming: bool
    started_at: datetime
    elapsed_s: float
    ttfb_ms: int | None = None


@dataclass(frozen=True, slots=True)
class WaitingRequest:
    """A request queued for a slot (``gate.waiting[]``)."""

    request_id: str
    client: str
    model: str
    enqueued_at: datetime
    waited_s: float
    position: int


@dataclass(frozen=True, slots=True)
class GateSnapshot:
    """Current and cumulative view of concurrency (``gate``)."""

    capacity: int
    in_flight: int
    queued: int
    available_slots: int
    admitted_total: int
    rejected_full_total: int
    rejected_timeout_total: int
    cancelled_while_waiting_total: int
    oldest_queue_wait_s: float | None
    current: tuple[InFlightRequest, ...] = ()
    waiting: tuple[WaitingRequest, ...] = ()


@dataclass(frozen=True, slots=True)
class ThroughputSnapshot:
    """Aggregate throughput over the session (``throughput``)."""

    requests_total: int = 0
    requests_completed: int = 0
    requests_failed: int = 0
    requests_aborted: int = 0
    prompt_tokens_total: int = 0
    completion_tokens_total: int = 0
    tokens_per_second_recent: float = 0.0
    requests_per_minute_recent: float = 0.0


@dataclass(frozen=True, slots=True)
class LatencyDistribution:
    """Quantiles in milliseconds over a ring buffer, computed on read."""

    p50: int = 0
    p90: int = 0
    p95: int = 0
    p99: int = 0
    max: int = 0


@dataclass(frozen=True, slots=True)
class LatencySnapshot:
    """Latency distributions (``latency``)."""

    queue_wait_ms: LatencyDistribution = field(default_factory=LatencyDistribution)
    time_to_first_byte_ms: LatencyDistribution = field(default_factory=LatencyDistribution)
    total_duration_ms: LatencyDistribution = field(default_factory=LatencyDistribution)


@dataclass(frozen=True, slots=True)
class ErrorEpisode:
    """One classified failure (FR-024, FR-025).

    ``gate_in_flight``/``gate_capacity`` are the admission state at record time:
    ``in_flight <= capacity`` means the gate never over-admitted, which is what
    separates a provider throttle of an in-limit request from a real gate fault.
    """

    at: datetime
    kind: ErrorKind
    status: int | None = None
    model: str | None = None
    client: str | None = None
    provider_correlation_id: str | None = None
    gate_in_flight: int | None = None
    gate_capacity: int | None = None


@dataclass(frozen=True, slots=True)
class ErrorSnapshot:
    """Error totals and breakdown (``errors``)."""

    total: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)
    recent: tuple[ErrorEpisode, ...] = ()


@dataclass(frozen=True, slots=True)
class ClientSnapshot:
    """Per-client activity (``clients[]``)."""

    client: str
    user_agent: str
    requests: int = 0
    failed: int = 0
    aborted: int = 0
    avg_queue_wait_ms: float = 0.0
    avg_duration_ms: float = 0.0
    last_seen_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class UpstreamSnapshot:
    """Upstream provider status (``upstream``)."""

    base_url: str
    retries_total: int = 0
    last_error: str | None = None
    models: tuple[ModelInfo, ...] = ()
    models_checked_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class BindAddress:
    host: str
    port: int


@dataclass(frozen=True, slots=True)
class SessionInfo:
    """Session identity (``session``)."""

    started_at: datetime
    uptime_s: float
    package_version: str
    python_version: str
    bind: BindAddress


@dataclass(frozen=True, slots=True)
class DashboardDocument:
    """Top level of ``GET /dashboard`` (FR-019)."""

    generated_at: datetime
    session: SessionInfo
    subscription: SubscriptionInfo
    gate: GateSnapshot
    throughput: ThroughputSnapshot
    latency: LatencySnapshot
    errors: ErrorSnapshot
    upstream: UpstreamSnapshot
    clients: tuple[ClientSnapshot, ...] = ()
    recent_requests: tuple[RequestRecord, ...] = ()
