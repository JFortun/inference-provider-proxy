"""In-memory metrics aggregation.

Ring buffers hold the recent samples used for latency quantiles; cumulative
counters track the whole process session; every error category is present and
zero-filled so the dashboard document is always complete.
Also holds subscription and upstream state so the dashboard has one source.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime

from inference_provider_proxy.application.ports.clock import ClockPort
from inference_provider_proxy.domain.errors import ErrorKind
from inference_provider_proxy.domain.metrics import (
    ClientSnapshot,
    ErrorEpisode,
    ErrorSnapshot,
    LatencyDistribution,
    LatencySnapshot,
    RequestRecord,
    ThroughputSnapshot,
    UpstreamSnapshot,
)
from inference_provider_proxy.domain.models import ClientIdentity, ModelInfo, SubscriptionInfo

_RECENT_WINDOW_S = 60.0
_MAX_RECENT_ERRORS = 200


def _quantile(sorted_values: list[int], q: float) -> int:
    if not sorted_values:
        return 0
    if len(sorted_values) == 1:
        return int(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return int(sorted_values[low])
    value = sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (pos - low)
    return int(value)


@dataclass(slots=True)
class _ClientAgg:
    user_agent: str = "unknown"
    requests: int = 0
    failed: int = 0
    aborted: int = 0
    queue_wait_sum_ms: int = 0
    duration_sum_ms: int = 0
    last_seen_at: datetime | None = None


class MemoryMetrics:
    """Implements ``MetricsPort`` and ``SubscriptionPort`` over in-memory state."""

    def __init__(self, *, base_url: str, retention: int, clock: ClockPort) -> None:
        self._clock = clock
        self._base_url = base_url
        self._queue_waits: deque[int] = deque(maxlen=retention)
        self._ttfbs: deque[int] = deque(maxlen=retention)
        self._durations: deque[int] = deque(maxlen=retention)
        self._recent: deque[RequestRecord] = deque(maxlen=retention)
        self._recent_errors: deque[ErrorEpisode] = deque(maxlen=_MAX_RECENT_ERRORS)
        self._by_kind: dict[str, int] = {kind.value: 0 for kind in ErrorKind}
        self._completed = 0
        self._failed = 0
        self._aborted = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._retries = 0
        self._clients: dict[str, _ClientAgg] = {}
        self._last_error: str | None = None
        self._models: tuple[ModelInfo, ...] = ()
        self._models_checked_at: datetime | None = None
        self._subscription = SubscriptionInfo()

    # -- MetricsPort ---------------------------------------------------------

    def note_client(self, client: ClientIdentity) -> None:
        agg = self._clients.setdefault(client.address, _ClientAgg())
        if client.user_agent and client.user_agent != "unknown":
            agg.user_agent = client.user_agent

    def record_request(self, record: RequestRecord) -> None:
        if record.aborted:
            self._aborted += 1
        elif record.error_kind is not None:
            self._failed += 1
        else:
            self._completed += 1

        if record.queue_wait_ms:
            self._queue_waits.append(record.queue_wait_ms)
        if record.ttfb_ms is not None:
            self._ttfbs.append(record.ttfb_ms)
        if record.duration_ms:
            self._durations.append(record.duration_ms)
        if record.prompt_tokens:
            self._prompt_tokens += record.prompt_tokens
        if record.completion_tokens:
            self._completion_tokens += record.completion_tokens

        self._recent.append(record)

        agg = self._clients.setdefault(record.client, _ClientAgg())
        agg.requests += 1
        if record.aborted:
            agg.aborted += 1
        elif record.error_kind is not None:
            agg.failed += 1
        agg.queue_wait_sum_ms += record.queue_wait_ms
        agg.duration_sum_ms += record.duration_ms
        agg.last_seen_at = record.at

    def record_error(self, episode: ErrorEpisode) -> None:
        self._by_kind[episode.kind.value] = self._by_kind.get(episode.kind.value, 0) + 1
        self._recent_errors.append(episode)

    def record_retry(self) -> None:
        self._retries += 1

    def set_models(self, models: list[ModelInfo], *, checked_at: datetime) -> None:
        self._models = tuple(models)
        self._models_checked_at = checked_at

    def set_upstream_error(self, message: str | None) -> None:
        self._last_error = message

    def throughput(self) -> ThroughputSnapshot:
        recent = self._recent_window()
        tokens = sum((r.prompt_tokens or 0) + (r.completion_tokens or 0) for r in recent)
        return ThroughputSnapshot(
            requests_total=self._completed + self._failed + self._aborted,
            requests_completed=self._completed,
            requests_failed=self._failed,
            requests_aborted=self._aborted,
            prompt_tokens_total=self._prompt_tokens,
            completion_tokens_total=self._completion_tokens,
            tokens_per_second_recent=round(tokens / _RECENT_WINDOW_S, 3),
            requests_per_minute_recent=float(len(recent)),
        )

    def latency(self) -> LatencySnapshot:
        return LatencySnapshot(
            queue_wait_ms=self._distribution(self._queue_waits),
            time_to_first_byte_ms=self._distribution(self._ttfbs),
            total_duration_ms=self._distribution(self._durations),
        )

    def errors(self) -> ErrorSnapshot:
        return ErrorSnapshot(
            total=sum(self._by_kind.values()),
            by_kind=dict(self._by_kind),
            recent=tuple(reversed(self._recent_errors)),
        )

    def clients(self) -> list[ClientSnapshot]:
        rows: list[ClientSnapshot] = []
        for address, agg in self._clients.items():
            requests = agg.requests or 1
            rows.append(
                ClientSnapshot(
                    client=address,
                    user_agent=agg.user_agent,
                    requests=agg.requests,
                    failed=agg.failed,
                    aborted=agg.aborted,
                    avg_queue_wait_ms=round(agg.queue_wait_sum_ms / requests, 2),
                    avg_duration_ms=round(agg.duration_sum_ms / requests, 2),
                    last_seen_at=agg.last_seen_at,
                )
            )
        rows.sort(
            key=lambda row: row.last_seen_at.timestamp() if row.last_seen_at else 0.0, reverse=True
        )
        return rows

    def recent_requests(self) -> list[RequestRecord]:
        return list(reversed(self._recent))

    def upstream(self) -> UpstreamSnapshot:
        return UpstreamSnapshot(
            base_url=self._base_url,
            retries_total=self._retries,
            last_error=self._last_error,
            models=self._models,
            models_checked_at=self._models_checked_at,
        )

    # -- SubscriptionPort ----------------------------------------------------

    def current(self) -> SubscriptionInfo:
        return self._subscription

    def update(self, info: SubscriptionInfo) -> None:
        self._subscription = info

    def mark_stale(self, stale: bool = True) -> None:
        self._subscription = replace(self._subscription, stale=stale)

    # -- internals -----------------------------------------------------------

    def _recent_window(self) -> list[RequestRecord]:
        now = self._clock.now()
        return [r for r in self._recent if (now - r.at).total_seconds() <= _RECENT_WINDOW_S]

    @staticmethod
    def _distribution(values: deque[int]) -> LatencyDistribution:
        if not values:
            return LatencyDistribution()
        ordered = sorted(values)
        return LatencyDistribution(
            p50=_quantile(ordered, 0.50),
            p90=_quantile(ordered, 0.90),
            p95=_quantile(ordered, 0.95),
            p99=_quantile(ordered, 0.99),
            max=int(ordered[-1]),
        )
