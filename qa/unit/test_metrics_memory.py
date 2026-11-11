"""Metric aggregation helpers behind the dashboard."""

from __future__ import annotations

from inference_provider_proxy.adapters.outbound.metrics.memory import MemoryMetrics, _quantile
from inference_provider_proxy.domain.errors import ErrorKind
from inference_provider_proxy.domain.metrics import RequestRecord
from inference_provider_proxy.domain.models import SubscriptionInfo
from qa.fakes import FakeClock


def _record(
    clock: FakeClock,
    request_id: str,
    status: int,
    *,
    aborted: bool = False,
    kind: ErrorKind | None = None,
) -> RequestRecord:
    return RequestRecord(
        request_id=request_id,
        at=clock.now(),
        client="10.0.0.1",
        model="m",
        streaming=False,
        status=status,
        queue_wait_ms=0,
        queue_position=None,
        duration_ms=10,
        aborted=aborted,
        error_kind=kind,
    )


def test_quantile_when_no_samples_then_is_zero() -> None:
    samples: list[float] = []

    value = _quantile(samples, 0.5)

    assert value == 0


def test_client_counters_when_requests_are_recorded_then_reconcile_with_throughput() -> None:
    clock = FakeClock()
    metrics = MemoryMetrics(base_url="https://provider.test", retention=10, clock=clock)

    metrics.record_request(_record(clock, "ok", 200))
    metrics.record_request(_record(clock, "bad", 404, kind=ErrorKind.UNKNOWN))
    metrics.record_request(_record(clock, "gone", 499, aborted=True, kind=ErrorKind.CLIENT_ABORTED))
    row = next(iter(metrics.clients()))
    throughput = metrics.throughput()

    assert (row.requests, row.failed, row.aborted) == (3, 1, 1)
    assert row.requests == throughput.requests_total
    assert row.failed == throughput.requests_failed
    assert row.aborted == throughput.requests_aborted


def test_mark_stale_when_called_then_flips_only_the_staleness_flag() -> None:
    clock = FakeClock()
    metrics = MemoryMetrics(base_url="https://provider.test", retention=10, clock=clock)
    metrics.update(SubscriptionInfo(pool="core", status="active", stale=False))

    metrics.mark_stale()

    current = metrics.current()
    assert current.stale is True
    assert current.pool == "core"
    assert current.status == "active"
