"""Metric aggregation helpers behind the dashboard (FR-019, FR-020)."""

from __future__ import annotations

from inference_provider_proxy.adapters.outbound.metrics.memory import MemoryMetrics, _quantile
from inference_provider_proxy.domain.errors import ErrorKind
from inference_provider_proxy.domain.metrics import RequestRecord
from inference_provider_proxy.domain.models import SubscriptionInfo
from qa.fakes import FakeClock


def test_quantile_of_no_samples_is_zero() -> None:
    assert _quantile([], 0.5) == 0


def test_client_counters_reconcile_with_throughput() -> None:
    """An abort is one label in both sections: aborted, never failed as well (F4)."""
    clock = FakeClock()
    metrics = MemoryMetrics(base_url="https://provider.test", retention=10, clock=clock)

    def record(
        request_id: str, status: int, *, aborted: bool = False, kind: ErrorKind | None = None
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

    metrics.record_request(record("ok", 200))
    metrics.record_request(record("bad", 404, kind=ErrorKind.UNKNOWN))
    metrics.record_request(record("gone", 499, aborted=True, kind=ErrorKind.CLIENT_ABORTED))

    throughput = metrics.throughput()
    (row,) = metrics.clients()
    assert (row.requests, row.failed, row.aborted) == (3, 1, 1)
    assert row.requests == throughput.requests_total
    assert row.failed == throughput.requests_failed
    assert row.aborted == throughput.requests_aborted


def test_mark_stale_flips_only_the_staleness_flag() -> None:
    clock = FakeClock()
    metrics = MemoryMetrics(base_url="https://provider.test", retention=10, clock=clock)
    metrics.update(SubscriptionInfo(pool="core", status="active", stale=False))

    metrics.mark_stale()

    current = metrics.current()
    assert current.stale is True
    assert current.pool == "core"
    assert current.status == "active"
