"""Metric aggregation helpers behind the dashboard (FR-019, FR-020)."""

from __future__ import annotations

from inference_provider_proxy.adapters.outbound.metrics.memory import MemoryMetrics, _quantile
from inference_provider_proxy.domain.models import SubscriptionInfo
from qa.fakes import FakeClock


def test_quantile_of_no_samples_is_zero() -> None:
    assert _quantile([], 0.5) == 0


def test_mark_stale_flips_only_the_staleness_flag() -> None:
    clock = FakeClock()
    metrics = MemoryMetrics(base_url="https://provider.test", retention=10, clock=clock)
    metrics.update(SubscriptionInfo(pool="core", status="active", stale=False))

    metrics.mark_stale()

    current = metrics.current()
    assert current.stale is True
    assert current.pool == "core"
    assert current.status == "active"
