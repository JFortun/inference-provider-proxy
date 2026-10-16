"""Driven ports that exist only under ``TYPE_CHECKING`` stay importable (FR-015)."""

from __future__ import annotations

import importlib


def test_metrics_port_module_declares_its_protocols() -> None:
    module = importlib.import_module("inference_provider_proxy.application.ports.metrics")

    assert module.MetricsPort is not None
    assert module.SubscriptionPort is not None
