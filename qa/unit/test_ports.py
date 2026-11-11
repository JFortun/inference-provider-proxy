"""Driven ports that exist only under ``TYPE_CHECKING`` stay importable."""

from __future__ import annotations

import importlib


def test_metrics_port_module_when_imported_then_declares_its_protocols() -> None:
    module = importlib.import_module("inference_provider_proxy.application.ports.metrics")

    metrics = module.MetricsPort
    subscription = module.SubscriptionPort

    assert metrics is not None
    assert subscription is not None
