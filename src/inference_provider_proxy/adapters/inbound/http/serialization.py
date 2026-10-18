"""Serialize the dashboard document to its pinned JSON shape (FR-023).

Keys here are exactly the ones the JSON Schema declares — it uses
``additionalProperties: false``, so anything extra is a contract violation.
"""

from __future__ import annotations

from typing import Any

from inference_provider_proxy.domain.metrics import (
    ClientSnapshot,
    DashboardDocument,
    ErrorEpisode,
    GateSnapshot,
    InFlightRequest,
    LatencyDistribution,
    RequestRecord,
    UpstreamSnapshot,
    WaitingRequest,
)
from inference_provider_proxy.domain.timestamps import iso_z as _iso


def _latency(distribution: LatencyDistribution) -> dict[str, int]:
    return {
        "p50": distribution.p50,
        "p90": distribution.p90,
        "p95": distribution.p95,
        "p99": distribution.p99,
        "max": distribution.max,
    }


def _in_flight(item: InFlightRequest) -> dict[str, Any]:
    return {
        "request_id": item.request_id,
        "client": item.client,
        "model": item.model,
        "streaming": item.streaming,
        "started_at": _iso(item.started_at),
        "elapsed_s": item.elapsed_s,
        "ttfb_ms": item.ttfb_ms,
    }


def _waiting(item: WaitingRequest) -> dict[str, Any]:
    return {
        "request_id": item.request_id,
        "client": item.client,
        "model": item.model,
        "enqueued_at": _iso(item.enqueued_at),
        "waited_s": item.waited_s,
        "position": item.position,
    }


def _gate(gate: GateSnapshot) -> dict[str, Any]:
    return {
        "capacity": gate.capacity,
        "in_flight": gate.in_flight,
        "queued": gate.queued,
        "available_slots": gate.available_slots,
        "admitted_total": gate.admitted_total,
        "rejected_full_total": gate.rejected_full_total,
        "rejected_timeout_total": gate.rejected_timeout_total,
        "cancelled_while_waiting_total": gate.cancelled_while_waiting_total,
        "oldest_queue_wait_s": gate.oldest_queue_wait_s,
        "current": [_in_flight(item) for item in gate.current],
        "waiting": [_waiting(item) for item in gate.waiting],
    }


def _error_episode(item: ErrorEpisode) -> dict[str, Any]:
    return {
        "at": _iso(item.at),
        "kind": item.kind.value,
        "status": item.status,
        "model": item.model,
        "client": item.client,
        "provider_correlation_id": item.provider_correlation_id,
        "gate_in_flight": item.gate_in_flight,
        "gate_capacity": item.gate_capacity,
    }


def _client(item: ClientSnapshot) -> dict[str, Any]:
    return {
        "client": item.client,
        "user_agent": item.user_agent,
        "requests": item.requests,
        "failed": item.failed,
        "aborted": item.aborted,
        "avg_queue_wait_ms": item.avg_queue_wait_ms,
        "avg_duration_ms": item.avg_duration_ms,
        "last_seen_at": _iso(item.last_seen_at) or "",
    }


def _request(item: RequestRecord) -> dict[str, Any]:
    return {
        "request_id": item.request_id,
        "at": _iso(item.at),
        "client": item.client,
        "model": item.model,
        "streaming": item.streaming,
        "status": item.status,
        "queue_wait_ms": item.queue_wait_ms,
        "queue_position": item.queue_position,
        "ttfb_ms": item.ttfb_ms,
        "duration_ms": item.duration_ms,
        "prompt_tokens": item.prompt_tokens,
        "completion_tokens": item.completion_tokens,
        "aborted": item.aborted,
        "provider_id": item.provider_id,
        "provider_correlation_id": item.provider_correlation_id,
    }


def _upstream(upstream: UpstreamSnapshot) -> dict[str, Any]:
    return {
        "base_url": upstream.base_url,
        "retries_total": upstream.retries_total,
        "last_error": upstream.last_error,
        "models": [
            {"id": model.id, "context_length": model.context_length} for model in upstream.models
        ],
        "models_checked_at": _iso(upstream.models_checked_at),
    }


def document_to_dict(document: DashboardDocument) -> dict[str, Any]:
    """Render the document exactly as ``dashboard.schema.json`` requires."""
    return {
        "generated_at": _iso(document.generated_at),
        "session": {
            "started_at": _iso(document.session.started_at),
            "uptime_s": document.session.uptime_s,
            "package_version": document.session.package_version,
            "python_version": document.session.python_version,
            "bind": {
                "host": document.session.bind.host,
                "port": document.session.bind.port,
            },
        },
        "subscription": {
            "pool": document.subscription.pool,
            "status": document.subscription.status,
            "expires_at": _iso(document.subscription.expires_at),
            "seconds_until_expiry": document.subscription.seconds_until_expiry,
            "key_name": document.subscription.key_name,
            "last_checked_at": _iso(document.subscription.last_checked_at),
            "stale": document.subscription.stale,
        },
        "gate": _gate(document.gate),
        "throughput": {
            "requests_total": document.throughput.requests_total,
            "requests_completed": document.throughput.requests_completed,
            "requests_failed": document.throughput.requests_failed,
            "requests_aborted": document.throughput.requests_aborted,
            "prompt_tokens_total": document.throughput.prompt_tokens_total,
            "completion_tokens_total": document.throughput.completion_tokens_total,
            "tokens_per_second_recent": document.throughput.tokens_per_second_recent,
            "requests_per_minute_recent": document.throughput.requests_per_minute_recent,
        },
        "latency": {
            "queue_wait_ms": _latency(document.latency.queue_wait_ms),
            "time_to_first_byte_ms": _latency(document.latency.time_to_first_byte_ms),
            "total_duration_ms": _latency(document.latency.total_duration_ms),
        },
        "errors": {
            "total": document.errors.total,
            "by_kind": dict(document.errors.by_kind),
            "recent": [_error_episode(item) for item in document.errors.recent],
        },
        "upstream": _upstream(document.upstream),
        "clients": [_client(item) for item in document.clients],
        "recent_requests": [_request(item) for item in document.recent_requests],
    }
