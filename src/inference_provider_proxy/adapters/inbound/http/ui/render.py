"""Fill the page defined by ``ui/index.html`` with one dashboard document.

Markup and labels live in the HTML file; this module only computes values,
table rows and pills, then substitutes them with ``string.Template``. A missing
placeholder raises, so the template and the pinned dashboard contract cannot
drift apart silently.
"""

from __future__ import annotations

from html import escape
from string import Template
from typing import Any

from inference_provider_proxy.adapters.inbound.http.ui.assets import INDEX_HTML

__all__ = ["render_dashboard"]

_TEMPLATE = Template(INDEX_HTML)
_QUANTILES = ("p50", "p90", "p95", "p99", "max")
_HARD_ERRORS = frozenset(
    {
        "provider_busy",
        "queue_full",
        "queue_timeout",
        "upstream_auth",
        "upstream_5xx",
        "upstream_unreachable",
    }
)


def _e(value: Any) -> str:
    """Escape a document value for markup; null or absent renders as a dash."""
    if value is None or value == "":
        return "—"
    return escape(str(value), quote=True)


def _pct(value: float, maximum: float) -> int:
    if maximum <= 0:
        return 0
    return max(0, min(100, round(value / maximum * 100)))


def _fmt_ms(value: int | float | None) -> str:
    if value is None:
        return "—"
    if value >= 1000:
        return f"{value / 1000:.1f} s"
    return f"{value:.0f} ms"


def _fmt_s(value: float | None) -> str:
    if value is None:
        return "—"
    if value >= 86400:
        return f"{value // 86400:.0f}d {(value % 86400) // 3600:02.0f}h"
    if value >= 3600:
        return f"{value // 3600:.0f}h {(value % 3600) // 60:02.0f}m"
    if value >= 60:
        return f"{value // 60:.0f}m {value % 60:02.0f}s"
    return f"{value:.1f} s"


def _time(value: str | None) -> str:
    if not value or len(value) < 19:
        return "—"
    return value[11:19]


def _pill(text: str, tone: str, *, title: str | None = None) -> str:
    attr = "" if title is None else f' title="{escape(title, quote=True)}"'
    return f'<span class="pill pill--{tone}"{attr}>{text}</span>'


def _status_pill(status: int) -> str:
    tone = "ok" if status < 400 else "warn" if status < 500 else "bad"
    return _pill(str(status), tone)


def _banner(label: str, value: Any) -> str:
    if not value:
        return ""
    return f'<p class="banner">{label}: <span class="mono">{_e(value)}</span></p>'


def _dist_values(dist: dict[str, int]) -> dict[str, str]:
    maximum = dist["max"]
    return {
        "width": str(_pct(dist["p50"], maximum)),
        "tick": str(_pct(dist["p99"], maximum)),
        "stats": " · ".join(f"{name} {_fmt_ms(dist[name])}" for name in _QUANTILES),
    }


def _inflight_row(item: dict[str, Any]) -> str:
    streaming = _pill("stream", "ok") if item["streaming"] else _pill("buffered", "neutral")
    return (
        f'<tr><td class="mono">{_e(item["request_id"])}</td><td>{_e(item["client"])}</td>'
        f"<td>{_e(item['model'])}</td><td>{streaming}</td>"
        f'<td class="num">{_fmt_s(item["elapsed_s"])}</td>'
        f'<td class="num">{_fmt_ms(item["ttfb_ms"])}</td>'
        f'<td class="mono">{_time(item["started_at"])}</td></tr>'
    )


def _waiting_row(item: dict[str, Any], oldest: float) -> str:
    width = _pct(item["waited_s"], oldest)
    return (
        f'<li><span class="pos">{item["position"]}</span>'
        f'<span class="mono">{_e(item["request_id"])}</span>'
        f'<span>{_e(item["client"])}</span><span class="muted">{_e(item["model"])}</span>'
        f'<span class="bar"><i style="width: {width}%"></i></span>'
        f'<span class="num">{_fmt_s(item["waited_s"])}</span></li>'
    )


def _gate_cell(in_flight: int | None, capacity: int | None) -> str:
    """``in_flight/capacity`` at failure time: over capacity (the only real
    gate fault) renders bad, within capacity renders neutral."""
    text = "—" if in_flight is None or capacity is None else f"{in_flight}/{capacity}"
    over = bool(in_flight is not None and capacity is not None and in_flight > capacity)
    return _pill(text, "bad" if over else "neutral")


def _episode_row(item: dict[str, Any]) -> str:
    tone = "bad" if item["kind"] in _HARD_ERRORS else "warn"
    return (
        f'<tr><td class="mono">{_time(item["at"])}</td><td>{_pill(_e(item["kind"]), tone)}</td>'
        f'<td class="num">{_e(item["status"])}</td>'
        f'<td class="num">{_gate_cell(item["gate_in_flight"], item["gate_capacity"])}</td>'
        f"<td>{_e(item['model'])}</td>"
        f'<td>{_e(item["client"])}</td><td class="mono">{_e(item["provider_correlation_id"])}</td>'
        "</tr>"
    )


def _request_row(item: dict[str, Any]) -> str:
    badges = " ".join(
        badge
        for badge in (
            _pill("stream", "ok") if item["streaming"] else "",
            _pill("aborted", "bad") if item["aborted"] else "",
        )
        if badge
    )
    return (
        f'<tr><td class="mono">{_time(item["at"])}</td><td>{_status_pill(item["status"])}</td>'
        f"<td>{_e(item['model'])}</td><td>{_e(item['client'])}</td><td>{badges}</td>"
        f'<td class="num">{_fmt_ms(item["queue_wait_ms"])}</td>'
        f'<td class="num">{_fmt_ms(item["ttfb_ms"])}</td>'
        f'<td class="num">{_fmt_ms(item["duration_ms"])}</td>'
        f'<td class="num">{_e(item["prompt_tokens"])}+{_e(item["completion_tokens"])}</td>'
        "</tr>"
    )


def _client_row(item: dict[str, Any], worst: int) -> str:
    width = _pct(item["requests"], worst)
    return (
        f"<tr><td>{_e(item['client'])}</td>"
        f'<td class="small" title="{_e(item["user_agent"])}">{_e(item["user_agent"])}</td>'
        f'<td class="num">{item["requests"]}</td>'
        f'<td class="bar"><i style="width: {width}%"></i></td>'
        f'<td class="num">{item["failed"]}</td><td class="num">{item["aborted"]}</td>'
        f'<td class="num">{_fmt_ms(item["avg_queue_wait_ms"])}</td>'
        f'<td class="num">{_fmt_ms(item["avg_duration_ms"])}</td>'
        f'<td class="mono">{_time(item["last_seen_at"])}</td></tr>'
    )


def _model_chip(model: dict[str, Any]) -> str:
    context = model["context_length"]
    suffix = f'<span class="ctx">{context // 1024}k</span>' if context else ""
    return f'<span class="chip">{_e(model["id"])}{suffix}</span>'


def render_dashboard(document: dict[str, Any]) -> str:
    """Fill ``index.html`` for one dashboard document."""
    session = document["session"]
    subscription = document["subscription"]
    gate = document["gate"]
    throughput = document["throughput"]
    latency = document["latency"]
    errors = document["errors"]
    upstream = document["upstream"]
    ttfb = latency["time_to_first_byte_ms"]

    capacity = gate["capacity"]
    in_flight = gate["in_flight"]
    waiting = gate["waiting"]
    clients = document["clients"]
    models = upstream["models"]

    busy = errors["by_kind"]["provider_busy"]
    worst_kind = max(errors["by_kind"].values())
    oldest = gate["oldest_queue_wait_s"] or max((item["waited_s"] for item in waiting), default=0.0)

    values: dict[str, str] = {
        "generated": _time(document["generated_at"]),
        "uptime": _fmt_s(session["uptime_s"]),
        "version": _e(session["package_version"]),
        "python": _e(session["python_version"]),
        "bind": f"{_e(session['bind']['host'])}:{session['bind']['port']}",
        "in_flight": str(in_flight),
        "capacity": str(capacity),
        "meter_cells": (
            "".join('<i class="on"></i>' if i < in_flight else "<i></i>" for i in range(capacity))
            if capacity <= 64
            else ""
        ),
        "queued": str(gate["queued"]),
        "free": str(gate["available_slots"]),
        "rpm": f"{throughput['requests_per_minute_recent']:.0f}",
        "tok_per_s": f"{throughput['tokens_per_second_recent']:.1f}",
        "requests_total": f"{throughput['requests_total']:,}",
        "failed": str(throughput["requests_failed"]),
        "aborted": str(throughput["requests_aborted"]),
        "ttfb_p50": _fmt_ms(ttfb["p50"]),
        "ttfb_p95": _fmt_ms(ttfb["p95"]),
        "ttfb_p99": _fmt_ms(ttfb["p99"]),
        "prompt_tokens": f"{throughput['prompt_tokens_total']:,}",
        "completion_tokens": f"{throughput['completion_tokens_total']:,}",
        "errors_total": str(errors["total"]),
        "busy_pill": (
            _pill("provider busy 0 · ok", "ok")
            if busy == 0
            else _pill(
                f"provider busy {busy}",
                "bad",
                title="Provider 429s of requests admitted within capacity, "
                "not gate faults; the episodes table shows each in-flight/capacity.",
            )
        ),
        "retries": str(upstream["retries_total"]),
        "sub_status": _e(subscription["status"]),
        "sub_pool": _e(subscription["pool"]),
        "expires": _fmt_s(subscription["seconds_until_expiry"]),
        "stale_pill": _pill("stale", "warn") if subscription["stale"] else _pill("fresh", "ok"),
        "last_checked": _time(subscription["last_checked_at"]),
        "key_name": _e(subscription["key_name"]),
        "admitted": f"{gate['admitted_total']:,}",
        "rejected_full": str(gate["rejected_full_total"]),
        "rejected_timeout": str(gate["rejected_timeout_total"]),
        "cancelled_waiting": str(gate["cancelled_while_waiting_total"]),
        "oldest_wait": _fmt_s(gate["oldest_queue_wait_s"]),
        "current_rows": "".join(_inflight_row(item) for item in gate["current"]),
        "queue_rows": "".join(_waiting_row(item, oldest) for item in waiting),
        "episode_rows": "".join(_episode_row(item) for item in errors["recent"]),
        "request_rows": "".join(_request_row(item) for item in document["recent_requests"]),
        "client_rows": "".join(
            _client_row(item, max((c["requests"] for c in clients), default=0)) for item in clients
        ),
        "base_url": _e(upstream["base_url"]),
        "last_error_banner": _banner("last error", upstream["last_error"]),
        "model_count": str(len(models)),
        "models_checked": _time(upstream["models_checked_at"]),
        "model_chips": "".join(_model_chip(model) for model in models),
    }
    for prefix, dist in (
        ("qw", latency["queue_wait_ms"]),
        ("ttfb", ttfb),
        ("td", latency["total_duration_ms"]),
    ):
        for part, value in _dist_values(dist).items():
            values[f"{prefix}_{part}"] = value
    for kind, count in errors["by_kind"].items():
        values[f"count_{kind}"] = str(count)
        values[f"width_{kind}"] = str(_pct(count, worst_kind))

    return _TEMPLATE.substitute(values)
