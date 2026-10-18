"""``GET /`` serves the dashboard as HTML + CSS with no client-side script."""

from __future__ import annotations

import httpx

from inference_provider_proxy.adapters.inbound.http.ui.render import render_dashboard
from qa.integration.conftest import build_harness


def _handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(404, json={"error": "not found"})


async def test_index_renders_the_zero_traffic_page(asgi_client_factory) -> None:
    harness = build_harness(_handler)
    client = asgi_client_factory(harness.app)

    response = await client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    body = response.text
    # Formatter-agnostic: the on-disk HTML is normalized on save.
    assert 'http-equiv="refresh"' in body and 'content="5"' in body
    assert 'rel="stylesheet"' in body and 'href="/styles.css"' in body
    assert 'href="/dashboard"' in body
    assert "<script" not in body.lower()
    # Everything substituted; nothing left as a raw placeholder.
    assert "$" not in body
    # Empty states rather than raw nulls or missing sections.
    assert "slot free" in body
    assert "queue empty" in body
    assert "no requests yet" in body
    assert "no clients yet" in body
    assert "no error episodes" in body
    assert "expected 0" in body
    assert "null" not in body
    assert ">None<" not in body


async def test_styles_css_is_served_separately(asgi_client_factory) -> None:
    harness = build_harness(_handler)
    client = asgi_client_factory(harness.app)

    response = await client.get("/styles.css")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/css")
    assert ".pill--bad" in response.text
    # Empty rows are hidden by CSS so the notes in index.html take over.
    assert "table:has(tbody:empty)" in response.text
    assert "+ .empty" in response.text


async def test_populated_page_renders_every_section_and_escapes(
    asgi_client_factory,
) -> None:
    harness = build_harness(_handler)
    client = asgi_client_factory(harness.app)
    document = (await client.get("/dashboard")).json()

    document["gate"].update(
        {
            "capacity": 2,
            "in_flight": 1,
            "queued": 1,
            "available_slots": 0,
            "oldest_queue_wait_s": 125.0,
            "current": [
                {
                    "request_id": "<script>alert(1)</script>",
                    "client": "host-a",
                    "model": "m",
                    "streaming": True,
                    "started_at": "2026-01-01T00:00:00Z",
                    "elapsed_s": 3.2,
                    "ttfb_ms": None,
                }
            ],
            "waiting": [
                {
                    "request_id": "req-2",
                    "client": "host-b",
                    "model": "m",
                    "enqueued_at": "2026-01-01T00:00:00Z",
                    "waited_s": 8.5,
                    "position": 1,
                }
            ],
        }
    )
    document["errors"]["by_kind"]["provider_busy"] = 2
    document["errors"]["by_kind"]["upstream_5xx"] = 3
    document["errors"]["total"] = 5
    document["errors"]["recent"] = [
        {
            "at": "2026-01-01T00:00:00Z",
            "kind": "upstream_5xx",
            "status": 502,
            "model": "m",
            "client": "host-a",
            "provider_correlation_id": None,
            "gate_in_flight": 1,
            "gate_capacity": 2,
        },
        {
            # The only shape a genuine gate fault would have: over capacity.
            "at": "2026-01-01T00:00:01Z",
            "kind": "provider_busy",
            "status": 429,
            "model": "m",
            "client": "host-b",
            "provider_correlation_id": "corr-9",
            "gate_in_flight": 3,
            "gate_capacity": 1,
        },
    ]
    document["clients"] = [
        {
            "client": "host-a",
            "user_agent": "curl/8.5",
            "requests": 4,
            "failed": 1,
            "aborted": 1,
            "avg_queue_wait_ms": 12.5,
            "avg_duration_ms": 810.0,
            "last_seen_at": "",
        }
    ]
    document["recent_requests"] = [
        {
            "request_id": "r1",
            "at": "2026-01-01T00:00:00Z",
            "client": "host-a",
            "model": "m",
            "streaming": True,
            "status": 200,
            "queue_wait_ms": 12,
            "queue_position": None,
            "ttfb_ms": None,
            "duration_ms": 4200,
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "aborted": True,
            "provider_id": "p",
            "provider_correlation_id": "corr-1",
        },
        {
            "request_id": "r2",
            "at": "2026-01-01T00:00:01Z",
            "client": "host-b",
            "model": "m",
            "streaming": False,
            "status": 429,
            "queue_wait_ms": 300,
            "queue_position": 3,
            "ttfb_ms": 300,
            "duration_ms": 420,
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "aborted": False,
            "provider_id": None,
            "provider_correlation_id": None,
        },
    ]
    document["upstream"]["models"] = [
        {"id": "<b>model</b>", "context_length": 131072},
        {"id": "tiny", "context_length": None},
    ]
    document["upstream"]["last_error"] = "<script>alert(2)</script>"
    document["subscription"].update(
        {"pool": "free", "status": "active", "stale": False, "seconds_until_expiry": 3725}
    )

    body = render_dashboard(document)

    assert "<script" not in body
    assert "&lt;script&gt;" in body
    assert "&lt;b&gt;model&lt;/b&gt;" in body
    assert "$" not in body
    assert "pill--bad" in body  # provider_busy 2 and status 502
    # The busy pill explains itself when non-zero.
    assert 'title="Provider 429s of requests admitted within capacity' in body
    assert "2m 05s" in body  # oldest queue wait, minute branch
    assert "4.2 s" in body  # duration above one second
    assert "stream" in body and "aborted" in body
    # Rows rendered: queue waiter, holder, episodes, requests, clients, models.
    assert "req-2" in body
    assert "<script>alert(1)</script>" not in body  # request_id, escaped
    assert "upstream_5xx" in body  # episode row kind pill
    # Gate column on every episode: within capacity is neutral, over capacity (the
    # only true gate fault) is bad, and the column explains itself on the page.
    assert ">1/2<" in body
    assert ">3/1<" in body
    assert "in-flight / capacity" in body
    assert "curl/8.5" in body
    assert "128k" in body
