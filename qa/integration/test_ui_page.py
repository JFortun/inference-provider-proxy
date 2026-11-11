"""The HTML UI: one shell with a sidebar, one route per section, no script."""

from __future__ import annotations

import httpx
import pytest

from inference_provider_proxy.adapters.inbound.http.ui.render import render_page
from qa.integration.conftest import build_harness

ROUTES = {
    "dashboard": "/",
    "live": "/live",
    "requests": "/requests",
    "errors": "/errors",
    "clients": "/clients",
    "upstream": "/upstream",
}

CONTAINS = {
    "/": ("<h2>Gate</h2>", "<h2>Latency</h2>", "provider busy 0 · ok"),
    "/live": (
        "<h2>Concurrency</h2>",
        "Holding the key",
        "slot free",
        "queue empty",
        "<tbody></tbody>",
    ),
    "/requests": (
        "<h2>Recent requests</h2>",
        "no requests yet",
        "<tbody></tbody>",
        '<p class="note"></p>',
    ),
    "/errors": (
        "<h2>Errors by kind</h2>",
        "no error episodes",
        "expected 0",
        "<tbody></tbody>",
        '<p class="note"></p>',
    ),
    "/clients": ("<h2>Clients</h2>", "no clients yet", "<tbody></tbody>"),
    "/upstream": ("<h2>Upstream</h2>", "retries", '<div class="chips">'),
}
ABSENT = {
    "/": ("<h2>Recent requests</h2>", "<h2>Clients</h2>", "Holding the key"),
    "/live": ("<h2>Latency</h2>", "<h2>Recent requests</h2>", "grid2"),
    "/requests": ("<h2>Clients</h2>", "<h2>Errors by kind</h2>"),
    "/errors": ("<h2>Recent requests</h2>", "<h2>Clients</h2>"),
    "/clients": ("<h2>Upstream</h2>", "<h2>Recent requests</h2>"),
    "/upstream": ("<h2>Clients</h2>", "<h2>Recent requests</h2>"),
}


def _handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(404, json={"error": "not found"})


async def test_index_when_fetched_then_renders_the_zero_traffic_overview(
    asgi_client_factory,
) -> None:
    harness = build_harness(_handler)
    client = asgi_client_factory(harness.app)

    response = await client.get("/")
    body = response.text

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert 'http-equiv="refresh"' in body and 'content="5"' in body
    assert 'rel="stylesheet"' in body and 'href="/styles.css"' in body
    assert 'href="/dashboard"' in body
    assert "<script" not in body.lower()
    assert "$" not in body
    assert "generated" in body and "uptime" in body
    assert "null" not in body
    assert ">None<" not in body
    assert 'href="/" aria-current="page"' in body


@pytest.mark.parametrize("page,route", list(ROUTES.items()))
async def test_section_page_when_visited_then_is_isolated_and_marks_itself_active(
    asgi_client_factory, page: str, route: str
) -> None:
    harness = build_harness(_handler)
    client = asgi_client_factory(harness.app)

    response = await client.get(route)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    body = response.text
    assert "<script" not in body.lower()
    assert "$" not in body
    assert "null" not in body
    assert ">None<" not in body
    assert 'http-equiv="refresh"' in body
    assert '<nav class="sidebar" aria-label="Sections">' in body
    assert body.count('aria-current="page"') == 1
    assert f'href="{route}" aria-current="page"' in body
    for other in ROUTES.values():
        assert f'href="{other}"' in body
    for marker in CONTAINS[route]:
        assert marker in body, f"{route} lost {marker!r}"
    for marker in ABSENT[route]:
        assert marker not in body, f"{route} leaked {marker!r}"


async def test_styles_css_when_served_then_carries_the_layout_rules(
    asgi_client_factory,
) -> None:
    harness = build_harness(_handler)
    client = asgi_client_factory(harness.app)

    response = await client.get("/styles.css")
    text = response.text

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/css")
    assert ".pill--bad" in text
    assert "table:has(tbody:empty)" in text
    assert "+ .empty" in text
    assert "max-height" in text
    assert "position: sticky" in text
    assert ".note:empty" in text
    assert "scrollbar-color" in text
    assert "::-webkit-scrollbar" in text
    assert ".sidebar" in text
    assert 'a[aria-current="page"]' in text
    assert ".nav-burger" in text
    assert ".nav-toggle" in text
    assert "@media (max-width: 56.25em)" in text
    assert ".nav-toggle:checked ~ .sidebar" in text
    assert "position: fixed" in text
    assert "margin: auto" not in text
    assert "font: 0.8125rem/1.5" in text
    assert "13px" not in text
    assert "grid-template-columns: 1fr auto 1fr" in text
    assert "justify-self: center" in text
    assert ".grid2" not in text


async def test_populated_document_when_rendered_then_escapes_every_page(
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

    body = "".join(render_page(page, document) for page in ROUTES)

    assert "<script" not in body
    assert "&lt;script&gt;" in body
    assert "&lt;b&gt;model&lt;/b&gt;" in body
    assert "$" not in body
    assert "pill--bad" in body
    assert 'title="Provider 429s of requests admitted within capacity' in body
    assert "2m 05s" in body
    assert "4.2 s" in body
    assert "stream" in body and "aborted" in body
    assert "req-2" in body
    assert "<script>alert(1)</script>" not in body
    assert "upstream_5xx" in body
    assert ">1/2<" in body
    assert ">3/1<" in body
    assert "in-flight / capacity" in body
    assert "curl/8.5" in body
    assert '<th class="share">Share</th>' in body
    assert '<td class="share"><span class="bar"><i style="width: 100%">' in body
    assert "128k" in body
    assert 'class="card card--fill"' in body
    assert '<p class="note"></p>' in body
    assert "showing" not in body


def _request_row_source(i: int) -> dict[str, object]:
    return {
        "request_id": f"r{i}",
        "at": "2026-01-01T00:00:00Z",
        "client": f"host-{i}",
        "model": "m",
        "streaming": False,
        "status": 200,
        "queue_wait_ms": 1,
        "queue_position": None,
        "ttfb_ms": 1,
        "duration_ms": 1,
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "aborted": False,
        "provider_id": None,
        "provider_correlation_id": None,
    }


def _episode_row_source(i: int) -> dict[str, object]:
    return {
        "at": "2026-01-01T00:00:00Z",
        "kind": "upstream_5xx",
        "status": 502,
        "model": "m",
        "client": f"ep-{i}",
        "provider_correlation_id": None,
        "gate_in_flight": 1,
        "gate_capacity": 2,
    }


async def test_tables_when_over_the_cap_then_render_100_rows_newest_first(
    asgi_client_factory,
) -> None:
    harness = build_harness(_handler)
    client = asgi_client_factory(harness.app)
    document = (await client.get("/dashboard")).json()
    document["recent_requests"] = [_request_row_source(i) for i in range(150)]
    document["errors"]["recent"] = [_episode_row_source(i) for i in range(120)]

    body = render_page("requests", document) + render_page("errors", document)

    assert "host-0" in body and "host-99" in body
    assert "host-100" not in body and "host-149" not in body
    assert "ep-99" in body and "ep-119" not in body
    assert body.count("showing 100 of 150") == 1
    assert body.count("showing 100 of 120") == 1


async def test_styles_css_when_rendered_then_stretches_cards_to_their_row(
    asgi_client_factory,
) -> None:
    harness = build_harness(_handler)
    client = asgi_client_factory(harness.app)

    response = await client.get("/styles.css")

    assert response.status_code == 200
    text = response.text
    assert "align-items: start" not in text
    assert "contain: size" in text
    assert "min-height: 27rem" in text
    assert ".card--fill .tablewrap" in text
