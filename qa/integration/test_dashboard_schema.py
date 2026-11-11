"""The dashboard document matches its pinned schema."""

from __future__ import annotations

import json
import time
from importlib.resources import files

import httpx
import jsonschema

from inference_provider_proxy.domain.errors import ErrorKind, ProxyError
from inference_provider_proxy.domain.models import ClientIdentity
from qa.integration.conftest import build_harness

SCHEMA = json.loads(
    files("inference_provider_proxy.contracts").joinpath("dashboard.schema.json").read_text()
)

EXPECTED_KINDS = {kind.value for kind in ErrorKind}

_PAYLOAD = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


def _handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/chat/completions"):
        return httpx.Response(
            200,
            json={
                "id": "c1",
                "object": "chat.completion",
                "model": "m",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )
    return httpx.Response(404, json={"error": "not found"})


def _throttled(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/chat/completions"):
        return httpx.Response(
            429,
            headers={"retry-after": "0"},
            json={"error": {"message": "c", "type": "throttling_error", "code": "429"}},
        )
    return httpx.Response(404, json={"error": "not found"})


def _not_found(request: httpx.Request) -> httpx.Response:
    return httpx.Response(404, json={"error": "not found"})


class _FailingCatalog:
    async def models(self):
        raise ProxyError(
            kind=ErrorKind.UPSTREAM_UNREACHABLE,
            http_status=502,
            code="upstream_unreachable",
            message="provider is down",
            retryable=False,
        )


async def test_zero_traffic_document_when_fetched_then_is_complete_and_schema_valid(
    asgi_client_factory,
) -> None:
    harness = build_harness(_handler)
    client = asgi_client_factory(harness.app)

    response = await client.get("/dashboard")
    document = response.json()

    assert response.status_code == 200
    jsonschema.validate(document, SCHEMA)
    assert document["gate"]["in_flight"] == 0
    assert document["gate"]["queued"] == 0
    assert document["gate"]["current"] == []
    assert document["gate"]["waiting"] == []
    assert document["gate"]["oldest_queue_wait_s"] is None
    assert document["clients"] == []
    assert document["recent_requests"] == []
    assert document["errors"]["total"] == 0
    assert set(document["errors"]["by_kind"]) == EXPECTED_KINDS
    assert document["errors"]["by_kind"]["provider_busy"] == 0
    assert document["subscription"]["stale"] is True
    assert document["throughput"]["requests_total"] == 0


async def test_dashboard_when_key_is_busy_then_returns_within_one_second(
    asgi_client_factory,
) -> None:
    harness = build_harness(_handler)
    client = asgi_client_factory(harness.app)
    slot_cm = harness.gate.slot(
        request_id="holder",
        client=ClientIdentity(address="10.0.0.2"),
        model="m",
        streaming=True,
        max_wait_s=5.0,
    )
    await slot_cm.__aenter__()

    started = time.monotonic()
    response = await client.get("/dashboard")
    elapsed = time.monotonic() - started
    await slot_cm.__aexit__(None, None, None)
    document = response.json()

    assert response.status_code == 200
    assert elapsed < 1.0
    assert document["gate"]["in_flight"] == 1
    assert document["gate"]["current"][0]["request_id"] == "holder"


async def test_dashboard_when_traffic_is_served_then_reflects_it(
    asgi_client_factory,
) -> None:
    harness = build_harness(_handler)
    client = asgi_client_factory(harness.app)

    await client.post("/v1/chat/completions", json=_PAYLOAD)
    document = (await client.get("/dashboard")).json()

    jsonschema.validate(document, SCHEMA)
    assert document["throughput"]["requests_total"] == 1
    assert document["throughput"]["requests_completed"] == 1
    assert len(document["recent_requests"]) == 1
    assert document["recent_requests"][0]["status"] == 200
    assert document["recent_requests"][0]["queue_position"] is None


async def test_error_episode_when_recorded_then_carries_the_gate_admission_state(
    asgi_client_factory,
) -> None:
    harness = build_harness(_throttled, max_retries=0)
    client = asgi_client_factory(harness.app)

    response = await client.post("/v1/chat/completions", json=_PAYLOAD)
    document = (await client.get("/dashboard")).json()
    episode = document["errors"]["recent"][0]

    assert response.status_code == 429
    jsonschema.validate(document, SCHEMA)
    assert episode["kind"] == ErrorKind.PROVIDER_BUSY.value
    assert episode["gate_in_flight"] == 1
    assert episode["gate_capacity"] == 1
    assert episode["gate_in_flight"] <= episode["gate_capacity"]


async def test_route_rejection_before_chat_service_when_raised_then_reaches_the_dashboard(
    asgi_client_factory,
) -> None:
    harness = build_harness(_handler)
    client = asgi_client_factory(harness.app)

    for payload in ("{not json", "[1,2,3]"):
        response = await client.post(
            "/v1/chat/completions",
            content=payload,
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 400
    document = (await client.get("/dashboard")).json()
    jsonschema.validate(document, SCHEMA)

    rows = [row for row in document["recent_requests"] if row["status"] == 400]
    throughput = document["throughput"]
    episode = document["errors"]["recent"][0]
    assert len(rows) == 2
    assert {row["model"] for row in rows} == {""}
    assert throughput["requests_total"] == 2
    assert throughput["requests_failed"] == 2
    assert document["errors"]["by_kind"]["unknown"] == 2
    assert document["errors"]["total"] == 2
    assert episode["model"] is None
    assert (episode["gate_in_flight"], episode["gate_capacity"]) == (None, None)
    assert throughput["requests_total"] == (
        throughput["requests_completed"]
        + throughput["requests_failed"]
        + throughput["requests_aborted"]
    )


async def test_only_chat_posts_when_other_routes_fail_then_are_counted_without_doubling(
    asgi_client_factory,
) -> None:
    harness = build_harness(_not_found)
    client = asgi_client_factory(harness.app)

    chat_404 = await client.post("/v1/chat/completions", json=_PAYLOAD)
    after_chat = (await client.get("/dashboard")).json()
    harness.app.state.services.catalog = _FailingCatalog()
    models_502 = await client.get("/v1/models")
    after_other = (await client.get("/dashboard")).json()
    chat_400 = await client.post(
        "/v1/chat/completions",
        content="{not json",
        headers={"content-type": "application/json"},
    )
    final = (await client.get("/dashboard")).json()

    assert chat_404.status_code == 404
    assert models_502.status_code == 502
    assert chat_400.status_code == 400
    assert after_chat["throughput"]["requests_total"] == 1
    assert after_other["throughput"]["requests_total"] == 1
    assert after_other["errors"]["total"] == 1
    assert final["throughput"]["requests_total"] == 2
    assert final["errors"]["by_kind"]["unknown"] == 2
    assert final["errors"]["total"] == 2
