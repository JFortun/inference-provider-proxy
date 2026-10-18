"""The dashboard document matches its pinned schema (US3 / T034)."""

from __future__ import annotations

import json
import time
from importlib.resources import files

import httpx
import jsonschema

from inference_provider_proxy.domain.errors import ErrorKind, ProxyError
from inference_provider_proxy.domain.models import ClientIdentity
from qa.integration.conftest import build_harness

# The pinned contract ships as package data, so a clean checkout — or a wheel
# install — always has it (specs/ is git-ignored scaffolding, not a home for a
# contract the tests must read).
SCHEMA = json.loads(
    files("inference_provider_proxy.contracts").joinpath("dashboard.schema.json").read_text()
)

EXPECTED_KINDS = {kind.value for kind in ErrorKind}


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


async def test_zero_traffic_document_is_complete_and_schema_valid(asgi_client_factory) -> None:
    harness = build_harness(_handler)
    client = asgi_client_factory(harness.app)

    response = await client.get("/dashboard")
    assert response.status_code == 200
    document = response.json()

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


async def test_dashboard_returns_within_one_second_while_the_key_is_busy(
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
    try:
        started = time.monotonic()
        response = await client.get("/dashboard")
        elapsed = time.monotonic() - started
    finally:
        await slot_cm.__aexit__(None, None, None)

    assert response.status_code == 200
    assert elapsed < 1.0
    document = response.json()
    assert document["gate"]["in_flight"] == 1
    assert document["gate"]["current"][0]["request_id"] == "holder"


async def test_dashboard_reflects_served_traffic(asgi_client_factory) -> None:
    harness = build_harness(_handler)
    client = asgi_client_factory(harness.app)

    await client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    document = (await client.get("/dashboard")).json()

    jsonschema.validate(document, SCHEMA)
    assert document["throughput"]["requests_total"] == 1
    assert document["throughput"]["requests_completed"] == 1
    assert len(document["recent_requests"]) == 1
    assert document["recent_requests"][0]["status"] == 200
    # Admitted immediately, so there is no wait position to report (SC-008, FR-013).
    assert document["recent_requests"][0]["queue_position"] is None


async def test_error_episodes_carry_the_gate_admission_state(asgi_client_factory) -> None:
    """An episode records the gate's in-flight/capacity, so `provider_busy` reads
    as "within capacity, not a gate fault" (F1)."""

    def throttled(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(
                429,
                headers={"retry-after": "0"},
                json={"error": {"message": "c", "type": "throttling_error", "code": "429"}},
            )
        return httpx.Response(404, json={"error": "not found"})

    harness = build_harness(throttled, max_retries=0)
    client = asgi_client_factory(harness.app)

    response = await client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 429

    document = (await client.get("/dashboard")).json()
    # Episodes are part of the pinned contract, gate state included.
    jsonschema.validate(document, SCHEMA)

    episode = document["errors"]["recent"][0]
    assert episode["kind"] == ErrorKind.PROVIDER_BUSY.value
    assert episode["gate_in_flight"] == 1
    assert episode["gate_capacity"] == 1
    assert episode["gate_in_flight"] <= episode["gate_capacity"]


async def test_route_rejections_before_the_chat_service_reach_the_dashboard(
    asgi_client_factory,
) -> None:
    """Body rejections are logged and answered, so they must also be counted (F3)."""
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
    assert len(rows) == 2  # both rejections, matching the access log
    assert {row["model"] for row in rows} == {""}  # no ChatRequest existed
    throughput = document["throughput"]
    assert throughput["requests_total"] == 2
    assert throughput["requests_failed"] == 2
    assert document["errors"]["by_kind"]["unknown"] == 2
    assert document["errors"]["total"] == 2
    episode = document["errors"]["recent"][0]
    assert episode["model"] is None
    assert (episode["gate_in_flight"], episode["gate_capacity"]) == (None, None)
    assert throughput["requests_total"] == (
        throughput["requests_completed"]
        + throughput["requests_failed"]
        + throughput["requests_aborted"]
    )


async def test_only_never_entered_chat_posts_are_recorded(asgi_client_factory) -> None:
    """No double count for chat-path failures; other routes stay out of chat totals."""
    harness = build_harness(lambda request: httpx.Response(404, json={"error": "not found"}))
    client = asgi_client_factory(harness.app)

    response = await client.post(
        "/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert response.status_code == 404  # passthrough, recorded by the chat service
    document = (await client.get("/dashboard")).json()
    assert document["throughput"]["requests_total"] == 1  # once, not twice
    assert document["errors"]["by_kind"]["unknown"] == 1
    assert document["errors"]["total"] == 1

    # A failure raised on another route is not chat traffic.
    harness.app.state.services.catalog = _FailingCatalog()
    assert (await client.get("/v1/models")).status_code == 502
    document = (await client.get("/dashboard")).json()
    assert document["throughput"]["requests_total"] == 1
    assert document["errors"]["total"] == 1

    # While a rejection the chat service never entered still lands.
    response = await client.post(
        "/v1/chat/completions",
        content="{not json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 400
    document = (await client.get("/dashboard")).json()
    assert document["throughput"]["requests_total"] == 2
    assert document["errors"]["by_kind"]["unknown"] == 2


class _FailingCatalog:
    async def models(self):
        raise ProxyError(
            kind=ErrorKind.UPSTREAM_UNREACHABLE,
            http_status=502,
            code="upstream_unreachable",
            message="provider is down",
            retryable=False,
        )
