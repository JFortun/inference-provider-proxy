"""Metadata is served ungated and cached."""

from __future__ import annotations

import httpx

from inference_provider_proxy.adapters.outbound.subscription.poller import SubscriptionPoller
from inference_provider_proxy.domain.models import ClientIdentity
from qa.integration.conftest import FakeClock, build_harness


def _metadata_handler() -> tuple[object, dict[str, int]]:
    calls = {"models": 0, "usage": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/v1/models"):
            calls["models"] += 1
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {"id": "deepseek-v4.1-flash", "object": "model", "context_length": 128000},
                        {"id": "mimo-v2.5", "object": "model", "context_length": 32000},
                    ],
                },
            )
        if path.endswith("/v1/usage"):
            calls["usage"] += 1
            return httpx.Response(
                200,
                json={
                    "object": "subscription",
                    "pool": "core",
                    "status": "active",
                    "expires_at": "2026-12-31T00:00:00Z",
                    "key_name": "house",
                },
            )
        return httpx.Response(404, json={"error": "not found"})

    return handler, calls


async def test_metadata_succeeds_while_the_key_is_busy(asgi_client_factory) -> None:
    handler, calls = _metadata_handler()
    harness = build_harness(handler)
    client = asgi_client_factory(harness.app)

    slot_cm = harness.gate.slot(
        request_id="holder",
        client=ClientIdentity(address="10.0.0.3"),
        model="m",
        streaming=True,
        max_wait_s=5.0,
    )
    await slot_cm.__aenter__()
    try:
        models = await client.get("/v1/models")
        usage = await client.get("/v1/usage")
        snapshot = harness.gate.snapshot()
    finally:
        await slot_cm.__aexit__(None, None, None)

    assert models.status_code == 200
    assert usage.status_code == 200
    assert [item["id"] for item in models.json()["data"]] == ["deepseek-v4.1-flash", "mimo-v2.5"]
    assert usage.json()["pool"] == "core"

    # Never waited behind, and never consumed, chat capacity.
    assert snapshot.in_flight == 1
    assert snapshot.queued == 0
    assert snapshot.rejected_full_total == 0
    assert snapshot.rejected_timeout_total == 0


async def test_metadata_is_served_from_cache_within_ttl(asgi_client_factory) -> None:
    handler, calls = _metadata_handler()
    harness = build_harness(handler, metadata_cache_ttl=60)
    client = asgi_client_factory(harness.app)

    first = await client.get("/v1/models")
    second = await client.get("/v1/models")

    assert first.status_code == second.status_code == 200
    assert calls["models"] == 1, "the second read must be served from cache"

    await client.get("/v1/usage")
    await client.get("/v1/usage")
    assert calls["usage"] == 1


async def test_usage_refresh_updates_subscription_state(asgi_client_factory) -> None:
    handler, _ = _metadata_handler()
    harness = build_harness(handler)
    client = asgi_client_factory(harness.app)

    await client.get("/v1/usage")
    document = (await client.get("/dashboard")).json()

    assert document["subscription"]["pool"] == "core"
    assert document["subscription"]["status"] == "active"
    assert document["subscription"]["stale"] is False


async def test_cache_entry_expires_after_the_ttl(asgi_client_factory) -> None:
    handler, calls = _metadata_handler()
    clock = FakeClock()
    harness = build_harness(handler, clock=clock, metadata_cache_ttl=5)
    client = asgi_client_factory(harness.app)

    await client.get("/v1/models")
    clock.advance(10)  # past the TTL
    await client.get("/v1/models")

    assert calls["models"] == 2, "an expired entry must be refetched"


async def test_metadata_routes_forward_the_source_payloads(asgi_client_factory) -> None:
    """Nothing is dropped or re-dialected between the provider and the client."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/v1/models"):
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "id": "deepseek-v4.1-flash",
                            "object": "model",
                            "created": 1759872000,
                            "owned_by": "deepseek",
                            "type": "chat",
                            "context_length": 128000,
                        }
                    ],
                },
            )
        if path.endswith("/v1/usage"):
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "data": {
                        "plan": {
                            "slug": "core",
                            "status": "active",
                            "expires_at": "2030-01-01T00:00:00.000Z",
                        },
                        "key": {
                            "name": "Core key",
                            "type": "subscription",
                            "created_at": "2026-09-09T10:10:52.449Z",
                        },
                    },
                },
            )
        return httpx.Response(404, json={"error": "not found"})

    harness = build_harness(handler)
    client = asgi_client_factory(harness.app)

    # The provider's model object arrives verbatim, only pinned keys added.
    assert (await client.get("/v1/models")).json()["data"][0] == {
        "id": "deepseek-v4.1-flash",
        "object": "model",
        "created": 1759872000,
        "owned_by": "deepseek",
        "type": "chat",
        "context_length": 128000,
    }

    usage = (await client.get("/v1/usage")).json()
    assert usage["success"] is True
    assert usage["key"] == {
        "name": "Core key",
        "type": "subscription",
        "created_at": "2026-09-09T10:10:52.449Z",
    }
    assert usage["key_name"] == "Core key"  # derived, kept alongside key.*
    assert usage["pool"] == "core"  # slug renamed by the contract, not dropped
    assert usage["object"] == "subscription"
    assert isinstance(usage["seconds_until_expiry"], int)  # derived
    # One timestamp dialect: same Z suffix the dashboard renders.
    assert usage["expires_at"] == "2030-01-01T00:00:00Z"
    assert "+00:00" not in usage["expires_at"]


async def test_background_refresh_populates_the_dashboard_without_a_client(
    asgi_client_factory,
) -> None:
    """A fresh proxy shows the real catalogue once the poller ran, not "0 models"."""
    handler, calls = _metadata_handler()
    harness = build_harness(handler)
    client = asgi_client_factory(harness.app)

    document = (await client.get("/dashboard")).json()
    assert document["upstream"]["models"] == []  # nothing fetched yet
    assert document["upstream"]["models_checked_at"] is None

    poller = SubscriptionPoller(
        provider=harness.provider,
        subscription=harness.metrics,
        interval_s=30.0,
        clock=FakeClock(),
    )
    await poller.refresh_once()

    document = (await client.get("/dashboard")).json()
    assert [model["id"] for model in document["upstream"]["models"]] == [
        "deepseek-v4.1-flash",
        "mimo-v2.5",
    ]
    assert document["upstream"]["models_checked_at"] is not None
    assert calls["models"] == 1  # only the poller's fetch; the client route never ran
