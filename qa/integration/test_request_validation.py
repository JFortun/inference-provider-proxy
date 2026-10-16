"""Malformed request bodies are rejected with a classified 400 (FR-025)."""

from __future__ import annotations

import httpx

from qa.integration.conftest import build_harness


def _unused_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"id": "c1", "choices": []})


async def test_non_json_body_is_a_classified_400(asgi_client_factory) -> None:
    harness = build_harness(_unused_handler)
    client = asgi_client_factory(harness.app)

    response = await client.post(
        "/v1/chat/completions",
        content=b"{not json",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request_error"


async def test_json_that_is_not_an_object_is_a_classified_400(asgi_client_factory) -> None:
    harness = build_harness(_unused_handler)
    client = asgi_client_factory(harness.app)

    response = await client.post("/v1/chat/completions", json=[1, 2, 3])

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "The request body must be a JSON object."
