"""Conditions the proxy does not understand are forwarded verbatim."""

from __future__ import annotations

import httpx

from qa.integration.conftest import build_harness

_PROVIDER_BODY = b'{"error": {"message": "teapot", "type": "vendor_specific"}}'


def _teapot(request: httpx.Request) -> httpx.Response:
    return httpx.Response(418, headers={"content-type": "application/json"}, content=_PROVIDER_BODY)


async def test_unrecognised_provider_error_when_returned_then_is_passed_through_verbatim(
    asgi_client_factory,
) -> None:
    harness = build_harness(_teapot, max_retries=0)
    client = asgi_client_factory(harness.app)

    response = await client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 418
    assert response.content == _PROVIDER_BODY
    assert response.headers["X-Queue-Wait-Ms"] == "0"
