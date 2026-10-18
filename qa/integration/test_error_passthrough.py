"""Conditions the proxy does not understand are forwarded verbatim."""

from __future__ import annotations

import httpx

from qa.integration.conftest import build_harness

_PROVIDER_BODY = b'{"error": {"message": "teapot", "type": "vendor_specific"}}'


async def test_unrecognised_provider_error_is_passed_through(asgi_client_factory) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            418, headers={"content-type": "application/json"}, content=_PROVIDER_BODY
        )

    harness = build_harness(handler, max_retries=0)
    client = asgi_client_factory(harness.app)

    response = await client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 418
    # The provider's own bytes reach the client untouched, not rewritten.
    assert response.content == _PROVIDER_BODY
    # Transparency headers ride along on the pass-through too.
    assert response.headers["X-Queue-Wait-Ms"] == "0"
