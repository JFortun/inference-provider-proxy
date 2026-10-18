"""The paid key never reaches a client response or a log record."""

from __future__ import annotations

import logging

import httpx

from qa.integration.conftest import build_harness

KEY = "sekret-provider-key-abc123"


def _handler(seen: dict[str, str]):
    def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers.get("authorization", "")
        seen["user_agent"] = request.headers.get("user-agent", "")
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

    return handler


async def test_key_is_substituted_outbound_but_never_returned(asgi_client_factory, caplog) -> None:
    seen: dict[str, str] = {}
    harness = build_harness(_handler(seen), upstream_api_key=KEY)
    client = asgi_client_factory(harness.app)

    with caplog.at_level(logging.INFO):
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        dashboard = await client.get("/dashboard")

    assert response.status_code == 200

    # Substituted server-side on the outbound call.
    assert seen["authorization"] == f"Bearer {KEY}"
    assert seen["user_agent"] == harness.settings.user_agent

    # Absent from every client-facing surface.
    assert KEY not in response.text
    assert KEY not in dashboard.text
    for name, value in response.headers.items():
        assert KEY not in f"{name}: {value}"

    # Absent from logs.
    assert KEY not in caplog.text
    assert "Bearer" not in caplog.text


async def test_client_supplied_credential_is_ignored(asgi_client_factory) -> None:
    seen: dict[str, str] = {}
    harness = build_harness(_handler(seen), upstream_api_key=KEY)
    client = asgi_client_factory(harness.app)

    await client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer client-supplied-nonsense"},
    )

    # The client's own credential is never forwarded upstream.
    assert seen["authorization"] == f"Bearer {KEY}"
    assert "client-supplied-nonsense" not in seen["authorization"]
