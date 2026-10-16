"""Provider payload parsing against frozen live bodies (R3, FR-015)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from inference_provider_proxy.adapters.outbound.provider.cheapestinference import (
    CheapestInferenceProvider,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _provider(handler) -> CheapestInferenceProvider:
    return CheapestInferenceProvider(
        base_url="https://provider.test",
        api_key="k",
        user_agent="ipp/0.1",
        timeout_s=10,
        stream_idle_timeout_s=10,
        transport=httpx.MockTransport(handler),
    )


async def test_usage_nested_payload_is_understood() -> None:
    body = (FIXTURES / "usage_200.json").read_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=json.loads(body))

    provider = _provider(handler)
    try:
        info = await provider.fetch_usage()
    finally:
        await provider.aclose()

    assert info.pool == "core"
    assert info.status == "active"
    assert info.key_name == "Core key"
    assert info.expires_at is not None
    assert info.expires_at.year == 2026
    assert info.seconds_until_expiry is not None
    assert info.stale is False


async def test_models_are_parsed_with_context_length() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "mimo-v2.5", "object": "model", "context_length": 1048576},
                    {"id": "deepseek-v4.1-flash", "object": "model", "context_length": 1000000},
                ]
            },
        )

    provider = _provider(handler)
    try:
        models = await provider.list_models()
    finally:
        await provider.aclose()

    assert [(m.id, m.context_length) for m in models] == [
        ("mimo-v2.5", 1048576),
        ("deepseek-v4.1-flash", 1000000),
    ]


async def test_non_json_usage_body_marks_stale() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>not json</html>")

    provider = _provider(handler)
    try:
        info = await provider.fetch_usage()
    finally:
        await provider.aclose()

    assert info.stale is True
    assert info.pool is None


@pytest.mark.parametrize("raw", ["3", "Wed, 21 Oct 2026 07:28:00 GMT"])
async def test_retry_after_accepts_seconds_and_http_date(raw: str) -> None:
    assert CheapestInferenceProvider._retry_after({"retry-after": raw}) is not None
    assert CheapestInferenceProvider._retry_after({}) is None
