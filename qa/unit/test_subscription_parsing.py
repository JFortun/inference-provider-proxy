"""Provider payload parsing against frozen live bodies."""

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


def _usage_fixture(request: httpx.Request) -> httpx.Response:
    body = (FIXTURES / "usage_200.json").read_bytes()
    return httpx.Response(200, json=json.loads(body))


def _models_fixture(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "data": [
                {
                    "id": "mimo-v2.5",
                    "object": "model",
                    "context_length": 1048576,
                    "created": 1735689600,
                    "owned_by": "xiaomi",
                    "type": "chat",
                },
                {"id": "deepseek-v4.1-flash", "object": "model", "context_length": 1000000},
            ]
        },
    )


def _html_usage(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=b"<html>not json</html>")


async def test_usage_payload_when_nested_then_is_fully_understood() -> None:
    provider = _provider(_usage_fixture)

    info = await provider.fetch_usage()
    await provider.aclose()

    assert info.pool == "core"
    assert info.status == "active"
    assert info.key_name == "Core key"
    assert info.expires_at is not None
    assert info.expires_at.year == 2026
    assert info.seconds_until_expiry is not None
    assert info.stale is False
    assert info.success is True
    assert info.key == {
        "name": "Core key",
        "type": "subscription",
        "created_at": "2026-09-09T10:10:52.449Z",
    }


async def test_models_when_parsed_then_carry_context_length_and_source() -> None:
    provider = _provider(_models_fixture)

    models = await provider.list_models()
    await provider.aclose()

    assert [(m.id, m.context_length) for m in models] == [
        ("mimo-v2.5", 1048576),
        ("deepseek-v4.1-flash", 1000000),
    ]
    assert set(models[0].source) == {
        "id",
        "object",
        "context_length",
        "created",
        "owned_by",
        "type",
    }


async def test_usage_when_body_is_not_json_then_is_marked_stale() -> None:
    provider = _provider(_html_usage)

    info = await provider.fetch_usage()
    await provider.aclose()

    assert info.stale is True
    assert info.pool is None


@pytest.mark.parametrize("raw", ["3", "Wed, 21 Oct 2026 07:28:00 GMT"])
async def test_retry_after_when_seconds_or_http_date_then_is_accepted(raw: str) -> None:
    headers = {"retry-after": raw}

    parsed = CheapestInferenceProvider._retry_after(headers)
    missing = CheapestInferenceProvider._retry_after({})

    assert parsed is not None
    assert missing is None
