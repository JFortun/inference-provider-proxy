"""Error classification covers every row of the mapping contract.

Driven from the frozen probe fixtures so it never needs the provider.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from inference_provider_proxy.adapters.outbound.provider.cheapestinference import (
    CheapestInferenceProvider,
)
from inference_provider_proxy.domain.errors import ErrorKind, ProxyError

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

CASES = [
    ("throttling_429.json", 429, {"retry-after": "3"}, ErrorKind.PROVIDER_BUSY, "429", True),
    ("auth_401.json", 401, {}, ErrorKind.UPSTREAM_AUTH, "invalid_api_key", False),
    ("edge_block_403.txt", 403, {}, ErrorKind.EDGE_BLOCK, "upstream_edge_blocked", False),
    ("not_reserved_402.json", 402, {}, ErrorKind.NOT_RESERVED, "402", False),
    ("upstream_500.json", 500, {}, ErrorKind.UPSTREAM_5XX, "internal_error", True),
    ("unknown_418.json", 418, {}, ErrorKind.UNKNOWN, "418", False),
]


def _provider(handler) -> CheapestInferenceProvider:
    return CheapestInferenceProvider(
        base_url="https://provider.test",
        api_key="k",
        user_agent="ipp/0.1",
        timeout_s=10,
        stream_idle_timeout_s=10,
        transport=httpx.MockTransport(handler),
    )


@pytest.mark.parametrize("name,status,headers,kind,code,retryable", CASES)
async def test_each_row_of_the_mapping_table(name, status, headers, kind, code, retryable) -> None:
    body = (FIXTURES / name).read_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, headers=headers, content=body)

    provider = _provider(handler)
    try:
        with pytest.raises(ProxyError) as excinfo:
            await provider.complete({"model": "m", "messages": []})
    finally:
        await provider.aclose()

    error = excinfo.value
    assert error.kind is kind
    assert error.code == code
    assert isinstance(error.code, str)
    assert error.provider_status == status
    assert error.retryable is retryable


async def test_provider_busy_honours_retry_after_header() -> None:
    body = (FIXTURES / "throttling_429.json").read_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "7"}, content=body)

    provider = _provider(handler)
    try:
        with pytest.raises(ProxyError) as excinfo:
            await provider.complete({"model": "m", "messages": []})
    finally:
        await provider.aclose()
    assert excinfo.value.retry_after_s == 7.0


async def test_edge_block_is_never_reported_as_auth() -> None:
    body = (FIXTURES / "edge_block_403.txt").read_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, content=body)

    provider = _provider(handler)
    try:
        with pytest.raises(ProxyError) as excinfo:
            await provider.complete({"model": "m", "messages": []})
    finally:
        await provider.aclose()
    assert excinfo.value.kind is ErrorKind.EDGE_BLOCK
    assert excinfo.value.kind is not ErrorKind.UPSTREAM_AUTH
    assert excinfo.value.code == "upstream_edge_blocked"


async def test_unrecognised_shape_is_passed_through_not_rewritten() -> None:
    body = (FIXTURES / "unknown_418.json").read_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(418, content=body)

    provider = _provider(handler)
    try:
        with pytest.raises(ProxyError) as excinfo:
            await provider.complete({"model": "m", "messages": []})
    finally:
        await provider.aclose()
    assert excinfo.value.kind is ErrorKind.UNKNOWN
    assert excinfo.value.provider_body == body


async def test_transport_error_is_unreachable_and_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    provider = _provider(handler)
    try:
        with pytest.raises(ProxyError) as excinfo:
            await provider.complete({"model": "m", "messages": []})
    finally:
        await provider.aclose()
    assert excinfo.value.kind is ErrorKind.UPSTREAM_UNREACHABLE
    assert excinfo.value.retryable is True
