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


def _status_handler(status: int, headers: dict | None = None, body: bytes | None = None):
    payload = body if body is not None else b""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, headers=headers or {}, content=payload)

    return handler


def _error_body(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


async def _classify(handler) -> ProxyError:
    provider = _provider(handler)
    try:
        with pytest.raises(ProxyError) as excinfo:
            await provider.complete({"model": "m", "messages": []})
    finally:
        await provider.aclose()
    return excinfo.value


@pytest.mark.parametrize("name,status,headers,kind,code,retryable", CASES)
async def test_each_row_of_the_mapping_table_when_classified_then_matches_the_contract(
    name, status, headers, kind, code, retryable
) -> None:
    body = _error_body(name)
    handler = _status_handler(status, headers, body)

    error = await _classify(handler)

    assert error.kind is kind
    assert error.code == code
    assert isinstance(error.code, str)
    assert error.provider_status == status
    assert error.retryable is retryable


async def test_provider_busy_when_retry_after_is_sent_then_honours_it() -> None:
    body = _error_body("throttling_429.json")
    handler = _status_handler(429, {"retry-after": "7"}, body)

    error = await _classify(handler)

    assert error.retry_after_s == 7.0


async def test_edge_block_when_returned_then_is_never_reported_as_auth() -> None:
    body = _error_body("edge_block_403.txt")
    handler = _status_handler(403, body=body)

    error = await _classify(handler)

    assert error.kind is ErrorKind.EDGE_BLOCK
    assert error.kind is not ErrorKind.UPSTREAM_AUTH
    assert error.code == "upstream_edge_blocked"


async def test_unrecognised_status_when_returned_then_is_passed_through_not_rewritten() -> None:
    body = _error_body("unknown_418.json")
    handler = _status_handler(418, body=body)

    error = await _classify(handler)

    assert error.kind is ErrorKind.UNKNOWN
    assert error.provider_body == body


async def test_transport_error_when_connect_fails_then_is_unreachable_and_retryable() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    error = await _classify(refuse)

    assert error.kind is ErrorKind.UPSTREAM_UNREACHABLE
    assert error.retryable is True
