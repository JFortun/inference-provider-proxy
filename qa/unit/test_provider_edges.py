"""Provider branches beyond the classification fixtures."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from inference_provider_proxy.adapters.outbound.provider.cheapestinference import (
    CheapestInferenceProvider,
    _OpenStream,
    _parse_timestamp,
)
from inference_provider_proxy.domain.errors import ErrorKind, ProxyError


class _StallingResponse(httpx.Response):
    """A response whose body never yields, to trigger the stream idle timeout."""

    def __init__(self) -> None:
        super().__init__(200)

    async def aiter_bytes(self, chunk_size: int | None = None):
        await asyncio.sleep(30)
        yield b"never"


def _provider(handler, *, idle_s: float = 10.0) -> CheapestInferenceProvider:
    return CheapestInferenceProvider(
        base_url="https://provider.test",
        api_key="k",
        user_agent="ipp/0.1",
        timeout_s=10,
        stream_idle_timeout_s=idle_s,
        transport=httpx.MockTransport(handler),
    )


async def test_stalled_stream_is_cut_off_as_unreachable() -> None:
    stream = _OpenStream(_StallingResponse(), idle_timeout_s=0.01)

    with pytest.raises(ProxyError) as excinfo:
        async for _ in stream.aiter_bytes():
            pass

    assert excinfo.value.kind is ErrorKind.UPSTREAM_UNREACHABLE
    assert excinfo.value.retryable is True


def test_oversized_partial_line_is_trimmed() -> None:
    stream = _OpenStream(_StallingResponse(), idle_timeout_s=1.0)

    stream._scan(b"x" * 70000)  # no newline, so nothing is parsed and the buffer must stay bounded

    assert len(stream._buffer) == 4096


def test_non_json_and_non_object_payloads_are_ignored() -> None:
    stream = _OpenStream(_StallingResponse(), idle_timeout_s=1.0)

    stream._scan(b"data: not-json\n")
    stream._scan(b"data: [1, 2]\n")

    assert stream.provider_id is None
    assert stream.usage is None


async def test_stream_transport_error_is_classified_as_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    provider = _provider(handler)
    try:
        with pytest.raises(ProxyError) as excinfo:
            async with provider.open_stream({"model": "m", "messages": []}) as stream:
                async for _ in stream.aiter_bytes():
                    pass
    finally:
        await provider.aclose()

    assert excinfo.value.kind is ErrorKind.UPSTREAM_UNREACHABLE


async def test_catalogue_body_that_is_not_json_yields_no_models() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>nope</html>")

    provider = _provider(handler)
    try:
        assert await provider.list_models() == []
    finally:
        await provider.aclose()


async def test_catalogue_error_status_is_classified() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "boom", "type": "internal"}})

    provider = _provider(handler)
    try:
        with pytest.raises(ProxyError) as excinfo:
            await provider.list_models()
    finally:
        await provider.aclose()

    assert excinfo.value.kind is ErrorKind.UPSTREAM_5XX


async def test_completion_without_a_usage_block_has_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "c1", "choices": []})

    provider = _provider(handler)
    try:
        completion = await provider.complete({"model": "m", "messages": []})
    finally:
        await provider.aclose()

    assert completion.provider_id == "c1"
    assert completion.usage is None


async def test_usage_body_that_is_json_but_not_an_object_is_stale() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"[1, 2, 3]")

    provider = _provider(handler)
    try:
        info = await provider.fetch_usage()
    finally:
        await provider.aclose()

    assert info.stale is True


def test_retry_after_unparseable_value_is_none() -> None:
    assert CheapestInferenceProvider._retry_after({"retry-after": "not-a-date"}) is None


def test_retry_after_accepts_a_naive_http_date() -> None:
    value = CheapestInferenceProvider._retry_after({"retry-after": "Wed, 21 Oct 2099 07:28:00"})

    assert value is not None
    assert value > 0


def test_parse_timestamp_handles_non_strings_and_bad_values() -> None:
    assert _parse_timestamp(123) is None
    assert _parse_timestamp("not-a-date") is None


def test_parse_timestamp_assumes_utc_for_naive_values() -> None:
    parsed = _parse_timestamp("2026-12-31T00:00:00")

    assert parsed is not None
    assert parsed.tzinfo is not None
