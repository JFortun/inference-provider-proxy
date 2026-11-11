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


def _connect_error(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("refused")


def _html_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=b"<html>nope</html>")


def _server_error(request: httpx.Request) -> httpx.Response:
    return httpx.Response(500, json={"error": {"message": "boom", "type": "internal"}})


def _completion_without_usage(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"id": "c1", "choices": []})


def _array_usage(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=b"[1, 2, 3]")


async def test_stalled_stream_when_idle_timeout_hits_then_is_cut_off_as_unreachable() -> None:
    stream = _OpenStream(_StallingResponse(), idle_timeout_s=0.01)

    with pytest.raises(ProxyError) as excinfo:
        async for _ in stream.aiter_bytes():
            pass

    assert excinfo.value.kind is ErrorKind.UPSTREAM_UNREACHABLE
    assert excinfo.value.retryable is True


def test_partial_line_when_oversized_without_newline_then_is_trimmed_to_bound() -> None:
    stream = _OpenStream(_StallingResponse(), idle_timeout_s=1.0)

    stream._scan(b"x" * 70000)

    assert len(stream._buffer) == 4096


def test_non_object_payloads_when_scanned_then_are_ignored() -> None:
    stream = _OpenStream(_StallingResponse(), idle_timeout_s=1.0)

    stream._scan(b"data: not-json\n")
    stream._scan(b"data: [1, 2]\n")

    assert stream.provider_id is None
    assert stream.usage is None


async def test_stream_transport_error_when_connect_fails_then_is_unreachable() -> None:
    provider = _provider(_connect_error)

    with pytest.raises(ProxyError) as excinfo:
        async with provider.open_stream({"model": "m", "messages": []}) as stream:
            async for _ in stream.aiter_bytes():
                pass
    await provider.aclose()

    assert excinfo.value.kind is ErrorKind.UPSTREAM_UNREACHABLE


async def test_catalogue_when_body_is_not_json_then_yields_no_models() -> None:
    provider = _provider(_html_response)

    models = await provider.list_models()
    await provider.aclose()

    assert models == []


async def test_catalogue_when_status_is_an_error_then_is_classified() -> None:
    provider = _provider(_server_error)

    with pytest.raises(ProxyError) as excinfo:
        await provider.list_models()
    await provider.aclose()

    assert excinfo.value.kind is ErrorKind.UPSTREAM_5XX


async def test_completion_when_usage_block_is_absent_then_has_none() -> None:
    provider = _provider(_completion_without_usage)

    completion = await provider.complete({"model": "m", "messages": []})
    await provider.aclose()

    assert completion.provider_id == "c1"
    assert completion.usage is None


async def test_usage_when_body_is_json_array_then_is_marked_stale() -> None:
    provider = _provider(_array_usage)

    info = await provider.fetch_usage()
    await provider.aclose()

    assert info.stale is True


def test_retry_after_when_value_is_unparseable_then_is_none() -> None:
    headers = {"retry-after": "not-a-date"}

    value = CheapestInferenceProvider._retry_after(headers)

    assert value is None


def test_retry_after_when_value_is_naive_http_date_then_is_positive() -> None:
    headers = {"retry-after": "Wed, 21 Oct 2099 07:28:00"}

    value = CheapestInferenceProvider._retry_after(headers)

    assert value is not None
    assert value > 0


def test_parse_timestamp_when_not_a_string_or_invalid_then_is_none() -> None:
    raw_values: list[object] = [123, "not-a-date"]

    parsed = [_parse_timestamp(value) for value in raw_values]

    assert parsed == [None, None]


def test_parse_timestamp_when_naive_then_assumes_utc() -> None:
    raw = "2026-12-31T00:00:00"

    parsed = _parse_timestamp(raw)

    assert parsed is not None
    assert parsed.tzinfo is not None
