"""CheapestInference provider adapter.

Owns the single ``httpx.AsyncClient``, substitutes the real provider credential
server-side, identifies itself with an explicit non-default User-Agent, passes
the provider's status and body through for conditions it does not fully
understand, and classifies every failure into exactly one category. No
request-size ceiling is imposed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from inference_provider_proxy.application.ports.provider import (
    ProviderCompletion,
    TokenUsage,
)
from inference_provider_proxy.domain.errors import ErrorKind, ProxyError
from inference_provider_proxy.domain.models import ModelInfo, SubscriptionInfo

logger = logging.getLogger("ipp.provider")

_CHAT_PATH = "/v1/chat/completions"
_MODELS_PATH = "/v1/models"
_USAGE_PATH = "/v1/usage"
_EDGE_MARKER = "1010"


def _token_usage(value: Any) -> TokenUsage | None:
    """Extract prompt/completion token counts from a provider ``usage`` object."""
    if not isinstance(value, dict):
        return None
    return TokenUsage(
        prompt_tokens=value.get("prompt_tokens"),
        completion_tokens=value.get("completion_tokens"),
    )


class _OpenStream:
    """Wraps an upstream SSE response, scanning chunks for ids and usage.

    Each read is bounded by the configured stream idle timeout, so a stalled
    upstream cannot hold the single slot open indefinitely.
    """

    def __init__(self, response: httpx.Response, *, idle_timeout_s: float) -> None:
        self._response = response
        self._idle_timeout_s = idle_timeout_s
        self.provider_id: str | None = None
        self.provider_correlation_id: str | None = response.headers.get("x-correlation-id")
        self.usage: TokenUsage | None = None
        self._buffer = b""

    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        iterator = self._response.aiter_bytes()
        while True:
            try:
                chunk = await asyncio.wait_for(anext(iterator), timeout=self._idle_timeout_s)
            except StopAsyncIteration:
                return
            except TimeoutError as exc:
                raise self._idle_timeout() from exc
            self._scan(chunk)
            yield chunk

    def _idle_timeout(self) -> ProxyError:
        logger.warning("provider stream stalled for %ss, cutting it off", self._idle_timeout_s)
        return ProxyError(
            kind=ErrorKind.UPSTREAM_UNREACHABLE,
            http_status=502,
            code="upstream_unreachable",
            message=(
                "The provider stream stalled longer than the configured idle "
                f"timeout ({self._idle_timeout_s:g}s) and was cut off."
            ),
            retryable=True,
            retry_after_s=1.0,
        )

    def _scan(self, chunk: bytes) -> None:
        self._buffer += chunk
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            text = line.decode("utf-8", "replace").strip()
            if not text.startswith("data:"):
                continue
            payload = text[len("data:") :].strip()
            if not payload or payload == "[DONE]":
                continue
            self._observe(payload)
        # Keep only a bounded tail so a malformed stream cannot grow unbounded.
        if len(self._buffer) > 65536:
            self._buffer = self._buffer[-4096:]

    def _observe(self, payload: str) -> None:
        try:
            data = json.loads(payload)
        except ValueError, TypeError:
            return
        if not isinstance(data, dict):
            return
        if isinstance(data.get("id"), str):
            self.provider_id = data["id"]
        usage = _token_usage(data.get("usage"))
        if usage is not None:
            self.usage = usage


class CheapestInferenceProvider:
    """Implements ``LLMProviderPort`` against the CheapestInference API."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        user_agent: str,
        timeout_s: float,
        stream_idle_timeout_s: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._stream_idle_timeout_s = stream_idle_timeout_s
        timeout = httpx.Timeout(timeout_s, connect=min(timeout_s, 10.0))
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout,
            transport=transport,
            headers={"User-Agent": user_agent},
        )

    # -- LLMProviderPort -----------------------------------------------------

    async def complete(self, body: Mapping[str, Any]) -> ProviderCompletion:
        headers = self._outbound_headers()
        started = time.monotonic()
        try:
            response = await self._client.post(_CHAT_PATH, json=dict(body), headers=headers)
        except httpx.TransportError as exc:
            raise self._unreachable(exc) from exc
        logger.debug(
            "POST %s status=%d in %dms",
            _CHAT_PATH,
            response.status_code,
            int((time.monotonic() - started) * 1000),
        )

        if response.status_code >= 400:
            raise self._classify(response.status_code, response.headers, response.content)

        try:
            payload = response.json()
        except ValueError as exc:
            logger.error("provider returned malformed JSON: %s", type(exc).__name__)
            raise
        return ProviderCompletion(
            payload=payload,
            status_code=response.status_code,
            provider_id=payload.get("id") if isinstance(payload, dict) else None,
            provider_correlation_id=response.headers.get("x-correlation-id"),
            usage=self._usage_from(payload),
        )

    @asynccontextmanager
    async def open_stream(self, body: Mapping[str, Any]) -> AsyncGenerator[_OpenStream]:
        headers = self._outbound_headers(stream=True)
        started = time.monotonic()
        try:
            async with self._client.stream(
                "POST", _CHAT_PATH, json=dict(body), headers=headers
            ) as response:
                logger.debug(
                    "POST %s status=%d in %dms",
                    _CHAT_PATH,
                    response.status_code,
                    int((time.monotonic() - started) * 1000),
                )
                if response.status_code >= 400:
                    raw = await response.aread()
                    raise self._classify(response.status_code, response.headers, raw)
                yield _OpenStream(response, idle_timeout_s=self._stream_idle_timeout_s)
        except httpx.TransportError as exc:
            raise self._unreachable(exc) from exc

    async def list_models(self) -> list[ModelInfo]:
        raw = await self._get_bytes(_MODELS_PATH)
        try:
            data = json.loads(raw)
        except ValueError:
            logger.warning("provider catalogue is not JSON; serving an empty list")
            return []
        models: list[ModelInfo] = []
        for item in data.get("data", []) if isinstance(data, dict) else []:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                models.append(
                    ModelInfo(
                        id=item["id"],
                        context_length=item.get("context_length"),
                        source=item,
                    )
                )
        return models

    async def fetch_usage(self) -> SubscriptionInfo:
        return self._subscription_from(await self._get_bytes(_USAGE_PATH))

    async def aclose(self) -> None:
        await self._client.aclose()

    # -- internals -----------------------------------------------------------

    async def _get_bytes(self, path: str) -> bytes:
        """GET one provider path, classifying any failure."""
        started = time.monotonic()
        async with self._client.stream("GET", path, headers=self._outbound_headers()) as response:
            logger.debug(
                "GET %s status=%d in %dms",
                path,
                response.status_code,
                int((time.monotonic() - started) * 1000),
            )
            if response.status_code >= 400:
                raw = await response.aread()
                raise self._classify(response.status_code, response.headers, raw)
            return await response.aread()

    def _outbound_headers(self, *, stream: bool = False) -> dict[str, str]:
        """Build outbound headers, substituting the real key server-side.

        The client's own ``Authorization`` header is never read or forwarded.
        """
        headers = {"Authorization": f"Bearer {self._api_key}"}
        headers["Accept"] = "text/event-stream" if stream else "application/json"
        return headers

    @staticmethod
    def _usage_from(payload: Any) -> TokenUsage | None:
        return _token_usage(payload.get("usage")) if isinstance(payload, dict) else None

    @staticmethod
    def _subscription_from(raw: bytes) -> SubscriptionInfo:
        now = datetime.now(UTC)
        try:
            data = json.loads(raw)
        except ValueError:
            logger.warning("provider usage payload is not JSON; marking the view stale")
            return SubscriptionInfo(last_checked_at=now, stale=True)
        if not isinstance(data, dict):
            logger.warning("provider usage payload is not an object; marking the view stale")
            return SubscriptionInfo(last_checked_at=now, stale=True)
        # The provider wraps the payload: {"success": true, "data": {...}},
        # with the plan under "plan" and the key label under "key".
        raw_success = data.get("success")
        if isinstance(data.get("data"), dict):
            data = data["data"]
        plan = data.get("plan") if isinstance(data.get("plan"), dict) else {}
        key = data.get("key") if isinstance(data.get("key"), dict) else {}

        expires_at = _parse_timestamp(plan.get("expires_at") or data.get("expires_at"))
        seconds = data.get("seconds_until_expiry")
        if seconds is None and expires_at is not None:
            seconds = int((expires_at - now).total_seconds())
        return SubscriptionInfo(
            pool=plan.get("slug") or data.get("pool"),
            status=plan.get("status") or data.get("status"),
            expires_at=expires_at,
            seconds_until_expiry=seconds,
            key_name=key.get("name") or data.get("key_name") or data.get("name"),
            success=raw_success if isinstance(raw_success, bool) else None,
            key=key,
            last_checked_at=now,
            stale=False,
        )

    @staticmethod
    def _retry_after(headers: Mapping[str, str]) -> float | None:
        raw = headers.get("retry-after")
        if not raw:
            return None
        try:
            return max(0.0, float(raw))
        except ValueError:
            try:
                when = parsedate_to_datetime(raw)
            except TypeError, ValueError:
                return None
            if when.tzinfo is None:
                when = when.replace(tzinfo=UTC)
            return max(0.0, (when - datetime.now(UTC)).total_seconds())

    def _unreachable(self, exc: Exception) -> ProxyError:
        # The exception class only: its message may quote parts of a request,
        # and nothing about the credential may reach a record.
        logger.warning("provider unreachable: %s", type(exc).__name__)
        return ProxyError(
            kind=ErrorKind.UPSTREAM_UNREACHABLE,
            http_status=502,
            code="upstream_unreachable",
            message=f"The provider could not be reached: {type(exc).__name__}.",
            retryable=True,
            retry_after_s=1.0,
        )

    def _classify(self, status: int, headers: Mapping[str, str], raw: bytes) -> ProxyError:
        """Map one provider failure onto exactly one category."""
        text = raw.decode("utf-8", "replace")
        payload: Any = None
        try:
            payload = json.loads(raw)
        except ValueError:
            payload = None
        correlation = headers.get("x-correlation-id")
        retry_after = self._retry_after(headers)
        error_type = ""
        code = ""
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                error_type = str(error.get("type", ""))
                code = str(error.get("code", ""))  # string-safe

        # Upstream concurrency collision: 429 + throttling.
        if status == 429:
            detail = f" ({error_type})" if error_type else ""
            return ProxyError(
                kind=ErrorKind.PROVIDER_BUSY,
                http_status=429,
                code="429",
                message=f"The provider reported a concurrency collision{detail}.",
                retryable=True,
                retry_after_s=retry_after if retry_after is not None else 1.0,
                provider_status=status,
                provider_correlation_id=correlation,
            )

        # Credential rejection (JSON body) vs edge-layer block (non-JSON).
        if status in (401, 403):
            if payload is not None:
                logger.error(
                    "provider rejected the credential status=%d correlation=%s",
                    status,
                    correlation,
                )
                return ProxyError(
                    kind=ErrorKind.UPSTREAM_AUTH,
                    http_status=status,
                    code=code or "invalid_api_key",
                    message="The provider rejected the proxy credential.",
                    retryable=False,
                    provider_status=status,
                    provider_correlation_id=correlation,
                )
            return ProxyError(
                kind=ErrorKind.EDGE_BLOCK,
                http_status=status,
                code="upstream_edge_blocked",
                message=(
                    "The provider's edge layer blocked the request"
                    + (f" ({_EDGE_MARKER})" if _EDGE_MARKER in text else "")
                    + "."
                ),
                retryable=False,
                provider_status=status,
                provider_correlation_id=correlation,
            )

        # Outside the reserved block / payment required: pass through and tag.
        if status == 402:
            return ProxyError(
                kind=ErrorKind.NOT_RESERVED,
                http_status=402,
                code=code or "402",
                message="The subscription is outside its reserved window or unpaid.",
                retryable=False,
                provider_status=status,
                provider_correlation_id=correlation,
                provider_body=raw,
            )

        if 500 <= status < 600:
            return ProxyError(
                kind=ErrorKind.UPSTREAM_5XX,
                http_status=status,
                code=code or "upstream_error",
                message="The provider returned a server error.",
                retryable=True,
                # A retryable condition must carry a retry hint even when the
                # provider did not send one.
                retry_after_s=retry_after if retry_after is not None else 1.0,
                provider_status=status,
                provider_correlation_id=correlation,
            )

        # Anything else: pass the provider's status and body through, tagged.
        return ProxyError(
            kind=ErrorKind.UNKNOWN,
            http_status=status,
            code=code or str(status),
            message="The provider returned an unrecognised response.",
            retryable=False,
            provider_status=status,
            provider_correlation_id=correlation,
            provider_body=raw,
        )


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed
