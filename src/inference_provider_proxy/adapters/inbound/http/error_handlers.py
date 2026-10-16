"""``ProxyError`` -> OpenAI-shaped envelope (FR-025, FR-029).

Registered as a skeleton here; the per-category construction of ``ProxyError``
happens where each condition is detected (the provider adapter and the chat
service), so this module only renders what it is given. Conditions whose shape
the proxy does not fully understand are passed through untouched rather than
rewritten (FR-025, FR-026).
"""

from __future__ import annotations

import math

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from inference_provider_proxy.domain.errors import ErrorKind, ProxyError

_PASSTHROUGH_KINDS = frozenset({ErrorKind.UNKNOWN, ErrorKind.NOT_RESERVED})


def queue_transparency_headers(
    queue_wait_ms: int | None, queue_position: int | None
) -> dict[str, str]:
    """Queue transparency headers, on success and error responses alike (FR-013)."""
    headers = {"X-Queue-Wait-Ms": str(queue_wait_ms or 0)}
    if queue_position is not None:
        headers["X-Queue-Position"] = str(queue_position)
    return headers


def envelope(error: ProxyError) -> dict[str, object]:
    """The OpenAI-shaped error body."""
    return {
        "error": {
            "message": error.message,
            "type": error.kind.value,
            "code": error.code,
            "param": None,
        }
    }


async def proxy_error_handler(request: Request, exc: Exception) -> Response:
    assert isinstance(exc, ProxyError)
    # Transparency headers are documented for *all* responses, including errors
    # (contracts/openai-compatible.md), so they are emitted here as well as on
    # the success path (FR-013).
    headers = queue_transparency_headers(exc.queue_wait_ms, exc.queue_position)
    if exc.retry_after_s is not None:
        headers["retry-after"] = str(max(0, math.ceil(exc.retry_after_s)))

    # Preserve the provider's own status and body for conditions that are not
    # fully understood, rather than rewriting them (FR-025).
    if exc.provider_body is not None and exc.kind in _PASSTHROUGH_KINDS:
        return Response(
            content=exc.provider_body,
            status_code=exc.provider_status or exc.http_status,
            media_type="application/json",
            headers=headers,
        )

    return JSONResponse(status_code=exc.http_status, content=envelope(exc), headers=headers)


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ProxyError, proxy_error_handler)
