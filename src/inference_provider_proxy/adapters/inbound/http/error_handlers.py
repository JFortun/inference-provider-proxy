"""``ProxyError`` -> OpenAI-shaped envelope (FR-025, FR-029), plus one log line.

Registered as a skeleton here; the per-category construction of ``ProxyError``
happens where each condition is detected (the provider adapter and the chat
service), so this module only renders what it is given — and says what it was
given, at a level that matches the taxonomy: client mistakes stay at INFO, load
shedding is a WARNING, provider trouble is an ERROR. Conditions whose shape the
proxy does not fully understand are passed through untouched rather than
rewritten (FR-025, FR-026); anything that reaches the server without being
classified at all is logged with its traceback and rendered as a 500.
"""

from __future__ import annotations

import logging
import math

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from inference_provider_proxy.domain.errors import ErrorKind, ProxyError

_PASSTHROUGH_KINDS = frozenset({ErrorKind.UNKNOWN, ErrorKind.NOT_RESERVED})

# Failures the operator must see: the provider or the proxy itself is at fault.
_ERROR_KINDS = frozenset(
    {
        ErrorKind.UPSTREAM_AUTH,
        ErrorKind.EDGE_BLOCK,
        ErrorKind.NOT_RESERVED,
        ErrorKind.UPSTREAM_5XX,
        ErrorKind.UPSTREAM_UNREACHABLE,
    }
)

# Conditions where the proxy shed load or compensated: worth a WARNING.
_WARNING_KINDS = frozenset({ErrorKind.QUEUE_FULL, ErrorKind.QUEUE_TIMEOUT, ErrorKind.PROVIDER_BUSY})

logger = logging.getLogger("ipp.http")


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


def _level_for(error: ProxyError) -> int:
    """Map one classified failure onto INFO, WARNING or ERROR."""
    if error.kind in _ERROR_KINDS or error.http_status >= 500:
        return logging.ERROR
    if error.kind in _WARNING_KINDS:
        return logging.WARNING
    return logging.INFO


def _log(request: Request, error: ProxyError) -> None:
    """One line per failed request: what, to whom, and how long it waited.

    The provider's own body, when it is passed through verbatim, is only ever
    quoted at DEBUG and truncated, so an operator reading at INFO sees ids and
    statuses and nothing that could hold a credential (FR-018).
    """
    logger.log(
        _level_for(error),
        "%d %s request_id=%s queue_wait_ms=%s provider_status=%s correlation=%s",
        error.http_status,
        error.kind,
        getattr(request.state, "request_id", ""),
        error.queue_wait_ms or 0,
        error.provider_status,
        error.provider_correlation_id,
    )
    if error.provider_body is not None:
        preview = error.provider_body[:512].decode("utf-8", "replace")
        logger.debug("passing the provider body through untouched: %s", preview)


async def proxy_error_handler(request: Request, exc: Exception) -> Response:
    assert isinstance(exc, ProxyError)
    _log(request, exc)
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


async def unhandled_error_handler(request: Request, exc: Exception) -> Response:
    """Classify nothing, hide nothing: log the traceback, answer in the dialect.

    The exception is re-raised by Starlette afterwards, so the server logs its
    own copy too; this line is the one carrying the ``request_id`` that ties the
    traceback to the access log.
    """
    logger.error(
        "unhandled %s request_id=%s: %s",
        type(exc).__name__,
        getattr(request.state, "request_id", ""),
        exc,
        exc_info=exc,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": f"Unexpected proxy failure: {type(exc).__name__}.",
                "type": "server_error",
                "code": "server_error",
                "param": None,
            }
        },
    )


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ProxyError, proxy_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)
