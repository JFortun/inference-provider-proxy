"""OpenAI-compatible routes (FR-001, FR-002, FR-013, FR-014, FR-017).

Chat is serialized through the gate; catalogue and usage bypass it entirely. No
route requires a credential (FR-016).
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from inference_provider_proxy.adapters.inbound.http.deps import services_of
from inference_provider_proxy.adapters.inbound.http.error_handlers import (
    queue_transparency_headers,
)
from inference_provider_proxy.domain.errors import ErrorKind, ProxyError
from inference_provider_proxy.domain.models import ChatRequest, ClientIdentity

router = APIRouter()

# How often the pre-response watcher asks whether the client is still there.
_DISCONNECT_POLL_S = 0.25


async def _discard[T](task: asyncio.Future[T]) -> None:
    """Cancel ``task`` if it is still running and wait for it to unwind."""
    if not task.done():
        task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def _await_pre_response[T](request: Request, awaitable: Awaitable[T]) -> T:
    """Await a chat call, aborting it if the client disconnects first.

    Nothing else reads the receive channel before the response starts, so the
    watcher may poll ``is_disconnected()``; it stops at the response, where
    Starlette's own disconnect handling takes over (R2).
    """
    call = asyncio.ensure_future(awaitable)
    gone = False

    async def _watch() -> None:
        nonlocal gone
        while not call.done():
            if await request.is_disconnected():
                gone = True
                call.cancel()
                return
            await asyncio.sleep(_DISCONNECT_POLL_S)

    watcher = asyncio.ensure_future(_watch())
    try:
        return await call
    except asyncio.CancelledError:
        if not gone:
            # Cancelled from outside (shutdown): stop the use case too, then let
            # the cancellation through untouched.
            await _discard(call)
            raise
        # The client is gone: answer in the dialect (499) so the access log, the
        # envelope and the dashboard agree, instead of handing the server a
        # traceback for a socket nobody is reading.
        raise ProxyError(
            kind=ErrorKind.CLIENT_ABORTED,
            http_status=499,
            code="client_aborted",
            message="The client disconnected before the response completed.",
            retryable=False,
        ) from None
    finally:
        await _discard(watcher)


def _client_identity(request: Request) -> ClientIdentity:
    address = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent") or "unknown"
    return ClientIdentity(address=address, user_agent=user_agent)


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "") or uuid4().hex


def _transparency_headers(
    request_id: str, queue_wait_ms: int, position: int | None
) -> dict[str, str]:
    return {
        "X-Proxy-Request-Id": request_id,
        **queue_transparency_headers(queue_wait_ms, position),
    }


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request) -> Response:
    services = services_of(request)

    try:
        body = await request.json()
    except ValueError as exc:
        raise ProxyError(
            kind=ErrorKind.UNKNOWN,
            http_status=400,
            code="invalid_request_error",
            message="The request body is not valid JSON.",
        ) from exc
    if not isinstance(body, dict):
        raise ProxyError(
            kind=ErrorKind.UNKNOWN,
            http_status=400,
            code="invalid_request_error",
            message="The request body must be a JSON object.",
        )

    chat_request = ChatRequest(
        request_id=_request_id(request),
        client=_client_identity(request),
        model=str(body.get("model", "")),
        stream=bool(body.get("stream", False)),
        # The body is forwarded verbatim: the proxy never inspects it (FR-001).
        body=body,
    )
    # From here the chat service owns this request's dashboard record; anything
    # raised earlier is recorded by the error handler instead (FR-020).
    request.state.chat_entered = True

    if chat_request.stream:
        result = await _await_pre_response(request, services.chat.stream(chat_request))
        # Handed to Starlette unwrapped: the watcher above has stopped, so the
        # single receive channel is free for Starlette's own disconnect handling
        # from the first chunk onwards (research.md R2).
        return StreamingResponse(
            result.chunks,
            media_type=result.media_type,
            headers=_transparency_headers(
                result.request_id, result.queue_wait_ms, result.queue_position
            ),
        )

    completed = await _await_pre_response(request, services.chat.complete(chat_request))
    return JSONResponse(
        status_code=completed.status_code,
        content=completed.payload,
        headers=_transparency_headers(
            completed.request_id, completed.queue_wait_ms, completed.queue_position
        ),
    )


@router.get("/v1/models")
async def models(request: Request) -> JSONResponse:
    return JSONResponse(content=await services_of(request).catalog.models())


@router.get("/v1/usage")
async def usage(request: Request) -> JSONResponse:
    return JSONResponse(content=await services_of(request).catalog.usage())
