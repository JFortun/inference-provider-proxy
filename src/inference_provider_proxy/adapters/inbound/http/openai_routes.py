"""OpenAI-compatible routes (FR-001, FR-002, FR-013, FR-014, FR-017).

Chat is serialized through the gate; catalogue and usage bypass it entirely. No
route requires a credential (FR-016).
"""

from __future__ import annotations

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

    if chat_request.stream:
        result = await services.chat.stream(chat_request)
        # The chunk generator is handed to Starlette unwrapped on purpose. When
        # the client disconnects, Starlette raises CancelledError/GeneratorExit
        # into it, which the chat service turns into an eager upstream cancel
        # and a bounded slot release (FR-010, research.md R2). Wrapping it here
        # to poll ``request.is_disconnected()`` would compete for the
        # single-consumer ASGI receive channel and could weaken that signal.
        return StreamingResponse(
            result.chunks,
            media_type=result.media_type,
            headers=_transparency_headers(
                result.request_id, result.queue_wait_ms, result.queue_position
            ),
        )

    completed = await services.chat.complete(chat_request)
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
