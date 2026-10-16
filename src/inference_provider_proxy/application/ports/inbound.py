"""Inbound ports: what the HTTP adapters are allowed to ask the application to do."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from inference_provider_proxy.domain.metrics import DashboardDocument
    from inference_provider_proxy.domain.models import ChatRequest


@dataclass(slots=True)
class ChatResult:
    """A completed, non-streaming chat completion plus transparency metadata."""

    request_id: str
    payload: dict[str, Any]
    queue_wait_ms: int
    queue_position: int | None = None
    status_code: int = 200


@dataclass(slots=True)
class ChatStreamResult:
    """An opened streaming chat completion plus transparency metadata.

    ``chunks`` must be iterated to completion (or closed) so the gate slot is
    released (FR-008, FR-010).
    """

    request_id: str
    chunks: AsyncIterator[bytes]
    queue_wait_ms: int
    queue_position: int | None = None
    media_type: str = "text/event-stream; charset=utf-8"


class ChatCompletionUseCase(Protocol):
    """The serialized chat path (FR-004, FR-008)."""

    async def complete(self, request: ChatRequest) -> ChatResult: ...

    async def stream(self, request: ChatRequest) -> ChatStreamResult: ...


class CatalogUseCase(Protocol):
    """Ungated catalogue and usage retrieval (FR-002, FR-014)."""

    async def models(self) -> dict[str, Any]: ...

    async def usage(self) -> dict[str, Any]: ...


class DashboardUseCase(Protocol):
    """Assembles the single session document (FR-019)."""

    def document(self) -> DashboardDocument: ...
