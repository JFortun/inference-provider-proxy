"""Driven port for the upstream provider (FR-001, FR-002, FR-025)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from inference_provider_proxy.domain.models import ModelInfo, SubscriptionInfo


@dataclass(slots=True)
class TokenUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(slots=True)
class ProviderCompletion:
    """A non-streaming provider response."""

    payload: dict[str, Any]
    status_code: int = 200
    provider_id: str | None = None
    provider_correlation_id: str | None = None
    usage: TokenUsage | None = None


class ProviderStream(Protocol):
    """An opened streaming provider response.

    ``usage``, ``provider_id`` and ``provider_correlation_id`` are populated as
    the stream is consumed, so read them after iteration finishes.
    """

    provider_id: str | None
    provider_correlation_id: str | None
    usage: TokenUsage | None

    def aiter_bytes(self) -> AsyncIterator[bytes]: ...


class LLMProviderPort(Protocol):
    """Everything the application needs from the upstream provider."""

    async def complete(self, body: Mapping[str, Any]) -> ProviderCompletion: ...

    def open_stream(
        self, body: Mapping[str, Any]
    ) -> AbstractAsyncContextManager[ProviderStream]: ...

    async def list_models(self) -> list[ModelInfo]: ...

    async def fetch_usage(self) -> SubscriptionInfo: ...

    async def aclose(self) -> None: ...
