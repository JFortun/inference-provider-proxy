"""Internal helpers of the chat service (FR-010)."""

from __future__ import annotations

from inference_provider_proxy.application.services.chat import ChatService


class _ExplodingContextManager:
    async def __aexit__(self, *exc: object) -> bool:
        raise RuntimeError("teardown failed")


async def test_close_swallows_teardown_errors() -> None:
    # Closing must never mask the outcome the slot is being released for (FR-010).
    await ChatService._close(_ExplodingContextManager())
