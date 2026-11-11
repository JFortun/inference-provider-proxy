"""Internal helpers of the chat service."""

from __future__ import annotations

from inference_provider_proxy.application.services.chat import ChatService


class _ExplodingContextManager:
    def __init__(self) -> None:
        self.closed = False

    async def __aexit__(self, *exc: object) -> bool:
        self.closed = True
        raise RuntimeError("teardown failed")


async def test_close_when_teardown_raises_then_swallows_the_error() -> None:
    exploding = _ExplodingContextManager()

    await ChatService._close(exploding)

    assert exploding.closed is True
