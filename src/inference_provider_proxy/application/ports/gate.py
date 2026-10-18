"""Driven port for the concurrency gate."""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from datetime import datetime

    from inference_provider_proxy.domain.metrics import GateSnapshot
    from inference_provider_proxy.domain.models import ClientIdentity


@dataclass(slots=True)
class QueueTicket:
    """The reservation a request holds while waiting or in flight.

    ``position`` is the 1-based FIFO place the request occupied while waiting,
    or ``None`` when it was admitted immediately.
    """

    request_id: str
    client: ClientIdentity
    enqueued_at: datetime
    position: int | None
    wait_ms: int


class ConcurrencyGatePort(Protocol):
    """A cancellation-aware, bounded, FIFO gate."""

    def slot(
        self,
        *,
        request_id: str,
        client: ClientIdentity,
        model: str,
        streaming: bool,
        max_wait_s: float,
    ) -> AbstractAsyncContextManager[QueueTicket]:
        """Yield a ticket once admitted, holding capacity until the block exits.

        Raises a ``queue_full`` or ``queue_timeout`` :class:`ProxyError` when a
        bound is exceeded. A waiter cancelled before admission never
        consumes capacity.
        """
        ...

    def mark_ttfb(self, *, request_id: str, ttfb_ms: int) -> None:
        """Record the time to first upstream byte for an in-flight request."""
        ...

    def snapshot(self) -> GateSnapshot: ...
