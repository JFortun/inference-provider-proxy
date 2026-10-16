"""The asyncio FIFO gate (R1).

A hand-written, cancellation-aware, bounded, capacity-N queue. It provides what
``asyncio.Semaphore`` cannot: strict arrival order (FR-005), bounded depth and
wait (FR-006), waiter cancellation before admission (FR-009), position/wait
reporting (FR-013), configurable capacity (FR-007), and a live snapshot for the
dashboard (FR-019).

The slot is held by the caller for the whole response — from just before the
provider call until the last byte is delivered (FR-008) — because the caller
exits the ``slot`` context manager only when it is finished with the provider.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from inference_provider_proxy.application.ports.gate import QueueTicket
from inference_provider_proxy.domain.errors import ErrorKind, ProxyError
from inference_provider_proxy.domain.metrics import GateSnapshot, InFlightRequest, WaitingRequest
from inference_provider_proxy.domain.models import ClientIdentity

if TYPE_CHECKING:
    from inference_provider_proxy.application.ports.clock import ClockPort

_RETRY_HINT_S = 1.0


@dataclass(slots=True)
class _Waiter:
    request_id: str
    client: ClientIdentity
    model: str
    streaming: bool
    enqueued_mono: float
    enqueued_at: datetime
    position: int
    event: asyncio.Event = field(default_factory=asyncio.Event)
    ticket: QueueTicket | None = None


@dataclass(slots=True)
class _Active:
    request_id: str
    client: ClientIdentity
    model: str
    streaming: bool
    started_mono: float
    started_at: datetime
    ttfb_ms: int | None = None


class AsyncioGate:
    """A FIFO, bounded, capacity-N gate behind ``ConcurrencyGatePort``."""

    def __init__(self, *, capacity: int, max_depth: int, clock: ClockPort) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._capacity = capacity
        self._max_depth = max_depth
        self._clock = clock
        self._in_flight = 0
        self._waiters: list[_Waiter] = []
        self._active: dict[str, _Active] = {}
        self._lock = asyncio.Lock()
        self._admitted_total = 0
        self._rejected_full_total = 0
        self._rejected_timeout_total = 0
        self._cancelled_while_waiting_total = 0

    @asynccontextmanager
    async def slot(
        self,
        *,
        request_id: str,
        client: ClientIdentity,
        model: str,
        streaming: bool,
        max_wait_s: float,
    ) -> AsyncGenerator[QueueTicket]:
        ticket = await self._acquire(
            request_id=request_id,
            client=client,
            model=model,
            streaming=streaming,
            max_wait_s=max_wait_s,
        )
        try:
            yield ticket
        finally:
            async with self._lock:
                self._release_locked(request_id)

    def mark_ttfb(self, *, request_id: str, ttfb_ms: int) -> None:
        active = self._active.get(request_id)
        if active is not None:
            active.ttfb_ms = ttfb_ms

    def snapshot(self) -> GateSnapshot:
        now_mono = self._clock.monotonic()
        current = tuple(
            InFlightRequest(
                request_id=a.request_id,
                client=a.client.address,
                model=a.model,
                streaming=a.streaming,
                started_at=a.started_at,
                elapsed_s=round(now_mono - a.started_mono, 3),
                ttfb_ms=a.ttfb_ms,
            )
            for a in self._active.values()
        )
        waiting = tuple(
            WaitingRequest(
                request_id=w.request_id,
                client=w.client.address,
                model=w.model,
                enqueued_at=w.enqueued_at,
                waited_s=round(now_mono - w.enqueued_mono, 3),
                position=index + 1,
            )
            for index, w in enumerate(self._waiters)
        )
        return GateSnapshot(
            capacity=self._capacity,
            in_flight=self._in_flight,
            queued=len(self._waiters),
            available_slots=self._capacity - self._in_flight,
            admitted_total=self._admitted_total,
            rejected_full_total=self._rejected_full_total,
            rejected_timeout_total=self._rejected_timeout_total,
            cancelled_while_waiting_total=self._cancelled_while_waiting_total,
            oldest_queue_wait_s=(
                round(now_mono - self._waiters[0].enqueued_mono, 3) if self._waiters else None
            ),
            current=current,
            waiting=waiting,
        )

    # -- internals -----------------------------------------------------------

    async def _acquire(
        self,
        *,
        request_id: str,
        client: ClientIdentity,
        model: str,
        streaming: bool,
        max_wait_s: float,
    ) -> QueueTicket:
        async with self._lock:
            if self._in_flight < self._capacity:
                self._start_locked(request_id, client, model, streaming)
                return QueueTicket(
                    request_id=request_id,
                    client=client,
                    enqueued_at=self._clock.now(),
                    position=None,
                    wait_ms=0,
                )
            if len(self._waiters) >= self._max_depth:
                self._rejected_full_total += 1
                raise ProxyError(
                    kind=ErrorKind.QUEUE_FULL,
                    http_status=429,
                    code="429",
                    message="The proxy queue is full; please retry shortly.",
                    retryable=True,
                    retry_after_s=_RETRY_HINT_S,
                )
            waiter = _Waiter(
                request_id=request_id,
                client=client,
                model=model,
                streaming=streaming,
                enqueued_mono=self._clock.monotonic(),
                enqueued_at=self._clock.now(),
                position=len(self._waiters) + 1,
            )
            self._waiters.append(waiter)

        try:
            await asyncio.wait_for(waiter.event.wait(), timeout=max_wait_s)
        except TimeoutError:
            async with self._lock:
                waited_ms = int((self._clock.monotonic() - waiter.enqueued_mono) * 1000)
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
                    self._rejected_timeout_total += 1
            raise ProxyError(
                kind=ErrorKind.QUEUE_TIMEOUT,
                http_status=429,
                code="429",
                message="The request waited longer than the configured bound; please retry.",
                retryable=True,
                retry_after_s=_RETRY_HINT_S,
                queue_wait_ms=waited_ms,
                queue_position=waiter.position,
            ) from None
        except asyncio.CancelledError:
            async with self._lock:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
                    self._cancelled_while_waiting_total += 1
                elif waiter.ticket is not None:
                    # Admitted at the same instant the waiter was cancelled:
                    # hand the slot straight back so it cannot leak.
                    self._release_locked(request_id)
            raise

        if waiter.ticket is None:  # pragma: no cover - defensive
            raise RuntimeError("waiter woken without an admission ticket")
        return waiter.ticket

    def _start_locked(
        self,
        request_id: str,
        client: ClientIdentity,
        model: str,
        streaming: bool,
    ) -> None:
        self._in_flight += 1
        self._admitted_total += 1
        self._active[request_id] = _Active(
            request_id=request_id,
            client=client,
            model=model,
            streaming=streaming,
            started_mono=self._clock.monotonic(),
            started_at=self._clock.now(),
        )

    def _release_locked(self, request_id: str) -> None:
        if request_id in self._active:
            del self._active[request_id]
            self._in_flight -= 1
        self._admit_waiters_locked()

    def _admit_waiters_locked(self) -> None:
        while self._waiters and self._in_flight < self._capacity:
            waiter = self._waiters.pop(0)
            self._start_locked(waiter.request_id, waiter.client, waiter.model, waiter.streaming)
            wait_ms = int((self._clock.monotonic() - waiter.enqueued_mono) * 1000)
            waiter.ticket = QueueTicket(
                request_id=waiter.request_id,
                client=waiter.client,
                enqueued_at=waiter.enqueued_at,
                position=waiter.position,
                wait_ms=wait_ms,
            )
            waiter.event.set()
