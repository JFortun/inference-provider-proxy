"""Unit tests for the asyncio FIFO gate (FR-004-FR-009, SC-003).

The gate is the load-bearing component: T024 implements it, these tests prove it.
"""

from __future__ import annotations

import asyncio
import random

import pytest

from inference_provider_proxy.adapters.outbound.gate.asyncio_gate import AsyncioGate
from inference_provider_proxy.domain.errors import ErrorKind, ProxyError
from inference_provider_proxy.domain.models import ClientIdentity
from qa.fakes import FakeClock

CLIENT = ClientIdentity(address="10.0.0.1", user_agent="pytest")


def _slot(gate: AsyncioGate, request_id: str, *, max_wait_s: float = 5.0):
    return gate.slot(
        request_id=request_id,
        client=CLIENT,
        model="deepseek-v4.1-flash",
        streaming=False,
        max_wait_s=max_wait_s,
    )


async def _settle() -> None:
    """Give the event loop enough turns to enqueue/observe waiters."""
    for _ in range(20):
        await asyncio.sleep(0)


async def test_capacity_one_serializes_in_fifo_order() -> None:
    clock = FakeClock()
    gate = AsyncioGate(capacity=1, max_depth=10, clock=clock)
    order: list[str] = []
    release_first = asyncio.Event()

    async def holder(request_id: str, *, block: bool) -> None:
        async with _slot(gate, request_id) as ticket:
            order.append(request_id)
            if block:
                await release_first.wait()
            clock.advance(0.25)
            assert ticket.wait_ms >= 0

    tasks = [asyncio.create_task(holder("r0", block=True))]
    await _settle()
    tasks += [asyncio.create_task(holder(rid, block=False)) for rid in ("r1", "r2", "r3")]
    await _settle()

    snapshot = gate.snapshot()
    assert snapshot.in_flight == 1
    assert snapshot.queued == 3
    assert [w.position for w in snapshot.waiting] == [1, 2, 3]

    release_first.set()
    await asyncio.gather(*tasks)

    assert order == ["r0", "r1", "r2", "r3"]
    final = gate.snapshot()
    assert final.in_flight == 0
    assert final.queued == 0
    assert final.admitted_total == 4


async def test_capacity_n_admits_up_to_capacity() -> None:
    clock = FakeClock()
    gate = AsyncioGate(capacity=3, max_depth=10, clock=clock)
    release = asyncio.Event()

    async def holder(request_id: str) -> None:
        async with _slot(gate, request_id):
            await release.wait()

    tasks = [asyncio.create_task(holder(f"r{i}")) for i in range(5)]
    await _settle()

    snapshot = gate.snapshot()
    assert snapshot.capacity == 3
    assert snapshot.in_flight == 3
    assert snapshot.queued == 2
    assert snapshot.available_slots == 0

    release.set()
    await asyncio.gather(*tasks)
    assert gate.snapshot().in_flight == 0


async def test_max_depth_rejects_with_queue_full() -> None:
    clock = FakeClock()
    gate = AsyncioGate(capacity=1, max_depth=2, clock=clock)
    release = asyncio.Event()

    async def holder(request_id: str) -> None:
        async with _slot(gate, request_id):
            await release.wait()

    tasks = [asyncio.create_task(holder("r0"))]
    await _settle()
    tasks += [asyncio.create_task(holder(rid)) for rid in ("r1", "r2")]
    await _settle()

    with pytest.raises(ProxyError) as excinfo:
        async with _slot(gate, "r3"):
            pass

    assert excinfo.value.kind is ErrorKind.QUEUE_FULL
    assert excinfo.value.http_status == 429
    assert excinfo.value.code == "429"
    assert excinfo.value.retryable is True
    assert excinfo.value.retry_after_s is not None
    assert gate.snapshot().rejected_full_total == 1

    release.set()
    await asyncio.gather(*tasks)


async def test_max_wait_times_out_with_queue_timeout() -> None:
    clock = FakeClock()
    gate = AsyncioGate(capacity=1, max_depth=10, clock=clock)
    release = asyncio.Event()

    async def holder() -> None:
        async with _slot(gate, "r0"):
            await release.wait()

    task = asyncio.create_task(holder())
    await _settle()

    with pytest.raises(ProxyError) as excinfo:
        async with _slot(gate, "r1", max_wait_s=0.05):
            pass

    assert excinfo.value.kind is ErrorKind.QUEUE_TIMEOUT
    assert excinfo.value.retryable is True
    snapshot = gate.snapshot()
    assert snapshot.rejected_timeout_total == 1
    assert snapshot.queued == 0

    release.set()
    await task


async def test_waiter_cancelled_before_admission_never_consumes_capacity() -> None:
    clock = FakeClock()
    gate = AsyncioGate(capacity=1, max_depth=10, clock=clock)
    release = asyncio.Event()
    admitted: list[str] = []

    async def holder(request_id: str) -> None:
        async with _slot(gate, request_id):
            admitted.append(request_id)
            await release.wait()

    first = asyncio.create_task(holder("r0"))
    await _settle()

    second = asyncio.create_task(holder("r1"))
    await _settle()
    assert gate.snapshot().queued == 1

    second.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second
    await _settle()

    snapshot = gate.snapshot()
    assert snapshot.queued == 0
    assert snapshot.cancelled_while_waiting_total == 1

    release.set()
    await first
    assert admitted == ["r0"]
    assert gate.snapshot().in_flight == 0


async def test_slot_released_when_body_raises() -> None:
    clock = FakeClock()
    gate = AsyncioGate(capacity=1, max_depth=10, clock=clock)

    with pytest.raises(RuntimeError):
        async with _slot(gate, "r0"):
            raise RuntimeError("boom")

    assert gate.snapshot().in_flight == 0
    assert gate.snapshot().available_slots == 1

    # The freed slot is immediately usable.
    async with _slot(gate, "r1"):
        assert gate.snapshot().in_flight == 1


async def test_randomised_acquire_cancel_loop_leaks_nothing() -> None:
    clock = FakeClock()
    gate = AsyncioGate(capacity=2, max_depth=4, clock=clock)
    rng = random.Random(20260921)

    async def worker(request_id: str) -> None:
        try:
            async with _slot(gate, request_id, max_wait_s=0.2):
                await asyncio.sleep(0)
                clock.advance(0.01)
        except ProxyError, asyncio.CancelledError:
            pass

    for i in range(1000):
        tasks = [asyncio.create_task(worker(f"r{i}-{j}")) for j in range(rng.randint(1, 4))]
        if rng.random() < 0.5:
            await _settle()
            for task in tasks:
                if rng.random() < 0.4:
                    task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    await _settle()
    snapshot = gate.snapshot()
    assert snapshot.in_flight == 0
    assert snapshot.queued == 0
    assert snapshot.available_slots == snapshot.capacity
