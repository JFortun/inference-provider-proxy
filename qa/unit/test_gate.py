"""Unit tests for the asyncio FIFO gate.

The gate is the load-bearing component: these tests prove it.
"""

from __future__ import annotations

import asyncio
import logging
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
    for _ in range(20):
        await asyncio.sleep(0)


async def _hold(gate: AsyncioGate, release: asyncio.Event, request_id: str, admitted=None) -> None:
    async with _slot(gate, request_id):
        if admitted is not None:
            admitted.append(request_id)
        await release.wait()


async def _pass_through(gate: AsyncioGate, request_id: str) -> None:
    async with _slot(gate, request_id):
        pass


async def _fifo_slot(
    gate: AsyncioGate,
    clock: FakeClock,
    order: list[str],
    release: asyncio.Event,
    request_id: str,
    *,
    block: bool,
) -> None:
    async with _slot(gate, request_id) as ticket:
        order.append(request_id)
        if block:
            await release.wait()
        clock.advance(0.25)
        assert ticket.wait_ms >= 0


async def test_capacity_one_when_more_arrive_then_serializes_them_behind_one_holder() -> None:
    clock = FakeClock()
    gate = AsyncioGate(capacity=1, max_depth=10, clock=clock)
    order: list[str] = []
    release = asyncio.Event()
    tasks = [asyncio.create_task(_fifo_slot(gate, clock, order, release, "r0", block=True))]
    await _settle()
    tasks += [
        asyncio.create_task(_fifo_slot(gate, clock, order, release, rid, block=False))
        for rid in ("r1", "r2", "r3")
    ]
    await _settle()

    snapshot = gate.snapshot()
    release.set()
    await asyncio.gather(*tasks)

    assert snapshot.in_flight == 1
    assert snapshot.queued == 3
    assert [w.position for w in snapshot.waiting] == [1, 2, 3]


async def test_waiters_when_admitted_on_release_then_enter_in_fifo_order() -> None:
    clock = FakeClock()
    gate = AsyncioGate(capacity=1, max_depth=10, clock=clock)
    order: list[str] = []
    release = asyncio.Event()
    tasks = [asyncio.create_task(_fifo_slot(gate, clock, order, release, "r0", block=True))]
    await _settle()
    tasks += [
        asyncio.create_task(_fifo_slot(gate, clock, order, release, rid, block=False))
        for rid in ("r1", "r2", "r3")
    ]
    await _settle()

    release.set()
    await asyncio.gather(*tasks)
    final = gate.snapshot()

    assert order == ["r0", "r1", "r2", "r3"]
    assert final.in_flight == 0
    assert final.queued == 0
    assert final.admitted_total == 4


async def test_a_long_queue_wait_when_admitted_then_is_logged_at_info(caplog) -> None:
    clock = FakeClock()
    gate = AsyncioGate(capacity=1, max_depth=10, clock=clock)
    release = asyncio.Event()
    first = asyncio.create_task(_hold(gate, release, "r0"))
    await _settle()
    second = asyncio.create_task(_pass_through(gate, "r1"))
    await _settle()
    queued = gate.snapshot().queued
    clock.advance(5.0)

    with caplog.at_level(logging.DEBUG, logger="ipp.gate"):
        release.set()
        await asyncio.gather(first, second)

    admitted = [
        record
        for record in caplog.records
        if record.name == "ipp.gate" and record.levelno == logging.INFO
    ]
    message = admitted[0].getMessage()
    assert queued == 1
    assert len(admitted) == 1
    assert "admitted after waiting" in message
    assert "request_id=r1" in message
    assert "waited_ms=5000" in message


async def test_capacity_n_when_more_waiters_than_slots_then_admits_up_to_capacity() -> None:
    gate = AsyncioGate(capacity=3, max_depth=10, clock=FakeClock())
    release = asyncio.Event()
    tasks = [asyncio.create_task(_hold(gate, release, f"r{i}")) for i in range(5)]
    await _settle()
    snapshot = gate.snapshot()

    release.set()
    await asyncio.gather(*tasks)
    drained = gate.snapshot()

    assert snapshot.capacity == 3
    assert snapshot.in_flight == 3
    assert snapshot.queued == 2
    assert snapshot.available_slots == 0
    assert drained.in_flight == 0


async def test_max_depth_when_queue_is_full_then_rejects_the_next_waiter() -> None:
    gate = AsyncioGate(capacity=1, max_depth=2, clock=FakeClock())
    release = asyncio.Event()
    tasks = [asyncio.create_task(_hold(gate, release, "r0"))]
    await _settle()
    tasks += [asyncio.create_task(_hold(gate, release, rid)) for rid in ("r1", "r2")]
    await _settle()

    with pytest.raises(ProxyError) as excinfo:
        async with _slot(gate, "r3"):
            pass
    release.set()
    await asyncio.gather(*tasks)

    assert excinfo.value.kind is ErrorKind.QUEUE_FULL
    assert excinfo.value.http_status == 429
    assert excinfo.value.code == "429"
    assert excinfo.value.retryable is True
    assert excinfo.value.retry_after_s is not None
    assert gate.snapshot().rejected_full_total == 1


async def test_max_wait_when_exceeded_then_times_out_the_queueing_waiter() -> None:
    gate = AsyncioGate(capacity=1, max_depth=10, clock=FakeClock())
    release = asyncio.Event()
    holder = asyncio.create_task(_hold(gate, release, "r0"))
    await _settle()

    with pytest.raises(ProxyError) as excinfo:
        async with _slot(gate, "r1", max_wait_s=0.05):
            pass
    release.set()
    await holder

    assert excinfo.value.kind is ErrorKind.QUEUE_TIMEOUT
    assert excinfo.value.retryable is True
    snapshot = gate.snapshot()
    assert snapshot.rejected_timeout_total == 1
    assert snapshot.queued == 0


async def _hold_recording(
    gate: AsyncioGate, release: asyncio.Event, admitted: list[str], request_id: str
) -> None:
    async with _slot(gate, request_id):
        admitted.append(request_id)
        await release.wait()


async def test_waiter_when_cancelled_before_admission_then_never_consumes_capacity() -> None:
    clock = FakeClock()
    gate = AsyncioGate(capacity=1, max_depth=10, clock=clock)
    release = asyncio.Event()
    admitted: list[str] = []
    first = asyncio.create_task(_hold_recording(gate, release, admitted, "r0"))
    await _settle()
    second = asyncio.create_task(_hold_recording(gate, release, admitted, "r1"))
    await _settle()
    queued = gate.snapshot().queued

    second.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second
    await _settle()
    after_cancel = gate.snapshot()
    release.set()
    await first
    final = gate.snapshot()

    assert queued == 1
    assert after_cancel.queued == 0
    assert after_cancel.cancelled_while_waiting_total == 1
    assert admitted == ["r0"]
    assert final.in_flight == 0


async def test_slot_when_body_raises_then_is_released() -> None:
    gate = AsyncioGate(capacity=1, max_depth=10, clock=FakeClock())

    with pytest.raises(RuntimeError):
        async with _slot(gate, "r0"):
            raise RuntimeError("boom")
    freed = gate.snapshot()
    async with _slot(gate, "r1"):
        reused = gate.snapshot().in_flight

    assert freed.in_flight == 0
    assert freed.available_slots == 1
    assert reused == 1


async def _random_worker(gate: AsyncioGate, clock: FakeClock, request_id: str) -> None:
    try:
        async with _slot(gate, request_id, max_wait_s=0.2):
            await asyncio.sleep(0)
            clock.advance(0.01)
    except ProxyError, asyncio.CancelledError:
        pass


async def test_randomised_acquire_cancel_loop_when_run_then_leaks_nothing() -> None:
    clock = FakeClock()
    gate = AsyncioGate(capacity=2, max_depth=4, clock=clock)
    rng = random.Random(20260921)

    for i in range(1000):
        tasks = [
            asyncio.create_task(_random_worker(gate, clock, f"r{i}-{j}"))
            for j in range(rng.randint(1, 4))
        ]
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
