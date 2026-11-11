"""Gate edge cases: the constructor guard and the cancel-vs-admit race."""

from __future__ import annotations

import asyncio

import pytest

from inference_provider_proxy.adapters.outbound.gate.asyncio_gate import AsyncioGate
from inference_provider_proxy.domain.models import ClientIdentity
from qa.fakes import FakeClock

CLIENT = ClientIdentity(address="10.0.0.1", user_agent="pytest")


def _slot(gate: AsyncioGate, request_id: str):
    return gate.slot(
        request_id=request_id, client=CLIENT, model="m", streaming=False, max_wait_s=5.0
    )


async def _settle(turns: int = 20) -> None:
    for _ in range(turns):
        await asyncio.sleep(0)


async def _hold(gate: AsyncioGate, release: asyncio.Event) -> None:
    async with _slot(gate, "holder"):
        await release.wait()


async def _queue(gate: AsyncioGate) -> None:
    async with _slot(gate, "waiter"):
        await asyncio.sleep(0)


def test_capacity_below_one_when_constructed_then_is_rejected() -> None:
    capacity = 0

    with pytest.raises(ValueError) as excinfo:
        AsyncioGate(capacity=capacity, max_depth=1, clock=FakeClock())

    assert str(excinfo.value) == "capacity must be >= 1"


async def test_second_waiter_when_gate_is_full_then_queues_behind_the_holder() -> None:
    gate = AsyncioGate(capacity=1, max_depth=4, clock=FakeClock())
    release = asyncio.Event()
    asyncio.create_task(_hold(gate, release))
    await _settle()

    waiter = asyncio.create_task(_queue(gate))
    await _settle()
    queued = gate.snapshot().queued
    release.set()
    await waiter

    assert queued == 1
    assert gate.snapshot().in_flight == 0


async def test_waiter_when_cancelled_in_the_admit_turn_then_slot_is_handed_back() -> None:
    """Releasing the holder admits the queued waiter and marks its event ready, but the
    waiter has not resumed yet. Cancelling it in the same event-loop turn must hand the
    just-granted slot back rather than leak it; ``release.set()`` schedules the holder
    first, so the callback queued after it observes the admitted-waiter state.
    """
    gate = AsyncioGate(capacity=1, max_depth=4, clock=FakeClock())
    release = asyncio.Event()
    holder = asyncio.create_task(_hold(gate, release))
    await _settle()
    waiter = asyncio.create_task(_queue(gate))
    await _settle()
    assert gate.snapshot().queued == 1

    release.set()
    asyncio.get_running_loop().call_soon(waiter.cancel)
    await _settle()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    await holder
    snapshot = gate.snapshot()

    assert snapshot.in_flight == 0
    assert snapshot.queued == 0
