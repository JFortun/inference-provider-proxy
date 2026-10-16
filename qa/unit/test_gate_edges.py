"""Gate edge cases: the constructor guard and the cancel-vs-admit race (FR-004, FR-010)."""

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


def test_capacity_below_one_is_rejected() -> None:
    with pytest.raises(ValueError):
        AsyncioGate(capacity=0, max_depth=1, clock=FakeClock())


async def test_cancelling_a_waiter_admitted_in_the_same_instant_releases_the_slot() -> None:
    gate = AsyncioGate(capacity=1, max_depth=4, clock=FakeClock())
    release = asyncio.Event()

    async def holder() -> None:
        async with _slot(gate, "holder"):
            await release.wait()

    async def waiter() -> None:
        async with _slot(gate, "waiter"):
            await asyncio.sleep(0)

    holder_task = asyncio.create_task(holder())
    await _settle()
    assert gate.snapshot().in_flight == 1

    waiter_task = asyncio.create_task(waiter())
    await _settle()
    assert gate.snapshot().queued == 1

    # Releasing the holder admits the queued waiter (its ticket is set) and marks
    # its event ready, but the waiter has not resumed yet. Cancelling it in the
    # same event-loop turn must hand the just-granted slot straight back rather
    # than leaking it (FR-010). ``release.set()`` schedules the holder first, so
    # the callback queued after it observes the admitted-waiter state.
    release.set()
    asyncio.get_running_loop().call_soon(waiter_task.cancel)
    await _settle()

    with pytest.raises(asyncio.CancelledError):
        await waiter_task
    await holder_task

    snapshot = gate.snapshot()
    assert snapshot.in_flight == 0
    assert snapshot.queued == 0
