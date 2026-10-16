"""Background subscription refresher (FR-015, T041).

Refreshes the entitlement view on a timer without ever touching the gate, so it
can never consume chat capacity (FR-014). When a refresh fails the last known
value is kept and marked stale rather than being cleared.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inference_provider_proxy.application.ports.clock import ClockPort
    from inference_provider_proxy.application.ports.metrics import SubscriptionPort
    from inference_provider_proxy.application.ports.provider import LLMProviderPort

logger = logging.getLogger("ipp.subscription")


class SubscriptionPoller:
    """Periodically refreshes ``/v1/usage`` state in the background."""

    def __init__(
        self,
        *,
        provider: LLMProviderPort,
        subscription: SubscriptionPort,
        interval_s: float,
        clock: ClockPort,
    ) -> None:
        self._provider = provider
        self._subscription = subscription
        self._interval_s = max(1.0, interval_s)
        self._clock = clock
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def refresh_once(self) -> None:
        """One refresh attempt; failures mark the view stale (FR-015)."""
        try:
            info = await self._provider.fetch_usage()
        except Exception as exc:  # noqa: BLE001 - any failure means "stale"
            logger.warning("subscription refresh failed: %s", type(exc).__name__)
            self._subscription.mark_stale(True)
            return
        self._subscription.update(info)

    async def _run(self) -> None:
        while True:
            await self.refresh_once()
            await asyncio.sleep(self._interval_s)
