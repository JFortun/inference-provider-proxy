"""Background upstream refresher.

Refreshes the entitlement view and the model catalogue on a timer without ever
touching the gate, so it can never consume chat capacity. When a
refresh fails the last known value is kept and marked stale rather than being
cleared.

A dead provider must not bury the log: the first failure and every tenth after
it are WARNINGs, the rest are DEBUG, and recovery is announced once.
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
    """Periodically refreshes the upstream view: usage and the model catalogue."""

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
        self._consecutive_failures = 0

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())
            logger.info("subscription poller started interval=%.0fs", self._interval_s)

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
        logger.info("subscription poller stopped")

    async def refresh_once(self) -> None:
        """One refresh attempt; failures mark the view stale."""
        try:
            info = await self._provider.fetch_usage()
            models = await self._provider.list_models()
        except Exception as exc:  # noqa: BLE001 - any failure means "stale"
            self._consecutive_failures += 1
            failures = self._consecutive_failures
            log = logger.warning if failures == 1 or failures % 10 == 0 else logger.debug
            log("subscription refresh failed: %s (attempt %d)", type(exc).__name__, failures)
            self._subscription.mark_stale(True)
            return
        if self._consecutive_failures:
            logger.info(
                "subscription refresh recovered after %d failed attempts",
                self._consecutive_failures,
            )
            self._consecutive_failures = 0
        logger.debug("subscription refresh ok pool=%s stale=%s", info.pool, info.stale)
        self._subscription.update(info)
        # The dashboard's catalogue must not depend on a client asking.
        self._subscription.set_models(models, checked_at=self._clock.now())

    async def _run(self) -> None:
        while True:
            await self.refresh_once()
            await asyncio.sleep(self._interval_s)
