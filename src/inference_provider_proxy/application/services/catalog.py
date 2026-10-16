"""Ungated catalogue and usage retrieval with a short TTL cache (FR-002, FR-014).

Neither path ever touches the gate: metadata must not wait behind, or consume,
chat capacity (FR-014, SC-012).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from inference_provider_proxy.domain.models import SubscriptionInfo

if TYPE_CHECKING:
    from inference_provider_proxy.application.ports.clock import ClockPort
    from inference_provider_proxy.application.ports.metrics import MetricsPort, SubscriptionPort
    from inference_provider_proxy.application.ports.provider import LLMProviderPort
    from inference_provider_proxy.config import Settings


class CatalogService:
    """Implements ``CatalogUseCase``."""

    def __init__(
        self,
        *,
        provider: LLMProviderPort,
        metrics: MetricsPort,
        settings: Settings,
        clock: ClockPort,
        subscription: SubscriptionPort,
    ) -> None:
        self._provider = provider
        self._metrics = metrics
        self._settings = settings
        self._clock = clock
        self._subscription = subscription
        self._cache: dict[str, tuple[float, dict[str, Any]]] = {}

    async def models(self) -> dict[str, Any]:
        cached = self._get("models")
        if cached is not None:
            return cached
        models = await self._provider.list_models()
        payload = {
            "object": "list",
            "data": [
                {"id": model.id, "object": "model", "context_length": model.context_length}
                for model in models
            ],
        }
        self._metrics.set_models(models, checked_at=self._clock.now())
        self._put("models", payload)
        return payload

    async def usage(self) -> dict[str, Any]:
        cached = self._get("usage")
        if cached is not None:
            return cached
        info = await self._provider.fetch_usage()
        self._subscription.update(info)
        payload = self._subscription_payload(info)
        self._put("usage", payload)
        return payload

    # -- cache ---------------------------------------------------------------

    def _get(self, key: str) -> dict[str, Any] | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        stored_at, payload = entry
        if self._clock.monotonic() - stored_at > self._settings.metadata_cache_ttl:
            return None
        return payload

    def _put(self, key: str, payload: dict[str, Any]) -> None:
        self._cache[key] = (self._clock.monotonic(), payload)

    @staticmethod
    def _subscription_payload(info: SubscriptionInfo) -> dict[str, Any]:
        return {
            "object": "subscription",
            "pool": info.pool,
            "status": info.status,
            "expires_at": info.expires_at.isoformat() if info.expires_at else None,
            "seconds_until_expiry": info.seconds_until_expiry,
            "key_name": info.key_name,
        }
