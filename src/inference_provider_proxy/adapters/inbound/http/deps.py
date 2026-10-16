"""Shared dependency container for the HTTP adapters.

Kept in its own module so route modules can import it without importing the app
factory (which would be circular). Holds ports only — never concrete outbound
adapters — so the inbound layer stays on the application boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi import Request

if TYPE_CHECKING:
    from inference_provider_proxy.application.ports.inbound import (
        CatalogUseCase,
        ChatCompletionUseCase,
        DashboardUseCase,
    )
    from inference_provider_proxy.application.ports.provider import LLMProviderPort
    from inference_provider_proxy.config import Settings


@dataclass(slots=True)
class Services:
    """Everything the inbound adapters need, wired once at composition time."""

    settings: Settings
    provider: LLMProviderPort
    chat: ChatCompletionUseCase
    catalog: CatalogUseCase
    dashboard: DashboardUseCase


def services_of(request: Request) -> Services:
    """Return the wired services for the running app."""
    return request.app.state.services
