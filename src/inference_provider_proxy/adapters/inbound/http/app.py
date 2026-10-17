"""FastAPI app factory and route wiring (FR-003, FR-016, FR-021)."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from inference_provider_proxy import __version__
from inference_provider_proxy.adapters.inbound.http import (
    admin_routes,
    dashboard_routes,
    openai_routes,
    ui_routes,
)
from inference_provider_proxy.adapters.inbound.http.deps import Services
from inference_provider_proxy.adapters.inbound.http.error_handlers import register_error_handlers
from inference_provider_proxy.adapters.inbound.http.middleware import RequestContextMiddleware


def create_app(services: Services, *, lifespan: Any = None) -> FastAPI:
    """Build the ASGI application around already-wired services."""
    app = FastAPI(
        title="inference-provider-proxy",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.services = services

    register_error_handlers(app)
    app.add_middleware(RequestContextMiddleware, access_log=services.settings.access_log)

    app.include_router(admin_routes.router)
    app.include_router(openai_routes.router)
    app.include_router(dashboard_routes.router)
    app.include_router(ui_routes.router)
    return app
