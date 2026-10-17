"""The HTML dashboard page: the ``GET /dashboard`` document, rendered as a page (FR-019).

Reads metrics only, so like ``/dashboard`` it is never blocked by the gate. The
browser gets plain HTML and a stylesheet — no client-side script.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, Response

from inference_provider_proxy.adapters.inbound.http.deps import services_of
from inference_provider_proxy.adapters.inbound.http.serialization import document_to_dict
from inference_provider_proxy.adapters.inbound.http.ui.assets import STYLESHEET
from inference_provider_proxy.adapters.inbound.http.ui.render import render_dashboard

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    services = services_of(request)
    return HTMLResponse(render_dashboard(document_to_dict(services.dashboard.document())))


@router.get("/styles.css", response_class=Response)
async def styles() -> Response:
    return Response(
        content=STYLESHEET,
        media_type="text/css",
        headers={"Cache-Control": "max-age=60"},
    )
