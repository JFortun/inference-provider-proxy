"""The HTML dashboard: the ``GET /dashboard`` document, rendered as pages.

One shell (``GET /``) carries the header and the sidebar; each section has its
own route so the UI is read one category at a time. Reads metrics only, so
like ``/dashboard`` it is never blocked by the gate. The browser gets plain
HTML and a stylesheet — no client-side script.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, Response

from inference_provider_proxy.adapters.inbound.http.deps import services_of
from inference_provider_proxy.adapters.inbound.http.serialization import document_to_dict
from inference_provider_proxy.adapters.inbound.http.ui.assets import STYLESHEET
from inference_provider_proxy.adapters.inbound.http.ui.render import render_page

router = APIRouter()


def _page(name: str, request: Request) -> HTMLResponse:
    document = document_to_dict(services_of(request).dashboard.document())
    return HTMLResponse(render_page(name, document))


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return _page("dashboard", request)


@router.get("/live", response_class=HTMLResponse)
async def live(request: Request) -> HTMLResponse:
    return _page("live", request)


@router.get("/requests", response_class=HTMLResponse)
async def requests(request: Request) -> HTMLResponse:
    return _page("requests", request)


@router.get("/errors", response_class=HTMLResponse)
async def errors(request: Request) -> HTMLResponse:
    return _page("errors", request)


@router.get("/clients", response_class=HTMLResponse)
async def clients(request: Request) -> HTMLResponse:
    return _page("clients", request)


@router.get("/upstream", response_class=HTMLResponse)
async def upstream(request: Request) -> HTMLResponse:
    return _page("upstream", request)


@router.get("/styles.css", response_class=Response)
async def styles() -> Response:
    return Response(
        content=STYLESHEET,
        media_type="text/css",
        headers={"Cache-Control": "max-age=60"},
    )
