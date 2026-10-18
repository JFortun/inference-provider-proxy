"""The dashboard route: one JSON document, never blocked by the gate."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from inference_provider_proxy.adapters.inbound.http.deps import services_of
from inference_provider_proxy.adapters.inbound.http.serialization import document_to_dict

router = APIRouter()


@router.get("/dashboard")
async def dashboard(request: Request) -> JSONResponse:
    services = services_of(request)
    return JSONResponse(content=document_to_dict(services.dashboard.document()))
