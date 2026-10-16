"""Domain value models (pure: no I/O, no framework imports).

Field names and rules follow the pinned dashboard contract —
``inference_provider_proxy/contracts/dashboard.schema.json`` — wherever they
surface on ``GET /dashboard``. The request body is carried through opaquely;
the proxy never inspects it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class ClientIdentity:
    """Request origin, inferred from the network only — there is no auth."""

    address: str
    user_agent: str = "unknown"


@dataclass(slots=True)
class ChatRequest:
    """An incoming chat completion request plus its lifecycle metadata."""

    request_id: str
    client: ClientIdentity
    model: str
    stream: bool = False
    # The original body, forwarded to the provider verbatim (FR-001).
    body: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """An available model's identifier and context length."""

    id: str
    context_length: int | None = None


@dataclass(frozen=True, slots=True)
class SubscriptionInfo:
    """Provider entitlement, refreshed by the background poller (FR-015)."""

    pool: str | None = None
    status: str | None = None
    expires_at: datetime | None = None
    seconds_until_expiry: int | None = None
    key_name: str | None = None
    last_checked_at: datetime | None = None
    stale: bool = True
