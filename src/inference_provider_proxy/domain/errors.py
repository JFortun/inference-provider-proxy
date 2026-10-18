"""The error taxonomy: exactly one category per failure."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ErrorKind(StrEnum):
    """The closed set of failure categories surfaced by the dashboard."""

    PROVIDER_BUSY = "provider_busy"
    QUEUE_TIMEOUT = "queue_timeout"
    QUEUE_FULL = "queue_full"
    UPSTREAM_AUTH = "upstream_auth"
    EDGE_BLOCK = "edge_block"
    NOT_RESERVED = "not_reserved"
    UPSTREAM_5XX = "upstream_5xx"
    UPSTREAM_UNREACHABLE = "upstream_unreachable"
    CLIENT_ABORTED = "client_aborted"
    UNKNOWN = "unknown"


@dataclass(eq=False)
class ProxyError(Exception):
    """A classified failure rendered to clients in the OpenAI dialect.

    ``code`` is always a string — the provider itself returns ``"429"`` as a
    string, so matching and rendering are string-safe. When the condition is not
    fully understood (``unknown``, ``not_reserved``), ``provider_body`` carries
    the provider's own bytes for pass-through rather than rewriting.
    """

    kind: ErrorKind
    http_status: int
    code: str
    message: str
    retryable: bool = False
    retry_after_s: float | None = None
    provider_status: int | None = None
    provider_correlation_id: str | None = None
    provider_body: bytes | None = None
    # Queue transparency for error responses too (contract: transparency
    # headers on *all* responses, including errors). ``None`` means "not
    # known", which the HTTP layer renders as zero/omitted.
    queue_wait_ms: int | None = None
    queue_position: int | None = None

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)
