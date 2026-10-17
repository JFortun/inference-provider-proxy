"""Request-id assignment and redacted access logging.

Pure ASGI middleware: it must not buffer the response body, or streaming and
client-disconnect detection would break (R2). There is **no authentication**
(FR-016); the only credential the proxy holds is the outbound provider key,
which is never logged or echoed (FR-018). Because we log only method, path,
status, client address, and duration — never headers or bodies — the provider
key cannot leak into a log record.
"""

from __future__ import annotations

import logging
import time
from uuid import uuid4

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger("ipp.access")

# Browser-facing pages are polled and reloaded; only API traffic belongs on the
# INFO lane. Everything else still lands in the log, one line below.
_QUIET_PATHS = frozenset({"/", "/styles.css", "/dashboard", "/healthz"})


class RequestContextMiddleware:
    """Assign a request id and emit one redacted access-log line per request."""

    def __init__(self, app: ASGIApp, *, access_log: bool = True) -> None:
        self.app = app
        self.access_log = access_log

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = uuid4().hex
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        started = time.monotonic()
        status = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                MutableHeaders(scope=message)["X-Proxy-Request-Id"] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            if self.access_log:
                client = scope.get("client")
                address = client[0] if client else "unknown"
                duration_ms = int((time.monotonic() - started) * 1000)
                emit = logger.debug if scope.get("path") in _QUIET_PATHS else logger.info
                emit(
                    "%s %s %s %dms client=%s request_id=%s",
                    scope.get("method"),
                    scope.get("path"),
                    status,
                    duration_ms,
                    address,
                    request_id,
                )
