"""The serialized chat path (FR-004–FR-013, FR-048 retry budget).

The gate slot is acquired before the provider call and released only after the
last byte is delivered (FR-008), so a mid-response failure or client disconnect
cannot leak capacity.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypeVar

from inference_provider_proxy.application.ports.inbound import ChatResult, ChatStreamResult
from inference_provider_proxy.application.ports.provider import ProviderStream, TokenUsage
from inference_provider_proxy.domain.errors import ErrorKind, ProxyError
from inference_provider_proxy.domain.metrics import ErrorEpisode, RequestRecord

if TYPE_CHECKING:
    from inference_provider_proxy.application.ports.clock import ClockPort
    from inference_provider_proxy.application.ports.gate import ConcurrencyGatePort, QueueTicket
    from inference_provider_proxy.application.ports.metrics import MetricsPort
    from inference_provider_proxy.application.ports.provider import LLMProviderPort
    from inference_provider_proxy.config import Settings
    from inference_provider_proxy.domain.models import ChatRequest

logger = logging.getLogger("ipp.chat")

_MAX_RETRY_DELAY_S = 5.0
_ABORT_STATUS = 499

_T = TypeVar("_T")


class ChatService:
    """Implements ``ChatCompletionUseCase``."""

    def __init__(
        self,
        *,
        gate: ConcurrencyGatePort,
        provider: LLMProviderPort,
        metrics: MetricsPort,
        clock: ClockPort,
        settings: Settings,
    ) -> None:
        self._gate = gate
        self._provider = provider
        self._metrics = metrics
        self._clock = clock
        self._settings = settings

    async def complete(self, request: ChatRequest) -> ChatResult:
        self._metrics.note_client(request.client)
        started = self._clock.monotonic()
        slot_cm = self._gate.slot(
            request_id=request.request_id,
            client=request.client,
            model=request.model,
            streaming=False,
            max_wait_s=self._settings.queue_timeout,
        )
        ticket: QueueTicket | None = None
        try:
            ticket = await slot_cm.__aenter__()
            completion = await self._call_with_retries(request.request_id, request.body)
        except asyncio.CancelledError:
            # The client left before the response started (FR-009, FR-010): record
            # the abort rather than let the request vanish or be counted as
            # completed; the finally below still frees the slot.
            self._record_client_gone(request, ticket, started, streaming=False)
            raise
        except ProxyError as exc:
            # Also catches a gate rejection (queue full or wait bound exceeded),
            # which never ran: it is still a classified failure and must be
            # surfaced in the dashboard taxonomy (FR-020, FR-024).
            self._record_failure(request, exc, ticket, started, streaming=False)
            raise
        except Exception as exc:  # noqa: BLE001 - converted to a classified error
            error = self._unknown(exc)
            self._record_failure(request, error, ticket, started, streaming=False)
            raise error from exc
        finally:
            if ticket is not None:
                await self._close(slot_cm)

        self._emit(
            request,
            streaming=False,
            status=completion.status_code,
            started=started,
            ticket=ticket,
            usage=completion.usage,
            provider_id=completion.provider_id,
            provider_correlation_id=completion.provider_correlation_id,
        )
        return ChatResult(
            request_id=request.request_id,
            payload=completion.payload,
            queue_wait_ms=ticket.wait_ms,
            queue_position=ticket.position,
            status_code=completion.status_code,
        )

    async def stream(self, request: ChatRequest) -> ChatStreamResult:
        self._metrics.note_client(request.client)
        started = self._clock.monotonic()
        slot_cm = self._gate.slot(
            request_id=request.request_id,
            client=request.client,
            model=request.model,
            streaming=True,
            max_wait_s=self._settings.queue_timeout,
        )
        try:
            ticket = await slot_cm.__aenter__()
        except asyncio.CancelledError:
            self._record_client_gone(request, None, started, streaming=True)
            raise
        except ProxyError as exc:
            # A gate rejection consumes no capacity and no upstream call, but it
            # is still a classified failure (FR-020, FR-024).
            self._record_failure(request, exc, None, started, streaming=True)
            raise
        try:
            stream_cm, stream = await self._with_retries(
                request.request_id,
                lambda: self._open_stream_once(request.body),
            )
        except asyncio.CancelledError:
            await self._close(slot_cm)
            self._record_client_gone(request, ticket, started, streaming=True)
            raise
        except ProxyError as exc:
            await self._close(slot_cm)
            self._record_failure(request, exc, ticket, started, streaming=True)
            raise
        except Exception as exc:  # noqa: BLE001 - converted to a classified error
            await self._close(slot_cm)
            error = self._unknown(exc)
            self._record_failure(request, error, ticket, started, streaming=True)
            raise error from exc

        chunks = self._iter_stream(slot_cm, stream_cm, stream, request, ticket, started)
        return ChatStreamResult(
            request_id=request.request_id,
            chunks=chunks,
            queue_wait_ms=ticket.wait_ms,
            queue_position=ticket.position,
        )

    # -- internals -----------------------------------------------------------

    async def _iter_stream(
        self,
        slot_cm: Any,
        stream_cm: Any,
        stream: ProviderStream,
        request: ChatRequest,
        ticket: QueueTicket,
        started: float,
    ) -> AsyncIterator[bytes]:
        ttfb_ms: int | None = None
        outcome = "success"
        error: ProxyError | None = None
        try:
            async for chunk in stream.aiter_bytes():
                if ttfb_ms is None:
                    ttfb_ms = self._elapsed_ms(started)
                    self._gate.mark_ttfb(request_id=request.request_id, ttfb_ms=ttfb_ms)
                yield chunk
        except GeneratorExit, asyncio.CancelledError:
            outcome = "aborted"
            raise
        except ProxyError as exc:
            outcome = "failed"
            error = exc
            raise
        except Exception as exc:  # noqa: BLE001 - converted to a classified error
            outcome = "failed"
            error = self._unknown(exc)
            raise error from exc
        finally:
            # Closing the upstream stream aborts the provider request eagerly, so
            # a disconnect cannot leave the provider busy (FR-010, T031). That
            # teardown is bounded so a stalled upstream cannot hold the single
            # slot past the configured disconnect-to-release bound; the slot is
            # released only afterwards, and only after the outcome is recorded,
            # so it spans the whole response (FR-008).
            await self._abort_upstream(stream_cm)
            if outcome == "success":
                self._emit(
                    request,
                    streaming=True,
                    status=200,
                    started=started,
                    ticket=ticket,
                    ttfb_ms=ttfb_ms,
                    usage=stream.usage,
                    provider_id=stream.provider_id,
                    provider_correlation_id=stream.provider_correlation_id,
                )
            elif outcome == "aborted":
                self._emit(
                    request,
                    streaming=True,
                    status=_ABORT_STATUS,
                    started=started,
                    ticket=ticket,
                    aborted=True,
                    error=self._aborted_error(),
                    ttfb_ms=ttfb_ms,
                )
            elif error is not None:
                self._emit(
                    request,
                    streaming=True,
                    status=error.http_status,
                    started=started,
                    ticket=ticket,
                    error=error,
                    ttfb_ms=ttfb_ms,
                )
            await self._close(slot_cm)

    async def _call_with_retries(self, request_id: str, body: dict[str, Any]) -> Any:
        return await self._with_retries(request_id, lambda: self._provider.complete(body))

    async def _open_stream_once(self, body: dict[str, Any]) -> tuple[Any, ProviderStream]:
        """Open one upstream stream, returning its context manager and stream."""
        stream_cm = self._provider.open_stream(body)
        stream = await stream_cm.__aenter__()
        return stream_cm, stream

    async def _with_retries(self, request_id: str, call: Callable[[], Awaitable[_T]]) -> _T:
        """Run ``call`` under the bounded retry budget (FR-012).

        Transient provider failures are retried, honouring ``retry-after`` and
        only while capacity permits (the slot is already held here), until the
        budget is exhausted, at which point the classified error is raised so
        the request fails fast rather than waiting for recovery. Each attempt is
        announced, and giving up after having retried is an ERROR, because by
        then the client is about to be told the provider is unusable.
        """
        attempt = 0
        while True:
            try:
                return await call()
            except ProxyError as exc:
                if not exc.retryable or attempt >= self._settings.max_retries:
                    if attempt:
                        logger.error(
                            "retry budget exhausted after %d attempts request_id=%s "
                            "kind=%s provider_status=%s",
                            attempt,
                            request_id,
                            exc.kind,
                            exc.provider_status,
                        )
                    raise
                attempt += 1
                self._metrics.record_retry()
                logger.warning(
                    "retry %d/%d request_id=%s kind=%s provider_status=%s retry_after=%s",
                    attempt,
                    self._settings.max_retries,
                    request_id,
                    exc.kind,
                    exc.provider_status,
                    exc.retry_after_s,
                )
                if exc.retry_after_s:
                    await asyncio.sleep(min(exc.retry_after_s, _MAX_RETRY_DELAY_S))

    def _emit(
        self,
        request: ChatRequest,
        *,
        streaming: bool,
        status: int,
        started: float,
        ticket: QueueTicket | None = None,
        aborted: bool = False,
        error: ProxyError | None = None,
        ttfb_ms: int | None = None,
        usage: TokenUsage | None = None,
        provider_id: str | None = None,
        provider_correlation_id: str | None = None,
    ) -> None:
        """Record one request's outcome and, when it failed, its error episode.

        A gate rejection arrives before a ticket exists (FR-006) and already
        carries its own queue wait; anything else takes the ticket's values. The
        resolved values are written back onto the error too, because that object
        is raised to the client and its transparency headers are read there
        (FR-013).
        """
        if ticket is not None:
            queue_wait_ms: int | None = ticket.wait_ms
            queue_position = ticket.position
        else:
            queue_wait_ms = error.queue_wait_ms if error is not None else 0
            queue_position = error.queue_position if error is not None else None
        if error is not None:
            error.queue_wait_ms = queue_wait_ms
            error.queue_position = queue_position

        if aborted:
            error_kind: ErrorKind | None = ErrorKind.CLIENT_ABORTED
        elif error is not None:
            error_kind = error.kind
        else:
            error_kind = None

        self._metrics.record_request(
            RequestRecord(
                request_id=request.request_id,
                at=self._now(),
                client=request.client.address,
                model=request.model,
                streaming=streaming,
                status=status,
                queue_wait_ms=queue_wait_ms or 0,
                queue_position=queue_position,
                duration_ms=self._elapsed_ms(started),
                aborted=aborted,
                ttfb_ms=ttfb_ms,
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
                provider_id=provider_id,
                provider_correlation_id=provider_correlation_id,
                error_kind=error_kind,
            )
        )
        if error is None:
            return

        # in_flight <= capacity on a provider throttle shows the gate never
        # over-admitted, so `provider_busy` is not read as a gate fault (FR-008:
        # the slot is still held here; a gate rejection is recorded pre-admission).
        gate = self._gate.snapshot()
        self._metrics.record_error(
            ErrorEpisode(
                at=self._now(),
                kind=error.kind,
                status=status,
                model=request.model,
                client=request.client.address,
                provider_correlation_id=error.provider_correlation_id,
                gate_in_flight=gate.in_flight,
                gate_capacity=gate.capacity,
            )
        )
        if error.provider_status is not None or error.kind in {
            ErrorKind.UPSTREAM_UNREACHABLE,
            ErrorKind.UPSTREAM_5XX,
        }:
            self._metrics.set_upstream_error(error.message)

    def _record_failure(
        self,
        request: ChatRequest,
        error: ProxyError,
        ticket: QueueTicket | None,
        started: float,
        *,
        streaming: bool,
        ttfb_ms: int | None = None,
        aborted: bool = False,
    ) -> None:
        self._emit(
            request,
            streaming=streaming,
            status=error.http_status,
            started=started,
            ticket=ticket,
            error=error,
            ttfb_ms=ttfb_ms,
            aborted=aborted,
        )

    def _record_client_gone(
        self,
        request: ChatRequest,
        ticket: QueueTicket | None,
        started: float,
        *,
        streaming: bool,
    ) -> None:
        """Record a disconnect seen before the response started (FR-009, FR-010)."""
        logger.warning(
            "client disconnected while %s request_id=%s",
            "queued" if ticket is None else "in flight",
            request.request_id,
        )
        self._record_failure(
            request, self._aborted_error(), ticket, started, streaming=streaming, aborted=True
        )

    async def _abort_upstream(self, stream_cm: Any) -> None:
        """Close the upstream stream within the disconnect-to-release bound.

        The bound is a maximum, not a target: a teardown that stalls, times out,
        or is cancelled by the disconnect itself must never skip the capacity
        release that follows (FR-010).
        """
        try:
            await asyncio.wait_for(
                self._close(stream_cm), timeout=self._settings.disconnect_release
            )
        except Exception, asyncio.CancelledError:  # noqa: BLE001 - release must proceed
            pass

    @staticmethod
    async def _close(cm: Any) -> None:
        try:
            await cm.__aexit__(None, None, None)
        except Exception:  # noqa: BLE001 - closing must never mask the outcome
            pass

    @staticmethod
    def _aborted_error() -> ProxyError:
        return ProxyError(
            kind=ErrorKind.CLIENT_ABORTED,
            http_status=_ABORT_STATUS,
            code="client_aborted",
            message="The client disconnected before the response completed.",
            retryable=False,
        )

    @staticmethod
    def _unknown(exc: Exception) -> ProxyError:
        # The traceback exists only here: the caller re-raises a ProxyError, so
        # without this line the original failure would vanish from the log.
        logger.error(
            "unexpected failure classified as unknown: %s",
            type(exc).__name__,
            exc_info=exc,
        )
        return ProxyError(
            kind=ErrorKind.UNKNOWN,
            http_status=502,
            code="unknown",
            message=f"Unexpected proxy failure: {type(exc).__name__}.",
            retryable=False,
        )

    def _elapsed_ms(self, started: float) -> int:
        return int((self._clock.monotonic() - started) * 1000)

    def _now(self) -> datetime:
        return self._clock.now()
