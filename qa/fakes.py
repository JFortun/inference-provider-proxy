"""Shared test doubles."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta


class FakeClock:
    """A deterministic clock: monotonic seconds for durations, wall time for display."""

    def __init__(
        self,
        *,
        start: float = 0.0,
        wall: datetime | None = None,
    ) -> None:
        self._mono = start
        self._wall = wall or datetime(2026, 1, 1, tzinfo=UTC)

    def now(self) -> datetime:
        return self._wall

    def monotonic(self) -> float:
        return self._mono

    def advance(self, seconds: float) -> None:
        self._mono += seconds
        self._wall += timedelta(seconds=seconds)
