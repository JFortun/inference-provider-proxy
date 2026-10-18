"""Driven port for time, so the gate is unit-testable against a fake clock."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Protocol


class ClockPort(Protocol):
    """Wall-clock timestamps for display, monotonic time for durations."""

    def now(self) -> datetime:
        """Current UTC wall-clock time."""
        ...

    def monotonic(self) -> float:
        """Monotonic seconds, suitable for measuring durations."""
        ...


class SystemClock:
    """The real clock."""

    def now(self) -> datetime:
        return datetime.now(UTC)

    def monotonic(self) -> float:
        return time.monotonic()
