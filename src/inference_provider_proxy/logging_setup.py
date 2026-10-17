"""Process-wide logging: one colored, section-tagged stderr stream (FR-031).

The proxy holds exactly one credential and it is never logged (FR-018): every
record is built from method, path, status, ids and timings only, so the output
can afford to be readable instead of machine-only:

* ``HH:MM:SS.mmm`` timestamp, dimmed when colors are on;
* a fixed five-character level badge (``INFO `` / ``WARN `` / ``ERROR``);
* a ``[section]`` tag padded to one common width, so a full log reads as
  aligned columns regardless of which section spoke;
* the message body, never colorized, so it survives a pipe or a grep.

Colors are enabled automatically only on a terminal; ``NO_COLOR`` and
``FORCE_COLOR`` are honoured and ``IPP_LOG_COLOR=always|never`` overrides both.

Configuration is idempotent: the console entry point and the ASGI factory both
call it, in either order, without duplicating a single line. Uvicorn is handed
the same formatter, so server lines and proxy lines are one format on one
stream rather than two interleaved dialects.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any

# Accepted ``IPP_LOG_LEVEL`` / ``--log-level`` names. ``trace`` is accepted for
# parity with uvicorn's own flag and means "everything Python calls DEBUG".
LOG_LEVELS = {
    "trace": logging.DEBUG,
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

# Every ``[section]`` tag is padded to this width so columns never mismatch.
SECTION_WIDTH = len("subscription")

_RESET = "\x1b[0m"
_DIM = "\x1b[2m"

_LEVEL_COLORS = {
    "DEBUG": "\x1b[90m",
    "INFO": "\x1b[32m",
    "WARNING": "\x1b[33m",
    "ERROR": "\x1b[31m",
    "CRITICAL": "\x1b[1;97;41m",
}

# One color per log section, so a reader can pick out a lane at a glance.
_SECTION_COLORS = {
    "access": "\x1b[36m",
    "http": "\x1b[36m",
    "gate": "\x1b[35m",
    "provider": "\x1b[34m",
    "chat": "\x1b[34m",
    "subscription": "\x1b[33m",
    "app": "\x1b[37m",
    "uvicorn": "\x1b[37m",
}

_LEVEL_BADGES = {
    "INFO": "INFO ",
    "WARNING": "WARN ",
    "CRITICAL": "CRIT ",
}


def _section(name: str) -> str:
    """The ``[section]`` label for a logger name (padding happens in the formatter).

    The proxy's own sections are used verbatim, uvicorn's several loggers speak
    as a single ``uvicorn`` section, and anything else contributes its last
    segment so a third-party dotted name cannot shift the columns.
    """
    if name.startswith("ipp."):
        return name[4:]
    if name.startswith("uvicorn"):
        return "uvicorn"
    return name.rsplit(".", 1)[-1]


class _SectionFormatter(logging.Formatter):
    """``HH:MM:SS.mmm LEVEL [section  ] message``, optionally colorized."""

    def __init__(self, *, color: bool) -> None:
        super().__init__()
        self._color = color

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"
        when = time.localtime(record.created)
        stamp = f"{when.tm_hour:02d}:{when.tm_min:02d}:{when.tm_sec:02d}.{int(record.msecs):03d}"
        badge = f"{_LEVEL_BADGES.get(record.levelname, record.levelname):<5.5}"
        section = _section(record.name)
        tag = f"[{section:<{SECTION_WIDTH}}]"
        if not self._color:
            return f"{stamp} {badge} {tag} {message}"
        level_color = _LEVEL_COLORS.get(record.levelname, "")
        section_color = _SECTION_COLORS.get(section, "")
        return (
            f"{_DIM}{stamp}{_RESET} {level_color}{badge}{_RESET} "
            f"{section_color}{tag}{_RESET} {message}"
        )


class _ProxyLogHandler(logging.StreamHandler):
    """Marker subclass, so re-configuration can find the handler it installed."""

    def __init__(self) -> None:
        super().__init__(sys.stderr)


def _should_color(mode: str) -> bool:
    """Resolve ``auto|always|never`` against the environment and the terminal."""
    if mode == "always":
        return True
    if mode == "never":
        return False
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR") is not None:
        return True
    return sys.stderr.isatty()


def configure_logging(level: str = "info", *, color: str = "auto") -> None:
    """Install the shared stderr handler (once) and set the process log level.

    Idempotent by design: a second call reuses the installed handler and only
    refreshes its formatter, so ``cli.main`` and ``build_app`` can both call it.
    """
    root = logging.getLogger()
    handler = next((h for h in root.handlers if isinstance(h, _ProxyLogHandler)), None)
    if handler is None:
        handler = _ProxyLogHandler()
        root.addHandler(handler)
    handler.setFormatter(_SectionFormatter(color=_should_color(color)))
    root.setLevel(LOG_LEVELS.get(level.lower(), logging.INFO))


def uvicorn_log_config(*, color: str = "auto") -> dict[str, Any]:
    """A ``log_config`` that makes uvicorn's own lines match ours.

    Without it the server speaks in a different dialect on a different stream,
    and a full log reads as two interleaved formats. ``uvicorn.access`` is
    configured for completeness only: the entry point always runs uvicorn with
    ``access_log=False``, because the request line belongs to ``ipp.access``,
    which carries the request id — one line per request, not two.
    """
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "inference_provider_proxy.logging_setup._SectionFormatter",
                "color": _should_color(color),
            }
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            }
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error": {"level": "INFO"},
            "uvicorn.access": {"handlers": ["default"], "level": "INFO", "propagate": False},
        },
    }
