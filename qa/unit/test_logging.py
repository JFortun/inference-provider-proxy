"""The logging foundation: layout, colors, and idempotent configuration.

The formatter is what an operator actually reads, so its layout is asserted
literally: fixed-width timestamp, five-character badge, and a ``[section]`` tag
padded to the width shared by every ``ipp.*`` section.
"""

from __future__ import annotations

import logging
import re
import sys

from inference_provider_proxy.logging_setup import (
    LOG_LEVELS,
    SECTION_WIDTH,
    _ProxyLogHandler,
    _section,
    _SectionFormatter,
    _should_color,
    configure_logging,
    uvicorn_log_config,
)


def _record(
    name: str = "ipp.access",
    level: int = logging.INFO,
    msg: str = "hello",
    exc_info: tuple | None = None,
) -> logging.LogRecord:
    return logging.LogRecord(
        name=name, level=level, pathname=__file__, lineno=1, args=(), msg=msg, exc_info=exc_info
    )


def _columns(*names: str, color: bool = False) -> set[int]:
    formatter = _SectionFormatter(color=color)
    return {formatter.format(_record(name=name)).index("hello") for name in names}


# -- layout ------------------------------------------------------------------


def test_level_names_when_read_then_match_the_python_levels() -> None:
    expected = {
        "trace": logging.DEBUG,
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    levels = LOG_LEVELS

    assert levels == expected


def test_plain_record_when_formatted_then_is_timestamp_badge_tag_message() -> None:
    formatter = _SectionFormatter(color=False)

    out = formatter.format(_record())
    tag = f"[{'access':<{SECTION_WIDTH}}]"

    assert re.fullmatch(rf"\d{{2}}:\d{{2}}:\d{{2}}\.\d{{3}} INFO  {re.escape(tag)} hello", out)


def test_every_section_line_when_formatted_then_puts_message_in_one_column() -> None:
    names = ("access", "subscription", "gate")

    lines = [_SectionFormatter(color=False).format(_record(name=f"ipp.{name}")) for name in names]

    columns = {line.index("hello") for line in lines}
    assert len(columns) == 1
    assert all(
        f"[{name:<{SECTION_WIDTH}}]" in line for name, line in zip(names, lines, strict=True)
    )


def test_each_standard_level_when_formatted_then_renders_a_five_char_badge() -> None:
    formatter = _SectionFormatter(color=False)
    cases = [
        (logging.DEBUG, "DEBUG"),
        (logging.INFO, "INFO"),
        (logging.WARNING, "WARN"),
        (logging.ERROR, "ERROR"),
        (logging.CRITICAL, "CRIT"),
    ]

    badges = set()
    for level, name in cases:
        line = formatter.format(_record(level=level))
        match = re.match(r"\d{2}:\d{2}:\d{2}\.\d{3} (.{5}) ", line)
        badges.add(match.group(1))
        assert match.group(1).strip() == name

    assert len(badges) == 5


def test_unnamed_level_when_formatted_then_renders_without_raising() -> None:
    formatter = _SectionFormatter(color=False)

    out = formatter.format(_record(name="ipp.weird", level=25))

    assert "Level" in out
    assert "hello" in out


def test_section_names_when_derived_then_keep_every_line_in_one_column() -> None:
    names = ("ipp.subscription", "uvicorn.error", "starlette.exceptions", "httpx")

    sections = [_section(name) for name in names]

    assert sections == ["subscription", "uvicorn", "exceptions", "httpx"]
    assert len(_columns(*names)) == 1


def test_uvicorn_config_when_requested_then_uses_our_formatter_on_stderr() -> None:
    color = "never"

    config = uvicorn_log_config(color=color)
    formatter = config["formatters"]["default"]
    handlers = config["handlers"]["default"]

    assert formatter["()"] == "inference_provider_proxy.logging_setup._SectionFormatter"
    assert formatter["color"] is False
    assert handlers["stream"] == "ext://sys.stderr"
    assert config["disable_existing_loggers"] is False
    assert config["loggers"]["uvicorn.access"]["level"] == "INFO"
    assert "handlers" not in config["loggers"]["uvicorn.error"]


def test_record_with_exception_when_formatted_then_appends_the_traceback() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        exc_info = sys.exc_info()

    out = _SectionFormatter(color=False).format(_record(msg="failed", exc_info=exc_info))

    assert "failed\nTraceback" in out
    assert "ValueError: boom" in out


# -- colors ------------------------------------------------------------------


def test_warning_record_when_colorized_then_keeps_the_message_plain() -> None:
    formatter = _SectionFormatter(color=True)

    out = formatter.format(_record(name="ipp.gate", level=logging.WARNING, msg="queue full"))

    assert "\x1b[33m" in out
    assert "\x1b[35m" in out
    assert "\x1b[2m" in out
    assert out.endswith("queue full")


def test_each_section_when_colorized_then_uses_its_own_color_and_level() -> None:
    formatter = _SectionFormatter(color=True)

    critical = formatter.format(_record(name="ipp.app", level=logging.CRITICAL, msg="x"))
    debug = formatter.format(_record(name="ipp.http", level=logging.DEBUG, msg="y"))
    info = formatter.format(_record(name="ipp.access", msg="z"))

    assert "\x1b[1;97;41m" in critical and "CRIT " in critical
    assert "\x1b[90m" in debug
    assert "\x1b[36m" in info


class _Pipe:
    def isatty(self) -> bool:
        return False


class _Terminal:
    def isatty(self) -> bool:
        return True


def test_color_decision_when_forced_always_or_never_then_ignores_the_terminal(
    monkeypatch,
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setattr(sys, "stderr", _Pipe())

    always = _should_color("always")
    never = _should_color("never")
    auto_on_pipe = _should_color("auto")

    assert always is True
    assert never is False
    assert auto_on_pipe is False


def test_color_decision_when_auto_hits_a_pipe_then_emits_no_escape_codes(monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setattr(sys, "stderr", _Pipe())

    decision = _should_color("auto")

    assert decision is False


def test_color_decision_when_force_color_is_set_then_auto_colors(monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(sys, "stderr", _Pipe())
    monkeypatch.setenv("FORCE_COLOR", "1")

    decision = _should_color("auto")

    assert decision is True


def test_color_decision_when_auto_hits_a_terminal_then_colors(monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setattr(sys, "stderr", _Terminal())

    decision = _should_color("auto")

    assert decision is True


def test_no_color_env_when_set_then_auto_emits_no_escape_codes(monkeypatch) -> None:
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setattr(sys, "stderr", _Terminal())
    monkeypatch.setenv("NO_COLOR", "1")

    decision = _should_color("auto")

    assert decision is False


# -- configuration -----------------------------------------------------------


def test_configure_logging_when_enabled_then_reaches_stderr_and_hides_debug(capsys) -> None:
    configure_logging("info", color="never")
    logging.getLogger("ipp.access").info("GET /v1/models 200 3ms")
    logging.getLogger("ipp.access").debug("a quiet line")

    err = capsys.readouterr().err

    assert "GET /v1/models 200 3ms" in err
    assert "a quiet line" not in err
    assert re.match(r"^\d{2}:\d{2}:\d{2}\.\d{3} INFO  \[access\s+\] ", err.splitlines()[0])


def test_configure_logging_when_called_twice_then_keeps_one_handler_at_debug(capsys) -> None:
    configure_logging("info", color="never")
    configure_logging("debug", color="never")

    root = logging.getLogger()
    ours = [handler for handler in root.handlers if isinstance(handler, _ProxyLogHandler)]

    assert len(ours) == 1
    assert root.level == logging.DEBUG


def test_a_bad_level_when_configured_then_falls_back_to_info() -> None:
    configure_logging("loud", color="never")

    level = logging.getLogger().level

    assert level == logging.INFO
