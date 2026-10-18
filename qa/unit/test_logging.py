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


# -- layout ------------------------------------------------------------------


def test_level_names_are_the_python_levels() -> None:
    assert LOG_LEVELS == {
        "trace": logging.DEBUG,
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }


def test_plain_layout_is_timestamp_badge_tag_message() -> None:
    out = _SectionFormatter(color=False).format(_record())
    tag = f"[{'access':<{SECTION_WIDTH}}]"

    assert re.fullmatch(rf"\d{{2}}:\d{{2}}:\d{{2}}\.\d{{3}} INFO  {re.escape(tag)} hello", out)


def test_every_section_line_puts_its_message_in_the_same_column() -> None:
    formatter = _SectionFormatter(color=False)
    names = ("access", "subscription", "gate")
    lines = [formatter.format(_record(name=f"ipp.{name}")) for name in names]

    columns = {line.index("hello") for line in lines}
    assert len(columns) == 1
    # The tag really is padded out to the shared width, not merely equal here.
    assert all(
        f"[{name:<{SECTION_WIDTH}}]" in line for name, line in zip(names, lines, strict=True)
    )


def test_badges_are_five_characters_for_every_standard_level() -> None:
    formatter = _SectionFormatter(color=False)
    badges = set()
    for level, name in [
        (logging.DEBUG, "DEBUG"),
        (logging.INFO, "INFO"),
        (logging.WARNING, "WARN"),
        (logging.ERROR, "ERROR"),
        (logging.CRITICAL, "CRIT"),
    ]:
        line = formatter.format(_record(level=level))
        match = re.match(r"\d{2}:\d{2}:\d{2}\.\d{3} (.{5}) ", line)
        assert match is not None
        assert match.group(1).strip() == name
        badges.add(match.group(1))

    assert len(badges) == 5


def test_unnamed_levels_render_without_raising() -> None:
    out = _SectionFormatter(color=False).format(_record(name="ipp.weird", level=25))

    assert "Level" in out
    assert "hello" in out


def test_section_names_keep_every_line_in_the_same_column() -> None:
    assert _section("ipp.subscription") == "subscription"  # ours, verbatim
    assert _section("uvicorn.error") == "uvicorn"  # the server speaks as one
    assert _section("starlette.exceptions") == "exceptions"  # third parties
    assert _section("httpx") == "httpx"

    formatter = _SectionFormatter(color=False)
    names = ("ipp.subscription", "uvicorn.error", "starlette.exceptions", "httpx")
    columns = {formatter.format(_record(name=name)).index("hello") for name in names}

    assert len(columns) == 1


def test_the_uvicorn_config_uses_our_formatter_on_our_stream() -> None:
    config = uvicorn_log_config(color="never")

    formatter = config["formatters"]["default"]
    assert formatter["()"] == "inference_provider_proxy.logging_setup._SectionFormatter"
    assert formatter["color"] is False
    assert config["handlers"]["default"]["stream"] == "ext://sys.stderr"
    assert config["disable_existing_loggers"] is False
    assert config["loggers"]["uvicorn.access"]["level"] == "INFO"
    # ``uvicorn.error`` rides on the parent's handler, as uvicorn itself does.
    assert "handlers" not in config["loggers"]["uvicorn.error"]


def test_traceback_is_appended_under_the_message() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        exc_info = sys.exc_info()

    out = _SectionFormatter(color=False).format(_record(msg="failed", exc_info=exc_info))

    assert "failed\nTraceback" in out
    assert "ValueError: boom" in out


# -- colors ------------------------------------------------------------------


def test_colorized_line_keeps_the_message_plain() -> None:
    out = _SectionFormatter(color=True).format(
        _record(name="ipp.gate", level=logging.WARNING, msg="queue full")
    )

    assert "\x1b[33m" in out  # warning is yellow
    assert "\x1b[35m" in out  # the gate section is magenta
    assert "\x1b[2m" in out  # the timestamp is dimmed
    assert out.endswith("queue full")  # the body carries no escape codes


def test_each_section_has_its_own_color_and_levels_too() -> None:
    formatter = _SectionFormatter(color=True)
    critical = formatter.format(_record(name="ipp.app", level=logging.CRITICAL, msg="x"))
    debug = formatter.format(_record(name="ipp.http", level=logging.DEBUG, msg="y"))
    info = formatter.format(_record(name="ipp.access", msg="z"))

    assert "\x1b[1;97;41m" in critical and "CRIT " in critical
    assert "\x1b[90m" in debug
    assert "\x1b[36m" in info  # access and http share cyan


def test_color_decision_honours_modes_environment_and_terminal(monkeypatch) -> None:
    class _Pipe:
        def isatty(self) -> bool:
            return False

    class _Terminal:
        def isatty(self) -> bool:
            return True

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setattr(sys, "stderr", _Pipe())

    assert _should_color("always") is True
    assert _should_color("never") is False
    assert _should_color("auto") is False  # a pipe gets no escape codes

    monkeypatch.setenv("NO_COLOR", "1")
    assert _should_color("auto") is False
    monkeypatch.delenv("NO_COLOR")

    monkeypatch.setenv("FORCE_COLOR", "1")
    assert _should_color("auto") is True
    monkeypatch.delenv("FORCE_COLOR")

    monkeypatch.setattr(sys, "stderr", _Terminal())
    assert _should_color("auto") is True


# -- configuration -----------------------------------------------------------


def test_configure_logging_sets_the_level_and_reaches_stderr(capsys) -> None:
    configure_logging("info", color="never")
    logging.getLogger("ipp.access").info("GET /v1/models 200 3ms")
    logging.getLogger("ipp.access").debug("a quiet line")

    err = capsys.readouterr().err

    assert "GET /v1/models 200 3ms" in err
    assert "a quiet line" not in err
    line = next(text for text in err.splitlines() if "GET" in text)
    assert re.match(r"^\d{2}:\d{2}:\d{2}\.\d{3} INFO  \[access\s+\] ", line)


def test_configure_logging_is_idempotent() -> None:
    configure_logging("info", color="never")
    configure_logging("debug", color="never")

    root = logging.getLogger()
    ours = [handler for handler in root.handlers if isinstance(handler, _ProxyLogHandler)]

    assert len(ours) == 1
    assert root.level == logging.DEBUG


def test_an_unrecognised_level_falls_back_to_info() -> None:
    configure_logging("loud", color="never")

    assert logging.getLogger().level == logging.INFO
