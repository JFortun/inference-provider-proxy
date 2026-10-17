"""The console and module entry points, exercised without starting a server."""

from __future__ import annotations

import sys

from inference_provider_proxy import cli
from inference_provider_proxy.config import Settings
from inference_provider_proxy.logging_setup import uvicorn_log_config


def test_main_wires_uvicorn_from_arguments(monkeypatch) -> None:
    captured: dict[str, object] = {}
    settings = Settings(_env_file=None)

    def fake_run(app_path: str, **kwargs: object) -> None:
        captured["app_path"] = app_path
        captured.update(kwargs)

    monkeypatch.setattr(cli.uvicorn, "run", fake_run)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(
        sys,
        "argv",
        ["ipp", "--host", "1.2.3.4", "--port", "9999", "--reload", "--log-level", "debug"],
    )

    cli.main()

    assert captured["app_path"] == "inference_provider_proxy.main:build_app"
    assert captured["factory"] is True
    assert captured["host"] == "1.2.3.4"
    assert captured["port"] == 9999
    assert captured["reload"] is True
    assert captured["log_level"] == "debug"
    # The command line wins over the environment and is written back, so the
    # startup line reports the bind and the level that were really chosen.
    assert settings.log_level == "debug"
    assert settings.bind_host == "1.2.3.4"
    assert settings.bind_port == 9999
    # One line per request: uvicorn's access log is replaced by ``ipp.access``.
    assert captured["access_log"] is False
    assert captured["log_config"] == uvicorn_log_config(color=settings.log_color)


def test_main_defaults_to_the_configured_bind(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(cli.uvicorn, "run", lambda *args, **kwargs: captured.update(kwargs))
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(sys, "argv", ["ipp"])

    cli.main()

    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 13306
    assert captured["reload"] is False
    # Without the flag, uvicorn and the proxy both use IPP_LOG_LEVEL.
    assert captured["log_level"] == "info"
    assert captured["access_log"] is False
    assert captured["log_config"]["loggers"]["uvicorn"]["propagate"] is False


def test_module_entry_delegates_to_cli_main() -> None:
    import inference_provider_proxy.__main__ as entry

    assert entry.main is cli.main
