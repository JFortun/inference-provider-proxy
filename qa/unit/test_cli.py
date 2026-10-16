"""The console and module entry points, exercised without starting a server."""

from __future__ import annotations

import sys

from inference_provider_proxy import cli
from inference_provider_proxy.config import Settings


def test_main_wires_uvicorn_from_arguments(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(app_path: str, **kwargs: object) -> None:
        captured["app_path"] = app_path
        captured.update(kwargs)

    monkeypatch.setattr(cli.uvicorn, "run", fake_run)
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(
        sys,
        "argv",
        ["ipp", "--host", "1.2.3.4", "--port", "9999", "--reload", "--log-level", "debug"],
    )

    cli.main()

    assert captured == {
        "app_path": "inference_provider_proxy.main:build_app",
        "factory": True,
        "host": "1.2.3.4",
        "port": 9999,
        "reload": True,
        "log_level": "debug",
    }


def test_main_defaults_to_the_configured_bind(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(cli.uvicorn, "run", lambda *args, **kwargs: captured.update(kwargs))
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(sys, "argv", ["ipp"])

    cli.main()

    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 13306
    assert captured["reload"] is False


def test_module_entry_delegates_to_cli_main() -> None:
    import inference_provider_proxy.__main__ as entry

    assert entry.main is cli.main
