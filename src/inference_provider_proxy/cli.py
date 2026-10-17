"""The ``ipp`` console entry point: parse arguments and run uvicorn."""

from __future__ import annotations

import argparse

import uvicorn

from inference_provider_proxy.config import get_settings
from inference_provider_proxy.logging_setup import configure_logging, uvicorn_log_config


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ipp",
        description="Inference provider concurrency proxy (OpenAI-compatible, LAN).",
    )
    parser.add_argument("--host", default=None, help="bind host (overrides IPP_BIND_HOST)")
    parser.add_argument(
        "--port", type=int, default=None, help="bind port (overrides IPP_BIND_PORT)"
    )
    parser.add_argument("--reload", action="store_true", help="enable auto-reload (development)")
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="log level for the proxy and uvicorn (overrides IPP_LOG_LEVEL)",
    )
    args = parser.parse_args()

    # The command line wins over the environment, and it is written back so the
    # startup line reports the bind that was actually chosen.
    settings = get_settings()
    if args.log_level is not None:
        settings.log_level = args.log_level
    if args.host is not None:
        settings.bind_host = args.host
    if args.port is not None:
        settings.bind_port = args.port
    configure_logging(settings.log_level, color=settings.log_color)
    uvicorn.run(
        "inference_provider_proxy.main:build_app",
        factory=True,
        host=settings.bind_host,
        port=settings.bind_port,
        reload=args.reload,
        log_level=args.log_level or settings.log_level,
        # Uvicorn's own access line is replaced by ``ipp.access``, which carries
        # the request id: one line per request, in one format, on one stream.
        access_log=False,
        log_config=uvicorn_log_config(color=settings.log_color),
    )


if __name__ == "__main__":  # pragma: no cover - module entry point
    main()
