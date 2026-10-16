"""The ``ipp`` console entry point: parse arguments and run uvicorn."""

from __future__ import annotations

import argparse

import uvicorn

from inference_provider_proxy.config import get_settings


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
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="uvicorn log level",
    )
    args = parser.parse_args()

    settings = get_settings()
    uvicorn.run(
        "inference_provider_proxy.main:build_app",
        factory=True,
        host=args.host or settings.bind_host,
        port=args.port or settings.bind_port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":  # pragma: no cover - module entry point
    main()
