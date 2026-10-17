"""Operator-tunable configuration (FR-030).

Read from the environment or a git-ignored ``.env`` file, using the ``IPP_``
prefix. Secrets are held as :class:`~pydantic.SecretStr` so they are redacted in
reprs and tracebacks. There is deliberately no client token: the proxy is open
on a trusted LAN (FR-016).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed settings for the proxy process."""

    model_config = SettingsConfigDict(
        env_prefix="IPP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Provider
    upstream_base_url: str = "https://api.example.com"
    upstream_api_key: SecretStr = SecretStr("")

    # Bind
    bind_host: str = "0.0.0.0"
    bind_port: int = Field(default=13306, ge=1, le=65535)

    # Concurrency and queueing (FR-004, FR-006, FR-007, FR-010)
    provider_concurrency: int = Field(default=1, ge=1)
    max_queue_depth: int = Field(default=64, ge=0)
    queue_timeout: float = Field(default=900.0, gt=0)
    disconnect_release: float = Field(default=15.0, gt=0)

    # Upstream timeouts and retries (FR-012, FR-030)
    upstream_timeout: float = Field(default=600.0, gt=0)
    stream_idle_timeout: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    user_agent: str = "ipp/0.1"

    # Metrics and logging (FR-030, FR-031)
    metrics_retention: int = Field(default=2000, ge=1)
    metadata_cache_ttl: float = Field(default=30.0, ge=0)
    access_log: bool = True
    # Level for the ``ipp.*`` sections and ANSI coloring of the stream
    # (``NO_COLOR``/``FORCE_COLOR`` are honoured by ``auto``).
    log_level: Literal["critical", "error", "warning", "info", "debug", "trace"] = "info"
    log_color: Literal["auto", "always", "never"] = "auto"

    @property
    def provider_key(self) -> str:
        """The real provider credential, for outbound substitution only (FR-017)."""
        return self.upstream_api_key.get_secret_value()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
