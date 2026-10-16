"""Settings never expose the provider key (FR-017, FR-018 / T042)."""

from __future__ import annotations

from inference_provider_proxy.config import Settings


def test_provider_key_is_redacted_in_repr() -> None:
    settings = Settings(_env_file=None, upstream_api_key="super-secret-value")
    assert "super-secret-value" not in repr(settings)
    assert "super-secret-value" not in str(settings)
    assert settings.provider_key == "super-secret-value"


def test_documented_defaults_match_the_spec() -> None:
    settings = Settings(_env_file=None)
    assert settings.provider_concurrency == 1  # FR-004
    assert settings.max_queue_depth == 64  # FR-006
    assert settings.queue_timeout == 900  # FR-006
    assert settings.disconnect_release == 15  # FR-010
    assert settings.max_retries == 2  # FR-012
    assert settings.metrics_retention == 2000  # FR-030
    assert settings.metadata_cache_ttl == 30  # FR-002
    assert settings.access_log is True


def test_no_client_token_setting_exists() -> None:
    # The no-auth decision removed the LAN token entirely (R11).
    fields = set(Settings.model_fields)
    assert not any("token" in name for name in fields)
    assert not any("auth" in name for name in fields)
