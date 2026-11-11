"""Settings never expose the provider key."""

from __future__ import annotations

from inference_provider_proxy.config import Settings


def test_provider_key_when_set_then_never_appears_in_repr_or_str() -> None:
    settings = Settings(_env_file=None, upstream_api_key="super-secret-value")

    as_repr = repr(settings)
    as_str = str(settings)

    assert "super-secret-value" not in as_repr
    assert "super-secret-value" not in as_str
    assert settings.provider_key == "super-secret-value"


def test_default_settings_when_loaded_then_match_the_documented_spec() -> None:
    settings = Settings(_env_file=None)

    defaults = {
        "provider_concurrency": settings.provider_concurrency,
        "max_queue_depth": settings.max_queue_depth,
        "queue_timeout": settings.queue_timeout,
        "disconnect_release": settings.disconnect_release,
        "max_retries": settings.max_retries,
        "metrics_retention": settings.metrics_retention,
        "metadata_cache_ttl": settings.metadata_cache_ttl,
        "access_log": settings.access_log,
        "log_level": settings.log_level,
        "log_color": settings.log_color,
    }

    assert defaults == {
        "provider_concurrency": 1,
        "max_queue_depth": 64,
        "queue_timeout": 900,
        "disconnect_release": 15,
        "max_retries": 2,
        "metrics_retention": 2000,
        "metadata_cache_ttl": 30,
        "access_log": True,
        "log_level": "info",
        "log_color": "auto",
    }


def test_settings_fields_when_inspected_then_contain_no_token_or_auth() -> None:
    fields = set(Settings.model_fields)

    has_token = any("token" in name for name in fields)
    has_auth = any("auth" in name for name in fields)

    assert not has_token
    assert not has_auth
