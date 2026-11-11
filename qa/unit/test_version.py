"""``__version__`` comes from the installed metadata, with a usable fallback.

A hardcoded literal in ``__init__.py`` is a second version source: ``uv
version`` never touches it, so the dashboard reported a stale version after a
successful install and the deploy probe failed against it.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from importlib.metadata import PackageNotFoundError, version
from unittest import mock

import inference_provider_proxy


def test_package_version_when_read_then_matches_installed_metadata() -> None:
    installed = version("inference-provider-proxy")

    reported = inference_provider_proxy.__version__

    assert reported == installed


def test_package_version_when_not_installed_then_falls_back_to_unknown() -> None:
    with mock.patch(
        "importlib.metadata.version",
        side_effect=PackageNotFoundError("inference-provider-proxy"),
    ):
        reloaded = importlib.reload(inference_provider_proxy)

    assert reloaded.__version__ == "0+unknown"

    importlib.reload(inference_provider_proxy)
