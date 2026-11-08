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


def test_version_is_the_installed_one() -> None:
    assert inference_provider_proxy.__version__ == version("inference-provider-proxy")


def test_version_falls_back_when_the_package_is_not_installed() -> None:
    with mock.patch(
        "importlib.metadata.version",
        side_effect=PackageNotFoundError("inference-provider-proxy"),
    ):
        reloaded = importlib.reload(inference_provider_proxy)
        assert reloaded.__version__ == "0+unknown"

    importlib.reload(inference_provider_proxy)  # restore the real version
    assert inference_provider_proxy.__version__ == version("inference-provider-proxy")
