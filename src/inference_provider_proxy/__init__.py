"""inference_provider_proxy (``ipp``).

A LAN proxy that owns a single provider key, serializes chat completions so
that overlapping requests queue instead of colliding with the provider's
concurrency limit, and exposes an OpenAI-compatible surface plus a session
dashboard.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    __version__ = version("inference-provider-proxy")
except PackageNotFoundError:  # imported from a bare source tree, never installed
    __version__ = "0+unknown"
