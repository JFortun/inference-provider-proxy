"""inference_provider_proxy (``ipp``).

A LAN proxy that owns a single provider key, serializes chat completions so
that overlapping requests queue instead of colliding with the provider's
concurrency limit, and exposes an OpenAI-compatible surface plus a session
dashboard.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
