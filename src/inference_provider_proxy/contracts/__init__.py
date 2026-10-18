"""Pinned contract artefacts shipped with the package.

``dashboard.schema.json`` is the source of truth for the ``GET /dashboard``
document: ``adapters/inbound/http/serialization.py`` emits exactly the keys it
declares, and both the integration suite and the live smoke run validate
against it. Keeping it here — not in spec-kit's git-ignored ``specs/`` tree —
means a clean checkout, an sdist, or a wheel install all carry the contract.
"""
