"""The proxy's one timestamp dialect: ISO-8601 UTC ending in ``Z``.

Every JSON endpoint renders through this helper so one instant never appears in
two formats across the API (the dashboard and ``/v1/usage`` previously drifted
into ``Z`` and ``+00:00`` dialects).
"""

from __future__ import annotations

from datetime import datetime


def iso_z(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat().replace("+00:00", "Z")
