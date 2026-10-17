"""Page assets for the HTML dashboard, read from disk once at import.

Kept out of the route and render modules so the markup and stylesheet ship as
real ``.html`` / ``.css`` files (package data) rather than Python string
literals, while request handling still does no file I/O.
"""

from __future__ import annotations

from importlib.resources import files

__all__ = ["INDEX_HTML", "STYLESHEET"]


def _read(name: str) -> str:
    return files(__package__).joinpath(name).read_text(encoding="utf-8")


INDEX_HTML = _read("index.html")
STYLESHEET = _read("styles.css")
