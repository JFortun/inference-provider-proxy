"""Pytest configuration for the ``ipp`` test suite.

The end-to-end suite talks to a real running proxy and therefore to the real
provider. It is excluded unless ``--run-e2e`` is passed explicitly: a marker on
its own excludes nothing, so the gate is enforced here rather than in
``addopts`` (where ``-m 'not e2e'`` would collide with an explicit ``-m e2e``).
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-e2e",
        action="store_true",
        default=False,
        help="run the e2e suite that talks to a real provider through a running proxy",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-e2e"):
        return
    skip_e2e = pytest.mark.skip(reason="need --run-e2e to run provider-backed e2e tests")
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip_e2e)
