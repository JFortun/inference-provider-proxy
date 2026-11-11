"""Hexagon boundaries are enforced mechanically, not by review.

Four rules, each verified to *fail* when deliberately violated so a rename or a
boundary break cannot pass silently:

1. Hexagon layering ``adapters -> application -> domain``.
2. Domain purity - no framework/HTTP libraries inside ``domain``.
3. Inbound adapters must never reach outbound adapters.
4. No import cycles at module level.

ArchUnitPython is pointed at the ``src`` project root, because the package lives
under ``src/`` and its import resolver otherwise treats those modules as
external. ``ignore_type_checking_imports=True`` is required: ports and route
modules reference domain types under ``if TYPE_CHECKING:`` only, which is not a
runtime coupling.
"""

from __future__ import annotations

import ast
from graphlib import CycleError, TopologicalSorter
from pathlib import Path

import pytest
from archunitpython import (
    CheckOptions,
    assert_passes,
    project_files,
    project_layers,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_PATH = "src"
PKG = REPO_ROOT / "src" / "inference_provider_proxy"

OPTIONS = CheckOptions(ignore_type_checking_imports=True)

DOMAIN = f"{PKG}/domain/**"
APPLICATION = f"{PKG}/application/**"
INBOUND = f"{PKG}/adapters/inbound/**"
OUTBOUND = f"{PKG}/adapters/outbound/**"

FRAMEWORKS = "fastapi|starlette|httpx|pydantic|pydantic_settings|uvicorn"


def _hexagon_layers():
    layers = project_layers(PROJECT_PATH)
    layers.layer("domain").defined_by(DOMAIN)
    layers.layer("application").defined_by(APPLICATION)
    layers.layer("inbound").defined_by(INBOUND)
    layers.layer("outbound").defined_by(OUTBOUND)
    layers.where_layer("domain").may_not_depend_on_layers("application", "inbound", "outbound")
    layers.where_layer("application").may_only_depend_on_layers("domain")
    layers.where_layer("inbound").may_only_depend_on_layers("application", "domain")
    layers.where_layer("outbound").may_only_depend_on_layers("application", "domain")
    return layers


# -- rule 1: hexagon layering -------------------------------------------------


def test_hexagon_layering_when_checked_then_has_no_violations() -> None:
    layers = _hexagon_layers()

    violations = layers.check(OPTIONS)

    assert violations == []


def test_inbound_layer_when_forbidden_from_domain_then_the_rule_detects_the_violation() -> None:
    broken = project_layers(PROJECT_PATH)
    broken.layer("domain").defined_by(DOMAIN)
    broken.layer("inbound").defined_by(INBOUND)
    broken.where_layer("inbound").may_not_depend_on_layers("domain")

    with pytest.raises(AssertionError) as excinfo:
        assert_passes(broken, OPTIONS)

    assert "violation" in str(excinfo.value)


# -- rule 2: domain purity ----------------------------------------------------


def _domain_purity(modules: str):
    return (
        project_files(PROJECT_PATH)
        .in_path(DOMAIN)
        .should_not()
        .depend_on_external_modules()
        .matching(modules)
    )


def test_domain_when_scanned_for_frameworks_then_is_pure() -> None:
    condition = _domain_purity(FRAMEWORKS)

    violations = condition.check(OPTIONS)

    assert violations == []


def test_domain_purity_rule_when_typing_is_forbidden_then_detects_the_violation() -> None:
    condition = _domain_purity("typing")

    with pytest.raises(AssertionError) as excinfo:
        assert_passes(condition, OPTIONS)

    assert "violation" in str(excinfo.value)


# -- rule 3: inbound must not reach outbound ----------------------------------


def test_inbound_when_checked_against_outbound_then_never_reaches_it() -> None:
    condition = (
        project_files(PROJECT_PATH)
        .in_path(INBOUND)
        .should_not()
        .depend_on_files()
        .in_path(OUTBOUND)
    )

    violations = condition.check(OPTIONS)

    assert violations == []


def test_route_boundary_when_outbound_forbidden_from_domain_then_detects_violation() -> None:
    condition = (
        project_files(PROJECT_PATH).in_path(OUTBOUND).should_not().depend_on_files().in_path(DOMAIN)
    )

    with pytest.raises(AssertionError) as excinfo:
        assert_passes(condition, OPTIONS)

    assert "violation" in str(excinfo.value)


# -- rule 4: no import cycles -------------------------------------------------


def _module_name(path: Path) -> str:
    relative = path.relative_to(REPO_ROOT / "src").with_suffix("")
    parts = [part for part in relative.parts if part != "__init__"]
    return ".".join(parts)


def _internal_imports(tree: ast.AST, module: str) -> set[str]:
    """Resolve ``inference_provider_proxy`` imports to fully-qualified module names."""
    found: set[str] = set()
    package = module.rsplit(".", 1)[0] if "." in module else module
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("inference_provider_proxy"):
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                target = package
            elif node.level:
                base = package
                for _ in range(node.level - 1):
                    base = base.rsplit(".", 1)[0]
                target = f"{base}.{node.module}"
            else:
                target = node.module
            if target.startswith("inference_provider_proxy"):
                found.add(target)
    return found


def _import_graph() -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {}
    for path in sorted(PKG.rglob("*.py")):
        module = _module_name(path)
        tree = ast.parse(path.read_text(), filename=str(path))
        graph[module] = _internal_imports(tree, module)
    return graph


def test_import_graph_when_sorted_then_contains_no_cycle() -> None:
    graph = _import_graph()
    sorter = TopologicalSorter(graph)

    try:
        sorter.prepare()
    except CycleError as exc:  # pragma: no cover - only on a real cycle
        raise AssertionError(f"import cycle detected: {exc.args[1]}") from exc

    assert sorter.is_active() is True


def test_topological_sorter_when_fed_a_cyclic_graph_then_raises_cycle_error() -> None:
    cyclic = {"a": {"b"}, "b": {"c"}, "c": {"a"}}
    acyclic = {"a": {"b"}, "b": {"c"}}

    with pytest.raises(CycleError):
        TopologicalSorter(cyclic).prepare()
    TopologicalSorter(acyclic).prepare()

    assert cyclic["a"] == {"b"}
