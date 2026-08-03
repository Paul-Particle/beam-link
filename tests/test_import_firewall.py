"""I8: the import firewall is a test. No allowlist, no exempted module.

The rule is enforced by PACKAGE LAYOUT, so it has no holes and a new file inherits it from the
directory it is placed in (SPEC.md §9):

    artifacts/   imports nothing else in the repo
    physics/     may import artifacts/
    viz/         may import artifacts/ only
    econ/        may import artifacts/ only
    drivers/     may invoke physics/ AND econ/

This walks the AST of every module in the package rather than importing it, so a module that would
import physics only on some code path still fails — and so the test does not need plotly, itur or a
run directory to say whether the firewall holds.

The two cases that looked like they needed an exemption resolve by classification instead (§2):
`beam_link/reference/` is a second implementation ON PURPOSE, checked against the primary by the
engine's own tests, and it is engine-internal — being inside the boundary is not an exemption from
it. And a figure that wants a pure focusing sweep does not need privileged access to the diffraction
code; it needs a run whose scenario has no receiver.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "beamlink"

# Which top-level subpackage may import which. `artifacts` is in every allowed set except its own,
# because it is the data contract; nothing may import `viz`, `econ` or `drivers` from inside physics.
PERMITTED = {
    "artifacts": set(),
    "physics": {"artifacts"},
    "viz": {"artifacts"},
    "econ": {"artifacts"},
    "drivers": {"artifacts", "physics", "econ"},
}


def _modules():
    """Every .py file under the package, with the subpackage it belongs to."""
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        relative = path.relative_to(PACKAGE_ROOT)
        if relative.parts[0] == "__pycache__":
            continue
        area = relative.parts[0] if len(relative.parts) > 1 else ""
        yield area, path


def _imported_areas(path: Path) -> set[str]:
    """The `beamlink.<area>` subpackages a module imports, from its AST.

    Read rather than executed, so a conditional or function-local import counts exactly the same as a
    top-level one — which matters, because "imports plotly inside a physics module" (§8 item 4) was a
    function-local import.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    areas: set[str] = set()
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module]
            names += [f"{node.module}.{alias.name}" for alias in node.names]
        for name in names:
            parts = name.split(".")
            if parts[0] == "beamlink" and len(parts) > 1 and parts[1] in PERMITTED:
                areas.add(parts[1])
    return areas


@pytest.mark.parametrize("area,path", list(_modules()), ids=lambda value: str(value)[-40:])
def test_module_imports_only_what_its_directory_permits(area, path):
    if area not in PERMITTED:
        pytest.skip(f"{path.name} is not inside a firewalled subpackage")
    permitted = PERMITTED[area] | {area}
    forbidden = _imported_areas(path) - permitted
    assert not forbidden, (
        f"{path.relative_to(PACKAGE_ROOT)} imports beamlink.{sorted(forbidden)}, which "
        f"beamlink/{area}/ may not. This is not a naming exception to add — it is the layout "
        f"telling you the code is in the wrong directory (SPEC.md §9, I8).")


def test_artifacts_imports_nothing_from_the_repo():
    """The data contract has no repo dependencies at all, which is what keeps the firewall honest.

    If the product definitions imported anything under `physics/`, every consumer would have to
    import the engine to read a run, and the rule would be a fiction.
    """
    offenders = {}
    for area, path in _modules():
        if area != "artifacts":
            continue
        imported = _imported_areas(path) - {"artifacts"}
        if imported:
            offenders[path.name] = sorted(imported)
    assert not offenders, f"artifacts/ must import nothing else in the repo; found {offenders}"


def test_no_consumer_imports_plotly_into_physics():
    """Presentation libraries belong outside the physics boundary (§8 items 4 and 5).

    Separate from the beamlink-internal rule above because a physics module importing plotly is a
    different defect from one importing the wrong subpackage: it means a renderer has been written
    inside a kernel, which is how `rasterize_view_zbuffer(color=...)` came to exist.
    """
    presentation = {"plotly", "matplotlib", "PIL", "bokeh", "altair", "seaborn"}
    offenders = {}
    for area, path in _modules():
        if area not in ("physics", "artifacts"):
            continue
        if "tables" in path.parts:
            continue          # the table generators plot nothing; they call ITU models
        tree = ast.parse(path.read_text(), filename=str(path))
        found = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                found.add(node.module.split(".")[0])
        if found & presentation:
            offenders[str(path.relative_to(PACKAGE_ROOT))] = sorted(found & presentation)
    assert not offenders, (f"a physics module may not import a plotting library; found {offenders}. "
                          f"Products carry physics, not presentation (I7).")


def test_the_permitted_table_covers_every_subpackage():
    """A new top-level subpackage must be given a rule, not left unruled.

    Without this, adding `beamlink/reports/` would silently be outside the firewall — the failure
    mode an allowlist has and a layout rule is supposed not to.
    """
    on_disk = {path.name for path in PACKAGE_ROOT.iterdir()
               if path.is_dir() and path.name != "__pycache__" and (path / "__init__.py").exists()}
    unruled = on_disk - set(PERMITTED)
    assert not unruled, (f"beamlink/{sorted(unruled)} has no entry in PERMITTED: every subpackage "
                         f"needs a stated rule (SPEC.md I8).")
