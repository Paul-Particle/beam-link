"""The manifest: the contract a consumer learns a run from (SPEC.md §4).

A consumer pointed at a run directory learns from the manifest alone what exists, at what
fidelity, and what was asked for but not produced. Nothing else in the run answers those
questions, and nothing in a figure is allowed to answer them from a hand-written caption (§7).

A STUDY manifest is the same idea one level up (D6): it names the runs it aggregated and the
aggregation definition it applied, so "who derived this number and how" is answerable for a
corridor figure exactly as it is for a pass figure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from beamlink.artifacts.spec import AggregateDefinition

STATUSES = ("produced", "skipped", "failed")


@dataclass(frozen=True)
class ProductStatus:
    """What became of one requested product."""

    name: str
    status: str
    reason: str = ""
    """Required for "skipped" and "failed". A skip is a legitimate outcome — a scenario with no
    receiver has no silhouette — and the reason is what tells a consumer which."""

    tier: str = ""
    method: str = ""
    """The method actually used, not the one requested. The closed-form capture tier and the
    overlap integral are different derivations of the same quantity, and mixing them within one
    presented result has to be detectable (I5)."""

    resolution: Mapping[str, Any] = field(default_factory=dict)
    """Whatever numbers set the accuracy of this product at this tier: mesh face count, capture
    grid size, aperture samples, iteration counts."""

    inputs: tuple[str, ...] = ()
    """For a derived product, the products it came from. A derived product that cannot name its
    inputs is illegal (I3), so this is empty only for a primary."""

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {self.status!r}")
        if self.status in ("skipped", "failed") and not self.reason:
            raise ValueError(f"{self.name}: a {self.status} product must give a reason")


@dataclass(frozen=True)
class Manifest:
    """Everything about a run that is not an array."""

    run_id: str
    spec: Mapping[str, Any]
    """The spec verbatim, as JSON. Arrays (the trajectory) are included in full: §6 rules output
    size out as a design constraint, and a manifest that omitted the trajectory would not be the
    contract it claims to be."""

    engine_sha: str
    engine_version: str
    created_utc: str
    products: Mapping[str, ProductStatus]
    candidate_axes: Mapping[str, tuple]
    """The swept parameter paths and their per-candidate values, so the leading candidate axis of
    every array is interpretable without re-reading the driver that produced it (§6a)."""

    n_candidates: int = 1
    n_samples: int = 0
    platform: Mapping[str, Any] = field(default_factory=dict)
    notes: str = ""

    def status(self, name: str) -> ProductStatus | None:
        return self.products.get(name)

    def was_produced(self, name: str) -> bool:
        entry = self.products.get(name)
        return entry is not None and entry.status == "produced"

    def absent(self) -> Mapping[str, str]:
        """Requested but not produced, name → the reason to render on the page."""
        return {name: entry.reason for name, entry in self.products.items()
                if entry.status != "produced"}


@dataclass(frozen=True)
class StudyManifest:
    """A cross-run aggregation, owned by the driver (D6).

    The engine derives within a run, the driver derives across runs, and a consumer never derives
    at all. The line is not who is allowed to do arithmetic — it is that whoever does it is named
    in the output, which is what this file is for.
    """

    study_id: str
    run_dirs: tuple[str, ...]
    """The runs aggregated, in the order the study's leading axis indexes them."""

    definitions: Mapping[str, AggregateDefinition]
    """The aggregation definitions applied, by result name. Same object P9 carries, so a corridor
    number is as answerable as a pass number."""

    driver: str
    """Which driver produced this, and its git sha — a study's structure is a consequence of a
    particular econ objective and the physics it depends on (D4), so naming it matters."""

    engine_sha: str
    created_utc: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    """The driver's own config: the sweep axis, the search settings, the econ objective."""

    notes: str = ""
