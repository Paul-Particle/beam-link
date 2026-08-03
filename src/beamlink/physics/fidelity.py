"""Fidelity tiers: method and resolution, never shape, units or meaning (I5).

Run time against accuracy is the only trade axis, and every point on it is set from here (§6).
Output size is deliberately NOT a lever: the one place it ever mattered was a 15 MB cap on a
published phone artifact, which is a property of that publishing route and not of the engine.

The levers, in the measured order of value recorded in §6:

  capture mesh face count       8× on the dominant cost for 0.34 pp of capture (5704 → 612 faces)
  closed-form capture           0.17 ms against 21–26 ms, biased high by 0.4–4.8 pp, one-sided
  taper search resolution       ×1.5, worth up to 4.8 pp of capture, so not free
  density-cap solve tolerance   free where the cap does not bind, ×3.5 where it does
  field sampling density        set by the sampling request, not by the tier

and the measured NON-levers, recorded so they are not tried again: the silhouette raster's output
grid (200 → 64 buys 25 % and adds ±1 pp of resampling noise) and the capture integral's own
transform grid (stable to <0.001 pp from 256 to 1536).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

TIERS = ("full", "standard", "fast", "coarse")

# Products whose accuracy is set by the FIELD solve rather than by a method of their own. A tier
# override on any of these would split one solve into two at different methods, which I1 forbids
# and I5 would make undetectable, so the engine rejects it (D8).
FIELD_BOUND_PRODUCTS = ("field", "capture")


@dataclass(frozen=True)
class FidelityConfig:
    """One resolved tier. Everything an engine kernel is allowed to vary lives here."""

    tier: str

    # --- airframe geometry -----------------------------------------------------------------
    mesh_n_span: int = 71
    mesh_n_theta: int = 40
    """The loft's own resolution, which sets the face count — the dominant cost of a cold pass and
    the single biggest lever measured."""

    raster_n_grid: int = 200
    """The z-buffer's output grid. A measured non-lever: keep it."""

    look_angle_bin_deg: float = 0.5
    """Look angles are quantised to this before the silhouette raster is cached, so a pass reuses
    ~40 rasters instead of 219. Free of any fidelity cost at this size because the silhouette
    changes slowly with arrival angle; 0.1° was the pre-SPEC bin and bought nothing."""

    # --- capture ---------------------------------------------------------------------------
    capture_method: str = "overlap_integral"
    """"overlap_integral" integrates the solved field over the silhouette elements;
    "closed_form_goubau" uses the Goubau τ on the equal-area disc, as the relay up-leg already
    does. The second is a DIFFERENT derivation, not a cheaper version of the same one, and P6
    names its inputs accordingly."""

    capture_grid_n: int = 384
    """Transverse samples per axis of the field grid the silhouette elements are interpolated
    from. A measured non-lever above ~256; kept generous."""

    taper_search_n: int = 9
    """Points in the taper grid searched for maximum capture. A real lever."""

    density_bisection_steps: int = 12
    """Fixed number of defocus bisection steps, run for EVERY sample rather than exited per
    sample on a tolerance. Marginally more arithmetic, uniform cost, and it vectorises (I9)."""

    # --- the field solve -------------------------------------------------------------------
    aperture_min_samples_across_minor: int = 32
    """Aperture-plane samples across the minor semi-axis. Sets the pitch together with the
    focusing phase and the alias-free window the queries need."""

    aperture_n_max: int = 3072
    """Cap on the aperture grid. Past this the near-aperture planes are reported unresolved rather
    than paid for with an n² array — and the defocus floor is chosen analytically so that the
    density-cap solve never asks for one (D9)."""

    fresnel_floor_diameters: float = 2.5
    """Planes closer to the aperture than this many aperture DIAMETERS are reported unresolved:
    the paraxial Fresnel approximation itself fails there, and the honest answer is a validity
    mask rather than a geometric fill that makes the picture whole (§8 item 14)."""

    field_dtype: str = "complex128"
    chunk_bytes: int = 400_000_000
    """Working memory is the binding resource, not output size (§6a). A batch that would not fit
    is computed in blocks rather than refused."""

    # --- atmosphere and the mass fixed point ------------------------------------------------
    atmosphere_path_samples: int = 64
    mass_iterations: int = 40
    """Lockstep iterations of the mass cascade. Convergence is geometric, so a fixed count costs
    little and removes the data-dependent early exit (D5, I9)."""

    def resolution(self) -> Mapping[str, object]:
        """What goes in the manifest as this product's resolution."""
        return {
            "tier": self.tier, "mesh_n_span": self.mesh_n_span, "mesh_n_theta": self.mesh_n_theta,
            "raster_n_grid": self.raster_n_grid, "look_angle_bin_deg": self.look_angle_bin_deg,
            "capture_grid_n": self.capture_grid_n, "taper_search_n": self.taper_search_n,
            "density_bisection_steps": self.density_bisection_steps,
            "aperture_min_samples_across_minor": self.aperture_min_samples_across_minor,
            "aperture_n_max": self.aperture_n_max, "field_dtype": self.field_dtype,
            "atmosphere_path_samples": self.atmosphere_path_samples,
            "mass_iterations": self.mass_iterations,
        }


_PRESETS = {
    "full": FidelityConfig(
        tier="full", mesh_n_span=71, mesh_n_theta=40, raster_n_grid=256,
        look_angle_bin_deg=0.1, capture_grid_n=512, taper_search_n=17,
        density_bisection_steps=20, aperture_min_samples_across_minor=48,
        aperture_n_max=4096, field_dtype="complex128", atmosphere_path_samples=128,
        mass_iterations=80),
    "standard": FidelityConfig(tier="standard"),
    "fast": FidelityConfig(
        tier="fast", mesh_n_span=41, mesh_n_theta=26, raster_n_grid=200,
        look_angle_bin_deg=0.5, capture_grid_n=256, taper_search_n=7,
        density_bisection_steps=8, aperture_min_samples_across_minor=24,
        aperture_n_max=2048, field_dtype="complex64", atmosphere_path_samples=48,
        mass_iterations=30),
    "coarse": FidelityConfig(
        tier="coarse", mesh_n_span=25, mesh_n_theta=16, raster_n_grid=128,
        look_angle_bin_deg=1.0, capture_method="closed_form_goubau", capture_grid_n=192,
        taper_search_n=5, density_bisection_steps=6,
        aperture_min_samples_across_minor=16, aperture_n_max=1024, field_dtype="complex64",
        atmosphere_path_samples=32, mass_iterations=20),
}


def config(tier: str) -> FidelityConfig:
    if tier not in _PRESETS:
        raise KeyError(f"unknown fidelity tier {tier!r}; have {TIERS}")
    return _PRESETS[tier]


def resolve(spec_tier: str, overrides: Mapping[str, str]) -> Mapping[str, FidelityConfig]:
    """Per-product configs for a run, rejecting an override that would split a shared solve.

    D8: the field product's tier governs every evaluation of the field, and `capture` samples that
    same solve, so those two share a tier by construction. Products with a genuinely independent
    method — the silhouette mesh, the mass fixed point, the atmosphere path integral — may be
    overridden freely, and the manifest records what each one ran at.
    """
    base = config(spec_tier)
    field_bound = {name for name in FIELD_BOUND_PRODUCTS if name in overrides}
    if field_bound and len({overrides[name] for name in field_bound} | {spec_tier}) > 1:
        raise ValueError(
            f"fidelity override on {sorted(field_bound)} would split one field solve into two at "
            f"different methods, which I1 forbids and I5 would make undetectable. Set the run's "
            f"tier instead (SPEC.md D8).")
    from beamlink.artifacts.spec import PRODUCTS
    resolved = {name: base for name in PRODUCTS}
    for name, tier in overrides.items():
        if name not in resolved:
            raise KeyError(f"fidelity override for unknown product {name!r}")
        resolved[name] = config(tier)
    return resolved


def with_changes(base: FidelityConfig, **changes) -> FidelityConfig:
    """A one-off variation, for the calibration study that measures each lever on its own."""
    return replace(base, **changes)
