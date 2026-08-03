"""The receiver's shadow: what the aircraft removes from the beam behind it.

An OPTION on the field solve, not a product of its own — same shape, same units (§5 P5). The
pre-SPEC version was parameterised per display cut (`half_major_m`/`half_minor_m`, one number per
panel), which made it a 2-D thing that only existed inside a figure; here it is a 3-D occluder the
solver applies to whatever query points lie past the receiver (§8 item 17).

100 % capture and no reflection: whatever the silhouette intercepts is gone from the beam
downstream, and the geometric overlap P6 reports is how much that is. Two limits bracket what the
shadow looks like, and this blends between them:

  CLOSE BEHIND, ray optics. The emitting aperture has half-extents (R_x, R_y) on the two transverse
  axes, so the shadow has an umbra (no part of the aperture visible) inside a penumbra (part of it):

      umbra_j(z)    = max(a_j + (a_j − R_j)·Δz/d, 0)
      penumbra_j(z) = a_j + (a_j + R_j)·Δz/d

  for silhouette half-extent a_j and aircraft range d. An aircraft narrower than the aperture has
  an umbra that CLOSES at Δz = d·a/(R − a) and is all penumbra beyond, which is why the dark core
  fades on its own even before diffraction acts.

  FAR BEHIND, diffraction. The surrounding beam spreads across the shadow once the Fresnel scale
  √(λ·Δz) exceeds the aircraft, i.e. beyond Δz ≈ a²/λ (~17 km for a 30 m half-span at 5.8 GHz).
  There the beam has its original shape again, carrying (1 − captured) of its power.

What this is NOT is an obstructed-diffraction calculation: it has no shadow-edge fringes, and the
transition between the two limits is an interpolation rather than a propagation. The rigorous
version is Babinet with the silhouette approximated by a separable rectangle — the aperture
illumination and the residual focusing phase are both separable, so the blocked field is a
difference of two separable terms and each propagates as a pair of 1-D transforms, which is exactly
what `aperture.plane_intensity` already does with a singleton output axis. That is the upgrade path
if the shadow ever has to carry more than an exposure read, and the method name in P5 says which
one produced a given run.
"""

from __future__ import annotations

import numpy as np

METHOD = "ray_umbra_penumbra_fresnel_blend"

_BISECTION_STEPS = 24
"""Fixed number of steps for the elliptical shadow-boundary solve below. Data-dependent iteration
goes lockstep (I9): every point pays the same 24 steps, which is a few hundred nanoseconds and
vectorizes, instead of a per-point convergence test that does not."""


def _elliptical_lit_fraction(x, y, umbra, penumbra):
    """How lit a point is between two centred, axis-aligned ellipses, in [0, 1].

    Interpolating a scalar "which shadow ellipse is this point on" rather than blending the two
    axes independently: the family of ellipses with half-axes umbra + λ·(penumbra − umbra) sweeps
    the penumbral annulus exactly once as λ goes 0 → 1, and

        f(λ) = (x/(u_x + λΔ_x))² + (y/(u_y + λΔ_y))² − 1

    is monotone decreasing in λ, so λ is the lit fraction and a fixed bisection finds it. Blending
    per axis instead would put a diamond-shaped seam through the penumbra.
    """
    delta = penumbra - umbra
    low = np.zeros_like(x)
    high = np.ones_like(x)

    def residual(lam):
        semi_x = np.maximum(umbra[..., 0] + lam * delta[..., 0], 1e-12)
        semi_y = np.maximum(umbra[..., 1] + lam * delta[..., 1], 1e-12)
        return (x / semi_x) ** 2 + (y / semi_y) ** 2 - 1.0

    inside_umbra = residual(low) <= 0.0        # inside the umbra at λ=0 → fully shadowed
    outside_penumbra = residual(high) > 0.0    # still outside at λ=1 → fully lit
    for _ in range(_BISECTION_STEPS):
        middle = 0.5 * (low + high)
        too_far_in = residual(middle) > 0.0    # outside this ellipse → the boundary is inward
        low = np.where(too_far_in, low, middle)
        high = np.where(too_far_in, high, middle)
    lit = 0.5 * (low + high)
    return np.where(inside_umbra, 0.0, np.where(outside_penumbra, 1.0, lit))


def transmission(points_m, receiver_range_m, silhouette_half_extent_m, aperture_half_extent_m,
                 wavelength_m, captured_fraction):
    """Fraction of the beam surviving the receiver at each query point.

    All arguments broadcast over a leading batch axis of (candidate × sample) members:
      `points_m`                    (m, n_point, 3) in the aperture frame
      `receiver_range_m`            (m,)
      `silhouette_half_extent_m`    (m, 2) on the aperture's (x, y) axes
      `aperture_half_extent_m`      (m, 2) — the aperture's own semi-axes
      `wavelength_m`                (m,)
      `captured_fraction`           (m,) — P6's GEOMETRIC capture

    1.0 everywhere at or ahead of the receiver, so the same call covers a whole side view.
    """
    points = np.asarray(points_m, dtype=float)
    x = points[..., 0]
    y = points[..., 1]
    beyond = points[..., 2] - np.asarray(receiver_range_m, dtype=float)[:, None]
    range_to_receiver = np.maximum(np.asarray(receiver_range_m, dtype=float), 1e-9)[:, None, None]

    silhouette = np.maximum(np.asarray(silhouette_half_extent_m, dtype=float), 1e-9)[:, None, :]
    aperture = np.asarray(aperture_half_extent_m, dtype=float)[:, None, :]
    step = np.clip(beyond, 0.0, None)[..., None] / range_to_receiver
    umbra = np.clip(silhouette + (silhouette - aperture) * step, 0.0, None)
    penumbra = silhouette + (silhouette + aperture) * step
    lit = _elliptical_lit_fraction(x, y, umbra, penumbra)

    # The blend weight puts each end at its honest limit, with the crossover where the physics says
    # it is: the Fresnel scale reaching the smaller silhouette half-extent.
    smallest = np.min(silhouette, axis=-1)
    fade = np.clip(1.0 - np.sqrt(np.asarray(wavelength_m, dtype=float)[:, None]
                                 * np.clip(beyond, 0.0, None)) / smallest, 0.0, 1.0)
    survives = 1.0 - np.asarray(captured_fraction, dtype=float)[:, None]
    shadowed = fade * lit + (1.0 - fade) * survives
    return np.where(beyond > 0.0, shadowed, 1.0)


def parameters(captured_fraction) -> dict:
    """What goes in the manifest so the shadow model is answerable from the product."""
    return {"method": METHOD, "bisection_steps": _BISECTION_STEPS,
            "capture_source": "capture.geometric",
            "far_field_transmission": "1 - geometric capture",
            "blend": "fade = 1 - sqrt(lambda*dz)/min(silhouette half-extent)"}
