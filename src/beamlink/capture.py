"""Capture: how much of a radiated beam the aircraft silhouette actually intercepts.

The physics of the overlap itself, deliberately free of `LinkConfig` and of the link chain:
everything here takes primitives (a silhouette raster, an aperture radius, a wavelength, a
range) and returns numbers. That is what makes this the file to edit when refining capture -
it cannot reach the architecture catalogue or the cost model, and they cannot reach into it.
The cfg-aware glue that picks a taper, enforces the density cap and assembles the reported
stages stays in `beam_link`, which re-exports these names as the engine's public face.
"""

from __future__ import annotations

import numpy as np

from beamlink import relations
from beamlink import diffraction
from beamlink import bwb_model

def equivalent_disc_radius(peak_intensity):
    """Single length scale for a spot of any shape: the radius of the uniform disc that would
    carry all the power at the observed peak intensity (pi*r^2*I_peak = 1). Takes the
    unit-power normalised intensity from `diffraction`, so 1/m^2 in, m out."""
    return float(1.0 / np.sqrt(np.pi * peak_intensity)) if peak_intensity > 0 else np.inf


def overlap_capture(raster, emitter_radius_m, wavelength_m, slant_m, shaped=False,
                    incidence_floor=0.90, aspect_cap=8.0, taper_p=0.0,
                    focus_range_m=None, pointing_offset_m=0.0, fixed_aspect=None,
                    n_grid=384, return_field=False):
    """Fraction of radiated power the aircraft silhouette actually intercepts.

    Integrates the diffracted intensity over the receiver elements of a raster from
    bwb_model. Returns (geometric capture, capture including the incidence derate, peak
    intensity in 1/m^2), all per unit radiated power, so the first two are fractions <= 1.

    With `shaped`, the emitter aperture is made elliptical and aligned with the silhouette:
    the aperture is stretched along whichever direction the target is *narrow* in, because
    spot width scales as 1/aperture width, giving a spot elongated the same way as the
    target. `aspect_cap` bounds how far that is pushed.
    """
    u, v, inc, proj = raster["u"], raster["v"], raster["incidence"], raster["proj_area"]
    focus = float(slant_m if focus_range_m is None else focus_range_m)
    if u.size == 0:
        empty = dict(u=u, v=v, beam=np.array([]), incidence=inc, proj_area=proj)
        return (0.0, 0.0, 0.0, empty) if return_field else (0.0, 0.0, 0.0)

    # Rectenna angular response only -- the cos(theta) area foreshortening is already in
    # proj_area. The 0.90 default is a conformal phased-array rectenna, whose elements re-phase
    # toward the arrival angle and so keep most of their off-normal match; a fixed patch array
    # would sit nearer 0.70.
    inc_eff = incidence_floor + (1.0 - incidence_floor) * inc
    # Principal axes of the silhouette (for the shaped-ellipse orientation AND the pointing
    # offset — a pointing error across the thin minor axis is the worst case).
    if u.size >= 3:
        # principal axes of the silhouette, weighted by the area each element presents
        evals, evecs = np.linalg.eigh(np.cov(np.vstack([u, v]), aweights=np.maximum(proj, 1e-12)))
    else:
        evals, evecs = np.array([1.0, 1.0]), np.eye(2)
    # eigh sorts ascending, so column 0 is the least-spread (narrow) direction. Offsetting
    # the target along it is the worst case for a pointing error.
    minor_axis = evecs[:, 0]
    du, dv = pointing_offset_m * minor_axis[0], pointing_offset_m * minor_axis[1]
    pc = evecs.T @ np.vstack([u - du, v - dv])         # rows: [along minor, along major]

    if not shaped or u.size < 3:
        aspect = 1.0
    elif fixed_aspect is not None:                      # FIXED elliptical emitter (built once)
        aspect = max(float(fixed_aspect), 1.0)
    else:                                               # adaptive: aspect = silhouette elongation
        aspect = float(np.clip(np.sqrt(max(evals[1], 1e-12) / max(evals[0], 1e-12)), 1.0, aspect_cap))
    # aperture semi-axes at constant area: major along pc[0], the silhouette's narrow axis
    ap_major, ap_minor = emitter_radius_m * np.sqrt(aspect), emitter_radius_m / np.sqrt(aspect)

    # Sampling grid spans the silhouette with 15% slack. The floors only stop it degenerating
    # when the target subtends almost nothing; each axis takes 5% of the *conjugate* aperture
    # semi-axis, which carries the same aspect scaling as the spot on that axis (the spot on x
    # goes as 1/ap_major, i.e. aspect^-1/2, and so does ap_minor).
    rx = max(float(np.abs(pc[0]).max()), ap_minor * 0.05, 1.0) * 1.15
    ry = max(float(np.abs(pc[1]).max()), ap_major * 0.05, 1.0) * 1.15
    x_ax = np.linspace(-rx, rx, n_grid)
    y_ax = np.linspace(-ry, ry, n_grid)
    intensity = diffraction.focal_intensity_grid(slant_m, focus, ap_major, ap_minor, wavelength_m,
                                                 x_ax, y_ax, taper_p=taper_p)
    beam = diffraction.sample_focal_intensity(x_ax, y_ax, intensity, pc[0], pc[1])

    geom = min(float(np.sum(beam * proj)), 1.0)
    eff = min(float(np.sum(beam * proj * inc_eff)), 1.0)
    # Peak areal intensity (for the W/m² density cap) at the spot centre — evaluate it exactly rather
    # than take the fine-grid max (the centre need not land on a grid node).
    peak = float(diffraction.focal_intensity_at(slant_m, focus, ap_major, ap_minor, wavelength_m,
                                                np.zeros(1), np.zeros(1), taper_p=taper_p)[0])
    if return_field:
        return geom, eff, peak, dict(u=u, v=v, beam=beam, incidence=inc, proj_area=proj,
                                     x_axis=x_ax, y_axis=y_ax, intensity=intensity,
                                     aspect=aspect, taper_p=float(taper_p))
    return geom, eff, peak


def receiver_radius_from_silhouette(silhouette_area_m2):
    """Equal-area circular radius, so a silhouette can be fed to the circular-aperture
    coupling formulas in napkin."""
    return np.sqrt(silhouette_area_m2 / np.pi)


def size_emitter_diameter(freq_ghz, slant_m, receiver_radius_m, tau_target=2.0):
    """Emitter diameter for a target near-field coupling τ (τ=2 → ~98% capture)."""
    return 2.0 * relations.required_emitter_radius(freq_ghz * 1e9, slant_m, receiver_radius_m, tau_target)


def _overlap_taper_grid(raster, emitter_radius_m, wavelength_m, slant_m, shaped,
                        incidence_floor, aspect_cap, taper_ps, pointing_offset_m,
                        fixed_aspect, n_grid=256):
    """`overlap_capture` for a whole set of taper exponents at once, in focus.

    Same geometry setup, but one batched FFT on a shared grid instead of a per-case DFT on a
    grid fitted to the target: cheap enough to scan taper, accurate enough to pick a winner,
    which is then re-evaluated properly.
    """
    u, v, inc, proj = raster["u"], raster["v"], raster["incidence"], raster["proj_area"]
    taper_ps = np.atleast_1d(np.asarray(taper_ps, dtype=float))
    if u.size == 0:
        z = np.zeros(taper_ps.size)
        return z, z, z.copy()
    inc_eff = incidence_floor + (1.0 - incidence_floor) * inc
    if u.size >= 3:
        evals, evecs = np.linalg.eigh(np.cov(np.vstack([u, v]), aweights=np.maximum(proj, 1e-12)))
    else:
        evals, evecs = np.array([1.0, 1.0]), np.eye(2)
    minor_axis = evecs[:, 0]
    du, dv = pointing_offset_m * minor_axis[0], pointing_offset_m * minor_axis[1]
    pc = evecs.T @ np.vstack([u - du, v - dv])
    if not shaped or u.size < 3:
        aspect = 1.0
    elif fixed_aspect is not None:
        aspect = max(float(fixed_aspect), 1.0)
    else:
        aspect = float(np.clip(np.sqrt(max(evals[1], 1e-12) / max(evals[0], 1e-12)), 1.0, aspect_cap))
    ap_major, ap_minor = emitter_radius_m * np.sqrt(aspect), emitter_radius_m / np.sqrt(aspect)
    x_ax, y_ax, stack = diffraction.plane_intensity_stack(
        slant_m, slant_m, ap_major, ap_minor, wavelength_m, taper_ps, n_grid=n_grid)
    beam = diffraction.sample_stack(x_ax, y_ax, stack, pc[0], pc[1])   # (K, n_samples)
    geoms = np.minimum((beam * proj).sum(axis=1), 1.0)
    effs = np.minimum((beam * proj * inc_eff).sum(axis=1), 1.0)
    peaks = stack.reshape(stack.shape[0], -1).max(axis=1)
    return geoms, effs, peaks


def _peak_density(delivered_mw, capture, rect_total, peak_intensity):
    """Peak power density (W/m^2) on the aircraft skin for a given delivered DC power.

    Dividing the delivered DC power by capture and the rectenna terms gives the beam power
    crossing the aircraft, and `peak_intensity` is normalised per unit power, so the
    atmospheric and transmitter losses upstream of this point drop out of the ratio.
    """
    if delivered_mw is None or capture <= 0:
        return 0.0
    p_spot_w = delivered_mw * 1e6 / max(capture * rect_total, 1e-9)
    return p_spot_w * peak_intensity
