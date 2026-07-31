"""Scalar-diffraction beam fields for the microwave power link.

The emitter is a phase-conjugated (focused) phased array of finite extent. The field
it delivers at the plane is NOT a Gaussian — it is the diffraction pattern of the
aperture: a main lobe carrying <100 % of the radiated power, surrounded by sidelobe
rings. This module computes that field exactly (scalar Fresnel/Fraunhofer optics) so
both the capture integral and the visualization run on the SAME physical field, with
no effective-spot calibration.

Two routines:
  • `focal_intensity`      — the field in the TARGET plane (z = focus range), where a
                             lens-focused aperture forms its Fraunhofer pattern. This is
                             what the plane actually flies through, and what capture
                             integrates over the silhouette.
  • `axial_intensity_cut`  — a (transverse × range) slice of the field along the whole
                             path, for the visualization (converging cone → waist →
                             diverging cone, sidelobes and all).

Aperture taper is the single physical knob (the array's amplitude weighting):
    A(ρ) = (1 − ρ²)^p ,  ρ = elliptical aperture radius ∈ [0, 1]
p = 0 is a uniform (top-hat) aperture → the Airy pattern (first sidelobe −17.6 dB, 83.8 %
of the power in the main lobe); larger p lowers the sidelobes at the cost of a wider main
lobe (p = 1 → −24.6 dB, p = 2 → −30.6 dB). The array picks p per geometry (see the
adaptive-taper optimizer in beam_link).

Aperture ↔ spot axes are Fourier-inverse: a LONG aperture axis makes a NARROW spot axis.
The elliptical aperture is described by its own semi-axes; the caller orients it.
"""
from __future__ import annotations

import numpy as np
from scipy.special import j0, j1
from scipy.ndimage import map_coordinates


def aperture_amplitude(x, y, semi_major_m, semi_minor_m, taper_p):
    """Aperture amplitude A(x,y): (1−ρ²)^p inside the ellipse, 0 outside.
    Aperture major axis is along x, minor along y (the caller rotates into this frame)."""
    rho2 = (x / semi_major_m) ** 2 + (y / semi_minor_m) ** 2
    inside = rho2 <= 1.0
    return np.where(inside, np.clip(1.0 - rho2, 0.0, None) ** float(taper_p), 0.0)


def _aperture_spacing(semi_major_m, semi_minor_m, wavelength_m, z_m, focus_range_m):
    """Aperture-plane sample spacing d for a single-FFT propagation to plane z of an aperture
    focused at `focus_range_m`. Two Nyquist constraints:
      • Focal-plane spacing dx = λ·z/(N·d) and extent = λ·z/d, so the SPOT (widest along the
        SHORT aperture axis) sets d = min_semi/12 → focal window ~12 spot-scales wide.
      • Off-focus, the aperture carries a residual quadratic phase exp(iπ·(1/z−1/f)·ρ²/λ) whose
        edge frequency is max_semi·|1/z−1/f|/λ; d must resolve it (d ≤ λ/(4·max_semi·|1/z−1/f|)).
    The tighter of the two wins (the phase term vanishes at z = f, so the spot term rules there)."""
    min_semi, max_semi = min(semi_major_m, semi_minor_m), max(semi_major_m, semi_minor_m)
    d_spot = min_semi / 12.0
    inv = abs(1.0 / max(z_m, 1e-9) - 1.0 / max(focus_range_m, 1e-9))
    d_phase = wavelength_m / (4.0 * max_semi * inv) if inv > 0 else np.inf
    return min(d_spot, d_phase)


def plane_intensity(z_m, focus_range_m, semi_major_m, semi_minor_m, wavelength_m,
                    taper_p=0.0, n_grid=1024):
    """Power-normalized scalar-diffraction intensity in the plane at range `z_m` of an
    elliptical aperture phase-conjugated to focus at `focus_range_m`. Single-FFT Fresnel:
    the field at z is the 2-D Fourier transform of the aperture amplitude carrying the
    residual quadratic phase exp(iπ(1/z − 1/f)ρ²/λ), with aperture frequency mapped to
    transverse position x = λ·z·f_freq. At z = f this collapses to the Fraunhofer (focal)
    pattern. Aperture major axis is along x → the spot is NARROW along x (Fourier inverse).
    Returns (x_m, y_m, intensity) with ∫∫ I dx dy = 1 (fraction of radiated power / m²)."""
    d = _aperture_spacing(semi_major_m, semi_minor_m, wavelength_m, z_m, focus_range_m)
    coord = (np.arange(n_grid) - n_grid // 2) * d
    X, Y = np.meshgrid(coord, coord, indexing="ij")
    A = aperture_amplitude(X, Y, semi_major_m, semi_minor_m, taper_p).astype(complex)
    inv = 1.0 / max(z_m, 1e-9) - 1.0 / max(focus_range_m, 1e-9)
    if inv != 0.0:
        A *= np.exp(1j * np.pi / wavelength_m * inv * (X ** 2 + Y ** 2))

    field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(A)))
    intensity = np.abs(field) ** 2

    dx = wavelength_m * z_m / (n_grid * d)                # observation-plane sample spacing (m)
    x = (np.arange(n_grid) - n_grid // 2) * dx
    total = intensity.sum() * dx * dx
    intensity = intensity / total if total > 0 else intensity
    return x, x.copy(), intensity


def plane_intensity_stack(z_m, focus_range_m, semi_major_m, semi_minor_m, wavelength_m,
                          taper_ps, n_grid=384):
    """`plane_intensity` for a whole LIST of aperture tapers at once. The aperture spacing,
    grid and residual focusing phase depend only on the geometry (not the taper), so all
    tapers share them and the transform is a SINGLE batched 2-D FFT over the stack — far
    cheaper than one FFT per taper. Returns (x_m, y_m, stack) with stack[k] the
    power-normalised intensity (∫∫ I dA = 1) for taper_ps[k]."""
    taper_ps = np.atleast_1d(np.asarray(taper_ps, dtype=float))
    d = _aperture_spacing(semi_major_m, semi_minor_m, wavelength_m, z_m, focus_range_m)
    coord = (np.arange(n_grid) - n_grid // 2) * d
    X, Y = np.meshgrid(coord, coord, indexing="ij")
    rho2 = (X / semi_major_m) ** 2 + (Y / semi_minor_m) ** 2
    inside = rho2 <= 1.0
    base = np.where(inside, np.clip(1.0 - rho2, 0.0, None), 0.0)
    stack = np.stack([np.where(inside, base ** p, 0.0) for p in taper_ps]).astype(np.complex64)
    inv = 1.0 / max(z_m, 1e-9) - 1.0 / max(focus_range_m, 1e-9)
    if inv != 0.0:
        stack *= np.exp(1j * np.pi / wavelength_m * inv * (X ** 2 + Y ** 2)).astype(np.complex64)

    field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(stack, axes=(1, 2)), axes=(1, 2)), axes=(1, 2))
    intensity = np.abs(field) ** 2

    dx = wavelength_m * z_m / (n_grid * d)
    x = (np.arange(n_grid) - n_grid // 2) * dx
    total = intensity.sum(axis=(1, 2)) * dx * dx
    intensity = intensity / np.where(total > 0, total, 1.0)[:, None, None]
    return x, x.copy(), intensity


def focal_intensity(semi_major_m, semi_minor_m, wavelength_m, focus_range_m,
                    taper_p=0.0, n_grid=1024):
    """Intensity in the target plane (z = focus_range) — plane_intensity at exact focus."""
    return plane_intensity(focus_range_m, focus_range_m, semi_major_m, semi_minor_m,
                           wavelength_m, taper_p=taper_p, n_grid=n_grid)


def axial_cut(z_m, focus_range_m, semi_major_m, semi_minor_m, wavelength_m, radial_m,
              taper_p=0.0, axis="x", n_grid=512):
    """(len(z), len(radial)) slice of the diffraction field ALONG the beam: for each plane z,
    the field is sampled along a principal `axis` ('x' = aperture-major axis → the NARROW spot
    cut; 'y' = aperture-minor axis → the WIDE spot cut) onto the common `radial_m` grid. Shows
    the converging cone → diffraction-limited waist → diverging cone, sidelobe rings included.

    Returns (intensity, enclosed): `intensity` is normalised to its own global peak (so the
    waist is brightest and the beam dims through the divergence); `enclosed` is the
    enclosed-energy-along-the-cut fraction E(z, r) = ∫_{|r'|≤r} I / ∫ I, for the funnel overlay.
    Planes so close to the aperture that its width would be clipped by the transform grid are
    left NaN (the near-aperture field isn't fabricated)."""
    z_m = np.asarray(z_m, dtype=float)
    radial_m = np.asarray(radial_m, dtype=float)
    intensity = np.full((z_m.size, radial_m.size), np.nan)
    enclosed = np.full((z_m.size, radial_m.size), np.nan)
    order = np.argsort(np.abs(radial_m))
    max_semi = max(semi_major_m, semi_minor_m)
    for i, z in enumerate(z_m):
        d = _aperture_spacing(semi_major_m, semi_minor_m, wavelength_m, z, focus_range_m)
        if n_grid * d < 2.4 * max_semi:                # aperture would be clipped → skip
            continue
        x_ax, y_ax, field = plane_intensity(z, focus_range_m, semi_major_m, semi_minor_m,
                                            wavelength_m, taper_p=taper_p, n_grid=n_grid)
        mid = field.shape[0] // 2
        line = field[:, mid] if axis == "x" else field[mid, :]
        ax = x_ax if axis == "x" else y_ax
        row = np.interp(radial_m, ax, line, left=0.0, right=0.0)
        intensity[i] = row
        total = row.sum()
        if total > 0:
            cdf = np.empty_like(row)
            cdf[order] = np.cumsum(row[order])
            enclosed[i] = cdf / total
    peak = np.nanmax(intensity)
    if peak and peak > 0:
        intensity = intensity / peak
    # Round for compact serialization (3 decimals is ample for a 0–1 display field).
    return np.round(intensity, 3), np.round(enclosed, 3)


def sample_focal_intensity(x_axis, y_axis, intensity, xq, yq):
    """Bilinearly sample a focal-plane intensity grid at query points (xq, yq) in metres
    (aperture principal frame: x along aperture major axis). Out-of-grid points → 0."""
    dx = x_axis[1] - x_axis[0]
    dy = y_axis[1] - y_axis[0]
    ix = (np.asarray(xq) - x_axis[0]) / dx
    iy = (np.asarray(yq) - y_axis[0]) / dy
    return map_coordinates(intensity, np.vstack([ix.ravel(), iy.ravel()]),
                           order=1, mode="constant", cval=0.0).reshape(np.shape(xq))


def sample_stack(x_axis, y_axis, stack, xq, yq):
    """Bilinearly sample a STACK of intensity grids (K, N, N) at the SAME query points
    (xq, yq) — the taper grid shares one silhouette, so the fractional indices are computed
    once and applied to every layer with a single vectorized gather. Returns (K, n_query).
    Out-of-grid points → 0."""
    dx = x_axis[1] - x_axis[0]
    dy = y_axis[1] - y_axis[0]
    n = stack.shape[1]
    fx = (np.asarray(xq, dtype=float) - x_axis[0]) / dx
    fy = (np.asarray(yq, dtype=float) - y_axis[0]) / dy
    x0, y0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
    wx, wy = fx - x0, fy - y0
    valid = (x0 >= 0) & (x0 < n - 1) & (y0 >= 0) & (y0 < n - 1)
    xi, yi = np.clip(x0, 0, n - 2), np.clip(y0, 0, n - 2)
    out = (stack[:, xi, yi] * ((1 - wx) * (1 - wy))
           + stack[:, xi + 1, yi] * (wx * (1 - wy))
           + stack[:, xi, yi + 1] * ((1 - wx) * wy)
           + stack[:, xi + 1, yi + 1] * (wx * wy))
    return out * valid


# --- Analytic references (validation) ----------------------------------------

def airy_encircled_energy(radius_m, aperture_radius_m, wavelength_m, focus_range_m):
    """Fraction of power within `radius_m` of the axis in the focal plane of a UNIFORM
    circular aperture (Airy pattern): E(r) = 1 − J0(v)² − J1(v)², v = 2π·R·r/(λ·f).
    First dark ring at v = 3.8317 (r = 0.610 λf/R); E there = 0.838 (main-lobe power)."""
    v = 2.0 * np.pi * aperture_radius_m * np.asarray(radius_m) / (wavelength_m * focus_range_m)
    v = np.where(v == 0.0, 1e-12, v)
    return 1.0 - j0(v) ** 2 - j1(v) ** 2


def encircled_energy(x_axis, y_axis, intensity, radii_m):
    """Numerically integrate a focal-plane intensity grid to the encircled-energy curve
    E(r) for a set of radii (m), centred on the grid origin."""
    dx = x_axis[1] - x_axis[0]
    dy = y_axis[1] - y_axis[0]
    X, Y = np.meshgrid(x_axis, y_axis, indexing="ij")
    R = np.sqrt(X ** 2 + Y ** 2)
    return np.array([float((intensity[R <= r]).sum() * dx * dy) for r in np.atleast_1d(radii_m)])
