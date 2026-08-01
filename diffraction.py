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
import scipy.fft as sfft
from scipy.special import j0, j1
from scipy.ndimage import map_coordinates


def _fft_workers(n_grid):
    """FFT thread count by grid size: the small channel transforms (n≤256) are fastest
    single-threaded (thread setup dominates the work); the larger capture/figure grids
    (n≥320) pay for multi-threading. Measured crossover sits between 256 and 384."""
    return -1 if n_grid >= 320 else 1


def aperture_amplitude(x, y, semi_major_m, semi_minor_m, taper_p):
    """Aperture amplitude A(x,y): (1−ρ²)^p inside the ellipse, 0 outside.
    Aperture major axis is along x, minor along y (the caller rotates into this frame)."""
    rho2 = (x / semi_major_m) ** 2 + (y / semi_minor_m) ** 2
    inside = rho2 <= 1.0
    return np.where(inside, np.clip(1.0 - rho2, 0.0, None) ** float(taper_p), 0.0)


def _aperture_spacing(semi_major_m, semi_minor_m, wavelength_m, z_m, focus_range_m, n_grid=1024):
    """Aperture-plane sample spacing d for a single-FFT propagation to plane z of an aperture
    focused at `focus_range_m`. Two Nyquist constraints:
      • Focal-plane spacing dx = λ·z/(N·d) and extent = λ·z/d, so the SPOT (widest along the
        SHORT aperture axis) sets d = min_semi/12 → focal window ~12 spot-scales wide. But the
        aperture-plane window is N·d, so d is floored at 2.5·max_semi/N so the LONG aperture
        axis always fits the transform grid (else an elongated aperture is spuriously clipped
        and the plane is dropped by the near-aperture guard in axial_cut).
      • Off-focus, the aperture carries a residual quadratic phase exp(iπ·(1/z−1/f)·ρ²/λ) whose
        edge frequency is max_semi·|1/z−1/f|/λ; d must resolve it (d ≤ λ/(4·max_semi·|1/z−1/f|)).
    The tighter of the two wins (the phase term vanishes at z = f, so the spot term rules there)."""
    min_semi, max_semi = min(semi_major_m, semi_minor_m), max(semi_major_m, semi_minor_m)
    d_spot = max(min_semi / 12.0, 2.5 * max_semi / n_grid)
    inv = abs(1.0 / max(z_m, 1e-9) - 1.0 / max(focus_range_m, 1e-9))
    d_phase = wavelength_m / (4.0 * max_semi * inv) if inv > 0 else np.inf
    return min(d_spot, d_phase)


def plane_intensity(z_m, focus_range_m, semi_major_m, semi_minor_m, wavelength_m,
                    taper_p=0.0, n_grid=1024, dtype=np.complex128):
    """Power-normalized scalar-diffraction intensity in the plane at range `z_m` of an
    elliptical aperture phase-conjugated to focus at `focus_range_m`. Single-FFT Fresnel:
    the field at z is the 2-D Fourier transform of the aperture amplitude carrying the
    residual quadratic phase exp(iπ(1/z − 1/f)ρ²/λ), with aperture frequency mapped to
    transverse position x = λ·z·f_freq. At z = f this collapses to the Fraunhofer (focal)
    pattern. Aperture major axis is along x → the spot is NARROW along x (Fourier inverse).
    Returns (x_m, y_m, intensity) with ∫∫ I dx dy = 1 (fraction of radiated power / m²).
    `dtype` sets the FFT precision — complex128 for the model, complex64 for the (display-only)
    channel cuts where the 2× cheaper transform is visually indistinguishable."""
    d = _aperture_spacing(semi_major_m, semi_minor_m, wavelength_m, z_m, focus_range_m, n_grid)
    coord = (np.arange(n_grid) - n_grid // 2) * d
    X, Y = np.meshgrid(coord, coord, indexing="ij")
    A = aperture_amplitude(X, Y, semi_major_m, semi_minor_m, taper_p).astype(dtype)
    inv = 1.0 / max(z_m, 1e-9) - 1.0 / max(focus_range_m, 1e-9)
    if inv != 0.0:
        A *= np.exp(1j * np.pi / wavelength_m * inv * (X ** 2 + Y ** 2)).astype(dtype)

    field = sfft.fftshift(sfft.fft2(sfft.ifftshift(A), workers=_fft_workers(n_grid)))
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
    d = _aperture_spacing(semi_major_m, semi_minor_m, wavelength_m, z_m, focus_range_m, n_grid)
    coord = (np.arange(n_grid) - n_grid // 2) * d
    X, Y = np.meshgrid(coord, coord, indexing="ij")
    rho2 = (X / semi_major_m) ** 2 + (Y / semi_minor_m) ** 2
    inside = rho2 <= 1.0
    base = np.where(inside, np.clip(1.0 - rho2, 0.0, None), 0.0)
    stack = np.stack([np.where(inside, base ** p, 0.0) for p in taper_ps]).astype(np.complex64)
    inv = 1.0 / max(z_m, 1e-9) - 1.0 / max(focus_range_m, 1e-9)
    if inv != 0.0:
        stack *= np.exp(1j * np.pi / wavelength_m * inv * (X ** 2 + Y ** 2)).astype(np.complex64)

    field = sfft.fftshift(sfft.fft2(sfft.ifftshift(stack, axes=(1, 2)), axes=(1, 2),
                                    workers=_fft_workers(n_grid)), axes=(1, 2))
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


def focal_intensity_at(z_m, focus_range_m, semi_major_m, semi_minor_m, wavelength_m,
                       qx, qy, taper_p=0.0, n_ap=512):
    """Scalar-diffraction focal intensity evaluated DIRECTLY at arbitrary query points (qx, qy) in
    the plane at range `z_m` (aperture principal frame: x = aperture-major axis → spot-MINOR).

    A direct DFT from the aperture samples to the query points — NOT an FFT on a fixed grid. So the
    spot is resolved at the QUERY resolution however fine (evaluate on the silhouette's own pixels),
    and no ±λz/d window is spent computing empty space. `n_ap` sets the aperture sampling grid (the
    residual focusing phase and taper are carried on it); only its non-zero samples enter the sum.
    Power-normalised consistently with `plane_intensity` (∫∫ I dA = 1 over all space), verified to
    match it at shared grid points. `qx`, `qy` are broadcast/flattened together; returns |E|² at each.

    Cost is O(n_query · n_aperture_samples); the query loop is chunked to bound memory."""
    d = _aperture_spacing(semi_major_m, semi_minor_m, wavelength_m, z_m, focus_range_m, n_ap)
    coord = (np.arange(n_ap) - n_ap // 2) * d
    X, Y = np.meshgrid(coord, coord, indexing="ij")
    amp = aperture_amplitude(X, Y, semi_major_m, semi_minor_m, taper_p)
    inside = amp > 0.0
    ax, ay = X[inside], Y[inside]
    av = amp[inside].astype(np.complex128)
    inv = 1.0 / max(z_m, 1e-9) - 1.0 / max(focus_range_m, 1e-9)
    if inv != 0.0:                                         # residual quadratic (defocus) phase
        av = av * np.exp(1j * np.pi / wavelength_m * inv * (ax ** 2 + ay ** 2))
    qx = np.asarray(qx, dtype=float).ravel()
    qy = np.asarray(qy, dtype=float).ravel()
    k = 2.0 * np.pi / (wavelength_m * z_m)                 # aperture-freq → focal-position kernel
    norm = float((amp ** 2).sum()) * (wavelength_m * z_m / d) ** 2   # Parseval total (∫∫I dA = 1)
    out = np.empty(qx.size, dtype=float)
    chunk = max(1, int(4.0e6 / max(av.size, 1)))          # cap the (chunk × n_ap) complex block
    for s in range(0, qx.size, chunk):
        phase = -k * (np.outer(qx[s:s + chunk], ax) + np.outer(qy[s:s + chunk], ay))
        field = np.exp(1j * phase) @ av
        out[s:s + chunk] = field.real ** 2 + field.imag ** 2
    return out / norm if norm > 0 else out


def focal_intensity_grid(z_m, focus_range_m, semi_major_m, semi_minor_m, wavelength_m,
                         x_out, y_out, taper_p=0.0, n_ap=512):
    """Direct scalar-diffraction focal intensity on an axis-aligned OUTPUT grid (x_out × y_out,
    aperture principal frame: x = aperture-major → spot-MINOR). Because the output is a grid the DFT
    is SEPARABLE — E = Wx · A · Wy, two matrix products instead of a per-point sum — so it ZOOMS to
    whatever window/resolution you pass (e.g. the silhouette's own pixel grid, high-res on the spot)
    for the cost of a couple of BLAS calls, no ±λz/d FFT window wasted on empty space. Power-
    normalised like `plane_intensity` (∫∫ I dA = 1). Returns intensity[i, j] at (x_out[i], y_out[j])."""
    d = _aperture_spacing(semi_major_m, semi_minor_m, wavelength_m, z_m, focus_range_m, n_ap)
    coord = (np.arange(n_ap) - n_ap // 2) * d
    X, Y = np.meshgrid(coord, coord, indexing="ij")
    amp = aperture_amplitude(X, Y, semi_major_m, semi_minor_m, taper_p)
    A = amp.astype(np.complex128)
    inv = 1.0 / max(z_m, 1e-9) - 1.0 / max(focus_range_m, 1e-9)
    if inv != 0.0:
        A = A * np.exp(1j * np.pi / wavelength_m * inv * (X ** 2 + Y ** 2))
    k = 2.0 * np.pi / (wavelength_m * z_m)
    Wx = np.exp(-1j * k * np.outer(np.asarray(x_out, float), coord))   # (nx, N)
    Wy = np.exp(-1j * k * np.outer(coord, np.asarray(y_out, float)))   # (N, ny)
    field = Wx @ A @ Wy                                               # (nx, ny)
    norm = float((amp ** 2).sum()) * (wavelength_m * z_m / d) ** 2
    inten = field.real ** 2 + field.imag ** 2
    return inten / norm if norm > 0 else inten


def _cut_enclosed(line, coord, dcoord, marginal, radial_m, r_query):
    """One plane's cut resampled onto radial_m, plus its strip-marginal enclosed-energy row
    E(r) = fraction of the plane's TOTAL power in the strip |coord| < r (∫∫ I dA = 1, so it
    only reaches 1 where the beam is that contained)."""
    intensity_row = np.interp(radial_m, coord, line, left=0.0, right=0.0)
    o = np.argsort(np.abs(coord))
    abs_sorted = np.abs(coord)[o]
    cum = np.cumsum(marginal[o] * dcoord)
    # Linearly INTERPOLATE the enclosed-energy curve onto the query radii (not nearest-index via
    # searchsorted): otherwise the iso-level radius snaps to this plane's grid spacing dx, which
    # varies plane-to-plane, and the funnel contours staircase along the path.
    enclosed_row = np.interp(r_query, abs_sorted, cum, left=0.0,
                             right=(cum[-1] if cum.size else 0.0))
    return intensity_row, enclosed_row


def _finalize_cut(intensity, enclosed, normalize):
    """Peak-normalise (when asked) and round for compact serialization. The raw field
    (normalize=False) is areal intensity ~1e-4 W-fraction/m² and would collapse to 0 at 3
    decimals, so it is left full-precision for the caller to normalise across legs."""
    finite = np.isfinite(intensity)
    peak = intensity[finite].max() if finite.any() else 0.0
    if normalize and peak > 0:
        intensity = intensity / peak
    if normalize:
        intensity = np.round(intensity, 3)
    return intensity, np.round(np.clip(enclosed, 0.0, 1.0), 3)


def _cut_from_field(field, dx_ax, radial_m, r_query):
    """Both principal center-line cuts + strip-marginals of one plane's intensity `field`
    (square grid, spacing dx_ax on both axes; aperture major along axis 0). Returns
    (ix, ex, iy, ey) rows on `radial_m`. Shared by the batched and per-plane paths."""
    mid = field.shape[0] // 2
    coord = (np.arange(field.shape[0]) - mid) * dx_ax
    ix, ex = _cut_enclosed(field[:, mid], coord, dx_ax, field.sum(axis=1) * dx_ax, radial_m, r_query)
    iy, ey = _cut_enclosed(field[mid, :], coord, dx_ax, field.sum(axis=0) * dx_ax, radial_m, r_query)
    return ix, ex, iy, ey


def _axial_cuts(z_m, focus_range_m, semi_major_m, semi_minor_m, wavelength_m, radial_m,
                taper_p, n_grid, dtype):
    """Core of the axial-cut routines: for every plane z, both principal center-line cuts of the
    diffraction field, on the common `radial_m` grid, RAW (per-plane ∫∫ I dA = 1, unnormalised).

    The grid spacing d = min(d_spot, d_phase) (see `_aperture_spacing`) is pinned to the CONSTANT
    d_spot for every plane except the strongly-defocused near-aperture ones (where the residual
    quadratic phase forces a finer d_phase). All the d_spot planes therefore share one aperture
    grid, so they are computed as a SINGLE batched FFT with the aperture amplitude and radius²
    grid built once — far cheaper than one FFT + one aperture raster per plane, and numerically
    identical to the per-plane path (only the FFT accumulation order differs, ~1e-7). The few
    d_phase planes keep the exact per-plane path. Returns (ix, ex, iy, ey), NaN where a plane is
    too close to the aperture for its width to fit the transform grid."""
    z_m = np.asarray(z_m, dtype=float)
    radial_m = np.asarray(radial_m, dtype=float)
    r_query = np.abs(radial_m)
    shape = (z_m.size, radial_m.size)
    ix = np.full(shape, np.nan); ex = np.full(shape, np.nan)
    iy = np.full(shape, np.nan); ey = np.full(shape, np.nan)
    min_semi, max_semi = min(semi_major_m, semi_minor_m), max(semi_major_m, semi_minor_m)
    d_spot = max(min_semi / 12.0, 2.5 * max_semi / n_grid)

    inv = 1.0 / np.maximum(z_m, 1e-9) - 1.0 / max(focus_range_m, 1e-9)
    abs_inv = np.abs(inv)
    d_phase = np.full_like(abs_inv, np.inf)            # exact-focus plane (inv = 0) → d_spot rules
    nonzero = abs_inv > 0.0
    d_phase[nonzero] = wavelength_m / (4.0 * max_semi * abs_inv[nonzero])
    is_spot = d_phase >= d_spot                        # this plane's spacing is pinned to d_spot

    # --- batched d_spot planes: one aperture raster + radius² grid, one batched FFT ------------
    # The residual quadratic phase varies plane-to-plane only through the scalar `inv`, and the
    # aperture masks everything outside the (typically ~few-percent-fill) ellipse, so the phase is
    # evaluated ONLY at the aperture samples and scattered into the zero-padded transform grid —
    # the exp() cost then scales with the aperture area, not the whole grid. Phase in complex64
    # (float32 argument): a ~1e-4 rad rounding, invisible in the display field.
    spot_idx = np.nonzero(is_spot)[0]
    if spot_idx.size and n_grid * d_spot >= 2.4 * max_semi:
        coord = (np.arange(n_grid) - n_grid // 2) * d_spot
        X, Y = np.meshgrid(coord, coord, indexing="ij")
        mask = aperture_amplitude(X, Y, semi_major_m, semi_minor_m, taper_p) > 0.0
        taper_in = aperture_amplitude(X[mask], Y[mask], semi_major_m, semi_minor_m,
                                      taper_p).astype(np.float32)
        radius2_in = ((X[mask] ** 2 + Y[mask] ** 2) * (np.pi / wavelength_m)).astype(np.float32)
        chunk = max(1, int(24e6 / (n_grid * n_grid)))  # cap the padded stack ~200 MB (complex64)
        for start in range(0, spot_idx.size, chunk):
            gidx = spot_idx[start:start + chunk]
            invs = inv[gidx].astype(np.float32)
            vals = taper_in[None] * np.exp(1j * (invs[:, None] * radius2_in[None]))  # (chunk, n_ap)
            stack = np.zeros((gidx.size, n_grid, n_grid), dtype=np.complex64)
            stack[:, mask] = vals
            field = sfft.fftshift(sfft.fft2(sfft.ifftshift(stack, axes=(1, 2)), axes=(1, 2),
                                            workers=_fft_workers(n_grid)), axes=(1, 2))
            inten = field.real ** 2 + field.imag ** 2  # |field|² without the wasted sqrt of abs()
            dxs = wavelength_m * z_m[gidx] / (n_grid * d_spot)     # per-plane observation spacing
            totals = inten.sum(axis=(1, 2)) * dxs ** 2
            inten /= np.where(totals > 0, totals, 1.0)[:, None, None]
            for k, gi in enumerate(gidx):
                ix[gi], ex[gi], iy[gi], ey[gi] = _cut_from_field(inten[k], dxs[k], radial_m, r_query)

    # --- the few strongly-defocused planes: exact per-plane path ------------------------------
    for gi in np.nonzero(~is_spot)[0]:
        if n_grid * d_phase[gi] < 2.4 * max_semi:      # near-aperture (phase-limited) → skip (NaN)
            continue
        x_ax, _, field = plane_intensity(z_m[gi], focus_range_m, semi_major_m, semi_minor_m,
                                         wavelength_m, taper_p=taper_p, n_grid=n_grid, dtype=dtype)
        ix[gi], ex[gi], iy[gi], ey[gi] = _cut_from_field(field, x_ax[1] - x_ax[0], radial_m, r_query)
    return ix, ex, iy, ey


def axial_cut(z_m, focus_range_m, semi_major_m, semi_minor_m, wavelength_m, radial_m,
              taper_p=0.0, axis="x", n_grid=512, normalize=True, dtype=np.complex128):
    """(len(z), len(radial)) slice of the diffraction field ALONG the beam: for each plane z,
    the field is sampled along a principal `axis` ('x' = aperture-major axis → the NARROW spot
    cut; 'y' = aperture-minor axis → the WIDE spot cut) onto the common `radial_m` grid. Shows
    the converging cone → diffraction-limited waist → diverging cone, sidelobe rings included.

    Returns (intensity, enclosed): `intensity` is the areal intensity along the cut (fraction of
    radiated power / m² per plane), divided by its own global peak when `normalize` (so the
    waist is brightest and the beam dims through divergence); pass normalize=False to keep the
    raw per-plane intensity, which stays comparable across legs of a multi-hop channel. `enclosed`
    is the enclosed-energy-along-the-cut fraction E(z, r) = ∫_{|r'|≤r} I / ∫ I, for the funnel.
    Planes so close to the aperture that its width would be clipped by the transform grid are
    left NaN (the near-aperture field isn't fabricated). For BOTH cuts at once use axial_cut_xy."""
    ix, ex, iy, ey = _axial_cuts(z_m, focus_range_m, semi_major_m, semi_minor_m, wavelength_m,
                                 radial_m, taper_p, n_grid, dtype)
    return _finalize_cut(*(( ix, ex) if axis == "x" else (iy, ey)), normalize)


def axial_cut_xy(z_m, focus_range_m, semi_major_m, semi_minor_m, wavelength_m, radial_m,
                 taper_p=0.0, n_grid=512, normalize=True, dtype=np.complex128):
    """BOTH principal cuts of the axial field in ONE pass — the x-cut (narrow spot) and y-cut
    (wide spot) share every diffraction transform. Returns (ix, ex, iy, ey) — see axial_cut for
    the per-cut meaning. The planes at the common d_spot grid spacing are computed as one batched
    FFT (see `_axial_cuts`)."""
    ix, ex, iy, ey = _axial_cuts(z_m, focus_range_m, semi_major_m, semi_minor_m, wavelength_m,
                                 radial_m, taper_p, n_grid, dtype)
    ix, ex = _finalize_cut(ix, ex, normalize)
    iy, ey = _finalize_cut(iy, ey, normalize)
    return ix, ex, iy, ey


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
