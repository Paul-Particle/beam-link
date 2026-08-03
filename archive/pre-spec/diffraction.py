"""Scalar diffraction from an elliptical, tapered, focused aperture.

The intensity routines return intensity normalised to unit radiated power: the integral over
the observation plane is 1, so values have units of m^-2 and W/m^2 comes from multiplying by
the beam power. `axial_cut`/`axial_cut_xy` are the exception -- they peak-normalise by default
and are dimensionless; `axial_cuts_direct`, despite the name, is unit-power like the rest. `focus_range_m` is the range the aperture phase is curved for; setting
it equal to `z_m` means "in focus at z".
"""

from __future__ import annotations

import numpy as np
import scipy.fft as sfft
from scipy.special import j0, j1
from scipy.ndimage import map_coordinates


def _fft_workers(n_grid):
    return -1 if n_grid >= 320 else 1  # threading only pays off on the larger transforms


def aperture_amplitude(x, y, semi_major_m, semi_minor_m, taper_p):
    """Illumination amplitude (1 - rho^2)^p inside the ellipse, 0 outside. p=0 is uniform."""
    rho2 = (x / semi_major_m) ** 2 + (y / semi_minor_m) ** 2
    inside = rho2 <= 1.0
    return np.where(inside, np.clip(1.0 - rho2, 0.0, None) ** float(taper_p), 0.0)


def _aperture_spacing(semi_major_m, semi_minor_m, wavelength_m, z_m, focus_range_m, n_grid=1024):
    """Aperture-plane sample pitch (m), from two competing constraints.

    A smaller pitch widens the observation window (n*dx = lambda*z/d) at the cost of a coarser
    pitch within it (dx = lambda*z/(n*d)) and less room for the aperture itself (n*d):
      d_spot  - min_semi/12, which makes the window exactly 12 spot scales wide; floored at
                2.5*max_semi/n_grid so the long aperture axis always fits the grid.
      d_phase - the focusing phase pi/lambda*inv*r^2 must advance < pi/2 per sample at the
                aperture edge; unconstrained when in focus (inv = 0).
    """
    min_semi, max_semi = min(semi_major_m, semi_minor_m), max(semi_major_m, semi_minor_m)
    d_spot = max(min_semi / 12.0, 2.5 * max_semi / n_grid)
    inv = abs(1.0 / max(z_m, 1e-9) - 1.0 / max(focus_range_m, 1e-9))
    d_phase = wavelength_m / (4.0 * max_semi * inv) if inv > 0 else np.inf
    return min(d_spot, d_phase)


N_AP_FIT_CAP = 4096


class ApertureTooFine(ValueError):
    """The aperture cannot be sampled finely enough to be represented at this focus and range.

    Its own type so callers can back off from a geometry the sampling cannot reach without also
    swallowing every other ValueError the capture chain might raise. Subclasses ValueError so
    existing `except ValueError` handlers keep working.
    """


def _aperture_grid(semi_major_m, semi_minor_m, wavelength_m, z_m, focus_range_m, n_ap,
                   cap=N_AP_FIT_CAP):
    """Sample pitch AND a grid size that actually contains the aperture: (d, n).

    Only `_aperture_spacing`'s d_spot branch carries the 2.5*max_semi/n_grid floor that keeps the
    aperture inside the grid. When the focusing phase binds instead (d_phase < d_spot — deep
    defocus, z far from the focus) there is no such guarantee, and building the aperture on an
    n_ap-wide grid would silently CLIP it: the field returned would be a smaller emitter's, with no
    error. So grow n until the aperture fits at the pitch chosen for that n (the pitch can shrink as
    n grows, hence the loop; it converges because d is floored by min(min_semi/12, d_phase)), and
    refuse past `cap` rather than allocating an n^2 aperture array.
    """
    max_semi = max(float(semi_major_m), float(semi_minor_m))
    n = int(n_ap)
    for _ in range(8):
        d = _aperture_spacing(semi_major_m, semi_minor_m, wavelength_m, z_m, focus_range_m, n)
        n_need = int(np.ceil(2.5 * max_semi / d))
        if n_need <= n:
            return d, n
        if n_need > cap:
            raise ApertureTooFine(
                f"aperture would be clipped: semi-axes ({semi_major_m:.3g}, {semi_minor_m:.3g}) m at "
                f"z={z_m:.4g} m, focus={focus_range_m:.4g} m need n_ap >= {n_need} (cap {cap}). The "
                f"focusing phase sets the pitch here; raise the cap or evaluate no closer to the "
                f"aperture than the Fresnel regime supports.")
        n = n_need
    return d, n


def plane_intensity(z_m, focus_range_m, semi_major_m, semi_minor_m, wavelength_m,
                    taper_p=0.0, n_grid=1024, dtype=np.complex128):
    """Intensity over a full plane at range z, via one FFT of the aperture field.

    Returns (x_axis, y_axis, intensity) with the axes in metres on the plane. The 1/(lambda z)
    amplitude factor and the observation-plane quadratic phase are dropped: the explicit
    normalisation to unit total power absorbs the former, and only |field|^2 is used.
    """
    d = _aperture_spacing(semi_major_m, semi_minor_m, wavelength_m, z_m, focus_range_m, n_grid)
    coord = (np.arange(n_grid) - n_grid // 2) * d
    X, Y = np.meshgrid(coord, coord, indexing="ij")
    A = aperture_amplitude(X, Y, semi_major_m, semi_minor_m, taper_p).astype(dtype)
    inv = 1.0 / max(z_m, 1e-9) - 1.0 / max(focus_range_m, 1e-9)
    if inv != 0.0:
        A *= np.exp(1j * np.pi / wavelength_m * inv * (X ** 2 + Y ** 2)).astype(dtype)

    field = sfft.fftshift(sfft.fft2(sfft.ifftshift(A), workers=_fft_workers(n_grid)))
    intensity = np.abs(field) ** 2

    dx = wavelength_m * z_m / (n_grid * d)  # Fraunhofer/Fresnel scaling of the FFT bins
    x = (np.arange(n_grid) - n_grid // 2) * dx
    total = intensity.sum() * dx * dx
    intensity = intensity / total if total > 0 else intensity
    return x, x.copy(), intensity


def plane_intensity_stack(z_m, focus_range_m, semi_major_m, semi_minor_m, wavelength_m,
                          taper_ps, n_grid=384):
    """`plane_intensity` batched over taper exponents (one shared aperture grid, one batched
    FFT, single precision). Used to scan taper cheaply; the winner is then re-evaluated."""
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
    """Intensity at arbitrary scattered points (qx, qy) on the plane at z, by direct DFT
    over the illuminated aperture samples only. Same normalisation as `plane_intensity`:
    `norm` is the analytic total power sum|A|^2 * (lambda z/d)^2, which is what Parseval
    gives for the FFT route.

    `n_ap` is a floor: `_aperture_grid` raises it if the aperture would not fit the grid."""
    d, n_ap = _aperture_grid(semi_major_m, semi_minor_m, wavelength_m, z_m, focus_range_m, n_ap)
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
    chunk = max(1, int(4.0e6 / max(av.size, 1)))  # cap the outer-product block at ~4M entries
    for s in range(0, qx.size, chunk):
        phase = -k * (np.outer(qx[s:s + chunk], ax) + np.outer(qy[s:s + chunk], ay))
        field = np.exp(1j * phase) @ av
        out[s:s + chunk] = field.real ** 2 + field.imag ** 2
    return out / norm if norm > 0 else out


def focal_intensity_grid(z_m, focus_range_m, semi_major_m, semi_minor_m, wavelength_m,
                         x_out, y_out, taper_p=0.0, n_ap=512):
    """As `focal_intensity_at` but on a caller-chosen separable output grid, so the DFT
    factorises into two matrix products. intensity[i, j] is at (x_out[i], y_out[j]);
    x_out corresponds to the semi_major aperture axis.

    `n_ap` is a floor: `_aperture_grid` raises it if the aperture would not fit the grid."""
    d, n_ap = _aperture_grid(semi_major_m, semi_minor_m, wavelength_m, z_m, focus_range_m, n_ap)
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


def axial_cuts_direct(z_m, focus_range_m, semi_major_m, semi_minor_m, wavelength_m, radial_m,
                      taper_p=0.0, n_ap_base=384, n_ap_max=3072, fresnel_floor_diam=2.5):
    """Both principal AXIAL cuts (x = aperture-major axis → NARROW spot; y = minor → WIDE spot) of the
    scalar-diffraction field, evaluated by a DIRECT zoom-DFT from the aperture — with NO ±λz/d
    single-FFT observation window. This resolves the NEAR field close to the aperture, where
    `axial_cut_xy`'s transform grid can no longer hold the aperture and it returns NaN.

    Trick: a cut ALONG a principal axis THROUGH the origin needs only a 1-D transform. With the
    orthogonal output coordinate pinned at 0 the transverse kernel is unity, so the on-axis field is
    the 1-D DFT of the aperture's MARGINAL projection onto that axis. The residual focusing phase
    exp(iπ(1/z−1/f)ρ²/λ) is separable (p(x)·p(y)), so each marginal is `p·(amp·p)` — O(n_ap²) to form
    the aperture and marginalize, O(n_radial·n_ap) for the transform, cheap even at the large `n_ap`
    the near field needs. Same Fresnel kernel and power normalisation (∫∫ I dA = 1) as
    `plane_intensity` / `focal_intensity_grid`, so the cuts splice seamlessly onto `axial_cut_xy`'s
    in the overlap region.

    Sampling: near the aperture the residual phase forces a fine spacing d ∝ z, and the aperture only
    fits the grid if n_ap ≳ 2.5·max_semi/d; n_ap is raised per plane to meet that, capped at
    `n_ap_max`. Planes closer than `fresnel_floor_diam` aperture DIAMETERS — where the paraxial
    Fresnel approximation itself fails and the true field is ≈ the aperture illumination — OR that
    would need n_ap > n_ap_max are returned NaN, for the caller to fill geometrically. (The rigorous
    field in that sub-Fresnel sliver, e.g. to the SIDE of the beam in the aperture plane for an
    exposure estimate, needs an angular-spectrum / Rayleigh–Sommerfeld pass, not this paraxial tool.)

    Aliasing: sampling the aperture at spacing d band-limits the reconstruction to the window
    |r| < λz/2d — evaluating beyond it WRAPS (aliases) the spot, NOT a bigger field of view. So the
    cut is truncated to zero outside that window (as the single-FFT path zero-fills outside its own
    grid). At the phase-resolving d this window is ≈ ±2·max_semi, which contains the aperture-sized
    near beam, so only far sidelobes are lost.

    Returns (ix, iy): (len(z), len(radial)) RAW areal intensity (fraction of radiated power / m²,
    comparable across planes/legs) of the narrow (x) and wide (y) cuts; 0 outside the alias-free
    window, NaN on planes below the Fresnel/`n_ap_max` floor."""
    z = np.atleast_1d(np.asarray(z_m, dtype=float))
    r = np.asarray(radial_m, dtype=float)
    max_semi = max(float(semi_major_m), float(semi_minor_m))
    z_floor = fresnel_floor_diam * 2.0 * max_semi
    ix = np.full((z.size, r.size), np.nan)
    iy = np.full((z.size, r.size), np.nan)
    for i, zi in enumerate(z):
        if not (zi > 0.0) or zi < z_floor:
            continue
        inv = 1.0 / zi - 1.0 / max(float(focus_range_m), 1e-9)
        # Close in, the focusing phase sets the pitch and the aperture only fits a much larger grid;
        # `_aperture_grid` sizes it. Past n_ap_max that is the practical near-aperture floor — skip the
        # plane (NaN) and let the caller fill it geometrically rather than pay an n^2 array for it.
        try:
            d, n_ap = _aperture_grid(semi_major_m, semi_minor_m, wavelength_m, zi, focus_range_m,
                                     n_ap_base, cap=n_ap_max)
        except ApertureTooFine:
            continue
        coord = (np.arange(n_ap) - n_ap // 2) * d
        X, Y = np.meshgrid(coord, coord, indexing="ij")
        amp = aperture_amplitude(X, Y, semi_major_m, semi_minor_m, taper_p)
        Ac = amp.astype(np.complex128)
        p = (np.exp(1j * np.pi / wavelength_m * inv * coord ** 2) if inv != 0.0
             else np.ones(n_ap, dtype=np.complex128))     # separable residual focusing phase p(x)·p(y)
        mx = p * (Ac @ p)                                   # marginal onto x → source for E(x, 0)
        my = p * (p @ Ac)                                   # marginal onto y → source for E(0, y)
        k = 2.0 * np.pi / (wavelength_m * zi)
        W = np.exp(-1j * k * np.outer(r, coord))            # shared kernel (symmetric grid)
        fx, fy = W @ mx, W @ my
        norm = float((amp ** 2).sum()) * (wavelength_m * zi / d) ** 2
        if norm > 0.0:
            outside = np.abs(r) > 0.95 * (wavelength_m * zi) / (2.0 * d)   # beyond the alias-free window
            gx = (fx.real ** 2 + fx.imag ** 2) / norm
            gy = (fy.real ** 2 + fy.imag ** 2) / norm
            gx[outside] = 0.0
            gy[outside] = 0.0
            ix[i], iy[i] = gx, gy
    return ix, iy


def _cut_enclosed(line, coord, dcoord, marginal, radial_m, r_query):
    """One plane's contribution to an axial cut: the intensity along a centre line, and the
    enclosed power. `enclosed` integrates the marginal, so it is the power in the strip
    |coord| <= r -- not inside a circle of radius r."""
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
    """Peak-normalise if asked, and round for compact serialization into the animation's HTML.

    The un-normalised branch is left full precision on purpose: the raw field is ~1e-4 m^-2 and
    would round to zero at 3 decimals, and it is the branch that stays comparable across legs.
    """
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
    """x and y cuts plus enclosed power for a whole sweep of ranges z_m, shape (n_z, n_r).

    Ranges where the focusing phase is mild enough that d_spot is the binding pitch share
    one aperture grid and go through a single batched FFT; the rest fall back to one
    `plane_intensity` each. Rows the grid cannot span the aperture for are left NaN.
    """
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
    is_spot = d_phase >= d_spot

    # Both branches below check that the grid still contains the aperture. On the d_spot branch
    # the 2.5*max_semi/n_grid floor in _aperture_spacing guarantees it, so that test never
    # fires; on the d_phase branch it does, and those near-aperture planes are left NaN rather
    # than fabricated. `focal_intensity_grid`/`focal_intensity_at` share the d_phase exposure
    # and make no such test -- they are safe today only because deep defocus needs a short
    # range while a high aperture aspect only arises at low elevation, i.e. long range, so the
    # two do not coincide at the geometries this model runs. Nothing enforces that.
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
    """Bilinear sample of every plane in a stack at the same query points; zero outside."""
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
    """Closed-form 1 - J0(v)^2 - J1(v)^2 for a uniform circular aperture in focus.
    Reference for the numerical routines above."""
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
