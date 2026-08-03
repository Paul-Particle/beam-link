"""Scalar diffraction from a tapered, elliptical, focused aperture — the kernel of the one solve.

There is exactly ONE way the engine computes a field (I1), and this is it: a direct Fresnel
transform from the sampled aperture to wherever the caller asks. `solver.py` turns a sampling
request into query points; this evaluates them. Nothing above this file re-implements propagation,
and there is no second grid, no second normalisation and no second window convention to reconcile.

Units: intensity comes out normalised to unit radiated power, so ∫∫ I dA = 1 and the values are
m⁻². Absolute W/m² is one multiplication by the leg's radiated power, applied once, in the engine
(P5/P8). `focus_range_m` is the range the aperture phase is curved for; equal to z means in focus.

**The sampling is derived from the query, per axis.** Three physical constraints bound the
aperture-plane pitch, and every one of them is an upper bound, so taking the smallest is safe:

    edge      dx ≤ semi_x / N                      — N samples across that semi-axis, so the
              illumination taper and the rim are resolved (N is a fidelity parameter)
    phase     dx ≤ λ / (4·semi_x·|1/z − 1/f|)      — the residual focusing phase must advance less
              than π/2 per sample at the aperture edge, or the aperture is aliased
    window    dx ≤ λ·z / (2·r_query_x)             — sampling at dx band-limits the reconstruction
              to |x| < λz/2dx; evaluating beyond that WRAPS the spot rather than showing a wider
              field of view

and the grid then only has to CONTAIN the aperture: n_x·dx ≥ 2.2·semi_x. Two consequences are what
make this fast enough to be the only route.

**Anisotropic sampling makes the cost independent of aperture aspect.** With one common pitch, an
8:1 aperture needs the minor axis's fine pitch across the major axis's full length — n = 576 where a
round aperture of the same area needs 71. Per-axis pitch gives n_x = n_y ≈ 2.2·N for ANY aspect,
because each axis is sampled at its own scale: 20× fewer aperture samples on the elongated
geometries, which are exactly the low-elevation samples a pass spends most of its time at.

**The focusing phase is separable, so it never touches the aperture array.** exp(iπ/λ·inv·(x²+y²))
factorises as p(x)·p(y), and (amp · p⊗p) sandwiched between two transform kernels equals
(kernel_x·p) · amp · (p·kernel_y) — so the phase is folded into the two O(n·n_out) kernels and the
aperture amplitude stays REAL and constant across a whole defocus search. The pre-SPEC near-field
cut used this factorisation as a special case for on-axis cuts; here it is how every evaluation
works.

Two regions are reported UNRESOLVED rather than filled (§5 P5, §8 item 14):

    sub_fresnel            closer to the aperture than `fresnel_floor_diameters` aperture
                           diameters, where the paraxial approximation itself fails and the true
                           field is ≈ the aperture illumination. The rigorous field there needs an
                           angular-spectrum / Rayleigh-Sommerfeld pass, not this paraxial tool.
    aperture_unsamplable   the focusing phase demands a finer pitch than `aperture_n_max` allows.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from beamlink.physics.fidelity import FidelityConfig

UNRESOLVED_REASONS = {
    0: "resolved",
    1: "sub_fresnel: closer to the aperture than the paraxial Fresnel approximation holds",
    2: "aperture_unsamplable: the focusing phase needs a finer pitch than the cap allows",
    3: "no_aperture: the aperture has zero area at this sample",
}

# Pitch is quantised down onto a geometric ladder so that many batch members share one aperture
# grid and therefore one batched transform. Rounding DOWN is always safe (every constraint is an
# upper bound); the ratio bounds the overshoot in n, and hence the wasted work, at 19 %.
_PITCH_LADDER_RATIO = 2.0 ** 0.25
_GRID_QUANTUM = 16          # n is rounded up to a multiple of this, to limit the number of groups
_APERTURE_MARGIN = 2.2      # n·d ≥ this × the semi-axis, so the grid contains the aperture


@dataclass(frozen=True)
class ApertureState:
    """A batch of aperture states, flat over one batch axis of length m.

    A batched call is the normal call and a single aperture is a batch of one (I9): the members are
    (candidate × sample) pairs, which is the axis the expensive internals should see. The pre-SPEC
    kernels were already batch-shaped — over propagation planes, and over taper exponents — but were
    re-entered once per sample, so the batching was spent on the wrong axis (§6a).
    """

    semi_major_m: np.ndarray
    semi_minor_m: np.ndarray
    taper_exponent: np.ndarray
    focus_range_m: np.ndarray
    wavelength_m: np.ndarray

    def __post_init__(self):
        names = ("semi_major_m", "semi_minor_m", "taper_exponent", "focus_range_m", "wavelength_m")
        arrays = [np.atleast_1d(np.asarray(getattr(self, name), dtype=float)).ravel()
                  for name in names]
        size = max(array.size for array in arrays)
        for name, array in zip(names, arrays):
            object.__setattr__(self, name,
                               np.broadcast_to(array, (size,)).copy() if array.size == 1 else array)
        if len({getattr(self, name).size for name in names}) != 1:
            raise ValueError("ApertureState fields must broadcast to one common length")

    @property
    def size(self) -> int:
        return int(self.semi_major_m.size)

    @property
    def semi_axes_m(self) -> np.ndarray:
        """(m, 2) — the two semi-axes, in the aperture frame's (x, y) order."""
        return np.stack([self.semi_major_m, self.semi_minor_m], axis=1)

    @property
    def max_semi_m(self) -> np.ndarray:
        return np.maximum(self.semi_major_m, self.semi_minor_m)

    def take(self, index) -> "ApertureState":
        return ApertureState(self.semi_major_m[index], self.semi_minor_m[index],
                             self.taper_exponent[index], self.focus_range_m[index],
                             self.wavelength_m[index])

    def with_focus(self, focus_range_m) -> "ApertureState":
        return replace(self, focus_range_m=np.broadcast_to(
            np.asarray(focus_range_m, dtype=float), (self.size,)).astype(float))

    def with_taper(self, taper_exponent) -> "ApertureState":
        return replace(self, taper_exponent=np.broadcast_to(
            np.asarray(taper_exponent, dtype=float), (self.size,)).astype(float))


def required_pitch(state: ApertureState, z_m, query_half_extent_m, config: FidelityConfig):
    """Per-axis aperture pitch and grid size each batch member needs at range z.

    `query_half_extent_m` is (m,) for a common extent or (m, 2) per axis. Returns
    (pitch (m, 2), n (m, 2) int, resolvable (m,) bool, reason (m,) int) — see UNRESOLVED_REASONS.
    """
    z = np.maximum(np.asarray(z_m, dtype=float), 1e-9)
    extent = np.asarray(query_half_extent_m, dtype=float)
    if extent.ndim == 1:
        extent = np.repeat(extent[:, None], 2, axis=1)
    extent = np.maximum(extent, 1e-9)
    semi = np.maximum(state.semi_axes_m, 1e-9)

    inverse_curvature = np.abs(1.0 / z - 1.0 / np.maximum(state.focus_range_m, 1e-9))[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        pitch_phase = np.where(inverse_curvature > 0.0,
                               state.wavelength_m[:, None] / (4.0 * semi * inverse_curvature),
                               np.inf)
    pitch_edge = semi / float(config.aperture_min_samples_across_minor)
    pitch_window = state.wavelength_m[:, None] * z[:, None] / (2.0 * extent)
    pitch = np.minimum(np.minimum(pitch_phase, pitch_edge), pitch_window)

    exponent = np.floor(np.log(pitch) / np.log(_PITCH_LADDER_RATIO))
    pitch = _PITCH_LADDER_RATIO ** exponent
    needed = np.ceil(_APERTURE_MARGIN * semi / pitch)
    n = (np.ceil(needed / _GRID_QUANTUM) * _GRID_QUANTUM).astype(int)

    fresnel_floor = config.fresnel_floor_diameters * 2.0 * state.max_semi_m
    sub_fresnel = z < fresnel_floor
    unsamplable = n.max(axis=1) > config.aperture_n_max
    degenerate = np.min(state.semi_axes_m, axis=1) <= 0.0
    reason = np.where(degenerate, 3, np.where(sub_fresnel, 1, np.where(unsamplable, 2, 0)))
    return pitch, n, reason == 0, reason


def defocus_floor(state: ApertureState, z_m, query_half_extent_m, config: FidelityConfig):
    """The SHORTEST focus range whose aperture the sampling can still represent, in closed form.

    The density-cap defocus search needs a compliant lower bracket, and the pre-SPEC engine found it
    by trying `slant/8` and multiplying by 1.5 until the field model stopped refusing — a
    data-dependent loop with an exception as its control flow, which cannot be vectorized. It does
    not have to be a search: the binding constraint is the focusing phase on the LONGER semi-axis, so

        n = 2.2·semi/d  with  d = λ/(4·semi·|1/z − 1/f|)  ⇒  n = 8.8·semi²·|1/z − 1/f| / λ

    is monotone in the defocus, and setting n = n_max inverts directly for f (D9). The other two
    pitch constraints do not depend on the focus at all, so they cannot move this bound; the result
    is clipped so it never exceeds the in-focus case and never asks for more defocus than z/8, which
    is the operating limit the cap search was always given.

    The budget is `n_max` discounted by the pitch ladder's own rounding and by the grid quantum,
    because both round in the direction that COSTS samples: the ladder takes a pitch down by up to
    19 % and the quantum takes n up to the next multiple of 16. Without that discount the closed form
    returns a focus that `required_pitch` then declines to sample — which is the same class of bug as
    the try/except loop it replaces, only quieter.
    """
    z = np.maximum(np.asarray(z_m, dtype=float), 1e-9)
    max_semi = np.maximum(state.max_semi_m, 1e-9)
    budget = max(config.aperture_n_max / _PITCH_LADDER_RATIO - _GRID_QUANTUM, 1.0)
    max_curvature = budget * state.wavelength_m / (_APERTURE_MARGIN * 4.0 * max_semi ** 2)
    representable = 1.0 / (1.0 / z + max_curvature)
    floor = np.maximum(representable, z / 8.0)
    # A grid that cannot even hold the in-focus aperture leaves no defocus available at all.
    _, _, resolvable, _ = required_pitch(state.with_focus(z), z, query_half_extent_m, config)
    return np.where(resolvable, np.minimum(floor, z), z)


def _complex_dtype(config: FidelityConfig):
    return np.complex64 if config.field_dtype == "complex64" else np.complex128


def _groups(pitch, n):
    """Members that share an aperture grid, as ((pitch_x, pitch_y, n_x, n_y), member indices)."""
    keyed: dict[tuple, list[int]] = {}
    for index in range(pitch.shape[0]):
        key = (float(pitch[index, 0]), float(pitch[index, 1]),
               int(n[index, 0]), int(n[index, 1]))
        keyed.setdefault(key, []).append(index)
    return [(key, np.asarray(members)) for key, members in keyed.items()]


def _amplitude(state: ApertureState, coordinate_x, coordinate_y, cache):
    """The REAL aperture illumination (1 − ρ²)^p, (m, n_x, n_y), plus Σ|A|².

    Real and focus-independent because the focusing phase is folded into the transform kernels
    instead — which is what lets a whole defocus search reuse one amplitude array.
    """
    key = None
    if cache is not None:
        key = (coordinate_x.shape[0], coordinate_y.shape[0], float(coordinate_x[1] - coordinate_x[0])
               if coordinate_x.size > 1 else 0.0,
               float(coordinate_y[1] - coordinate_y[0]) if coordinate_y.size > 1 else 0.0,
               state.semi_major_m.tobytes(), state.semi_minor_m.tobytes(),
               state.taper_exponent.tobytes())
        hit = cache.get(key)
        if hit is not None:
            return hit
    normalised = ((coordinate_x[None, :, None] / state.semi_major_m[:, None, None]) ** 2
                  + (coordinate_y[None, None, :] / state.semi_minor_m[:, None, None]) ** 2)
    amplitude = np.where(normalised <= 1.0, np.clip(1.0 - normalised, 0.0, None)
                         ** state.taper_exponent[:, None, None], 0.0)
    power_sum = np.sum(amplitude.reshape(state.size, -1) ** 2, axis=1)
    result = (amplitude, power_sum)
    if cache is not None:
        cache[key] = result
    return result


def plane_intensity(state: ApertureState, z_m, x_out, y_out, config: FidelityConfig,
                    amplitude_cache=None):
    """Intensity on a separable transverse grid at range z, per batch member.

    `z_m` is (m,), `x_out` is (m, n_x_out) and `y_out` is (m, n_y_out), all in the aperture frame.
    Returns (intensity (m, n_x_out, n_y_out) in m⁻², valid, reason (m,), pitch (m, 2), n (m, 2)).

    A cut ALONG a principal axis through the origin is the same call with a singleton orthogonal
    axis, and it costs O(n_x·n_y) rather than O(n_x·n_y·n_out) because `amp @ kernel_y` collapses to
    the aperture's marginal projection first — the trick the pre-SPEC near-field cut needed a
    separate code path for.

    `amplitude_cache` is an optional dict the caller keeps across a search: the amplitude depends on
    the semi-axes and the taper only, so a defocus bisection recomputes nothing.
    """
    z = np.atleast_1d(np.asarray(z_m, dtype=float))
    x_out = np.atleast_2d(np.asarray(x_out, dtype=float))
    y_out = np.atleast_2d(np.asarray(y_out, dtype=float))
    members, n_x_out, n_y_out = state.size, x_out.shape[1], y_out.shape[1]
    extent = np.stack([np.abs(x_out).max(axis=1), np.abs(y_out).max(axis=1)], axis=1)
    pitch, grid_n, resolvable, reason = required_pitch(state, z, extent, config)

    intensity = np.zeros((members, n_x_out, n_y_out))
    valid = np.zeros((members, n_x_out, n_y_out), dtype=bool)
    dtype = _complex_dtype(config)

    for (pitch_x, pitch_y, n_x, n_y), members_in_group in _groups(pitch, grid_n):
        indices = members_in_group[resolvable[members_in_group]]
        if indices.size == 0:
            continue
        coordinate_x = (np.arange(n_x) - n_x // 2) * pitch_x
        coordinate_y = (np.arange(n_y) - n_y // 2) * pitch_y
        # Working memory is the binding resource, not output size (§6a): the amplitude plus the two
        # kernels and the transform temporaries are what a chunk has to fit.
        per_member = (n_x * n_y + n_x * n_x_out + n_y * n_y_out) * np.dtype(dtype).itemsize * 4
        chunk = max(1, int(config.chunk_bytes // max(per_member, 1)))
        for start in range(0, indices.size, chunk):
            block = indices[start:start + chunk]
            sub = state.take(block)
            amplitude, power_sum = _amplitude(sub, coordinate_x, coordinate_y, amplitude_cache)
            inverse_curvature = (1.0 / np.maximum(z[block], 1e-9)
                                 - 1.0 / np.maximum(sub.focus_range_m, 1e-9))
            chirp = (np.pi / sub.wavelength_m * inverse_curvature)[:, None]
            phase_x = np.exp(1j * chirp * coordinate_x[None, :] ** 2)
            phase_y = np.exp(1j * chirp * coordinate_y[None, :] ** 2)
            wavenumber = (2.0 * np.pi / (sub.wavelength_m * z[block]))[:, None, None]
            kernel_x = (np.exp(-1j * wavenumber * x_out[block][:, :, None]
                               * coordinate_x[None, None, :])
                        * phase_x[:, None, :]).astype(dtype)
            kernel_y = (np.exp(-1j * wavenumber * coordinate_y[None, :, None]
                               * y_out[block][:, None, :])
                        * phase_y[:, :, None]).astype(dtype)
            transformed = kernel_x @ amplitude.astype(dtype) @ kernel_y
            block_intensity = (transformed.real.astype(np.float64) ** 2
                               + transformed.imag.astype(np.float64) ** 2)
            normalisation = (power_sum * (sub.wavelength_m * z[block]) ** 2
                             / (pitch_x * pitch_y))
            intensity[block] = block_intensity / np.where(normalisation > 0, normalisation,
                                                          1.0)[:, None, None]
            # Truncate outside the alias-free window: the pitch was chosen to contain the query
            # extent, but the ladder quantisation and a caller-supplied grid can still reach past it.
            window_x = 0.95 * sub.wavelength_m * z[block] / (2.0 * pitch_x)
            window_y = 0.95 * sub.wavelength_m * z[block] / (2.0 * pitch_y)
            inside = ((np.abs(x_out[block]) <= window_x[:, None])[:, :, None]
                      & (np.abs(y_out[block]) <= window_y[:, None])[:, None, :])
            intensity[block] = np.where(inside, intensity[block], 0.0)
            valid[block] = inside
    return intensity, valid, reason, pitch, grid_n


def scattered_intensity(state: ApertureState, z_m, x_query, y_query, config: FidelityConfig):
    """Intensity at arbitrary scattered points, by direct transform over the illuminated samples.

    The exact route: no output grid and therefore no resampling error. It costs
    n_query × n_illuminated, which is affordable for a modest query cloud and prohibitive for a
    large one, so it is the REFERENCE the recompute checks bound the grid-and-resample route against,
    and the sampler for a caller-supplied point cloud, rather than a fidelity tier of its own (D10).

    `x_query`/`y_query` are (m, n_query).
    """
    z = np.atleast_1d(np.asarray(z_m, dtype=float))
    x_query = np.atleast_2d(np.asarray(x_query, dtype=float))
    y_query = np.atleast_2d(np.asarray(y_query, dtype=float))
    extent = np.stack([np.abs(x_query).max(axis=1), np.abs(y_query).max(axis=1)], axis=1)
    pitch, grid_n, resolvable, reason = required_pitch(state, z, extent, config)
    intensity = np.zeros(x_query.shape)
    valid = np.zeros(x_query.shape, dtype=bool)

    for member in range(state.size):
        if not resolvable[member]:
            continue
        one = state.take(np.array([member]))
        n_x, n_y = int(grid_n[member, 0]), int(grid_n[member, 1])
        coordinate_x = (np.arange(n_x) - n_x // 2) * pitch[member, 0]
        coordinate_y = (np.arange(n_y) - n_y // 2) * pitch[member, 1]
        amplitude, power_sum = _amplitude(one, coordinate_x, coordinate_y, None)
        grid_x, grid_y = np.meshgrid(coordinate_x, coordinate_y, indexing="ij")
        illuminated = amplitude[0] > 0.0
        aperture_x, aperture_y = grid_x[illuminated], grid_y[illuminated]
        values = amplitude[0][illuminated].astype(np.complex128)
        inverse_curvature = 1.0 / z[member] - 1.0 / max(one.focus_range_m[0], 1e-9)
        if inverse_curvature != 0.0:
            values = values * np.exp(1j * np.pi / one.wavelength_m[0] * inverse_curvature
                                     * (aperture_x ** 2 + aperture_y ** 2))
        wavenumber = 2.0 * np.pi / (one.wavelength_m[0] * z[member])
        normalisation = float(power_sum[0]) * (one.wavelength_m[0] * z[member]) ** 2 / (
            pitch[member, 0] * pitch[member, 1])
        chunk = max(1, int(4.0e6 / max(values.size, 1)))
        for start in range(0, x_query.shape[1], chunk):
            slab = slice(start, start + chunk)
            phase = -wavenumber * (np.outer(x_query[member, slab], aperture_x)
                                   + np.outer(y_query[member, slab], aperture_y))
            transformed = np.exp(1j * phase) @ values
            intensity[member, slab] = transformed.real ** 2 + transformed.imag ** 2
        if normalisation > 0:
            intensity[member] /= normalisation
        window_x = 0.95 * one.wavelength_m[0] * z[member] / (2.0 * pitch[member, 0])
        window_y = 0.95 * one.wavelength_m[0] * z[member] / (2.0 * pitch[member, 1])
        inside = (np.abs(x_query[member]) <= window_x) & (np.abs(y_query[member]) <= window_y)
        intensity[member] = np.where(inside, intensity[member], 0.0)
        valid[member] = inside
    return intensity, valid, reason


def aperture_phase(state: ApertureState, z_m, grid_n: int = 128):
    """The aperture phase as a field over the aperture (P4, on request).

    Emitted instead of the zone-plate draw-parameters — a distance, a radius, a wavelength and a
    which-way-up flag — that the pre-SPEC frame payload carried for a figure (§8 item 9).
    """
    z = np.atleast_1d(np.asarray(z_m, dtype=float))
    half = state.max_semi_m
    axis = np.linspace(-1.0, 1.0, grid_n)[None, :] * half[:, None]
    x_grid = axis[:, :, None]
    y_grid = axis[:, None, :]
    inverse_curvature = 1.0 / np.maximum(z, 1e-9) - 1.0 / np.maximum(state.focus_range_m, 1e-9)
    phase = (np.pi / state.wavelength_m[:, None, None] * inverse_curvature[:, None, None]
             * (x_grid ** 2 + y_grid ** 2))
    inside = (((x_grid / state.semi_major_m[:, None, None]) ** 2
               + (y_grid / state.semi_minor_m[:, None, None]) ** 2) <= 1.0)
    return np.where(inside, np.angle(np.exp(1j * phase)), np.nan), axis


# --- analytic references (used by the tests, not by the chain) -----------------------------


def airy_encircled_energy(radius_m, aperture_radius_m, wavelength_m, focus_range_m):
    """Closed-form 1 − J0(v)² − J1(v)² for a uniform circular aperture in focus."""
    from scipy.special import j0, j1
    v = (2.0 * np.pi * aperture_radius_m * np.asarray(radius_m, dtype=float)
         / (wavelength_m * focus_range_m))
    v = np.where(v == 0.0, 1e-12, v)
    return 1.0 - j0(v) ** 2 - j1(v) ** 2
