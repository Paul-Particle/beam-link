"""The one solve, evaluated wherever the spec asks: sampling requests → query points → intensity.

This is the collapse of the two pre-SPEC field paths into one (§10 item 4). `capture.overlap_capture`
solved the focal plane on a 2-D grid; `flyby_frames.channel_field` solved the whole path with 1-D
axial cuts, and the aircraft plane lies INSIDE the path — two independent solves of the same field,
agreeing on peak intensity to 15 %, held together only by a tolerance chosen when the comparison was
first written (§1). There is one solver now, and the silhouette overlap and the side view are two
samplings of it, which is not two solves (I1).

It is also new capability rather than a move: the pre-SPEC engine had no concept of a field away
from the aircraft, so points and volumes did not exist. All the samplers ship together so the
capability is honest, and a routine run simply does not request the expensive ones (D2).

Everything is batched over a flat member axis of (candidate × sample) pairs (I9). Intensity comes
back normalised to unit radiated power; the engine multiplies by P8's radiated power once.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field

import numpy as np

from beamlink.artifacts.spec import FieldSampling
from beamlink.physics.beam_link import aperture as aperture_module
from beamlink.physics.beam_link import occlusion
from beamlink.physics.fidelity import FidelityConfig

# How wide to fit an unstated extent. The transverse extent of a cut has to come from somewhere;
# these are the two physical scales available, and the larger wins, so a fitted window contains the
# beam near the aperture (where it is aperture-wide) and at focus (where it is diffraction-wide).
_RECEIVER_EXTENT_SLACK = 1.15      # around the silhouette, matching the capture grid's own slack
_CUT_EXTENT_SLACK = 3.0            # around the receiver, for a side view that shows the skirt


@dataclass
class SolveContext:
    """Everything the samplers need that is not in the sampling request itself.

    All arrays have a leading member axis of length m = candidate × sample.
    """

    state: aperture_module.ApertureState
    receiver_range_m: np.ndarray
    """(m,) — the range of the plane the aircraft is in, i.e. the last leg's slant range."""

    element_offsets_m: np.ndarray | None = None
    """(m, n_element, 2) — silhouette element coordinates in the APERTURE transverse frame, already
    the same frame the aperture's own axes define (see `airframe.silhouette`)."""

    element_valid: np.ndarray | None = None
    receiver_half_extent_m: np.ndarray | None = None
    """(m, 2) — the silhouette's principal half-extents, used to fit an unstated extent and as the
    occluder's size."""

    pointing_offset_m: np.ndarray | None = None
    """(m,) — the beam's offset from the silhouette centroid, applied along the aperture's MAJOR
    axis, which is the silhouette's narrow direction and so the worst case for a given magnitude."""

    captured_geometric: np.ndarray | None = None
    """(m,) — P6's geometric capture, needed only by an occluded sample set."""

    amplitude_cache: dict = dataclass_field(default_factory=dict)
    """Kept across a search. The aperture amplitude depends on the semi-axes and the taper only —
    the focus enters through the transform kernels — so a whole defocus bisection reuses one
    amplitude array per (grid, taper). Correctness-neutral and keyed on its full input, per D4."""

    extras: dict = dataclass_field(default_factory=dict)

    @property
    def size(self) -> int:
        return self.state.size


@dataclass
class SolvedSampleSet:
    """One sampling, solved. Intensity is per unit radiated power at this stage."""

    sampling: FieldSampling
    points_m: np.ndarray
    intensity_per_m2: np.ndarray
    valid: np.ndarray
    grid_shape: tuple[int, ...]
    grid_axes: dict
    occluder_transmission: np.ndarray | None = None
    unresolved_reason: np.ndarray | None = None
    enclosed_fraction: np.ndarray | None = None
    aperture_pitch_m: np.ndarray | None = None
    aperture_samples: np.ndarray | None = None
    """The pitch and grid size the sampling actually needed, per member. Recorded per SAMPLE SET
    rather than per product because the pitch is derived from the query extent (see
    `aperture.required_pitch`): a one-point on-axis query and a 200 m-wide side view of the same
    solve legitimately need different aperture grids, and I5 wants the resolution each one ran at."""


def solve(sampling: FieldSampling, context: SolveContext, config: FidelityConfig
          ) -> SolvedSampleSet:
    """Evaluate the one solve at the points `sampling` asks for."""
    builder = {
        "silhouette": _solve_silhouette,
        "transverse_plane": _solve_transverse_plane,
        "axial_plane": _solve_axial_plane,
        "points": _solve_points,
        "volume": _solve_volume,
    }[sampling.kind]
    solved = builder(sampling, context, config)
    if sampling.occluders == "receiver":
        solved = _apply_occluder(solved, context)
    return solved


def method_name(config: FidelityConfig) -> str:
    """One name for the route, recorded per product (I5). See D10 for why there is only one."""
    return f"fresnel_dft_separable[{config.field_dtype}]"


# --- the samplers -------------------------------------------------------------------------


def _fitted_transverse_extent(context: SolveContext, slack: float) -> np.ndarray:
    """(m, 2) half-extents to use when the request states none.

    The two physical scales: the aperture's own semi-axes, which bound the beam near the aperture,
    and the receiver's principal half-extents, which is the region a capture figure is about.
    """
    receiver = context.receiver_half_extent_m
    if receiver is None:
        receiver = np.zeros((context.size, 2))
    aperture_semi = np.stack([context.state.semi_major_m, context.state.semi_minor_m], axis=1)
    return np.maximum(receiver * slack, aperture_semi)


def _requested_extent(sampling: FieldSampling, context: SolveContext, slack: float) -> np.ndarray:
    stated = np.asarray(sampling.extent_m, dtype=float)
    fitted = _fitted_transverse_extent(context, slack)
    return np.where(stated[None, :] > 0.0, np.broadcast_to(stated, fitted.shape), fitted)


def _centred_axis(n: int, half_extent: np.ndarray) -> np.ndarray:
    """(m, n) transverse coordinates spanning ±half_extent, centred.

    A one-sample axis is the CENTRE, not the low edge: an n=1 transverse grid is how the on-axis
    peak is asked for, and `linspace` would put it at −half_extent.
    """
    if n == 1:
        return np.zeros((half_extent.size, 1))
    return half_extent[:, None] * np.linspace(-1.0, 1.0, n)[None, :]


def _solve_silhouette(sampling, context: SolveContext, config) -> SolvedSampleSet:
    """The receiver's own visible surface elements, at the receiver range.

    Evaluated on a separable grid fitted to the element cloud and bilinearly resampled onto the
    elements. That resampling is a property of the SAMPLING, not a second solve, and the recompute
    check bounds it against `aperture.scattered_intensity`, which is exact at the element positions
    (D10).
    """
    if context.element_offsets_m is None:
        raise ValueError("silhouette sampling needs a receiver: the run's scenario has none")
    members = context.size
    offsets = context.element_offsets_m.copy()
    valid_elements = (context.element_valid if context.element_valid is not None
                      else np.ones(offsets.shape[:2], dtype=bool))
    if context.pointing_offset_m is not None:
        offsets[..., 0] = offsets[..., 0] - context.pointing_offset_m[:, None]

    # Grid spanning the cloud with the same 15 % slack the pre-SPEC capture grid used, and floors
    # that only stop it degenerating when the target subtends almost nothing. Each axis's floor
    # takes 5 % of the CONJUGATE aperture semi-axis, which carries the same aspect scaling as the
    # spot on that axis.
    masked = np.where(valid_elements[..., None], np.abs(offsets), 0.0)
    reach_x = np.maximum(masked[..., 0].max(axis=1), context.state.semi_minor_m * 0.05)
    reach_y = np.maximum(masked[..., 1].max(axis=1), context.state.semi_major_m * 0.05)
    half_x = np.maximum(reach_x, 1.0) * _RECEIVER_EXTENT_SLACK
    half_y = np.maximum(reach_y, 1.0) * _RECEIVER_EXTENT_SLACK
    unit_axis = np.linspace(-1.0, 1.0, config.capture_grid_n)
    x_axis = half_x[:, None] * unit_axis[None, :]
    y_axis = half_y[:, None] * unit_axis[None, :]

    grid_intensity, grid_valid, reason, pitch, grid_n = aperture_module.plane_intensity(
        context.state, context.receiver_range_m, x_axis, y_axis, config,
        amplitude_cache=context.amplitude_cache)
    intensity = _bilinear_sample(x_axis, y_axis, grid_intensity, offsets[..., 0], offsets[..., 1])
    sampled_valid = _bilinear_sample(x_axis, y_axis, grid_valid.astype(float),
                                     offsets[..., 0], offsets[..., 1]) > 0.999
    resolved = (reason == 0)[:, None]
    valid = valid_elements & sampled_valid & resolved
    intensity = np.where(valid, intensity, 0.0)
    points = np.concatenate(
        [offsets, np.broadcast_to(context.receiver_range_m[:, None, None], offsets.shape[:2] + (1,))],
        axis=2)
    return SolvedSampleSet(
        sampling=sampling, points_m=points, intensity_per_m2=intensity, valid=valid,
        grid_shape=(), grid_axes={"grid_half_extent_x_m": half_x, "grid_half_extent_y_m": half_y},
        unresolved_reason=reason, aperture_pitch_m=pitch, aperture_samples=grid_n)


def _solve_transverse_plane(sampling, context: SolveContext, config) -> SolvedSampleSet:
    """An x-y grid at each requested fraction of the receiver range."""
    half = _requested_extent(sampling, context, _RECEIVER_EXTENT_SLACK)
    n_x, n_y = sampling.n_transverse
    x_axis = _centred_axis(n_x, half[:, 0])
    y_axis = _centred_axis(n_y, half[:, 1])
    fractions = np.asarray(sampling.z_fractions, dtype=float)
    return _stack_over_planes(sampling, context, config, fractions, x_axis, y_axis,
                              grid_shape=(fractions.size, n_x, n_y))


def _solve_axial_plane(sampling, context: SolveContext, config) -> SolvedSampleSet:
    """A side view: one of the aperture's own two principal planes.

    The default cuts (D2). `axis="major"` puts the radial coordinate on the aperture's MAJOR axis,
    which by the Fourier inverse is the NARROW spot; `axis="minor"` gives the wide one. Both are
    physically defined by the aperture, which is what the pre-SPEC F/F₂ pair was not.
    """
    half = _requested_extent(sampling, context, _CUT_EXTENT_SLACK)
    on_major = sampling.axis == "major"
    radial_axis = _centred_axis(sampling.n_radial, half[:, 0] if on_major else half[:, 1])
    singleton = np.zeros((context.size, 1))
    low, high = min(sampling.z_fractions), max(sampling.z_fractions)
    fractions = np.linspace(low, high, sampling.n_axial)
    x_axis, y_axis = (radial_axis, singleton) if on_major else (singleton, radial_axis)
    solved = _stack_over_planes(sampling, context, config, fractions, x_axis, y_axis,
                                grid_shape=(sampling.n_axial, sampling.n_radial))
    solved.grid_axes["radial_m"] = radial_axis
    solved.grid_axes["radial_on"] = np.array([0 if on_major else 1])
    return solved


def _solve_volume(sampling, context: SolveContext, config) -> SolvedSampleSet:
    """A 3-D box grid. Built in v1 rather than deferred, so the capability is honest (D2)."""
    half = _requested_extent(sampling, context, _CUT_EXTENT_SLACK)
    n_x, n_y = sampling.n_transverse
    x_axis = _centred_axis(n_x, half[:, 0])
    y_axis = _centred_axis(n_y, half[:, 1])
    low, high = min(sampling.z_fractions or (0.0, 1.0)), max(sampling.z_fractions or (0.0, 1.0))
    fractions = np.linspace(low, high, sampling.n_axial)
    return _stack_over_planes(sampling, context, config, fractions, x_axis, y_axis,
                              grid_shape=(sampling.n_axial, n_x, n_y))


def _solve_points(sampling, context: SolveContext, config) -> SolvedSampleSet:
    """Arbitrary points in the aperture frame, by the exact scattered transform.

    A caller-supplied cloud is not separable, so this is the one sampler that uses
    `aperture.scattered_intensity` — exact at the query points, at a cost of n_query ×
    n_illuminated. Points sharing a range are one call, because the aperture grid depends on the
    range and nothing else.

    The z coordinates come from the spec and are therefore the same for every member, which is what
    lets the grouping be over columns rather than over members.
    """
    points = np.asarray(sampling.points_m, dtype=float)
    if points.ndim == 2:
        points = np.broadcast_to(points, (context.size,) + points.shape).copy()
    intensity = np.zeros(points.shape[:2])
    valid = np.zeros(points.shape[:2], dtype=bool)
    reason = np.zeros(context.size, dtype=int)
    ranges = np.round(points[0, :, 2], 6)
    for plane_range in np.unique(ranges):
        columns = np.nonzero(ranges == plane_range)[0]
        plane_z = np.full(context.size, float(plane_range))
        plane_intensity, plane_valid, plane_reason = aperture_module.scattered_intensity(
            context.state, plane_z, points[:, columns, 0], points[:, columns, 1], config)
        intensity[:, columns] = plane_intensity
        valid[:, columns] = plane_valid
        reason = np.where((reason == 0) & (plane_reason != 0), plane_reason, reason)
        pitch, grid_n, _, _ = aperture_module.required_pitch(
            context.state, plane_z, np.abs(points[:, columns, :2]).max(axis=(1, 2)), config)
    unresolved = reason != 0
    valid[unresolved] = False
    intensity = np.where(valid, intensity, 0.0)
    return SolvedSampleSet(sampling=sampling, points_m=points, intensity_per_m2=intensity,
                           valid=valid, grid_shape=(), grid_axes={}, unresolved_reason=reason,
                           aperture_pitch_m=pitch, aperture_samples=grid_n)


def _stack_over_planes(sampling, context: SolveContext, config, fractions, x_axis, y_axis,
                       grid_shape) -> SolvedSampleSet:
    """Evaluate a shared transverse grid at each of `fractions` × the receiver range.

    Every plane is its own batched call because the aperture pitch depends on the range, which is
    exactly the axis `aperture.required_pitch` groups over — so a side view of 160 planes is 160
    batched transforms over all members, not 160 × m separate ones.
    """
    members = context.size
    n_x, n_y = x_axis.shape[1], y_axis.shape[1]
    n_point = int(fractions.size * n_x * n_y)
    intensity = np.zeros((members, n_point))
    valid = np.zeros((members, n_point), dtype=bool)
    points = np.zeros((members, n_point, 3))
    reason = np.zeros(members, dtype=int)
    ranges = np.zeros((members, fractions.size))

    grid_x = np.broadcast_to(x_axis[:, :, None], (members, n_x, n_y))
    grid_y = np.broadcast_to(y_axis[:, None, :], (members, n_x, n_y))
    for plane_index, fraction in enumerate(fractions):
        plane_z = np.maximum(context.receiver_range_m * float(fraction), 1.0)
        ranges[:, plane_index] = plane_z
        plane_intensity, plane_valid, plane_reason, pitch, grid_n = aperture_module.plane_intensity(
            context.state, plane_z, x_axis, y_axis, config,
            amplitude_cache=context.amplitude_cache)
        window = slice(plane_index * n_x * n_y, (plane_index + 1) * n_x * n_y)
        intensity[:, window] = plane_intensity.reshape(members, -1)
        valid[:, window] = plane_valid.reshape(members, -1)
        points[:, window, 0] = grid_x.reshape(members, -1)
        points[:, window, 1] = grid_y.reshape(members, -1)
        points[:, window, 2] = plane_z[:, None]
        # A plane the method could not resolve is UNRESOLVED, not zero: the mask is the product
        # (§5 P5), and the sub-Fresnel geometric fill that used to hide this gap is gone (§8 item 14).
        unresolved = plane_reason != 0
        if unresolved.any():
            valid[np.nonzero(unresolved)[0][:, None], np.arange(window.start, window.stop)] = False
            reason = np.where(unresolved & (reason == 0), plane_reason, reason)
    intensity = np.where(valid, intensity, 0.0)
    return SolvedSampleSet(
        sampling=sampling, points_m=points, intensity_per_m2=intensity, valid=valid,
        grid_shape=grid_shape,
        grid_axes={"z_m": ranges, "z_fraction": fractions, "x_m": x_axis, "y_m": y_axis},
        enclosed_fraction=_enclosed_fraction(intensity, points, grid_shape, fractions.size,
                                             n_x, n_y),
        unresolved_reason=reason, aperture_pitch_m=pitch, aperture_samples=grid_n)


def _enclosed_fraction(intensity, points, grid_shape, n_plane, n_x, n_y):
    """The enclosed-energy field: at each range, the fraction of that plane's power inside |r|.

    Emitted because §8 item 15 says so. The pre-SPEC engine returned NaN-separated CONTOUR POLYLINES
    from a physics module — a drawing, produced inside the model — and the verdict was to emit the
    field the contours come from and let the viewer contour it (which §7 explicitly permits). So this
    is the funnel, as data.

    It is the STRIP enclosed fraction along the sampling's own transverse axes — the power within
    |x| ≤ r of the line, not within a circle of radius r — because that is what a side view's radial
    axis means. Normalised to each plane's total ON THIS GRID, so a beam wider than the requested
    window correctly never reaches a high level instead of being renormalised into one.

    Unoccluded: the occluder is applied afterwards, so what the receiver absorbs SHOWS UP as the
    high-level rings closing at the receiver plane rather than being renormalised away.
    """
    members = intensity.shape[0]
    if n_x > 1 and n_y > 1:
        return None      # a 2-D transverse grid has no single radial axis to enclose along
    radial_count = max(n_x, n_y)
    if radial_count < 2:
        return None
    block = intensity.reshape(members, n_plane, radial_count)
    coordinate = points.reshape(members, n_plane, radial_count, 3)
    radial = coordinate[..., 0] if n_x > 1 else coordinate[..., 1]
    order = np.argsort(np.abs(radial), axis=-1)
    step = np.abs(np.diff(radial, axis=-1)[..., :1])
    sorted_intensity = np.take_along_axis(block, order, axis=-1)
    cumulative = np.cumsum(sorted_intensity, axis=-1) * step
    total = cumulative[..., -1:]
    enclosed = np.divide(cumulative, total, out=np.zeros_like(cumulative), where=total > 0)
    restored = np.empty_like(enclosed)
    np.put_along_axis(restored, order, enclosed, axis=-1)
    return restored.reshape(members, n_plane * radial_count)


# --- shared plumbing ----------------------------------------------------------------------


def _bilinear_sample(x_axis, y_axis, grid, x_query, y_query):
    """Batched bilinear sample of (m, n_x, n_y) grids at (m, n_query) points; zero outside.

    Written out rather than taken from scipy so that it is batched over members: a per-member
    `map_coordinates` call is exactly the kind of loop wearing a vectorized interface that §6a is
    about.
    """
    members = grid.shape[0]
    n_x, n_y = grid.shape[1], grid.shape[2]
    pitch_x = (x_axis[:, -1] - x_axis[:, 0]) / max(n_x - 1, 1)
    pitch_y = (y_axis[:, -1] - y_axis[:, 0]) / max(n_y - 1, 1)
    fractional_x = (x_query - x_axis[:, 0:1]) / np.where(pitch_x > 0, pitch_x, 1.0)[:, None]
    fractional_y = (y_query - y_axis[:, 0:1]) / np.where(pitch_y > 0, pitch_y, 1.0)[:, None]
    low_x = np.floor(fractional_x).astype(int)
    low_y = np.floor(fractional_y).astype(int)
    weight_x = fractional_x - low_x
    weight_y = fractional_y - low_y
    inside = (low_x >= 0) & (low_x < n_x - 1) & (low_y >= 0) & (low_y < n_y - 1)
    clipped_x = np.clip(low_x, 0, n_x - 2)
    clipped_y = np.clip(low_y, 0, n_y - 2)
    rows = np.arange(members)[:, None]
    corner_00 = grid[rows, clipped_x, clipped_y]
    corner_10 = grid[rows, clipped_x + 1, clipped_y]
    corner_01 = grid[rows, clipped_x, clipped_y + 1]
    corner_11 = grid[rows, clipped_x + 1, clipped_y + 1]
    interpolated = (corner_00 * (1 - weight_x) * (1 - weight_y)
                    + corner_10 * weight_x * (1 - weight_y)
                    + corner_01 * (1 - weight_x) * weight_y
                    + corner_11 * weight_x * weight_y)
    return np.where(inside, interpolated, 0.0)


def _apply_occluder(solved: SolvedSampleSet, context: SolveContext) -> SolvedSampleSet:
    if context.captured_geometric is None or context.receiver_half_extent_m is None:
        raise ValueError("occluders='receiver' needs a receiver and its capture: solve the "
                         "silhouette sampling and P6 first")
    aperture_half = np.stack([context.state.semi_major_m, context.state.semi_minor_m], axis=1)
    surviving = occlusion.transmission(
        solved.points_m, context.receiver_range_m, context.receiver_half_extent_m, aperture_half,
        context.state.wavelength_m, context.captured_geometric)
    solved.occluder_transmission = surviving
    solved.intensity_per_m2 = solved.intensity_per_m2 * surviving
    return solved
