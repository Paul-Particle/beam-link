"""Capture: how much of a radiated beam the airframe intercepts, and the aperture that maximises it.

Two derivations of the same quantity, and P6 names which one produced a run:

  overlap_integral      the solved field integrated over the silhouette elements. Inputs: P5, P3.
  closed_form_goubau    1 − exp(−τ²) on the equal-area disc, as the relay up-leg already uses.
                        Inputs: P2, P3 — it never touches the field, so it is a DIFFERENT derivation
                        rather than a cheaper version of the same one, and §6 measures its bias
                        (0.17 ms against 21–26 ms, high by 0.4–4.8 pp, one-sided).

Everything here is batched over (candidate × sample) and every search runs LOCKSTEP (I9). Three
things make that affordable rather than 126× the work:

  the taper grid becomes part of the batch axis  — the search is ONE batched call over
    (member × taper), which is the pre-SPEC coarse scan with the batching spent on the right axis
  the defocus bisection runs a fixed step count  — set by the fidelity config as a tolerance, not
    exited per sample, and only over the members where the cap actually binds
  the searches run at a COARSER capture grid     — a scan and a bisection are RANKING and
    COMPLIANCE decisions; the winner is then re-evaluated once at full resolution, which is what
    the pre-SPEC code did with 256 against 384 and is why the grid is a measured non-lever

and the defocus floor is closed form (D9) rather than a try/except back-off loop.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from beamlink.artifacts.spec import FieldSampling
from beamlink.physics.beam_link import aperture as aperture_module
from beamlink.physics.beam_link import relations, solver
from beamlink.physics.fidelity import FidelityConfig

_SILHOUETTE_SAMPLING = FieldSampling(kind="silhouette", name="silhouette")
# The on-axis peak is a one-point TRANSVERSE PLANE at the receiver range, not a `points` cloud:
# the receiver range differs per member, and a `points` request carries absolute coordinates shared
# by every member. A tiny stated extent keeps the aperture pitch off the query-window constraint, so
# this costs one aperture build and nothing else.
_ON_AXIS_SAMPLING = FieldSampling(kind="transverse_plane", name="on_axis_peak",
                                  z_fractions=(1.0,), n_transverse=(1, 1),
                                  extent_m=(1e-6, 1e-6))


@dataclass(frozen=True)
class DensityModel:
    """How the skin power density follows from the field, given how the run's power is specified.

    With a DELIVERED target the beam power crossing the aircraft is the delivered power divided back
    out through capture and the rectenna, so the density depends on capture as well as on the peak —
    which is why the defocus search has to track both. With a fixed RADIATED power it does not.
    Either way the atmospheric loss between aperture and aircraft is the only upstream term that
    survives, because the peak is normalised per unit radiated power.
    """

    delivered_target_w: float | None
    radiated_w: float | None
    rectenna_total: np.ndarray
    transmittance: np.ndarray
    cap_w_m2: float | None

    def density(self, peak_areal_density_per_m2, capture):
        peak = np.asarray(peak_areal_density_per_m2, dtype=float)
        if self.delivered_target_w is not None:
            crossing = self.delivered_target_w / np.maximum(
                np.asarray(capture, dtype=float) * self.rectenna_total, 1e-12)
            return crossing * peak
        return self.radiated_w * self.transmittance * peak

    def tile(self, repeats: int) -> "DensityModel":
        return replace(self, rectenna_total=np.repeat(self.rectenna_total, repeats),
                       transmittance=np.repeat(self.transmittance, repeats))


@dataclass
class CaptureResult:
    """One resolved capture, batched over members."""

    geometric: np.ndarray
    incidence_weighted: np.ndarray
    peak_areal_density_per_m2: np.ndarray
    taper_exponent: np.ndarray
    focus_range_m: np.ndarray
    density_cap_exceeded: np.ndarray
    method: str
    silhouette_set: solver.SolvedSampleSet | None = None
    """The field sample set the overlap integrated. Kept so that P5 emits the same solve P6 derived
    from rather than a second evaluation of it (I1)."""


def equivalent_disc_radius(peak_areal_density_per_m2):
    """Radius of the uniform disc that would carry all the power at the observed peak, so one length
    scale describes a spot of any shape. Unit-power intensity in (1/m²), metres out.

    A reported measure only: it is not a Gaussian 1/e² waist and nothing in the chain uses it.
    """
    peak = np.asarray(peak_areal_density_per_m2, dtype=float)
    return np.where(peak > 0, 1.0 / np.sqrt(np.pi * np.maximum(peak, 1e-300)), np.inf)


def skin_power_density(radiated_w, peak_areal_density_per_m2, transmittance):
    """Peak power density on the skin, in W/m² — the ABSOLUTE number, once P8 exists.

    `peak_areal_density_per_m2` is per unit radiated power, so the atmospheric loss between aperture
    and aircraft is the one thing that has to be reapplied here; everything upstream of the aperture
    is already in `radiated_w`. Distinct from `DensityModel.density`, which answers the cap question
    DURING the search from the spec's power target, before a radiated power exists.
    """
    return (np.asarray(radiated_w, dtype=float) * np.asarray(transmittance, dtype=float)
            * np.asarray(peak_areal_density_per_m2, dtype=float))


# --- the overlap integral -----------------------------------------------------------------


def overlap(context: solver.SolveContext, elements, incidence_floor, config: FidelityConfig):
    """One evaluation of the overlap integral at the aperture state in `context`.

    Returns (geometric, incidence-weighted, peak areal density, the solved sample set). The rectenna
    angular response is applied to the incidence cosine only — the cos(theta) area foreshortening is
    already in the element's projected area, so applying it here would double-count it.
    """
    solved = solver.solve(_SILHOUETTE_SAMPLING, context, config)
    intensity = solved.intensity_per_m2
    area = np.where(solved.valid, elements["projected_area_m2"], 0.0)
    response = incidence_floor + (1.0 - incidence_floor) * elements["incidence_cos"]
    geometric = np.minimum(np.sum(intensity * area, axis=1), 1.0)
    weighted = np.minimum(np.sum(intensity * area * response, axis=1), 1.0)
    # The peak is at the spot centre, which need not land on a grid node, so evaluate it exactly
    # rather than taking the fine-grid maximum. Same solve, sampled at one point.
    on_axis = solver.solve(_ON_AXIS_SAMPLING, context, config)
    peak = np.where(on_axis.valid[:, 0], on_axis.intensity_per_m2[:, 0], 0.0)
    return geometric, weighted, peak, solved


# --- the closed form ----------------------------------------------------------------------


def closed_form_capture(slant_m, frequency_ghz, semi_major_m, semi_minor_m,
                        equal_area_receiver_radius_m):
    """Goubau collection on the equal-area disc, at the equal-area emitter radius.

    Inputs are P2 and P3 only: no field, no taper, no focusing and no pointing error, which is
    exactly why it is fast and why it is biased HIGH — it assumes the optimal illumination the
    overlap integral has to earn.
    """
    emitter_radius = np.sqrt(np.asarray(semi_major_m, dtype=float)
                             * np.asarray(semi_minor_m, dtype=float))
    tau = relations.coupling_tau(np.asarray(frequency_ghz, dtype=float) * 1e9, slant_m,
                                 emitter_radius, equal_area_receiver_radius_m)
    return relations.collection_efficiency(tau)


def closed_form_peak(slant_m, frequency_ghz, semi_major_m, semi_minor_m):
    """Peak areal density of a uniform elliptical aperture in focus, π·a·b/(λz)², per unit power.

    The closed-form tier's counterpart to the field's on-axis sample: the density cap still has to
    be answerable at a tier that does not solve the field.
    """
    wavelength = relations.wavelength(np.asarray(frequency_ghz, dtype=float) * 1e9)
    return (np.pi * np.asarray(semi_major_m, dtype=float) * np.asarray(semi_minor_m, dtype=float)
            / (wavelength * np.asarray(slant_m, dtype=float)) ** 2)


def resolve_closed_form(slant_m, frequency_ghz, state: aperture_module.ApertureState,
                        equal_area_receiver_radius_m, elements, incidence_floor,
                        density: DensityModel) -> CaptureResult:
    """The closed-form tier. It cannot defocus, so an exceeded cap is reported and not fixed.

    That is a real difference in what the tier can answer, not a rounding difference, and it is the
    reason `capture_method` is part of the product's identity (I5): a run at this tier holds the
    density cap only where the geometry happens to comply.
    """
    geometric = closed_form_capture(slant_m, frequency_ghz, state.semi_major_m, state.semi_minor_m,
                                    equal_area_receiver_radius_m)
    # The incidence derate is the projected-area-weighted mean of the rectenna's angular response
    # over the silhouette, which P3 already carries: the closed form skips the field, not the
    # receiver.
    valid_area = np.where(elements["valid"], elements["projected_area_m2"], 0.0)
    response = incidence_floor + (1.0 - incidence_floor) * elements["incidence_cos"]
    mean_response = (np.sum(valid_area * response, axis=1)
                     / np.maximum(np.sum(valid_area, axis=1), 1e-12))
    peak = closed_form_peak(slant_m, frequency_ghz, state.semi_major_m, state.semi_minor_m)
    weighted = geometric * mean_response
    exceeded = np.zeros(state.size, dtype=bool)
    if density.cap_w_m2 is not None:
        exceeded = density.density(peak, weighted) > density.cap_w_m2
    return CaptureResult(geometric, weighted, peak, state.taper_exponent.copy(),
                         np.asarray(slant_m, dtype=float).copy(), exceeded,
                         "closed_form_goubau", None)


# --- the searched, field-based route ------------------------------------------------------


def resolve(context: solver.SolveContext, elements, incidence_floor, density: DensityModel,
            adaptive_taper: bool, taper_p: float, taper_p_max: float,
            config: FidelityConfig) -> CaptureResult:
    """Choose the aperture taper, then satisfy the skin power-density cap. Batched, lockstep.

    The taper that maximises capture is found on a grid; if that choice breaches the cap it has to
    be defocused, which costs capture, so heavier tapers — which lower the peak on their own — are
    ranked at their POST-defocus capture and the best survivor wins. Unlike the pre-SPEC version
    nothing stops early: every binding member walks the whole bisection, which is why cost is now
    uniform across a pass rather than geometry-dependent.

    The taper grid is a LOOP over a config-set count with the member batch kept whole, not a second
    batch axis. Folding it into the batch was the obvious reading of §6a and measurably the wrong
    one: it copies the padded element cloud once per taper — 127 MB for an 81-sample pass at nine
    tapers — and the memory traffic costs more than the transforms it was meant to amortise (D11).
    """
    tapers = (np.linspace(0.0, float(taper_p_max), config.taper_search_n) if adaptive_taper
              else np.atleast_1d(np.asarray(taper_p, dtype=float)))
    members = context.size
    scan_config = replace(config, capture_grid_n=max(96, config.capture_grid_n // 3))
    slant = context.receiver_range_m
    query_extent = np.maximum(context.receiver_half_extent_m.max(axis=1), 1.0)
    floor = aperture_module.defocus_floor(context.state.with_focus(slant), slant, query_extent,
                                          config)

    capture_by_taper = np.zeros((members, tapers.size))
    focus_by_taper = np.zeros((members, tapers.size))
    exceeded_by_taper = np.zeros((members, tapers.size), dtype=bool)
    for column, taper in enumerate(tapers):
        at_taper = _with_state(context, context.state.with_taper(np.full(members, float(taper)))
                               .with_focus(slant))
        _, weighted, peak, _ = overlap(at_taper, elements, incidence_floor, scan_config)
        chosen_focus = slant.copy()
        exceeded = np.zeros(members, dtype=bool)
        if density.cap_w_m2 is not None:
            binds = density.density(peak, weighted) > density.cap_w_m2
            if binds.any():
                chosen_focus, exceeded, weighted = _defocus_to_cap(
                    at_taper, elements, incidence_floor, density, binds, floor, weighted,
                    scan_config, config)
        capture_by_taper[:, column] = weighted
        focus_by_taper[:, column] = chosen_focus
        exceeded_by_taper[:, column] = exceeded

    # A compliant taper beats a non-compliant one outright; between two of the same compliance the
    # higher post-defocus capture wins.
    score = np.where(exceeded_by_taper, -1.0e3, 0.0) + capture_by_taper
    best = np.argmax(score, axis=1)
    rows = np.arange(members)
    winning_taper = tapers[best]
    winning_focus = focus_by_taper[rows, best]

    # Re-evaluate the winner once, at full resolution.
    final = _with_state(context, context.state.with_taper(winning_taper).with_focus(winning_focus))
    geometric, weighted, peak, solved = overlap(final, elements, incidence_floor, config)
    return CaptureResult(geometric, weighted, peak, winning_taper, winning_focus,
                         exceeded_by_taper[rows, best], "overlap_integral", solved)


def _with_state(context: solver.SolveContext, state) -> solver.SolveContext:
    """The same context at a different aperture state. Shares the element arrays by reference: the
    padded cloud is the largest thing in a batch and must never be copied per search step."""
    return solver.SolveContext(
        state=state, receiver_range_m=context.receiver_range_m,
        element_offsets_m=context.element_offsets_m, element_valid=context.element_valid,
        receiver_half_extent_m=context.receiver_half_extent_m,
        pointing_offset_m=context.pointing_offset_m,
        captured_geometric=context.captured_geometric,
        amplitude_cache=context.amplitude_cache, extras=context.extras)


def _defocus_to_cap(context: solver.SolveContext, elements, incidence_floor,
                    density: DensityModel, binds, floor, weighted, scan_config, config):
    """Bisect the focus range down to the longest one that meets the cap, for the binding members.

    Focusing SHORT of the target spreads the spot, so the longest complying focus is the highest
    capture that complies. `low` is the most-spread geometry the sampling can represent and is
    TESTED rather than assumed compliant: whether it can meet the cap depends on delivered power,
    aperture, range and frequency together, not on emitter size alone (400 MW through a 40 m relay
    aperture overhead sits ~1.3× the cap at every focus).

    Every member of the batch walks the same fixed number of steps; only the members where the cap
    actually binds are in the batch at all, which is a selection, not an early exit.
    """
    index = np.nonzero(binds)[0]
    subset = _subset(context, index)
    subset_elements = {name: value[index] for name, value in elements.items()}
    subset_density = replace(density, rectenna_total=density.rectenna_total[index],
                             transmittance=density.transmittance[index])
    slant = subset.receiver_range_m
    low, high = floor[index].copy(), slant.copy()

    at_floor = overlap(_with_state(subset, subset.state.with_focus(low)), subset_elements,
                       incidence_floor, scan_config)
    unmeetable = subset_density.density(at_floor[2], at_floor[1]) > density.cap_w_m2

    for _ in range(config.density_bisection_steps):
        middle = 0.5 * (low + high)
        step = overlap(_with_state(subset, subset.state.with_focus(middle)), subset_elements,
                       incidence_floor, scan_config)
        too_hot = subset_density.density(step[2], step[1]) > density.cap_w_m2
        high = np.where(too_hot, middle, high)          # still too hot → defocus more
        low = np.where(too_hot, low, middle)
    # Where the cap cannot be met at any allowed defocus, return the most-spread geometry and FLAG
    # it. By decision this surfaces but does not block, and it deliberately does not touch
    # feasibility (P7).
    chosen = np.where(unmeetable, floor[index], low)
    final = overlap(_with_state(subset, subset.state.with_focus(chosen)), subset_elements,
                    incidence_floor, scan_config)

    focus_out = context.receiver_range_m.copy()
    exceeded_out = np.zeros(context.size, dtype=bool)
    weighted_out = np.asarray(weighted, dtype=float).copy()
    focus_out[index] = chosen
    exceeded_out[index] = unmeetable
    weighted_out[index] = final[1]
    return focus_out, exceeded_out, weighted_out


def _subset(context: solver.SolveContext, index) -> solver.SolveContext:
    return solver.SolveContext(
        state=context.state.take(index), receiver_range_m=context.receiver_range_m[index],
        element_offsets_m=None if context.element_offsets_m is None
        else context.element_offsets_m[index],
        element_valid=None if context.element_valid is None else context.element_valid[index],
        receiver_half_extent_m=context.receiver_half_extent_m[index],
        pointing_offset_m=None if context.pointing_offset_m is None
        else context.pointing_offset_m[index],
        captured_geometric=context.captured_geometric, extras=context.extras)
        # NB: a subset gets a FRESH amplitude cache — its member set differs, so the parent's
        # entries would not match its keys anyway.
