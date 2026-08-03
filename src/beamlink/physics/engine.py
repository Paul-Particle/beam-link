"""`evaluate(spec) -> (Products, Manifest)` and `write(...)` — the engine's whole public surface.

D4 makes these two separate calls on purpose. An optimiser calls the first and reads its objective
straight off the emitted aggregate; the archival path calls both. One physics path, two uses — not
two paths. And nothing here is process-global: caches are correctness-neutral and explicitly keyed
(`physics.cache`), so parallel calls cannot interfere.

The engine's unit of work is ONE scenario, including its trajectory, batched over a candidate axis.
Anything that varies the scenario is a driver, and a driver must be able to treat this as a
black-box callable.

Order of derivation, which is also the order I3's dependency claims run in:

    P1 trajectory  ← the spec
    P2 geometry    ← P1, per leg (elevation ground-referenced at the emitting end, D1)
    P10 atmosphere ← P2
    P3 silhouette  ← P2's arrival angles
    P4 aperture    ← P2, P3 (the aspect from P3's rms extents; the frame from P3's basis)
    P5 field       ← P4, sampled where the spec asks (D2), scaled by the radiated power
    P6 capture     ← P5 × P3, or P2 × P3 at the closed-form tier
    P7 chain       ← P6, P10
    P8 power       ← P7 and the spec's power target
    P9 aggregates  ← P7, P8
    P11–P13        ← P3 (belly area → rectenna mass) and the airframe spec; independent of the link

Everything per-sample carries a leading candidate axis. Internally the two are flattened into one
member axis of length n_candidates × n_samples, because that is the axis the field kernels batch over
(I9); products are reshaped back on the way out.
"""

from __future__ import annotations

import dataclasses
import platform
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from beamlink import artifacts
from beamlink.artifacts import products as product_types
from beamlink.artifacts import serialize
from beamlink.artifacts.manifest import Manifest, ProductStatus
from beamlink.artifacts.spec import FieldSampling, RunSpec, ScenarioSpec
from beamlink.physics import aggregate, fidelity, geo
from beamlink.physics.airframe import flight
from beamlink.physics.airframe import loft as loft_module
from beamlink.physics.airframe import silhouette as silhouette_module
from beamlink.physics.beam_link import aperture as aperture_module
from beamlink.physics.beam_link import atmosphere
from beamlink.physics.beam_link import capture as capture_module
from beamlink.physics.beam_link import chain, occlusion, relations, solver

ENGINE_VERSION = "0.2.0"

_LINK_PRODUCTS = ("silhouette", "capture", "chain", "power", "aggregates")
_NO_RECEIVER_REASON = ("the scenario has no receiver, so there is nothing to capture on "
                       "(SPEC.md §2, §5) — the aperture and field products still exist, which is "
                       "the general form of the pure focusing sweep")


# --- candidate expansion ------------------------------------------------------------------


def _apply_override(scenario, dotted_path: str, value):
    """Set one dotted parameter on a copy of a scenario, e.g. "aperture.fixed_aspect"."""
    head, _, tail = dotted_path.partition(".")
    if not hasattr(scenario, head):
        raise KeyError(f"scenario has no parameter {head!r} (from {dotted_path!r})")
    if not tail:
        return replace(scenario, **{head: value})
    nested = getattr(scenario, head)
    if nested is None:
        raise ValueError(f"cannot set {dotted_path!r}: {head} is None in this scenario")
    return replace(scenario, **{head: _apply_override(nested, tail, value)})


def candidate_scenarios(spec: RunSpec) -> tuple[ScenarioSpec, ...]:
    """The scenario per candidate. Axes are ZIPPED, not crossed — building a cross product is the
    driver's job, because only the driver knows whether a full grid is affordable (D3, D4)."""
    scenarios = []
    for index in range(spec.n_candidates):
        scenario = spec.scenario
        for axis in spec.candidates:
            scenario = _apply_override(scenario, axis.parameter, axis.values[index])
        scenarios.append(scenario)
    return tuple(scenarios)


def _require_homogeneous(scenarios) -> None:
    """One run is one architecture. A candidate axis may not change the number of link legs.

    D7: leg-indexed products (P2, P8, P10) would otherwise have a ragged leg axis, and "relay" and
    "direct" are not the same model at different parameter values — they are different models, and
    comparing them is a study over two runs, which D6 already makes the driver's job.
    """
    if len({scenario.relay is None for scenario in scenarios}) > 1:
        raise ValueError("a candidate axis may not switch between the direct and relay "
                         "architectures: that is two runs and a study, not one run (SPEC.md D7)")
    if len({scenario.receiver is None for scenario in scenarios}) > 1:
        raise ValueError("a candidate axis may not add or remove the receiver: the set of products a "
                         "run produces would differ per candidate (SPEC.md D7)")


def leg_frequencies(scenario: ScenarioSpec) -> tuple[float, ...]:
    """The frequency each leg radiates at. A passive relay cannot retune, so it re-emits f_in."""
    if scenario.relay is None:
        return (scenario.frequency_ghz,)
    relay = scenario.relay
    return (relay.f_in_ghz, relay.f_in_ghz if relay.mode == "passive" else relay.f_out_ghz)


# --- P1 / P2 / P10 ------------------------------------------------------------------------


def _trajectory(spec: RunSpec) -> product_types.Trajectory:
    trajectory = spec.trajectory
    positions_ecef = geo.positions_to_ecef(trajectory.positions_m)
    speed = np.zeros(trajectory.n_samples)
    if trajectory.n_samples > 1:
        steps = np.linalg.norm(np.diff(positions_ecef, axis=0), axis=1)
        intervals = np.diff(trajectory.times_s)
        rate = steps / np.where(np.abs(intervals) > 0, intervals, np.inf)
        speed = np.concatenate([[rate[0]], 0.5 * (rate[1:] + rate[:-1]), [rate[-1]]])
    return product_types.Trajectory(
        positions_m=trajectory.positions_m, attitudes_deg=trajectory.attitudes_deg,
        times_s=trajectory.times_s, positions_ecef_m=positions_ecef,
        ground_speed_ms=np.abs(speed), name=trajectory.name)


def _geometry(scenarios, trajectory, config) -> product_types.LinkGeometry:
    """Per-leg slant, ground-referenced elevation, blockage, and the arrival angles (D1)."""
    n_candidates, n_samples = len(scenarios), trajectory.times_s.size
    relayed = scenarios[0].relay is not None
    leg_names = ("station->relay", "relay->aircraft") if relayed else ("station->aircraft",)
    n_legs = len(leg_names)

    slant = np.zeros((n_candidates, n_legs, n_samples))
    elevation = np.zeros((n_candidates, n_legs, n_samples))
    blocked = np.zeros((n_candidates, n_legs, n_samples), dtype=bool)
    emitter_ecef = np.zeros((n_candidates, n_legs, n_samples, 3))
    beta = np.zeros((n_candidates, n_samples))
    alpha = np.zeros((n_candidates, n_samples))
    aircraft = trajectory.positions_ecef_m

    for index, scenario in enumerate(scenarios):
        ends = [geo.positions_to_ecef(np.asarray(scenario.station_position_m, dtype=float))]
        if relayed:
            ends.append(geo.positions_to_ecef(np.asarray(scenario.relay.position_m, dtype=float)))
        for leg, emitting_end in enumerate(ends):
            emitting = np.broadcast_to(emitting_end, (n_samples, 3))
            receiving = (np.broadcast_to(ends[leg + 1], (n_samples, 3)) if leg + 1 < len(ends)
                         else aircraft)
            # Elevation is measured AT THE EMITTING END, above its own local horizon — one
            # definition applied per leg, which is what dissolves the pre-SPEC double key (D1).
            slant[index, leg], elevation[index, leg] = geo.link(emitting, receiving)
            blocked[index, leg] = geo.horizon_blocked(emitting, receiving,
                                                     config.atmosphere_path_samples)
            emitter_ecef[index, leg] = emitting
        towards_emitter = geo.direction_in_local_enu(aircraft, emitter_ecef[index, -1])
        beta[index], alpha[index] = loft_module.beta_alpha_from_enu(towards_emitter,
                                                                  trajectory.attitudes_deg)
    return product_types.LinkGeometry(
        leg_names=leg_names, slant_m=slant, elevation_deg=elevation, horizon_blocked=blocked,
        arrival_beta_deg=beta, arrival_alpha_deg=alpha, emitter_position_ecef_m=emitter_ecef)


def _atmosphere(scenarios, trajectory, geometry, config) -> product_types.Atmosphere:
    n_candidates, n_legs, n_samples = geometry.slant_m.shape
    loss = np.zeros((n_candidates, n_legs, n_samples))
    gaseous, rain, cloud = (np.zeros_like(loss) for _ in range(3))
    spectrum_loss, frequency_grid = [], np.zeros(0)
    for index, scenario in enumerate(scenarios):
        frequencies = leg_frequencies(scenario)
        for leg in range(n_legs):
            emitting = geometry.emitter_position_ecef_m[index, leg]
            receiving = (geometry.emitter_position_ecef_m[index, leg + 1] if leg + 1 < n_legs
                         else trajectory.positions_ecef_m)
            parts = atmosphere.path_loss_db(frequencies[leg], emitting, receiving,
                                            scenario.weather, config.atmosphere_path_samples)
            gaseous[index, leg], rain[index, leg], cloud[index, leg] = parts
            loss[index, leg] = sum(parts)
        frequency_grid, zenith = atmosphere.zenith_spectrum_db(
            float(np.max(trajectory.positions_m[:, 2])), scenario.weather,
            config.atmosphere_path_samples)
        spectrum_loss.append(zenith)
    return product_types.Atmosphere(
        inputs=("geometry",), loss_db=loss,
        transmittance=atmosphere.transmittance(loss, geometry.horizon_blocked),
        gaseous_db=gaseous, rain_db=rain, cloud_db=cloud,
        spectrum_frequency_ghz=frequency_grid, spectrum_zenith_loss_db=np.stack(spectrum_loss),
        profile_source=atmosphere.profile().source)


# --- P3 -----------------------------------------------------------------------------------


def _silhouette(scenarios, geometry, config) -> product_types.Silhouette:
    loft = scenarios[0].airframe.loft
    vertices, faces, normals = loft_module.mesh(loft)
    cloud = silhouette_module.element_cloud(loft, geometry.arrival_beta_deg,
                                            geometry.arrival_alpha_deg, config)
    return product_types.Silhouette(
        mesh_vertices_m=vertices, mesh_faces=faces, mesh_face_normals=normals,
        element_offsets_m=cloud["offsets_m"],
        element_projected_area_m2=cloud["projected_area_m2"],
        element_incidence_cos=cloud["incidence_cos"], element_valid=cloud["valid"],
        projected_area_m2=cloud["area_m2"],
        equal_area_radius_m=silhouette_module.equal_area_radius(cloud["area_m2"]),
        view_basis_body=cloud["basis_body"], principal_half_extent_m=cloud["half_extent_m"],
        principal_sigma_m=cloud["sigma_m"], principal_skew_m3=cloud["skew_m3"],
        max_half_extent_m=silhouette_module.max_half_extent(loft, config))


# --- P4 -----------------------------------------------------------------------------------


def _aperture_state(scenarios, geometry, silhouette, trajectory, config):
    """The aperture per member, plus its frame in the earth frame.

    P4 and P5 describe the leg that SERVES THE AIRCRAFT. A relay's up-leg is aperture-to-aperture and
    the engine has no field solve for it — its capture is the closed-form Goubau coupling in
    `chain.up_leg_capture`. The pre-SPEC animation synthesized an up-leg aperture inside the figure
    module in order to draw it, which is exactly the consumer-computes-physics defect §1 is about; an
    up-leg field is new products, not a figure's local calculation (D12).
    """
    n_candidates, n_samples = len(scenarios), trajectory.times_s.size
    members = n_candidates * n_samples
    slant = geometry.slant_m[:, -1, :].reshape(members)
    frequency = np.concatenate([np.full(n_samples, leg_frequencies(s)[-1]) for s in scenarios])
    has_receiver = scenarios[0].receiver is not None

    equal_area_diameter = np.zeros(members)
    aspect = np.ones(members)
    for index, scenario in enumerate(scenarios):
        window = slice(index * n_samples, (index + 1) * n_samples)
        if scenario.relay is not None:
            diameter = np.full(n_samples, float(scenario.relay.aperture_diam_m))
        elif scenario.aperture.diameter_m is not None:
            diameter = np.full(n_samples, float(scenario.aperture.diameter_m))
        elif has_receiver:
            # No stated diameter: size to the coupling target at THIS sample's geometry, which is a
            # per-sample resize no real station can do. Legal for a design study; a run meant to be
            # one built array states a diameter.
            diameter = relations.emitter_diameter_for_tau(
                frequency[window], slant[window], silhouette.equal_area_radius_m[index],
                scenario.aperture.tau_target)
        else:
            raise ValueError("a scenario with no receiver must state aperture.diameter_m: there is "
                             "no silhouette to size the aperture against (SPEC.md §2, D2)")
        equal_area_diameter[window] = diameter
        if scenario.aperture.fixed_aspect is not None:
            aspect[window] = max(float(scenario.aperture.fixed_aspect), 1.0)
        elif scenario.aperture.shaped and has_receiver:
            # The aperture is stretched along whichever direction the target is NARROW in, because
            # spot width goes as 1/aperture width. The elongation is the ratio of P3's rms extents.
            sigma = silhouette.principal_sigma_m[index]
            aspect[window] = np.clip(sigma[:, 1] / np.maximum(sigma[:, 0], 1e-9), 1.0,
                                     scenario.aperture.aspect_cap)

    radius = equal_area_diameter / 2.0
    state = aperture_module.ApertureState(
        semi_major_m=radius * np.sqrt(aspect), semi_minor_m=radius / np.sqrt(aspect),
        taper_exponent=np.full(members, float(scenarios[0].aperture.taper_p)),
        focus_range_m=slant, wavelength_m=relations.wavelength(frequency * 1e9))
    frame, origin = _aperture_frame(scenarios, geometry, silhouette, trajectory, has_receiver)
    return state, frame, origin, equal_area_diameter


def _aperture_frame(scenarios, geometry, silhouette, trajectory, has_receiver):
    """(members, 3, 3) aperture-frame axes in the earth frame, and the emitting origin.

    With a receiver the frame IS P3's view basis, lifted from the airframe body frame into the earth
    frame — one frame, defined once, shared by P3, P4 and P5. Without a receiver the transverse axes
    come from the scenario's declared `aim_reference_enu`, orthogonalised against the propagation
    direction: there is no silhouette to align to, so the reference has to be an input (D2).
    """
    n_candidates, n_samples = len(scenarios), trajectory.times_s.size
    members = n_candidates * n_samples
    frame = np.zeros((members, 3, 3))
    origin = np.zeros((members, 3))
    for index, scenario in enumerate(scenarios):
        window = slice(index * n_samples, (index + 1) * n_samples)
        emitting = geometry.emitter_position_ecef_m[index, -1]
        origin[window] = emitting
        if has_receiver:
            body_to_enu = loft_module.body_to_enu_matrix(trajectory.attitudes_deg)
            basis_enu = silhouette.view_basis_body[index] @ body_to_enu
            frame[window] = geo.enu_to_ecef(basis_enu, trajectory.positions_ecef_m[:, None, :])
        else:
            propagation = geo.unit(trajectory.positions_ecef_m - emitting)
            reference = geo.enu_to_ecef(
                np.broadcast_to(np.asarray(scenario.aim_reference_enu, dtype=float),
                                (n_samples, 3)), emitting)
            x_axis, y_axis = geo.transverse_basis(propagation, reference)
            frame[window] = np.stack([x_axis, y_axis, propagation], axis=1)
    return frame, origin


# --- P6 / P7 / P8 -------------------------------------------------------------------------


def _rectenna_terms(scenarios, trajectory):
    """(members,) rectenna RF→DC and polarisation efficiency; per-sample only through roll."""
    n_samples = trajectory.times_s.size
    rectenna, polarization = [], []
    for scenario in scenarios:
        rectenna.append(np.full(n_samples,
                                float(relations.rectenna_efficiency(leg_frequencies(scenario)[-1]))))
        polarization.append(relations.polarization_efficiency(
            scenario.polarization, trajectory.attitudes_deg[:, 2]))
    return np.concatenate(rectenna), np.concatenate(polarization)


def _solve_context(scenarios, geometry, silhouette, state, members):
    slant = geometry.slant_m[:, -1, :].reshape(members)
    if silhouette is None:
        return solver.SolveContext(state=state, receiver_range_m=slant,
                                  receiver_half_extent_m=np.zeros((members, 2)))
    # The offset is a MAGNITUDE in the spec; its direction is the one that costs more, taken from the
    # projection's own third moment so the answer does not depend on the frame's gauge (D13). Zero
    # skew (a symmetric projection) makes the two directions equivalent, so either will do.
    skew = silhouette.principal_skew_m3.reshape(members, 2)[:, 0]
    direction = np.where(skew >= 0.0, -1.0, 1.0)
    return solver.SolveContext(
        state=state, receiver_range_m=slant,
        element_offsets_m=silhouette.element_offsets_m.reshape(members, -1, 2),
        element_valid=silhouette.element_valid.reshape(members, -1),
        receiver_half_extent_m=silhouette.principal_half_extent_m.reshape(members, 2),
        pointing_offset_m=direction * abs(float(scenarios[0].aperture.pointing_error_m)))


def _capture(spec, scenarios, geometry, silhouette, state, context, atmosphere_product, rectenna,
             polarization, config):
    """P6, and the field sample set it derived from."""
    n_candidates, n_samples = len(scenarios), geometry.slant_m.shape[2]
    members = n_candidates * n_samples
    slant = geometry.slant_m[:, -1, :].reshape(members)
    scenario = scenarios[0]
    receiver = scenario.receiver
    elements = dict(
        projected_area_m2=silhouette.element_projected_area_m2.reshape(members, -1),
        incidence_cos=silhouette.element_incidence_cos.reshape(members, -1),
        valid=silhouette.element_valid.reshape(members, -1))
    density = capture_module.DensityModel(
        delivered_target_w=spec.power.delivered_target_w, radiated_w=spec.power.radiated_w,
        rectenna_total=rectenna * polarization,
        transmittance=atmosphere_product.transmittance[:, -1, :].reshape(members),
        cap_w_m2=receiver.max_power_density_w_m2)

    if config.capture_method == "closed_form_goubau":
        frequency = np.concatenate([np.full(n_samples, leg_frequencies(s)[-1]) for s in scenarios])
        result = capture_module.resolve_closed_form(
            slant, frequency, state, silhouette.equal_area_radius_m.reshape(members), elements,
            receiver.incidence_floor, density)
        return result, ("geometry", "silhouette")
    result = capture_module.resolve(
        context, elements, receiver.incidence_floor, density, scenario.aperture.adaptive_taper,
        scenario.aperture.taper_p, scenario.aperture.taper_p_max, config)
    return result, ("field", "silhouette")


def _chain(scenarios, geometry, atmosphere_product, capture_result, rectenna, polarization):
    n_candidates, n_legs, n_samples = geometry.slant_m.shape
    members = n_candidates * n_samples
    scenario = scenarios[0]
    transmittance = atmosphere_product.transmittance
    serving = transmittance[:, -1, :].reshape(members)
    if scenario.relay is None:
        factors = chain.direct_factors(scenario, capture_result.incidence_weighted, serving,
                                      polarization, rectenna)
    else:
        capture_up, _ = chain.up_leg_capture(scenario, geometry.slant_m[:, 0, :].reshape(members))
        factors = chain.relay_factors(scenario, capture_up,
                                      transmittance[:, 0, :].reshape(members),
                                      capture_result.incidence_weighted, serving, polarization,
                                      rectenna)
    blocked = geometry.horizon_blocked.transpose(1, 0, 2).reshape(n_legs, members)
    names, stacked, eta, feasible = chain.assemble(factors, blocked)
    shape = (n_candidates, n_samples)
    product = product_types.Chain(
        inputs=("capture", "atmosphere", "geometry"), factor_names=names,
        factors=stacked.reshape((len(names),) + shape).transpose(1, 0, 2),
        eta_end_to_end=eta.reshape(shape), feasible=feasible.reshape(shape))
    by_name = {name: np.asarray(value, dtype=float).reshape(shape) for name, value in factors}
    return product, by_name


def _power(spec, chain_product, factors, capture_result, atmosphere_product, geometry):
    """P8. One conversion, one place.

    Under the delivered-target basis the station is rated so the target arrives at the BEST sample of
    the trajectory and holds that grid draw — which is what a station with one rated draw does, and
    what makes `grid_w` constant along a pass instead of chasing 1/η.
    """
    n_candidates, n_samples = chain_product.eta_end_to_end.shape
    n_legs = geometry.slant_m.shape[1]
    eta = chain_product.eta_end_to_end
    station_side = factors["grid_to_rf"] * factors["emitter_aperture"]
    if spec.power.delivered_target_w is not None:
        peak = np.max(eta, axis=1)
        rated = spec.power.delivered_target_w / np.where(peak > 0, peak, np.inf)
        grid = np.broadcast_to(rated[:, None], eta.shape).copy()
        basis = "delivered_target_at_best_sample"
    else:
        rated = None
        grid = np.broadcast_to(spec.power.radiated_w / np.where(station_side > 0, station_side,
                                                               np.inf), eta.shape).copy()
        basis = "radiated_fixed"

    radiated = np.zeros((n_candidates, n_legs, n_samples))
    radiated[:, 0] = grid * station_side
    if n_legs == 2:
        radiated[:, 1] = (radiated[:, 0] * factors["capture_station_to_relay"]
                          * factors["atmosphere_station_to_relay"] * factors["relay_convert"]
                          * factors["relay_aperture"])
    at_receiver = (radiated[:, -1] * capture_result.incidence_weighted.reshape(eta.shape)
                   * atmosphere_product.transmittance[:, -1, :])
    delivered = grid * eta
    return product_types.Power(
        inputs=("chain", "atmosphere", "capture"),
        reference_planes=artifacts.POWER_REFERENCE_PLANES, grid_w=grid, radiated_w=radiated,
        at_receiver_w=at_receiver, rectenna_terminals_w=delivered, waste_w=grid - delivered,
        rating_basis=basis, rated_grid_w=rated)


def _radiated_without_chain(spec, scenarios, trajectory):
    """(members,) radiated power for a run with no receiver, hence no chain and no P8.

    §2 and D2 make a no-receiver scenario legal, and P5 is emitted in absolute units, so the run has
    to state a radiated power rather than a delivered target — there is nothing to deliver to.
    """
    if spec.power.radiated_w is None:
        raise ValueError("a scenario with no receiver must specify power.radiated_w, not a "
                         "delivered target: there is no receiver to deliver to (SPEC.md §2)")
    n_samples = trajectory.times_s.size
    station_side = np.concatenate([
        np.full(n_samples, float(relations.eta_grid_to_rf(leg_frequencies(s)[-1], s.pae_scenario)
                                 * relations.emitter_aperture_efficiency(
                                     s.aperture.error_efficiency)))
        for s in scenarios])
    return np.full(station_side.size, float(spec.power.radiated_w))


# --- P5 -----------------------------------------------------------------------------------


def _field(request, context, capture_result, radiated_serving_w, config, n_candidates, n_samples):
    """P5: the solved silhouette set plus whatever else the spec asked for, in absolute W/m².

    The silhouette set is the one P6 integrated — not a second evaluation of it (I1). Everything is
    scaled by the SERVING leg's radiated power, so the field is absolute, which is what an exposure
    level can be compared against and what keeps two legs of a relay in true relative brightness.
    """
    sets: dict[str, product_types.FieldSampleSet] = {}
    reasons: dict[str, str] = {}
    requested = list(request.sampling) if request is not None else []
    if capture_result is not None and capture_result.silhouette_set is not None:
        requested = [s for s in requested if s.kind != "silhouette"]
        sets["silhouette"] = _to_product(capture_result.silhouette_set, radiated_serving_w,
                                        n_candidates, n_samples, reasons)
    occluding = replace(context, captured_geometric=None if capture_result is None
                        else capture_result.geometric)
    for sampling in requested:
        sets[sampling.name] = _to_product(solver.solve(sampling, occluding, config),
                                         radiated_serving_w, n_candidates, n_samples, reasons)
    return product_types.Field(sets=sets, method=solver.method_name(config),
                               unresolved_reason=reasons)


def _to_product(solved, radiated_w, n_candidates, n_samples, reasons):
    shape = (n_candidates, n_samples)
    n_point = solved.intensity_per_m2.shape[1]
    absolute = solved.intensity_per_m2 * np.asarray(radiated_w, dtype=float)[:, None]
    codes = solved.unresolved_reason if solved.unresolved_reason is not None else np.zeros(1, int)
    for code in np.unique(codes):
        if int(code) != 0:
            reasons[solved.sampling.name] = aperture_module.UNRESOLVED_REASONS[int(code)]
    return product_types.FieldSampleSet(
        sampling=solved.sampling, points_m=solved.points_m.reshape(shape + (n_point, 3)),
        intensity_w_m2=absolute.reshape(shape + (n_point,)),
        valid=solved.valid.reshape(shape + (n_point,)),
        grid_shape=tuple(int(v) for v in solved.grid_shape),
        grid_axes={name: np.asarray(value) for name, value in solved.grid_axes.items()},
        occluder_transmission=None if solved.occluder_transmission is None
        else solved.occluder_transmission.reshape(shape + (n_point,)),
        aperture_pitch_m=None if solved.aperture_pitch_m is None
        else solved.aperture_pitch_m.reshape(shape + (2,)),
        aperture_samples=None if solved.aperture_samples is None
        else solved.aperture_samples.reshape(shape + (2,)))


# --- P11–P13 ------------------------------------------------------------------------------


def _flight_products(scenarios, trajectory, config):
    """The airframe's own products. Independent of the link: coupled only through the spec (D5)."""
    airframe = scenarios[0].airframe
    belly = silhouette_module.silhouette_at(airframe.loft, 0.0, 0.0, config)
    density = np.asarray([s.receiver.rectenna_areal_density_kg_m2 if s.receiver is not None else 0.0
                          for s in scenarios], dtype=float)
    rectenna_kg = flight.rectenna_mass_kg(float(belly["area"]), density)
    design_speed = float(np.median(trajectory.ground_speed_ms)) if trajectory.times_s.size else 0.0
    speeds = np.full(len(scenarios), design_speed)
    cascade = flight.mass_cascade(airframe, rectenna_kg, speeds, config.mass_iterations)

    mass = product_types.Mass(
        inputs=("silhouette",), payload_t=cascade["payload_kg"] / 1e3,
        structure_t=cascade["structure_kg"] / 1e3, systems_t=cascade["systems_kg"] / 1e3,
        motor_t=cascade["motor_kg"] / 1e3, reserve_battery_t=cascade["reserve_battery_kg"] / 1e3,
        rectenna_t=cascade["rectenna_kg"] / 1e3, mtow_t=cascade["mtow_kg"] / 1e3,
        empty_fraction=(cascade["mtow_kg"] - cascade["payload_kg"]) / cascade["mtow_kg"],
        iterations=int(config.mass_iterations), residual_kg=cascade["residual_kg"])

    cruise_shaft = flight.cruise_shaft_power_w(cascade["mtow_kg"], speeds, airframe.lift_to_drag)
    peak_shaft = flight.climb_shaft_power_w(cascade["mtow_kg"], speeds, airframe.lift_to_drag,
                                            airframe.climb_rate_ms)
    per_sample = flight.cruise_shaft_power_w(cascade["mtow_kg"][:, None],
                                             trajectory.ground_speed_ms[None, :],
                                             airframe.lift_to_drag)
    flight_power = product_types.FlightPower(
        inputs=("mass", "trajectory"), reference_plane="rectenna_terminals",
        cruise_shaft_w=cruise_shaft, peak_shaft_w=peak_shaft,
        required_w=per_sample / airframe.drivetrain_efficiency,
        peak_required_w=peak_shaft / airframe.drivetrain_efficiency,
        lift_to_drag=np.full(len(scenarios), airframe.lift_to_drag))
    reserve = product_types.Reserve(inputs=("mass",),
                                    **flight.reserve(airframe, cascade, speeds))
    return mass, flight_power, reserve, float(belly["area"])


# --- P9 -----------------------------------------------------------------------------------


def _aggregates(serve_eta_floor, chain_product, power, capture_product, trajectory):
    entries = {}
    available = {"chain.eta_end_to_end": chain_product.eta_end_to_end,
                 "power.rectenna_terminals_w": power.rectenna_terminals_w,
                 "power.grid_w": power.grid_w}
    if capture_product is not None:
        available["capture.incidence_weighted"] = capture_product.incidence_weighted
    for name, definition in aggregate.definitions(trajectory.name, serve_eta_floor).items():
        if definition.statistic == "energy_ratio":
            entries[name] = aggregate.build(
                definition, power.rectenna_terminals_w, chain_product.eta_end_to_end,
                chain_product.feasible, trajectory.times_s, denominator=power.grid_w)
        elif definition.quantity in available:
            entries[name] = aggregate.build(definition, available[definition.quantity],
                                            chain_product.eta_end_to_end, chain_product.feasible,
                                            trajectory.times_s)
    return product_types.Aggregates(entries=entries)


# --- the entry points ---------------------------------------------------------------------


def evaluate(spec: RunSpec) -> tuple[product_types.Products, Manifest]:
    """One RunSpec in, products and a manifest out. Writes nothing (D4)."""
    started = time.time()
    scenarios = candidate_scenarios(spec)
    _require_homogeneous(scenarios)
    configs = fidelity.resolve(spec.fidelity.tier, spec.fidelity.overrides)
    n_candidates, n_samples = len(scenarios), spec.trajectory.n_samples
    members = n_candidates * n_samples
    shape = (n_candidates, n_samples)
    has_receiver = scenarios[0].receiver is not None
    wanted = {request.name for request in spec.products}

    built: dict[str, object] = {}
    statuses: dict[str, ProductStatus] = {}

    def record(name, product, inputs=(), method="", resolution=None, tier=None):
        built[name] = product
        statuses[name] = ProductStatus(
            name=name, status="produced", tier=tier or configs[name].tier, method=method,
            resolution=resolution or configs[name].resolution(), inputs=tuple(inputs))

    def skip(name, reason):
        statuses[name] = ProductStatus(name=name, status="skipped", reason=reason,
                                       tier=configs[name].tier)

    for name in artifacts.PRODUCTS:
        if name not in wanted:
            skip(name, "not requested in this run")

    # ---- P1, P2, P10 -----------------------------------------------------------------------
    trajectory = _trajectory(spec)
    if "trajectory" in wanted:
        record("trajectory", trajectory, method="spec_echo")
    geometry = _geometry(scenarios, trajectory, configs["geometry"])
    if "geometry" in wanted:
        record("geometry", geometry, inputs=("trajectory",),
               method="spherical_earth_per_leg_ground_referenced")
    atmosphere_product = _atmosphere(scenarios, trajectory, geometry, configs["atmosphere"])
    if "atmosphere" in wanted:
        record("atmosphere", atmosphere_product, inputs=("geometry",),
               method=f"itu_profile_path_integral[{atmosphere_product.profile_source}]")

    # ---- P3 --------------------------------------------------------------------------------
    silhouette = None
    if has_receiver:
        silhouette = _silhouette(scenarios, geometry, configs["silhouette"])
        if "silhouette" in wanted:
            record("silhouette", silhouette, inputs=("geometry",),
                   method="zbuffer_raster_principal_frame",
                   resolution={**configs["silhouette"].resolution(),
                               "n_faces": int(silhouette.mesh_faces.shape[0]),
                               "n_elements_padded": int(silhouette.element_valid.shape[-1])})
    else:
        for name in _LINK_PRODUCTS:
            if name in wanted:
                skip(name, _NO_RECEIVER_REASON)

    # ---- P4, P6 ----------------------------------------------------------------------------
    field_config = configs["field"]
    state, frame, origin, equal_area_diameter = _aperture_state(
        scenarios, geometry, silhouette, trajectory, field_config)
    context = _solve_context(scenarios, geometry, silhouette, state, members)

    capture_result = capture_inputs = None
    rectenna = polarization = None
    if has_receiver:
        rectenna, polarization = _rectenna_terms(scenarios, trajectory)
        capture_result, capture_inputs = _capture(
            spec, scenarios, geometry, silhouette, state, context, atmosphere_product, rectenna,
            polarization, configs["capture"])
        state = replace(state, taper_exponent=capture_result.taper_exponent,
                        focus_range_m=capture_result.focus_range_m)
        context = replace(context, state=state)

    if "aperture" in wanted:
        request = spec.requested("aperture")
        phase = phase_axis = None
        if request is not None and request.emit_aperture_phase:
            values, coordinate = aperture_module.aperture_phase(state, context.receiver_range_m)
            phase = values.reshape(shape + values.shape[1:])
            phase_axis = coordinate.reshape(shape + coordinate.shape[1:])
        record("aperture", product_types.Aperture(
            semi_major_m=state.semi_major_m.reshape(shape),
            semi_minor_m=state.semi_minor_m.reshape(shape),
            equal_area_diameter_m=equal_area_diameter.reshape(shape),
            taper_exponent=state.taper_exponent.reshape(shape),
            focus_range_m=state.focus_range_m.reshape(shape),
            aspect=(state.semi_major_m / state.semi_minor_m).reshape(shape),
            frame_axes_ecef=frame.reshape(shape + (3, 3)),
            origin_ecef_m=origin.reshape(shape + (3,)),
            phase_rad=phase, phase_axis_m=phase_axis),
            inputs=("geometry", "silhouette") if has_receiver else ("geometry",),
            method="equal_area_ellipse_aligned_to_silhouette" if has_receiver
            else "equal_area_ellipse_on_declared_reference", tier=field_config.tier)

    # ---- P7, P8 ----------------------------------------------------------------------------
    chain_product = power = capture_product = None
    if capture_result is not None:
        chain_product, factors = _chain(scenarios, geometry, atmosphere_product, capture_result,
                                        rectenna, polarization)
        power = _power(spec, chain_product, factors, capture_result, atmosphere_product, geometry)
        radiated_serving = power.radiated_w[:, -1, :].reshape(members)
        if "chain" in wanted:
            record("chain", chain_product, inputs=chain_product.inputs, method="multiplicative")
        if "power" in wanted:
            record("power", power, inputs=power.inputs, method=power.rating_basis)
        if "capture" in wanted:
            capture_product = product_types.Capture(
                inputs=capture_inputs, method=capture_result.method,
                geometric=capture_result.geometric.reshape(shape),
                incidence_weighted=capture_result.incidence_weighted.reshape(shape),
                peak_areal_density_per_m2=capture_result.peak_areal_density_per_m2.reshape(shape),
                equivalent_disc_radius_m=capture_module.equivalent_disc_radius(
                    capture_result.peak_areal_density_per_m2).reshape(shape),
                density_cap_exceeded=capture_result.density_cap_exceeded.reshape(shape),
                skin_power_density_w_m2=capture_module.skin_power_density(
                    radiated_serving, capture_result.peak_areal_density_per_m2,
                    atmosphere_product.transmittance[:, -1, :].reshape(members)).reshape(shape))
            record("capture", capture_product, inputs=capture_product.inputs,
                   method=capture_product.method, tier=field_config.tier)
    else:
        radiated_serving = _radiated_without_chain(spec, scenarios, trajectory)

    # ---- P5 --------------------------------------------------------------------------------
    if "field" in wanted:
        field_product = _field(spec.requested("field"), context, capture_result, radiated_serving,
                               field_config, n_candidates, n_samples)
        if not field_product.sets:
            skip("field", "no field sampling was requested and the scenario has no receiver, so "
                          "there is no default sampling to fall back on (SPEC.md D2)")
        else:
            record("field", field_product, inputs=("aperture", "power") if power is not None
                   else ("aperture",), method=field_product.method,
                   resolution={**field_config.resolution(),
                               "sample_sets": sorted(field_product.sets),
                               "occluder_method": occlusion.METHOD},
                   tier=field_config.tier)

    # ---- P9 --------------------------------------------------------------------------------
    if chain_product is not None and power is not None and "aggregates" in wanted:
        request = spec.requested("aggregates")
        floor = float(request.serve_eta_floor) if request is not None else 0.02
        aggregates = _aggregates(floor, chain_product, power, capture_product, trajectory)
        record("aggregates", aggregates, inputs=("chain", "power", "trajectory"),
               method="named_reductions",
               resolution={"serve_eta_floor": floor, "names": sorted(aggregates.entries)})

    # ---- P11–P13 ---------------------------------------------------------------------------
    if any(name in wanted for name in ("mass", "flight_power", "reserve")):
        mass, flight_power, reserve, belly_area = _flight_products(scenarios, trajectory,
                                                                  configs["mass"])
        if "mass" in wanted:
            record("mass", mass, inputs=mass.inputs, method="lockstep_fixed_point",
                   resolution={"iterations": mass.iterations,
                               "belly_silhouette_area_m2": round(belly_area, 3),
                               "rectenna_area_source": "belly-on silhouette (beta=0, alpha=0)"})
        if "flight_power" in wanted:
            record("flight_power", flight_power, inputs=flight_power.inputs,
                   method="closed_form_at_rectenna_terminals")
        if "reserve" in wanted:
            record("reserve", reserve, inputs=reserve.inputs,
                   method="glide_and_battery_closed_form")

    manifest = Manifest(
        run_id=spec.run_id or f"run-{int(started)}", spec=_spec_payload(spec),
        engine_sha=engine_sha(), engine_version=ENGINE_VERSION,
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        products=statuses,
        candidate_axes={axis.parameter: tuple(axis.values) for axis in spec.candidates},
        n_candidates=n_candidates, n_samples=n_samples,
        platform={"python": platform.python_version(), "numpy": np.__version__,
                  "seconds": round(time.time() - started, 3)},
        notes=spec.notes)
    return product_types.Products(**built), manifest


def _spec_payload(spec: RunSpec):
    """The spec verbatim, as JSON. Arrays included in full — §6 rules output size out as a design
    constraint, and a manifest that omitted the trajectory would not be the contract it claims."""
    return serialize.to_json_safe({field.name: getattr(spec, field.name)
                                   for field in dataclasses.fields(spec)})


def write(products: product_types.Products, manifest: Manifest, run_dir: str | Path) -> Path:
    """Persist a run. Separate from `evaluate` because an optimiser reads its objective straight off
    the emitted aggregate and never wants a directory (D4)."""
    return artifacts.save_run(run_dir, products, manifest)


def run(spec: RunSpec, run_dir: str | Path):
    """The archival path: evaluate then write. A convenience, not the contract."""
    products, manifest = evaluate(spec)
    write(products, manifest, run_dir)
    return products, manifest


def engine_sha() -> str:
    """The git sha the manifest stamps. "unknown" outside a checkout rather than an exception."""
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                                capture_output=True, text=True, timeout=5)
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"
