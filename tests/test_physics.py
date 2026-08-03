"""Behavioural checks for the engine: invariants, closed-form limits, and internal consistency.

Mostly not regressions against stored values — they pin down monotonicity, symmetry, agreement with
analytic limits, and the properties SPEC.md's invariants claim. Each assertion carries the numbers
behind it in its message, so a failure reads off the detail rather than off a bare comparison.

`tests/test_import_firewall.py` covers I8 and `test_contract.py` covers the consumer contract; this
file is the physics.
"""

from __future__ import annotations

import numpy as np
import pytest

from beamlink import artifacts as A
from beamlink import physics
from beamlink.physics import fidelity, geo, recompute
from beamlink.physics.airframe import flight
from beamlink.physics.airframe import loft as loft_module
from beamlink.physics.airframe import silhouette as silhouette_module
from beamlink.physics.beam_link import aperture as aperture_module
from beamlink.physics.beam_link import atmosphere as atmosphere_module
from beamlink.physics.beam_link import capture as capture_module
from beamlink.physics.beam_link import relations, solver
from beamlink.physics.beam_link.reference import chain_reference, hankel

STANDARD = fidelity.config("standard")


# --- the airframe -------------------------------------------------------------------------


def test_silhouette_behaves_like_a_flat_lifting_body():
    """Belly-on beats edge-on several times over, opposite views agree for a closed body, and area
    falls monotonically as the beam moves off the belly."""
    loft = A.LoftSpec()
    _, _, normals, areas = loft_module.mesh_with_areas(loft, STANDARD)
    belly = float(loft_module.convex_projected_area(
        loft_module.direction_from_beta_alpha(0.0, 0.0), normals, areas))
    edge = float(loft_module.convex_projected_area(
        loft_module.direction_from_beta_alpha(90.0, 90.0), normals, areas))
    assert belly > 3 * edge, f"belly {belly:.0f} vs edge {edge:.0f} m²"

    direction = loft_module.direction_from_beta_alpha(35.0, 20.0)
    forward = float(loft_module.convex_projected_area(direction, normals, areas))
    backward = float(loft_module.convex_projected_area(-direction, normals, areas))
    assert abs(forward - backward) / forward < 0.02, f"{forward:.1f} vs {backward:.1f} m²"

    betas = np.linspace(0.0, 90.0, 19)
    table = silhouette_module.convex_area(loft, STANDARD, betas, np.zeros(19))
    assert np.all(np.diff(table) <= 1e-6), f"not monotone: {np.round(table, 1)}"


def test_the_raster_removes_the_convex_over_count():
    """Self-occlusion is real: side-on at low elevation the winglets and the far wing overlap in
    projection, and the z-buffer sees less area than the convex identity claims."""
    loft = A.LoftSpec()
    convex = float(silhouette_module.convex_area(loft, STANDARD, 88.0, 90.0))
    raster = float(silhouette_module.silhouette_at(loft, 88.0, 90.0, STANDARD)["area"])
    assert raster < 0.93 * convex, f"raster {raster:.0f} vs convex {convex:.0f} m²"


def test_the_silhouette_frame_is_a_deterministic_right_handed_gauge():
    """The frame is orthonormal, right-handed, and a function of the arrival angle alone.

    D13: the SIGN of x̂ is a gauge choice, not a continuous one — the narrow principal axis rotates
    ~90° over a pass (fore-aft belly-on, near-vertical at the horizon), so it crosses the pin's
    fallback thresholds and the sign can flip there. What must hold is that the frame is well-formed
    and reproducible; that nothing physical depends on the sign is the next test.
    """
    products, _, _ = _run(A.ScenarioSpec(aperture=A.ApertureSpec(diameter_m=100.0)), n_samples=21)
    frames = products.silhouette.view_basis_body[0]
    assert np.allclose(np.linalg.norm(frames, axis=2), 1.0, atol=1e-9), "axes must be unit"
    for i, j in ((0, 1), (0, 2), (1, 2)):
        assert np.allclose(np.einsum("ni,ni->n", frames[:, i], frames[:, j]), 0.0, atol=1e-9)
    handedness = np.einsum("ni,ni->n", np.cross(frames[:, 0], frames[:, 1]), frames[:, 2])
    assert np.allclose(handedness, 1.0, atol=1e-9), f"not right-handed: {handedness.min():.6f}"

    # ẑ is the propagation direction, so it must point AWAY from the emitter in the body frame:
    # its belly component is positive whenever the beam arrives from below.
    assert np.all(frames[:, 2, 2] < 0.0), "propagation must point from the emitter to the aircraft"

    loft = products.silhouette
    repeat = silhouette_module.silhouette_at(A.LoftSpec(), 30.0, 180.0, STANDARD)["basis"]
    again = silhouette_module.silhouette_at(A.LoftSpec(), 30.0, 180.0, STANDARD)["basis"]
    assert np.array_equal(repeat, again), "the same arrival angle must give the same frame"


def test_capture_is_invariant_to_the_frame_gauge():
    """Nothing physical may depend on the sign of x̂ (D13) — including the pointing error.

    The aperture is an ellipse with even illumination, so the FIELD is gauge-invariant outright. The
    pointing offset is not: a signed offset along an arbitrarily-signed axis is one of two directions
    that differ by ~7 pp of capture on this loft, because the projection is not symmetric about its
    narrow axis. The engine therefore takes the direction from the projection's own third moment,
    which flips with the gauge and so describes the same physical offset either way.

    Mirroring the element cloud is the direct test: capture must not move.
    """
    loft = A.LoftSpec()
    raster = silhouette_module.silhouette_at(loft, 60.0, 180.0, STANDARD)
    count = raster["element_x"].size
    offsets = np.stack([raster["element_x"], raster["element_y"]], axis=1)[None]
    elements = dict(projected_area_m2=raster["projected_area"][None],
                    incidence_cos=raster["incidence"][None],
                    valid=np.ones((1, count), dtype=bool))
    slant = np.array([11_000.0 / np.cos(np.radians(60.0))])
    state = aperture_module.ApertureState([70.7], [35.4], [1.0], slant,
                                          [float(relations.wavelength(5.8e9))])
    magnitude, captures = 2.0, []
    for sign in (1.0, -1.0):
        mirrored = offsets * np.array([sign, 1.0])
        skew = float(np.sum(raster["projected_area"] * (sign * raster["element_x"]) ** 3))
        signed = magnitude * (-1.0 if skew >= 0.0 else 1.0)
        context = solver.SolveContext(
            state=state, receiver_range_m=slant, element_offsets_m=mirrored,
            element_valid=elements["valid"],
            receiver_half_extent_m=raster["half_extent"][None],
            pointing_offset_m=np.array([signed]))
        captures.append(float(capture_module.overlap(context, elements, 0.90, STANDARD)[1][0]))
    assert abs(captures[0] - captures[1]) < 1e-6, (
        f"capture must not depend on the frame's gauge: {captures[0]:.6f} vs {captures[1]:.6f}")


def test_the_pointing_offset_is_the_worse_of_the_two_directions():
    """D13's other half: the chosen direction has to be the one that costs more, or "worst case for a
    given magnitude" is just a label."""
    loft = A.LoftSpec()
    raster = silhouette_module.silhouette_at(loft, 60.0, 180.0, STANDARD)
    count = raster["element_x"].size
    offsets = np.stack([raster["element_x"], raster["element_y"]], axis=1)[None]
    elements = dict(projected_area_m2=raster["projected_area"][None],
                    incidence_cos=raster["incidence"][None],
                    valid=np.ones((1, count), dtype=bool))
    slant = np.array([11_000.0 / np.cos(np.radians(60.0))])
    state = aperture_module.ApertureState([70.7], [35.4], [1.0], slant,
                                          [float(relations.wavelength(5.8e9))])

    def capture_at(signed_offset):
        context = solver.SolveContext(
            state=state, receiver_range_m=slant, element_offsets_m=offsets,
            element_valid=elements["valid"],
            receiver_half_extent_m=raster["half_extent"][None],
            pointing_offset_m=np.array([signed_offset]))
        return float(capture_module.overlap(context, elements, 0.90, STANDARD)[1][0])

    skew = float(np.sum(raster["projected_area"] * raster["element_x"] ** 3))
    chosen = -2.0 if skew >= 0.0 else 2.0
    assert capture_at(chosen) <= capture_at(-chosen) + 1e-9, (
        f"the third-moment rule picked the kinder direction: {capture_at(chosen):.5f} > "
        f"{capture_at(-chosen):.5f} (skew {skew:+.3g} m³)")


def test_the_mass_cascade_converges_lockstep_and_reproduces_the_pre_spec_anchor():
    """A fixed iteration count reaches the same fixed point a convergence test would, and MTOW lands
    where the pre-SPEC scalar loop put it (~80 t for a 40 t payload)."""
    airframe = A.AirframeSpec()
    rectenna = flight.rectenna_mass_kg(2095.0, 1.0)
    coarse = flight.mass_cascade(airframe, np.array([rectenna]), np.array([250.0]), 10)
    fine = flight.mass_cascade(airframe, np.array([rectenna]), np.array([250.0]), 80)
    assert fine["residual_kg"][0] < 1e-6, f"residual {fine['residual_kg'][0]:.3g} kg"
    assert abs(coarse["mtow_kg"][0] - fine["mtow_kg"][0]) < 1.0, "10 iterations should already be 1 kg"
    assert 75e3 < fine["mtow_kg"][0] < 85e3, f"MTOW {fine['mtow_kg'][0] / 1e3:.1f} t"


# --- the atmosphere -----------------------------------------------------------------------


def test_gaseous_loss_is_all_below_cruise_altitude():
    """The stratosphere is dry, so an 11 km and a 20 km zenith path cost the same."""
    ground = geo.point(0.0, 0.0, 0.0)
    clear = A.weather("clear")
    to_11 = sum(atmosphere_module.path_loss_db(30.0, ground, geo.point(0., 0., 11_000.), clear))
    to_20 = sum(atmosphere_module.path_loss_db(30.0, ground, geo.point(0., 0., 20_000.), clear))
    assert abs(float(to_20) - float(to_11)) < 0.05, f"{float(to_11):.3f} vs {float(to_20):.3f} dB"


def test_a_slant_path_costs_the_secant_of_its_elevation():
    ground = geo.point(0.0, 0.0, 0.0)
    clear = A.weather("clear")
    zenith = float(sum(atmosphere_module.path_loss_db(30.0, ground,
                                                     geo.point(0., 0., 11_000.), clear)))
    downrange = geo.downrange_for_elevation(30.0, 11_000.0)
    slant = float(sum(atmosphere_module.path_loss_db(30.0, ground,
                                                    geo.point(downrange, 0., 11_000.), clear)))
    assert abs(slant / zenith - 2.0) < 0.06, f"{slant:.3f}/{zenith:.3f} = {slant / zenith:.2f}"


def test_rain_hammers_ka_band_and_spares_c_band():
    """At the 30° reference geometry — the claim is about the BAND, so it needs a stated geometry:
    the same weather over a 150 km slant costs 5.8 GHz a factor of four as well, because the wet path
    is six times longer."""
    downrange = geo.downrange_for_elevation(30.0, 11_000.0)
    ground, plane = geo.point(0., 0., 0.), geo.point(downrange, 0., 11_000.)
    rain = A.weather("heavy_rain")
    low = float(relations.db_to_fraction(
        sum(atmosphere_module.path_loss_db(5.8, ground, plane, rain))))
    high = float(relations.db_to_fraction(
        sum(atmosphere_module.path_loss_db(30.0, ground, plane, rain))))
    assert low > 0.7 and high < 0.05, f"5.8 → {low:.3f}, 30 → {high:.3f}"


# --- curvature and the per-leg elevation convention (D1) ----------------------------------


def test_earth_curvature_is_in_the_geometry():
    ground = geo.point(0.0, 0.0, 0.0)
    ranges = np.array([20e3, 100e3, 200e3, 300e3, 400e3])
    _, elevation = geo.link(ground, geo.point(ranges, 0.0, 11_000.0))
    assert np.all(np.diff(elevation) < 0), f"{np.round(elevation, 1)} deg"
    assert geo.horizon_blocked(ground, geo.point(600e3, 0., 11e3))
    assert not geo.horizon_blocked(ground, geo.point(300e3, 0., 11e3))


def test_elevation_is_ground_referenced_per_leg():
    """D1: ground→aircraft is positive and +90° overhead; relay→aircraft is correctly NEGATIVE,
    the aircraft being below a relay's horizon. Both come from one definition applied per leg."""
    direct = _run(A.ScenarioSpec(aperture=A.ApertureSpec(diameter_m=100.0)), n_samples=5)
    overhead = int(np.argmax(direct[0].geometry.elevation_deg[0, 0]))
    assert abs(direct[0].geometry.elevation_deg[0, 0, overhead] - 90.0) < 1e-6, "overhead is +90°"
    assert np.all(direct[0].geometry.elevation_deg[0, 0] > 0.0), "a ground station looks up"

    relayed = _run(A.ScenarioSpec(relay=A.RelaySpec(position_m=(0.0, 0.0, 20_000.0),
                                                   mode="multiplier", f_in_ghz=5.8,
                                                   f_out_ghz=17.4, aperture_diam_m=40.0)),
                   n_samples=5)
    geometry = relayed[0].geometry
    assert geometry.leg_names == ("station->relay", "relay->aircraft")
    assert geometry.elevation_deg[0, 0, 0] > 0.0, "the station looks UP at the relay"
    assert np.all(geometry.elevation_deg[0, 1] < 0.0), (
        "a relay at 20 km looks DOWN at an aircraft at 11 km: "
        f"{np.round(geometry.elevation_deg[0, 1], 1)}")


# --- the one field solve ------------------------------------------------------------------


def test_the_field_reproduces_the_analytic_airy_encircled_energy():
    """A round uniform aperture on a disc must reproduce 1 − J0(2τ)² − J1(2τ)² exactly, sidelobe
    spill and all — i.e. the solve really is the Fresnel transform of the aperture."""
    wavelength = float(relations.wavelength(30e9))
    distance, emitter_radius, receiver_radius = 200e3, 40.0, 15.0
    state = aperture_module.ApertureState([emitter_radius], [emitter_radius], [0.0], [distance],
                                          [wavelength])
    grid = np.linspace(-receiver_radius, receiver_radius, 401)
    pitch = grid[1] - grid[0]
    intensity, _, _, _, _ = aperture_module.plane_intensity(state, np.array([distance]),
                                                            grid[None, :], grid[None, :], STANDARD)
    x_grid, y_grid = np.meshgrid(grid, grid, indexing="ij")
    inside = x_grid ** 2 + y_grid ** 2 <= receiver_radius ** 2
    numeric = float(intensity[0][inside].sum() * pitch * pitch)
    analytic = float(aperture_module.airy_encircled_energy(receiver_radius, emitter_radius,
                                                           wavelength, distance))
    assert abs(numeric - analytic) < 3e-3, f"numeric {numeric:.5f} vs Airy {analytic:.5f}"


def test_the_peak_matches_the_analytic_uniform_disc():
    """π·a²/(λz)² on axis, which is what the density cap is enforced against."""
    wavelength = float(relations.wavelength(30e9))
    distance, radius = 200e3, 40.0
    state = aperture_module.ApertureState([radius], [radius], [0.0], [distance], [wavelength])
    peak, _, _, _, _ = aperture_module.plane_intensity(state, np.array([distance]),
                                                       np.zeros((1, 1)), np.zeros((1, 1)), STANDARD)
    analytic = np.pi * radius ** 2 / (wavelength * distance) ** 2
    assert abs(peak[0, 0, 0] / analytic - 1.0) < 0.01, f"{peak[0, 0, 0]:.4e} vs {analytic:.4e}"


def test_anisotropic_sampling_costs_the_same_at_any_aperture_aspect():
    """The reason the one solve is affordable: per-axis pitch makes the aperture grid aspect-free.

    A common pitch would need the minor axis's fine spacing across the major axis's full length, which
    is where the pre-SPEC 512² grid with ~450 illuminated samples came from.
    """
    wavelength = float(relations.wavelength(5.8e9))
    sizes = []
    for aspect in (1.0, 4.0, 8.0):
        radius = 50.0
        state = aperture_module.ApertureState([radius * np.sqrt(aspect)], [radius / np.sqrt(aspect)],
                                              [1.0], [11_000.0], [wavelength])
        _, grid_n, _, _ = aperture_module.required_pitch(state, np.array([11_000.0]),
                                                         np.array([[45.0, 45.0]]), STANDARD)
        sizes.append(int(grid_n.max()))
    assert max(sizes) <= 2 * min(sizes), f"aperture grid grew with aspect: {sizes}"


def test_unresolvable_regions_are_reported_not_filled():
    """Sub-Fresnel planes come back INVALID with a stated reason, and the intensity there is zero
    rather than a geometric fill that makes the picture whole (§5 P5, §8 item 14)."""
    products, manifest, _ = _run(
        A.ScenarioSpec(aperture=A.ApertureSpec(diameter_m=100.0)), n_samples=3,
        field_sampling=A.default_axial_planes(n_axial=40, n_radial=41))
    sample_set = products.field.sets["axial_major"]
    near_aperture = sample_set.points_m[0, 1, :, 2] < 100.0
    assert near_aperture.any(), "the cut should reach close to the aperture"
    assert not sample_set.valid[0, 1][near_aperture].any(), "near-aperture planes must be invalid"
    assert np.all(sample_set.intensity_w_m2[0, 1][near_aperture] == 0.0)
    assert "sub_fresnel" in products.field.unresolved_reason["axial_major"]


def test_one_solve_serves_both_the_overlap_and_the_side_view():
    """I1: the field at the aircraft plane read off the SIDE VIEW agrees with the field the capture
    integral used, because they are the same solve sampled twice.

    This is the check the pre-SPEC code could not pass: two independent solves of the same field
    agreed on peak intensity to 15 %, held together by a tolerance chosen when the comparison was
    written (§1).
    """
    products, _, _ = _run(
        A.ScenarioSpec(aperture=A.ApertureSpec(diameter_m=100.0, adaptive_taper=False, taper_p=1.0)),
        n_samples=3,
        field_sampling=(A.FieldSampling(kind="transverse_plane", name="at_receiver",
                                        z_fractions=(1.0,), n_transverse=(1, 1),
                                        extent_m=(1e-6, 1e-6)),))
    on_axis = products.field.sets["at_receiver"].intensity_w_m2[0, 1, 0]
    radiated = products.power.radiated_w[0, -1, 1]
    peak_from_capture = products.capture.peak_areal_density_per_m2[0, 1] * radiated
    assert abs(on_axis / peak_from_capture - 1.0) < 1e-9, (
        f"the side view's on-axis value {on_axis:.6e} must BE the capture peak "
        f"{peak_from_capture:.6e} — one solve, sampled twice")


# --- capture ------------------------------------------------------------------------------


def test_the_adaptive_taper_beats_every_fixed_taper():
    base = A.ApertureSpec(diameter_m=100.0)
    adaptive, _, _ = _run(A.ScenarioSpec(aperture=base), n_samples=5)
    best_fixed = max(
        float(_run(A.ScenarioSpec(aperture=A.ApertureSpec(
            diameter_m=100.0, adaptive_taper=False, taper_p=taper)), n_samples=5
        )[0].capture.incidence_weighted[0, 2])
        for taper in (0.0, 1.0, 2.0, 4.0))
    achieved = float(adaptive.capture.incidence_weighted[0, 2])
    assert achieved >= best_fixed - 3e-3, f"adaptive {achieved:.4f} vs best fixed {best_fixed:.4f}"
    assert 0.0 <= adaptive.aperture.taper_exponent[0, 2] <= 4.0


def test_a_taper_lifts_capture_on_a_main_lobe_sized_receiver_and_lowers_the_peak():
    """Tapering moves power out of the sidelobes into the main lobe, and widens the main lobe — which
    is the lever the density cap pulls on."""
    wavelength = float(relations.wavelength(30e9))
    distance, radius = 200e3, 40.0
    first_null = 0.610 * wavelength * distance / radius
    grid = np.linspace(-first_null, first_null, 201)
    pitch = grid[1] - grid[0]
    x_grid, y_grid = np.meshgrid(grid, grid, indexing="ij")
    inside = x_grid ** 2 + y_grid ** 2 <= first_null ** 2

    def captured_and_peak(taper):
        state = aperture_module.ApertureState([radius], [radius], [taper], [distance], [wavelength])
        intensity, _, _, _, _ = aperture_module.plane_intensity(
            state, np.array([distance]), grid[None, :], grid[None, :], STANDARD)
        peak, _, _, _, _ = aperture_module.plane_intensity(
            state, np.array([distance]), np.zeros((1, 1)), np.zeros((1, 1)), STANDARD)
        return float(intensity[0][inside].sum() * pitch * pitch), float(peak[0, 0, 0])

    uniform_capture, uniform_peak = captured_and_peak(0.0)
    tapered_capture, tapered_peak = captured_and_peak(1.5)
    assert tapered_capture > uniform_capture, f"{uniform_capture:.3f} → {tapered_capture:.3f}"
    assert tapered_peak < uniform_peak, f"{uniform_peak:.3e} → {tapered_peak:.3e} /m²"


def test_the_closed_form_tier_is_biased_high_and_one_sided():
    """§6's measured claim, asserted: the Goubau closed form assumes the optimal illumination the
    overlap integral has to earn, so it can only over-state capture."""
    scenario = A.ScenarioSpec(aperture=A.ApertureSpec(diameter_m=100.0))
    overlap, _, _ = _run(scenario, n_samples=9)
    closed, _, _ = _run(scenario, n_samples=9, tier="coarse")
    assert closed.capture.method == "closed_form_goubau"
    assert closed.capture.inputs == ("geometry", "silhouette"), "it does not touch the field"
    difference = closed.capture.geometric - overlap.capture.geometric
    assert np.all(difference > -0.02), f"closed form dipped below the overlap: {np.round(difference, 3)}"


def test_the_density_cap_is_held_by_defocusing():
    """More delivered power forces a wider spot, because the only way to stay under the cap is to
    defocus — and the resulting density lands ON the cap, not under it by luck."""
    scenario = A.ScenarioSpec(aperture=A.ApertureSpec(diameter_m=100.0))
    low, _, _ = _run(scenario, n_samples=5, delivered_w=1e6)
    high, _, _ = _run(scenario, n_samples=5, delivered_w=60e6)
    cap = A.ReceiverSpec().max_power_density_w_m2
    assert (high.capture.equivalent_disc_radius_m[0, 2]
            > low.capture.equivalent_disc_radius_m[0, 2]), "the spot must spread"
    assert high.aperture.focus_range_m[0, 2] < 0.999 * high.geometry.slant_m[0, -1, 2], "defocused"
    assert np.all(high.capture.skin_power_density_w_m2 <= cap * 1.05), (
        f"cap breached: {high.capture.skin_power_density_w_m2.max() / 1e4:.1f} W/cm²")


def test_an_unmeetable_cap_is_flagged_rather_than_silently_exceeded():
    """Defocus cannot beat an asymptote, and when it cannot the flag is set rather than the number
    quietly returned.

    The asymptote is worth stating because it is why this case exists at all. Once the spot is wider
    than the aircraft, capture falls in step with the peak, so

        density = delivered / (capture · rectenna) · peak  →  delivered / (area · rectenna)

    which no amount of defocusing changes. For this loft that floor is reached at about
    2.5e5 W/m² × 2073 m² × 0.76 ≈ 390 MW delivered, so 400 MW sits within 2 % of it — which is why
    the pre-SPEC engine measured "~1.30× the cap at every focus" on very nearly this geometry. The
    test uses 800 MW so it is testing the mechanism rather than the edge of it.

    By decision this SURFACES and does not block, and it deliberately does not touch feasibility.
    """
    scenario = A.ScenarioSpec(
        relay=A.RelaySpec(position_m=(0.0, 0.0, 20_000.0), mode="multiplier", f_in_ghz=5.8,
                          f_out_ghz=17.4, aperture_diam_m=40.0))
    products, _, _ = _run(scenario, delivered_w=800e6,
                          trajectory=A.TrajectorySpec.straight_overflight(1.0, 11_000.0, 250.0,
                                                                         n_samples=3))
    assert products.capture.density_cap_exceeded.all(), (
        f"800 MW cannot comply at any defocus; densities "
        f"{np.round(products.capture.skin_power_density_w_m2[0] / 1e4, 1)} W/cm²")
    assert products.chain.feasible.all(), "an over-cap geometry is still an operating case"

    # And the compliant case is not flagged, which is the other half of the claim.
    modest, _, _ = _run(scenario, delivered_w=10e6,
                        trajectory=A.TrajectorySpec.straight_overflight(1.0, 11_000.0, 250.0,
                                                                       n_samples=3))
    assert not modest.capture.density_cap_exceeded.any()


def test_pointing_error_barely_costs_a_contained_spot():
    """A tiny spot well inside a large rectenna loses almost nothing to a 2 m offset."""
    perfect, _, _ = _run(A.ScenarioSpec(aperture=A.ApertureSpec(diameter_m=100.0,
                                                               pointing_error_m=0.0)), n_samples=3)
    offset, _, _ = _run(A.ScenarioSpec(aperture=A.ApertureSpec(diameter_m=100.0,
                                                              pointing_error_m=2.0)), n_samples=3)
    overhead = 1
    loss = abs(offset.chain.eta_end_to_end[0, overhead]
               - perfect.chain.eta_end_to_end[0, overhead]) / perfect.chain.eta_end_to_end[0, overhead]
    assert loss < 0.02, f"lost {100 * loss:.2f}% to a 2 m pointing error"


def test_a_shaped_aperture_never_captures_less_than_a_round_one():
    low_elevation = A.TrajectorySpec.straight_overflight(140e3, 11_000.0, 250.0, n_samples=3)
    shaped, _, _ = _run(A.ScenarioSpec(aperture=A.ApertureSpec(diameter_m=100.0, shaped=True)),
                        trajectory=low_elevation)
    round_, _, _ = _run(A.ScenarioSpec(aperture=A.ApertureSpec(diameter_m=100.0, shaped=False)),
                        trajectory=low_elevation)
    assert np.all(shaped.capture.geometric >= round_.capture.geometric - 1e-6), (
        f"shaped {np.round(shaped.capture.geometric[0], 4)} vs "
        f"round {np.round(round_.capture.geometric[0], 4)}")


# --- the chain, the pass, and the two definitions of §1 -----------------------------------


def test_the_pass_profile_peaks_overhead_and_is_near_symmetric():
    products, _, _ = _run(A.ScenarioSpec(aperture=A.ApertureSpec(diameter_m=100.0)), n_samples=41)
    delivered = products.power.rectenna_terminals_w[0]
    along = products.trajectory.positions_m[:, 0]
    peak = int(np.argmax(delivered))
    assert abs(along[peak]) < 10e3, f"peak at {along[peak] / 1e3:.1f} km"
    left, right = delivered[along < 0], delivered[along > 0][::-1]
    count = min(left.size, right.size)
    asymmetry = np.max(np.abs(left[-count:] - right[:count]) / np.maximum(right[:count], 1e-9))
    assert asymmetry < 0.08, f"fore/aft asymmetry {100 * asymmetry:.1f}%"


def test_the_two_pass_efficiencies_of_section_one_are_both_named_and_do_diverge():
    """The whole point of P9.

    §12 measured econ's `energy_eta` against the figure's panel-D curve at 0.32× in a storm at 400 km
    station spacing. Both reductions are emitted now, both carry their definition, and this asserts
    that they are genuinely different numbers on that scenario rather than two names for one.
    """
    storm = A.ScenarioSpec(aperture=A.ApertureSpec(diameter_m=100.0), weather=A.weather("storm"))
    products, _, _ = _run(storm, trajectory=A.TrajectorySpec.straight_overflight(
        200e3, 11_000.0, 250.0, n_samples=41))
    served = products.aggregates.entries["pass_eta_mean_served"]
    whole = products.aggregates.entries["pass_eta_mean_whole_span"]
    assert served.definition.serve_policy == "eta_floor"
    assert whole.definition.serve_policy == "always"
    assert served.definition.serve_eta_floor == pytest.approx(0.02)
    ratio = float(whole.value[0] / served.value[0])
    assert ratio < 0.9, (
        f"in a storm over ±200 km the two definitions must differ: served "
        f"{served.value[0]:.4f} vs whole-span {whole.value[0]:.4f} (ratio {ratio:.2f})")
    assert served.served.sum() < served.served.size, "the serve policy must actually exclude samples"


def test_the_serve_floor_moves_the_headline_and_is_recorded_with_it():
    """§8 item 20: the floor is not derived from anything and it sets the headline. It is now a
    declared input, and the number it produced carries it."""
    storm = A.ScenarioSpec(aperture=A.ApertureSpec(diameter_m=100.0), weather=A.weather("storm"))
    trajectory = A.TrajectorySpec.straight_overflight(200e3, 11_000.0, 250.0, n_samples=41)
    values = {}
    for floor in (0.0, 0.10):
        products, _, _ = _run(storm, trajectory=trajectory, serve_eta_floor=floor)
        entry = products.aggregates.entries["pass_eta_mean_served"]
        assert entry.definition.serve_eta_floor == pytest.approx(floor)
        values[floor] = float(entry.value[0])
    assert values[0.10] > 1.5 * values[0.0], (
        f"raising the floor discards the worst samples: {values[0.0]:.4f} → {values[0.10]:.4f}")


def test_the_relay_beats_direct_through_a_storm():
    """The relay's emitter is above the weather, which is its physical claim to keep."""
    trajectory = A.TrajectorySpec.straight_overflight(100e3, 11_000.0, 250.0, n_samples=17)
    direct, _, _ = _run(A.ScenarioSpec(frequency_ghz=5.8, weather=A.weather("storm"),
                                       aperture=A.ApertureSpec(diameter_m=100.0)),
                        trajectory=trajectory)
    relay, _, _ = _run(A.ScenarioSpec(
        weather=A.weather("storm"),
        relay=A.RelaySpec(position_m=(0.0, 0.0, 20_000.0), mode="multiplier", f_in_ghz=5.8,
                          f_out_ghz=17.4, aperture_diam_m=40.0)), trajectory=trajectory)
    direct_harmonic = float(direct.aggregates.entries["harmonic_mean_eta"].value[0])
    relay_harmonic = float(relay.aggregates.entries["harmonic_mean_eta"].value[0])
    assert relay_harmonic > direct_harmonic, (
        f"relay {relay_harmonic:.4f} vs direct {direct_harmonic:.4f} on continuous delivery")


# --- the receiver's shadow ----------------------------------------------------------------


def test_the_aircraft_casts_a_shadow_that_washes_out_to_one_minus_capture():
    """100 % capture and no reflection says two things, at the shadow's two ends: the axis goes dark
    right behind the aircraft, and far behind — once diffraction has washed the shadow out — the beam
    has its own shape again carrying (1 − capture) of its power, NOT full brightness."""
    products, _, _ = _run(
        A.ScenarioSpec(aperture=A.ApertureSpec(diameter_m=100.0)),
        trajectory=A.TrajectorySpec.straight_overflight(1.0, 11_000.0, 250.0, n_samples=3),
        field_sampling=(A.FieldSampling(kind="axial_plane", name="shadowed", axis="major",
                                        n_axial=120, n_radial=81, z_fractions=(0.5, 6.0),
                                        occluders="receiver"),))
    sample_set = products.field.sets["shadowed"]
    transmission = sample_set.occluder_transmission[0, 1]
    ranges = sample_set.points_m[0, 1, :, 2]
    radial = sample_set.points_m[0, 1, :, 0]
    slant = products.geometry.slant_m[0, -1, 1]
    on_axis = np.abs(radial) < 1.0

    just_behind = on_axis & (ranges > slant * 1.02) & (ranges < slant * 1.2)
    assert transmission[just_behind].max() < 0.1, (
        f"the axis must go dark right behind the aircraft: {transmission[just_behind].max():.3f}")
    far_behind = on_axis & (ranges > slant * 5.0)
    survives = float(np.median(transmission[far_behind]))
    expected = 1.0 - float(products.capture.geometric[0, 1])
    assert abs(survives - expected) < 0.25 * max(expected, 0.02), (
        f"far behind, {100 * survives:.2f}% survives against 1 − capture = {100 * expected:.2f}%")


# --- the independent Hankel reference (§2) ------------------------------------------------


def test_the_hankel_propagator_conserves_power_and_follows_the_spreading_law():
    wavelength = float(relations.wavelength(34.8e9))
    grid = hankel.HankelGrid(30 * 82.5, 512)
    waist = 20.0
    rayleigh = np.pi * waist ** 2 / wavelength
    start = np.exp(-(grid.radius / waist) ** 2).astype(complex)
    for distance in (0.25 * rayleigh, rayleigh, 3 * rayleigh):
        out = grid.propagate(start, distance, wavelength)
        assert abs(grid.power(out) / grid.power(start) - 1.0) < 1e-9
        intensity = np.abs(out) ** 2 / np.abs(out[0]) ** 2
        fitted = np.interp(np.exp(-2.0), intensity[::-1], grid.radius[::-1])
        expected = waist * np.sqrt(1 + (distance / rayleigh) ** 2)
        assert abs(fitted / expected - 1.0) < 0.04, f"{fitted:.3f} vs {expected:.3f} m"


@pytest.mark.parametrize("tau_target", [0.5, 1.0, 2.0, 3.0])
def test_the_true_truncated_field_tracks_goubau_from_below(tau_target):
    """Truncation can only lose power the closed form assumes arrives, so the SIGN of the gap is
    physics rather than tolerance. This is what makes the sizing rule the engine uses checked rather
    than merely self-consistent."""
    wavelength = float(relations.wavelength(34.8e9))
    length, emitter_radius = 300e3, 82.5
    receiver_radius = tau_target * wavelength * length / (np.pi * emitter_radius)
    grid, _, planes, tau, launched = hankel.hop_field(emitter_radius, receiver_radius, length,
                                                     wavelength, n_planes=2, n_points=1024)
    captured = grid.power_within(planes[-1], receiver_radius) / launched
    goubau = float(relations.collection_efficiency(tau))
    assert captured <= goubau + 1e-6, f"field {captured:.4f} above Goubau {goubau:.4f}"
    assert abs(captured - goubau) < 0.03, f"field {captured:.4f} vs Goubau {goubau:.4f}"


def test_a_single_hop_reference_chain_matches_the_closed_form():
    result = chain_reference.run_chain(34.8e9, 300e3, [82.5, 19.943], [0.0, 300e3], verbose=False)
    assert result["total_true"] <= result["total_goubau"] + 1e-9
    assert abs(result["total_true"] - result["total_goubau"]) < 0.03, (
        f"true {result['total_true']:.4%} vs Goubau {result['total_goubau']:.4%}")


def test_chaining_relays_compounds_truncation_loss_the_closed_form_cannot_see():
    """A measured finding, pinned so it is not rediscovered as a bug.

    Three relays each sized for τ=2 give a Goubau product of ~95 %, and the true field delivers ~54 %.
    The closed form assumes the MATCHED mode arrives at every element; after the first truncation the
    field is Airy-like, so each subsequent rim cuts sidelobes the closed form already counted as
    captured. The gap is therefore physics and it compounds with hop count — which is a real argument
    against long relay chains, and not something a per-hop efficiency product can express.
    """
    chained = chain_reference.run_chain(34.8e9, 300e3, [82.5, 20.0, 20.5, 20.5, 20.0],
                                       [0.0, 75e3, 150e3, 225e3, 300e3], verbose=False)
    assert chained["total_true"] <= chained["total_goubau"] + 1e-9, "truncation can only lose"
    assert 0.4 < chained["total_true"] < 0.7, f"true {chained['total_true']:.4%}"
    assert chained["total_goubau"] > 0.9, f"Goubau product {chained['total_goubau']:.4%}"


# --- I6 and I9 ----------------------------------------------------------------------------


def test_every_derived_product_recomputes_from_its_inputs():
    """I6, on a run with both architectures' worth of derivations exercised."""
    products, manifest, spec = _run(A.ScenarioSpec(aperture=A.ApertureSpec(diameter_m=100.0)),
                                    n_samples=13)
    checks = recompute.check_all(products, manifest, spec)
    failures = [str(check) for check in checks if not check.passed]
    assert not failures, "\n".join(failures)
    assert len(checks) >= 15, f"only {len(checks)} checks ran"


def test_the_relay_architecture_recomputes_too():
    products, manifest, spec = _run(
        A.ScenarioSpec(relay=A.RelaySpec(position_m=(0.0, 0.0, 20_000.0), mode="multiplier",
                                         f_in_ghz=5.8, f_out_ghz=17.4, aperture_diam_m=40.0)),
        n_samples=9)
    failures = [str(c) for c in recompute.check_all(products, manifest, spec) if not c.passed]
    assert not failures, "\n".join(failures)


def test_a_batch_of_candidates_gives_the_same_answer_as_the_runs_it_batches():
    """I9: a batched call is the normal call and a single scenario is a batch of one — which is only
    a legitimate claim if the batch is numerically the same as the runs it replaces."""
    trajectory = A.TrajectorySpec.straight_overflight(100e3, 11_000.0, 250.0, n_samples=9)
    diameters = (80.0, 100.0, 140.0)
    batched = A.RunSpec(
        scenario=A.ScenarioSpec(aperture=A.ApertureSpec(diameter_m=diameters[0])),
        trajectory=trajectory,
        candidates=(A.CandidateAxis(parameter="aperture.diameter_m", values=diameters),),
        run_id="batched")
    batch_products, _ = physics.evaluate(batched)
    assert batch_products.chain.eta_end_to_end.shape == (3, 9)
    for index, diameter in enumerate(diameters):
        single, _, _ = _run(A.ScenarioSpec(aperture=A.ApertureSpec(diameter_m=diameter)),
                            trajectory=trajectory)
        assert np.allclose(batch_products.chain.eta_end_to_end[index],
                           single.chain.eta_end_to_end[0], rtol=1e-12), (
            f"candidate {index} (Ø{diameter} m) differs from the same scenario run alone")


def test_a_candidate_axis_may_not_change_the_architecture():
    """D7: that is two runs and a study, not one run."""
    with pytest.raises(ValueError, match="two runs and a study"):
        physics.evaluate(A.RunSpec(
            candidates=(A.CandidateAxis(
                parameter="relay",
                values=(None, A.RelaySpec(position_m=(0.0, 0.0, 20_000.0)))),),
            trajectory=A.TrajectorySpec.straight_overflight(10e3, 11_000.0, 250.0, n_samples=3)))


def test_a_fidelity_override_may_not_split_the_field_solve():
    """D8: `field` and `capture` share one solve, so they share a tier by construction."""
    with pytest.raises(ValueError, match="split one field solve"):
        fidelity.resolve("standard", {"capture": "fast"})
    resolved = fidelity.resolve("standard", {"silhouette": "fast", "mass": "full"})
    assert resolved["silhouette"].tier == "fast" and resolved["mass"].tier == "full"
    assert resolved["field"].tier == "standard"


def test_the_defocus_floor_is_closed_form_and_representable():
    """D9: the shortest representable focus, without a try/except back-off loop.

    The pre-SPEC bracket of slant/8 is genuinely out of reach for a large aperture at short range,
    which is what the back-off existed for; the closed form has to land above it and still be
    evaluable.
    """
    wavelength = float(relations.wavelength(5.8e9))
    slant, radius = 11_000.0, 200.0
    state = aperture_module.ApertureState([radius], [radius], [0.0], [slant], [wavelength])
    extent = np.array([45.0])
    floor = aperture_module.defocus_floor(state, np.array([slant]), extent, STANDARD)
    assert slant / 8.0 <= floor[0] <= slant, f"floor {floor[0] / 1e3:.2f} km of {slant / 1e3:.0f} km"
    _, _, resolvable, reason = aperture_module.required_pitch(state.with_focus(floor),
                                                              np.array([slant]), extent, STANDARD)
    assert resolvable[0], f"the closed-form floor must be evaluable, got reason {reason[0]}"


# --- helper -------------------------------------------------------------------------------


def _run(scenario, n_samples=9, trajectory=None, tier="standard", delivered_w=10e6,
         field_sampling=(), serve_eta_floor=0.02):
    """Evaluate one scenario and hand back (products, manifest, spec).

    The tests go through the engine's own entry point rather than reaching into its internals, which
    is the same access a driver has — so a test cannot pass by using a path a caller could not.
    """
    if trajectory is None:
        trajectory = A.TrajectorySpec.straight_overflight(100e3, 11_000.0, 250.0,
                                                          n_samples=n_samples)
    requests = []
    for name in A.PRODUCTS:
        if name == "field":
            requests.append(A.ProductRequest(name=name, sampling=tuple(field_sampling)))
        elif name == "aggregates":
            requests.append(A.ProductRequest(name=name, serve_eta_floor=serve_eta_floor))
        else:
            requests.append(A.ProductRequest(name=name))
    spec = A.RunSpec(scenario=scenario, trajectory=trajectory, products=tuple(requests),
                     fidelity=A.FidelitySpec(tier=tier),
                     power=A.PowerSpec(delivered_target_w=delivered_w), run_id="test")
    products, manifest = physics.evaluate(spec)
    return products, manifest, spec
