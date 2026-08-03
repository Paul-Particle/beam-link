"""The consumer contract (§7), the run directory as a contract (§4), and I7.

These are the properties the whole architecture exists to provide, so they are tested directly rather
than being left implied by the physics passing:

  a run directory answers what exists, at what fidelity, and what was asked for but not produced
  a consumer handles absence as a rendering and refuses substitution
  products carry physics, not presentation
  a study is loaded through the same loaders a run is
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from beamlink import artifacts as A
from beamlink import econ, physics, viz
from beamlink.drivers import corridor, sweep


@pytest.fixture(scope="module")
def reference_run(tmp_path_factory):
    """One archived run with every product and three field samplings."""
    root = tmp_path_factory.mktemp("runs")
    sampling = A.default_axial_planes(n_axial=40, n_radial=41, z_fractions=(0.0, 1.5)) + (
        A.FieldSampling(kind="axial_plane", name="shadowed", axis="major", n_axial=40,
                        n_radial=41, z_fractions=(0.0, 1.5), occluders="receiver"),
        A.FieldSampling(kind="volume", name="box", n_transverse=(7, 7), n_axial=4,
                        z_fractions=(0.6, 1.0)))
    spec = A.RunSpec(
        scenario=A.ScenarioSpec(aperture=A.ApertureSpec(diameter_m=100.0)),
        trajectory=A.TrajectorySpec.straight_overflight(100e3, 11_000.0, 250.0, n_samples=11),
        products=tuple(A.ProductRequest(name=name, sampling=sampling) if name == "field"
                       else A.ProductRequest(name=name) for name in A.PRODUCTS),
        run_id="contract-reference", notes="the reference case")
    run_dir = root / "contract-reference"
    physics.run(spec, run_dir)
    return run_dir


@pytest.fixture(scope="module")
def no_receiver_run(tmp_path_factory):
    """A scenario with no receiver — the general form of the pure focusing sweep (§2, D2)."""
    root = tmp_path_factory.mktemp("runs")
    spec = A.RunSpec(
        scenario=A.ScenarioSpec(receiver=None, aperture=A.ApertureSpec(diameter_m=100.0)),
        power=A.PowerSpec(delivered_target_w=None, radiated_w=50e6),
        trajectory=A.TrajectorySpec.straight_overflight(60e3, 11_000.0, 250.0, n_samples=5),
        products=tuple(
            A.ProductRequest(name=name, sampling=A.default_axial_planes(n_axial=30, n_radial=31))
            if name == "field" else A.ProductRequest(name=name) for name in A.PRODUCTS),
        run_id="pure-focusing-sweep")
    run_dir = root / "pure-focusing-sweep"
    physics.run(spec, run_dir)
    return run_dir


# --- the run directory is the contract (§4) -----------------------------------------------


def test_the_manifest_answers_what_exists_at_what_fidelity_and_what_did_not(reference_run):
    manifest = A.load_manifest(reference_run)
    assert manifest.engine_sha and manifest.engine_version and manifest.created_utc
    for name in A.PRODUCTS:
        status = manifest.status(name)
        assert status is not None, f"{name} is not accounted for at all"
        assert status.status in A.STATUSES
        if status.status != "produced":
            assert status.reason, f"{name} was {status.status} without a reason"
        else:
            assert status.tier and status.method, f"{name} produced without a tier or a method"
    # Every DERIVED product names its inputs (I3); a primary legitimately has none.
    for name in ("capture", "chain", "power", "aggregates", "mass", "flight_power", "reserve"):
        assert manifest.status(name).inputs, f"{name} is derived and must name its inputs (I3)"


def test_the_spec_in_the_manifest_is_verbatim(reference_run):
    """Including the trajectory arrays. A manifest that omitted them would not be the contract."""
    spec = A.load_spec(reference_run)
    assert spec.trajectory.n_samples == 11
    assert spec.scenario.aperture.diameter_m == 100.0
    assert spec.notes == "the reference case"
    payload = json.loads((Path(reference_run) / "manifest.json").read_text())
    assert len(payload["spec"]["trajectory"]["positions_m"]) == 11


def test_a_run_directory_is_readable_with_numpy_alone(reference_run):
    """The .npz files are a pile of numbers, so a reader that has never heard of this package can
    still get at them. That is what makes the products a contract rather than an API."""
    with np.load(Path(reference_run) / "chain.npz", allow_pickle=False) as archive:
        assert "eta_end_to_end" in archive.files
        assert archive["eta_end_to_end"].shape == (1, 11)
    with np.load(Path(reference_run) / "field.npz", allow_pickle=False) as archive:
        assert any(key.endswith("intensity_w_m2") for key in archive.files)


def test_products_round_trip_through_disk_bit_for_bit(reference_run):
    products, manifest = A.load_run(reference_run)
    again, _ = A.load_run(reference_run)
    assert np.array_equal(products.chain.factors, again.chain.factors)
    assert products.chain.factor_names == again.chain.factor_names
    entry = products.aggregates.entries["pass_eta_mean_served"]
    assert entry.definition.serve_policy == "eta_floor"
    assert entry.definition.note, "a definition without its note is a definition half-recorded"


# --- absence is a rendering, not an exception (§7) -----------------------------------------


def test_a_scenario_with_no_receiver_still_produces_the_field_and_the_aperture(no_receiver_run):
    """§2's resolution of the "pure focusing sweep" case: it needs a run without a receiver, not
    privileged access to the diffraction code."""
    products, manifest = A.load_run(no_receiver_run)
    assert products.field is not None and products.aperture is not None
    assert products.silhouette is None and products.capture is None and products.chain is None
    absent = manifest.absent()
    for name in ("silhouette", "capture", "chain", "power", "aggregates"):
        assert "no receiver" in absent[name], f"{name}: {absent[name]}"
    assert products.field.sets["axial_major"].intensity_w_m2.max() > 0.0, (
        "the field must still be in absolute units, from the run's stated radiated power")


def test_a_no_receiver_run_must_state_radiated_power_rather_than_a_delivered_target():
    with pytest.raises(ValueError, match="no receiver to deliver to"):
        physics.evaluate(A.RunSpec(
            scenario=A.ScenarioSpec(receiver=None, aperture=A.ApertureSpec(diameter_m=100.0)),
            trajectory=A.TrajectorySpec.straight_overflight(10e3, 11_000.0, 250.0, n_samples=3)))


def test_require_refuses_rather_than_substituting(no_receiver_run):
    products, manifest = A.load_run(no_receiver_run)
    with pytest.raises(A.RunIncomplete) as raised:
        A.require(products, manifest, "capture", "chain")
    assert "no receiver" in str(raised.value)


def test_the_report_renders_absence_and_stamps_provenance(reference_run, no_receiver_run):
    page = viz.report(reference_run, channel_sets=("axial_major", "shadowed"))
    html = page.read_text()
    manifest = A.load_manifest(reference_run)
    assert manifest.engine_sha[:12] in html, "the engine sha must be on the page"
    assert manifest.run_id in html
    assert "Provenance" in html and "pass_eta_mean_served" in html
    assert "serve policy" in html, "an aggregate shown without its definition is §1 again"

    bare = viz.report(no_receiver_run).read_text()
    assert "not generated in this run" in bare
    assert "no receiver" in bare
    assert "would be a second modelling pass presented as the model" in bare


def test_the_report_refuses_a_cut_the_run_did_not_sample(reference_run):
    with pytest.raises(A.RunIncomplete, match="no field sample set"):
        viz.report(reference_run, channel_sets=("a_cut_nobody_asked_for",))


# --- products carry physics, not presentation (I7) -----------------------------------------


def test_the_field_is_absolute_and_not_peak_normalised(reference_run):
    """The pre-SPEC cut peak-normalised by default and rounded for HTML (§8 item 1, item 12).

    Absolute units are what an exposure level can be compared against, and what keeps two legs of a
    relay in their true relative brightness — so a field whose maximum happened to be 1.0 would be a
    symptom, not a coincidence.
    """
    products, _ = A.load_run(reference_run)
    for name, sample_set in products.field.sets.items():
        peak = float(sample_set.intensity_w_m2.max())
        assert peak > 100.0, f"{name}: peak {peak:.3g} looks normalised, not W/m²"
        assert sample_set.intensity_w_m2.dtype == np.float64, f"{name}: precision reduced"
        assert not np.isnan(sample_set.intensity_w_m2).any(), (
            f"{name}: NaN in an emitted field — unresolved regions belong in `valid`, not in the "
            f"numbers, and NaN-separated polylines are exactly what I7 forbids")


def test_no_product_carries_a_display_label_or_a_normalised_axis(reference_run):
    """A crude but load-bearing check: the .npz files hold numbers.

    Label strings, colours and 0–1 normalised axes were all real defects in the pre-SPEC outputs. The
    NAMES in P7 and P9 are data and live in the manifest's metadata, not in the arrays.
    """
    for path in sorted(Path(reference_run).glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            for key in archive.files:
                array = archive[key]
                assert array.dtype.kind in "fiub", (
                    f"{path.name}:{key} has dtype {array.dtype} — products are numbers (I7)")


def test_unresolved_regions_are_masked_rather_than_filled(reference_run):
    products, _ = A.load_run(reference_run)
    sample_set = products.field.sets["axial_major"]
    assert not sample_set.valid.all(), "this sampling reaches the sub-Fresnel region"
    assert np.all(sample_set.intensity_w_m2[~sample_set.valid] == 0.0), (
        "an unresolved region must be reported unresolved, never filled to make a picture whole")
    assert products.field.unresolved_reason, "and the reason must be stated"


# --- econ selects, and refuses (§7, I4) ----------------------------------------------------


def test_econ_selects_from_products_and_publishes_what_it_selected(reference_run):
    products, manifest = A.load_run(reference_run)
    selection = econ.select(products, manifest, n_stations=28, station_spacing_km=200.0)
    result = econ.verdict(selection)
    assert result["beamed_total_usd"].shape == (1,)
    # Every physical input must be traceable to the products it came from.
    assert selection.pass_energy_efficiency[0] == pytest.approx(
        products.aggregates.entries["pass_energy_efficiency"].value[0])
    assert selection.required_w[0] == pytest.approx(products.flight_power.required_w[0].mean())
    stamp = econ.provenance(selection)
    assert stamp["serve_policy"].startswith("eta_floor")
    assert stamp["engine_sha"] == manifest.engine_sha[:12]


def test_econ_refuses_a_run_that_cannot_support_it(no_receiver_run):
    products, manifest = A.load_run(no_receiver_run)
    with pytest.raises(A.RunIncomplete):
        econ.select(products, manifest, n_stations=28, station_spacing_km=200.0)


def test_econ_refuses_an_efficiency_it_was_not_given(reference_run):
    products, manifest = A.load_run(reference_run)
    with pytest.raises(A.RunIncomplete, match="new named"):
        econ.select(products, manifest, 28, 200.0, efficiency_aggregate="whatever_i_prefer")


# --- studies load through the same loaders (§4, D6) ----------------------------------------


def test_a_sweep_writes_a_study_that_names_its_objective_and_definition(tmp_path):
    base = A.RunSpec(scenario=A.ScenarioSpec(aperture=A.ApertureSpec(diameter_m=100.0)),
                     trajectory=A.TrajectorySpec.straight_overflight(60e3, 11_000.0, 250.0,
                                                                     n_samples=7))
    result = sweep.sweep(base, "aperture.fixed_aspect", (1.0, 3.0, 6.0), "delivered_energy_j")
    assert result.objective_values.shape == (3,)
    assert result.objective_values[1] > result.objective_values[0], (
        "an elongated aperture must beat a round one on this loft")
    study_dir = sweep.write_sweep_study(result, tmp_path / "study", tmp_path / "runs")
    arrays, manifest = A.load_study(study_dir)
    assert "objective" in arrays and manifest.driver.startswith("drivers.sweep@")
    assert manifest.definitions["delivered_energy_j"].statistic == "time_integral"
    assert manifest.parameters["swept_parameter"] == "aperture.fixed_aspect"


def test_the_corridor_study_records_the_runs_and_the_reduction_it_applied(tmp_path):
    """D6: the driver derives across runs and is named in the output, exactly as P9 is within one."""
    base = A.RunSpec(scenario=A.ScenarioSpec(aperture=A.ApertureSpec(diameter_m=100.0)))
    layout = corridor.layout(altitude_m=11_000.0, min_elevation_deg=15.0, route_km=1000.0)
    results, study_dir = corridor.run_corridor(base, layout, tmp_path / "corridor",
                                               tmp_path / "runs",
                                               weathers=("clear", "heavy_rain"), n_samples=9)
    arrays, manifest = A.load_study(study_dir)
    assert len(manifest.run_dirs) == 2
    assert manifest.driver.startswith("drivers.corridor@")
    assert set(manifest.definitions) >= {"route_energy_efficiency", "route_delivered_energy_j"}
    ratio = manifest.definitions["route_energy_efficiency"]
    assert ratio.statistic == "energy_ratio" and "route" in ratio.span
    # The route efficiency must be the energy-weighted ratio, not the mean of the per-pass numbers.
    delivered = arrays["route_delivered_energy_j"]
    grid = arrays["route_grid_energy_j"]
    assert results["route_energy_efficiency"].value[0] == pytest.approx(
        float(delivered[0] / grid[0]))
    # And it must sit between the best and worst station, which is what makes it an average at all.
    assert (results["route_worst_pass_efficiency"].value[0]
            <= results["route_energy_efficiency"].value[0] + 1e-9)


def test_the_corridor_layout_comes_from_the_real_geometry_not_a_flat_earth_fit():
    """econ used 2·alt/tan(elevation), which is short at the low elevations that decide the case."""
    layout = corridor.layout(altitude_m=11_000.0, min_elevation_deg=15.0)
    flat_earth = 2.0 * 11.0 / np.tan(np.radians(15.0))
    assert layout.station_spacing_km < flat_earth, (
        f"spherical {layout.station_spacing_km:.1f} km vs flat-earth {flat_earth:.1f} km")
    assert abs(layout.station_spacing_km / flat_earth - 1.0) < 0.1
    assert layout.serving_half_span_m() == pytest.approx(layout.station_spacing_km * 1e3 / 2.0)


# --- the caches are correctness-neutral and persistable (D4) --------------------------------


def test_the_raster_cache_is_content_keyed_and_survives_the_process(tmp_path, monkeypatch):
    """The first of the two free speedups (§10 item 1), and the property D4 requires of it.

    A cache that could change a result would be worse than no cache. Three things have to hold: a
    disk read reproduces the computed value exactly, a different mesh does not collide with it, and
    dropping the whole thing costs only time.
    """
    from beamlink.physics import cache, fidelity
    from beamlink.physics.airframe import silhouette as silhouette_module

    monkeypatch.setenv("BEAMLINK_CACHE_DIR", str(tmp_path))
    standard = fidelity.config("standard")
    loft = A.LoftSpec()

    computed = silhouette_module.silhouette_at(loft, 42.0, 137.0, standard)
    cache.clear_process_memo()
    from_disk = silhouette_module.silhouette_at(loft, 42.0, 137.0, standard)
    for key in ("element_x", "element_y", "incidence", "projected_area", "basis", "sigma", "skew"):
        assert np.array_equal(computed[key], from_disk[key]), f"{key} changed on a cache round trip"

    # A different loft must not read this entry back: the pre-SPEC dict was keyed on the look angle
    # alone, so a coarser mesh could and did read a finer mesh's raster.
    stubbier = A.LoftSpec(full_span_m=60.0)
    assert (silhouette_module.raster_key(loft, 42.0, 137.0, standard)
            != silhouette_module.raster_key(stubbier, 42.0, 137.0, standard))
    coarser = fidelity.config("coarse")
    assert (silhouette_module.raster_key(loft, 42.0, 137.0, standard)
            != silhouette_module.raster_key(loft, 42.0, 137.0, coarser))

    # And the tessellation is in the key too, which is what makes the mesh a fidelity lever (D14).
    assert silhouette_module.silhouette_at(loft, 42.0, 137.0, coarser)["area"] != computed["area"]

    for entry in tmp_path.glob("*.npz"):
        entry.unlink()
    cache.clear_process_memo()
    recomputed = silhouette_module.silhouette_at(loft, 42.0, 137.0, standard)
    assert np.array_equal(computed["element_x"], recomputed["element_x"]), (
        "deleting the cache must cost time and nothing else")


# --- the documented array shapes are the actual array shapes (I5, §5) -----------------------

# What each product's arrays must lead with, from the docstrings in `artifacts/products.py`. This is a
# transcription of the schema, kept here so the schema is asserted rather than described: a product
# that returns a scalar where it promised a candidate axis makes a consumer branch, which is the kind
# of surprise I5 exists to prevent (and P13 did exactly that until this test was written).
_LEADING_AXES = {
    "trajectory": {"positions_m": ("sample", 3), "attitudes_deg": ("sample", 3),
                   "times_s": ("sample",), "positions_ecef_m": ("sample", 3),
                   "ground_speed_ms": ("sample",)},
    "geometry": {"slant_m": ("candidate", "leg", "sample"),
                 "elevation_deg": ("candidate", "leg", "sample"),
                 "horizon_blocked": ("candidate", "leg", "sample"),
                 "arrival_beta_deg": ("candidate", "sample"),
                 "arrival_alpha_deg": ("candidate", "sample")},
    "silhouette": {"projected_area_m2": ("candidate", "sample"),
                   "equal_area_radius_m": ("candidate", "sample"),
                   "view_basis_body": ("candidate", "sample", 3, 3),
                   "principal_half_extent_m": ("candidate", "sample", 2),
                   "principal_sigma_m": ("candidate", "sample", 2),
                   "principal_skew_m3": ("candidate", "sample", 2)},
    "aperture": {"semi_major_m": ("candidate", "sample"), "semi_minor_m": ("candidate", "sample"),
                 "equal_area_diameter_m": ("candidate", "sample"),
                 "taper_exponent": ("candidate", "sample"), "focus_range_m": ("candidate", "sample"),
                 "aspect": ("candidate", "sample"),
                 "frame_axes_ecef": ("candidate", "sample", 3, 3),
                 "origin_ecef_m": ("candidate", "sample", 3)},
    "capture": {"geometric": ("candidate", "sample"), "incidence_weighted": ("candidate", "sample"),
                "peak_areal_density_per_m2": ("candidate", "sample"),
                "equivalent_disc_radius_m": ("candidate", "sample"),
                "density_cap_exceeded": ("candidate", "sample"),
                "skin_power_density_w_m2": ("candidate", "sample")},
    "chain": {"eta_end_to_end": ("candidate", "sample"), "feasible": ("candidate", "sample")},
    "power": {"grid_w": ("candidate", "sample"), "radiated_w": ("candidate", "leg", "sample"),
              "at_receiver_w": ("candidate", "sample"),
              "rectenna_terminals_w": ("candidate", "sample"),
              "waste_w": ("candidate", "sample"), "rated_grid_w": ("candidate",)},
    "atmosphere": {"loss_db": ("candidate", "leg", "sample"),
                   "transmittance": ("candidate", "leg", "sample"),
                   "gaseous_db": ("candidate", "leg", "sample"),
                   "rain_db": ("candidate", "leg", "sample"),
                   "cloud_db": ("candidate", "leg", "sample")},
    "mass": {"payload_t": ("candidate",), "structure_t": ("candidate",), "systems_t": ("candidate",),
             "motor_t": ("candidate",), "reserve_battery_t": ("candidate",),
             "rectenna_t": ("candidate",), "mtow_t": ("candidate",),
             "empty_fraction": ("candidate",), "residual_kg": ("candidate",)},
    "flight_power": {"cruise_shaft_w": ("candidate",), "peak_shaft_w": ("candidate",),
                     "required_w": ("candidate", "sample"), "peak_required_w": ("candidate",),
                     "lift_to_drag": ("candidate",)},
    "reserve": {"glide_range_km": ("candidate",), "glide_duration_s": ("candidate",),
                "battery_duration_s": ("candidate",), "powered_fraction": ("candidate",)},
}


def test_every_product_has_the_leading_axes_it_documents():
    """Checked on a run with TWO candidates and an unequal sample count, so a shape that only happens
    to be right when both are 1 fails."""
    trajectory = A.TrajectorySpec.straight_overflight(60e3, 11_000.0, 250.0, n_samples=7)
    spec = A.RunSpec(
        scenario=A.ScenarioSpec(aperture=A.ApertureSpec(diameter_m=100.0)),
        trajectory=trajectory,
        candidates=(A.CandidateAxis(parameter="aperture.diameter_m", values=(80.0, 120.0)),),
        run_id="shapes")
    products, manifest = physics.evaluate(spec)
    sizes = {"candidate": 2, "sample": 7, "leg": 1}
    for product_name, fields in _LEADING_AXES.items():
        product = products.get(product_name)
        assert product is not None, f"{product_name} was not produced"
        for field_name, axes in fields.items():
            array = getattr(product, field_name)
            assert array is not None, f"{product_name}.{field_name} is None"
            expected = tuple(sizes.get(axis, axis) for axis in axes)
            assert np.shape(array) == expected, (
                f"{product_name}.{field_name} is {np.shape(array)}, and §5 says {expected}")

    # And the field's sample sets, whose point axis is the one thing that varies by sampling.
    for name, sample_set in products.field.sets.items():
        assert sample_set.intensity_w_m2.shape[:2] == (2, 7), f"{name}: {sample_set.intensity_w_m2.shape}"
        assert sample_set.points_m.shape[:2] == (2, 7)
        assert sample_set.points_m.shape[-1] == 3
        assert sample_set.valid.shape == sample_set.intensity_w_m2.shape
