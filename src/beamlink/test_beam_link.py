"""Behavioural checks for the beam link model. Run as a script; every `check` prints the
numbers behind it, so a failure is read off the detail string rather than a message.

These are mostly invariants and sanity relations, not regressions against stored values:
they pin down monotonicity, symmetry, agreement with closed-form limits, and internal
consistency of the reported efficiency chain.
"""

import numpy as np

from beamlink import relations
from beamlink import geo
from beamlink import beam_link as bl
from beamlink import sweeps
from beamlink import diffraction as dfr
from beamlink import bwb_model as m
from beamlink import field as fl
from beamlink.flyby_frames import channel_field, frame_physics
from beamlink.architectures import standard_architectures
from beamlink.bwb_model import BWBSilhouette, beta_alpha_from_enu, projected_area, make_bwb_mesh, \
    triangle_normals_areas, direction_from_beta_alpha

PASS, FAIL = "  ok ", "FAIL "
_failures = []


def check(name, condition, detail=""):
    print(f"{PASS if condition else FAIL} {name}{('  — ' + detail) if detail else ''}")
    if not condition:
        _failures.append(name)


# --- Silhouette ---------------------------------------------------------------

def test_silhouette():
    """Silhouette area behaves like a flat lifting body: belly-on beats edge-on several
    times over, opposite views agree (closed body), area falls monotonically as the beam
    moves off the belly, and banking away from the beam costs area."""
    print("\n[silhouette]")
    verts, faces = make_bwb_mesh()
    normals, areas = triangle_normals_areas(verts, faces)
    belly = projected_area(direction_from_beta_alpha(0.0, 0.0), normals, areas)
    edge = projected_area(direction_from_beta_alpha(90.0, 90.0), normals, areas)
    check("belly-on silhouette exceeds edge-on", belly > 3 * edge,
          f"{belly:.0f} vs {edge:.0f} m²")

    # Convex projected-area identity: closed surface must give equal area for a
    # direction and its opposite (front area = back area for a closed body).
    d = direction_from_beta_alpha(35.0, 20.0)
    a_fwd = projected_area(d, normals, areas)
    a_back = projected_area(-d, normals, areas)
    check("closed-surface front/back areas match", abs(a_fwd - a_back) / a_fwd < 0.02,
          f"{a_fwd:.1f} vs {a_back:.1f} m²")

    sil = BWBSilhouette()
    check("silhouette decreases monotonically 0→90° from belly",
          np.all(np.diff(sil.area(np.linspace(0, 90, 19), np.zeros(19))) <= 1e-6))

    # Banking away from a straight-down emitter must reduce captured area.
    a_level = sil.area(*beta_alpha_from_enu([0, 0, -1], roll_deg=0))
    a_banked = sil.area(*beta_alpha_from_enu([0, 0, -1], roll_deg=40))
    check("banking 40° reduces belly capture", a_banked < a_level,
          f"{float(a_level):.0f} → {float(a_banked):.0f} m²")


# --- Atmosphere ---------------------------------------------------------------

def test_atmosphere():
    """Gaseous loss is all below cruise altitude (11 km and 20 km zenith paths agree), a
    30 degree slant path costs 1/sin(30) = 2x the zenith path, and heavy rain is survivable
    at 5.8 GHz but not at 30 GHz."""
    print("\n[atmosphere — curved-path integration]")
    atmos = bl.Atmosphere()
    S = geo.point(0.0, 0.0, 0.0)
    # Gaseous loss to 11 km ≈ loss to 20 km (stratosphere is dry).
    g11 = float(atmos.path_loss_db(30.0, S, geo.point(0.0, 0.0, 11_000.0), "clear"))
    g20 = float(atmos.path_loss_db(30.0, S, geo.point(0.0, 0.0, 20_000.0), "clear"))
    check("stratosphere adds ~no gaseous loss above 11 km", abs(g20 - g11) < 0.05,
          f"{g11:.3f} vs {g20:.3f} dB")

    # Slant ≈ doubles the vertical loss at 30° elevation.
    d30 = geo.downrange_for_elevation(30.0, 11_000.0)
    slant = float(atmos.path_loss_db(30.0, S, geo.point(d30, 0.0, 11_000.0), "clear"))
    check("30° slant ≈ 2× vertical loss", abs(slant / g11 - 2.0) < 0.06,
          f"{slant:.3f}/{g11:.3f} = {slant/g11:.2f}")

    # Rain hammers 30 GHz but spares 5.8 GHz.
    st, pl = sweeps.reference_geometry(30.0)
    r58 = bl.evaluate(bl.LinkConfig(station_pos=st, plane_pos=pl, f_emit_plane_ghz=5.8,
                                    weather="heavy_rain"), 10.0)["atmos"]
    r30 = bl.evaluate(bl.LinkConfig(station_pos=st, plane_pos=pl, f_emit_plane_ghz=30.0,
                                    weather="heavy_rain"), 10.0)["atmos"]
    check("5.8 GHz survives heavy rain, 30 GHz does not", r58 > 0.7 and r30 < 0.05,
          f"5.8→{float(r58):.3f}  30→{float(r30):.3f}")


def test_curvature():
    """Earth curvature is really in the geometry: elevation falls with downrange, a 600 km
    link is over the horizon while 300 km is not, and from the aircraft the station is below
    the local horizontal."""
    print("\n[earth curvature]")
    S = geo.point(0.0, 0.0, 0.0)
    # Elevation of a level plane falls with downrange and eventually goes below horizon.
    ranges = np.array([20e3, 100e3, 200e3, 300e3, 400e3])
    _, elev = geo.link(S, geo.point(ranges, 0.0, 11_000.0))
    check("elevation decreases monotonically with range", np.all(np.diff(elev) < 0),
          f"{np.round(elev, 1)} deg")

    # A plane past the ~374 km line-of-sight horizon is infeasible.
    near = bl.evaluate(bl.LinkConfig(station_pos=(0., 0, 0), plane_pos=(300e3, 0, 11e3),
                                     f_emit_plane_ghz=30.0), 10.0)
    far = bl.evaluate(bl.LinkConfig(station_pos=(0., 0, 0), plane_pos=(600e3, 0, 11e3),
                                    f_emit_plane_ghz=30.0), 10.0)
    check("plane within horizon feasible, beyond horizon not",
          bool(near["feasible"]) and not bool(far["feasible"]))

    # Curvature tilts a level plane's aspect: at range it sees the station below its
    # own local horizon (belly no longer points at the emitter).
    d = geo.downrange_for_elevation(20.0, 11_000.0)
    up_comp = geo.direction_in_local_enu(geo.point(d, 0, 11_000.0), S)[2]
    check("level plane sees distant station below its local horizon", up_comp < 0,
          f"local-up component {float(up_comp):.3f}")


# --- Near-field coupling ------------------------------------------------------

def test_coupling():
    """Aperture sizing follows the Goubau rule: tau=2 collects 98%, required diameter scales
    with wavelength (so 5.8 GHz needs 30/5.8 times the 30 GHz aperture), and a low-elevation
    pass needs a bigger aperture than a high one because the slant range is longer."""
    print("\n[near-field coupling]")
    check("collection η(τ=2) ≈ 0.982", abs(relations.collection_efficiency(2.0) - 0.9817) < 1e-3)

    # Sizing to τ=2 at any frequency yields the same capture; the differentiator is
    # emitter DIAMETER, which scales as λ (∝ 1/f).
    st, pl = sweeps.reference_geometry(30.0)
    d58 = bl.evaluate(bl.LinkConfig(station_pos=st, plane_pos=pl, f_emit_plane_ghz=5.8), 10.0)["emitter_diam_m"]
    d30 = bl.evaluate(bl.LinkConfig(station_pos=st, plane_pos=pl, f_emit_plane_ghz=30.0), 10.0)["emitter_diam_m"]
    ratio = float(d58) / float(d30)
    check("emitter Ø scales ~as 1/frequency", abs(ratio - 30.0 / 5.8) / (30.0 / 5.8) < 0.05,
          f"Ø 5.8={float(d58):.0f} m, 30={float(d30):.0f} m, ratio {ratio:.2f} vs {30/5.8:.2f}")

    # Lower elevation → longer slant + smaller silhouette → bigger emitter.
    d_high = bl.evaluate(bl.LinkConfig(*sweeps.reference_geometry(45.0), f_emit_plane_ghz=30.0), 10.0)["emitter_diam_m"]
    d_low = bl.evaluate(bl.LinkConfig(*sweeps.reference_geometry(10.0), f_emit_plane_ghz=30.0), 10.0)["emitter_diam_m"]
    check("low elevation needs a bigger emitter", float(d_low) > float(d_high),
          f"45°→{float(d_high):.0f} m, 10°→{float(d_low):.0f} m")


# --- End-to-end bookkeeping ---------------------------------------------------

def test_end_to_end():
    """The reported chain is self-consistent for all three architectures: every stage is a
    factor in (0, 1], their product is exactly eta_end_to_end, and grid/waste power follow
    from it. Last check: at 30 GHz a passive mirror beats an active relay, i.e. the
    conversion loss of rectifying and re-transmitting costs more than it buys."""
    print("\n[end-to-end]")
    st, pl = sweeps.reference_geometry(30.0)
    for arch, cfg in {
        "direct": bl.LinkConfig(station_pos=st, plane_pos=pl, f_emit_plane_ghz=30.0),
        "relay-active": bl.LinkConfig(station_pos=st, plane_pos=pl,
                                      relay_pos=(pl[0] / 2, 0, 20_000.0), relay_mode="active"),
        "relay-passive": bl.LinkConfig(station_pos=st, plane_pos=pl,
                                       relay_pos=(pl[0] / 2, 0, 20_000.0), relay_mode="passive",
                                       f_emit_relay_ghz=30.0),
    }.items():
        s = bl.evaluate(cfg, delivered_mw=10.0)
        stage_effs = [e for _, e in bl.waterfall(s)]
        check(f"[{arch}] every stage efficiency in (0,1]",
              all(0.0 < e <= 1.0 for e in stage_effs))
        product = float(np.prod(stage_effs))
        check(f"[{arch}] η_end_to_end == product of stages",
              abs(product - float(s["eta_end_to_end"])) < 1e-9,
              f"{product:.4f} vs {float(s['eta_end_to_end']):.4f}")
        check(f"[{arch}] grid = delivered / η and waste ≥ 0",
              abs(float(s["grid_mw"]) - 10.0 / float(s["eta_end_to_end"])) < 1e-6
              and float(s["waste_mw"]) >= 0)

    # Passive relay beats active relay in clear air (no rectify+reamp penalty).
    active = bl.evaluate(bl.LinkConfig(station_pos=st, plane_pos=pl,
                                       relay_pos=(pl[0] / 2, 0, 20_000.0), relay_mode="active"), 10.0)
    passive = bl.evaluate(bl.LinkConfig(station_pos=st, plane_pos=pl,
                                        relay_pos=(pl[0] / 2, 0, 20_000.0), relay_mode="passive",
                                        f_emit_relay_ghz=30.0), 10.0)
    check("passive relay > active relay in clear air",
          float(passive["eta_end_to_end"]) > float(active["eta_end_to_end"]),
          f"passive {float(passive['eta_end_to_end']):.3f} vs active {float(active['eta_end_to_end']):.3f}")


# --- Vectorization ------------------------------------------------------------

def test_vectorized():
    """Array inputs keep their shape and stay monotone: loss rises with frequency, silhouette
    area falls as the beam moves off the belly."""
    print("\n[vectorization]")
    atmos = bl.Atmosphere()
    freqs = np.array([5.8, 12.0, 20.0, 30.0, 34.8])
    S, P = geo.point(0.0, 0.0, 0.0), geo.point(20e3, 0.0, 11_000.0)
    loss = atmos.path_loss_db(freqs, S, P, "cloudy")
    check("path_loss_db vectorizes over frequency", loss.shape == freqs.shape
          and np.all(np.diff(loss) > 0), f"{np.round(loss, 2)} dB (rising with f)")

    sil = BWBSilhouette()
    betas = np.array([0.0, 30.0, 60.0, 90.0])
    areas = sil.area(betas, np.zeros_like(betas))
    check("silhouette lookup vectorizes", areas.shape == betas.shape and np.all(np.diff(areas) < 0))


def test_pass_profile():
    """Shape of a single overflight: delivery peaks overhead, is near-symmetric fore and aft
    (a swept planform is not exactly symmetric, hence the 8% tolerance), and sizing the
    emitter for a lower reference elevation buys a wider useful window at the price of a
    bigger aperture. Capture is near-perfect overhead and poor at the edges."""
    print("\n[pass profile / fixed emitter]")
    p = sweeps.pass_profile(frequency_ghz=30.0, reference_elevation_deg=60.0)
    peak_idx = int(np.argmax(p["deliverable_mw"]))
    overhead_idx = int(np.argmin(np.abs(p["alongtrack_km"])))
    dmax = p["deliverable_mw"].max()
    check("peak delivery is in the central plateau", abs(p["alongtrack_km"][peak_idx]) < 10.0,
          f"peak at {p['alongtrack_km'][peak_idx]:.1f} km")
    check("overhead is on the plateau (within 3% of peak)",
          p["deliverable_mw"][overhead_idx] > 0.97 * dmax,
          f"overhead {p['deliverable_mw'][overhead_idx]/dmax:.3f} of peak")
    left = p["deliverable_mw"][p["alongtrack_km"] < 0]
    right = p["deliverable_mw"][p["alongtrack_km"] > 0][::-1]
    n = min(len(left), len(right))
    # Only APPROXIMATELY symmetric: the swept wing + aft winglets make nose-on and
    # tail-on differ by a few %.
    asym = np.max(np.abs(left[-n:] - right[:n]) / np.maximum(right[:n], 1e-9))
    check("pass profile ~symmetric about overhead (swept-wing asymmetry small)",
          asym < 0.08, f"max fore/aft asymmetry {100*asym:.1f}%")

    # A larger emitter (built for a lower reference elevation) widens the window.
    def window_km(ref):
        pp = sweeps.pass_profile(frequency_ghz=30.0, reference_elevation_deg=ref)
        norm = pp["deliverable_mw"] / pp["deliverable_mw"].max()
        xin = pp["alongtrack_km"][norm >= 0.5]
        return float(xin.max() - xin.min()), pp["emitter_diam_m"]
    w60, d60 = window_km(60.0)
    w20, d20 = window_km(20.0)
    check("bigger emitter → wider powering window", w20 > w60 and d20 > d60,
          f"Ø{d60:.0f}m→{w60:.0f}km vs Ø{d20:.0f}m→{w20:.0f}km")

    # Fixed emitter: capture is high near overhead (defocused spot fills the silhouette)
    # and DROPS toward the low-elevation edges (the spot overspills the sliver).
    check("fixed-emitter capture falls off-overhead",
          p["capture"][peak_idx] > 0.85 and np.min(p["capture"]) < 0.3,
          f"peak {p['capture'][peak_idx]:.2f}, min {np.min(p['capture']):.2f}")


# --- Rasterized silhouette + 2-D overlap capture ------------------------------

def test_overlap():
    """The raster overlap against cases with known answers.

    On a flat disc receiver it must reproduce the Airy encircled energy, and with the taper
    free to vary it must at least reach the Goubau prediction. Then on the aircraft: the
    z-buffer sees less than the convex projected area at grazing angles (self-occlusion), the
    incidence derate barely bites belly-on but clearly does edge-on, and shaping the aperture
    to the silhouette never captures less than a round one.
    """
    print("\n[overlap capture — the 2-D beam × silhouette model]")

    # (1) The overlap of a ROUND uniform aperture on a round rectenna must reproduce the
    # exact analytic Airy encircled energy E = 1 − J0(2τ)² − J1(2τ)² — i.e. the diffraction
    # field really is the Fraunhofer pattern of the aperture, sidelobe spill and all. (This
    # replaces the old Goubau 1−e^(−τ²) check: Goubau is the optimal-taper envelope, which the
    # ADAPTIVE taper approaches to ~1–2 %, but a fixed uniform aperture loses its sidelobes.)
    def disk_raster(r_r, n=400):
        g = np.linspace(-r_r, r_r, n)
        uu, vv = np.meshgrid(g, g)
        mask = uu ** 2 + vv ** 2 <= r_r ** 2
        cell = (g[1] - g[0]) ** 2
        return dict(u=uu[mask], v=vv[mask], incidence=np.ones(int(mask.sum())),
                    proj_area=np.full(int(mask.sum()), cell), area=int(mask.sum()) * cell)
    lam, D, r_t, r_r = relations.wavelength(30e9), 200e3, 40.0, 15.0
    airy = float(dfr.airy_encircled_energy(r_r, r_t, lam, D))
    geom, _, _ = bl.overlap_capture(disk_raster(r_r), r_t, lam, D, shaped=False,
                                    taper_p=0.0, incidence_floor=1.0)
    check("round uniform aperture on a disk reproduces the Airy encircled energy",
          abs(geom - airy) < 1.5e-2, f"overlap {geom:.4f} vs Airy {airy:.4f}")

    # (1b) The adaptive taper recovers most of the sidelobe loss, tracking the Goubau
    # optimal-coupling envelope 1−e^(−τ²) to within a couple of points on the same disk.
    tau = relations.coupling_tau(30e9, D, r_t, r_r)
    goubau = relations.collection_efficiency(tau)
    best = max(bl.overlap_capture(disk_raster(r_r), r_t, lam, D, shaped=False,
                                  taper_p=p, incidence_floor=1.0)[0]
               for p in (0.0, 0.5, 1.0, 1.5, 2.0))
    check("adaptive taper approaches the Goubau optimal-coupling envelope",
          best >= goubau - 0.03, f"best-taper {best:.4f} vs Goubau {goubau:.4f}")

    # (2) The rasterized (self-occluded) silhouette is SMALLER than the convex identity
    # side-on at low elevation, where winglets + far wing overlap in projection.
    verts, faces = make_bwb_mesh()
    normals, areas = triangle_normals_areas(verts, faces)
    samples, snorm, swt = m.get_surface_samples()
    d = direction_from_beta_alpha(88.0, 90.0); d = d / np.linalg.norm(d)
    convex = float(projected_area(d, normals, areas))
    raster = m.rasterize_silhouette(d, samples, snorm, swt)["area"]
    check("rasterized silhouette removes convex over-count (side, low elev)",
          raster < 0.93 * convex, f"raster {raster:.0f} < convex {convex:.0f} m²")

    # (3) The incidence penalty EMERGES from geometry: incidence-weighted capture ≈ the
    # geometric overlap overhead, but is markedly lower at grazing (low) elevation.
    def caps(beta):
        dd = direction_from_beta_alpha(beta, 0.0); dd = dd / np.linalg.norm(dd)
        r = m.rasterize_silhouette(dd, samples, snorm, swt)
        # Pin the floor low so this tests the MECHANISM, not the (milder, conformal-
        # rectenna) default floor.
        return bl.overlap_capture(r, 30.0, lam, 250e3, shaped=False, incidence_floor=0.70)
    g_over, e_over, _ = caps(5.0)     # near belly-on (overhead)
    g_low, e_low, _ = caps(85.0)      # grazing (low elevation)
    check("incidence penalty ~absent overhead", e_over / g_over > 0.9,
          f"eff/geom = {e_over/g_over:.2f}")
    check("incidence penalty bites at low elevation", e_low / g_low < 0.85,
          f"eff/geom = {e_low/g_low:.2f}")

    # (4) A shaped (phased-array) spot captures ≥ a round spot on an elongated silhouette.
    dd = direction_from_beta_alpha(80.0, 0.0); dd = dd / np.linalg.norm(dd)
    r = m.rasterize_silhouette(dd, samples, snorm, swt)
    g_round = bl.overlap_capture(r, 20.0, lam, 250e3, shaped=False)[0]
    g_shaped = bl.overlap_capture(r, 20.0, lam, 250e3, shaped=True)[0]
    check("shaped spot ≥ round spot on an elongated silhouette", g_shaped >= g_round - 1e-6,
          f"shaped {g_shaped:.3f} ≥ round {g_round:.3f}")


# --- Trajectory-integrated flyby ----------------------------------------------

def test_flyby():
    """Corridor-level behaviour: the three pass averages order as harmonic <= arithmetic <=
    peak; a 20 km tether emitter is above the weather so a storm barely touches it; a relay
    beats going direct through a storm; and a 30 GHz ground station loses the link outright in
    a storm, which is what makes its constant-delivery energy infinite."""
    print("\n[flyby — trajectory-integrated over a pass]")
    archs = standard_architectures()

    # Ordering: harmonic ≤ arithmetic mean ≤ peak, always.
    r = sweeps.flyby(archs["5.8 direct"], station_spacing_km=200.0)
    check("harmonic ≤ mean ≤ peak η", r["hmean_eta"] <= r["mean_eta"] + 1e-9 <= r["peak_eta"] + 1e-9,
          f"h {r['hmean_eta']:.3f} ≤ mean {r['mean_eta']:.3f} ≤ peak {r['peak_eta']:.3f}")

    # The tether (emitter above the weather) is ~weather-independent over the pass.
    t_clear = sweeps.flyby({**archs["tether 30 direct"], "weather": "clear"}, station_spacing_km=200.0)["mean_eta"]
    t_storm = sweeps.flyby({**archs["tether 30 direct"], "weather": "storm"}, station_spacing_km=200.0)["mean_eta"]
    check("tether high-f is ~weather-independent", abs(t_clear - t_storm) / t_clear < 0.05,
          f"clear {t_clear:.3f} vs storm {t_storm:.3f}")

    # In a storm the relay feeds continuously far better than 5.8-direct (whose
    # low-elevation pass edges rain out) — the relay's value over the FULL flyby.
    relay = sweeps.flyby({**archs["relay ×3 mult 5.8→17.4"], "weather": "storm"}, station_spacing_km=200.0)
    direct = sweeps.flyby({**archs["5.8 direct"], "weather": "storm"}, station_spacing_km=200.0)
    check("relay beats 5.8-direct on continuous delivery in storm",
          relay["hmean_eta"] > direct["hmean_eta"],
          f"relay hmean {relay['hmean_eta']:.3f} vs direct {direct['hmean_eta']:.3f}")

    # Ground direct-30 rains out somewhere on the pass → cannot feed continuously (∞).
    g30 = sweeps.flyby({**archs["30 direct (ground)"], "weather": "storm"}, station_spacing_km=200.0)
    check("ground direct-30 cannot feed continuously in a storm",
          not np.isfinite(g30["grid_energy_per_pass_gj"]))


def test_pointing_density():
    """A 2 m pointing error costs under 2% of end-to-end efficiency; the skin power density
    respects max_power_density_w_m2 (2% slack); and asking for more delivered power forces a
    wider spot, since the only way to stay under the density cap is to defocus."""
    print("\n[pointing folded into overlap + power-density cap]")
    st, pl, relay = (0, 0, 0), (0, 0, 11e3), (0, 0, 20e3)
    base = dict(station_pos=st, plane_pos=pl, relay_pos=relay, relay_mode="multiplier",
                f_emit_relay_ghz=5.8, f_relay_plane_ghz=17.4, relay_aperture_diam_m=40,
                max_power_density_w_m2=None)
    # A tiny spot well inside a huge rectenna loses ~nothing to a 2 m pointing error —
    # the old exp(−2(δ/w)²) point-target factor wrongly tanked this.
    e0 = bl.evaluate(bl.LinkConfig(**base, pointing_error_m=0.0), 10.0)["eta_end_to_end"]
    e2 = bl.evaluate(bl.LinkConfig(**base, pointing_error_m=2.0), 10.0)["eta_end_to_end"]
    check("tiny contained spot ~immune to pointing error", abs(e2 - e0) / e0 < 0.02,
          f"0 m → {e0:.3f}, 2 m → {e2:.3f}")

    # The power density is actually held under the cap (spot spread as needed).
    s = bl.evaluate(bl.LinkConfig(station_pos=st, plane_pos=pl, relay_pos=relay,
                    relay_mode="multiplier", f_emit_relay_ghz=5.8, f_relay_plane_ghz=17.4,
                    relay_aperture_diam_m=40), 10.0)
    check("incident power density held under the 25 W/cm² cap",
          s["power_density_w_m2"] <= 2.5e5 * 1.02, f"{s['power_density_w_m2']/1e4:.1f} W/cm²")
    check("a compliant case is not flagged", s["density_cap_exceeded"] is False)

    # When the cap CANNOT be met at any allowed defocus, that is reported rather than hidden:
    # 400 MW through the same 40 m relay aperture overhead sits ~1.3× the cap at every focus.
    hot = bl.evaluate(bl.LinkConfig(station_pos=st, plane_pos=pl, relay_pos=relay,
                      relay_mode="multiplier", f_emit_relay_ghz=5.8, f_relay_plane_ghz=17.4,
                      relay_aperture_diam_m=40), 400.0)
    check("unmeetable density cap is flagged, not silently exceeded",
          hot["density_cap_exceeded"] is True and hot["power_density_w_m2"] > 2.5e5,
          f"{hot['power_density_w_m2']/1e4:.1f} W/cm² = {hot['power_density_w_m2']/2.5e5:.2f}× cap")
    # More delivered power → the cap forces a larger spot (lower peak-power geometry).
    lo = bl.evaluate(bl.LinkConfig(station_pos=st, plane_pos=pl, f_emit_plane_ghz=5.8,
                     emitter_diam_m=78.0), 1.0)["equivalent_disc_radius_m"]
    hi = bl.evaluate(bl.LinkConfig(station_pos=st, plane_pos=pl, f_emit_plane_ghz=5.8,
                     emitter_diam_m=78.0), 40.0)["equivalent_disc_radius_m"]
    check("higher delivered power → larger enforced spot", hi >= lo,
          f"1 MW → {lo:.1f} m, 40 MW → {hi:.1f} m")


def test_defocus_floor():
    """A big aperture cannot be focused as short as the density cap's `slant/8` floor: the field
    model refuses (rather than silently clip the aperture) because the focusing phase would need a
    finer sample pitch than it will allocate. The floor has to back off to a focus it can
    represent instead of letting that refusal escape.

    The animation's own defaults land here — they size the emitter for a low-elevation reference
    and then fly it overhead, so a several-hundred-metre aperture sits 11 km from the aircraft.
    """
    print("\n[defocus floor — an unrepresentable focus backs off, it does not raise]")
    station, plane, diam = (0.0, 0.0, 0.0), (0.0, 0.0, 11_000.0), 400.0
    cfg = bl.LinkConfig(station_pos=station, plane_pos=plane, f_emit_plane_ghz=5.8,
                        emitter_diam_m=diam)
    slant = float(bl.geometry(np.asarray(plane, float), np.asarray(station, float))[0])

    # Precondition, so the rest of this is exercising the back-off rather than a geometry that
    # never needed one: the intended floor really is out of the field model's reach here.
    try:
        bl.plane_capture(cfg, station, 5.8, diam, taper_p=0.0, focus_range_m=slant / 8.0)
        refused = False
    except dfr.ApertureTooFine:
        refused = True
    check(f"slant/8 is unrepresentable for a {diam:.0f} m aperture at {slant/1e3:.0f} km", refused)

    focused = bl.plane_capture(cfg, station, 5.8, diam, taper_p=0.0)
    focus, pc = bl._defocus_floor(cfg, station, 5.8, diam, 0.0, slant, focused, False)
    check("the floor backs off to a longer focus", slant / 8.0 < focus <= slant,
          f"{focus/1e3:.2f} km vs slant/8 = {slant/8e3:.2f} km")
    check("...and the capture there is real", pc["capture"] > 0.0, f"capture {pc['capture']:.3f}")

    s = bl.evaluate(cfg, 10.0)              # used to propagate ApertureTooFine out of evaluate
    check("a large overhead aperture evaluates end-to-end", 0.0 < s["eta_end_to_end"] < 1.0,
          f"η {s['eta_end_to_end']:.3f} at focus {s['focus_range_m']/1e3:.1f} km")


def test_taper():
    """What tapering the aperture buys, on a receiver whose radius is the first Airy null
    (0.61*lambda*D/a): it moves
    power out of the sidelobes into the main lobe (capture up) while lowering the peak
    intensity, which is the lever the density cap pulls on."""
    print("\n[aperture taper — sidelobe spill vs main-lobe width]")
    def disk_raster(r_r, n=400):
        g = np.linspace(-r_r, r_r, n)
        uu, vv = np.meshgrid(g, g)
        mask = uu ** 2 + vv ** 2 <= r_r ** 2
        cell = (g[1] - g[0]) ** 2
        return dict(u=uu[mask], v=vv[mask], incidence=np.ones(int(mask.sum())),
                    proj_area=np.full(int(mask.sum()), cell), area=int(mask.sum()) * cell)
    lam, D, r_t = relations.wavelength(30e9), 200e3, 40.0
    # On a receiver the size of the uniform aperture's main lobe (first Airy null), the
    # uniform pattern spills ~16 % into sidelobes that miss the disk; a taper trades main-lobe
    # width for far lower sidelobes and so lands MORE of the power on the disk.
    r_null = 0.610 * lam * D / r_t
    r = disk_raster(r_null)
    g_unif = bl.overlap_capture(r, r_t, lam, D, shaped=False, taper_p=0.0, incidence_floor=1.0)[0]
    g_taper = bl.overlap_capture(r, r_t, lam, D, shaped=False, taper_p=1.5, incidence_floor=1.0)[0]
    check("taper lifts capture on a main-lobe-sized receiver (kills sidelobe spill)",
          g_taper > g_unif, f"uniform {g_unif:.3f} → taper {g_taper:.3f}")
    # A heavier taper widens the main lobe → lower peak areal intensity for the same power.
    p_unif = bl.overlap_capture(r, r_t, lam, D, shaped=False, taper_p=0.0, incidence_floor=1.0)[2]
    p_taper = bl.overlap_capture(r, r_t, lam, D, shaped=False, taper_p=2.0, incidence_floor=1.0)[2]
    check("heavier taper lowers the peak areal intensity", p_taper < p_unif,
          f"uniform {p_unif:.2e} → taper {p_taper:.2e} /m²")


def test_adaptive_taper():
    """The adaptive search is at least as good as any fixed taper it could have chosen (3e-3
    slack for the coarse scan grid), stays inside [0, taper_p_max], and switching it off
    really does use cfg.taper_p."""
    print("\n[adaptive aperture taper — chosen per geometry for max capture under the cap]")
    st, relay = (0, 0, 0), (0, 0, 20e3)
    base = dict(station_pos=st, relay_pos=relay, relay_mode="multiplier", f_emit_relay_ghz=5.8,
                f_relay_plane_ghz=17.4, relay_aperture_diam_m=40, plane_pos=(0, 0, 11e3))
    # The adaptive taper optimises over the (1−ρ²)^p family, so it must capture ≥ ANY fixed
    # taper at the same geometry.
    adap = bl.evaluate(bl.LinkConfig(adaptive_taper=True, **base), 10.0)
    fixed_caps = [bl.evaluate(bl.LinkConfig(adaptive_taper=False, taper_p=p, **base), 10.0)["capture"]
                  for p in (0.0, 1.0, 2.0, 4.0)]
    check("adaptive taper ≥ every fixed taper", adap["capture"] >= max(fixed_caps) - 3e-3,
          f"adaptive {adap['capture']:.4f} vs best fixed {max(fixed_caps):.4f}")
    check("chosen taper reported within [0, taper_p_max]", 0.0 <= adap["aperture_taper_p"] <= 4.0,
          f"p = {adap['aperture_taper_p']:.2f}")
    # Turning adaptation off pins the configured taper.
    fixed = bl.evaluate(bl.LinkConfig(adaptive_taper=False, taper_p=2.0, **base), 10.0)
    check("adaptive off pins the taper", abs(fixed["aperture_taper_p"] - 2.0) < 1e-9,
          f"p = {fixed['aperture_taper_p']:.2f}")


def test_energy_balance():
    """`flyby`'s energy bookkeeping closes: waste is grid minus delivered, energy_eta is the
    ratio of the two, and the served fraction is a fraction."""
    print("\n[flyby energy balance — plane banks everything]")
    archs = standard_architectures(weather="clear")
    r = sweeps.flyby(archs["relay ×3 mult 5.8→17.4"], station_spacing_km=200.0)
    check("waste = grid − delivered energy", abs(r["waste_gj"] - (r["grid_gj"] - r["delivered_gj"])) < 1e-9,
          f"grid {r['grid_gj']:.2f} − deliv {r['delivered_gj']:.2f} = {r['grid_gj']-r['delivered_gj']:.2f}")
    check("energy η = delivered / grid over served window",
          abs(r["energy_eta"] - r["delivered_gj"] / r["grid_gj"]) < 1e-6,
          f"energy_eta {r['energy_eta']:.3f} vs deliv/grid {r['delivered_gj']/r['grid_gj']:.3f}")
    check("0 ≤ served fraction ≤ 1", 0.0 <= r["served_frac"] <= 1.0, f"{r['served_frac']:.2f}")


def test_hankel_reference():
    """`field`'s Hankel propagator, as the independent ground truth for the coupling closed form.

    It is a separate implementation from `diffraction` (quasi-discrete Hankel transform on a disc
    rather than an FFT/DFT on a square grid), so it is the check that the Goubau law the engine
    sizes apertures with is not just internally consistent. Three things: the propagator conserves
    power, an untruncated Gaussian follows the analytic spreading law, and a truncated aperture's
    captured fraction tracks 1-exp(-tau^2) while sitting BELOW it - truncation can only lose power
    that the closed form assumes arrives, so the sign of that gap is physics, not tolerance.
    """
    print("\n[Hankel ground truth — the coupling law from an independent propagator]")
    lam = relations.wavelength(34.8e9)

    # Untruncated Gaussian: power conserved, width follows w0*sqrt(1+(z/zR)^2).
    grid = fl.HankelGrid(30 * 82.5, 512)
    w0 = 20.0
    zr = np.pi * w0 ** 2 / lam
    start = np.exp(-(grid.radius / w0) ** 2).astype(complex)
    worst_power, worst_width = 0.0, 0.0
    for dist in (0.25 * zr, zr, 3 * zr):
        out = grid.propagate(start, dist, lam)
        worst_power = max(worst_power, abs(grid.power(out) / grid.power(start) - 1.0))
        inten = np.abs(out) ** 2 / np.abs(out[0]) ** 2
        w_fit = np.interp(np.exp(-2.0), inten[::-1], grid.radius[::-1])
        worst_width = max(worst_width, abs(w_fit / (w0 * np.sqrt(1 + (dist / zr) ** 2)) - 1.0))
    check("Hankel propagation conserves power", worst_power < 1e-9, f"max drift {worst_power:.2e}")
    check("untruncated Gaussian follows the analytic spreading law", worst_width < 0.04,
          f"max width error {100*worst_width:.1f}%")

    # Truncated aperture vs Goubau, over the tau range the engine actually sizes into.
    L, r_a = 300e3, 82.5
    for tau_target in (0.5, 1.0, 2.0, 3.0):
        r_b = tau_target * lam * L / (np.pi * r_a)
        g, _, planes, tau, launched = fl.hop_field(r_a, r_b, L, lam, n_planes=2, n_points=1024)
        captured = g.power_within(planes[-1], r_b) / launched
        goubau = relations.collection_efficiency(tau)
        check(f"τ={tau:.1f}: true truncated field tracks Goubau, from below",
              captured <= goubau + 1e-6 and abs(captured - goubau) < 0.03,
              f"field {captured:.4f} vs 1−exp(−τ²) {goubau:.4f}  ({captured-goubau:+.4f})")


def test_relay_channel():
    """The RELAY rendering path stays alive while we iterate on the direct case.

    A relay is an opt-in, so nothing in the default build exercises its two-leg channel — this
    covers it: the frame carries an up-leg and a down-leg whose lengths sum to the total path,
    the two legs really differ (own aperture, focus and wavelength), and the channel field
    resolves intensity in BOTH of them rather than only the leg nearest the emitter.
    """
    print("\n[relay channel — the opt-in two-leg render path]")
    arch = standard_architectures(weather="clear")["relay ×3 mult 5.8→17.4"]
    basis = (geo.point(0.0, 0.0, 0.0), *geo.local_enu(geo.point(0.0, 0.0, 0.0)))
    cv, cf = m.make_bwb_mesh(n_span=41, n_theta=26)
    cn, _ = m.triangle_normals_areas(cv, cf)
    f = frame_physics(arch, -60e3, 11_000.0, basis, (cv, cf, cn))

    legs = f["chan_legs"]
    check("relay frame has two channel legs", len(legs) == 2, f"{len(legs)} legs")
    total = sum(leg["length_m"] for leg in legs)
    check("leg lengths sum to the total path length",
          abs(total - f["chan_total"]) < 1.0, f"{total/1e3:.1f} vs {f['chan_total']/1e3:.1f} km")
    check("the two legs are physically distinct (wavelength and aperture)",
          legs[0]["lam"] != legs[1]["lam"] and legs[0]["ap_major"] != legs[1]["ap_major"],
          f"λ {legs[0]['lam']*100:.1f}/{legs[1]['lam']*100:.1f} cm, "
          f"ap {legs[0]['ap_major']:.0f}/{legs[1]['ap_major']:.0f} m")

    dist = np.linspace(0.0, 1.0, 160)
    radial = np.linspace(-200.0, 200.0, 161)
    ix, _, iy, _ = channel_field(dist, radial, legs, f["chan_total"], f["chan_total"])
    s = dist * f["chan_total"]
    up = np.isfinite(ix[s < legs[0]["length_m"]]).any()
    down = np.isfinite(ix[s > legs[0]["length_m"]]).any()
    check("channel field resolves BOTH legs", bool(up and down), f"up={up} down={down}")


if __name__ == "__main__":
    test_silhouette()
    test_atmosphere()
    test_curvature()
    test_coupling()
    test_end_to_end()
    test_vectorized()
    test_pass_profile()
    test_overlap()
    test_flyby()
    test_pointing_density()
    test_defocus_floor()
    test_taper()
    test_adaptive_taper()
    test_energy_balance()
    test_hankel_reference()
    test_relay_channel()
    print(f"\n{'ALL CHECKS PASSED' if not _failures else 'FAILURES: ' + ', '.join(_failures)}")
    raise SystemExit(1 if _failures else 0)
