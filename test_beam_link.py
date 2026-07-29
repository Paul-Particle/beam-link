"""Physics-review self-tests for the beam-link engine.

Each check encodes a physical expectation and traces one link in the chain, so a
green run is evidence the engine behaves the way the physics says it should — not
just that it runs. Run:  uv run python test_beam_link.py
"""
import numpy as np

import napkin
import geo
import beam_link as bl
from bwb_model import BWBSilhouette, beta_alpha_from_enu, projected_area, make_bwb_mesh, \
    triangle_normals_areas, direction_from_beta_alpha

PASS, FAIL = "  ok ", "FAIL "
_failures = []


def check(name, condition, detail=""):
    print(f"{PASS if condition else FAIL} {name}{('  — ' + detail) if detail else ''}")
    if not condition:
        _failures.append(name)


# --- Silhouette ---------------------------------------------------------------

def test_silhouette():
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
    st, pl = bl.reference_geometry(30.0)
    r58 = bl.evaluate(bl.LinkConfig(station_pos=st, plane_pos=pl, f_emit_plane_ghz=5.8,
                                    weather="heavy_rain"), 10.0)["atmos"]
    r30 = bl.evaluate(bl.LinkConfig(station_pos=st, plane_pos=pl, f_emit_plane_ghz=30.0,
                                    weather="heavy_rain"), 10.0)["atmos"]
    check("5.8 GHz survives heavy rain, 30 GHz does not", r58 > 0.7 and r30 < 0.05,
          f"5.8→{float(r58):.3f}  30→{float(r30):.3f}")


def test_curvature():
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
    print("\n[near-field coupling]")
    # τ=2 → the design capture ceiling ~0.98.
    check("collection η(τ=2) ≈ 0.982", abs(napkin.collection_efficiency(2.0) - 0.9817) < 1e-3)

    # Sizing to τ=2 at any frequency yields the same capture; the differentiator is
    # emitter DIAMETER, which scales as λ (∝ 1/f).
    st, pl = bl.reference_geometry(30.0)
    d58 = bl.evaluate(bl.LinkConfig(station_pos=st, plane_pos=pl, f_emit_plane_ghz=5.8), 10.0)["emitter_diam_m"]
    d30 = bl.evaluate(bl.LinkConfig(station_pos=st, plane_pos=pl, f_emit_plane_ghz=30.0), 10.0)["emitter_diam_m"]
    ratio = float(d58) / float(d30)
    check("emitter Ø scales ~as 1/frequency", abs(ratio - 30.0 / 5.8) / (30.0 / 5.8) < 0.05,
          f"Ø 5.8={float(d58):.0f} m, 30={float(d30):.0f} m, ratio {ratio:.2f} vs {30/5.8:.2f}")

    # Lower elevation → longer slant + smaller silhouette → bigger emitter.
    d_high = bl.evaluate(bl.LinkConfig(*bl.reference_geometry(45.0), f_emit_plane_ghz=30.0), 10.0)["emitter_diam_m"]
    d_low = bl.evaluate(bl.LinkConfig(*bl.reference_geometry(10.0), f_emit_plane_ghz=30.0), 10.0)["emitter_diam_m"]
    check("low elevation needs a bigger emitter", float(d_low) > float(d_high),
          f"45°→{float(d_high):.0f} m, 10°→{float(d_low):.0f} m")


# --- End-to-end bookkeeping ---------------------------------------------------

def test_end_to_end():
    print("\n[end-to-end]")
    st, pl = bl.reference_geometry(30.0)
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
    print("\n[pass profile / fixed emitter]")
    # A fixed emitter delivers most overhead and falls off both ways symmetrically.
    p = bl.pass_profile(frequency_ghz=30.0, reference_elevation_deg=60.0)
    peak_idx = int(np.argmax(p["deliverable_mw"]))
    check("peak delivery is overhead (x≈0)", abs(p["alongtrack_km"][peak_idx]) < 2.0,
          f"peak at {p['alongtrack_km'][peak_idx]:.1f} km")
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
        pp = bl.pass_profile(frequency_ghz=30.0, reference_elevation_deg=ref)
        norm = pp["deliverable_mw"] / pp["deliverable_mw"].max()
        xin = pp["alongtrack_km"][norm >= 0.5]
        return float(xin.max() - xin.min()), pp["emitter_diam_m"]
    w60, d60 = window_km(60.0)
    w20, d20 = window_km(20.0)
    check("bigger emitter → wider powering window", w20 > w60 and d20 > d60,
          f"Ø{d60:.0f}m→{w60:.0f}km vs Ø{d20:.0f}m→{w20:.0f}km")

    # Fixed emitter: capture DROPS off-overhead (vs sized-to-τ which holds ~0.98).
    check("fixed-emitter capture falls off-overhead",
          p["capture"][peak_idx] > 0.95 and np.min(p["capture"]) < 0.3)


if __name__ == "__main__":
    test_silhouette()
    test_atmosphere()
    test_curvature()
    test_coupling()
    test_end_to_end()
    test_vectorized()
    test_pass_profile()
    print(f"\n{'ALL CHECKS PASSED' if not _failures else 'FAILURES: ' + ', '.join(_failures)}")
    raise SystemExit(1 if _failures else 0)
