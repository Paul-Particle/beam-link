"""Second econ pass: the low-density / long-range VERDICT with the real physics wired in.

This is the only econ driver. It replaced a first-pass comparison (now deleted, in git history)
that got two things wrong for this test:
  (1) it sized a ROUND emitter (`emitter_diameter_m`) — the BWB silhouette is ~4.3:1, so a
      shaped (phased-array) elliptical spot needs ~1/3 the aperture AREA for the same capture;
  (2) it used a FLAT `ETA_BEAM = 0.80` for the beam — the beam-link engine computes the real
      delivery η (weather, self-occluded silhouette, elliptical spot, incidence) over the pass.

Here we size the ELLIPTICAL aperture by inverting the shaped-spot capture at the design edge
(min-elevation, τ≈2-equivalent → ~90% capture), run a beam-link `flyby` over the served window
to get the real `energy_eta`, and feed BOTH (aperture area + η) into the `econ` cost primitives.
The test is the conservative one we agreed on: does beamed beat fossil at LOW density (few
planes → low utilization) and LONG range (few, widely spaced stations)?  Sweeps frequency ×
elevation × weather.  `station_musd_elliptical` below is now the single station cost model.

Run:  PYTHONPATH=$PWD uv run python econ_pass2.py
"""
import numpy as np

from beamlink import beam_link as bl
from beamlink import bwb_model as m
from beamlink import geo
from beamlink import airframe
from beamlink import econ
from beamlink import relations

ALT_M = 11_000.0
ROUTE_KM = 5500.0
SPEED_MS = 250.0
CAPTURE_TARGET = 0.90          # elliptical-aperture sizing target at the design edge (τ≈2-ish)
DELIVERED_MW_CEIL = 10.0       # plane absorption ceiling for the flyby FoM (η is ratio, ceiling cancels)

# Station-capex coefficients. These began as a mirror of a round-aperture cost model in `econ`;
# that one is gone, so this is the only copy and there is nothing to keep in sync.
FIXED_MUSD = 50.0
ARRAY_MUSD_PER_RF_MW = 2.5
SUPPLY_MUSD_PER_GRID_MW = 1.2
COOLING_MUSD_PER_WASTE_MW = 1.0
APERTURE_MUSD_PER_M2 = 0.005

mb = airframe.mass_buildup(payload_t=40.0)
CRUISE_MW = mb["cruise_mw"]

# --- static geometry helpers -------------------------------------------------
_basis = None
_coarse = None


def _edge_raster(freq_ghz, min_elev_deg):
    """A silhouette raster at the design edge (plane at min elevation, overhead pass)."""
    x_edge = geo.downrange_for_elevation(min_elev_deg, ALT_M)
    plane = np.array([x_edge, 0.0, ALT_M])
    station = np.array([0.0, 0.0, 0.0])
    slant, elev, unit = bl.geometry(plane, station)
    beta, alpha = m.beta_alpha_from_enu(unit, 0.0, 0.0)      # level plane, zero heading
    raster = bl._cached_raster(beta, alpha)
    lam = bl.relations.SPEED_OF_LIGHT / (freq_ghz * 1e9)
    r_recv = bl.receiver_radius_from_silhouette(raster["area"])
    return raster, float(slant), lam, float(r_recv)


def size_apertures(freq_ghz, min_elev_deg):
    """Round τ≈2 design diameter, and the smaller SHAPED (elliptical) diameter that matches
    its capture. Capture-vs-aperture is unimodal (over-large aperture → tiny over-focused
    spot), so we anchor on the round τ≈2 design and scan DOWN for the smallest shaped D that
    still matches its capture. Both cases use the diffraction beam at a uniform taper (p=0) as a
    STABLE sizing reference; the engine itself now optimises the aperture taper per geometry, so
    the aperture sizing here needs its own reconciliation pass against it. #(verify)"""
    raster, slant, lam, r_recv = _edge_raster(freq_ghz, min_elev_deg)
    D_round = relations.emitter_diameter_m(freq_ghz, slant / 1e3, r_recv)
    eff_round = bl.overlap_capture(raster, D_round / 2.0, lam, slant, shaped=False,
                                   taper_p=0.0, incidence_floor=0.90)[1]
    D_ellip = D_round
    for D in np.linspace(0.2 * D_round, D_round, 50):         # smallest shaped D ≥ round capture
        e = bl.overlap_capture(raster, D / 2.0, lam, slant, shaped=True,
                               taper_p=0.0, incidence_floor=0.90)[1]
        if e >= eff_round:
            D_ellip = D
            break
    return float(D_ellip), float(D_round), float(eff_round)


def delivery_eta(freq_ghz, diam_m, min_elev_deg, weather):
    """Real grid→drivetrain delivery η over the served window, from a beam-link flyby."""
    spacing_km = econ.station_spacing_km(ALT_M / 1e3, min_elev_deg)
    arch = dict(station_pos=(0.0, 0.0, 0.0), relay_pos=None, f_emit_plane_ghz=freq_ghz,
                emitter_diam_m=diam_m, shaped_spot=True, weather=weather)
    fb = bl.flyby(arch, plane_alt_m=ALT_M, station_spacing_km=spacing_km,
                  speed_ms=SPEED_MS, delivered_mw=DELIVERED_MW_CEIL)
    return fb["energy_eta"], fb["served_frac"]


def station_musd_elliptical(deliver_mw, energy_eta, freq_ghz, aperture_area_m2):
    """Station cost with the elliptical aperture and beam-link η. grid draw from the real η;
    station-side grid→RF split from napkin's band model."""
    eta_grid_rf = relations.eta_grid_to_rf_band(freq_ghz)
    grid_mw = deliver_mw / max(energy_eta, 1e-6)
    rf_mw = grid_mw * eta_grid_rf
    waste_mw = grid_mw - rf_mw
    musd = (FIXED_MUSD + ARRAY_MUSD_PER_RF_MW * rf_mw + SUPPLY_MUSD_PER_GRID_MW * grid_mw
            + COOLING_MUSD_PER_WASTE_MW * waste_mw + APERTURE_MUSD_PER_M2 * aperture_area_m2)
    return musd, grid_mw


MIN_FEASIBLE_ETA = 0.02       # below this the link is rained/spilled out → not an operating case
MIN_FEASIBLE_SERVED = 0.5     # must serve most of the window to bank a pass's energy


def verdict(freq_ghz, min_elev_deg, weather, airborne_per_station):
    D, D_round, eff_round = size_apertures(freq_ghz, min_elev_deg)
    eta, served = delivery_eta(freq_ghz, D, min_elev_deg, weather)
    n_st = econ.station_count(ROUTE_KM, ALT_M / 1e3, min_elev_deg)
    spacing = econ.station_spacing_km(ALT_M / 1e3, min_elev_deg)
    feasible = (eta >= MIN_FEASIBLE_ETA) and (served >= MIN_FEASIBLE_SERVED)
    if not feasible:                                          # rained/spilled out — no verdict
        return dict(D=D, D_round=D_round, eta=eta, served=served, n_st=n_st, spacing=spacing,
                    feasible=False, st_musd=None, grid_mw=None, infra=None, energy=None, verdict=None)
    transit_h = econ.flight_hours(ROUTE_KM, SPEED_MS)
    throughput = econ.corridor_throughput_per_day(airborne_per_station, n_st, transit_h)
    deliver_mw = econ.station_power_mw(airborne_per_station, CRUISE_MW)
    area = np.pi * (D / 2.0) ** 2
    st_musd, grid_mw = station_musd_elliptical(deliver_mw, eta, freq_ghz, area)
    infra = econ.infra_cost_per_flight(n_st, st_musd, throughput)
    energy = econ.energy_cost_per_flight(CRUISE_MW, ROUTE_KM, SPEED_MS, eta_delivery=eta)
    beamed_ac = econ.aircraft_cost_per_flight(100.0)
    fossil = econ.fuel_cost_per_flight()["cost_usd"] + econ.aircraft_cost_per_flight(150.0)
    beamed_total = energy["cost_usd"] + infra["cost_usd"] + beamed_ac
    return dict(D=D, D_round=D_round, eta=eta, served=served, n_st=n_st, spacing=spacing,
                feasible=True, throughput=throughput, st_musd=st_musd, grid_mw=grid_mw,
                infra=infra["cost_usd"], energy=energy["cost_usd"], verdict=beamed_total - fossil)


if __name__ == "__main__":
    print(f"MTOW {mb['mtow_t']:.0f} t  cruise {CRUISE_MW:.1f} MW   "
          f"fossil ref: fuel ${econ.fuel_cost_per_flight()['cost_usd']:.0f} + "
          f"a/c ${econ.aircraft_cost_per_flight(150.0):.0f}\n")
    A = 1.0     # LOW density: one plane per station (conservative)
    for weather in ("clear", "heavy_rain"):
        print(f"################  weather = {weather}   (airborne/station = {A})  ################")
        print(f"{'f_GHz':>6} {'elev':>5} {'slant':>6} {'spacing':>7} {'n_st':>5} {'Dround':>7} {'Dellip':>7} "
              f"{'eta%':>6} {'srv%':>5} {'grid_MW':>7} {'st_$M':>7} {'infra$':>9} {'energy$':>8} {'verdict$':>11}")
        for f in (5.8, 17.4, 34.8):
            for elev in (6, 12, 20):
                r = verdict(f, elev, weather, A)
                slant = ALT_M / 1e3 / np.sin(np.radians(elev))
                if not r["feasible"]:
                    print(f"{f:6.1f} {elev:5d} {slant:6.0f} {r['spacing']:7.0f} {r['n_st']:5d} "
                          f"{r['D_round']:7.0f} {r['D']:7.0f} {r['eta']*100:6.1f} {r['served']*100:5.0f} "
                          f"{'':>7} {'':>7} {'':>9} {'':>8} {'  RAINS OUT':>11}")
                    continue
                win = "WIN" if r["verdict"] < 0 else "lose"
                print(f"{f:6.1f} {elev:5d} {slant:6.0f} {r['spacing']:7.0f} {r['n_st']:5d} "
                      f"{r['D_round']:7.0f} {r['D']:7.0f} {r['eta']*100:6.1f} {r['served']*100:5.0f} "
                      f"{r['grid_mw']:7.1f} {r['st_musd']:7.0f} {r['infra']:9.0f} {r['energy']:8.0f} "
                      f"{r['verdict']:11.0f} {win}")
        print()
