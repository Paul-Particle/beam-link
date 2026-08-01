"""First-pass techno-economics: what a beamed flight costs against jet fuel.

Order-of-magnitude throughout, with the levers exposed as arguments. `ETA_BEAM` here is a
flat placeholder; `econ_pass2` is the pass that drives the same primitives from the engine's
computed eta instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .airframe import BEAMED_LIFT_TO_DRAG, mass_buildup, powered_fraction
from .relations import (emitter_diameter_m, eta_grid_to_rf_band, pae_dc_to_rf,
                        rectenna_efficiency_band)

# --- First-pass techno-economics ----------------------------------------------
# The comparison the whole project must beat: jet fuel, unsubsidized, on a
# transatlantic widebody. We compare only the terms that DIFFER — energy source
# and ground infrastructure — because crew, maintenance, navigation, and airport
# fees are ~equal for the same route/aircraft class and cancel in the difference.
# Everything here is #(verify): order-of-magnitude, levers exposed for turning.

def flight_hours(route_km=5500.0, speed_ms=250.0):
    return route_km * 1e3 / speed_ms / 3600.0


# --- efficiency chain ------------------------------------------------------------------

ETA_GRID_TO_RF = 0.55
ETA_BEAM = 0.80       # placeholder for the beam-link engine's computed eta; feeding flyby's
                      # per-pass eta into econ_comparison in its place is the tracked next step
ETA_RECTENNA = 0.80


def delivery_efficiency(eta_grid_to_rf=ETA_GRID_TO_RF, eta_beam=ETA_BEAM,
                        eta_rectenna=ETA_RECTENNA):
    """Grid busbar → rectenna DC output. The onboard motor/drivetrain (~0.85) is already
    folded into cruise_mw, so this stops at the rectenna terminals — do NOT double-count it."""
    return eta_grid_to_rf * eta_beam * eta_rectenna


# --- economics -------------------------------------------------------------------------

def energy_cost_per_flight(cruise_mw, route_km=5500.0, speed_ms=250.0,
                           electricity_usd_per_mwh=30.0, eta_delivery=None):
    """Grid-energy cost to fly one beamed flight. Delivered (drivetrain-input) energy is
    cruise power × block time; grid draw is that over the delivery chain."""
    if eta_delivery is None:
        eta_delivery = delivery_efficiency()
    delivered_mwh = cruise_mw * flight_hours(route_km, speed_ms)
    grid_mwh = delivered_mwh / eta_delivery
    return dict(delivered_mwh=delivered_mwh, grid_mwh=grid_mwh,
                cost_usd=grid_mwh * electricity_usd_per_mwh)


def fuel_cost_per_flight(fuel_tonnes=37.0, fuel_usd_per_tonne=850.0):
    """Fossil comparator's block-fuel cost. 37 t #(verify) ≈ a 787-class widebody over
    5500 km (~5.6 t/h × 6.1 h + climb); Jet-A ~$850/t (volatile: $600-1200)."""
    return dict(fuel_tonnes=fuel_tonnes, cost_usd=fuel_tonnes * fuel_usd_per_tonne)


def station_spacing_km(aircraft_alt_km=11.0, min_elevation_deg=15.0):
    """Along-track window in which one station stays above `min_elevation_deg`, flat earth:
    the aircraft is in view for alt/tan(el) either side of overhead."""
    return 2.0 * aircraft_alt_km / np.tan(np.radians(min_elevation_deg))


def station_count(route_km=5500.0, aircraft_alt_km=11.0, min_elevation_deg=15.0):
    spacing = station_spacing_km(aircraft_alt_km, min_elevation_deg)
    return int(np.ceil(route_km / spacing)) + 1


def capital_recovery_factor(rate=0.08, years=25):
    """Annual payment per unit of capital that amortises it over `years` at `rate`."""
    r = rate
    return r * (1 + r) ** years / ((1 + r) ** years - 1)


# --- Utilization & the boost-glide operating concept --------------------------
# The plane is POWERED ("boosted") only while passing over a station at high
# elevation (good geometry, full aperture), and GLIDES through the low-elevation
# gap between stations — so we never beam at low elevation, and the sawtooth is
# synchronized to station spacing by design. `airborne_per_station` is the target
# packing: ~2 (one boosting + one coasting) is "minimum-slack" operation.

def corridor_throughput_per_day(airborne_per_station, n_stations, transit_h):
    """Flights per day the corridor sustains: aircraft in the air at once, over transit time."""
    return airborne_per_station * n_stations / transit_h * 24.0


def boost_power_mw(cruise_mw, powered_fraction=0.5):
    """Beam power needed if the aircraft is only illuminated for part of a climb-glide cycle
    but must average cruise power over the whole of it."""
    return cruise_mw / powered_fraction


def station_power_mw(airborne_per_station, cruise_mw):
    """Delivered (drivetrain-input) power a station must radiate. It serves (φ·a) planes
    at once, each drawing (cruise/φ) → a·cruise total, INDEPENDENT of the duty φ. At
    minimum-slack utilization the station runs here continuously (capacity factor ≈ 1)."""
    return airborne_per_station * cruise_mw


def station_capex_musd(deliver_mw, frequency_ghz=34.8, aircraft_alt_km=11.0,
                       min_elevation_deg=15.0, rectenna_radius_m=20.0, fixed_musd=50.0,
                       array_musd_per_rf_mw=2.5, supply_musd_per_grid_mw=1.2,
                       cooling_musd_per_waste_mw=1.0, aperture_musd_per_m2=0.005):
    """Cost of one ground station able to deliver `deliver_mw` onboard, working the chain
    backwards to RF and grid power and charging for array, supply, cooling and aperture."""
    eta_rect = rectenna_efficiency_band(frequency_ghz)
    eta_grid_rf = eta_grid_to_rf_band(frequency_ghz)
    rf_mw = deliver_mw / (ETA_BEAM * eta_rect)
    grid_mw = rf_mw / eta_grid_rf
    waste_mw = grid_mw - rf_mw
    slant_km = aircraft_alt_km / np.sin(np.radians(min_elevation_deg))  # worst-case range
    emitter_diam = emitter_diameter_m(frequency_ghz, slant_km, rectenna_radius_m)
    aperture_area_m2 = np.pi * (emitter_diam / 2) ** 2
    musd = (fixed_musd + array_musd_per_rf_mw * rf_mw + supply_musd_per_grid_mw * grid_mw
            + cooling_musd_per_waste_mw * waste_mw + aperture_musd_per_m2 * aperture_area_m2)
    return dict(musd=musd, rf_mw=rf_mw, grid_mw=grid_mw, waste_mw=waste_mw,
                emitter_diam_m=emitter_diam, eta_grid_rf=eta_grid_rf, eta_rect=eta_rect)


def infra_cost_per_flight(n_stations, station_musd, throughput_per_day,
                          rate=0.08, life_years=25, om_fraction=0.03):
    """Ground-infrastructure charge per flight = corridor capex, annualized, over the
    flights the corridor carries. With station cost ∝ its power and station power ∝
    utilization, the POWER-scaling part of this is utilization- and count-invariant (a
    capacity rental); only the FIXED per-station part is reduced by packing planes in."""
    capex_usd = n_stations * station_musd * 1e6
    annual_usd = capex_usd * (capital_recovery_factor(rate, life_years) + om_fraction)
    annual_flights = throughput_per_day * 365.0
    return dict(capex_busd=capex_usd / 1e9, annual_busd=annual_usd / 1e9,
                annual_flights=annual_flights, cost_usd=annual_usd / annual_flights)


def aircraft_cost_per_flight(aircraft_musd, flights_per_year=500.0, rate=0.08,
                             life_years=25, om_fraction=0.0):
    """Ownership charge per flight for one airframe. Beamed airframe is far lighter
    (~80 t vs ~250 t MTOW) but novel/low-volume; fossil widebody ~$150M #(verify)."""
    annual_usd = aircraft_musd * 1e6 * (capital_recovery_factor(rate, life_years) + om_fraction)
    return annual_usd / flights_per_year


def econ_comparison(cruise_mw=None, route_km=5500.0, speed_ms=250.0,
                    electricity_usd_per_mwh=30.0, fuel_usd_per_tonne=850.0,
                    fuel_tonnes=37.0, aircraft_alt_km=11.0, min_elevation_deg=15.0,
                    frequency_ghz=34.8, airborne_per_station=2.0, powered_fraction=0.5,
                    station_fixed_musd=50.0, beamed_aircraft_musd=100.0,
                    fossil_aircraft_musd=150.0, flights_per_year=500.0):
    """Per-flight cost of a beamed corridor against a fossil aircraft, USD.

    Only the terms that differ between the two are counted (energy, aircraft ownership,
    beam infrastructure), so the totals are not full operating costs and only
    `verdict_usd` = beamed - fossil is meaningful.
    """
    if cruise_mw is None:
        cruise_mw = mass_buildup()["cruise_mw"]
    n = station_count(route_km, aircraft_alt_km, min_elevation_deg)
    transit_h = flight_hours(route_km, speed_ms)
    throughput = corridor_throughput_per_day(airborne_per_station, n, transit_h)
    st_deliver_mw = station_power_mw(airborne_per_station, cruise_mw)
    st = station_capex_musd(st_deliver_mw, frequency_ghz=frequency_ghz,
                            aircraft_alt_km=aircraft_alt_km, min_elevation_deg=min_elevation_deg,
                            fixed_musd=station_fixed_musd)
    eta_deliv = st["eta_grid_rf"] * ETA_BEAM * st["eta_rect"]

    energy = energy_cost_per_flight(cruise_mw, route_km, speed_ms, electricity_usd_per_mwh,
                                    eta_delivery=eta_deliv)
    fuel = fuel_cost_per_flight(fuel_tonnes, fuel_usd_per_tonne)
    infra = infra_cost_per_flight(n, st["musd"], throughput)
    beamed_ac = aircraft_cost_per_flight(beamed_aircraft_musd, flights_per_year)
    fossil_ac = aircraft_cost_per_flight(fossil_aircraft_musd, flights_per_year)
    beamed_total = energy["cost_usd"] + infra["cost_usd"] + beamed_ac
    fossil_total = fuel["cost_usd"] + fossil_ac
    return dict(cruise_mw=cruise_mw, n_stations=n, throughput_per_day=throughput,
                frequency_ghz=frequency_ghz, eta_delivery=eta_deliv,
                emitter_diam_m=st["emitter_diam_m"], station_waste_mw=st["waste_mw"],
                station_deliver_mw=st_deliver_mw, boost_mw=boost_power_mw(cruise_mw, powered_fraction),
                station_musd=st["musd"], infra_capex_busd=infra["capex_busd"],
                beamed_energy_usd=energy["cost_usd"], beamed_grid_mwh=energy["grid_mwh"],
                beamed_infra_usd=infra["cost_usd"], beamed_aircraft_usd=beamed_ac,
                beamed_total_usd=beamed_total,
                fossil_fuel_usd=fuel["cost_usd"], fossil_aircraft_usd=fossil_ac,
                fossil_total_usd=fossil_total,
                verdict_usd=beamed_total - fossil_total)


def frequency_tradeoff(frequencies=(5.8, 11.6, 17.4, 23.2, 34.8)):
    """Sweep the master dial: emitter aperture (favours high f) vs PAE/rectenna/thermal
    (favour low f), everything else at the minimum-slack baseline. Shows why the choice
    is a genuine two-sided optimum, not a free win either way."""
    rows = []
    for f in frequencies:
        r = econ_comparison(frequency_ghz=f)
        rows.append((f, round(r["emitter_diam_m"]), round(pae_dc_to_rf(f), 2),
                     round(rectenna_efficiency_band(f), 2), round(r["eta_delivery"], 3),
                     round(r["station_waste_mw"], 1), round(r["station_musd"]),
                     round(r["beamed_energy_usd"]), round(r["verdict_usd"])))
    return pd.DataFrame(rows, columns=["f_GHz", "emitter_m", "PAE", "eta_rect",
                                       "deliv_eff", "waste_MW", "station_$M",
                                       "energy_$", "verdict_$"])


def econ_table():
    """Per-flight cost breakdown at the baseline (minimum-slack) assumptions."""
    e = econ_comparison()
    rows = [
        ("energy: fuel (fossil)", e["fossil_fuel_usd"]),
        ("energy: electricity (beamed)", e["beamed_energy_usd"]),
        ("aircraft ownership (fossil)", e["fossil_aircraft_usd"]),
        ("aircraft ownership (beamed)", e["beamed_aircraft_usd"]),
        (f"beam infra ({e['n_stations']} stn × ${e['station_musd']:.0f}M = "
         f"${e['infra_capex_busd']:.0f}B, {e['throughput_per_day']:.0f}/day)",
         e["beamed_infra_usd"]),
        ("— beamed total (diverging terms)", e["beamed_total_usd"]),
        ("— fossil total (diverging terms)", e["fossil_total_usd"]),
        ("— verdict (beamed − fossil)", e["verdict_usd"]),
    ]
    return pd.DataFrame(rows, columns=["term", "usd_per_flight"]).round(0)


def econ_sensitivity():
    """One-at-a-time swing of the levers around the minimum-slack baseline (verdict
    $/flight; negative = beamed wins). Baseline: 15° elevation, 2 airborne/station,
    $30/MWh, $50M fixed/station. Watch that utilization and elevation now move the
    POWER-scaling infra cost too — the count/utilization cancellation leaves only the
    fixed-per-station part and the capacity rental."""
    base = dict(min_elevation_deg=15.0, airborne_per_station=2.0,
                electricity_usd_per_mwh=30.0, station_fixed_musd=50.0)
    sweeps = {
        "min_elevation_deg": [8, 12, 15, 20, 30],
        "airborne_per_station": [0.5, 1.0, 2.0, 3.0],
        "electricity_usd_per_mwh": [15, 30, 60, 100],
        "station_fixed_musd": [20, 50, 100, 200],
    }
    rows = []
    for lever, values in sweeps.items():
        for v in values:
            kw = dict(base); kw[lever] = v
            r = econ_comparison(**kw)
            rows.append((lever, v, r["n_stations"], round(r["throughput_per_day"]),
                         round(r["verdict_usd"])))
    return pd.DataFrame(rows, columns=["lever", "value", "n_stations", "per_day", "verdict_usd"])
