"""Cost primitives for comparing a beamed flight against jet fuel.

Order-of-magnitude throughout, with the levers exposed as arguments. These are primitives
only: `econ_pass2` is the driver that assembles them, sizing the elliptical aperture and
taking the delivery eta from the engine's own flyby rather than a flat placeholder.
"""

from __future__ import annotations

import numpy as np

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
ETA_BEAM = 0.80       # flat placeholder, and only a fallback: pass the engine's per-pass eta in
                      # as `eta_delivery` (which is what econ_pass2 does) to bypass it
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


def station_power_mw(airborne_per_station, cruise_mw):
    """Delivered (drivetrain-input) power a station must radiate. It serves (φ·a) planes
    at once, each drawing (cruise/φ) → a·cruise total, INDEPENDENT of the duty φ. At
    minimum-slack utilization the station runs here continuously (capacity factor ≈ 1)."""
    return airborne_per_station * cruise_mw


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
