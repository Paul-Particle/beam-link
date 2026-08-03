"""Cost primitives: money, and nothing else.

Order-of-magnitude throughout, with every lever an argument. What is NOT here is the point: no
efficiency, no aperture sizing, no station spacing, no cruise power. Those are physics, and econ may
not compute them (I4) — it selects them from products.

Three things the pre-SPEC econ did that this cannot:

  `ETA_BEAM = 0.80` as a reachable default. 0.55 × 0.80 × 0.80 = 0.352 against the engine's ~0.21,
  biased in favour of the technology, and reachable simply by not passing the engine's number
  (§8 item 22). There is no beam efficiency in this module at all; it arrives as an argument.

  Sizing the aperture. `econ_pass2` inverted the shaped-spot capture itself to find the smallest
  elliptical diameter that matched a round one — a second modelling pass, in the cost layer. That is
  now a sweep over a declared design parameter against a named aggregate, which is a driver's job
  (D3).

  Station spacing from altitude and minimum elevation. That is geometry. The driver decides the
  corridor layout using `physics.geo` and records it in the study manifest; econ reads the count.

Everything is vectorized over a candidate axis, because the engine is and a search loop alternating
between the two would otherwise be bottlenecked here (§6a).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- the comparison -----------------------------------------------------------------------
# The comparison the whole project must beat: jet fuel, unsubsidised, on a transatlantic widebody.
# Only the terms that DIFFER are compared — energy source and ground infrastructure — because crew,
# maintenance, navigation and airport fees are ~equal for the same route and aircraft class and cancel
# in the difference. Every number here is #(verify): order-of-magnitude, levers exposed for turning.


@dataclass(frozen=True)
class EconConfig:
    """Every economic assumption, in one declared object.

    A cost model with defaults scattered across module constants is a cost model whose assumptions
    cannot be listed, which is the same defect as an unnamed derived quantity.
    """

    electricity_usd_per_mwh: float = 30.0
    route_km: float = 5500.0

    fossil_fuel_tonnes: float = 37.0
    """Block fuel for a 787-class widebody over 5500 km (~5.6 t/h × 6.1 h plus climb). #(verify)"""

    fuel_usd_per_tonne: float = 850.0
    """Jet-A. Volatile: $600–1200."""

    airborne_per_station: float = 1.0
    """Aircraft in the air per station — the corridor packing. ~2 (one boosting, one coasting) is
    minimum-slack operation; 1 is the conservative low-density case."""

    discount_rate: float = 0.08
    life_years: int = 25
    station_om_fraction: float = 0.03

    station_fixed_musd: float = 50.0
    station_musd_per_rf_mw: float = 2.5
    station_musd_per_grid_mw: float = 1.2
    station_musd_per_waste_mw: float = 1.0
    station_musd_per_aperture_m2: float = 0.005

    beamed_aircraft_musd: float = 100.0
    """Far lighter than a fossil widebody (~80 t against ~250 t MTOW) but novel and low-volume."""

    fossil_aircraft_musd: float = 150.0
    flights_per_year_per_aircraft: float = 500.0


def flight_hours(route_km, speed_ms):
    """Block time. `speed_ms` comes from P1's own ground speed, not from an econ constant."""
    return np.asarray(route_km, dtype=float) * 1e3 / np.asarray(speed_ms, dtype=float) / 3600.0


def capital_recovery_factor(rate, years):
    """Annual payment per unit of capital that amortises it over `years` at `rate`."""
    rate = np.asarray(rate, dtype=float)
    return rate * (1 + rate) ** years / ((1 + rate) ** years - 1)


def energy_cost_per_flight(required_w, route_km, speed_ms, pass_energy_efficiency,
                           electricity_usd_per_mwh):
    """Grid-energy cost to fly one beamed flight.

    `required_w` is the aircraft's requirement AT THE RECTENNA TERMINALS (P12), and
    `pass_energy_efficiency` is grid busbar to the same plane (P9) — so the two compose without a
    drivetrain term, and there is nowhere to double-count one (D5).
    """
    hours = flight_hours(route_km, speed_ms)
    delivered_mwh = np.asarray(required_w, dtype=float) / 1e6 * hours
    efficiency = np.maximum(np.asarray(pass_energy_efficiency, dtype=float), 1e-9)
    grid_mwh = delivered_mwh / efficiency
    return dict(block_hours=hours, delivered_mwh=delivered_mwh, grid_mwh=grid_mwh,
                cost_usd=grid_mwh * electricity_usd_per_mwh)


def fuel_cost_per_flight(config: EconConfig):
    """The fossil comparator's block-fuel cost."""
    return dict(fuel_tonnes=config.fossil_fuel_tonnes,
                cost_usd=config.fossil_fuel_tonnes * config.fuel_usd_per_tonne)


def station_power(delivered_w, pass_energy_efficiency, grid_to_rf):
    """What one station must draw and radiate to serve its share of the corridor.

    It serves (duty × airborne) aircraft at once, each drawing (requirement / duty), so the total is
    airborne × requirement — INDEPENDENT of the duty cycle. At minimum-slack utilisation the station
    runs here continuously.

    `grid_to_rf` is read off P7's own factor of that name, not re-derived from a band model: the
    station-side split is physics and econ selects it.
    """
    delivered = np.asarray(delivered_w, dtype=float)
    efficiency = np.maximum(np.asarray(pass_energy_efficiency, dtype=float), 1e-9)
    grid = delivered / efficiency
    radiated = grid * np.asarray(grid_to_rf, dtype=float)
    return dict(delivered_w=delivered, grid_w=grid, rf_w=radiated, waste_w=grid - radiated)


def station_capex_musd(power, aperture_area_m2, config: EconConfig):
    """One station's capital cost, from its power and its aperture area.

    Both come from products: the power from P8/P9 and the area from P4's emitted semi-axes. The
    aperture is where the elliptical shaping shows up in the cost — a shaped array of the same capture
    needs roughly a third of the round one's AREA, which is a physics result the cost model consumes
    rather than reproduces.
    """
    return (config.station_fixed_musd
            + config.station_musd_per_rf_mw * power["rf_w"] / 1e6
            + config.station_musd_per_grid_mw * power["grid_w"] / 1e6
            + config.station_musd_per_waste_mw * power["waste_w"] / 1e6
            + config.station_musd_per_aperture_m2 * np.asarray(aperture_area_m2, dtype=float))


def corridor_throughput_per_day(airborne_per_station, n_stations, block_hours):
    """Flights per day the corridor sustains: aircraft in the air at once, over transit time."""
    return (np.asarray(airborne_per_station, dtype=float) * np.asarray(n_stations, dtype=float)
            / np.asarray(block_hours, dtype=float) * 24.0)


def infrastructure_cost_per_flight(n_stations, station_musd, throughput_per_day,
                                   config: EconConfig):
    """Ground-infrastructure charge per flight: corridor capex, annualised, over the flights it
    carries.

    With station cost ∝ its power and station power ∝ utilisation, the POWER-scaling part of this is
    utilisation- and count-invariant — a capacity rental. Only the FIXED per-station part is reduced
    by packing aircraft in, which is why `station_fixed_musd` is the lever that responds to density.
    """
    capex_usd = np.asarray(n_stations, dtype=float) * np.asarray(station_musd, dtype=float) * 1e6
    annual = capex_usd * (capital_recovery_factor(config.discount_rate, config.life_years)
                          + config.station_om_fraction)
    annual_flights = np.asarray(throughput_per_day, dtype=float) * 365.0
    return dict(capex_busd=capex_usd / 1e9, annual_busd=annual / 1e9,
                annual_flights=annual_flights,
                cost_usd=annual / np.maximum(annual_flights, 1e-9))


def aircraft_cost_per_flight(aircraft_musd, config: EconConfig, om_fraction: float = 0.0):
    """Ownership charge per flight for one airframe."""
    annual = (np.asarray(aircraft_musd, dtype=float) * 1e6
              * (capital_recovery_factor(config.discount_rate, config.life_years) + om_fraction))
    return annual / config.flights_per_year_per_aircraft
