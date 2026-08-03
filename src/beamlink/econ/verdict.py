"""The verdict: does beamed beat fossil, on numbers a run already produced.

This is the whole of econ's contact with the physics, and it is SELECTION (§7). Every physical
quantity is read from a named product, and the code below does no arithmetic that is not money:

    pass efficiency        P9 `pass_energy_efficiency` — grid busbar to rectenna terminals, with its
                           span, serve policy and statistic carried alongside it
    power required         P12 `required_w` at `rectenna_terminals`, the same plane
    grid→RF split          P7's own factor of that name
    aperture area          P4's emitted semi-axes
    block speed            P1's ground speed
    corridor layout        the study manifest's parameters, because it spans runs (D6)

The test is the conservative one: does beamed beat fossil at LOW density (few aircraft, so low
utilisation) and LONG range (few, widely spaced stations)?

Everything comes back vectorized over the candidate axis, and `refuse` is a first-class outcome:
pointed at a run that lacks a product it was told to use, econ raises rather than substituting or
recomputing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from beamlink import artifacts
from beamlink.econ import cost
from beamlink.econ.cost import EconConfig

REQUIRED_PRODUCTS = ("aggregates", "flight_power", "chain", "aperture", "trajectory", "mass")


@dataclass(frozen=True)
class Selection:
    """The physical numbers econ selected, kept separately from the money it made of them.

    Emitted so a figure can show WHAT econ read as well as what it concluded — which is the whole
    remedy for §1's failure mode, where a page presented one modelling pass and the numbers came from
    another.
    """

    pass_energy_efficiency: np.ndarray
    efficiency_definition: artifacts.AggregateDefinition
    required_w: np.ndarray
    grid_to_rf: np.ndarray
    aperture_area_m2: np.ndarray
    block_speed_ms: float
    mtow_t: np.ndarray
    n_stations: int
    station_spacing_km: float
    run_id: str
    engine_sha: str
    efficiency_product_tier: str


def select(products: artifacts.Products, manifest: artifacts.Manifest, n_stations: int,
           station_spacing_km: float,
           efficiency_aggregate: str = "pass_energy_efficiency") -> Selection:
    """Read what econ needs off one run, refusing loudly if it is not there."""
    artifacts.require(products, manifest, *REQUIRED_PRODUCTS)
    entry = products.aggregates.entries.get(efficiency_aggregate)
    if entry is None:
        raise artifacts.RunIncomplete(
            f"run {manifest.run_id} has no aggregate named {efficiency_aggregate!r}; it produced "
            f"{sorted(products.aggregates.entries)}. A differently-defined efficiency is a new named "
            f"engine product, not a local calculation (I4).")
    factor = products.chain.factor_names.index("grid_to_rf")
    # The station-side split varies per sample only through nothing at all, but it is read rather than
    # assumed constant, so a scenario that made it vary would not silently be averaged away.
    grid_to_rf = np.mean(products.chain.factors[:, factor, :], axis=1)
    area = np.pi * np.mean(products.aperture.semi_major_m * products.aperture.semi_minor_m, axis=1)
    return Selection(
        pass_energy_efficiency=entry.value, efficiency_definition=entry.definition,
        required_w=np.mean(products.flight_power.required_w, axis=1), grid_to_rf=grid_to_rf,
        aperture_area_m2=area,
        block_speed_ms=float(np.median(products.trajectory.ground_speed_ms)),
        mtow_t=products.mass.mtow_t, n_stations=int(n_stations),
        station_spacing_km=float(station_spacing_km), run_id=manifest.run_id,
        engine_sha=manifest.engine_sha,
        efficiency_product_tier=manifest.status("aggregates").tier)


def verdict(selection: Selection, config: EconConfig | None = None) -> dict:
    """Cost per flight, beamed against fossil, per candidate. Negative `verdict_usd` means beamed wins.

    The station is sized for the corridor's demand, not for one aircraft: `airborne_per_station`
    aircraft each need `required_w`, so the station delivers their sum.
    """
    config = config or EconConfig()
    delivered = config.airborne_per_station * selection.required_w
    power = cost.station_power(delivered, selection.pass_energy_efficiency, selection.grid_to_rf)
    station_musd = cost.station_capex_musd(power, selection.aperture_area_m2, config)

    energy = cost.energy_cost_per_flight(selection.required_w, config.route_km,
                                         selection.block_speed_ms,
                                         selection.pass_energy_efficiency,
                                         config.electricity_usd_per_mwh)
    throughput = cost.corridor_throughput_per_day(config.airborne_per_station,
                                                  selection.n_stations, energy["block_hours"])
    infrastructure = cost.infrastructure_cost_per_flight(selection.n_stations, station_musd,
                                                         throughput, config)
    beamed_ownership = cost.aircraft_cost_per_flight(config.beamed_aircraft_musd, config)
    fossil_fuel = cost.fuel_cost_per_flight(config)["cost_usd"]
    fossil_ownership = cost.aircraft_cost_per_flight(config.fossil_aircraft_musd, config)

    beamed_total = energy["cost_usd"] + infrastructure["cost_usd"] + beamed_ownership
    fossil_total = fossil_fuel + fossil_ownership
    return dict(
        selection=selection, station_musd=station_musd, station_grid_mw=power["grid_w"] / 1e6,
        station_rf_mw=power["rf_w"] / 1e6, block_hours=energy["block_hours"],
        grid_mwh_per_flight=energy["grid_mwh"], energy_usd=energy["cost_usd"],
        infrastructure_usd=infrastructure["cost_usd"], corridor_capex_busd=infrastructure["capex_busd"],
        throughput_per_day=throughput, beamed_aircraft_usd=np.full_like(
            np.asarray(selection.required_w, dtype=float), beamed_ownership),
        beamed_total_usd=beamed_total, fossil_total_usd=np.full_like(beamed_total, fossil_total),
        verdict_usd=beamed_total - fossil_total, beamed_wins=beamed_total < fossil_total)


def verdict_from_run(run_dir, n_stations: int, station_spacing_km: float,
                     config: EconConfig | None = None,
                     efficiency_aggregate: str = "pass_energy_efficiency") -> dict:
    """The one-call path for a consumer: point it at a run directory."""
    products, manifest = artifacts.load_run(run_dir)
    selection = select(products, manifest, n_stations, station_spacing_km, efficiency_aggregate)
    return verdict(selection, config)


def provenance(selection: Selection) -> dict:
    """What a page must print in the margin next to any of these numbers (§7).

    Not a nicety: the efficiency's DEFINITION is what distinguishes the two numbers of §1, so a figure
    that shows a cost without it is showing a cost whose meaning cannot be recovered.
    """
    definition = selection.efficiency_definition
    return {
        "run": selection.run_id,
        "engine_sha": selection.engine_sha[:12],
        "aggregates_tier": selection.efficiency_product_tier,
        "efficiency": definition.quantity,
        "efficiency_statistic": definition.statistic,
        "efficiency_span": definition.span,
        "serve_policy": f"{definition.serve_policy}"
                        + (f" >= {definition.serve_eta_floor:g}"
                           if definition.serve_policy == "eta_floor" else ""),
        "weighting": definition.weighting,
        "corridor": f"{selection.n_stations} stations at "
                    f"{selection.station_spacing_km:.0f} km spacing",
    }
