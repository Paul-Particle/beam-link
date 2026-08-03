"""Corridor-level numbers: energy per flight over a route served by N stations.

D6: route- and corridor-level numbers span several engine runs, and they are exactly what econ
consumes. The driver owns them. It uses the same reduction machinery the engine's P9 uses and records
the same way — a study manifest naming the runs it aggregated and the aggregation definition it
applied — so "who derived this number and how" is answerable for a corridor figure exactly as it is
for a pass figure, and econ still only selects.

The corridor LAYOUT is also the driver's: how far apart the stations sit follows from the aircraft's
altitude and the lowest elevation worth serving, which is geometry. The pre-SPEC econ computed it
itself, flat-earth, from a minimum elevation — econ may not (I4), and it does not have to, because the
driver can ask `physics.geo` and put the answer in the study manifest for econ to read.

One run per station is not needed for a uniform corridor — every station sees the same relative pass —
but the machinery is written for the general case, because a real corridor has stations at different
altitudes and under different weather, and that is several runs by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from beamlink import artifacts, physics
from beamlink.artifacts.spec import AggregateDefinition, RunSpec, TrajectorySpec
from beamlink.physics import aggregate, geo

DRIVER_NAME = "drivers.corridor"


@dataclass(frozen=True)
class CorridorLayout:
    """How the stations are placed. Geometry, so it comes from `physics.geo`, not from econ."""

    station_spacing_km: float
    n_stations: int
    min_elevation_deg: float
    altitude_m: float
    route_km: float

    def serving_half_span_m(self) -> float:
        """Half the station spacing: the along-track window one station owns.

        This is what a run's trajectory span MUST be for its pass aggregates to mean "one station's
        share of the corridor". Three pre-SPEC modules had three different spans, which is how §1's
        two pass efficiencies came to disagree — so here the span is derived from the layout and
        handed to the run, rather than each layer choosing its own.
        """
        return self.station_spacing_km * 1e3 / 2.0


def layout(altitude_m: float = 11_000.0, min_elevation_deg: float = 15.0,
           route_km: float = 5500.0) -> CorridorLayout:
    """Station spacing on the real spherical earth, from the lowest elevation worth serving.

    `geo.downrange_for_elevation` inverts the true geometry, where the pre-SPEC econ used
    2·alt/tan(elevation) — a flat-earth approximation that is 4 % short at 15° and 11 km, and worse
    as the elevation falls, which is exactly where the corridor's economics are decided.
    """
    half_span = float(geo.downrange_for_elevation(min_elevation_deg, altitude_m))
    spacing_km = 2.0 * half_span / 1e3
    return CorridorLayout(station_spacing_km=spacing_km,
                          n_stations=int(np.ceil(route_km / spacing_km)) + 1,
                          min_elevation_deg=min_elevation_deg, altitude_m=altitude_m,
                          route_km=route_km)


def station_spec(base: RunSpec, corridor: CorridorLayout, speed_ms: float = 250.0,
                 n_samples: int = 41, run_id: str = "station") -> RunSpec:
    """One station's run: its own serving window as the trajectory span."""
    from dataclasses import replace
    trajectory = TrajectorySpec.straight_overflight(
        half_span_m=corridor.serving_half_span_m(), altitude_m=corridor.altitude_m,
        speed_ms=speed_ms, n_samples=n_samples,
        name=f"station_window_{corridor.station_spacing_km:.0f}km")
    return replace(base, trajectory=trajectory, run_id=run_id)


# --- the cross-run reduction ---------------------------------------------------------------


def route_definitions(corridor: CorridorLayout) -> dict[str, AggregateDefinition]:
    """What the driver derives ACROSS runs, stated the same way P9 states what it derives within one.

    The span is the route, not a pass, and that difference is the whole reason these are separate
    named products rather than a rescaled pass number.
    """
    span = (f"route {corridor.route_km:.0f} km, {corridor.n_stations} stations at "
            f"{corridor.station_spacing_km:.0f} km")
    return {
        "route_delivered_energy_j": AggregateDefinition(
            quantity="aggregates.delivered_energy_j", statistic="time_integral", span=span,
            serve_policy="always", weighting="sample", inputs=("aggregates",), unit="J",
            note="Sum of each station's per-pass delivered energy over the stations the route "
                 "crosses. A SUM over runs, not a mean: the aircraft is served by each station in "
                 "turn, so the route's energy is the total of the passes."),
        "route_grid_energy_j": AggregateDefinition(
            quantity="aggregates.grid_energy_j", statistic="time_integral", span=span,
            serve_policy="always", weighting="sample", inputs=("aggregates",), unit="J",
            note="The same sum for grid energy drawn."),
        "route_energy_efficiency": AggregateDefinition(
            quantity="aggregates.delivered_energy_j/aggregates.grid_energy_j",
            statistic="energy_ratio", span=span, serve_policy="always", weighting="sample",
            inputs=("aggregates",), unit="1",
            note="Route delivered energy over route grid energy. This is what econ consumes for a "
                 "corridor, and it is NOT the mean of the per-pass efficiencies: it is weighted by "
                 "each pass's energy, which is what makes it the number that closes the balance."),
        "route_worst_pass_efficiency": AggregateDefinition(
            quantity="aggregates.pass_energy_efficiency", statistic="minimum", span=span,
            serve_policy="always", weighting="sample", inputs=("aggregates",), unit="1",
            note="The weakest station on the route — the one that sets whether the corridor has a "
                 "gap, which an average cannot show."),
    }


def aggregate_route(per_station: list[artifacts.Products], corridor: CorridorLayout):
    """Reduce several runs' P9 aggregates to route level, using the engine's own machinery.

    The station axis takes the place of the sample axis, and the candidate axis is preserved — so a
    corridor of swept candidates reduces per candidate, which is what makes an economic search over a
    corridor affordable.
    """
    definitions = route_definitions(corridor)
    stacked = {}
    for name in ("delivered_energy_j", "grid_energy_j", "pass_energy_efficiency"):
        stacked[name] = np.stack([products.aggregates.entries[name].value
                                  for products in per_station], axis=1)
    n_candidates, n_stations = stacked["delivered_energy_j"].shape
    ones = np.ones((n_candidates, n_stations))
    feasible = np.ones((n_candidates, n_stations), dtype=bool)
    station_index = np.arange(n_stations, dtype=float)

    results = {}
    for name, definition in definitions.items():
        source = definition.quantity.split(".")[-1]
        if definition.statistic == "energy_ratio":
            entry = aggregate.build(definition, stacked["delivered_energy_j"], ones, feasible,
                                    station_index, denominator=stacked["grid_energy_j"])
        else:
            entry = aggregate.build(definition, stacked[source], ones, feasible, station_index)
        results[name] = entry
    return results, definitions


def run_corridor(base: RunSpec, corridor: CorridorLayout, study_dir: str | Path,
                 run_root: str | Path, weathers=("clear",), speed_ms: float = 250.0,
                 n_samples: int = 41, notes: str = "") -> tuple[dict, Path]:
    """Evaluate one run per corridor segment, reduce to route level, and write the study.

    `weathers` is the one genuinely per-station variation this reference driver exposes: a route does
    not sit under one sky, and weather changes the number of link legs not at all, so it is a run
    axis rather than a candidate axis (D7 makes the architecture the thing that cannot vary within a
    run).
    """
    from dataclasses import replace
    per_station, run_dirs = [], []
    for index, weather_name in enumerate(weathers):
        scenario = replace(base.scenario, weather=artifacts.weather(weather_name))
        spec = station_spec(replace(base, scenario=scenario), corridor, speed_ms, n_samples,
                            run_id=f"corridor-{index:02d}-{weather_name}")
        products, manifest = physics.evaluate(spec)
        run_dir = Path(run_root) / manifest.run_id
        physics.write(products, manifest, run_dir)
        per_station.append(products)
        run_dirs.append(str(run_dir))

    results, definitions = aggregate_route(per_station, corridor)
    manifest = artifacts.StudyManifest(
        study_id=Path(study_dir).name, run_dirs=tuple(run_dirs), definitions=definitions,
        driver=f"{DRIVER_NAME}@{physics.engine_sha()[:12]}", engine_sha=physics.engine_sha(),
        created_utc=per_station and _now() or _now(),
        parameters={"station_spacing_km": corridor.station_spacing_km,
                    "n_stations": corridor.n_stations,
                    "min_elevation_deg": corridor.min_elevation_deg,
                    "route_km": corridor.route_km, "altitude_m": corridor.altitude_m,
                    "segments": list(weathers),
                    "layout_source": "physics.geo.downrange_for_elevation on a spherical earth"},
        notes=notes)
    arrays = {name: entry.value for name, entry in results.items()}
    arrays["station_served_fraction"] = np.stack(
        [products.aggregates.entries["served_fraction"].value for products in per_station], axis=1)
    return results, artifacts.save_study(study_dir, arrays, manifest)


def _now() -> str:
    import time
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
