"""The fidelity calibration table: every tier's error against `full`, on a fixed grid of scenarios.

§6: "Calibration is part of the deliverable. A fixed grid of scenarios run at every tier against
`full`, reporting error in η and in the trajectory-level aggregates econ consumes, published in the
README. A tier without a published error bar is not usable for a headline number."

So this is not a diagnostic — it is what makes the cheap tiers usable at all, and it is a driver
because it varies the scenario (D4).

Two tables come out:

  the TIER table       each preset against `full`, on every scenario, in η and in the aggregates
  the LEVER table      each lever moved on its own from `full`, so the measured order of value in §6
                       is re-measurable rather than remembered

    uv run python -m beamlink.drivers.calibrate            # print both, and update README.md
    uv run python -m beamlink.drivers.calibrate --quick    # a two-scenario smoke version
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from beamlink import artifacts, physics
from beamlink.artifacts.spec import ApertureSpec, RelaySpec, RunSpec, ScenarioSpec, TrajectorySpec
from beamlink.physics import fidelity

# The grid. Chosen to span what actually moves the answer: the frequency (which sets both the aperture
# and the electronics), the weather (which is the relay's whole argument), and the architecture. Each
# is a separate run because weather and architecture are run axes, not candidate axes (D7).
SCENARIOS = {
    "5.8 direct, clear": ScenarioSpec(frequency_ghz=5.8,
                                      aperture=ApertureSpec(diameter_m=100.0),
                                      weather=artifacts.weather("clear")),
    "5.8 direct, storm": ScenarioSpec(frequency_ghz=5.8,
                                      aperture=ApertureSpec(diameter_m=100.0),
                                      weather=artifacts.weather("storm")),
    "30 direct, clear": ScenarioSpec(frequency_ghz=30.0,
                                     aperture=ApertureSpec(diameter_m=20.0),
                                     weather=artifacts.weather("clear")),
    "relay x3 5.8->17.4, storm": ScenarioSpec(
        weather=artifacts.weather("storm"),
        relay=RelaySpec(position_m=(0.0, 0.0, 20_000.0), mode="multiplier", f_in_ghz=5.8,
                        f_out_ghz=17.4, aperture_diam_m=40.0)),
}

# The aggregates econ consumes, which is what §6 says the error has to be reported in — a tier whose η
# is close but whose pass energy is not would be no use to the layer that reads it.
REPORTED = ("pass_energy_efficiency", "delivered_energy_j", "peak_eta", "served_fraction")

LEVERS = {
    "mesh 5704 -> 612 faces": dict(mesh_n_span=25, mesh_n_theta=16),
    "closed-form capture": dict(capture_method="closed_form_goubau"),
    "taper search 17 -> 5": dict(taper_search_n=5),
    "density solve 20 -> 6 steps": dict(density_bisection_steps=6),
    "aperture samples 48 -> 16": dict(aperture_min_samples_across_minor=16),
    "single precision": dict(field_dtype="complex64"),
    "look-angle bin 0.1 -> 1.0 deg": dict(look_angle_bin_deg=1.0),
    "raster grid 256 -> 64": dict(raster_n_grid=64),
}


@dataclass
class Row:
    """One (scenario, variant) measurement against the `full` reference."""

    scenario: str
    variant: str
    seconds: float
    eta_max_error_pp: float
    eta_mean_error_pp: float
    aggregate_error: dict[str, float]
    faces: int


def _spec(scenario: ScenarioSpec, tier: str, n_samples: int, delivered_w: float) -> RunSpec:
    return RunSpec(
        scenario=scenario,
        trajectory=TrajectorySpec.straight_overflight(100e3, 11_000.0, 250.0, n_samples=n_samples,
                                                     name="calibration_pass"),
        fidelity=artifacts.FidelitySpec(tier=tier),
        power=artifacts.PowerSpec(delivered_target_w=delivered_w),
        run_id=f"calibrate-{tier}")


def _measure(products, reference, seconds: float, scenario: str, variant: str) -> Row:
    eta = products.chain.eta_end_to_end[0]
    reference_eta = reference.chain.eta_end_to_end[0]
    errors = {}
    for name in REPORTED:
        got = products.aggregates.entries.get(name)
        want = reference.aggregates.entries.get(name)
        if got is None or want is None:
            continue
        base = abs(float(want.value[0]))
        errors[name] = (100.0 * abs(float(got.value[0]) - float(want.value[0])) / base
                        if base > 0 else 0.0)
    return Row(scenario=scenario, variant=variant, seconds=seconds,
               eta_max_error_pp=100.0 * float(np.max(np.abs(eta - reference_eta))),
               eta_mean_error_pp=100.0 * float(np.mean(np.abs(eta - reference_eta))),
               aggregate_error=errors,
               faces=int(products.silhouette.mesh_faces.shape[0]))


def _run_variant(scenario: ScenarioSpec, tier: str, n_samples: int, changes=None,
                 delivered_w: float = 10e6):
    """Evaluate one variant and time it WARM. Optionally with individual levers moved off a preset.

    A lever is applied by registering a single-use preset rather than by inventing a tier, so the
    table measures the lever and not a bundle of them.

    The timing is the SECOND evaluation. The first pays for whatever silhouette rasters this variant's
    mesh and look-angle bin have not cached yet, and every lever that touches either changes the cache
    key — so a cold timing measures cache population, which is a one-off, and reports a coarser tier
    as slower than the fine one it is meant to beat. §6's own numbers are warm; these have to be too.
    """
    if changes:
        fidelity._PRESETS["_calibration"] = fidelity.with_changes(fidelity.config(tier), **changes)
        tier = "_calibration"
    spec = _spec(scenario, tier, n_samples, delivered_w)
    physics.evaluate(spec)                       # warm the caches this variant needs
    started = time.time()
    products, _ = physics.evaluate(spec)
    return products, time.time() - started


def tier_table(n_samples: int = 25, tiers=("standard", "fast", "coarse")) -> list[Row]:
    rows: list[Row] = []
    for name, scenario in SCENARIOS.items():
        reference, reference_seconds = _run_variant(scenario, "full", n_samples)
        rows.append(_measure(reference, reference, reference_seconds, name, "full (reference)"))
        for tier in tiers:
            products, seconds = _run_variant(scenario, tier, n_samples)
            rows.append(_measure(products, reference, seconds, name, tier))
    return rows


# Two lever cases, because one of the levers only exists in one of them: the density-cap solve is
# free where the cap does not bind (§6), so measuring it at 10 MW measures nothing. 60 MW delivered
# through a 100 m aperture binds it over the middle of the pass.
LEVER_CASES = (("5.8 direct, clear", 10e6), ("5.8 direct, clear @ 60 MW", 60e6))


def lever_table(n_samples: int = 25) -> list[Row]:
    rows: list[Row] = []
    for label, delivered_w in LEVER_CASES:
        scenario = SCENARIOS["5.8 direct, clear"]
        reference, reference_seconds = _run_variant(scenario, "full", n_samples,
                                                   delivered_w=delivered_w)
        rows.append(_measure(reference, reference, reference_seconds, label, "full (reference)"))
        for lever, changes in LEVERS.items():
            products, seconds = _run_variant(scenario, "full", n_samples, changes,
                                             delivered_w=delivered_w)
            rows.append(_measure(products, reference, seconds, label, lever))
    return rows


def markdown(rows: list[Row], title: str, reference_seconds: float | None = None) -> str:
    """The table as it goes in the README. Speed is quoted as a factor, which is what a tier is for."""
    header = ("| scenario | variant | s | speed | faces | Δη max (pp) | Δη mean (pp) | "
              + " | ".join(f"Δ{name} (%)" for name in REPORTED) + " |")
    divider = "|" + "---|" * (7 + len(REPORTED))
    lines = [f"**{title}**", "", header, divider]
    baseline = {}
    for row in rows:
        if row.variant.startswith("full"):
            baseline[row.scenario] = row.seconds
        speed = baseline.get(row.scenario, row.seconds) / max(row.seconds, 1e-9)
        cells = [f"{row.aggregate_error.get(name, float('nan')):.2f}" for name in REPORTED]
        lines.append(f"| {row.scenario} | {row.variant} | {row.seconds:.2f} | {speed:.1f}× | "
                     f"{row.faces} | {row.eta_max_error_pp:.3f} | {row.eta_mean_error_pp:.3f} | "
                     + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                       help="two scenarios and fewer samples, for a smoke run")
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--write-readme", type=Path, default=None,
                       help="replace the calibration section of this README in place")
    arguments = parser.parse_args(argv)

    global SCENARIOS
    if arguments.quick:
        SCENARIOS = {name: SCENARIOS[name] for name in list(SCENARIOS)[:2]}

    tiers = tier_table(arguments.samples)
    levers = lever_table(arguments.samples)
    body = (markdown(tiers, "Fidelity tiers against `full`") + "\n\n"
            + markdown(levers, "Individual levers, moved off `full` one at a time"))
    print(body)
    if arguments.write_readme:
        _splice_readme(arguments.write_readme, body)
        print(f"\nupdated {arguments.write_readme}")
    return 0


_MARKER_START = "<!-- calibration:start -->"
_MARKER_END = "<!-- calibration:end -->"


def _splice_readme(path: Path, body: str) -> None:
    """Replace the marked calibration section, so the published table cannot go stale silently."""
    text = path.read_text()
    stamp = (f"_Measured {time.strftime('%Y-%m-%d')} on engine "
             f"{physics.engine_sha()[:12]}, {physics.ENGINE_VERSION}._")
    block = f"{_MARKER_START}\n\n{stamp}\n\n{body}\n\n{_MARKER_END}"
    if _MARKER_START in text and _MARKER_END in text:
        head = text.split(_MARKER_START)[0]
        tail = text.split(_MARKER_END)[1]
        path.write_text(head + block + tail)
    else:
        path.write_text(text.rstrip() + "\n\n" + block + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
