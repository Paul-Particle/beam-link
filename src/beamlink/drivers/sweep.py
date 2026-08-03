"""The reference driver: sweep a declared design parameter, and search on an economic objective.

Deliberately thin and not privileged (§10 item 7). It demonstrates the plug shape D4 requires — the
engine is a black-box callable, `evaluate(spec) -> Products`, and the objective is read straight off
an emitted aggregate — and it computes no physics and no cost of its own.

Two modes, both first-class (§9):

  `sweep`             one batched call over a candidate axis. This is what D3 replaces the pre-SPEC
                      bespoke design-optimisation product with: "best fixed aperture aspect" is a
                      sweep over `aperture.fixed_aspect` against pass-integrated delivered energy,
                      econ's aperture scan is a sweep over diameter against capture at the design
                      edge, and frequency and taper studies come free.
  `refine`            coarse-to-fine search, which is the same thing several times. Each round
                      proposes a whole bracket of candidates in ONE engine call, which is why I9
                      matters: a search that can only propose one candidate at a time wastes the
                      batching that makes the search affordable.

`search_economic` is the alternation D4 describes: ask the engine a physical question, ask econ what
the answer is worth, and propose the next bracket from that. A driver is allowed to be specific — its
workflow is dictated by the particular interplay of an econ objective and the physics it depends on —
and this is the one place in the repo where "oddly specific" is the correct shape.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from beamlink import artifacts, econ, physics
from beamlink.artifacts.spec import CandidateAxis, RunSpec

DRIVER_NAME = "drivers.sweep"


@dataclass
class SweepResult:
    """One sweep: the parameter values tried, the objective at each, and the products behind them."""

    parameter: str
    values: tuple
    objective: str
    objective_values: np.ndarray
    definition: artifacts.AggregateDefinition | None
    products: artifacts.Products
    manifest: artifacts.Manifest

    @property
    def best_index(self) -> int:
        return int(np.argmax(self.objective_values))

    @property
    def best_value(self):
        return self.values[self.best_index]

    def table(self) -> str:
        lines = [f"{self.parameter:>28}  {self.objective}"]
        for value, score in zip(self.values, self.objective_values):
            marker = " <-- best" if value == self.best_value else ""
            lines.append(f"{value!s:>28}  {score:.6g}{marker}")
        return "\n".join(lines)


def sweep(base: RunSpec, parameter: str, values, objective: str = "delivered_energy_j"
          ) -> SweepResult:
    """Evaluate one candidate axis in a single batched engine call.

    `objective` names an aggregate the engine already emits. The driver does not compute it and does
    not get to define it: a differently-defined objective is a new named P9 product (I4).
    """
    values = tuple(values)
    spec = replace(base, candidates=(CandidateAxis(parameter=parameter, values=values),),
                   run_id=base.run_id or f"sweep-{parameter.replace('.', '-')}")
    products, manifest = physics.evaluate(spec)
    scores, definition = _objective(products, objective)
    return SweepResult(parameter=parameter, values=values, objective=objective,
                       objective_values=scores, definition=definition, products=products,
                       manifest=manifest)


def refine(base: RunSpec, parameter: str, low: float, high: float,
           objective: str = "delivered_energy_j", per_round: int = 7, rounds: int = 3,
           log=None) -> SweepResult:
    """Coarse-to-fine search over one continuous parameter.

    Each round is ONE batched engine call over `per_round` candidates, and the next bracket is the
    interval around the winner. That keeps the search's unit of work the same shape as a sweep's,
    which is the property that makes a search affordable at all (§6a).
    """
    result = None
    for round_index in range(rounds):
        values = tuple(np.linspace(low, high, per_round))
        result = sweep(base, parameter, values, objective)
        if log is not None:
            log(f"round {round_index + 1}: [{low:.4g}, {high:.4g}] -> "
                f"{result.best_value:.6g} at {result.objective_values[result.best_index]:.6g}")
        step = (high - low) / (per_round - 1)
        centre = float(result.best_value)
        low, high = centre - step, centre + step
    return result


def search_economic(base: RunSpec, parameter: str, low: float, high: float,
                    n_stations: int, station_spacing_km: float,
                    econ_config: econ.EconConfig | None = None, per_round: int = 7,
                    rounds: int = 3, log=None):
    """Search a physical design parameter against an ECONOMIC objective.

    The alternation D4 describes, and the reason the driver is the only layer that sees both sides: an
    economic objective cannot be searched for in physics space without evaluating econ. Each round is
    one engine call plus one econ call, both vectorized over the same candidate axis — so the cost of
    consulting econ is one batched pass, not one pass per candidate.

    Returns (best parameter value, the round's SweepResult, the econ verdict for that round).
    """
    best = None
    for round_index in range(rounds):
        values = tuple(np.linspace(low, high, per_round))
        result = sweep(base, parameter, values, "pass_energy_efficiency")
        selection = econ.select(result.products, result.manifest, n_stations, station_spacing_km)
        verdict = econ.verdict(selection, econ_config)
        # Lower total cost per flight is better, so the objective is negated: the driver decides what
        # "best" means, and it says so here rather than in a comparison buried in a loop.
        index = int(np.argmin(verdict["beamed_total_usd"]))
        if log is not None:
            log(f"round {round_index + 1}: [{low:.4g}, {high:.4g}] -> {values[index]:.6g} at "
                f"${verdict['beamed_total_usd'][index]:,.0f}/flight "
                f"(eta {selection.pass_energy_efficiency[index]:.4f})")
        best = (values[index], result, verdict)
        step = (high - low) / (per_round - 1)
        low, high = values[index] - step, values[index] + step
    return best


# --- the study: several runs, one recorded aggregation (D6) --------------------------------


def write_sweep_study(result: SweepResult, study_dir: str | Path, run_root: str | Path,
                      notes: str = "") -> Path:
    """Archive a sweep: one run directory, plus a study manifest naming the objective it reduced.

    A sweep is several scenarios in ONE run (the candidate axis), so there is one run directory here
    and the study exists to record the aggregation applied across candidates — which is the driver's
    derivation, and therefore has to be named in the output (D6, I3).
    """
    run_dir = Path(run_root) / result.manifest.run_id
    physics.write(result.products, result.manifest, run_dir)
    manifest = artifacts.StudyManifest(
        study_id=Path(study_dir).name, run_dirs=(str(run_dir),),
        definitions={result.objective: result.definition} if result.definition else {},
        driver=f"{DRIVER_NAME}@{physics.engine_sha()[:12]}",
        engine_sha=physics.engine_sha(),
        created_utc=result.manifest.created_utc,
        parameters={"swept_parameter": result.parameter,
                    "values": [_json_safe(v) for v in result.values],
                    "objective": result.objective,
                    "best_value": _json_safe(result.best_value),
                    "reduction_across_candidates": "argmax over the candidate axis"},
        notes=notes)
    return artifacts.save_study(study_dir, {"objective": result.objective_values,
                                            "candidate_index": np.arange(len(result.values))},
                               manifest)


def _objective(products: artifacts.Products, name: str):
    if products.aggregates is None:
        raise artifacts.RunIncomplete(
            "the run produced no aggregates, so there is no objective to read. A driver reads its "
            "objective off an emitted aggregate; it does not compute one (I4).")
    entry = products.aggregates.entries.get(name)
    if entry is None:
        raise artifacts.RunIncomplete(
            f"no aggregate named {name!r}; the run emitted {sorted(products.aggregates.entries)}")
    return entry.value, entry.definition


def _json_safe(value):
    return value.item() if hasattr(value, "item") else value
