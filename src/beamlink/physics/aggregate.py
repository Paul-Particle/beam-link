"""The reduction machinery, and the named trajectory-level aggregates of P9.

I3 is worded as *derivation is owned and recorded* rather than "only the engine derives", because the
engine derives within a run and the driver derives across runs (D6). Both use this module, and both
record the same `AggregateDefinition` alongside the number — so "who derived this and how" is
answerable for a corridor figure exactly as it is for a pass figure. A consumer never derives at all,
which is why this lives under `physics/` and not under `artifacts/`.

**The two orphaned definitions of §1 are the first two named products here.** They disagreed by
0.32× in a storm at 400 km spacing, and neither recorded its span, serve policy, statistic or
sampling. Both are now emitted, both named, both carrying their full definition. Redundancy is not
the enemy; anonymity is.
"""

from __future__ import annotations

import numpy as np

from beamlink.artifacts.products import Aggregate
from beamlink.artifacts.spec import AggregateDefinition


def serve_mask(definition: AggregateDefinition, eta, feasible):
    """Which samples the serve policy admits, (candidate, sample) bool.

    A station rated for one grid draw is either serving or it is not; the policy says which, and the
    threshold is a declared input recorded with the number it produced. `min_serve_eta = 0.02` in the
    pre-SPEC sweep was "not derived from anything, and it sets the headline": moving it from 0 to
    0.10 took `energy_eta` from 0.090 to 0.240 for 5.8-direct in a storm (§8 item 20).
    """
    eta = np.asarray(eta, dtype=float)
    if definition.serve_policy == "always":
        return np.ones(eta.shape, dtype=bool)
    if definition.serve_policy == "feasible":
        return np.asarray(feasible, dtype=bool)
    return eta >= definition.serve_eta_floor


def sample_weights(definition: AggregateDefinition, times_s, n_candidates):
    """(candidate, sample) weights: the trajectory's own dt, or uniform.

    Time weighting uses the midpoint rule on the spec's times, so a non-uniformly sampled trajectory
    integrates correctly instead of being silently treated as uniform — which is what a reduction
    that assumed a constant speed would do.
    """
    times = np.asarray(times_s, dtype=float)
    if definition.weighting == "sample":
        weights = np.ones(times.shape)
    elif times.size < 2:
        weights = np.ones(times.shape)
    else:
        edges = np.concatenate([[times[0]], 0.5 * (times[1:] + times[:-1]), [times[-1]]])
        weights = np.diff(edges)
    return np.broadcast_to(weights, (n_candidates, times.size)).copy()


def reduce_quantity(definition: AggregateDefinition, values, weights, served,
                    denominator=None):
    """Apply one statistic over the served window, per candidate.

    `values` and `weights` are (candidate, sample); `denominator` is required by "energy_ratio",
    which is the only statistic that reduces two quantities rather than one.
    """
    values = np.asarray(values, dtype=float)
    active = weights * served
    total_weight = active.sum(axis=1)
    safe_weight = np.where(total_weight > 0, total_weight, 1.0)
    statistic = definition.statistic

    if statistic == "mean":
        return np.where(total_weight > 0, (values * active).sum(axis=1) / safe_weight, 0.0)
    if statistic == "harmonic_mean":
        positive = np.where(values > 0, values, np.inf)
        reciprocal = (active / positive).sum(axis=1)
        return np.where(reciprocal > 0, total_weight / np.where(reciprocal > 0, reciprocal, 1.0), 0.0)
    if statistic == "peak":
        return np.max(np.where(served, values, -np.inf), axis=1, initial=0.0)
    if statistic == "minimum":
        return np.min(np.where(served, values, np.inf), axis=1, initial=np.inf)
    if statistic == "time_integral":
        return (values * active).sum(axis=1)
    if statistic == "energy_ratio":
        numerator = (values * active).sum(axis=1)
        below = (np.asarray(denominator, dtype=float) * active).sum(axis=1)
        return np.where(below > 0, numerator / np.where(below > 0, below, 1.0), 0.0)
    if statistic == "served_fraction":
        weight_total = weights.sum(axis=1)
        return np.where(weight_total > 0, total_weight / np.where(weight_total > 0, weight_total, 1.0),
                        0.0)
    raise ValueError(f"unhandled statistic {statistic!r}")


def build(definition: AggregateDefinition, values, eta, feasible, times_s, denominator=None
          ) -> Aggregate:
    """One named aggregate, with the window it was taken over emitted alongside the number."""
    eta = np.atleast_2d(np.asarray(eta, dtype=float))
    served = serve_mask(definition, eta, feasible)
    weights = sample_weights(definition, times_s, eta.shape[0])
    value = reduce_quantity(definition, np.atleast_2d(values), weights, served, denominator)
    return Aggregate(definition=definition, value=value, served=served)


# --- the named catalogue ------------------------------------------------------------------


def definitions(span: str, serve_eta_floor: float) -> dict[str, AggregateDefinition]:
    """The aggregates every run with a chain and a power product emits.

    `span` names the trajectory the run was given, so a number carries the pass it belongs to instead
    of inheriting whichever span the module that computed it happened to use — the direct cause of
    the divergence in §1, where three modules had three different pass spans.
    """
    chain_inputs = ("chain", "trajectory")
    power_inputs = ("power", "chain", "trajectory")
    return {
        # ---- the two definitions from §1, both named, both defined ------------------------
        "pass_eta_mean_served": AggregateDefinition(
            quantity="chain.eta_end_to_end", statistic="mean", span=span,
            serve_policy="eta_floor", serve_eta_floor=serve_eta_floor, weighting="time",
            inputs=chain_inputs, unit="1",
            note="Time-weighted mean of end-to-end eta over the SERVED window only. This is the "
                 "pre-SPEC sweeps.flyby `energy_eta`, the number econ consumed. It rises as the "
                 "serve floor rises, because raising the floor discards the worst samples."),
        "pass_eta_mean_whole_span": AggregateDefinition(
            quantity="chain.eta_end_to_end", statistic="mean", span=span,
            serve_policy="always", weighting="time", inputs=chain_inputs, unit="1",
            note="Time-weighted mean of end-to-end eta over the WHOLE span, served or not. This is "
                 "the figure's panel-D curve reduced the way the panel implied. It is the same "
                 "physical quantity as pass_eta_mean_served and a different number: 0.32x of it in "
                 "a storm at 400 km station spacing (SPEC.md §12)."),
        # ---- energy, the thing econ actually needs -----------------------------------------
        "delivered_energy_j": AggregateDefinition(
            quantity="power.rectenna_terminals_w", statistic="time_integral", span=span,
            serve_policy="eta_floor", serve_eta_floor=serve_eta_floor, weighting="time",
            inputs=power_inputs, unit="J",
            note="Energy banked at the rectenna terminals over the served window."),
        "grid_energy_j": AggregateDefinition(
            quantity="power.grid_w", statistic="time_integral", span=span,
            serve_policy="eta_floor", serve_eta_floor=serve_eta_floor, weighting="time",
            inputs=power_inputs, unit="J",
            note="Grid energy drawn over the served window. The station holds its rated draw "
                 "whenever it serves, so this is the rated draw times the served time."),
        "pass_energy_efficiency": AggregateDefinition(
            quantity="power.rectenna_terminals_w/power.grid_w", statistic="energy_ratio",
            span=span, serve_policy="eta_floor", serve_eta_floor=serve_eta_floor,
            weighting="time", inputs=power_inputs, unit="1",
            note="Delivered energy over grid energy across the served window. Equals "
                 "pass_eta_mean_served when the grid draw is held constant while serving, and is "
                 "the definition to prefer because it stays correct if it is not."),
        # ---- shape of the pass -------------------------------------------------------------
        "peak_eta": AggregateDefinition(
            quantity="chain.eta_end_to_end", statistic="peak", span=span, serve_policy="always",
            weighting="sample", inputs=chain_inputs, unit="1",
            note="Best sample of the pass. Sets the station's rating under the delivered-target "
                 "power basis, so it is an input to P8 as well as a reported number."),
        "harmonic_mean_eta": AggregateDefinition(
            quantity="chain.eta_end_to_end", statistic="harmonic_mean", span=span,
            serve_policy="always", weighting="time", inputs=chain_inputs, unit="1",
            note="The average that governs CONSTANT-delivery operation: the grid draw needed to "
                 "hold delivered power flat goes as 1/eta, so its time average is harmonic. A "
                 "pessimistic bound that applies only to an aircraft that cannot bank energy."),
        "served_fraction": AggregateDefinition(
            quantity="chain.eta_end_to_end", statistic="served_fraction", span=span,
            serve_policy="eta_floor", serve_eta_floor=serve_eta_floor, weighting="time",
            inputs=chain_inputs, unit="1",
            note="Fraction of the span's TIME the serve policy admitted."),
        "minimum_capture": AggregateDefinition(
            quantity="capture.incidence_weighted", statistic="minimum", span=span,
            serve_policy="eta_floor", serve_eta_floor=serve_eta_floor, weighting="time",
            inputs=("capture", "chain"), unit="1",
            note="Worst capture inside the served window — the design edge a fixed aperture has to "
                 "reach."),
    }
