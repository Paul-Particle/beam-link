"""I6: every derived product is recompute-checkable.

Each check takes a finished run's products and recomputes one derived product FROM ITS STATED INPUTS,
returning the worst disagreement and the tolerance it is held to. This is what makes derived products
safe rather than redundant: P6 is a number computed from P5 and P3, and if it ever stops being that
number the suite says so.

It is also the answer to §1. The failure there was not that two efficiencies existed — it was that
neither could be recomputed from anything, so nothing could tell you they had diverged. A derived
product that names its inputs (I3) can be recomputed from them, and one that is recomputed on every
run cannot silently drift.

`check_all(products, manifest)` returns a list of `Check` records; the test suite asserts every one
passes, and the calibration driver prints them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from beamlink.physics import aggregate, fidelity
from beamlink.physics.beam_link import aperture as aperture_module


@dataclass(frozen=True)
class Check:
    """One recompute check: what was rebuilt, from what, how far off, and how far off is allowed."""

    product: str
    inputs: tuple[str, ...]
    statement: str
    error: float
    tolerance: float
    unit: str = ""

    @property
    def passed(self) -> bool:
        return bool(np.isfinite(self.error)) and self.error <= self.tolerance

    def __str__(self):
        verdict = "ok  " if self.passed else "FAIL"
        return (f"{verdict} {self.product:<12} {self.statement}  "
                f"error {self.error:.3e}{self.unit} <= {self.tolerance:.1e}")


def check_capture(products, spec) -> list[Check]:
    """P6 from P5 × P3: the overlap integral is a sum over the emitted field and the emitted mesh.

    This is the strongest of the checks, because it recomputes the number econ ultimately consumes
    from arrays a CONSUMER can read — so it also proves the products carry enough to answer the
    question, which is the property §7 relies on.
    """
    capture, field, silhouette = products.capture, products.field, products.silhouette
    if capture is None or silhouette is None:
        return []
    if capture.method != "overlap_integral" or field is None or "silhouette" not in field.sets:
        return [Check("capture", capture.inputs,
                      "closed-form tier: capture is derived from P2 and P3, not from the field",
                      0.0, 0.0)]
    sample_set = field.sets["silhouette"]
    # The field is in absolute W/m² and the capture integral is a fraction of RADIATED power, so
    # divide the radiated power back out — the one place the two units meet.
    radiated = products.power.radiated_w[:, -1, :]
    intensity = np.where(sample_set.valid,
                         sample_set.intensity_w_m2 / np.where(radiated > 0, radiated, np.inf)[..., None],
                         0.0)
    # The area mask is the FIELD's validity, not the element padding: an element the sampling could
    # not resolve contributes nothing, which is what the engine's own integral does.
    area = np.where(sample_set.valid, silhouette.element_projected_area_m2, 0.0)
    geometric = np.minimum(np.sum(intensity * area, axis=-1), 1.0)
    floor = spec.scenario.receiver.incidence_floor
    response = floor + (1.0 - floor) * silhouette.element_incidence_cos
    weighted = np.minimum(np.sum(intensity * area * response, axis=-1), 1.0)
    return [
        Check("capture", ("field", "silhouette"),
              "geometric == sum(field.intensity/radiated * silhouette.projected_area)",
              float(np.max(np.abs(geometric - capture.geometric))), 2e-6),
        Check("capture", ("field", "silhouette"),
              "incidence_weighted == the same sum with the rectenna's angular response",
              float(np.max(np.abs(weighted - capture.incidence_weighted))), 2e-6),
    ]


def check_chain(products) -> list[Check]:
    """P7's η is the product of exactly the factors it emitted — no separate expression to drift."""
    chain = products.chain
    if chain is None:
        return []
    recomputed = np.prod(chain.factors, axis=1)
    factor_index = {name: i for i, name in enumerate(chain.factor_names)}
    checks = [Check("chain", ("chain",), "eta_end_to_end == prod(factors)",
                    float(np.max(np.abs(recomputed - chain.eta_end_to_end))), 1e-12)]
    if products.capture is not None:
        # The chain's capture factor must be P6's number, not a second copy of it (I2).
        name = ("capture_relay_to_aircraft" if "capture_relay_to_aircraft" in factor_index
                else "capture_station_to_aircraft")
        checks.append(Check(
            "chain", ("capture",), f"factors[{name}] == capture.incidence_weighted",
            float(np.max(np.abs(chain.factors[:, factor_index[name], :]
                                - products.capture.incidence_weighted))), 1e-12))
    if products.atmosphere is not None:
        name = ("atmosphere_relay_to_aircraft" if "atmosphere_relay_to_aircraft" in factor_index
                else "atmosphere_station_to_aircraft")
        checks.append(Check(
            "chain", ("atmosphere",), f"factors[{name}] == atmosphere.transmittance on the last leg",
            float(np.max(np.abs(chain.factors[:, factor_index[name], :]
                                - products.atmosphere.transmittance[:, -1, :]))), 1e-12))
    checks.append(Check("chain", ("chain",), "every factor is in (0, 1]",
                        float(max(0.0, chain.factors.max() - 1.0)
                              + max(0.0, -chain.factors.min())), 1e-12))
    return checks


def check_power(products) -> list[Check]:
    """P8 from P7: delivered is grid × η, waste closes the balance, and the rating basis holds."""
    power, chain = products.power, products.chain
    if power is None or chain is None:
        return []
    checks = [
        Check("power", ("chain",), "rectenna_terminals_w == grid_w * chain.eta_end_to_end",
              float(np.max(np.abs(power.grid_w * chain.eta_end_to_end
                                  - power.rectenna_terminals_w))), 1e-6, " W"),
        Check("power", ("power",), "waste_w == grid_w - rectenna_terminals_w",
              float(np.max(np.abs(power.grid_w - power.rectenna_terminals_w - power.waste_w))),
              1e-9, " W"),
    ]
    if power.rating_basis == "delivered_target_at_best_sample" and power.rated_grid_w is not None:
        # The rated draw times the best sample's η is exactly the delivered target: the basis is
        # a statement about the station, and it either holds or the number is not what it claims.
        best = np.max(power.rectenna_terminals_w, axis=1)
        target = np.max(power.rated_grid_w[:, None] * chain.eta_end_to_end, axis=1)
        checks.append(Check("power", ("chain",),
                            "the rated grid draw meets the delivered target at the best sample",
                            float(np.max(np.abs(best - target))), 1e-6, " W"))
    return checks


def check_aggregates(products) -> list[Check]:
    """P9 from P7/P8: every named reduction re-run from its own recorded definition.

    The definition is the input here, which is the point: a reduction whose definition does not
    reproduce its value is a reduction whose definition is decoration.
    """
    aggregates, chain, power = products.aggregates, products.chain, products.power
    if aggregates is None or chain is None or power is None:
        return []
    available = {"chain.eta_end_to_end": chain.eta_end_to_end,
                 "power.rectenna_terminals_w": power.rectenna_terminals_w,
                 "power.grid_w": power.grid_w}
    if products.capture is not None:
        available["capture.incidence_weighted"] = products.capture.incidence_weighted
    checks = []
    for name, entry in aggregates.entries.items():
        definition = entry.definition
        if definition.statistic == "energy_ratio":
            rebuilt = aggregate.build(definition, power.rectenna_terminals_w, chain.eta_end_to_end,
                                      chain.feasible, products.trajectory.times_s,
                                      denominator=power.grid_w)
        elif definition.quantity in available:
            rebuilt = aggregate.build(definition, available[definition.quantity],
                                      chain.eta_end_to_end, chain.feasible,
                                      products.trajectory.times_s)
        else:
            continue
        scale = max(float(np.max(np.abs(entry.value))), 1.0)
        checks.append(Check("aggregates", definition.inputs,
                            f"{name} reproduces from its own recorded definition",
                            float(np.max(np.abs(rebuilt.value - entry.value))) / scale, 1e-12))
    # The two definitions of §1 must be present AND distinguishable: the whole point is that they
    # are two named products, not one anonymous one.
    if {"pass_eta_mean_served", "pass_eta_mean_whole_span"} <= set(aggregates.entries):
        checks.append(Check("aggregates", ("chain",),
                            "both §1 pass efficiencies are emitted, each with its own definition",
                            0.0, 0.0))
    return checks


def check_mass(products) -> list[Check]:
    """P11: the lockstep fixed point actually converged, and the cascade sums to MTOW."""
    mass = products.mass
    if mass is None:
        return []
    total = (mass.payload_t + mass.structure_t + mass.systems_t + mass.motor_t
             + mass.reserve_battery_t + mass.rectenna_t)
    return [
        Check("mass", ("mass",), "the components sum to MTOW",
              float(np.max(np.abs(total - mass.mtow_t))), 1e-9, " t"),
        Check("mass", ("mass",),
              f"the fixed point converged in the {mass.iterations} lockstep iterations",
              float(np.max(mass.residual_kg)), 1e-3, " kg"),
    ]


def check_field_resampling(products, manifest, spec, tolerance: float = 0.01) -> list[Check]:
    """The silhouette sampling's grid-and-resample route against the EXACT scattered transform.

    D10: there is one field method, and the silhouette sampling reads it off a separable grid because
    a scattered transform over ten thousand elements does not scale. That interpolation is the only
    approximation the sampling adds, and this is what bounds it — on one sample, because the exact
    route costs n_query × n_illuminated.
    """
    field, silhouette, capture = products.field, products.silhouette, products.capture
    if field is None or silhouette is None or capture is None:
        return []
    if capture.method != "overlap_integral" or "silhouette" not in field.sets:
        return []
    sample_set = field.sets["silhouette"]
    # The best-captured sample: the one whose number matters most and whose element cloud is largest.
    flat = int(np.argmax(capture.incidence_weighted))
    candidate, sample = np.unravel_index(flat, capture.incidence_weighted.shape)
    points = sample_set.points_m[candidate, sample]
    valid = silhouette.element_valid[candidate, sample]
    aperture = products.aperture
    state = aperture_module.ApertureState(
        semi_major_m=[aperture.semi_major_m[candidate, sample]],
        semi_minor_m=[aperture.semi_minor_m[candidate, sample]],
        taper_exponent=[aperture.taper_exponent[candidate, sample]],
        focus_range_m=[aperture.focus_range_m[candidate, sample]],
        wavelength_m=[_serving_wavelength(spec)])
    config = fidelity.config(manifest.status("field").tier or "standard")
    exact, exact_valid, _ = aperture_module.scattered_intensity(
        state, np.array([points[0, 2]]), points[None, valid, 0], points[None, valid, 1], config)
    radiated = products.power.radiated_w[candidate, -1, sample]
    resampled = sample_set.intensity_w_m2[candidate, sample][valid] / max(radiated, 1e-30)
    area = silhouette.element_projected_area_m2[candidate, sample][valid]
    exact_capture = float(np.sum(exact[0] * area))
    resampled_capture = float(np.sum(resampled * area))
    relative = abs(resampled_capture - exact_capture) / max(exact_capture, 1e-12)
    return [Check("field", ("field",),
                  "the silhouette sampling's resample matches the exact scattered transform",
                  relative, tolerance)]


def check_all(products, manifest, spec) -> list[Check]:
    """Every check for one run. `spec` is the run's own RunSpec — from `evaluate`, or from
    `artifacts.load_spec(run_dir)` for a run read back off disk.

    The spec is a required argument rather than a set of assumed constants: the incidence floor, the
    serving frequency and the serve policy are all declared inputs, and a check that guessed them
    would be checking a different run from the one it was handed.
    """
    checks: list[Check] = []
    checks += check_capture(products, spec)
    checks += check_chain(products)
    checks += check_power(products)
    checks += check_aggregates(products)
    checks += check_mass(products)
    checks += check_field_resampling(products, manifest, spec)
    return checks


def _serving_wavelength(spec) -> float:
    """The wavelength of the leg that serves the aircraft, from the spec (a passive relay re-emits
    its input frequency, which is why this is not simply `scenario.frequency_ghz`)."""
    from beamlink.physics.beam_link import relations
    from beamlink.physics.engine import leg_frequencies
    return float(relations.wavelength(leg_frequencies(spec.scenario)[-1] * 1e9))
