"""The efficiency chain: the named factors whose product is end-to-end η (P7).

Numeric only. The pre-SPEC engine returned display label strings from a hand-maintained mirror of
the chain that "has to be mirrored" on every change (§8 item 3); here the factor NAMES are data, the
order is the order they apply in, and I6 asserts that the product of the factors is the reported η —
so a factor that is added and not multiplied in fails the suite instead of quietly disappearing from
a waterfall.

The relay branch does NOT republish its down-leg terms under the direct branch's generic keys. The
pre-SPEC code did, "so sweep code can stay architecture-agnostic", which put two different quantities
under one name (§8 item 10, I2). A consumer that wants the aircraft-side capture reads the factor
named `capture_relay_to_aircraft`, and `factor_names` tells it which architecture it is looking at.
"""

from __future__ import annotations

import numpy as np

from beamlink.artifacts.spec import ScenarioSpec
from beamlink.physics.beam_link import relations

FEASIBLE_ETA_FLOOR = 1e-4
"""Below this the link is not an operating case. Deliberately separate from the power-density flag:
an over-cap geometry is reported and kept, a blocked or vanishing link is not feasible."""


def direct_factors(scenario: ScenarioSpec, capture, transmittance, polarization, rectenna):
    """The five factors of a direct station→aircraft link, station side first."""
    grid_to_rf = relations.eta_grid_to_rf(scenario.frequency_ghz, scenario.pae_scenario)
    aperture = relations.emitter_aperture_efficiency(scenario.aperture.error_efficiency)
    ones = np.ones_like(np.asarray(capture, dtype=float))
    return (
        ("grid_to_rf", grid_to_rf * ones),
        ("emitter_aperture", aperture * ones),
        ("capture_station_to_aircraft", np.asarray(capture, dtype=float)),
        ("atmosphere_station_to_aircraft", np.asarray(transmittance, dtype=float)),
        ("rectenna_rf_to_dc", np.asarray(rectenna, dtype=float)),
        ("polarization", np.asarray(polarization, dtype=float)),
    )


def relay_factors(scenario: ScenarioSpec, capture_up, transmittance_up, capture_down,
                  transmittance_down, polarization, rectenna):
    """The nine factors of a relayed station→relay→aircraft link, station side first."""
    relay = scenario.relay
    frequency_out = relay.f_in_ghz if relay.mode == "passive" else relay.f_out_ghz
    grid_to_rf = relations.eta_grid_to_rf(relay.f_in_ghz, scenario.pae_scenario)
    aperture = relations.emitter_aperture_efficiency(scenario.aperture.error_efficiency)
    convert = relations.relay_convert_efficiency(
        relay.f_in_ghz, frequency_out, relay.mode,
        reflectarray_reflectivity=relay.reflectarray_reflectivity,
        surface_error_wavelengths=relay.surface_error_wavelengths,
        stage_efficiency=relay.stage_efficiency, pae_scenario=scenario.pae_scenario)
    ones = np.ones_like(np.asarray(capture_down, dtype=float))
    return (
        ("grid_to_rf", grid_to_rf * ones),
        ("emitter_aperture", aperture * ones),
        ("capture_station_to_relay", np.asarray(capture_up, dtype=float)),
        ("atmosphere_station_to_relay", np.asarray(transmittance_up, dtype=float)),
        ("relay_convert", convert * ones),
        ("relay_aperture", aperture * ones),
        ("capture_relay_to_aircraft", np.asarray(capture_down, dtype=float)),
        ("atmosphere_relay_to_aircraft", np.asarray(transmittance_down, dtype=float)),
        ("rectenna_rf_to_dc", np.asarray(rectenna, dtype=float)),
        ("polarization", np.asarray(polarization, dtype=float)),
    )


def assemble(factors, horizon_blocked):
    """(names, stacked factors, η, feasible) from a sequence of (name, array) pairs.

    η is the product of exactly the factors emitted — there is no separate expression for it to drift
    away from, which is what I6's recompute check pins.
    """
    names = tuple(name for name, _ in factors)
    stacked = np.stack([np.asarray(value, dtype=float) for _, value in factors], axis=0)
    eta = np.prod(stacked, axis=0)
    blocked = np.any(np.asarray(horizon_blocked, dtype=bool), axis=0)
    return names, stacked, eta, (~blocked) & (eta > FEASIBLE_ETA_FLOOR)


def up_leg_capture(scenario: ScenarioSpec, slant_up_m):
    """Goubau coupling of the station→relay leg, where both ends are circular apertures.

    An aperture-to-aperture hop: the closed form is the right model here, and it is the same relation
    the closed-form capture tier uses on the aircraft leg. Taper, focusing and pointing error do not
    enter, which is a real limitation of this leg's model and not a tier choice — the relay is a
    fixed disc and the up-leg is sized to it.
    """
    relay = scenario.relay
    relay_radius = relay.aperture_diam_m / 2.0
    emitter_radius = relations.required_emitter_radius(
        relay.f_in_ghz * 1e9, slant_up_m, relay_radius, scenario.aperture.tau_target)
    tau = relations.coupling_tau(relay.f_in_ghz * 1e9, slant_up_m, emitter_radius, relay_radius)
    return relations.collection_efficiency(tau), 2.0 * emitter_radius
