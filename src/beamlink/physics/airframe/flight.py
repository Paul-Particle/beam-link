"""Flight power, the mass cascade and the unpowered reserve — P11–P13.

Inside the engine, as `physics.airframe` alongside `physics.beam_link` (D5): it is physics, so econ
may not compute it. The pre-SPEC econ driver took `CRUISE_MW` straight out of the mass cascade,
which the firewall forbids.

Two things the ruling changed, and both are here:

**The power reference plane is in the schema, not in a comment.** The cascade folds the onboard
motor/drivetrain into its power requirement, and the pre-SPEC econ carried a comment warning not to
double-count it. A number whose measurement point lives in a comment is the same defect as an
unnamed derived quantity. P12 states `reference_plane = "rectenna_terminals"` and P8 quotes
delivered power at the same plane, so the double count is structurally impossible.

**The fixed point runs lockstep.** `mass_buildup` was a scalar loop with a data-dependent early exit
(`max_iter=200`, `tol=1 kg`, `if converged: break`). Vectorized over candidates it runs a fixed
iteration count set by the fidelity config; convergence is geometric, so this costs little, and P11
emits the residual so the fixed count is auditable rather than asserted.

The two domains stay independent, coupled only through the spec. The apparent cross-domain fixed
point — delivered power sets what the aircraft can do, aircraft mass sets the power required — was
cut in the pre-SPEC code by a hardcoded `rectenna_mass_t = 2.0`, and it stays cut, because rectenna
mass follows from silhouette area and areal density (geometry) rather than from the link's η.
Closing the remaining scenario-level condition is a search over scenarios, which D4 puts in the
driver.
"""

from __future__ import annotations

import numpy as np

from beamlink.artifacts.spec import AirframeSpec

GRAVITY = 9.81   # m/s²


def cruise_shaft_power_w(mass_kg, speed_ms, lift_to_drag):
    """Propulsive power for level flight: thrust × speed, with thrust = weight / (L/D)."""
    return (np.asarray(mass_kg, dtype=float) * GRAVITY
            / np.asarray(lift_to_drag, dtype=float) * np.asarray(speed_ms, dtype=float))


def climb_shaft_power_w(mass_kg, speed_ms, lift_to_drag, climb_rate_ms):
    """Level-flight power plus the potential-energy rate m·g·v_climb."""
    mass = np.asarray(mass_kg, dtype=float)
    return (cruise_shaft_power_w(mass, speed_ms, lift_to_drag)
            + mass * GRAVITY * np.asarray(climb_rate_ms, dtype=float))


def glide_range_km(lift_to_drag, altitude_drop_m):
    return np.asarray(lift_to_drag, dtype=float) * np.asarray(altitude_drop_m, dtype=float) / 1e3


def powered_fraction(speed_ms, climb_rate_ms, lift_to_drag):
    """Fraction of a climb-then-glide cycle that needs the beam.

    Climbing height h covers h·V/v_climb of ground and gliding it back covers h·L/D, so the powered
    share is (V/v_c)/(V/v_c + L/D) — independent of h, and the same in time as in distance.
    """
    ratio = np.asarray(speed_ms, dtype=float) / np.maximum(np.asarray(climb_rate_ms, dtype=float),
                                                           1e-9)
    return ratio / (ratio + np.asarray(lift_to_drag, dtype=float))


def mass_cascade(spec: AirframeSpec, rectenna_mass_kg, design_speed_ms, iterations: int):
    """The converged mass cascade, batched over candidates, at a FIXED iteration count.

    Structure and systems scale with MTOW, motors with peak shaft power, and the reserve battery with
    `reserve_minutes` of cruise power. Iterating from twice the payload, the map MTOW → MTOW is a
    contraction with ratio (structure_fraction + systems_fraction + small terms) ≈ 0.4, so the error
    falls by ~0.4 per step and forty steps is 1e-16 of the payload. That is why a fixed count is
    cheaper than the convergence test it replaces, not just more vectorizable.

    Returns a dict of per-candidate arrays in kg, plus the last step's movement in MTOW.
    """
    payload = np.asarray(spec.payload_t, dtype=float) * 1e3
    rectenna = np.asarray(rectenna_mass_kg, dtype=float)
    speed = np.asarray(design_speed_ms, dtype=float)
    payload, rectenna, speed = np.broadcast_arrays(payload, rectenna, speed)
    mtow = payload * 2.0
    previous = mtow
    structure = systems = motor = battery = np.zeros_like(mtow)
    for _ in range(int(iterations)):
        # Both powers are quoted at the RECTENNA TERMINALS: shaft power over the drivetrain, which is
        # the plane P12 reports and P8 delivers at.
        cruise_terminals = (cruise_shaft_power_w(mtow, speed, spec.lift_to_drag)
                            / spec.drivetrain_efficiency)
        peak_terminals = (climb_shaft_power_w(mtow, speed, spec.lift_to_drag, spec.climb_rate_ms)
                          / spec.drivetrain_efficiency)
        motor = peak_terminals / (spec.motor_specific_power_kw_per_kg * 1e3)
        battery = (cruise_terminals * (spec.reserve_minutes / 60.0)
                   / spec.battery_specific_energy_wh_per_kg)
        structure = spec.structure_fraction * mtow
        systems = spec.systems_fraction * mtow
        previous = mtow
        mtow = payload + structure + systems + motor + battery + rectenna
    return dict(payload_kg=payload, structure_kg=structure, systems_kg=systems, motor_kg=motor,
                reserve_battery_kg=battery, rectenna_kg=rectenna, mtow_kg=mtow,
                residual_kg=np.abs(mtow - previous))


def rectenna_mass_kg(belly_silhouette_area_m2, areal_density_kg_m2):
    """Rectenna mass from the array's own area and a declared areal density.

    A gain D5 enables: the belly-on silhouette area is already a product (P3), so the pre-SPEC 2 t
    constant becomes area × areal density with the areal density as the declared input it actually
    is. The BELLY-ON area is used, not the per-sample illuminated area — the array is built once.
    """
    return (np.asarray(belly_silhouette_area_m2, dtype=float)
            * np.asarray(areal_density_kg_m2, dtype=float))


def reserve(spec: AirframeSpec, cascade, design_speed_ms):
    """P13's numbers: what the glide and the reserve battery buy, per candidate."""
    cruise_terminals = (cruise_shaft_power_w(cascade["mtow_kg"], design_speed_ms, spec.lift_to_drag)
                        / spec.drivetrain_efficiency)
    battery_energy_j = (cascade["reserve_battery_kg"] * spec.battery_specific_energy_wh_per_kg
                        * 3600.0)
    # The glide uses the BARE airframe's lift-to-drag: the electric-propulsion integration bonus is
    # an energy-path advantage and is not available with the beam off.
    range_km = glide_range_km(spec.airframe_lift_to_drag, spec.dropout_altitude_m)
    return dict(
        glide_range_km=range_km,
        glide_duration_s=range_km * 1e3 / np.maximum(np.asarray(design_speed_ms, dtype=float), 1e-9),
        battery_duration_s=battery_energy_j / np.maximum(cruise_terminals, 1e-9),
        powered_fraction=powered_fraction(design_speed_ms, spec.climb_rate_ms,
                                          spec.airframe_lift_to_drag))
