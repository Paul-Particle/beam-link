"""Flight power and the mass cascade for a beamed airframe.

Closed-form, and shared: the econ layer sizes stations from `mass_buildup`'s cruise power,
and the napkin notebook uses the same functions for its narrative figures.
"""

from __future__ import annotations

GRAVITY = 9.81             # m/s^2


# --- aircraft power and mass -----------------------------------------------------------

# 22 is the bare lifting-body/BWB airframe, shared with the fossil comparator so that airframe
# gains are not credited to beaming. The +2 is electric-propulsion integration -- boundary-layer
# ingestion, no podded nacelles, distributed fans -- which is an energy-path advantage and so is
# legitimately the concept's. ~+5-10%, unverified.
AIRFRAME_LIFT_TO_DRAG = 22.0
BEAMED_LIFT_TO_DRAG = 24.0
REFERENCE_LIFT_TO_DRAG = BEAMED_LIFT_TO_DRAG   # headline value used downstream


def glide_range_km(lift_to_drag, altitude_drop_m):
    return lift_to_drag * altitude_drop_m / 1e3


def powered_fraction(speed_ms, climb_rate_ms, lift_to_drag):
    """Fraction of a climb-then-glide cycle that needs the beam. Climbing height h covers
    h*V/climb_rate of ground, gliding it back covers h*L/D, so the powered share is
    (V/vc)/(V/vc + L/D) -- independent of h, and the same in time as in distance."""
    ratio = speed_ms / climb_rate_ms
    return ratio / (ratio + lift_to_drag)


def cruise_power_mw(mass_kg, speed_ms, lift_to_drag, eta_chain=0.78):
    """Electrical power (MW) for level flight; eta_chain covers rectenna DC to propulsive."""
    thrust = mass_kg * GRAVITY / lift_to_drag
    return thrust * speed_ms / eta_chain / 1e6


def climb_power_mw(mass_kg, speed_ms, lift_to_drag, climb_rate_ms, eta_chain=0.78):
    """As above plus the potential-energy rate m*g*vc."""
    prop = mass_kg * GRAVITY * speed_ms / lift_to_drag + mass_kg * GRAVITY * climb_rate_ms
    return prop / eta_chain / 1e6


def beam_power_mw(electrical_mw, rectenna_eff=0.85, collection_eff=0.90):
    """Radiated power needed for a given onboard electrical power."""
    return electrical_mw / rectenna_eff / collection_eff


# --- Mass buildup (the size cascade, made concrete) ---------------------------

def mass_buildup(payload_t=40.0, speed_ms=250.0, lift_to_drag=BEAMED_LIFT_TO_DRAG,
                 climb_rate_ms=4.0, eta_chain=0.85, struct_fraction=0.30,
                 systems_fraction=0.10, motor_specific_power_kw_per_kg=5.0,
                 battery_specific_energy_wh_per_kg=250.0, buffer_minutes=5.0,
                 rectenna_mass_t=2.0, max_iter=200, tol=1.0):
    """Fixed-point mass loop: structure and systems scale with MTOW, motors with peak power,
    battery with `buffer_minutes` of cruise power (the beam-dropout reserve). Iterates from
    2x payload until MTOW moves less than `tol` kg. Masses out in tonnes, power in MW."""
    payload = payload_t * 1e3
    rectenna = rectenna_mass_t * 1e3
    mtow = payload * 2.0
    for _ in range(max_iter):
        cruise_w = mtow * GRAVITY * speed_ms / lift_to_drag / eta_chain
        peak_w = cruise_w + mtow * GRAVITY * climb_rate_ms / eta_chain
        motor = peak_w / (motor_specific_power_kw_per_kg * 1e3)
        battery = cruise_w * (buffer_minutes / 60.0) / battery_specific_energy_wh_per_kg
        structure = struct_fraction * mtow
        systems = systems_fraction * mtow
        mtow_new = payload + structure + systems + motor + battery + rectenna
        converged = abs(mtow_new - mtow) < tol
        mtow = mtow_new
        if converged:
            break
    return dict(payload_t=payload / 1e3, structure_t=structure / 1e3,
                systems_t=systems / 1e3, motor_t=motor / 1e3,
                battery_t=battery / 1e3, rectenna_t=rectenna / 1e3,
                mtow_t=mtow / 1e3, cruise_mw=cruise_w / 1e6, peak_mw=peak_w / 1e6,
                empty_fraction=(mtow - payload) / mtow)
