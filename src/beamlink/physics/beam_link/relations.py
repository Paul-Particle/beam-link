"""Closed-form beam relations: coupling, frequency multiplication, rain, band efficiencies.

The engine's chain runs on these, so retuning a constant here moves the reported η. SI units
unless a name says otherwise; frequencies in GHz where the argument says so. Everything
broadcasts.
"""

from __future__ import annotations

import numpy as np

SPEED_OF_LIGHT = 2.998e8   # m/s

# Incident power density a member of the public may be exposed to, W/m². Above ~2 GHz ICNIRP 2020
# (whole-body average, 30 min), FCC OET-65 "general population / uncontrolled" (1 mW/cm²) and
# IEEE C95.1-2019 (unrestricted environment) all land on the same 10 W/m² plateau, so one constant
# covers every band this model uses. The controlled/occupational figure is 5× higher.
#
# These are REFERENCE LEVELS, not engine parameters: no product bakes one in. The pre-SPEC
# `exposure_boundary` did, together with a plot-window heuristic, which is why it is gone (§8 item
# 16) — a regulatory line is a viewer overlay on the emitted field, and P5 emits the field in
# absolute W/m² precisely so that overlay is possible.
PUBLIC_EXPOSURE_LIMIT_W_M2 = 10.0
OCCUPATIONAL_EXPOSURE_LIMIT_W_M2 = 50.0


# --- beam coupling (Goubau) ---------------------------------------------------------------


def wavelength(frequency_hz):
    return SPEED_OF_LIGHT / np.asarray(frequency_hz, dtype=float)


def coupling_tau(frequency_hz, distance_m, emitter_radius_m, receiver_radius_m):
    """Goubau coupling parameter τ = π·a_t·a_r/(λ·D), dimensionless. The single number that decides
    how much of a beam a finite receiver can collect."""
    return (np.pi * np.asarray(emitter_radius_m, dtype=float)
            * np.asarray(receiver_radius_m, dtype=float)
            / (wavelength(frequency_hz) * np.asarray(distance_m, dtype=float)))


def collection_efficiency(tau):
    """Fraction of radiated power intercepted, 1 − exp(−τ²). τ=2 gives 0.982."""
    return 1.0 - np.exp(-(np.asarray(tau, dtype=float) ** 2))


def required_emitter_radius(frequency_hz, distance_m, receiver_radius_m, tau_target=2.0):
    """`coupling_tau` solved for the emitter radius; the sizing rule used everywhere."""
    return (np.asarray(tau_target, dtype=float) * wavelength(frequency_hz)
            * np.asarray(distance_m, dtype=float)
            / (np.pi * np.maximum(np.asarray(receiver_radius_m, dtype=float), 1e-9)))


def emitter_diameter_for_tau(frequency_ghz, slant_m, receiver_radius_m, tau_target=2.0):
    """Emitter diameter for a target near-field coupling τ (τ=2 → ~98 % capture)."""
    return 2.0 * required_emitter_radius(np.asarray(frequency_ghz, dtype=float) * 1e9, slant_m,
                                         receiver_radius_m, tau_target)


def surface_precision(frequency_hz, fraction=1 / 16):
    """Element-position KNOWLEDGE needed for phase correction (~λ/16), in m.

    Not a mechanical flatness or rigidity requirement: a retrodirective array may flex freely
    provided each element's position is known to this accuracy. The distinction is what keeps a
    large tethered or aerostat-borne aperture on the table — 0.6 mm at 30 GHz is a metrology
    problem, not a structural one.
    """
    return wavelength(frequency_hz) * fraction


# --- passive frequency multiplication -----------------------------------------------------
# Manley-Rowe permits ~100 % f → n·f conversion for an IDEAL lossless nonlinear reactance
# (varactor); real multipliers fall well short, and a high order like 5.8 → 30 GHz (~×6) is built
# as a cascade whose per-stage losses compound.


def multiplier_order(input_ghz, output_ghz):
    """Harmonic order of a frequency-multiplying relay, rounded to an integer."""
    return int(round(float(output_ghz) / float(input_ghz)))


def prime_stages(order: int) -> list[int]:
    """Factor `order` into small prime stages (a ×6 → [2, 3])."""
    stages, divisor = [], 2
    remaining = int(order)
    while divisor * divisor <= remaining:
        while remaining % divisor == 0:
            stages.append(divisor)
            remaining //= divisor
        divisor += 1
    if remaining > 1:
        stages.append(remaining)
    return stages or [1]


def cascade_efficiency(order: int, stage_efficiency=0.7):
    """One stage per prime factor, each costing `stage_efficiency`. Returns (total, stages)."""
    stages = prime_stages(order)
    return float(stage_efficiency) ** len(stages), stages


# --- rain and gaseous attenuation ---------------------------------------------------------

# ITU-R P.838-3 curve fits for the rain coefficients k and α: a sum of Gaussians in log10(f/GHz)
# plus a linear term. Reproduces the published table (30 GHz: kh 0.2403, ah 0.9485, kv 0.2291,
# av 0.9129). Valid 1–1000 GHz.
_P838 = {
    "kh": dict(aj=[-5.33980, -0.35351, -0.23789, -0.94158],
               bj=[-0.10008, 1.2697, 0.86036, 0.64552],
               cj=[1.13098, 0.454, 0.15354, 0.16817], m=-0.18961, c=0.71147),
    "kv": dict(aj=[-3.80595, -3.44965, -0.39902, 0.50167],
               bj=[0.56934, -0.22911, 0.73042, 1.07319],
               cj=[0.81061, 0.51059, 0.11899, 0.27195], m=-0.16398, c=0.63297),
    "ah": dict(aj=[-0.14318, 0.29591, 0.32177, -5.37610, 16.1721],
               bj=[1.82442, 0.77564, 0.63773, -0.96230, -3.29980],
               cj=[-0.55187, 0.19822, 0.13164, 1.47828, 3.4399], m=0.67849, c=-1.95537),
    "av": dict(aj=[-0.07771, 0.56727, -0.20238, -48.2991, 48.5833],
               bj=[2.3384, 0.95545, 1.1452, 0.791669, 0.791459],
               cj=[-0.76284, 0.54039, 0.26809, 0.116226, 0.116479], m=-0.053739, c=0.83433),
}


def _p838_term(frequency_ghz, coefficients, is_k):
    """The fit is for log10(k) but for α directly, hence `is_k`."""
    log_frequency = np.log10(np.asarray(frequency_ghz, dtype=float))
    gaussians = sum(a * np.exp(-(((log_frequency - b) / c) ** 2))
                    for a, b, c in zip(coefficients["aj"], coefficients["bj"], coefficients["cj"]))
    value = gaussians + coefficients["m"] * log_frequency + coefficients["c"]
    return 10.0 ** value if is_k else value


def rain_coefficients(frequency_ghz, elevation_deg=90.0, polarization_tilt_deg=0.0):
    """(k, α) for the specific-attenuation law, combining the H and V fits for path elevation and
    polarisation tilt as in P.838. Both the zenith path and circular polarisation (tilt 45°)
    collapse this to the plain H/V mean."""
    kh = _p838_term(frequency_ghz, _P838["kh"], is_k=True)
    kv = _p838_term(frequency_ghz, _P838["kv"], is_k=True)
    ah = _p838_term(frequency_ghz, _P838["ah"], is_k=False)
    av = _p838_term(frequency_ghz, _P838["av"], is_k=False)
    cos_elevation_squared = np.cos(np.radians(elevation_deg)) ** 2
    cos_two_tilt = np.cos(np.radians(2 * polarization_tilt_deg))
    k = (kh + kv + (kh - kv) * cos_elevation_squared * cos_two_tilt) / 2.0
    alpha = ((kh * ah + kv * av + (kh * ah - kv * av) * cos_elevation_squared * cos_two_tilt)
             / (2.0 * k))
    return k, alpha


def specific_rain_attenuation(rain_rate_mmh, k, alpha):
    """γ_R = k·R^α in dB/km, R in mm/h (P.838)."""
    return k * np.asarray(rain_rate_mmh, dtype=float) ** alpha


def db_to_fraction(decibels):
    """Power fraction surviving `decibels` of attenuation: 10^(−dB/10)."""
    return 10.0 ** (-np.asarray(decibels, dtype=float) / 10.0)


# --- band efficiencies --------------------------------------------------------------------
# Frequency is the master dial. The DC→RF amplifier's efficiency (PAE) falls steeply with
# frequency; the required EMITTER aperture grows with wavelength (∝ 1/f) once the belly rectenna
# is capped by the airframe. So low frequency = efficient electronics but a huge emitter; high
# frequency = a small emitter but lossy, hot, less-mature electronics. The anchors are read off
# the 2019–2025 GaN literature (C-band saturated ~65 % PAE / 80 % rectenna; Ka-band ~40 % PAE /
# ~72 % rectenna) and are #(verify).

PAE_SCENARIOS = {
    "today":  (0.65, 0.40),   # current saturated-CW GaN literature
    "medium": (0.65, 0.53),   # a decade of dedicated high-band development
    "flat":   (0.65, 0.65),   # device parity across bands — isolates geometry + weather
}


def pae_dc_to_rf(frequency_ghz, scenario="today"):
    """Transmitter power-added efficiency, linear in frequency between the two band edges."""
    low, high = PAE_SCENARIOS[scenario]
    slope = (high - low) / (34.8 - 5.8)
    return np.clip(low + slope * (np.asarray(frequency_ghz, dtype=float) - 5.8),
                   min(low, high) - 0.05, 0.9)


def eta_grid_to_rf(frequency_ghz, scenario="today", eta_ac_dc=0.98):
    """Station grid→RF = SiC AC/DC front end (mature ~0.98) × GaN PA PAE(f, scenario)."""
    return eta_ac_dc * pae_dc_to_rf(frequency_ghz, scenario)


def rectenna_efficiency(frequency_ghz):
    """Rectenna RF→DC efficiency, same band-edge interpolation: 0.80 at 5.8, 0.72 at 34.8."""
    slope = (0.72 - 0.80) / (34.8 - 5.8)
    return np.clip(0.80 + slope * (np.asarray(frequency_ghz, dtype=float) - 5.8), 0.65, 0.82)


def emitter_aperture_efficiency(error_efficiency=0.95):
    """Illumination/phase-error efficiency of the emitting array.

    No scan roll-off. Every pre-SPEC call site passed scan_deg=0, which is only correct for a
    mechanically pointed aperture — and the baseline emitter is a phased array, so this is an
    unjustified omission rather than a modelled choice, and it flatters the low-elevation ends of
    a pass where the scan angle is largest. Carried forward as the same omission rather than given
    a new derivation nobody has checked; the `scan_rolloff = 1.2` the old signature offered had
    none on record either.
    """
    return np.asarray(error_efficiency, dtype=float)


def polarization_efficiency(polarization, roll_deg, axial_ratio_loss=0.95):
    """Circular polarisation pays a fixed axial-ratio penalty but is roll-independent; anything
    else is treated as linear and loses cos² of the aircraft roll."""
    roll = np.asarray(roll_deg, dtype=float)
    if polarization == "circular":
        return np.full(roll.shape, float(axial_ratio_loss))
    return np.cos(np.radians(roll)) ** 2


def relay_convert_efficiency(f_in_ghz, f_out_ghz, mode="active", reflectarray_reflectivity=0.90,
                             surface_error_wavelengths=1 / 50, stage_efficiency=0.70,
                             pae_scenario="today"):
    """RF-in to RF-out efficiency of the three relay types.

    active      rectify to DC and re-transmit at f_out, so both conversions are paid
    multiplier  harmonic multiplier chain behind a reflectarray, one stage per prime factor
    passive     reflect only; the loss is Ruze, exp(−(4π·ε)²) for an rms surface error of ε
                wavelengths — so it does not change frequency and f_out is ignored
    """
    if mode == "active":
        return float(rectenna_efficiency(f_in_ghz)) * float(pae_dc_to_rf(f_out_ghz, pae_scenario))
    if mode == "multiplier":
        chain, _ = cascade_efficiency(multiplier_order(f_in_ghz, f_out_ghz), stage_efficiency)
        return float(reflectarray_reflectivity) * chain
    ruze = float(np.exp(-((4 * np.pi * surface_error_wavelengths) ** 2)))
    return float(reflectarray_reflectivity) * ruze
