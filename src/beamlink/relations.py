"""Closed-form beam relations: coupling, frequency multiplication, rain, band efficiencies.

The engine's live chain runs on these -- `beam_link` calls the coupling, rain and rectenna
terms directly, so retuning a constant here moves the reported eta. SI units unless a name
says otherwise; frequencies in GHz where the argument says so.
"""

from __future__ import annotations

import numpy as np

SPEED_OF_LIGHT = 2.998e8   # m/s


# --- beam coupling (Goubau) ------------------------------------------------------------

def wavelength(frequency_hz):
    return SPEED_OF_LIGHT / frequency_hz


def coupling_tau(frequency_hz, distance_m, emitter_radius_m, receiver_radius_m):
    """Goubau coupling parameter tau = pi*a_t*a_r/(lambda*D), dimensionless. The single
    number that decides how much of a beam a finite receiver can collect."""
    return np.pi * emitter_radius_m * receiver_radius_m / (wavelength(frequency_hz) * distance_m)


def collection_efficiency(tau):
    """Fraction of radiated power intercepted, 1 - exp(-tau^2). tau=2 gives 0.982."""
    return 1.0 - np.exp(-(tau ** 2))


def required_emitter_radius(frequency_hz, distance_m, receiver_radius_m, tau_target=2.0):
    """`coupling_tau` solved for emitter radius; the sizing rule used everywhere."""
    return tau_target * wavelength(frequency_hz) * distance_m / (np.pi * receiver_radius_m)


def surface_precision(frequency_hz, fraction=1 / 16):
    """Element-position *knowledge* needed for phase correction (~lambda/16), in m.

    Not a mechanical flatness or rigidity requirement: a retrodirective array may flex freely
    provided each element's position is known to this accuracy. The distinction is what keeps a
    78 m tethered or aerostat-borne aperture on the table -- 0.6 mm at 30 GHz is a metrology
    problem, not a structural one.
    """
    return wavelength(frequency_hz) * fraction


def power_density_limited_diameter(power_w, density_wcm2=25.0):
    """Smallest aperture that can radiate `power_w` without exceeding a surface power
    density limit (W/cm^2)."""
    area_m2 = power_w / (density_wcm2 * 1e4)
    return 2.0 * np.sqrt(area_m2 / np.pi)


# --- Passive frequency multiplication (the "gearbox") -------------------------
# Manley-Rowe permits ~100% f -> n·f conversion for an IDEAL lossless nonlinear
# reactance (varactor); real multipliers fall well short, and a high order like
# 5.8 -> 30 GHz (~x6) is built as a cascade whose per-stage losses compound.

def multiplier_factor_prime(input_ghz=5.8, output_ghz=30.0):
    """Harmonic order of a frequency-multiplying relay, rounded to an integer."""
    return round(output_ghz / input_ghz)


def cascade_efficiency(order, stage_efficiency=0.7):
    """A multiplier of the given order is built from one stage per prime factor (x6 = x2
    then x3), each costing stage_efficiency. Returns (total, stage list)."""
    stages = _prime_stages(order)
    return stage_efficiency ** len(stages), stages


def _prime_stages(n):
    """Factor `n` into a list of small prime stages (a x6 -> [2, 3])."""
    stages, d = [], 2
    while d * d <= n:
        while n % d == 0:
            stages.append(d)
            n //= d
        d += 1
    if n > 1:
        stages.append(n)
    return stages or [1]


# --- rain and gaseous attenuation ------------------------------------------------------

# ITU-R P.838-3 curve fits for the rain coefficients k and alpha: a sum of Gaussians in
# log10(f/GHz) plus a linear term. Reproduces the published table (30 GHz: kh 0.2403,
# ah 0.9485, kv 0.2291, av 0.9129). Valid 1-1000 GHz.
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


def _p838_term(frequency_ghz, coeffs, is_k):
    """The fit is for log10(k) but for alpha directly, hence `is_k`."""
    log_f = np.log10(frequency_ghz)
    gaussians = sum(a * np.exp(-(((log_f - b) / c) ** 2))
                    for a, b, c in zip(coeffs["aj"], coeffs["bj"], coeffs["cj"]))
    value = gaussians + coeffs["m"] * log_f + coeffs["c"]
    return 10.0 ** value if is_k else value


def rain_coefficients(frequency_ghz, elevation_deg=90.0, polarization_tilt_deg=0.0):
    """(k, alpha) for the specific-attenuation law, combining the H and V fits for path
    elevation and polarisation tilt as in P.838. Both the zenith path (cos^2 el = 0) and
    circular polarisation (tilt 45 deg) collapse this to the plain H/V mean."""
    kh = _p838_term(frequency_ghz, _P838["kh"], is_k=True)
    kv = _p838_term(frequency_ghz, _P838["kv"], is_k=True)
    ah = _p838_term(frequency_ghz, _P838["ah"], is_k=False)
    av = _p838_term(frequency_ghz, _P838["av"], is_k=False)
    cos_el2 = np.cos(np.radians(elevation_deg)) ** 2
    cos_2tau = np.cos(np.radians(2 * polarization_tilt_deg))
    k = (kh + kv + (kh - kv) * cos_el2 * cos_2tau) / 2.0
    alpha = (kh * ah + kv * av + (kh * ah - kv * av) * cos_el2 * cos_2tau) / (2.0 * k)
    return k, alpha


RAIN_K, RAIN_ALPHA = rain_coefficients(30.0)  # the defaults below are 30 GHz values


def specific_rain_attenuation(rain_rate_mmh, k=RAIN_K, alpha=RAIN_ALPHA):
    """gamma_R = k*R^alpha in dB/km, R in mm/h (P.838)."""
    return k * rain_rate_mmh ** alpha


def rain_fade_db(rain_rate_mmh, rain_height_km, elevation_deg=90.0, k=RAIN_K, alpha=RAIN_ALPHA):
    """Total rain loss for a slant path through a uniform layer of the given depth."""
    path_km = rain_height_km / np.sin(np.radians(elevation_deg))
    return specific_rain_attenuation(rain_rate_mmh, k, alpha) * path_km


def db_to_fraction(db):
    """Power fraction surviving `db` of attenuation: 10^(−db/10)."""
    return 10.0 ** (-db / 10.0)

# --- Frequency as the master dial: the electronics vs aperture tradeoff --------
# The DC→RF amplifier's efficiency (PAE) falls steeply with frequency; the required
# EMITTER aperture GROWS with wavelength (∝ λ ∝ 1/f) once the belly rectenna is
# capped by the airframe. So low frequency = efficient electronics but a huge emitter;
# high frequency = tiny emitter but lossy, hot, less-mature electronics. Anchors are
# #(verify), read off the 2019-2025 GaN literature (C-band saturated ~65% PAE / 80%
# rectenna; Ka-band ~40% PAE / ~72% rectenna).

def pae_dc_to_rf(frequency_ghz):
    """Transmitter power-added efficiency, straight line through 0.65 at 5.8 GHz and 0.40 at
    34.8 GHz (the band edges used throughout)."""
    slope = (0.40 - 0.65) / (34.8 - 5.8)
    return float(np.clip(0.65 + slope * (frequency_ghz - 5.8), 0.30, 0.72))


def eta_grid_to_rf_band(frequency_ghz, eta_ac_dc=0.98):
    """Station grid→RF = SiC AC/DC front end (mature ~0.98) × GaN PA PAE(f)."""
    return eta_ac_dc * pae_dc_to_rf(frequency_ghz)


def rectenna_efficiency_band(frequency_ghz):
    """Rectenna RF-to-DC efficiency, same band-edge interpolation: 0.80 at 5.8, 0.72 at 34.8."""
    slope = (0.72 - 0.80) / (34.8 - 5.8)
    return float(np.clip(0.80 + slope * (frequency_ghz - 5.8), 0.65, 0.82))


def emitter_diameter_m(frequency_ghz, slant_km, rectenna_radius_m=20.0, tau_target=2.0):
    """Transmit-aperture diameter for near-full geometric capture (τ≈2 → ~98%) at the
    given slant range, with the belly rectenna capped at rectenna_radius_m by the
    airframe: r_t = τ·λ·D/(π·r_r) ∝ λ. Halving the frequency ~doubles this diameter."""
    return 2.0 * required_emitter_radius(frequency_ghz * 1e9, slant_km * 1e3,
                                         rectenna_radius_m, tau_target)
