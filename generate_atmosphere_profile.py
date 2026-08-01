"""Generate an ALTITUDE-RESOLVED atmospheric-absorption table for the beam-link engine.

Unlike generate_spectrum.py (which stores gaseous specific attenuation for a handful
of named surface states), this walks the ITU-R standard atmosphere from sea level up
through the stratosphere and stores gaseous specific attenuation as a function of BOTH
frequency and altitude. The beam-link engine integrates it along a slant path, so the
stratospheric dry-out (water vapour → ~0 above the tropopause) falls out of the profile
itself rather than being asserted.

itur is a heavy, generate-only dependency (astropy/scipy/pyproj), so it is NOT in
pyproject; run this with an ephemeral install:

    uv run --with itur python generate_atmosphere_profile.py

Output: data/atmosphere_profile.npz with
  frequency_ghz            (Nf,)          frequency grid
  altitude_m               (Nh,)          altitude grid
  gamma_gas_db_km          (Nh, Nf)       gaseous specific attenuation γ(h, f)
  gamma_gas_cumulative_db  (Nh, Nf)       ∫₀ʰ γ dh  (dB for a vertical path 0→h)
  cloud_kl_db_km_per_gm3   (Nf,)          cloud liquid-water specific-attenuation coeff Kl(f)
  pressure_hpa, temperature_k, water_vapour_gm3   (Nh,)  the profile used
Metadata (latitude, season) is stored as 0-d arrays.
"""
from pathlib import Path

import numpy as np
import astropy.units as u
from itur.models.itu676 import gamma_exact
from itur.models import itu835, itu840

LATITUDE_DEG = 45.0      # mid-latitude reference (transatlantic corridor band)
SEASON = "summer"        # wetter → conservative (more absorption) than winter

# Frequency grid: dense through the bands we actually use (5-40 GHz) and out to
# 100 GHz so the O2 60-GHz complex is represented for completeness.
FREQUENCY_GHZ = np.unique(np.concatenate([
    np.linspace(1.0, 45.0, 441),      # 0.1-GHz steps across the working bands
    np.logspace(np.log10(45.0), np.log10(100.0), 120),
]))

# Altitude grid: sea level to 30 km at 250 m — resolves the tropopause dry-out.
ALTITUDE_M = np.arange(0.0, 30_000.0 + 1.0, 250.0)


def _value(quantity):
    """Strip astropy units if present, else pass through."""
    return np.asarray(getattr(quantity, "value", quantity), dtype=float)


def build_profile():
    altitude_km = ALTITUDE_M / 1e3
    pressure_hpa = np.array([_value(itu835.pressure(LATITUDE_DEG, h, SEASON)) for h in altitude_km])
    temperature_k = np.array([_value(itu835.temperature(LATITUDE_DEG, h, SEASON)) for h in altitude_km])
    vapour_gm3 = np.array([_value(itu835.water_vapour_density(LATITUDE_DEG, h, SEASON)) for h in altitude_km])

    freq_q = FREQUENCY_GHZ * u.GHz
    gamma = np.empty((ALTITUDE_M.size, FREQUENCY_GHZ.size))
    for i in range(ALTITUDE_M.size):
        g = gamma_exact(freq_q, pressure_hpa[i] * u.hPa,
                        vapour_gm3[i] * u.g / u.m ** 3, temperature_k[i] * u.K)
        gamma[i] = _value(g)

    # Vertical cumulative integral ∫₀ʰ γ dh (dB), per frequency.
    cumulative = np.zeros_like(gamma)
    cumulative[1:] = np.cumsum(
        0.5 * (gamma[1:] + gamma[:-1]) * np.diff(ALTITUDE_M / 1e3)[:, None], axis=0)

    # Cloud liquid-water specific-attenuation coefficient Kl(f) [(dB/km)/(g/m³)],
    # evaluated at a representative cloud temperature (~0 °C).
    kl = np.array([float(_value(itu840.specific_attenuation_coefficients(f, T=273.15)))
                   for f in FREQUENCY_GHZ])

    return dict(pressure_hpa=pressure_hpa, temperature_k=temperature_k,
                vapour_gm3=vapour_gm3, gamma=gamma, cumulative=cumulative, kl=kl)


def main():
    p = build_profile()
    out = Path("data/atmosphere_profile.npz")
    out.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        out,
        frequency_ghz=FREQUENCY_GHZ,
        altitude_m=ALTITUDE_M,
        gamma_gas_db_km=p["gamma"],
        gamma_gas_cumulative_db=p["cumulative"],
        cloud_kl_db_km_per_gm3=p["kl"],
        pressure_hpa=p["pressure_hpa"],
        temperature_k=p["temperature_k"],
        water_vapour_gm3=p["vapour_gm3"],
        latitude_deg=np.array(LATITUDE_DEG),
        season=np.array(SEASON),
    )
    print(f"wrote {out}  ({FREQUENCY_GHZ.size} freqs × {ALTITUDE_M.size} altitudes)")
    # Spot-check: 30 GHz vertical loss 0→11 km and 0→20 km.
    j = int(np.argmin(np.abs(FREQUENCY_GHZ - 30.0)))
    for h_km in (11.0, 20.0):
        i = int(np.argmin(np.abs(ALTITUDE_M - h_km * 1e3)))
        print(f"  30 GHz vertical gaseous loss 0→{h_km:.0f} km: {p['cumulative'][i, j]:.3f} dB")


if __name__ == "__main__":
    main()
