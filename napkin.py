"""Napkin math for microwave-powered aviation — pushing on the envelope.

First-order relations behind the 'Premise & target envelope' section of
research-questions.md. SI units unless a name says otherwise. Every number this
produces is #(verify)-grade: order-of-magnitude sanity checks, not engineering.

Run:  uv run python napkin.py   → prints tables, writes out/pass_geometry.html
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go

SPEED_OF_LIGHT = 2.998e8   # m/s
GRAVITY = 9.81             # m/s^2
RHO_AIR_SEA = 1.225        # kg/m^3
SCALE_HEIGHT = 8500.0      # m; single-exponential atmosphere (approx; ~30% high near 20 km)

GAS_FRACTION = {"hydrogen": 2 / 29, "helium": 4 / 29}   # lifting-gas density / air density

TETHER_MATERIALS = {  # tensile strength (Pa), density (kg/m^3)
    "Dyneema (UHMWPE)": (3.5e9, 970),
    "Zylon (PBO)": (5.8e9, 1560),
    "Carbon (T1100)": (7.0e9, 1790),
    "HS steel": (2.0e9, 7850),
}


# --- Beam coupling ------------------------------------------------------------

def wavelength(frequency_hz):
    return SPEED_OF_LIGHT / frequency_hz


def coupling_tau(frequency_hz, distance_m, emitter_radius_m, receiver_radius_m):
    """Goubau aperture-coupling parameter τ = π·r_t·r_r/(λ·D)."""
    return np.pi * emitter_radius_m * receiver_radius_m / (wavelength(frequency_hz) * distance_m)


def collection_efficiency(tau):
    """Fraction of radiated power caught by the receiving aperture: η ≈ 1−exp(−τ²)."""
    return 1.0 - np.exp(-(tau ** 2))


def required_emitter_radius(frequency_hz, distance_m, receiver_radius_m, tau_target=2.0):
    """r_t = τ·λ·D/(π·r_r); τ≈2 → ~94% collection."""
    return tau_target * wavelength(frequency_hz) * distance_m / (np.pi * receiver_radius_m)


def surface_precision(frequency_hz, fraction=1 / 16):
    """Element-POSITION knowledge for phase correction (~λ/16) — not mechanical flatness."""
    return wavelength(frequency_hz) * fraction


def power_density_limited_diameter(power_w, density_wcm2=25.0):
    """Min aperture diameter to radiate/collect `power_w` without exceeding a surface
    power-density limit (~25 W/cm² for solid-state arrays and rectennas alike):
    d = 2·√(P/(π·S)). Small apertures hit this floor long before the diffraction
    minimum — which is why the 'diffraction-tiny' feed dish is really power-density-bound."""
    area_m2 = power_w / (density_wcm2 * 1e4)
    return 2.0 * np.sqrt(area_m2 / np.pi)


# --- Atmospheric attenuation (rain) -------------------------------------------

RAIN_K, RAIN_ALPHA = 0.1871, 1.0214   # ITU-R P.838, 30 GHz, horizontal pol (conservative)


def specific_rain_attenuation(rain_rate_mmh, k=RAIN_K, alpha=RAIN_ALPHA):
    """ITU-R P.838 specific attenuation γ = k·R^α [dB/km]. Defaults: 30 GHz, H-pol."""
    return k * rain_rate_mmh ** alpha


def rain_fade_db(rain_rate_mmh, rain_height_km, elevation_deg=90.0, k=RAIN_K, alpha=RAIN_ALPHA):
    """Rain attenuation over a slant path through a rain layer of depth `rain_height_km`
    at the given elevation: γ·h_R/sin(ε). 90° = vertical hop; low (grazing) elevation
    blows the path length — and hence the dB — up."""
    path_km = rain_height_km / np.sin(np.radians(elevation_deg))
    return specific_rain_attenuation(rain_rate_mmh, k, alpha) * path_km


def db_to_fraction(db):
    """Power fraction surviving `db` of attenuation: 10^(−db/10)."""
    return 10.0 ** (-db / 10.0)


# --- Flight profile -----------------------------------------------------------

def glide_range_km(lift_to_drag, altitude_drop_m):
    return lift_to_drag * altitude_drop_m / 1e3


def powered_fraction(speed_ms, climb_rate_ms, lift_to_drag):
    """Fraction of a climb-then-glide sawtooth under power = (v/vc)/(v/vc + L/D)."""
    ratio = speed_ms / climb_rate_ms
    return ratio / (ratio + lift_to_drag)


def cruise_power_mw(mass_kg, speed_ms, lift_to_drag, eta_chain=0.78):
    """Electrical power for level cruise: W·v/(L/D)/η. Drag is altitude-independent."""
    thrust = mass_kg * GRAVITY / lift_to_drag
    return thrust * speed_ms / eta_chain / 1e6


def climb_power_mw(mass_kg, speed_ms, lift_to_drag, climb_rate_ms, eta_chain=0.78):
    """Cruise + potential-energy rate (W·climb_rate) while banking altitude for the glide."""
    prop = mass_kg * GRAVITY * speed_ms / lift_to_drag + mass_kg * GRAVITY * climb_rate_ms
    return prop / eta_chain / 1e6


def beam_power_mw(electrical_mw, rectenna_eff=0.85, collection_eff=0.90):
    """What the emitter must put into the beam to deliver a given electrical power on board."""
    return electrical_mw / rectenna_eff / collection_eff


# --- Pass geometry & dwell ----------------------------------------------------

def pass_geometry(plane_alt_m, station_alt_m, alongtrack_m, lateral_offset_m=0.0):
    """Projected-area FRACTIONS (=beam direction cosines) for belly/sail/frontal surfaces."""
    height_gap = plane_alt_m - station_alt_m
    slant = np.sqrt(alongtrack_m**2 + lateral_offset_m**2 + height_gap**2)
    elevation = np.degrees(np.arcsin(height_gap / slant))
    return slant, elevation, height_gap / slant, np.abs(lateral_offset_m) / slant, np.abs(alongtrack_m) / slant


def dwell_seconds(speed_ms, plane_alt_m, station_alt_m, lateral_offset_m, max_slant_m):
    """Seconds in usable range (slant < max_slant) as the plane transits at cruise speed.

    Zero if the closest approach already exceeds max_slant.
    """
    fixed = (plane_alt_m - station_alt_m) ** 2 + lateral_offset_m**2
    if fixed >= max_slant_m**2:
        return 0.0
    half_window = np.sqrt(max_slant_m**2 - fixed)
    return 2.0 * half_window / speed_ms


# --- Tether: strength vs buoyancy vs emitter mass -----------------------------

def air_density(altitude_m):
    return RHO_AIR_SEA * np.exp(-altitude_m / SCALE_HEIGHT)


def breaking_length_km(sigma_pa, density_kgm3, safety_factor=1.0):
    return sigma_pa / (density_kgm3 * GRAVITY * safety_factor) / 1e3


def net_lift_per_m3(altitude_m, gas="hydrogen"):
    return air_density(altitude_m) * (1.0 - GAS_FRACTION[gas])


def emitter_mass_t(aperture_diam_m, areal_density_kgm2):
    return np.pi * (aperture_diam_m / 2) ** 2 * areal_density_kgm2 / 1e3


def aerostat_diameter_m(altitude_m, cable_kg_per_m, payload_kg, gas="hydrogen"):
    total_mass = cable_kg_per_m * altitude_m + payload_kg
    volume = total_mass / net_lift_per_m3(altitude_m, gas)
    return (6.0 * volume / np.pi) ** (1 / 3)


# --- Tables & figure ----------------------------------------------------------

def emitter_table():
    rows = [("baseline 5.8 GHz", 5.8, 300, 20), ("30 GHz", 30.0, 300, 25),
            ("30 GHz, 76 m rectenna", 30.0, 300, 38), ("30 GHz, D=195 km", 30.0, 195, 25)]
    return (
        pd.DataFrame(rows, columns=["scenario", "f_GHz", "D_km", "r_r_m"])
        .assign(emitter_diam_m=lambda d: 2 * required_emitter_radius(d["f_GHz"] * 1e9, d["D_km"] * 1e3, d["r_r_m"]))
        .round({"emitter_diam_m": 0})
    )


def power_table(mass_t=200, speed_ms=250, eta=0.78):
    mass = mass_t * 1e3
    rows = []
    for ld in (15, 20, 30):
        cruise = cruise_power_mw(mass, speed_ms, ld, eta)
        for vc in (5, 10):
            climb = climb_power_mw(mass, speed_ms, ld, vc, eta)
            rows.append((ld, vc, cruise, climb, beam_power_mw(climb),
                         powered_fraction(speed_ms, vc, ld) * 100))
    return pd.DataFrame(
        rows, columns=["L/D", "climb_ms", "cruise_elec_MW", "climb_elec_MW", "climb_beam_MW", "powered_%"]
    ).round(0)


def aerostat_table(altitude_km=20, aperture_diam_m=150):
    rows = []
    for label, areal, cable in [("reflectarray relay 1.5 kg/m² + structural tether", 1.5, 1.0),
                                ("active array 5 kg/m² + power cable", 5.0, 3.0)]:
        payload = emitter_mass_t(aperture_diam_m, areal) * 1e3
        rows.append((label, round(payload / 1e3), round(aerostat_diameter_m(altitude_km * 1e3, cable, payload))))
    return pd.DataFrame(rows, columns=[f"architecture @ {altitude_km} km, {aperture_diam_m} m aperture",
                                       "emitter_t", "H2_sphere_m"])


def pass_figure(plane_alt_km=12.0, station_alt_km=5.0, lateral_km=20.0, span_km=150.0,
                path="out/pass_geometry.html"):
    x = np.linspace(-span_km * 1e3, span_km * 1e3, 500)
    _, elev, belly, sail, frontal = pass_geometry(plane_alt_km * 1e3, station_alt_km * 1e3, x, lateral_km * 1e3)
    fig = go.Figure()
    fig.add_scatter(x=x / 1e3, y=belly, name="belly (down, low-drag)", line=dict(width=3))
    fig.add_scatter(x=x / 1e3, y=sail, name="sail (lateral, low-drag)", line=dict(width=3))
    fig.add_scatter(x=x / 1e3, y=frontal, name="frontal (fore/aft, HIGH-drag)", line=dict(width=3, dash="dash"))
    fig.add_scatter(x=x / 1e3, y=elev / 90, name="elevation / 90°", line=dict(width=1), opacity=0.4)
    fig.update_layout(title=f"Pass: plane {plane_alt_km} km, station {station_alt_km} km, offset {lateral_km} km",
                      xaxis_title="along-track offset (km)", yaxis_title="projected-area fraction",
                      template="plotly_white", legend=dict(orientation="h", y=-0.2))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.write_html(path, include_plotlyjs="inline")
    return path


if __name__ == "__main__":
    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 20)

    print("\n=== Emitter sizing ===\n", emitter_table().to_string(index=False))

    print("\n=== Power budget (200 t, 250 m/s) — cruise vs climb (banking altitude) ===\n",
          power_table().to_string(index=False))

    print("\n=== Dwell per pass (plane 12 km, station 5 km, usable slant 250 km) ===")
    for lat in (0, 20, 50, 100):
        print(f"  lateral offset {lat:3d} km -> {dwell_seconds(250, 12e3, 5e3, lat*1e3, 250e3):5.0f} s in range")

    print("\n=== Tether: strength is fine (breaking length, SF2) ===")
    for name, (s, r) in TETHER_MATERIALS.items():
        print(f"  {name:18s} {breaking_length_km(s, r, 2.0):5.0f} km")

    print("\n=== Aerostat size once the REAL emitter mass is loaded ===\n",
          aerostat_table().to_string(index=False))

    print("\n=== Pass figure ===\n ", pass_figure())
