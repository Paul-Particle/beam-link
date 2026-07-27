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
SCALE_HEIGHT = 8500.0      # m; single-exponential atmosphere (approx; ~30% high at 20 km)

# lifting-gas density as a fraction of air (molar-mass ratio)
GAS_FRACTION = {"hydrogen": 2 / 29, "helium": 4 / 29}

# tensile strength (Pa), density (kg/m^3)
TETHER_MATERIALS = {
    "Dyneema (UHMWPE)": (3.5e9, 970),
    "Zylon (PBO)": (5.8e9, 1560),
    "Kevlar-49": (3.6e9, 1440),
    "Carbon (T1100)": (7.0e9, 1790),
    "HS steel": (2.0e9, 7850),
}


# --- Beam coupling (the one fundamental limit) --------------------------------

def wavelength(frequency_hz: float) -> float:
    return SPEED_OF_LIGHT / frequency_hz


def coupling_tau(frequency_hz, distance_m, emitter_radius_m, receiver_radius_m):
    """Goubau coupling τ = π·r_t·r_r/(λ·D). τ≈1→~50%, τ≈2→~94%, τ≥2.5→~99%."""
    return np.pi * emitter_radius_m * receiver_radius_m / (wavelength(frequency_hz) * distance_m)


def collection_efficiency(tau):
    """η ≈ 1 − exp(−τ²). Intuition form; exact constant depends on taper."""
    return 1.0 - np.exp(-np.square(tau))


def required_emitter_radius(frequency_hz, distance_m, receiver_radius_m, tau_target=2.0):
    """r_t = τ·λ·D/(π·r_r)."""
    return tau_target * wavelength(frequency_hz) * distance_m / (np.pi * receiver_radius_m)


def surface_precision(frequency_hz, fraction=1 / 16):
    """Element-POSITION knowledge needed for phase correction (~λ/16) — not mechanical flatness."""
    return wavelength(frequency_hz) * fraction


# --- Flight profile: altitude buffer + passenger-comfort duty cycle -----------

def glide_range_km(lift_to_drag, altitude_drop_m):
    """Unpowered glide reach = (L/D)·Δh."""
    return lift_to_drag * altitude_drop_m / 1e3


def climb_angle_deg(speed_ms, climb_rate_ms):
    return np.degrees(np.arcsin(climb_rate_ms / speed_ms))


def powered_fraction(speed_ms, climb_rate_ms, lift_to_drag):
    """Fraction of a sawtooth (climb-under-beam, then glide) that must be under power.

    = (v/climb_rate) / (v/climb_rate + L/D).  Independent of the altitude swing —
    set only by comfort (climb rate) and aerodynamic quality (L/D).
    """
    climb_legs = speed_ms / climb_rate_ms
    return climb_legs / (climb_legs + lift_to_drag)


# --- Pass geometry: what the airframe presents to the beam --------------------

def pass_geometry(plane_alt_m, station_alt_m, alongtrack_m, lateral_offset_m=0.0):
    """Beam geometry as a plane passes a station.

    Returns slant (m), elevation (deg at station), and projected-area FRACTIONS
    for belly (normal down), sail (normal lateral, low-drag fin/fuselage side),
    and frontal (normal fore/aft, high-drag). The three fractions are the beam's
    direction cosines: belly²+sail²+frontal² = 1.
    """
    height_gap = plane_alt_m - station_alt_m
    slant = np.sqrt(alongtrack_m**2 + lateral_offset_m**2 + height_gap**2)
    elevation = np.degrees(np.arcsin(height_gap / slant))
    belly = height_gap / slant
    sail = np.abs(lateral_offset_m) / slant
    frontal = np.abs(alongtrack_m) / slant
    return slant, elevation, belly, sail, frontal


# --- Tether height: strength vs buoyancy --------------------------------------

def air_density(altitude_m):
    return RHO_AIR_SEA * np.exp(-altitude_m / SCALE_HEIGHT)


def breaking_length_km(sigma_pa, density_kgm3, safety_factor=1.0):
    """Self-support length of a hanging strand: σ/(ρ·g·SF)."""
    return sigma_pa / (density_kgm3 * GRAVITY * safety_factor) / 1e3


def net_lift_per_m3(altitude_m, gas="hydrogen"):
    return air_density(altitude_m) * (1.0 - GAS_FRACTION[gas])


def aerostat_diameter_m(altitude_m, cable_kg_per_m, payload_kg, gas="hydrogen"):
    """Sphere diameter needed to buoy a tether of given linear mass + payload at altitude."""
    total_mass = cable_kg_per_m * altitude_m + payload_kg
    volume = total_mass / net_lift_per_m3(altitude_m, gas)
    return (6.0 * volume / np.pi) ** (1 / 3)


# --- Scenarios ----------------------------------------------------------------

def emitter_table() -> pd.DataFrame:
    scenarios = [
        ("baseline  5.8 GHz, sea level", 5.8, 300, 20),
        ("+freq     30 GHz", 30.0, 300, 25),
        ("+freq+recv 30 GHz, 76 m rectenna", 30.0, 300, 38),
        ("comfort   30 GHz, D=195 km", 30.0, 195, 25),
    ]
    return (
        pd.DataFrame(scenarios, columns=["scenario", "f_GHz", "D_km", "r_r_m"])
        .assign(
            r_t_m=lambda d: required_emitter_radius(d["f_GHz"] * 1e9, d["D_km"] * 1e3, d["r_r_m"]),
            emitter_diam_m=lambda d: 2 * d["r_t_m"],
            pos_knowledge_mm=lambda d: surface_precision(d["f_GHz"] * 1e9) * 1e3,
        )
        .round({"r_t_m": 0, "emitter_diam_m": 0, "pos_knowledge_mm": 2})
    )


def tether_table(payload_kg=20000.0) -> pd.DataFrame:
    rows = []
    for altitude_km in (5, 10, 15, 20):
        for label, cable in (("structural ~1 kg/m (beam power up)", 1.0),
                             ("power cable ~3 kg/m (conduct up)", 3.0)):
            rows.append((
                altitude_km, label,
                aerostat_diameter_m(altitude_km * 1e3, cable, payload_kg, "hydrogen"),
            ))
    return pd.DataFrame(rows, columns=["alt_km", "tether", "H2_sphere_diam_m"]).round({"H2_sphere_diam_m": 0})


def pass_figure(plane_alt_km=12.0, station_alt_km=5.0, lateral_km=20.0,
                span_km=150.0, path="out/pass_geometry.html"):
    alongtrack = np.linspace(-span_km * 1e3, span_km * 1e3, 500)
    _, elevation, belly, sail, frontal = pass_geometry(
        plane_alt_km * 1e3, station_alt_km * 1e3, alongtrack, lateral_km * 1e3
    )
    x_km = alongtrack / 1e3
    fig = go.Figure()
    fig.add_scatter(x=x_km, y=belly, name="belly (down, low-drag)", line=dict(width=3))
    fig.add_scatter(x=x_km, y=sail, name="sail (lateral, low-drag)", line=dict(width=3))
    fig.add_scatter(x=x_km, y=frontal, name="frontal (fore/aft, HIGH-drag)", line=dict(width=3, dash="dash"))
    fig.add_scatter(x=x_km, y=elevation / 90.0, name="elevation / 90°", line=dict(width=1), opacity=0.4)
    fig.update_layout(
        title=f"Pass: plane {plane_alt_km} km, station {station_alt_km} km, lateral offset {lateral_km} km",
        xaxis_title="along-track offset (km)  [0 = abeam the station]",
        yaxis_title="projected-area fraction",
        template="plotly_white", legend=dict(orientation="h", y=-0.2),
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.write_html(path, include_plotlyjs="inline")
    return path


if __name__ == "__main__":
    pd.set_option("display.width", 170)
    pd.set_option("display.max_columns", 20)

    print("\n=== Emitter sizing ===\n")
    print(emitter_table().to_string(index=False))

    print("\n=== Comfort duty cycle (fraction of route that must be under beam) ===")
    print("  climb angle @ 250 m/s, 10 m/s climb: %.1f°" % climb_angle_deg(250, 10))
    for ld in (15, 20, 30):
        frac = powered_fraction(250, 10, ld)
        print(f"  L/D={ld:2d}: powered {frac*100:4.0f}% of route  (glide {100-frac*100:.0f}%)")

    print("\n=== Tether height: strength is NOT the limit ===")
    for name, (sigma, rho) in TETHER_MATERIALS.items():
        bl = breaking_length_km(sigma, rho, safety_factor=2.0)
        print(f"  {name:20s} breaking length (SF2) {bl:6.0f} km  -> 20 km uses {20/bl*100:4.1f}%")

    print("\n=== Tether height: BUOYANCY is the limit (H2 sphere to lift cable+20 t payload) ===")
    print(tether_table().to_string(index=False))

    print("\n=== Pass geometry figure ===")
    print("  wrote", pass_figure())
