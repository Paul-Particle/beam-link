"""Napkin math for microwave-powered aviation — pushing on the envelope.

First-order relations behind the 'Premise & target envelope' section of
research-questions.md. SI units unless a name says otherwise. Every number this
produces is #(verify)-grade: order-of-magnitude sanity checks, not engineering.

Run:  uv run python napkin.py   → prints scenario table, writes out/pass_geometry.html
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

SPEED_OF_LIGHT = 2.998e8  # m/s
GRAVITY = 9.81            # m/s^2


# --- Beam coupling (the one fundamental limit) --------------------------------

def wavelength(frequency_hz: float) -> float:
    return SPEED_OF_LIGHT / frequency_hz


def coupling_tau(frequency_hz, distance_m, emitter_radius_m, receiver_radius_m):
    """Goubau aperture-coupling parameter τ = π·r_t·r_r / (λ·D).

    τ≈1 → ~50% collected, τ≈2 → ~94%, τ≥2.5 → ~99%.
    """
    return (
        np.pi * emitter_radius_m * receiver_radius_m
        / (wavelength(frequency_hz) * distance_m)
    )


def collection_efficiency(tau):
    """η ≈ 1 − exp(−τ²). Closed-form intuition; exact constant depends on taper."""
    return 1.0 - np.exp(-np.square(tau))


def required_emitter_radius(frequency_hz, distance_m, receiver_radius_m, tau_target=2.0):
    """Invert τ for the emitter radius needed to hit a target coupling: r_t = τ·λ·D/(π·r_r)."""
    return tau_target * wavelength(frequency_hz) * distance_m / (np.pi * receiver_radius_m)


def surface_precision(frequency_hz, fraction=1 / 16):
    """Phased-array surface/phase accuracy needed across the aperture (~λ/16)."""
    return wavelength(frequency_hz) * fraction


# --- Route / buffer geometry --------------------------------------------------

def worst_case_link_distance(route_km, n_stations, plane_alt_m, station_alt_m):
    """Plane midway between two evenly spaced stations: slant range at that point."""
    spacing_m = (route_km * 1e3) / n_stations
    horizontal = spacing_m / 2.0
    return float(np.hypot(plane_alt_m - station_alt_m, horizontal))


def glide_range_km(lift_to_drag, altitude_drop_m):
    """Unpowered glide reach = (L/D)·Δh — the honest 'altitude battery' framing."""
    return lift_to_drag * altitude_drop_m / 1e3


def altitude_energy_kwh(mass_kg, altitude_gain_m):
    return mass_kg * GRAVITY * altitude_gain_m / 3.6e6


def pass_geometry(plane_alt_m, station_alt_m, alongtrack_m):
    """A plane passing a station along-track (0 = overhead).

    Returns slant distance (m), elevation angle (deg, at the station), and the
    projected-area FRACTION for a horizontal 'belly' rectenna vs a vertical
    fore/aft 'sail'. Flat-earth, single vertical plane.
    """
    height_gap = plane_alt_m - station_alt_m
    slant = np.hypot(height_gap, alongtrack_m)
    elevation = np.arcsin(height_gap / slant)
    belly_fraction = np.sin(elevation)   # horizontal surface, full overhead
    sail_fraction = np.cos(elevation)    # vertical fore/aft, full at the horizon
    return slant, np.degrees(elevation), belly_fraction, sail_fraction


# --- Scenarios ----------------------------------------------------------------

def scenario_table() -> pd.DataFrame:
    scenarios = [
        # name, freq GHz, D km, rectenna radius m, station alt km
        ("baseline  5.8 GHz, sea-level", 5.8, 300, 20, 0.0),
        ("+freq     30 GHz,  tethered ~5 km", 30.0, 300, 25, 5.0),
        ("+freq+recv 30 GHz, 76 m rectenna", 30.0, 300, 38, 20.0),
        ("comfort   30 GHz, D=195 km", 30.0, 195, 25, 20.0),
    ]
    return (
        pd.DataFrame(scenarios, columns=["scenario", "f_GHz", "D_km", "r_r_m", "stn_alt_km"])
        .assign(
            r_t_m=lambda d: required_emitter_radius(
                d["f_GHz"] * 1e9, d["D_km"] * 1e3, d["r_r_m"], tau_target=2.0
            ),
            emitter_diam_m=lambda d: 2 * d["r_t_m"],
            eta_at_tau2=lambda d: collection_efficiency(2.0),
            surface_mm=lambda d: surface_precision(d["f_GHz"] * 1e9) * 1e3,
        )
        .round({"r_t_m": 0, "emitter_diam_m": 0, "eta_at_tau2": 2, "surface_mm": 2})
    )


def pass_figure(plane_alt_km=12.0, station_alt_km=5.0, span_km=120.0, path="out/pass_geometry.html"):
    alongtrack = np.linspace(-span_km * 1e3, span_km * 1e3, 400)
    slant, elevation_deg, belly, sail = pass_geometry(
        plane_alt_km * 1e3, station_alt_km * 1e3, alongtrack
    )
    x_km = alongtrack / 1e3

    fig = go.Figure()
    fig.add_scatter(x=x_km, y=belly, name="belly rectenna  ∝ sin(elev)", line=dict(width=3))
    fig.add_scatter(x=x_km, y=sail, name="sail (fore/aft)  ∝ cos(elev)", line=dict(width=3, dash="dash"))
    fig.add_scatter(x=x_km, y=elevation_deg / 90.0, name="elevation / 90°", line=dict(width=1), opacity=0.5)
    fig.add_scatter(x=x_km, y=slant / slant.max(), name="slant / max", line=dict(width=1, dash="dot"), opacity=0.5)
    fig.update_layout(
        title=f"Single pass — plane {plane_alt_km} km, station {station_alt_km} km",
        xaxis_title="along-track offset from station (km)   [0 = overhead]",
        yaxis_title="projected-area fraction  /  normalized",
        template="plotly_white",
        legend=dict(orientation="h", y=-0.2),
    )
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.write_html(path, include_plotlyjs="inline")
    return path


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print("\n=== Emitter sizing: pushing the envelope ===\n")
    print(scenario_table().to_string(index=False))

    print("\n=== Altitude buffer (glide range = L/D × Δh) ===")
    for ld in (15, 20, 25):
        for drop_km in (5, 8, 10):
            print(f"  L/D={ld:2d}, drop {drop_km:2d} km  ->  glide {glide_range_km(ld, drop_km*1e3):5.0f} km")

    print("\n=== Pass geometry figure ===")
    print("  wrote", pass_figure())
