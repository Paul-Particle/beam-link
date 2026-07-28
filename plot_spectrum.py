"""Plot atmospheric microwave attenuation vs frequency: gaseous absorption
(ITU-R P.676, read from the committed CSV) overlaid with rain (ITU-R P.838,
computed from napkin.rain_coefficients). Writes a self-contained dark HTML.

    uv run python plot_spectrum.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import napkin

DATA = Path(__file__).parent / "data" / "atmospheric_attenuation.csv"
OUT = Path(__file__).parent / "figures" / "atmospheric_attenuation.html"

GASEOUS_LABELS = {
    "dry_cold": "Dry / cold (polar, 0.5 g/m³)",
    "temperate": "Temperate (ITU std, 7.5 g/m³)",
    "tropical_humid": "Tropical humid (20 g/m³)",
    "cruise_11km": "Cruise 11 km (thin dry air)",
    "relay_20km": "Relay 20 km (thinner still)",
}
GASEOUS_COLORS = {
    "dry_cold": "#7fb3ff",
    "temperate": "#4caf50",
    "tropical_humid": "#ff9800",
    "cruise_11km": "#b39ddb",
    "relay_20km": "#f48fb1",
}
RAIN_RATES = [(10.0, "Rain 10 mm/h (moderate)"),
              (50.0, "Rain 50 mm/h (heavy tropical)")]


def main():
    spectrum = pd.read_csv(DATA, comment="#")
    frequency_ghz = spectrum["frequency_ghz"].to_numpy()

    figure = go.Figure()

    # Gaseous absorption — one solid line per atmospheric state.
    for column, label in GASEOUS_LABELS.items():
        figure.add_trace(go.Scatter(
            x=frequency_ghz, y=spectrum[column],
            name=label, mode="lines",
            line=dict(color=GASEOUS_COLORS[column], width=2),
        ))

    # Rain — frequency-dependent specific attenuation (P.838), dashed.
    rain_k, rain_alpha = napkin.rain_coefficients(frequency_ghz)
    for rate_mmh, label in RAIN_RATES:
        figure.add_trace(go.Scatter(
            x=frequency_ghz,
            y=napkin.specific_rain_attenuation(rate_mmh, rain_k, rain_alpha),
            name=label, mode="lines",
            line=dict(color="#4dd0e1", width=1.6,
                      dash="dash" if rate_mmh == 10 else "dot"),
        ))

    # Vertical markers as Scatter primitives (traces use raw data coords on a log
    # axis reliably, unlike add_vline/add_vrect). Annotations DO take log10 coords.
    y_lo, y_hi = 2e-4, 2e2

    # O2 60 GHz absorption wall — shaded band.
    figure.add_trace(go.Scatter(
        x=[57, 63, 63, 57], y=[y_lo, y_lo, y_hi, y_hi],
        fill="toself", fillcolor="rgba(255,82,82,0.12)", line=dict(width=0),
        mode="lines", hoverinfo="skip", showlegend=False))
    figure.add_annotation(x=np.log10(60), y=np.log10(60), text="O₂ 60 GHz wall",
                          showarrow=False, font=dict(color="#ff8a80", size=11))

    # Operating frequencies (the frequency dial).
    for freq_ghz, text in [(5.8, "5.8 GHz uplink"), (30.0, "30 GHz downlink")]:
        figure.add_trace(go.Scatter(
            x=[freq_ghz, freq_ghz], y=[y_lo, y_hi], mode="lines",
            line=dict(color="#ffffff", width=1, dash="dot"), opacity=0.35,
            hoverinfo="skip", showlegend=False))
        figure.add_annotation(x=np.log10(freq_ghz), y=np.log10(40),
                              text=text, showarrow=False, textangle=-90,
                              font=dict(color="#cccccc", size=11), xshift=-8)

    # Spectral-line callouts.
    for freq_ghz, text in [(22.235, "H₂O 22 GHz"), (118.75, "O₂ 118 GHz"),
                           (183.31, "H₂O 183 GHz")]:
        figure.add_annotation(x=np.log10(freq_ghz), y=np.log10(0.0015), yref="y",
                              text=text, showarrow=False,
                              font=dict(color="#888888", size=9), textangle=-90)

    figure.update_layout(
        template="plotly_dark",
        title="Atmospheric microwave attenuation — gaseous (P.676) + rain (P.838)",
        xaxis=dict(title="Frequency (GHz)", type="log",
                   tickvals=[1, 2, 5, 10, 20, 30, 50, 60, 100, 200],
                   ticktext=["1", "2", "5", "10", "20", "30", "50", "60", "100", "200"]),
        yaxis=dict(title="Specific attenuation γ (dB/km)", type="log",
                   range=[np.log10(2e-4), np.log10(2e2)]),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.4)"),
        width=1000, height=640,
    )

    # Secondary axis: distance over which 90% of the energy survives, L90 = C/γ.
    # A pure inverse of γ, so on a log axis it is just a reversed, relabelled twin
    # (atmospheric absorption only — beam spreading is separate).
    loss_db_90 = -10.0 * np.log10(0.90)          # 0.4576 dB
    l90_km = [1000, 100, 10, 1, 0.1, 0.01]
    figure.add_trace(go.Scatter(                 # invisible anchor so y2 renders
        x=[frequency_ghz[0]], y=[loss_db_90 / y_lo],
        yaxis="y2", mode="markers", marker=dict(opacity=0),
        hoverinfo="skip", showlegend=False))
    figure.update_layout(yaxis2=dict(
        title="Distance to keep 90% of energy (absorption only)",
        overlaying="y", side="right", type="log",
        range=[np.log10(loss_db_90 / y_lo), np.log10(loss_db_90 / y_hi)],  # reversed
        tickvals=l90_km,
        ticktext=[f"{k:g} km" if k >= 1 else f"{int(k*1000)} m" for k in l90_km],
        showgrid=False))

    return figure


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    main().write_html(OUT, include_plotlyjs="cdn")
    print(f"wrote {OUT}")
