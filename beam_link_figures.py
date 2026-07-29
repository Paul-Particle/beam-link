"""Visualizations for the beam-link engine — every architecture we've weighed.

Writes self-contained HTML figures to out/. Run:  uv run python beam_link_figures.py

Figures:
  1. efficiency cascade (waterfall) for each architecture — where the power goes
  2. frequency sweep — the aperture-vs-electronics-vs-weather optimum
  3. geometry sweep — the low-elevation double penalty (big emitter, small aperture)
  4. architecture × weather map — why the relay is weather insurance, not efficiency
  5. attitude — how banking and pass geometry move the silhouette aperture
"""
import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import beam_link as bl

OUT = "out"
WEATHER_ORDER = ["clear", "cloudy", "light_rain", "heavy_rain", "storm"]
WEATHER_LABEL = {"clear": "clear", "cloudy": "cloudy", "light_rain": "light rain",
                 "heavy_rain": "heavy rain", "storm": "storm"}
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
PAE_SCENARIOS = ["today", "medium", "flat"]


def _relay_pos(station, plane, alt_m=20_000.0):
    return (station[0] * 0.5 + plane[0] * 0.5, 0.0, alt_m)


def architectures(station, plane, weather="clear", pae_scenario="today"):
    """The architectures we've weighed, at a common geometry, as {name: LinkConfig}."""
    relay = _relay_pos(station, plane)
    common = dict(station_pos=station, plane_pos=plane, weather=weather, pae_scenario=pae_scenario)
    return {
        "5.8 GHz direct": bl.LinkConfig(**common, f_emit_plane_ghz=5.8),
        "30 GHz direct": bl.LinkConfig(**common, f_emit_plane_ghz=30.0),
        "relay ×3 mult 5.8→17.4": bl.LinkConfig(**common, relay_pos=relay, relay_mode="multiplier",
                                                f_emit_relay_ghz=5.8, f_relay_plane_ghz=17.4),
        "relay ×6 mult 5.8→34.8": bl.LinkConfig(**common, relay_pos=relay, relay_mode="multiplier",
                                                f_emit_relay_ghz=5.8, f_relay_plane_ghz=34.8),
        "relay active 5.8→30": bl.LinkConfig(**common, relay_pos=relay, relay_mode="active",
                                             f_emit_relay_ghz=5.8, f_relay_plane_ghz=30.0),
        "relay passive 5.8": bl.LinkConfig(**common, relay_pos=relay, relay_mode="passive",
                                           f_emit_relay_ghz=5.8),
    }


# --- 1. Efficiency cascade ----------------------------------------------------

def figure_waterfall(elevation_deg=30.0, weather="clear", delivered_mw=10.0,
                     path=f"{OUT}/beam_link_waterfall.html"):
    station, plane = bl.reference_geometry(elevation_deg)
    cases = architectures(station, plane, weather)
    names = list(cases)
    fig = make_subplots(rows=3, cols=2, subplot_titles=names,
                        vertical_spacing=0.11, horizontal_spacing=0.09)
    for idx, (name, cfg) in enumerate(cases.items()):
        stages = bl.evaluate(cfg, delivered_mw)
        chain = bl.waterfall(stages)
        eta = float(stages["eta_end_to_end"])
        labels, effs = zip(*chain)
        cum = np.cumprod(effs)
        cum_before = np.concatenate([[1.0], cum[:-1]])
        losses_pct = (cum_before - cum) * 100.0
        x = ["grid"] + list(labels) + ["delivered"]
        measure = ["absolute"] + ["relative"] * len(labels) + ["total"]
        y = [100.0] + [-l for l in losses_pct] + [0.0]
        row, col = idx // 2 + 1, idx % 2 + 1
        fig.add_trace(go.Waterfall(
            x=x, y=y, measure=measure, orientation="v",
            decreasing=dict(marker=dict(color="#d62728")),
            increasing=dict(marker=dict(color="#2ca02c")),
            totals=dict(marker=dict(color="#1f77b4")),
            connector=dict(line=dict(color="rgba(120,120,120,0.4)")),
            text=[f"{eta*100:.1f}%" if m == "total" else "" for m in measure],
            textposition="outside", showlegend=False), row=row, col=col)
        fig.update_yaxes(range=[0, 105], row=row, col=col,
                         title_text="% of grid power" if col == 1 else None)
        fig.update_xaxes(tickangle=-40, row=row, col=col)
    fig.update_layout(
        template="plotly_white", height=1000, width=1200,
        title=(f"Where the power goes — efficiency cascade per architecture "
               f"(plane 11 km, {elevation_deg:.0f}° elevation, {WEATHER_LABEL[weather]}, "
               f"{delivered_mw:.0f} MW delivered)"))
    os.makedirs(OUT, exist_ok=True)
    fig.write_html(path, include_plotlyjs="inline")
    return path


# --- 2. Frequency sweep -------------------------------------------------------

def figure_frequency(elevation_deg=30.0, delivered_mw=10.0,
                     path=f"{OUT}/beam_link_frequency.html"):
    """Three panels: (A) efficiency vs freq under the three device-efficiency
    scenarios — how much of the low-frequency tilt is our PAE assumption vs physics;
    (B) PURE atmospheric transmittance vs freq per weather (ITU-R only, no device
    assumptions) — the honest weather picture; (C) emitter diameter vs freq."""
    station, plane = bl.reference_geometry(elevation_deg)
    freqs = np.linspace(5.8, 40.0, 70)
    fig = make_subplots(
        rows=1, cols=3, horizontal_spacing=0.075,
        subplot_titles=("η vs frequency — 3 device scenarios (clear air)",
                        "Atmospheric transmittance vs frequency (physics only)",
                        "Emitter diameter vs frequency (τ=2)"))

    for scen, color in zip(PAE_SCENARIOS, PALETTE):
        eta = [float(bl.evaluate(bl.LinkConfig(station_pos=station, plane_pos=plane,
               f_emit_plane_ghz=f, pae_scenario=scen), delivered_mw)["eta_end_to_end"])
               for f in freqs]
        fig.add_trace(go.Scatter(x=freqs, y=eta, mode="lines", name=f"PAE: {scen}",
                                 line=dict(color=color)), row=1, col=1)

    for weather, color in zip(["clear", "cloudy", "light_rain", "heavy_rain"], PALETTE):
        # 'atmos' from evaluate is the pure ITU-R transmittance — no device terms.
        atmos = [float(bl.evaluate(bl.LinkConfig(station_pos=station, plane_pos=plane,
                 f_emit_plane_ghz=f, weather=weather), delivered_mw)["atmos"]) for f in freqs]
        fig.add_trace(go.Scatter(x=freqs, y=atmos, mode="lines", name=WEATHER_LABEL[weather],
                                 line=dict(color=color, dash="dot")), row=1, col=2)

    diam = [float(bl.evaluate(bl.LinkConfig(station_pos=station, plane_pos=plane,
            f_emit_plane_ghz=f), delivered_mw)["emitter_diam_m"]) for f in freqs]
    fig.add_trace(go.Scatter(x=freqs, y=diam, mode="lines", name="emitter Ø",
                             line=dict(color="#111", width=3), showlegend=False), row=1, col=3)

    for f in (5.8, 22.0, 30.0):
        for c in (1, 2, 3):
            fig.add_vline(x=f, line=dict(color="grey", dash="dot", width=1), row=1, col=c)
    for c in (1, 2, 3):
        fig.update_xaxes(title_text="frequency (GHz)", row=1, col=c)
    fig.update_yaxes(title_text="η end-to-end", row=1, col=1, rangemode="tozero")
    fig.update_yaxes(title_text="atmospheric transmittance", row=1, col=2, range=[0, 1.02])
    fig.update_yaxes(title_text="emitter diameter (m)", type="log", row=1, col=3)
    fig.update_layout(template="plotly_white", height=520, width=1500,
                      title=(f"Frequency dial (plane 11 km, {elevation_deg:.0f}° elev): the "
                             f"efficiency tilt to low f is mostly the PAE assumption (A); the real "
                             f"physics is the weather cliff (B) and the emitter blow-up (C)"),
                      legend=dict(orientation="h", y=-0.2))
    os.makedirs(OUT, exist_ok=True)
    fig.write_html(path, include_plotlyjs="inline")
    return path


# --- 3. Geometry sweep --------------------------------------------------------

def figure_geometry(frequency_ghz=30.0, delivered_mw=10.0,
                    path=f"{OUT}/beam_link_geometry.html"):
    elevs = np.linspace(8.0, 60.0, 60)
    diam, silh, eta = [], [], []
    for e in elevs:
        station, plane = bl.reference_geometry(e)
        s = bl.evaluate(bl.LinkConfig(station_pos=station, plane_pos=plane,
                                      f_emit_plane_ghz=frequency_ghz), delivered_mw)
        diam.append(float(s["emitter_diam_m"]))
        silh.append(float(s["silhouette_m2"]))
        eta.append(float(s["eta_end_to_end"]))

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        specs=[[{"secondary_y": True}, {}]],
                        subplot_titles=("Emitter grows while plane silhouette shrinks",
                                        "End-to-end efficiency vs elevation"))
    fig.add_trace(go.Scatter(x=elevs, y=diam, name="emitter Ø (m)",
                             line=dict(color="#d62728", width=3)), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=elevs, y=silh, name="plane silhouette (m²)",
                             line=dict(color="#2ca02c", width=3)), row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=elevs, y=100 * np.array(eta), name="η (%)",
                             line=dict(color="#1f77b4", width=3)), row=1, col=2)
    fig.add_vline(x=15.0, line=dict(color="grey", dash="dash"), row=1, col=1)
    fig.add_vline(x=15.0, line=dict(color="grey", dash="dash"), row=1, col=2)
    fig.update_xaxes(title_text="elevation of plane from station (deg)", row=1, col=1)
    fig.update_xaxes(title_text="elevation of plane from station (deg)", row=1, col=2)
    fig.update_yaxes(title_text="emitter diameter (m)", type="log", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="silhouette (m²)", row=1, col=1, secondary_y=True, rangemode="tozero")
    fig.update_yaxes(title_text="η end-to-end (%)", row=1, col=2, rangemode="tozero")
    fig.update_layout(
        template="plotly_white", height=520, width=1150,
        title=(f"The low-elevation double penalty at {frequency_ghz:.0f} GHz: as the plane "
               f"sits lower, the emitter must GROW while the plane's aperture SHRINKS"),
        legend=dict(orientation="h", y=-0.18))
    os.makedirs(OUT, exist_ok=True)
    fig.write_html(path, include_plotlyjs="inline")
    return path


# --- 4. Architecture × weather map -------------------------------------------

def figure_architecture_weather(elevation_deg=30.0, delivered_mw=10.0,
                                path=f"{OUT}/beam_link_architecture_weather.html"):
    station, plane = bl.reference_geometry(elevation_deg)
    names = list(architectures(station, plane))
    matrix = np.zeros((len(names), len(WEATHER_ORDER)))
    for j, weather in enumerate(WEATHER_ORDER):
        cases = architectures(station, plane, weather)
        for i, name in enumerate(names):
            matrix[i, j] = float(bl.evaluate(cases[name], delivered_mw)["eta_end_to_end"])

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.52, 0.48], horizontal_spacing=0.12,
        subplot_titles=("End-to-end efficiency (%) by architecture × weather",
                        "Grid power to deliver 10 MW (MW; capped at 200)"))

    text = [[f"{100*v:.1f}" if v > 1e-3 else "✗" for v in row] for row in matrix]
    fig.add_trace(go.Heatmap(
        z=matrix * 100, x=[WEATHER_LABEL[w] for w in WEATHER_ORDER], y=names,
        text=text, texttemplate="%{text}", colorscale="Viridis",
        colorbar=dict(title="η %", x=0.44), zmin=0), row=1, col=1)

    grid = np.where(matrix > 1e-3, np.minimum(delivered_mw / np.maximum(matrix, 1e-9), 200.0), np.nan)
    for i, name in enumerate(names):
        fig.add_trace(go.Bar(x=[WEATHER_LABEL[w] for w in WEATHER_ORDER], y=grid[i],
                             name=name, marker_color=PALETTE[i]), row=1, col=2)
    fig.update_yaxes(title_text="grid power (MW)", row=1, col=2)
    fig.update_layout(
        template="plotly_white", height=560, width=1220, barmode="group",
        title=(f"Direct high-frequency links are rained out; the 5.8→30 relay keeps a "
               f"small-aperture plane fed through weather (plane 11 km, {elevation_deg:.0f}° elev)"),
        legend=dict(orientation="h", y=-0.18))
    os.makedirs(OUT, exist_ok=True)
    fig.write_html(path, include_plotlyjs="inline")
    return path


# --- 5. Attitude / pass geometry ---------------------------------------------

def figure_attitude(frequency_ghz=30.0, delivered_mw=10.0,
                    path=f"{OUT}/beam_link_attitude.html"):
    sil = bl.get_silhouette()
    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.12,
        subplot_titles=("Silhouette vs bank angle (emitter straight below)",
                        "Silhouette through a pass (level vs 20° bank into station)"))

    # Bank sweep with emitter directly below.
    from bwb_model import beta_alpha_from_enu
    rolls = np.linspace(0, 60, 61)
    area_bank = [float(sil.area(*beta_alpha_from_enu([0, 0, -1], roll_deg=r))) for r in rolls]
    fig.add_trace(go.Scatter(x=rolls, y=area_bank, mode="lines",
                             line=dict(color="#d62728", width=3), showlegend=False), row=1, col=1)
    fig.update_xaxes(title_text="bank (roll) angle (deg)", row=1, col=1)
    fig.update_yaxes(title_text="silhouette (m²)", row=1, col=1, rangemode="tozero")

    # Pass: plane at 11 km flies over a station along track; compute silhouette vs
    # along-track offset for level flight and a 20° bank toward the station.
    plane_alt = 11_000.0
    x = np.linspace(-80_000.0, 80_000.0, 200)
    for roll, label, color in [(0.0, "level", "#1f77b4"), (20.0, "20° bank in", "#ff7f0e")]:
        areas = []
        for xt in x:
            plane_pos = np.array([xt, 0.0, plane_alt])
            _, _, unit = bl.geometry(plane_pos, np.array([0.0, 0.0, 0.0]))  # plane→station
            beta, alpha = beta_alpha_from_enu(unit, heading_deg=0.0, pitch_deg=0.0,
                                              roll_deg=np.sign(xt) * roll)
            areas.append(float(sil.area(beta, alpha)))
        fig.add_trace(go.Scatter(x=x / 1e3, y=areas, mode="lines", name=label,
                                 line=dict(color=color, width=2)), row=1, col=2)
    fig.update_xaxes(title_text="along-track offset from station (km)", row=1, col=2)
    fig.update_yaxes(title_text="silhouette (m²)", row=1, col=2, rangemode="tozero")
    fig.update_layout(
        template="plotly_white", height=520, width=1150,
        title="Attitude matters: banking tilts the belly off the beam; a well-timed "
              "bank can present more aperture on the approach",
        legend=dict(orientation="h", y=-0.2))
    os.makedirs(OUT, exist_ok=True)
    fig.write_html(path, include_plotlyjs="inline")
    return path


# --- 6. Pass profile: the powering window ------------------------------------

def figure_pass(frequency_ghz=30.0, speed_ms=250.0, path=f"{OUT}/beam_link_pass.html"):
    station = (0.0, 0.0, 0.0)
    ref_elevs = [80.0, 60.0, 40.0, 20.0, 10.0]
    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.12,
        specs=[[{}, {"secondary_y": True}]],
        subplot_titles=("Deliverable power through a pass (fixed emitter, normalised)",
                        "Powering window & dwell vs emitter size"))

    widths_km, dwell_s, diams = [], [], []
    for ref, color in zip(ref_elevs, PALETTE):
        diam = bl.emitter_for_reference(frequency_ghz, station, reference_elevation_deg=ref)
        p = bl.pass_profile(frequency_ghz=frequency_ghz, emitter_diam_m=diam)
        norm = p["deliverable_mw"] / np.max(p["deliverable_mw"])
        fig.add_trace(go.Scatter(x=p["alongtrack_km"], y=norm, mode="lines",
                                 name=f"Ø {diam:.0f} m (built for {ref:.0f}°)",
                                 line=dict(color=color)), row=1, col=1)
        mask = norm >= 0.5
        xin = p["alongtrack_km"][mask]
        w = float(xin.max() - xin.min()) if mask.any() else 0.0
        widths_km.append(w); dwell_s.append(w * 1e3 / speed_ms); diams.append(diam)
    fig.add_trace(go.Scatter(x=[-120, 120], y=[0.5, 0.5], mode="lines",
                             line=dict(color="grey", dash="dot"), name="50%"), row=1, col=1)
    fig.update_xaxes(title_text="along-track offset from station (km)", row=1, col=1)
    fig.update_yaxes(title_text="deliverable power (fraction of overhead)", row=1, col=1)

    fig.add_trace(go.Scatter(x=diams, y=widths_km, mode="lines+markers",
                             name="≥50% window (km)", line=dict(color="#1f77b4")),
                  row=1, col=2, secondary_y=False)
    fig.add_trace(go.Scatter(x=diams, y=dwell_s, mode="lines+markers",
                             name=f"dwell @ {speed_ms:.0f} m/s (s)", line=dict(color="#d62728")),
                  row=1, col=2, secondary_y=True)
    fig.update_xaxes(title_text="emitter diameter (m)", row=1, col=2)
    fig.update_yaxes(title_text="≥50% window width (km)", row=1, col=2, secondary_y=False, rangemode="tozero")
    fig.update_yaxes(title_text=f"dwell @ {speed_ms:.0f} m/s (s)", row=1, col=2, secondary_y=True, rangemode="tozero")
    fig.update_layout(
        template="plotly_white", height=520, width=1180,
        title=(f"Boost-over-station window at {frequency_ghz:.0f} GHz: a bigger emitter buys a wider "
               f"powering window — the emitter-size ↔ station-spacing coupling"),
        legend=dict(orientation="h", y=-0.2))
    os.makedirs(OUT, exist_ok=True)
    fig.write_html(path, include_plotlyjs="inline")
    return path


# --- 7. Receive-side derate sensitivity --------------------------------------

def figure_derate_sensitivity(frequency_ghz=30.0, elevation_deg=30.0, delivered_mw=10.0,
                              path=f"{OUT}/beam_link_derate_sensitivity.html"):
    """How much do the 'simple approximation' receive-side factors move the answer?
    Sweep each around its default, holding the others, and show the swing in η."""
    station, plane = bl.reference_geometry(elevation_deg)
    base = dict(station_pos=station, plane_pos=plane, f_emit_plane_ghz=frequency_ghz)
    sweeps = {
        "spot_match": np.linspace(0.6, 1.0, 21),
        "incidence_floor": np.linspace(0.5, 1.0, 21),
        "pointing_error_m": np.linspace(0.0, 8.0, 21),
    }
    fig = go.Figure()
    for (lever, values), color in zip(sweeps.items(), PALETTE):
        eta = []
        for v in values:
            cfg = bl.LinkConfig(**base, **{lever: float(v)})
            eta.append(float(bl.evaluate(cfg, delivered_mw)["eta_end_to_end"]))
        # normalise x to 0..1 across each lever's range for a shared axis
        xn = (values - values.min()) / (values.max() - values.min())
        fig.add_trace(go.Scatter(x=xn, y=eta, mode="lines", name=lever, line=dict(color=color)))
    fig.update_layout(
        template="plotly_white", height=500, width=900,
        title=(f"Sensitivity of η to the 'simple approximation' receive-side factors "
               f"({frequency_ghz:.0f} GHz, {elevation_deg:.0f}° elev) — which knobs actually matter"),
        xaxis_title="each lever swept low→high across its plausible range (normalised)",
        yaxis_title="η end-to-end", legend=dict(orientation="h", y=-0.18))
    fig.update_yaxes(rangemode="tozero")
    os.makedirs(OUT, exist_ok=True)
    fig.write_html(path, include_plotlyjs="inline")
    return path


# --- 8. Spot shaping: silhouette elongation and the emitter shape it implies ---

def figure_spot_shape(path=f"{OUT}/beam_link_spot_shape.html"):
    """The silhouette is a wide-thin sliver at low elevation, so a round beam wastes
    most of its power. Panel A: silhouette aspect ratio vs elevation. Panel B: what a
    round vs a shape-matched (elliptical, phased-array) spot recovers — and the emitter
    aperture must be elongated PERPENDICULAR to the sliver to paint the matching spot."""
    import bwb_model as m
    betas = np.linspace(0.0, 88.0, 23)
    elevation = 90.0 - betas
    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.12,
        subplot_titles=("Silhouette elongation vs elevation",
                        "Spot-match: round vs shaped emitter (side aspect)"))

    for az, lbl, color in [(90.0, "side aspect (lateral pass)", "#1f77b4"),
                           (0.0, "nose aspect (head-on)", "#ff7f0e")]:
        aspect = [m.silhouette_aspect(b, az)[0] for b in betas]
        fig.add_trace(go.Scatter(x=elevation, y=aspect, mode="lines", name=lbl,
                                 line=dict(color=color)), row=1, col=1)
    fig.update_xaxes(title_text="elevation (deg) — 90=overhead, 0=on the horizon", row=1, col=1)
    fig.update_yaxes(title_text="silhouette aspect ratio (major/minor)", row=1, col=1, rangemode="tozero")

    aspect_side = np.array([m.silhouette_aspect(b, 90.0)[0] for b in betas])
    sm_round = bl.spot_match_from_aspect(aspect_side, shaped=False)
    sm_shaped = bl.spot_match_from_aspect(aspect_side, shaped=True) * np.ones_like(aspect_side)
    fig.add_trace(go.Scatter(x=elevation, y=sm_round, mode="lines", name="round emitter",
                             line=dict(color="#d62728")), row=1, col=2)
    fig.add_trace(go.Scatter(x=elevation, y=sm_shaped, mode="lines",
                             name="shaped (phased-array) emitter", line=dict(color="#2ca02c")),
                  row=1, col=2)
    fig.update_xaxes(title_text="elevation (deg)", row=1, col=2)
    fig.update_yaxes(title_text="spot-match efficiency", row=1, col=2, range=[0, 1.0])
    fig.update_layout(
        template="plotly_white", height=520, width=1150,
        title=("Spot shaping: at low elevation the silhouette is a wide-thin sliver — a round "
               "beam couples ~1/aspect, so the emitter must synthesise an elongated spot "
               "(a phased array can; a fixed aperture is a permanent portrait/landscape choice)"),
        legend=dict(orientation="h", y=-0.2))
    os.makedirs(OUT, exist_ok=True)
    fig.write_html(path, include_plotlyjs="inline")
    return path


def build_all():
    paths = [
        figure_waterfall(),
        figure_frequency(),
        figure_geometry(),
        figure_architecture_weather(),
        figure_attitude(),
        figure_pass(),
        figure_derate_sensitivity(),
        figure_spot_shape(),
    ]
    print("wrote:")
    for p in paths:
        print("  ", p)
    return paths


if __name__ == "__main__":
    build_all()
