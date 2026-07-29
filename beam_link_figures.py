"""Visualizations for the beam-link engine — trajectory-integrated over a flyby.

The engine now flies the plane through a station's serving window and integrates the
link, rather than scoring a single snapshot geometry. Capture is the real 2-D overlap
of a focused (optionally shaped) beam spot with the rasterized, self-occluded
silhouette. These figures show what that produces.

Writes self-contained HTML figures to out/. Run:  uv run python beam_link_figures.py

Figures:
  1. flyby profiles       — the sawtooth: deliverable power vs along-track, per arch
  2. corridor efficiency  — mean & harmonic-mean η by architecture × weather
  3. spacing sweep        — mean η & grid energy vs station spacing (emitter coupling)
  4. capture overlap      — the 2-D beam × silhouette overlap along a pass (round vs
                            shaped), the static precursor to the flyby animation
  5. occlusion            — rasterized (self-occluded) vs convex silhouette area
  6. spot shape           — silhouette elongation and the emitter shape it implies
  7. frequency (physics)  — atmospheric transmittance + PAE scenarios (reference)
"""
import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import beam_link as bl
import bwb_model as m

OUT = "out"
WEATHER_ORDER = ["clear", "cloudy", "light_rain", "heavy_rain", "storm"]
WEATHER_LABEL = {"clear": "clear", "cloudy": "cloudy", "light_rain": "light rain",
                 "heavy_rain": "heavy rain", "storm": "storm"}
PALETTE = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#d62728"]
CONTENDERS = ["5.8 direct", "relay ×3 mult 5.8→17.4", "tether 30 direct"]


def _color(name, names):
    return PALETTE[names.index(name) % len(PALETTE)]


# --- 1. Flyby profiles: the sawtooth -----------------------------------------

def figure_flyby_profiles(station_spacing_km=200.0, path=f"{OUT}/beam_link_flyby.html"):
    """One tooth of the sawtooth: deliverable power (fixed emitter) vs along-track for
    each architecture, in clear air and in a storm. The relay/tether stay flat and
    weatherproof; 5.8-direct peaks high overhead but droops at the low-elevation edges
    (and rains out there in a storm)."""
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.09,
                        subplot_titles=("Clear air", "Storm"))
    archs = bl.standard_architectures(include_reference=True)
    names = list(archs)
    for col, weather in enumerate(["clear", "storm"], start=1):
        for name in names:
            kw = {**archs[name], "weather": weather}
            r = bl.flyby(kw, station_spacing_km=station_spacing_km)
            dash = "solid" if name in CONTENDERS else "dot"
            fig.add_trace(go.Scatter(
                x=r["x_km"], y=r["deliverable_mw"], mode="lines", name=name,
                legendgroup=name, showlegend=(col == 1),
                line=dict(color=_color(name, names), dash=dash,
                          width=3 if name in CONTENDERS else 1.5)), row=1, col=col)
        fig.update_xaxes(title_text="along-track position past station (km)", row=1, col=col)
    fig.update_yaxes(title_text="deliverable power (MW, fixed emitter)", row=1, col=1)
    fig.update_yaxes(title_text="deliverable power (MW)", row=1, col=2)
    fig.update_layout(
        template="plotly_white", height=520, width=1200,
        title=(f"One tooth of the sawtooth ({station_spacing_km:.0f} km spacing, plane 11 km): "
               f"a fixed emitter's deliverable power over a pass — the relay & tether stay "
               f"flat and weatherproof; 5.8-direct droops at the edges (solid = contenders)"),
        legend=dict(orientation="h", y=-0.2))
    os.makedirs(OUT, exist_ok=True)
    fig.write_html(path, include_plotlyjs="inline")
    return path


# --- 2. Corridor efficiency: integrated over the pass ------------------------

def figure_corridor_efficiency(station_spacing_km=200.0,
                               path=f"{OUT}/beam_link_corridor.html"):
    """The flyby-integrated figure of merit that replaces the snapshot η heatmap:
    time-average η (battery-buffered plane) and harmonic-mean η (continuous
    must-deliver plane), by architecture × weather."""
    archs = bl.standard_architectures()
    names = list(archs)
    mean_mat = np.zeros((len(names), len(WEATHER_ORDER)))
    hmean_mat = np.zeros_like(mean_mat)
    for j, weather in enumerate(WEATHER_ORDER):
        for i, name in enumerate(names):
            r = bl.flyby({**archs[name], "weather": weather}, station_spacing_km=station_spacing_km)
            mean_mat[i, j] = r["mean_eta"] * 100
            hmean_mat[i, j] = r["hmean_eta"] * 100

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.16,
                        subplot_titles=("Time-average η over the pass (buffered plane)",
                                        "Harmonic-mean η (continuous must-deliver plane)"))
    for col, mat in ((1, mean_mat), (2, hmean_mat)):
        fig.add_trace(go.Heatmap(
            z=mat, x=[WEATHER_LABEL[w] for w in WEATHER_ORDER], y=names,
            text=[[f"{v:.1f}" for v in row] for row in mat], texttemplate="%{text}",
            colorscale="Viridis", zmin=0, showscale=(col == 2),
            colorbar=dict(title="η %")), row=1, col=col)
    fig.update_layout(
        template="plotly_white", height=520, width=1250,
        title=(f"Integrated over the flyby ({station_spacing_km:.0f} km spacing): the relay is the "
               f"weather-robust continuous winner; 5.8-direct wins clear-air average but its "
               f"harmonic mean collapses in storm; tether is flat across weather"))
    os.makedirs(OUT, exist_ok=True)
    fig.write_html(path, include_plotlyjs="inline")
    return path


# --- 3. Station-spacing sweep -------------------------------------------------

def figure_spacing_sweep(path=f"{OUT}/beam_link_spacing.html"):
    """The emitter-size ↔ station-spacing coupling, integrated: as stations spread out,
    the plane spends more of each pass at low elevation, so mean η falls and the grid
    energy per pass rises. Shown for the contenders in clear air and storm."""
    spacings = np.linspace(60.0, 400.0, 18)
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=("Mean η over the pass vs station spacing",
                                        "Grid energy per pass vs station spacing (storm)"))
    archs = bl.standard_architectures(include_reference=False)
    names = list(archs)
    for name in names:
        mean_clear, mean_storm, gj_storm = [], [], []
        for s in spacings:
            mean_clear.append(bl.flyby({**archs[name], "weather": "clear"}, station_spacing_km=s)["mean_eta"] * 100)
            rs = bl.flyby({**archs[name], "weather": "storm"}, station_spacing_km=s)
            mean_storm.append(rs["mean_eta"] * 100)
            gj_storm.append(rs["grid_energy_per_pass_gj"])
        c = _color(name, names)
        fig.add_trace(go.Scatter(x=spacings, y=mean_clear, mode="lines", name=name,
                                 legendgroup=name, line=dict(color=c)), row=1, col=1)
        fig.add_trace(go.Scatter(x=spacings, y=mean_storm, mode="lines", name=name,
                                 legendgroup=name, showlegend=False,
                                 line=dict(color=c, dash="dot")), row=1, col=1)
        gj_storm = np.array(gj_storm, dtype=float)
        fig.add_trace(go.Scatter(x=spacings, y=gj_storm, mode="lines", name=name,
                                 legendgroup=name, showlegend=False,
                                 line=dict(color=c)), row=1, col=2)
    fig.update_xaxes(title_text="station spacing (km)", row=1, col=1)
    fig.update_xaxes(title_text="station spacing (km)", row=1, col=2)
    fig.update_yaxes(title_text="mean η (%)  — solid clear, dot storm", row=1, col=1, rangemode="tozero")
    fig.update_yaxes(title_text="grid energy per pass (GJ, storm)", row=1, col=2, rangemode="tozero")
    fig.update_layout(
        template="plotly_white", height=520, width=1180,
        title=("Station spacing sets how much of each pass is spent at low elevation: "
               "wider spacing → lower mean η and higher grid energy (∞ where a pass rains out)"),
        legend=dict(orientation="h", y=-0.2))
    os.makedirs(OUT, exist_ok=True)
    fig.write_html(path, include_plotlyjs="inline")
    return path


# --- 4. Capture overlap: beam × silhouette (animation precursor) -------------

def _overlap_snapshot(arch_kwargs, x_along_m, plane_alt_m, shaped):
    """Beam field + silhouette cells for one plane position of one architecture."""
    plane = (x_along_m, 0.0, plane_alt_m)
    cfg = bl.LinkConfig(plane_pos=plane, shaped_spot=shaped, **arch_kwargs)
    # Plane-side emitter: relay position for a relay link, else the (possibly lofted) station.
    if cfg.relay_pos is not None:
        emitter, freq, diam = cfg.relay_pos, (cfg.f_emit_relay_ghz if cfg.relay_mode == "passive"
                                              else cfg.f_relay_plane_ghz), cfg.relay_aperture_diam_m
    else:
        emitter, freq, diam = cfg.station_pos, cfg.f_emit_plane_ghz, cfg.emitter_diam_m
    slant, elevation, unit = bl.geometry(np.asarray(plane, float), np.asarray(emitter, float))
    beta, alpha = m.beta_alpha_from_enu(unit, *cfg.plane_attitude)
    raster = bl._cached_raster(beta, alpha)
    lam = bl.napkin.SPEED_OF_LIGHT / (freq * 1e9)
    geom, eff, w0, field = bl.overlap_capture(raster, diam / 2.0, lam, slant, shaped=shaped,
                                              incidence_floor=cfg.incidence_floor, return_field=True)
    return field, dict(elevation=float(elevation), geom=geom, eff=eff, w0=w0,
                       area=raster["area"], beta=float(beta))


def figure_capture_overlap(arch_name="tether 30 direct", station_spacing_km=200.0,
                           path=f"{OUT}/beam_link_overlap.html"):
    """The heart of the new model, made visible: at three points of a pass (near
    overhead, mid, low edge), the focused beam spot (filled contours) laid over the
    self-occluded silhouette (points coloured by incidence). Top row a ROUND emitter,
    bottom row a SHAPED (phased-array elliptical) emitter — the static precursor to the
    flyby animation."""
    archs = bl.standard_architectures()
    kw = archs[arch_name]
    half = station_spacing_km * 1e3 / 2.0
    positions = [(0.0, "near overhead"), (0.45 * half, "mid pass"), (0.98 * half, "low edge")]
    fig = make_subplots(
        rows=2, cols=3, horizontal_spacing=0.06, vertical_spacing=0.13,
        row_titles=("round emitter", "shaped emitter"),
        subplot_titles=[f"{lbl}" for _, lbl in positions] * 2)

    for row, shaped in ((1, False), (2, True)):
        for col, (x, lbl) in enumerate(positions, start=1):
            field, meta = _overlap_snapshot(kw, x, 11_000.0, shaped)
            u, v = field["u"], field["v"]
            # beam filled contour on a background grid
            span = max(np.abs(np.concatenate([u, v])).max(), meta["w0"]) * 1.1
            g = np.linspace(-span, span, 80)
            uu, vv = np.meshgrid(g, g)
            w0 = meta["w0"]
            beam_bg = np.exp(-2.0 * (uu ** 2 + vv ** 2) / w0 ** 2)  # round reference ring
            fig.add_trace(go.Contour(
                x=g, y=g, z=beam_bg, showscale=False, contours=dict(coloring="heatmap"),
                colorscale="Hot", opacity=0.45, line=dict(width=0),
                hoverinfo="skip"), row=row, col=col)
            fig.add_trace(go.Scatter(
                x=u, y=v, mode="markers",
                marker=dict(size=2.5, color=field["incidence"], colorscale="Viridis",
                            cmin=0, cmax=1, showscale=(row == 1 and col == 3),
                            colorbar=dict(title="incidence", x=1.02)),
                showlegend=False, hoverinfo="skip"), row=row, col=col)
            cap = meta["eff"]
            fig.add_annotation(x=0, y=span * 0.86, text=f"elev {meta['elevation']:.0f}°  cap {cap*100:.0f}%",
                               showarrow=False, font=dict(size=11), row=row, col=col)
            fig.update_xaxes(scaleanchor=f"y{'' if (row-1)*3+col==1 else (row-1)*3+col}",
                             scaleratio=1, row=row, col=col)
    fig.update_layout(
        template="plotly_white", height=760, width=1200,
        title=(f"Beam × silhouette overlap through a pass — {arch_name} (heat = beam spot, "
               f"points = self-occluded skin coloured by incidence). A round beam overspills "
               f"the low-edge sliver; a shaped beam re-forms to fit it"))
    os.makedirs(OUT, exist_ok=True)
    fig.write_html(path, include_plotlyjs="inline")
    return path


# --- 5. Occlusion correction -------------------------------------------------

def figure_occlusion(path=f"{OUT}/beam_link_occlusion.html"):
    """The convex projected-area identity OVER-counts a non-convex body; the rasterized
    projection removes the self-occlusion. Biggest at side aspect, low elevation (the
    two winglets and the far wing overlap in projection)."""
    verts, faces = m.make_bwb_mesh()
    normals, areas = m.triangle_normals_areas(verts, faces)
    samples, snorm, _ = m.get_surface_samples()
    betas = np.linspace(0.0, 88.0, 23)
    elevation = 90.0 - betas
    fig = go.Figure()
    for az, lbl, color in [(90.0, "side aspect", "#1f77b4"), (0.0, "nose aspect", "#ff7f0e")]:
        convex, raster = [], []
        for b in betas:
            d = m.direction_from_beta_alpha(b, az); d = d / np.linalg.norm(d)
            convex.append(float(m.projected_area(d, normals, areas)))
            raster.append(m.rasterize_silhouette(d, samples, snorm, n_grid=120)["area"])
        fig.add_trace(go.Scatter(x=elevation, y=convex, mode="lines", name=f"{lbl} — convex identity",
                                 line=dict(color=color, dash="dot")))
        fig.add_trace(go.Scatter(x=elevation, y=raster, mode="lines", name=f"{lbl} — rasterized (occluded)",
                                 line=dict(color=color)))
    fig.update_layout(
        template="plotly_white", height=520, width=950,
        title=("Self-occlusion correction: the convex identity over-counts (dotted); the "
               "rasterized silhouette (solid) is the true outline — the gap is largest "
               "side-on at low elevation, where winglets and far wing overlap"),
        xaxis_title="elevation (deg) — 90 = overhead, 0 = on the horizon",
        yaxis_title="silhouette area (m²)", legend=dict(orientation="h", y=-0.2))
    fig.update_yaxes(rangemode="tozero")
    os.makedirs(OUT, exist_ok=True)
    fig.write_html(path, include_plotlyjs="inline")
    return path


# --- 6. Spot shape -----------------------------------------------------------

def figure_spot_shape(path=f"{OUT}/beam_link_spot_shape.html"):
    """Silhouette elongation vs elevation, and what a shaped vs round emitter recovers —
    computed with the real overlap model (a fixed 25 m aperture at 250 km, 30 GHz)."""
    samples, snorm, _ = m.get_surface_samples()
    betas = np.linspace(0.0, 88.0, 23)
    elevation = 90.0 - betas
    lam = bl.napkin.SPEED_OF_LIGHT / 30e9
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=("Silhouette elongation vs elevation",
                                        "Overlap capture: round vs shaped emitter (side aspect)"))
    for az, lbl, color in [(90.0, "side aspect", "#1f77b4"), (0.0, "nose aspect", "#ff7f0e")]:
        aspect = [m.silhouette_aspect(b, az)[0] for b in betas]
        fig.add_trace(go.Scatter(x=elevation, y=aspect, mode="lines", name=lbl,
                                 line=dict(color=color)), row=1, col=1)
    fig.update_xaxes(title_text="elevation (deg)", row=1, col=1)
    fig.update_yaxes(title_text="silhouette aspect ratio", row=1, col=1, rangemode="tozero")

    round_cap, shaped_cap = [], []
    for b in betas:
        d = m.direction_from_beta_alpha(b, 90.0); d = d / np.linalg.norm(d)
        r = m.rasterize_silhouette(d, samples, snorm, n_grid=120)
        round_cap.append(bl.overlap_capture(r, 12.5, lam, 250e3, shaped=False, incidence_floor=1.0)[0])
        shaped_cap.append(bl.overlap_capture(r, 12.5, lam, 250e3, shaped=True, incidence_floor=1.0)[0])
    fig.add_trace(go.Scatter(x=elevation, y=round_cap, mode="lines", name="round emitter",
                             line=dict(color="#d62728")), row=1, col=2)
    fig.add_trace(go.Scatter(x=elevation, y=shaped_cap, mode="lines", name="shaped emitter",
                             line=dict(color="#2ca02c")), row=1, col=2)
    fig.update_xaxes(title_text="elevation (deg)", row=1, col=2)
    fig.update_yaxes(title_text="geometric overlap capture", row=1, col=2, rangemode="tozero")
    fig.update_layout(
        template="plotly_white", height=520, width=1150,
        title=("At low elevation the silhouette is a wide-thin sliver; a shaped (phased-array) "
               "emitter re-forms its spot to fit it, recovering capture a round beam wastes"),
        legend=dict(orientation="h", y=-0.2))
    os.makedirs(OUT, exist_ok=True)
    fig.write_html(path, include_plotlyjs="inline")
    return path


# --- 7. Frequency physics (reference) ----------------------------------------

def figure_frequency(elevation_deg=30.0, path=f"{OUT}/beam_link_frequency.html"):
    """Reference (point-eval): pure atmospheric transmittance vs frequency per weather
    (physics only), and the DC→RF PAE scenarios — the two axes behind the architecture
    choice, isolated from geometry."""
    station, plane = bl.reference_geometry(elevation_deg)
    freqs = np.linspace(5.8, 40.0, 70)
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=("Atmospheric transmittance vs frequency (physics only)",
                                        "DC→RF PAE scenarios vs frequency"))
    for weather, color in zip(["clear", "cloudy", "light_rain", "heavy_rain"], PALETTE):
        atmos = [float(bl.evaluate(bl.LinkConfig(station_pos=station, plane_pos=plane,
                 f_emit_plane_ghz=f, weather=weather, emitter_diam_m=60.0))["atmos"]) for f in freqs]
        fig.add_trace(go.Scatter(x=freqs, y=atmos, mode="lines", name=WEATHER_LABEL[weather],
                                 line=dict(color=color)), row=1, col=1)
    for scen, color in zip(["today", "medium", "flat"], ["#d62728", "#ff7f0e", "#2ca02c"]):
        pae = bl.pae_dc_to_rf(freqs, scen)
        fig.add_trace(go.Scatter(x=freqs, y=pae, mode="lines", name=f"PAE: {scen}",
                                 line=dict(color=color, dash="dash")), row=1, col=2)
    for f in (5.8, 22.0, 30.0):
        for c in (1, 2):
            fig.add_vline(x=f, line=dict(color="grey", dash="dot", width=1), row=1, col=c)
    fig.update_xaxes(title_text="frequency (GHz)", row=1, col=1)
    fig.update_xaxes(title_text="frequency (GHz)", row=1, col=2)
    fig.update_yaxes(title_text="atmospheric transmittance", row=1, col=1, range=[0, 1.02])
    fig.update_yaxes(title_text="DC→RF PAE", row=1, col=2, range=[0, 0.8])
    fig.update_layout(
        template="plotly_white", height=500, width=1150,
        title=(f"The two axes behind the frequency choice (plane 11 km, {elevation_deg:.0f}° elev): "
               f"weather is physics (left); the low-f efficiency tilt is part device scaling, "
               f"part maturity (right)"),
        legend=dict(orientation="h", y=-0.22))
    os.makedirs(OUT, exist_ok=True)
    fig.write_html(path, include_plotlyjs="inline")
    return path


def build_all():
    paths = [
        figure_flyby_profiles(),
        figure_corridor_efficiency(),
        figure_spacing_sweep(),
        figure_capture_overlap(),
        figure_occlusion(),
        figure_spot_shape(),
        figure_frequency(),
    ]
    print("wrote:")
    for p in paths:
        print("  ", p)
    return paths


if __name__ == "__main__":
    build_all()
