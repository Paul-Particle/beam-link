"""Flyby animation for the beam-link engine — the whole link in one moving picture.

A time slider drives four synchronized views over a single pass:
  A (2-D)  the focused beam spot (contours) over the self-occluded silhouette
           (filled by incidence) — where the power actually lands, frame by frame.
  B (3-D)  the geometry on a SPHERICAL earth: ground arc, cruise & stratosphere
           levels, station / relay / plane, and the live beam — curvature visible.
  C (3-D)  the BWB itself, each facet coloured by how squarely the beam strikes it
           (incidence), with the beam-arrival arrow.
  D (2-D)  end-to-end η along the pass, with a marker at the current position.

The point is to spot inconsistencies at a glance (does the silhouette collapse at the
edges? does the spot overspill? does the far wing go dark at grazing incidence? does
the plane track the curve?) far faster than from tables.

Run:  uv run python beam_link_animation.py   → out/beam_link_flyby_animation.html
"""
import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import beam_link as bl
import bwb_model as m
import geo

OUT = "out"


# --- one frame's worth of physics --------------------------------------------

def _localize(downrange_m, crossrange_m, altitude_m, basis):
    """ECEF point → station-local (downrange, crossrange, up) metres, so earth
    curvature shows as the ground and cruise arcs falling away at the edges."""
    base, east, north, up = basis
    d = geo.point(downrange_m, crossrange_m, altitude_m) - base
    return np.array([np.dot(d, north), np.dot(d, east), np.dot(d, up)])


def _beam_on_grid(field, w0, shaped, gx, gy):
    """The beam intensity the overlap used, evaluated on a plotting grid — round, or
    the shaped ellipse aligned to the silhouette's principal axes (matching
    overlap_capture so the picture agrees with the capture number)."""
    uu, vv = np.meshgrid(gx, gy)
    if not shaped or field["u"].size < 3:
        return np.exp(-2.0 * (uu ** 2 + vv ** 2) / w0 ** 2)
    coords = np.vstack([field["u"], field["v"]])
    evals, evecs = np.linalg.eigh(np.cov(coords))
    aspect = np.clip(np.sqrt(max(evals[1], 1e-12) / max(evals[0], 1e-12)), 1.0, 8.0)
    w_major, w_minor = w0 * np.sqrt(aspect), w0 / np.sqrt(aspect)
    flat = np.vstack([uu.ravel(), vv.ravel()])
    proj = evecs.T @ flat                       # rows: [along minor, along major]
    beam = np.exp(-2.0 * (proj[1] ** 2 / w_major ** 2 + proj[0] ** 2 / w_minor ** 2))
    return beam.reshape(uu.shape)


def _frame_physics(arch_kwargs, x_along_m, plane_alt_m, basis, coarse_mesh):
    """Everything one frame needs, for one plane position of one architecture."""
    plane = (x_along_m, 0.0, plane_alt_m)
    cfg = bl.LinkConfig(plane_pos=plane, **arch_kwargs)
    if cfg.relay_pos is not None:
        emitter = np.asarray(cfg.relay_pos, float)
        freq = cfg.f_emit_relay_ghz if cfg.relay_mode == "passive" else cfg.f_relay_plane_ghz
        diam = cfg.relay_aperture_diam_m
    else:
        emitter = np.asarray(cfg.station_pos, float)
        freq, diam = cfg.f_emit_plane_ghz, cfg.emitter_diam_m

    slant, elevation, unit = bl.geometry(np.asarray(plane, float), emitter)
    beta, alpha = m.beta_alpha_from_enu(unit, *cfg.plane_attitude)
    look = m.direction_from_beta_alpha(float(beta), float(alpha))
    raster = bl._cached_raster(beta, alpha)
    lam = bl.napkin.SPEED_OF_LIGHT / (freq * 1e9)
    _, eff, w0, field = bl.overlap_capture(raster, diam / 2.0, lam, slant,
                                           shaped=cfg.shaped_spot,
                                           incidence_floor=cfg.incidence_floor, return_field=True)
    eta = float(bl.evaluate(cfg, 10.0)["eta_end_to_end"])

    verts, faces, normals = coarse_mesh
    face_incidence = np.clip(normals @ look, 0.0, 1.0)

    plane_loc = _localize(x_along_m, 0.0, plane_alt_m, basis)
    emitter_loc = _localize(emitter[0], emitter[1], emitter[2], basis)
    return dict(field=field, w0=w0, shaped=cfg.shaped_spot, eff=eff, eta=eta,
                elevation=float(elevation), slant_km=slant / 1e3, beta=float(beta),
                area=raster["area"], look=look, face_incidence=face_incidence,
                plane_loc=plane_loc, emitter_loc=emitter_loc)


# --- the animation -----------------------------------------------------------

def build_flyby_animation(arch_name="relay ×3 mult 5.8→17.4", weather="light_rain",
                          station_spacing_km=200.0, plane_alt_m=11_000.0, n_frames=31,
                          grid_n=70, path=f"{OUT}/beam_link_flyby_animation.html"):
    archs = bl.standard_architectures(weather=weather)
    if arch_name not in archs:
        raise KeyError(f"{arch_name!r} not in {list(archs)}")
    kw = archs[arch_name]

    basis = (geo.point(0.0, 0.0, 0.0), *geo.local_enu(geo.point(0.0, 0.0, 0.0)))
    coarse_v, coarse_f = m.make_bwb_mesh(n_span=25, n_theta=16)
    coarse_n, _ = m.triangle_normals_areas(coarse_v, coarse_f)
    coarse_mesh = (coarse_v, coarse_f, coarse_n)

    half = station_spacing_km * 1e3 / 2.0
    xs = np.linspace(-half, half, n_frames)
    frames_phys = [_frame_physics(kw, x, plane_alt_m, basis, coarse_mesh) for x in xs]

    # Common panel-A extent (stable axes across the pass).
    ext = max(max(np.abs(np.concatenate([f["field"]["u"], f["field"]["v"]])).max()
                  if f["field"]["u"].size else f["w0"], f["w0"]) for f in frames_phys) * 1.1
    gx = np.linspace(-ext, ext, grid_n)
    gy = np.linspace(-ext, ext, grid_n)

    def panel_a_grids(f):
        """Silhouette incidence (NaN outside) + the beam intensity, on the common grid."""
        field = f["field"]
        inc_grid = np.full((grid_n, grid_n), np.nan)
        if field["u"].size:
            iu = np.clip(np.searchsorted(gx, field["u"]) - 1, 0, grid_n - 1)
            iv = np.clip(np.searchsorted(gy, field["v"]) - 1, 0, grid_n - 1)
            acc = np.zeros((grid_n, grid_n)); cnt = np.zeros((grid_n, grid_n))
            np.add.at(acc, (iv, iu), field["incidence"])
            np.add.at(cnt, (iv, iu), 1.0)
            covered = cnt > 0
            inc_grid[covered] = acc[covered] / cnt[covered]
        beam_grid = _beam_on_grid(field, f["w0"], f["shaped"], gx, gy)
        return inc_grid, beam_grid

    # Static 3-D geometry (ground arc, cruise + stratosphere levels, station/relay).
    arc_dr = np.linspace(-half, half, 60)
    ground = np.array([_localize(d, 0, 0, basis) for d in arc_dr]) / 1e3
    cruise = np.array([_localize(d, 0, plane_alt_m, basis) for d in arc_dr]) / 1e3
    strato = np.array([_localize(d, 0, 20_000.0, basis) for d in arc_dr]) / 1e3
    w = bl.WEATHER[weather]
    rain_top_km = w["rain_top_m"] / 1e3
    station_loc = _localize(0, 0, 0, basis) / 1e3
    has_relay = kw.get("relay_pos") is not None
    relay_loc = frames_phys[0]["emitter_loc"] / 1e3 if has_relay else None

    eta_series = np.array([f["eta"] for f in frames_phys]) * 100

    # ---- assemble base figure ----
    fig = make_subplots(
        rows=2, cols=2, horizontal_spacing=0.08, vertical_spacing=0.10,
        specs=[[{"type": "xy"}, {"type": "scene"}], [{"type": "scene"}, {"type": "xy"}]],
        subplot_titles=("A · beam spot over silhouette (fill = incidence)",
                        "B · geometry on a spherical earth",
                        "C · BWB facets by beam incidence",
                        "D · end-to-end η along the pass"))

    inc0, beam0 = panel_a_grids(frames_phys[0])
    # trace 0: silhouette incidence heatmap
    fig.add_trace(go.Heatmap(x=gx, y=gy, z=inc0, colorscale="Viridis", zmin=0, zmax=1,
                             colorbar=dict(title="incid.", len=0.42, y=0.79, x=0.44),
                             hoverinfo="skip"), row=1, col=1)
    # trace 1: beam contour (lines)
    fig.add_trace(go.Contour(x=gx, y=gy, z=beam0, showscale=False, colorscale="Hot",
                             contours=dict(coloring="lines", start=0.1, end=0.9, size=0.2),
                             line=dict(width=1.5), hoverinfo="skip"), row=1, col=1)

    # traces 2-6: static 3-D geometry
    fig.add_trace(go.Scatter3d(x=ground[:, 0], y=ground[:, 1], z=ground[:, 2], mode="lines",
                               line=dict(color="#5b4636", width=4), name="ground",
                               hoverinfo="skip"), row=1, col=2)
    fig.add_trace(go.Scatter3d(x=cruise[:, 0], y=cruise[:, 1], z=cruise[:, 2], mode="lines",
                               line=dict(color="#9fb6cd", width=1, dash="dot"), name="11 km cruise",
                               hoverinfo="skip"), row=1, col=2)
    fig.add_trace(go.Scatter3d(x=strato[:, 0], y=strato[:, 1], z=strato[:, 2], mode="lines",
                               line=dict(color="#c9c9ff", width=1, dash="dot"), name="20 km",
                               hoverinfo="skip"), row=1, col=2)
    fig.add_trace(go.Scatter3d(x=[station_loc[0]], y=[station_loc[1]], z=[station_loc[2]],
                               mode="markers", marker=dict(size=5, color="#2ca02c", symbol="square"),
                               name="station", hoverinfo="skip"), row=1, col=2)
    if has_relay:
        fig.add_trace(go.Scatter3d(x=[relay_loc[0]], y=[relay_loc[1]], z=[relay_loc[2]],
                                   mode="markers", marker=dict(size=5, color="#ff7f0e", symbol="diamond"),
                                   name="relay", hoverinfo="skip"), row=1, col=2)
    else:
        fig.add_trace(go.Scatter3d(x=[station_loc[0]], y=[station_loc[1]], z=[station_loc[2]],
                                   mode="markers", marker=dict(size=0.1, opacity=0),
                                   showlegend=False, hoverinfo="skip"), row=1, col=2)

    # trace 7: plane marker (animated); trace 8: beam line (animated)
    p0 = frames_phys[0]["plane_loc"] / 1e3
    e0 = frames_phys[0]["emitter_loc"] / 1e3
    fig.add_trace(go.Scatter3d(x=[p0[0]], y=[p0[1]], z=[p0[2]], mode="markers",
                               marker=dict(size=5, color="#d62728"), name="plane",
                               hoverinfo="skip"), row=1, col=2)
    fig.add_trace(go.Scatter3d(x=[e0[0], p0[0]], y=[e0[1], p0[1]], z=[e0[2], p0[2]], mode="lines",
                               line=dict(color="#d62728", width=3), name="beam",
                               hoverinfo="skip"), row=1, col=2)

    # trace 9: BWB mesh coloured by incidence (animated); trace 10: beam arrow (animated)
    fig.add_trace(go.Mesh3d(x=coarse_v[:, 0], y=coarse_v[:, 1], z=coarse_v[:, 2],
                            i=coarse_f[:, 0], j=coarse_f[:, 1], k=coarse_f[:, 2],
                            intensity=frames_phys[0]["face_incidence"], intensitymode="cell",
                            colorscale="Viridis", cmin=0, cmax=1, showscale=False,
                            flatshading=True, name="BWB", hoverinfo="skip"), row=2, col=1)
    look0 = frames_phys[0]["look"]
    fig.add_trace(go.Scatter3d(x=[look0[0] * 60, look0[0] * 22], y=[look0[1] * 60, look0[1] * 22],
                               z=[look0[2] * 60, look0[2] * 22], mode="lines",
                               line=dict(color="#d62728", width=5), name="beam→",
                               showlegend=False, hoverinfo="skip"), row=2, col=1)

    # trace 11: η profile (static); trace 12: current marker (animated)
    fig.add_trace(go.Scatter(x=xs / 1e3, y=eta_series, mode="lines",
                             line=dict(color="#1f77b4", width=2), showlegend=False,
                             hoverinfo="skip"), row=2, col=2)
    fig.add_trace(go.Scatter(x=[xs[0] / 1e3], y=[eta_series[0]], mode="markers",
                             marker=dict(size=10, color="#d62728"), showlegend=False,
                             hoverinfo="skip"), row=2, col=2)

    ANIMATED = [0, 1, 7, 8, 9, 10, 12]

    frames = []
    for k, f in enumerate(frames_phys):
        inc_g, beam_g = panel_a_grids(f)
        p = f["plane_loc"] / 1e3
        e = f["emitter_loc"] / 1e3
        look = f["look"]
        frames.append(go.Frame(name=str(k), traces=ANIMATED, data=[
            go.Heatmap(x=gx, y=gy, z=inc_g),
            go.Contour(x=gx, y=gy, z=beam_g),
            go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]]),
            go.Scatter3d(x=[e[0], p[0]], y=[e[1], p[1]], z=[e[2], p[2]]),
            go.Mesh3d(intensity=f["face_incidence"], intensitymode="cell"),
            go.Scatter3d(x=[look[0] * 60, look[0] * 22], y=[look[1] * 60, look[1] * 22],
                         z=[look[2] * 60, look[2] * 22]),
            go.Scatter(x=[xs[k] / 1e3], y=[eta_series[k]]),
        ]))
    fig.frames = frames

    steps = [dict(method="animate", label=f"{xs[k]/1e3:.0f}",
                  args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))]) for k in range(n_frames)]
    fig.update_layout(
        sliders=[dict(active=0, y=-0.02, x=0.05, len=0.9, currentvalue=dict(
            prefix="along-track: ", suffix=" km"), steps=steps)],
        updatemenus=[dict(type="buttons", showactive=False, x=0.05, y=0.08, xanchor="right",
                          buttons=[dict(label="▶ play", method="animate",
                                        args=[None, dict(frame=dict(duration=120, redraw=True),
                                                         fromcurrent=True, mode="immediate")]),
                                   dict(label="❚❚ pause", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                           mode="immediate")])])])

    # panel A square + metres; D axes; 3-D scenes framed to show curvature
    fig.update_xaxes(title_text="view-plane u (m)", scaleanchor="y", scaleratio=1, row=1, col=1)
    fig.update_yaxes(title_text="v (m)", row=1, col=1)
    fig.update_xaxes(title_text="along-track (km)", row=2, col=2)
    fig.update_yaxes(title_text="η end-to-end (%)", rangemode="tozero", row=2, col=2)
    fig.update_scenes(dict(xaxis_title="downrange (km)", yaxis_title="cross (km)",
                           zaxis_title="altitude (km)", aspectmode="manual",
                           aspectratio=dict(x=3, y=0.5, z=1),
                           camera=dict(eye=dict(x=0.2, y=-2.4, z=0.6))), row=1, col=2)
    fig.update_scenes(dict(aspectmode="data",
                           camera=dict(eye=dict(x=1.4, y=-1.4, z=1.1))), row=2, col=1)
    fig.update_layout(
        template="plotly_white", height=880, width=1350,
        title=(f"Flyby animation — {arch_name}, {weather} weather, "
               f"{station_spacing_km:.0f} km spacing (drag the slider or press play)"),
        legend=dict(x=0.46, y=0.98, font=dict(size=10)))
    os.makedirs(OUT, exist_ok=True)
    fig.write_html(path, include_plotlyjs="inline", auto_play=False)
    return path


if __name__ == "__main__":
    for arch, weather in [("relay ×3 mult 5.8→17.4", "light_rain"),
                          ("5.8 direct", "storm"),
                          ("tether 30 direct", "clear")]:
        tag = arch.split()[0].replace(".", "")
        p = build_flyby_animation(arch_name=arch, weather=weather,
                                  path=f"{OUT}/beam_link_flyby_animation_{tag}.html")
        print("wrote", p)
