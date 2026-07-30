"""Phone-friendly view: just the two beam-intensity channel cuts (major & minor axis).

A compact, single-column animation of panels F / F₂ from the full flyby — the converging
beam channel in its two orthogonal cross-sections, with the encircled-energy funnel and the
plane slice — small enough to open on a phone. Reuses the full engine + frame physics from
beam_link_animation, so it stays in lock-step with the main figure.

Run:  uv run python beam_link_phone.py   → out/flyby_phone.html
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import beam_link as bl
import bwb_model as m
import geo
import beam_link_animation as anim

OUT = "out"


def build_phone_channels(arch_name="5.8 direct", weather="clear", lateral_offset_km=5.0,
                         emitter_diam_m=100.0, n_frames=25, grid_dist=60, grid_radial=90,
                         plane_alt_m=11_000.0, min_elevation_deg=2.0, delivered_mw=10.0,
                         write=True, path=f"{OUT}/flyby_phone.html"):
    kw = dict(bl.standard_architectures(weather=weather)[arch_name])
    kw["emitter_diam_m"] = emitter_diam_m
    lateral_m = lateral_offset_km * 1e3
    basis = (geo.point(0.0, 0.0, 0.0), *geo.local_enu(geo.point(0.0, 0.0, 0.0)))
    cv, cf = m.make_bwb_mesh(n_span=41, n_theta=26)
    cn, _ = m.triangle_normals_areas(cv, cf)
    mesh = (cv, cf, cn)
    half = float(geo.downrange_for_elevation(min_elevation_deg, plane_alt_m))
    xs = np.linspace(-half, half, n_frames)
    fp = [anim._frame_physics(kw, x, plane_alt_m, basis, mesh, lateral_offset_m=lateral_m,
                              delivered_mw=delivered_mw) for x in xs]

    max_total = max(f["chan_total"] for f in fp)
    R = 1.4 * max(max(f["chan_emit_radius_m"] for f in fp), max(f["sil_major_m"] for f in fp))
    dist = np.linspace(0.0, 1.0, grid_dist)
    radial = np.linspace(-R, R, grid_radial)
    r_t0 = fp[0]["chan_emit_radius_m"]
    geom = np.array([f["geom"] for f in fp]) * 100

    def chan(f, key):
        return anim._beam_channel(dist, radial, f[key], max_total, f["chan_total"],
                                  leg_norms=None).astype(np.float32)

    def enc(f, key):
        return anim._channel_encircled(dist, radial, f[key], max_total,
                                       f["chan_total"]).astype(np.float32)

    contour_opts = dict(coloring="lines", showlabels=True, start=0.1, end=0.9, size=0.2,
                        labelfont=dict(size=8, color="#c8ffe6"))
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.09,
                        subplot_titles=("major-axis cut (widest silhouette axis)",
                                        "minor-axis cut (thinnest axis)"))

    def add_cut(f, key, sil_key, row):
        heat_i = len(fig.data)
        fig.add_trace(go.Heatmap(x=radial, y=dist, z=chan(f, key), colorscale="Plasma",
                                 zmin=0, zmax=1, zauto=False, zsmooth="fast", showscale=False,
                                 hoverinfo="skip"), row=row, col=1)
        cont_i = len(fig.data)
        fig.add_trace(go.Contour(x=radial, y=dist, z=enc(f, key), showscale=False,
                                 colorscale=[[0.0, "#7dffce"], [1.0, "#00f5a0"]],
                                 contours=contour_opts, line=dict(width=1.0), opacity=0.5,
                                 hoverinfo="skip"), row=row, col=1)
        fig.add_trace(go.Scatter(x=[-r_t0, r_t0], y=[0.0, 0.0], mode="lines",
                                 line=dict(color="white", width=5), showlegend=False,
                                 hoverinfo="skip"), row=row, col=1)                # emitter aperture
        plane_i = len(fig.data)
        s0 = f[sil_key]
        yy = f["chan_total"] / max_total
        fig.add_trace(go.Scatter(x=[-s0, s0], y=[yy, yy], mode="lines",
                                 line=dict(color="#00e5ff", width=4), showlegend=False,
                                 hoverinfo="skip"), row=row, col=1)                # plane slice
        return heat_i, cont_i, plane_i

    i_maj = add_cut(fp[0], "chan_segments_major", "sil_major_m", 1)
    i_min = add_cut(fp[0], "chan_segments_minor", "sil_minor_m", 2)
    animated = [i_maj[0], i_maj[1], i_maj[2], i_min[0], i_min[1], i_min[2]]

    frames = []
    for k, f in enumerate(fp):
        yy = f["chan_total"] / max_total
        frames.append(go.Frame(name=str(k), traces=animated, data=[
            go.Heatmap(z=chan(f, "chan_segments_major")),
            go.Contour(x=radial, y=dist, z=enc(f, "chan_segments_major")),
            go.Scatter(x=[-f["sil_major_m"], f["sil_major_m"]], y=[yy, yy]),
            go.Heatmap(z=chan(f, "chan_segments_minor")),
            go.Contour(x=radial, y=dist, z=enc(f, "chan_segments_minor")),
            go.Scatter(x=[-f["sil_minor_m"], f["sil_minor_m"]], y=[yy, yy]),
        ]))
    fig.frames = frames

    steps = [dict(method="animate", label=f"{xs[k]/1e3:.0f}",
                  args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))]) for k in range(n_frames)]
    # Slider at the TOP (in the top margin) beside the ▶ play button, so on a phone the
    # controls are reachable without scrolling past both panels.
    fig.update_layout(sliders=[dict(active=0, y=1.0, yanchor="bottom", x=0.14, len=0.80,
                                    currentvalue=dict(prefix="", suffix=" km"),
                                    steps=steps)])
    for row in (1, 2):
        fig.update_xaxes(title_text="radial offset (m)", range=[-R, R], row=row, col=1)
        fig.update_yaxes(title_text="emitter → plane", range=[0, 1], row=row, col=1)
    fig.update_layout(template="plotly_dark", height=1120, width=260,
                      margin=dict(l=44, r=12, t=104, b=44),
                      title=dict(text=f"Beam channel<br><sub>{arch_name}, {emitter_diam_m:.0f} m, "
                                      f"{lateral_offset_km:.0f} km pass · green = enclosed energy · "
                                      f"cyan = plane</sub>", y=0.995, yanchor="top", font=dict(size=12)))
    if write:
        import os
        os.makedirs(OUT, exist_ok=True)
        post = anim._PLAYER_JS.replace("{n_frames}", str(n_frames))
        fig.write_html(path, include_plotlyjs="inline", full_html=False, auto_play=False,
                       post_script=post)
    return fig


def _optimal_fixed_aspect(base_kw, xs, plane_alt_m, basis, mesh, lateral_m, delivered_mw,
                          aspects=(1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0)):
    """The single fixed emitter aspect that maximises pass-integrated delivered energy
    (Σ η at constant grid power) for THIS pass — the best one elliptical array to build."""
    best = (-np.inf, 1.0)
    for asp in aspects:
        kw = {**base_kw, "shaped_spot": True, "fixed_emitter_aspect": asp}
        s = sum(anim._frame_physics(kw, x, plane_alt_m, basis, mesh, lateral_offset_m=lateral_m,
                                    delivered_mw=delivered_mw)["eta"] for x in xs)
        if s > best[0]:
            best = (s, asp)
    return best[1]


def build_phone_comparison(arch_name="5.8 direct", weather="clear", lateral_offset_km=20.0,
                           emitter_diam_m=100.0, fixed_aspect=None, n_frames=25,
                           grid_dist=60, grid_radial=80, plane_alt_m=11_000.0,
                           min_elevation_deg=2.0, delivered_mw=10.0,
                           write=True, path=f"{OUT}/flyby_phone.html"):
    """Phone-friendly comparison: a ROUND emitter vs a FIXED elliptical emitter of the same
    aperture area, both mechanically pointed (no per-frame reshaping), over one lateral pass.
    Each case shows its two beam perspectives — the major-axis and minor-axis channel cuts —
    stacked in a single column, with an overlaid capture-efficiency curve at the bottom.

    The fixed aspect is chosen (if not given) to maximise pass-integrated delivered energy —
    the best single elliptical array to build for THIS pass geometry.

    Run:  uv run python beam_link_phone.py   → out/flyby_phone.html"""
    base_kw = dict(bl.standard_architectures(weather=weather)[arch_name])
    base_kw["emitter_diam_m"] = emitter_diam_m
    lateral_m = lateral_offset_km * 1e3
    basis = (geo.point(0.0, 0.0, 0.0), *geo.local_enu(geo.point(0.0, 0.0, 0.0)))
    cv, cf = m.make_bwb_mesh(n_span=41, n_theta=26)
    cn, _ = m.triangle_normals_areas(cv, cf)
    mesh = (cv, cf, cn)
    half = float(geo.downrange_for_elevation(min_elevation_deg, plane_alt_m))
    xs = np.linspace(-half, half, n_frames)

    if fixed_aspect is None:
        xs_coarse = np.linspace(-half, half, 13)
        fixed_aspect = _optimal_fixed_aspect(base_kw, xs_coarse, plane_alt_m, basis, mesh,
                                             lateral_m, delivered_mw)
    r = emitter_diam_m / 2.0
    a_maj, a_min = r * np.sqrt(fixed_aspect), r / np.sqrt(fixed_aspect)   # ellipse semi-axes

    circ_kw = {**base_kw, "shaped_spot": False}
    ellf_kw = {**base_kw, "shaped_spot": True, "fixed_emitter_aspect": fixed_aspect}
    adap_kw = {**base_kw, "shaped_spot": True}

    fr_c = [anim._frame_physics(circ_kw, x, plane_alt_m, basis, mesh, lateral_offset_m=lateral_m, delivered_mw=delivered_mw) for x in xs]
    fr_e = [anim._frame_physics(ellf_kw, x, plane_alt_m, basis, mesh, lateral_offset_m=lateral_m, delivered_mw=delivered_mw) for x in xs]
    cap_adap = np.array([anim._frame_physics(adap_kw, x, plane_alt_m, basis, mesh, lateral_offset_m=lateral_m, delivered_mw=delivered_mw)["eff"] for x in xs]) * 100.0

    all_fr = fr_c + fr_e
    max_total = max(f["chan_total"] for f in all_fr)
    R = 1.4 * max(max(f["chan_emit_radius_m"] for f in all_fr), max(f["sil_major_m"] for f in all_fr))
    dist = np.linspace(0.0, 1.0, grid_dist)
    radial = np.linspace(-R, R, grid_radial)
    r_t0 = fr_c[0]["chan_emit_radius_m"]
    cap_c = np.array([f["eff"] for f in fr_c]) * 100.0
    cap_e = np.array([f["eff"] for f in fr_e]) * 100.0
    x_km = xs / 1e3

    def chan(f, key):
        return anim._beam_channel(dist, radial, f[key], max_total, f["chan_total"], leg_norms=None).astype(np.float32)

    def enc(f, key):
        return anim._channel_encircled(dist, radial, f[key], max_total, f["chan_total"]).astype(np.float32)

    contour_opts = dict(coloring="lines", showlabels=True, start=0.1, end=0.9, size=0.2,
                        labelfont=dict(size=7, color="#c8ffe6"))
    fig = make_subplots(rows=5, cols=1, vertical_spacing=0.055,
                        row_heights=[0.21, 0.21, 0.21, 0.21, 0.16],
                        subplot_titles=("round · major-axis cut", "round · minor-axis cut",
                                        f"fixed ellipse ({fixed_aspect:.1f}:1) · major-axis cut",
                                        f"fixed ellipse ({fixed_aspect:.1f}:1) · minor-axis cut",
                                        "capture efficiency along the pass"))

    def add_cut(f, key, sil_key, sil_color, row):
        heat_i = len(fig.data)
        fig.add_trace(go.Heatmap(x=radial, y=dist, z=chan(f, key), colorscale="Plasma",
                                 zmin=0, zmax=1, zauto=False, zsmooth="fast", showscale=False,
                                 hoverinfo="skip"), row=row, col=1)
        cont_i = len(fig.data)
        fig.add_trace(go.Contour(x=radial, y=dist, z=enc(f, key), showscale=False,
                                 colorscale=[[0.0, "#7dffce"], [1.0, "#00f5a0"]],
                                 contours=contour_opts, line=dict(width=1.0), opacity=0.5,
                                 hoverinfo="skip"), row=row, col=1)
        fig.add_trace(go.Scatter(x=[-r_t0, r_t0], y=[0.0, 0.0], mode="lines",
                                 line=dict(color="white", width=5), showlegend=False,
                                 hoverinfo="skip"), row=row, col=1)                # emitter aperture
        plane_i = len(fig.data)
        s0 = f[sil_key]
        yy = f["chan_total"] / max_total
        fig.add_trace(go.Scatter(x=[-s0, s0], y=[yy, yy], mode="lines",
                                 line=dict(color=sil_color, width=4), showlegend=False,
                                 hoverinfo="skip"), row=row, col=1)                # plane slice
        return heat_i, cont_i, plane_i

    i_cmaj = add_cut(fr_c[0], "chan_segments_major", "sil_major_m", "#f2b134", 1)
    i_cmin = add_cut(fr_c[0], "chan_segments_minor", "sil_minor_m", "#f2b134", 2)
    i_emaj = add_cut(fr_e[0], "chan_segments_major", "sil_major_m", "#00e5ff", 3)
    i_emin = add_cut(fr_e[0], "chan_segments_minor", "sil_minor_m", "#00e5ff", 4)

    fig.add_trace(go.Scatter(x=x_km, y=cap_adap, mode="lines", name="adaptive ceiling",
                             line=dict(color="#888888", width=1.2, dash="dot"), hoverinfo="skip"), row=5, col=1)
    fig.add_trace(go.Scatter(x=x_km, y=cap_c, mode="lines", name="round",
                             line=dict(color="#f2b134", width=2.5), hoverinfo="skip"), row=5, col=1)
    fig.add_trace(go.Scatter(x=x_km, y=cap_e, mode="lines", name=f"fixed ellipse {fixed_aspect:.1f}:1",
                             line=dict(color="#00e5ff", width=2.5), hoverinfo="skip"), row=5, col=1)
    i_mc = len(fig.data)
    fig.add_trace(go.Scatter(x=[x_km[0]], y=[cap_c[0]], mode="markers",
                             marker=dict(size=8, color="#f2b134"), showlegend=False, hoverinfo="skip"), row=5, col=1)
    i_me = len(fig.data)
    fig.add_trace(go.Scatter(x=[x_km[0]], y=[cap_e[0]], mode="markers",
                             marker=dict(size=8, color="#00e5ff"), showlegend=False, hoverinfo="skip"), row=5, col=1)

    animated = [i_cmaj[0], i_cmaj[1], i_cmaj[2], i_cmin[0], i_cmin[1], i_cmin[2],
                i_emaj[0], i_emaj[1], i_emaj[2], i_emin[0], i_emin[1], i_emin[2], i_mc, i_me]
    frames = []
    for k in range(n_frames):
        fc, fe = fr_c[k], fr_e[k]
        yc, ye = fc["chan_total"] / max_total, fe["chan_total"] / max_total
        frames.append(go.Frame(name=str(k), traces=animated, data=[
            go.Heatmap(z=chan(fc, "chan_segments_major")),
            go.Contour(x=radial, y=dist, z=enc(fc, "chan_segments_major")),
            go.Scatter(x=[-fc["sil_major_m"], fc["sil_major_m"]], y=[yc, yc]),
            go.Heatmap(z=chan(fc, "chan_segments_minor")),
            go.Contour(x=radial, y=dist, z=enc(fc, "chan_segments_minor")),
            go.Scatter(x=[-fc["sil_minor_m"], fc["sil_minor_m"]], y=[yc, yc]),
            go.Heatmap(z=chan(fe, "chan_segments_major")),
            go.Contour(x=radial, y=dist, z=enc(fe, "chan_segments_major")),
            go.Scatter(x=[-fe["sil_major_m"], fe["sil_major_m"]], y=[ye, ye]),
            go.Heatmap(z=chan(fe, "chan_segments_minor")),
            go.Contour(x=radial, y=dist, z=enc(fe, "chan_segments_minor")),
            go.Scatter(x=[-fe["sil_minor_m"], fe["sil_minor_m"]], y=[ye, ye]),
            go.Scatter(x=[x_km[k]], y=[cap_c[k]]),
            go.Scatter(x=[x_km[k]], y=[cap_e[k]]),
        ]))
    fig.frames = frames

    steps = [dict(method="animate", label=f"{xs[k]/1e3:.0f}",
                  args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))]) for k in range(n_frames)]
    fig.update_layout(sliders=[dict(active=0, y=1.0, yanchor="bottom", x=0.14, len=0.78,
                                    currentvalue=dict(prefix="", suffix=" km"), steps=steps)])
    for row in (1, 2, 3, 4):
        fig.update_xaxes(title_text="radial (m)", range=[-R, R], row=row, col=1)
        fig.update_yaxes(title_text="emitter → plane", range=[0, 1], row=row, col=1)
    d_top = max(float(cap_c.max()), float(cap_e.max()), float(cap_adap.max())) * 1.12 + 1e-9
    fig.update_xaxes(title_text="along-track (km)", range=[x_km[0], x_km[-1]], row=5, col=1)
    fig.update_yaxes(title_text="capture (%)", range=[0, d_top], row=5, col=1)
    fig.update_layout(template="plotly_dark", height=1560, width=300,
                      margin=dict(l=46, r=12, t=120, b=44),
                      legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center", font=dict(size=8)),
                      title=dict(text=(f"Round vs fixed ellipse<br><sub>same area · {emitter_diam_m:.0f} m equiv "
                                       f"({2*a_maj:.0f}×{2*a_min:.0f} m ellipse) · {lateral_offset_km:.0f} km pass · "
                                       f"amber=round, cyan=ellipse plane slice</sub>"),
                                 y=0.992, yanchor="top", font=dict(size=12)))
    if write:
        import os
        os.makedirs(OUT, exist_ok=True)
        post = anim._PLAYER_JS.replace("{n_frames}", str(n_frames))
        fig.write_html(path, include_plotlyjs="inline", full_html=False, auto_play=False, post_script=post)
    return fig


if __name__ == "__main__":
    build_phone_comparison()
    print("wrote", f"{OUT}/flyby_phone.html")
