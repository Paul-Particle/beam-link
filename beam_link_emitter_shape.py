"""Emitter-shape comparison: a ROUND 100 m emitter vs a SHAPE-MATCHED (elliptical) emitter
of the SAME aperture area, whose aspect matches the plane silhouette.

Why this figure exists — the honest version of the "shaped spot" question. `overlap_capture`
forms the elliptical spot by fixed-etendue reshaping: w_minor = w0/√aspect, i.e. TIGHTER in
the short axis than the circular diffraction limit w0. A round aperture physically CANNOT do
that — a 100 m circular array focuses no tighter than w0 in any direction. The only way to get
that tight minor axis is a genuinely ELONGATED aperture (wide along the axis where the spot
must be narrow, at the same total area). So "shaped_spot" has really been assuming an elliptical
emitter all along. This figure makes the two hardware choices explicit and compares them:

  • round emitter    — shaped_spot=False → round spot (what a 100 m dish actually does)
  • shaped emitter   — shaped_spot=True  → elliptical spot, same aperture area, aspect = plane

Four beam-channel cuts {round, elliptical} × {major-axis, minor-axis} plus one overlaid
capture-efficiency curve (a line per case) along the pass. Reuses the engine + frame physics
from beam_link_animation so it stays in lock-step with the main flyby figure.

Run:  uv run python beam_link_emitter_shape.py   → out/emitter_shape.html
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import beam_link as bl
import bwb_model as m
import geo
import beam_link_animation as anim

OUT = "out"

ROUND_C = "#f2b134"      # round-emitter accent (amber)
ELLIP_C = "#00e5ff"      # shaped-emitter accent (cyan)


def build_emitter_shape_comparison(arch_name="5.8 direct", weather="clear", lateral_offset_km=5.0,
                                   emitter_diam_m=100.0, n_frames=25, grid_dist=70, grid_radial=90,
                                   plane_alt_m=11_000.0, min_elevation_deg=2.0, delivered_mw=10.0,
                                   write=True, path=f"{OUT}/emitter_shape.html"):
    base_kw = dict(bl.standard_architectures(weather=weather)[arch_name])
    base_kw["emitter_diam_m"] = emitter_diam_m
    round_kw = {**base_kw, "shaped_spot": False}      # round aperture → round spot
    ellip_kw = {**base_kw, "shaped_spot": True}       # same area, elliptical → shaped spot
    lateral_m = lateral_offset_km * 1e3

    basis = (geo.point(0.0, 0.0, 0.0), *geo.local_enu(geo.point(0.0, 0.0, 0.0)))
    cv, cf = m.make_bwb_mesh(n_span=41, n_theta=26)
    cn, _ = m.triangle_normals_areas(cv, cf)
    mesh = (cv, cf, cn)
    half = float(geo.downrange_for_elevation(min_elevation_deg, plane_alt_m))
    xs = np.linspace(-half, half, n_frames)

    fr_round = [anim._frame_physics(round_kw, x, plane_alt_m, basis, mesh,
                                    lateral_offset_m=lateral_m, delivered_mw=delivered_mw) for x in xs]
    fr_ellip = [anim._frame_physics(ellip_kw, x, plane_alt_m, basis, mesh,
                                    lateral_offset_m=lateral_m, delivered_mw=delivered_mw) for x in xs]

    all_fr = fr_round + fr_ellip
    max_total = max(f["chan_total"] for f in all_fr)
    R = 1.4 * max(max(f["chan_emit_radius_m"] for f in all_fr),
                  max(f["sil_major_m"] for f in all_fr))
    dist = np.linspace(0.0, 1.0, grid_dist)
    radial = np.linspace(-R, R, grid_radial)
    r_t0 = fr_round[0]["chan_emit_radius_m"]

    # Capture efficiency (incidence-weighted overlap = what feeds η); geometry/silhouette are
    # identical between the two cases, so the gap is a PURE beam-shape effect.
    cap_round = np.array([f["eff"] for f in fr_round]) * 100.0
    cap_ellip = np.array([f["eff"] for f in fr_ellip]) * 100.0
    x_km = xs / 1e3

    _cc = {}

    def _all(f):
        k = id(f)
        if k not in _cc:
            _cc[k] = anim._channel_field(dist, radial, f["chan_legs"], max_total, f["chan_total"])
        return _cc[k]

    def chan(f, axis):
        ix, ex, iy, ey = _all(f)
        return iy if axis == "y" else ix

    def enc(f, axis):
        ix, ex, iy, ey = _all(f)
        return ey if axis == "y" else ex

    contour_opts = dict(coloring="lines", showlabels=True, start=0.1, end=0.9, size=0.2,
                        labelfont=dict(size=8, color="#c8ffe6"))
    fig = make_subplots(
        rows=3, cols=2, vertical_spacing=0.085, horizontal_spacing=0.09,
        row_heights=[0.35, 0.35, 0.30],
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy", "colspan": 2}, None]],
        subplot_titles=("round emitter · major-axis cut", "round emitter · minor-axis cut",
                        "shaped (elliptical) emitter · major-axis cut",
                        "shaped (elliptical) emitter · minor-axis cut",
                        "capture efficiency along the pass"))

    def add_cut(f, key, sil_key, sil_color, row, col):
        heat_i = len(fig.data)
        fig.add_trace(go.Heatmap(x=radial, y=dist, z=chan(f, key), colorscale="Plasma",
                                 zmin=0, zmax=1, zauto=False, zsmooth="fast", showscale=False,
                                 hoverinfo="skip"), row=row, col=col)
        cont_i = len(fig.data)
        fig.add_trace(go.Contour(x=radial, y=dist, z=enc(f, key), showscale=False,
                                 colorscale=[[0.0, "#7dffce"], [1.0, "#00f5a0"]],
                                 contours=contour_opts, line=dict(width=1.0), opacity=0.5,
                                 hoverinfo="skip"), row=row, col=col)
        fig.add_trace(go.Scatter(x=[-r_t0, r_t0], y=[0.0, 0.0], mode="lines",
                                 line=dict(color="white", width=5), showlegend=False,
                                 hoverinfo="skip"), row=row, col=col)                # emitter aperture
        plane_i = len(fig.data)
        s0 = f[sil_key]
        yy = f["chan_total"] / max_total
        fig.add_trace(go.Scatter(x=[-s0, s0], y=[yy, yy], mode="lines",
                                 line=dict(color=sil_color, width=4), showlegend=False,
                                 hoverinfo="skip"), row=row, col=col)                # plane slice
        return heat_i, cont_i, plane_i

    i_rmaj = add_cut(fr_round[0], "y", "sil_major_m", ROUND_C, 1, 1)
    i_rmin = add_cut(fr_round[0], "x", "sil_minor_m", ROUND_C, 1, 2)
    i_emaj = add_cut(fr_ellip[0], "y", "sil_major_m", ELLIP_C, 2, 1)
    i_emin = add_cut(fr_ellip[0], "x", "sil_minor_m", ELLIP_C, 2, 2)

    # ---- capture curves (row 3) : a line per case + a moving marker per case -------------
    fig.add_trace(go.Scatter(x=x_km, y=cap_round, mode="lines", name="round emitter",
                             line=dict(color=ROUND_C, width=2.5), hoverinfo="skip"), row=3, col=1)
    fig.add_trace(go.Scatter(x=x_km, y=cap_ellip, mode="lines", name="shaped (elliptical) emitter",
                             line=dict(color=ELLIP_C, width=2.5), hoverinfo="skip"), row=3, col=1)
    i_mround = len(fig.data)
    fig.add_trace(go.Scatter(x=[x_km[0]], y=[cap_round[0]], mode="markers",
                             marker=dict(size=10, color=ROUND_C), showlegend=False,
                             hoverinfo="skip"), row=3, col=1)
    i_mellip = len(fig.data)
    fig.add_trace(go.Scatter(x=[x_km[0]], y=[cap_ellip[0]], mode="markers",
                             marker=dict(size=10, color=ELLIP_C), showlegend=False,
                             hoverinfo="skip"), row=3, col=1)

    animated = [i_rmaj[0], i_rmaj[1], i_rmaj[2], i_rmin[0], i_rmin[1], i_rmin[2],
                i_emaj[0], i_emaj[1], i_emaj[2], i_emin[0], i_emin[1], i_emin[2],
                i_mround, i_mellip]
    frames = []
    for k in range(n_frames):
        fr, fe = fr_round[k], fr_ellip[k]
        yr, ye = fr["chan_total"] / max_total, fe["chan_total"] / max_total
        frames.append(go.Frame(name=str(k), traces=animated, data=[
            go.Heatmap(z=chan(fr, "y")),
            go.Contour(x=radial, y=dist, z=enc(fr, "y")),
            go.Scatter(x=[-fr["sil_major_m"], fr["sil_major_m"]], y=[yr, yr]),
            go.Heatmap(z=chan(fr, "x")),
            go.Contour(x=radial, y=dist, z=enc(fr, "x")),
            go.Scatter(x=[-fr["sil_minor_m"], fr["sil_minor_m"]], y=[yr, yr]),
            go.Heatmap(z=chan(fe, "y")),
            go.Contour(x=radial, y=dist, z=enc(fe, "y")),
            go.Scatter(x=[-fe["sil_major_m"], fe["sil_major_m"]], y=[ye, ye]),
            go.Heatmap(z=chan(fe, "x")),
            go.Contour(x=radial, y=dist, z=enc(fe, "x")),
            go.Scatter(x=[-fe["sil_minor_m"], fe["sil_minor_m"]], y=[ye, ye]),
            go.Scatter(x=[x_km[k]], y=[cap_round[k]]),
            go.Scatter(x=[x_km[k]], y=[cap_ellip[k]]),
        ]))
    fig.frames = frames

    steps = [dict(method="animate", label=f"{xs[k]/1e3:.0f}",
                  args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))]) for k in range(n_frames)]
    fig.update_layout(sliders=[dict(active=0, y=1.0, yanchor="bottom", x=0.13, len=0.80,
                                    currentvalue=dict(prefix="along-track: ", suffix=" km"),
                                    steps=steps)])

    for row in (1, 2):
        for col in (1, 2):
            fig.update_xaxes(title_text="radial offset (m)", range=[-R, R], row=row, col=col)
            fig.update_yaxes(title_text="emitter → plane", range=[0, 1], row=row, col=col)
    d_top = max(float(cap_round.max()), float(cap_ellip.max())) * 1.12 + 1e-9
    fig.update_xaxes(title_text="along-track (km)", range=[x_km[0], x_km[-1]], row=3, col=1)
    fig.update_yaxes(title_text="capture efficiency (%)", range=[0, d_top], row=3, col=1)

    fig.update_layout(
        template="plotly_dark", height=1180, width=940, margin=dict(t=140, l=60, r=30, b=55),
        title=dict(text=(f"Emitter shape — round vs shape-matched (same aperture area)<br>"
                         f"<sub>{arch_name}, {emitter_diam_m:.0f} m equivalent, {weather}, "
                         f"{lateral_offset_km:.0f} km pass · white = emitter aperture · "
                         f"amber/cyan = plane slice · green = enclosed-energy rings · "
                         f"the shaped spot needs an ELONGATED aperture (extent ∝ √aspect per axis)</sub>"),
                   y=0.985, yanchor="top", font=dict(size=15)),
        legend=dict(orientation="h", y=1.04, x=0.5, xanchor="center", font=dict(size=11)))

    if write:
        import os
        os.makedirs(OUT, exist_ok=True)
        post = anim._PLAYER_JS.replace("{n_frames}", str(n_frames))
        fig.write_html(path, include_plotlyjs="inline", full_html=False, auto_play=False,
                       post_script=post)
    return fig


if __name__ == "__main__":
    build_emitter_shape_comparison()
    print("wrote", f"{OUT}/emitter_shape.html")
