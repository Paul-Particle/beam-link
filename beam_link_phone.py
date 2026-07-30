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
    fig.update_layout(sliders=[dict(active=0, y=-0.03, x=0.08, len=0.86,
                                    currentvalue=dict(prefix="along-track: ", suffix=" km"),
                                    steps=steps)])
    for row in (1, 2):
        fig.update_xaxes(title_text="radial offset (m)", range=[-R, R], row=row, col=1)
        fig.update_yaxes(title_text="emitter → plane", range=[0, 1], row=row, col=1)
    fig.update_layout(template="plotly_dark", height=1120, width=520,
                      margin=dict(l=52, r=16, t=54, b=48),
                      title=dict(text=f"Beam channel — {arch_name}, {emitter_diam_m:.0f} m emitter, "
                                      f"{weather}, {lateral_offset_km:.0f} km pass<br>"
                                      "<sub>plane height = slant/max-slant · green = enclosed-energy "
                                      "fraction · cyan = plane surface</sub>", font=dict(size=13)))
    if write:
        import os
        os.makedirs(OUT, exist_ok=True)
        post = anim._PLAYER_JS.replace("{n_frames}", str(n_frames))
        fig.write_html(path, include_plotlyjs="inline", full_html=False, auto_play=False,
                       post_script=post)
    return fig


if __name__ == "__main__":
    build_phone_channels()
    print("wrote", f"{OUT}/flyby_phone.html")
