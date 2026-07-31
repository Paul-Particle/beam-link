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
import diffraction as dfr
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

    _cc = {}

    def _both(f, axis):
        k = (id(f), axis)
        if k not in _cc:
            _cc[k] = anim._channel_field(dist, radial, f["chan_legs"], max_total, f["chan_total"], axis)
        return _cc[k]

    def chan(f, axis):
        return _both(f, axis)[0]

    def enc(f, axis):
        return _both(f, axis)[1]

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

    i_maj = add_cut(fp[0], "y", "sil_major_m", 1)
    i_min = add_cut(fp[0], "x", "sil_minor_m", 2)
    animated = [i_maj[0], i_maj[1], i_maj[2], i_min[0], i_min[1], i_min[2]]

    frames = []
    for k, f in enumerate(fp):
        yy = f["chan_total"] / max_total
        frames.append(go.Frame(name=str(k), traces=animated, data=[
            go.Heatmap(z=chan(f, "y")),
            go.Contour(x=radial, y=dist, z=enc(f, "y")),
            go.Scatter(x=[-f["sil_major_m"], f["sil_major_m"]], y=[yy, yy]),
            go.Heatmap(z=chan(f, "x")),
            go.Contour(x=radial, y=dist, z=enc(f, "x")),
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

    _cc = {}

    def _both(f, axis):
        k = (id(f), axis)
        if k not in _cc:
            _cc[k] = anim._channel_field(dist, radial, f["chan_legs"], max_total, f["chan_total"], axis)
        return _cc[k]

    def chan(f, axis):
        return _both(f, axis)[0]

    def enc(f, axis):
        return _both(f, axis)[1]

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

    i_cmaj = add_cut(fr_c[0], "y", "sil_major_m", "#f2b134", 1)
    i_cmin = add_cut(fr_c[0], "x", "sil_minor_m", "#f2b134", 2)
    i_emaj = add_cut(fr_e[0], "y", "sil_major_m", "#00e5ff", 3)
    i_emin = add_cut(fr_e[0], "x", "sil_minor_m", "#00e5ff", 4)

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
            go.Heatmap(z=chan(fc, "y")),
            go.Contour(x=radial, y=dist, z=enc(fc, "y")),
            go.Scatter(x=[-fc["sil_major_m"], fc["sil_major_m"]], y=[yc, yc]),
            go.Heatmap(z=chan(fc, "x")),
            go.Contour(x=radial, y=dist, z=enc(fc, "x")),
            go.Scatter(x=[-fc["sil_minor_m"], fc["sil_minor_m"]], y=[yc, yc]),
            go.Heatmap(z=chan(fe, "y")),
            go.Contour(x=radial, y=dist, z=enc(fe, "y")),
            go.Scatter(x=[-fe["sil_major_m"], fe["sil_major_m"]], y=[ye, ye]),
            go.Heatmap(z=chan(fe, "x")),
            go.Contour(x=radial, y=dist, z=enc(fe, "x")),
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


def _cut_r50(radial_m, enclosed_row):
    """Radius containing 50 % of the cut's power, from an enclosed-energy-along-cut row."""
    r_abs = np.abs(radial_m)
    o = np.argsort(r_abs)
    return float(np.interp(0.5, enclosed_row[o], r_abs[o]))


def build_phone_focus(freq_ghz=5.8, emitter_diam_m=100.0, fixed_aspect=3.5, taper_p=0.0,
                      L_min_km=11.0, L_max_km=213.0, n_frames=17, grid_dist=44,
                      grid_radial=80, R_m=230.0, n_grid=384, write=True,
                      path=f"{OUT}/focus_phone.html"):
    """Pure FOCUSING sweep (no plane, no atmosphere): a circular array vs an elliptical array of
    the same aperture area, focusing from as close as possible to as far as possible. The field
    is computed DIRECTLY from exact scalar diffraction (diffraction.axial_cut) — the picture IS
    the model, no cartoon and no Gaussian idealisation: the main lobe carries <100 % of the power
    and is ringed by real Airy SIDELOBES. The vertical axis is the TRUE distance from the emitter
    (km), so the beam visibly converges to a diffraction-limited waist and diverges beyond it as
    the focus travels up-range — from overhead at cruise altitude (11 km) to the 2° horizon
    (213 km).

    Reads: the bright pinch is the diffraction waist; the sidelobe rings are the power that a
    finite aperture unavoidably throws outside the main lobe; the coloured bar is the delivered
    spot (50 % encircled radius) at the target range. At long range the waist sits well before
    the target and the delivered spot has already diverged — a fixed aperture cannot concentrate
    the beam far away. The circle is identical in both cuts; the ellipse (same area) focuses
    TIGHTER in its long-aperture (minor-spot) axis and looser in the other. Four cuts {circle,
    ellipse} × {major, minor} plus the delivered-spot-vs-range curves. `taper_p` is the aperture
    amplitude taper (0 = uniform → the sharpest sidelobes). Slider = focus distance.

    Run:  uv run python beam_link_phone.py   → out/focus_phone.html"""
    lam = bl.napkin.SPEED_OF_LIGHT / (freq_ghz * 1e9)
    r = emitter_diam_m / 2.0
    A = float(fixed_aspect)
    ap_long, ap_short = r * np.sqrt(A), r / np.sqrt(A)   # ellipse aperture semi-axes (long, short)
    Ls = np.linspace(L_min_km * 1e3, L_max_km * 1e3, n_frames)
    y_km = np.linspace(0.0, L_max_km, grid_dist)         # vertical axis = actual distance (km)
    radial = np.linspace(-R_m, R_m, grid_radial)
    z_m = y_km * 1e3

    # per-case: (title, aperture semi-major, semi-minor, cut axis, aperture half-width drawn, colour)
    #  circle: symmetric; ellipse major cut = WIDE spot (from the SHORT aperture axis, drawn as
    #  the y-cut of the elliptical field); ellipse minor cut = NARROW spot (from the LONG axis).
    cases = [("circular array · major-axis cut", r, r, "x", r, "#f2b134"),
             ("circular array · minor-axis cut", r, r, "x", r, "#f2b134"),
             (f"elliptical array ({A:.1f}:1) · major-axis cut", ap_long, ap_short, "y", ap_short, "#00e5ff"),
             (f"elliptical array ({A:.1f}:1) · minor-axis cut", ap_long, ap_short, "x", ap_long, "#00e5ff")]

    def cut(case, L):
        _, smaj, smin, axis, _, _ = case
        return dfr.axial_cut(z_m, L, smaj, smin, lam, radial, taper_p=taper_p, axis=axis, n_grid=n_grid)

    contour_opts = dict(coloring="lines", showlabels=True, start=0.1, end=0.9, size=0.2,
                        labelfont=dict(size=7, color="#c8ffe6"))
    fig = make_subplots(rows=5, cols=1, vertical_spacing=0.05,
                        row_heights=[0.21, 0.21, 0.21, 0.21, 0.16],
                        subplot_titles=tuple(c[0] for c in cases) + ("delivered spot (50% radius) vs focal range",))

    L0 = Ls[0]
    cut_idx = []
    for row, case in enumerate(cases, start=1):
        _, _, _, _, apline, color = case
        inten, enc = cut(case, L0)
        heat_i = len(fig.data)
        fig.add_trace(go.Heatmap(x=radial, y=y_km, z=inten, colorscale="Plasma",
                                 zmin=0, zmax=1, zauto=False, zsmooth="fast", showscale=False,
                                 hoverinfo="skip"), row=row, col=1)
        cont_i = len(fig.data)
        fig.add_trace(go.Contour(x=radial, y=y_km, z=enc, showscale=False,
                                 colorscale=[[0.0, "#7dffce"], [1.0, "#00f5a0"]],
                                 contours=contour_opts, line=dict(width=1.0), opacity=0.5,
                                 hoverinfo="skip"), row=row, col=1)
        fig.add_trace(go.Scatter(x=[-apline, apline], y=[0.0, 0.0], mode="lines",
                                 line=dict(color="white", width=5), showlegend=False,
                                 hoverinfo="skip"), row=row, col=1)                 # aperture (fixed)
        s0 = _cut_r50(radial, enc[np.argmin(np.abs(z_m - L0))])
        bar_i = len(fig.data)
        fig.add_trace(go.Scatter(x=[-s0, s0], y=[L0 / 1e3, L0 / 1e3], mode="lines",
                                 line=dict(color=color, width=4), showlegend=False,
                                 hoverinfo="skip"), row=row, col=1)                 # delivered spot at target
        cut_idx.append((heat_i, cont_i, bar_i))

    # ---- precompute every frame's cut images + delivered spot, so row 5 has the full curves ----
    frames_data = []          # per frame: list of (inten, enc) per case
    spot_at_target = []       # per frame: [r50 per case]
    for k in range(n_frames):
        per_case = [cut(case, Ls[k]) for case in cases]
        frames_data.append(per_case)
        zi = np.argmin(np.abs(z_m - Ls[k]))
        spot_at_target.append([_cut_r50(radial, enc[zi]) for (_, enc) in per_case])
    spot_at_target = np.array(spot_at_target)             # (n_frames, 4)
    Lkm = Ls / 1e3
    sc, smaj, smin = spot_at_target[:, 0], spot_at_target[:, 2], spot_at_target[:, 3]

    fig.add_trace(go.Scatter(x=[Lkm[0], Lkm[-1]], y=[r, r], mode="lines", name="aperture radius",
                             line=dict(color="#888888", width=1.2, dash="dot"), hoverinfo="skip"), row=5, col=1)
    fig.add_trace(go.Scatter(x=Lkm, y=smaj, mode="lines", name="ellipse major",
                             line=dict(color="#00e5ff", width=2, dash="dash"), hoverinfo="skip"), row=5, col=1)
    fig.add_trace(go.Scatter(x=Lkm, y=sc, mode="lines", name="circle",
                             line=dict(color="#f2b134", width=2.5), hoverinfo="skip"), row=5, col=1)
    fig.add_trace(go.Scatter(x=Lkm, y=smin, mode="lines", name="ellipse minor",
                             line=dict(color="#00e5ff", width=2.5), hoverinfo="skip"), row=5, col=1)
    m_maj = len(fig.data); fig.add_trace(go.Scatter(x=[Lkm[0]], y=[smaj[0]], mode="markers",
                             marker=dict(size=7, color="#00e5ff"), showlegend=False, hoverinfo="skip"), row=5, col=1)
    m_cir = len(fig.data); fig.add_trace(go.Scatter(x=[Lkm[0]], y=[sc[0]], mode="markers",
                             marker=dict(size=7, color="#f2b134"), showlegend=False, hoverinfo="skip"), row=5, col=1)
    m_min = len(fig.data); fig.add_trace(go.Scatter(x=[Lkm[0]], y=[smin[0]], mode="markers",
                             marker=dict(size=7, color="#00e5ff"), showlegend=False, hoverinfo="skip"), row=5, col=1)

    animated = [i for (h, c, b) in cut_idx for i in (h, c, b)] + [m_maj, m_cir, m_min]
    frames = []
    for k in range(n_frames):
        data = []
        for ci, case in enumerate(cases):
            inten, enc = frames_data[k][ci]
            s = spot_at_target[k][ci]
            data += [go.Heatmap(z=inten),
                     go.Contour(x=radial, y=y_km, z=enc),
                     go.Scatter(x=[-s, s], y=[Ls[k] / 1e3, Ls[k] / 1e3])]
        data += [go.Scatter(x=[Lkm[k]], y=[smaj[k]]),
                 go.Scatter(x=[Lkm[k]], y=[sc[k]]),
                 go.Scatter(x=[Lkm[k]], y=[smin[k]])]
        frames.append(go.Frame(name=str(k), traces=animated, data=data))
    fig.frames = frames

    steps = [dict(method="animate", label=f"{Ls[k]/1e3:.0f}",
                  args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))]) for k in range(n_frames)]
    fig.update_layout(sliders=[dict(active=0, y=1.0, yanchor="bottom", x=0.14, len=0.78,
                                    currentvalue=dict(prefix="focus at: ", suffix=" km"), steps=steps)])
    for row in (1, 2, 3, 4):
        fig.update_xaxes(title_text="radial (m)", range=[-R_m, R_m], row=row, col=1)
        fig.update_yaxes(title_text="distance from emitter (km)", range=[0, L_max_km * 1.02], row=row, col=1)
    fig.update_xaxes(title_text="focus distance (km)", type="log", row=5, col=1)
    fig.update_yaxes(title_text="delivered spot 50% radius (m)", type="log", row=5, col=1)
    taper_txt = "uniform aperture (Airy)" if taper_p == 0 else f"taper (1−ρ²)^{taper_p:g}"
    fig.update_layout(template="plotly_dark", height=1560, width=330,
                      paper_bgcolor="black", plot_bgcolor="black",
                      margin=dict(l=48, r=12, t=120, b=44),
                      legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center", font=dict(size=8)),
                      title=dict(text=(f"Focusing: circle vs ellipse<br><sub>exact scalar diffraction · {taper_txt} · "
                                       f"same area · {emitter_diam_m:.0f} m equiv ({2*ap_long:.0f}×{2*ap_short:.0f} m ellipse) · "
                                       f"white=aperture · bar=delivered 50% spot · "
                                       f"bright pinch = diffraction waist · rings = real sidelobes</sub>"),
                                 y=0.992, yanchor="top", font=dict(size=11)))
    if write:
        import os
        os.makedirs(OUT, exist_ok=True)
        post = anim._PLAYER_JS.replace("{n_frames}", str(n_frames))
        post += "\ndocument.body.style.background='#000';document.documentElement.style.background='#000';"
        fig.write_html(path, include_plotlyjs="inline", full_html=False, auto_play=False, post_script=post)
    return fig


if __name__ == "__main__":
    # Build the focusing sweep (circle vs ellipse) → out/focus_phone.html. The two flyby
    # channel views (build_phone_channels / build_phone_comparison) both write out/flyby_phone.html,
    # so call whichever you want directly rather than building both here.
    build_phone_focus()
    print("wrote", f"{OUT}/focus_phone.html")
