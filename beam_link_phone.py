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


def build_phone_focus(freq_ghz=5.8, emitter_diam_m=100.0, fixed_aspect=3.5,
                      L_min_km=11.0, L_max_km=213.0, n_frames=21, grid_dist=64,
                      grid_radial=64, R_m=230.0, write=True, path=f"{OUT}/focus_phone.html"):
    """Pure FOCUSING sweep (no plane, no atmosphere): a circular array vs an elliptical array of
    the same aperture area, focusing from as close as possible to as far as possible. The field
    is computed DIRECTLY from the exact Gaussian-beam model (bl.gaussian_beam_radius) — the
    picture IS the model, no cartoon. The vertical axis is the TRUE distance from the emitter
    (km), so the beam visibly converges to a diffraction-limited waist and diverges beyond it as
    the focus travels up-range — from overhead at cruise altitude (11 km) to the 2° horizon
    (213 km).

    Reads: the bright pinch is the true waist w0 = a·u/√(1+u²) at z_waist = L/(1+u²) ≤ L (u =
    L/zR, zR = πa²/λ); the coloured bar is the spot actually DELIVERED at the target range
    (w(L) = λL/πa). At long range the waist sits well before the target and the delivered spot
    has already diverged — a fixed aperture cannot concentrate the beam far away. The circle is
    identical in both cuts; the ellipse (same area) focuses TIGHTER in its long-aperture
    (minor-spot) axis and looser in the other. Four cuts {circle, ellipse} × {major, minor} plus
    the delivered-spot-vs-range curves. Slider = focus distance.

    Run:  uv run python beam_link_phone.py   → out/focus_phone.html"""
    lam = bl.napkin.SPEED_OF_LIGHT / (freq_ghz * 1e9)
    r = emitter_diam_m / 2.0
    A = float(fixed_aspect)
    ap_maj, ap_min = r / np.sqrt(A), r * np.sqrt(A)     # aperture semi-axis feeding major/minor cut
    Ls = np.linspace(L_min_km * 1e3, L_max_km * 1e3, n_frames)
    y_km = np.linspace(0.0, L_max_km, grid_dist)         # vertical axis = actual distance (km)
    radial = np.linspace(-R_m, R_m, grid_radial)

    z_m = y_km * 1e3                                     # along-beam distance grid (m)

    def field(ap, L):
        """Self-normalised beam intensity on the (z, radial) grid, computed DIRECTLY from the
        exact Gaussian model — w(z) from gaussian_beam_radius, intensity ∝ exp(−2r²/w(z)²)/w(z)²
        (power-conserving, so it brightens into the waist and dims through the divergence)."""
        w, _, _ = bl.gaussian_beam_radius(z_m, ap, lam, L)
        wz2 = (w ** 2)[:, None]
        inten = np.exp(-2.0 * radial[None, :] ** 2 / wz2) / (np.pi * wz2)
        mx = float(np.nanmax(inten))
        return (inten / mx if mx > 0 else inten).astype(np.float32)

    def encf(ap, L):
        """Enclosed-energy fraction within |radial| at each z: 1 − exp(−2r²/w(z)²)."""
        w, _, _ = bl.gaussian_beam_radius(z_m, ap, lam, L)
        wz2 = (w ** 2)[:, None]
        return (1.0 - np.exp(-2.0 * radial[None, :] ** 2 / wz2)).astype(np.float32)

    def spot_at(ap, L):
        """Delivered spot radius AT the target range L (= λL/πa)."""
        return float(bl.gaussian_beam_radius(np.array([float(L)]), ap, lam, L)[0][0])

    # per-case (title, aperture semi-axis, colour)
    cases = [("circular array · major-axis cut", r, "#f2b134"),
             ("circular array · minor-axis cut", r, "#f2b134"),
             (f"elliptical array ({A:.1f}:1) · major-axis cut", ap_maj, "#00e5ff"),
             (f"elliptical array ({A:.1f}:1) · minor-axis cut", ap_min, "#00e5ff")]

    contour_opts = dict(coloring="lines", showlabels=True, start=0.1, end=0.9, size=0.2,
                        labelfont=dict(size=7, color="#c8ffe6"))
    fig = make_subplots(rows=5, cols=1, vertical_spacing=0.05,
                        row_heights=[0.21, 0.21, 0.21, 0.21, 0.16],
                        subplot_titles=tuple(c[0] for c in cases) + ("focal-spot waist vs focal range",))

    L0 = Ls[0]
    cut_idx = []
    for row, (_, ap, color) in enumerate(cases, start=1):
        heat_i = len(fig.data)
        fig.add_trace(go.Heatmap(x=radial, y=y_km, z=field(ap, L0), colorscale="Plasma",
                                 zmin=0, zmax=1, zauto=False, zsmooth="fast", showscale=False,
                                 hoverinfo="skip"), row=row, col=1)
        cont_i = len(fig.data)
        fig.add_trace(go.Contour(x=radial, y=y_km, z=encf(ap, L0), showscale=False,
                                 colorscale=[[0.0, "#7dffce"], [1.0, "#00f5a0"]],
                                 contours=contour_opts, line=dict(width=1.0), opacity=0.5,
                                 hoverinfo="skip"), row=row, col=1)
        fig.add_trace(go.Scatter(x=[-ap, ap], y=[0.0, 0.0], mode="lines",
                                 line=dict(color="white", width=5), showlegend=False,
                                 hoverinfo="skip"), row=row, col=1)                 # aperture (fixed)
        bar_i = len(fig.data)
        s0 = spot_at(ap, L0)
        fig.add_trace(go.Scatter(x=[-s0, s0], y=[L0 / 1e3, L0 / 1e3], mode="lines",
                                 line=dict(color=color, width=4), showlegend=False,
                                 hoverinfo="skip"), row=row, col=1)                 # delivered spot at target
        cut_idx.append((heat_i, cont_i, bar_i, ap, color))

    # ---- delivered spot radius (at the target range) vs range (row 5) ----
    Lkm = Ls / 1e3
    sc = np.array([spot_at(r, L) for L in Ls])            # circle (both axes)
    smaj = np.array([spot_at(ap_maj, L) for L in Ls])     # ellipse major-spot axis (short aperture)
    smin = np.array([spot_at(ap_min, L) for L in Ls])     # ellipse minor-spot axis (long aperture)
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

    animated = [i for (h, c, b, _, _) in cut_idx for i in (h, c, b)] + [m_maj, m_cir, m_min]
    frames = []
    for k in range(n_frames):
        L = Ls[k]
        data = []
        for (h, c, b, ap, color) in cut_idx:
            s = spot_at(ap, L)
            data += [go.Heatmap(z=field(ap, L)),
                     go.Contour(x=radial, y=y_km, z=encf(ap, L)),
                     go.Scatter(x=[-s, s], y=[L / 1e3, L / 1e3])]
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
    fig.update_yaxes(title_text="delivered spot radius (m)", type="log", row=5, col=1)
    fig.update_layout(template="plotly_dark", height=1560, width=330,
                      paper_bgcolor="black", plot_bgcolor="black",
                      margin=dict(l=48, r=12, t=120, b=44),
                      legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center", font=dict(size=8)),
                      title=dict(text=(f"Focusing: circle vs ellipse<br><sub>exact Gaussian beam · same area · "
                                       f"{emitter_diam_m:.0f} m equiv ({2*ap_min:.0f}×{2*ap_maj:.0f} m ellipse) · "
                                       f"white=aperture · bar=spot delivered at the target · "
                                       f"bright pinch = true waist (before the target at long range)</sub>"),
                                 y=0.992, yanchor="top", font=dict(size=11)))
    if write:
        import os
        os.makedirs(OUT, exist_ok=True)
        post = anim._PLAYER_JS.replace("{n_frames}", str(n_frames))
        post += "\ndocument.body.style.background='#000';document.documentElement.style.background='#000';"
        fig.write_html(path, include_plotlyjs="inline", full_html=False, auto_play=False, post_script=post)
    return fig


if __name__ == "__main__":
    build_phone_comparison()
    print("wrote", f"{OUT}/flyby_phone.html")
    build_phone_focus()
    print("wrote", f"{OUT}/focus_phone.html")
