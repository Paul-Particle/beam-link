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


def _beam_on_grid(field, w0, shaped, gx, gy, profile_order=1.0):
    """The beam intensity the overlap used, evaluated on a plotting grid — round, or
    the shaped ellipse aligned to the silhouette's principal axes (matching
    overlap_capture so the picture agrees with the capture number). `profile_order` is
    the super-Gaussian order (1 = Gaussian, >1 = flat top)."""
    n = float(profile_order)
    uu, vv = np.meshgrid(gx, gy)
    if not shaped or field["u"].size < 3:
        return np.exp(-2.0 * ((uu ** 2 + vv ** 2) / w0 ** 2) ** n)
    coords = np.vstack([field["u"], field["v"]])
    evals, evecs = np.linalg.eigh(np.cov(coords))
    aspect = np.clip(np.sqrt(max(evals[1], 1e-12) / max(evals[0], 1e-12)), 1.0, 8.0)
    w_major, w_minor = w0 * np.sqrt(aspect), w0 / np.sqrt(aspect)
    flat = np.vstack([uu.ravel(), vv.ravel()])
    proj = evecs.T @ flat                       # rows: [along minor, along major]
    beam = np.exp(-2.0 * (proj[1] ** 2 / w_major ** 2 + proj[0] ** 2 / w_minor ** 2) ** n)
    return beam.reshape(uu.shape)


def _frame_physics(arch_kwargs, x_along_m, plane_alt_m, basis, coarse_mesh,
                   lateral_offset_m=0.0, delivered_mw=10.0):
    """Everything one frame needs, for one plane position of one architecture."""
    plane = (x_along_m, lateral_offset_m, plane_alt_m)
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
    # The engine is the single source of truth: it chooses the spot's SIZE (waist), SHAPE
    # (elliptical, in overlap_capture) and PROFILE (Gaussian/flat-top, adaptively) and
    # enforces the density cap. Read the winning waist + order back and re-form the SAME
    # spot here (with pointing) purely to draw panel A — so the picture matches panel D's η.
    s = bl.evaluate(cfg, delivered_mw)
    eta = float(s["eta_end_to_end"])
    w0 = float(s["spot_waist_m"])
    order = float(s["beam_profile_order"])
    _, eff, _, field = bl.overlap_capture(raster, diam / 2.0, lam, slant, shaped=cfg.shaped_spot,
                                          min_waist_m=w0, pointing_offset_m=cfg.pointing_error_m,
                                          incidence_floor=cfg.incidence_floor,
                                          profile_order=order, return_field=True)

    verts, faces, normals = coarse_mesh
    face_incidence = np.clip(normals @ look, 0.0, 1.0)

    plane_loc = _localize(x_along_m, lateral_offset_m, plane_alt_m, basis)
    emitter_loc = _localize(emitter[0], emitter[1], emitter[2], basis)
    return dict(field=field, w0=w0, shaped=cfg.shaped_spot, profile_order=order,
                eff=eff, eta=eta, elevation=float(elevation), slant_km=slant / 1e3, beta=float(beta),
                area=raster["area"], look=look, face_incidence=face_incidence,
                plane_loc=plane_loc, emitter_loc=emitter_loc)


# --- the animation -----------------------------------------------------------

def _size_emitter_far(kw, ref_elev_deg, plane_alt_m):
    """Size a direct/tether emitter to couple well at a FAR (low-elevation) reference
    geometry, then keep it fixed — so it reaches the plane out near the horizon and
    merely defocuses overhead (the opposite of over-focusing to a tiny overhead spot).
    Relay apertures are left as-is (fixed hardware; defocus handles the near field)."""
    kw = dict(kw)
    if kw.get("relay_pos") is not None:
        return kw
    dr = float(geo.downrange_for_elevation(ref_elev_deg, plane_alt_m))
    base = {k: v for k, v in kw.items() if k != "emitter_diam_m"}
    cfg = bl.LinkConfig(plane_pos=(dr, 0.0, plane_alt_m), emitter_diam_m=None, **base)
    kw["emitter_diam_m"] = float(bl.evaluate(cfg)["emitter_diam_m"])
    return kw


def build_flyby_animation(arch_name="relay ×3 mult 5.8→17.4", weather="light_rain",
                          plane_alt_m=11_000.0, lateral_offset_km=0.0, min_elevation_deg=2.0,
                          n_frames=81, size_ref_elev_deg=8.0, delivered_mw=10.0, grid_n=180,
                          path=f"{OUT}/beam_link_flyby_animation.html"):
    archs = bl.standard_architectures(weather=weather)
    if arch_name not in archs:
        raise KeyError(f"{arch_name!r} not in {list(archs)}")
    kw = _size_emitter_far(archs[arch_name], size_ref_elev_deg, plane_alt_m)
    lateral_m = lateral_offset_km * 1e3

    basis = (geo.point(0.0, 0.0, 0.0), *geo.local_enu(geo.point(0.0, 0.0, 0.0)))
    coarse_v, coarse_f = m.make_bwb_mesh(n_span=41, n_theta=26)
    coarse_n, _ = m.triangle_normals_areas(coarse_v, coarse_f)
    coarse_mesh = (coarse_v, coarse_f, coarse_n)

    # Horizon-to-horizon, sampled uniformly in ALONG-TRACK distance = uniformly in TIME
    # at constant speed, so playback is smooth and physically paced (independent of the
    # sampling). The plane's closest approach is at x=0 (offset laterally for a side pass).
    half = float(geo.downrange_for_elevation(min_elevation_deg, plane_alt_m))
    xs = np.linspace(-half, half, n_frames)
    frames_phys = [_frame_physics(kw, x, plane_alt_m, basis, coarse_mesh,
                                  lateral_offset_m=lateral_m, delivered_mw=delivered_mw) for x in xs]

    def frame_extent(f):
        """Half-width of panel A for THIS frame: enough to show the beam's 0.2-of-peak
        contour AND the whole silhouette. PER-FRAME (not a shared max) — otherwise the
        far-out frames' huge defocused waist sets a giant fixed window and every overhead
        frame collapses to an invisible central dot (the 'panel A empty' bug)."""
        sil = (np.abs(np.concatenate([f["field"]["u"], f["field"]["v"]])).max()
               if f["field"]["u"].size else 0.0)
        return max(1.3 * f["w0"], sil) * 1.15

    def panel_a_grids(f):
        """Silhouette incidence (NaN outside) + the beam intensity, on a per-frame grid
        sized to this frame's extent; returns the grid axes so the frame can rescale."""
        ext = frame_extent(f)
        gx = np.linspace(-ext, ext, grid_n)
        gy = np.linspace(-ext, ext, grid_n)
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
        beam_grid = _beam_on_grid(field, f["w0"], f["shaped"], gx, gy,
                                  profile_order=f["profile_order"])
        return inc_grid, beam_grid, gx, gy, ext

    # Static 3-D geometry (ground/cruise/stratosphere arcs; the emitter + its base).
    half = float(xs.max())
    arc_dr = np.linspace(-half, half, 90)
    ground = np.array([_localize(d, 0, 0, basis) for d in arc_dr]) / 1e3
    cruise = np.array([_localize(d, lateral_m, plane_alt_m, basis) for d in arc_dr]) / 1e3
    strato = np.array([_localize(d, 0, 20_000.0, basis) for d in arc_dr]) / 1e3
    rain_top_m = bl.WEATHER[weather]["rain_top_m"]
    rain = (np.array([_localize(d, 0, rain_top_m, basis) for d in arc_dr]) / 1e3
            if rain_top_m > 0 else None)

    has_relay = kw.get("relay_pos") is not None
    emit_dr, emit_cr, emit_alt = kw["relay_pos"] if has_relay else kw["station_pos"]
    base_loc = _localize(emit_dr, emit_cr, 0.0, basis) / 1e3
    emitter_loc = _localize(emit_dr, emit_cr, emit_alt, basis) / 1e3
    lofted = emit_alt > 1.0
    emit_name = "relay" if has_relay else ("tethered emitter" if lofted else "ground emitter")

    eta_series = np.array([f["eta"] for f in frames_phys]) * 100
    BOX = 45.0                                        # fixed half-extent of panel C (m)

    fig = make_subplots(
        rows=2, cols=2, horizontal_spacing=0.08, vertical_spacing=0.11,
        specs=[[{"type": "xy"}, {"type": "scene"}], [{"type": "scene"}, {"type": "xy"}]],
        subplot_titles=("A · beam spot over silhouette (fill = incidence)",
                        "B · geometry on a spherical earth (vertical exaggerated)",
                        "C · BWB facets by beam incidence",
                        "D · end-to-end η along the pass"))

    inc0, beam0, gx0, gy0, ext0 = panel_a_grids(frames_phys[0])
    contour_opts = dict(coloring="lines", showlabels=True, start=0.2, end=0.8, size=0.2,
                        labelfont=dict(size=9, color="#b30000"))
    # 0: silhouette incidence heatmap  1: beam contour (labelled = fraction of peak)
    fig.add_trace(go.Heatmap(x=gx0, y=gy0, z=inc0, colorscale="Viridis", zmin=0, zmax=1,
                             colorbar=dict(title="incidence", len=0.40, y=0.80, x=0.445),
                             hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Contour(x=gx0, y=gy0, z=beam0, showscale=False, colorscale="Reds",
                             contours=contour_opts, line=dict(width=1.4), hoverinfo="skip"),
                  row=1, col=1)

    # 2-4: arcs   5: rain-top layer   6: base station   7: tether/feed   8: emitter
    fig.add_trace(go.Scatter3d(x=ground[:, 0], y=ground[:, 1], z=ground[:, 2], mode="lines",
                               line=dict(color="#5b4636", width=4), name="ground", hoverinfo="skip"), row=1, col=2)
    fig.add_trace(go.Scatter3d(x=cruise[:, 0], y=cruise[:, 1], z=cruise[:, 2], mode="lines",
                               line=dict(color="#9fb6cd", width=1, dash="dot"), name="11 km cruise", hoverinfo="skip"), row=1, col=2)
    fig.add_trace(go.Scatter3d(x=strato[:, 0], y=strato[:, 1], z=strato[:, 2], mode="lines",
                               line=dict(color="#c9c9ff", width=1, dash="dot"), name="20 km", hoverinfo="skip"), row=1, col=2)
    if rain is not None:
        fig.add_trace(go.Scatter3d(x=rain[:, 0], y=rain[:, 1], z=rain[:, 2], mode="lines",
                                   line=dict(color="#4a90d9", width=6), opacity=0.4,
                                   name=f"rain top ({weather})", hoverinfo="skip"), row=1, col=2)
    else:
        fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="lines",
                                   showlegend=False, hoverinfo="skip"), row=1, col=2)
    fig.add_trace(go.Scatter3d(x=[base_loc[0]], y=[base_loc[1]], z=[base_loc[2]], mode="markers",
                               marker=dict(size=5, color="#2ca02c", symbol="square"),
                               name="base station", hoverinfo="skip"), row=1, col=2)
    tether_x = [base_loc[0], emitter_loc[0]] if lofted else [None]
    tether_y = [base_loc[1], emitter_loc[1]] if lofted else [None]
    tether_z = [base_loc[2], emitter_loc[2]] if lofted else [None]
    fig.add_trace(go.Scatter3d(x=tether_x, y=tether_y, z=tether_z, mode="lines",
                               line=dict(color="#7f7f7f", width=3, dash="dot"),
                               name="tether / feed", showlegend=lofted, hoverinfo="skip"), row=1, col=2)
    fig.add_trace(go.Scatter3d(
        x=[emitter_loc[0]] if lofted else [None], y=[emitter_loc[1]] if lofted else [None],
        z=[emitter_loc[2]] if lofted else [None], mode="markers",
        marker=dict(size=6, color="#ff7f0e", symbol="diamond"),
        name=emit_name, showlegend=lofted, hoverinfo="skip"), row=1, col=2)

    # 9: plane marker (anim)  10: beam line (anim)
    p0 = frames_phys[0]["plane_loc"] / 1e3
    fig.add_trace(go.Scatter3d(x=[p0[0]], y=[p0[1]], z=[p0[2]], mode="markers",
                               marker=dict(size=5, color="#d62728"), name="plane", hoverinfo="skip"), row=1, col=2)
    fig.add_trace(go.Scatter3d(x=[emitter_loc[0], p0[0]], y=[emitter_loc[1], p0[1]],
                               z=[emitter_loc[2], p0[2]], mode="lines",
                               line=dict(color="#d62728", width=3), name="beam", hoverinfo="skip"), row=1, col=2)

    # 11: BWB mesh by incidence (anim)  12: beam-arrival arrow (anim, fixed length)
    fig.add_trace(go.Mesh3d(x=coarse_v[:, 0], y=coarse_v[:, 1], z=coarse_v[:, 2],
                            i=coarse_f[:, 0], j=coarse_f[:, 1], k=coarse_f[:, 2],
                            intensity=frames_phys[0]["face_incidence"], intensitymode="cell",
                            colorscale="Viridis", cmin=0, cmax=1,
                            colorbar=dict(title="incidence", len=0.40, y=0.20, x=0.445),
                            flatshading=True, name="BWB", hoverinfo="skip"), row=2, col=1)
    look0 = frames_phys[0]["look"]
    fig.add_trace(go.Scatter3d(x=[look0[0] * 40, look0[0] * 15], y=[look0[1] * 40, look0[1] * 15],
                               z=[look0[2] * 40, look0[2] * 15], mode="lines",
                               line=dict(color="#d62728", width=6), name="beam→",
                               showlegend=False, hoverinfo="skip"), row=2, col=1)

    # 13: η profile (static)  14: current marker (anim)
    fig.add_trace(go.Scatter(x=xs / 1e3, y=eta_series, mode="lines",
                             line=dict(color="#1f77b4", width=2), showlegend=False, hoverinfo="skip"), row=2, col=2)
    fig.add_trace(go.Scatter(x=[xs[0] / 1e3], y=[eta_series[0]], mode="markers",
                             marker=dict(size=10, color="#d62728"), showlegend=False, hoverinfo="skip"), row=2, col=2)

    ANIMATED = [0, 1, 9, 10, 11, 12, 14]
    frames = []
    for k, f in enumerate(frames_phys):
        inc_g, beam_g, gxk, gyk, extk = panel_a_grids(f)
        p = f["plane_loc"] / 1e3
        look = f["look"]
        # Rescale panel A to this frame's extent (the beam waist shrinks ~40× overhead vs
        # near the horizon) so the spot/silhouette stay legible the whole pass.
        frames.append(go.Frame(name=str(k), traces=ANIMATED,
            layout=go.Layout(xaxis=dict(range=[-extk, extk]), yaxis=dict(range=[-extk, extk])),
            data=[
                go.Heatmap(x=gxk, y=gyk, z=inc_g),
                go.Contour(x=gxk, y=gyk, z=beam_g),
                go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]]),
                go.Scatter3d(x=[emitter_loc[0], p[0]], y=[emitter_loc[1], p[1]], z=[emitter_loc[2], p[2]]),
                go.Mesh3d(intensity=f["face_incidence"], intensitymode="cell"),
                go.Scatter3d(x=[look[0] * 40, look[0] * 15], y=[look[1] * 40, look[1] * 15],
                             z=[look[2] * 40, look[2] * 15]),
                go.Scatter(x=[xs[k] / 1e3], y=[eta_series[k]]),
            ]))
    fig.frames = frames

    # Uniform frame spacing = uniform time at constant speed; a short transition tweens
    # the marker/beam positions so playback is smooth, not steppy.
    dt_ms = 70
    steps = [dict(method="animate", label=f"{xs[k]/1e3:.0f}",
                  args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))]) for k in range(n_frames)]
    fig.update_layout(
        sliders=[dict(active=0, y=-0.04, x=0.05, len=0.9,
                      currentvalue=dict(prefix="along-track: ", suffix=" km"), steps=steps)],
        updatemenus=[dict(type="buttons", showactive=False, x=0.05, y=0.02, xanchor="right",
                          buttons=[dict(label="▶ play", method="animate",
                                        args=[None, dict(frame=dict(duration=dt_ms, redraw=True),
                                                         transition=dict(duration=dt_ms, easing="linear"),
                                                         fromcurrent=True, mode="immediate")]),
                                   dict(label="❚❚ pause", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                           mode="immediate")])])])

    fig.update_xaxes(title_text="view-plane u (m)", scaleanchor="y", scaleratio=1,
                     range=[-ext0, ext0], row=1, col=1)
    fig.update_yaxes(title_text="v (m)", range=[-ext0, ext0], row=1, col=1)
    fig.update_xaxes(title_text="along-track (km)", row=2, col=2)
    fig.update_yaxes(title_text="η end-to-end (%)", rangemode="tozero", row=2, col=2)
    # Panel B: the model curvature is real (physics uses it); the DISPLAY exaggerates
    # the vertical so the arc is visible over the ±few-hundred-km span (horizontal 1:1).
    fig.update_scenes(dict(xaxis_title="downrange (km)", yaxis_title="cross (km)",
                           zaxis_title="altitude (km, exaggerated)", aspectmode="manual",
                           aspectratio=dict(x=4.0, y=0.25, z=1.0),
                           camera=dict(eye=dict(x=0.1, y=-2.4, z=0.45), up=dict(x=0, y=0, z=1))),
                      row=1, col=2)
    # Panel C: FIXED box so it never rescales as the beam arrow swings.
    fig.update_scenes(dict(xaxis=dict(range=[-BOX, BOX], title="nose (m)"),
                           yaxis=dict(range=[-BOX, BOX], title="span (m)"),
                           zaxis=dict(range=[-BOX, BOX], title="vert (m)"),
                           aspectmode="cube",
                           camera=dict(eye=dict(x=1.4, y=-1.4, z=1.0))), row=2, col=1)
    fig.add_annotation(x=0.5, y=1.0, xref="x domain", yref="y domain", row=1, col=1,
                       text="red contours = beam intensity (fraction of peak)",
                       showarrow=False, font=dict(size=9, color="#b30000"))
    pass_kind = "head-on pass" if abs(lateral_m) < 1.0 else f"lateral pass ({lateral_offset_km:.0f} km offset)"
    fig.update_layout(
        template="plotly_white", height=900, width=1350,
        title=(f"Flyby animation — {arch_name}, {weather} weather, {pass_kind}, "
               f"horizon-to-horizon (±{half/1e3:.0f} km; smooth in time — drag the slider or press play)"),
        legend=dict(orientation="h", y=1.07, x=0.5, xanchor="center", font=dict(size=10)))
    os.makedirs(OUT, exist_ok=True)
    fig.write_html(path, include_plotlyjs="inline", auto_play=False)
    return path


if __name__ == "__main__":
    runs = [
        ("5.8 direct", "storm", 0.0, "58_headon"),
        ("5.8 direct", "storm", 60.0, "58_lateral"),
        ("relay ×3 mult 5.8→17.4", "light_rain", 0.0, "relay_headon"),
        ("relay ×3 mult 5.8→17.4", "light_rain", 60.0, "relay_lateral"),
    ]
    for arch, weather, lateral, tag in runs:
        p = build_flyby_animation(arch_name=arch, weather=weather, lateral_offset_km=lateral,
                                  path=f"{OUT}/beam_link_flyby_animation_{tag}.html")
        print("wrote", p)
