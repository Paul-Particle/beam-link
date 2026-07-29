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


_SIL_EXTENT = None


def max_silhouette_extent():
    """Largest half-extent (m) of the plane's silhouette over a full β/α sweep — a
    property of the airframe, so the SAME fixed value can frame panel A across every run,
    making silhouettes directly comparable. Cached."""
    global _SIL_EXTENT
    if _SIL_EXTENT is None:
        samples, snorm, swt = m.get_surface_samples()
        e = 0.0
        for beta in range(0, 91, 10):
            for alpha in range(0, 181, 30):
                d = m.direction_from_beta_alpha(beta, alpha); d = d / np.linalg.norm(d)
                r = m.rasterize_silhouette(d, samples, snorm, swt)
                if r["u"].size:
                    e = max(e, float(np.abs(r["u"]).max()), float(np.abs(r["v"]).max()))
        _SIL_EXTENT = e
    return _SIL_EXTENT


def _beam_channel(distance_norm, radial, slant_m, max_slant_m, emitter_radius_m, w0, order):
    """Intensity of the CONVERGING beam over a side view: y = along-beam distance (fraction
    of the pass's max slant), x = radial offset. The spot tapers from the emitter aperture
    (radius r_t at the emitter) to the focused waist w0 at the plane (distance = slant), so
    the beam is an hourglass narrowing onto the aircraft. Intensity ∝ 1/w(s)² (brightens as
    it focuses); NaN beyond the plane (no beam past the focus)."""
    C = bl.supergaussian_area_factor(order)
    dd, rr = np.meshgrid(distance_norm, radial, indexing="ij")   # (n_dist, n_radial)
    s = dd * max_slant_m
    frac = np.clip(s / max(slant_m, 1e-6), 0.0, 1.0)
    w = emitter_radius_m + (w0 - emitter_radius_m) * frac
    inten = np.exp(-2.0 * (rr ** 2 / w ** 2) ** order) / (np.pi * w ** 2 * C)
    inten[s > slant_m] = np.nan                          # nothing past the plane
    return inten


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
    geom, eff, _, field = bl.overlap_capture(raster, diam / 2.0, lam, slant, shaped=cfg.shaped_spot,
                                             min_waist_m=w0, pointing_offset_m=cfg.pointing_error_m,
                                             incidence_floor=cfg.incidence_floor,
                                             profile_order=order, return_field=True)
    wf = bl.waterfall(s)                              # [(stage label, efficiency), ...]

    # Orient panel A so the aircraft's DORSAL direction (−ẑ body = "up") points up in the
    # view — otherwise the view basis is arbitrary and the (small, dorsal) winglets can
    # project downward, disagreeing with panel C. Rotate the silhouette coords in-plane;
    # the beam is re-derived from them downstream, so it rotates consistently.
    u_ax, v_ax, _ = m._view_basis(look)
    up_u, up_v = float(np.array([0, 0, -1.0]) @ u_ax), float(np.array([0, 0, -1.0]) @ v_ax)
    if np.hypot(up_u, up_v) > 1e-3 and field["u"].size:
        phi = np.pi / 2.0 - np.arctan2(up_v, up_u)
        cph, sph = np.cos(phi), np.sin(phi)
        U, V = field["u"], field["v"]
        field = dict(field, u=cph * U - sph * V, v=sph * U + cph * V)
    sil_radial_m = (float(np.abs(np.concatenate([field["u"], field["v"]])).max())
                    if field["u"].size else 0.0)

    verts, faces, normals = coarse_mesh
    face_incidence = np.clip(normals @ look, 0.0, 1.0)

    plane_loc = _localize(x_along_m, lateral_offset_m, plane_alt_m, basis)
    emitter_loc = _localize(emitter[0], emitter[1], emitter[2], basis)
    return dict(field=field, w0=w0, shaped=cfg.shaped_spot, profile_order=order,
                geom=geom, eff=eff, eta=eta, elevation=float(elevation), slant_km=slant / 1e3,
                slant_m=float(slant), emitter_radius_m=float(diam / 2.0), sil_radial_m=sil_radial_m,
                beta=float(beta), area=raster["area"], look=look, face_incidence=face_incidence,
                plane_loc=plane_loc, emitter_loc=emitter_loc,
                wf_labels=[lbl for lbl, _ in wf], wf_values=[v for _, v in wf])


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
                          n_frames=81, size_ref_elev_deg=8.0, delivered_mw=10.0, grid_n=110,
                          scatter_n=4000, path=f"{OUT}/beam_link_flyby_animation.html"):
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

    # FIXED panel-A window = the airframe's max silhouette extent (same every frame AND
    # every run, so silhouettes compare directly). 1:1 aspect via scaleanchor below.
    EXT = max_silhouette_extent() * 1.08
    gx = np.linspace(-EXT, EXT, grid_n)
    gy = np.linspace(-EXT, EXT, grid_n)

    def encircled_grid(f):
        """The beam's ENCIRCLED-ENERGY field on the fixed grid: at each cell, the fraction
        of total beam energy carried by all cells at least as bright — so contours at
        0.1…0.9 are rings enclosing that fraction of the delivered energy (centre ≈ 0,
        spilling out to 1). (The silhouette itself is drawn as a high-detail sample scatter,
        not a binned grid — finer, and far smaller in the HTML.)"""
        beam_grid = _beam_on_grid(f["field"], f["w0"], f["shaped"], gx, gy,
                                  profile_order=f["profile_order"])
        flat = beam_grid.ravel()
        order = np.argsort(flat)[::-1]
        csum = np.cumsum(flat[order]) / max(flat.sum(), 1e-30)
        encircled = np.empty_like(flat)
        encircled[order] = csum
        return encircled.reshape(beam_grid.shape)

    def sub_field(f):
        """Down-sampled silhouette samples (u, v, incidence) for the panel-A scatter —
        the raster carries tens of thousands of samples; a few thousand render identically
        and keep the HTML small."""
        fld = f["field"]
        n = fld["u"].size
        if n <= scatter_n:
            return fld["u"], fld["v"], fld["incidence"]
        idx = np.linspace(0, n - 1, scatter_n).astype(int)
        return fld["u"][idx], fld["v"][idx], fld["incidence"][idx]

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
    capture_series = np.array([f["geom"] for f in frames_phys]) * 100   # intensity-weighted overlap
    wf_labels = frames_phys[0]["wf_labels"]
    wf_mat = np.array([f["wf_values"] for f in frames_phys])            # (n_frames, n_stages)
    stage_cum = np.cumprod(wf_mat, axis=1)                              # cumulative product per stage
    # Cumulative delivered energy over the pass ("eat everything": emitter at a fixed grid
    # power sized to hit delivered_mw at the best point; plane banks grid·η each instant).
    peak = max(eta_series.max() / 100.0, 1e-9)
    dt_s = (xs[1] - xs[0]) / 250.0                                       # 250 m/s cruise
    deliverable_mw = (delivered_mw / peak) * (eta_series / 100.0)
    cum_gj = np.cumsum(deliverable_mw * dt_s) / 1e3                      # MW·s = MJ → GJ
    # Panel F (beam channel) axes.
    slant_series = np.array([f["slant_m"] for f in frames_phys])
    max_slant = float(slant_series.max())
    R_max = max(max(f["emitter_radius_m"], f["w0"], f["sil_radial_m"]) for f in frames_phys) * 1.1
    dist_norm = np.linspace(0.0, 1.0, 110)
    radial = np.linspace(-R_max, R_max, 90)
    r_t0 = frames_phys[0]["emitter_radius_m"]
    BOX = 45.0                                        # fixed half-extent of panel C (m)
    stage_colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2", "#9d755d"]

    fig = make_subplots(
        rows=2, cols=3, horizontal_spacing=0.115, vertical_spacing=0.15,
        specs=[[{"type": "xy"}, {"type": "scene"}, {"type": "scene"}],
               [{"secondary_y": True}, {"type": "xy"}, {"type": "xy"}]],
        subplot_titles=("A · beam spot over silhouette (dots = incidence)",
                        "B · geometry on a spherical earth (vertical exaggerated)",
                        "C · BWB facets by beam incidence",
                        "D · capture, η & cumulative energy along the pass",
                        "E · cumulative efficiency after each stage",
                        "F · beam channel emitter→plane (radial exaggerated)"))

    # ---- Panel A: silhouette sample scatter (incidence) + encircled-energy contour ------
    su0, sv0, si0 = sub_field(frames_phys[0])
    i_scatA = len(fig.data)
    fig.add_trace(go.Scattergl(x=su0, y=sv0, mode="markers",
                               marker=dict(size=3, color=si0, colorscale="Viridis",
                                           cmin=0, cmax=1, showscale=False),
                               showlegend=False, hoverinfo="skip"), row=1, col=1)
    contour_opts = dict(coloring="lines", showlabels=True, start=0.1, end=0.9, size=0.2,
                        labelfont=dict(size=9, color="#b30000"))
    i_contA = len(fig.data)
    fig.add_trace(go.Contour(x=gx, y=gy, z=encircled_grid(frames_phys[0]), showscale=False,
                             colorscale="Reds", contours=contour_opts, line=dict(width=1.3),
                             hoverinfo="skip"), row=1, col=1)

    # ---- Panel B (3-D geometry): arcs, emitter/base, plane + beam ----------------------
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
    p0 = frames_phys[0]["plane_loc"] / 1e3
    i_plane = len(fig.data)
    fig.add_trace(go.Scatter3d(x=[p0[0]], y=[p0[1]], z=[p0[2]], mode="markers",
                               marker=dict(size=5, color="#d62728"), name="plane", hoverinfo="skip"), row=1, col=2)
    i_beam = len(fig.data)
    fig.add_trace(go.Scatter3d(x=[emitter_loc[0], p0[0]], y=[emitter_loc[1], p0[1]],
                               z=[emitter_loc[2], p0[2]], mode="lines",
                               line=dict(color="#d62728", width=3), name="beam", hoverinfo="skip"), row=1, col=2)

    # ---- Panel C (3-D BWB by incidence): z negated (body frame is z-DOWN, belly at +z) --
    i_mesh = len(fig.data)
    fig.add_trace(go.Mesh3d(x=coarse_v[:, 0], y=coarse_v[:, 1], z=-coarse_v[:, 2],
                            i=coarse_f[:, 0], j=coarse_f[:, 1], k=coarse_f[:, 2],
                            intensity=frames_phys[0]["face_incidence"], intensitymode="cell",
                            colorscale="Viridis", cmin=0, cmax=1, showscale=False,
                            flatshading=True, name="BWB", hoverinfo="skip"), row=1, col=3)
    look0 = frames_phys[0]["look"]
    i_arrow = len(fig.data)
    fig.add_trace(go.Scatter3d(x=[look0[0] * 40, look0[0] * 15], y=[look0[1] * 40, look0[1] * 15],
                               z=[-look0[2] * 40, -look0[2] * 15], mode="lines",
                               line=dict(color="#d62728", width=6), name="beam→",
                               showlegend=False, hoverinfo="skip"), row=1, col=3)

    # ---- Panel D (row2,col1): capture, η (left %) + cumulative energy (right GJ) --------
    fig.add_trace(go.Scatter(x=xs / 1e3, y=eta_series, mode="lines", name="η end-to-end (%)",
                             line=dict(color="#1f77b4", width=2), hoverinfo="skip"),
                  row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=xs / 1e3, y=capture_series, mode="lines", name="overlap capture (%)",
                             line=dict(color="#2ca02c", width=2, dash="dot"), hoverinfo="skip"),
                  row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=xs / 1e3, y=cum_gj, mode="lines", name="cumulative energy (GJ)",
                             line=dict(color="#ff7f0e", width=2), hoverinfo="skip"),
                  row=2, col=1, secondary_y=True)
    i_eta_m = len(fig.data)
    fig.add_trace(go.Scatter(x=[xs[0] / 1e3], y=[eta_series[0]], mode="markers",
                             marker=dict(size=9, color="#1f77b4"), showlegend=False, hoverinfo="skip"),
                  row=2, col=1, secondary_y=False)
    i_cap_m = len(fig.data)
    fig.add_trace(go.Scatter(x=[xs[0] / 1e3], y=[capture_series[0]], mode="markers",
                             marker=dict(size=9, color="#2ca02c"), showlegend=False, hoverinfo="skip"),
                  row=2, col=1, secondary_y=False)
    i_cum_m = len(fig.data)
    fig.add_trace(go.Scatter(x=[xs[0] / 1e3], y=[cum_gj[0]], mode="markers",
                             marker=dict(size=9, color="#ff7f0e"), showlegend=False, hoverinfo="skip"),
                  row=2, col=1, secondary_y=True)

    # ---- Panel E (row2,col2): cumulative efficiency after each stage (stacked curves).
    # Labelled inline at overhead (not in the legend, which would crowd B/D).
    mid = n_frames // 2
    fig.add_trace(go.Scatter(x=xs / 1e3, y=np.ones_like(xs), mode="lines",
                             line=dict(color="#bbbbbb", width=1, dash="dot"),
                             showlegend=False, hoverinfo="skip"), row=2, col=2)
    for si, lbl in enumerate(wf_labels):
        col = stage_colors[si % len(stage_colors)]
        fig.add_trace(go.Scatter(x=xs / 1e3, y=stage_cum[:, si], mode="lines",
                                 line=dict(color=col, width=2),
                                 showlegend=False, hoverinfo="skip"), row=2, col=2)
        fig.add_annotation(x=0.0, y=stage_cum[mid, si], text=f"×{lbl}", xanchor="center",
                           yanchor="bottom", showarrow=False, font=dict(size=8, color=col),
                           row=2, col=2)
    i_e_marker = len(fig.data)
    fig.add_trace(go.Scatter(x=[xs[0] / 1e3, xs[0] / 1e3], y=[0.0, 1.02], mode="lines",
                             line=dict(color="#888", width=1, dash="dot"),
                             showlegend=False, hoverinfo="skip"), row=2, col=2)

    # ---- Panel F (row2,col3): converging beam channel; white emitter line + plane slice -
    i_fheat = len(fig.data)
    fig.add_trace(go.Heatmap(x=radial, y=dist_norm,
                             z=_beam_channel(dist_norm, radial, slant_series[0], max_slant, r_t0,
                                             frames_phys[0]["w0"], frames_phys[0]["profile_order"]),
                             colorscale="Inferno", showscale=False, hoverinfo="skip"), row=2, col=3)
    fig.add_trace(go.Scatter(x=[-r_t0, r_t0], y=[0.0, 0.0], mode="lines",
                             line=dict(color="white", width=5), name="emitter aperture",
                             showlegend=False, hoverinfo="skip"), row=2, col=3)
    sil0 = frames_phys[0]["sil_radial_m"]
    i_fplane = len(fig.data)
    fig.add_trace(go.Scatter(x=[-sil0, sil0], y=[slant_series[0] / max_slant] * 2, mode="lines",
                             line=dict(color="#00e5ff", width=4), name="plane",
                             showlegend=False, hoverinfo="skip"), row=2, col=3)

    ANIMATED = [i_scatA, i_contA, i_plane, i_beam, i_mesh, i_arrow,
                i_eta_m, i_cap_m, i_cum_m, i_e_marker, i_fheat, i_fplane]
    frames = []
    for k, f in enumerate(frames_phys):
        p = f["plane_loc"] / 1e3
        look = f["look"]
        su, sv, sinc = sub_field(f)
        cur = xs[k] / 1e3
        beamF = _beam_channel(dist_norm, radial, f["slant_m"], max_slant,
                              f["emitter_radius_m"], f["w0"], f["profile_order"])
        yy = f["slant_m"] / max_slant
        frames.append(go.Frame(name=str(k), traces=ANIMATED, data=[
            go.Scattergl(x=su, y=sv, marker=dict(color=sinc)),
            go.Contour(x=gx, y=gy, z=encircled_grid(f)),
            go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]]),
            go.Scatter3d(x=[emitter_loc[0], p[0]], y=[emitter_loc[1], p[1]], z=[emitter_loc[2], p[2]]),
            go.Mesh3d(intensity=f["face_incidence"], intensitymode="cell"),
            go.Scatter3d(x=[look[0] * 40, look[0] * 15], y=[look[1] * 40, look[1] * 15],
                         z=[-look[2] * 40, -look[2] * 15]),
            go.Scatter(x=[cur], y=[eta_series[k]]),
            go.Scatter(x=[cur], y=[capture_series[k]]),
            go.Scatter(x=[cur], y=[cum_gj[k]]),
            go.Scatter(x=[cur, cur], y=[0.0, 1.02]),
            go.Heatmap(z=beamF),
            go.Scatter(x=[-f["sil_radial_m"], f["sil_radial_m"]], y=[yy, yy]),
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
                     range=[-EXT, EXT], row=1, col=1)
    fig.update_yaxes(title_text="v (m)", range=[-EXT, EXT], row=1, col=1)
    fig.update_xaxes(title_text="along-track (km)", row=2, col=1)
    fig.update_yaxes(title_text="capture / η (%)", rangemode="tozero", secondary_y=False, row=2, col=1)
    fig.update_yaxes(title_text="cumulative energy (GJ)", rangemode="tozero", secondary_y=True, row=2, col=1)
    fig.update_xaxes(title_text="along-track (km)", row=2, col=2)
    fig.update_yaxes(title_text="cumulative efficiency (×)", range=[0, 1.05], row=2, col=2)
    fig.update_xaxes(title_text="radial offset (m)", row=2, col=3)
    fig.update_yaxes(title_text="distance emitter→plane (norm.)", range=[0, 1], row=2, col=3)
    # Panel B: the model curvature is real (physics uses it); the DISPLAY exaggerates the
    # vertical so the arc is visible. Camera pulled back so the axes/labels are in view.
    fig.update_scenes(dict(xaxis_title="downrange (km)", yaxis_title="cross (km)",
                           zaxis_title="altitude (km, exaggerated)", aspectmode="manual",
                           aspectratio=dict(x=4.0, y=0.25, z=1.0),
                           camera=dict(eye=dict(x=0.15, y=-3.7, z=0.72), up=dict(x=0, y=0, z=1))),
                      row=1, col=2)
    # Panel C: FIXED box so it never rescales as the beam arrow swings.
    fig.update_scenes(dict(xaxis=dict(range=[-BOX, BOX], title="nose (m)"),
                           yaxis=dict(range=[-BOX, BOX], title="span (m)"),
                           zaxis=dict(range=[-BOX, BOX], title="vert (m, up)"),
                           aspectmode="cube",
                           camera=dict(eye=dict(x=1.7, y=-1.7, z=1.2))), row=1, col=3)
    fig.add_annotation(x=0.5, y=1.0, xref="x domain", yref="y domain", row=1, col=1,
                       text="red contours = enclosed beam-energy fraction (centre→edge)",
                       showarrow=False, font=dict(size=9, color="#b30000"))
    pass_kind = "head-on pass" if abs(lateral_m) < 1.0 else f"lateral pass ({lateral_offset_km:.0f} km offset)"
    fig.update_layout(
        template="plotly_white", height=940, width=1680,
        title=(f"Flyby animation — {arch_name}, {weather} weather, {pass_kind}, "
               f"horizon-to-horizon (±{half/1e3:.0f} km; smooth in time — drag the slider or press play)"),
        legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center", font=dict(size=9)))
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
