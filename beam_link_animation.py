"""Flyby animation for the beam-link engine — the whole link in one moving picture.

A time slider drives synchronized views over a single pass, stacked in one column:
  A (2-D)  the focused beam spot (encircled-energy contours) over the self-occluded
           silhouette (filled by incidence) — where the power lands, frame by frame.
  D (2-D)  capture, end-to-end η and cumulative delivered energy along the pass.
  F  (2-D) the converging beam channel emitter→plane, MAJOR-axis (wide) cut (radial
           exaggerated), with the bottom half of the emitter phase fused at its base.
  F₂ (2-D) the same channel, MINOR-axis (narrow) cut, with its emitter-phase half — the
           90°-rotated view of F's aperture (long axis along-radial instead of short).

The point is to spot inconsistencies at a glance (does the spot overspill? does the far
wing go dark? does the aperture phase tighten toward the waist?) far faster than tables.

Run:  uv run python beam_link_animation.py   → out/flyby.html
(the single reference case — see the __main__ block; call build_flyby_animation() directly
for other architectures, apertures, weathers or pass geometries.)
"""
import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import beam_link as bl
import diffraction as dfr
import bwb_model as m
import geo

OUT = "out"
# Diffraction transform grid for the beam-channel panels (F/F₂). 384 (not 192): the shaped
# emitter's elongated aperture (semi-major ≈ √aspect · r_emit, ~141 m at aspect 8) only fits
# the transform window with adequate margin at this size. At 192 the far/defocused frames alias
# badly (RMS ~17 %, peak ~40 % off a converged reference — visible as non-monotone banding);
# 384 converges (RMS <0.2 %). n≥320 also trips scipy's multi-threaded FFT (see _fft_workers).
CHANNEL_NGRID = 384

# Play button: a custom sequential stepper. Plotly's native sequenced play only redraws
# Heatmap/Waterfall on the FIRST frame (the rest freeze), and `Plotly.animate([k],redraw)`
# tears down + re-renders both WebGL 3-D scenes each tick → flicker. So this advances ONE
# frame per tick with `Plotly.restyle` on just the animated traces (in-place, in order, no
# teardown), which still redraws the heatmaps/waterfall. `{n_frames}` is injected (gd.frames
# isn't populated when this runs); `{plot_id}` is substituted by write_html.
_PLAYER_JS = """
var gd = document.getElementById("{plot_id}");
var N = {n_frames};
var cur = 0, playing = false, timer = null;
function frameByName(k) {
  var frs = (gd._transitionData && gd._transitionData._frames) || gd.frames || [];
  for (var i = 0; i < frs.length; i++) { if (frs[i] && frs[i].name === String(k)) return frs[i]; }
  return null;
}
function applyFrame(k) {
  var fr = frameByName(k);
  if (!fr) return Promise.resolve();
  var proms = [];
  for (var j = 0; j < fr.data.length; j++) {
    var d = fr.data[j], tr = fr.traces ? fr.traces[j] : j, upd = {};
    for (var key in d) { if (d.hasOwnProperty(key) && key !== 'type') { upd[key] = [d[key]]; } }
    proms.push(Plotly.restyle(gd, upd, [tr]));
  }
  cur = k;
  return Promise.all(proms);
}
function loop() { if (!playing) return; applyFrame((cur + 1) % N).then(function(){ if (playing) timer = setTimeout(loop, 8); }); }
var btn = document.createElement('button');
btn.textContent = '\\u25B6 play';
btn.style.cssText = 'position:absolute;left:6px;top:6px;z-index:999;font:13px sans-serif;padding:3px 12px;cursor:pointer;background:#2a2a2a;color:#eee;border:1px solid #666;border-radius:3px;';
function play() { if (playing) return; playing = true; btn.textContent = '\\u275A\\u275A pause'; loop(); }
function pause() { playing = false; if (timer) clearTimeout(timer); btn.textContent = '\\u25B6 play'; }
btn.onclick = function(){ playing ? pause() : play(); };
var host = gd.parentNode || gd;
if (getComputedStyle(host).position === 'static') host.style.position = 'relative';
host.appendChild(btn);
gd.on('plotly_sliderchange', function(e){ if (e && e.slider) cur = e.slider.active | 0; });
"""


def _box_sum(a, r):
    """Sum over a (2r+1)² neighbourhood at each cell (zero-padded) — a fast separable-ish
    box filter used to fill the gaps between Monte-Carlo silhouette samples."""
    height, width = a.shape
    padded = np.pad(a, r)
    out = np.zeros_like(a)
    for di in range(2 * r + 1):
        for dj in range(2 * r + 1):
            out += padded[di:di + height, dj:dj + width]
    return out


# --- one frame's worth of physics --------------------------------------------

def _localize(downrange_m, crossrange_m, altitude_m, basis):
    """ECEF point → station-local (downrange, crossrange, up) metres, so earth
    curvature shows as the ground and cruise arcs falling away at the edges."""
    base, east, north, up = basis
    d = geo.point(downrange_m, crossrange_m, altitude_m) - base
    return np.array([np.dot(d, north), np.dot(d, east), np.dot(d, up)])


def _beam_on_grid(field, gx, gy):
    """The emitter's SCALAR-DIFFRACTION target-plane intensity (from overlap_capture's
    returned field: x_axis/y_axis/intensity in the aperture principal frame) resampled onto
    the panel grid (gx, gy) and aligned to the DISPLAYED silhouette's principal axes — so the
    picture is exactly the field the capture integral used, sidelobes and all. Returns the
    beam grid with plotly's z[row=y, col=x] shape (len(gy), len(gx))."""
    uu, vv = np.meshgrid(gx, gy)                              # (len(gy), len(gx))
    if field["u"].size < 3 or "intensity" not in field:
        return np.zeros_like(uu)
    coords = np.vstack([field["u"], field["v"]])
    _, evecs = np.linalg.eigh(np.cov(coords))                # display-frame principal axes
    proj = evecs.T @ np.vstack([uu.ravel(), vv.ravel()])     # rows: [along minor, along major]
    # grid x-axis = spot minor (silhouette minor coord); grid y-axis = spot major.
    beam = dfr.sample_focal_intensity(field["x_axis"], field["y_axis"], field["intensity"],
                                      proj[0], proj[1])
    return beam.reshape(uu.shape)


def _channel_field(dist_norm, radial, legs, total_shown_m, current_total_m):
    """Scalar-diffraction beam along the whole path, as a (n_dist, n_radial) side view: the
    along-path axis (fraction of the longest total path over the pass) is laid out over the
    link's legs — one leg for a direct link, TWO for a relay (ground→relay up-leg focused at
    the relay, then relay→plane down-leg focused at the plane). Both principal cuts are computed
    in ONE diffraction pass per leg (`diffraction.axial_cut_xy`, complex64 — display precision),
    so the picture is the model: converging cone → diffraction waist → divergence, Airy sidelobes
    included. NaN beyond the current plane position. Returns (ix, ex, iy, ey): intensity of the
    narrow (x) and wide (y) cuts, each normalised to the frame peak, with the strip-marginal
    encircled-energy fraction for the funnel."""
    s = dist_norm * total_shown_m
    shp = (dist_norm.size, radial.size)
    ix = np.full(shp, np.nan); ex = np.full(shp, np.nan)
    iy = np.full(shp, np.nan); ey = np.full(shp, np.nan)
    cum = 0.0
    for leg in legs:
        sel = (s >= cum) & (s < cum + leg["length_m"]) & (s <= current_total_m)
        if leg["length_m"] > 0 and np.any(sel):
            local_z = s[sel] - cum                           # distance from this leg's emitter
            lix, lex, liy, ley = dfr.axial_cut_xy(
                local_z, leg["focus_m"], leg["ap_major"], leg["ap_minor"], leg["lam"], radial,
                taper_p=leg["taper_p"], n_grid=CHANNEL_NGRID, normalize=False, dtype=np.complex64)
            ix[sel] = lix; ex[sel] = lex; iy[sel] = liy; ey[sel] = ley
        cum += leg["length_m"]

    def _norm(inten):
        finite = np.isfinite(inten)
        peak = inten[finite].max() if finite.any() else 0.0
        return (inten / peak if peak > 0 else inten).astype(np.float32)
    return _norm(ix), ex.astype(np.float32), _norm(iy), ey.astype(np.float32)


_SIL_EXTENT = None


def max_silhouette_extent():
    """Largest silhouette half-extent (m) over a full β/α sweep — an airframe property, so the
    SAME fixed value frames panel A across every run (identical scale, comparable silhouettes).
    Cached."""
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
    # The engine is the single source of truth: it chooses the aperture TAPER (adaptively) and
    # the spot SHAPE (elliptical) and enforces the density cap (possibly DEFOCUSING to hold the
    # W/m² cap). evaluate_with_field hands back the exact scalar-diffraction field the capture
    # integral used (winning taper + focus + pointing), so panel A IS that field — no recompute —
    # and matches panel D's η.
    s, field = bl.evaluate_with_field(cfg, delivered_mw)
    eta = float(s["eta_end_to_end"])
    taper_p = float(s["aperture_taper_p"])
    focus = float(s.get("focus_range_m", slant))
    geom = float(s.get("capture_geom", s.get("capture_down_geom", s["capture"])))
    eff = float(s["capture"])
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

    # The silhouette's TRUE major/minor extents (computed regardless of spot shape, so a
    # ROUND-emitter figure still shows the beam over the elongated outline). The emitter
    # aperture aspect is what overlap_capture actually used (round, fixed, or adaptive).
    if field["u"].size >= 3:
        coords = np.vstack([field["u"], field["v"]])
        ev, evec = np.linalg.eigh(np.cov(coords))
        proj = evec.T @ coords                           # rows: [along minor, along major]
        sil_minor_m = float(np.abs(proj[0]).max())
        sil_major_m = float(np.abs(proj[1]).max())
    else:
        sil_major_m = sil_minor_m = sil_radial_m
    aspect = float(field.get("aspect", 1.0))
    r_emit = float(diam / 2.0)
    # Long aperture axis → narrow spot (Fourier inverse). Major-axis cut ('y') = wide spot from
    # the SHORT aperture axis; minor-axis cut ('x') = narrow spot from the LONG axis.
    ap_major, ap_minor = r_emit * np.sqrt(aspect), r_emit / np.sqrt(aspect)

    # Panel-F beam-channel legs (end-to-end along the path). Relay = a ground→relay up-leg (round
    # ground aperture auto-sized to the relay at that range, focused at the relay) followed by
    # the focused relay→plane down-leg (the relay's elliptical aperture); direct = one leg. Each
    # leg carries its aperture semi-axes, focus range, wavelength and taper — the channel
    # renderer computes the exact diffraction field per leg.
    down_leg = dict(ap_major=ap_major, ap_minor=ap_minor, focus_m=focus,
                    length_m=float(slant), lam=lam, taper_p=taper_p)
    if cfg.relay_pos is not None:
        station = np.asarray(cfg.station_pos, float)
        L1 = float(bl.geometry(station, np.asarray(cfg.relay_pos, float))[0])   # ground→relay
        r_relay = cfg.relay_aperture_diam_m / 2.0
        r_up = bl.size_emitter_diameter(cfg.f_emit_relay_ghz, L1, r_relay, cfg.tau_target) / 2.0
        lam_up = bl.napkin.SPEED_OF_LIGHT / (cfg.f_emit_relay_ghz * 1e9)
        up_leg = dict(ap_major=float(r_up), ap_minor=float(r_up), focus_m=L1,
                      length_m=L1, lam=lam_up, taper_p=0.0)
        chan_legs = [up_leg, down_leg]
        chan_total = L1 + float(slant)
        chan_relay_m, chan_relay_radius_m, chan_emit_radius_m = L1, r_relay, float(r_up)
    else:
        chan_legs = [down_leg]
        chan_total = float(slant)
        chan_relay_m, chan_relay_radius_m, chan_emit_radius_m = None, None, r_emit

    plane_loc = _localize(x_along_m, lateral_offset_m, plane_alt_m, basis)
    emitter_loc = _localize(emitter[0], emitter[1], emitter[2], basis)
    return dict(field=field, aspect=aspect, taper_p=taper_p,
                geom=geom, eff=eff, eta=eta, elevation=float(elevation), slant_km=slant / 1e3,
                slant_m=float(slant), emitter_radius_m=r_emit, sil_radial_m=sil_radial_m,
                chan_legs=chan_legs,
                chan_total=chan_total, chan_relay_m=chan_relay_m,
                chan_relay_radius_m=chan_relay_radius_m, chan_emit_radius_m=chan_emit_radius_m,
                sil_major_m=sil_major_m, sil_minor_m=sil_minor_m,
                beta=float(beta), area=raster["area"], look=look, face_incidence=face_incidence,
                plane_loc=plane_loc, emitter_loc=emitter_loc,
                phase_d=(np.asarray(plane, float) - emitter), phase_R=float(diam / 2.0),
                phase_lam=lam, phase_normal_up=(cfg.relay_pos is None),
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
                          n_frames=121, size_ref_elev_deg=8.0, delivered_mw=10.0, grid_n=110,
                          grid_nA=200, emitter_diam_m=None, arch_kwargs=None,
                          write=True, path=f"{OUT}/beam_link_flyby_animation.html"):
    """Build (and, if `write`, save) the 6-panel flyby animation, returning the figure.

    `arch_name` selects a named case from `standard_architectures`; pass `arch_kwargs`
    (a LinkConfig kwarg dict, `plane_pos` omitted) instead to animate an architecture not
    in that catalogue (e.g. a passive same-frequency relay), with `arch_name` used only as
    the display label. `emitter_diam_m` sets the emitter aperture EXPLICITLY (direct →
    `emitter_diam_m`, relay → `relay_aperture_diam_m`) and bypasses the far-reference
    auto-sizing — so emitter size can be swept as the single variable. `grid_nA` is the
    panel-A silhouette-grid resolution (lower it for lighter embedded figures)."""
    if arch_kwargs is None:
        archs = bl.standard_architectures(weather=weather)
        if arch_name not in archs:
            raise KeyError(f"{arch_name!r} not in {list(archs)}")
        kw = dict(archs[arch_name])
    else:
        kw = dict(arch_kwargs)
        kw["weather"] = weather                       # the weather arg governs (arcs + physics)
    if emitter_diam_m is not None:
        if kw.get("relay_pos") is not None:
            kw["relay_aperture_diam_m"] = float(emitter_diam_m)
        else:
            kw["emitter_diam_m"] = float(emitter_diam_m)
    else:
        kw = _size_emitter_far(kw, size_ref_elev_deg, plane_alt_m)
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

    # Panel-A window: a FIXED zoom, IDENTICAL in every figure (so silhouettes compare at one
    # scale), sized so the airframe's largest silhouette dimension fills ~ZOOM_FILL of the
    # panel's long (horizontal) axis. Wide window (aspect PANEL_A_ASPECT) matched to the wide
    # panel so it fills the row; scaleanchor below keeps it 1:1, so the spot is undistorted.
    # A heavily defocused beam can exceed this window and clip at the edges — the silhouette
    # framing is the priority.
    ZOOM_FILL = 0.75
    PANEL_A_ASPECT = 2.0                             # panel A width:height (spans both columns)
    EXT_H = max_silhouette_extent() / ZOOM_FILL
    EXT_V = EXT_H / PANEL_A_ASPECT
    box_r = max(2, round(grid_nA / 75))              # smoothing radius scales with resolution
    gx = np.linspace(-EXT_H, EXT_H, grid_n)          # contour grid (smooth beam)
    gy = np.linspace(-EXT_V, EXT_V, grid_n)
    gxA = np.linspace(-EXT_H, EXT_H, grid_nA)        # fine grid for the silhouette heatmap
    gyA = np.linspace(-EXT_V, EXT_V, grid_nA)

    def encircled_grid(f):
        """The beam's ENCIRCLED-ENERGY field on the fixed grid: at each cell, the fraction
        of the beam's TOTAL radiated energy carried by all cells at least as bright — so
        contours at 0.1…0.9 are rings enclosing that fraction of the radiated beam. The
        diffraction field is power-normalised (∫∫ I dA = 1 over all space), so the divisor is
        1: when the beam overspills the panel window the outer rings correctly fail to appear
        (the beam is shown as uncontained), matching the low capture number."""
        beam_grid = _beam_on_grid(f["field"], gx, gy)
        flat = beam_grid.ravel()
        order = np.argsort(flat)[::-1]
        dcell = (gx[1] - gx[0]) * (gy[1] - gy[0])
        csum = np.clip(np.cumsum(flat[order]) * dcell, 0.0, 1.0)   # total radiated power = 1
        encircled = np.empty_like(flat)
        encircled[order] = csum
        return encircled.reshape(beam_grid.shape).astype(np.float32)

    def incidence_grid(f):
        """The silhouette's incidence on the fine panel-A grid, BOX-SMOOTHED to fill the
        gaps between Monte-Carlo samples: a clean, high-res filled outline instead of
        per-cell speckle. NaN only where no sample lies within the smoothing radius (so
        the true outline is preserved). An SVG heatmap so the contour can overlay it (a
        WebGL scatter would render ABOVE the contour and hide it, and left holes)."""
        fld = f["field"]
        if fld["u"].size == 0:
            return np.full((grid_nA, grid_nA), np.nan)
        iu = np.clip(((fld["u"] + EXT_H) / (2 * EXT_H) * (grid_nA - 1)).astype(int), 0, grid_nA - 1)
        iv = np.clip(((fld["v"] + EXT_V) / (2 * EXT_V) * (grid_nA - 1)).astype(int), 0, grid_nA - 1)
        cell = iv * grid_nA + iu                       # z[row=v, col=u]
        ssum = np.zeros(grid_nA * grid_nA); cnt = np.zeros(grid_nA * grid_nA)
        np.add.at(ssum, cell, fld["incidence"]); np.add.at(cnt, cell, 1.0)
        ss = _box_sum(ssum.reshape(grid_nA, grid_nA), box_r)
        cc = _box_sum(cnt.reshape(grid_nA, grid_nA), box_r)
        z = np.where(cc > 0.5, ss / np.maximum(cc, 1e-9), np.nan)
        return z.astype(np.float32)                   # float32 halves the base64 payload

    half = float(xs.max())

    eta_series = np.array([f["eta"] for f in frames_phys]) * 100
    capture_series = np.array([f["geom"] for f in frames_phys]) * 100   # intensity-weighted overlap
    # Cumulative delivered energy over the pass ("eat everything": emitter at a fixed grid
    # power sized to hit delivered_mw at the best point; plane banks grid·η each instant).
    peak = max(eta_series.max() / 100.0, 1e-9)
    dt_s = (xs[1] - xs[0]) / 250.0                                       # 250 m/s cruise
    deliverable_mw = (delivered_mw / peak) * (eta_series / 100.0)
    cum_gj = np.cumsum(deliverable_mw * dt_s) / 1e3                      # MW·s = MJ → GJ
    # Panels F/F₂ (beam channel): the along-path axis is normalised to the LONGEST path over the
    # pass, so the plane's position along it encodes actual slant (far = near the far end, close
    # = near the emitter) — that distance cue is what keeps F/F₂ from being redundant with A.
    # Radial extent anchored to the EMITTER aperture and the silhouette (both bounded), NOT to
    # the beam waist: far out the defocused elliptical major waist is hundreds of m, which would
    # shrink the emitter and all near-field detail to a blob. A beam wider than this window
    # simply fills the panel with no contours — the correct read for a badly overspilling beam.
    max_total = max(f["chan_total"] for f in frames_phys)
    r_emit = max(f["chan_emit_radius_m"] for f in frames_phys)
    r_sil = max(f["sil_major_m"] for f in frames_phys)
    R_max = 1.4 * max(r_emit, r_sil)
    dist_norm = np.linspace(0.0, 1.0, 110)   # along-path samples (one diffraction pass each)
    radial = np.linspace(-R_max, R_max, 121)  # radial samples — cheap (interp only); smooths contours
    r_t0 = frames_phys[0]["chan_emit_radius_m"]

    # Intensity self-normalises per frame so the beam shape is always visible (a pass-peak
    # scale left all but the closest frames dark blue); the along-path axis spans max_total and
    # the beam is masked beyond the current plane position (current_total_m = this slant). The
    # encircled-energy contours are absolute fractions, so brightness normalisation is moot for
    # reading "how much lands on the plane".
    _chan_cache = {}

    def _channel_all(f):                              # (ix, ex, iy, ey) — both cuts, one pass
        key = id(f)
        if key not in _chan_cache:
            _chan_cache[key] = _channel_field(dist_norm, radial, f["chan_legs"], max_total,
                                              f["chan_total"])
        return _chan_cache[key]

    def channel(f, axis):                             # axis 'y' = major (wide), 'x' = minor (narrow)
        ix, ex, iy, ey = _channel_all(f)
        return iy if axis == "y" else ix

    def channel_enc(f, axis):
        ix, ex, iy, ey = _channel_all(f)
        return ey if axis == "y" else ex

    # Emitter phase, split under each beam map (panels F/F₂), in its OWN 1:1 subplot so the
    # aperture is undistorted. The disk is halved (bottom semicircle only). The two halves show
    # the SAME elliptical aperture rotated 90°: F's radial axis is Fourier-conjugate to the SHORT
    # aperture axis (short axis along-radial, long axis running down), F₂'s to the LONG axis. Both
    # are drawn to a common metres scale ±PHASE_L that fits the LARGEST aperture dimension, so
    # their sizes and the 90° rotation are literal. Rings are the symmetric focusing zone plate
    # φ = wrap(−(2π/λ)(√(f²+ρ²) − f)) — it tightens as the focus nears.
    _ap = [(f["chan_legs"][-1]["ap_major"], f["chan_legs"][-1]["ap_minor"]) for f in frames_phys]
    ap_short = min(min(a) for a in _ap)
    ap_long = max(max(a) for a in _ap)
    PHASE_L = 1.12 * ap_long                           # metres half-extent (fits the long axis)
    phase_x = np.linspace(-PHASE_L, PHASE_L, 161)
    phase_y_m = np.linspace(-PHASE_L, 0.0, 81)         # bottom half only

    def half_phase(f, ap_radial_m, ap_depth_m):        # bottom-half zone plate, metres grid
        U, V = np.meshgrid(phase_x, phase_y_m)
        inside = (U / ap_radial_m) ** 2 + (V / ap_depth_m) ** 2 <= 1.0
        lam = float(f["chan_legs"][-1]["lam"]); foc = float(f["chan_legs"][-1]["focus_m"])
        phi = -(2.0 * np.pi / lam) * (np.sqrt(foc ** 2 + U ** 2 + V ** 2) - foc)
        phi = (phi + np.pi) % (2.0 * np.pi) - np.pi
        return np.where(inside, phi, np.nan).astype(np.float32)

    def half_outline(ap_radial_m, ap_depth_m):         # thin white boundary: arc + top diameter
        t = np.linspace(np.pi, 2.0 * np.pi, 160)       # so the dark phase values don't vanish
        ax, ay = ap_radial_m * np.cos(t), ap_depth_m * np.sin(t)
        return np.append(ax, ax[0]), np.append(ay, ay[0])   # close (chord = the top diameter)

    # Layout: a single stacked column — beam spot over the silhouette (A), the along-pass curves
    # (D), then each beam-channel map with its 1:1 emitter-phase half directly below, so the beam
    # grows out of its own (undistorted) half of the aperture.
    fig = make_subplots(
        rows=6, cols=1, vertical_spacing=0.035,
        row_heights=[0.185, 0.10, 0.16, 0.12, 0.16, 0.12],
        specs=[[{"type": "xy"}],
               [{"secondary_y": True}],
               [{"type": "xy"}],
               [{"type": "xy"}],
               [{"type": "xy"}],
               [{"type": "xy"}]],
        subplot_titles=("A · beam spot over silhouette (fill = incidence)",
                        "D · capture, η & cumulative energy along the pass",
                        "F · beam channel, major-axis (wide) cut",
                        "emitter phase — short axis along radial (1:1)",
                        "F₂ · beam channel, minor-axis (narrow) cut",
                        "emitter phase — long axis along radial (1:1, 90° rotated)"))

    # ---- Panel A (1,1): incidence heatmap (filled outline) + encircled-energy contour ---
    i_incA = len(fig.data)
    fig.add_trace(go.Heatmap(x=gxA, y=gyA, z=incidence_grid(frames_phys[0]),
                             colorscale="Plasma", zmin=0, zmax=1, zauto=False, zsmooth="fast",
                             showscale=False, hoverinfo="skip"), row=1, col=1)
    contour_opts = dict(coloring="lines", showlabels=True, start=0.1, end=0.9, size=0.2,
                        labelfont=dict(size=9, color="#c8ffe6"))
    i_contA = len(fig.data)
    # Bright mint-green: reads cleanly over Plasma (which has no green), unlike the old red.
    # Toned down (opacity) so the beam overlay sits lightly over the silhouette fill.
    fig.add_trace(go.Contour(x=gx, y=gy, z=encircled_grid(frames_phys[0]), showscale=False,
                             colorscale=[[0.0, "#7dffce"], [1.0, "#00f5a0"]],
                             contours=contour_opts, line=dict(width=1.1), opacity=0.45,
                             hoverinfo="skip"), row=1, col=1)
    # Live capture % in the corner of panel A — the beam fraction the silhouette intercepts.
    i_capA = len(fig.data)
    fig.add_trace(go.Scatter(x=[-EXT_H * 0.94], y=[EXT_V * 0.82], mode="text",
                             text=[f"{capture_series[0]:.0f}% captured"], textposition="middle right",
                             textfont=dict(size=16, color="#ffffff"), cliponaxis=False,
                             showlegend=False, hoverinfo="skip"), row=1, col=1)

    # ---- Panel D (row 2): capture, η (left %) + cumulative energy (right GJ) -------------
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

    yy0 = frames_phys[0]["chan_total"] / max_total

    def add_pair(f0, cut, sil_key, ap_radial_m, ap_depth_m, map_row, phase_row):
        """A beam-channel map (emitter at y=0, beam up to the plane slice at y = slant/max_slant)
        plus, in the row below, the bottom half of the emitter phase in its OWN 1:1 subplot.
        Returns (i_heat, i_cont, i_plane, i_phase)."""
        i_heat = len(fig.data)
        fig.add_trace(go.Heatmap(x=radial, y=dist_norm, z=channel(f0, cut),
                                 colorscale="Plasma", zmin=0, zmax=1, zauto=False, zsmooth="fast",
                                 showscale=False, hoverinfo="skip"), row=map_row, col=1)
        i_cont = len(fig.data)
        fig.add_trace(go.Contour(x=radial, y=dist_norm, z=channel_enc(f0, cut),
                                 showscale=False, colorscale=[[0.0, "#7dffce"], [1.0, "#00f5a0"]],
                                 contours=contour_opts, line=dict(width=1.0), opacity=0.45,
                                 hoverinfo="skip"), row=map_row, col=1)
        fig.add_trace(go.Scatter(x=[-ap_radial_m, ap_radial_m], y=[0.0, 0.0], mode="lines",
                                 line=dict(color="white", width=5), showlegend=False,
                                 hoverinfo="skip"), row=map_row, col=1)                 # aperture line
        i_plane = len(fig.data)
        s0 = f0[sil_key]
        fig.add_trace(go.Scatter(x=[-s0, s0], y=[yy0, yy0], mode="lines",               # plane slice
                                 line=dict(color="#00e5ff", width=4), showlegend=False,
                                 hoverinfo="skip"), row=map_row, col=1)
        i_phase = len(fig.data)                                # emitter phase (bottom half), 1:1 row
        fig.add_trace(go.Heatmap(x=phase_x, y=phase_y_m, z=half_phase(f0, ap_radial_m, ap_depth_m),
                                 colorscale="Phase", zmin=-np.pi, zmax=np.pi, zauto=False,
                                 showscale=False, hoverinfo="skip"), row=phase_row, col=1)
        ox, oy = half_outline(ap_radial_m, ap_depth_m)
        fig.add_trace(go.Scatter(x=ox, y=oy, mode="lines", line=dict(color="white", width=1.2),
                                 showlegend=False, hoverinfo="skip"), row=phase_row, col=1)  # outline
        return i_heat, i_cont, i_plane, i_phase

    # Panel F (rows 3–4): major-axis (wide) cut — the SHORT aperture axis lies along the radial.
    i_fheat, i_fcont, i_fplane, i_fphase = add_pair(
        frames_phys[0], "y", "sil_major_m", ap_short, ap_long, 3, 4)
    # Panel F₂ (rows 5–6): minor-axis (narrow) cut — the LONG aperture axis lies along the radial
    # (its emitter half is the 90°-rotated view of F's).
    i_f2heat, i_f2cont, i_f2plane, i_f2phase = add_pair(
        frames_phys[0], "x", "sil_minor_m", ap_long, ap_short, 5, 6)

    ANIMATED = [i_incA, i_contA, i_capA, i_eta_m, i_cap_m, i_cum_m,
                i_fheat, i_fcont, i_fplane, i_fphase, i_f2heat, i_f2cont, i_f2plane, i_f2phase]
    frames = []
    for k, f in enumerate(frames_phys):
        cur = xs[k] / 1e3
        yy = f["chan_total"] / max_total
        frames.append(go.Frame(name=str(k), traces=ANIMATED, data=[
            go.Heatmap(z=incidence_grid(f)),
            go.Contour(x=gx, y=gy, z=encircled_grid(f)),
            go.Scatter(text=[f"{capture_series[k]:.0f}% captured"]),
            go.Scatter(x=[cur], y=[eta_series[k]]),
            go.Scatter(x=[cur], y=[capture_series[k]]),
            go.Scatter(x=[cur], y=[cum_gj[k]]),
            go.Heatmap(z=channel(f, "y")),
            go.Contour(x=radial, y=dist_norm, z=channel_enc(f, "y")),
            go.Scatter(x=[-f["sil_major_m"], f["sil_major_m"]], y=[yy, yy]),
            go.Heatmap(z=half_phase(f, ap_short, ap_long)),
            go.Heatmap(z=channel(f, "x")),
            go.Contour(x=radial, y=dist_norm, z=channel_enc(f, "x")),
            go.Scatter(x=[-f["sil_minor_m"], f["sil_minor_m"]], y=[yy, yy]),
            go.Heatmap(z=half_phase(f, ap_long, ap_short)),
        ]))
    fig.frames = frames

    # Slider steps animate to a SINGLE frame (mode immediate, redraw True) — the one path that
    # reliably redraws Heatmap/Waterfall. Plotly's native sequenced *play* only honours redraw
    # on the first frame (freezing A/E/F), so the ▶ play button is the custom sequential
    # stepper injected via post_script instead of updatemenus.
    steps = [dict(method="animate", label=f"{xs[k]/1e3:.0f}",
                  args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))]) for k in range(n_frames)]
    # Slider at the TOP (in the top margin, just under the title) alongside the ▶ play button,
    # so the controls are reachable without scrolling to the bottom of this very tall figure.
    slider = dict(active=0, y=1.0, yanchor="bottom", x=0.13, len=0.80,
                  currentvalue=dict(prefix="along-track: ", suffix=" km"), steps=steps)
    fig.update_layout(sliders=[slider])

    # Panel A: 1:1 so the spot is undistorted over the silhouette.
    fig.update_xaxes(title_text="view-plane u (m)", scaleanchor="y", scaleratio=1,
                     range=[-EXT_H, EXT_H], row=1, col=1)
    fig.update_yaxes(title_text="v (m)", range=[-EXT_V, EXT_V], row=1, col=1)
    # Panel D: FIXED ranges (x, both y) so nothing re-autoranges as the markers move.
    d_left = max(float(eta_series.max()), float(capture_series.max())) * 1.08
    fig.update_xaxes(title_text="along-track (km)", range=[xs[0] / 1e3, xs[-1] / 1e3], row=2, col=1)
    fig.update_yaxes(title_text="capture / η (%)", range=[0, d_left], secondary_y=False, row=2, col=1)
    fig.update_yaxes(title_text="cumulative energy (GJ)", range=[0, float(cum_gj[-1]) * 1.08 + 1e-9],
                     secondary_y=True, row=2, col=1)
    # Maps F/F₂ (rows 3, 5): radial x, along-path y (0 = emitter → 1 = far end of the pass).
    for r in (3, 5):
        fig.update_xaxes(title_text="radial offset (m)", range=[-R_max, R_max], row=r, col=1)
        fig.update_yaxes(title_text="emitter → plane (norm.)", range=[0, 1], row=r, col=1)
    # Emitter-phase rows (4, 6): 1:1 (scaleanchor) so the aperture is undistorted; common metres
    # scale ±PHASE_L fitting the long axis, so F's and F₂'s halves are literally the same ellipse
    # rotated 90°. scaleanchor must reference each subplot's OWN x-axis id.
    for i_ph, pr in ((i_fphase, 4), (i_f2phase, 6)):
        fig.update_xaxes(title_text="aperture (m)", range=[-PHASE_L, PHASE_L],
                         constrain="domain", row=pr, col=1)
        fig.update_yaxes(range=[-PHASE_L, 0.06 * PHASE_L], scaleanchor=fig.data[i_ph].xaxis,
                         scaleratio=1, constrain="domain", row=pr, col=1)
    fig.add_annotation(x=0.02, y=0.03, xref="x domain", yref="y domain", row=1, col=1,
                       xanchor="left", yanchor="bottom",
                       text="green contours = enclosed beam-energy fraction (centre→edge)",
                       showarrow=False, font=dict(size=9, color="#8fffd0"))
    pass_kind = "head-on pass" if abs(lateral_m) < 1.0 else f"lateral pass ({lateral_offset_km:.0f} km offset)"
    aperture_m = kw.get("emitter_diam_m") or kw.get("relay_aperture_diam_m") or 0.0
    fig.update_layout(
        template="plotly_dark", height=2100, width=900, margin=dict(t=240, l=60, r=30, b=55),
        title=dict(text=(f"Flyby animation — {arch_name}, {aperture_m:.0f} m emitter, {weather} weather, "
                         f"{pass_kind}, horizon-to-horizon (±{half/1e3:.0f} km; uniform in time — "
                         f"press ▶ play or drag the slider above)"), y=0.995, yanchor="top"),
        legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center", font=dict(size=9)))
    if write:
        os.makedirs(OUT, exist_ok=True)
        # The ▶ play button is the custom sequential stepper (see _PLAYER_JS): plotly's native
        # sequenced play only honours redraw on the first frame, freezing panels A/E/F.
        post = _PLAYER_JS.replace("{n_frames}", str(n_frames))
        fig.write_html(path, include_plotlyjs="inline", auto_play=False, post_script=post)
    return fig


# The single reference case: 5.8 GHz weatherproof low band straight from a 100 m ground
# emitter, clear air, a close 5 km lateral pass (the plane crossing near-overhead rather than
# dead-on, so the geometry stays legible). Drag the slider or press ▶ play to scrub the pass.
if __name__ == "__main__":
    build_flyby_animation(arch_name="5.8 direct", weather="clear", lateral_offset_km=5.0,
                          emitter_diam_m=100.0, path=f"{OUT}/flyby.html")
    print("wrote", f"{OUT}/flyby.html")
