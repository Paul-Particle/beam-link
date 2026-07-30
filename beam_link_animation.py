"""Flyby animation for the beam-link engine — the whole link in one moving picture.

A time slider drives six synchronized views over a single pass. Left is the 2×2 core
(A, B, C, D); the two extra plots (E, F) stack in a right-hand column:
  A (2-D)  the focused beam spot (encircled-energy contours) over the self-occluded
           silhouette (filled by incidence) — where the power lands, frame by frame.
  B (3-D)  the geometry on a SPHERICAL earth: ground arc, cruise & stratosphere
           levels, station / relay / plane, and the live beam — curvature visible.
  C (3-D)  the BWB itself, each facet coloured by how squarely the beam strikes it
           (incidence), with the beam-arrival arrow.
  D (2-D)  capture, end-to-end η and cumulative delivered energy along the pass.
  E (2-D)  the efficiency waterfall for the current instant (1.0 → η, block per stage).
  F (2-D)  the converging beam channel emitter→plane (radial exaggerated).

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

# Custom play button. Two problems ruled out the obvious approaches: (1) Plotly's native
# sequenced play only redraws Heatmap/Waterfall on the FIRST frame (the rest freeze); (2)
# stepping with `Plotly.animate([k], redraw:true)` DOES redraw them but a full redraw tears
# down and re-renders both WebGL 3-D scenes every tick → wild flicker. So this stepper applies
# each frame's data with `Plotly.restyle` on just the animated traces — an IN-PLACE update (no
# full-figure teardown, no flicker) that still redraws heatmaps/waterfall. Frame count injected
# (`{n_frames}`) since `gd.frames` isn't populated when this runs; `{plot_id}` via write_html.
_PLAYER_JS = """
var gd = document.getElementById("{plot_id}");
var N = {n_frames};
var FPS = {play_fps};                 // pass-frames shown per wall-clock second
var cur = 0, playing = false, raf = null, busy = false, startTs = null, startFrame = 0;
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
// Wall-clock playback: the frame shown is chosen from elapsed time, not incremented one per
// tick — so if a frame is slow to draw the loop SKIPS ahead and motion stays real-time and
// smooth instead of stuttering in slow-motion. `busy` prevents overlapping restyles.
function loop(ts) {
  if (!playing) return;
  if (startTs === null) { startTs = ts; startFrame = cur; }
  var target = (startFrame + Math.round((ts - startTs) / 1000 * FPS)) % N;
  if (target !== cur && !busy) { busy = true; applyFrame(target).then(function(){ busy = false; }); }
  raf = requestAnimationFrame(loop);
}
var btn = document.createElement('button');
btn.textContent = '\\u25B6 play';
btn.style.cssText = 'position:absolute;left:6px;bottom:6px;z-index:999;font:13px sans-serif;padding:3px 12px;cursor:pointer;background:#2a2a2a;color:#eee;border:1px solid #666;border-radius:3px;';
function play() { if (playing) return; playing = true; startTs = null; btn.textContent = '\\u275A\\u275A pause'; raf = requestAnimationFrame(loop); }
function pause() { playing = false; if (raf) cancelAnimationFrame(raf); btn.textContent = '\\u25B6 play'; }
btn.onclick = function(){ playing ? pause() : play(); };
var host = gd.parentNode || gd;
if (getComputedStyle(host).position === 'static') host.style.position = 'relative';
host.appendChild(btn);
gd.on('plotly_sliderchange', function(e){ if (e && e.slider) { cur = e.slider.active | 0; startTs = null; } });
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


def _beam_channel(distance_norm, radial, segments, total_shown_m, current_total_m):
    """Intensity of the CONVERGING beam over a side view: y = along-PATH distance (fraction
    of the longest total path over the pass), x = radial offset. `segments` is a list of
    (w_start, w_end, length_m, order) legs laid end to end along y — one leg for a direct
    link, TWO for a relay (ground→relay up-leg, then relay→plane down-leg). Each leg tapers
    linearly from its emitting aperture to its focal waist; intensity ∝ 1/w(s)² is normalised
    PER LEG so both the wide, faint up-leg and the focused, bright down-leg stay visible.
    NaN beyond the current plane position (no beam past the focus)."""
    dd, rr = np.meshgrid(distance_norm, radial, indexing="ij")   # (n_dist, n_radial)
    s = dd * total_shown_m
    z = np.full(s.shape, np.nan)
    cum = 0.0
    for w_start, w_end, seg_len, order in segments:
        in_seg = (s >= cum) & (s < cum + seg_len) & (s <= current_total_m)
        if seg_len > 0 and np.any(in_seg):
            frac = np.clip((s - cum) / seg_len, 0.0, 1.0)
            w = w_start + (w_end - w_start) * frac
            C = bl.supergaussian_area_factor(order)
            inten = np.where(in_seg, np.exp(-2.0 * (rr ** 2 / w ** 2) ** order) / (np.pi * w ** 2 * C), np.nan)
            mx = np.nanmax(inten)
            if mx and mx > 0:
                inten = inten / mx
            z = np.where(in_seg, inten, z)
        cum += seg_len
    return z


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

    # Panel-F beam-channel legs (end-to-end along the path). Relay = a ground→relay up-leg
    # (its ground aperture auto-sized to match the relay at that range) followed by the
    # focused relay→plane down-leg; direct = a single ground→plane leg.
    if cfg.relay_pos is not None:
        station = np.asarray(cfg.station_pos, float)
        L1 = float(bl.geometry(station, np.asarray(cfg.relay_pos, float))[0])   # ground→relay
        r_relay = cfg.relay_aperture_diam_m / 2.0
        r_up = bl.size_emitter_diameter(cfg.f_emit_relay_ghz, L1, r_relay, cfg.tau_target) / 2.0
        chan_segments = [(float(r_up), r_relay, L1, 1.0), (r_relay, w0, float(slant), order)]
        chan_total = L1 + float(slant)
        chan_relay_m, chan_relay_radius_m, chan_emit_radius_m = L1, r_relay, float(r_up)
    else:
        chan_segments = [(float(diam / 2.0), w0, float(slant), order)]
        chan_total = float(slant)
        chan_relay_m, chan_relay_radius_m, chan_emit_radius_m = None, None, float(diam / 2.0)

    plane_loc = _localize(x_along_m, lateral_offset_m, plane_alt_m, basis)
    emitter_loc = _localize(emitter[0], emitter[1], emitter[2], basis)
    return dict(field=field, w0=w0, shaped=cfg.shaped_spot, profile_order=order,
                geom=geom, eff=eff, eta=eta, elevation=float(elevation), slant_km=slant / 1e3,
                slant_m=float(slant), emitter_radius_m=float(diam / 2.0), sil_radial_m=sil_radial_m,
                chan_segments=chan_segments, chan_total=chan_total, chan_relay_m=chan_relay_m,
                chan_relay_radius_m=chan_relay_radius_m, chan_emit_radius_m=chan_emit_radius_m,
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
                          n_frames=121, size_ref_elev_deg=8.0, delivered_mw=10.0, grid_n=110,
                          grid_nA=200, emitter_diam_m=None, arch_kwargs=None, play_fps=24,
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

    # Panel-A window: cover BOTH the self-occluded silhouette AND the (possibly elongated)
    # beam spot across the whole pass, so the encircled-energy contours are never clipped
    # at the frame edge. Use a WIDE window (separate horizontal/vertical half-extents) so
    # the plot fills its full-width row instead of letterboxing to a small central square;
    # scaleanchor below keeps it 1:1, so the spot shape is not distorted. The elliptical
    # spot's major axis reaches ≈3.0·w0 at the 0.9 encircled ring (w_major = w0·√aspect,
    # aspect capped at 8), so size the horizontal extent to clear that.
    def _fld_ext(axis):
        return max((float(np.abs(f["field"][axis]).max()) if f["field"][axis].size else 0.0)
                   for f in frames_phys)
    max_w0 = max(f["w0"] for f in frames_phys)
    EXT_H = max(_fld_ext("u"), 3.2 * max_w0) * 1.1
    EXT_V = max(_fld_ext("v"), 1.7 * max_w0) * 1.1
    EXT_V = max(EXT_V, EXT_H / 2.2)                  # cap aspect so 1:1 mostly fills the wide row
    gx = np.linspace(-EXT_H, EXT_H, grid_n)          # contour grid (smooth beam)
    gy = np.linspace(-EXT_V, EXT_V, grid_n)
    gxA = np.linspace(-EXT_H, EXT_H, grid_nA)        # fine grid for the silhouette heatmap
    gyA = np.linspace(-EXT_V, EXT_V, grid_nA)

    def encircled_grid(f):
        """The beam's ENCIRCLED-ENERGY field on the fixed grid: at each cell, the fraction
        of total beam energy carried by all cells at least as bright — so contours at
        0.1…0.9 are rings enclosing that fraction of the delivered energy (centre ≈ 0,
        spilling out to 1). Drawn as an SVG contour over the silhouette heatmap."""
        beam_grid = _beam_on_grid(f["field"], f["w0"], f["shaped"], gx, gy,
                                  profile_order=f["profile_order"])
        flat = beam_grid.ravel()
        order = np.argsort(flat)[::-1]
        csum = np.cumsum(flat[order]) / max(flat.sum(), 1e-30)
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
        ss = _box_sum(ssum.reshape(grid_nA, grid_nA), 2)
        cc = _box_sum(cnt.reshape(grid_nA, grid_nA), 2)
        z = np.where(cc > 0.5, ss / np.maximum(cc, 1e-9), np.nan)
        return z.astype(np.float32)                   # float32 halves the base64 payload

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
    # Panel F (beam channel) axes — y spans the whole path (relay = up-leg + down-leg), so
    # normalise to the longest total path over the pass; the relay node sits at a fixed y.
    max_total = max(f["chan_total"] for f in frames_phys)
    R_max = max(max(f["chan_emit_radius_m"], f["chan_relay_radius_m"] or 0.0, f["w0"], f["sil_radial_m"])
                for f in frames_phys) * 1.1
    dist_norm = np.linspace(0.0, 1.0, 96)
    radial = np.linspace(-R_max, R_max, 76)
    r_t0 = frames_phys[0]["chan_emit_radius_m"]
    relay_m0 = frames_phys[0]["chan_relay_m"]
    relay_r0 = frames_phys[0]["chan_relay_radius_m"]
    BOX = 45.0                                        # fixed half-extent of panel C (m)

    def channel(f):
        return _beam_channel(dist_norm, radial, f["chan_segments"], max_total, f["chan_total"]).astype(np.float32)

    # Panel E is an efficiency WATERFALL of the current instant (cascade of blocks from
    # 1.0 down through each stage's loss to η) — most stages are constant along the pass,
    # so a live waterfall reads far better than the stacked cumulative curves did.
    wf_x = ["input"] + list(wf_labels) + ["η"]
    wf_measure = ["absolute"] + ["relative"] * len(wf_labels) + ["total"]

    def waterfall_y(k):
        cum = stage_cum[k]
        prev = np.concatenate([[1.0], cum[:-1]])
        return [1.0] + list(-(prev - cum)) + [0.0]    # start 1.0, minus each stage's loss

    # Layout: A (beam spot) and B (geometry) span the FULL width, stacked at the top and
    # given the most height — they're the two views that most reward size (the wide belly
    # silhouette; the horizon-to-horizon arc). The remaining four sit below as a 2×2 grid,
    # each larger than in the old 2×3 packing: C | D on one row, E | F on the next.
    fig = make_subplots(
        rows=4, cols=2, horizontal_spacing=0.09, vertical_spacing=0.06,
        row_heights=[0.28, 0.27, 0.225, 0.225],
        specs=[[{"type": "xy", "colspan": 2}, None],
               [{"type": "scene", "colspan": 2}, None],
               [{"type": "scene"}, {"secondary_y": True}],
               [{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=("A · beam spot over silhouette (fill = incidence)",
                        "B · geometry on a spherical earth (vertical exaggerated)",
                        "C · BWB facets by beam incidence",
                        "D · capture, η & cumulative energy along the pass",
                        "E · efficiency waterfall (this instant)",
                        "F · beam channel emitter→plane (radial exaggerated)"))

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

    # ---- Panel B (1,2) (3-D geometry): arcs, emitter/base, plane + beam -----------------
    fig.add_trace(go.Scatter3d(x=ground[:, 0], y=ground[:, 1], z=ground[:, 2], mode="lines",
                               line=dict(color="#5b4636", width=4), name="ground", hoverinfo="skip"), row=2, col=1)
    fig.add_trace(go.Scatter3d(x=cruise[:, 0], y=cruise[:, 1], z=cruise[:, 2], mode="lines",
                               line=dict(color="#9fb6cd", width=1, dash="dot"), name="11 km cruise", hoverinfo="skip"), row=2, col=1)
    fig.add_trace(go.Scatter3d(x=strato[:, 0], y=strato[:, 1], z=strato[:, 2], mode="lines",
                               line=dict(color="#c9c9ff", width=1, dash="dot"), name="20 km", hoverinfo="skip"), row=2, col=1)
    if rain is not None:
        fig.add_trace(go.Scatter3d(x=rain[:, 0], y=rain[:, 1], z=rain[:, 2], mode="lines",
                                   line=dict(color="#4a90d9", width=6), opacity=0.4,
                                   name=f"rain top ({weather})", hoverinfo="skip"), row=2, col=1)
    else:
        fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="lines",
                                   showlegend=False, hoverinfo="skip"), row=2, col=1)
    fig.add_trace(go.Scatter3d(x=[base_loc[0]], y=[base_loc[1]], z=[base_loc[2]], mode="markers",
                               marker=dict(size=5, color="#2ca02c", symbol="square"),
                               name="base station", hoverinfo="skip"), row=2, col=1)
    tether_x = [base_loc[0], emitter_loc[0]] if lofted else [None]
    tether_y = [base_loc[1], emitter_loc[1]] if lofted else [None]
    tether_z = [base_loc[2], emitter_loc[2]] if lofted else [None]
    fig.add_trace(go.Scatter3d(x=tether_x, y=tether_y, z=tether_z, mode="lines",
                               line=dict(color="#7f7f7f", width=3, dash="dot"),
                               name="tether / feed", showlegend=lofted, hoverinfo="skip"), row=2, col=1)
    fig.add_trace(go.Scatter3d(
        x=[emitter_loc[0]] if lofted else [None], y=[emitter_loc[1]] if lofted else [None],
        z=[emitter_loc[2]] if lofted else [None], mode="markers",
        marker=dict(size=6, color="#ff7f0e", symbol="diamond"),
        name=emit_name, showlegend=lofted, hoverinfo="skip"), row=2, col=1)
    p0 = frames_phys[0]["plane_loc"] / 1e3
    i_plane = len(fig.data)
    fig.add_trace(go.Scatter3d(x=[p0[0]], y=[p0[1]], z=[p0[2]], mode="markers",
                               marker=dict(size=5, color="#d62728"), name="plane", hoverinfo="skip"), row=2, col=1)
    i_beam = len(fig.data)
    fig.add_trace(go.Scatter3d(x=[emitter_loc[0], p0[0]], y=[emitter_loc[1], p0[1]],
                               z=[emitter_loc[2], p0[2]], mode="lines",
                               line=dict(color="#d62728", width=3), name="beam", hoverinfo="skip"), row=2, col=1)

    # ---- Panel E (1,3): efficiency waterfall of the current instant --------------------
    i_wf = len(fig.data)
    fig.add_trace(go.Waterfall(x=wf_x, measure=wf_measure, y=waterfall_y(0),
                               connector=dict(line=dict(color="#bbbbbb", width=1)),
                               decreasing=dict(marker=dict(color="#e45756")),
                               increasing=dict(marker=dict(color="#54a24b")),
                               totals=dict(marker=dict(color="#4c78a8")),
                               textposition="none", showlegend=False, hoverinfo="skip"),
                  row=4, col=1)

    # ---- Panel C (2,1) (3-D BWB by incidence): z negated (body frame is z-DOWN) ---------
    i_mesh = len(fig.data)
    fig.add_trace(go.Mesh3d(x=coarse_v[:, 0], y=coarse_v[:, 1], z=-coarse_v[:, 2],
                            i=coarse_f[:, 0], j=coarse_f[:, 1], k=coarse_f[:, 2],
                            intensity=frames_phys[0]["face_incidence"], intensitymode="cell",
                            colorscale="Plasma", cmin=0, cmax=1, showscale=False,
                            flatshading=True, name="BWB", hoverinfo="skip"), row=3, col=1)
    look0 = frames_phys[0]["look"]
    i_arrow = len(fig.data)
    fig.add_trace(go.Scatter3d(x=[look0[0] * 40, look0[0] * 15], y=[look0[1] * 40, look0[1] * 15],
                               z=[-look0[2] * 40, -look0[2] * 15], mode="lines",
                               line=dict(color="#d62728", width=6), name="beam→",
                               showlegend=False, hoverinfo="skip"), row=3, col=1)

    # ---- Panel D (2,2): capture, η (left %) + cumulative energy (right GJ) --------------
    fig.add_trace(go.Scatter(x=xs / 1e3, y=eta_series, mode="lines", name="η end-to-end (%)",
                             line=dict(color="#1f77b4", width=2), hoverinfo="skip"),
                  row=3, col=2, secondary_y=False)
    fig.add_trace(go.Scatter(x=xs / 1e3, y=capture_series, mode="lines", name="overlap capture (%)",
                             line=dict(color="#2ca02c", width=2, dash="dot"), hoverinfo="skip"),
                  row=3, col=2, secondary_y=False)
    fig.add_trace(go.Scatter(x=xs / 1e3, y=cum_gj, mode="lines", name="cumulative energy (GJ)",
                             line=dict(color="#ff7f0e", width=2), hoverinfo="skip"),
                  row=3, col=2, secondary_y=True)
    i_eta_m = len(fig.data)
    fig.add_trace(go.Scatter(x=[xs[0] / 1e3], y=[eta_series[0]], mode="markers",
                             marker=dict(size=9, color="#1f77b4"), showlegend=False, hoverinfo="skip"),
                  row=3, col=2, secondary_y=False)
    i_cap_m = len(fig.data)
    fig.add_trace(go.Scatter(x=[xs[0] / 1e3], y=[capture_series[0]], mode="markers",
                             marker=dict(size=9, color="#2ca02c"), showlegend=False, hoverinfo="skip"),
                  row=3, col=2, secondary_y=False)
    i_cum_m = len(fig.data)
    fig.add_trace(go.Scatter(x=[xs[0] / 1e3], y=[cum_gj[0]], mode="markers",
                             marker=dict(size=9, color="#ff7f0e"), showlegend=False, hoverinfo="skip"),
                  row=3, col=2, secondary_y=True)

    # ---- Panel F (2,3): converging beam channel; emitter + (relay) apertures + plane slice
    i_fheat = len(fig.data)
    fig.add_trace(go.Heatmap(x=radial, y=dist_norm, z=channel(frames_phys[0]),
                             colorscale="Plasma", zmin=0, zmax=1, zauto=False, zsmooth="fast",
                             showscale=False, hoverinfo="skip"), row=4, col=2)
    fig.add_trace(go.Scatter(x=[-r_t0, r_t0], y=[0.0, 0.0], mode="lines",
                             line=dict(color="white", width=5), name="emitter aperture",
                             showlegend=False, hoverinfo="skip"), row=4, col=2)
    if relay_m0 is not None:                          # relay node between the two legs
        yr = relay_m0 / max_total
        fig.add_trace(go.Scatter(x=[-relay_r0, relay_r0], y=[yr, yr], mode="lines",
                                 line=dict(color="#ffd24a", width=5), name="relay aperture",
                                 showlegend=False, hoverinfo="skip"), row=4, col=2)
        fig.add_annotation(x=0.0, y=yr, row=4, col=2, text="relay", showarrow=False,
                           yshift=8, font=dict(size=9, color="#ffd24a"))
    sil0 = frames_phys[0]["sil_radial_m"]
    i_fplane = len(fig.data)
    fig.add_trace(go.Scatter(x=[-sil0, sil0], y=[frames_phys[0]["chan_total"] / max_total] * 2, mode="lines",
                             line=dict(color="#00e5ff", width=4), name="plane",
                             showlegend=False, hoverinfo="skip"), row=4, col=2)

    ANIMATED = [i_incA, i_contA, i_capA, i_plane, i_beam, i_wf, i_mesh, i_arrow,
                i_eta_m, i_cap_m, i_cum_m, i_fheat, i_fplane]
    frames = []
    for k, f in enumerate(frames_phys):
        p = f["plane_loc"] / 1e3
        look = f["look"]
        cur = xs[k] / 1e3
        yy = f["chan_total"] / max_total
        frames.append(go.Frame(name=str(k), traces=ANIMATED, data=[
            go.Heatmap(z=incidence_grid(f)),
            go.Contour(x=gx, y=gy, z=encircled_grid(f)),
            go.Scatter(text=[f"{capture_series[k]:.0f}% captured"]),
            go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]]),
            go.Scatter3d(x=[emitter_loc[0], p[0]], y=[emitter_loc[1], p[1]], z=[emitter_loc[2], p[2]]),
            go.Waterfall(y=waterfall_y(k)),
            go.Mesh3d(intensity=f["face_incidence"], intensitymode="cell"),
            go.Scatter3d(x=[look[0] * 40, look[0] * 15], y=[look[1] * 40, look[1] * 15],
                         z=[-look[2] * 40, -look[2] * 15]),
            go.Scatter(x=[cur], y=[eta_series[k]]),
            go.Scatter(x=[cur], y=[capture_series[k]]),
            go.Scatter(x=[cur], y=[cum_gj[k]]),
            go.Heatmap(z=channel(f)),
            go.Scatter(x=[-f["sil_radial_m"], f["sil_radial_m"]], y=[yy, yy]),
        ]))
    fig.frames = frames

    # Slider steps redraw a SINGLE frame (mode immediate, redraw True) — the one path that
    # reliably redraws Heatmap/Waterfall. Plotly's NATIVE sequenced play only honours redraw
    # on the first frame of the sequence (subsequent frames skip the full redraw, freezing
    # A/E/F), so the play button is a custom stepper injected below instead of updatemenus.
    steps = [dict(method="animate", label=f"{xs[k]/1e3:.0f}",
                  args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))]) for k in range(n_frames)]
    fig.update_layout(
        sliders=[dict(active=0, y=-0.04, x=0.10, len=0.85,
                      currentvalue=dict(prefix="along-track: ", suffix=" km"), steps=steps)])

    fig.update_xaxes(title_text="view-plane u (m)", scaleanchor="y", scaleratio=1,
                     range=[-EXT_H, EXT_H], row=1, col=1)
    fig.update_yaxes(title_text="v (m)", range=[-EXT_V, EXT_V], row=1, col=1)
    fig.update_yaxes(title_text="efficiency (×)", range=[0, 1.05], row=4, col=1)
    fig.update_xaxes(tickangle=-40, tickfont=dict(size=8), row=4, col=1)
    # Panel D: FIXED ranges (x, both y) so nothing re-autoranges as the markers move —
    # otherwise the plot area jitters/scrolls slightly frame to frame.
    d_left = max(float(eta_series.max()), float(capture_series.max())) * 1.08
    fig.update_xaxes(title_text="along-track (km)", range=[xs[0] / 1e3, xs[-1] / 1e3], row=3, col=2)
    fig.update_yaxes(title_text="capture / η (%)", range=[0, d_left], secondary_y=False, row=3, col=2)
    fig.update_yaxes(title_text="cumulative energy (GJ)", range=[0, float(cum_gj[-1]) * 1.08 + 1e-9],
                     secondary_y=True, row=3, col=2)
    fig.update_xaxes(title_text="radial offset (m)", range=[-R_max, R_max], row=4, col=2)
    fig.update_yaxes(title_text="distance emitter→plane (norm.)", range=[0, 1], row=4, col=2)
    # Panel B: the model curvature is real (physics uses it); the DISPLAY exaggerates the
    # vertical so the arc is visible. Narrower box (less extreme x aspect) + closer camera
    # so the arc fills the viewport without clipping the axis labels.
    fig.update_scenes(dict(xaxis_title="downrange (km)", yaxis_title="cross (km)",
                           zaxis_title="altitude (km, exaggerated)", aspectmode="manual",
                           aspectratio=dict(x=2.0, y=0.35, z=0.9),
                           camera=dict(eye=dict(x=0.05, y=-1.3, z=0.42), up=dict(x=0, y=0, z=1))),
                      row=2, col=1)
    # Panel C: FIXED box so it never rescales as the beam arrow swings.
    fig.update_scenes(dict(xaxis=dict(range=[-BOX, BOX], title="nose (m)"),
                           yaxis=dict(range=[-BOX, BOX], title="span (m)"),
                           zaxis=dict(range=[-BOX, BOX], title="vert (m, up)"),
                           aspectmode="cube",
                           camera=dict(eye=dict(x=1.7, y=-1.7, z=1.2))), row=3, col=1)
    fig.add_annotation(x=0.5, y=1.0, xref="x domain", yref="y domain", row=1, col=1,
                       text="green contours = enclosed beam-energy fraction (centre→edge)",
                       showarrow=False, font=dict(size=9, color="#8fffd0"))
    pass_kind = "head-on pass" if abs(lateral_m) < 1.0 else f"lateral pass ({lateral_offset_km:.0f} km offset)"
    aperture_m = kw.get("emitter_diam_m") or kw.get("relay_aperture_diam_m") or 0.0
    fig.update_layout(
        template="plotly_dark", height=2100, width=1140,
        title=(f"Flyby animation — {arch_name}, {aperture_m:.0f} m emitter, {weather} weather, {pass_kind}, "
               f"horizon-to-horizon (±{half/1e3:.0f} km; smooth in time — drag the slider or press play)"),
        legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center", font=dict(size=9)))
    if write:
        os.makedirs(OUT, exist_ok=True)
        fig.write_html(path, include_plotlyjs="inline", auto_play=False,
                       post_script=(_PLAYER_JS.replace("{n_frames}", str(n_frames))
                                    .replace("{play_fps}", str(play_fps))))
    return fig


# Relay vs direct at a common, meaningful emitter aperture (100 m everywhere), normal weather,
# lateral pass — the head-to-head with the emitter no longer confounding the comparison.
STUDY_WEATHER = "clear"
STUDY_LATERAL_KM = 60.0
EMITTER_M = 100.0
DIRECT_ARCH = "5.8 direct"
RELAY_ARCH = "relay ×3 mult 5.8→17.4"

# (label, arch_name, emitter_diam_m, tag).
STUDY_CASES = [
    ("Relay ×3 — 100 m aperture", RELAY_ARCH, EMITTER_M, "relay_100m"),
    ("5.8 direct — 100 m emitter", DIRECT_ARCH, EMITTER_M, "direct_100m"),
]


if __name__ == "__main__":
    for _label, arch, diam, tag in STUDY_CASES:
        build_flyby_animation(arch_name=arch, weather=STUDY_WEATHER, lateral_offset_km=STUDY_LATERAL_KM,
                              emitter_diam_m=diam, path=f"{OUT}/beam_link_flyby_animation_{tag}.html")
        print("wrote", f"{OUT}/beam_link_flyby_animation_{tag}.html")
