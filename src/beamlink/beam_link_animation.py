"""Flyby animation for the beam-link engine — the whole link in one moving picture.

A time slider drives synchronized views over a single pass. A and D span the full width; the
two beam-channel rows read left→right, each with its emitter-phase plot in a narrow column to
the left (sharing the map's vertical radial axis, so aperture and beam line up):
  A (2-D)  the focused beam spot (encircled-energy contours) over the self-occluded
           silhouette (filled by incidence) — where the power lands, frame by frame.
  D (2-D)  capture, end-to-end η and cumulative delivered energy along the pass.
  F  (2-D) the converging beam channel emitter→plane, MAJOR-axis (wide) cut (radial
           exaggerated), with the emitter phase to its left — the SHORT aperture axis lies
           along the radial here, so the aperture shows as a sliver (long axis cropped off).
  F₂ (2-D) the same channel, MINOR-axis (narrow) cut — here the LONG aperture axis lies along
           the radial, so the emitter phase fills the height (the 90°-rotated view of F's).

The point is to spot inconsistencies at a glance (does the spot overspill? does the far
wing go dark? does the aperture phase tighten toward the waist?) far faster than tables.

This module is the PICTURE: layout, colour, contour and PNG baking, and the stepper JS. The
numbers it draws come from `flyby_frames` — one call per frame, no physics here.

Run:  uv run python -m beamlink.beam_link_animation   → out/flyby.html
(the single reference case — see the __main__ block; call build_flyby_animation() directly
for other architectures, apertures, weathers or pass geometries.)
"""
import os
import io
import base64

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib as mpl
import matplotlib.image as mpimg
import contourpy

from beamlink import beam_link as bl
from beamlink import diffraction as dfr
from beamlink import bwb_model as m
from beamlink import geo
from beamlink.architectures import standard_architectures
from beamlink.flyby_frames import (channel_field, frame_physics, funnel_contour,
                                   max_silhouette_extent, size_emitter_far)

OUT = "out"


def _build_stamp():
    """Short git commit + dirty flag + timestamp, stamped on every HTML so we can be sure which
    build a figure came from when discussing it."""
    import subprocess, datetime
    here = os.path.dirname(os.path.abspath(__file__)) or "."
    try:
        h = subprocess.run(["git", "-C", here, "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True).stdout.strip() or "?"
        dirty = subprocess.run(["git", "-C", here, "status", "--porcelain"],
                               capture_output=True, text=True).stdout.strip()
        h += "+dirty" if dirty else ""
    except Exception:
        h = "?"
    return f"{h} · {datetime.datetime.now():%Y-%m-%d %H:%M}"


# Play button: a custom sequential stepper. Each frame swaps the five baked background images
# (one `Plotly.relayout`) and restyles the handful of live vector traces (contours, plane slices,
# curve markers) — no z-grid re-raster and no per-frame contour recompute, so playback stays
# smooth. Plotly's native sequenced play is unreliable here (it only redraws some traces past the
# first frame), so this advances ONE frame per tick in-place instead. `{n_frames}` is injected
# (gd.frames isn't populated when this runs); `{plot_id}` is substituted by write_html.
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
  if (fr.layout && fr.layout.images) { proms.push(Plotly.relayout(gd, {images: fr.layout.images})); }
  cur = k;
  return Promise.all(proms);
}
function loop() { if (!playing) return; applyFrame((cur + 1) % N).then(function(){ if (playing) timer = setTimeout(loop, 8); }); }
var btn = document.createElement('button');
btn.textContent = '\\u25B6 play';
btn.style.cssText = 'position:absolute;left:6px;top:{btn_top}px;z-index:999;font:13px sans-serif;padding:3px 12px;cursor:pointer;background:#2a2a2a;color:#eee;border:1px solid #666;border-radius:3px;';
function play() { if (playing) return; playing = true; btn.textContent = '\\u275A\\u275A pause'; loop(); }
function pause() { playing = false; if (timer) clearTimeout(timer); btn.textContent = '\\u25B6 play'; }
btn.onclick = function(){ playing ? pause() : play(); };
var host = gd.parentNode || gd;
if (getComputedStyle(host).position === 'static') host.style.position = 'relative';
host.appendChild(btn);
gd.on('plotly_sliderchange', function(e){ if (e && e.slider) cur = e.slider.active | 0; });
"""


# Beam maps (F/F₂): three enclosed-energy rings. The outer ring is 0.8, NOT 0.9 — the 0.9 contour
# rides the physical (but visually noisy) sidelobe skirt, so we stop just inside it.
_BEAM_LEVELS = (0.1, 0.5, 0.8)
# Panel A (spot over silhouette): the same three as SOLID rings, plus 0.3/0.7 as DOTTED in-betweens.
_SIL_SOLID = (0.1, 0.5, 0.8)
_SIL_DOTTED = (0.3, 0.7)

# Beam-map overlay line weight: the emitter half-aperture outline, the white aperture (z=0) line and
# the plane slice line ALL share this one thin weight (matched to the ellipse border) and are all
# white, so aperture + plane + ellipse read as a single coherent frame rather than chunky markers.
_MAP_EDGE_W = 1.0


def _field_to_png(field01, transpose=False, flip_rows=False, cmap="plasma"):
    """A 2-D field already scaled to [0, 1] → a base64 PNG data-URI: ONE pixel per sample, with
    NO interpolation or anti-aliasing (a direct colour-mapped pixel dump). Any smoothness must
    come from dense sampling, never from smoothing — so the picture stays a faithful display of
    the computed field. NaN samples become fully transparent (the dark page shows through).
    `transpose`/`flip_rows` orient the array so image row 0 lands at the top of the panel.
    `cmap` is any matplotlib colormap — `plasma` (sequential) for intensity magnitudes, a CYCLIC
    map like `twilight` for phase, whose endpoints match so the 0↔2π wrap has no false seam."""
    a = field01.T if transpose else field01
    if flip_rows:
        a = a[::-1]
    rgba = mpl.colormaps[cmap](np.clip(np.nan_to_num(a, nan=0.0), 0.0, 1.0))
    rgba[..., 3] = np.where(np.isfinite(a), 1.0, 0.0)
    buf = io.BytesIO()
    mpimg.imsave(buf, rgba, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# --- the animation -----------------------------------------------------------

def build_flyby_animation(arch_name="5.8 direct", weather="clear",
                          plane_alt_m=11_000.0, lateral_offset_km=0.0, min_elevation_deg=2.0,
                          n_frames=121, size_ref_elev_deg=8.0, delivered_mw=10.0,
                          grid_nA=200, emitter_diam_m=None, arch_kwargs=None,
                          orientation="landscape", phone=False, beam_full_range=False,
                          write=True, path=f"{OUT}/beam_link_flyby_animation.html"):
    """Build (and, if `write`, save) the flyby animation (A, D, F, F₂), returning the figure.

    `arch_name` selects a named case from `standard_architectures`; pass `arch_kwargs`
    (a LinkConfig kwarg dict, `plane_pos` omitted) instead to animate an architecture not
    in that catalogue (e.g. a passive same-frequency relay), with `arch_name` used only as
    the display label. The default is the DIRECT case, which is the one being iterated on;
    a RELAY is an explicit opt-in (`arch_name="relay ×3 mult 5.8→17.4"`) and renders the
    channel as two legs, ground→relay then relay→plane, each with its own aperture, focus
    and wavelength. `emitter_diam_m` sets the emitter aperture EXPLICITLY (direct →
    `emitter_diam_m`, relay → `relay_aperture_diam_m`) and bypasses the far-reference
    auto-sizing — so emitter size can be swept as the single variable. `grid_nA` is the
    panel-A silhouette-grid resolution (lower it for lighter embedded figures).

    `orientation`: "landscape" (desktop — beam maps read left→right with the emitter phase
    in a column to their LEFT) or "portrait" (the same views rotated 90° — beam maps read
    bottom→top with the emitter phase directly BELOW; a single column that fits a phone).
    `phone=True` selects the portrait layout at a phone-width size with a black background."""
    if phone:
        orientation = "portrait"
    portrait = (orientation == "portrait")
    if arch_kwargs is None:
        archs = standard_architectures(weather=weather)
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
        kw = size_emitter_far(kw, size_ref_elev_deg, plane_alt_m)
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
    frames_phys = [frame_physics(kw, x, plane_alt_m, basis, coarse_mesh,
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
    gxA = np.linspace(-EXT_H, EXT_H, grid_nA)        # fine grid for the z-buffered silhouette image
    gyA = np.linspace(-EXT_V, EXT_V, grid_nA)
    # Panel-A silhouette mesh = the ENGINE's own capture mesh (`get_mesh`), so the displayed outline is
    # exactly the surface the capture integral sums over — no finer, no coarser. Normals are per-face
    # (flat incidence per pixel).
    silA_v, silA_f, silA_n = m.get_mesh()
    # PER-FRAME airframe projection into each beam-cut plane, drawn at the plane line. Unlike a fixed
    # top/side view, this reflects the aircraft's TRUE orientation relative to the cut we are viewing at
    # that instant (and would follow a banking manoeuvre if modelled), and it is coloured by BEAM
    # incidence (n̂·look, exactly like panel A). Each beam map is a cut of the 3-D beam containing the
    # propagation axis and one transverse axis: F = MAJOR transverse (radial ← major, along ← beam
    # propagation, VIEWED along the minor transverse), F₂ = MINOR transverse (radial ← minor, viewed
    # along the major). The transverse axes are the silhouette's principal axes in the panel-A view
    # plane, so the projection foreshortens/rotates with the geometry. Uses the coarse channel mesh
    # (fast; the projection is a small on-screen element). `cut` = 'y' (F, major) / 'x' (F₂, minor).
    proj_v, proj_f, proj_n = coarse_mesh
    _blank_png = _field_to_png(np.full((2, 2), np.nan))
    _proj_cache = {}

    def _plane_proj(f, cut):
        """(png, radial_half_m, along_half_m) for the plane's projection into the F ('y') / F₂ ('x')
        beam-cut plane at frame `f`, coloured by beam incidence. Cached per (frame, cut)."""
        key = (id(f), cut)
        if key not in _proj_cache:
            look = np.asarray(f["look"], float); look = look / (np.linalg.norm(look) + 1e-30)
            vu, vv = np.asarray(f["view_u"], float), np.asarray(f["view_v"], float)
            coords = np.vstack([f["field"]["u"], f["field"]["v"]])
            if coords.shape[1] >= 3:
                _, ev = np.linalg.eigh(np.cov(coords))    # cols [minor, major] in the (view_u,view_v) plane
                major = ev[0, 1] * vu + ev[1, 1] * vv
                minor = ev[0, 0] * vu + ev[1, 0] * vv
            else:
                major, minor = vu, vv
            radial_ax, view_ax = (major, minor) if cut == "y" else (minor, major)
            beam_inc = np.clip(proj_n @ look, 0.0, 1.0)   # COLOUR: beam incidence (like panel A)
            view_inc = proj_n @ view_ax                    # CULL/occlude by the viewing direction
            front = view_inc > 1e-6
            if not front.any():
                return _blank_png, 0.5, 0.5
            fv = np.unique(proj_f[front])
            R, A = proj_v @ radial_ax, proj_v @ look       # radial (transverse) & along (propagation)
            r_half = max(0.5 * (R[fv].max() - R[fv].min()) * 1.05, 0.3)
            a_half = max(0.5 * (A[fv].max() - A[fv].min()) * 1.05, 0.3)
            r_mid, a_mid = 0.5 * (R[fv].min() + R[fv].max()), 0.5 * (A[fv].min() + A[fv].max())
            r_lin = np.linspace(r_mid - r_half, r_mid + r_half, 170)
            a_lin = np.linspace(a_mid - a_half, a_mid + a_half, 170)
            incid = m.rasterize_view_zbuffer(radial_ax, look, view_ax, proj_v, proj_f, view_inc,
                                             r_lin, a_lin, color=beam_inc)
            png = _field_to_png(incid, transpose=not portrait, flip_rows=True, cmap="plasma")
            _proj_cache[key] = (png, float(r_half), float(a_half))
        return _proj_cache[key]

    _incidA_cache = {}

    def incidence_grid(f):
        """The silhouette's incidence on the panel-A grid (gxA×gyA), z-BUFFERED over the ENGINE's
        capture mesh: per pixel, the incidence cosine (n̂·ŵ) of the nearest visible triangle, NaN where
        uncovered. Deterministic, self-occluded outline — the baked panel-A background image."""
        key = id(f)
        if key not in _incidA_cache:
            _incidA_cache[key] = m.rasterize_view_zbuffer(f["view_u"], f["view_v"], f["view_w"],
                                                          silA_v, silA_f, silA_n @ f["view_w"], gxA, gyA)
        return _incidA_cache[key]

    half = float(xs.max())

    # Panel-D curves are sampled MUCH more finely than the animation frames. The frames are limited
    # by the per-frame diffraction + image-baking cost, but each point of the η/capture/energy curve
    # is only a scalar link evaluation (no field baked), so a dense grid is cheap — and it removes
    # the visible kinks (notably in the delivered-energy curve) from drawing a smooth quantity as a
    # coarse polyline. One vectorized evaluate() maps the scalar core over the whole dense grid.
    n_curve = max(n_frames, 400)
    xs_curve = np.linspace(-half, half, n_curve)
    plane_arr = np.column_stack([xs_curve, np.full(n_curve, lateral_m), np.full(n_curve, plane_alt_m)])
    s_curve = bl.evaluate(bl.LinkConfig(plane_pos=plane_arr, **kw), delivered_mw)
    eta_series = np.asarray(s_curve["eta_end_to_end"], float) * 100
    _geom = s_curve.get("capture_geom", s_curve.get("capture_down_geom", s_curve["capture"]))
    capture_series = np.asarray(_geom, float) * 100                     # intensity-weighted overlap
    # Cumulative delivered energy over the pass ("eat everything": emitter at a fixed grid
    # power sized to hit delivered_mw at the best point; plane banks grid·η each instant).
    peak = max(eta_series.max() / 100.0, 1e-9)
    dt_s = (xs_curve[1] - xs_curve[0]) / 250.0                          # 250 m/s cruise
    deliverable_mw = (delivered_mw / peak) * (eta_series / 100.0)
    cum_gj = np.cumsum(deliverable_mw * dt_s) / 1e3                      # MW·s = MJ → GJ
    # "Windowed pass efficiency": if the beam only operates while the plane is within ±d of closest
    # approach, this is total delivered energy ÷ total input energy over that window. Input power is
    # fixed (constant grid power), so it reduces to the mean of η over |along-track| ≤ d. At d = 0 it
    # equals peak η; it falls as the window widens to swallow low-elevation (inefficient) geometry —
    # i.e. "how good is my average efficiency if I'm willing to beam out to this distance?".
    _order = np.argsort(np.abs(xs_curve))                              # points from closest approach out
    _cummean = np.cumsum((eta_series / 100.0)[_order]) / np.arange(1, n_curve + 1)
    windowed_eta = np.empty_like(eta_series)
    windowed_eta[_order] = _cummean * 100.0                             # mean η over |x| ≤ |x_k|, in %
    # Companion (same ±d window): TOTAL energy delivered if the beam only operates within ±d —
    # the delivered energy summed over |x| ≤ d. Rises monotonically to the full-pass total. Reads
    # "how much energy do I actually land if I'm willing to beam out to this distance?".
    windowed_gj = np.empty_like(eta_series)
    windowed_gj[_order] = np.cumsum((deliverable_mw * dt_s)[_order]) / 1e3   # GJ within |x| ≤ |x_k|
    # Frame-resolution values for the moving markers + panel-A capture text: sample the dense curves
    # at each frame's along-track position, so a marker always rides exactly on its (smooth) line.
    eta_mk = np.interp(xs, xs_curve, eta_series)
    cap_mk = np.interp(xs, xs_curve, capture_series)
    cum_mk = np.interp(xs, xs_curve, cum_gj)
    win_eta_mk = np.interp(xs, xs_curve, windowed_eta)
    win_gj_mk = np.interp(xs, xs_curve, windowed_gj)
    # Panel-A "% captured" = the ENGINE's geometric capture (`f["geom"]`) — the very number panel D and
    # the econ use. The displayed silhouette is rasterized from the same mesh the engine integrates, so
    # the picture IS a faithful readout of what this number sums over (integration cells = shown pixels),
    # with no parallel on-grid re-integration to drift from it.
    cap_pct_frame = np.array([f["geom"] for f in frames_phys]) * 100
    # Panels F/F₂ (beam channel): the along-path axis is normalised to the LONGEST path over the
    # pass, so the plane's position along it encodes actual slant (far = near the far end, close
    # = near the emitter) — that distance cue is what keeps F/F₂ from being redundant with A.
    # Radial extent anchored to the EMITTER aperture and the silhouette (both bounded), NOT to
    # the beam waist: far out the defocused elliptical major waist is hundreds of m, which would
    # shrink the emitter and all near-field detail to a blob. A beam wider than this window
    # simply fills the panel with no contours — the correct read for a badly overspilling beam.
    max_total = max(f["chan_total"] for f in frames_phys)
    # ONE shared RADIAL scale for both channel cuts AND their emitter overlays: fit the LARGEST
    # emitter dimension (the long aperture semi-axis) over all frames, + a small margin. So the
    # emitter aperture and the white aperture-line on each beam map read at exactly the same scale
    # and line up. BOTH panels draw a HALF aperture cut along the RADIAL axis (cut at cross = 0, on
    # the map side; arc bulging to negative cross), pre-distorted to render isotropically (_iso_lo).
    _ap = [(f["chan_legs"][-1]["ap_major"], f["chan_legs"][-1]["ap_minor"]) for f in frames_phys]
    ap_short = min(min(a) for a in _ap)
    ap_long = max(max(a) for a in _ap)
    PHASE_L = 1.1 * ap_long                    # shared radial half-extent (m), fits the long axis
    # Dense sampling so the baked map images are smooth WITHOUT interpolation: radial is
    # interpolation-only (free), along-path is one diffraction pass each (linear cost, kept
    # moderate — the propagation envelope varies slowly). Along-path is EVALUATED on a grid with
    # EXTRA DENSITY near the aperture (`dist_norm`) — the near field converges fast and the geometric
    # near-field fill lives there — while the IMAGE is baked on a UNIFORM grid (`dist_disp`) so the
    # along/distance axis stays physically linear (regular km ticks); the beam is resampled from the
    # dense evaluation onto the uniform display grid. Funnel contours carry their own z, so they use
    # the dense grid directly (no resample).
    dist_norm = np.unique(np.concatenate([
        np.linspace(0.0, 0.14, 150),           # extra near-aperture density (evaluation/integration)
        np.linspace(0.0, 1.0, 300)]))          # regular coverage over the whole path
    dist_disp = np.linspace(0.0, 1.0, 360)     # UNIFORM grid for the baked image (linear distance axis)
    radial = np.linspace(-PHASE_L, PHASE_L, 600)  # radial samples — cheap (interp only)
    r_t0 = frames_phys[0]["chan_emit_radius_m"]

    def _resample_along(field2d):
        """Resample a (len(dist_norm), n_radial) field from the dense (near-aperture-heavy) evaluation
        grid onto the UNIFORM display grid `dist_disp`, per radial column (both grids ascending). NaN
        (beyond the current plane) is preserved so the mask stays intact."""
        idx = np.clip(np.searchsorted(dist_norm, dist_disp, side="right") - 1, 0, dist_norm.size - 2)
        x0, x1 = dist_norm[idx], dist_norm[idx + 1]
        w = np.where(x1 > x0, (dist_disp - x0) / (x1 - x0), 0.0)[:, None]
        lo, hi = field2d[idx], field2d[idx + 1]
        out = lo * (1.0 - w) + hi * w
        return np.where(np.isnan(lo) | np.isnan(hi), np.nan, out)

    # The along-path axis spans max_total, so the sample SPACING (Δz = max_total/(N−1)) is CONSTANT
    # across frames and the sample count effectively grows with slant (near/overhead frames fill only
    # a short low band, far frames the whole axis). Intensity self-normalises per frame so the beam
    # shape is always visible; the encircled-energy contours are absolute fractions, so normalisation
    # is moot for reading "how much lands on the plane".
    #   beam_full_range=False (default): the beam is masked beyond the current plane position, so the
    #     map shows only emitter→plane. True: render the field over the WHOLE path (out to max_total),
    #     so the energy that MISSES the plane — sidelobes + the diverging beam BEHIND the aircraft —
    #     is visible for an exposure/limits check (the plane line still marks the aircraft position).
    _chan_cache = {}

    def _channel_all(f):                              # (ix, ex, iy, ey) — both cuts, one pass
        key = id(f)
        if key not in _chan_cache:
            shown_to = max_total if beam_full_range else f["chan_total"]
            _chan_cache[key] = channel_field(dist_norm, radial, f["chan_legs"], max_total, shown_to)
        return _chan_cache[key]

    def channel(f, axis):                             # axis 'y' = major (wide), 'x' = minor (narrow)
        ix, ex, iy, ey = _channel_all(f)
        return iy if axis == "y" else ix

    def channel_enc(f, axis):
        ix, ex, iy, ey = _channel_all(f)
        return ey if axis == "y" else ex

    # Emitter phase, drawn INTO each beam map's plot (in the along < 0 region), sharing the map's
    # radial axis so the aperture's straight cut lands on the map's white aperture-line. The image is
    # rendered over its OWN cross×radial metre grid (cross ∈ [−ap_cross, 0], radial ∈ ±ap_radial), then
    # placed into the isotropically pre-distorted rect (see _iso_lo). The two panels show the SAME
    # elliptical aperture rotated 90°: F's radial axis is Fourier-conjugate to the SHORT aperture axis
    # (long axis along cross), F₂'s to the LONG axis (short axis along cross). CYCLIC `twilight`
    # colormap (phase wraps 0↔2π, so a sequential map puts a false bright→dark cliff at the seam).
    # Rings are the symmetric zone plate φ = wrap(−(2π/λ)(√(f²+ρ²) − f)), tightening as focus nears.
    def emitter_phase(f, ap_radial_m, ap_cross_m, cross_lin):   # zone plate over the aperture (m grid)
        rad_lin = np.linspace(-ap_radial_m, ap_radial_m, 400)
        cross, rad = np.meshgrid(cross_lin, rad_lin)   # cross = x, rad = y
        inside = (rad / ap_radial_m) ** 2 + (cross / ap_cross_m) ** 2 <= 1.0
        lam = float(f["chan_legs"][-1]["lam"]); foc = float(f["chan_legs"][-1]["focus_m"])
        phi = -(2.0 * np.pi / lam) * (np.sqrt(foc ** 2 + cross ** 2 + rad ** 2) - foc)
        phi = (phi + np.pi) % (2.0 * np.pi) - np.pi
        return np.where(inside, phi, np.nan).astype(np.float32)

    def emitter_outline(ap_radial_m, ap_cross_m):      # thin white aperture edge; returns (cross, radial)
        # Half aperture (both panels): D-shape with the straight cut along the RADIAL axis at cross=0
        # (so it coincides with the map's white aperture-line) and the arc bulging to NEGATIVE cross.
        t = np.linspace(-np.pi / 2.0, np.pi / 2.0, 200)
        return (np.append(-ap_cross_m * np.cos(t), 0.0),      # 0 → −ap_cross → 0, then close the cut
                np.append(ap_radial_m * np.sin(t), -ap_radial_m))

    # Layout: A (beam spot over silhouette) and D (along-pass curves) come first, then the two
    # beam-channel maps each with its emitter-phase plot sharing the RADIAL axis (so the aperture
    # and the map's white aperture-line read at one scale and line up).
    #   landscape — A/D span the full width; each map reads left→right (x = emitter→plane) with
    #               its emitter phase in a narrow column to its LEFT.
    #   portrait  — a single column; each map reads bottom→top (y = emitter→plane) with its
    #               emitter phase directly BELOW (the 90°-rotated view; fits a phone).
    # Layout: four stacked panels (A, D, F, F₂). The emitter phase is drawn INTO each beam map's own
    # plot, in the region just before the emitter end (along < 0), rather than as a separate subplot —
    # so the aperture's straight cut lands exactly on the map's white aperture-line (touch + align) and
    # the shared radial axis needs no reconciliation. Because that plot's two axes carry DIFFERENT
    # units (along = normalised emitter→plane distance, radial = metres), the aperture's cross extent
    # is pre-distorted (see _iso_lo) so the half-ellipse still renders ISOTROPICALLY on screen.
    if portrait:
        FIG_W, FIG_H, MARG = 402, 1750, dict(t=150, l=54, r=10, b=40)
        f_title = "F · beam channel · top view · wide cut"
        f2_title = "F₂ · beam channel · side view · narrow cut"
        row_heights, vspace = [0.18, 0.24, 0.29, 0.29], 0.05
    else:
        FIG_W, FIG_H, MARG = 1150, 1550, dict(t=140, l=66, r=25, b=55)
        f_title = "F · beam channel, top view — major-axis (wide) cut, emitter phase at left"
        f2_title = "F₂ · beam channel, side view — minor-axis (narrow) cut, emitter phase at left"
        row_heights, vspace = [0.22, 0.24, 0.27, 0.27], 0.075
    fig = make_subplots(
        rows=4, cols=1, vertical_spacing=vspace, row_heights=row_heights,
        specs=[[{"type": "xy"}], [{"secondary_y": True}], [{"type": "xy"}], [{"type": "xy"}]],
        subplot_titles=("A · beam spot over silhouette (fill = incidence)",
                        "D · capture, η & cumulative energy along the pass", f_title, f2_title))
    if portrait:
        for ann in fig.layout.annotations:
            ann.font.size = 12
    A_rc, D_rc, F_rc, F2_rc = (1, 1), (2, 1), (3, 1), (4, 1)
    inner_w, inner_h = FIG_W - MARG["l"] - MARG["r"], FIG_H - MARG["t"] - MARG["b"]

    # ---- offline renderers: heatmaps → honest no-AA PNG backgrounds; contours → vector lines --
    def map_png(f, cut):                                   # beam intensity → background image
        # Resample the dense (near-aperture-heavy) evaluation onto the uniform display grid so the
        # baked image has a physically-linear distance axis.
        return _field_to_png(_resample_along(channel(f, cut)), transpose=not portrait, flip_rows=portrait)
    def map_contour(f, cut):                               # encircled-energy funnel (per-plane inversion)
        # The enclosed field is integrated on a common radial grid (see flyby_frames.channel_field),
        # so it is already smooth along the path; the surviving waviness is real Fresnel structure.
        r_vals, z_vals = funnel_contour(channel_enc(f, cut), dist_norm, radial, _BEAM_LEVELS)  # r_L(z)=E⁻¹
        return (r_vals, z_vals) if portrait else (z_vals, r_vals)     # portrait x=radial; landscape x=dist
    def phase_png(f, ar, ac):                              # emitter zone-plate → background image
        cross_lin = np.linspace(-ac, 0.0, 220)             # cross ∈ [−ap_cross, 0] (cut at 0, map side)
        # portrait: transpose so cross→y, and flip so the cut (cross=0) is the TOP row (abuts the map
        # above). landscape: cross→x already; radial is symmetric so row order is irrelevant.
        return _field_to_png((emitter_phase(f, ar, ac, cross_lin) + np.pi) / (2.0 * np.pi),
                             transpose=portrait, flip_rows=portrait, cmap="twilight")
    def incid_png(f):                                      # silhouette incidence → background image
        return _field_to_png(incidence_grid(f), transpose=False, flip_rows=True)
    _encA_cache = {}

    def _panelA_enc(f):
        """High-res encircled-energy field for panel A, cached per frame. The focal intensity is
        evaluated by a direct separable ZOOM-DFT (`focal_intensity_grid`) on a window sized to CONTAIN
        the beam (main lobe + several sidelobe rings) at high resolution: the ±λz/d single-FFT window
        undersamples the tiny spot (jagged contours) and a panel-sized window truncates a spread beam
        (edge-clip loops). Returns (enc[ix,iy], x_axis, y_axis, evecs); evecs = display principal axes
        [minor, major] = [x_axis, y_axis] directions."""
        key = id(f)
        if key not in _encA_cache:
            leg = f["chan_legs"][-1]
            z, foc, lam = f["slant_m"], leg["focus_m"], leg["lam"]
            apmaj, apmin = leg["ap_major"], leg["ap_minor"]
            defocus = abs(1.0 - z / max(foc, 1e-9))
            wx = 6.0 * lam * z / max(apmaj, 1e-3) + 1.5 * apmaj * defocus   # narrow (minor) spot dir
            wy = 6.0 * lam * z / max(apmin, 1e-3) + 1.5 * apmin * defocus   # wide (major) spot dir
            n = 640
            xa = np.linspace(-wx, wx, n)
            ya = np.linspace(-wy, wy, n)
            I = dfr.focal_intensity_grid(z, foc, apmaj, apmin, lam, xa, ya, taper_p=leg["taper_p"])
            flat = I.ravel(); order = np.argsort(flat)[::-1]
            enc = np.empty_like(flat)
            enc[order] = np.clip(np.cumsum(flat[order]) * (xa[1] - xa[0]) * (ya[1] - ya[0]), 0.0, 1.0)
            coords = np.vstack([f["field"]["u"], f["field"]["v"]])
            _, evecs = np.linalg.eigh(np.cov(coords))
            _encA_cache[key] = (enc.reshape(I.shape), xa, ya, evecs)
        return _encA_cache[key]

    def incid_contour(f, levels):
        """Encircled-energy contours for panel A: iso-levels of the cumulative-energy field on the
        beam's NATIVE (recomputed, finer) diffraction grid, transformed into the display (u, v) frame.
        The native grid contains the whole beam and decays smoothly — NOT the clipped panel window,
        whose edge-truncation of a spread beam produced the spurious hooks/loops. Each level encloses
        that fraction of the beam's TOTAL radiated energy (∫∫ I dA = 1)."""
        fld = f["field"]
        if fld["u"].size < 3 or "intensity" not in fld:
            return np.array([]), np.array([])
        enc, xa, ya, evecs = _panelA_enc(f)
        gen = contourpy.contour_generator(xa, ya, enc.T)  # z[row=ya, col=xa]; segs in (x_axis, y_axis)
        uo, vo = [], []
        for level in levels:
            for seg in gen.lines(float(level)):
                uv = evecs @ np.vstack([seg[:, 0], seg[:, 1]])   # (x_axis, y_axis) → display (u, v)
                uo.extend(uv[0]); uo.append(np.nan)
                vo.extend(uv[1]); vo.append(np.nan)
        return np.asarray(uo), np.asarray(vo)

    # ---- Panel A: incidence image (added to layout.images below) + encircled contours (vector) --
    # Mint-green reads cleanly over Plasma (which has no green). The heatmap itself is a baked
    # background image; only the two contour traces (solid 0.1/0.5/0.8, dotted 0.3/0.7) and the
    # capture text are live vector traces here.
    i_contA = len(fig.data)
    cAx, cAy = incid_contour(frames_phys[0], _SIL_SOLID)
    fig.add_trace(go.Scatter(x=cAx, y=cAy, mode="lines", line=dict(color="#7dffce", width=1.1),
                             opacity=0.4, showlegend=False, hoverinfo="skip"), row=A_rc[0], col=A_rc[1])
    i_contA_dot = len(fig.data)
    dAx, dAy = incid_contour(frames_phys[0], _SIL_DOTTED)
    fig.add_trace(go.Scatter(x=dAx, y=dAy, mode="lines", line=dict(color="#7dffce", width=1.0, dash="dot"),
                             opacity=0.3, showlegend=False, hoverinfo="skip"), row=A_rc[0], col=A_rc[1])
    # Live capture % in the corner of panel A — the beam fraction the silhouette intercepts.
    i_capA = len(fig.data)
    fig.add_trace(go.Scatter(x=[-EXT_H * 0.94], y=[EXT_V * 0.82], mode="text",
                             text=[f"{cap_pct_frame[0]:.0f}% captured"], textposition="middle right",
                             textfont=dict(size=16, color="#ffffff"), cliponaxis=False,
                             showlegend=False, hoverinfo="skip"), row=A_rc[0], col=A_rc[1])

    # ---- Panel D: capture, η (left %) + cumulative energy (right GJ) --------------------------
    # Short legend labels on the phone so the horizontal legend fits one line at 402 px width.
    nm_eta = "η now (%)" if portrait else "η end-to-end, instantaneous (%)"
    nm_cap = "capture (%)" if portrait else "overlap capture (%)"
    nm_cum = "energy (GJ)" if portrait else "cumulative energy (GJ)"
    nm_win = "pass η, ±d (%)" if portrait else "pass η if beaming within ±d (%)"
    nm_weg = "energy, ±d (GJ)" if portrait else "energy delivered if beaming within ±d (GJ)"
    fig.add_trace(go.Scatter(x=xs_curve / 1e3, y=eta_series, mode="lines", name=nm_eta,
                             line=dict(color="#1f77b4", width=2), hoverinfo="skip"),
                  row=D_rc[0], col=D_rc[1], secondary_y=False)
    fig.add_trace(go.Scatter(x=xs_curve / 1e3, y=windowed_eta, mode="lines", name=nm_win,
                             line=dict(color="#9467bd", width=2, dash="dash"), hoverinfo="skip"),
                  row=D_rc[0], col=D_rc[1], secondary_y=False)
    fig.add_trace(go.Scatter(x=xs_curve / 1e3, y=capture_series, mode="lines", name=nm_cap,
                             line=dict(color="#2ca02c", width=2, dash="dot"), hoverinfo="skip"),
                  row=D_rc[0], col=D_rc[1], secondary_y=False)
    fig.add_trace(go.Scatter(x=xs_curve / 1e3, y=cum_gj, mode="lines", name=nm_cum,
                             line=dict(color="#ff7f0e", width=2), hoverinfo="skip"),
                  row=D_rc[0], col=D_rc[1], secondary_y=True)
    fig.add_trace(go.Scatter(x=xs_curve / 1e3, y=windowed_gj, mode="lines", name=nm_weg,
                             line=dict(color="#e377c2", width=2, dash="dash"), hoverinfo="skip"),
                  row=D_rc[0], col=D_rc[1], secondary_y=True)
    i_eta_m = len(fig.data)
    fig.add_trace(go.Scatter(x=[xs[0] / 1e3], y=[eta_mk[0]], mode="markers",
                             marker=dict(size=9, color="#1f77b4"), showlegend=False, hoverinfo="skip"),
                  row=D_rc[0], col=D_rc[1], secondary_y=False)
    i_win_m = len(fig.data)
    fig.add_trace(go.Scatter(x=[xs[0] / 1e3], y=[win_eta_mk[0]], mode="markers",
                             marker=dict(size=9, color="#9467bd"), showlegend=False, hoverinfo="skip"),
                  row=D_rc[0], col=D_rc[1], secondary_y=False)
    i_cap_m = len(fig.data)
    fig.add_trace(go.Scatter(x=[xs[0] / 1e3], y=[cap_mk[0]], mode="markers",
                             marker=dict(size=9, color="#2ca02c"), showlegend=False, hoverinfo="skip"),
                  row=D_rc[0], col=D_rc[1], secondary_y=False)
    i_cum_m = len(fig.data)
    fig.add_trace(go.Scatter(x=[xs[0] / 1e3], y=[cum_mk[0]], mode="markers",
                             marker=dict(size=9, color="#ff7f0e"), showlegend=False, hoverinfo="skip"),
                  row=D_rc[0], col=D_rc[1], secondary_y=True)
    i_wgj_m = len(fig.data)
    fig.add_trace(go.Scatter(x=[xs[0] / 1e3], y=[win_gj_mk[0]], mode="markers",
                             marker=dict(size=9, color="#e377c2"), showlegend=False, hoverinfo="skip"),
                  row=D_rc[0], col=D_rc[1], secondary_y=True)

    yy0 = frames_phys[0]["chan_total"] / max_total

    def add_channel(f0, cut, sil_key, ap_radial_m, rc):
        """Vector overlays for one channel, all in the SAME subplot `rc`: encircled contour + the
        white aperture-line + the plane slice (the emitter outline is added later, once the isotropic
        cross extent is known). Heatmaps are baked images. Returns the trace indices."""
        mr, mc = rc
        s0 = f0[sil_key]
        i_cont = len(fig.data)
        cxx, cyy = map_contour(f0, cut)
        fig.add_trace(go.Scatter(x=cxx, y=cyy, mode="lines", line=dict(color="#7dffce", width=1.1),
                                 opacity=0.45, showlegend=False, hoverinfo="skip"), row=mr, col=mc)
        if portrait:
            ap_x, ap_y, pl_x, pl_y = [-ap_radial_m, ap_radial_m], [0.0, 0.0], [-s0, s0], [yy0, yy0]
        else:
            ap_x, ap_y, pl_x, pl_y = [0.0, 0.0], [-ap_radial_m, ap_radial_m], [yy0, yy0], [-s0, s0]
        fig.add_trace(go.Scatter(x=ap_x, y=ap_y, mode="lines", line=dict(color="white", width=_MAP_EDGE_W),
                                 showlegend=False, hoverinfo="skip"), row=mr, col=mc)   # aperture (static)
        i_plane = len(fig.data)
        fig.add_trace(go.Scatter(x=pl_x, y=pl_y, mode="lines", line=dict(color="white", width=_MAP_EDGE_W),
                                 showlegend=False, hoverinfo="skip"), row=mr, col=mc)   # plane slice
        return dict(cont=i_cont, plane=i_plane)

    # F: major-axis (wide) cut — SHORT aperture axis along the radial. F₂: minor-axis (narrow) cut —
    # LONG axis along the radial (the 90°-rotated view of F's aperture). ap_cross is the OTHER axis.
    F = add_channel(frames_phys[0], "y", "sil_major_m", ap_short, F_rc)
    F2 = add_channel(frames_phys[0], "x", "sil_minor_m", ap_long, F2_rc)

    def _ax(idx):
        d = fig.data[idx]
        return (d.xaxis or "x"), (d.yaxis or "y")
    axF, axF2, axA = _ax(F["cont"]), _ax(F2["cont"]), _ax(i_contA)

    ALONG_GUT = 0.03      # data-space gutter (fraction of the map width) beyond the arc, for the axis
    ALONG_TOP = 1.2       # along-axis upper bound (> 1): room BEHIND the plane for its projection image

    def _panel_px(ax):
        """(along_px, radial_px): the panel's on-screen size in pixels along each map axis, from its
        fixed paper domain — the basis for isotropic pre-distortion of the emitter AND the plane."""
        xd = fig.layout[("xaxis" + ax[0][1:])].domain if ax[0] != "x" else fig.layout["xaxis"].domain
        yd = fig.layout[("yaxis" + ax[1][1:])].domain if ax[1] != "y" else fig.layout["yaxis"].domain
        px_w, px_h = (xd[1] - xd[0]) * inner_w, (yd[1] - yd[0]) * inner_h
        return (px_h, px_w) if portrait else (px_w, px_h)

    # ONE isotropic layout shared by BOTH beam maps (F and F₂ have the same paper domain → same pixels),
    # so they get the SAME along-axis range: the emitter face (along=0) lands at the same figure-x, and
    # the map (0→1) is identical, with room past the plane (1→ALONG_TOP) for the projection images. `k`
    # converts METRES to normalised-along units PER unit range-width (radial px/m ÷ along px/norm). The
    # along lower bound is set from the LARGER emitter (ap_long, cross of F) so it renders isotropically
    # with a gutter; the smaller emitter (ap_short) then occupies a sub-region with empty space beside it.
    along_px, radial_px = _panel_px(axF)
    k = (radial_px / (2.0 * PHASE_L)) / along_px
    B = min(ap_long * k, 0.6)                              # along-fraction the big emitter fills (capped)
    along_lo = -(B * ALONG_TOP + ALONG_GUT) / (1.0 - B)   # COMMON lower bound for both panels
    range_w = ALONG_TOP - along_lo
    lo_F, lo_F2 = ap_long * k * range_w, ap_short * k * range_w   # isotropic emitter half-regions (norm)
    npm = k * range_w                                     # normalised-along per METRE (isotropic)

    def add_emitter(rc, ap_radial_m, ap_cross_m, lo):
        """Emitter half-aperture outline, in the map subplot, pre-distorted so it reads isotropic:
        cross ∈ [−ap_cross, 0] maps to along ∈ [−lo, 0], radial is 1:1 metres."""
        oc, orad = emitter_outline(ap_radial_m, ap_cross_m)          # (cross ∈ [−ap_cross,0], radial)
        along = oc * (lo / ap_cross_m)
        ox, oy = (orad, along) if portrait else (along, orad)
        fig.add_trace(go.Scatter(x=ox, y=oy, mode="lines", line=dict(color="white", width=_MAP_EDGE_W),
                                 showlegend=False, hoverinfo="skip"), row=rc[0], col=rc[1])
    add_emitter(F_rc, ap_short, ap_long, lo_F)
    add_emitter(F2_rc, ap_long, ap_short, lo_F2)

    # The per-frame projection images (see _plane_proj) are placed at the plane line, isotropically
    # pre-distorted (radial 1:1 metres, along stretched by `npm`) and set BEHIND the slice line (far
    # side, along > plane) with a gap, into the 1→ALONG_TOP space so even the farthest frame has room.
    def _plane_rect(rhalf_m, ahalf_m, yy):
        ahalf = ahalf_m * npm                             # along half-extent (isotropic, norm units)
        gap = max(0.008, 0.3 * ahalf)
        mid = min(yy + ahalf + gap, ALONG_TOP - ahalf)    # behind the line, kept inside the top edge
        if portrait:
            return (-rhalf_m, mid - ahalf, 2.0 * rhalf_m, 2.0 * ahalf)   # x=radial, y=along
        return (mid - ahalf, -rhalf_m, 2.0 * ahalf, 2.0 * rhalf_m)       # x=along, y=radial

    # ---- background images (baked heatmaps), each positioned in its subplot's own data coords --
    def _map_rect():                                      # beam map over along ∈ [0,1], radial ±PHASE_L
        return (-PHASE_L, 0.0, 2 * PHASE_L, 1.0) if portrait else (0.0, -PHASE_L, 1.0, 2 * PHASE_L)
    def _em_rect(ap_radial_m, lo):                        # phase over along ∈ [−lo,0], radial ±ap_radial
        return ((-ap_radial_m, -lo, 2 * ap_radial_m, lo) if portrait
                else (-lo, -ap_radial_m, lo, 2 * ap_radial_m))
    A_rect = (-EXT_H, -EXT_V, 2 * EXT_H, 2 * EXT_V)

    def _image(src, ax, rect):
        return dict(source=src, xref=ax[0], yref=ax[1], x=rect[0], y=rect[1], sizex=rect[2],
                    sizey=rect[3], xanchor="left", yanchor="bottom", sizing="stretch", layer="below")

    def frame_images(f):                                  # baked backgrounds for one frame
        yy = f["chan_total"] / max_total
        pt_png, pt_rh, pt_ah = _plane_proj(f, "y")        # per-frame projection into F  (major cut)
        ps_png, ps_rh, ps_ah = _plane_proj(f, "x")        # per-frame projection into F₂ (minor cut)
        return [_image(incid_png(f), axA, A_rect),
                _image(map_png(f, "y"), axF, _map_rect()),
                _image(map_png(f, "x"), axF2, _map_rect()),
                _image(phase_png(f, ap_short, ap_long), axF, _em_rect(ap_short, lo_F)),
                _image(phase_png(f, ap_long, ap_short), axF2, _em_rect(ap_long, lo_F2)),
                _image(pt_png, axF, _plane_rect(pt_rh, pt_ah, yy)),
                _image(ps_png, axF2, _plane_rect(ps_rh, ps_ah, yy))]

    fig.update_layout(images=frame_images(frames_phys[0]))

    # Each frame swaps the 5 background images (a cheap relayout) and restyles the live vector
    # traces (contours, plane slices, D markers, capture text) — no per-frame raster or contour
    # recompute in the browser, so playback stays fast.
    ANIMATED = [i_contA, i_contA_dot, i_capA, i_eta_m, i_win_m, i_cap_m, i_cum_m, i_wgj_m,
                F["cont"], F["plane"], F2["cont"], F2["plane"]]
    frames = []
    for k, f in enumerate(frames_phys):
        cur = xs[k] / 1e3
        yy = f["chan_total"] / max_total
        cAx, cAy = incid_contour(f, _SIL_SOLID)
        dAx, dAy = incid_contour(f, _SIL_DOTTED)
        fcx, fcy = map_contour(f, "y"); f2cx, f2cy = map_contour(f, "x")
        if portrait:
            fpl = go.Scatter(x=[-f["sil_major_m"], f["sil_major_m"]], y=[yy, yy])
            f2pl = go.Scatter(x=[-f["sil_minor_m"], f["sil_minor_m"]], y=[yy, yy])
        else:
            fpl = go.Scatter(x=[yy, yy], y=[-f["sil_major_m"], f["sil_major_m"]])
            f2pl = go.Scatter(x=[yy, yy], y=[-f["sil_minor_m"], f["sil_minor_m"]])
        frames.append(go.Frame(name=str(k), traces=ANIMATED,
            data=[go.Scatter(x=cAx, y=cAy),
                  go.Scatter(x=dAx, y=dAy),
                  go.Scatter(text=[f"{cap_pct_frame[k]:.0f}% captured"]),
                  go.Scatter(x=[cur], y=[eta_mk[k]]),
                  go.Scatter(x=[cur], y=[win_eta_mk[k]]),
                  go.Scatter(x=[cur], y=[cap_mk[k]]),
                  go.Scatter(x=[cur], y=[cum_mk[k]]),
                  go.Scatter(x=[cur], y=[win_gj_mk[k]]),
                  go.Scatter(x=fcx, y=fcy), fpl,
                  go.Scatter(x=f2cx, y=f2cy), f2pl],
            layout=go.Layout(images=frame_images(f))))
    fig.frames = frames

    # Slider steps animate to a SINGLE frame (mode immediate, redraw True) — the one path that
    # reliably redraws Heatmap. Plotly's native sequenced *play* only honours redraw on the first
    # frame, so the ▶ play button is the custom sequential stepper injected via post_script.
    steps = [dict(method="animate", label=f"{xs[k]/1e3:.0f}",
                  args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))]) for k in range(n_frames)]
    # Slider at the TOP (in the top margin, under the title) alongside the ▶ play button.
    slider = dict(active=0, y=(1.035 if portrait else 1.02), yanchor="bottom",
                  x=(0.16 if portrait else 0.13), len=(0.74 if portrait else 0.80),
                  currentvalue=dict(prefix="along-track: ", suffix=" km"), steps=steps)
    fig.update_layout(sliders=[slider])

    # Panel A: 1:1 so the spot is undistorted; constrain="domain" shrinks the box to a 2:1 strip.
    fig.update_xaxes(title_text="view-plane u (m)", scaleanchor="y", scaleratio=1,
                     range=[-EXT_H, EXT_H], constrain="domain", row=A_rc[0], col=A_rc[1])
    fig.update_yaxes(title_text="v (m)", range=[-EXT_V, EXT_V], row=A_rc[0], col=A_rc[1])
    # Panel D: FIXED ranges (x, both y) so nothing re-autoranges as the markers move.
    d_left = max(float(eta_series.max()), float(capture_series.max())) * 1.08
    fig.update_xaxes(title_text="along-track (km)", range=[xs[0] / 1e3, xs[-1] / 1e3],
                     row=D_rc[0], col=D_rc[1])
    fig.update_yaxes(title_text="capture / η (%)", range=[0, d_left], secondary_y=False,
                     row=D_rc[0], col=D_rc[1])
    fig.update_yaxes(title_text="cumulative energy (GJ)", range=[0, float(cum_gj[-1]) * 1.08 + 1e-9],
                     secondary_y=True, row=D_rc[0], col=D_rc[1])
    # Beam maps (with the emitter merged in): the ALONG axis carries DISTANCE FROM THE EMITTER in km
    # (0 at the array face → max_total at the far horizon; the plane slice slides along it per frame).
    # Ticks are placed only in the physical map region (along ≥ 0) at regular km intervals — the
    # negative region is the emitter aperture (the isotropically-compressed near-array zone), NOT
    # distance, so it gets no distance ticks. The RADIAL axis (±PHASE_L) carries the metre ticks +
    # title at the plot's far edge (LEFT of the ellipses landscape / BELOW portrait). The two axes are
    # NOT scale-anchored (different units); emitter + plane isotropy is baked into `lo` / `npm`. Both
    # maps share the SAME along range [along_lo, ALONG_TOP], so the emitter face (0) and the plane
    # projection space (1→ALONG_TOP) align across F and F₂.
    max_km = max_total / 1e3
    dk = next(d for d in (10, 20, 25, 50, 100, 150, 200, 250) if max_km / d <= 7)
    km_ticks = np.arange(0.0, max_km + 1e-9, dk)
    km_tickvals = (km_ticks * 1e3 / max_total).tolist()          # normalised positions in [0, 1]
    km_ticktext = [f"{v:.0f}" for v in km_ticks]
    for rc in (F_rc, F2_rc):
        along_kw = dict(title_text="distance from emitter (km)", range=[along_lo, ALONG_TOP],
                        tickvals=km_tickvals, ticktext=km_ticktext, showgrid=False, zeroline=False)
        radial_kw = dict(title_text="radial (m)", range=[-PHASE_L, PHASE_L],
                         showgrid=False, zeroline=False)
        if portrait:
            fig.update_yaxes(**along_kw, row=rc[0], col=rc[1])
            fig.update_xaxes(**radial_kw, row=rc[0], col=rc[1])
        else:
            fig.update_xaxes(**along_kw, row=rc[0], col=rc[1])
            fig.update_yaxes(**radial_kw, row=rc[0], col=rc[1])
    fig.add_annotation(x=0.02, y=0.03, xref="x domain", yref="y domain", row=A_rc[0], col=A_rc[1],
                       xanchor="left", yanchor="bottom",
                       text="green = enclosed beam-energy fraction",
                       showarrow=False, font=dict(size=8, color="#8fffd0"))
    pass_kind = "head-on pass" if abs(lateral_m) < 1.0 else f"lateral pass ({lateral_offset_km:.0f} km offset)"
    aperture_m = kw.get("emitter_diam_m") or kw.get("relay_aperture_diam_m") or 0.0
    stamp = _build_stamp()
    # Legend lives INSIDE panel D (the line graph), anchored to its top-left interior — where every
    # curve is near zero (far along-track), so it overlaps nothing. Position from D's actual paper
    # domain (row 2), robust to the row-height / orientation differences.
    d_xdom, d_ydom = fig.layout["xaxis2"].domain, fig.layout["yaxis2"].domain
    legend = dict(x=d_xdom[0] + 0.010, y=d_ydom[1] - 0.008, xanchor="left", yanchor="top",
                  orientation="v", font=dict(size=(8 if portrait else 11)),
                  bgcolor="rgba(0,0,0,0.4)", bordercolor="rgba(255,255,255,0.18)", borderwidth=1)
    btn_top = 46 if portrait else 60          # clear of the (now multi-line) headline
    if portrait:
        fig.update_layout(
            template="plotly_dark", height=FIG_H, width=FIG_W, paper_bgcolor="black", plot_bgcolor="black",
            margin=MARG,
            title=dict(text=(f"Flyby — {arch_name}<br><sub>{aperture_m:.0f} m emitter · {weather} · "
                             f"{pass_kind} · ±{half/1e3:.0f} km · ▶ play or drag"
                             f"<br>build {stamp}</sub>"),
                       y=0.992, yanchor="top", x=0.5, xanchor="center", font=dict(size=12)),
            legend=legend)
    else:
        # Headline wrapped to a few lines so it fits the graph width instead of one long line.
        fig.update_layout(
            template="plotly_dark", height=FIG_H, width=FIG_W, margin=MARG,
            title=dict(text=(f"Flyby animation — {arch_name} · {aperture_m:.0f} m emitter · {weather} · {pass_kind}"
                             f"<br><sub>horizon-to-horizon ±{half/1e3:.0f} km · uniform in time · build {stamp}"
                             f"<br>▶ play or drag the slider above</sub>"),
                       y=0.988, yanchor="top", x=0.5, xanchor="center", font=dict(size=15)),
            legend=legend)
    if write:
        os.makedirs(OUT, exist_ok=True)
        # The ▶ play button is the custom sequential stepper (see _PLAYER_JS): it swaps the baked
        # background images + restyles the live vector traces per frame, without a native replay.
        post = _PLAYER_JS.replace("{n_frames}", str(n_frames)).replace("{btn_top}", str(btn_top))
        fig.write_html(path, include_plotlyjs="inline", auto_play=False, post_script=post)
    return fig


# The single reference case: 5.8 GHz weatherproof low band straight from a 100 m ground
# emitter, clear air, a close 5 km lateral pass (the plane crossing near-overhead rather than
# dead-on, so the geometry stays legible). Drag the slider or press ▶ play to scrub the pass.
if __name__ == "__main__":
    build_flyby_animation(arch_name="5.8 direct", weather="clear", lateral_offset_km=5.0,
                          emitter_diam_m=100.0, path=f"{OUT}/flyby.html")
    print("wrote", f"{OUT}/flyby.html")
