"""Per-frame physics for the flyby animation — everything one frame of the picture needs to
know about the beam, computed from the engine, with no plotly in sight.

The animation module owns layout, colour and baking; this owns the numbers it draws:
  `frame_physics` — one plane position of one architecture through `beam_link.evaluate_with_field`,
    plus the view basis, the silhouette's principal extents, and the beam-channel legs.
  `channel_field` — the scalar-diffraction beam along the WHOLE path as a side view, leg by leg,
    in ABSOLUTE W/m², with the encircled-energy fraction the funnel contours are inverted from
    and, optionally, the aircraft's own shadow in the beam past it.
  `exposure_boundary` — the outermost radius per plane at which that field still exceeds an
    exposure limit: the beam's keep-out tube.
  `beam_cut_axes` — the body-frame axes of one beam-cut plane, which is what orients the airframe
    projection the animation draws on top of that cut.

Reached into by `beam_link_animation`, `beam_link_phone` and the test suite, so the names here
are public: they are the seam a rewrite of either side would meet at.
"""

from __future__ import annotations

import numpy as np

from beamlink import beam_link as bl
from beamlink import bwb_model as m
from beamlink import diffraction as dfr
from beamlink import geo


# Diffraction transform grid for the beam-channel panels (F/F₂). 384 (not 192): the shaped
# emitter's elongated aperture (semi-major ≈ √aspect · r_emit, ~141 m at aspect 8) only fits
# the transform window with adequate margin at this size. At 192 the far/defocused frames alias
# badly (RMS ~17 %, peak ~40 % off a converged reference — visible as non-monotone banding);
# 384 converges (RMS <0.2 %). n≥320 also trips scipy's multi-threaded FFT (see _fft_workers).
CHANNEL_NGRID = 384
# Aperture-sample cap for the near-field zoom-DFT (`diffraction.axial_cuts_direct`) that resolves the
# beam map close to the emitter, below the single-FFT's transverse window. It sets how near the array
# we compute real diffraction before handing off to the geometric aperture fill: n_ap ∝ 1/z, so this
# cap ⇒ a first-resolved plane of ~a few hundred m for a compact aperture to ~1 km for the elongated
# far-frame aperture (both « the ~200-300 km map width). Higher = closer + slower; 3072 is offline-cheap.
N_AP_MAX_CHANNEL = 3072


def localize(downrange_m, crossrange_m, altitude_m, basis):
    """ECEF point → station-local (downrange, crossrange, up) metres, so earth
    curvature shows as the ground and cruise arcs falling away at the edges."""
    base, east, north, up = basis
    d = geo.point(downrange_m, crossrange_m, altitude_m) - base
    return np.array([np.dot(d, north), np.dot(d, east), np.dot(d, up)])


def _shadow_transmission(local_z, radial, plane_dist_m, silhouette_half_m, aperture_half_m,
                         wavelength_m, captured_fraction):
    """Fraction of the beam that survives the aircraft, plane by plane — the shadow it casts.

    100 % capture and no reflection: whatever the silhouette intercepts is gone from the beam
    downstream, and `captured_fraction` (the engine's geometric overlap, the same number panel A
    prints) is how much that is. Two limits bracket what the shadow looks like, and this blends
    between them:

      CLOSE BEHIND, ray optics. The emitting aperture is `aperture_half_m` wide on this cut, so the
      shadow has an umbra (no part of the aperture visible) inside a penumbra (part of it):
        r_umbra(z) = a + (a − R)·(z − d)/d        r_penumbra(z) = a + (a + R)·(z − d)/d
      for silhouette half-extent a, aperture half-extent R, aircraft range d. An aircraft narrower
      than the aperture (a < R) has an umbra that CLOSES at z = d·a/(R − a) and is all penumbra
      beyond — which is why the dark core fades on its own even before diffraction acts.

      FAR BEHIND, diffraction. The surrounding beam spreads across the shadow once the Fresnel scale
      √(λ·Δz) exceeds the aircraft, i.e. beyond Δz ≈ a²/λ (~17 km for a 30 m half-span at 5.8 GHz).
      There the beam has its original shape again, carrying (1 − captured_fraction) of its power.

    The blend weight is `fade` = 1 − √(λ·Δz)/a, so both ends are the honest limit and the crossover
    sits where the physics says it does. What this is NOT is an obstructed-diffraction calculation:
    it has no shadow-edge fringes, and the transition between the two limits is an interpolation
    rather than a propagation. The rigorous version is Babinet with the silhouette approximated by a
    separable rectangle — the aperture illumination and the residual focusing phase are both
    separable, so the blocked field is a difference of two separable terms and each propagates as a
    pair of 1-D transforms (the same trick `diffraction.axial_cuts_direct` already uses). That is
    the upgrade path if the shadow ever has to carry more than the exposure read.
    """
    dz = np.asarray(local_z, float)[:, None] - float(plane_dist_m)      # (n_z, 1), past the aircraft
    r = np.abs(np.asarray(radial, float))[None, :]
    a, R = max(float(silhouette_half_m), 1e-9), float(aperture_half_m)
    d = max(float(plane_dist_m), 1e-9)
    umbra = np.maximum(a + (a - R) * dz / d, 0.0)
    penumbra = a + (a + R) * dz / d
    lit = np.clip((r - umbra) / np.maximum(penumbra - umbra, 1e-9), 0.0, 1.0)   # ray-optics shadow
    fade = np.clip(1.0 - np.sqrt(float(wavelength_m) * np.maximum(dz, 0.0)) / a, 0.0, 1.0)
    survives = 1.0 - float(captured_fraction)
    return np.where(dz > 0.0, fade * lit + (1.0 - fade) * survives, 1.0)


def exposure_boundary(intensity_w_m2, dist, radial, level_w_m2, edge_margin=0.9):
    """The beam's keep-out tube: per plane, the OUTERMOST radius at which the beam still exceeds
    `level_w_m2`. Inverted per plane like `funnel_contour` (not by marching squares), but from the
    outside in, because intensity is not monotonic in r — sidelobes cross the level several times
    and it is the last crossing that bounds the volume to keep clear.

    What is inverted is the sidelobe ENVELOPE, not the raw profile: max over [r, r_max], which is
    monotone by construction, so the boundary follows the skirt instead of jumping between the nulls
    of individual sidelobes. Keeping people out of a null between two lobes that both exceed the
    limit is not a real option anyway.

    A plane contributes a gap (NaN) when the level is exceeded nowhere, and also when the boundary
    falls outside `edge_margin` of the radial window — where it is effectively off the map, and
    drawing it would claim a tube the picture cannot see. The margin is not just tidiness: near the
    aperture the boundary sits AT the window edge for tens of km, and without it the line breaks into
    a dashed mess as consecutive planes flip either side of the edge. Same convention as the funnel —
    no line means the beam is wider than the window. Returns (r_vals, z_vals) flat arrays with NaN
    gaps, the +side then the −side."""
    pos = radial >= 0
    order = np.argsort(radial[pos])
    rp = radial[pos][order]                           # 0 → +max, ascending
    r_lvl = np.full(intensity_w_m2.shape[0], np.nan)
    for i in range(intensity_w_m2.shape[0]):
        row = np.nan_to_num(intensity_w_m2[i, pos][order], nan=0.0)
        env = np.maximum.accumulate(row[::-1])[::-1]  # max over [r, r_max] — non-increasing in r
        if env[0] <= level_w_m2:
            continue                                  # nowhere above the level on this plane
        r = float(np.interp(level_w_m2, env[::-1], rp[::-1]))
        if r < edge_margin * rp[-1]:
            r_lvl[i] = r
    r_out, z_out = [], []
    for sign in (1.0, -1.0):
        r_out.extend(sign * r_lvl); r_out.append(np.nan)
        z_out.extend(dist);         z_out.append(np.nan)
    return np.asarray(r_out), np.asarray(z_out)


def channel_field(dist_norm, radial, legs, total_shown_m, current_total_m, shadow=None):
    """Scalar-diffraction beam along the whole path, as a (n_dist, n_radial) side view: the
    along-path axis (fraction of the longest total path over the pass) is laid out over the
    link's legs — one leg for a direct link, TWO for a relay (ground→relay up-leg focused at
    the relay, then relay→plane down-leg focused at the plane). Both principal cuts are computed
    in ONE diffraction pass per leg (`diffraction.axial_cut_xy`, complex64 — display precision),
    so the picture is the model: converging cone → diffraction waist → divergence, Airy sidelobes
    included. Each leg is scaled by its own radiated power (`leg["power_w"]`), so the field is in
    ABSOLUTE W/m² before display normalisation — which is what an exposure limit can be compared
    against, and what keeps the two legs of a relay in their true relative brightness.

    `current_total_m` is how far along the path to render. Passing the aircraft's own range stops
    the beam there (nothing past it is drawn); passing more extends the LAST leg beyond the aircraft
    — the same Fresnel evaluation at z > focus, the beam diverging again — which is what shows the
    power that MISSES the aircraft and how far the beam stays hazardous. `shadow`, then, is what the
    aircraft removes from it: dict(captured_fraction=, half_major_m=, half_minor_m=) — the engine's
    geometric overlap plus the silhouette's half-extents on the wide and narrow cuts (the same
    numbers the white plane-slice line is drawn from). See `_shadow_transmission`. Without it the
    beam past the aircraft is drawn as if the aircraft were not there.

    Returns (ix, ex, iy, ey, peak_w_m2): intensity of the narrow (x) and wide (y) cuts, each divided
    by `peak_w_m2` (the shared frame peak, so multiplying by it recovers W/m²), with the
    strip-marginal encircled-energy fraction for the funnel."""
    s = dist_norm * total_shown_m
    shp = (dist_norm.size, radial.size)
    ix = np.full(shp, np.nan); ex = np.full(shp, np.nan)
    iy = np.full(shp, np.nan); ey = np.full(shp, np.nan)
    # Enclosed-energy is integrated on ONE common wide radial grid, NOT each plane's native
    # diffraction grid (spacing λz/(N·d), which jumps along z and put a numerical STAIRCASE in
    # E(z) — the steps in the funnel contours). Sampling every plane's intensity onto the same
    # grid and integrating there makes E(z) smooth (only real Fresnel structure survives); the
    # grid is 3× the display half-extent so it captures the beam and E normalises to the true
    # total (a beam overspilling the display window then correctly shows no high-level contours).
    half = float(radial.max())
    r_wide = np.linspace(-3.0 * half, 3.0 * half, 3 * radial.size)
    o = np.argsort(np.abs(r_wide)); dw = r_wide[1] - r_wide[0]
    cum = 0.0
    for index, leg in enumerate(legs):
        last = index == len(legs) - 1
        # Each leg owns the planes between its own aperture and its end — except the LAST leg, which
        # runs on to `current_total_m`, so asking for more than the path length draws the beam PAST
        # the aircraft rather than truncating it there.
        sel = (s >= cum) & (s <= current_total_m)
        if not last:
            sel &= s < cum + leg["length_m"]
        if leg["length_m"] > 0 and np.any(sel):
            local_z = s[sel] - cum                           # distance from this leg's emitter
            lix, _, liy, _ = dfr.axial_cut_xy(
                local_z, leg["focus_m"], leg["ap_major"], leg["ap_minor"], leg["lam"], r_wide,
                taper_p=leg["taper_p"], n_grid=CHANNEL_NGRID, normalize=False, dtype=np.complex64)
            lix, liy = np.array(lix, dtype=np.float64), np.array(liy, dtype=np.float64)
            # NEAR FIELD (Fresnel zoom-DFT): the single-FFT above runs out of transverse window and
            # returns NaN close to the aperture. Resolve those planes DIRECTLY from the aperture down
            # to the Fresnel validity floor (~a few aperture diameters) — real converging near-field
            # cone, no fabricated shape. Below the floor the geometric fill below still takes over, but
            # over a much thinner sliver. See diffraction.axial_cuts_direct.
            unresolved = ~np.isfinite(lix).any(axis=1)
            if unresolved.any():
                nfx, nfy = dfr.axial_cuts_direct(
                    local_z[unresolved], leg["focus_m"], leg["ap_major"], leg["ap_minor"],
                    leg["lam"], r_wide, taper_p=leg["taper_p"], n_ap_max=N_AP_MAX_CHANNEL)
                lix[unresolved], liy[unresolved] = nfx, nfy
            idx = np.where(sel)[0]
            taper_p = float(leg["taper_p"])
            sh = shadow or {}
            for inten_w, i_out, e_out, ap_cut, sil_half in (
                    (lix, ix, ex, leg["ap_major"], sh.get("half_minor_m")),
                    (liy, iy, ey, leg["ap_minor"], sh.get("half_major_m"))):
                # W/m²: the raw cut is a fraction of radiated power per m², so scaling by this leg's
                # radiated power makes the whole channel absolute (and relay legs comparable).
                Iw = np.abs(np.nan_to_num(np.asarray(inten_w, dtype=np.float64), nan=0.0)) \
                    * float(leg["power_w"])
                filled = np.isfinite(inten_w).any(axis=1)    # planes with a resolved diffraction cut
                # SUB-FRESNEL FILL: planes closer than the Fresnel floor (below what the zoom above can
                # resolve) come back NaN. There the paraxial field IS ≈ the aperture illumination, so
                # synthesize them as the aperture's OWN pattern — a taper² top-hat of half-width
                # `ap_cut` — crossfaded (∝ z) into the first RESOLVED plane so there is no seam. This
                # honest fill now spans only the thin sub-Fresnel sliver, not the old km-scale gap. (A
                # rigorous field there — e.g. beside the beam in the aperture plane for an EXPOSURE
                # estimate — needs an angular-spectrum / Rayleigh–Sommerfeld pass; see axial_cuts_direct.)
                if filled.any() and (~filled).any():
                    fv = int(np.argmax(filled))              # first resolved plane (planes sort by z)
                    z_h = max(float(local_z[fv]), 1e-9)
                    rho = np.abs(r_wide) / max(float(ap_cut), 1e-9)
                    ap_shape = np.where(rho < 1.0, np.clip(1.0 - rho ** 2, 0.0, None) ** (2.0 * taper_p), 0.0)
                    ssa = ap_shape.sum()
                    ap_prof = ap_shape * (Iw[fv].sum() / ssa) if ssa > 0 else Iw[fv].copy()
                    near = np.nonzero(~filled)[0]
                    near = near[local_z[near] < z_h]         # only the near-aperture side of the gap
                    for a in near:
                        w = float(np.clip(local_z[a] / z_h, 0.0, 1.0))
                        Iw[a] = (1.0 - w) * ap_prof + w * Iw[fv]
                    filled[near] = True
                # The funnel's denominator is the plane total BEFORE the aircraft takes its share, so
                # what the aircraft absorbs SHOWS UP as the high-level rings closing at the plane
                # line. Normalising to the surviving power instead would renormalise the loss away
                # and draw a confident funnel around a beam that is no longer there.
                tot = (Iw.sum(axis=1) * dw)[:, None]
                if shadow is not None and last:
                    Iw = Iw * _shadow_transmission(local_z, r_wide, leg["length_m"], sil_half,
                                                   ap_cut, leg["lam"], shadow["captured_fraction"])
                c = np.cumsum(Iw[:, o] * dw, axis=1)         # enclosed from the axis outward
                Ew = np.divide(c, tot, out=np.full_like(c, np.nan), where=tot > 0)
                Ew_r = np.empty_like(Ew); Ew_r[:, o] = Ew    # undo the |r| sort → r_wide order
                for a, zi in enumerate(idx):
                    if not filled[a]:
                        continue
                    i_out[zi] = np.interp(radial, r_wide, Iw[a])     # display intensity (common grid)
                    e_out[zi] = np.interp(radial, r_wide, Ew_r[a])   # smooth enclosed fraction
        cum += leg["length_m"]

    # Normalise BOTH cuts to ONE shared peak over the whole frame (not each to its own peak), so the
    # relative brightness between the wide (F) and narrow (F₂) cuts is preserved — the narrow cut
    # focuses tighter so it is genuinely brighter at the waist. The near field is then dim in both
    # (its power is spread over the aperture width) — that is honest, per the display convention.
    fin_x, fin_y = np.isfinite(ix), np.isfinite(iy)
    peak = max(float(ix[fin_x].max()) if fin_x.any() else 0.0,
               float(iy[fin_y].max()) if fin_y.any() else 0.0)
    scale = peak if peak > 0 else 1.0
    return ((ix / scale).astype(np.float32), ex.astype(np.float32),
            (iy / scale).astype(np.float32), ey.astype(np.float32), float(scale))


def funnel_contour(enc, dist, radial, levels):
    """Enclosed-energy funnel contours by DIRECT per-plane inversion, not 2-D marching squares. At
    each plane the enclosed fraction E(|r|) rises monotonically from 0 (on axis) to its window
    total, so the level-L contour radius is simply r_L = E⁻¹(L) — one 1-D interpolation per plane.
    This yields a smooth r_L(z) with no cell-quantised staircase and no spurious segments along the
    data boundary (both artefacts of running a contour generator over a discretised, partly-NaN
    grid). Where the beam is wider than the window (E never reaches L) or the plane is near-field
    NaN, r_L is left as a gap. `enc` is (len(dist), len(radial)); returns (r_vals, z_vals) flat
    arrays (metres, normalised-distance) with NaN gaps — the +side then the −side of each level."""
    pos = np.argsort(radial[radial >= 0])
    rp = radial[radial >= 0][pos]                     # 0 → +max, ascending
    r_out, z_out = [], []
    for level in levels:
        r_lvl = np.full(enc.shape[0], np.nan)
        for i in range(enc.shape[0]):
            e = enc[i, radial >= 0][pos]
            fin = np.isfinite(e)
            if fin.sum() < 2:
                continue
            ei, ri = e[fin], rp[fin]
            if ei[-1] < level:                        # beam overspills the window at this plane
                continue
            r_lvl[i] = np.interp(level, ei, ri)       # E is monotonic ↑ in r → invert
        for sign in (1.0, -1.0):
            r_out.extend(sign * r_lvl); r_out.append(np.nan)
            z_out.extend(dist);         z_out.append(np.nan)
    return np.asarray(r_out), np.asarray(z_out)


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


def frame_physics(arch_kwargs, x_along_m, plane_alt_m, basis, coarse_mesh,
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
    lam = bl.relations.SPEED_OF_LIGHT / (freq * 1e9)
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
    u_ax, v_ax, w_ax = m._view_basis(look)
    up_u, up_v = float(np.array([0, 0, -1.0]) @ u_ax), float(np.array([0, 0, -1.0]) @ v_ax)
    phi = (np.pi / 2.0 - np.arctan2(up_v, up_u)) if np.hypot(up_u, up_v) > 1e-3 else 0.0
    cph, sph = np.cos(phi), np.sin(phi)
    if phi and field["u"].size:
        U, V = field["u"], field["v"]
        field = dict(field, u=cph * U - sph * V, v=sph * U + cph * V)
    # The SAME in-plane rotation as an explicit view basis, so the z-buffer rasterizer (panel A)
    # renders the silhouette in exactly the orientation of the MC field above (dorsal up).
    view_u, view_v, view_w = cph * u_ax - sph * v_ax, sph * u_ax + cph * v_ax, w_ax
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
    # Radiated power per leg, so the channel map can be drawn in absolute W/m² (exposure limits are
    # absolute). The field is normalised to unit radiated power per plane, so the beam power at the
    # aircraft is the delivered DC divided back out through capture and the rectenna, and the LAUNCH
    # power adds the atmospheric loss back on — the worst case anywhere on that leg. The floors keep
    # a horizon-blocked frame (atmos → 0, so η → 0 and grid power → ∞) finite instead of poisoning
    # the whole map with infinities.
    p_at_plane = delivered_mw * 1e6 / max(eff * float(s["rectenna_total"]), 1e-9)
    p_rad_down = p_at_plane / max(float(s["atmos"]), 1e-3)
    down_leg = dict(ap_major=ap_major, ap_minor=ap_minor, focus_m=focus,
                    length_m=float(slant), lam=lam, taper_p=taper_p, power_w=p_rad_down)
    if cfg.relay_pos is not None:
        station = np.asarray(cfg.station_pos, float)
        L1 = float(bl.geometry(station, np.asarray(cfg.relay_pos, float))[0])   # ground→relay
        r_relay = cfg.relay_aperture_diam_m / 2.0
        r_up = bl.size_emitter_diameter(cfg.f_emit_relay_ghz, L1, r_relay, cfg.tau_target) / 2.0
        lam_up = bl.relations.SPEED_OF_LIGHT / (cfg.f_emit_relay_ghz * 1e9)
        # The up-leg carries MORE power than the down-leg: walk the down-leg's launch power back up
        # through the relay's conversion and aperture and the up-leg's own capture and atmosphere.
        up_chain = (float(s["relay_convert"]) * float(s["relay_aperture"])
                    * float(s["capture_up"]) * float(s["atmos_up"]))
        up_leg = dict(ap_major=float(r_up), ap_minor=float(r_up), focus_m=L1,
                      length_m=L1, lam=lam_up, taper_p=0.0,
                      power_w=p_rad_down / max(up_chain, 1e-6))
        chan_legs = [up_leg, down_leg]
        chan_total = L1 + float(slant)
        chan_relay_m, chan_relay_radius_m, chan_emit_radius_m = L1, r_relay, float(r_up)
    else:
        chan_legs = [down_leg]
        chan_total = float(slant)
        chan_relay_m, chan_relay_radius_m, chan_emit_radius_m = None, None, r_emit

    plane_loc = localize(x_along_m, lateral_offset_m, plane_alt_m, basis)
    emitter_loc = localize(emitter[0], emitter[1], emitter[2], basis)
    return dict(field=field, aspect=aspect, taper_p=taper_p,
                geom=geom, eff=eff, eta=eta, elevation=float(elevation), slant_km=slant / 1e3,
                slant_m=float(slant), emitter_radius_m=r_emit, sil_radial_m=sil_radial_m,
                chan_legs=chan_legs,
                chan_total=chan_total, chan_relay_m=chan_relay_m,
                chan_relay_radius_m=chan_relay_radius_m, chan_emit_radius_m=chan_emit_radius_m,
                sil_major_m=sil_major_m, sil_minor_m=sil_minor_m,
                beta=float(beta), area=raster["area"], look=look, face_incidence=face_incidence,
                view_u=view_u, view_v=view_v, view_w=view_w,
                plane_loc=plane_loc, emitter_loc=emitter_loc,
                phase_d=(np.asarray(plane, float) - emitter), phase_R=float(diam / 2.0),
                phase_lam=lam, phase_normal_up=(cfg.relay_pos is None),
                wf_labels=[lbl for lbl, _ in wf], wf_values=[v for _, v in wf])


def _pin_transverse_sign(axis):
    """Pin a silhouette principal axis to the AIRFRAME, in body coordinates.

    `np.linalg.eigh` fixes each principal DIRECTION but not its sign, and the sign it happens to
    return flips as the pass crosses overhead — which mirrors whatever is drawn on that axis.
    Anchor to DORSAL (−ẑ body: the same "dorsal up" convention panel A is rotated into), and fall
    back to STARBOARD, then NOSE, for the axes perpendicular to it — the span axis always, and the
    near-vertical axis at overhead, where it swings into the nose direction. A unit axis always has
    a sizeable component on one of the three, so the chain resolves.
    """
    candidates = (-axis[2], axis[1], axis[0])                     # dorsal, starboard, nose
    reference = next((c for c in candidates if abs(c) > 0.1), 1.0)
    return axis if reference >= 0.0 else -axis


def beam_cut_axes(frame, cut, portrait=False):
    """(radial, along, view) body-frame unit axes of one beam-cut plane of `frame`.

    A beam map is a cut of the 3-D beam containing the propagation axis and one transverse axis:
    `cut='y'` (F) takes the MAJOR silhouette axis as radial and views along the minor, `cut='x'`
    (F₂) the other way round. The transverse axes are the silhouette's principal axes in the
    panel-A view plane, so a cut foreshortens and rotates with the geometry.

      `along`  is the PROPAGATION direction, emitter → plane, matching the map's along axis
               (0 at the aperture, 1 at the plane). Note `frame["look"]` runs the OTHER way —
               it points at the emitter, which is what incidence needs.
      `radial` is the chosen transverse axis, sign-pinned to the airframe.
      `view`   is DERIVED from the other two rather than taken as the leftover eigenvector, so
               the render is a true view of the cut plane and not its mirror: the viewer sits at
               +view, which has to be (screen right × screen up). `portrait` swaps those two,
               putting radial on the screen's horizontal instead of its vertical, so it also
               swaps the side the plane is viewed from.

    A pass turns the airframe by ~170° in the beam's frame (the beam arrives ahead of the aircraft
    inbound, from below at overhead, behind it outbound), so in the vertical cut it CANNOT be drawn
    dorsal-up throughout and continuously — one or the other has to give. Pinning to the airframe
    picks dorsal-up: every frame reads as an aircraft the right way up, at the cost of the drawn
    attitude mirroring once, at the overhead frame where the beam is belly-on and up-vs-down in the
    cut is degenerate anyway. The horizontal cut has no such choice to make (dorsal is its normal),
    and comes out continuous: plan view inbound, head-on at overhead, from below outbound.
    """
    look = np.asarray(frame["look"], float)
    look = look / (np.linalg.norm(look) + 1e-30)
    along = -look
    view_u, view_v = np.asarray(frame["view_u"], float), np.asarray(frame["view_v"], float)
    coords = np.vstack([frame["field"]["u"], frame["field"]["v"]])
    if coords.shape[1] >= 3:
        _, evec = np.linalg.eigh(np.cov(coords))     # cols [minor, major] in the (view_u, view_v) plane
        major = evec[0, 1] * view_u + evec[1, 1] * view_v
        minor = evec[0, 0] * view_u + evec[1, 0] * view_v
    else:
        major, minor = view_u, view_v
    radial = _pin_transverse_sign(major if cut == "y" else minor)
    view = np.cross(radial, along) if portrait else np.cross(along, radial)
    return radial, along, view


def size_emitter_far(kw, ref_elev_deg, plane_alt_m):
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
