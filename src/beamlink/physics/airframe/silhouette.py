"""The silhouette: what surface the capture integral sums over, per beam-arrival angle.

The convex projected-area identity over-counts a non-convex body (a winglet behind the body, a
far wing seen edge-on) because it has no occlusion test. For the honest overlap we rasterize the
ACTUAL projection with a z-buffer over the mesh triangles: each covered pixel is one receiver
element of area du·dv — already a projected area — and the incidence of the skin the beam
actually reaches, cell by cell.

Two things about this module are load-bearing for the rest of the engine.

**It is the one kernel that does not vectorize for free** (§6a): a z-buffer loop over triangles,
and the dominant cost of a cold pass. What makes it affordable is not batching it but noticing
that it depends on the ARRIVAL ANGLE alone — not on frequency, range, weather or power — so
binning the angle and caching by content key turns 241 samples into ~40 rasters, and a second run
of the same loft into ~1 ms per angle. That is the whole of §10 item 1, and it costs no fidelity.

**It defines the one transverse frame the link uses.** The silhouette's own principal axes, sign-
pinned to the airframe, are the aperture's principal axes and therefore the frame P5's field is
sampled and emitted in. The pre-SPEC code computed those axes twice — once in the capture integral
and once in the figure's frame payload, the second time in a display basis — which is what made
the F/F₂ cut pair viz-shaped (§8 item 13). There is one computation now, in P3, and everything
downstream reads it. The sign of that frame is a gauge choice rather than a continuous one; see
`_pin_to_airframe` for why, and for what does and does not depend on it.
"""

from __future__ import annotations

import numpy as np

from beamlink.artifacts.spec import LoftSpec
from beamlink.physics import cache
from beamlink.physics.airframe import loft as loft_module
from beamlink.physics.fidelity import FidelityConfig


def _auxiliary_basis(look_dir):
    """A deterministic orthonormal (u, v, w) with w along the look direction (towards the emitter).

    Only ever used to rasterize into; the emitted basis is the PRINCIPAL one derived below, so the
    arbitrariness of the reference vector here never reaches a product.
    """
    w = np.asarray(look_dir, dtype=float)
    w = w / (np.linalg.norm(w) + 1e-30)
    reference = np.array([0.0, 0.0, 1.0]) if abs(w[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(reference, w)
    u /= (np.linalg.norm(u) + 1e-30)
    return u, np.cross(w, u), w


def rasterize(u_axis, v_axis, w_axis, vertices, faces, face_incidence, u_lin, v_lin):
    """Barycentric z-buffer over the front-facing triangles. Returns an image of face incidence,
    NaN where nothing was drawn. Depth increases towards the emitter, so the largest depth wins.

    No `color` argument: the pre-SPEC version took one and was used only by the animation, which
    made a renderer out of a physics kernel (§8 item 5). A viewer that wants to shade the mesh
    has the mesh, from P3.
    """
    projected_u = vertices @ u_axis
    projected_v = vertices @ v_axis
    depth = vertices @ w_axis
    n_u, n_v = u_lin.size, v_lin.size
    du = (u_lin[-1] - u_lin[0]) / (n_u - 1)
    dv = (v_lin[-1] - v_lin[0]) / (n_v - 1)
    pixel_x = (projected_u - u_lin[0]) / du
    pixel_y = (projected_v - v_lin[0]) / dv
    z_buffer = np.full((n_v, n_u), -np.inf)
    incidence_image = np.full((n_v, n_u), np.nan, dtype=np.float32)
    for triangle, incidence in zip(faces, face_incidence):
        if incidence <= 1e-6:
            continue                                  # back-facing → not visible
        i0, i1, i2 = triangle
        x0, y0, d0 = pixel_x[i0], pixel_y[i0], depth[i0]
        x1, y1, d1 = pixel_x[i1], pixel_y[i1], depth[i1]
        x2, y2, d2 = pixel_x[i2], pixel_y[i2], depth[i2]
        low_x = max(int(np.floor(min(x0, x1, x2))), 0)
        high_x = min(int(np.ceil(max(x0, x1, x2))), n_u - 1)
        low_y = max(int(np.floor(min(y0, y1, y2))), 0)
        high_y = min(int(np.ceil(max(y0, y1, y2))), n_v - 1)
        if low_x > high_x or low_y > high_y:
            continue                                  # triangle off-grid
        denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denominator) < 1e-12:
            continue                                  # degenerate (edge-on) projection
        grid_x, grid_y = np.meshgrid(np.arange(low_x, high_x + 1), np.arange(low_y, high_y + 1))
        bary_a = ((y1 - y2) * (grid_x - x2) + (x2 - x1) * (grid_y - y2)) / denominator
        bary_b = ((y2 - y0) * (grid_x - x2) + (x0 - x2) * (grid_y - y2)) / denominator
        bary_c = 1.0 - bary_a - bary_b
        inside = (bary_a >= 0) & (bary_b >= 0) & (bary_c >= 0)
        if not inside.any():
            continue
        rows, columns = grid_y[inside], grid_x[inside]
        interpolated_depth = (bary_a[inside] * d0 + bary_b[inside] * d1 + bary_c[inside] * d2)
        nearer = interpolated_depth > z_buffer[rows, columns]
        z_buffer[rows[nearer], columns[nearer]] = interpolated_depth[nearer]
        incidence_image[rows[nearer], columns[nearer]] = incidence
    return incidence_image


def _pin_to_airframe(axis_body):
    """Pin a principal axis's SIGN to the airframe, in body coordinates.

    `np.linalg.eigh` fixes each principal DIRECTION but not its sign, and the sign it happens to
    return depends on nothing at all — so without a pin, the frame's handedness and everything
    expressed in it flip arbitrarily between neighbouring samples. Anchor to DORSAL (−ẑ body),
    falling back to STARBOARD then NOSE for the axes perpendicular to it.

    **This is a GAUGE choice, not a continuous one, and the difference matters (D13).** The narrow
    principal axis genuinely rotates by ~90° over a pass — it is fore-aft belly-on and near-vertical
    at the horizon — so it crosses the fallback thresholds, and the pinned sign can flip there. No
    fixed 3-D reference avoids that: pinning a direction in a rotating transverse plane always has a
    degenerate locus somewhere.

    What makes it harmless is that no PHYSICAL quantity depends on the sign. The aperture is an
    ellipse and its illumination is even, so the field is invariant under x̂ → −x̂; the only asymmetry
    is the pointing offset, and mirroring it is the same worst case. `test_capture_is_invariant_to_the
    _frame_gauge` asserts exactly that. A viewer that wants a CONTINUOUS drawn orientation re-pins
    along the trajectory using the emitted per-sample frames — which is a decision about which way up
    to draw, i.e. presentation, and belongs to the viewer (I7).
    """
    candidates = (-axis_body[2], axis_body[1], axis_body[0])         # dorsal, starboard, nose
    reference = next((c for c in candidates if abs(c) > 0.1), 1.0)
    return axis_body if reference >= 0.0 else -axis_body


def _principal_frame(offsets_u, offsets_v, weights, aux_u, aux_v, look_dir):
    """The silhouette's own transverse frame, in body coordinates.

    Returns (element x, element y, basis) where `basis` has rows (x̂, ŷ, ẑ) with ẑ the PROPAGATION
    direction (emitter → aircraft, i.e. −look), x̂ the silhouette's least-spread ("narrow")
    principal axis sign-pinned to the airframe, and ŷ = ẑ × x̂ so the frame is right-handed.

    x̂ is the narrow axis because that is the direction the aperture is stretched ALONG: spot width
    goes as 1/aperture width, so a target that is narrow on x̂ wants the aperture long on x̂. The
    aperture frame and the silhouette frame are therefore the same frame, and no product anywhere
    has to convert between two of them.
    """
    if offsets_u.size >= 3:
        covariance = np.cov(np.vstack([offsets_u, offsets_v]),
                            aweights=np.maximum(weights, 1e-12))
        _, eigenvectors = np.linalg.eigh(covariance)        # ascending: column 0 is least-spread
        narrow_in_plane = eigenvectors[:, 0]
    else:
        narrow_in_plane = np.array([1.0, 0.0])
    narrow_body = narrow_in_plane[0] * aux_u + narrow_in_plane[1] * aux_v
    x_axis = _pin_to_airframe(narrow_body / (np.linalg.norm(narrow_body) + 1e-30))
    propagation = -np.asarray(look_dir, dtype=float)
    propagation = propagation / (np.linalg.norm(propagation) + 1e-30)
    x_axis = x_axis - np.dot(x_axis, propagation) * propagation
    x_axis /= (np.linalg.norm(x_axis) + 1e-30)
    y_axis = np.cross(propagation, x_axis)
    element_x = offsets_u * np.dot(aux_u, x_axis) + offsets_v * np.dot(aux_v, x_axis)
    element_y = offsets_u * np.dot(aux_u, y_axis) + offsets_v * np.dot(aux_v, y_axis)
    return element_x, element_y, np.stack([x_axis, y_axis, propagation])


_SCHEMA_VERSION = 2
"""Bumped whenever the set of arrays a cached raster carries changes.

Part of the key, because "explicitly keyed" (D4) has to mean keyed on the OUTPUT schema as well as
on the inputs: an entry written before a field was added is not a hit for a caller that needs it, and
the failure mode without this is a KeyError deep inside a batched gather rather than a recompute."""


def raster_key(loft: LoftSpec, beta_deg: float, alpha_deg: float, config: FidelityConfig) -> str:
    return cache.content_key(
        "silhouette", schema=_SCHEMA_VERSION,
        mesh=loft_module.mesh_key(loft, config.mesh_n_span, config.mesh_n_theta),
        beta=float(beta_deg), alpha=float(alpha_deg), n_grid=int(config.raster_n_grid))


def bin_look_angles(beta_deg, alpha_deg, bin_deg: float):
    """Quantise arrival angles to the cache bin, and give the unique ones and an index into them.

    This is the second of the two free speedups (§10 item 1): the silhouette changes slowly with
    arrival angle, so a 0.5° bin turns a 241-sample pass into ~40 rasters. Unlike the pre-SPEC
    rounding, the bin size is a recorded fidelity parameter and it is part of the cache key, so a
    run at a different bin cannot read another bin's entries.
    """
    beta = np.round(np.asarray(beta_deg, dtype=float) / bin_deg) * bin_deg
    alpha = np.round(np.asarray(alpha_deg, dtype=float) / bin_deg) * bin_deg
    pairs = np.stack([beta.ravel(), alpha.ravel()], axis=1)
    unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
    return unique, inverse.reshape(np.shape(beta))


def silhouette_at(loft: LoftSpec, beta_deg: float, alpha_deg: float, config: FidelityConfig):
    """One arrival angle's visible-surface element cloud, in the silhouette's own frame.

    Keys are content keys over (loft, angle, grid), so nothing about this depends on process
    state (D4). Returns a dict of arrays; `element_x`/`element_y` are offsets from the
    projected-area centroid in the frame `basis` describes.
    """
    def build():
        vertices, faces, normals = loft_module.mesh(loft, config)
        look = loft_module.direction_from_beta_alpha(float(beta_deg), float(alpha_deg))
        look = look / np.linalg.norm(look)
        aux_u, aux_v, aux_w = _auxiliary_basis(look)
        face_incidence = normals @ aux_w
        front = face_incidence > 1e-6
        if not front.any():
            return _empty_raster(look)
        front_vertices = np.unique(faces[front])
        projected_u = vertices @ aux_u
        projected_v = vertices @ aux_v
        u_mid = 0.5 * (projected_u[front_vertices].min() + projected_u[front_vertices].max())
        v_mid = 0.5 * (projected_v[front_vertices].min() + projected_v[front_vertices].max())
        half = 0.5 * 1.06 * max(np.ptp(projected_u[front_vertices]),
                                np.ptp(projected_v[front_vertices]), 1e-6)
        u_lin = np.linspace(u_mid - half, u_mid + half, config.raster_n_grid)
        v_lin = np.linspace(v_mid - half, v_mid + half, config.raster_n_grid)
        image = rasterize(aux_u, aux_v, aux_w, vertices, faces, face_incidence, u_lin, v_lin)
        covered = np.isfinite(image)
        if not covered.any():
            return _empty_raster(look)
        du = (u_lin[-1] - u_lin[0]) / (config.raster_n_grid - 1)
        dv = (v_lin[-1] - v_lin[0]) / (config.raster_n_grid - 1)
        grid_u, grid_v = np.meshgrid(u_lin, v_lin)
        offsets_u, offsets_v = grid_u[covered], grid_v[covered]
        incidence = image[covered].astype(float)
        projected_area = np.full(offsets_u.size, du * dv)
        centroid_u, centroid_v = float(offsets_u.mean()), float(offsets_v.mean())
        offsets_u = offsets_u - centroid_u
        offsets_v = offsets_v - centroid_v
        element_x, element_y, basis = _principal_frame(offsets_u, offsets_v, projected_area,
                                                       aux_u, aux_v, look)
        weight = projected_area / max(projected_area.sum(), 1e-12)
        sigma = np.sqrt([float(np.sum(weight * element_x ** 2)),
                         float(np.sum(weight * element_y ** 2))])
        # The THIRD moment is what makes the pointing worst case well posed (D13). The offset is
        # applied along x̂, whose sign is a gauge choice, so a SIGNED offset would be one arbitrary
        # side of two that differ by ~7 pp of capture on this loft. Skew flips sign with the gauge,
        # so choosing the direction from it gives the same PHYSICAL offset either way — and points it
        # at the thin side of the projection, which is the direction that actually costs more.
        skew = np.array([float(np.sum(weight * element_x ** 3)),
                         float(np.sum(weight * element_y ** 3))])
        return dict(
            element_x=element_x, element_y=element_y, incidence=incidence,
            projected_area=projected_area,
            area=np.array(float(projected_area.sum())),
            basis=basis,
            half_extent=np.array([float(np.abs(element_x).max()),
                                  float(np.abs(element_y).max())]),
            sigma=sigma, skew=skew,
            look=look)

    return cache.memoized(raster_key(loft, beta_deg, alpha_deg, config), build)


def _empty_raster(look):
    """A look angle at which no skin faces the emitter. Legal, and reported as zero area."""
    aux_u, aux_v, _ = _auxiliary_basis(look)
    propagation = -np.asarray(look, dtype=float)
    return dict(element_x=np.zeros(0), element_y=np.zeros(0), incidence=np.zeros(0),
                projected_area=np.zeros(0), area=np.array(0.0),
                basis=np.stack([aux_u, aux_v, propagation]),
                half_extent=np.zeros(2), sigma=np.zeros(2), skew=np.zeros(2),
                look=np.asarray(look, dtype=float))


def element_cloud(loft: LoftSpec, beta_deg, alpha_deg, config: FidelityConfig):
    """Batched, padded element clouds for a whole (candidate, sample) grid of arrival angles.

    Ragged inner axes need a padding convention (§6a): the element cloud has a different size at
    every look angle, so batching across samples means padding plus a validity mask — the same
    device P5 uses for unresolved field regions, applied in the schema generally.
    """
    beta = np.asarray(beta_deg, dtype=float)
    unique_angles, index = bin_look_angles(beta, alpha_deg, config.look_angle_bin_deg)
    rasters = [silhouette_at(loft, float(b), float(a), config) for b, a in unique_angles]
    max_elements = max((r["element_x"].size for r in rasters), default=1)
    max_elements = max(max_elements, 1)

    shape = beta.shape
    offsets = np.zeros(shape + (max_elements, 2))
    projected_area = np.zeros(shape + (max_elements,))
    incidence = np.zeros(shape + (max_elements,))
    valid = np.zeros(shape + (max_elements,), dtype=bool)
    area = np.zeros(shape)
    basis = np.zeros(shape + (3, 3))
    half_extent = np.zeros(shape + (2,))
    sigma = np.zeros(shape + (2,))
    skew = np.zeros(shape + (2,))

    for flat_index in np.ndindex(shape):
        raster = rasters[int(index[flat_index])]
        count = raster["element_x"].size
        offsets[flat_index + (slice(0, count), 0)] = raster["element_x"]
        offsets[flat_index + (slice(0, count), 1)] = raster["element_y"]
        projected_area[flat_index + (slice(0, count),)] = raster["projected_area"]
        incidence[flat_index + (slice(0, count),)] = raster["incidence"]
        valid[flat_index + (slice(0, count),)] = True
        area[flat_index] = float(raster["area"])
        basis[flat_index] = raster["basis"]
        half_extent[flat_index] = raster["half_extent"]
        sigma[flat_index] = raster["sigma"]
        skew[flat_index] = raster["skew"]
    return dict(offsets_m=offsets, projected_area_m2=projected_area, incidence_cos=incidence,
                valid=valid, area_m2=area, basis_body=basis, half_extent_m=half_extent,
                sigma_m=sigma, skew_m3=skew, n_rasters=len(rasters))


def max_half_extent(loft: LoftSpec, config: FidelityConfig) -> float:
    """Largest silhouette half-extent over a full arrival-angle sweep.

    A genuine airframe property, not a plot window: the same fixed value for every run of the same
    loft, which is what makes silhouettes from different runs comparable (§8 item 19).
    """
    def build():
        extent = 0.0
        for beta in range(0, 181, 10):
            for alpha in range(0, 360, 30):
                raster = silhouette_at(loft, float(beta), float(alpha), config)
                extent = max(extent, float(raster["half_extent"].max()))
        return dict(max_half_extent=np.array(extent))

    key = cache.content_key(
        "max_half_extent", schema=_SCHEMA_VERSION,
        mesh=loft_module.mesh_key(loft, config.mesh_n_span, config.mesh_n_theta),
        n_grid=config.raster_n_grid)
    return float(cache.memoized(key, build)["max_half_extent"])


# --- the convex, occlusion-free table -----------------------------------------------------


def convex_area_table(loft: LoftSpec, config: FidelityConfig, beta_step_deg: float = 2.0,
                      alpha_step_deg: float = 5.0):
    """Silhouette area on a (β, α) grid by the convex projected-area identity.

    Occlusion-free and fully vectorized, which is why §6 keeps it: it is the closed-form capture
    tier's receiver area and the alternative to batching the z-buffer. It OVER-counts a non-convex
    body — measured at >7 % side-on at low elevation — and the tier calibration publishes that.
    """
    def build():
        vertices, faces, normals, areas = loft_module.mesh_with_areas(loft, config)
        beta = np.arange(0.0, 180.0 + 1e-9, beta_step_deg)
        alpha = np.arange(0.0, 360.0 + 1e-9, alpha_step_deg)
        beta_grid, alpha_grid = np.meshgrid(beta, alpha, indexing="ij")
        directions = loft_module.direction_from_beta_alpha(beta_grid, alpha_grid)
        table = loft_module.convex_projected_area(directions, normals, areas)
        planform = loft_module.convex_projected_area(np.array([0.0, 0.0, 1.0]), normals, areas)
        return dict(beta_deg=beta, alpha_deg=alpha, area_m2=table,
                    planform_area_m2=np.array(float(planform)))

    key = cache.content_key(
        "convex_area", mesh=loft_module.mesh_key(loft, config.mesh_n_span, config.mesh_n_theta),
        beta_step=float(beta_step_deg), alpha_step=float(alpha_step_deg))
    return cache.memoized(key, build)


def convex_area(loft: LoftSpec, config: FidelityConfig, beta_deg, alpha_deg):
    """Bilinear lookup into `convex_area_table`, vectorized over a batch of arrival angles."""
    table = convex_area_table(loft, config)
    beta_axis, alpha_axis, areas = table["beta_deg"], table["alpha_deg"], table["area_m2"]
    beta = np.clip(np.asarray(beta_deg, dtype=float), 0.0, 180.0)
    alpha = np.mod(np.asarray(alpha_deg, dtype=float), 360.0)
    bi = np.clip(np.searchsorted(beta_axis, beta) - 1, 0, beta_axis.size - 2)
    ai = np.clip(np.searchsorted(alpha_axis, alpha) - 1, 0, alpha_axis.size - 2)
    beta_low, beta_high = beta_axis[bi], beta_axis[bi + 1]
    alpha_low, alpha_high = alpha_axis[ai], alpha_axis[ai + 1]
    tb = np.where(beta_high > beta_low, (beta - beta_low) / (beta_high - beta_low), 0.0)
    ta = np.where(alpha_high > alpha_low, (alpha - alpha_low) / (alpha_high - alpha_low), 0.0)
    return ((1 - tb) * (1 - ta) * areas[bi, ai] + (1 - tb) * ta * areas[bi, ai + 1]
            + tb * (1 - ta) * areas[bi + 1, ai] + tb * ta * areas[bi + 1, ai + 1])


def equal_area_radius(area_m2):
    """Radius of the circle of the same area — what the closed-form coupling relations take as
    the receiver."""
    return np.sqrt(np.maximum(np.asarray(area_m2, dtype=float), 0.0) / np.pi)
