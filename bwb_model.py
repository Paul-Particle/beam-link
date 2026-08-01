"""Parametric blended-wing-body geometry and its silhouette area vs viewing direction.

Body frame throughout: +x nose, +y starboard, +z belly (so "down" for level flight). A
look direction is given as (beta, alpha): beta is the angle off the belly axis (0 = straight
below the aircraft) and alpha the azimuth about it, from the nose towards starboard.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


# --- Parametric mesh ----------------------------------------------------------

def _cuboid(center, size):
    """Dead: no caller anywhere in the tree. Only its 12-triangle face table survives, copied
    into `_winglet_prism`."""
    cx, cy, cz = center
    sx, sy, sz = np.array(size) / 2.0
    v = np.array([[cx - sx, cy - sy, cz - sz], [cx + sx, cy - sy, cz - sz],
                  [cx + sx, cy + sy, cz - sz], [cx - sx, cy + sy, cz - sz],
                  [cx - sx, cy - sy, cz + sz], [cx + sx, cy - sy, cz + sz],
                  [cx + sx, cy + sy, cz + sz], [cx - sx, cy + sy, cz + sz]])
    f = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1],
                  [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]])
    return v, f


def _winglet_prism(tip_center_x, y_tip, root_chord, tip_chord, height,
                   fin_sweep, cant_deg, thickness):
    """Flat tapered fin at a wing tip, canted outboard by cant_deg and rising away from the
    belly (-z), swept aft by fin_sweep. Same 12-triangle topology as `_cuboid`."""
    side = 1.0 if y_tip >= 0 else -1.0
    cant = np.radians(cant_deg)
    rise = np.array([0.0, side * np.sin(cant), -np.cos(cant)])       # up-and-outboard
    root_center = np.array([tip_center_x, y_tip, 0.0])
    tip_center = root_center + height * rise + np.array([-fin_sweep, 0.0, 0.0])
    chord = np.array([1.0, 0.0, 0.0])
    r_le, r_te = root_center + chord * root_chord / 2, root_center - chord * root_chord / 2
    t_le, t_te = tip_center + chord * tip_chord / 2, tip_center - chord * tip_chord / 2
    normal = np.cross(chord, rise); normal /= np.linalg.norm(normal)  # ≈ spanwise (thin)
    off = normal * thickness / 2.0
    quad = np.array([r_le, r_te, t_te, t_le])
    v = np.vstack([quad - off, quad + off])
    f = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1],
                  [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]])
    return v, f


def make_bwb_mesh(full_span_m=80.0, root_chord_m=40.0, tip_chord_m=3.0,
                  root_thickness_m=6.0, tip_thickness_m=0.4, sweep_m=25.0,
                  body_fraction=0.25, n_span=71, n_theta=40,
                  winglet_height_m=3.0, winglet_root_chord_m=3.0, winglet_tip_chord_m=1.2,
                  winglet_sweep_m=1.5, winglet_cant_deg=20.0, winglet_thickness_m=0.3):
    """Closed triangle mesh: elliptical cross-sections lofted along the span.

    Chord and thickness are constant over the centre body (|u| < body_fraction of the
    half-span) and taper linearly to the tip beyond it; section centres move aft with |u|
    to give the sweep. Returns (vertices Nx3 m, faces Mx3 indices).
    """
    half_span = full_span_m / 2.0
    u = np.linspace(-1.0, 1.0, n_span)              # spanwise fraction (−1..1)
    au = np.abs(u)
    taper = np.clip((au - body_fraction) / (1.0 - body_fraction), 0.0, 1.0)
    chord = root_chord_m - (root_chord_m - tip_chord_m) * taper
    thickness = root_thickness_m - (root_thickness_m - tip_thickness_m) * taper
    y_center = -sweep_m * au                          # nose forward at centre, tips aft
    x_span = u * half_span

    psi = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)  # around each section
    cos_psi, sin_psi = np.cos(psi), np.sin(psi)

    # Vertex grid: (n_span, n_theta, 3) in body axes x̂=nose, ŷ=starboard, ẑ=vertical.
    verts = np.empty((n_span, n_theta, 3))
    verts[..., 0] = y_center[:, None] + (chord[:, None] / 2.0) * cos_psi[None, :]   # nose-tail
    verts[..., 1] = x_span[:, None]                                                 # span
    verts[..., 2] = (thickness[:, None] / 2.0) * sin_psi[None, :]                   # vertical

    vertices = verts.reshape(-1, 3)

    def vid(i, j):
        return i * n_theta + (j % n_theta)

    faces = []
    for i in range(n_span - 1):
        for j in range(n_theta):
            a, b, c, d = vid(i, j), vid(i, j + 1), vid(i + 1, j + 1), vid(i + 1, j)
            faces.append((a, b, c))
            faces.append((a, c, d))

    for i in (0, n_span - 1):  # cap both wing tips with a fan, wound to face outboard
        ring = verts[i]
        center = ring.mean(axis=0)
        c_idx = len(vertices)
        vertices = np.vstack([vertices, center])
        for j in range(n_theta):
            a, b = vid(i, j), vid(i, j + 1)
            faces.append((c_idx, a, b) if i == 0 else (c_idx, b, a))

    faces = np.asarray(faces, dtype=int)

    # Wingtip winglets: realistic tapered/swept/canted dorsal fins. They present some area
    # to a SIDE aspect at low elevation, but are modest — the large side-area radiator is
    # the (future) sail / vertical stabiliser, added separately.
    if winglet_height_m > 0:
        tip_x = -sweep_m                                   # nose-tail centre of the swept tip
        for y_tip in (-full_span_m / 2.0, full_span_m / 2.0):
            wv, wf = _winglet_prism(tip_x, y_tip, winglet_root_chord_m, winglet_tip_chord_m,
                                    winglet_height_m, winglet_sweep_m, winglet_cant_deg,
                                    winglet_thickness_m)
            faces = np.vstack([faces, wf + len(vertices)])
            vertices = np.vstack([vertices, wv])

    return vertices, faces


def _connected_components(vertices, faces):
    """Union-find label per vertex. Needed because the mesh is three disjoint bodies
    (centre body plus two winglets) and each must be oriented about its own centroid."""
    parent = np.arange(len(vertices))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, c in faces:
        ra, rb, rc = find(a), find(b), find(c)
        if ra != rb:
            parent[rb] = ra
        if ra != rc:
            parent[find(c)] = ra
    roots = np.array([find(i) for i in range(len(vertices))])
    _, comp = np.unique(roots, return_inverse=True)
    return comp


def triangle_normals_areas(vertices, faces):
    """Per-face unit normals (forced outward) and areas in m^2, ignoring winding order."""
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    normal = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + 1e-30)

    comp = _connected_components(vertices, faces)
    comp_sum = np.zeros((comp.max() + 1, 3))
    np.add.at(comp_sum, comp, vertices)
    comp_centroid = comp_sum / np.bincount(comp)[:, None]
    ref = comp_centroid[comp[faces[:, 0]]]                 # each face's own-part centroid

    tri_centroid = (v0 + v1 + v2) / 3.0
    outward = np.sum(normal * (tri_centroid - ref), axis=1) < 0  # i.e. currently inward
    normal[outward] *= -1.0
    return normal, area


def projected_area(view_dirs, normals, areas):
    """Sum of area*max(n.v, 0) in m^2 -- the silhouette area for a convex closed body.
    Self-occlusion is not modelled, so concave shadowing is overcounted (see the
    z-buffer rasterisers below)."""
    view = np.asarray(view_dirs, dtype=float)
    view = view / (np.linalg.norm(view, axis=-1, keepdims=True) + 1e-30)
    dots = view @ normals.T                              # (..., M)
    return np.sum(np.clip(dots, 0.0, None) * areas, axis=-1)


# --- Body-frame angles ↔ direction --------------------------------------------

def direction_from_beta_alpha(beta_deg, alpha_deg):
    """Unit beam-arrival direction in body frame from (β from belly +ẑ, α from nose)."""
    b, a = np.radians(beta_deg), np.radians(alpha_deg)
    return np.stack([np.sin(b) * np.cos(a),      # nose  (x̂)
                     np.sin(b) * np.sin(a),      # starboard (ŷ)
                     np.cos(b)], axis=-1)         # belly (ẑ)


def silhouette_aspect(beta_deg, alpha_deg, mesh_kwargs=None):
    """(major/minor, major, minor) extents in m of the outline projected on the view plane,
    measured along the principal axes of the projected vertex cloud."""
    verts, _ = make_bwb_mesh(**(mesh_kwargs or {}))
    d = direction_from_beta_alpha(float(beta_deg), float(alpha_deg))
    d = d / np.linalg.norm(d)
    ref = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(d, ref); e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    pts = np.stack([verts @ e1, verts @ e2], axis=1)
    pts = pts - pts.mean(0)
    eigval, eigvec = np.linalg.eigh(np.cov(pts.T))
    extents = np.ptp(pts @ eigvec, axis=0)
    minor, major = np.sort(extents)
    return major / max(minor, 1e-6), major, minor


# --- Occlusion-aware projection (rasterized silhouette + incidence) -----------
# The convex identity above OVER-counts a non-convex body (a winglet behind the
# body, a far wing seen edge-on) because it has no occlusion test. For the honest
# spot-in-silhouette overlap we rasterize the ACTUAL projection: sample the surface
# densely, project onto the view plane, and keep — per grid cell — only the sample
# nearest the emitter (a z-buffer). That gives the true (self-occluded) outline AND
# the incidence angle of the skin the beam actually reaches, cell by cell.

def surface_samples(mesh_kwargs=None, target_points=120_000, seed=0):
    """Area-weighted uniform point cloud over the skin: (points, per-point normal,
    per-point surface area m^2). Weights sum to the total wetted area."""
    vertices, faces = make_bwb_mesh(**(mesh_kwargs or {}))
    normals, areas = triangle_normals_areas(vertices, faces)
    counts = np.maximum(1, np.round(target_points * areas / areas.sum()).astype(int))
    rng = np.random.default_rng(seed)
    pts, nrm, wts = [], [], []
    for tri, n, normal, area in zip(faces, counts, normals, areas):
        v0, v1, v2 = vertices[tri]
        r1, r2 = rng.random(n), rng.random(n)
        su = np.sqrt(r1)                              # uniform barycentric sampling
        a, b, c = 1.0 - su, su * (1.0 - r2), su * r2
        pts.append(a[:, None] * v0 + b[:, None] * v1 + c[:, None] * v2)
        nrm.append(np.broadcast_to(normal, (n, 3)))
        wts.append(np.full(n, area / n))
    return np.vstack(pts), np.vstack(nrm), np.concatenate(wts)


def _view_basis(look_dir):
    """Orthonormal (u, v, w) with w along the look direction, i.e. towards the emitter, so
    that depth along w increases towards it. u, v span the illuminated plane."""
    w = np.asarray(look_dir, dtype=float)
    w = w / (np.linalg.norm(w) + 1e-30)
    ref = np.array([0.0, 0.0, 1.0]) if abs(w[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(ref, w); u /= (np.linalg.norm(u) + 1e-30)
    v = np.cross(w, u)
    return u, v, w


def rasterize_silhouette(look_dir, samples, sample_normals, sample_weights,
                         n_grid=150, depth_tol_m=1.0):
    """Visible-surface point cloud as seen from `look_dir`, occlusion by depth-sorting the
    surface samples into view-plane cells.

    Returns the receiver-element description consumed by beam_link.overlap_capture:
      u, v         in-plane offsets (m) from the projected-area centroid
      incidence    cos of the angle between the local normal and the look direction
      proj_area    area (m^2) each element presents to the beam, i.e. skin area * incidence
      area         sum of proj_area == illuminated silhouette area (m^2)
    """
    u_ax, v_ax, w_ax = _view_basis(look_dir)
    incidence = sample_normals @ w_ax
    front = incidence > 1e-6                          # skin facing the emitter
    U = samples[front] @ u_ax
    V = samples[front] @ v_ax
    depth = samples[front] @ w_ax                     # larger = nearer the emitter
    inc = incidence[front]
    wt = sample_weights[front]
    if U.size == 0:
        return dict(u=np.array([]), v=np.array([]), incidence=np.array([]),
                    proj_area=np.array([]), area=0.0, n_grid=n_grid)

    u_lo, u_hi, v_lo, v_hi = U.min(), U.max(), V.min(), V.max()
    span_u, span_v = max(u_hi - u_lo, 1e-6), max(v_hi - v_lo, 1e-6)
    iu = np.clip(((U - u_lo) / span_u * (n_grid - 1)).astype(int), 0, n_grid - 1)
    iv = np.clip(((V - v_lo) / span_v * (n_grid - 1)).astype(int), 0, n_grid - 1)
    cell = iu * n_grid + iv

    cell_max_depth = np.full(n_grid * n_grid, -np.inf)
    np.maximum.at(cell_max_depth, cell, depth)
    # keep whatever is within depth_tol_m of the nearest-to-emitter sample in its cell
    visible = depth >= cell_max_depth[cell] - depth_tol_m

    U, V, inc, proj = U[visible], V[visible], inc[visible], (wt[visible] * inc[visible])
    cu = np.sum(U * proj) / max(proj.sum(), 1e-12)
    cv = np.sum(V * proj) / max(proj.sum(), 1e-12)
    return dict(u=U - cu, v=V - cv, incidence=inc, proj_area=proj,
                area=float(proj.sum()), n_grid=n_grid)


def rasterize_view_zbuffer(u_ax, v_ax, w_ax, vertices, faces, face_incidence,
                           u_lin, v_lin, color=None):
    """Barycentric z-buffer rasteriser over the front-facing triangles. Returns an image of
    face incidence (or of `color`), NaN where nothing was drawn. Depth increases towards the
    emitter, so the largest depth wins."""
    U = vertices @ u_ax
    V = vertices @ v_ax
    depth = vertices @ w_ax                              # larger = nearer the emitter
    stored = face_incidence if color is None else np.asarray(color)
    n_u, n_v = u_lin.size, v_lin.size
    du = (u_lin[-1] - u_lin[0]) / (n_u - 1)
    dv = (v_lin[-1] - v_lin[0]) / (n_v - 1)
    px = (U - u_lin[0]) / du                             # fractional pixel coords of each vertex
    py = (V - v_lin[0]) / dv
    zbuf = np.full((n_v, n_u), -np.inf)
    incid = np.full((n_v, n_u), np.nan, dtype=np.float32)
    for tri, inc, col in zip(faces, face_incidence, stored):
        if inc <= 1e-6:
            continue                                     # back-facing (to the cull axis) → not visible
        i0, i1, i2 = tri
        x0, y0, d0 = px[i0], py[i0], depth[i0]
        x1, y1, d1 = px[i1], py[i1], depth[i1]
        x2, y2, d2 = px[i2], py[i2], depth[i2]
        lo_x = max(int(np.floor(min(x0, x1, x2))), 0)
        hi_x = min(int(np.ceil(max(x0, x1, x2))), n_u - 1)
        lo_y = max(int(np.floor(min(y0, y1, y2))), 0)
        hi_y = min(int(np.ceil(max(y0, y1, y2))), n_v - 1)
        if lo_x > hi_x or lo_y > hi_y:
            continue                                     # triangle off-grid
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-12:
            continue                                     # degenerate (edge-on) projection
        gx, gy = np.meshgrid(np.arange(lo_x, hi_x + 1), np.arange(lo_y, hi_y + 1))
        a = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / denom
        b = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / denom
        c = 1.0 - a - b
        inside = (a >= 0) & (b >= 0) & (c >= 0)
        if not inside.any():
            continue
        gyi, gxi = gy[inside], gx[inside]
        di = a[inside] * d0 + b[inside] * d1 + c[inside] * d2
        win = di > zbuf[gyi, gxi]                         # nearer than the current occupant
        wy, wx = gyi[win], gxi[win]
        zbuf[wy, wx] = di[win]
        incid[wy, wx] = col
    return incid


def rasterize_silhouette_zbuffer(look_dir, vertices, faces, face_normals, n_grid=200, margin=1.06):
    """Same output schema as `rasterize_silhouette`, but from the mesh instead of a point
    cloud: each covered pixel is one receiver element of area du*dv, already a projected
    area, and `area` is the pixel-counted silhouette area. The square window is fitted to
    the front-facing vertices with `margin` slack."""
    u_ax, v_ax, w_ax = _view_basis(look_dir)
    U = vertices @ u_ax
    V = vertices @ v_ax
    face_inc = face_normals @ w_ax
    front = face_inc > 1e-6
    empty = dict(u=np.array([]), v=np.array([]), incidence=np.array([]),
                 proj_area=np.array([]), area=0.0, n_grid=n_grid)
    if not front.any():
        return empty
    fv = np.unique(faces[front])
    u_mid, v_mid = 0.5 * (U[fv].min() + U[fv].max()), 0.5 * (V[fv].min() + V[fv].max())
    half = 0.5 * margin * max(U[fv].max() - U[fv].min(), V[fv].max() - V[fv].min(), 1e-6)
    u_lin = np.linspace(u_mid - half, u_mid + half, n_grid)
    v_lin = np.linspace(v_mid - half, v_mid + half, n_grid)
    incid = rasterize_view_zbuffer(u_ax, v_ax, w_ax, vertices, faces, face_inc, u_lin, v_lin)
    covered = np.isfinite(incid)
    if not covered.any():
        return empty
    du = (u_lin[-1] - u_lin[0]) / (n_grid - 1)
    dv = (v_lin[-1] - v_lin[0]) / (n_grid - 1)
    uu, vv = np.meshgrid(u_lin, v_lin)
    u_c, v_c, inc_c = uu[covered], vv[covered], incid[covered].astype(float)
    cu, cv = float(u_c.mean()), float(v_c.mean())          # projected-area centroid (uniform dA)
    return dict(u=u_c - cu, v=v_c - cv, incidence=inc_c,
                proj_area=np.full(u_c.size, du * dv), area=float(covered.sum() * du * dv),
                n_grid=n_grid)


# Cached mesh (verts, faces, per-face normals) for the default airframe — used by the z-buffer
# silhouette rasterizer (the capture path). Separate from the point cloud `_SAMPLES`.
_MESH = None


def get_mesh(mesh_kwargs=None):
    """Cached (verts, faces, normals) at the FULL body loft (71×40, `make_bwb_mesh` default). This
    is the capture-path mesh AND the panel-A display mesh — they are deliberately the SAME so the
    on-screen silhouette is a faithful picture of exactly the surface the capture integral sums over
    (integration cells = displayed pixels). The silhouette AREA is grid-limited, not mesh-limited, so
    the finer body loft is cosmetic for capture magnitude; it is worth the ~2.6× rasterization cost
    to keep the displayed and integrated silhouettes identical (winglets/edges at their true width)."""
    global _MESH
    if _MESH is None or mesh_kwargs is not None:
        v, f = make_bwb_mesh(**(mesh_kwargs or {}))
        n, _ = triangle_normals_areas(v, f)
        _MESH = (v, f, n)
    return _MESH


# Cached sample cloud for the default airframe.
_SAMPLES = None


def get_surface_samples(mesh_kwargs=None):
    global _SAMPLES
    if _SAMPLES is None or mesh_kwargs is not None:
        _SAMPLES = surface_samples(mesh_kwargs)
    return _SAMPLES


# --- Lookup table -------------------------------------------------------------

_TABLE_PATH = Path(__file__).with_name("data") / "bwb_silhouette.npz"


def build_table(beta_step_deg=2.0, alpha_step_deg=5.0, mesh_kwargs=None, save=True):
    """Precompute silhouette area (m^2) on a (beta, alpha) grid, plus the belly-on planform
    area. Convex projected area only -- no occlusion, unlike the rasterisers."""
    vertices, faces = make_bwb_mesh(**(mesh_kwargs or {}))
    normals, areas = triangle_normals_areas(vertices, faces)
    beta = np.arange(0.0, 180.0 + 1e-9, beta_step_deg)
    alpha = np.arange(0.0, 360.0 + 1e-9, alpha_step_deg)
    bb, aa = np.meshgrid(beta, alpha, indexing="ij")
    dirs = direction_from_beta_alpha(bb, aa)             # (Nb, Na, 3)
    table = projected_area(dirs, normals, areas)         # (Nb, Na)
    planform = projected_area(np.array([0.0, 0.0, 1.0]), normals, areas)
    if save:
        _TABLE_PATH.parent.mkdir(exist_ok=True)
        np.savez_compressed(_TABLE_PATH, beta_deg=beta, alpha_deg=alpha,
                            area_m2=table, planform_area_m2=np.array(planform),
                            mesh_kwargs=np.array(str(mesh_kwargs or {})))
    return beta, alpha, table, planform


def load_table():
    if not _TABLE_PATH.exists():
        build_table()
    d = np.load(_TABLE_PATH, allow_pickle=True)
    return d["beta_deg"], d["alpha_deg"], d["area_m2"], float(d["planform_area_m2"])


class BWBSilhouette:
    """Fast vectorized silhouette lookup with numpy bilinear interpolation."""

    def __init__(self):
        self.beta, self.alpha, self.table, self.planform_area = load_table()

    def area(self, beta_deg, alpha_deg):
        """Interpolated silhouette area in m^2. The beta fold is defensive only: every in-tree
        caller derives beta from an arccos and is therefore already in 0..180, so the fold --
        and the alpha flip a true polar wrap would also need, which it omits -- never fires."""
        beta_deg = np.mod(np.asarray(beta_deg, dtype=float), 360.0)
        beta_deg = np.where(beta_deg > 180.0, 360.0 - beta_deg, beta_deg)   # fold to 0..180
        alpha_deg = np.mod(np.asarray(alpha_deg, dtype=float), 360.0)
        return self._bilinear(beta_deg, alpha_deg)

    def _bilinear(self, beta, alpha):
        bi = np.clip(np.searchsorted(self.beta, beta) - 1, 0, self.beta.size - 2)
        ai = np.clip(np.searchsorted(self.alpha, alpha) - 1, 0, self.alpha.size - 2)
        b0, b1 = self.beta[bi], self.beta[bi + 1]
        a0, a1 = self.alpha[ai], self.alpha[ai + 1]
        tb = np.where(b1 > b0, (beta - b0) / (b1 - b0), 0.0)
        ta = np.where(a1 > a0, (alpha - a0) / (a1 - a0), 0.0)
        f00 = self.table[bi, ai]; f01 = self.table[bi, ai + 1]
        f10 = self.table[bi + 1, ai]; f11 = self.table[bi + 1, ai + 1]
        return ((1 - tb) * (1 - ta) * f00 + (1 - tb) * ta * f01
                + tb * (1 - ta) * f10 + tb * ta * f11)


# --- Attitude: world (ENU) beam direction → body-frame (β, α) ------------------

def _rodrigues(vec, axis, angle_rad):
    """Rotate `vec` about unit `axis` by `angle_rad` (Rodrigues)."""
    axis = axis / (np.linalg.norm(axis) + 1e-30)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return vec * c + np.cross(axis, vec) * s + axis * np.dot(axis, vec) * (1 - c)


def body_axes_enu(heading_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
    """Body axes expressed in local ENU. Heading is measured from north towards east;
    positive pitch is nose-up (about starboard), positive roll is right-wing-down
    (about the nose), applied in that order."""
    psi = np.radians(heading_deg)
    nose = np.array([np.sin(psi), np.cos(psi), 0.0])
    star = np.array([np.cos(psi), -np.sin(psi), 0.0])
    belly = np.array([0.0, 0.0, -1.0])
    nose = _rodrigues(nose, star, np.radians(pitch_deg))
    belly = _rodrigues(belly, star, np.radians(pitch_deg))
    star = _rodrigues(star, nose, np.radians(roll_deg))
    belly = _rodrigues(belly, nose, np.radians(roll_deg))
    return nose, star, belly


def beta_alpha_from_enu(dir_enu, heading_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
    """ENU direction (aircraft -> emitter) to body-frame (beta, alpha) in degrees."""
    nose, star, belly = body_axes_enu(heading_deg, pitch_deg, roll_deg)
    d = np.asarray(dir_enu, dtype=float)
    d = d / (np.linalg.norm(d, axis=-1, keepdims=True) + 1e-30)
    dx, dy, dz = d @ nose, d @ star, d @ belly           # body components
    beta = np.degrees(np.arccos(np.clip(dz, -1.0, 1.0)))
    alpha = np.degrees(np.arctan2(dy, dx)) % 360.0
    return beta, alpha


# --- Figure -------------------------------------------------------------------

def silhouette_figure(path="out/bwb_silhouette.html"):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    vertices, faces = make_bwb_mesh()
    normals, areas = triangle_normals_areas(vertices, faces)
    beta, alpha, table, planform = build_table(save=False)

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.45, 0.55],
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        subplot_titles=(f"BWB mesh (planform silhouette {planform:.0f} m²)",
                        "Silhouette area vs beam-arrival angle"))

    fig.add_trace(go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color="lightsteelblue", opacity=0.85, flatshading=True, name="BWB"),
        row=1, col=1)

    # Silhouette vs β for a few azimuths (nose-on, quarter, side-on, tail-on).
    for az, label in [(0.0, "α=0° nose-on"), (45.0, "α=45°"), (90.0, "α=90° side-on"),
                      (180.0, "α=180° tail-on")]:
        ai = int(np.argmin(np.abs(alpha - az)))
        fig.add_trace(go.Scatter(x=beta, y=table[:, ai], mode="lines",
                                 name=label), row=1, col=2)
    fig.add_trace(go.Scatter(x=[beta[0], beta[-1]], y=[planform, planform],
                             mode="lines", line=dict(dash="dot", color="grey"),
                             name="belly-on planform"), row=1, col=2)
    fig.update_xaxes(title_text="β: angle from belly (deg) — 0=overhead, 90=on the horizon",
                     row=1, col=2)
    fig.update_yaxes(title_text="silhouette area (m²)", row=1, col=2)
    fig.update_scenes(aspectmode="data")
    fig.update_layout(template="plotly_white", height=520,
                      title="Blended-wing-body silhouette: the collapsing aperture")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.write_html(path, include_plotlyjs="inline")
    return path


if __name__ == "__main__":
    vertices, faces = make_bwb_mesh()
    normals, areas = triangle_normals_areas(vertices, faces)
    beta, alpha, table, planform = build_table()
    print(f"BWB mesh: {len(vertices)} verts, {len(faces)} faces")
    print(f"belly-on planform silhouette : {planform:6.0f} m²  "
          f"(effective radius {np.sqrt(planform/np.pi):4.1f} m)")
    for b in (0, 30, 60, 75, 90):
        a_nose = projected_area(direction_from_beta_alpha(b, 0.0), normals, areas)
        a_side = projected_area(direction_from_beta_alpha(b, 90.0), normals, areas)
        print(f"  β={b:3d}°  nose-az area {a_nose:6.0f} m²   side-az area {a_side:6.0f} m²")
    print("wrote", _TABLE_PATH)
    print("figure:", silhouette_figure())
