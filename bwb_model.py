"""Blended-wing-body silhouette model for the beam-link engine.

The receiving aperture on the aircraft is NOT constant: with a fully conformal
rectenna over all surfaces, the geometric power a beam can deliver is set by the
aircraft's SILHOUETTE (shadow) area toward the emitter — which collapses from the
full planform (belly-on, emitter overhead) toward a thin edge (side/nose-on, low
elevation). This module builds a parametric BWB as a closed triangulated surface
and computes that silhouette area for arbitrary beam-arrival directions.

Method: for a closed, (near-)convex body the projected (shadow) area toward a unit
direction d is  A(d) = Σ_i max(0, n̂_i·d)·A_i  over outward-facing triangles — the
standard convex projected-area identity, exact for a convex body and a good
approximation for the smooth blended lens here (self-occlusion among front faces is
the only error, small for a shallow body). Fully vectorized over directions.

Body frame:  x̂ = nose (forward), ŷ = starboard (right), ẑ = belly (down).
A beam-arrival direction (aircraft → emitter) is described by
  β  = polar angle from the belly axis +ẑ  (β=0 → belly-on, maximum area),
  α  = azimuth from the nose +x̂ toward starboard +ŷ.

Run:  uv run python bwb_model.py   → builds data/bwb_silhouette.npz + a 3-view figure.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


# --- Parametric mesh ----------------------------------------------------------

def make_bwb_mesh(full_span_m=80.0, root_chord_m=40.0, tip_chord_m=3.0,
                  root_thickness_m=6.0, tip_thickness_m=0.4, sweep_m=25.0,
                  body_fraction=0.25, n_span=61, n_theta=28):
    """A blended-wing-body as a closed lofted lens.

    Spanwise the chord and thickness are constant over the centre body
    (`body_fraction` of the half-span) then taper to the tip; the leading edge
    sweeps aft by `sweep_m` from root to tip. Each spanwise station's cross-section
    (in the length–thickness plane) is an ellipse, so the body is a smooth closed
    surface. Returns (vertices (N,3), faces (M,3) int)."""
    half_span = full_span_m / 2.0
    u = np.linspace(-1.0, 1.0, n_span)              # spanwise fraction (−1..1)
    au = np.abs(u)
    taper = np.clip((au - body_fraction) / (1.0 - body_fraction), 0.0, 1.0)
    chord = root_chord_m - (root_chord_m - tip_chord_m) * taper
    thickness = root_thickness_m - (root_thickness_m - tip_thickness_m) * taper
    y_center = -sweep_m * au                          # nose forward at centre, tips aft
    x_span = u * half_span

    psi = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    cos_psi, sin_psi = np.cos(psi), np.sin(psi)

    # Vertex grid: (n_span, n_theta, 3)
    verts = np.empty((n_span, n_theta, 3))
    verts[..., 0] = x_span[:, None]
    verts[..., 1] = y_center[:, None] + (chord[:, None] / 2.0) * cos_psi[None, :]
    verts[..., 2] = (thickness[:, None] / 2.0) * sin_psi[None, :]

    vertices = verts.reshape(-1, 3)

    def vid(i, j):
        return i * n_theta + (j % n_theta)

    faces = []
    for i in range(n_span - 1):
        for j in range(n_theta):
            a, b, c, d = vid(i, j), vid(i, j + 1), vid(i + 1, j + 1), vid(i + 1, j)
            faces.append((a, b, c))
            faces.append((a, c, d))

    # Cap the two spanwise ends (fan to the ring centroid) so the surface is closed.
    for i in (0, n_span - 1):
        ring = verts[i]
        center = ring.mean(axis=0)
        c_idx = len(vertices)
        vertices = np.vstack([vertices, center])
        for j in range(n_theta):
            a, b = vid(i, j), vid(i, j + 1)
            faces.append((c_idx, a, b) if i == 0 else (c_idx, b, a))

    return vertices, np.asarray(faces, dtype=int)


def triangle_normals_areas(vertices, faces):
    """Outward unit normals and areas for each triangle (outward via body-centroid test)."""
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    normal = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + 1e-30)
    body_centroid = vertices.mean(axis=0)
    tri_centroid = (v0 + v1 + v2) / 3.0
    outward = np.sum(normal * (tri_centroid - body_centroid), axis=1) < 0
    normal[outward] *= -1.0
    return normal, area


def projected_area(view_dirs, normals, areas):
    """Silhouette area toward each unit direction: Σ max(0, n̂·d)·A.

    `view_dirs` may be (3,) or (...,3); returns scalar or (...) matching the lead
    dimensions. Fully vectorized."""
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


# --- Lookup table -------------------------------------------------------------

_TABLE_PATH = Path(__file__).with_name("data") / "bwb_silhouette.npz"


def build_table(beta_step_deg=2.0, alpha_step_deg=5.0, mesh_kwargs=None, save=True):
    """Precompute silhouette area over (β, α) and (optionally) cache to npz."""
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
        """Silhouette area (m²) for arrays of (β, α) — bilinear on the cached grid."""
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
    """Body axes (nose, starboard, belly) expressed in ENU (x=E, y=N, z=up).

    heading: 0 = nose toward North, increasing clockwise from above (toward East).
    pitch: nose-up positive. roll: right-wing-down positive."""
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
    """(β, α) in body frame for a beam-arrival direction given in ENU."""
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

    # Silhouette vs β for a few azimuths (nose-on, quarter, side-on).
    for az, label in [(0.0, "α=0° nose-on"), (45.0, "α=45°"), (90.0, "α=90° side-on")]:
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
