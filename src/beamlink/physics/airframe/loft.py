"""Parametric blended-wing-body geometry: the loft, its face normals, and the body-frame angles.

Body frame throughout: +x nose, +y starboard, +z belly (so "down" for level flight). A beam
arrival direction is given as (beta, alpha): beta is the angle off the belly axis (0 = straight
below the aircraft) and alpha the azimuth about it, from the nose towards starboard.

This describes the AIRCRAFT, and both physics domains need it — the link to integrate capture over
the silhouette, the airframe to size a rectenna from its area. The dependency runs one way: the
link imports the aircraft, never the reverse (SPEC.md §9).
"""

from __future__ import annotations

import numpy as np

from beamlink.artifacts.spec import LoftSpec
from beamlink.physics import cache


def _winglet_prism(tip_center_x, y_tip, root_chord, tip_chord, height, fin_sweep, cant_deg,
                   thickness):
    """Flat tapered fin at a wing tip, canted outboard by cant_deg and rising away from the belly
    (−z), swept aft by fin_sweep. Twelve triangles, cuboid topology."""
    side = 1.0 if y_tip >= 0 else -1.0
    cant = np.radians(cant_deg)
    rise = np.array([0.0, side * np.sin(cant), -np.cos(cant)])          # up-and-outboard
    root_center = np.array([tip_center_x, y_tip, 0.0])
    tip_center = root_center + height * rise + np.array([-fin_sweep, 0.0, 0.0])
    chord = np.array([1.0, 0.0, 0.0])
    root_leading, root_trailing = root_center + chord * root_chord / 2, root_center - chord * root_chord / 2
    tip_leading, tip_trailing = tip_center + chord * tip_chord / 2, tip_center - chord * tip_chord / 2
    normal = np.cross(chord, rise)
    normal /= np.linalg.norm(normal)                                     # ≈ spanwise (thin)
    offset = normal * thickness / 2.0
    quad = np.array([root_leading, root_trailing, tip_trailing, tip_leading])
    vertices = np.vstack([quad - offset, quad + offset])
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1],
                      [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]])
    return vertices, faces


def make_mesh(loft: LoftSpec, n_span: int, n_theta: int) -> tuple[np.ndarray, np.ndarray]:
    """Closed triangle mesh: elliptical cross-sections lofted along the span.

    Chord and thickness are constant over the centre body (|u| < body_fraction of the half-span)
    and taper linearly to the tip beyond it; section centres move aft with |u| to give the sweep.
    Returns (vertices (N, 3) m, faces (M, 3) indices).

    `n_span` and `n_theta` are the TESSELLATION, from the fidelity config — the shape is `loft`, and
    the two are separate so the face count can be the lever §6 measures it to be (D14).
    """
    half_span = loft.full_span_m / 2.0
    spanwise = np.linspace(-1.0, 1.0, n_span)
    magnitude = np.abs(spanwise)
    taper = np.clip((magnitude - loft.body_fraction) / (1.0 - loft.body_fraction), 0.0, 1.0)
    chord = loft.root_chord_m - (loft.root_chord_m - loft.tip_chord_m) * taper
    thickness = loft.root_thickness_m - (loft.root_thickness_m - loft.tip_thickness_m) * taper
    nose_tail_center = -loft.sweep_m * magnitude          # nose forward at centre, tips aft
    span_station = spanwise * half_span

    angle = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    cos_angle, sin_angle = np.cos(angle), np.sin(angle)

    ring = np.empty((n_span, n_theta, 3))
    ring[..., 0] = nose_tail_center[:, None] + (chord[:, None] / 2.0) * cos_angle[None, :]
    ring[..., 1] = span_station[:, None]
    ring[..., 2] = (thickness[:, None] / 2.0) * sin_angle[None, :]
    vertices = ring.reshape(-1, 3)

    def vertex_id(i, j):
        return i * n_theta + (j % n_theta)

    faces = []
    for i in range(n_span - 1):
        for j in range(n_theta):
            a, b, c, d = vertex_id(i, j), vertex_id(i, j + 1), vertex_id(i + 1, j + 1), vertex_id(i + 1, j)
            faces.append((a, b, c))
            faces.append((a, c, d))

    for i in (0, n_span - 1):             # cap both wing tips with a fan, wound to face outboard
        center_index = len(vertices)
        vertices = np.vstack([vertices, ring[i].mean(axis=0)])
        for j in range(n_theta):
            a, b = vertex_id(i, j), vertex_id(i, j + 1)
            faces.append((center_index, a, b) if i == 0 else (center_index, b, a))

    faces = np.asarray(faces, dtype=int)

    # Wingtip winglets: tapered, swept, canted dorsal fins. They present some area to a SIDE
    # aspect at low elevation, but are modest — the large side-area radiator would be a vertical
    # stabiliser, which this loft does not have.
    if loft.winglet_height_m > 0:
        tip_x = -loft.sweep_m                          # nose-tail centre of the swept tip
        for y_tip in (-half_span, half_span):
            winglet_vertices, winglet_faces = _winglet_prism(
                tip_x, y_tip, loft.winglet_root_chord_m, loft.winglet_tip_chord_m,
                loft.winglet_height_m, loft.winglet_sweep_m, loft.winglet_cant_deg,
                loft.winglet_thickness_m)
            faces = np.vstack([faces, winglet_faces + len(vertices)])
            vertices = np.vstack([vertices, winglet_vertices])

    return vertices, faces


def _connected_components(vertices, faces):
    """Union-find label per vertex. Needed because the mesh is three disjoint bodies (centre body
    plus two winglets) and each must be oriented about its own centroid."""
    parent = np.arange(len(vertices))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, c in faces:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a
        if root_a != find(c):
            parent[find(c)] = root_a
    roots = np.array([find(i) for i in range(len(vertices))])
    _, component = np.unique(roots, return_inverse=True)
    return component


def face_normals_areas(vertices, faces):
    """Per-face unit normals (forced outward) and areas in m², ignoring winding order."""
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    normal = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + 1e-30)

    component = _connected_components(vertices, faces)
    component_sum = np.zeros((component.max() + 1, 3))
    np.add.at(component_sum, component, vertices)
    component_centroid = component_sum / np.bincount(component)[:, None]
    reference = component_centroid[component[faces[:, 0]]]      # each face's own-part centroid

    triangle_centroid = (v0 + v1 + v2) / 3.0
    inward = np.sum(normal * (triangle_centroid - reference), axis=1) < 0
    normal[inward] *= -1.0
    return normal, area


def convex_projected_area(view_dirs, normals, areas):
    """Sum of area·max(n·v, 0) in m² — the silhouette area for a CONVEX closed body.

    Self-occlusion is not modelled, so concave shadowing is overcounted. Kept because it is the
    vectorized-but-occlusion-free path §6 names as an alternative to batching the z-buffer, and
    because it is the closed-form capture tier's receiver area.
    """
    view = np.asarray(view_dirs, dtype=float)
    view = view / (np.linalg.norm(view, axis=-1, keepdims=True) + 1e-30)
    dots = view @ normals.T
    return np.sum(np.clip(dots, 0.0, None) * areas, axis=-1)


# --- body-frame angles ↔ direction --------------------------------------------------------


def direction_from_beta_alpha(beta_deg, alpha_deg):
    """Unit beam-arrival direction in the body frame from (β off the belly +ẑ, α from the nose)."""
    beta, alpha = np.radians(beta_deg), np.radians(alpha_deg)
    return np.stack([np.sin(beta) * np.cos(alpha),      # nose (x̂)
                     np.sin(beta) * np.sin(alpha),      # starboard (ŷ)
                     np.cos(beta) * np.ones_like(alpha)], axis=-1)   # belly (ẑ)


def _rodrigues(vector, axis, angle_rad):
    axis = axis / (np.linalg.norm(axis, axis=-1, keepdims=True) + 1e-30)
    cos, sin = np.cos(angle_rad), np.sin(angle_rad)
    return (vector * cos[..., None] + np.cross(axis, vector) * sin[..., None]
            + axis * np.sum(axis * vector, axis=-1)[..., None] * (1 - cos)[..., None])


def body_axes_enu(heading_deg, pitch_deg, roll_deg):
    """Body axes expressed in local ENU, broadcasting over a batch of attitudes.

    Heading is measured from north towards east; positive pitch is nose-up (about starboard),
    positive roll is right-wing-down (about the nose), applied in that order.
    """
    heading = np.radians(np.asarray(heading_deg, dtype=float))
    pitch = np.radians(np.asarray(pitch_deg, dtype=float))
    roll = np.radians(np.asarray(roll_deg, dtype=float))
    zero = np.zeros_like(heading)
    nose = np.stack([np.sin(heading), np.cos(heading), zero], axis=-1)
    starboard = np.stack([np.cos(heading), -np.sin(heading), zero], axis=-1)
    belly = np.stack([zero, zero, -np.ones_like(heading)], axis=-1)
    nose = _rodrigues(nose, starboard, pitch)
    belly = _rodrigues(belly, starboard, pitch)
    starboard = _rodrigues(starboard, nose, roll)
    belly = _rodrigues(belly, nose, roll)
    return nose, starboard, belly


def beta_alpha_from_enu(direction_enu, attitudes_deg):
    """ENU direction (aircraft → emitter) to body-frame (β, α) in degrees, batched.

    `attitudes_deg` is (..., 3) as (heading, pitch, roll) and `direction_enu` is (..., 3).
    """
    attitudes = np.asarray(attitudes_deg, dtype=float)
    nose, starboard, belly = body_axes_enu(attitudes[..., 0], attitudes[..., 1], attitudes[..., 2])
    direction = np.asarray(direction_enu, dtype=float)
    direction = direction / (np.linalg.norm(direction, axis=-1, keepdims=True) + 1e-30)
    along_nose = np.sum(direction * nose, axis=-1)
    along_starboard = np.sum(direction * starboard, axis=-1)
    along_belly = np.sum(direction * belly, axis=-1)
    beta = np.degrees(np.arccos(np.clip(along_belly, -1.0, 1.0)))
    alpha = np.degrees(np.arctan2(along_starboard, along_nose)) % 360.0
    return beta, alpha


def body_to_enu_matrix(attitudes_deg):
    """(..., 3, 3) whose ROWS are the body axes (nose, starboard, belly) in local ENU.

    Left-multiplying a body-frame vector by its transpose puts it in ENU, which is how P3's view
    basis is lifted into the earth frame for P4/P5.
    """
    attitudes = np.asarray(attitudes_deg, dtype=float)
    nose, starboard, belly = body_axes_enu(attitudes[..., 0], attitudes[..., 1], attitudes[..., 2])
    return np.stack([nose, starboard, belly], axis=-2)


# --- the cached mesh ----------------------------------------------------------------------


_SCHEMA_VERSION = 2
"""Bumped whenever the set of arrays a cached mesh entry carries changes; see `silhouette`."""


def mesh_key(loft: LoftSpec, n_span: int, n_theta: int) -> str:
    """The content key every cached quantity derived from this mesh is filed under.

    Both the SHAPE and the TESSELLATION are in the key, so a run at a coarser mesh tier can never
    read a finer mesh's raster back out — which the pre-SPEC look-angle dict could do, because it was
    keyed on the look angle alone (see `physics.cache`).
    """
    return cache.content_key("mesh", schema=_SCHEMA_VERSION, loft=loft, n_span=int(n_span),
                             n_theta=int(n_theta))


def _mesh_entry(loft: LoftSpec, n_span: int, n_theta: int):
    vertices, faces = make_mesh(loft, n_span, n_theta)
    normals, areas = face_normals_areas(vertices, faces)
    return dict(vertices=vertices, faces=faces, normals=normals, areas=areas)


def mesh(loft: LoftSpec, config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(vertices, faces, face normals) at this tier's tessellation, cached by content key.

    This is BOTH the capture-path mesh and the mesh P3 emits, so the surface a viewer draws is
    exactly the surface the capture integral summed over — by construction rather than by paying
    2.6× the rasterisation cost to keep two of them equal (§8 item 6).
    """
    entry = cache.memoized(mesh_key(loft, config.mesh_n_span, config.mesh_n_theta),
                           lambda: _mesh_entry(loft, config.mesh_n_span, config.mesh_n_theta))
    return entry["vertices"], entry["faces"], entry["normals"]


def mesh_with_areas(loft: LoftSpec, config):
    """As `mesh`, plus per-face areas — what the convex, occlusion-free area table integrates."""
    entry = cache.memoized(mesh_key(loft, config.mesh_n_span, config.mesh_n_theta),
                           lambda: _mesh_entry(loft, config.mesh_n_span, config.mesh_n_theta))
    return entry["vertices"], entry["faces"], entry["normals"], entry["areas"]
