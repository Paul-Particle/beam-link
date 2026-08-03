"""Compare the extracted `bwb` package against beamlink's physics.airframe originals.

Run with beam-link's venv python, with beamed-aviation/src prepended to sys.path.
Nothing is allowed to differ beyond float noise; mesh topology must be exactly equal.
"""

import sys

import numpy as np

from beamlink.artifacts.spec import LoftSpec
from beamlink.physics import fidelity
from beamlink.physics.airframe import loft as o_loft
from beamlink.physics.airframe import silhouette as o_sil

from bwb import frames as n_frames
from bwb import loft as n_loft
from bwb import silhouette as n_sil
from bwb.loft import LoftShape

failures = []
checks = 0


def check(name, ok, detail=""):
    global checks
    checks += 1
    if not ok:
        failures.append(f"{name}: {detail}")
    print(f"{'ok  ' if ok else 'FAIL'}  {name}  {detail}")


def close(a, b, rtol=0.0, atol=0.0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return False, f"shape {a.shape} vs {b.shape}"
    if a.size == 0:
        return True, "empty"
    err = np.abs(a - b)
    scale = np.maximum(np.abs(a), np.abs(b))
    rel = np.where(scale > 0, err / np.maximum(scale, 1e-300), 0.0)
    ok = bool(np.all((err <= atol) | (rel <= rtol)))
    return ok, f"max abs {err.max():.3e}, max rel {rel.max():.3e}"


# --- the defaults are the same aircraft -----------------------------------------------------

spec, shape = LoftSpec(), LoftShape()
for field in spec.__dataclass_fields__:
    check(f"default {field}", getattr(spec, field) == getattr(shape, field),
          f"{getattr(spec, field)} vs {getattr(shape, field)}")

config = fidelity.config("standard")
N_SPAN, N_THETA, N_GRID = config.mesh_n_span, config.mesh_n_theta, config.raster_n_grid
check("tessellation defaults match", (N_SPAN, N_THETA) == (n_loft.DEFAULT_N_SPAN, n_loft.DEFAULT_N_THETA)
      and N_GRID == n_sil.DEFAULT_RASTER_N_GRID, f"{N_SPAN},{N_THETA},{N_GRID}")

# --- the mesh, exactly ----------------------------------------------------------------------

for tess in [(N_SPAN, N_THETA), (25, 16), (41, 26)]:
    ov, of = o_loft.make_mesh(spec, *tess)
    nv, nf = n_loft.make_mesh(shape, *tess)
    check(f"mesh vertices {tess}", *close(ov, nv, atol=0.0, rtol=0.0))
    check(f"mesh faces {tess} exactly equal", of.shape == nf.shape and np.array_equal(of, nf),
          f"{of.shape} vs {nf.shape}")
    on, oa = o_loft.face_normals_areas(ov, of)
    nn, na = n_loft.face_normals_areas(nv, nf)
    check(f"face normals {tess}", *close(on, nn, atol=0.0, rtol=0.0))
    check(f"face areas {tess}", *close(oa, na, atol=0.0, rtol=0.0))

# --- a shape that is NOT the default, and one with no winglets ------------------------------

for kwargs in [dict(full_span_m=60.0, root_chord_m=30.0, sweep_m=18.0, body_fraction=0.4),
               dict(winglet_height_m=0.0),
               dict(winglet_cant_deg=45.0, winglet_height_m=6.0, tip_thickness_m=0.9)]:
    ov, of = o_loft.make_mesh(LoftSpec(**kwargs), 31, 20)
    nv, nf = n_loft.make_mesh(LoftShape(**kwargs), 31, 20)
    check(f"mesh vertices {kwargs}", *close(ov, nv))
    check(f"mesh faces {kwargs}", np.array_equal(of, nf), f"{of.shape} vs {nf.shape}")

# --- frames ---------------------------------------------------------------------------------

beta_grid = np.array([0.0, 17.0, 45.0, 90.0, 133.0, 180.0])
alpha_grid = np.array([0.0, 30.0, 90.0, 210.0, 359.0])
bb, aa = np.meshgrid(beta_grid, alpha_grid, indexing="ij")
check("direction_from_beta_alpha",
      *close(o_loft.direction_from_beta_alpha(bb, aa), n_frames.direction_from_beta_alpha(bb, aa)))

attitudes = np.array([[0.0, 0.0, 0.0], [42.0, 5.0, -8.0], [270.0, -3.0, 20.0], [123.0, 11.0, 0.0]])
o_axes = o_loft.body_axes_enu(attitudes[:, 0], attitudes[:, 1], attitudes[:, 2])
n_axes = n_frames.body_axes_enu(attitudes[:, 0], attitudes[:, 1], attitudes[:, 2])
for label, o_ax, n_ax in zip(("nose", "starboard", "belly"), o_axes, n_axes):
    check(f"body_axes_enu {label}", *close(o_ax, n_ax))

check("body_to_enu_matrix",
      *close(o_loft.body_to_enu_matrix(attitudes), n_frames.body_to_enu_matrix(attitudes)))

rng = np.random.default_rng(7)
dirs = rng.normal(size=(4, 3))
o_beta, o_alpha = o_loft.beta_alpha_from_enu(dirs, attitudes)
n_beta, n_alpha = n_frames.beta_alpha_from_enu(dirs, attitudes)
check("beta_alpha_from_enu beta", *close(o_beta, n_beta))
check("beta_alpha_from_enu alpha", *close(o_alpha, n_alpha))

# --- convex projected area and the table ----------------------------------------------------

ov, of, on, oa = o_loft.mesh_with_areas(spec, config)
nv, nf, nn, na = n_loft.mesh_with_areas(shape, N_SPAN, N_THETA)
check("mesh_with_areas normals", *close(on, nn))
check("mesh_with_areas areas", *close(oa, na))

view = n_frames.direction_from_beta_alpha(bb, aa)
check("convex_projected_area",
      *close(o_loft.convex_projected_area(view, on, oa),
             n_loft.convex_projected_area(view, nn, na)))

o_table = o_sil.convex_area_table(spec, config)
n_table = n_sil.convex_area_table(shape, N_SPAN, N_THETA)
check("convex_area_table area", *close(o_table["area_m2"], n_table["area_m2"]))
check("convex_area_table planform", *close(o_table["planform_area_m2"], n_table["planform_area_m2"]))
check("convex_area lookup",
      *close(o_sil.convex_area(spec, config, bb, aa), n_sil.convex_area(shape, bb, aa, N_SPAN, N_THETA)))
check("equal_area_radius",
      *close(o_sil.equal_area_radius(o_table["area_m2"]), n_sil.equal_area_radius(n_table["area_m2"])))

# --- the silhouette raster, the one that matters --------------------------------------------

ANGLES = [(0.0, 0.0), (10.0, 0.0), (30.0, 45.0), (55.0, 30.0), (55.0, -30.0), (85.0, 90.0),
          (90.0, 0.0), (120.0, 200.0), (170.0, 15.0), (180.0, 0.0), (2.5, 317.0), (67.5, 123.5)]
for beta, alpha in ANGLES:
    o_r = o_sil.silhouette_at(spec, beta, alpha, config)
    n_r = n_sil.silhouette_at(shape, beta, alpha, N_SPAN, N_THETA, N_GRID)
    tag = f"({beta}, {alpha})"
    check(f"silhouette area {tag}", *close(o_r["area"], n_r["area"]))
    check(f"silhouette n_elements {tag}",
          o_r["element_x"].size == n_r["element_x"].size,
          f"{o_r['element_x'].size} vs {n_r['element_x'].size}")
    for key in ("element_x", "element_y", "incidence", "projected_area", "basis",
                "half_extent", "sigma", "skew", "look"):
        check(f"silhouette {key} {tag}", *close(o_r[key], n_r[key]))

# --- max_half_extent, at a coarse tessellation so it finishes -------------------------------

coarse = fidelity.with_changes(config, mesh_n_span=25, mesh_n_theta=16, raster_n_grid=96)
check("max_half_extent",
      *close(o_sil.max_half_extent(spec, coarse),
             n_sil.max_half_extent(shape, 25, 16, 96)))

print()
print(f"{checks} checks, {len(failures)} failed")
for line in failures:
    print("  FAIL " + line)
sys.exit(1 if failures else 0)
