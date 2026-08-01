"""Winglet-geometry validation renders.

Purpose: let a human eyeball whether the wingtip winglets are modelled as intended — thin fins
whose surfaces are PARALLEL to the forward (nose) direction, so that:
  * head-on (looking straight at the nose, orthographic) they appear EDGE-ON (near-vertical slivers);
  * after a small yaw both winglets read as SHIFTED (translated, not mirrored) copies — you see the
    inside face of one and the outside face of the other, and both leading-edge "points" swing the
    same way.

Body frame (see bwb_model): x̂ = nose (forward), ŷ = starboard (right), ẑ = vertical BELLY-DOWN,
so screen-up = −ẑ. The canonical validation view looks along −x̂ with ŷ → screen-x, −ẑ → screen-y.

Outputs:
  out/winglet_headon.png        — matplotlib: head-on ortho + yaw-15° ortho, high-res, winglets cyan
  out/winglet_3d_ortho.html     — interactive plotly Mesh3d, ORTHOGRAPHIC camera at the nose
  out/winglet_3d_persp.html     — interactive plotly Mesh3d, PERSPECTIVE camera at the nose

Run:  PYTHONPATH="$PWD" uv run --with kaleido python winglet_validation.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
import plotly.graph_objects as go

import bwb_model as m

OUT = "out"
N_WINGLET_FACES = 24          # last 24 faces are the two winglets (12 each) — see make_bwb_mesh


def _yaw_about_vertical(verts, deg):
    """Yaw the airframe by `deg` about the vertical body axis ẑ (component 2). Positive = nose
    swings toward starboard (+ŷ). Rotates the x̂/ŷ (nose/starboard) components; ẑ unchanged."""
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return verts @ R.T


def _render_headon(ax, verts, faces, yaw_deg, title):
    """Orthographic head-on (camera in front of the nose on +x̂, looking −x̂). Screen-x = ŷ
    (starboard), screen-y = −ẑ (up). Depth for painter's sort = x̂ (nose is nearest the camera).
    Body drawn as a faint wireframe so it never hides the winglets; winglets filled cyan."""
    v = _yaw_about_vertical(verts, yaw_deg)
    tri = v[faces]                                        # (M, 3, 3)
    sx = tri[..., 1]                                      # starboard → screen x
    sy = -tri[..., 2]                                     # up (−ẑ) → screen y
    polys = np.stack([sx, sy], axis=-1)                  # (M, 3, 2)
    is_wing = np.zeros(len(faces), bool); is_wing[-N_WINGLET_FACES:] = True

    # Body: faint edges only (wireframe), so the thin winglets are always visible over/through it.
    body_segs = []
    for p in polys[~is_wing]:
        body_segs += [[p[0], p[1]], [p[1], p[2]], [p[2], p[0]]]
    ax.add_collection(LineCollection(body_segs, colors="#3a6ea5", linewidths=0.25, alpha=0.35))

    # Winglets: filled cyan with dark edges, depth-sorted among themselves (nose-nearest last).
    wing_polys = polys[is_wing]
    wing_depth = tri[is_wing][..., 0].mean(axis=1)       # mean x̂ per face
    order = np.argsort(wing_depth)
    pc = PolyCollection(wing_polys[order], facecolors="#00d5e0", edgecolors="#003b40",
                        linewidths=0.6, alpha=0.9)
    ax.add_collection(pc)

    ax.set_aspect("equal")
    pad = 3.0
    ax.set_xlim(sx.min() - pad, sx.max() + pad)
    ax.set_ylim(sy.min() - pad, sy.max() + pad)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("starboard ŷ (m)  →"); ax.set_ylabel("up  −ẑ (m)  ↑")
    ax.axhline(0, color="#888", lw=0.4, ls=":"); ax.axvline(0, color="#888", lw=0.4, ls=":")


def render_headon_png(path=f"{OUT}/winglet_headon.png"):
    verts, faces = m.make_bwb_mesh()
    fig, axs = plt.subplots(1, 2, figsize=(18, 8))
    _render_headon(axs[0], verts, faces, 0.0,
                   "HEAD-ON, orthographic (looking at the nose, wings horizontal)\n"
                   "winglets should be EDGE-ON — near-vertical slivers")
    _render_headon(axs[1], verts, faces, 15.0,
                   "YAWED 15° about vertical (nose → starboard), orthographic\n"
                   "both winglets = shifted copies: inside of one, outside of the other")
    fig.suptitle("Winglet validation — BWB body frame x̂=nose, ŷ=starboard, ẑ=belly-down",
                 fontsize=13)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)
    print("wrote", path)


def _mesh3d_traces(verts, faces):
    is_wing = np.zeros(len(faces), bool); is_wing[-N_WINGLET_FACES:] = True
    out = []
    for mask, color, opacity, name in ((~is_wing, "#4a7fb5", 0.45, "body"),
                                       (is_wing, "#00d5e0", 1.0, "winglets")):
        f = faces[mask]
        out.append(go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                             i=f[:, 0], j=f[:, 1], k=f[:, 2],
                             color=color, opacity=opacity, name=name, showscale=False,
                             flatshading=True, hoverinfo="name"))
    return out


def render_3d_html(projection, path):
    """Interactive 3D Mesh3d viewed from in front of the nose. `projection` ∈ {orthographic,
    perspective}. z is belly-down, so up = −z: we set the camera's up vector to (0,0,-1)."""
    verts, faces = m.make_bwb_mesh()
    fig = go.Figure(_mesh3d_traces(verts, faces))
    # Camera in FRONT of the nose (+x̂), looking back toward −x̂; up = −ẑ.
    cam = dict(eye=dict(x=2.2, y=0.0, z=-0.15), center=dict(x=0, y=0, z=0),
               up=dict(x=0, y=0, z=-1), projection=dict(type=projection))
    fig.update_layout(
        title=f"BWB winglet check — 3D {projection} · drag to rotate "
              f"(x̂=nose, ŷ=starboard, ẑ=belly-down)",
        scene=dict(camera=cam, aspectmode="data",
                   xaxis_title="x̂ nose (m)", yaxis_title="ŷ starboard (m)",
                   zaxis_title="ẑ down (m)"),
        margin=dict(l=0, r=0, t=40, b=0), height=800)
    os.makedirs(OUT, exist_ok=True)
    fig.write_html(path, include_plotlyjs="inline")
    print("wrote", path)


if __name__ == "__main__":
    render_headon_png()
    render_3d_html("orthographic", f"{OUT}/winglet_3d_ortho.html")
    render_3d_html("perspective", f"{OUT}/winglet_3d_persp.html")
