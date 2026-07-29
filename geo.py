"""Spherical-earth geometry for the beam-link engine.

Curvature is intrinsic here, not a correction bolted onto a flat model: positions
live in an earth-centred (ECEF-like) frame, "level flight" means the belly points at
the local nadir (earth centre), and the emitter's and plane's local verticals differ
by the central angle between them. So the effects the flat model missed fall out for
free — the atmosphere path rides up as the surface "falls away", a target sinks below
the local horizon at range, and a level plane presents a tilted aspect to a distant
station.

Frame: the station's ground point sits at the "top" of the sphere, S₀=(0,0,R). The
station's local ENU is then east=+x, north=+y, up=+z. A node is placed by its
downrange arc (northward, +y), crossrange arc (eastward, +x), and altitude. All
functions are vectorized: pass (...,3) position arrays or arrays of arc distances.
"""
from __future__ import annotations

import numpy as np

R_EARTH = 6.371e6      # m (matches napkin.line_of_sight_km)
_NORTH_REF = np.array([0.0, 1.0, 0.0])


def point(downrange_m, crossrange_m=0.0, altitude_m=0.0):
    """ECEF position of a node at (downrange north, crossrange east, altitude).

    Radial unit is built in closed form (no matrices): tilt +ẑ toward +ŷ by the
    downrange arc angle and toward +x̂ by the crossrange arc angle."""
    an = np.asarray(downrange_m, dtype=float) / R_EARTH      # northward tilt
    ac = np.asarray(crossrange_m, dtype=float) / R_EARTH     # eastward tilt
    up = np.stack([np.cos(an) * np.sin(ac),
                   np.sin(an),
                   np.cos(an) * np.cos(ac)], axis=-1)
    return (R_EARTH + np.asarray(altitude_m, dtype=float))[..., None] * up \
        if np.ndim(altitude_m) or np.ndim(an) else (R_EARTH + altitude_m) * up


def _unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-30)


def local_enu(position_ecef):
    """Local (east, north, up) basis vectors at an ECEF position."""
    p = np.asarray(position_ecef, dtype=float)
    up = _unit(p)
    east = _unit(np.cross(np.broadcast_to(_NORTH_REF, up.shape), up))
    north = np.cross(up, east)
    return east, north, up


def altitude_of(position_ecef):
    return np.linalg.norm(np.asarray(position_ecef, dtype=float), axis=-1) - R_EARTH


def link(from_ecef, to_ecef):
    """Slant range (m) and the elevation (deg) of `to` above `from`'s local horizontal.

    Elevation is measured against the local horizon at `from`, so curvature makes a
    distant target's elevation drop (and go negative below the horizon)."""
    a = np.asarray(from_ecef, dtype=float)
    b = np.asarray(to_ecef, dtype=float)
    d = b - a
    slant = np.linalg.norm(d, axis=-1)
    dhat = d / (slant[..., None] + 1e-30)
    _, _, up_a = local_enu(a)
    sin_el = np.sum(dhat * up_a, axis=-1)
    elevation = np.degrees(np.arcsin(np.clip(sin_el, -1.0, 1.0)))
    return slant, elevation


def direction_in_local_enu(from_ecef, to_ecef):
    """Unit from→to vector expressed in `from`'s local ENU frame (E, N, U components).

    Feed this (with the plane at `from`) to bwb_model.beta_alpha_from_enu together with
    the plane's attitude — the curvature-induced local-vertical tilt is already baked in."""
    a = np.asarray(from_ecef, dtype=float)
    b = np.asarray(to_ecef, dtype=float)
    dhat = _unit(b - a)
    east, north, up = local_enu(a)
    return np.stack([np.sum(dhat * east, axis=-1),
                     np.sum(dhat * north, axis=-1),
                     np.sum(dhat * up, axis=-1)], axis=-1)


def sample_altitudes(from_ecef, to_ecef, n=64):
    """Altitudes above the sphere at `n` points along the straight ECEF chord, and the
    per-segment length ds (m). The chord's altitude profile is curved — the surface
    falls away between the endpoints — which is exactly the atmospheric-path effect."""
    a = np.asarray(from_ecef, dtype=float)
    b = np.asarray(to_ecef, dtype=float)
    t = np.linspace(0.0, 1.0, n)
    pts = a[..., None, :] + t[..., :, None] * (b - a)[..., None, :]   # (...,n,3)
    alt = np.linalg.norm(pts, axis=-1) - R_EARTH                       # (...,n)
    ds = np.linalg.norm(b - a, axis=-1) / (n - 1)                      # (...)
    return alt, ds


def horizon_blocked(from_ecef, to_ecef, n=64, clearance_m=0.0):
    """True if the straight path dips into (or within clearance of) the earth — i.e.
    the two points are below each other's horizon."""
    alt, _ = sample_altitudes(from_ecef, to_ecef, n)
    return np.min(alt, axis=-1) < clearance_m


def downrange_for_elevation(elevation_deg, altitude_m, station_altitude_m=0.0):
    """Downrange arc distance (m) at which a node at `altitude_m` is seen at
    `elevation_deg` from the station. Inverts the monotone elevation(range) relation
    numerically; vectorized over `elevation_deg`."""
    station = point(0.0, 0.0, station_altitude_m)
    grid = np.linspace(1.0, np.pi / 2 * R_EARTH, 4000)               # up to a quarter arc
    nodes = point(grid, 0.0, altitude_m)
    _, elev = link(station, nodes)                                   # decreasing in range
    order = np.argsort(elev)
    return np.interp(np.asarray(elevation_deg, dtype=float), elev[order], grid[order])
