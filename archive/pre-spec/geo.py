"""Spherical-earth geometry for ground-station / aircraft links.

Positions are cartesian metres in a fixed earth frame whose +y axis is the pole of the
(downrange, crossrange) surface parameterisation and whose +z axis passes through the
origin of that parameterisation. All angles in/out are degrees, all lengths metres.
"""

from __future__ import annotations

import numpy as np

R_EARTH = 6.371e6
_NORTH_REF = np.array([0.0, 1.0, 0.0])  # pole of the parameterisation below == +downrange


def point(downrange_m, crossrange_m=0.0, altitude_m=0.0):
    """Surface arc lengths + altitude -> cartesian position.

    downrange/crossrange are distances measured along the surface, so an/ac are the
    latitude/longitude-like angles about the +y pole.
    """
    an = np.asarray(downrange_m, dtype=float) / R_EARTH
    ac = np.asarray(crossrange_m, dtype=float) / R_EARTH
    up = np.stack([np.cos(an) * np.sin(ac),
                   np.sin(an),
                   np.cos(an) * np.cos(ac)], axis=-1)
    # two branches only to keep the scalar case a plain scalar-times-vector
    return (R_EARTH + np.asarray(altitude_m, dtype=float))[..., None] * up \
        if np.ndim(altitude_m) or np.ndim(an) else (R_EARTH + altitude_m) * up


def _unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-30)


def local_enu(position_ecef):
    """Local east/north/up unit vectors. "North" is the +downrange direction."""
    p = np.asarray(position_ecef, dtype=float)
    up = _unit(p)
    east = _unit(np.cross(np.broadcast_to(_NORTH_REF, up.shape), up))
    north = np.cross(up, east)
    return east, north, up


def altitude_of(position_ecef):
    return np.linalg.norm(np.asarray(position_ecef, dtype=float), axis=-1) - R_EARTH


def link(from_ecef, to_ecef):
    """(slant range m, elevation deg) of `to` as seen from `from`. Negative = below horizontal."""
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
    """Altitudes (m) at n points along the straight chord, plus the step length ds (m).

    Altitude goes negative where the chord cuts below the surface, which is what makes
    this double as the line-of-sight test below.
    """
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
    """Inverse of `link`: downrange (m) at which a target at `altitude_m` is seen at
    the given elevation. Tabulate-and-interpolate out to a quarter circumference."""
    station = point(0.0, 0.0, station_altitude_m)
    grid = np.linspace(1.0, np.pi / 2 * R_EARTH, 4000)               # up to a quarter arc
    nodes = point(grid, 0.0, altitude_m)
    _, elev = link(station, nodes)
    order = np.argsort(elev)  # elevation falls monotonically with downrange; flip for interp
    return np.interp(np.asarray(elevation_deg, dtype=float), elev[order], grid[order])
