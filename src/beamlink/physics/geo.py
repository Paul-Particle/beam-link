"""Spherical-earth geometry, shared by both physics domains.

Positions are cartesian metres in a fixed earth frame whose +y axis is the pole of the
(downrange, crossrange) surface parameterisation and whose +z axis passes through the origin of
that parameterisation. All angles in and out are degrees, all lengths metres.

Everything here broadcasts: a batch of (candidate, sample) endpoints is the normal call, and a
single pair is a batch of one (I9).
"""

from __future__ import annotations

import numpy as np

R_EARTH = 6.371e6
_NORTH_REF = np.array([0.0, 1.0, 0.0])   # pole of the parameterisation below == +downrange


def point(downrange_m, crossrange_m=0.0, altitude_m=0.0):
    """Surface arc lengths plus altitude → cartesian position, shape (..., 3).

    downrange/crossrange are distances measured along the surface, so they are the
    latitude/longitude-like angles about the +y pole once divided by the earth radius.
    """
    along = np.asarray(downrange_m, dtype=float) / R_EARTH
    across = np.broadcast_to(np.asarray(crossrange_m, dtype=float), np.shape(along)) / R_EARTH
    altitude = np.broadcast_to(np.asarray(altitude_m, dtype=float), np.shape(along))
    up = np.stack([np.cos(along) * np.sin(across),
                   np.sin(along),
                   np.cos(along) * np.cos(across)], axis=-1)
    return (R_EARTH + altitude)[..., None] * up


def positions_to_ecef(positions):
    """(..., 3) of (downrange, crossrange, altitude) → (..., 3) earth-frame cartesian."""
    p = np.asarray(positions, dtype=float)
    return point(p[..., 0], p[..., 1], p[..., 2])


def unit(vectors):
    v = np.asarray(vectors, dtype=float)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-30)


def local_enu(position_ecef):
    """Local (east, north, up) unit vectors. "North" is the +downrange direction."""
    p = np.asarray(position_ecef, dtype=float)
    up = unit(p)
    east = unit(np.cross(np.broadcast_to(_NORTH_REF, up.shape), up))
    north = np.cross(up, east)
    return east, north, up


def link(from_ecef, to_ecef):
    """(slant range m, elevation deg) of `to` as seen from `from`.

    Elevation is measured at the EMITTING end, above its own local horizon, and is negative when
    `to` is below that horizon. This is the one definition the engine has (D1): a per-leg
    quantity, evaluated at whichever end emits, so ground→aircraft comes out positive and
    relay→aircraft correctly negative without a second sign convention anywhere.
    """
    a = np.asarray(from_ecef, dtype=float)
    b = np.asarray(to_ecef, dtype=float)
    delta = b - a
    slant = np.linalg.norm(delta, axis=-1)
    direction = delta / (slant[..., None] + 1e-30)
    _, _, up_at_a = local_enu(a)
    sin_elevation = np.sum(direction * up_at_a, axis=-1)
    return slant, np.degrees(np.arcsin(np.clip(sin_elevation, -1.0, 1.0)))


def direction_in_local_enu(from_ecef, to_ecef):
    """Unit from→to vector in `from`'s local ENU frame, as (east, north, up) components.

    Fed (with the aircraft at `from`) to the airframe's body-frame conversion, this already has
    the curvature-induced local-vertical tilt baked in.
    """
    a = np.asarray(from_ecef, dtype=float)
    b = np.asarray(to_ecef, dtype=float)
    direction = unit(b - a)
    east, north, up = local_enu(a)
    return np.stack([np.sum(direction * east, axis=-1),
                     np.sum(direction * north, axis=-1),
                     np.sum(direction * up, axis=-1)], axis=-1)


def enu_to_ecef(vectors_enu, at_ecef):
    """A direction given in `at_ecef`'s local ENU frame, expressed back in the earth frame."""
    east, north, up = local_enu(at_ecef)
    v = np.asarray(vectors_enu, dtype=float)
    return v[..., 0:1] * east + v[..., 1:2] * north + v[..., 2:3] * up


def sample_altitudes(from_ecef, to_ecef, n=64):
    """Altitudes (m) at n points along the straight chord, plus the step length ds (m).

    Altitude goes negative where the chord cuts below the surface, which is what makes this
    double as the line-of-sight test below.
    """
    a = np.asarray(from_ecef, dtype=float)
    b = np.asarray(to_ecef, dtype=float)
    fraction = np.linspace(0.0, 1.0, n)
    points = a[..., None, :] + fraction[..., :, None] * (b - a)[..., None, :]
    altitude = np.linalg.norm(points, axis=-1) - R_EARTH
    step = np.linalg.norm(b - a, axis=-1) / (n - 1)
    return altitude, step


def horizon_blocked(from_ecef, to_ecef, n=64, clearance_m=0.0):
    """True where the straight path dips into (or within clearance of) the earth."""
    altitude, _ = sample_altitudes(from_ecef, to_ecef, n)
    return np.min(altitude, axis=-1) < clearance_m


def downrange_for_elevation(elevation_deg, altitude_m, station_altitude_m=0.0):
    """Inverse of `link`: the downrange (m) at which a target at `altitude_m` is seen at the given
    elevation from the station. Tabulate-and-interpolate out to a quarter circumference."""
    station = point(0.0, 0.0, station_altitude_m)
    grid = np.linspace(1.0, np.pi / 2 * R_EARTH, 4000)
    _, elevation = link(station, point(grid, 0.0, altitude_m))
    order = np.argsort(elevation)   # elevation falls monotonically with downrange; flip for interp
    return np.interp(np.asarray(elevation_deg, dtype=float), elevation[order], grid[order])


def transverse_basis(propagation_ecef, reference_ecef):
    """An orthonormal (x̂, ŷ) transverse to `propagation_ecef`, with x̂ along `reference_ecef`.

    Used where there is no receiver to align the aperture to: the reference direction is a
    declared spec input, orthogonalised against the propagation direction, so the aperture frame
    is defined by something the run states rather than by whatever `np.cross` happened to return
    (D2).
    """
    w = unit(propagation_ecef)
    reference = np.asarray(reference_ecef, dtype=float)
    projected = reference - np.sum(reference * w, axis=-1, keepdims=True) * w
    degenerate = np.linalg.norm(projected, axis=-1, keepdims=True) < 1e-9
    fallback = np.broadcast_to(np.array([0.0, 0.0, 1.0]), w.shape)
    fallback = fallback - np.sum(fallback * w, axis=-1, keepdims=True) * w
    x_axis = unit(np.where(degenerate, fallback, projected))
    return x_axis, np.cross(w, x_axis)
