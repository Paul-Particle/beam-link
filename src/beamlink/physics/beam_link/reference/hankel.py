"""True Fresnel field along a beam, by quasi-discrete Hankel transform — the independent reference.

Ground truth for the closed forms the engine sizes apertures with (Guizar-Sicairos & Gutierrez-Vega
2004). Circularly symmetric, on a disc, sharing NO code with `beam_link.aperture`'s Fresnel transform
on a square grid — which is what makes the check on the Goubau coupling law a check rather than a
tautology (SPEC.md §2).

Engine-internal, and exercised by the test suite only. Being inside the boundary is not an exemption
from it: this is a second implementation on purpose, and the engine's own tests are what hold the two
together.

Scope, measured rather than assumed: accurate for SMOOTH illumination (the matched Gaussian below,
whose amplitude is already small at the rim) and for INTEGRATED quantities — power is conserved to
~1e-16 and captured fractions track 1 - exp(-tau^2) to a few percent. NOT reliable for a hard-edged
uniform disc's PEAK intensity: the discontinuity is not band-limited on this grid, so the on-axis
value does not converge — it wanders between ~0.47x and ~1.25x the analytic pi*a^2/(lambda*L)^2 as
n_points and the pad are varied, getting worse with finer sampling. Use the engine's own transform for
peak intensity and hard apertures; use this for coupling and as the cross-check.
"""

from __future__ import annotations

import numpy as np
from scipy.special import j0, j1, jn_zeros

C_LIGHT = 299_792_458.0


class HankelGrid:
    """Order-zero QDHT on a disc of radius r_max, N sample points."""

    def __init__(self, r_max: float, n_points: int = 256):
        zeros = jn_zeros(0, n_points + 1)
        self.j_zeros = zeros[:n_points]
        self.s_max = zeros[n_points]
        self.n_points = n_points
        self.r_max = r_max
        self.nu_max = self.s_max / (2 * np.pi * r_max)

        self.radius = self.j_zeros * r_max / self.s_max          # spatial samples
        self.nu = self.j_zeros / (2 * np.pi * r_max)             # frequency samples

        j1_abs = np.abs(j1(self.j_zeros))
        self.weights = j1_abs
        # involutory transform matrix; forward and inverse share it
        self.matrix = (
            2.0
            / self.s_max
            * j0(np.outer(self.j_zeros, self.j_zeros) / self.s_max)
            / np.outer(j1_abs, j1_abs)
        )
        # quadrature weights for integral f -> 2*pi*int f r dr over the disc
        self.area_weights = 4 * np.pi * r_max**2 / (self.s_max**2 * j1_abs**2)

    def forward(self, field: np.ndarray) -> np.ndarray:
        return (self.matrix @ (field / self.weights)) * self.weights * (
            self.r_max / self.nu_max
        )

    def inverse(self, spectrum: np.ndarray) -> np.ndarray:
        return (self.matrix @ (spectrum / self.weights)) * self.weights * (
            self.nu_max / self.r_max
        )

    def power(self, field: np.ndarray) -> float:
        """Total power in the field, 2*pi*int |E|^2 r dr."""
        return float(np.sum(self.area_weights * np.abs(field) ** 2))

    def power_within(self, field: np.ndarray, radius: float) -> float:
        mask = self.radius <= radius
        return float(np.sum(self.area_weights[mask] * np.abs(field[mask]) ** 2))

    def propagate(self, field: np.ndarray, distance: float, wavelength: float):
        """Paraxial angular-spectrum step."""
        spectrum = self.forward(field)
        spectrum = spectrum * np.exp(-1j * np.pi * wavelength * distance * self.nu**2)
        return self.inverse(spectrum)


def gaussian_illumination(grid: HankelGrid, waist: float, aperture_radius: float,
                          focus_distance: float, wavelength: float) -> np.ndarray:
    """Gaussian of the matched mode, hard-truncated at the rim, focused downrange."""
    field = np.exp(-(grid.radius / waist) ** 2).astype(complex)
    field[grid.radius > aperture_radius] = 0.0
    field *= np.exp(-1j * np.pi * grid.radius**2 / (wavelength * focus_distance))
    return field


def matched_waists(r_a: float, r_b: float, length: float, wavelength: float):
    """w_a*w_b = lam*L/pi with w_a/w_b = r_a/r_b, i.e. w = r/sqrt(tau)."""
    tau = np.pi * r_a * r_b / (wavelength * length)
    return r_a / np.sqrt(tau), r_b / np.sqrt(tau), tau


def hop_field(r_a, r_b, length, wavelength, n_planes=48, n_points=512,
              pad=8.0, incident=None, grid=None):
    """Propagate one hop; return (grid, z samples, field per plane, tau, launched power)."""
    w_a, w_b, tau = matched_waists(r_a, r_b, length, wavelength)
    if grid is None:
        grid = HankelGrid(pad * max(r_a, r_b), n_points)
    if incident is None:
        field = gaussian_illumination(grid, w_a, r_a, length, wavelength)
    else:
        field = incident
    launched = grid.power(field)

    z = np.linspace(0.0, length, n_planes)
    planes = np.empty((n_planes, grid.n_points), dtype=complex)
    planes[0] = field
    for i, zi in enumerate(z[1:], start=1):
        planes[i] = grid.propagate(field, zi, wavelength)
    return grid, z, planes, tau, launched
