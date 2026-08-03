"""Ground-truth chain propagation: every element truncates and refocuses.

The multi-hop counterpart to `hankel`: propagate the real field through a chain of apertures, each
truncating what lands on it and writing a fresh outgoing phase, and compare the result with the
product of per-hop Goubau coupling factors. Engine-internal, tests only (SPEC.md §2).
"""

import numpy as np
from scipy.integrate import simpson

from beamlink.physics.beam_link.reference.hankel import (C_LIGHT, HankelGrid,
                                                        matched_waists)

N_POINTS = 1024
PAD = 6.0


def captured(grid, field, radius, n_fine=4001):
    inten = np.abs(field) ** 2
    rr = np.linspace(0.0, radius, n_fine)
    return 2 * np.pi * simpson(np.interp(rr, grid.radius, inten) * rr, x=rr)


def run_chain(freq_hz, total_m, radii_m, positions_m, n_points=N_POINTS,
              n_planes_per_hop=40, verbose=True):
    """radii_m = [emitter, relay..., receiver] radii; positions_m matches."""
    lam = C_LIGHT / freq_hz
    hops = len(radii_m) - 1
    grid_r = PAD * max(radii_m)
    grid = HankelGrid(grid_r, n_points)

    # emitter illumination: matched Gaussian for hop 1, truncated at the rim
    L0 = positions_m[1] - positions_m[0]
    w_a, _, _ = matched_waists(radii_m[0], radii_m[1], L0, lam)
    field = np.exp(-(grid.radius / w_a) ** 2).astype(complex)
    field[grid.radius > radii_m[0]] = 0.0
    launched = grid.power(field)

    out = {"lam": lam, "hops": [], "planes": [], "z": []}
    running = 1.0
    for h in range(hops):
        L = positions_m[h + 1] - positions_m[h]
        r_next = radii_m[h + 1]
        # A reflectarray is phase-only: it keeps the amplitude that lands on it
        # and writes a fresh outgoing phase. The matched mode wants wavefront
        # curvature R = L at every launching aperture, whatever the ratio of
        # the two apertures, so that is the phase to write.
        focused = np.abs(field) * np.exp(-1j * np.pi * grid.radius**2 / (lam * L))
        p_in = grid.power(focused)

        zs = np.linspace(0.0, L, n_planes_per_hop)
        for zi in zs:
            out["planes"].append(grid.propagate(focused, zi, lam) if zi > 0 else focused)
            out["z"].append(positions_m[h] + zi)

        arrived = grid.propagate(focused, L, lam)
        cap = captured(grid, arrived, r_next)
        eta_true = cap / p_in
        tau = np.pi * radii_m[h] * r_next / (lam * L)
        out["hops"].append({
            "tau": tau, "goubau": 1 - np.exp(-(tau**2)), "true": eta_true,
            "L": L, "r_a": radii_m[h], "r_b": r_next})
        running *= eta_true

        # the element keeps what it caught and re-launches it
        field = arrived.copy()
        field[grid.radius > r_next] = 0.0

    out["total_true"] = running
    out["total_goubau"] = float(np.prod([h["goubau"] for h in out["hops"]]))
    out["launched"] = launched
    if verbose:
        print(f"  lam {lam*1000:.3f} mm   grid R {grid_r:.0f} m   N {n_points}")
        for i, h in enumerate(out["hops"]):
            print(f"  hop {i+1}: L {h['L']/1000:8.1f} km  r {h['r_a']:7.2f}->{h['r_b']:7.2f} m"
                  f"  tau {h['tau']:6.3f}  Goubau {h['goubau']:8.4%}  true {h['true']:8.4%}"
                  f"  delta {h['true']-h['goubau']:+7.4%}")
        print(f"  chain: Goubau {out['total_goubau']:.4%}   true {out['total_true']:.4%}"
              f"   delta {out['total_true']-out['total_goubau']:+.4%}")
    return out
