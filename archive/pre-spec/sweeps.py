"""Sweeps of the link engine over a trajectory: one overflight of one station, and one
station's share of a corridor.

`beam_link.evaluate` answers "what does this link deliver at this instant". Everything here
places the aircraft along a path, maps that call over it, and integrates — so this is the layer
where a station's fixed aperture, its rated grid draw, and the energy balance of a whole pass
appear. It sits on top of the engine and the architecture catalogue; the engine knows nothing
of it.

    uv run python -m beamlink.sweeps     # the corridor table, per weather
"""

from __future__ import annotations

import numpy as np

from beamlink import geo
from beamlink.architectures import standard_architectures
from beamlink.beam_link import (LinkConfig, evaluate, geometry, get_silhouette,
                                receiver_radius_from_silhouette, size_emitter_diameter)
from beamlink.bwb_model import beta_alpha_from_enu


def reference_geometry(elevation_deg=30.0, plane_alt_km=11.0):
    """(station_pos, plane_pos) placing the aircraft at a chosen elevation from the station."""
    downrange = float(geo.downrange_for_elevation(elevation_deg, plane_alt_km * 1e3))
    return (0.0, 0.0, 0.0), (downrange, 0.0, plane_alt_km * 1e3)


def emitter_for_reference(frequency_ghz, station_pos, plane_alt_m=11_000.0,
                          reference_elevation_deg=60.0, tau_target=2.0,
                          silhouette=None):
    """Emitter diameter sized once, for the aircraft seen at `reference_elevation_deg`.

    A real station cannot resize per position, so a whole pass is flown with one aperture; the
    reference elevation is the design point that choice is made at.

    Uses the convex silhouette table rather than the occluded raster, which is safe here because
    this call site only ever evaluates tail-on: the aircraft is placed downrange at zero
    crossrange and zero attitude, so alpha comes out at 180 deg for every reference elevation,
    and along alpha=180 the table and the raster agree to a few parts in a thousand.
    """
    downrange = float(geo.downrange_for_elevation(reference_elevation_deg, plane_alt_m))
    plane = (station_pos[0] + downrange, station_pos[1], plane_alt_m)
    slant, _, unit = geometry(plane, station_pos)
    beta, alpha = beta_alpha_from_enu(unit, 0.0, 0.0, 0.0)
    sil = silhouette or get_silhouette()
    r_recv = receiver_radius_from_silhouette(sil.area(beta, alpha))
    return float(size_emitter_diameter(frequency_ghz, slant, r_recv, tau_target))


def pass_profile(frequency_ghz=30.0, plane_alt_m=11_000.0, lateral_offset_m=0.0,
                 alongtrack_m=None, weather="clear", delivered_mw=10.0,
                 emitter_diam_m=None, reference_elevation_deg=60.0):
    """One straight overflight of a single station, direct architecture.

    The station is rated so that `delivered_mw` arrives at the best point of the pass;
    `deliverable_mw` is then what that same fixed grid draw delivers everywhere else.
    """
    if alongtrack_m is None:
        alongtrack_m = np.linspace(-120_000.0, 120_000.0, 241)
    station = (0.0, 0.0, 0.0)
    if emitter_diam_m is None:
        emitter_diam_m = emitter_for_reference(frequency_ghz, station, plane_alt_m,
                                               reference_elevation_deg)
    x = np.asarray(alongtrack_m, dtype=float)
    plane = np.stack([x, np.full_like(x, lateral_offset_m),
                      np.full_like(x, plane_alt_m)], axis=1)
    cfg = LinkConfig(station_pos=station, plane_pos=plane, f_emit_plane_ghz=frequency_ghz,
                     weather=weather, emitter_diam_m=emitter_diam_m)
    s = evaluate(cfg, delivered_mw)
    # Power the fixed emitter can actually deliver, if it radiates the SAME RF it
    # would need to make `delivered_mw` at the reference geometry (capture-limited).
    eta_ref = np.max(s["eta_end_to_end"])
    grid_ref = delivered_mw / eta_ref
    deliverable_mw = grid_ref * s["eta_end_to_end"]
    return dict(alongtrack_km=x / 1e3, elevation_deg=s["elevation_deg"],
                eta_end_to_end=s["eta_end_to_end"], capture=s["capture"],
                silhouette_m2=s["silhouette_m2"], deliverable_mw=deliverable_mw,
                emitter_diam_m=emitter_diam_m, eta_ref=eta_ref)


def flyby(arch_kwargs, plane_alt_m=11_000.0, lateral_offset_m=0.0, station_spacing_km=200.0,
          speed_ms=250.0, delivered_mw=10.0, n_points=81, min_serve_eta=0.02):
    """Energy accounting for one station's share of a corridor: the aircraft crosses half the
    station spacing either side of overhead.

    Two accountings, deliberately not the same thing:
      grid_gj / delivered_gj   station rated for `delivered_mw` at the best point and left at
                               that draw whenever eta clears min_serve_eta, so energy_eta is
                               the mean eta over the served window
      grid_energy_per_pass_gj  the opposite convention -- hold delivered power constant and
                               let the grid draw chase 1/eta, which diverges if eta ever
                               reaches zero, hence the inf

    The first is the operating model: the aircraft is an energy sink, banking whatever arrives as
    altitude or speed, so no battery is modelled and altitude is the buffer. The second is a
    pessimistic bound that would only apply to an aircraft that could not bank.

    `min_serve_eta = 0.02` is not derived from anything, and it sets the headline: for
    `5.8 direct` in a storm, moving it from 0 to 0.10 takes `energy_eta` 0.090 -> 0.240 and
    `grid_gj` 23.8 -> 7.4.
    """
    half = station_spacing_km * 1e3 / 2.0
    x = np.linspace(-half, half, n_points)
    plane = np.stack([x, np.full_like(x, lateral_offset_m), np.full_like(x, plane_alt_m)], axis=1)
    cfg = LinkConfig(plane_pos=plane, **arch_kwargs)
    s = evaluate(cfg, delivered_mw)
    eta = np.asarray(s["eta_end_to_end"], dtype=float)
    dt = (x[1] - x[0]) / speed_ms
    peak = float(np.max(eta))
    # Fixed emitter grid power, sized so the plane hits its absorption ceiling
    # (delivered_mw) at the best point; delivered then rides the η curve.
    grid_rated_mw = delivered_mw / peak if peak > 0 else 0.0
    deliverable_mw = grid_rated_mw * eta
    # Serve only where it's worth it (η above the floor); banked energy = ∫delivered,
    # grid energy = ∫grid over that window; the rest of the pass is coasted on altitude.
    served = eta >= min_serve_eta
    delivered_gj = float(grid_rated_mw * np.sum(eta[served]) * dt / 1e3)
    grid_gj = float(grid_rated_mw * np.count_nonzero(served) * dt / 1e3)
    energy_eta = float(np.mean(eta[served])) if np.any(served) else 0.0
    # Legacy: constant-delivery (no-bank) bound — grid to keep the plane fed at a FLAT
    # delivered_mw all pass; ∞ if any point is unfeasible (can't hold constant delivery).
    with np.errstate(divide="ignore"):
        grid_energy_gj = (float(delivered_mw * np.sum(1.0 / eta) * dt / 1e3)
                          if np.all(eta > 1e-4) else np.inf)
    return dict(
        x_km=x / 1e3, eta=eta, elevation_deg=np.asarray(s["elevation_deg"], float),
        silhouette_m2=np.asarray(s["silhouette_m2"], float),
        capture=np.asarray(s["capture"], float), equivalent_disc_radius_m=np.asarray(s["equivalent_disc_radius_m"], float),
        deliverable_mw=deliverable_mw, mean_eta=float(np.mean(eta)),
        hmean_eta=float(len(eta) / np.sum(1.0 / np.maximum(eta, 1e-9))),
        peak_eta=peak, energy_eta=energy_eta, served_frac=float(np.mean(served)),
        delivered_gj=delivered_gj, grid_gj=grid_gj, waste_gj=grid_gj - delivered_gj,
        grid_energy_per_pass_gj=grid_energy_gj, dt=dt)


def corridor_table(station_spacing_km=200.0, weather="clear", pae_scenario="today",
                   plane_alt_m=11_000.0, delivered_mw=10.0, include_reference=True):
    """Run every standard architecture over one flyby and return {name: flyby-result}."""
    archs = standard_architectures(weather=weather, pae_scenario=pae_scenario,
                                   include_reference=include_reference)
    return {name: flyby(kw, plane_alt_m=plane_alt_m, station_spacing_km=station_spacing_km,
                        delivered_mw=delivered_mw) for name, kw in archs.items()}


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    print("Beam-link engine — flyby-integrated over one station's serving window "
          "(plane 11 km, 200 km spacing, 10 MW delivered)\n")
    print(f"{'architecture':26s} {'weather':11s} {'energy η':>8s} {'peak η':>7s} "
          f"{'served':>7s} {'deliv GJ':>9s} {'grid GJ':>8s} {'waste GJ':>9s}")
    for weather in ("clear", "heavy_rain", "storm"):
        for name, r in corridor_table(weather=weather).items():
            print(f"  {name:24s} {weather:11s} {r['energy_eta']*100:7.1f}% {r['peak_eta']*100:6.1f}% "
                  f"{r['served_frac']*100:6.0f}% {r['delivered_gj']:9.2f} {r['grid_gj']:8.2f} {r['waste_gj']:9.2f}")
        print()
