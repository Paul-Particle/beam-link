"""Beam-link engine for microwave-powered aviation.

Given the 3-D positions and orientations of a ground/sea STATION, an optional
RELAY, and the PLANE, plus the link frequencies and the delivered power the plane
needs, this computes the end-to-end power-transfer efficiency and the grid power a
station must draw — tracing every loss in the chain so the physics can be audited
stage by stage. Everything is vectorized: pass numpy arrays for any input to sweep.

The chain (grid busbar → shaft-input DC at the plane):

  grid ──DC→RF (PAE)──▶ emitter aperture (errors, scan) ──▶ radiated RF
       ──atmosphere (gas+rain+cloud)──▶ ──near-field capture τ──▶ RF on receiver
       ──[relay: rx-capture · convert · re-emit]──▶ ──atmosphere──▶ capture
       ──rectenna: RF→DC · silhouette-match · incidence · polarization──▶ delivered

Two architectures, chosen by `relay=None` (direct) or a relay position:
  • direct   station → plane at f_emit_plane
  • relayed  station → relay at f_emit_relay, then relay → plane at f_relay_plane
    with the relay either an ACTIVE frequency-converting repeater (default) or a
    PASSIVE reflectarray (then f_relay_plane is forced equal to f_emit_relay).
Direct and relayed are ALTERNATIVES — the engine evaluates whichever you ask for;
`best_path()` runs both and picks the better under the given weather.

Physical models are #(verify)-grade and reuse napkin.py wherever it already has the
relation. Near-field coupling is the Goubau aperture-coupling law (NOT far-field
spreading — the link sits deep inside the Fresnel region). Atmospheric absorption is
integrated from the ITU-R standard-atmosphere table built by
generate_atmosphere_profile.py; rain/cloud are layered on per weather scenario.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import napkin
import geo
import bwb_model
from bwb_model import (BWBSilhouette, beta_alpha_from_enu, direction_from_beta_alpha,
                       get_surface_samples, rasterize_silhouette)


# --- Weather scenarios --------------------------------------------------------
# rain_rate mm/h, rain_top m (melting layer), cloud liquid-water g/m³ over a band.
# #(verify) representative values; "stratospheric" is implicit — any path segment
# above the rain/cloud bands and the (dry) tropopause simply picks up ~nothing.
WEATHER = {
    "clear":      dict(rain_rate_mmh=0.0,  rain_top_m=0.0,    cloud_lwc_gm3=0.0, cloud_band_m=(0.0, 0.0)),
    "cloudy":     dict(rain_rate_mmh=0.0,  rain_top_m=0.0,    cloud_lwc_gm3=0.5, cloud_band_m=(1000.0, 3000.0)),
    "light_rain": dict(rain_rate_mmh=5.0,  rain_top_m=3000.0, cloud_lwc_gm3=0.5, cloud_band_m=(1000.0, 3000.0)),
    "heavy_rain": dict(rain_rate_mmh=25.0, rain_top_m=4000.0, cloud_lwc_gm3=1.0, cloud_band_m=(1000.0, 4000.0)),
    "storm":      dict(rain_rate_mmh=50.0, rain_top_m=5000.0, cloud_lwc_gm3=2.0, cloud_band_m=(1000.0, 5000.0)),
}


# --- Atmosphere ---------------------------------------------------------------

class Atmosphere:
    """Path attenuation from the altitude-resolved ITU-R absorption table.

    Gaseous, rain, and cloud losses are integrated along the actual (curved) slant
    path sampled by geo.sample_altitudes: at each point the altitude above the sphere
    sets the local specific attenuation, so the 'surface falls away' effect and the
    stratospheric dry-out are intrinsic. All methods are vectorized over the batch and
    over frequency."""

    def __init__(self, path="data/atmosphere_profile.npz"):
        p = Path(path) if Path(path).is_absolute() else Path(__file__).parent / path
        d = np.load(p)
        self.freq = d["frequency_ghz"]            # (Nf,)
        self.alt = d["altitude_m"]                # (Nh,)
        self.gamma = d["gamma_gas_db_km"]         # (Nh, Nf) specific gaseous atten
        self.kl = d["cloud_kl_db_km_per_gm3"]     # (Nf,) cloud liquid coeff

    def _bilinear_alt_freq(self, table, freq_ghz, altitude_m):
        """Bilinear interp of a (Nh, Nf) table at (altitude, freq); broadcasts."""
        f, h = np.broadcast_arrays(np.asarray(freq_ghz, dtype=float),
                                   np.asarray(altitude_m, dtype=float))
        h = np.clip(h, self.alt[0], self.alt[-1])
        fi = np.clip(np.searchsorted(self.freq, f) - 1, 0, self.freq.size - 2)
        hi = np.clip(np.searchsorted(self.alt, h) - 1, 0, self.alt.size - 2)
        f0, f1 = self.freq[fi], self.freq[fi + 1]
        h0, h1 = self.alt[hi], self.alt[hi + 1]
        tf = np.where(f1 > f0, (f - f0) / (f1 - f0), 0.0)
        th = np.where(h1 > h0, (h - h0) / (h1 - h0), 0.0)
        return ((1 - th) * (1 - tf) * table[hi, fi] + (1 - th) * tf * table[hi, fi + 1]
                + th * (1 - tf) * table[hi + 1, fi] + th * tf * table[hi + 1, fi + 1])

    def _kl_at(self, freq_ghz):
        return np.interp(np.asarray(freq_ghz, dtype=float), self.freq, self.kl)

    def path_loss_db(self, freq_ghz, from_ecef, to_ecef, weather="clear", n_path=64):
        """Total path attenuation (dB) integrated along the curved slant path."""
        alt, ds = geo.sample_altitudes(from_ecef, to_ecef, n_path)      # (...,n), (...)
        f = np.asarray(freq_ghz, dtype=float)
        f_col = f[..., None] if f.ndim else f
        ds_km = ds / 1e3

        gamma = self._bilinear_alt_freq(self.gamma, f_col, alt)          # (...,n) dB/km
        gaseous = np.sum(0.5 * (gamma[..., 1:] + gamma[..., :-1]), axis=-1) * ds_km

        w = WEATHER[weather] if isinstance(weather, str) else weather
        rain = np.zeros_like(gaseous)
        if w["rain_rate_mmh"] > 0:
            # circular pol (tilt 45°) → P.838 coeffs elevation-independent
            k, a = napkin.rain_coefficients(f, elevation_deg=90.0, polarization_tilt_deg=45.0)
            specific = napkin.specific_rain_attenuation(w["rain_rate_mmh"], k, a)   # dB/km
            seg_in = (alt[..., 1:] < w["rain_top_m"]) & (alt[..., :-1] < w["rain_top_m"])
            rain = specific * np.sum(seg_in, axis=-1) * ds_km
        cloud = np.zeros_like(gaseous)
        if w["cloud_lwc_gm3"] > 0:
            specific = self._kl_at(f) * w["cloud_lwc_gm3"]                          # dB/km
            b_lo, b_hi = w["cloud_band_m"]
            in_band = (alt >= b_lo) & (alt <= b_hi)
            seg_in = in_band[..., 1:] & in_band[..., :-1]
            cloud = specific * np.sum(seg_in, axis=-1) * ds_km
        return gaseous + rain + cloud

    def transmittance(self, freq_ghz, from_ecef, to_ecef, weather="clear"):
        return napkin.db_to_fraction(self.path_loss_db(freq_ghz, from_ecef, to_ecef, weather))


# Shared, lazily-loaded singletons so sweeps don't reload the .npz tables per call.
_ATMOSPHERE = None
_SILHOUETTE = None


def get_atmosphere():
    global _ATMOSPHERE
    if _ATMOSPHERE is None:
        _ATMOSPHERE = Atmosphere()
    return _ATMOSPHERE


def get_silhouette():
    global _SILHOUETTE
    if _SILHOUETTE is None:
        _SILHOUETTE = BWBSilhouette()
    return _SILHOUETTE


# --- Geometry (spherical earth, via geo.py) -----------------------------------
# Node positions are given as (downrange_m north, crossrange_m east, altitude_m)
# relative to the station's ground point; _to_ecef maps them onto the sphere so that
# curvature is intrinsic everywhere downstream.

def _to_ecef(pos):
    """(downrange, crossrange, altitude) → ECEF. Accepts (...,3) and broadcasts."""
    p = np.asarray(pos, dtype=float)
    return geo.point(p[..., 0], p[..., 1], p[..., 2])


def geometry(from_pos, to_pos):
    """Slant range (m), elevation of `to` above `from`'s local horizon (deg), and the
    from→to unit direction expressed in `from`'s LOCAL ENU. Spherical earth."""
    a, b = _to_ecef(from_pos), _to_ecef(to_pos)
    slant, elevation = geo.link(a, b)
    return slant, elevation, geo.direction_in_local_enu(a, b)


# --- Emitter / rectenna / relay efficiency pieces -----------------------------

def emitter_aperture_efficiency(scan_deg=0.0, error_efficiency=0.95, scan_rolloff=1.2):
    """Aperture efficiency beyond the optimal near-field taper (already in the
    Goubau capture law): random phase/amplitude errors (Ruze, lumped into
    `error_efficiency`) × electronic scan loss ∝ cos(scan)^rolloff when steering off
    the array boresight (0 for a mechanically-pointed aperture)."""
    scan = np.clip(np.cos(np.radians(scan_deg)), 0.0, 1.0) ** scan_rolloff
    return error_efficiency * scan


def pointing_efficiency(pointing_error_m, spot_radius_m):
    """Focused-spot pointing loss: η = exp(−2·(δ/w)²) for a Gaussian spot of
    radius w and boresight error δ."""
    return np.exp(-2.0 * (pointing_error_m / np.maximum(spot_radius_m, 1e-6)) ** 2)


def polarization_efficiency(polarization="circular", roll_deg=0.0, axial_ratio_loss=0.95):
    """Circular pol: orientation-robust, a flat axial-ratio penalty. Linear pol:
    cos²(roll) mismatch as the airframe banks relative to the beam's E-field."""
    if polarization == "circular":
        return np.full_like(np.asarray(roll_deg, dtype=float), axial_ratio_loss) if np.ndim(roll_deg) else axial_ratio_loss
    return np.cos(np.radians(roll_deg)) ** 2


def rectenna_derate(freq_ghz, beta_deg, polarization="circular", roll_deg=0.0,
                    spot_match=0.90, incidence_floor=0.70):
    """The whole receive-side derate on a fully-conformal rectenna, folded into one
    factor (each piece can only LOWER efficiency, per the design brief):

      RF→DC(f)  · spot_match · incidence(β) · polarization

    • RF→DC(f)   Schottky-rectifier conversion, from napkin's frequency band.
    • spot_match how well a shaped beam fills the (irregular) silhouette + soft-edge
                 (beam-edge) taper — a single settable overlap efficiency.
    • incidence  off-normal incidence over the curved skin detunes elements; mild
                 roll-off with β (angle from belly): 1 overhead → `incidence_floor`
                 edge-on. (The area foreshortening itself is already in the
                 silhouette, so this is only the impedance/scan penalty on top.)
    • polarization  see polarization_efficiency."""
    conv = napkin.rectenna_efficiency_band(freq_ghz) if np.ndim(freq_ghz) == 0 else \
        np.array([napkin.rectenna_efficiency_band(f) for f in np.atleast_1d(freq_ghz)]).reshape(np.shape(freq_ghz))
    # Fold topside (β>90) onto the equivalent belly angle — the airframe is
    # symmetric, so a beam 47° off the top normal is as good as 47° off the belly.
    beta_folded = np.where(np.asarray(beta_deg) > 90.0, 180.0 - np.asarray(beta_deg), beta_deg)
    incidence = incidence_floor + (1.0 - incidence_floor) * np.cos(np.radians(beta_folded))
    pol = polarization_efficiency(polarization, roll_deg)
    return conv * spot_match * incidence * pol, dict(rf_to_dc=conv, spot_match=spot_match,
                                                     incidence=incidence, polarization=pol)


def spot_match_from_aspect(aspect_ratio, shaped=False, fill=0.90):
    """Physically-grounded spot-match efficiency from the silhouette's elongation.

    A ROUND emitter makes a round spot; to cover an elongated silhouette (major:minor
    = aspect) it must span the major axis, wasting the corners → overlap ≈ minor/major
    = 1/aspect (times a soft-edge `fill`). A SHAPE-MATCHED (elliptical) emitter — which
    a phased array can synthesise and re-orient electronically — recovers to `fill`.
    This is the honest version of the constant `spot_match` knob.

    NOTE: superseded for the main link by `overlap_capture`, which does the real 2-D
    beam×silhouette integral. Kept for the illustrative spot-shape figure."""
    aspect = np.maximum(aspect_ratio, 1.0)
    return fill if shaped else fill / aspect


def focused_spot_waist(wavelength_m, slant_m, emitter_radius_m):
    """1/e² intensity waist of a focused aperture at range D: w = √2·λ·D/(π·r_t).

    Chosen so that a ROUND spot on a uniform round rectenna of radius r_r reproduces
    the Goubau capture 1−e^(−τ²) exactly (τ = π·r_t·r_r/(λD)) — the overlap model is a
    strict generalisation of the near-field coupling law to real silhouette shapes."""
    return np.sqrt(2.0) * wavelength_m * slant_m / (np.pi * np.maximum(emitter_radius_m, 1e-6))


def overlap_capture(raster, emitter_radius_m, wavelength_m, slant_m, shaped=False,
                    incidence_floor=0.70, aspect_cap=8.0, return_field=False):
    """Capture as the 2-D overlap of a focused beam spot with the true (rasterized,
    self-occluded) silhouette — replacing √(A/π) sizing + τ-capture + a scalar
    spot_match with one honest integral.

    The spot is centred on the silhouette centroid. A ROUND emitter paints a round
    spot; a SHAPED (phased-array) emitter re-shapes it — at FIXED aperture area, i.e.
    fixed spot etendue w_u·w_v = w0² — into an ellipse aligned to the silhouette's
    principal axes (spot aspect = silhouette aspect, capped at what an array can
    realistically synthesise). Returns:
      geom  — fraction of radiated beam power intercepted by the airframe outline
      eff   — same, weighted by the per-cell incidence response (grazing skin counts
              less: floor + (1−floor)·cos θ_incidence) — the incidence penalty now
              EMERGES from the geometry instead of being a scalar knob.
      w0    — the round-spot waist (m), for reporting/pointing."""
    u, v, inc = raster["u"], raster["v"], raster["incidence"]
    cell_area = raster["cell_area"]
    w0 = focused_spot_waist(wavelength_m, slant_m, emitter_radius_m)
    p_total = 0.5 * np.pi * w0 ** 2                    # ∫∫ exp(−2ρ²/w0²) dA
    if u.size == 0:
        return 0.0, 0.0, float(w0)

    inc_eff = incidence_floor + (1.0 - incidence_floor) * inc
    if not shaped:
        beam = np.exp(-2.0 * (u ** 2 + v ** 2) / w0 ** 2)
    else:
        coords = np.vstack([u, v])
        cov = np.cov(coords) if u.size > 2 else np.eye(2)
        evals, evecs = np.linalg.eigh(cov)            # ascending: [minor, major]
        aspect = np.clip(np.sqrt(max(evals[1], 1e-12) / max(evals[0], 1e-12)), 1.0, aspect_cap)
        w_major, w_minor = w0 * np.sqrt(aspect), w0 / np.sqrt(aspect)
        proj = evecs.T @ coords                        # rows: [along minor, along major]
        beam = np.exp(-2.0 * (proj[1] ** 2 / w_major ** 2 + proj[0] ** 2 / w_minor ** 2))
    geom = float(np.sum(beam) * cell_area / p_total)
    eff = float(np.sum(beam * inc_eff) * cell_area / p_total)
    if return_field:
        return geom, eff, float(w0), dict(u=u, v=v, beam=beam, incidence=inc,
                                          incidence_eff=inc_eff, w0=float(w0))
    return geom, eff, float(w0)


# --- Device-efficiency scenarios ---------------------------------------------
# The DC→RF PAE falls with frequency partly from device physics (smaller high-band
# devices, lower breakdown, more parasitic loss — a REAL floor) and partly from market
# maturity (few efficient 10–30 GHz PAs today). We can't assume the physics away, but
# the maturity gap could close, so treat the high-band PAE as a scenario. All three
# share the SAME low-frequency anchor (0.65 @ 5.8 GHz — mature GaN/magnetron); only the
# 34.8 GHz anchor moves, so the scenarios differ ONLY in the high-band slope. #(verify)
PAE_SCENARIOS = {
    "today":  (0.65, 0.40),   # current saturated-CW GaN literature
    "medium": (0.65, 0.53),   # a decade of dedicated high-band development
    "flat":   (0.65, 0.65),   # device parity across bands — isolates geometry+weather
}


def pae_dc_to_rf(frequency_ghz, scenario="today"):
    """GaN power-amplifier PAE vs frequency under a named device scenario."""
    lo, hi = PAE_SCENARIOS[scenario]
    slope = (hi - lo) / (34.8 - 5.8)
    return np.clip(lo + slope * (np.asarray(frequency_ghz, dtype=float) - 5.8),
                   min(lo, hi) - 0.05, 0.9)


def eta_grid_to_rf(frequency_ghz, scenario="today", eta_ac_dc=0.98):
    """Station grid→RF = SiC AC/DC front end (mature ~0.98) × GaN PA PAE(f, scenario)."""
    return eta_ac_dc * pae_dc_to_rf(frequency_ghz, scenario)


def relay_convert_efficiency(f_in_ghz, f_out_ghz, mode="active", reflectarray_reflectivity=0.90,
                             surface_error_frac=1 / 50, stage_efficiency=0.70,
                             pae_scenario="today"):
    """Relay conversion efficiency.
      active     : rectify the incoming RF (RF→DC at f_in) then re-amplify (DC→RF PAE
                   at f_out). Runs off its own rectified DC, so only the PA's PAE
                   applies on re-emit — NOT the grid AC/DC stage.
      multiplier : a (more-or-less) PASSIVE nonlinear-reactance harmonic multiplier
                   (Manley-Rowe permits ~100% for an ideal reactance; realized as ×2 /
                   ×2·×3 stages at `stage_efficiency` each, from napkin.cascade_efficiency)
                   × an aperture/coupling factor. Up-converts 5.8→n·5.8 with NO
                   rectify+re-amp cliff. f_out sets the harmonic order.
      passive    : reflectarray reflectivity × Ruze surface-tolerance loss; no frequency
                   change (caller must set f_out = f_in)."""
    if mode == "active":
        return napkin.rectenna_efficiency_band(f_in_ghz) * pae_dc_to_rf(f_out_ghz, pae_scenario)
    if mode == "multiplier":
        order = napkin.multiplier_factor_prime(f_in_ghz, f_out_ghz)
        chain, _ = napkin.cascade_efficiency(order, stage_efficiency)
        return reflectarray_reflectivity * chain
    ruze = np.exp(-((4 * np.pi * surface_error_frac) ** 2))
    return reflectarray_reflectivity * ruze


# --- Aperture sizing ----------------------------------------------------------

def receiver_radius_from_silhouette(silhouette_area_m2):
    """Effective circular-aperture radius of the conformal rectenna = √(A_sil/π)."""
    return np.sqrt(silhouette_area_m2 / np.pi)


def size_emitter_diameter(freq_ghz, slant_m, receiver_radius_m, tau_target=2.0):
    """Emitter diameter for a target near-field coupling τ (τ=2 → ~98% capture)."""
    return 2.0 * napkin.required_emitter_radius(freq_ghz * 1e9, slant_m, receiver_radius_m, tau_target)


# --- One hop ------------------------------------------------------------------

@dataclass
class HopResult:
    freq_ghz: float
    slant_m: float
    elevation_deg: float
    emitter_diam_m: float
    tau: float
    capture: float
    atmos_transmittance: float
    atmos_loss_db: float
    emitter_aperture_eff: float
    blocked: object = False

    def radiated_to_incident(self):
        """Fraction of radiated RF that arrives as RF on the receiver aperture."""
        return self.capture * self.atmos_transmittance


def hop(atmos, freq_ghz, from_pos, to_pos, receiver_radius_m, weather="clear",
        tau_target=2.0, emitter_diam_m=None, scan_deg=0.0, error_efficiency=0.95,
        n_path=64):
    """One emitter→receiver hop on the spherical earth: size (or accept) the emitter
    aperture, compute near-field capture, curved-path atmospheric transmittance (zeroed
    if the path is blocked by the horizon), and emitter aperture efficiency.
    `from_pos`/`to_pos` are (downrange, crossrange, altitude) triples."""
    a, b = _to_ecef(from_pos), _to_ecef(to_pos)
    slant, elevation = geo.link(a, b)

    if emitter_diam_m is None:
        emitter_diam_m = size_emitter_diameter(freq_ghz, slant, receiver_radius_m, tau_target)
    emitter_radius = np.asarray(emitter_diam_m) / 2.0
    tau = napkin.coupling_tau(freq_ghz * 1e9, slant, emitter_radius, receiver_radius_m)
    capture = napkin.collection_efficiency(tau)
    blocked = geo.horizon_blocked(a, b, n_path)
    loss_db = atmos.path_loss_db(freq_ghz, a, b, weather, n_path)
    transmittance = napkin.db_to_fraction(loss_db) * np.where(blocked, 0.0, 1.0)
    ap_eff = emitter_aperture_efficiency(scan_deg, error_efficiency)
    return HopResult(freq_ghz=freq_ghz, slant_m=slant, elevation_deg=elevation,
                     emitter_diam_m=np.asarray(emitter_diam_m), tau=tau, capture=capture,
                     atmos_transmittance=transmittance, atmos_loss_db=loss_db,
                     emitter_aperture_eff=ap_eff, blocked=blocked)


# --- Full link ----------------------------------------------------------------

@dataclass
class LinkConfig:
    """Everything needed to evaluate one link geometry + tech set."""
    station_pos: tuple = (0.0, 0.0, 0.0)          # ENU metres (sea level station)
    plane_pos: tuple = (150_000.0, 0.0, 11_000.0)  # 150 km downrange at 11 km
    plane_attitude: tuple = (0.0, 0.0, 0.0)        # heading, pitch, roll (deg)
    relay_pos: tuple | None = None                 # None → direct link

    f_emit_plane_ghz: float = 30.0                 # direct station→plane
    f_emit_relay_ghz: float = 5.8                  # station→relay (weather-robust up)
    f_relay_plane_ghz: float = 30.0                # relay→plane (small plane aperture)

    relay_mode: str = "active"                     # "active" | "multiplier" | "passive"
    relay_aperture_diam_m: float = 60.0            # relay receive/transmit aperture
    relay_stage_efficiency: float = 0.70           # per-stage eff of a multiplier relay
    weather: str = "clear"
    pae_scenario: str = "today"                    # "today" | "medium" | "flat"

    # Emitter aperture: fixed physical diameter (m) if given, else sized to tau_target
    # per geometry. A real station's emitter is FIXED, so capture DROPS at low
    # elevation instead of the emitter growing — set this for realistic pass profiles.
    emitter_diam_m: float | None = None
    tau_target: float = 2.0
    polarization: str = "circular"
    shaped_spot: bool = True                        # phased-array elliptical spot (vs round aperture)
    spot_match: float = 0.90                        # deprecated: overlap_capture supersedes it
    incidence_floor: float = 0.70                   # rectenna element angular-response floor
    emitter_error_efficiency: float = 0.95
    pointing_error_m: float = 2.0

    _silhouette: BWBSilhouette = field(default=None, repr=False)

    def silhouette(self):
        if self._silhouette is None:
            self._silhouette = get_silhouette()
        return self._silhouette


# Silhouette rasters depend only on the look direction (β, α) — not on frequency,
# range, or weather — so cache them: a weather/architecture sweep reuses one raster
# per look angle instead of re-projecting 10⁵ points every call.
_RASTER_CACHE = {}


def _cached_raster(beta_deg, alpha_deg, n_grid=120):
    key = (round(float(beta_deg), 1), round(float(alpha_deg), 1), n_grid)
    raster = _RASTER_CACHE.get(key)
    if raster is None:
        look = direction_from_beta_alpha(key[0], key[1])
        samples, sample_normals, _ = get_surface_samples()
        raster = rasterize_silhouette(look, samples, sample_normals, n_grid=n_grid)
        _RASTER_CACHE[key] = raster
    return raster


def plane_capture(cfg, emitter_pos, freq_ghz, emitter_diam_m=None, n_grid=120):
    """The real receive-side geometry the plane presents to an emitter at `emitter_pos`:
    rasterize the (self-occluded) silhouette toward the emitter and overlap the focused
    beam spot with it. Returns a dict with the true silhouette area, the incidence-
    weighted capture (`capture`), the bare geometric overlap, the spot waist, the
    emitter diameter used (sized to τ_target on the equal-area disk if not fixed), and
    the look geometry. Scalar geometry only (one plane position)."""
    plane = np.asarray(cfg.plane_pos, dtype=float)
    emitter = np.asarray(emitter_pos, dtype=float)
    slant, elevation, unit_plane_to_emitter = geometry(plane, emitter)
    heading, pitch, roll = cfg.plane_attitude
    beta, alpha = beta_alpha_from_enu(unit_plane_to_emitter, heading, pitch, roll)
    raster = _cached_raster(beta, alpha, n_grid)
    area = raster["area"]
    r_recv = receiver_radius_from_silhouette(max(area, 1e-6))

    if emitter_diam_m is None:                     # size to τ_target on the equal-area disk
        emitter_diam_m = float(size_emitter_diameter(freq_ghz, slant, r_recv, cfg.tau_target))
    wavelength = napkin.SPEED_OF_LIGHT / (freq_ghz * 1e9)
    geom, capture, w0 = overlap_capture(raster, emitter_diam_m / 2.0, wavelength, slant,
                                        shaped=cfg.shaped_spot, incidence_floor=cfg.incidence_floor)
    return dict(area=area, r_recv=r_recv, beta=float(beta), roll=float(roll),
                slant=float(slant), elevation=float(elevation), capture=capture,
                capture_geom=geom, spot_waist_m=w0, emitter_diam_m=float(emitter_diam_m))


def _evaluate_scalar(cfg: LinkConfig, delivered_mw=None):
    """End-to-end evaluation for ONE plane position (scalar geometry)."""
    atmos = get_atmosphere()
    station = np.asarray(cfg.station_pos, dtype=float)
    plane = np.asarray(cfg.plane_pos, dtype=float)

    if cfg.relay_pos is None:
        # ---- direct emitter → plane (emitter at station_pos; lofted for the tether case) ----
        f = cfg.f_emit_plane_ghz
        pc = plane_capture(cfg, station, f, cfg.emitter_diam_m)
        a, b = _to_ecef(station), _to_ecef(plane)
        blocked = bool(geo.horizon_blocked(a, b, 64))
        atmos_t = float(napkin.db_to_fraction(atmos.path_loss_db(f, a, b, cfg.weather))) * (0.0 if blocked else 1.0)
        ap_eff = float(emitter_aperture_efficiency(0.0, cfg.emitter_error_efficiency))
        point = float(pointing_efficiency(cfg.pointing_error_m, pc["spot_waist_m"]))
        conv = float(napkin.rectenna_efficiency_band(f))
        pol = float(polarization_efficiency(cfg.polarization, pc["roll"]))
        eta_dc_rf = float(eta_grid_to_rf(f, cfg.pae_scenario))

        eta = eta_dc_rf * ap_eff * pc["capture"] * atmos_t * point * conv * pol
        stages = dict(
            architecture="direct", dc_to_rf=eta_dc_rf, emitter_aperture=ap_eff,
            capture=pc["capture"], capture_geom=pc["capture_geom"], atmos=atmos_t,
            pointing=point, rectenna_rf_to_dc=conv, polarization=pol,
            rectenna_total=conv * pol, silhouette_m2=pc["area"],
            receiver_radius_m=pc["r_recv"], beta_deg=pc["beta"], spot_waist_m=pc["spot_waist_m"],
            emitter_diam_m=pc["emitter_diam_m"], slant_km=pc["slant"] / 1e3,
            elevation_deg=pc["elevation"], blocked=blocked)

    else:
        # ---- relayed station → relay → plane ----
        relay = np.asarray(cfg.relay_pos, dtype=float)
        r_relay = cfg.relay_aperture_diam_m / 2.0
        f1 = cfg.f_emit_relay_ghz
        f2 = f1 if cfg.relay_mode == "passive" else cfg.f_relay_plane_ghz

        # up-leg: emitter → relay is a plain circular aperture (Goubau τ-capture).
        h1 = hop(atmos, f1, station, relay, r_relay, cfg.weather, cfg.tau_target,
                 error_efficiency=cfg.emitter_error_efficiency)
        # down-leg: relay → plane uses the real silhouette overlap; the emitting
        # aperture is the relay's own (fixed) transmit array.
        pc = plane_capture(cfg, relay, f2, cfg.relay_aperture_diam_m)
        a2, b2 = _to_ecef(relay), _to_ecef(plane)
        blocked2 = bool(geo.horizon_blocked(a2, b2, 64))
        atmos_dn = float(napkin.db_to_fraction(atmos.path_loss_db(f2, a2, b2, cfg.weather))) * (0.0 if blocked2 else 1.0)

        convert = float(relay_convert_efficiency(f1, f2, cfg.relay_mode,
                        stage_efficiency=cfg.relay_stage_efficiency, pae_scenario=cfg.pae_scenario))
        point = float(pointing_efficiency(cfg.pointing_error_m, pc["spot_waist_m"]))
        conv = float(napkin.rectenna_efficiency_band(f2))
        pol = float(polarization_efficiency(cfg.polarization, pc["roll"]))
        eta_dc_rf = float(eta_grid_to_rf(f1, cfg.pae_scenario))

        eta = (eta_dc_rf * float(h1.emitter_aperture_eff) * float(h1.capture) * float(h1.atmos_transmittance)
               * convert * float(emitter_aperture_efficiency(0.0, cfg.emitter_error_efficiency))
               * pc["capture"] * atmos_dn * point * conv * pol)
        stages = dict(
            architecture=f"relay-{cfg.relay_mode}", dc_to_rf=eta_dc_rf,
            emitter_aperture=float(h1.emitter_aperture_eff),
            capture_up=float(h1.capture), atmos_up=float(h1.atmos_transmittance),
            relay_convert=convert, relay_aperture=float(emitter_aperture_efficiency(0.0, cfg.emitter_error_efficiency)),
            capture_down=pc["capture"], capture_down_geom=pc["capture_geom"], atmos_down=atmos_dn,
            pointing=point, rectenna_rf_to_dc=conv, polarization=pol, rectenna_total=conv * pol,
            capture=pc["capture"], atmos=atmos_dn,
            silhouette_m2=pc["area"], receiver_radius_m=pc["r_recv"], beta_deg=pc["beta"],
            spot_waist_m=pc["spot_waist_m"], emitter_diam_m=pc["emitter_diam_m"],
            emitter_diam_up_m=float(h1.emitter_diam_m),
            slant_up_km=h1.slant_m / 1e3, slant_down_km=pc["slant"] / 1e3,
            elevation_up_deg=float(h1.elevation_deg), elevation_down_deg=pc["elevation"],
            elevation_deg=pc["elevation"], blocked=bool(np.logical_or(h1.blocked, blocked2)))

    stages["eta_end_to_end"] = eta
    stages["feasible"] = (not stages["blocked"]) and eta > 1e-4
    if delivered_mw is not None:
        stages["delivered_mw"] = delivered_mw
        stages["grid_mw"] = delivered_mw / eta if eta > 0 else np.inf
        stages["waste_mw"] = stages["grid_mw"] - delivered_mw
    return stages


def evaluate(cfg: LinkConfig, delivered_mw=None):
    """Evaluate a link end to end (grid busbar → DC at the rectenna terminals).

    Capture is the real 2-D overlap of the focused beam spot with the rasterized,
    self-occluded silhouette (see `plane_capture`/`overlap_capture`) — the incidence
    penalty and the spot-vs-silhouette shape mismatch are computed, not assumed. If
    `plane_pos` is an (N,3) array the scalar core is mapped over the N positions and
    every array-valued field is stacked (silhouette rasterization is per-geometry)."""
    plane = np.asarray(cfg.plane_pos, dtype=float)
    if plane.ndim == 1:
        return _evaluate_scalar(cfg, delivered_mw)

    rows = [_evaluate_scalar(_with_plane(cfg, p), delivered_mw) for p in plane]
    out = {}
    for key in rows[0]:
        vals = [r[key] for r in rows]
        out[key] = vals[0] if isinstance(vals[0], str) else np.array(vals)
    return out


def _with_plane(cfg: LinkConfig, plane_pos):
    """Copy `cfg` with a single plane position (used to map evaluate over a trajectory)."""
    return LinkConfig(**{**cfg.__dict__, "plane_pos": tuple(np.asarray(plane_pos, float)),
                         "_silhouette": cfg._silhouette})


def best_path(cfg: LinkConfig, delivered_mw=None):
    """Evaluate direct and relayed (given the relay position) and return the better,
    plus both, so weather-driven path selection is explicit."""
    direct_cfg = LinkConfig(**{**cfg.__dict__, "relay_pos": None, "_silhouette": cfg.silhouette()})
    direct = evaluate(direct_cfg, delivered_mw)
    if cfg.relay_pos is None:
        return dict(chosen=direct, direct=direct, relayed=None)
    relayed = evaluate(cfg, delivered_mw)
    chosen = direct if direct["eta_end_to_end"] >= relayed["eta_end_to_end"] else relayed
    return dict(chosen=chosen, direct=direct, relayed=relayed)


# --- Pass profile (the boost-over-station window) ------------------------------

def emitter_for_reference(frequency_ghz, station_pos, plane_alt_m=11_000.0,
                          reference_elevation_deg=60.0, tau_target=2.0,
                          silhouette=None):
    """Physical emitter diameter that hits `tau_target` at a chosen reference
    elevation (e.g. near overhead) — the FIXED aperture a station is built with."""
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
    """Sweep a plane along a straight over-flight past a fixed-emitter station and
    return per-position link metrics — the realistic 'how big is the powering
    window' picture behind the boost-over-station concept. Fully vectorized.

    The emitter is FIXED (built for `reference_elevation_deg`), so capture and hence
    delivered-power capability fall away from overhead. Returns a dict of arrays."""
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


# --- Architectures + trajectory-integrated flyby ------------------------------

def standard_architectures(station_pos=(0.0, 0.0, 0.0), weather="clear", pae_scenario="today",
                           relay_alt_m=20_000.0, tether_alt_m=20_000.0,
                           emitter_5p8_m=78.0, relay_aperture_m=40.0, emitter_tether_m=15.0,
                           relay_multiplier_out_ghz=17.4, include_reference=True):
    """The realistic contenders + labelled reference cases, as {name: cfg-kwargs} with
    plane_pos left OUT (the flyby fills it per point). Emitters are FIXED apertures.

    Contenders:
      • 5.8 direct              — weatherproof low-f from the ground (big emitter).
      • relay ×N mult 5.8→hi-f  — ground 5.8→relay aloft→passive multiplier→hi-f down
                                  (keeps the hi-f leg in dry stratospheric air).
      • tether hi-f direct      — hi-f emitter lofted on a tether (no relay, no
                                  frequency conversion; short dry path). No altitude
                                  restriction here — a modelling probe, not yet costed.
    Reference (not real contenders, kept for contrast):
      • 30 direct (ground)      — rains out; illustrative only.
      • relay active 5.8→30     — dominated by the passive multiplier in every weather."""
    relay = (station_pos[0], station_pos[1], relay_alt_m)          # aloft, above the station
    tether = (station_pos[0], station_pos[1], tether_alt_m)
    common = dict(weather=weather, pae_scenario=pae_scenario)
    contenders = {
        "5.8 direct": dict(station_pos=station_pos, f_emit_plane_ghz=5.8, emitter_diam_m=emitter_5p8_m),
        f"relay ×{round(relay_multiplier_out_ghz/5.8)} mult 5.8→{relay_multiplier_out_ghz:g}":
            dict(station_pos=station_pos, relay_pos=relay, relay_mode="multiplier",
                 f_emit_relay_ghz=5.8, f_relay_plane_ghz=relay_multiplier_out_ghz,
                 relay_aperture_diam_m=relay_aperture_m),
        "tether 30 direct": dict(station_pos=tether, f_emit_plane_ghz=30.0, emitter_diam_m=emitter_tether_m),
    }
    reference = {
        "30 direct (ground)": dict(station_pos=station_pos, f_emit_plane_ghz=30.0, emitter_diam_m=15.0),
        "relay active 5.8→30": dict(station_pos=station_pos, relay_pos=relay, relay_mode="active",
                                    f_emit_relay_ghz=5.8, f_relay_plane_ghz=30.0,
                                    relay_aperture_diam_m=relay_aperture_m),
    }
    archs = {**contenders, **(reference if include_reference else {})}
    return {name: {**common, **kw} for name, kw in archs.items()}


def flyby(arch_kwargs, plane_alt_m=11_000.0, lateral_offset_m=0.0, station_spacing_km=200.0,
          speed_ms=250.0, delivered_mw=10.0, n_points=81):
    """Fly the plane through the window a single station serves (±spacing/2 along track,
    passing overhead at x=0) and evaluate the link at every step — the actual rising-
    and-falling tooth of the sawtooth, not a snapshot. The emitter is FIXED, so capture
    falls away from overhead as the geometry (elevation, silhouette, spot-fit) worsens.

    Figures of merit over the served window:
      mean_eta   — time-average end-to-end η (a battery-buffered plane; delivered energy
                   per grid energy).
      hmean_eta  — harmonic-mean η (a plane that must be fed CONTINUOUSLY at constant
                   power: dominated by the worst points).
      grid_energy_per_pass_gj — grid energy a station must spend to keep the plane fed
                   at constant `delivered_mw` for one pass (∝ 1/hmean): a fuel-cost
                   proxy, lower is better. ∞ if any point of the pass is unfeasible."""
    half = station_spacing_km * 1e3 / 2.0
    x = np.linspace(-half, half, n_points)
    plane = np.stack([x, np.full_like(x, lateral_offset_m), np.full_like(x, plane_alt_m)], axis=1)
    cfg = LinkConfig(plane_pos=plane, **arch_kwargs)
    s = evaluate(cfg, delivered_mw)
    eta = np.asarray(s["eta_end_to_end"], dtype=float)
    dt = (x[1] - x[0]) / speed_ms
    peak = float(np.max(eta))
    # Deliverable profile at a fixed emitter grid power (sized to hit delivered_mw at the
    # best point of the pass) — the shape of one tooth of the sawtooth.
    deliverable_mw = (delivered_mw / peak if peak > 0 else np.inf) * eta
    # Continuous feeding is only possible if the link stays feasible all pass; a single
    # rained-out / below-horizon point makes constant-power delivery impossible → ∞.
    with np.errstate(divide="ignore"):
        grid_energy_gj = (float(delivered_mw * np.sum(1.0 / eta) * dt / 1e3)
                          if np.all(eta > 1e-4) else np.inf)
    return dict(
        x_km=x / 1e3, eta=eta, elevation_deg=np.asarray(s["elevation_deg"], float),
        silhouette_m2=np.asarray(s["silhouette_m2"], float),
        capture=np.asarray(s["capture"], float), spot_waist_m=np.asarray(s["spot_waist_m"], float),
        deliverable_mw=deliverable_mw, mean_eta=float(np.mean(eta)),
        hmean_eta=float(len(eta) / np.sum(1.0 / np.maximum(eta, 1e-9))),
        peak_eta=peak, grid_energy_per_pass_gj=grid_energy_gj, dt=dt)


def corridor_table(station_spacing_km=200.0, weather="clear", pae_scenario="today",
                   plane_alt_m=11_000.0, delivered_mw=10.0, include_reference=True):
    """Run every standard architecture over one flyby and return {name: flyby-result}."""
    archs = standard_architectures(weather=weather, pae_scenario=pae_scenario,
                                   include_reference=include_reference)
    return {name: flyby(kw, plane_alt_m=plane_alt_m, station_spacing_km=station_spacing_km,
                        delivered_mw=delivered_mw) for name, kw in archs.items()}


# --- Reporting ----------------------------------------------------------------

def waterfall(stages):
    """Ordered (stage, efficiency) chain for a result dict — for printing/plotting."""
    if stages["architecture"] == "direct":
        keys = [("DC→RF (PAE)", "dc_to_rf"), ("emitter aperture", "emitter_aperture"),
                ("near-field capture", "capture"), ("atmosphere", "atmos"),
                ("pointing", "pointing"), ("rectenna RF→DC+match", "rectenna_total")]
    else:
        keys = [("DC→RF (PAE)", "dc_to_rf"), ("emitter aperture", "emitter_aperture"),
                ("capture up", "capture_up"), ("atmosphere up", "atmos_up"),
                ("relay convert", "relay_convert"), ("relay aperture", "relay_aperture"),
                ("capture down", "capture_down"), ("atmosphere down", "atmos_down"),
                ("pointing", "pointing"), ("rectenna RF→DC+match", "rectenna_total")]
    return [(label, float(stages[k])) for label, k in keys]


def print_report(name, stages):
    print(f"\n=== {name}  [{stages['architecture']}] ===")
    running = 1.0
    for label, eff in waterfall(stages):
        running *= eff
        print(f"  {label:24s} × {eff:6.3f}   → cumulative {running:6.3f}")
    print(f"  {'END-TO-END':24s} = {float(stages['eta_end_to_end']):6.3f}")
    if "grid_mw" in stages:
        print(f"  delivered {float(stages['delivered_mw']):5.2f} MW → grid "
              f"{float(stages['grid_mw']):6.2f} MW  (waste {float(stages['waste_mw']):5.2f} MW)")
    ed = stages.get("emitter_diam_m", stages.get("emitter_diam_down_m"))
    print(f"  emitter Ø {float(np.atleast_1d(ed)[0]):.0f} m,  silhouette "
          f"{float(np.atleast_1d(stages['silhouette_m2'])[0]):.0f} m² "
          f"(r_recv {float(np.atleast_1d(stages['receiver_radius_m'])[0]):.1f} m), "
          f"β {float(np.atleast_1d(stages['beta_deg'])[0]):.0f}°")


# --- Named tech cases ---------------------------------------------------------

def reference_geometry(elevation_deg=30.0, plane_alt_km=11.0):
    """Station at origin (sea level); plane placed downrange (on the sphere) so the
    station sees it at `elevation_deg`. Returns (station_pos, plane_pos) in
    (downrange, crossrange, altitude) coordinates."""
    downrange = float(geo.downrange_for_elevation(elevation_deg, plane_alt_km * 1e3))
    return (0.0, 0.0, 0.0), (downrange, 0.0, plane_alt_km * 1e3)


def tech_cases(elevation_deg=30.0, weather="clear", delivered_mw=10.0):
    """Snapshot evaluation of the standard architectures at one elevation (kept for the
    waterfall/spot-check figures; the flyby is the primary figure of merit)."""
    station, plane = reference_geometry(elevation_deg)
    archs = standard_architectures(station_pos=station, weather=weather)
    return {name: evaluate(LinkConfig(plane_pos=plane, **kw), delivered_mw)
            for name, kw in archs.items()}


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    print("Beam-link engine — flyby-integrated over one station's serving window "
          "(plane 11 km, 200 km spacing, 10 MW delivered)\n")
    print(f"{'architecture':26s} {'weather':11s} {'mean η':>7s} {'harm η':>7s} "
          f"{'peak η':>7s} {'grid GJ/pass':>13s}")
    for weather in ("clear", "heavy_rain", "storm"):
        for name, r in corridor_table(weather=weather).items():
            gj = f"{r['grid_energy_per_pass_gj']:12.1f}" if np.isfinite(r['grid_energy_per_pass_gj']) else f"{'∞':>12s}"
            print(f"  {name:24s} {weather:11s} {r['mean_eta']*100:6.1f}% {r['hmean_eta']*100:6.1f}% "
                  f"{r['peak_eta']*100:6.1f}% {gj}")
        print()
