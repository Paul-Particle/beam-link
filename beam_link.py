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
from bwb_model import BWBSilhouette, beta_alpha_from_enu


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

    Gaseous loss is the vertical cumulative integral between the two path
    altitudes, divided by sin(elevation) for the slant. Rain and cloud are added
    only over the altitude bands where they exist. All methods are vectorized."""

    def __init__(self, path="data/atmosphere_profile.npz"):
        p = Path(path) if Path(path).is_absolute() else Path(__file__).parent / path
        d = np.load(p)
        self.freq = d["frequency_ghz"]            # (Nf,)
        self.alt = d["altitude_m"]                # (Nh,)
        self.gamma = d["gamma_gas_db_km"]         # (Nh, Nf)
        self.cumulative = d["gamma_gas_cumulative_db"]   # (Nh, Nf) vertical dB 0→h
        self.kl = d["cloud_kl_db_km_per_gm3"]     # (Nf,)

    def _cumulative_at(self, freq_ghz, altitude_m):
        """Bilinear-interpolate vertical cumulative gaseous dB at (altitude, freq)."""
        f = np.asarray(freq_ghz, dtype=float)
        h = np.asarray(altitude_m, dtype=float)
        fi = np.clip(np.searchsorted(self.freq, f) - 1, 0, self.freq.size - 2)
        hi = np.clip(np.searchsorted(self.alt, h) - 1, 0, self.alt.size - 2)
        f0, f1 = self.freq[fi], self.freq[fi + 1]
        h0, h1 = self.alt[hi], self.alt[hi + 1]
        tf = np.where(f1 > f0, (f - f0) / (f1 - f0), 0.0)
        th = np.where(h1 > h0, (h - h0) / (h1 - h0), 0.0)
        c = self.cumulative
        return ((1 - th) * (1 - tf) * c[hi, fi] + (1 - th) * tf * c[hi, fi + 1]
                + th * (1 - tf) * c[hi + 1, fi] + th * tf * c[hi + 1, fi + 1])

    def _kl_at(self, freq_ghz):
        f = np.asarray(freq_ghz, dtype=float)
        return np.interp(f, self.freq, self.kl)

    def gaseous_db(self, freq_ghz, h_low_m, h_high_m, elevation_deg):
        delta = self._cumulative_at(freq_ghz, h_high_m) - self._cumulative_at(freq_ghz, h_low_m)
        # Path length depends on the angle from horizontal; a downward (depression)
        # leg has the same slant/vertical ratio as the equivalent upward one.
        grazing = np.clip(np.abs(elevation_deg), 0.5, 90.0)
        return np.abs(delta) / np.sin(np.radians(grazing))

    def _band_path_km(self, h_low_m, h_high_m, band_low_m, band_high_m, elevation_deg):
        """Slant path length (km) that a link between two altitudes spends inside an
        altitude band [band_low, band_high]."""
        lo = np.maximum(np.minimum(h_low_m, h_high_m), band_low_m)
        hi = np.minimum(np.maximum(h_low_m, h_high_m), band_high_m)
        vertical = np.clip(hi - lo, 0.0, None)
        grazing = np.clip(np.abs(elevation_deg), 0.5, 90.0)
        return vertical / np.sin(np.radians(grazing)) / 1e3

    def rain_db(self, freq_ghz, h_low_m, h_high_m, elevation_deg, weather):
        w = WEATHER[weather] if isinstance(weather, str) else weather
        if w["rain_rate_mmh"] <= 0:
            return np.zeros_like(np.asarray(freq_ghz, dtype=float) + np.asarray(elevation_deg, dtype=float))
        # Circular polarization (tilt 45°) → the P.838 coefficients become
        # elevation-independent; the slant geometry is handled by _band_path_km.
        k, alpha = napkin.rain_coefficients(freq_ghz, elevation_deg=90.0, polarization_tilt_deg=45.0)
        specific = napkin.specific_rain_attenuation(w["rain_rate_mmh"], k, alpha)   # dB/km
        path_km = self._band_path_km(h_low_m, h_high_m, 0.0, w["rain_top_m"], elevation_deg)
        return specific * path_km

    def cloud_db(self, freq_ghz, h_low_m, h_high_m, elevation_deg, weather):
        w = WEATHER[weather] if isinstance(weather, str) else weather
        if w["cloud_lwc_gm3"] <= 0:
            return np.zeros_like(np.asarray(freq_ghz, dtype=float) + np.asarray(elevation_deg, dtype=float))
        specific = self._kl_at(freq_ghz) * w["cloud_lwc_gm3"]                       # dB/km
        b_lo, b_hi = w["cloud_band_m"]
        path_km = self._band_path_km(h_low_m, h_high_m, b_lo, b_hi, elevation_deg)
        return specific * path_km

    def path_loss_db(self, freq_ghz, h_low_m, h_high_m, elevation_deg, weather="clear"):
        return (self.gaseous_db(freq_ghz, h_low_m, h_high_m, elevation_deg)
                + self.rain_db(freq_ghz, h_low_m, h_high_m, elevation_deg, weather)
                + self.cloud_db(freq_ghz, h_low_m, h_high_m, elevation_deg, weather))

    def transmittance(self, freq_ghz, h_low_m, h_high_m, elevation_deg, weather="clear"):
        return napkin.db_to_fraction(self.path_loss_db(freq_ghz, h_low_m, h_high_m, elevation_deg, weather))


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


# --- Geometry -----------------------------------------------------------------

def geometry(from_pos, to_pos):
    """Slant range (m), elevation of `to` seen from `from` (deg), and the unit
    direction from→to, all in ENU. Accepts (...,3) arrays and broadcasts."""
    a = np.asarray(from_pos, dtype=float)
    b = np.asarray(to_pos, dtype=float)
    delta = b - a
    slant = np.linalg.norm(delta, axis=-1)
    horizontal = np.linalg.norm(delta[..., :2], axis=-1)
    elevation = np.degrees(np.arctan2(delta[..., 2], horizontal))
    unit = delta / (slant[..., None] + 1e-30)
    return slant, elevation, unit


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


def relay_convert_efficiency(f_in_ghz, f_out_ghz, mode="active", reflectarray_reflectivity=0.90,
                             surface_error_frac=1 / 50):
    """Relay conversion efficiency.
      active  : rectify the incoming RF (RF→DC at f_in) then re-amplify (DC→RF PAE at
                f_out). The relay runs off its own rectified DC, so only the PA's PAE
                applies on the re-emit side — NOT the grid AC/DC front-end stage.
      passive : reflectarray reflectivity × Ruze surface-tolerance loss at f_in; no
                frequency change (caller must set f_out = f_in)."""
    if mode == "active":
        return napkin.rectenna_efficiency_band(f_in_ghz) * napkin.pae_dc_to_rf(f_out_ghz)
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

    def radiated_to_incident(self):
        """Fraction of radiated RF that arrives as RF on the receiver aperture."""
        return self.capture * self.atmos_transmittance


def hop(atmos, freq_ghz, from_pos, to_pos, receiver_radius_m, weather="clear",
        tau_target=2.0, emitter_diam_m=None, scan_deg=0.0, error_efficiency=0.95):
    """One emitter→receiver hop: size (or accept) the emitter aperture, compute
    near-field capture, atmospheric transmittance, and emitter aperture efficiency."""
    slant, elevation, _ = geometry(from_pos, to_pos)
    h_low = np.minimum(np.asarray(from_pos)[..., 2], np.asarray(to_pos)[..., 2])
    h_high = np.maximum(np.asarray(from_pos)[..., 2], np.asarray(to_pos)[..., 2])

    if emitter_diam_m is None:
        emitter_diam_m = size_emitter_diameter(freq_ghz, slant, receiver_radius_m, tau_target)
    emitter_radius = np.asarray(emitter_diam_m) / 2.0
    tau = napkin.coupling_tau(freq_ghz * 1e9, slant, emitter_radius, receiver_radius_m)
    capture = napkin.collection_efficiency(tau)
    loss_db = atmos.path_loss_db(freq_ghz, h_low, h_high, elevation, weather)
    transmittance = napkin.db_to_fraction(loss_db)
    ap_eff = emitter_aperture_efficiency(scan_deg, error_efficiency)
    return HopResult(freq_ghz=freq_ghz, slant_m=slant, elevation_deg=elevation,
                     emitter_diam_m=np.asarray(emitter_diam_m), tau=tau, capture=capture,
                     atmos_transmittance=transmittance, atmos_loss_db=loss_db,
                     emitter_aperture_eff=ap_eff)


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

    relay_mode: str = "active"                     # "active" | "passive"
    relay_aperture_diam_m: float = 60.0            # relay receive/transmit aperture
    weather: str = "clear"

    # Emitter aperture: fixed physical diameter (m) if given, else sized to tau_target
    # per geometry. A real station's emitter is FIXED, so capture DROPS at low
    # elevation instead of the emitter growing — set this for realistic pass profiles.
    emitter_diam_m: float | None = None
    tau_target: float = 2.0
    polarization: str = "circular"
    spot_match: float = 0.90
    incidence_floor: float = 0.70
    emitter_error_efficiency: float = 0.95
    pointing_error_m: float = 2.0

    _silhouette: BWBSilhouette = field(default=None, repr=False)

    def silhouette(self):
        if self._silhouette is None:
            self._silhouette = get_silhouette()
        return self._silhouette


def _plane_receiver(cfg, emitter_pos):
    """Silhouette area and effective rectenna radius the plane presents to an emitter
    at `emitter_pos`, given the plane's attitude."""
    plane = np.asarray(cfg.plane_pos, dtype=float)
    _, _, unit_plane_to_emitter = geometry(plane, emitter_pos)
    heading, pitch, roll = cfg.plane_attitude
    beta, alpha = beta_alpha_from_enu(unit_plane_to_emitter, heading, pitch, roll)
    area = cfg.silhouette().area(beta, alpha)
    return area, receiver_radius_from_silhouette(area), beta, roll


def evaluate(cfg: LinkConfig, delivered_mw=None):
    """Evaluate a link end to end. Returns a dict with every stage efficiency and,
    if `delivered_mw` is given, the RF and grid power a station must supply.

    Efficiency is defined grid busbar → DC delivered at the rectenna terminals (the
    onboard drivetrain is NOT included here — napkin folds it into cruise_mw)."""
    atmos = get_atmosphere()
    station = np.asarray(cfg.station_pos, dtype=float)
    plane = np.asarray(cfg.plane_pos, dtype=float)
    stages = {}

    if cfg.relay_pos is None:
        # ---- direct station → plane ----
        f = cfg.f_emit_plane_ghz
        area, r_recv, beta, roll = _plane_receiver(cfg, station)
        h = hop(atmos, f, station, plane, r_recv, cfg.weather, cfg.tau_target,
                emitter_diam_m=cfg.emitter_diam_m,
                scan_deg=0.0, error_efficiency=cfg.emitter_error_efficiency)
        rect, rect_parts = rectenna_derate(f, beta, cfg.polarization, roll,
                                           cfg.spot_match, cfg.incidence_floor)
        point = pointing_efficiency(cfg.pointing_error_m, r_recv)
        eta_dc_rf = napkin.eta_grid_to_rf_band(f)

        stages = dict(
            architecture="direct",
            dc_to_rf=eta_dc_rf,
            emitter_aperture=h.emitter_aperture_eff,
            capture=h.capture, atmos=h.atmos_transmittance, pointing=point,
            rectenna_total=rect, rectenna_parts=rect_parts,
            silhouette_m2=area, receiver_radius_m=r_recv, beta_deg=beta,
            emitter_diam_m=h.emitter_diam_m, slant_km=h.slant_m / 1e3,
            elevation_deg=h.elevation_deg, atmos_loss_db=h.atmos_loss_db)
        eta = eta_dc_rf * h.emitter_aperture_eff * h.capture * h.atmos_transmittance * point * rect

    else:
        # ---- relayed station → relay → plane ----
        relay = np.asarray(cfg.relay_pos, dtype=float)
        r_relay = cfg.relay_aperture_diam_m / 2.0
        f1 = cfg.f_emit_relay_ghz
        f2 = f1 if cfg.relay_mode == "passive" else cfg.f_relay_plane_ghz

        h1 = hop(atmos, f1, station, relay, r_relay, cfg.weather, cfg.tau_target,
                 error_efficiency=cfg.emitter_error_efficiency)
        area, r_recv, beta, roll = _plane_receiver(cfg, relay)
        h2 = hop(atmos, f2, relay, plane, r_recv, cfg.weather, cfg.tau_target,
                 error_efficiency=cfg.emitter_error_efficiency)

        convert = relay_convert_efficiency(f1, f2, cfg.relay_mode)
        rect, rect_parts = rectenna_derate(f2, beta, cfg.polarization, roll,
                                           cfg.spot_match, cfg.incidence_floor)
        point = pointing_efficiency(cfg.pointing_error_m, r_recv)
        eta_dc_rf = napkin.eta_grid_to_rf_band(f1)

        stages = dict(
            architecture=f"relay-{cfg.relay_mode}",
            dc_to_rf=eta_dc_rf,
            emitter_aperture=h1.emitter_aperture_eff,
            capture_up=h1.capture, atmos_up=h1.atmos_transmittance,
            relay_convert=convert,
            relay_aperture=h2.emitter_aperture_eff,
            capture_down=h2.capture, atmos_down=h2.atmos_transmittance,
            pointing=point, rectenna_total=rect, rectenna_parts=rect_parts,
            silhouette_m2=area, receiver_radius_m=r_recv, beta_deg=beta,
            emitter_diam_up_m=h1.emitter_diam_m, emitter_diam_down_m=h2.emitter_diam_m,
            slant_up_km=h1.slant_m / 1e3, slant_down_km=h2.slant_m / 1e3,
            elevation_up_deg=h1.elevation_deg, elevation_down_deg=h2.elevation_deg,
            atmos_loss_up_db=h1.atmos_loss_db, atmos_loss_down_db=h2.atmos_loss_db)
        eta = (eta_dc_rf * h1.emitter_aperture_eff * h1.capture * h1.atmos_transmittance
               * convert * h2.emitter_aperture_eff * h2.capture * h2.atmos_transmittance
               * point * rect)

    stages["eta_end_to_end"] = eta
    if delivered_mw is not None:
        stages["delivered_mw"] = delivered_mw
        stages["grid_mw"] = delivered_mw / eta
        stages["waste_mw"] = delivered_mw / eta - delivered_mw
        stages["incident_power_density_kw_m2"] = (
            delivered_mw / stages["rectenna_total"] * 1e3 / np.maximum(stages["silhouette_m2"], 1e-6))
    return stages


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
    ground = plane_alt_m / np.tan(np.radians(reference_elevation_deg))
    plane = np.array([station_pos[0] + ground, station_pos[1], plane_alt_m])
    _, _, unit = geometry(plane, np.asarray(station_pos, dtype=float))
    beta, alpha = beta_alpha_from_enu(unit, 0.0, 0.0, 0.0)
    sil = silhouette or get_silhouette()
    r_recv = receiver_radius_from_silhouette(sil.area(beta, alpha))
    slant = np.linalg.norm(plane - np.asarray(station_pos, dtype=float))
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
    """Station at origin (sea level); plane placed downrange so the emitter sees it
    at `elevation_deg`. Returns (station_pos, plane_pos)."""
    ground_range = plane_alt_km * 1e3 / np.tan(np.radians(elevation_deg))
    return (0.0, 0.0, 0.0), (ground_range, 0.0, plane_alt_km * 1e3)


def tech_cases(elevation_deg=30.0, weather="clear", delivered_mw=10.0):
    """The architectures the project has been weighing, at a common geometry."""
    station, plane = reference_geometry(elevation_deg)
    relay = (station[0] * 0.5 + plane[0] * 0.5, 0.0, 20_000.0)   # relay aloft, mid-span
    common = dict(station_pos=station, plane_pos=plane, weather=weather)
    cases = {
        "5.8 GHz direct":            LinkConfig(**common, f_emit_plane_ghz=5.8),
        "30 GHz direct":             LinkConfig(**common, f_emit_plane_ghz=30.0),
        "34.8 GHz direct":           LinkConfig(**common, f_emit_plane_ghz=34.8),
        "relay up-convert 5.8→30":   LinkConfig(**common, relay_pos=relay, relay_mode="active",
                                                f_emit_relay_ghz=5.8, f_relay_plane_ghz=30.0),
        "relay passive 30 GHz":      LinkConfig(**common, relay_pos=relay, relay_mode="passive",
                                                f_emit_relay_ghz=30.0),
    }
    return {name: evaluate(cfg, delivered_mw) for name, cfg in cases.items()}


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    print("Beam-link engine — reference geometry: plane at 11 km, 30° elevation, 10 MW delivered\n")
    for name, stages in tech_cases(elevation_deg=30.0, weather="clear").items():
        print_report(name, stages)

    print("\n\n########## Weather stress (30 GHz direct, 30° elevation) ##########")
    station, plane = reference_geometry(30.0)
    for wx in ("clear", "cloudy", "light_rain", "heavy_rain", "storm"):
        s = evaluate(LinkConfig(station_pos=station, plane_pos=plane,
                                f_emit_plane_ghz=30.0, weather=wx), delivered_mw=10.0)
        print(f"  {wx:11s}: atmos {float(s['atmos']):.3f}  end-to-end {float(s['eta_end_to_end']):.3f}"
              f"  grid {float(s['grid_mw']):.1f} MW")
