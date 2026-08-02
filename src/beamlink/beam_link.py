"""End-to-end efficiency of a microwave power link from a ground station (or a relay) to an
aircraft, at one instant.

Positions are (downrange_m, crossrange_m, altitude_m) triples, converted to earth-frame
cartesian by geo.point. Every "efficiency" here is a dimensionless factor in a single
multiplicative chain; `evaluate` returns the factors and their product, and `waterfall`
lists them in the order they apply.

Layers: Atmosphere (table lookup) -> per-stage efficiency terms -> spot/silhouette overlap
(via diffraction + bwb_model) -> the chain -> reporting. The named cases to compare live in
`architectures`, and sweeping a trajectory is `sweeps`; neither is imported here, so the
engine stays free of them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from beamlink import relations
from beamlink import geo
from beamlink.bwb_model import (BWBSilhouette, beta_alpha_from_enu, direction_from_beta_alpha,
                                get_mesh, rasterize_silhouette_zbuffer)
# The capture physics proper lives in `capture`, which knows nothing of LinkConfig or of this
# chain, so it can be reworked without touching either. Re-exported because this module is the
# engine's public face: bl.overlap_capture and bl.size_emitter_diameter are what the econ
# driver, the figures and the tests call.
from beamlink.capture import (equivalent_disc_radius, overlap_capture, _overlap_taper_grid,
                              _peak_density, receiver_radius_from_silhouette,
                              size_emitter_diameter)

_LOG = logging.getLogger(__name__)


# Weather presets. Rain fills everything below rain_top_m at a uniform rate; cloud fills
# the altitude band at a uniform liquid water content. Rain rates are the usual
# light/heavy/storm brackets (mm/h) and the layer tops rise with intensity.
WEATHER = {
    "clear":      dict(rain_rate_mmh=0.0,  rain_top_m=0.0,    cloud_lwc_gm3=0.0, cloud_band_m=(0.0, 0.0)),
    "cloudy":     dict(rain_rate_mmh=0.0,  rain_top_m=0.0,    cloud_lwc_gm3=0.5, cloud_band_m=(1000.0, 3000.0)),
    "light_rain": dict(rain_rate_mmh=5.0,  rain_top_m=3000.0, cloud_lwc_gm3=0.5, cloud_band_m=(1000.0, 3000.0)),
    "heavy_rain": dict(rain_rate_mmh=25.0, rain_top_m=4000.0, cloud_lwc_gm3=1.0, cloud_band_m=(1000.0, 4000.0)),
    "storm":      dict(rain_rate_mmh=50.0, rain_top_m=5000.0, cloud_lwc_gm3=2.0, cloud_band_m=(1000.0, 5000.0)),
}


# --- Atmosphere ---------------------------------------------------------------

class Atmosphere:
    """Path loss from a precomputed profile: gaseous specific attenuation on an
    (altitude, frequency) grid plus a frequency-only cloud coefficient."""

    def __init__(self, path="data/atmosphere_profile.npz"):
        p = Path(path) if Path(path).is_absolute() else Path(__file__).parent / path
        d = np.load(p)
        self.freq = d["frequency_ghz"]          # GHz
        self.alt = d["altitude_m"]              # m
        self.gamma = d["gamma_gas_db_km"]       # dB/km, indexed [altitude, frequency]
        self.kl = d["cloud_kl_db_km_per_gm3"]   # dB/km per g/m^3 of liquid water

    def _bilinear_alt_freq(self, table, freq_ghz, altitude_m):
        """Bilinear interp of a (Nh, Nf) table at (altitude, freq); broadcasts."""
        f, h = np.broadcast_arrays(np.asarray(freq_ghz, dtype=float),
                                   np.asarray(altitude_m, dtype=float))
        h = np.clip(h, self.alt[0], self.alt[-1])  # hold the top row above the table (30 km)
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
        """One-way loss in dB along the straight path, integrated over the altitudes it
        crosses. Vectorised over frequency (add an axis) as well as over endpoints."""
        alt, ds = geo.sample_altitudes(from_ecef, to_ecef, n_path)
        f = np.asarray(freq_ghz, dtype=float)
        f_col = f[..., None] if f.ndim else f  # broadcast frequency against the path samples
        ds_km = ds / 1e3

        gamma = self._bilinear_alt_freq(self.gamma, f_col, alt)          # (...,n) dB/km
        gaseous = np.sum(0.5 * (gamma[..., 1:] + gamma[..., :-1]), axis=-1) * ds_km

        w = WEATHER[weather] if isinstance(weather, str) else weather
        rain = np.zeros_like(gaseous)
        if w["rain_rate_mmh"] > 0:
            # 45 deg tilt = circular polarisation, so k and alpha are the H/V mean
            k, a = relations.rain_coefficients(f, elevation_deg=90.0, polarization_tilt_deg=45.0)
            specific = relations.specific_rain_attenuation(w["rain_rate_mmh"], k, a)
            # count whole steps with both ends in the layer: quantises the wet path length
            # to ds and biases it low by up to one step
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
        return relations.db_to_fraction(self.path_loss_db(freq_ghz, from_ecef, to_ecef, weather))


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
    """(slant m, elevation deg, unit direction in `from`'s local ENU) for a position pair."""
    a, b = _to_ecef(from_pos), _to_ecef(to_pos)
    slant, elevation = geo.link(a, b)
    return slant, elevation, geo.direction_in_local_enu(a, b)


# --- per-stage efficiency terms ---------------------------------------------------------

def emitter_aperture_efficiency(scan_deg=0.0, error_efficiency=0.95, scan_rolloff=1.2):
    """Illumination/phase-error efficiency of the emitter, times a cos^p roll-off for steering
    off array boresight.

    OPEN: every call site passes scan_deg=0, which is only correct for a mechanically-pointed
    aperture -- and the baseline emitter is a phased array, so this is an unjustified omission
    rather than a modelled choice. It flatters the low-elevation ends of a pass, where the scan
    angle is largest. `scan_rolloff = 1.2` has no derivation on record either.
    """
    scan = np.clip(np.cos(np.radians(scan_deg)), 0.0, 1.0) ** scan_rolloff
    return error_efficiency * scan


def polarization_efficiency(polarization="circular", roll_deg=0.0, axial_ratio_loss=0.95):
    """Circular polarisation pays a fixed axial-ratio penalty but is roll-independent;
    anything else is treated as linear and loses cos^2 of the aircraft roll."""
    if polarization == "circular":
        return np.full_like(np.asarray(roll_deg, dtype=float), axial_ratio_loss) if np.ndim(roll_deg) else axial_ratio_loss
    return np.cos(np.radians(roll_deg)) ** 2


# --- source, relay and receiver efficiency ---------------------------------------------

# Transmitter power-added efficiency at (5.8 GHz, 34.8 GHz). "today" is the observed
# roll-off with frequency; "flat" assumes it is engineered away, "medium" splits the two.
PAE_SCENARIOS = {
    "today":  (0.65, 0.40),   # current saturated-CW GaN literature
    "medium": (0.65, 0.53),   # a decade of dedicated high-band development
    "flat":   (0.65, 0.65),   # device parity across bands — isolates geometry+weather
}


def pae_dc_to_rf(frequency_ghz, scenario="today"):
    """Linear in frequency between the two band edges above, extrapolated with a floor."""
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
    """RF-in to RF-out efficiency of the three relay types.

    active      rectify to DC and re-transmit at f_out, so both conversions are paid
    multiplier  harmonic multiplier chain behind a reflectarray, one stage per prime factor
    passive     reflect only; the loss is Ruze, exp(-(4*pi*eps)^2) for an rms surface error
                of eps wavelengths -- so it does not change frequency and f_out is ignored
    """
    if mode == "active":
        return relations.rectenna_efficiency_band(f_in_ghz) * pae_dc_to_rf(f_out_ghz, pae_scenario)
    if mode == "multiplier":
        order = relations.multiplier_factor_prime(f_in_ghz, f_out_ghz)
        chain, _ = relations.cascade_efficiency(order, stage_efficiency)
        return reflectarray_reflectivity * chain
    ruze = np.exp(-((4 * np.pi * surface_error_frac) ** 2))
    return reflectarray_reflectivity * ruze


# --- Aperture sizing ----------------------------------------------------------


# --- architectures ----------------------------------------------------------------------

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
    """One aperture-to-aperture hop, used for ground-to-relay where both ends are circular
    discs. Coupling comes from the analytic Goubau formula, not the raster overlap, so
    taper, focusing and pointing error do not enter. Sizes the emitter for `tau_target`
    unless a diameter is given."""
    a, b = _to_ecef(from_pos), _to_ecef(to_pos)
    slant, elevation = geo.link(a, b)

    if emitter_diam_m is None:
        emitter_diam_m = size_emitter_diameter(freq_ghz, slant, receiver_radius_m, tau_target)
    emitter_radius = np.asarray(emitter_diam_m) / 2.0
    tau = relations.coupling_tau(freq_ghz * 1e9, slant, emitter_radius, receiver_radius_m)
    capture = relations.collection_efficiency(tau)
    blocked = geo.horizon_blocked(a, b, n_path)
    loss_db = atmos.path_loss_db(freq_ghz, a, b, weather, n_path)
    transmittance = relations.db_to_fraction(loss_db) * np.where(blocked, 0.0, 1.0)
    ap_eff = emitter_aperture_efficiency(scan_deg, error_efficiency)
    return HopResult(freq_ghz=freq_ghz, slant_m=slant, elevation_deg=elevation,
                     emitter_diam_m=np.asarray(emitter_diam_m), tau=tau, capture=capture,
                     atmos_transmittance=transmittance, atmos_loss_db=loss_db,
                     emitter_aperture_eff=ap_eff, blocked=blocked)


# --- Full link ----------------------------------------------------------------

@dataclass
class LinkConfig:
    """One link case. `relay_pos = None` selects the direct architecture, in which case only
    f_emit_plane_ghz and emitter_diam_m matter; with a relay the plane-side emitter is the
    relay aperture and emitter_diam_m is unused."""

    station_pos: tuple = (0.0, 0.0, 0.0)          # (downrange, crossrange, altitude) m
    plane_pos: tuple = (150_000.0, 0.0, 11_000.0)  # may be an (n, 3) array; see `evaluate`
    plane_attitude: tuple = (0.0, 0.0, 0.0)        # heading, pitch, roll in degrees
    relay_pos: tuple | None = None

    f_emit_plane_ghz: float = 30.0
    f_emit_relay_ghz: float = 5.8
    f_relay_plane_ghz: float = 30.0

    relay_mode: str = "active"
    relay_aperture_diam_m: float = 60.0
    relay_stage_efficiency: float = 0.70
    weather: str = "clear"
    pae_scenario: str = "today"                    # "today" | "medium" | "flat"

    emitter_diam_m: float | None = None  # None sizes it from tau_target at this geometry
    tau_target: float = 2.0              # 2 == 98% Goubau collection
    polarization: str = "circular"
    shaped_spot: bool = True                        # phased-array elliptical spot (vs round aperture)
    # If set, the emitter is a FIXED elliptical aperture of THIS aspect ratio (major:minor),
    # same area as the round one, aligned to the silhouette axes — a single built array, NOT
    # reshaped per geometry. None → the adaptive per-geometry aspect (silhouette elongation).
    fixed_emitter_aspect: float | None = None
    spot_match: float = 0.90       # dead: overlap_capture counts spill directly, so this would
                                   # double-count. Nothing reads it.
    incidence_floor: float = 0.90  # conformal phased-array rectenna; see overlap_capture
    taper_p: float = 1.0
    adaptive_taper: bool = True          # if set, taper_p is chosen per geometry, not used
    taper_p_max: float = 4.0
    emitter_error_efficiency: float = 0.95
    pointing_error_m: float = 2.0
    max_power_density_w_m2: float | None = 2.5e5  # 25 W/cm^2 at the aircraft skin

    _silhouette: BWBSilhouette = field(default=None, repr=False)

    def silhouette(self):
        if self._silhouette is None:
            self._silhouette = get_silhouette()
        return self._silhouette


# Silhouette rasters depend only on the look direction (β, α) — not on frequency,
# range, or weather — so cache them: a weather/architecture sweep reuses one raster
# per look angle instead of re-projecting 10⁵ points every call.
_RASTER_CACHE = {}


def _cached_raster(beta_deg, alpha_deg, n_grid=200):
    """Rasterising a view costs far more than the rest of a case, so views are cached --
    which quantises every result to 0.1 degrees of arrival angle."""
    key = (round(float(beta_deg), 1), round(float(alpha_deg), 1), n_grid)
    raster = _RASTER_CACHE.get(key)
    if raster is None:
        look = direction_from_beta_alpha(key[0], key[1])
        verts, faces, normals = get_mesh()
        raster = rasterize_silhouette_zbuffer(look, verts, faces, normals, n_grid=n_grid)
        _RASTER_CACHE[key] = raster
    return raster


def _capture_setup(cfg, emitter_pos, freq_ghz, emitter_diam_m, raster_n_grid=200):
    """Everything the overlap needs for one (emitter, aircraft) pair: the silhouette raster
    as seen from the emitter, the equal-area receiver radius, and the emitter diameter.

    The geometry is taken from the *aircraft* outwards, so `shared["elevation"]` -- which
    becomes `stages["elevation_deg"]` -- is the angle to the emitter as seen from the aircraft:
    negative for a ground emitter, -90 deg overhead, positive for a relay above.

    UNRESOLVED: the relay branch also reports `elevation_up_deg`, measured at the station, so two
    keys in the same dict carry opposite sign conventions, and `pass_profile` passes this one
    through as a bare `elevation_deg`. Whether that is the intended convention or a sign error is
    a decision nobody has taken.
    """
    plane = np.asarray(cfg.plane_pos, dtype=float)
    emitter = np.asarray(emitter_pos, dtype=float)
    slant, elevation, unit_plane_to_emitter = geometry(plane, emitter)
    heading, pitch, roll = cfg.plane_attitude
    beta, alpha = beta_alpha_from_enu(unit_plane_to_emitter, heading, pitch, roll)
    raster = _cached_raster(beta, alpha, raster_n_grid)
    r_recv = receiver_radius_from_silhouette(max(raster["area"], 1e-6))
    if emitter_diam_m is None:                     # size to τ_target on the equal-area disk
        emitter_diam_m = float(size_emitter_diameter(freq_ghz, slant, r_recv, cfg.tau_target))
    wavelength = relations.SPEED_OF_LIGHT / (freq_ghz * 1e9)
    shared = dict(area=raster["area"], r_recv=r_recv, beta=float(beta), roll=float(roll),
                  slant=float(slant), elevation=float(elevation), emitter_diam_m=float(emitter_diam_m))
    return raster, float(slant), wavelength, float(emitter_diam_m), shared


def _capture_dict(shared, geom, eff, peak, taper_p, focus_range_m):
    return dict(**shared, capture=eff, capture_geom=geom, peak_intensity=peak,
                equivalent_disc_radius_m=equivalent_disc_radius(peak), taper_p=float(taper_p),
                focus_range_m=float(focus_range_m), density_cap_exceeded=False)


def plane_capture(cfg, emitter_pos, freq_ghz, emitter_diam_m=None, taper_p=None,
                  focus_range_m=None, raster_n_grid=200, return_field=False):
    """The real receive-side geometry the plane presents to an emitter at `emitter_pos`:
    rasterize the (self-occluded) silhouette and overlap the emitter's scalar-diffraction
    field with it. Returns a dict with the silhouette area, incidence-weighted capture
    (`capture`), bare geometric overlap, peak areal intensity (for the density cap), a
    report-only effective waist, the aperture taper, the emitter diameter used, and the look
    geometry. Scalar geometry only. `focus_range_m` shorter than the slant defocuses (spreads
    the spot) for the density cap; None focuses on the plane."""
    raster, slant, wavelength, emitter_diam_m, shared = _capture_setup(
        cfg, emitter_pos, freq_ghz, emitter_diam_m, raster_n_grid)
    p = cfg.taper_p if taper_p is None else float(taper_p)
    result = overlap_capture(
        raster, emitter_diam_m / 2.0, wavelength, slant, shaped=cfg.shaped_spot,
        incidence_floor=cfg.incidence_floor, taper_p=p, focus_range_m=focus_range_m,
        pointing_offset_m=cfg.pointing_error_m, fixed_aspect=cfg.fixed_emitter_aspect,
        return_field=return_field)
    geom, eff, peak = result[:3]
    focus_used = slant if focus_range_m is None else float(focus_range_m)
    out = _capture_dict(shared, geom, eff, peak, p, focus_used)
    if return_field:
        # overlap_capture computed the field regardless; retaining it lets the animation draw
        # panel A from the winner without a second FFT.
        out["field"] = result[3]
    return out


def _defocus_floor(cfg, emitter_pos, freq_ghz, emitter_diam_m, taper_p, slant, return_field):
    """Capture at the shortest focus range the defocus search may use — the most-spread spot.

    `slant / 8` is the intended floor, but focusing a very large aperture that short leaves the
    regime the field is sampled in: the aperture pitch goes as the residual focusing phase, so
    `diffraction._aperture_grid` refuses rather than clip the aperture. Walk the floor back out
    until it can be evaluated; a longer focus always needs a coarser pitch, so everything above
    the floor is then evaluable too. Returns (focus_range_m, capture dict).
    """
    focus = slant / 8.0
    while focus < slant:
        try:
            return focus, plane_capture(cfg, emitter_pos, freq_ghz, emitter_diam_m,
                                        taper_p=taper_p, focus_range_m=focus,
                                        return_field=return_field)
        except ValueError:
            _LOG.debug("defocus floor %.3g m unevaluable for a %.3g m emitter at %.3g m slant; "
                       "backing off", focus, emitter_diam_m or 0.0, slant)
            focus *= 1.5
    return slant, plane_capture(cfg, emitter_pos, freq_ghz, emitter_diam_m, taper_p=taper_p,
                                focus_range_m=slant, return_field=return_field)


def _capture_for_taper(cfg, emitter_pos, freq_ghz, emitter_diam_m, delivered_mw, rect_total,
                       taper_p, pc=None, return_field=False):
    """Capture at a fixed taper, deliberately defocused if the skin power density would
    otherwise exceed cfg.max_power_density_w_m2.

    Focusing short of the target spreads the spot, so the bisection looks for the longest
    (least defocused, highest capture) focus range that still complies.
    """
    if pc is None:
        pc = plane_capture(cfg, emitter_pos, freq_ghz, emitter_diam_m, taper_p=taper_p,
                           return_field=return_field)
    cap = cfg.max_power_density_w_m2
    if cap is None or delivered_mw is None or pc["capture"] <= 0:
        return pc
    if _peak_density(delivered_mw, pc["capture"], rect_total, pc["peak_intensity"]) <= cap:
        return pc
    slant = pc["slant"]
    # The bisection needs `lo` to be a COMPLIANT focus range and `hi` a too-hot one, so test `lo`
    # instead of assuming it: whether the most defocused geometry we allow can meet the cap depends
    # on delivered power, aperture, range and frequency together, not on emitter size alone (400 MW
    # through a 40 m relay aperture overhead sits ~1.30x the cap at every focus). When even `lo` is
    # over there is no compliant geometry in range, so log it and mark the result rather than
    # returning a quietly non-compliant one; by decision this SURFACES but does not block, and
    # `feasible` is deliberately left alone.
    lo, pc_lo = _defocus_floor(cfg, emitter_pos, freq_ghz, emitter_diam_m, taper_p, slant,
                               return_field)
    hi = slant
    density_lo = _peak_density(delivered_mw, pc_lo["capture"], rect_total, pc_lo["peak_intensity"])
    if density_lo > cap:
        _LOG.warning(
            "power-density cap unmet: %.2fx cap (%.3g vs %.3g W/m2) at maximum defocus "
            "(focus %.3g m of %.3g m slant, %.3g MW delivered, taper %.2f, emitter %.3g m) - "
            "returning the most-spread geometry with density_cap_exceeded set",
            density_lo / cap, density_lo, cap, lo, slant, delivered_mw, taper_p,
            pc_lo["emitter_diam_m"])
        pc_lo["density_cap_exceeded"] = True
        return pc_lo
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        pcm = plane_capture(cfg, emitter_pos, freq_ghz, emitter_diam_m, taper_p=taper_p, focus_range_m=mid)
        if _peak_density(delivered_mw, pcm["capture"], rect_total, pcm["peak_intensity"]) > cap:
            hi = mid                                   # still too hot → defocus more (shorter focus)
        else:
            lo = mid
    return plane_capture(cfg, emitter_pos, freq_ghz, emitter_diam_m, taper_p=taper_p,
                         focus_range_m=lo, return_field=return_field)


def _capture_with_caps(cfg, emitter_pos, freq_ghz, emitter_diam_m, delivered_mw, rect_total,
                       n_taper=9, return_field=False):
    """Pick the aperture taper, then satisfy the power-density cap.

    The taper that maximises capture is found on a coarse grid. If that choice breaches the
    cap it has to be defocused, which costs capture, so heavier tapers -- which lower the
    peak on their own -- are tried too and the best surviving capture wins. Stops once a
    taper needs no defocus at all, since heavier ones can then only lose capture.
    """
    if not cfg.adaptive_taper:
        return _capture_for_taper(cfg, emitter_pos, freq_ghz, emitter_diam_m,
                                  delivered_mw, rect_total, cfg.taper_p, return_field=return_field)

    raster, slant, wavelength, emitter_diam_m, _ = _capture_setup(
        cfg, emitter_pos, freq_ghz, emitter_diam_m)
    taper_grid = np.linspace(0.0, cfg.taper_p_max, n_taper)
    _, effs, _ = _overlap_taper_grid(                  # coarse batched search for the best taper
        raster, emitter_diam_m / 2.0, wavelength, slant, cfg.shaped_spot,
        cfg.incidence_floor, 8.0, taper_grid, cfg.pointing_error_m, cfg.fixed_emitter_aspect)
    best_k = int(np.argmax(effs))
    # Recompute the winner at full resolution (accurate capture + peak for the cap decision).
    result = plane_capture(cfg, emitter_pos, freq_ghz, emitter_diam_m,
                           taper_p=float(taper_grid[best_k]), return_field=return_field)
    cap = cfg.max_power_density_w_m2
    if cap is None or delivered_mw is None or \
            _peak_density(delivered_mw, result["capture"], rect_total, result["peak_intensity"]) <= cap:
        return result
    # Cap binds → defocus the winner (reusing the focused solve above), then re-check the
    # heavier tapers (they need less defocus). Heavier tapers have lower FOCUSED capture, so
    # once one needs NO defocus every heavier one only loses capture — stop there.
    focused_winner = result
    result = _capture_for_taper(cfg, emitter_pos, freq_ghz, emitter_diam_m, delivered_mw,
                                rect_total, float(taper_grid[best_k]), pc=focused_winner,
                                return_field=return_field)
    slant = result["slant"]
    for p in taper_grid[best_k + 1:]:
        r = _capture_for_taper(cfg, emitter_pos, freq_ghz, emitter_diam_m, delivered_mw,
                               rect_total, float(p), return_field=return_field)
        if r["capture"] > result["capture"]:
            result = r
        if r["focus_range_m"] >= slant - 1e-6:     # no defocus needed → heavier tapers only worse
            break
    return result


def _evaluate_scalar(cfg: LinkConfig, delivered_mw=None, return_field=False):
    """The efficiency chain for a single aircraft position; see `evaluate`.

    The relay branch reports the down-leg terms under both their own names and the direct
    branch's generic ones (`capture`, `atmos`, ...) so that sweep code can stay
    architecture-agnostic.
    """
    atmos = get_atmosphere()
    station = np.asarray(cfg.station_pos, dtype=float)
    plane = np.asarray(cfg.plane_pos, dtype=float)

    if cfg.relay_pos is None:
        # ---- direct emitter → plane (emitter at station_pos; lofted for the tether case) ----
        f = cfg.f_emit_plane_ghz
        conv = float(relations.rectenna_efficiency_band(f))
        pol = float(polarization_efficiency(cfg.polarization, cfg.plane_attitude[2]))  # roll
        # Choose the spot profile per geometry and enforce the W/m² cap.
        pc = _capture_with_caps(cfg, station, f, cfg.emitter_diam_m, delivered_mw, conv * pol,
                                return_field=return_field)
        a, b = _to_ecef(station), _to_ecef(plane)
        blocked = bool(geo.horizon_blocked(a, b, 64))
        atmos_t = float(relations.db_to_fraction(atmos.path_loss_db(f, a, b, cfg.weather))) * (0.0 if blocked else 1.0)
        ap_eff = float(emitter_aperture_efficiency(0.0, cfg.emitter_error_efficiency))
        eta_dc_rf = float(eta_grid_to_rf(f, cfg.pae_scenario))

        eta = eta_dc_rf * ap_eff * pc["capture"] * atmos_t * conv * pol
        stages = dict(
            architecture="direct", dc_to_rf=eta_dc_rf, emitter_aperture=ap_eff,
            capture=pc["capture"], capture_geom=pc["capture_geom"], atmos=atmos_t,
            rectenna_rf_to_dc=conv, polarization=pol,
            rectenna_total=conv * pol, silhouette_m2=pc["area"],
            receiver_radius_m=pc["r_recv"], beta_deg=pc["beta"], equivalent_disc_radius_m=pc["equivalent_disc_radius_m"],
            aperture_taper_p=pc["taper_p"], focus_range_m=pc["focus_range_m"],
            power_density_w_m2=_peak_density(delivered_mw, pc["capture"], conv * pol,
                                            pc["peak_intensity"]),
            density_cap_exceeded=pc["density_cap_exceeded"],
            emitter_diam_m=pc["emitter_diam_m"], slant_km=pc["slant"] / 1e3,
            elevation_deg=pc["elevation"], blocked=blocked)

    else:
        # ---- relayed station → relay → plane ----
        relay = np.asarray(cfg.relay_pos, dtype=float)
        r_relay = cfg.relay_aperture_diam_m / 2.0
        f1 = cfg.f_emit_relay_ghz
        f2 = f1 if cfg.relay_mode == "passive" else cfg.f_relay_plane_ghz  # a mirror cannot retune

        # up-leg: emitter → relay is a plain circular aperture (Goubau τ-capture).
        h1 = hop(atmos, f1, station, relay, r_relay, cfg.weather, cfg.tau_target,
                 error_efficiency=cfg.emitter_error_efficiency)
        # down-leg: relay → plane uses the real silhouette overlap; the emitting
        # aperture is the relay's own (fixed) transmit array.
        conv = float(relations.rectenna_efficiency_band(f2))
        pol = float(polarization_efficiency(cfg.polarization, cfg.plane_attitude[2]))  # roll
        pc = _capture_with_caps(cfg, relay, f2, cfg.relay_aperture_diam_m, delivered_mw, conv * pol,
                                return_field=return_field)
        a2, b2 = _to_ecef(relay), _to_ecef(plane)
        blocked2 = bool(geo.horizon_blocked(a2, b2, 64))
        atmos_dn = float(relations.db_to_fraction(atmos.path_loss_db(f2, a2, b2, cfg.weather))) * (0.0 if blocked2 else 1.0)

        convert = float(relay_convert_efficiency(f1, f2, cfg.relay_mode,
                        stage_efficiency=cfg.relay_stage_efficiency, pae_scenario=cfg.pae_scenario))
        eta_dc_rf = float(eta_grid_to_rf(f1, cfg.pae_scenario))

        eta = (eta_dc_rf * float(h1.emitter_aperture_eff) * float(h1.capture) * float(h1.atmos_transmittance)
               * convert * float(emitter_aperture_efficiency(0.0, cfg.emitter_error_efficiency))
               * pc["capture"] * atmos_dn * conv * pol)
        stages = dict(
            architecture=f"relay-{cfg.relay_mode}", dc_to_rf=eta_dc_rf,
            emitter_aperture=float(h1.emitter_aperture_eff),
            capture_up=float(h1.capture), atmos_up=float(h1.atmos_transmittance),
            relay_convert=convert, relay_aperture=float(emitter_aperture_efficiency(0.0, cfg.emitter_error_efficiency)),
            capture_down=pc["capture"], capture_down_geom=pc["capture_geom"], atmos_down=atmos_dn,
            rectenna_rf_to_dc=conv, polarization=pol, rectenna_total=conv * pol,
            capture=pc["capture"], atmos=atmos_dn,
            silhouette_m2=pc["area"], receiver_radius_m=pc["r_recv"], beta_deg=pc["beta"],
            equivalent_disc_radius_m=pc["equivalent_disc_radius_m"], aperture_taper_p=pc["taper_p"],
            focus_range_m=pc["focus_range_m"], emitter_diam_m=pc["emitter_diam_m"],
            power_density_w_m2=_peak_density(delivered_mw, pc["capture"], conv * pol,
                                            pc["peak_intensity"]),
            density_cap_exceeded=pc["density_cap_exceeded"],
            emitter_diam_up_m=float(h1.emitter_diam_m),
            slant_up_km=h1.slant_m / 1e3, slant_down_km=pc["slant"] / 1e3,
            elevation_up_deg=float(h1.elevation_deg), elevation_down_deg=pc["elevation"],
            elevation_deg=pc["elevation"], blocked=bool(np.logical_or(h1.blocked, blocked2)))

    stages["eta_end_to_end"] = eta
    stages["feasible"] = (not stages["blocked"]) and eta > 1e-4
    if delivered_mw is not None:
        # `delivered_mw` is onboard DC; grid_mw is what the station must draw for it
        stages["delivered_mw"] = delivered_mw
        stages["grid_mw"] = delivered_mw / eta if eta > 0 else np.inf
        stages["waste_mw"] = stages["grid_mw"] - delivered_mw
    if return_field:
        return stages, pc.get("field")
    return stages


def evaluate(cfg: LinkConfig, delivered_mw=None):
    """Efficiency chain for the configured link, as a dict of named stage factors plus
    `eta_end_to_end`. An (n, 3) `plane_pos` is looped over and the results stacked, so every
    numeric entry becomes an array of length n.
    """
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


def evaluate_with_field(cfg: LinkConfig, delivered_mw=None):
    """Scalar `evaluate` that ALSO returns the winning capture field dict (the exact
    scalar-diffraction field behind the capture number — winning taper, focus and pointing), so
    the animation can draw panel A from it without recomputing overlap_capture. Returns
    (stages, field). `cfg.plane_pos` must be a single position."""
    return _evaluate_scalar(cfg, delivered_mw, return_field=True)


# --- reporting --------------------------------------------------------------------------

def waterfall(stages):
    """The stage factors whose product is eta_end_to_end, in the order they apply. Any change
    to the chain in `_evaluate_scalar` has to be mirrored here."""
    if stages["architecture"] == "direct":
        keys = [("DC→RF (PAE)", "dc_to_rf"), ("emitter aperture", "emitter_aperture"),
                ("overlap capture", "capture"), ("atmosphere", "atmos"),
                ("rectenna RF→DC+incidence", "rectenna_total")]
    else:
        keys = [("DC→RF (PAE)", "dc_to_rf"), ("emitter aperture", "emitter_aperture"),
                ("capture up", "capture_up"), ("atmosphere up", "atmos_up"),
                ("relay convert", "relay_convert"), ("relay aperture", "relay_aperture"),
                ("overlap capture", "capture_down"), ("atmosphere down", "atmos_down"),
                ("rectenna RF→DC+incidence", "rectenna_total")]
    return [(label, float(stages[k])) for label, k in keys]
