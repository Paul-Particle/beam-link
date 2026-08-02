"""End-to-end efficiency of a microwave power link from a ground station (or a relay) to an
aircraft, and the pass/corridor sweeps built on it.

Positions are (downrange_m, crossrange_m, altitude_m) triples, converted to earth-frame
cartesian by geo.point. Every "efficiency" here is a dimensionless factor in a single
multiplicative chain; `evaluate` returns the factors and their product, and `waterfall`
lists them in the order they apply.

Layers: Atmosphere (table lookup) -> per-stage efficiency terms -> spot/silhouette overlap
(via diffraction + bwb_model) -> architectures -> pass and corridor sweeps -> reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from beamlink import relations
from beamlink import geo
from beamlink import diffraction
from beamlink import bwb_model
from beamlink.bwb_model import (BWBSilhouette, beta_alpha_from_enu, direction_from_beta_alpha,
                       get_mesh, rasterize_silhouette_zbuffer)


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


def equivalent_disc_radius(peak_intensity):
    """Single length scale for a spot of any shape: the radius of the uniform disc that would
    carry all the power at the observed peak intensity (pi*r^2*I_peak = 1). Takes the
    unit-power normalised intensity from `diffraction`, so 1/m^2 in, m out."""
    return float(1.0 / np.sqrt(np.pi * peak_intensity)) if peak_intensity > 0 else np.inf


def overlap_capture(raster, emitter_radius_m, wavelength_m, slant_m, shaped=False,
                    incidence_floor=0.90, aspect_cap=8.0, taper_p=0.0,
                    focus_range_m=None, pointing_offset_m=0.0, fixed_aspect=None,
                    n_grid=384, return_field=False):
    """Fraction of radiated power the aircraft silhouette actually intercepts.

    Integrates the diffracted intensity over the receiver elements of a raster from
    bwb_model. Returns (geometric capture, capture including the incidence derate, peak
    intensity in 1/m^2), all per unit radiated power, so the first two are fractions <= 1.

    With `shaped`, the emitter aperture is made elliptical and aligned with the silhouette:
    the aperture is stretched along whichever direction the target is *narrow* in, because
    spot width scales as 1/aperture width, giving a spot elongated the same way as the
    target. `aspect_cap` bounds how far that is pushed.
    """
    u, v, inc, proj = raster["u"], raster["v"], raster["incidence"], raster["proj_area"]
    focus = float(slant_m if focus_range_m is None else focus_range_m)
    if u.size == 0:
        empty = dict(u=u, v=v, beam=np.array([]), incidence=inc, proj_area=proj)
        return (0.0, 0.0, 0.0, empty) if return_field else (0.0, 0.0, 0.0)

    # Rectenna angular response only -- the cos(theta) area foreshortening is already in
    # proj_area. The 0.90 default is a conformal phased-array rectenna, whose elements re-phase
    # toward the arrival angle and so keep most of their off-normal match; a fixed patch array
    # would sit nearer 0.70.
    inc_eff = incidence_floor + (1.0 - incidence_floor) * inc
    # Principal axes of the silhouette (for the shaped-ellipse orientation AND the pointing
    # offset — a pointing error across the thin minor axis is the worst case).
    if u.size >= 3:
        # principal axes of the silhouette, weighted by the area each element presents
        evals, evecs = np.linalg.eigh(np.cov(np.vstack([u, v]), aweights=np.maximum(proj, 1e-12)))
    else:
        evals, evecs = np.array([1.0, 1.0]), np.eye(2)
    # eigh sorts ascending, so column 0 is the least-spread (narrow) direction. Offsetting
    # the target along it is the worst case for a pointing error.
    minor_axis = evecs[:, 0]
    du, dv = pointing_offset_m * minor_axis[0], pointing_offset_m * minor_axis[1]
    pc = evecs.T @ np.vstack([u - du, v - dv])         # rows: [along minor, along major]

    if not shaped or u.size < 3:
        aspect = 1.0
    elif fixed_aspect is not None:                      # FIXED elliptical emitter (built once)
        aspect = max(float(fixed_aspect), 1.0)
    else:                                               # adaptive: aspect = silhouette elongation
        aspect = float(np.clip(np.sqrt(max(evals[1], 1e-12) / max(evals[0], 1e-12)), 1.0, aspect_cap))
    # aperture semi-axes at constant area: major along pc[0], the silhouette's narrow axis
    ap_major, ap_minor = emitter_radius_m * np.sqrt(aspect), emitter_radius_m / np.sqrt(aspect)

    # Sampling grid spans the silhouette with 15% slack. The floors only stop it degenerating
    # when the target subtends almost nothing; each axis takes 5% of the *conjugate* aperture
    # semi-axis, which carries the same aspect scaling as the spot on that axis (the spot on x
    # goes as 1/ap_major, i.e. aspect^-1/2, and so does ap_minor).
    rx = max(float(np.abs(pc[0]).max()), ap_minor * 0.05, 1.0) * 1.15
    ry = max(float(np.abs(pc[1]).max()), ap_major * 0.05, 1.0) * 1.15
    x_ax = np.linspace(-rx, rx, n_grid)
    y_ax = np.linspace(-ry, ry, n_grid)
    intensity = diffraction.focal_intensity_grid(slant_m, focus, ap_major, ap_minor, wavelength_m,
                                                 x_ax, y_ax, taper_p=taper_p)
    beam = diffraction.sample_focal_intensity(x_ax, y_ax, intensity, pc[0], pc[1])

    geom = min(float(np.sum(beam * proj)), 1.0)
    eff = min(float(np.sum(beam * proj * inc_eff)), 1.0)
    # Peak areal intensity (for the W/m² density cap) at the spot centre — evaluate it exactly rather
    # than take the fine-grid max (the centre need not land on a grid node).
    peak = float(diffraction.focal_intensity_at(slant_m, focus, ap_major, ap_minor, wavelength_m,
                                                np.zeros(1), np.zeros(1), taper_p=taper_p)[0])
    if return_field:
        return geom, eff, peak, dict(u=u, v=v, beam=beam, incidence=inc, proj_area=proj,
                                     x_axis=x_ax, y_axis=y_ax, intensity=intensity,
                                     aspect=aspect, taper_p=float(taper_p))
    return geom, eff, peak


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

def receiver_radius_from_silhouette(silhouette_area_m2):
    """Equal-area circular radius, so a silhouette can be fed to the circular-aperture
    coupling formulas in napkin."""
    return np.sqrt(silhouette_area_m2 / np.pi)


def size_emitter_diameter(freq_ghz, slant_m, receiver_radius_m, tau_target=2.0):
    """Emitter diameter for a target near-field coupling τ (τ=2 → ~98% capture)."""
    return 2.0 * relations.required_emitter_radius(freq_ghz * 1e9, slant_m, receiver_radius_m, tau_target)


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
                focus_range_m=float(focus_range_m))


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


def _overlap_taper_grid(raster, emitter_radius_m, wavelength_m, slant_m, shaped,
                        incidence_floor, aspect_cap, taper_ps, pointing_offset_m,
                        fixed_aspect, n_grid=256):
    """`overlap_capture` for a whole set of taper exponents at once, in focus.

    Same geometry setup, but one batched FFT on a shared grid instead of a per-case DFT on a
    grid fitted to the target: cheap enough to scan taper, accurate enough to pick a winner,
    which is then re-evaluated properly.
    """
    u, v, inc, proj = raster["u"], raster["v"], raster["incidence"], raster["proj_area"]
    taper_ps = np.atleast_1d(np.asarray(taper_ps, dtype=float))
    if u.size == 0:
        z = np.zeros(taper_ps.size)
        return z, z, z.copy()
    inc_eff = incidence_floor + (1.0 - incidence_floor) * inc
    if u.size >= 3:
        evals, evecs = np.linalg.eigh(np.cov(np.vstack([u, v]), aweights=np.maximum(proj, 1e-12)))
    else:
        evals, evecs = np.array([1.0, 1.0]), np.eye(2)
    minor_axis = evecs[:, 0]
    du, dv = pointing_offset_m * minor_axis[0], pointing_offset_m * minor_axis[1]
    pc = evecs.T @ np.vstack([u - du, v - dv])
    if not shaped or u.size < 3:
        aspect = 1.0
    elif fixed_aspect is not None:
        aspect = max(float(fixed_aspect), 1.0)
    else:
        aspect = float(np.clip(np.sqrt(max(evals[1], 1e-12) / max(evals[0], 1e-12)), 1.0, aspect_cap))
    ap_major, ap_minor = emitter_radius_m * np.sqrt(aspect), emitter_radius_m / np.sqrt(aspect)
    x_ax, y_ax, stack = diffraction.plane_intensity_stack(
        slant_m, slant_m, ap_major, ap_minor, wavelength_m, taper_ps, n_grid=n_grid)
    beam = diffraction.sample_stack(x_ax, y_ax, stack, pc[0], pc[1])   # (K, n_samples)
    geoms = np.minimum((beam * proj).sum(axis=1), 1.0)
    effs = np.minimum((beam * proj * inc_eff).sum(axis=1), 1.0)
    peaks = stack.reshape(stack.shape[0], -1).max(axis=1)
    return geoms, effs, peaks


def _peak_density(delivered_mw, capture, rect_total, peak_intensity):
    """Peak power density (W/m^2) on the aircraft skin for a given delivered DC power.

    Dividing the delivered DC power by capture and the rectenna terms gives the beam power
    crossing the aircraft, and `peak_intensity` is normalised per unit power, so the
    atmospheric and transmitter losses upstream of this point drop out of the ratio.
    """
    if delivered_mw is None or capture <= 0:
        return 0.0
    p_spot_w = delivered_mw * 1e6 / max(capture * rect_total, 1e-9)
    return p_spot_w * peak_intensity


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
    # OPEN: `lo` is meant to hold a compliant focus range, but slant/8 is assumed compliant and
    # never tested. It is not always: whenever that much defocus still cannot spread the spot
    # enough -- a function of delivered power, aperture, range and frequency together, not of
    # emitter size alone (400 MW through the 40 m relay aperture overhead reports 1.30x the cap)
    # -- an over-limit density is returned unflagged, no caller checks it, and
    # `stages["feasible"]` ignores it. What should happen instead is undecided.
    lo, hi = slant / 8.0, slant
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


# --- Architectures + trajectory-integrated flyby ------------------------------

def standard_architectures(station_pos=(0.0, 0.0, 0.0), weather="clear", pae_scenario="today",
                           relay_alt_m=20_000.0, tether_alt_m=20_000.0,
                           emitter_5p8_m=78.0, relay_aperture_m=40.0, emitter_tether_m=15.0,
                           relay_multiplier_out_ghz=17.4, include_reference=True):
    """The architectures compared throughout, as LinkConfig keyword sets.

    Emitter and relay diameters are fixed inputs here, not sized from the geometry, so the
    same aperture is flown across a whole pass. `include_reference` adds two further cases
    kept for comparison rather than proposed.
    """
    relay = (station_pos[0], station_pos[1], relay_alt_m)
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


def reference_geometry(elevation_deg=30.0, plane_alt_km=11.0):
    """(station_pos, plane_pos) placing the aircraft at a chosen elevation from the station."""
    downrange = float(geo.downrange_for_elevation(elevation_deg, plane_alt_km * 1e3))
    return (0.0, 0.0, 0.0), (downrange, 0.0, plane_alt_km * 1e3)


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
