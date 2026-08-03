"""Path attenuation from a precomputed profile, and the spectrum it comes from (P10).

Gaseous specific attenuation on an (altitude, frequency) grid, plus a frequency-only cloud
coefficient and the ITU-R P.838 rain law. The table is committed; `tables/` regenerates it.

Batched over endpoints and over frequency: a whole (candidate, leg, sample) grid of paths is one
call, which is what keeps the atmosphere off the critical path of a vectorized sweep (I9).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from beamlink.artifacts.spec import Weather
from beamlink.physics import geo
from beamlink.physics.beam_link import relations

_PROFILE_PATH = Path(__file__).with_name("data") / "atmosphere_profile.npz"


class AtmosphereProfile:
    """The committed (altitude, frequency) attenuation tables, loaded once per process.

    Read-only and content-identified, so sharing one across parallel engine calls cannot change a
    result (D4): `source` is the file name and a hash of its bytes, and it goes in P10 so a run
    records which table it was solved against.
    """

    def __init__(self, path: str | Path = _PROFILE_PATH):
        path = Path(path)
        data = np.load(path)
        self.frequency_ghz = data["frequency_ghz"]
        self.altitude_m = data["altitude_m"]
        self.gamma_gas_db_km = data["gamma_gas_db_km"]        # [altitude, frequency]
        self.cloud_kl_db_km_per_gm3 = data["cloud_kl_db_km_per_gm3"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        self.source = f"{path.name}:{digest}"

    def _bilinear(self, table, frequency_ghz, altitude_m):
        """Bilinear interpolation of an (n_altitude, n_frequency) table; broadcasts."""
        frequency, altitude = np.broadcast_arrays(np.asarray(frequency_ghz, dtype=float),
                                                  np.asarray(altitude_m, dtype=float))
        altitude = np.clip(altitude, self.altitude_m[0], self.altitude_m[-1])
        fi = np.clip(np.searchsorted(self.frequency_ghz, frequency) - 1, 0,
                     self.frequency_ghz.size - 2)
        hi = np.clip(np.searchsorted(self.altitude_m, altitude) - 1, 0, self.altitude_m.size - 2)
        f_low, f_high = self.frequency_ghz[fi], self.frequency_ghz[fi + 1]
        h_low, h_high = self.altitude_m[hi], self.altitude_m[hi + 1]
        tf = np.where(f_high > f_low, (frequency - f_low) / (f_high - f_low), 0.0)
        th = np.where(h_high > h_low, (altitude - h_low) / (h_high - h_low), 0.0)
        return ((1 - th) * (1 - tf) * table[hi, fi] + (1 - th) * tf * table[hi, fi + 1]
                + th * (1 - tf) * table[hi + 1, fi] + th * tf * table[hi + 1, fi + 1])

    def cloud_coefficient(self, frequency_ghz):
        return np.interp(np.asarray(frequency_ghz, dtype=float), self.frequency_ghz,
                         self.cloud_kl_db_km_per_gm3)


_PROFILE: AtmosphereProfile | None = None


def profile() -> AtmosphereProfile:
    """The shared read-only profile. Loading it twice would only cost time, never a result."""
    global _PROFILE
    if _PROFILE is None:
        _PROFILE = AtmosphereProfile()
    return _PROFILE


def path_loss_db(frequency_ghz, from_ecef, to_ecef, weather: Weather, n_path=64):
    """One-way loss along the straight path, split into its three named parts.

    Returns (gaseous, rain, cloud) in dB, each broadcast over the endpoint batch. Split rather
    than summed because P10 emits the parts: "the path attenuation actually applied" is a
    different statement from "17 dB happened".
    """
    table = profile()
    altitude, step = geo.sample_altitudes(from_ecef, to_ecef, n_path)
    frequency = np.asarray(frequency_ghz, dtype=float)
    frequency_column = frequency[..., None] if frequency.ndim else frequency
    step_km = step / 1e3

    gamma = table._bilinear(table.gamma_gas_db_km, frequency_column, altitude)
    gaseous = np.sum(0.5 * (gamma[..., 1:] + gamma[..., :-1]), axis=-1) * step_km

    rain = np.zeros_like(gaseous)
    if weather.rain_rate_mmh > 0:
        # 45° tilt == circular polarisation, so k and α are the H/V mean.
        k, alpha = relations.rain_coefficients(frequency, elevation_deg=90.0,
                                               polarization_tilt_deg=45.0)
        specific = relations.specific_rain_attenuation(weather.rain_rate_mmh, k, alpha)
        # Count whole steps with BOTH ends in the layer: this quantises the wet path length to ds
        # and biases it low by up to one step, which is the honest direction for a loss term.
        in_layer = (altitude[..., 1:] < weather.rain_top_m) & (altitude[..., :-1] < weather.rain_top_m)
        rain = specific * np.sum(in_layer, axis=-1) * step_km
    cloud = np.zeros_like(gaseous)
    if weather.cloud_lwc_gm3 > 0:
        specific = table.cloud_coefficient(frequency) * weather.cloud_lwc_gm3
        band_low, band_high = weather.cloud_band_m
        in_band = (altitude >= band_low) & (altitude <= band_high)
        in_layer = in_band[..., 1:] & in_band[..., :-1]
        cloud = specific * np.sum(in_layer, axis=-1) * step_km
    return gaseous, rain, cloud


def transmittance(loss_db, blocked):
    """The surviving power fraction, with horizon blockage folded in as a hard zero."""
    return relations.db_to_fraction(loss_db) * np.where(blocked, 0.0, 1.0)


def zenith_spectrum_db(altitude_m, weather: Weather, n_path=64):
    """Zenith loss from the ground to `altitude_m` against frequency, under this weather.

    The spectrum the per-sample numbers came from (P10), emitted so that a figure about frequency
    choice reads a product instead of re-deriving the table — which is how a viewer ends up owning
    a second copy of the physics.
    """
    table = profile()
    ground = geo.point(0.0, 0.0, 0.0)
    top = geo.point(0.0, 0.0, float(altitude_m))
    gaseous, rain, cloud = path_loss_db(table.frequency_ghz, ground, top, weather, n_path)
    return table.frequency_ghz, gaseous + rain + cloud
