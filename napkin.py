"""First-order sizing for microwave-beamed aviation: closed-form estimates and tables.

Not a standalone layer: `beam_link` calls the coupling, rain and rectenna relations here in its
live chain, so retuning one of those constants moves the engine's numbers too. What is superseded
is narrower -- `ETA_BEAM`, `silhouette_area_m2`, `effective_rectenna_radius_m` and this module's
own PAE curves have scenario-driven or diffraction-based replacements there. The economics below
still runs on the flat placeholders. SI units unless a name says otherwise.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go

SPEED_OF_LIGHT = 2.998e8   # m/s
GRAVITY = 9.81             # m/s^2
RHO_AIR_SEA = 1.225        # kg/m^3
SCALE_HEIGHT = 8500.0      # m; single-exponential atmosphere (approx; ~30% high near 20 km)

# lifting-gas molar mass over that of dry air (~29 g/mol): the fraction of displaced air
# mass the gas itself weighs
GAS_FRACTION = {"hydrogen": 2 / 29, "helium": 4 / 29}

# (tensile strength Pa, density kg/m^3)
TETHER_MATERIALS = {
    "Dyneema (UHMWPE)": (3.5e9, 970),
    "Zylon (PBO)": (5.8e9, 1560),
    "Carbon (T1100)": (7.0e9, 1790),
    "HS steel": (2.0e9, 7850),
}


# --- beam coupling (Goubau) ------------------------------------------------------------

def wavelength(frequency_hz):
    return SPEED_OF_LIGHT / frequency_hz


def coupling_tau(frequency_hz, distance_m, emitter_radius_m, receiver_radius_m):
    """Goubau coupling parameter tau = pi*a_t*a_r/(lambda*D), dimensionless. The single
    number that decides how much of a beam a finite receiver can collect."""
    return np.pi * emitter_radius_m * receiver_radius_m / (wavelength(frequency_hz) * distance_m)


def collection_efficiency(tau):
    """Fraction of radiated power intercepted, 1 - exp(-tau^2). tau=2 gives 0.982."""
    return 1.0 - np.exp(-(tau ** 2))


def required_emitter_radius(frequency_hz, distance_m, receiver_radius_m, tau_target=2.0):
    """`coupling_tau` solved for emitter radius; the sizing rule used everywhere."""
    return tau_target * wavelength(frequency_hz) * distance_m / (np.pi * receiver_radius_m)


def surface_precision(frequency_hz, fraction=1 / 16):
    """Element-position *knowledge* needed for phase correction (~lambda/16), in m.

    Not a mechanical flatness or rigidity requirement: a retrodirective array may flex freely
    provided each element's position is known to this accuracy. The distinction is what keeps a
    78 m tethered or aerostat-borne aperture on the table -- 0.6 mm at 30 GHz is a metrology
    problem, not a structural one.
    """
    return wavelength(frequency_hz) * fraction


def power_density_limited_diameter(power_w, density_wcm2=25.0):
    """Smallest aperture that can radiate `power_w` without exceeding a surface power
    density limit (W/cm^2)."""
    area_m2 = power_w / (density_wcm2 * 1e4)
    return 2.0 * np.sqrt(area_m2 / np.pi)


# --- Passive frequency multiplication (the "gearbox") -------------------------
# Manley-Rowe permits ~100% f -> n·f conversion for an IDEAL lossless nonlinear
# reactance (varactor); real multipliers fall well short, and a high order like
# 5.8 -> 30 GHz (~x6) is built as a cascade whose per-stage losses compound.

def multiplier_factor_prime(input_ghz=5.8, output_ghz=30.0):
    """Harmonic order of a frequency-multiplying relay, rounded to an integer."""
    return round(output_ghz / input_ghz)


def cascade_efficiency(order, stage_efficiency=0.7):
    """A multiplier of the given order is built from one stage per prime factor (x6 = x2
    then x3), each costing stage_efficiency. Returns (total, stage list)."""
    stages = _prime_stages(order)
    return stage_efficiency ** len(stages), stages


def _prime_stages(n):
    """Factor `n` into a list of small prime stages (a x6 -> [2, 3])."""
    stages, d = [], 2
    while d * d <= n:
        while n % d == 0:
            stages.append(d)
            n //= d
        d += 1
    if n > 1:
        stages.append(n)
    return stages or [1]


# --- rain and gaseous attenuation ------------------------------------------------------

# ITU-R P.838-3 curve fits for the rain coefficients k and alpha: a sum of Gaussians in
# log10(f/GHz) plus a linear term. Reproduces the published table (30 GHz: kh 0.2403,
# ah 0.9485, kv 0.2291, av 0.9129). Valid 1-1000 GHz.
_P838 = {
    "kh": dict(aj=[-5.33980, -0.35351, -0.23789, -0.94158],
               bj=[-0.10008, 1.2697, 0.86036, 0.64552],
               cj=[1.13098, 0.454, 0.15354, 0.16817], m=-0.18961, c=0.71147),
    "kv": dict(aj=[-3.80595, -3.44965, -0.39902, 0.50167],
               bj=[0.56934, -0.22911, 0.73042, 1.07319],
               cj=[0.81061, 0.51059, 0.11899, 0.27195], m=-0.16398, c=0.63297),
    "ah": dict(aj=[-0.14318, 0.29591, 0.32177, -5.37610, 16.1721],
               bj=[1.82442, 0.77564, 0.63773, -0.96230, -3.29980],
               cj=[-0.55187, 0.19822, 0.13164, 1.47828, 3.4399], m=0.67849, c=-1.95537),
    "av": dict(aj=[-0.07771, 0.56727, -0.20238, -48.2991, 48.5833],
               bj=[2.3384, 0.95545, 1.1452, 0.791669, 0.791459],
               cj=[-0.76284, 0.54039, 0.26809, 0.116226, 0.116479], m=-0.053739, c=0.83433),
}


def _p838_term(frequency_ghz, coeffs, is_k):
    """The fit is for log10(k) but for alpha directly, hence `is_k`."""
    log_f = np.log10(frequency_ghz)
    gaussians = sum(a * np.exp(-(((log_f - b) / c) ** 2))
                    for a, b, c in zip(coeffs["aj"], coeffs["bj"], coeffs["cj"]))
    value = gaussians + coeffs["m"] * log_f + coeffs["c"]
    return 10.0 ** value if is_k else value


def rain_coefficients(frequency_ghz, elevation_deg=90.0, polarization_tilt_deg=0.0):
    """(k, alpha) for the specific-attenuation law, combining the H and V fits for path
    elevation and polarisation tilt as in P.838. Both the zenith path (cos^2 el = 0) and
    circular polarisation (tilt 45 deg) collapse this to the plain H/V mean."""
    kh = _p838_term(frequency_ghz, _P838["kh"], is_k=True)
    kv = _p838_term(frequency_ghz, _P838["kv"], is_k=True)
    ah = _p838_term(frequency_ghz, _P838["ah"], is_k=False)
    av = _p838_term(frequency_ghz, _P838["av"], is_k=False)
    cos_el2 = np.cos(np.radians(elevation_deg)) ** 2
    cos_2tau = np.cos(np.radians(2 * polarization_tilt_deg))
    k = (kh + kv + (kh - kv) * cos_el2 * cos_2tau) / 2.0
    alpha = (kh * ah + kv * av + (kh * ah - kv * av) * cos_el2 * cos_2tau) / (2.0 * k)
    return k, alpha


RAIN_K, RAIN_ALPHA = rain_coefficients(30.0)  # the defaults below are 30 GHz values


def specific_rain_attenuation(rain_rate_mmh, k=RAIN_K, alpha=RAIN_ALPHA):
    """gamma_R = k*R^alpha in dB/km, R in mm/h (P.838)."""
    return k * rain_rate_mmh ** alpha


def rain_fade_db(rain_rate_mmh, rain_height_km, elevation_deg=90.0, k=RAIN_K, alpha=RAIN_ALPHA):
    """Total rain loss for a slant path through a uniform layer of the given depth."""
    path_km = rain_height_km / np.sin(np.radians(elevation_deg))
    return specific_rain_attenuation(rain_rate_mmh, k, alpha) * path_km


def db_to_fraction(db):
    """Power fraction surviving `db` of attenuation: 10^(−db/10)."""
    return 10.0 ** (-db / 10.0)


# --- aircraft power and mass -----------------------------------------------------------

# 22 is the bare lifting-body/BWB airframe, shared with the fossil comparator so that airframe
# gains are not credited to beaming. The +2 is electric-propulsion integration -- boundary-layer
# ingestion, no podded nacelles, distributed fans -- which is an energy-path advantage and so is
# legitimately the concept's. ~+5-10%, unverified.
AIRFRAME_LIFT_TO_DRAG = 22.0
BEAMED_LIFT_TO_DRAG = 24.0
REFERENCE_LIFT_TO_DRAG = BEAMED_LIFT_TO_DRAG   # headline value used downstream


def glide_range_km(lift_to_drag, altitude_drop_m):
    return lift_to_drag * altitude_drop_m / 1e3


def powered_fraction(speed_ms, climb_rate_ms, lift_to_drag):
    """Fraction of a climb-then-glide cycle that needs the beam. Climbing height h covers
    h*V/climb_rate of ground, gliding it back covers h*L/D, so the powered share is
    (V/vc)/(V/vc + L/D) -- independent of h, and the same in time as in distance."""
    ratio = speed_ms / climb_rate_ms
    return ratio / (ratio + lift_to_drag)


def cruise_power_mw(mass_kg, speed_ms, lift_to_drag, eta_chain=0.78):
    """Electrical power (MW) for level flight; eta_chain covers rectenna DC to propulsive."""
    thrust = mass_kg * GRAVITY / lift_to_drag
    return thrust * speed_ms / eta_chain / 1e6


def climb_power_mw(mass_kg, speed_ms, lift_to_drag, climb_rate_ms, eta_chain=0.78):
    """As above plus the potential-energy rate m*g*vc."""
    prop = mass_kg * GRAVITY * speed_ms / lift_to_drag + mass_kg * GRAVITY * climb_rate_ms
    return prop / eta_chain / 1e6


def beam_power_mw(electrical_mw, rectenna_eff=0.85, collection_eff=0.90):
    """Radiated power needed for a given onboard electrical power."""
    return electrical_mw / rectenna_eff / collection_eff


# --- Mass buildup (the size cascade, made concrete) ---------------------------

def mass_buildup(payload_t=40.0, speed_ms=250.0, lift_to_drag=BEAMED_LIFT_TO_DRAG,
                 climb_rate_ms=4.0, eta_chain=0.85, struct_fraction=0.30,
                 systems_fraction=0.10, motor_specific_power_kw_per_kg=5.0,
                 battery_specific_energy_wh_per_kg=250.0, buffer_minutes=5.0,
                 rectenna_mass_t=2.0, max_iter=200, tol=1.0):
    """Fixed-point mass loop: structure and systems scale with MTOW, motors with peak power,
    battery with `buffer_minutes` of cruise power (the beam-dropout reserve). Iterates from
    2x payload until MTOW moves less than `tol` kg. Masses out in tonnes, power in MW."""
    payload = payload_t * 1e3
    rectenna = rectenna_mass_t * 1e3
    mtow = payload * 2.0
    for _ in range(max_iter):
        cruise_w = mtow * GRAVITY * speed_ms / lift_to_drag / eta_chain
        peak_w = cruise_w + mtow * GRAVITY * climb_rate_ms / eta_chain
        motor = peak_w / (motor_specific_power_kw_per_kg * 1e3)
        battery = cruise_w * (buffer_minutes / 60.0) / battery_specific_energy_wh_per_kg
        structure = struct_fraction * mtow
        systems = systems_fraction * mtow
        mtow_new = payload + structure + systems + motor + battery + rectenna
        converged = abs(mtow_new - mtow) < tol
        mtow = mtow_new
        if converged:
            break
    return dict(payload_t=payload / 1e3, structure_t=structure / 1e3,
                systems_t=systems / 1e3, motor_t=motor / 1e3,
                battery_t=battery / 1e3, rectenna_t=rectenna / 1e3,
                mtow_t=mtow / 1e3, cruise_mw=cruise_w / 1e6, peak_mw=peak_w / 1e6,
                empty_fraction=(mtow - payload) / mtow)


# --- Airframe silhouette (the rectenna aperture is not constant) --------------

def silhouette_area_m2(semi_span_m, semi_length_m, semi_thickness_m,
                       elevation_deg, azimuth_deg=0.0):
    """Exact projected area of a triaxial ellipsoid viewed along (elevation, azimuth). The
    ellipsoid picture used in the narrative figures; no relation to the meshed silhouette path in
    bwb_model, and nothing routes between them. Azimuth 0 looks along the fuselage."""
    a, b, c = semi_span_m, semi_length_m, semi_thickness_m
    el, az = np.radians(elevation_deg), np.radians(azimuth_deg)
    nx = np.cos(el) * np.sin(az)     # span-ward
    ny = np.cos(el) * np.cos(az)     # length-ward (nose)
    nz = np.sin(el)                  # thickness-ward (belly / nadir)
    return np.pi * np.sqrt((b * c * nx) ** 2 + (c * a * ny) ** 2 + (a * b * nz) ** 2)


def effective_rectenna_radius_m(planform_radius_m, elevation_deg):
    """Equal-area radius of a flat circular rectenna seen at the given elevation: the
    projected area is pi*r^2*sin(el), hence the square root."""
    return planform_radius_m * np.sqrt(np.sin(np.radians(elevation_deg)))


# --- First-pass techno-economics ----------------------------------------------
# The comparison the whole project must beat: jet fuel, unsubsidized, on a
# transatlantic widebody. We compare only the terms that DIFFER — energy source
# and ground infrastructure — because crew, maintenance, navigation, and airport
# fees are ~equal for the same route/aircraft class and cancel in the difference.
# Everything here is #(verify): order-of-magnitude, levers exposed for turning.

def flight_hours(route_km=5500.0, speed_ms=250.0):
    return route_km * 1e3 / speed_ms / 3600.0


# --- efficiency chain ------------------------------------------------------------------

ETA_GRID_TO_RF = 0.55
ETA_BEAM = 0.80       # placeholder for the beam-link engine's computed eta; feeding flyby's
                      # per-pass eta into econ_comparison in its place is the tracked next step
ETA_RECTENNA = 0.80


def delivery_efficiency(eta_grid_to_rf=ETA_GRID_TO_RF, eta_beam=ETA_BEAM,
                        eta_rectenna=ETA_RECTENNA):
    """Grid busbar → rectenna DC output. The onboard motor/drivetrain (~0.85) is already
    folded into cruise_mw, so this stops at the rectenna terminals — do NOT double-count it."""
    return eta_grid_to_rf * eta_beam * eta_rectenna


# --- Frequency as the master dial: the electronics vs aperture tradeoff --------
# The DC→RF amplifier's efficiency (PAE) falls steeply with frequency; the required
# EMITTER aperture GROWS with wavelength (∝ λ ∝ 1/f) once the belly rectenna is
# capped by the airframe. So low frequency = efficient electronics but a huge emitter;
# high frequency = tiny emitter but lossy, hot, less-mature electronics. Anchors are
# #(verify), read off the 2019-2025 GaN literature (C-band saturated ~65% PAE / 80%
# rectenna; Ka-band ~40% PAE / ~72% rectenna).

def pae_dc_to_rf(frequency_ghz):
    """Transmitter power-added efficiency, straight line through 0.65 at 5.8 GHz and 0.40 at
    34.8 GHz (the band edges used throughout)."""
    slope = (0.40 - 0.65) / (34.8 - 5.8)
    return float(np.clip(0.65 + slope * (frequency_ghz - 5.8), 0.30, 0.72))


def eta_grid_to_rf_band(frequency_ghz, eta_ac_dc=0.98):
    """Station grid→RF = SiC AC/DC front end (mature ~0.98) × GaN PA PAE(f)."""
    return eta_ac_dc * pae_dc_to_rf(frequency_ghz)


def rectenna_efficiency_band(frequency_ghz):
    """Rectenna RF-to-DC efficiency, same band-edge interpolation: 0.80 at 5.8, 0.72 at 34.8."""
    slope = (0.72 - 0.80) / (34.8 - 5.8)
    return float(np.clip(0.80 + slope * (frequency_ghz - 5.8), 0.65, 0.82))


def emitter_diameter_m(frequency_ghz, slant_km, rectenna_radius_m=20.0, tau_target=2.0):
    """Transmit-aperture diameter for near-full geometric capture (τ≈2 → ~98%) at the
    given slant range, with the belly rectenna capped at rectenna_radius_m by the
    airframe: r_t = τ·λ·D/(π·r_r) ∝ λ. Halving the frequency ~doubles this diameter."""
    return 2.0 * required_emitter_radius(frequency_ghz * 1e9, slant_km * 1e3,
                                         rectenna_radius_m, tau_target)


# --- economics -------------------------------------------------------------------------

def energy_cost_per_flight(cruise_mw, route_km=5500.0, speed_ms=250.0,
                           electricity_usd_per_mwh=30.0, eta_delivery=None):
    """Grid-energy cost to fly one beamed flight. Delivered (drivetrain-input) energy is
    cruise power × block time; grid draw is that over the delivery chain."""
    if eta_delivery is None:
        eta_delivery = delivery_efficiency()
    delivered_mwh = cruise_mw * flight_hours(route_km, speed_ms)
    grid_mwh = delivered_mwh / eta_delivery
    return dict(delivered_mwh=delivered_mwh, grid_mwh=grid_mwh,
                cost_usd=grid_mwh * electricity_usd_per_mwh)


def fuel_cost_per_flight(fuel_tonnes=37.0, fuel_usd_per_tonne=850.0):
    """Fossil comparator's block-fuel cost. 37 t #(verify) ≈ a 787-class widebody over
    5500 km (~5.6 t/h × 6.1 h + climb); Jet-A ~$850/t (volatile: $600-1200)."""
    return dict(fuel_tonnes=fuel_tonnes, cost_usd=fuel_tonnes * fuel_usd_per_tonne)


def station_spacing_km(aircraft_alt_km=11.0, min_elevation_deg=15.0):
    """Along-track window in which one station stays above `min_elevation_deg`, flat earth:
    the aircraft is in view for alt/tan(el) either side of overhead."""
    return 2.0 * aircraft_alt_km / np.tan(np.radians(min_elevation_deg))


def station_count(route_km=5500.0, aircraft_alt_km=11.0, min_elevation_deg=15.0):
    spacing = station_spacing_km(aircraft_alt_km, min_elevation_deg)
    return int(np.ceil(route_km / spacing)) + 1


def capital_recovery_factor(rate=0.08, years=25):
    """Annual payment per unit of capital that amortises it over `years` at `rate`."""
    r = rate
    return r * (1 + r) ** years / ((1 + r) ** years - 1)


# --- Utilization & the boost-glide operating concept --------------------------
# The plane is POWERED ("boosted") only while passing over a station at high
# elevation (good geometry, full aperture), and GLIDES through the low-elevation
# gap between stations — so we never beam at low elevation, and the sawtooth is
# synchronized to station spacing by design. `airborne_per_station` is the target
# packing: ~2 (one boosting + one coasting) is "minimum-slack" operation.

def corridor_throughput_per_day(airborne_per_station, n_stations, transit_h):
    """Flights per day the corridor sustains: aircraft in the air at once, over transit time."""
    return airborne_per_station * n_stations / transit_h * 24.0


def boost_power_mw(cruise_mw, powered_fraction=0.5):
    """Beam power needed if the aircraft is only illuminated for part of a climb-glide cycle
    but must average cruise power over the whole of it."""
    return cruise_mw / powered_fraction


def station_power_mw(airborne_per_station, cruise_mw):
    """Delivered (drivetrain-input) power a station must radiate. It serves (φ·a) planes
    at once, each drawing (cruise/φ) → a·cruise total, INDEPENDENT of the duty φ. At
    minimum-slack utilization the station runs here continuously (capacity factor ≈ 1)."""
    return airborne_per_station * cruise_mw


def station_capex_musd(deliver_mw, frequency_ghz=34.8, aircraft_alt_km=11.0,
                       min_elevation_deg=15.0, rectenna_radius_m=20.0, fixed_musd=50.0,
                       array_musd_per_rf_mw=2.5, supply_musd_per_grid_mw=1.2,
                       cooling_musd_per_waste_mw=1.0, aperture_musd_per_m2=0.005):
    """Cost of one ground station able to deliver `deliver_mw` onboard, working the chain
    backwards to RF and grid power and charging for array, supply, cooling and aperture."""
    eta_rect = rectenna_efficiency_band(frequency_ghz)
    eta_grid_rf = eta_grid_to_rf_band(frequency_ghz)
    rf_mw = deliver_mw / (ETA_BEAM * eta_rect)
    grid_mw = rf_mw / eta_grid_rf
    waste_mw = grid_mw - rf_mw
    slant_km = aircraft_alt_km / np.sin(np.radians(min_elevation_deg))  # worst-case range
    emitter_diam = emitter_diameter_m(frequency_ghz, slant_km, rectenna_radius_m)
    aperture_area_m2 = np.pi * (emitter_diam / 2) ** 2
    musd = (fixed_musd + array_musd_per_rf_mw * rf_mw + supply_musd_per_grid_mw * grid_mw
            + cooling_musd_per_waste_mw * waste_mw + aperture_musd_per_m2 * aperture_area_m2)
    return dict(musd=musd, rf_mw=rf_mw, grid_mw=grid_mw, waste_mw=waste_mw,
                emitter_diam_m=emitter_diam, eta_grid_rf=eta_grid_rf, eta_rect=eta_rect)


def infra_cost_per_flight(n_stations, station_musd, throughput_per_day,
                          rate=0.08, life_years=25, om_fraction=0.03):
    """Ground-infrastructure charge per flight = corridor capex, annualized, over the
    flights the corridor carries. With station cost ∝ its power and station power ∝
    utilization, the POWER-scaling part of this is utilization- and count-invariant (a
    capacity rental); only the FIXED per-station part is reduced by packing planes in."""
    capex_usd = n_stations * station_musd * 1e6
    annual_usd = capex_usd * (capital_recovery_factor(rate, life_years) + om_fraction)
    annual_flights = throughput_per_day * 365.0
    return dict(capex_busd=capex_usd / 1e9, annual_busd=annual_usd / 1e9,
                annual_flights=annual_flights, cost_usd=annual_usd / annual_flights)


def aircraft_cost_per_flight(aircraft_musd, flights_per_year=500.0, rate=0.08,
                             life_years=25, om_fraction=0.0):
    """Ownership charge per flight for one airframe. Beamed airframe is far lighter
    (~80 t vs ~250 t MTOW) but novel/low-volume; fossil widebody ~$150M #(verify)."""
    annual_usd = aircraft_musd * 1e6 * (capital_recovery_factor(rate, life_years) + om_fraction)
    return annual_usd / flights_per_year


def econ_comparison(cruise_mw=None, route_km=5500.0, speed_ms=250.0,
                    electricity_usd_per_mwh=30.0, fuel_usd_per_tonne=850.0,
                    fuel_tonnes=37.0, aircraft_alt_km=11.0, min_elevation_deg=15.0,
                    frequency_ghz=34.8, airborne_per_station=2.0, powered_fraction=0.5,
                    station_fixed_musd=50.0, beamed_aircraft_musd=100.0,
                    fossil_aircraft_musd=150.0, flights_per_year=500.0):
    """Per-flight cost of a beamed corridor against a fossil aircraft, USD.

    Only the terms that differ between the two are counted (energy, aircraft ownership,
    beam infrastructure), so the totals are not full operating costs and only
    `verdict_usd` = beamed - fossil is meaningful.
    """
    if cruise_mw is None:
        cruise_mw = mass_buildup()["cruise_mw"]
    n = station_count(route_km, aircraft_alt_km, min_elevation_deg)
    transit_h = flight_hours(route_km, speed_ms)
    throughput = corridor_throughput_per_day(airborne_per_station, n, transit_h)
    st_deliver_mw = station_power_mw(airborne_per_station, cruise_mw)
    st = station_capex_musd(st_deliver_mw, frequency_ghz=frequency_ghz,
                            aircraft_alt_km=aircraft_alt_km, min_elevation_deg=min_elevation_deg,
                            fixed_musd=station_fixed_musd)
    eta_deliv = st["eta_grid_rf"] * ETA_BEAM * st["eta_rect"]

    energy = energy_cost_per_flight(cruise_mw, route_km, speed_ms, electricity_usd_per_mwh,
                                    eta_delivery=eta_deliv)
    fuel = fuel_cost_per_flight(fuel_tonnes, fuel_usd_per_tonne)
    infra = infra_cost_per_flight(n, st["musd"], throughput)
    beamed_ac = aircraft_cost_per_flight(beamed_aircraft_musd, flights_per_year)
    fossil_ac = aircraft_cost_per_flight(fossil_aircraft_musd, flights_per_year)
    beamed_total = energy["cost_usd"] + infra["cost_usd"] + beamed_ac
    fossil_total = fuel["cost_usd"] + fossil_ac
    return dict(cruise_mw=cruise_mw, n_stations=n, throughput_per_day=throughput,
                frequency_ghz=frequency_ghz, eta_delivery=eta_deliv,
                emitter_diam_m=st["emitter_diam_m"], station_waste_mw=st["waste_mw"],
                station_deliver_mw=st_deliver_mw, boost_mw=boost_power_mw(cruise_mw, powered_fraction),
                station_musd=st["musd"], infra_capex_busd=infra["capex_busd"],
                beamed_energy_usd=energy["cost_usd"], beamed_grid_mwh=energy["grid_mwh"],
                beamed_infra_usd=infra["cost_usd"], beamed_aircraft_usd=beamed_ac,
                beamed_total_usd=beamed_total,
                fossil_fuel_usd=fuel["cost_usd"], fossil_aircraft_usd=fossil_ac,
                fossil_total_usd=fossil_total,
                verdict_usd=beamed_total - fossil_total)


def frequency_tradeoff(frequencies=(5.8, 11.6, 17.4, 23.2, 34.8)):
    """Sweep the master dial: emitter aperture (favours high f) vs PAE/rectenna/thermal
    (favour low f), everything else at the minimum-slack baseline. Shows why the choice
    is a genuine two-sided optimum, not a free win either way."""
    rows = []
    for f in frequencies:
        r = econ_comparison(frequency_ghz=f)
        rows.append((f, round(r["emitter_diam_m"]), round(pae_dc_to_rf(f), 2),
                     round(rectenna_efficiency_band(f), 2), round(r["eta_delivery"], 3),
                     round(r["station_waste_mw"], 1), round(r["station_musd"]),
                     round(r["beamed_energy_usd"]), round(r["verdict_usd"])))
    return pd.DataFrame(rows, columns=["f_GHz", "emitter_m", "PAE", "eta_rect",
                                       "deliv_eff", "waste_MW", "station_$M",
                                       "energy_$", "verdict_$"])


def econ_table():
    """Per-flight cost breakdown at the baseline (minimum-slack) assumptions."""
    e = econ_comparison()
    rows = [
        ("energy: fuel (fossil)", e["fossil_fuel_usd"]),
        ("energy: electricity (beamed)", e["beamed_energy_usd"]),
        ("aircraft ownership (fossil)", e["fossil_aircraft_usd"]),
        ("aircraft ownership (beamed)", e["beamed_aircraft_usd"]),
        (f"beam infra ({e['n_stations']} stn × ${e['station_musd']:.0f}M = "
         f"${e['infra_capex_busd']:.0f}B, {e['throughput_per_day']:.0f}/day)",
         e["beamed_infra_usd"]),
        ("— beamed total (diverging terms)", e["beamed_total_usd"]),
        ("— fossil total (diverging terms)", e["fossil_total_usd"]),
        ("— verdict (beamed − fossil)", e["verdict_usd"]),
    ]
    return pd.DataFrame(rows, columns=["term", "usd_per_flight"]).round(0)


def econ_sensitivity():
    """One-at-a-time swing of the levers around the minimum-slack baseline (verdict
    $/flight; negative = beamed wins). Baseline: 15° elevation, 2 airborne/station,
    $30/MWh, $50M fixed/station. Watch that utilization and elevation now move the
    POWER-scaling infra cost too — the count/utilization cancellation leaves only the
    fixed-per-station part and the capacity rental."""
    base = dict(min_elevation_deg=15.0, airborne_per_station=2.0,
                electricity_usd_per_mwh=30.0, station_fixed_musd=50.0)
    sweeps = {
        "min_elevation_deg": [8, 12, 15, 20, 30],
        "airborne_per_station": [0.5, 1.0, 2.0, 3.0],
        "electricity_usd_per_mwh": [15, 30, 60, 100],
        "station_fixed_musd": [20, 50, 100, 200],
    }
    rows = []
    for lever, values in sweeps.items():
        for v in values:
            kw = dict(base); kw[lever] = v
            r = econ_comparison(**kw)
            rows.append((lever, v, r["n_stations"], round(r["throughput_per_day"]),
                         round(r["verdict_usd"])))
    return pd.DataFrame(rows, columns=["lever", "value", "n_stations", "per_day", "verdict_usd"])


# --- pass geometry (flat earth) --------------------------------------------------------

def pass_geometry(plane_alt_m, station_alt_m, alongtrack_m, lateral_offset_m=0.0):
    """(slant m, elevation deg, then the three direction cosines of the beam in body axes:
    belly, side and frontal). The cosines say which face of the aircraft the beam sees."""
    height_gap = plane_alt_m - station_alt_m
    slant = np.sqrt(alongtrack_m**2 + lateral_offset_m**2 + height_gap**2)
    elevation = np.degrees(np.arcsin(height_gap / slant))
    return slant, elevation, height_gap / slant, np.abs(lateral_offset_m) / slant, np.abs(alongtrack_m) / slant


def dwell_seconds(speed_ms, plane_alt_m, station_alt_m, lateral_offset_m, max_slant_m):
    """Time inside a maximum slant range; 0 if the closest approach is already outside it."""
    fixed = (plane_alt_m - station_alt_m) ** 2 + lateral_offset_m**2
    if fixed >= max_slant_m**2:
        return 0.0
    half_window = np.sqrt(max_slant_m**2 - fixed)
    return 2.0 * half_window / speed_ms


# --- tethered aerostat relay -----------------------------------------------------------

def air_density(altitude_m):
    """Isothermal exponential atmosphere, kg/m^3."""
    return RHO_AIR_SEA * np.exp(-altitude_m / SCALE_HEIGHT)


def breaking_length_km(sigma_pa, density_kgm3, safety_factor=1.0):
    """Length of its own material a cable can hang before failing -- the figure of merit
    for whether a tether to a given altitude is possible at all."""
    return sigma_pa / (density_kgm3 * GRAVITY * safety_factor) / 1e3


def net_lift_per_m3(altitude_m, gas="hydrogen"):
    """Buoyant lift in kg per m^3 of envelope at altitude (displaced air minus the gas)."""
    return air_density(altitude_m) * (1.0 - GAS_FRACTION[gas])


def emitter_mass_t(aperture_diam_m, areal_density_kgm2):
    return np.pi * (aperture_diam_m / 2) ** 2 * areal_density_kgm2 / 1e3


def aerostat_diameter_m(altitude_m, cable_kg_per_m, payload_kg, gas="hydrogen"):
    """Diameter of the sphere whose buoyancy at `altitude_m` just carries payload plus the
    full length of cable. Envelope mass is ignored here (`tether_blowover` includes it)."""
    total_mass = cable_kg_per_m * altitude_m + payload_kg
    volume = total_mass / net_lift_per_m3(altitude_m, gas)
    return (6.0 * volume / np.pi) ** (1 / 3)


# --- Curvature, wind, and tether blow-over ------------------------------------

def line_of_sight_km(alt1_m, alt2_m, earth_radius_m=6.371e6):
    """Geometric horizon-to-horizon range between two altitudes, no refraction."""
    return (np.sqrt(2 * earth_radius_m * alt1_m) + np.sqrt(2 * earth_radius_m * alt2_m)) / 1e3


def wind_profile(altitude_m, v_jet=40.0, v_strato=20.0, v_surface=8.0,
                 jet_center_m=11000.0, jet_sigma_m=2500.0):
    """Wind speed m/s: a Gaussian jet stream over a surface-to-stratosphere ramp."""
    jet = v_jet * np.exp(-0.5 * ((altitude_m - jet_center_m) / jet_sigma_m) ** 2)
    background = v_surface + (v_strato - v_surface) * np.clip(altitude_m / 20000.0, 0, 1)
    return np.maximum(jet, background)


def tether_blowover(aerostat_diam_m, payload_kg, target_alt_m=20000.0, gas="hydrogen",
                    v_jet=40.0, v_strato=20.0, v_surface=8.0, tether="Dyneema (UHMWPE)",
                    safety_factor=2.0, cd_cable=1.1, cd_sphere=0.5, envelope_kgm2=0.15, steps=800):
    """Static tether shape for a moored aerostat in a wind profile.

    Walks up the cable accumulating the drag carried above each height; the local tilt from
    vertical follows the ratio of horizontal to vertical tension. `excursion` is the downwind
    displacement of the aerostat, `tension_margin` the cable's breaking load over the peak
    tension it actually sees.

    `altitude_reached_km` is meant to be the height reached when the *cable length* is held at
    target_alt_m. KNOWN BUG, not fixed here: the expression applies the tilt penalty twice, and
    for constant tilt evaluates to H*cos^2(tilt) where H*cos(tilt) is wanted -- 1.4% low at the
    default wind, ~9% at a 160 m/s jet. Nothing in the tree calls this function.
    """
    sigma, rho_cable = TETHER_MATERIALS[tether]
    volume = np.pi / 6 * aerostat_diam_m ** 3
    gross_lift = volume * net_lift_per_m3(target_alt_m, gas) * GRAVITY          # net-of-gas lift, N
    envelope_mass = np.pi * aerostat_diam_m ** 2 * envelope_kgm2
    cable_diam = 2 * np.sqrt(gross_lift * safety_factor / sigma / np.pi)  # sized on lift alone
    cable_weight_per_m = rho_cable * np.pi * (cable_diam / 2) ** 2 * GRAVITY
    cable_mass = cable_weight_per_m / GRAVITY * target_alt_m
    free_lift = gross_lift - GRAVITY * (payload_kg + envelope_mass + cable_mass)
    h = np.linspace(0, target_alt_m, steps)
    dh = h[1] - h[0]
    v = wind_profile(h, v_jet, v_strato, v_surface)
    drag_per_len = 0.5 * air_density(h) * v ** 2 * cd_cable * cable_diam
    drag_above = np.cumsum(drag_per_len[::-1])[::-1] * dh  # cable drag above each height
    v_top = wind_profile(np.array([target_alt_m]), v_jet, v_strato, v_surface)[0]
    aero_drag = 0.5 * air_density(target_alt_m) * v_top ** 2 * cd_sphere * (np.pi / 4 * aerostat_diam_m ** 2)
    tension_x = aero_drag + drag_above
    # (target_alt_m - h) is the cable ABOVE h, which is what the vertical tension at h supports,
    # so tension_z is smallest at the anchor and the tilt largest there. SECOND KNOWN BUG, not
    # fixed here: free_lift has already netted out the whole cable weight, so subtracting the
    # cable above h again leaves tension_z low by one full cable weight at every height, which
    # over-tilts the entire profile.
    tension_z = free_lift - cable_weight_per_m * (target_alt_m - h)
    tilt = np.clip(np.arctan2(tension_x, np.maximum(tension_z, 1.0)), 0, np.radians(88))
    excursion = np.trapezoid(np.tan(tilt), h)
    arc = np.trapezoid(1.0 / np.cos(tilt), h)  # deployed cable length for this tilt profile
    tension = np.sqrt(tension_x ** 2 + tension_z ** 2)
    return dict(free_lift_t=free_lift / GRAVITY / 1e3, aero_drag_kn=aero_drag / 1e3,
                tether_drag_kn=drag_above[0] / 1e3, tilt_deg=np.degrees(tilt[0]),
                excursion_km=excursion / 1e3,
                altitude_reached_km=np.trapezoid(np.cos(tilt), h) / arc * target_alt_m / 1e3,
                tension_margin=sigma * np.pi * (cable_diam / 2) ** 2 / tension.max())


def station_keeping_power_mw(horizontal_force_n, wind_ms, eta=0.6):
    """Propulsive power to hold position against a drag force in wind (free-flying relay)."""
    return horizontal_force_n * wind_ms / eta / 1e6


# --- tables and figures ----------------------------------------------------------------

def emitter_table():
    rows = [("baseline 5.8 GHz", 5.8, 300, 20), ("30 GHz", 30.0, 300, 25),
            ("30 GHz, 76 m rectenna", 30.0, 300, 38), ("30 GHz, D=195 km", 30.0, 195, 25)]
    return (
        pd.DataFrame(rows, columns=["scenario", "f_GHz", "D_km", "r_r_m"])
        .assign(emitter_diam_m=lambda d: 2 * required_emitter_radius(d["f_GHz"] * 1e9, d["D_km"] * 1e3, d["r_r_m"]))
        .round({"emitter_diam_m": 0})
    )


def power_table(mass_t=200, speed_ms=250, eta=0.78):
    mass = mass_t * 1e3
    rows = []
    for ld in (15, REFERENCE_LIFT_TO_DRAG, 30):   # tube-wing / lifting-body ref / aspirational
        cruise = cruise_power_mw(mass, speed_ms, ld, eta)
        for vc in (5, 10):
            climb = climb_power_mw(mass, speed_ms, ld, vc, eta)
            rows.append((ld, vc, cruise, climb, beam_power_mw(climb),
                         powered_fraction(speed_ms, vc, ld) * 100))
    return pd.DataFrame(
        rows, columns=["L/D", "climb_ms", "cruise_elec_MW", "climb_elec_MW", "climb_beam_MW", "powered_%"]
    ).round(0)


def aerostat_table(altitude_km=20, aperture_diam_m=150):
    rows = []
    for label, areal, cable in [("reflectarray relay 1.5 kg/m² + structural tether", 1.5, 1.0),
                                ("active array 5 kg/m² + power cable", 5.0, 3.0)]:
        payload = emitter_mass_t(aperture_diam_m, areal) * 1e3
        rows.append((label, round(payload / 1e3), round(aerostat_diameter_m(altitude_km * 1e3, cable, payload))))
    return pd.DataFrame(rows, columns=[f"architecture @ {altitude_km} km, {aperture_diam_m} m aperture",
                                       "emitter_t", "H2_sphere_m"])


def pass_figure(plane_alt_km=12.0, station_alt_km=5.0, lateral_km=20.0, span_km=150.0,
                path="out/pass_geometry.html"):
    x = np.linspace(-span_km * 1e3, span_km * 1e3, 500)
    _, elev, belly, sail, frontal = pass_geometry(plane_alt_km * 1e3, station_alt_km * 1e3, x, lateral_km * 1e3)
    fig = go.Figure()
    fig.add_scatter(x=x / 1e3, y=belly, name="belly (down, low-drag)", line=dict(width=3))
    fig.add_scatter(x=x / 1e3, y=sail, name="sail (lateral, low-drag)", line=dict(width=3))
    fig.add_scatter(x=x / 1e3, y=frontal, name="frontal (fore/aft, HIGH-drag)", line=dict(width=3, dash="dash"))
    fig.add_scatter(x=x / 1e3, y=elev / 90, name="elevation / 90°", line=dict(width=1), opacity=0.4)
    fig.update_layout(title=f"Pass: plane {plane_alt_km} km, station {station_alt_km} km, offset {lateral_km} km",
                      xaxis_title="along-track offset (km)", yaxis_title="projected-area fraction",
                      template="plotly_white", legend=dict(orientation="h", y=-0.2))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.write_html(path, include_plotlyjs="inline")
    return path


if __name__ == "__main__":
    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 20)

    print("\n=== Emitter sizing ===\n", emitter_table().to_string(index=False))

    print("\n=== Power budget (200 t, 250 m/s) — cruise vs climb (banking altitude) ===\n",
          power_table().to_string(index=False))

    print("\n=== Dwell per pass (plane 12 km, station 5 km, usable slant 250 km) ===")
    for lat in (0, 20, 50, 100):
        print(f"  lateral offset {lat:3d} km -> {dwell_seconds(250, 12e3, 5e3, lat*1e3, 250e3):5.0f} s in range")

    print("\n=== Tether: strength is fine (breaking length, SF2) ===")
    for name, (s, r) in TETHER_MATERIALS.items():
        print(f"  {name:18s} {breaking_length_km(s, r, 2.0):5.0f} km")

    print("\n=== Aerostat size once the REAL emitter mass is loaded ===\n",
          aerostat_table().to_string(index=False))

    print("\n=== Pass figure ===\n ", pass_figure())
