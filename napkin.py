"""Napkin math for microwave-powered aviation — pushing on the envelope.

First-order relations behind the 'Premise & target envelope' section of
research-questions.md. SI units unless a name says otherwise. Every number this
produces is #(verify)-grade: order-of-magnitude sanity checks, not engineering.

Run:  uv run python napkin.py   → prints tables, writes out/pass_geometry.html
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

GAS_FRACTION = {"hydrogen": 2 / 29, "helium": 4 / 29}   # lifting-gas density / air density

TETHER_MATERIALS = {  # tensile strength (Pa), density (kg/m^3)
    "Dyneema (UHMWPE)": (3.5e9, 970),
    "Zylon (PBO)": (5.8e9, 1560),
    "Carbon (T1100)": (7.0e9, 1790),
    "HS steel": (2.0e9, 7850),
}


# --- Beam coupling ------------------------------------------------------------

def wavelength(frequency_hz):
    return SPEED_OF_LIGHT / frequency_hz


def coupling_tau(frequency_hz, distance_m, emitter_radius_m, receiver_radius_m):
    """Goubau aperture-coupling parameter τ = π·r_t·r_r/(λ·D)."""
    return np.pi * emitter_radius_m * receiver_radius_m / (wavelength(frequency_hz) * distance_m)


def collection_efficiency(tau):
    """Fraction of radiated power caught by the receiving aperture: η ≈ 1−exp(−τ²)."""
    return 1.0 - np.exp(-(tau ** 2))


def required_emitter_radius(frequency_hz, distance_m, receiver_radius_m, tau_target=2.0):
    """r_t = τ·λ·D/(π·r_r); τ≈2 → ~94% collection."""
    return tau_target * wavelength(frequency_hz) * distance_m / (np.pi * receiver_radius_m)


def surface_precision(frequency_hz, fraction=1 / 16):
    """Element-POSITION knowledge for phase correction (~λ/16) — not mechanical flatness."""
    return wavelength(frequency_hz) * fraction


def power_density_limited_diameter(power_w, density_wcm2=25.0):
    """Min aperture diameter to radiate/collect `power_w` without exceeding a surface
    power-density limit (~25 W/cm² for solid-state arrays and rectennas alike):
    d = 2·√(P/(π·S)). Small apertures hit this floor long before the diffraction
    minimum — which is why the 'diffraction-tiny' feed dish is really power-density-bound."""
    area_m2 = power_w / (density_wcm2 * 1e4)
    return 2.0 * np.sqrt(area_m2 / np.pi)


# --- Passive frequency multiplication (the "gearbox") -------------------------
# Manley-Rowe permits ~100% f -> n·f conversion for an IDEAL lossless nonlinear
# reactance (varactor); real multipliers fall well short, and a high order like
# 5.8 -> 30 GHz (~x6) is built as a cascade whose per-stage losses compound.

def multiplier_factor_prime(input_ghz=5.8, output_ghz=30.0):
    """Required integer multiplication order (rounded) to reach `output_ghz`."""
    return round(output_ghz / input_ghz)


def cascade_efficiency(order, stage_efficiency=0.7):
    """End-to-end efficiency of a multiplier chain realizing `order` as a product
    of small prime stages (x2/x3), each at `stage_efficiency`. A x6 = x2·x3 is two
    stages; x5 (5.8->30, if taken as prime) is one hard high-order stage. Returns
    (efficiency, [prime stages])."""
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


# --- Atmospheric attenuation (rain) -------------------------------------------

# ITU-R P.838-3 curve-fit coefficients (Gaussian sums in log-frequency). Valid
# 1–1000 GHz. Ported so k, α can be evaluated at ANY frequency, not just 30 GHz.
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
    log_f = np.log10(frequency_ghz)
    gaussians = sum(a * np.exp(-(((log_f - b) / c) ** 2))
                    for a, b, c in zip(coeffs["aj"], coeffs["bj"], coeffs["cj"]))
    value = gaussians + coeffs["m"] * log_f + coeffs["c"]
    return 10.0 ** value if is_k else value


def rain_coefficients(frequency_ghz, elevation_deg=90.0, polarization_tilt_deg=0.0):
    """ITU-R P.838-3 (k, α) for rain specific attenuation at arbitrary frequency,
    path elevation, and polarization tilt (τ=0 horizontal, 90 vertical, 45 circular)."""
    kh = _p838_term(frequency_ghz, _P838["kh"], is_k=True)
    kv = _p838_term(frequency_ghz, _P838["kv"], is_k=True)
    ah = _p838_term(frequency_ghz, _P838["ah"], is_k=False)
    av = _p838_term(frequency_ghz, _P838["av"], is_k=False)
    cos_el2 = np.cos(np.radians(elevation_deg)) ** 2
    cos_2tau = np.cos(np.radians(2 * polarization_tilt_deg))
    k = (kh + kv + (kh - kv) * cos_el2 * cos_2tau) / 2.0
    alpha = (kh * ah + kv * av + (kh * ah - kv * av) * cos_el2 * cos_2tau) / (2.0 * k)
    return k, alpha


RAIN_K, RAIN_ALPHA = rain_coefficients(30.0)   # 30 GHz, vertical path (project reference)


def specific_rain_attenuation(rain_rate_mmh, k=RAIN_K, alpha=RAIN_ALPHA):
    """ITU-R P.838 specific attenuation γ = k·R^α [dB/km]. Defaults: 30 GHz, H-pol."""
    return k * rain_rate_mmh ** alpha


def rain_fade_db(rain_rate_mmh, rain_height_km, elevation_deg=90.0, k=RAIN_K, alpha=RAIN_ALPHA):
    """Rain attenuation over a slant path through a rain layer of depth `rain_height_km`
    at the given elevation: γ·h_R/sin(ε). 90° = vertical hop; low (grazing) elevation
    blows the path length — and hence the dB — up."""
    path_km = rain_height_km / np.sin(np.radians(elevation_deg))
    return specific_rain_attenuation(rain_rate_mmh, k, alpha) * path_km


def db_to_fraction(db):
    """Power fraction surviving `db` of attenuation: 10^(−db/10)."""
    return 10.0 ** (-db / 10.0)


# --- Flight profile -----------------------------------------------------------

# Lifting body / BWB. The bare AIRFRAME L/D is shared with the fossil comparator
# (airframe gains are not a beaming benefit — tube-and-wing ~17-19). The BEAMED
# plane earns a modest bump from its ELECTRIC-PROPULSION INTEGRATION — boundary-
# layer ingestion, no podded nacelles, distributed fans — which is an energy-path
# advantage, so it is legitimately credited to the concept (#(verify): ~+5-10%).
AIRFRAME_LIFT_TO_DRAG = 22.0
BEAMED_LIFT_TO_DRAG = 24.0
REFERENCE_LIFT_TO_DRAG = BEAMED_LIFT_TO_DRAG   # headline value used downstream


def glide_range_km(lift_to_drag, altitude_drop_m):
    return lift_to_drag * altitude_drop_m / 1e3


def powered_fraction(speed_ms, climb_rate_ms, lift_to_drag):
    """Fraction of a climb-then-glide sawtooth under power = (v/vc)/(v/vc + L/D)."""
    ratio = speed_ms / climb_rate_ms
    return ratio / (ratio + lift_to_drag)


def cruise_power_mw(mass_kg, speed_ms, lift_to_drag, eta_chain=0.78):
    """Electrical power for level cruise: W·v/(L/D)/η. Drag is altitude-independent."""
    thrust = mass_kg * GRAVITY / lift_to_drag
    return thrust * speed_ms / eta_chain / 1e6


def climb_power_mw(mass_kg, speed_ms, lift_to_drag, climb_rate_ms, eta_chain=0.78):
    """Cruise + potential-energy rate (W·climb_rate) while banking altitude for the glide."""
    prop = mass_kg * GRAVITY * speed_ms / lift_to_drag + mass_kg * GRAVITY * climb_rate_ms
    return prop / eta_chain / 1e6


def beam_power_mw(electrical_mw, rectenna_eff=0.85, collection_eff=0.90):
    """What the emitter must put into the beam to deliver a given electrical power on board."""
    return electrical_mw / rectenna_eff / collection_eff


# --- Mass buildup (the size cascade, made concrete) ---------------------------

def mass_buildup(payload_t=40.0, speed_ms=250.0, lift_to_drag=BEAMED_LIFT_TO_DRAG,
                 climb_rate_ms=4.0, eta_chain=0.85, struct_fraction=0.30,
                 systems_fraction=0.10, motor_specific_power_kw_per_kg=5.0,
                 battery_specific_energy_wh_per_kg=250.0, buffer_minutes=5.0,
                 rectenna_mass_t=2.0, max_iter=200, tol=1.0):
    """No-fuel beamed-electric mass buildup by fixed-point iteration `#(verify)`.

    The point that dooms battery-electric aviation — hours of stored energy at
    ~250 Wh/kg — does NOT apply here: energy arrives continuously, so the battery is
    only a *buffer* (minutes; the real reserve is the glide, L/D·Δh). MTOW closes on
    itself: structure & systems scale with MTOW, the motor with the peak (climb)
    power that mass demands, the buffer battery with the cruise power. Contrast a
    fuel burner, whose ~40% takeoff fuel inflates MTOW and hence structure — remove
    it and the whole aircraft shrinks. Returns component masses (t), MTOW, powers."""
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
    """Orthogonal silhouette (shadow) area of a BWB approximated as an ellipsoid with
    semi-axes (span, length, thickness), viewed from `elevation_deg` above the wing
    plane and `azimuth_deg` around it (0 = nose-on, 90 = side-on). Exact for an
    ellipsoid: A = π·√((b·c·nx)² + (c·a·ny)² + (a·b·nz)²)."""
    a, b, c = semi_span_m, semi_length_m, semi_thickness_m
    el, az = np.radians(elevation_deg), np.radians(azimuth_deg)
    nx = np.cos(el) * np.sin(az)     # span-ward
    ny = np.cos(el) * np.cos(az)     # length-ward (nose)
    nz = np.sin(el)                  # thickness-ward (belly / nadir)
    return np.pi * np.sqrt((b * c * nx) ** 2 + (c * a * ny) ** 2 + (a * b * nz) ** 2)


def effective_rectenna_radius_m(planform_radius_m, elevation_deg):
    """Effective radius of a flat belly rectenna seen by an emitter at elevation ε:
    projected area ∝ sin ε (cosine law off the nadir), so radius ∝ √(sin ε). Full
    value directly overhead (ε=90°); it collapses toward the grazing pass — i.e. the
    beam-budget r_r ≈ 20 m holds only near nadir."""
    return planform_radius_m * np.sqrt(np.sin(np.radians(elevation_deg)))


# --- First-pass techno-economics ----------------------------------------------
# The comparison the whole project must beat: jet fuel, unsubsidized, on a
# transatlantic widebody. We compare only the terms that DIFFER — energy source
# and ground infrastructure — because crew, maintenance, navigation, and airport
# fees are ~equal for the same route/aircraft class and cancel in the difference.
# Everything here is #(verify): order-of-magnitude, levers exposed for turning.

def flight_hours(route_km=5500.0, speed_ms=250.0):
    return route_km * 1e3 / speed_ms / 3600.0


# Delivery-chain efficiencies, grid busbar → rectenna DC output (#(verify)):
ETA_GRID_TO_RF = 0.55    # AC→DC→RF at the station (GaN PA-dominated, ~40-60% observed)
ETA_BEAM = 0.80          # transmit aperture → captured at rectenna (atmos + spillover + geometry)
ETA_RECTENNA = 0.80      # RF→DC rectification (~72-82% demonstrated)


def delivery_efficiency(eta_grid_to_rf=ETA_GRID_TO_RF, eta_beam=ETA_BEAM,
                        eta_rectenna=ETA_RECTENNA):
    """Grid busbar → rectenna DC output. The onboard motor/drivetrain (~0.85) is already
    folded into cruise_mw, so this stops at the rectenna terminals — do NOT double-count it."""
    return eta_grid_to_rf * eta_beam * eta_rectenna


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
    """Along-track spacing that keeps the aircraft above `min_elevation_deg` from the
    nearest station: 2·h/tan(ε). Low ε → wide spacing (few stations) but long slant
    path and collapsed rectenna aperture (√sin ε); high ε → many stations. This single
    knob dominates station count, and station count dominates the economics."""
    return 2.0 * aircraft_alt_km / np.tan(np.radians(min_elevation_deg))


def station_count(route_km=5500.0, aircraft_alt_km=11.0, min_elevation_deg=15.0):
    spacing = station_spacing_km(aircraft_alt_km, min_elevation_deg)
    return int(np.ceil(route_km / spacing)) + 1


def capital_recovery_factor(rate=0.08, years=25):
    """Annual fraction of a capex to recover principal + interest over `years`."""
    r = rate
    return r * (1 + r) ** years / ((1 + r) ** years - 1)


# --- Utilization & the boost-glide operating concept --------------------------
# The plane is POWERED ("boosted") only while passing over a station at high
# elevation (good geometry, full aperture), and GLIDES through the low-elevation
# gap between stations — so we never beam at low elevation, and the sawtooth is
# synchronized to station spacing by design. `airborne_per_station` is the target
# packing: ~2 (one boosting + one coasting) is "minimum-slack" operation.

def corridor_throughput_per_day(airborne_per_station, n_stations, transit_h):
    """Flights/day the corridor delivers: a fleet of (airborne_per_station·n) planes,
    each clearing the corridor in transit_h, completes at fleet/transit rate."""
    return airborne_per_station * n_stations / transit_h * 24.0


def boost_power_mw(cruise_mw, powered_fraction=0.5):
    """Instantaneous power a plane must receive while boosting: it takes on all of a
    cycle's energy during the powered fraction φ, so boost = cruise/φ (the glide returns
    the climbed potential energy). φ≈0.5 is the one-boost/one-coast sawtooth."""
    return cruise_mw / powered_fraction


def station_power_mw(airborne_per_station, cruise_mw):
    """Delivered (drivetrain-input) power a station must radiate. It serves (φ·a) planes
    at once, each drawing (cruise/φ) → a·cruise total, INDEPENDENT of the duty φ. At
    minimum-slack utilization the station runs here continuously (capacity factor ≈ 1)."""
    return airborne_per_station * cruise_mw


def station_capex_musd(deliver_mw, fixed_musd=50.0, array_musd_per_rf_mw=2.5,
                       supply_musd_per_grid_mw=1.2):
    """Station cost split by what drives it: a FIXED platform/siting cost (per station,
    the mid-ocean pain — a floating structure exists whether it radiates 1 or 100 MW),
    plus a phased array ∝ RF transmit power, plus power provisioning ∝ grid draw
    (mid-ocean = a local floating plant). Anchors ~$190M at 19 MW delivered #(verify)."""
    rf_mw = deliver_mw / (ETA_BEAM * ETA_RECTENNA)
    grid_mw = rf_mw / ETA_GRID_TO_RF
    return fixed_musd + array_musd_per_rf_mw * rf_mw + supply_musd_per_grid_mw * grid_mw


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
                    airborne_per_station=2.0, powered_fraction=0.5,
                    station_fixed_musd=50.0, beamed_aircraft_musd=100.0,
                    fossil_aircraft_musd=150.0, flights_per_year=500.0):
    """Per-flight cost difference on the terms that diverge, with the operating concept
    baked in: station power scales with utilization, throughput is derived from packing,
    and station cost = fixed + power-scaling. `verdict_usd` = beamed − fossil (neg = win)."""
    if cruise_mw is None:
        cruise_mw = mass_buildup()["cruise_mw"]
    n = station_count(route_km, aircraft_alt_km, min_elevation_deg)
    transit_h = flight_hours(route_km, speed_ms)
    throughput = corridor_throughput_per_day(airborne_per_station, n, transit_h)
    st_deliver_mw = station_power_mw(airborne_per_station, cruise_mw)
    st_musd = station_capex_musd(st_deliver_mw, fixed_musd=station_fixed_musd)

    energy = energy_cost_per_flight(cruise_mw, route_km, speed_ms, electricity_usd_per_mwh)
    fuel = fuel_cost_per_flight(fuel_tonnes, fuel_usd_per_tonne)
    infra = infra_cost_per_flight(n, st_musd, throughput)
    beamed_ac = aircraft_cost_per_flight(beamed_aircraft_musd, flights_per_year)
    fossil_ac = aircraft_cost_per_flight(fossil_aircraft_musd, flights_per_year)
    beamed_total = energy["cost_usd"] + infra["cost_usd"] + beamed_ac
    fossil_total = fuel["cost_usd"] + fossil_ac
    return dict(cruise_mw=cruise_mw, n_stations=n, throughput_per_day=throughput,
                station_deliver_mw=st_deliver_mw, boost_mw=boost_power_mw(cruise_mw, powered_fraction),
                station_musd=st_musd, infra_capex_busd=infra["capex_busd"],
                beamed_energy_usd=energy["cost_usd"], beamed_grid_mwh=energy["grid_mwh"],
                beamed_infra_usd=infra["cost_usd"], beamed_aircraft_usd=beamed_ac,
                beamed_total_usd=beamed_total,
                fossil_fuel_usd=fuel["cost_usd"], fossil_aircraft_usd=fossil_ac,
                fossil_total_usd=fossil_total,
                verdict_usd=beamed_total - fossil_total)


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


# --- Pass geometry & dwell ----------------------------------------------------

def pass_geometry(plane_alt_m, station_alt_m, alongtrack_m, lateral_offset_m=0.0):
    """Projected-area FRACTIONS (=beam direction cosines) for belly/sail/frontal surfaces."""
    height_gap = plane_alt_m - station_alt_m
    slant = np.sqrt(alongtrack_m**2 + lateral_offset_m**2 + height_gap**2)
    elevation = np.degrees(np.arcsin(height_gap / slant))
    return slant, elevation, height_gap / slant, np.abs(lateral_offset_m) / slant, np.abs(alongtrack_m) / slant


def dwell_seconds(speed_ms, plane_alt_m, station_alt_m, lateral_offset_m, max_slant_m):
    """Seconds in usable range (slant < max_slant) as the plane transits at cruise speed.

    Zero if the closest approach already exceeds max_slant.
    """
    fixed = (plane_alt_m - station_alt_m) ** 2 + lateral_offset_m**2
    if fixed >= max_slant_m**2:
        return 0.0
    half_window = np.sqrt(max_slant_m**2 - fixed)
    return 2.0 * half_window / speed_ms


# --- Tether: strength vs buoyancy vs emitter mass -----------------------------

def air_density(altitude_m):
    return RHO_AIR_SEA * np.exp(-altitude_m / SCALE_HEIGHT)


def breaking_length_km(sigma_pa, density_kgm3, safety_factor=1.0):
    return sigma_pa / (density_kgm3 * GRAVITY * safety_factor) / 1e3


def net_lift_per_m3(altitude_m, gas="hydrogen"):
    return air_density(altitude_m) * (1.0 - GAS_FRACTION[gas])


def emitter_mass_t(aperture_diam_m, areal_density_kgm2):
    return np.pi * (aperture_diam_m / 2) ** 2 * areal_density_kgm2 / 1e3


def aerostat_diameter_m(altitude_m, cable_kg_per_m, payload_kg, gas="hydrogen"):
    total_mass = cable_kg_per_m * altitude_m + payload_kg
    volume = total_mass / net_lift_per_m3(altitude_m, gas)
    return (6.0 * volume / np.pi) ** (1 / 3)


# --- Curvature, wind, and tether blow-over ------------------------------------

def line_of_sight_km(alt1_m, alt2_m, earth_radius_m=6.371e6):
    """Geometric line-of-sight range between two altitudes over a spherical Earth:
    √(2R·h1)+√(2R·h2). Caps ground-station reach; quantifies the reach benefit of a relay."""
    return (np.sqrt(2 * earth_radius_m * alt1_m) + np.sqrt(2 * earth_radius_m * alt2_m)) / 1e3


def wind_profile(altitude_m, v_jet=40.0, v_strato=20.0, v_surface=8.0,
                 jet_center_m=11000.0, jet_sigma_m=2500.0):
    """Stylized wind speed vs altitude: a Gaussian jet-stream bump at the tropopause over a
    background rising from surface to stratosphere. Napkin only — real profiles vary widely."""
    jet = v_jet * np.exp(-0.5 * ((altitude_m - jet_center_m) / jet_sigma_m) ** 2)
    background = v_surface + (v_strato - v_surface) * np.clip(altitude_m / 20000.0, 0, 1)
    return np.maximum(jet, background)


def tether_blowover(aerostat_diam_m, payload_kg, target_alt_m=20000.0, gas="hydrogen",
                    v_jet=40.0, v_strato=20.0, v_surface=8.0, tether="Dyneema (UHMWPE)",
                    safety_factor=2.0, cd_cable=1.1, cd_sphere=0.5, envelope_kgm2=0.15, steps=800):
    """Static blow-over of a tethered stratospheric aerostat under a wind profile.

    Net buoyancy/free lift tensions the cable; the aerostat's bluff-body drag plus the tether's
    distributed drag tilt it. Returns free lift, the two drag terms, tilt at the anchor, downwind
    excursion, the altitude actually reached on a target-length cable, and the breaking/peak-tension
    margin. STATIC only — no vortex-induced vibration, gusts, or dynamics (the real open risk).
    """
    sigma, rho_cable = TETHER_MATERIALS[tether]
    volume = np.pi / 6 * aerostat_diam_m ** 3
    gross_lift = volume * net_lift_per_m3(target_alt_m, gas) * GRAVITY          # net-of-gas lift, N
    envelope_mass = np.pi * aerostat_diam_m ** 2 * envelope_kgm2
    cable_diam = 2 * np.sqrt(gross_lift * safety_factor / sigma / np.pi)
    cable_weight_per_m = rho_cable * np.pi * (cable_diam / 2) ** 2 * GRAVITY
    cable_mass = cable_weight_per_m / GRAVITY * target_alt_m
    free_lift = gross_lift - GRAVITY * (payload_kg + envelope_mass + cable_mass)
    h = np.linspace(0, target_alt_m, steps)
    dh = h[1] - h[0]
    v = wind_profile(h, v_jet, v_strato, v_surface)
    drag_per_len = 0.5 * air_density(h) * v ** 2 * cd_cable * cable_diam
    drag_above = np.cumsum(drag_per_len[::-1])[::-1] * dh                       # drag from h up to top
    v_top = wind_profile(np.array([target_alt_m]), v_jet, v_strato, v_surface)[0]
    aero_drag = 0.5 * air_density(target_alt_m) * v_top ** 2 * cd_sphere * (np.pi / 4 * aerostat_diam_m ** 2)
    tension_x = aero_drag + drag_above
    tension_z = free_lift - cable_weight_per_m * (target_alt_m - h)
    tilt = np.clip(np.arctan2(tension_x, np.maximum(tension_z, 1.0)), 0, np.radians(88))
    excursion = np.trapezoid(np.tan(tilt), h)
    arc = np.trapezoid(1.0 / np.cos(tilt), h)
    tension = np.sqrt(tension_x ** 2 + tension_z ** 2)
    return dict(free_lift_t=free_lift / GRAVITY / 1e3, aero_drag_kn=aero_drag / 1e3,
                tether_drag_kn=drag_above[0] / 1e3, tilt_deg=np.degrees(tilt[0]),
                excursion_km=excursion / 1e3,
                altitude_reached_km=np.trapezoid(np.cos(tilt), h) / arc * target_alt_m / 1e3,
                tension_margin=sigma * np.pi * (cable_diam / 2) ** 2 / tension.max())


def station_keeping_power_mw(horizontal_force_n, wind_ms, eta=0.6):
    """Power to hold station against a steady wind = thrust·wind/η, and thrust = the horizontal
    drag → the power scales as wind³. Must be delivered aloft. η optimistic for thin-air props."""
    return horizontal_force_n * wind_ms / eta / 1e6


# --- Tables & figure ----------------------------------------------------------

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
