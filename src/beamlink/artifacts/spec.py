"""The run spec: everything the engine is told, as frozen dataclasses.

One `RunSpec` in, one run directory out (SPEC.md §4). Everything that used to be a convention
buried in a sweep or a figure is an explicit input here — the trajectory (§8 item 21), the
serve policy that sets the headline pass efficiency (§8 item 20), the sampling geometry the
field is evaluated at (D2), and the aperture's areal density that replaces a hardcoded
rectenna mass (D5).

The spec imports nothing but numpy, and it holds no physics: it names quantities and their
units, and the engine decides what they mean.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

import numpy as np

# --- weather -----------------------------------------------------------------------------
# Rain fills everything below rain_top_m at a uniform rate; cloud fills the altitude band at a
# uniform liquid water content. Rain rates are the usual light/heavy/storm brackets in mm/h and
# the layer tops rise with intensity. A named preset is a shorthand for a `Weather` value; the
# engine only ever sees the value, so a scenario may carry an unnamed one.


@dataclass(frozen=True)
class Weather:
    """The atmospheric state a path is integrated through."""

    name: str = "clear"
    rain_rate_mmh: float = 0.0
    rain_top_m: float = 0.0
    cloud_lwc_gm3: float = 0.0
    cloud_band_m: tuple[float, float] = (0.0, 0.0)


WEATHER_PRESETS: Mapping[str, Weather] = {
    "clear": Weather("clear"),
    "cloudy": Weather("cloudy", cloud_lwc_gm3=0.5, cloud_band_m=(1000.0, 3000.0)),
    "light_rain": Weather("light_rain", rain_rate_mmh=5.0, rain_top_m=3000.0,
                          cloud_lwc_gm3=0.5, cloud_band_m=(1000.0, 3000.0)),
    "heavy_rain": Weather("heavy_rain", rain_rate_mmh=25.0, rain_top_m=4000.0,
                          cloud_lwc_gm3=1.0, cloud_band_m=(1000.0, 4000.0)),
    "storm": Weather("storm", rain_rate_mmh=50.0, rain_top_m=5000.0,
                     cloud_lwc_gm3=2.0, cloud_band_m=(1000.0, 5000.0)),
}


def weather(name: str) -> Weather:
    """A named weather preset. Raises on an unknown name rather than defaulting to clear."""
    if name not in WEATHER_PRESETS:
        raise KeyError(f"unknown weather {name!r}; have {sorted(WEATHER_PRESETS)}")
    return WEATHER_PRESETS[name]


# --- the aircraft ------------------------------------------------------------------------


@dataclass(frozen=True)
class LoftSpec:
    """Blended-wing-body outer SHAPE. Lengths only — no tessellation.

    How finely the shape is discretised is a numerical choice, not a property of the aircraft, so
    `n_span`/`n_theta` live in the fidelity config and not here. Keeping them apart is what lets the
    face count be a fidelity LEVER at all (§6 measures it as the largest one: 8× on the dominant cost
    for 0.34 pp of capture) — while it sat in the loft, changing the tier changed nothing (D14).
    """

    full_span_m: float = 80.0
    root_chord_m: float = 40.0
    tip_chord_m: float = 3.0
    root_thickness_m: float = 6.0
    tip_thickness_m: float = 0.4
    sweep_m: float = 25.0
    body_fraction: float = 0.25
    winglet_height_m: float = 3.0
    winglet_root_chord_m: float = 3.0
    winglet_tip_chord_m: float = 1.2
    winglet_sweep_m: float = 1.5
    winglet_cant_deg: float = 20.0
    winglet_thickness_m: float = 0.3


@dataclass(frozen=True)
class ReceiverSpec:
    """The receiving side of the link. A scenario may have none — then P5 and P4 exist and
    P3/P6/P7 do not, which is the general form of the "pure focusing sweep" figure (D2, §2)."""

    incidence_floor: float = 0.90
    """Rectenna angular response at grazing incidence. 0.90 is a conformal phased array whose
    elements re-phase toward the arrival angle; a fixed patch array would sit nearer 0.70. The
    cos(theta) area foreshortening is NOT here — it is in the silhouette's projected area."""

    rectenna_areal_density_kg_m2: float = 1.0
    """Rectenna mass per unit of BELLY-ON silhouette area — the area of the array as built, which
    is an airframe property and not the per-sample illuminated area. D5: this replaces the
    hardcoded `rectenna_mass_t = 2.0`, so rectenna mass follows from geometry (P3's area) and a
    declared input rather than from an assumption. 1 kg/m² over this loft's 2095 m² belly-on
    silhouette reproduces the old 2 t, which is what makes it a like-for-like replacement rather
    than a quiet change to the mass cascade."""

    max_power_density_w_m2: float | None = 2.5e5
    """Skin power-density cap, enforced by defocusing. None disables the cap entirely."""


@dataclass(frozen=True)
class AirframeSpec:
    """The flight-mechanics inputs behind P11–P13. Independent of the link: neither domain
    imports the other, and they meet only here and at the named power reference plane (D5)."""

    loft: LoftSpec = field(default_factory=LoftSpec)
    payload_t: float = 40.0
    lift_to_drag: float = 24.0
    """Beamed airframe. 22 is the bare lifting-body/BWB shared with the fossil comparator; the
    +2 is electric-propulsion integration, which is an energy-path advantage and so is
    legitimately the concept's."""

    airframe_lift_to_drag: float = 22.0
    """The bare airframe, used for the unpowered glide reserve (P13) — the propulsion-integration
    bonus is not available with the beam off."""

    climb_rate_ms: float = 4.0
    drivetrain_efficiency: float = 0.85
    """Rectenna terminals → propulsive power. Named here because P12 quotes its requirement at
    the rectenna terminals and P8 quotes delivered power at the same plane (D5), which is what
    makes the double count structurally impossible rather than warned against in a comment."""

    structure_fraction: float = 0.30
    systems_fraction: float = 0.10
    motor_specific_power_kw_per_kg: float = 5.0
    battery_specific_energy_wh_per_kg: float = 250.0
    reserve_minutes: float = 5.0
    dropout_altitude_m: float = 11_000.0
    """Altitude the unpowered glide in P13 starts from."""


# --- the emitting side -------------------------------------------------------------------


@dataclass(frozen=True)
class ApertureSpec:
    """The emitting aperture and the policy that shapes it per geometry."""

    diameter_m: float | None = None
    """Equal-area circular diameter. None sizes it from `tau_target` at each sample's geometry,
    which is a per-sample resize a real station cannot do — use it for design studies, and give
    an explicit diameter for anything meant to be one built array."""

    tau_target: float = 2.0
    """Goubau coupling target used when `diameter_m` is None. τ=2 → ~98 % collection."""

    shaped: bool = True
    """Make the aperture elliptical and align it to the receiver: stretched along whichever
    direction the target is narrow in, because spot width goes as 1/aperture width."""

    fixed_aspect: float | None = None
    """A FIXED elliptical aperture of this major:minor ratio at the same area — one built array,
    not reshaped per geometry. None → the adaptive per-geometry aspect."""

    aspect_cap: float = 8.0
    taper_p: float = 1.0
    """Illumination taper exponent of (1 - rho^2)^p. Used directly when `adaptive_taper` is off."""

    adaptive_taper: bool = True
    taper_p_max: float = 4.0
    error_efficiency: float = 0.95
    """Illumination/phase-error efficiency of the array. Does NOT include a scan roll-off: every
    call site in the pre-SPEC engine passed scan_deg=0, which is only correct for a mechanically
    pointed aperture, and the baseline emitter is a phased array. Carried forward as the same
    unjustified omission rather than silently given a new derivation."""

    pointing_error_m: float = 2.0
    """Beam offset MAGNITUDE at the receiver, applied across the receiver's NARROW principal axis,
    which is where both the spot and the target are narrowest.

    Its direction is not an input: the engine points it at the thin side of the projection, from P3's
    own third moment, so it is genuinely the worse of the two directions and does not depend on the
    sign convention of the transverse frame (D13)."""


@dataclass(frozen=True)
class RelaySpec:
    """A relay between the station and the aircraft. The up-leg is aperture-to-aperture; the
    down-leg emits from the relay's own array."""

    position_m: tuple[float, float, float]
    mode: str = "active"
    """active     rectify to DC and re-transmit at f_out, so both conversions are paid
       multiplier harmonic multiplier chain behind a reflectarray, one stage per prime factor
       passive    reflect only; the loss is Ruze, and it cannot retune, so f_out is ignored"""

    aperture_diam_m: float = 60.0
    f_in_ghz: float = 5.8
    f_out_ghz: float = 30.0
    stage_efficiency: float = 0.70
    reflectarray_reflectivity: float = 0.90
    surface_error_wavelengths: float = 1.0 / 50.0


@dataclass(frozen=True)
class ScenarioSpec:
    """One link case: geometry, frequency, aperture, weather, and the aircraft it serves."""

    station_position_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """(downrange north, crossrange east, altitude) in metres, relative to the station's ground
    point. Mapped onto a spherical earth by the engine, so curvature is intrinsic downstream."""

    frequency_ghz: float = 5.8
    """The frequency of the leg that serves the aircraft. With a relay the aircraft-side
    frequency is the relay's `f_out_ghz` and this is ignored."""

    aperture: ApertureSpec = field(default_factory=ApertureSpec)
    relay: RelaySpec | None = None
    weather: Weather = field(default_factory=Weather)
    pae_scenario: str = "today"
    """Transmitter power-added efficiency against frequency: "today" | "medium" | "flat"."""

    polarization: str = "circular"
    receiver: ReceiverSpec | None = field(default_factory=ReceiverSpec)
    airframe: AirframeSpec = field(default_factory=AirframeSpec)
    aim_reference_enu: tuple[float, float, float] = (0.0, 0.0, 1.0)
    """With no receiver there is no silhouette to align the aperture to, so the aperture's own
    transverse frame needs a declared reference direction; the major axis is this vector
    orthogonalised against the propagation direction. Ignored when a receiver is present."""


# --- trajectory --------------------------------------------------------------------------


@dataclass(frozen=True)
class TrajectorySpec:
    """Where and how the aircraft was, sample by sample — an input object, not a convention.

    Three different pass spans in three modules is how the two orphaned pass efficiencies in
    §1 came to disagree, so the span is now something a run states rather than something a
    sweep decides (§8 item 21).
    """

    positions_m: np.ndarray
    """(n_samples, 3) as (downrange, crossrange, altitude) in metres."""

    attitudes_deg: np.ndarray
    """(n_samples, 3) as (heading from north towards east, pitch nose-up, roll right-wing-down)."""

    times_s: np.ndarray
    """(n_samples,) seconds. Sets every time integral in P9, so it is required, not derived
    from a speed the aggregate would otherwise have to assume."""

    name: str = "trajectory"

    def __post_init__(self):
        for attribute in ("positions_m", "attitudes_deg", "times_s"):
            object.__setattr__(self, attribute,
                               np.asarray(getattr(self, attribute), dtype=float))
        if self.positions_m.ndim != 2 or self.positions_m.shape[1] != 3:
            raise ValueError(f"positions_m must be (n, 3), got {self.positions_m.shape}")
        if self.attitudes_deg.shape != self.positions_m.shape:
            raise ValueError("attitudes_deg must match positions_m; attitude is an input, not a "
                             "convention (SPEC.md §8 item 21)")
        if self.times_s.shape != (self.positions_m.shape[0],):
            raise ValueError(f"times_s must be (n,) with n = {self.positions_m.shape[0]}")

    @property
    def n_samples(self) -> int:
        return int(self.positions_m.shape[0])

    @classmethod
    def straight_overflight(cls, half_span_m: float, altitude_m: float, speed_ms: float,
                            n_samples: int = 81, lateral_offset_m: float = 0.0,
                            heading_deg: float = 0.0, name: str = "straight_overflight"):
        """A level straight pass through overhead, `half_span_m` either side, at constant speed.

        A convenience constructor, not a default: a run that uses it has said so, and the span
        and sample count are in the spec where an aggregate's definition can name them.
        """
        along = np.linspace(-float(half_span_m), float(half_span_m), int(n_samples))
        positions = np.stack([along, np.full_like(along, float(lateral_offset_m)),
                              np.full_like(along, float(altitude_m))], axis=1)
        attitudes = np.zeros_like(positions)
        attitudes[:, 0] = float(heading_deg)
        return cls(positions_m=positions, attitudes_deg=attitudes,
                   times_s=(along - along[0]) / float(speed_ms), name=name)


# --- power -------------------------------------------------------------------------------

POWER_REFERENCE_PLANES = ("grid", "radiated", "at_receiver", "rectenna_terminals")
"""The named planes P8 quotes power at, station-side first. `rectenna_terminals` is onboard DC
out of the rectenna, BEFORE the motor/drivetrain — the same plane P12 states the airframe's
requirement at, which is what makes the drivetrain double-count structurally impossible (D5)."""


@dataclass(frozen=True)
class PowerSpec:
    """What sets the absolute scale of the run. Exactly one of the two is given."""

    delivered_target_w: float | None = 10.0e6
    """Target power at `rectenna_terminals`. The station is then rated to deliver this at the
    best sample of the trajectory and left at that grid draw (see the serve policy in P9)."""

    radiated_w: float | None = None
    """Radiated RF instead, if the run is about a fixed transmitter rather than a fixed demand."""

    def __post_init__(self):
        given = [v for v in (self.delivered_target_w, self.radiated_w) if v is not None]
        if len(given) != 1:
            raise ValueError("give exactly one of delivered_target_w or radiated_w")


# --- sampling (D2) -----------------------------------------------------------------------

FIELD_SAMPLING_KINDS = ("silhouette", "transverse_plane", "axial_plane", "points", "volume")


@dataclass(frozen=True)
class FieldSampling:
    """Where the one field solve is evaluated. All samplers ship together (D2): no sampling is
    privileged by the engine, and a routine run simply does not request the expensive ones.

    Coordinates are in the APERTURE frame — z along the propagation direction from the emitting
    aperture, x along the aperture's major axis, y along its minor axis. The frame's axes in the
    earth frame are emitted with the field, so a consumer can place a cut in space without
    reconstructing any of this.

        silhouette         the receiver's own visible surface elements, at the receiver range
        transverse_plane   an x-y grid at one or more fractions of the receiver range
        axial_plane        a side view: one of the aperture's own two principal planes, which
                           are the beam's transverse principal axes. NOT the silhouette's
                           principal axes in a display basis, which is what made the pre-SPEC
                           F/F₂ cut pair viz-shaped (§8 item 13, D2)
        points             arbitrary points, given directly in the aperture frame
        volume             a 3-D box grid, built in v1 so the capability is honest
    """

    kind: str
    name: str = ""
    occluders: str = "none"
    """"none" or "receiver". The receiver's shadow is an option on the solve, not a product of
    its own: same shape, same units (§5 P5)."""

    # transverse_plane / axial_plane / volume
    z_fractions: tuple[float, ...] = ()
    """Ranges to evaluate at, as fractions of the receiver range. 1.0 is the receiver plane;
    greater than 1.0 is past it, which is where the power that missed the aircraft goes."""

    n_axial: int = 0
    """axial_plane / volume only: number of ranges, spread over `z_fractions`' min..max."""

    axis: str = "major"
    """axial_plane only: which of the aperture's principal planes — "major" is the plane
    containing the aperture's major axis (the narrow spot cut), "minor" the other."""

    n_transverse: tuple[int, int] = (0, 0)
    """(n_x, n_y) for transverse_plane and volume; (n_radial, 1) is implied for axial_plane."""

    n_radial: int = 0
    extent_m: tuple[float, float] = (0.0, 0.0)
    """(half-extent on x, half-extent on y) in metres. (0, 0) asks the engine to fit the extent
    to the beam at the receiver plane and record what it chose."""

    points_m: np.ndarray | None = None
    """(n_points, 3) aperture-frame points for kind="points"."""

    def __post_init__(self):
        if self.kind not in FIELD_SAMPLING_KINDS:
            raise ValueError(f"unknown sampling kind {self.kind!r}; have {FIELD_SAMPLING_KINDS}")
        if not self.name:
            object.__setattr__(self, "name", self.kind)
        if self.occluders not in ("none", "receiver"):
            raise ValueError(f"occluders must be 'none' or 'receiver', got {self.occluders!r}")
        if self.points_m is not None:
            object.__setattr__(self, "points_m", np.asarray(self.points_m, dtype=float))
        if self.kind == "points" and self.points_m is None:
            raise ValueError("kind='points' needs points_m")
        if self.kind == "axial_plane":
            if self.axis not in ("major", "minor"):
                raise ValueError(f"axis must be 'major' or 'minor', got {self.axis!r}")
            if self.n_axial < 1 or self.n_radial < 1:
                raise ValueError("axial_plane needs n_axial and n_radial")
        if self.kind == "transverse_plane" and (min(self.n_transverse) < 1
                                                or not self.z_fractions):
            raise ValueError("transverse_plane needs n_transverse and z_fractions")
        if self.kind == "volume" and (min(self.n_transverse) < 1 or self.n_axial < 1):
            raise ValueError("volume needs n_transverse and n_axial")


def default_axial_planes(n_axial: int = 160, n_radial: int = 201,
                         z_fractions: tuple[float, float] = (0.0, 1.0),
                         occluders: str = "none") -> tuple[FieldSampling, ...]:
    """The two physically-defined default cuts (D2): the aperture's own principal planes."""
    return tuple(
        FieldSampling(kind="axial_plane", name=f"axial_{axis}", axis=axis, n_axial=n_axial,
                      n_radial=n_radial, z_fractions=z_fractions, occluders=occluders)
        for axis in ("major", "minor"))


# --- products, fidelity, aggregates ------------------------------------------------------

PRODUCTS = ("trajectory", "geometry", "silhouette", "aperture", "field", "capture", "chain",
            "power", "aggregates", "atmosphere", "mass", "flight_power", "reserve")
"""P1–P13 in SPEC.md §5 order. A name not in this tuple is a spec error, not a silent skip."""


@dataclass(frozen=True)
class ProductRequest:
    """One requested product, with its per-product sampling and fidelity."""

    name: str
    sampling: tuple[FieldSampling, ...] = ()
    """`field` only. Empty asks for the engine's default: silhouette elements when there is a
    receiver, and nothing else."""

    tier: str | None = None
    """Overrides the run's fidelity tier for this product. Rejected where it would split a
    shared solve — see D8."""

    emit_aperture_phase: bool = False
    """`aperture` only: also emit the aperture phase as a field over the aperture (P4)."""

    serve_eta_floor: float = 0.02
    """`aggregates` only: the η below which the station is not served. An OPERATING policy, so it
    belongs to the reduction that uses it rather than to the scenario. Not derived from anything, and
    it sets the headline — which is why every aggregate that applies it records it (§8 item 20)."""

    def __post_init__(self):
        if self.name not in PRODUCTS:
            raise ValueError(f"unknown product {self.name!r}; have {PRODUCTS}")


@dataclass(frozen=True)
class FidelitySpec:
    """Which tier each product runs at (I5). A tier varies method and resolution; it never
    varies shape, units or meaning, and every tier's error against `full` is published."""

    tier: str = "standard"
    overrides: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateAxis:
    """One swept design parameter (D3). `parameter` is a dotted path into the scenario, e.g.
    "aperture.fixed_aspect" or "frequency_ghz"; `values` has one entry per candidate.

    Several axes are zipped, not crossed: they all have length n_candidates and index the same
    candidate list. Building a cross product is the driver's job, because only the driver knows
    whether a full grid is affordable (D4).
    """

    parameter: str
    values: tuple


@dataclass(frozen=True)
class RunSpec:
    """One run: one scenario batch, one trajectory, one set of requested products."""

    scenario: ScenarioSpec = field(default_factory=ScenarioSpec)
    trajectory: TrajectorySpec = field(
        default_factory=lambda: TrajectorySpec.straight_overflight(
            half_span_m=100_000.0, altitude_m=11_000.0, speed_ms=250.0, n_samples=81))
    products: tuple[ProductRequest, ...] = ()
    fidelity: FidelitySpec = field(default_factory=FidelitySpec)
    power: PowerSpec = field(default_factory=PowerSpec)
    candidates: tuple[CandidateAxis, ...] = ()
    run_id: str = ""
    notes: str = ""

    def __post_init__(self):
        lengths = {len(axis.values) for axis in self.candidates}
        if len(lengths) > 1:
            raise ValueError(f"candidate axes are zipped, so they must be the same length; "
                             f"got {sorted(lengths)}")
        if not self.products:
            object.__setattr__(self, "products", default_products())

    @property
    def n_candidates(self) -> int:
        return len(self.candidates[0].values) if self.candidates else 1

    def requested(self, name: str) -> ProductRequest | None:
        return next((r for r in self.products if r.name == name), None)

    def with_scenario(self, **changes) -> "RunSpec":
        return replace(self, scenario=replace(self.scenario, **changes))


def default_products() -> tuple[ProductRequest, ...]:
    """Everything except the expensive optional field samplings and the aperture phase.

    A run that wants a side view or a volume asks for it; the point of listing the cheap
    products by default is that a run directory a consumer is pointed at has the primaries
    every derived product names as its inputs (I3).
    """
    return tuple(ProductRequest(name=name) for name in PRODUCTS)


# --- aggregate definitions (P9, and the driver's cross-run reductions in D6) --------------

SERVE_POLICIES = ("always", "eta_floor", "feasible")
STATISTICS = ("mean", "harmonic_mean", "peak", "minimum", "time_integral", "energy_ratio",
              "served_fraction")


@dataclass(frozen=True)
class AggregateDefinition:
    """What a named reduction actually did. Carried WITH the number, as data.

    This is the object whose absence made the two pass efficiencies in §1 indistinguishable:
    neither recorded its span, serve policy, statistic or sampling, so "the pass efficiency"
    named two different numbers that happened to agree in clear air at 200 km spacing and
    differed by 0.32× in a storm at 400 km.
    """

    quantity: str
    """The per-sample quantity reduced, named as it appears in the product it came from."""

    statistic: str
    span: str
    """The sample range reduced over, as a statement about the trajectory: "trajectory" is all
    of it, and a run whose trajectory is half a station spacing says so in its own spec."""

    serve_policy: str = "always"
    serve_eta_floor: float = 0.0
    """The η below which the station is not served. NOT derived from anything, and it sets the
    headline: 0 → 0.10 moved the pre-SPEC `energy_eta` from 0.090 to 0.240 for 5.8-direct in a
    storm. Now a declared input recorded with the number it produced (§8 item 20)."""

    weighting: str = "time"
    """"time" weights by the trajectory's own dt, "sample" counts samples equally. The two
    differ as soon as a trajectory is not uniformly sampled."""

    inputs: tuple[str, ...] = ()
    """The products this was derived from. A derived product that cannot name its inputs is
    illegal (I3)."""

    unit: str = ""
    note: str = ""

    def __post_init__(self):
        if self.serve_policy not in SERVE_POLICIES:
            raise ValueError(f"unknown serve policy {self.serve_policy!r}")
        if self.statistic not in STATISTICS:
            raise ValueError(f"unknown statistic {self.statistic!r}")
        if self.weighting not in ("time", "sample"):
            raise ValueError(f"unknown weighting {self.weighting!r}")
