"""The products: P1–P13 of SPEC.md §5, as frozen dataclasses.

Every field states the physical question it answers. SI units throughout, full precision, no
presentation (I7): no label strings, no colours, no normalised axes, no NaN-separated
polylines, no "how far to draw".

Array shapes. Every per-sample array carries a leading CANDIDATE axis and then a SAMPLE axis,
so a single scenario is a candidate axis of length 1 and there is one code path rather than two
(I9). Where a quantity is per link leg the axis order is (candidate, leg, sample). Ragged inner
axes — the silhouette element cloud is a different size at every look angle — are padded and
carry a validity mask, the same device P5 uses for unresolved field regions (§6a).

These definitions live in `artifacts/` and not under `physics/` on purpose (SPEC.md §9): if they
lived with the engine, every consumer would have to import the engine to read a run and the
firewall would be a fiction. Physics still owns what is in them — it decides the products and
their fields — but the package that holds them has no repo dependencies and no physics in it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from beamlink.artifacts.spec import AggregateDefinition, FieldSampling


# --- P1 trajectory -----------------------------------------------------------------------


@dataclass(frozen=True)
class Trajectory:
    """P1 — where and how the aircraft was, sample by sample.

    The spec's trajectory echoed as data, so every other product can be indexed against it
    without a consumer having to re-read the spec and trust that the engine used it verbatim.
    """

    positions_m: np.ndarray
    """(sample, 3) — (downrange, crossrange, altitude) in metres from the station's ground point."""

    attitudes_deg: np.ndarray
    """(sample, 3) — (heading from north towards east, pitch nose-up, roll right-wing-down)."""

    times_s: np.ndarray
    """(sample,) — the clock every time integral in P9 uses."""

    positions_ecef_m: np.ndarray
    """(sample, 3) — the same positions on the spherical earth the engine actually solved on,
    so curvature is visible in the data rather than only inside the engine."""

    ground_speed_ms: np.ndarray
    """(sample,) — speed along the path, from the positions and times. What P12 costs to fly."""

    name: str = "trajectory"


# --- P2 geometry -------------------------------------------------------------------------


@dataclass(frozen=True)
class LinkGeometry:
    """P2 — what the link geometry was per sample, per leg.

    Elevation is GROUND-REFERENCED and defined per leg (D1): the elevation of a leg's receiving
    end as seen from its emitting end, above that end's local horizon. Ground→aircraft is
    positive and +90° overhead; relay→aircraft is correctly negative, the aircraft being below a
    relay's horizon. The pre-SPEC engine emitted an aircraft-referenced `elevation_deg` (−65.6°
    overhead) alongside a station-referenced `elevation_up_deg` in the same dict; the per-leg
    definition dissolves that double key rather than renaming it.
    """

    leg_names: tuple[str, ...]
    """One name per leg, emitting end first: ("station->aircraft",) or
    ("station->relay", "relay->aircraft")."""

    slant_m: np.ndarray
    """(candidate, leg, sample) — straight-line range between the leg's two ends."""

    elevation_deg: np.ndarray
    """(candidate, leg, sample) — see the class note. Ground-referenced, per leg."""

    horizon_blocked: np.ndarray
    """(candidate, leg, sample) bool — the straight path dips into the earth."""

    arrival_beta_deg: np.ndarray
    """(candidate, sample) — angle of the beam's arrival direction off the aircraft's belly axis,
    in the AIRFRAME body frame. 0 is straight below the aircraft. This is what the aircraft
    actually presents to the beam, and it is why nothing is lost by referencing elevation to the
    ground instead of the airframe (D1)."""

    arrival_alpha_deg: np.ndarray
    """(candidate, sample) — azimuth of the same direction about the belly axis, from the nose
    towards starboard."""

    emitter_position_ecef_m: np.ndarray
    """(candidate, leg, sample, 3) — where each leg emitted from, on the sphere."""


# --- P3 silhouette -----------------------------------------------------------------------


@dataclass(frozen=True)
class Silhouette:
    """P3 — what surface the capture integral summed over.

    The mesh AS INTEGRATED is emitted, which is what lets a viewer draw exactly the surface used
    instead of a lookalike; the pre-SPEC engine paid 2.6× the rasterisation cost to hold a figure
    requirement that this makes true by construction (§8 item 6).
    """

    mesh_vertices_m: np.ndarray
    """(n_vertices, 3) — the loft in the airframe body frame (+x nose, +y starboard, +z belly)."""

    mesh_faces: np.ndarray
    """(n_faces, 3) int — vertex indices of each triangle."""

    mesh_face_normals: np.ndarray
    """(n_faces, 3) — outward unit normals, so incidence is reproducible from the mesh alone."""

    element_offsets_m: np.ndarray
    """(candidate, sample, element, 2) — each visible element's (u, v) offset from the
    projected-area centroid, in the view basis below. Padded over the element axis."""

    element_projected_area_m2: np.ndarray
    """(candidate, sample, element) — the area each element presents TO the beam, i.e. skin area
    times incidence. The cos(theta) foreshortening is here and not in the incidence derate."""

    element_incidence_cos: np.ndarray
    """(candidate, sample, element) — cosine of the angle between the local surface normal and
    the arrival direction. The rectenna's angular response is applied to this by P6."""

    element_valid: np.ndarray
    """(candidate, sample, element) bool — the padding mask. The element cloud is a different
    size at every look angle, so batching over samples means padding (§6a)."""

    projected_area_m2: np.ndarray
    """(candidate, sample) — the illuminated, self-occluded silhouette area."""

    equal_area_radius_m: np.ndarray
    """(candidate, sample) — radius of the circle of the same area, which is what the closed-form
    coupling relations take as the receiver."""

    view_basis_body: np.ndarray
    """(candidate, sample, 3, 3) — rows (x̂, ŷ, ẑ) in the airframe body frame. ẑ is the PROPAGATION
    direction (emitter → aircraft); x̂ is the silhouette's least-spread ("narrow") principal axis in
    the plane transverse to it, sign-pinned to the airframe (dorsal, then starboard, then nose); and
    ŷ = ẑ × x̂.

    This is the SAME frame the aperture's principal axes define and the same frame P5's field is
    sampled in, which is the point: the pre-SPEC code derived these axes twice, once in the capture
    integral and once in a figure's display basis, and the second one is what made the F/F₂ cut pair
    viz-shaped (§8 item 13). It is recorded so a consumer never has to guess a basis — eigenvector
    signs are arbitrary and would otherwise flip between neighbouring samples.

    The SIGN of x̂ is a gauge choice (D13): the narrow axis rotates ~90° over a pass, so any fixed
    reference has a locus where the pin flips. Nothing physical depends on it — the aperture is an
    ellipse with even illumination — but a viewer that wants a continuous drawn orientation re-pins
    along the trajectory from these frames, which is presentation and therefore the viewer's (I7)."""

    principal_half_extent_m: np.ndarray
    """(candidate, sample, 2) — the outermost element along each of (x, y). What frames the outline
    and what sizes the receiver as an occluder (§5 P5)."""

    principal_sigma_m: np.ndarray
    """(candidate, sample, 2) — the projected-area-weighted rms extent along each of (x, y). The
    aperture's ASPECT is the ratio of these, not of the half-extents: rms is what the elongation of a
    distribution means, and a single outlying winglet pixel must not reshape the emitter. Emitted as
    a product because the aperture is chosen from it, so it is not a local calculation (I4)."""

    principal_skew_m3: np.ndarray
    """(candidate, sample, 2) — the projected-area-weighted third moment along each of (x, y).

    Which side of the projection is the THIN one, and therefore which direction a pointing error costs
    most in. The aperture's pointing offset is applied along −sign(skew_x), which is what makes the
    "worst case for a given magnitude" a well-posed statement rather than one arbitrary side of two
    that differ by ~7 pp of capture on this loft (D13). A product, not a local calculation, for the
    same reason the rms extents are one: the aperture is chosen from it (I4)."""

    max_half_extent_m: float
    """The largest silhouette half-extent over a full look-angle sweep — an airframe property,
    the same for every run of the same loft (§8 item 19)."""


# --- P4 aperture -------------------------------------------------------------------------


@dataclass(frozen=True)
class Aperture:
    """P4 — what the emitter did per sample."""

    semi_major_m: np.ndarray
    """(candidate, sample) — the long semi-axis of the elliptical aperture actually radiated."""

    semi_minor_m: np.ndarray
    """(candidate, sample)."""

    equal_area_diameter_m: np.ndarray
    """(candidate, sample) — diameter of the round aperture of the same area, which is the number
    a station is specified by."""

    taper_exponent: np.ndarray
    """(candidate, sample) — p in the illumination (1 - rho^2)^p that was used."""

    focus_range_m: np.ndarray
    """(candidate, sample) — the range the aperture phase was curved for. Shorter than the slant
    means deliberately defocused to hold the skin power-density cap."""

    aspect: np.ndarray
    """(candidate, sample) — semi_major / semi_minor. 1 is a round aperture."""

    frame_axes_ecef: np.ndarray
    """(candidate, sample, 3, 3) — rows (x̂, ŷ, ẑ) of the APERTURE frame in the earth frame: ẑ is
    the propagation direction, x̂ the aperture's major axis. This is what places a field sample
    set in space, and it is a physical basis rather than a camera (§8 items 7–9)."""

    origin_ecef_m: np.ndarray
    """(candidate, sample, 3) — the emitting aperture's centre."""

    phase_rad: np.ndarray | None = None
    """(candidate, sample, n, n) — the aperture phase as a field over the aperture, on request.
    Emitted instead of the zone-plate draw-parameters the pre-SPEC frame payload carried."""

    phase_axis_m: np.ndarray | None = None
    """(candidate, sample, n) — the aperture-plane coordinate `phase_rad` is sampled on, along
    both axes (the grid is square and centred)."""


# --- P5 field ----------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldSampleSet:
    """One named place the single field solve was evaluated.

    Sampling one solve at different locations is not two solves (I1), so the sample sets in one
    `Field` product all come from the same solver, at the same method and precision; what differs
    between them is where they asked.
    """

    sampling: FieldSampling
    """The request verbatim, so the sampling is answerable from the product."""

    points_m: np.ndarray
    """(candidate, sample, point, 3) — query points in the APERTURE frame (P4's `frame_axes_ecef`
    places them). For a plane or volume the grid is flattened over the point axis, and the axes
    below say how to fold it back."""

    intensity_w_m2: np.ndarray
    """(candidate, sample, point) — areal intensity in ABSOLUTE W/m², scaled by the radiated power
    P8 reports for the emitting leg. Absolute because an exposure level is absolute and because
    two legs of a relay are only comparable this way."""

    valid: np.ndarray
    """(candidate, sample, point) bool — False where the METHOD could not resolve the field, not
    where the field is small. Regions the method cannot reach are reported as unresolved, never
    filled to make a picture whole (§5 P5, §8 item 14)."""

    grid_shape: tuple[int, ...] = ()
    """How to fold the point axis back into a grid: () for a scattered cloud, (n_z, n_r) for an
    axial plane, (n_x, n_y) per z for a transverse plane, (n_z, n_x, n_y) for a volume."""

    grid_axes: Mapping[str, np.ndarray] = field(default_factory=dict)
    """The coordinate values of each grid axis, in metres, keyed "x"/"y"/"z". Per (candidate,
    sample) where the extent is fitted to the geometry."""

    occluder_transmission: np.ndarray | None = None
    """(candidate, sample, point) — the fraction the receiver's shadow let through, when
    occluders="receiver". Emitted separately from the intensity so the unobstructed field and the
    shadow are both answerable from one solve rather than two."""

    enclosed_fraction: np.ndarray | None = None
    """(candidate, sample, point) — at each range, the fraction of that plane's power inside |r|.

    The funnel, as DATA. The pre-SPEC engine returned NaN-separated contour polylines from a physics
    module; the verdict was to emit the field the contours come from and let the viewer contour it
    (§8 item 15, and §7 permits contouring). It is the STRIP enclosed fraction along the sampling's
    own transverse axis, normalised to each plane's total on the requested window — so a beam wider
    than the window never reaches a high level instead of being renormalised into one — and it is
    computed BEFORE the occluder, so what the receiver absorbs shows up as the high-level rings
    closing at the receiver plane rather than being renormalised away.

    None for a 2-D transverse grid, which has no single radial axis to enclose along."""

    aperture_pitch_m: np.ndarray | None = None
    """(candidate, sample, 2) — the aperture-plane sample pitch on each axis."""

    aperture_samples: np.ndarray | None = None
    """(candidate, sample, 2) — the aperture grid size on each axis.

    Per SAMPLE SET rather than per product, because the pitch is derived from the query extent: a
    one-point on-axis query and a 200 m-wide side view of the SAME solve legitimately need different
    aperture grids, and I5 wants the resolution each one actually ran at."""


@dataclass(frozen=True)
class Field:
    """P5 — the one solve. Intensity in absolute units wherever the spec asked for it."""

    sets: Mapping[str, FieldSampleSet]
    method: str
    """The solver route. One name, because there is one route (D10): a direct Fresnel transform from
    the anisotropically-sampled aperture, at the precision the tier sets."""

    unresolved_reason: Mapping[str, str] = field(default_factory=dict)
    """Why the method declines where it declines, by name, so an unresolved region is a stated
    limit rather than a mystery: "sub_fresnel" (closer to the aperture than the paraxial
    approximation holds), "aperture_unsamplable" (the focusing phase needs a finer pitch than the
    cap allows), "outside_alias_free_window"."""


# --- P6 capture --------------------------------------------------------------------------


@dataclass(frozen=True)
class Capture:
    """P6 — what fraction of radiated power the airframe intercepted. Derived from P5 × P3."""

    inputs: tuple[str, ...]
    """The products this was derived from: ("field", "silhouette") for the overlap integral, or
    ("geometry", "silhouette") for the closed-form tier, which does not touch the field at all
    and is therefore a differently-derived product rather than a cheaper version of the same one."""

    method: str
    """"overlap_integral" or "closed_form_goubau"."""

    geometric: np.ndarray
    """(candidate, sample) — the bare overlap: intensity integrated over the projected area."""

    incidence_weighted: np.ndarray
    """(candidate, sample) — the same integral with the rectenna's angular response applied. This
    is the number the efficiency chain uses."""

    peak_areal_density_per_m2: np.ndarray
    """(candidate, sample) — peak intensity at the receiver per unit radiated power, in 1/m².
    Multiplied by the beam power crossing the aircraft it gives the skin power density."""

    equivalent_disc_radius_m: np.ndarray
    """(candidate, sample) — radius of the uniform disc that would carry all the power at the
    observed peak, i.e. one length scale for a spot of any shape. A reported measure, not a
    Gaussian waist and not used in any chain."""

    density_cap_exceeded: np.ndarray
    """(candidate, sample) bool — the skin power-density cap could not be met at any allowed
    defocus. Surfaced rather than blocking, and deliberately not folded into feasibility."""

    skin_power_density_w_m2: np.ndarray
    """(candidate, sample) — the peak density actually on the skin, in absolute units."""


# --- P7 chain ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Chain:
    """P7 — the efficiency factors whose product is end-to-end η, per sample, as named numeric
    data. Derived.

    The pre-SPEC engine returned display label strings from a hand-maintained mirror of the chain
    that "has to be mirrored" on every change (§8 item 3); the names here are data, the order is
    the order they apply in, and the product identity in `eta_end_to_end` is checked by I6.
    """

    inputs: tuple[str, ...]
    factor_names: tuple[str, ...]
    """In the order they apply, station side first."""

    factors: np.ndarray
    """(candidate, factor, sample) — each factor, dimensionless, in (0, 1]."""

    eta_end_to_end: np.ndarray
    """(candidate, sample) — the product of the factors. Grid busbar to rectenna terminals."""

    feasible: np.ndarray
    """(candidate, sample) bool — not horizon-blocked and η above a floor. The power-density flag
    is deliberately NOT part of this."""


# --- P8 power ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Power:
    """P8 — power at each named reference plane, per sample. Derived. One conversion, one place.

    `rectenna_terminals` is onboard DC out of the rectenna, BEFORE the motor and drivetrain — the
    same plane P12 states the airframe's requirement at. Stating both in the schema is what makes
    the drivetrain double-count structurally impossible rather than warned against in a comment
    (D5).
    """

    inputs: tuple[str, ...]
    reference_planes: tuple[str, ...]
    grid_w: np.ndarray
    """(candidate, sample) — what the station draws from the grid."""

    radiated_w: np.ndarray
    """(candidate, leg, sample) — RF leaving each leg's emitting aperture. This is what P5's
    absolute intensity is scaled by."""

    at_receiver_w: np.ndarray
    """(candidate, sample) — RF intercepted by the airframe, before RF→DC."""

    rectenna_terminals_w: np.ndarray
    """(candidate, sample) — onboard DC out of the rectenna."""

    waste_w: np.ndarray
    """(candidate, sample) — grid minus delivered, i.e. everything the chain lost."""

    rating_basis: str = ""
    """How the absolute scale was set: "delivered_target_at_best_sample" holds the grid draw at
    the value that meets the spec's delivered target at the best sample of the trajectory, which
    is what a station with one rated draw actually does; "radiated_fixed" holds radiated power."""

    rated_grid_w: np.ndarray | None = None
    """(candidate,) — the single rated grid draw the basis above produced."""


# --- P9 aggregates -----------------------------------------------------------------------


@dataclass(frozen=True)
class Aggregate:
    """One named trajectory-level reduction and the definition that produced it."""

    definition: AggregateDefinition
    value: np.ndarray
    """(candidate,) — the reduced value per candidate."""

    served: np.ndarray
    """(candidate, sample) bool — which samples the serve policy admitted. Emitted so the window
    a number was taken over is answerable from the product, not only describable in prose."""


@dataclass(frozen=True)
class Aggregates:
    """P9 — named trajectory-level reductions, each carrying its definition.

    This is where pass efficiency and delivered energy live, and where the two orphaned
    definitions of §1 become two named products that can both be right.
    """

    entries: Mapping[str, Aggregate]


# --- P10 atmosphere ----------------------------------------------------------------------


@dataclass(frozen=True)
class Atmosphere:
    """P10 — the path attenuation actually applied per sample, and the spectrum it came from."""

    inputs: tuple[str, ...]
    loss_db: np.ndarray
    """(candidate, leg, sample) — total one-way loss along the leg."""

    transmittance: np.ndarray
    """(candidate, leg, sample) — the surviving power fraction, horizon blockage included."""

    gaseous_db: np.ndarray
    """(candidate, leg, sample) — the gaseous share, integrated over the altitudes crossed."""

    rain_db: np.ndarray
    cloud_db: np.ndarray

    spectrum_frequency_ghz: np.ndarray
    """(n_freq,) — the frequency grid of the profile the loss was interpolated from."""

    spectrum_zenith_loss_db: np.ndarray
    """(candidate, n_freq) — zenith loss from the ground to the trajectory's altitude against
    frequency, under this scenario's weather. The spectrum the per-sample numbers came from, so a
    figure about frequency choice reads a product rather than re-deriving the table."""

    profile_source: str = ""
    """Which committed table this is, and its content hash."""


# --- P11–P13 the flight products (D5) ----------------------------------------------------


@dataclass(frozen=True)
class Mass:
    """P11 — what the aircraft weighs and where the weight is.

    The converged mass cascade. `rectenna_t` is DERIVED from P3's silhouette area and the
    declared areal density, not assumed: the fixed 2 t the pre-SPEC cascade carried is what cut
    the apparent cross-domain fixed point, and it stays cut, because rectenna mass follows from
    geometry rather than from the link's η (D5).
    """

    inputs: tuple[str, ...]
    payload_t: np.ndarray
    structure_t: np.ndarray
    systems_t: np.ndarray
    motor_t: np.ndarray
    reserve_battery_t: np.ndarray
    rectenna_t: np.ndarray
    mtow_t: np.ndarray
    """(candidate,) each. Mass is a design property, so there is no sample axis."""

    empty_fraction: np.ndarray
    iterations: int = 0
    """The lockstep iteration count the fidelity config set. The cascade is a fixed point; it runs
    a fixed number of steps for every candidate rather than breaking on a per-candidate tolerance,
    because convergence is geometric and a data-dependent early exit does not vectorise (I9, D5)."""

    residual_kg: np.ndarray | None = None
    """(candidate,) — how much MTOW moved on the last iteration, so the fixed count is auditable."""


@dataclass(frozen=True)
class FlightPower:
    """P12 — what it costs to fly the profile, quoted at a named reference plane."""

    inputs: tuple[str, ...]
    reference_plane: str
    """Always "rectenna_terminals" — see the note on P8."""

    cruise_shaft_w: np.ndarray
    """(candidate,) — propulsive power for level flight at the design speed."""

    peak_shaft_w: np.ndarray
    """(candidate,) — cruise plus the potential-energy rate of the design climb."""

    required_w: np.ndarray
    """(candidate, sample) — power required AT THE RECTENNA TERMINALS to fly this sample of the
    trajectory, i.e. shaft power over the drivetrain efficiency. Composes directly with P8's
    `rectenna_terminals_w` because both are quoted at the same plane."""

    peak_required_w: np.ndarray
    """(candidate,) — the same for the climb case."""

    lift_to_drag: np.ndarray


@dataclass(frozen=True)
class Reserve:
    """P13 — how far it gets unpowered, and what the reserve mass in P11 actually buys."""

    inputs: tuple[str, ...]
    glide_range_km: np.ndarray
    """(candidate,) — from the beam-dropout altitude at the bare airframe's lift-to-drag: the
    propulsion-integration bonus is not available with the beam off."""

    glide_duration_s: np.ndarray
    battery_duration_s: np.ndarray
    """(candidate,) — how long the reserve battery holds cruise power."""

    powered_fraction: np.ndarray
    """(candidate,) — the share of a climb-then-glide cycle that needs the beam. Independent of
    the height traded, and the same in time as in distance."""


# --- the bundle --------------------------------------------------------------------------


@dataclass(frozen=True)
class Products:
    """Everything one run produced. A product is None when it was not produced, and the manifest
    says whether that was a skip (and why) or a failure (and why).

    A consumer handles absence as a first-class rendering — "not generated in this run" — rather
    than as an exception (§7).
    """

    trajectory: Trajectory | None = None
    geometry: LinkGeometry | None = None
    silhouette: Silhouette | None = None
    aperture: Aperture | None = None
    field: Field | None = None
    capture: Capture | None = None
    chain: Chain | None = None
    power: Power | None = None
    aggregates: Aggregates | None = None
    atmosphere: Atmosphere | None = None
    mass: Mass | None = None
    flight_power: FlightPower | None = None
    reserve: Reserve | None = None

    def get(self, name: str):
        return getattr(self, name)

    def produced(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.__dataclass_fields__.values()
                     if getattr(self, f.name) is not None)


PRODUCT_TYPES = {
    "trajectory": Trajectory, "geometry": LinkGeometry, "silhouette": Silhouette,
    "aperture": Aperture, "field": Field, "capture": Capture, "chain": Chain, "power": Power,
    "aggregates": Aggregates, "atmosphere": Atmosphere, "mass": Mass,
    "flight_power": FlightPower, "reserve": Reserve,
}
