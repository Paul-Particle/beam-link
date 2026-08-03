# beamlink — physics engine spec

Status: **implemented, ready for review as a finished version** (§11). This document is both the target
shape and the record of every decision behind it — D1–D6 were taken before any code was written, D7–D14
are the ones the implementation forced, and §8's backlog now carries what was actually done to each
item. §12 keeps the pre-implementation measurements and adds the ones taken since.

The pre-SPEC modules are in `archive/pre-spec/` rather than deleted, so the two can be compared line by
line where a number is in question.

## 1. What this is for

The engine is the first-class citizen. It answers physical questions about beamed power to an
aircraft, emits the answers as data, and knows nothing about who reads them. Visualization and
economics are consumers: they select from what a run produced and display or cost it. Neither may
compute physics, and neither owns the shape of the data.

The failure mode this exists to prevent: **a consumer performing a second modelling pass and
presenting it as the model.** A figure that says "this is how the physics went" while the technical
and economic outputs were computed differently is worse than a missing figure, because it is
convincing. Two instances were measured in the current code:

- Two definitions of "pass efficiency" — `sweeps.flyby`'s `energy_eta` (which econ consumed) and
  panel D's own curve. They agree to 1.01× in clear air at 200 km station spacing and diverge to
  **0.32×** in a storm at 400 km (4.99 % shown against 15.45 % used). Neither recorded its span,
  serve policy, statistic or sampling.
- Two independent solves of the same field. `capture.overlap_capture` solves the focal plane on a
  2-D grid; `flyby_frames.channel_field` solves the whole path with 1-D axial cuts, and the aircraft
  plane lies inside the path. They agree on peak intensity to **15 %**, held together only by a
  tolerance chosen when the comparison was first written.

Neither was a wrong published number. Both were one scenario away from being one.

## 2. The boundary

Inside the engine: the physics, its numerics, its reference implementations, and the emission of
products. Outside: everything that reads products.

There is **no allowlist and no exempted module.** The two cases that looked like they needed one
resolve by classification instead:

- `field` / `chain_reference` (Hankel-transform ground truth) are a *second implementation on
  purpose*, checked against the primary by the engine's own tests. They are engine-internal. Being
  inside the boundary is not an exemption from it.
- A "pure focusing sweep, no aircraft, no atmosphere" figure does not need privileged access to
  `diffraction`. It needs a run whose scenario has **no receiver** — see §5, P5. The field product
  exists, the capture and silhouette products do not, and the viewer reports them as not produced.

Enforced by test, not convention (I8 below).

## 3. Invariants

| # | invariant |
|---|---|
| I1 | **One solve per physical quantity per run.** Two code paths may not both compute the same field. Sampling one solve at different locations is not two solves. |
| I2 | **No quantity appears under two names or two units.** No convenience aliases for consumers that would rather not branch. |
| I3 | **Derivation is owned and recorded.** Within a run the engine derives (P6–P9); across runs the driver does (D6). Every derived product names the inputs it came from and carries its full definition (span, weighting, policy, statistic, sampling) as data. A derived product that cannot name its inputs is illegal. |
| I4 | **Consumers never derive.** They select, display, and cost. A consumer needing a different aggregate is a request for a new named engine product, not a local calculation. |
| I5 | **Fidelity is part of product identity.** Same shape, same units, same semantics across tiers; method and resolution recorded per product. Mixing tiers within one presented result must be detectable. |
| I6 | **Every derived product is recompute-checkable.** CI recomputes it from its primary and asserts agreement to a stated tolerance. This is what makes derived products safe rather than redundant. |
| I7 | **Products carry physics, not presentation.** No label strings, no colours, no 0–1 normalised axes, no per-frame peak normalisation, no NaN-separated polylines, no layout or "how far to draw" arguments, no precision reduced for file size. |
| I8 | **The import firewall is a test.** Consumers importing any physics module fails the suite. No exemptions. |
| I9 | **The engine is vectorized over samples *and* over scenario candidates.** A batched call is the normal call; a single scenario is a batch of one. Data-dependent iteration runs lockstep with a config-set step count rather than branching per sample. Working memory is bounded by config-set chunking. See §6a. |

Redundancy is not the enemy; **anonymity** is. A run may legitimately contain two efficiency figures
provided each is named and defined. Two numbers both called "the pass efficiency" is the defect.

## 4. Run model

One `RunSpec` in, one run directory out.

    RunSpec:
      scenario(s)     — station/relay geometry, frequency, aperture, weather, attitude policy
      trajectory      — sampled positions AND attitudes with times (an input object, not a
                        convention; see §8 item 21)
      products        — which products to emit, with per-product sampling requests
      fidelity        — per-product tier
      power spec      — delivered target or radiated power, for the absolute-units products

    runs/<id>/
      manifest.json   — the spec verbatim, engine git sha, per-product fidelity + method +
                        resolution, and for every requested product: produced | skipped(reason) |
                        failed(reason)
      <product>.npz   — arrays, SI units, full precision

The manifest is the contract. A consumer pointed at a run learns from it alone what exists, at what
fidelity, and what was asked for but not produced.

A **study** is the same idea one level up (D6): `studies/<id>/manifest.json` names the runs it
aggregated and the aggregation definition applied, with its results alongside in the same array format.
Consumers load a study through the same `artifacts/` loaders they load a run through.

## 5. Product catalogue

Each product states the physical question it answers. A field whose only justification is a figure
does not belong in a product. Products are independent (I1/I2): none republishes another's content.

- **P1 `trajectory`** — where and how the aircraft was, sample by sample: position, attitude, time.
  The spec's trajectory echoed as data so every other product can be indexed against it.
- **P2 `geometry`** — what the link geometry was per sample: slant range, arrival direction in the
  airframe body frame, elevation. *(sign convention: §9 D1)*
- **P3 `silhouette`** — what surface the capture integral summed over: the mesh as integrated, plus
  per-sample projected area and per-element incidence. Emitting the integrated mesh is what lets a
  viewer draw exactly the surface used, instead of a lookalike.
- **P4 `aperture`** — what the emitter did per sample: semi-axes, taper exponent, focus range, and
  on request the aperture phase as a field over the aperture.
- **P5 `field`** — the one solve. Intensity in absolute units at the sample locations the spec asks
  for — silhouette elements, named planes, or a volume — each with a **validity mask** (regions the
  method could not resolve are reported as unresolved, never filled to make a picture whole).
  Options: occluders present or absent (this is where the receiver's shadow lives — same shape, same
  units, an option on the solve rather than a product of its own).
- **P6 `capture`** — what fraction of radiated power the airframe intercepted. Derived from P5 × P3;
  geometric and incidence-weighted, and the peak areal density.
- **P7 `chain`** — the efficiency factors whose product is end-to-end η, per sample, as named
  numeric data. Derived.
- **P8 `power`** — radiated, at-receiver, delivered and grid power per sample for the spec's power
  target. Derived. One conversion, one place.
- **P9 `aggregates`** — named trajectory-level reductions, each carrying its definition: span, serve
  policy, statistic, sampling. Derived from P7/P8. This is where pass efficiency and delivered
  energy live, and where the two orphaned definitions in §1 become two named products.
- **P10 `atmosphere`** — the path attenuation actually applied per sample, and the spectrum it came
  from.

Scenarios without a receiver are legal: P5 and P4 exist, P3/P6/P7 do not, and consumers see that in
the manifest. That is the general form of the "pure focusing sweep" figure.

Flight-mechanics products (`physics.airframe`, per D5). They share the aircraft geometry the link
products index against, and are indexed the same way, but depend on nothing in `beam_link`:

- **P11 `mass`** — what the aircraft weighs and where the weight is: the converged mass cascade
  (payload, structure, systems, motors, reserve, rectenna) and MTOW.
- **P12 `flight_power`** — what it costs to fly the profile: cruise and peak shaft power, and the
  power required **at a named reference plane** (rectenna terminals), so it composes with P8 without
  double-counting the drivetrain.
- **P13 `reserve`** — how far it gets unpowered: glide range from the beam-dropout altitude, and the
  buffer the reserve mass in P11 actually buys.

## 6. Fidelity

### 6a. Vectorization

This is expected to be the single largest lever on both run time and iteration speed, and it is a
capability the driver and econ layers are entitled to rely on: a sweep evaluates as one batched call,
and a search algorithm evaluates many candidate next steps at once.

**Today it is a loop wearing a vectorized interface.** `evaluate`'s own docstring: *"An (n, 3)
`plane_pos` is looped over and the results stacked, so every numeric entry becomes an array of length
n."* That is why a pass costs 56–95 ms per sample. The encouraging part is that the expensive
internals are already batch-shaped — `_axial_cuts` batches its FFT stack over propagation planes,
`_overlap_taper_grid` batches over taper exponents — they are simply re-entered once per sample, so
the batching is spent on the wrong axis. The work is to make (candidates × samples) the batch axis
those kernels see, not to invent batching.

Consequences, binding on the design:

- **The scenario spec is batchable.** Design parameters may be arrays; products carry a leading
  candidate axis; a run directory records that axis and the per-candidate parameter values. A single
  scenario is a candidate axis of length 1, so there is one code path, not two.
- **Data-dependent iteration goes lockstep.** The adaptive taper search already evaluates a full grid
  and needs no change. The density-cap defocus bisection currently early-exits per sample; vectorized,
  it runs a fixed number of bisection steps for every sample, with the count set by the fidelity
  config as a tolerance. Marginally more arithmetic per sample, uniform cost, and it vectorizes.
- **Ragged inner axes need a padding convention.** The silhouette element cloud has a different size
  at every look angle, so batching across samples means padding plus a validity mask — the same
  device P5 already uses for unresolved field regions, applied in the schema generally.
- **Working memory is the binding resource, not output size.** Chunk sizes come from config (the
  existing heuristic in `_axial_cuts` generalises), so a batch that would not fit is computed in
  blocks rather than refused.
- **The silhouette raster is the one kernel that does not vectorize for free** — it is a z-buffer loop
  over mesh triangles, and the dominant cost. Batching it over look angles is a real piece of work;
  the fidelity ladder's coarse-mesh and closed-form tiers (§6) are the alternatives, and the existing
  convex silhouette table is the vectorized-but-occlusion-free path.

Econ is expected to be vectorized in the same shape, which makes it an obligation on the artifacts
layer: loaders expose arrays with named axes (candidate, sample, …), never per-sample objects that
force a consumer to loop.

**What it cost, measured after the fact.** A warm 81-sample pass at `standard` runs at **18 ms per
sample** against the 56–95 ms above, and the batching is only part of it. Three changes, in order of
what they were worth:

- **Anisotropic aperture sampling.** The pitch is set per axis, so each axis is sampled at its own
  scale and the aperture grid is ~2.2·N on both regardless of aspect: n = 624 → 80 for an 8:1 aperture,
  which is 20× fewer aperture samples on exactly the low-elevation geometries a pass spends most of its
  time at. A common pitch had to use the minor axis's fine spacing across the major axis's full length.
- **The focusing phase folded into the transform kernels.** exp(iπ/λ·inv·(x²+y²)) factorises as
  p(x)·p(y), so it multiplies the two O(n·n_out) kernels instead of the O(n²) aperture array — and the
  aperture amplitude then stays real and constant across a whole defocus search, so a 12-step bisection
  builds it once. The pre-SPEC near-field cut used this factorisation as a special case for on-axis
  cuts; here it is how every evaluation works.
- **The taper grid as a loop rather than a second batch axis** (D11), which is the one place the
  obvious reading of this section was measurably wrong.

Sizing the aperture grid to the APERTURE rather than to a transform window is what makes the first of
those available at all. The pre-SPEC path carried an `n_ap = 512` floor at a pitch of min_semi/12,
putting a 100 m aperture inside a grid 21 diameters wide — a 512² array of which ~450 samples were
illuminated — because the size was inherited from a single-FFT observation window it did not use.

**Run time against accuracy is the only trade axis, and every point on it is set from the config.**
Nothing else — output size in particular — is a design constraint: the one place size ever mattered
was the 15 MB cap on a published phone artifact, which is a property of that publishing route and not
of the engine. A product is emitted at full precision in SI units and that is the end of it.

Tiers vary method and resolution; they never vary shape, units or meaning (I5). Levers, in measured
order of value:

- capture mesh face count — **8× on the dominant cost for 0.34 pp of capture** (5704 → 612 faces)
- closed-form capture (Goubau τ, as the relay up-leg already uses) instead of the overlap integral —
  **0.17 ms against 21–26 ms**, biased high by 0.4–4.8 pp, one-sided
- taper search resolution — ×1.5, worth up to 4.8 pp of capture, so not free
- density-cap solve tolerance — free where the cap does not bind, ×3.5 where it does
- field sampling density and extent

Measured non-levers, recorded so they are not tried again: the silhouette raster's output grid
(200 → 64 buys 25 % and adds ±1 pp of resampling noise), and the capture integral's own transform
grid (stable to <0.001 pp from n_grid 256 to 1536).

**Re-measured on the implementation** (`drivers.calibrate`, published in the README). Three of those
claims changed shape, and the reasons are worth having:

- the **mesh face count** is no longer a run-time lever at all, because the cost it drove was paying
  for silhouette rasters and those are now cached across runs. 5704 → 824 faces costs 0.2 % of the
  aggregates and buys nothing on a warm pass; the 8× is a one-off on the first run of a new loft, which
  belongs to the cache and not to the tier. It stays in the config because that first run still matters.
- **closed-form capture** is confirmed one-sided and biased high, at 5.2 pp of η on the clear reference
  case for 78×. What was not in the original measurement: it **cannot defocus**, so where the skin
  power-density cap binds its error roughly doubles — 7.6 % → 13.2 % on pass efficiency at 60 MW
  delivered. That is a limit on what the tier can ANSWER, not a rounding difference, which is why P6
  names its inputs differently at that tier (I5).
- the **taper search** is the one real accuracy lever left: 17 → 5 points is 0.9–1.3 % on the
  aggregates for 2.7–3.3×.
- and one new non-lever: **single precision** is 5 % of run time for no measurable error, because the
  cost is in the transforms' memory traffic rather than their arithmetic.

Which leaves the honest summary: after the vectorization work, `standard` and `full` differ by 3–4× and
0.1 pp, so the tier ladder matters much less than it did — the speed came from the sampling being right
rather than from being allowed to be wrong.

Free of any fidelity cost, and therefore first: a persistent on-disk raster cache (77 ms → ~1 ms per
look angle) and a coarser look-angle bin (0.1° → ~0.5°, so a pass reuses ~40 rasters instead of 219).

**Calibration is part of the deliverable.** A fixed grid of scenarios run at every tier against
`full`, reporting error in η and in the trajectory-level aggregates econ consumes, published in the
README. A tier without a published error bar is not usable for a headline number.

## 7. Consumer contract

Consumers import one read module exposing frozen dataclasses (`load(run_dir)`), and nothing else.
They must:

- handle any product being absent, and say so on the page — "not generated in this run" is a
  first-class rendering, not an exception;
- stamp provenance visibly: run id, engine sha, and the fidelity of each product shown, in the
  margin, from the manifest rather than hand-written per figure;
- refuse loudly when pointed at a run lacking what they were told to show, rather than substituting
  or recomputing;
- do no arithmetic beyond selection, unit presentation, and contouring/scaling of emitted fields.

Econ is rewritten from scratch against this contract; nothing in the current econ constrains the
design. Visualization is rebuilt against it too, and is expected to be broken in the interim.

## 8. Backlog: what to cut, rework, or promote

From the audit of 2026-08-03. Verdicts are against §3 and §5. **All 22 items are closed**; the
"outcome" column says how, and the two that were not simply done as written are 16 and 18 — one stayed
parked and one was retired by D3, both as the verdict called for.

| # | outcome |
|---|---|
| 1 | both cut. No rounding and no peak normalisation anywhere; `test_the_field_is_absolute_and_not_peak_normalised` asserts it |
| 2 | gone. There is no "with field" return mode: the field is P5, and the sample set P6 integrated is the one P5 emits |
| 3 | cut. P7 emits `factor_names` as data and the viewer's waterfall reads them, so there is no mirror to keep |
| 4, 5 | moved out. `physics/` imports no plotting library — the firewall test asserts it — and the z-buffer takes no `color` |
| 6 | holds by construction. One cached mesh serves the integral and P3, so "displayed == integrated" is not a cost any more |
| 7, 8, 9 | cut. No camera, no pre-projected positions, no which-way-up flag. P4 emits the aperture frame in the earth frame and the aperture phase as a field |
| 10 | cut. The relay's factors are named per leg; nothing is republished under a generic key |
| 11 | reworked. `FieldSampling(kind="axial_plane", axis=...)` names a physical plane; no layout argument exists |
| 12 | reworked into P5, absolute and masked |
| 13 | reworked. The default cuts are the APERTURE's principal planes (D2), and there is one frame in the repo, defined in P3 |
| 14 | reworked. The sub-Fresnel region is `valid=False` with a stated reason; nothing is filled |
| 15 | cut, and replaced: P5 emits `enclosed_fraction` and the viewer contours it with `contourpy` |
| 16 | stays parked, as the verdict said. `PUBLIC_EXPOSURE_LIMIT_W_M2` lives in `viz/panels.py` and is drawn as an overlay on P5's absolute field; no product knows it exists |
| 17 | reworked. A 3-D occluder option on the solve, parameterised by the receiver's own extents, applied to any sampling |
| 18 | retired by D3, as the verdict allowed. "Best fixed aperture aspect" is `drivers.sweep(base, "aperture.fixed_aspect", …)` against a named aggregate |
| 19 | kept. `Silhouette.max_half_extent_m`, cached per loft |
| 20 | reworked. The serve floor is `ProductRequest(name="aggregates", serve_eta_floor=…)`, recorded in every definition that applies it, and `speed_ms` is now the trajectory's own times |
| 21 | reworked into P1. `TrajectorySpec` carries positions, attitudes AND times, and every aggregate's span names it |
| 22 | dead with the econ rewrite. There is no beam efficiency constant in `econ/` at all |

**Presentation inside the physics modules**

| # | item | verdict |
|---|---|---|
| 1 | `diffraction._finalize_cut` rounds output "for compact serialization into the animation's HTML"; `normalize=True` (peak-normalise) is the default | cut both — I7 |
| 2 | `beam_link.evaluate_with_field` exists so "the animation can draw panel A without a second FFT" | rework — the field is P5, not a return mode |
| 3 | `beam_link.waterfall` returns display label strings from a hand-maintained mirror of the chain that "has to be mirrored" on every change | cut — P7 is numeric |
| 4 | `bwb_model.silhouette_figure` imports plotly inside a physics module | move out |
| 5 | `bwb_model.rasterize_view_zbuffer(color=…)` is a renderer, used only by the animation | move out |
| 6 | `get_mesh` resolution set by a figure requirement (2.6× cost to keep displayed and integrated silhouettes identical) | rework — P3 emits the integrated mesh, so the property holds by construction |

**The de-facto product (`flyby_frames.frame_physics`) is a panel payload**

| # | item | verdict |
|---|---|---|
| 7 | `view_u/view_v/view_w` — a display basis rotated so dorsal points up on screen | cut — a camera |
| 8 | `plane_loc`, `emitter_loc` — positions pre-projected into a plotting frame | cut — P1 emits model-frame positions |
| 9 | `phase_d/phase_R/phase_lam/phase_normal_up` — draw-parameters for the zone-plate image, including a which-way-up flag | cut — P4 emits a field |
| 10 | `slant_km` and `slant_m`; relay branch republishing down-leg terms under the direct branch's generic keys "so sweep code can stay architecture-agnostic" | cut — I2 |
| 11 | `beam_cut_axes(frame, cut, portrait=False)` — a geometry function taking a screen-layout argument | rework — emit the physical basis |

**Products that are viz-shaped**

| # | item | verdict |
|---|---|---|
| 12 | `channel_field` — normalised 0–1 plot axis in, "how far to draw" bounds, frame-peak normalisation, display precision, HTML rounding | rework into P5. The physics is general; the shape is not |
| 13 | the F/F₂ two-cut pair — "the silhouette's principal transverse axes in the display view basis" as the sampling | rework into a declared sampling. **§9 D2** |
| 14 | the sub-Fresnel geometric fill, which exists so the image has no hole | rework — P5 emits a validity mask instead |
| 15 | `funnel_contour` returns NaN-separated polylines | cut — emit the enclosed-energy field, contouring is the viewer's |
| 16 | `exposure_boundary` — bakes in a regulatory level and an `edge_margin` plot-window heuristic | cut. Parked; if it returns, the level is a viewer overlay on an emitted field |
| 17 | `_shadow_transmission` parameterised per display cut (`half_major_m`/`half_minor_m`) | rework — a 3-D occluder option on the P5 solve |
| 18 | `_optimal_fixed_aspect` — a design optimisation ("the best one array to build") inside a phone figure | promote to an engine product or cut. **§9 D3** |
| 19 | `max_silhouette_extent` — exists to frame a panel, but is a genuine airframe property | keep |

**Policy and convention inside the physics layer**

| # | item | verdict |
|---|---|---|
| 20 | `sweeps.flyby` owns an operating policy: `min_serve_eta = 0.02` ("not derived from anything, and it sets the headline" — 0 → 0.10 moves `energy_eta` 0.090 → 0.240), `speed_ms`, and two hardcoded accounting conventions | rework — P9 records policy as part of a named aggregate's definition; the policy itself is a declared input |
| 21 | the trajectory is a convention, not an input — three different pass spans in three modules, which is how §1's divergence arose | rework into P1 |
| 22 | `econ.ETA_BEAM = 0.80` reachable as a default (0.55 × 0.80 × 0.80 = 0.352 against the engine's ~0.21, biased in favour of the technology) | dies with the econ rewrite; until then, make the argument required |

**Not defects** — `field`/`chain_reference` as engine-internal ground truth (§2), and the engine's
own transform-grid resolution, measured adequate in §6.

## 9. Decisions taken

Settled 2026-08-03. Recorded here rather than in a transcript, because the engine is to be built to
completion against them and reviewed as a finished version (§11).

- **D1 — elevation is ground-referenced, and defined per link leg.** The elevation of a leg's
  receiving end as seen from its emitting end, above that end's local horizon. Ground→aircraft is
  positive, +90° overhead; relay→aircraft is correctly negative, the aircraft being below a relay's
  horizon. Rationale: every elevation *input* in the repo (`min_elevation_deg`, reference elevations,
  `downrange_for_elevation`) is already ground-referenced and positive, while the emitted
  `elevation_deg` was aircraft-referenced (−65.6° overhead) because `_capture_setup` measures from
  the aircraft outwards — and the relay up-leg, measured at the station, came out positive in the
  same dict. Per-leg definition dissolves that double key instead of renaming it. Nothing is lost on
  the aircraft side: the arrival direction in the airframe frame is P2's (β, α).
- **D2 — sampling geometry is a declared input, and all samplers ship together.** The run spec asks
  for silhouette elements, planes, arbitrary points, or a volume; the engine solves once and
  evaluates wherever asked. No sampling is privileged by the engine. The **default planes are
  physically defined** — the aperture's own principal axes, which set the beam's transverse principal
  axes — not the silhouette's principal axes in a display basis, which is what made the current F/F₂
  pair viz-shaped. Rotational symmetry is only slightly broken, so those axes are the natural cut.
  The volume sampler is built in v1 rather than deferred, so the capability is honest; routine runs
  simply do not request it.
- **D3 — no bespoke design-optimisation product.** A run may sweep a declared design parameter and
  emit the objective per candidate, where the objective is a named aggregate (P9). "Best fixed
  aperture aspect" is then a sweep over `fixed_emitter_aspect` against pass-integrated delivered
  energy; econ's bespoke aperture scan becomes a sweep over diameter against capture at the design
  edge; frequency and taper studies come free. Retires backlog item 18 and the `#(verify)` sizing
  pass in econ together. A sweep is several scenarios in one study, products indexed by scenario —
  the same shape at different parameter values, which is the permitted kind of repetition.
- **D4 — the sweep loop is OUTSIDE the engine, and the engine plugs into any generic one.** The
  engine's unit of work is *one scenario, including its trajectory* → products. Anything that varies
  the scenario is a driver, and must be able to treat the engine as a black-box callable. Three
  consequences, all binding on the design:
  - `evaluate(spec) -> Products` and `write(products, run_dir)` are separate. An optimiser calls the
    first and reads its objective straight off the emitted aggregate; the archival path calls both.
    One physics path, two uses — not two paths.
  - **No process-global state may affect a result.** Caches (the raster cache, atmosphere tables) are
    correctness-neutral, explicitly keyed, and safe to share or persist across parallel calls.
  - Trajectory-level reduction stays inside the unit of work (P9), because it is a definition-bearing
    derived product. Only *scenario variation* moves out.

### Consequences for the firewall

The firewall is enforced by **package layout**, not by naming exceptions, so it has no holes and a
new file inherits the rule from its directory:

    artifacts/            — the data contract: product dataclasses, load and save. Imports
                            NOTHING else in the repo.
    physics/              — the engine. Imports artifacts/ (to build and write products).
      geo.py              — earth and ENU geometry, shared by both domains
      airframe/           — the aircraft: loft, mesh and silhouette table; flight power and
                            the mass cascade
      beam_link/          — the link: chain, capture, diffraction, closed-form relations,
                            atmosphere; and reference/ (Hankel ground truth, tests only)
    viz/                  — may import artifacts/ only
    econ/                 — may import artifacts/ only
    drivers/              — may invoke physics/ AND econ/; may not reimplement any part of either

Two placements are worth justifying because neither is the obvious one:

- **The product dataclasses live in `artifacts/`, not inside `physics/`.** Physics still *owns* what
  is in them — it decides the products and their fields — but if the definitions lived under
  `physics/`, every consumer would have to import `physics/` to read a run, and the firewall would be
  a fiction. A package with no repo dependencies and no physics in it is the data contract, and the
  import rule stays structural: nothing outside `physics/` and `drivers/` may import `physics/`, and
  `artifacts/` may import nothing from the repo at all. One definition, no hole.
- **Aircraft geometry belongs to `airframe/`, and `beam_link/` imports it.** The BWB loft, mesh and
  silhouette table describe the aircraft, and both domains need them: the link to integrate capture
  over the silhouette, the airframe to size a rectenna. The dependency runs one way — link depends on
  aircraft, never the reverse.

(`physics.beam_link` inside a package called `beamlink` is mildly redundant; `physics.link` reads
better. Taking the names as given unless you'd rather have the shorter one.)

A driver that calls `engine.evaluate()` computes no physics itself, so this is a classification, not
an exemption — the same resolution as the Hankel ground truth in §2. Viz and econ never appear on the
permitted side of any rule.

**The driver owns sweeps, optimisation, and the orchestration between them.** It is the only layer
that sees both sides, and it has to: an economic objective cannot be searched for in physics space
without evaluating econ, and exhaustive sweeps are affordable only in the simplest cases, so anything
else needs a search loop that alternates between asking the engine a physical question and asking econ
what the answer is worth. Both modes are first-class — sweep *and* search — which is why I9 matters:
a search that can only propose one candidate at a time wastes the batching that makes the search
affordable.

A driver is therefore allowed to be **specific**. Its workflow is dictated by the particular interplay
of a given econ objective and the physics it depends on, and it takes that specificity through its
config. This is the one place in the repo where "oddly specific" is the correct shape, and it is why
the driver is written last: its structure is a consequence of the finished engine and econ, not an
input to either.

### D5 — the airframe is inside the engine

`physics.airframe` alongside `physics.beam_link`: cruise and climb power, the mass cascade, glide
reserve, lift-to-drag, and the aircraft's own geometry. It is physics, so econ may not compute it —
`econ_pass2` currently takes `CRUISE_MW` straight out of `airframe.mass_buildup`, which the firewall
forbids. §5 gains flight products (P11–P13). Three consequences fall out of the ruling:

- **The two domains stay independent, coupled only through the spec.** Neither imports the other; both
  use the shared aircraft geometry. The apparent cross-domain fixed point — delivered power sets what
  the aircraft can do, aircraft mass sets the power required — is *cut today by a hardcoded
  `rectenna_mass_t = 2.0`*, and it should stay cut, because rectenna mass follows from silhouette area
  and areal density (geometry) rather than from the link's η. Closing the remaining scenario-level
  condition — "the aperture and frequency that deliver the power this aircraft needs" — is a search
  over scenarios, which D4 already puts in the driver. No engine-internal cross-domain solve, so I1
  stays clean and either domain remains usable on its own.
- **`rectenna_mass_t` becomes derived rather than assumed.** A gain the ruling enables: silhouette area
  is already a product (P3), so the 2 t constant becomes area × areal density, with the areal density
  as the declared input it actually is.
- **`mass_buildup`'s fixed point must go lockstep** (I9). It is currently a scalar loop with a
  data-dependent early exit (`max_iter=200`, `tol=1 kg`, `if converged: break`). Vectorized over
  candidates it runs a fixed iteration count set by the fidelity config; convergence is geometric, so
  this costs little.

**And one bookkeeping hazard the ruling exposes.** `airframe.mass_buildup` folds the onboard
motor/drivetrain (`eta_chain=0.85`) into `cruise_mw`, and `econ.delivery_efficiency` carries a comment
warning not to double-count it. A number whose measurement point lives in a comment is the same defect
as an unnamed derived quantity. Both domains must state their **power reference point** in the schema —
the airframe's requirement and the link's delivered power quoted at the same named plane (rectenna
terminals) — so the double count is structurally impossible rather than warned against.

### D6 — cross-run aggregation is the driver's

Route- and corridor-level numbers — energy per flight over a route served by N stations — span several
engine runs, and they are exactly what econ consumes. The driver owns them, which follows from D4
(it already owns cross-scenario work) and is now an explicit ruling rather than an inference.

It uses the same reduction machinery and records the same way: a **study manifest** naming the runs it
aggregated and the aggregation definition it applied. So "who derived this number and how" is
answerable for a corridor figure exactly as it is for a pass figure, and econ still only selects.

This is why I3 is worded as *derivation is owned and recorded* rather than "only the engine derives":
the engine derives within a run, the driver derives across runs, and a consumer never derives at all.
The line is not who is allowed to do arithmetic — it is that whoever does it is named in the output.

### D7 — one run is one architecture

A candidate axis may not switch between the direct and relay architectures, and may not add or remove
the receiver. Two reasons, and the second is the real one: the leg-indexed products (P2, P8, P10)
would acquire a ragged leg axis, and *direct* and *relay* are not one model at different parameter
values — they are different models. Comparing them is a study over two runs, which D6 already makes
the driver's job. The engine raises rather than producing a run whose per-candidate product set
differs.

Weather is not in this class and stays a run axis by convention rather than by rule: it changes no
shape, so a candidate axis over weather would work. It is a run axis in the reference driver because a
route does not sit under one sky, and that is a statement about corridors, not about the engine.

### D8 — the field solve's tier is one per run

`field` and `capture` share a fidelity tier by construction, because P6 is a sampling of P5. A
per-product override on either would split one solve into two at different methods — which I1 forbids
outright, and which I5 could not make detectable, since both would be labelled with the product they
came out of rather than the solve they came from. `fidelity.resolve` rejects it.

Products with a genuinely independent method may be overridden freely: the silhouette mesh, the mass
fixed point, the atmosphere path integral. The manifest records what each one ran at, so a page
showing two of them at different tiers can say so (it does — see the mixed-tier warning in the run
report).

### D9 — the defocus floor is closed form, not a back-off loop

The density-cap search needs a compliant lower bracket on focus range. The pre-SPEC engine found it by
trying `slant/8` and multiplying by 1.5 until the field model stopped raising `ApertureTooFine` — a
data-dependent loop with an exception as its control flow, which cannot be vectorized. It does not
have to be a search. The binding constraint is the focusing phase on the longer semi-axis, so

    n = 2.2·semi/d  with  d = λ/(4·semi·|1/z − 1/f|)   ⇒   n = 8.8·semi²·|1/z − 1/f| / λ

is monotone in the defocus and inverts directly at `n = n_max`. Two details are load-bearing and were
found by the test rather than by inspection: the budget must be discounted by the pitch ladder's own
rounding (which takes a pitch *down* by up to 19 %) and by the grid quantum, or the closed form
returns a focus that the sampling then declines to represent — the same class of bug as the loop it
replaces, only quieter.

### D10 — one field method; the silhouette sampling's resampling is bounded, not tiered

There is one route from aperture to query point (a direct Fresnel transform), at the precision the
tier sets. The silhouette sampling reads it off a separable grid fitted to the element cloud and
interpolates onto the elements, because an exact scattered transform over ~11 000 elements costs
n_query × n_illuminated and does not scale.

That interpolation is a property of the SAMPLING, not a second solve, and it is not a fidelity tier
either — it is bounded. `physics.recompute.check_field_resampling` re-evaluates the best-captured
sample with `aperture.scattered_intensity`, which is exact at the element positions, and asserts the
resulting capture agrees to 1 %. Measured on the reference case: **2.0e-6 relative**.

### D11 — the taper grid is a loop over a config-set count, not a second batch axis

Folding the taper search into the batch axis is the obvious reading of §6a and measurably the wrong
one. Tiling (member × taper) copies the padded silhouette element cloud once per taper — 127 MB for an
81-sample pass at nine tapers — and the memory traffic costs more than the transforms it was meant to
amortise. Measured: 85.5 ms/sample tiled against 72.5 ms/sample looped, before the other two
optimisations below.

The member batch stays whole and the taper grid is a loop of nine. That is still lockstep: the count
is config-set and every member walks all of it. What §6a actually asks for is that the *expensive
internals* see (candidate × sample) as their batch axis, and they do.

### D12 — P4 and P5 describe the leg that serves the aircraft

A relay's up-leg is aperture-to-aperture and the engine has no field solve for it at all — its capture
is the closed-form Goubau coupling, which is the right model for two circular apertures. So the
aperture and field products describe the down-leg, and a relay run's P5 is the relay's beam.

This is a scope statement, not an oversight, and it is recorded because the pre-SPEC animation *drew*
an up-leg by synthesizing an up-leg aperture inside the figure module — a consumer computing physics,
which is precisely the defect §1 is about. An up-leg field is new products (a leg axis on P4/P5, or a
second named aperture), not a figure's local calculation.

### D13 — the transverse frame's sign is a gauge, and the pointing error's direction is not an input

The silhouette's narrow principal axis rotates by ~90° over a pass — fore-aft belly-on, near-vertical
at the horizon — so any fixed reference used to pin its sign has a locus where the pin flips. That is
not fixable: pinning a direction in a rotating transverse plane always has a degenerate locus. The
frame is therefore a **deterministic gauge**, reproducible from the arrival angle, and not continuous.

Nothing physical may depend on that gauge, and checking it found something that did. The aperture is an
ellipse with even illumination, so the FIELD is gauge-invariant outright — but a *signed* pointing
offset along an arbitrarily-signed axis is not. Measured on this loft at β=60°: the two directions
differ by **7 pp of capture** (0.857 against 0.928), because the projection is not symmetric about its
narrow axis. "The worst case for a given magnitude" was therefore a label on one arbitrary side of two.

The fix makes it well posed. `pointing_error_m` is a MAGNITUDE; the direction comes from the
projection's own third moment (P3 `principal_skew_m3`), which points the beam at the thin side. Skew
flips sign with the gauge, so the same physical offset is chosen either way, and it is demonstrably the
worse of the two directions. Two tests pin both halves.

A viewer that wants a continuously-oriented drawn airframe re-pins along the trajectory from the
emitted per-sample frames. That is a decision about which way up to draw, i.e. presentation, and I7
puts it on the viewer — which is also the honest resolution of the pre-SPEC belly-up figure: the
mirror was never a physics defect, it was a viewer without enough information, and now it has it.

### D14 — the loft is the aircraft's shape; the tessellation is a fidelity parameter

`n_span`/`n_theta` moved out of `LoftSpec` and into `FidelityConfig`. While they sat in the loft, the
mesh face count — §6's largest measured lever — was a scenario input, so changing the tier changed
nothing, and the calibration table reported 5704 faces at every tier including `coarse`. The shape is
physics; how finely it is discretised is numerics; separating them is what makes the lever a lever.

### Process: the suite is pytest, in `tests/`

The pre-SPEC checks were a `python -m` script whose every assertion printed the numbers behind it, and
that property is worth keeping — so the assertion messages carry the numbers. What pytest buys is
parametrisation (the firewall test is one case per module in the package, so a new file that breaks the
rule names itself) and fixtures (one archived reference run shared by every contract test). Added as a
`dev` dependency group; `uv sync` picks it up.

### Nothing open

All decisions needed to start were taken before implementation; D7–D14 are the ones the
implementation forced, recorded here as §11 requires.

## 10. Order of work

1. The two free speedups (§6) — no fidelity cost, and they make everything after this bearable. The
   raster cache becomes explicitly keyed and persistable here, which D4 requires anyway.
2. Package layout of §9, and the import firewall test (I8), before there is anything to firewall.
3. Schema and manifest: write §5 down as dataclasses, with a stated physical question per field.
4. One solve: collapse `overlap_capture` and `channel_field` into a single field solver with
   requested sampling — silhouette elements, planes, points and volume together (D2). This is new
   physics capability, not a move: the engine has no concept of a field away from the aircraft today.
   Largest single piece.
5. `evaluate(spec) -> Products` and `write(products, run_dir)` (D4), P1–P8, the recompute checks of I6.
6. P9 aggregates, with the two definitions from §1 as the first two named products, and their policy
   constants as declared inputs.
7. A minimal reference driver that sweeps a declared parameter (D3) — a demonstration of the plug
   shape, deliberately thin and not privileged.
8. Rebuild viz against the contract; rewrite econ from scratch.

Fidelity tiers land with the schema (I5), not after it.

**All eight are done.** One deviation from the order, worth recording: items 1 and 2 were done
together rather than 1-then-2, because the raster cache lives inside the new layout and building it
twice would have been the only way to keep them apart. The measured effect of item 1 is unchanged —
**93 ms cold to 3 ms from disk** per look angle, and free from the in-process memo after that.

Where the finished code sits against §9's layout:

    artifacts/   spec.py products.py manifest.py store.py serialize.py
    physics/     engine.py aggregate.py recompute.py fidelity.py cache.py geo.py
                 airframe/  loft.py silhouette.py flight.py
                 beam_link/ aperture.py solver.py occlusion.py capture.py chain.py
                            atmosphere.py relations.py  reference/ hankel.py chain_reference.py
    viz/         run_report.py panels.py provenance.py
    econ/        cost.py verdict.py
    drivers/     sweep.py corridor.py calibrate.py
    tests/       test_physics.py test_contract.py test_import_firewall.py

`physics.beam_link` was kept over the shorter `physics.link`, as §9 offered — the recorded layout is
worth more than the two syllables.

One name in §9's sketch did not survive: there is no `physics/airframe/mass.py`. The mass cascade,
the flight power and the reserve are one file, `flight.py`, because they are one fixed point and its
consequences — splitting them would have meant `mass.py` importing `flight.py` for the power the
cascade iterates on, or duplicating it.

## 11. Working mode

The engine is built to completion against §9 and reviewed as a finished version, rather than
supervised step by step. That puts the burden on this document: **every decision taken along the way
is recorded here**, with its rationale and the measurement behind it where there is one, so that the
review can be a discussion of recorded choices rather than an archaeology of the diff. A choice made
in passing and not written down is a defect in the same way an unnamed derived quantity is.

**Physics first, in the shape agreed in §9. Driver and econ second, from what the finished physics
actually look like.** Viz is also second, but is expected to need comparatively little rewriting: it
was already reading per-frame physics, and most of its work becomes selection.

Viz and econ are expected to be broken while this happens. Econ is rewritten from scratch afterwards
and nothing in the present econ constrains the design.

Building physics before its consumers is safe for a specific reason worth stating, since it is doing
real work here: **derived products are additive.** A consumer that later needs a differently-defined
aggregate is a new named P9 product over unchanged primaries, which I3/I6 make cheap and checkable.
The ordering only bites if a consumer needs a physical quantity no primary carries — which is why the
airframe question above has to be settled now rather than discovered later, and why the field sampler
is general enough (D2) that the parked exposure work can return as a sampling request rather than as
new physics, should exclusion-zone footprint turn out to be a siting cost.

## 12. Evidence

Measurements behind §1 and §6, taken 2026-08-03 on `92623bd`, reference case 5.8 GHz direct,
100 m emitter, 11 km cruise, 10 MW delivered, clear air unless stated.

Pass efficiency, econ's `energy_eta` against the figure's panel-D curve:

| weather | spacing | econ | figure | ratio |
|---|---|---|---|---|
| clear | 200 km | 21.32 % | 21.50 % | 1.01× |
| clear | 400 km | 16.45 % | 11.71 % | 0.71× |
| heavy_rain | 400 km | 17.46 % | 8.86 % | 0.51× |
| storm | 400 km | 15.45 % | 4.99 % | 0.32× |

Cost structure: one `evaluate` 21–26 ms warm; silhouette raster 77–90 ms per new look angle (~75 %
of a cold pass, and linear in mesh faces); 241-point pass 23.4 s cold, 5.9 s warm.

Emitter sizing, three code paths, one scenario at 8° reference elevation: 419.5 m (figure, via the
engine), 420.1 m (`sweeps`, τ=2 on the convex table), 419.5 m round / 220.9 m elliptical
(`econ_pass2`, own scan at forced `taper_p=0`). Agreement is by construction, not by test, and the
live divergence is the taper regime rather than the diameter.

### Measured since, on the implementation

Reference case as above unless stated; a "pass" is 81 samples over ±100 km at 11 km, `standard` tier.

| what | measured |
|---|---|
| warm pass, whole engine | **18 ms per sample** (was 56–95); 0.89 s for a 41-sample run with every product |
| silhouette raster | 93 ms cold, **3 ms** from the on-disk cache, free from the process memo |
| aperture grid at 8:1 aspect | n = 624 with one common pitch, **80** with per-axis pitch |
| field against the analytic Airy encircled energy | within **0.13 pp** at n_ap = 80; the pre-SPEC test's tolerance was 1.5 pp at n_ap = 512 |
| on-axis peak against π·a²/(λz)² | **0.4 %** |
| silhouette resample against the exact scattered transform | **2.0e-6** relative in capture (D10) |
| the two pre-SPEC field solves, now one | the side view's on-axis value **IS** the capture peak, asserted to 1e-9, where the two independent solves agreed to 15 % |
| pointing-error direction, the two gauges | **7 pp of capture** apart at β=60° — which is what made D13 necessary |
| MTOW, against the pre-SPEC scalar cascade | 79.95 t; rectenna 2.095 t from area × 1 kg/m² against the old hardcoded 2.0 t |
| mass fixed point, 10 lockstep iterations | within **1 kg** of 80; residual at 40 iterations is 1.5e-11 kg |
| station spacing at 15°, 11 km | **81.0 km** on the sphere against econ's flat-earth 82.1 km |
| the taper grid as a batch axis (D11) | 85.5 ms/sample tiled against 72.5 ms/sample looped |

And one finding that is physics rather than engineering, pinned by
`test_chaining_relays_compounds_truncation_loss_the_closed_form_cannot_see`: three relays each sized for
τ=2 give a Goubau product of **94.8 %** and the true propagated field delivers **54.2 %**. The closed
form assumes the matched mode arrives at every element; after the first truncation the field is
Airy-like, so each subsequent rim cuts sidelobes the closed form has already counted as captured. The
gap compounds with hop count, which is a real argument against long relay chains and is not something a
product of per-hop efficiencies can express. The engine's own relay is one hop, where the same reference
agrees to 3 %.
