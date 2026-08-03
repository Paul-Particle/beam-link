# beamlink — physics engine spec

Status: **outline, agreed in principle, not yet implemented.** This document is the target shape and
the record of every decision behind it (§9). The current code does not conform; §8 is the backlog that
gets it there. Built to completion and reviewed as a finished version — see §11.

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

From the audit of 2026-08-03. Verdicts are against §3 and §5.

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

### Nothing open

All decisions needed to start are taken. Later ones get appended here as they are made (§11).

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
