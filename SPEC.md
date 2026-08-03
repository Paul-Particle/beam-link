# beamlink — physics engine spec

Status: **outline, agreed in principle, not yet implemented.** This document is the target shape.
The current code does not conform; §8 is the backlog that gets it there and §9 lists the decisions
still outstanding.

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
| I3 | **Only the engine derives.** Every derived product names the primary it came from and carries its full definition (span, weighting, policy, statistic, sampling) as data. A derived product that cannot name its primary is illegal. |
| I4 | **Consumers never derive.** They select, display, and cost. A consumer needing a different aggregate is a request for a new named engine product, not a local calculation. |
| I5 | **Fidelity is part of product identity.** Same shape, same units, same semantics across tiers; method and resolution recorded per product. Mixing tiers within one presented result must be detectable. |
| I6 | **Every derived product is recompute-checkable.** CI recomputes it from its primary and asserts agreement to a stated tolerance. This is what makes derived products safe rather than redundant. |
| I7 | **Products carry physics, not presentation.** No label strings, no colours, no 0–1 normalised axes, no per-frame peak normalisation, no NaN-separated polylines, no layout or "how far to draw" arguments, no precision reduced for file size. |
| I8 | **The import firewall is a test.** Consumers importing any physics module fails the suite. No exemptions. |

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

## 6. Fidelity

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

## 9. Decisions outstanding

- **D1 — `elevation_deg` sign.** Currently opposite between the ground-emitter and relay branches,
  recorded in the code as an open question. It cannot stay open in a schema: P2 needs one convention.
- **D2 — field sampling: declared planes or a volume.** Cuts are cheap (~280 MB per run at today's
  frame count); an honest volume is orders of magnitude larger. The cheap option is the viz-shaped
  one, so this trade should be made deliberately, with a disk budget.
- **D3 — is "best fixed aperture aspect over a trajectory" a product?** If yes it is an engine
  optimisation with recorded objective; if no, the figure that needs it goes.
- **D4 — is `sweeps` inside the engine?** Assumed yes throughout this document: it becomes the
  trajectory and aggregate layer (P1, P9). Unconfirmed.

## 10. Order of work

1. The two free speedups (§6) — no fidelity cost, and they make everything after this bearable.
2. Schema and manifest: write §5 down as dataclasses, with a stated physical question per field,
   before any engine change. Review gate.
3. One solve: collapse `overlap_capture` and `channel_field` into a single field solver with
   requested sampling (I1). This is new physics capability, not a move — the engine has no concept
   of a field away from the aircraft today. Largest single piece.
4. P1–P8 emission plus the manifest; the recompute checks of I6; the import firewall test of I8.
5. P9 aggregates, with the two definitions from §1 as the first two named products.
6. Rebuild viz against the contract; rewrite econ from scratch.

Fidelity tiers land with the schema (I5), not after it.

## 11. Evidence

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
