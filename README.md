# beamlink

A physics engine for beamed microwave power to an aircraft in flight, and the layers that read it.

Split out of [`microwave-power-transmission`](https://github.com/Paul-Particle/microwave-power-transmission),
which keeps the research: the literature review, the argument, the papers, the aerostat and tether
sizing, and the napkin notebook. This repo is the model those depend on.

**If you move or rename this directory, two things break.** The research repo depends on it by
*relative path* — `[tool.uv.sources] beamlink = { path = "../beam-link", editable = true }` in its
`pyproject.toml`, resolved against that file's location — so a move needs that line updated and a
`uv sync` to relock, plus the `../beam-link` links in its README. The import name `beamlink` is
independent of the directory name and does not have to change with it.

## The shape of it

**[SPEC.md](SPEC.md) is the architecture**, and the record of every decision behind it. The engine is
the first-class citizen: it answers physical questions, emits the answers as data, and knows nothing
about who reads them. Visualisation and economics are consumers — they select from what a run produced
and display or cost it. Neither may compute physics, and neither owns the shape of the data.

    one RunSpec  ->  runs/<id>/manifest.json    the spec verbatim, engine sha, and for every
                                                requested product: produced | skipped(why) | failed(why)
                     runs/<id>/<product>.npz    arrays, SI units, full precision

The package layout IS the import firewall, enforced by a test that walks the import graph, so a new
file inherits the rule from the directory it is placed in:

| package | may import | what it is |
|---|---|---|
| `artifacts/` | nothing else in the repo | the data contract: the run spec, products P1–P13, the manifest, load and save |
| `physics/` | `artifacts/` | the engine. `geo`, `airframe/` (loft, silhouette, mass, flight power, reserve), `beam_link/` (the field solve, capture, the chain, the atmosphere, and a Hankel ground truth) |
| `viz/` | `artifacts/` only | figures |
| `econ/` | `artifacts/` only | cost against jet fuel |
| `drivers/` | `physics/` and `econ/` | sweeps, search, corridors, calibration — the only layer that sees both sides |

## Running it

    uv sync
    uv run pytest -q                                    # 105 checks: physics, the contract, the firewall
    uv run python -m beamlink.drivers.calibrate         # the fidelity error table below
    uv run python -m beamlink.viz.run_report runs/<id>  # one self-contained page for a run

A run, end to end:

```python
from beamlink import artifacts as A, physics, viz

spec = A.RunSpec(
    scenario=A.ScenarioSpec(frequency_ghz=5.8, aperture=A.ApertureSpec(diameter_m=100.0),
                            weather=A.weather("clear")),
    trajectory=A.TrajectorySpec.straight_overflight(half_span_m=100e3, altitude_m=11_000.0,
                                                    speed_ms=250.0, n_samples=41),
    products=tuple(A.ProductRequest(name=name, sampling=A.default_axial_planes())
                   if name == "field" else A.ProductRequest(name=name) for name in A.PRODUCTS),
    power=A.PowerSpec(delivered_target_w=10e6), run_id="reference")

products, manifest = physics.evaluate(spec)          # writes nothing
physics.write(products, manifest, "runs/reference")   # separate on purpose — see SPEC.md D4
viz.report("runs/reference", channel_sets=("axial_major", "axial_minor"))
```

Reading one back needs `artifacts` and nothing else, which is what makes the products a contract
rather than an API:

```python
products, manifest = A.load_run("runs/reference")
manifest.absent()                       # {name: why it is not here}
products.aggregates.entries["pass_energy_efficiency"].definition   # span, serve policy, statistic
```

## Two things worth knowing before you read a number

**Every derived product names its inputs and is recomputed on every test run.** P6 capture is a sum
over P5's emitted field and P3's emitted mesh; P7's η is the product of exactly the factors it emits;
P9's aggregates re-run from their own recorded definitions. 22 such checks. This exists because the
defect it prevents already happened twice — see SPEC.md §1.

**"The pass efficiency" is not a number, it is several, and each one says which it is.** P9 emits
`pass_eta_mean_served` and `pass_eta_mean_whole_span` side by side, each carrying its span, serve
policy, statistic and weighting. They agree in clear air over a short span and differ by ~0.7× in a
storm over ±200 km. Two numbers that disagree and say why is the fix; one anonymous number was the bug.

## Direct by default, relay by opt-in

The case under active iteration is the direct link. A relay is fully supported but explicitly asked
for, and it is a genuinely different path — the channel becomes two legs, each with its own aperture,
focus and wavelength — so the suite exercises it on every run to stop it rotting:

```python
A.ScenarioSpec(relay=A.RelaySpec(position_m=(0.0, 0.0, 20_000.0), mode="multiplier",
                                 f_in_ghz=5.8, f_out_ghz=17.4, aperture_diam_m=40.0))
```

It also earned its keep physically: its emitter is above the weather, so it beats going direct through
a storm on continuous delivery. Note SPEC.md D12 — P4 and P5 describe the leg that serves the aircraft,
so a relay run's beam maps are the relay's beam, and the up-leg's capture is closed-form coupling.

## Scenarios with no receiver are legal

A "pure focusing sweep, no aircraft, no atmosphere" figure does not need privileged access to the
diffraction code. It needs a run whose scenario has no receiver: P4 and P5 exist, P3/P6/P7/P8/P9 do
not, and the manifest says so in as many words. That is the general form, and the report renders the
absences rather than leaving blanks.

## Exposure: the beam is a keep-out volume, not a hazard that ends somewhere

P5 emits intensity in absolute W/m², which is what lets a regulatory line be drawn *by the viewer* on
the model's own field — `viz.panels.PUBLIC_EXPOSURE_LIMIT_W_M2` is 10 W/m², where ICNIRP 2020, FCC
OET-65 uncontrolled and IEEE C95.1-2019 all agree above ~2 GHz. No product knows that number exists.

What the line says for the reference case: nothing along the beam is remotely close to compliant, the
keep-out tube's radius tracks the beam envelope, and **the aircraft is the beam dump** — overhead the
silhouette intercepts ~99.9 % and what continues past is below the public limit, while at 120 km
along-track it catches a few percent of a much larger beam and what continues past is hundreds of times
the limit. The exposure argument therefore rests on capture, which is why the shadow is an option on the
solve (`occluders="receiver"`) rather than a figure's own calculation. The shadow model is a ray-optics
umbra/penumbra blended into the diffraction limit over the Fresnel length; it has no edge fringes, and
`beam_link/occlusion.py` records what a rigorous version would take.

## Fidelity: tiers, and what each one costs you

Tiers vary method and resolution; they never vary shape, units or meaning. Run time against accuracy is
the only trade axis, and output size is deliberately not a lever. A tier without a published error bar
is not usable for a headline number, so here they are, regenerated by
`uv run python -m beamlink.drivers.calibrate --write-readme README.md`.

What the table says, in short:

- **`standard` is the default and is honest to ~0.1 pp of η** and under 0.7 % on the aggregates econ
  consumes, for 3–4× the speed of `full`. Use it for anything published.
- **`fast` costs almost nothing more** (~0.1 pp, ~0.6 %) at 5–7×, because the two things it coarsens —
  the mesh and the aperture sampling — are both weak levers. It is the sensible default for a sweep.
- **`coarse` is a different derivation, not a coarser one.** It uses the closed-form Goubau capture, so
  it never touches the field: 80–160× faster, 2–5 pp of η, 6–11 % on the aggregates, and *one-sided* —
  it can only over-state capture, because it assumes the optimal illumination the overlap integral has
  to earn. It also cannot defocus, so at 60 MW delivered (where the skin power-density cap binds) its
  error on pass efficiency goes from 7 % to 13 %. Use it to find the shape of a search space, never for
  a number.
- **Two levers are not worth pulling.** Single precision buys ~5 % of run time for no measurable error,
  and the density-cap solve's step count is free wherever the cap does not bind. The taper search is the
  one lever that really costs: 17 → 5 points is 0.9–1.3 % on the aggregates for 2.7–3.3×.
- **The mesh face count shows no speed-up here, and that is the measurement working.** Its cost is
  paying for silhouette rasters, and these timings are warm — so 5704 → 824 faces buys nothing on a
  second run and 0.2 % of accuracy. Its real payoff is the *cold* first run of a new loft, which is a
  one-off and belongs to the cache rather than to the tier.

<!-- calibration:start -->

_Measured 2026-08-03 on engine 29053e8dcf12, 0.2.0._

**Fidelity tiers against `full`**

| scenario | variant | s | speed | faces | Δη max (pp) | Δη mean (pp) | Δpass_energy_efficiency (%) | Δdelivered_energy_j (%) | Δpeak_eta (%) | Δserved_fraction (%) |
|---|---|---|---|---|---|---|---|---|---|---|
| 5.8 direct, clear | full (reference) | 1.40 | 1.0× | 5704 | 0.000 | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 |
| 5.8 direct, clear | standard | 0.41 | 3.4× | 5704 | 0.550 | 0.100 | 0.27 | 0.27 | 0.00 | 0.00 |
| 5.8 direct, clear | fast | 0.27 | 5.2× | 2156 | 0.585 | 0.114 | 0.53 | 0.53 | 0.00 | 0.00 |
| 5.8 direct, clear | coarse | 0.02 | 77.9× | 824 | 5.066 | 1.578 | 7.57 | 7.86 | 0.27 | 0.00 |
| 5.8 direct, storm | full (reference) | 1.40 | 1.0× | 5704 | 0.000 | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 |
| 5.8 direct, storm | standard | 0.40 | 3.5× | 5704 | 0.194 | 0.075 | 0.69 | 0.39 | 0.29 | 0.00 |
| 5.8 direct, storm | fast | 0.27 | 5.2× | 2156 | 0.100 | 0.044 | 0.32 | 0.16 | 0.16 | 0.00 |
| 5.8 direct, storm | coarse | 0.01 | 94.3× | 824 | 2.057 | 0.590 | 5.91 | 6.39 | 0.45 | 0.00 |
| 30 direct, clear | full (reference) | 1.50 | 1.0× | 5704 | 0.000 | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 |
| 30 direct, clear | standard | 0.42 | 3.5× | 5704 | 0.251 | 0.052 | 0.25 | 0.25 | 0.00 | 0.00 |
| 30 direct, clear | fast | 0.28 | 5.4× | 2156 | 0.361 | 0.068 | 0.58 | 0.58 | 0.01 | 0.00 |
| 30 direct, clear | coarse | 0.02 | 92.9× | 824 | 2.725 | 0.828 | 7.37 | 7.68 | 0.29 | 0.00 |
| relay x3 5.8->17.4, storm | full (reference) | 2.84 | 1.0× | 5704 | 0.000 | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 |
| relay x3 5.8->17.4, storm | standard | 0.70 | 4.1× | 5704 | 0.147 | 0.075 | 0.52 | 0.24 | 0.77 | 0.00 |
| relay x3 5.8->17.4, storm | fast | 0.40 | 7.2× | 2156 | 0.291 | 0.111 | 0.60 | 0.65 | 1.26 | 0.00 |
| relay x3 5.8->17.4, storm | coarse | 0.02 | 164.0× | 824 | 3.024 | 1.038 | 11.08 | 8.91 | 1.99 | 0.00 |

**Individual levers, moved off `full` one at a time**

| scenario | variant | s | speed | faces | Δη max (pp) | Δη mean (pp) | Δpass_energy_efficiency (%) | Δdelivered_energy_j (%) | Δpeak_eta (%) | Δserved_fraction (%) |
|---|---|---|---|---|---|---|---|---|---|---|
| 5.8 direct, clear | full (reference) | 1.41 | 1.0× | 5704 | 0.000 | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 |
| 5.8 direct, clear | mesh 5704 -> 612 faces | 1.48 | 1.0× | 824 | 0.241 | 0.053 | 0.23 | 0.22 | 0.01 | 0.00 |
| 5.8 direct, clear | closed-form capture | 0.02 | 56.5× | 5704 | 5.172 | 1.551 | 7.44 | 7.72 | 0.26 | 0.00 |
| 5.8 direct, clear | taper search 17 -> 5 | 0.52 | 2.7× | 5704 | 1.217 | 0.179 | 0.87 | 0.87 | 0.00 | 0.00 |
| 5.8 direct, clear | density solve 20 -> 6 steps | 1.27 | 1.1× | 5704 | 0.000 | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 |
| 5.8 direct, clear | aperture samples 48 -> 16 | 1.15 | 1.2× | 5704 | 0.062 | 0.022 | 0.07 | 0.07 | 0.00 | 0.00 |
| 5.8 direct, clear | single precision | 1.40 | 1.0× | 5704 | 0.000 | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 |
| 5.8 direct, clear | look-angle bin 0.1 -> 1.0 deg | 1.50 | 0.9× | 5704 | 0.340 | 0.127 | 0.24 | 0.24 | 0.00 | 0.00 |
| 5.8 direct, clear | raster grid 256 -> 64 | 0.78 | 1.8× | 5704 | 0.453 | 0.129 | 0.00 | 0.00 | 0.00 | 0.00 |
| 5.8 direct, clear @ 60 MW | full (reference) | 6.69 | 1.0× | 5704 | 0.000 | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 |
| 5.8 direct, clear @ 60 MW | mesh 5704 -> 612 faces | 6.43 | 1.0× | 824 | 0.241 | 0.077 | 0.37 | 0.35 | 0.02 | 0.00 |
| 5.8 direct, clear @ 60 MW | closed-form capture | 0.02 | 299.7× | 5704 | 6.327 | 2.573 | 13.17 | 3.64 | 9.19 | 0.00 |
| 5.8 direct, clear @ 60 MW | taper search 17 -> 5 | 2.01 | 3.3× | 5704 | 2.494 | 0.252 | 1.29 | 1.29 | 0.00 | 0.00 |
| 5.8 direct, clear @ 60 MW | density solve 20 -> 6 steps | 3.16 | 2.1× | 5704 | 0.460 | 0.047 | 0.24 | 0.16 | 0.40 | 0.00 |
| 5.8 direct, clear @ 60 MW | aperture samples 48 -> 16 | 5.06 | 1.3× | 5704 | 0.062 | 0.021 | 0.07 | 0.07 | 0.00 | 0.00 |
| 5.8 direct, clear @ 60 MW | single precision | 5.87 | 1.1× | 5704 | 0.000 | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 |
| 5.8 direct, clear @ 60 MW | look-angle bin 0.1 -> 1.0 deg | 6.22 | 1.1× | 5704 | 0.340 | 0.137 | 0.26 | 0.26 | 0.00 | 0.00 |
| 5.8 direct, clear @ 60 MW | raster grid 256 -> 64 | 3.22 | 2.1× | 5704 | 0.490 | 0.146 | 0.19 | 0.07 | 0.12 | 0.00 |

<!-- calibration:end -->

## Known open items

Carried forward deliberately, and named rather than hidden:

- **No scan roll-off on the emitting aperture.** Every pre-SPEC call site passed `scan_deg=0`, which is
  only correct for a mechanically pointed aperture, and the baseline emitter is a phased array. It
  flatters the low-elevation ends of a pass. Carried forward as the same omission rather than given a
  new derivation nobody has checked; `ApertureSpec.error_efficiency` is where a real one would go.
- **Whether a scan-loss term belongs in the chain at all** is the same question, deferred.
- **The relay up-leg has no field solve** (SPEC.md D12), so no beam map and no shaped aperture on that
  leg. Closed-form coupling is the right model for two circular apertures; what is missing is the
  capability, not the number.
- **The exclusion-zone footprint is parked.** If siting cost turns out to need it, it comes back as a
  sampling request on P5 — which is why the volume sampler exists in v1 (SPEC.md D2).
- `tether_blowover`, which stayed in the research repo, applies its tilt penalty twice and
  double-counts cable weight.

Resolved since the pre-SPEC version, beyond SPEC.md §8's twenty-two items:

- The `elevation_deg` sign flip between the ground-emitter and relay cases was a genuine ambiguity, not
  a bug or a convention: it is now one definition applied per leg (SPEC.md D1), so ground→aircraft is
  +90° overhead and relay→aircraft is correctly negative, and the double key is gone rather than renamed.
- The defocus floor's back-off loop is a closed form (D9), and the density-cap search runs lockstep.
- `rectenna_mass_t = 2.0` is derived from the belly-on silhouette area and a declared areal density.
- The pointing error's *direction* was an arbitrary gauge choice worth 7 pp of capture; it now comes from
  the projection's own third moment (D13).

## History

`relations`, `airframe` and `econ` were one 719-line `napkin.py` before the split, and the pre-SPEC
modules are in `archive/pre-spec/` rather than deleted, so a number can be traced to the code that used
to produce it. Git's default rename detection does not follow the napkin split:

    git log --follow -M20% -- archive/pre-spec/relations.py
