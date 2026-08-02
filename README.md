# beamlink

Link budget and first-pass economics for beaming microwave power to an aircraft in flight.

Split out of [`microwave-power-transmission`](https://github.com/Paul-Particle/microwave-power-transmission),
which keeps the research: the literature review, the argument, the papers, the aerostat and
tether sizing, and the napkin notebook. This repo is the model those depend on.

**If you move or rename this directory, two things break.** The research repo depends on it by
*relative path* — `[tool.uv.sources] beamlink = { path = "../beam-link", editable = true }` in its
`pyproject.toml`, resolved against that file's location — so a move needs that line updated and a
`uv sync` to relock, plus the `../beam-link` links in its README. The import name `beamlink` is
independent of the directory name and does not have to change with it.

    uv sync
    uv run python -m beamlink.test_beam_link      # ALL CHECKS PASSED
    uv run python -m beamlink.sweeps              # the corridor table, per weather
    uv run python -m beamlink.econ_pass2
    uv run python -m beamlink.beam_link_animation # out/flyby.html
    uv run python -m beamlink.beam_link_phone     # out/flyby_phone.html + the artifact fragment

## Direct by default, relay by opt-in

The case under active iteration is the DIRECT link (`5.8 direct`), and that is what every
entry point builds unless told otherwise. A RELAY is fully supported but deliberately opt-in:

    build_flyby_animation(arch_name="relay ×3 mult 5.8→17.4", weather="clear")

It is a genuinely different render path — the channel becomes two legs, ground→relay then
relay→plane, each with its own aperture, focus and wavelength — so `test_relay_channel` exercises
it on every run to stop it rotting while the direct case moves. Relay also earned its keep
physically: at a matched aperture it roughly doubled the direct case's delivered energy, and it
is the weather-robust option, so it is not to be quietly dropped.

## Layout

| module | what it is |
|---|---|
| `beam_link` | the engine: one scenario → link budget, taper choice, efficiency chain |
| `capture` | the overlap physics — beam × silhouette, free of `LinkConfig` and the chain |
| `architectures` | the named cases compared, as `LinkConfig` keyword sets — imports nothing |
| `sweeps` | the engine mapped over a trajectory: one pass, one corridor, the energy balance |
| `diffraction` | scalar diffraction from tapered elliptical apertures |
| `bwb_model` | blended-wing-body silhouette mesh and its projected capture area |
| `geo` | ENU pass geometry |
| `relations` | closed-form beam terms: Goubau coupling, the prime multiplier chain, ITU-R P.838 rain, band efficiencies |
| `airframe` | flight power and the mass cascade |
| `econ` | cost primitives: energy, fuel, station spacing/count, infra and airframe ownership |
| `econ_pass2` | the only econ driver — assembles those primitives on the engine's computed η |
| `field`, `chain_reference` | Hankel-transform ground truth the closed forms are checked against |
| `flyby_frames` | per-frame physics for the figure: one frame's beam (in W/m²), the channel side view, the aircraft's shadow, the exposure boundary, the beam-cut axes |
| `beam_link_animation`, `beam_link_phone` | the flyby figure itself, landscape and portrait |
| `plot_spectrum` | the atmospheric-attenuation figure the research notebook embeds |
| `generate_atmosphere_profile`, `generate_spectrum` | regenerate the committed ITU tables (`--extra generators`) |

`relations`, `airframe` and `econ` were one 719-line `napkin.py` before the split. Its history
is in this repo, but the similarity is too low for git's default rename detection:

    git log --follow -M20% -- src/beamlink/relations.py

## Exposure: the beam is a keep-out volume, not a hazard that ends somewhere

The beam maps are drawn in absolute W/m² (each channel leg scaled by its own radiated power), so
they can carry a regulatory line: `relations.PUBLIC_EXPOSURE_LIMIT_W_M2` = 10 W/m², where ICNIRP
2020, FCC OET-65 uncontrolled and IEEE C95.1-2019 all agree above ~2 GHz. What that line says for
the reference case (5.8 GHz, 100 m emitter, 10 MW delivered, clear air):

- **Nothing along the beam is remotely close to compliant.** Mean density across the emitter face
  is 1.7 kW/m² at the overhead frame — 171× the public limit, 34× occupational — and 19 kW/m² at
  100 km along-track (the engine radiates 148 MW there to deliver 10 MW). There is no distance at
  which the beam "becomes safe": it is a keep-out tube from the array face to the aircraft and on
  past it.
- **The tube's radius is what the picture shows**: ~20 m at the overhead waist, a few hundred m at
  mid-pass, 1.6 km (wide axis) at 100 km along-track and ~4 km at the horizon frames. It is set by
  where the sidelobe skirt drops through the limit, so it tracks the beam envelope: on the far
  frames it is several times wider than the map window and the amber line correctly disappears.
- **The occupational limit is not worth drawing separately** — 5× in level is <20 % in radius at
  these gradients, i.e. the same boundary one line-width out.
- **The aircraft is the beam dump, and that is the one place the hazard does end.** Overhead the
  silhouette intercepts 99.94 % of the beam, so past it the axis is 0.6 W/m² — below the public
  limit. At 120 km along-track it catches 7 % of a 220 MW beam, so what continues past is still
  400× the limit. The exposure argument therefore rests on capture: a well-aimed beam has its
  hazard terminated by the receiver, a poorly-aimed one irradiates whatever is behind it.
- The skin cap (`LinkConfig.max_power_density_w_m2`, 2.5e5 W/m²) is never the binding constraint
  anywhere in the channel — peak on the skin is 1.3e5 W/m² at its worst.

The shadow behind the aircraft is ray optics (umbra/penumbra from the finite aperture) blended into
the diffraction limit ((1 − capture) of the power, original beam shape) over the Fresnel length
a²/λ ≈ 17 km. It is not an obstructed-diffraction calculation and has no edge fringes;
`flyby_frames._shadow_transmission` records what a rigorous version would take.

## Known open items

Carried over deliberately unfixed, from the comment-recovery review that preceded the split:

- `airframe`/`relations` are clean, but `tether_blowover` — which stayed in the research repo —
  applies its tilt penalty twice and double-counts cable weight.

Resolved since:

- The uniform-disc measure is now named for what it is: `equivalent_disc_radius`, reported as
  `equivalent_disc_radius_m`. It was never a Gaussian 1/e² waist, and the two functions whose
  √2 the old `spot_waist_m` name echoed (`focused_spot_waist`, `gaussian_beam_radius`) are gone
  with the rest of the pre-diffraction spot model.
- `focal_intensity_grid` / `_at` no longer clip the aperture on the `d_phase` branch:
  `_aperture_grid` sizes the grid to hold it and refuses past a cap. This was NOT in fact
  unreachable — 70×20 m semi-axes focused at 50 km already clip at n_ap=512 by z≈1.5 km, which
  is squarely where the near-field beam maps evaluate.

- When the power-density cap cannot be met at any allowed defocus, that is now logged (a warning
  naming how far over, at what focus and power) and reported as `density_cap_exceeded` on the
  stages. It deliberately does NOT block, and does not touch `feasible` — the decision was to
  surface it and keep going. The bisection's lower bracket is tested rather than assumed
  compliant, which is what used to let a non-compliant geometry return unremarked. That bracket
  (`slant / 8`) is also no longer assumed EVALUABLE: focusing a very large auto-sized aperture
  that short leaves the sampled Fresnel regime, so `_defocus_floor` walks the floor back out to
  the most-spread spot the field model can represent. Before that, `build_flyby_animation()` on
  its own defaults — no explicit `emitter_diam_m`, so the emitter is sized for the horizon —
  raised `ValueError` from the aperture guard at the overhead frames.

- The airframe drawn on the beam maps faces the way it is flying. Its along axis was
  `frame["look"]`, which points AT the emitter (what incidence needs) while the map's along axis
  runs emitter → plane — so the aircraft was mirrored front-to-back, nose pointing away from the
  emitter on the inbound half and back at it on the outbound half. The cut's three axes now come
  from `flyby_frames.beam_cut_axes`, which also pins the transverse sign to the airframe (eigenvector
  signs are arbitrary and flip at the overhead crossover, which drew the aircraft belly-up over half
  the pass) and derives the viewing direction, per layout, so each panel is a true view of its cut
  rather than the mirror image.

- The beam maps no longer stop at the aircraft: the last channel leg is rendered over the whole
  along axis with the aircraft's shadow cast into it (`beam_past_plane`, on by default). The old
  `beam_full_range` flag never worked — planes were assigned to a leg only within its own length, so
  asking for the full range still left everything past the aircraft NaN. The funnel's denominator is
  now the plane total BEFORE the aircraft takes its share, so capture shows up as the high-level
  rings closing at the plane line instead of being renormalised away (which drew confident contours
  around a beam that was no longer there). Cost: every frame now evaluates every plane, ~2.2× the
  build time.

Design questions still recorded in the code as open rather than rationalised: whether the
`elevation_deg` sign flip between the ground-emitter and relay cases is a bug or a convention,
and whether a scan-loss term belongs in the chain at all (deferred as later refinement).
