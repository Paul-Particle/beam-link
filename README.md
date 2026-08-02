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
| `beam_link` | the engine: scenario → link budget, taper choice, flyby energy balance |
| `capture` | the overlap physics — beam × silhouette, free of `LinkConfig` and the chain |
| `diffraction` | scalar diffraction from tapered elliptical apertures |
| `bwb_model` | blended-wing-body silhouette mesh and its projected capture area |
| `geo` | ENU pass geometry |
| `relations` | closed-form beam terms: Goubau coupling, the prime multiplier chain, ITU-R P.838 rain, band efficiencies |
| `airframe` | flight power and the mass cascade |
| `econ` | cost primitives: energy, fuel, station spacing/count, infra and airframe ownership |
| `econ_pass2` | the only econ driver — assembles those primitives on the engine's computed η |
| `field`, `chain_reference` | Hankel-transform ground truth the closed forms are checked against |
| `beam_link_animation`, `beam_link_phone` | the flyby figure, landscape and portrait |
| `plot_spectrum` | the atmospheric-attenuation figure the research notebook embeds |
| `generate_atmosphere_profile`, `generate_spectrum` | regenerate the committed ITU tables (`--extra generators`) |

`relations`, `airframe` and `econ` were one 719-line `napkin.py` before the split. Its history
is in this repo, but the similarity is too low for git's default rename detection:

    git log --follow -M20% -- src/beamlink/relations.py

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
  compliant, which is what used to let a non-compliant geometry return unremarked.

Design questions still recorded in the code as open rather than rationalised: whether the
`elevation_deg` sign flip between the ground-emitter and relay cases is a bug or a convention,
and whether a scan-loss term belongs in the chain at all (deferred as later refinement).
