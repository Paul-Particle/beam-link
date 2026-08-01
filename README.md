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

## Layout

| module | what it is |
|---|---|
| `beam_link` | the engine: scenario → link budget, taper choice, flyby energy balance |
| `diffraction` | scalar diffraction from tapered elliptical apertures |
| `bwb_model` | blended-wing-body silhouette mesh and its projected capture area |
| `geo` | ENU pass geometry |
| `relations` | closed-form beam terms: Goubau coupling, the prime multiplier chain, ITU-R P.838 rain, band efficiencies |
| `airframe` | flight power and the mass cascade |
| `econ` | first-pass cost comparison against jet fuel |
| `econ_pass2` | the same cost primitives driven by the engine's computed η instead of a flat placeholder |
| `field`, `chain_reference` | Hankel-transform ground truth the closed forms are checked against |
| `beam_link_figures`, `beam_link_animation`, `beam_link_phone`, `beam_link_emitter_shape` | plotly output |
| `winglet_validation` | matplotlib diagnostics for the mesh (`--extra validation`) |
| `generate_atmosphere_profile`, `generate_spectrum` | regenerate the committed ITU tables (`--extra generators`) |

`relations`, `airframe` and `econ` were one 719-line `napkin.py` before the split. Its history
is in this repo, but the similarity is too low for git's default rename detection:

    git log --follow -M20% -- src/beamlink/relations.py

## Known open items

Carried over deliberately unfixed, from the comment-recovery review that preceded the split:

- `airframe`/`relations` are clean, but `tether_blowover` — which stayed in the research repo —
  applies its tilt penalty twice and double-counts cable weight.
- `diffraction.effective_waist` computes the uniform-disc equivalent radius, not a Gaussian
  1/e² waist; the √2 discrepancy propagates into the name `spot_waist_m`.
- `focal_intensity_grid` / `_at` can clip the aperture on the `d_phase` branch with no guard.
  Unreachable through the public API only because deep defocus and high aperture aspect are
  anti-correlated, and nothing enforces that.

Three design questions are recorded in the code as open rather than rationalised: what should
happen when the power-density cap cannot be met, whether the `elevation_deg` sign flip between
the ground-emitter and relay cases is a bug or a convention, and whether a scan-loss term
belongs in the chain at all.
