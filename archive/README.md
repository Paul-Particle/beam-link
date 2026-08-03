# Archive record

This repository was archived on 2026-08-03, together with the research repo it was split out of,
[`microwave-power-transmission`](https://github.com/Paul-Particle/microwave-power-transmission).
Neither is being continued. That repo's `ARCHIVE/README.md` explains why and holds the agent-memory
record for the project as a whole; this file covers only what is specific to the model.

This directory predates the archiving — `pre-spec/` below is the generation of the model that the
SPEC rebuild replaced, kept at that point for reference. The two files added alongside it are the
archive record proper.

**`main` was fast-forwarded to `spec-engine` when the repo was archived**, so a plain clone gives the
final state of the engine. Before that, `main` still pointed at the generation preceding the SPEC
rebuild, which would have been the misleading thing to hand a future reader. The `spec-engine` branch
is left in place and points at the same commit. The earlier generation is still in the history, and
its files are also kept as they stood in `pre-spec/` next to this file.

## What was carried forward

Only the blended-wing-body geometry, and only to `research/beamed-aviation` as a standalone `bwb`
package: `physics/airframe/loft.py` and `physics/airframe/silhouette.py`, reduced to a numpy-only
dependency. The shape became a plain dataclass and the tessellation an argument, which is what let
`artifacts.spec.LoftSpec`, `physics.cache` and `physics.fidelity.FidelityConfig` be left behind.

Deliberately not carried: `flight.py` (the mass cascade and flight power), the whole field-solve and
capture chain, econ, viz, drivers, and the product/manifest data contract.

`verify_bwb_extraction.py` is the script that checked the extraction, kept because it cannot be
re-run once this tree is gone from disk and it is the evidence the port was faithful. It imports both
`beamlink` and the extracted `bwb`, and compares meshes at four tessellations, three off-default
shapes, every frame conversion, the convex area table, and full rasters at twelve viewing angles.
**179 comparisons, all bit-identical** — max absolute and max relative difference both exactly zero,
not merely within a tolerance. To run it again:

    git clone https://github.com/Paul-Particle/beam-link && cd beam-link
    uv sync
    PYTHONPATH=/path/to/beamed-aviation/src .venv/bin/python archive/verify_bwb_extraction.py

## Worth knowing if the model is ever revisited

`SPEC.md` is the architecture and the record of every decision behind it, and it is the densest thing
in either repo. The import firewall it describes — the package layout being the rule, enforced by a
test that walks the import graph — is the part most likely to be worth reusing independently of the
subject matter. The known-open bugs and the three design decisions that were never resolved are in
the archived memory entries in the research repo, not here.
