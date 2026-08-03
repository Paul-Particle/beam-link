"""The engine. Imports `artifacts/` to build and write products, and nothing else in the repo.

    evaluate(spec) -> (Products, Manifest)     the physics. Writes nothing.
    write(products, manifest, run_dir)         the archival path.
    run(spec, run_dir)                         both, for convenience.

Those first two are separate because an optimiser calls `evaluate` and reads its objective straight
off the emitted aggregate, while the archival path calls both — one physics path, two uses, not two
paths (SPEC.md D4). No process-global state may affect a result: the caches under `physics.cache` are
correctness-neutral, explicitly keyed, and safe to share across parallel calls.

Only `physics/` and `drivers/` may import this. That is enforced by `tests/test_import_firewall.py`,
not by convention.
"""

from beamlink.physics.engine import (ENGINE_VERSION, candidate_scenarios, engine_sha, evaluate,
                                     leg_frequencies, run, write)
from beamlink.physics.fidelity import TIERS, FidelityConfig, config as fidelity_config

__all__ = ["ENGINE_VERSION", "FidelityConfig", "TIERS", "candidate_scenarios", "engine_sha",
           "evaluate", "fidelity_config", "leg_frequencies", "run", "write"]
