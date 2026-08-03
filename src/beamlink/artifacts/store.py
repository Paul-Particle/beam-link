"""Reading and writing runs and studies: `save_run`, `load_run`, `save_study`, `load_study`.

This is the ONLY module a consumer imports to read a run (§7). It exposes frozen dataclasses and
arrays with named axes; it never returns a per-sample object that would force a consumer to loop,
because econ is expected to be vectorized in the same shape the engine is (§6a).

    runs/<id>/
      manifest.json        the spec verbatim, engine sha, per-product status/fidelity/method
      <product>.npz        arrays, SI units, full precision

    studies/<id>/
      manifest.json        the runs aggregated and the aggregation definitions applied
      results.npz          the aggregated arrays, same format as a run's
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from beamlink.artifacts import serialize
from beamlink.artifacts.manifest import Manifest, ProductStatus, StudyManifest
from beamlink.artifacts.products import PRODUCT_TYPES, Products
from beamlink.artifacts.spec import RunSpec

MANIFEST_NAME = "manifest.json"
STUDY_RESULTS_NAME = "results.npz"


class RunIncomplete(RuntimeError):
    """A consumer was pointed at a run that lacks what it was told to show.

    Raised by `require`, so a figure refuses loudly rather than substituting or recomputing (§7).
    """


# --- writing -----------------------------------------------------------------------------


def save_run(run_dir: str | Path, products: Products, manifest: Manifest) -> Path:
    """Write one run directory. Products the manifest calls absent are not written."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {}
    for name, product in ((n, products.get(n)) for n in PRODUCT_TYPES):
        if product is None:
            continue
        arrays, meta = serialize.to_parts(product)
        metadata[name] = meta
        _savez_atomic(run_dir / f"{name}.npz", arrays)
    payload = _manifest_payload(manifest)
    payload["product_metadata"] = metadata
    _write_json_atomic(run_dir / MANIFEST_NAME, payload)
    return run_dir


def save_study(study_dir: str | Path, results: Mapping[str, np.ndarray],
               manifest: StudyManifest) -> Path:
    """Write one study directory: aggregated arrays plus the manifest that defines them."""
    study_dir = Path(study_dir)
    study_dir.mkdir(parents=True, exist_ok=True)
    _savez_atomic(study_dir / STUDY_RESULTS_NAME, dict(results))
    _write_json_atomic(study_dir / MANIFEST_NAME,
                       {"kind": "study", **serialize.to_json_safe(_as_dict(manifest))})
    return study_dir


# --- reading -----------------------------------------------------------------------------


def load_manifest(run_dir: str | Path) -> Manifest:
    payload = json.loads((Path(run_dir) / MANIFEST_NAME).read_text())
    return _manifest_from_payload(payload)


def load_run(run_dir: str | Path, products: tuple[str, ...] | None = None
             ) -> tuple[Products, Manifest]:
    """Load a run. `products` limits what is read from disk; absent products come back None.

    The manifest is always fully loaded, because a consumer has to be able to say what was asked
    for and not produced even when it is not reading that product's arrays.
    """
    run_dir = Path(run_dir)
    payload = json.loads((run_dir / MANIFEST_NAME).read_text())
    manifest = _manifest_from_payload(payload)
    metadata = payload.get("product_metadata", {})
    wanted = tuple(PRODUCT_TYPES) if products is None else products
    loaded: dict[str, Any] = {}
    for name in wanted:
        path = run_dir / f"{name}.npz"
        if name not in metadata or not path.exists():
            continue
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        loaded[name] = serialize.from_parts(PRODUCT_TYPES[name], arrays, metadata[name])
    return Products(**loaded), manifest


def load_spec(run_dir: str | Path) -> RunSpec:
    """The run's own spec, rebuilt from the manifest. What the run was asked to be."""
    payload = json.loads((Path(run_dir) / MANIFEST_NAME).read_text())
    return serialize.from_parts(RunSpec, {}, payload["spec"])


def load_study(study_dir: str | Path) -> tuple[dict[str, np.ndarray], StudyManifest]:
    """A study loads through the same loaders a run does (§4)."""
    study_dir = Path(study_dir)
    payload = json.loads((study_dir / MANIFEST_NAME).read_text())
    manifest = serialize.from_parts(StudyManifest, {}, payload)
    with np.load(study_dir / STUDY_RESULTS_NAME, allow_pickle=False) as archive:
        results = {key: archive[key] for key in archive.files}
    return results, manifest


def require(products: Products, manifest: Manifest, *names: str) -> None:
    """Refuse loudly when pointed at a run lacking what the caller was told to show (§7).

    The alternative a consumer must never take is substituting a different product or computing
    the missing one locally, which is exactly the second modelling pass SPEC.md §1 exists to
    prevent.
    """
    missing = {}
    for name in names:
        if name not in PRODUCT_TYPES:
            raise KeyError(f"unknown product {name!r}")
        if products.get(name) is None:
            entry = manifest.status(name)
            missing[name] = entry.reason if entry else "not requested in this run"
    if missing:
        detail = "; ".join(f"{name}: {reason}" for name, reason in missing.items())
        raise RunIncomplete(f"run {manifest.run_id} does not carry {detail}")


# --- plumbing ----------------------------------------------------------------------------


def _as_dict(record) -> dict[str, Any]:
    return {f.name: getattr(record, f.name) for f in dataclasses.fields(record)}


def _manifest_payload(manifest: Manifest) -> dict[str, Any]:
    payload = serialize.to_json_safe(_as_dict(manifest))
    payload["kind"] = "run"
    payload["products"] = {name: serialize.to_json_safe(_as_dict(status))
                           for name, status in manifest.products.items()}
    return payload


def _manifest_from_payload(payload: Mapping[str, Any]) -> Manifest:
    fields = {f.name for f in dataclasses.fields(Manifest)}
    kwargs = {k: v for k, v in payload.items() if k in fields}
    kwargs["products"] = {name: ProductStatus(**entry)
                          for name, entry in payload.get("products", {}).items()}
    kwargs["candidate_axes"] = {name: tuple(values)
                                for name, values in payload.get("candidate_axes", {}).items()}
    return Manifest(**kwargs)


def _savez_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write via a sibling temp file and rename, so a reader never sees a half-written product.

    Caches and outputs both have to be safe to write from parallel calls (D4), and on this repo's
    shared mount a partially-flushed file is a real failure mode rather than a theoretical one.
    """
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=False))
    temporary.replace(path)
