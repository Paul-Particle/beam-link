"""An explicitly-keyed, persistable array cache.

D4 forbids process-global state that can affect a result: caches must be correctness-neutral,
explicitly keyed, and safe to share or persist across parallel calls. That rules out the
pre-SPEC pattern of a module-level dict keyed on a rounded look angle, because the rounding was
the key AND the quantisation, and nothing recorded which mesh the entry was built from — so a
run with a coarser loft could read a finer loft's raster back out.

What makes it safe here:

  the key is the full input      a content key over every argument that changes the result,
                                 including the loft parameters and the grid size, so a different
                                 mesh cannot collide
  the value is arrays only       no objects, no pickle: an entry is an .npz a reader can open
                                 with numpy alone
  writes are atomic              temp file plus rename, so a parallel reader never sees a
                                 half-written entry (and this repo's shared mount makes that a
                                 real failure mode, not a theoretical one)
  a miss is never an error       an unreadable or stale entry is recomputed, so deleting the
                                 cache directory can only cost time

The measured win is the reason it exists: rasterising a look angle costs 77–90 ms and is ~75 % of
a cold pass, against ~1 ms to read one back (§6).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

_ENVIRONMENT_VARIABLE = "BEAMLINK_CACHE_DIR"

# In-process memo in front of the disk, so a pass that revisits the same look angle does not pay
# even the ~1 ms read. Keyed by the SAME content key as the disk entry, which is what makes it
# correctness-neutral: there is no state here that the key does not cover.
_MEMO: dict[str, dict[str, np.ndarray]] = {}


def cache_root() -> Path:
    """Where entries live. Overridable per process so tests never touch a real cache."""
    override = os.environ.get(_ENVIRONMENT_VARIABLE)
    return Path(override) if override else Path.home() / ".cache" / "beamlink"


def content_key(namespace: str, **parts: Any) -> str:
    """A stable hash over every argument that changes the result.

    Sorted keys and a canonical JSON encoding, so the key does not depend on call order, and
    floats go through `repr` so two bit-identical inputs never hash differently.
    """
    payload = json.dumps(_canonical(parts), sort_keys=True, separators=(",", ":"))
    return f"{namespace}-{hashlib.sha256(payload.encode()).hexdigest()[:32]}"


def _canonical(value: Any) -> Any:
    if isinstance(value, np.generic):
        return repr(value.item())
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, np.ndarray):
        return [_canonical(v) for v in value.tolist()]
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if hasattr(value, "__dataclass_fields__"):
        return {name: _canonical(getattr(value, name))
                for name in sorted(value.__dataclass_fields__)}
    return value


def load(key: str) -> dict[str, np.ndarray] | None:
    """The cached arrays for `key`, or None. Never raises: a bad entry is a miss."""
    hit = _MEMO.get(key)
    if hit is not None:
        return hit
    path = cache_root() / f"{key}.npz"
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
    except Exception:
        return None                      # truncated or written by an older layout → recompute
    _MEMO[key] = arrays
    return arrays


def store(key: str, arrays: Mapping[str, np.ndarray]) -> None:
    """Write an entry. A failure to write is not a failure of the run."""
    entry = {name: np.asarray(value) for name, value in arrays.items()}
    _MEMO[key] = entry
    root = cache_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
        temporary = root / f"{key}.{os.getpid()}.tmp"
        with temporary.open("wb") as handle:
            np.savez(handle, **entry)
        temporary.replace(root / f"{key}.npz")
    except Exception:
        pass


def memoized(key: str, compute: Callable[[], Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """`compute()` unless `key` is already cached. The whole public surface, in practice."""
    hit = load(key)
    if hit is not None:
        return hit
    arrays = dict(compute())
    store(key, arrays)
    return arrays


def clear_process_memo() -> None:
    """Drop the in-process half only. For tests that want to measure a disk read."""
    _MEMO.clear()
