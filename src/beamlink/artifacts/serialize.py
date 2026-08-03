"""Generic dataclass ↔ (arrays, JSON) walker, so every product serialises the same way.

A product or spec object is a frozen dataclass whose fields are one of four things, and this
module is the whole reason the schema can grow without a matching pair of hand-written
save/load functions per product (which is how field sets drift out of sync):

    ndarray                      → one entry in the run's .npz, full precision, SI units
    JSON scalar / tuple / dict   → one entry in the manifest's metadata for that product
    a nested dataclass           → recursed, under a dotted key prefix
    Mapping[str, dataclass]      → recursed per key, with the key list recorded in metadata

Nothing here knows what a product means; it only knows the four shapes. Arrays and metadata
are kept in separate files on purpose: the .npz stays a pile of numbers a consumer can read
with numpy alone, and the manifest stays human-readable (SPEC.md §4 — "the manifest is the
contract").
"""

from __future__ import annotations

import collections.abc
import dataclasses
import typing
from typing import Any, Mapping

import numpy as np

# Fields whose value is a mapping of name → dataclass record the key order under this suffix,
# because a dict's keys are data (which named aggregates a run produced) and the loader has to
# rebuild the same mapping.
_KEYS_SUFFIX = "__keys__"


def is_record(obj: Any) -> bool:
    """A dataclass instance (not the class itself)."""
    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)


def _resolved_hints(cls) -> dict[str, Any]:
    """Field type hints with `from __future__ import annotations` strings resolved."""
    return typing.get_type_hints(cls)


def _is_mapping_hint(hint) -> bool:
    """Whether `hint` is a `Mapping[...]`/`dict[...]` of any flavour.

    `typing.get_origin(Mapping[str, X])` is `collections.abc.Mapping`, which is NOT a `dict`
    subclass — so a check written against `dict` silently misses every `Mapping`-annotated field, and
    the values come back as plain dicts instead of the dataclasses they were written from. That is how
    a study manifest's aggregation definitions lost their type on the way back off disk.
    """
    origin = typing.get_origin(hint)
    if origin is None:
        return False
    try:
        return issubclass(origin, collections.abc.Mapping)
    except TypeError:
        return False


def _mapping_value_type(hint) -> Any | None:
    """For a `Mapping[str, X]`/`dict[str, X]` hint where X is a dataclass, return X."""
    if not _is_mapping_hint(hint):
        return None
    args = typing.get_args(hint)
    if len(args) != 2:
        return None
    return args[1] if dataclasses.is_dataclass(args[1]) else None


def _optional_inner(hint) -> Any:
    """Unwrap `X | None` to X; anything else is returned unchanged."""
    if typing.get_origin(hint) is typing.Union:
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    # PEP 604 unions come back with types.UnionType as the origin
    args = typing.get_args(hint)
    if args and type(None) in args:
        rest = [a for a in args if a is not type(None)]
        if len(rest) == 1:
            return rest[0]
    return hint


def to_parts(record: Any, prefix: str = "") -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Split a dataclass into (arrays keyed by dotted path, JSON-safe metadata tree)."""
    arrays: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {}
    for field in dataclasses.fields(record):
        value = getattr(record, field.name)
        key = f"{prefix}{field.name}"
        if isinstance(value, np.ndarray):
            arrays[key] = value
        elif is_record(value):
            sub_arrays, sub_meta = to_parts(value, prefix=f"{key}.")
            arrays.update(sub_arrays)
            meta[field.name] = sub_meta
        elif isinstance(value, Mapping) and value and all(is_record(v) for v in value.values()):
            meta[field.name] = {_KEYS_SUFFIX: list(value.keys())}
            for name, item in value.items():
                sub_arrays, sub_meta = to_parts(item, prefix=f"{key}.{name}.")
                arrays.update(sub_arrays)
                meta[field.name][name] = sub_meta
        else:
            meta[field.name] = to_json_safe(value)
    return arrays, meta


def from_parts(cls, arrays: Mapping[str, np.ndarray], meta: Mapping[str, Any],
               prefix: str = "") -> Any:
    """Rebuild a dataclass of type `cls` from the two halves `to_parts` produced."""
    hints = _resolved_hints(cls)
    kwargs: dict[str, Any] = {}
    for field in dataclasses.fields(cls):
        key = f"{prefix}{field.name}"
        hint = _optional_inner(hints.get(field.name, Any))
        if key in arrays:
            kwargs[field.name] = arrays[key]
            continue
        sub_meta = meta.get(field.name)
        value_type = _mapping_value_type(hint)
        if value_type is not None and isinstance(sub_meta, Mapping) and _KEYS_SUFFIX in sub_meta:
            kwargs[field.name] = {
                name: from_parts(value_type, arrays, sub_meta[name], prefix=f"{key}.{name}.")
                for name in sub_meta[_KEYS_SUFFIX]}
        elif dataclasses.is_dataclass(hint) and isinstance(sub_meta, Mapping):
            kwargs[field.name] = from_parts(hint, arrays, sub_meta, prefix=f"{key}.")
        elif sub_meta is None and field.default is not dataclasses.MISSING:
            kwargs[field.name] = field.default
        else:
            kwargs[field.name] = from_json_safe(sub_meta, hint)
    return cls(**kwargs)


def to_json_safe(value: Any) -> Any:
    """numpy scalars, tuples and nested containers to something `json.dump` accepts."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_record(value):
        return {f.name: to_json_safe(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]
    return value


def from_json_safe(value: Any, hint: Any) -> Any:
    """Undo `to_json_safe` for the few container hints the schema uses.

    JSON has no tuple, so a `tuple[...]` field comes back as a list and has to be re-tupled —
    which matters because the specs are frozen dataclasses used as cache keys and compared for
    equality, and (1, 2) != [1, 2].
    """
    origin = typing.get_origin(hint)
    if value is None:
        return None
    if hint is np.ndarray:
        # Spec arrays (a trajectory's positions, attitudes and times) live in the manifest as
        # JSON lists so that "the spec verbatim" really is verbatim and self-contained; §6 rules
        # output size out as a design constraint, so a few tens of kB of numbers is free.
        return np.asarray(value, dtype=float)
    if origin is tuple:
        args = typing.get_args(hint)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(from_json_safe(v, args[0]) for v in value)
        if args:
            return tuple(from_json_safe(v, a) for v, a in zip(value, args))
        return tuple(value)
    if _is_mapping_hint(hint):
        args = typing.get_args(hint)
        value_hint = args[1] if len(args) == 2 else Any
        return {k: from_json_safe(v, value_hint) for k, v in value.items()}
    if dataclasses.is_dataclass(hint) and isinstance(value, Mapping):
        return from_parts(hint, {}, value)
    if isinstance(value, list):
        return tuple(value)          # unhinted sequences round-trip as tuples, never lists
    return value
