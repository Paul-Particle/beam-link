"""The data contract: product dataclasses, the run spec, load and save.

This package imports NOTHING else in the repo, which is what keeps the import firewall
structural rather than a naming convention (SPEC.md §9). A consumer imports this and nothing
else; the engine imports it to build and write products.
"""

from beamlink.artifacts.manifest import (Manifest, ProductStatus, STATUSES, StudyManifest)
from beamlink.artifacts.products import (Aggregate, Aggregates, Aperture, Atmosphere, Capture,
                                         Chain, Field, FieldSampleSet, FlightPower, LinkGeometry,
                                         Mass, PRODUCT_TYPES, Power, Products, Reserve,
                                         Silhouette, Trajectory)
from beamlink.artifacts.spec import (AggregateDefinition, AirframeSpec, ApertureSpec,
                                     CandidateAxis, FIELD_SAMPLING_KINDS, FidelitySpec,
                                     FieldSampling, LoftSpec, POWER_REFERENCE_PLANES, PRODUCTS,
                                     PowerSpec, ProductRequest, ReceiverSpec, RelaySpec, RunSpec,
                                     SERVE_POLICIES, STATISTICS, ScenarioSpec, TrajectorySpec,
                                     WEATHER_PRESETS, Weather, default_axial_planes,
                                     default_products, weather)
from beamlink.artifacts.store import (MANIFEST_NAME, RunIncomplete, load_manifest, load_run,
                                      load_spec, load_study, require, save_run, save_study)

__all__ = [
    "Aggregate", "AggregateDefinition", "Aggregates", "AirframeSpec", "Aperture", "ApertureSpec",
    "Atmosphere", "CandidateAxis", "Capture", "Chain", "FIELD_SAMPLING_KINDS", "Field",
    "FieldSampleSet", "FieldSampling", "FidelitySpec", "FlightPower", "LinkGeometry", "LoftSpec",
    "MANIFEST_NAME", "Manifest", "Mass", "PRODUCTS", "PRODUCT_TYPES",
    "POWER_REFERENCE_PLANES", "Power", "PowerSpec", "ProductRequest", "ProductStatus", "Products",
    "ReceiverSpec", "RelaySpec", "Reserve", "RunIncomplete", "RunSpec", "SERVE_POLICIES",
    "STATISTICS", "STATUSES", "ScenarioSpec", "Silhouette", "StudyManifest", "Trajectory",
    "TrajectorySpec", "WEATHER_PRESETS", "Weather", "default_axial_planes", "default_products",
    "load_manifest", "load_run", "load_spec", "load_study", "require", "save_run", "save_study",
    "weather",
]
