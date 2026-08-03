"""The aircraft: loft, mesh and silhouette table; flight power, the mass cascade and the reserve.

Inside the engine (SPEC.md D5), because it is physics and econ may not compute it. The two physics
domains stay independent — neither imports the other — and both use the aircraft geometry here: the
link to integrate capture over the silhouette, the airframe to size a rectenna from its area. The
dependency runs one way: `beam_link` imports `airframe`, never the reverse.
"""
