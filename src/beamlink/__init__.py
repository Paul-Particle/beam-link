"""beamlink — a physics engine for beamed power to an aircraft, and the layers that read it.

The package layout IS the import firewall (SPEC.md §9). Nothing here re-exports anything, so
importing `beamlink` never drags physics into a consumer:

    beamlink.artifacts   the data contract — product dataclasses, the run spec, load and save.
                         Imports nothing else in this package.
    beamlink.physics     the engine. Imports artifacts/ to build and write products.
    beamlink.viz         may import artifacts/ only.
    beamlink.econ        may import artifacts/ only.
    beamlink.drivers     may invoke physics/ AND econ/; reimplements neither.

`tests/test_import_firewall.py` enforces that by walking the import graph, so a new file
inherits the rule from the directory it is placed in.
"""

__version__ = "0.2.0"
