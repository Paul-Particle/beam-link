"""Cost comparison against jet fuel. May import `artifacts/` only.

Econ SELECTS from products and turns them into money (§7, I4). It computes no physics: no beam
efficiency, no aperture sizing, no station spacing, no cruise power. Where the pre-SPEC econ had all
four, the corresponding numbers now arrive from named products, and a differently-defined efficiency
is a request for a new named engine product rather than a local calculation.
"""

from beamlink.econ.cost import EconConfig
from beamlink.econ.verdict import (REQUIRED_PRODUCTS, Selection, provenance, select, verdict,
                                   verdict_from_run)

__all__ = ["EconConfig", "REQUIRED_PRODUCTS", "Selection", "provenance", "select", "verdict",
           "verdict_from_run"]
