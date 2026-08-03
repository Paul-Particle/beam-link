"""Figures. May import `artifacts/` only.

A consumer: it SELECTS from products, presents units, and contours emitted fields — and does nothing
else (§7). It handles any product being absent and says so on the page, stamps provenance from the
manifest rather than from a hand-written caption, and refuses loudly when pointed at a run that
lacks what it was told to show.
"""

from beamlink.viz.run_report import report

__all__ = ["report"]
