"""Drivers: sweeps, optimisation, and the orchestration between physics and econ.

The only layer that sees both sides, and it has to: an economic objective cannot be searched for in
physics space without evaluating econ, and exhaustive sweeps are affordable only in the simplest
cases (SPEC.md §9). A driver may invoke `physics/` and `econ/` and may reimplement neither.

It is also allowed to be SPECIFIC. Its workflow is dictated by the particular interplay of a given
econ objective and the physics it depends on, and it takes that specificity through its config — the
one place in the repo where "oddly specific" is the correct shape, and the reason the driver is
written last: its structure is a consequence of the finished engine and econ, not an input to either.

    sweep      one declared parameter, one batched engine call (D3)
    corridor   route-level numbers across several runs, with a study manifest (D6)
    calibrate  the fidelity error table §6 requires as part of the deliverable
"""
