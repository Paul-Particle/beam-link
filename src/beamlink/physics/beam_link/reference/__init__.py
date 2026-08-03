"""Independent ground truth for the closed forms, exercised by the test suite only.

A second implementation on purpose (SPEC.md §2): a quasi-discrete Hankel transform on a disc, sharing
no code with the FFT/DFT propagation on a square grid that the engine runs on. That is what makes the
check on the Goubau coupling law a check rather than a tautology.
"""
