"""Regenerate the committed ITU table the atmosphere model reads. Needs `--extra generators`.

    uv run --extra generators python -m beamlink.physics.beam_link.tables.generate_atmosphere_profile

One output, `beam_link/data/atmosphere_profile.npz`, and P10 emits both the per-sample loss and the
zenith spectrum from it — so there is one table and one place it is interpreted. The pre-SPEC repo also
committed a separate specific-attenuation CSV for a standalone figure to read; that figure is now P10's
panel in the run report, so the CSV and its generator are in `archive/pre-spec/` rather than being a
second copy of the same physics on disk.

These scripts are engine-internal but not engine code: they call ITU-R reference models offline and
write numbers. Nothing imports them at run time, which is why `itur` and `astropy` are an optional
extra and not dependencies.
"""
