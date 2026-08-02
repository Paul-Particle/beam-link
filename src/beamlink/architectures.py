"""The named architectures the study compares, as `LinkConfig` keyword sets.

This is the catalogue of inputs, not a model: adding a case to compare means adding an entry
here. It deliberately imports nothing — the keywords are handed to `beam_link.LinkConfig` by
whoever runs them (`sweeps.corridor_table`, the econ driver, the flyby animation).
"""

from __future__ import annotations


def standard_architectures(station_pos=(0.0, 0.0, 0.0), weather="clear", pae_scenario="today",
                           relay_alt_m=20_000.0, tether_alt_m=20_000.0,
                           emitter_5p8_m=78.0, relay_aperture_m=40.0, emitter_tether_m=15.0,
                           relay_multiplier_out_ghz=17.4, include_reference=True):
    """The architectures compared throughout, as LinkConfig keyword sets.

    Emitter and relay diameters are fixed inputs here, not sized from the geometry, so the
    same aperture is flown across a whole pass. `include_reference` adds two further cases
    kept for comparison rather than proposed.
    """
    relay = (station_pos[0], station_pos[1], relay_alt_m)
    tether = (station_pos[0], station_pos[1], tether_alt_m)
    common = dict(weather=weather, pae_scenario=pae_scenario)
    contenders = {
        "5.8 direct": dict(station_pos=station_pos, f_emit_plane_ghz=5.8, emitter_diam_m=emitter_5p8_m),
        f"relay ×{round(relay_multiplier_out_ghz/5.8)} mult 5.8→{relay_multiplier_out_ghz:g}":
            dict(station_pos=station_pos, relay_pos=relay, relay_mode="multiplier",
                 f_emit_relay_ghz=5.8, f_relay_plane_ghz=relay_multiplier_out_ghz,
                 relay_aperture_diam_m=relay_aperture_m),
        "tether 30 direct": dict(station_pos=tether, f_emit_plane_ghz=30.0, emitter_diam_m=emitter_tether_m),
    }
    reference = {
        "30 direct (ground)": dict(station_pos=station_pos, f_emit_plane_ghz=30.0, emitter_diam_m=15.0),
        "relay active 5.8→30": dict(station_pos=station_pos, relay_pos=relay, relay_mode="active",
                                    f_emit_relay_ghz=5.8, f_relay_plane_ghz=30.0,
                                    relay_aperture_diam_m=relay_aperture_m),
    }
    archs = {**contenders, **(reference if include_reference else {})}
    return {name: {**common, **kw} for name, kw in archs.items()}
