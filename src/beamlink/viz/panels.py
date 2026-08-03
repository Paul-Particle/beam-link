"""Panels: one figure each, built by SELECTING from products.

The rule every function here obeys (§7): no arithmetic beyond selection, unit presentation, and
contouring or scaling of emitted fields. Concretely —

  units          W/m² → W/cm², J → GJ, m → km. Presentation.
  scaling        a log colour axis over an emitted field. Presentation.
  contouring     `contourpy` over P5's emitted `enclosed_fraction`. Explicitly permitted, and the
                 reason the engine emits that field instead of returning polylines (§8 item 15).
  selection      picking a candidate, a sample, a named factor, a named aggregate.

What is NOT here, and used to be: a camera basis, positions pre-projected into a plotting frame, a
which-way-up flag, a geometric fill so the image has no hole, a regulatory level baked into a
boundary, and a peak-normalised intensity. The frames P4 emits place every cut in space; the validity
masks say where the method could not reach; and the exposure limit is an overlay this module draws on
an absolute field, which is what makes it a viewer's line rather than the model's.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from beamlink import artifacts

# ICNIRP 2020 / FCC OET-65 uncontrolled / IEEE C95.1-2019 all land on 10 W/m² above ~2 GHz. It lives
# HERE, in the viewer, as an overlay on an emitted absolute field — never inside a product (§8 item 16).
PUBLIC_EXPOSURE_LIMIT_W_M2 = 10.0

_TEMPLATE = "plotly_white"


def pass_profile(products: artifacts.Products, candidate: int = 0) -> go.Figure:
    """The shape of a pass: efficiency, capture, delivered power and elevation against along-track.

    Four selections from four products, on one along-track axis taken from P1. The pre-SPEC panel D
    plotted an efficiency curve that no other output used; this plots P7's η, which IS the number P8
    and P9 are built from, so the curve and the headline cannot disagree.
    """
    along_km = products.trajectory.positions_m[:, 0] / 1e3
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=along_km, y=products.chain.eta_end_to_end[candidate],
                                name="η end-to-end", mode="lines", yaxis="y"))
    figure.add_trace(go.Scatter(x=along_km, y=products.capture.incidence_weighted[candidate],
                                name="capture (incidence-weighted)", mode="lines", yaxis="y"))
    figure.add_trace(go.Scatter(x=along_km,
                                y=products.power.rectenna_terminals_w[candidate] / 1e6,
                                name="delivered at rectenna terminals (MW)", mode="lines",
                                yaxis="y2"))
    figure.add_trace(go.Scatter(x=along_km, y=products.geometry.elevation_deg[candidate, -1],
                                name=f"elevation, {products.geometry.leg_names[-1]} (deg)",
                                mode="lines", line=dict(dash="dot"), yaxis="y3"))
    figure.update_layout(
        template=_TEMPLATE, height=430, title="Pass profile",
        xaxis=dict(title="along-track (km)"),
        yaxis=dict(title="fraction", rangemode="tozero", domain=[0.0, 1.0]),
        yaxis2=dict(title="MW", overlaying="y", side="right", rangemode="tozero"),
        yaxis3=dict(overlaying="y", side="right", position=1.0, showticklabels=False,
                    rangemode="tozero"),
        legend=dict(orientation="h", y=-0.25))
    return figure


def silhouette_with_beam(products: artifacts.Products, candidate: int = 0, sample: int | None = None
                         ) -> go.Figure:
    """The surface the capture integral summed over, with the field on it.

    The elements are P3's own — the mesh AS INTEGRATED — and the colour is P5's absolute intensity at
    exactly those elements. Because they are the same solve and the same surface, the picture cannot be
    a lookalike of the integral; it is the integral. The frame is fixed by P3's emitted basis, and the
    scale by its `max_half_extent_m`, which is an airframe property, so silhouettes from different runs
    are comparable.
    """
    sample = _peak_sample(products, candidate) if sample is None else sample
    offsets = products.silhouette.element_offsets_m[candidate, sample]
    valid = products.silhouette.element_valid[candidate, sample]
    sample_set = products.field.sets["silhouette"]
    intensity = sample_set.intensity_w_m2[candidate, sample]
    resolved = valid & sample_set.valid[candidate, sample]
    extent = products.silhouette.max_half_extent_m * 1.05

    figure = go.Figure()
    figure.add_trace(go.Scattergl(
        x=offsets[resolved, 0], y=offsets[resolved, 1], mode="markers",
        marker=dict(size=3, color=intensity[resolved] / 1e4, colorscale="Inferno",
                    colorbar=dict(title="W/cm²"), cmin=0.0),
        name="illuminated elements",
        hovertemplate="x %{x:.1f} m<br>y %{y:.1f} m<br>%{marker.color:.2f} W/cm²<extra></extra>"))
    unresolved = valid & ~sample_set.valid[candidate, sample]
    if unresolved.any():
        figure.add_trace(go.Scattergl(
            x=offsets[unresolved, 0], y=offsets[unresolved, 1], mode="markers",
            marker=dict(size=3, color="#9aa0a6", symbol="x"),
            name="unresolved by the method"))
    area = products.silhouette.projected_area_m2[candidate, sample]
    capture = products.capture.incidence_weighted[candidate, sample]
    figure.update_layout(
        template=_TEMPLATE, height=430,
        title=(f"Silhouette as integrated — {area:.0f} m², capture {100 * capture:.2f} %"
               f" (sample {sample})"),
        xaxis=dict(title="x, aperture major axis (m)", range=[-extent, extent],
                   constrain="domain"),
        yaxis=dict(title="y (m)", range=[-extent, extent], scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", y=-0.2))
    return figure


def beam_channel(products: artifacts.Products, set_name: str = "axial_major", candidate: int = 0,
                 sample: int | None = None, enclosed_levels=(0.5, 0.8, 0.95),
                 show_exposure_limit: bool = True) -> go.Figure:
    """The beam along the path, as a side view in absolute W/m².

    One of the aperture's own principal planes (D2), so the cut is physically defined rather than
    being the silhouette's axes in a display basis. Three things this draws that are all selection or
    presentation:

      the heatmap        log10 of P5's absolute intensity, with the unresolved region left blank
                         because P5 says it is unresolved and this page does not fill it in
      the funnel         contours of P5's emitted `enclosed_fraction`, drawn with `contourpy`
      the exposure line  a contour of the emitted field at 10 W/m², which is a VIEWER's overlay on
                         absolute data, not a level baked into a product (§8 item 16)
    """
    sample = _peak_sample(products, candidate) if sample is None else sample
    sample_set = products.field.sets[set_name]
    n_axial, n_radial = sample_set.grid_shape
    points = sample_set.points_m[candidate, sample].reshape(n_axial, n_radial, 3)
    intensity = sample_set.intensity_w_m2[candidate, sample].reshape(n_axial, n_radial)
    valid = sample_set.valid[candidate, sample].reshape(n_axial, n_radial)
    radial_axis = points[0, :, 0] if np.ptp(points[0, :, 0]) > 0 else points[0, :, 1]
    range_km = points[:, 0, 2] / 1e3

    shown = np.where(valid & (intensity > 0), intensity, np.nan)
    figure = go.Figure()
    figure.add_trace(go.Heatmap(
        x=radial_axis, y=range_km, z=np.log10(shown), colorscale="Inferno",
        colorbar=dict(title="log₁₀ W/m²"), hoverongaps=False,
        hovertemplate="r %{x:.0f} m<br>range %{y:.1f} km<br>10^%{z:.2f} W/m²<extra></extra>"))

    if sample_set.enclosed_fraction is not None and enclosed_levels:
        enclosed = sample_set.enclosed_fraction[candidate, sample].reshape(n_axial, n_radial)
        for level, line in zip(enclosed_levels, ("dot", "dash", "solid")):
            for path in _contours(radial_axis, range_km, np.where(valid, enclosed, np.nan), level):
                figure.add_trace(go.Scatter(
                    x=path[:, 0], y=path[:, 1], mode="lines", showlegend=path is None,
                    line=dict(color="#9ecbff", width=1, dash=line),
                    name=f"{100 * level:.0f} % enclosed", legendgroup=f"enc{level}",
                    hoverinfo="skip"))
    if show_exposure_limit:
        for path in _contours(radial_axis, range_km, shown, PUBLIC_EXPOSURE_LIMIT_W_M2):
            figure.add_trace(go.Scatter(x=path[:, 0], y=path[:, 1], mode="lines",
                                        line=dict(color="#ffb74d", width=1.5),
                                        name=f"{PUBLIC_EXPOSURE_LIMIT_W_M2:.0f} W/m² public limit",
                                        legendgroup="limit", hoverinfo="skip"))
    slant_km = products.geometry.slant_m[candidate, -1, sample] / 1e3
    figure.add_hline(y=slant_km, line=dict(color="white", width=1),
                     annotation_text="aircraft", annotation_position="top left")
    occluded = sample_set.sampling.occluders == "receiver"
    figure.update_layout(
        template=_TEMPLATE, height=560,
        title=(f"Beam channel — {set_name}, aperture {sample_set.sampling.axis} plane"
               f"{', receiver shadow applied' if occluded else ''} (sample {sample})"),
        xaxis=dict(title="transverse (m)"),
        yaxis=dict(title="range from the aperture (km)"),
        legend=dict(orientation="h", y=-0.18))
    return figure


def chain_waterfall(products: artifacts.Products, candidate: int = 0, sample: int | None = None
                    ) -> go.Figure:
    """The efficiency chain at one sample, from P7's own factor names and values.

    No label mirror. The pre-SPEC waterfall read display strings from a hand-maintained copy of the
    chain that "has to be mirrored" on every change (§8 item 3); these labels ARE the product's
    `factor_names`, so a factor added to the chain appears here without anyone remembering to add it.
    """
    sample = _peak_sample(products, candidate) if sample is None else sample
    names = list(products.chain.factor_names)
    values = products.chain.factors[candidate, :, sample]
    eta = products.chain.eta_end_to_end[candidate, sample]
    cumulative = np.cumprod(values)
    figure = go.Figure()
    figure.add_trace(go.Bar(x=names, y=values, name="factor",
                            text=[f"{v:.3f}" for v in values], textposition="outside"))
    figure.add_trace(go.Scatter(x=names, y=cumulative, name="cumulative", mode="lines+markers",
                                line=dict(color="#d62728")))
    figure.update_layout(
        template=_TEMPLATE, height=430,
        title=f"Efficiency chain at sample {sample} — η end-to-end {eta:.4f}",
        yaxis=dict(title="factor / cumulative", range=[0, 1.08]),
        xaxis=dict(tickangle=-30), legend=dict(orientation="h", y=-0.35))
    return figure


def aggregates_table(products: artifacts.Products, candidate: int = 0) -> str:
    """P9 as HTML, with each number's DEFINITION beside it.

    The definition is the point. §1's two pass efficiencies were indistinguishable because neither
    recorded its span, serve policy, statistic or sampling; here both appear, both labelled, and a
    reader can see why they differ.
    """
    rows = []
    for name, entry in products.aggregates.entries.items():
        definition = entry.definition
        policy = definition.serve_policy
        if policy == "eta_floor":
            policy += f" ≥ {definition.serve_eta_floor:g}"
        served = 100.0 * entry.served[candidate].mean()
        rows.append(
            f"<tr><td><code>{name}</code></td><td class='num'>{entry.value[candidate]:.6g}</td>"
            f"<td>{definition.unit or '—'}</td><td>{definition.statistic}</td>"
            f"<td>{_escape(definition.span)}</td><td>{policy}</td>"
            f"<td class='num'>{served:.0f} %</td><td>{definition.weighting}</td>"
            f"<td>{', '.join(definition.inputs)}</td></tr>"
            f"<tr class='note-row'><td colspan='9'>{_escape(definition.note)}</td></tr>")
    return f"""
<section class="panel">
  <h3>Named aggregates (P9)</h3>
  <table class="aggregates">
    <thead><tr><th>name</th><th>value</th><th>unit</th><th>statistic</th><th>span</th>
      <th>serve policy</th><th>served</th><th>weighting</th><th>from</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
</section>"""


def mass_cascade(products: artifacts.Products, candidate: int = 0) -> go.Figure:
    """P11 as a stacked bar. The rectenna share is derived from P3's area, not assumed (D5)."""
    parts = [("payload", products.mass.payload_t), ("structure", products.mass.structure_t),
             ("systems", products.mass.systems_t), ("motors", products.mass.motor_t),
             ("reserve battery", products.mass.reserve_battery_t),
             ("rectenna", products.mass.rectenna_t)]
    figure = go.Figure()
    for name, values in parts:
        figure.add_trace(go.Bar(x=[name], y=[values[candidate]], name=name,
                                text=[f"{values[candidate]:.1f} t"], textposition="outside"))
    figure.update_layout(
        template=_TEMPLATE, height=380, showlegend=False,
        title=(f"Mass cascade — MTOW {products.mass.mtow_t[candidate]:.1f} t, "
               f"empty fraction {products.mass.empty_fraction[candidate]:.2f}, "
               f"{products.mass.iterations} lockstep iterations"),
        yaxis=dict(title="tonnes"))
    return figure


def atmosphere_spectrum(products: artifacts.Products, candidate: int = 0) -> go.Figure:
    """P10's own spectrum: zenith loss against frequency, under this run's weather.

    The atmospheric-attenuation figure used to be a standalone script that re-read the ITU table. It is
    a product now, so the figure shows the spectrum the run's per-sample numbers actually came from.
    """
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=products.atmosphere.spectrum_frequency_ghz,
                                y=products.atmosphere.spectrum_zenith_loss_db[candidate],
                                mode="lines", name="zenith loss"))
    for leg, name in enumerate(products.geometry.leg_names):
        loss = products.atmosphere.loss_db[candidate, leg]
        figure.add_trace(go.Scatter(
            x=[products.atmosphere.spectrum_frequency_ghz.min()], y=[loss.min()], mode="markers",
            marker=dict(size=9, symbol="diamond"),
            name=f"{name}: {loss.min():.2f}–{loss.max():.2f} dB actually applied"))
    figure.update_layout(
        template=_TEMPLATE, height=380,
        title=f"Atmosphere — {_escape(products.atmosphere.profile_source)}",
        xaxis=dict(title="frequency (GHz)", type="log"),
        yaxis=dict(title="zenith loss to cruise altitude (dB)", type="log"),
        legend=dict(orientation="h", y=-0.3))
    return figure


def candidate_objective(products: artifacts.Products, manifest: artifacts.Manifest,
                        aggregate_name: str = "delivered_energy_j") -> go.Figure | None:
    """A swept parameter against a named aggregate. None when the run swept nothing."""
    if not manifest.candidate_axes:
        return None
    entry = products.aggregates.entries.get(aggregate_name)
    if entry is None:
        return None
    parameter, values = next(iter(manifest.candidate_axes.items()))
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=[str(v) for v in values], y=entry.value, mode="lines+markers"))
    figure.update_layout(template=_TEMPLATE, height=380,
                         title=f"{aggregate_name} against {parameter}",
                         xaxis=dict(title=parameter),
                         yaxis=dict(title=f"{aggregate_name} ({entry.definition.unit or '—'})"))
    return figure


# --- plumbing -----------------------------------------------------------------------------


def _peak_sample(products: artifacts.Products, candidate: int) -> int:
    """The best-served sample. A selection, so the page says which one it picked."""
    if products.chain is not None:
        return int(np.argmax(products.chain.eta_end_to_end[candidate]))
    return 0


def _contours(x_axis, y_axis, values, level):
    """Contour paths of an emitted field. Contouring is the viewer's (§7, §8 item 15)."""
    from contourpy import contour_generator
    finite = np.where(np.isfinite(values), values, -np.inf)
    generator = contour_generator(x=x_axis, y=y_axis, z=finite, name="serial",
                                  fill_type="OuterCode", line_type="SeparateCode")
    lines, _ = generator.lines(level)
    return [np.asarray(path) for path in lines if len(path) > 1]


def _escape(text) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
