"""`report(run_dir, out_path)` — one self-contained page for one run.

The consumer contract in one function (§7): it imports one read module, handles any product being
absent and says so on the page, stamps provenance from the manifest, refuses loudly when told to show
something the run does not carry, and does no arithmetic beyond selection, unit presentation and
contouring.

    uv run python -m beamlink.viz.run_report runs/<id>
"""

from __future__ import annotations

import sys
from pathlib import Path

from beamlink import artifacts
from beamlink.viz import panels, provenance

# Every panel: which products it needs, and the builder. A panel whose products are absent renders
# as a statement of absence rather than being skipped — the page has to SAY what it could not show.
_PANELS = (
    ("pass profile", ("trajectory", "chain", "capture", "power", "geometry"),
     lambda p, m, c: panels.pass_profile(p, c)),
    ("silhouette with the beam on it", ("silhouette", "field", "capture"),
     lambda p, m, c: panels.silhouette_with_beam(p, c)),
    ("efficiency chain", ("chain",), lambda p, m, c: panels.chain_waterfall(p, c)),
    ("mass cascade", ("mass",), lambda p, m, c: panels.mass_cascade(p, c)),
    ("atmosphere", ("atmosphere", "geometry"), lambda p, m, c: panels.atmosphere_spectrum(p, c)),
    ("swept objective", ("aggregates",), lambda p, m, c: panels.candidate_objective(p, m)),
)

_STYLE = """
:root { color-scheme: light dark; }
body { font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
       margin: 0 auto; max-width: 1080px; padding: 2rem 1.25rem 5rem; }
h1 { font-size: 1.6rem; margin-bottom: .2rem; }
h2 { font-size: 1.15rem; margin-top: 2.4rem; }
h3 { font-size: 1.02rem; margin: 0 0 .6rem; }
.lede { color: #5f6368; margin-top: 0; }
.panel { margin: 2rem 0; }
.absent-panel { border-left: 3px solid #c9a227; padding: .6rem 0 .6rem 1rem; background: #fffbf0; }
.absent-panel h3 { color: #8a6d1a; }
.note, .note-row td { color: #5f6368; font-size: .88rem; }
.provenance { margin-top: 3rem; border-top: 1px solid #dadce0; padding-top: 1rem; }
.provenance dl { display: grid; grid-template-columns: 7rem 1fr; gap: .1rem .8rem; margin: 0 0 1rem; }
.provenance dt { color: #5f6368; }
.warn { background: #fff4e5; border-left: 3px solid #c9a227; padding: .5rem .8rem; }
table { border-collapse: collapse; width: 100%; font-size: .86rem; }
th, td { text-align: left; padding: .3rem .5rem; border-bottom: 1px solid #eceff1; vertical-align: top; }
th { color: #5f6368; font-weight: 600; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
code { font-size: .95em; }
.absent { color: #8a6d1a; }
@media (prefers-color-scheme: dark) {
  body { background: #16181c; color: #e8eaed; }
  .lede, .provenance dt, .note, .note-row td, th { color: #9aa0a6; }
  .absent-panel { background: #241f10; border-color: #c9a227; }
  .absent-panel h3, .absent { color: #f0c674; }
  .warn { background: #241f10; }
  th, td { border-color: #2a2d31; }
  .provenance { border-color: #2a2d31; }
}
"""


def report(run_dir: str | Path, out_path: str | Path | None = None, candidate: int = 0,
           channel_sets: tuple[str, ...] = ()) -> Path:
    """Build the page. `channel_sets` names P5 sample sets to draw as side views.

    Passing a set the run did not produce is a refusal, not a silent omission: a figure told to show a
    cut must not quietly show a different one.
    """
    run_dir = Path(run_dir)
    products, manifest = artifacts.load_run(run_dir)
    out_path = Path(out_path) if out_path else run_dir / "report.html"

    body = [f"<h1>{_escape(manifest.run_id)}</h1>",
            f"<p class='lede'>{_escape(manifest.notes) or 'Run report'} &mdash; "
            f"{manifest.n_candidates} candidate(s), {manifest.n_samples} samples, "
            f"engine {_escape(manifest.engine_sha[:12])}.</p>"]
    shown: list[str] = []
    first = True

    for title, needed, build in _PANELS:
        missing = [name for name in needed if products.get(name) is None]
        if missing:
            body.append(provenance.absent_panel(missing[0], manifest))
            continue
        figure = build(products, manifest, candidate)
        if figure is None:
            continue                       # a panel that has nothing to say for this run, e.g. no sweep
        shown.extend(needed)
        body.append(f"<section class='panel'><h3>{_escape(title)}</h3>"
                    + figure.to_html(include_plotlyjs="inline" if first else False,
                                     full_html=False, config={"displaylogo": False})
                    + "</section>")
        first = False

    if channel_sets:
        provenance.refuse(manifest, ("field",))
        available = set(products.field.sets)
        unknown = [name for name in channel_sets if name not in available]
        if unknown:
            raise artifacts.RunIncomplete(
                f"run {manifest.run_id} has no field sample set(s) {unknown}; it produced "
                f"{sorted(available)}. Ask for a run that sampled them rather than a different cut.")
        for name in channel_sets:
            figure = panels.beam_channel(products, name, candidate)
            shown.append("field")
            body.append(f"<section class='panel'><h3>Beam channel &mdash; {_escape(name)}</h3>"
                        + figure.to_html(include_plotlyjs="inline" if first else False,
                                         full_html=False, config={"displaylogo": False})
                        + "</section>")
            first = False

    if products.aggregates is not None:
        body.append(panels.aggregates_table(products, candidate))
        shown.append("aggregates")
    else:
        body.append(provenance.absent_panel("aggregates", manifest))

    body.append(provenance.margin(manifest, tuple(dict.fromkeys(shown))))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        f"<!doctype html><meta charset='utf-8'><title>{_escape(manifest.run_id)}</title>"
        f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<style>{_STYLE}</style>" + "".join(body))
    return out_path


def _escape(text) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m beamlink.viz.run_report <run_dir> [out.html]")
    target = report(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None,
                    channel_sets=("axial_major", "axial_minor"))
    print(f"wrote {target}")
