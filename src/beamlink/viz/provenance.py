"""Provenance and absence: the two things §7 makes non-negotiable for a page.

**Provenance is stamped from the manifest, never hand-written per figure.** Run id, engine sha, and
the fidelity and method of each product shown. That is the direct remedy for §1: a figure that says
"this is how the physics went" has to be able to say WHICH physics, at what fidelity, from which run —
and a caption typed by hand cannot, because it does not change when the run does.

**"Not generated in this run" is a first-class rendering, not an exception.** A consumer handles any
product being absent and says so on the page. The alternative — substituting a different product, or
computing the missing one locally — is exactly the second modelling pass §1 exists to prevent.
"""

from __future__ import annotations

from beamlink import artifacts


def margin(manifest: artifacts.Manifest, products_shown: tuple[str, ...]) -> str:
    """The provenance block, as HTML. Everything in it comes from the manifest."""
    rows = []
    for name in products_shown:
        status = manifest.status(name)
        if status is None:
            rows.append(f"<tr><td>{name}</td><td colspan='3' class='absent'>not requested</td></tr>")
            continue
        if status.status != "produced":
            rows.append(f"<tr><td>{name}</td><td colspan='3' class='absent'>"
                        f"{status.status}: {_escape(status.reason)}</td></tr>")
            continue
        inputs = ", ".join(status.inputs) or "—"
        rows.append(f"<tr><td>{name}</td><td>{status.tier}</td>"
                    f"<td>{_escape(status.method)}</td><td>{inputs}</td></tr>")
    tiers = {manifest.status(name).tier for name in products_shown
             if manifest.was_produced(name)}
    mixed = ("<p class='warn'>Mixed fidelity tiers in one page: "
             + ", ".join(sorted(tiers)) + ". I5 requires that this be detectable — it is.</p>"
             if len(tiers) > 1 else "")
    return f"""
<section class="provenance">
  <h2>Provenance</h2>
  <dl>
    <dt>run</dt><dd>{_escape(manifest.run_id)}</dd>
    <dt>engine</dt><dd>{_escape(manifest.engine_version)} @ {_escape(manifest.engine_sha[:12])}</dd>
    <dt>produced</dt><dd>{_escape(manifest.created_utc)}</dd>
    <dt>candidates</dt><dd>{manifest.n_candidates} &times; {manifest.n_samples} samples</dd>
    <dt>swept</dt><dd>{_escape(", ".join(manifest.candidate_axes) or "nothing")}</dd>
  </dl>
  {mixed}
  <table>
    <thead><tr><th>product</th><th>tier</th><th>method</th><th>derived from</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
</section>"""


def absent_panel(name: str, manifest: artifacts.Manifest) -> str:
    """What a panel renders when its product is not there. A statement, not a blank."""
    status = manifest.status(name)
    reason = status.reason if status is not None else "not requested in this run"
    verb = status.status if status is not None else "absent"
    return f"""
<section class="panel absent-panel">
  <h3>{_escape(name)} &mdash; not generated in this run</h3>
  <p>{_escape(verb)}: {_escape(reason)}</p>
  <p class="note">This page will not substitute another product or compute this one locally; that
  would be a second modelling pass presented as the model (SPEC.md &sect;1).</p>
</section>"""


def refuse(manifest: artifacts.Manifest, wanted: tuple[str, ...]) -> None:
    """Raise if the run lacks anything the caller was told to show (§7)."""
    missing = {name: (manifest.status(name).reason if manifest.status(name) else "not requested")
               for name in wanted if not manifest.was_produced(name)}
    if missing:
        detail = "; ".join(f"{name}: {reason}" for name, reason in missing.items())
        raise artifacts.RunIncomplete(
            f"run {manifest.run_id} cannot support this figure — {detail}. Point the figure at a run "
            f"that produced these products, or ask for a figure that does not need them.")


def _escape(text) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
