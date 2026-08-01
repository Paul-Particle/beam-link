"""Build a single self-contained HTML that embeds the FOUR emitter-size study animations
(relay/direct × large/small emitter, all in normal weather on a lateral pass) with a case
selector — the exact Plotly figures, one plotly.js shared, rendered on demand so only one
figure (its two WebGL scenes) is live at a time.

Lighter than the standalone out/ HTMLs (fewer frames + coarser panel-A grid) so all four fit
in one artifact; the full-resolution versions remain in out/ from `beam_link_animation.py`.

Run:  uv run python emitter_study_artifact.py   → out/emitter_study_artifact.html
"""
import json

import plotly.io as pio
import plotly.offline as po

import beam_link_animation as bla

# Kept deliberately light so all four 6-panel figures + one shared plotly.js fit under the
# 16 MB artifact ceiling. Spatial detail (grid) is prioritised over frame count — seeing the
# spot-size difference matters more than temporal smoothness, and the slider still scrubs.
# The full-resolution 81-frame versions live in out/ from beam_link_animation.py.
N_FRAMES = 15
GRID_NA = 82
OUT_PATH = "out/emitter_study_artifact.html"

# Dark-only by design: the embedded figures are plotly_dark, so the chrome commits to the
# same world. Neutrals carry a slight warm bias and the accent is drawn from the Plasma ramp
# the figures use (magenta→orange), so the UI reads as part of the same instrument.
_CSS = """
:root {
  color-scheme: dark;
  --ground:#111014; --panel:#1a181e; --panel-hi:#26232b;
  --ink:#ece8ee; --muted:#9b939f; --line:#413c48;
  --accent:#f2934a; --accent-2:#d6486f;   /* plasma orange / magenta */
}
* { box-sizing: border-box; }
body { margin:0; background:var(--ground); color:var(--ink);
       font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
.wrap { padding: 16px 20px 28px; }
.eyebrow { font-size:11px; letter-spacing:.18em; text-transform:uppercase; color:var(--accent);
           font-weight:600; margin-bottom:4px; }
h1 { font-size:21px; font-weight:650; margin:0 0 4px; text-wrap:balance; letter-spacing:-.01em; }
.sub { color:var(--muted); font-size:13.5px; line-height:1.5; margin:0 0 14px; max-width:70ch; }
.sub b { color:var(--ink); font-weight:600; }
.controls { display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:12px; }
.cases { display:flex; gap:8px; flex-wrap:wrap; }
.cases button, #playbtn { font: inherit; font-size:13px; cursor:pointer; border-radius:7px;
           padding:7px 14px; transition: background .12s, border-color .12s, color .12s; }
.cases button { background:var(--panel); color:#d7d2db; border:1px solid var(--line); }
.cases button:hover { background:var(--panel-hi); }
.cases button.active { background:var(--accent); color:#1a1206; border-color:var(--accent); font-weight:600; }
#playbtn { background:var(--panel); color:var(--ink); border:1px solid var(--line); }
#playbtn:hover { background:var(--panel-hi); }
:focus-visible { outline:2px solid var(--accent-2); outline-offset:2px; }
#plotwrap { overflow-x:auto; border-top:1px solid var(--line); padding-top:8px; }
.note { color:var(--muted); font-size:12px; margin-top:10px; }
"""

_JS = """
var FIGS = __FIGS__;
var div = document.getElementById('plotdiv');
var playBtn = document.getElementById('playbtn');
var casesEl = document.getElementById('cases');
var playing = false, cur = 0, timer = null, N = 0;

function frames() { return (div._transitionData && div._transitionData._frames) || []; }
function applyFrame(k) {
  var frs = frames(), fr = null;
  for (var i = 0; i < frs.length; i++) { if (frs[i] && frs[i].name === String(k)) { fr = frs[i]; break; } }
  if (!fr) return Promise.resolve();
  var proms = [];
  for (var j = 0; j < fr.data.length; j++) {
    var d = fr.data[j], tr = fr.traces ? fr.traces[j] : j, upd = {};
    for (var key in d) { if (d.hasOwnProperty(key) && key !== 'type') upd[key] = [d[key]]; }
    proms.push(Plotly.restyle(div, upd, [tr]));
  }
  cur = k;
  return Promise.all(proms);
}
function loop() { if (!playing) return; applyFrame((cur + 1) % N).then(function(){ if (playing) timer = setTimeout(loop, 12); }); }
function play() { if (playing || !N) return; playing = true; playBtn.textContent = '❚❚ pause'; loop(); }
function pause() { playing = false; if (timer) clearTimeout(timer); playBtn.textContent = '▶ play'; }
playBtn.onclick = function(){ playing ? pause() : play(); };

function render(i) {
  pause();
  var f = FIGS[i].fig;
  return Plotly.newPlot(div, f.data, f.layout, {responsive: false, displaylogo: false})
    .then(function(){ return Plotly.addFrames(div, f.frames); })
    .then(function(){
      N = f.frames.length; cur = 0;
      div.on('plotly_sliderchange', function(e){ if (e && e.slider) cur = e.slider.active | 0; });
    });
}
function setActive(i) { for (var c = 0; c < casesEl.children.length; c++) casesEl.children[c].className = (c === i ? 'active' : ''); }

FIGS.forEach(function(F, i) {
  var b = document.createElement('button');
  b.textContent = F.name;
  b.onclick = function(){ setActive(i); render(i); };
  casesEl.appendChild(b);
});
setActive(0);
render(0);
"""


def build():
    figs_js = []
    for label, arch, diam, _tag in bla.STUDY_CASES:
        fig = bla.build_flyby_animation(
            arch_name=arch, weather=bla.STUDY_WEATHER, lateral_offset_km=bla.STUDY_LATERAL_KM,
            emitter_diam_m=diam, n_frames=N_FRAMES, grid_nA=GRID_NA, write=False)
        figs_js.append('{"name": %s, "fig": %s}' % (json.dumps(label), pio.to_json(fig)))
    figs_array = "[\n" + ",\n".join(figs_js) + "\n]"

    body = (
        f"<style>{_CSS}</style>\n"
        '<div class="wrap">\n'
        '  <div class="eyebrow">Beam-link engine · flyby</div>\n'
        "  <h1>Emitter-size study</h1>\n"
        '  <div class="sub">Everything held fixed — normal (clear) weather, lateral pass '
        "(60 km offset), same delivered-power ceiling — and only the <b>emitter aperture</b> "
        "swept, for both architectures. The earlier 5.8-direct vs relay comparison confounded "
        "weather, frequency <i>and</i> a much larger direct dish; this isolates the emitter size, "
        "which dominates focusing (smaller spot → better capture).</div>\n"
        '  <div class="controls"><div class="cases" id="cases"></div>'
        '<button id="playbtn">▶ play</button></div>\n'
        '  <div id="plotwrap"><div id="plotdiv"></div></div>\n'
        '  <div class="note">Drag the slider to scrub the pass; ▶ play animates it. '
        "Each case re-renders on click (one figure live at a time).</div>\n"
        "</div>\n"
        f"<script>{po.get_plotlyjs()}</script>\n"
        f"<script>\n{_JS.replace('__FIGS__', figs_array)}\n</script>\n"
    )
    with open(OUT_PATH, "w") as fh:
        fh.write(body)
    import os
    print(f"wrote {OUT_PATH} ({os.path.getsize(OUT_PATH)/1e6:.1f} MB)")
    return OUT_PATH


if __name__ == "__main__":
    build()
