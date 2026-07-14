"""Interactive widgets embeddable in Academy articles.

An article's Markdown can contain a standalone marker line, e.g.::

    [[widget:aero-calculator]]

The parser turns that into ``<div data-widget="aero-calculator"></div>``; the
renderer looks the name up in :data:`WIDGETS` and swaps in the widget's HTML
(with its own scoped CSS/JS inlined). Widgets are self-contained — no external
JS — so a page stays a single server-rendered document.

Design rule for data-backed widgets: the numbers a widget computes with must be
**traceable to a cited source**, not invented to make the widget look good. The
aero calculator below uses CdA values measured in one study (van Druenen 2023,
so they are mutually comparable) and standard, textbook cycling-power physics;
its on-screen caption says exactly that.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Aero position calculator
# ---------------------------------------------------------------------------
# CdA (m^2) per position — ALL from a single study so they are comparable:
#   van Druenen & Blocken (2023), Computers & Fluids 257, measured at 15 m/s.
#   DP-high = dropped posture, high (more upright) torso  -> 0.266
#   DP-low  = dropped posture, low torso                  -> 0.231
#   TTP     = time-trial posture                          -> 0.213
# The physics (P = ½·rho·CdA·v³ + Crr·m·g·v) is standard; rho/Crr are typical
# flat-road, sea-level constants surfaced in the caption. Rider+bike mass is a
# slider. Validated: our CdA ratios (100/87/80 %) reproduce the study's own
# reported relative-drag figures exactly.
_AERO_CALC = """
<figure class="aero-calc" data-widget-ready="1">
  <figcaption class="wcap">
    <span class="weyebrow">Interactive</span>
    Aero position calculator — how much your torso angle is worth
  </figcaption>

  <div class="wgrid">
    <div class="wcontrols">
      <label class="wctl">
        <span class="wlab">Position <b id="ac-posname">Drops · low torso</b></span>
        <input type="range" id="ac-pos" min="0" max="2" step="1" value="1">
        <span class="wscale"><span>Drops · high</span><span>Drops · low</span><span>Aero tuck</span></span>
      </label>

      <div class="wtabs" role="tablist" aria-label="What to hold fixed">
        <button id="ac-mode-power" role="tab" aria-selected="true">Fix power → see speed</button>
        <button id="ac-mode-speed" role="tab" aria-selected="false">Fix speed → see power</button>
      </div>

      <label class="wctl" id="ac-power-row">
        <span class="wlab">Your power <b><span id="ac-power-val">250</span> W</span></span>
        <input type="range" id="ac-power" min="120" max="400" step="5" value="250">
      </label>
      <label class="wctl wcollapsed" id="ac-speed-row">
        <span class="wlab">Your speed <b><span id="ac-speed-val">36</span> km/h</span></span>
        <input type="range" id="ac-speed" min="24" max="52" step="1" value="36">
      </label>

      <label class="wctl">
        <span class="wlab">Rider + bike <b><span id="ac-mass-val">78</span> kg</span></span>
        <input type="range" id="ac-mass" min="55" max="110" step="1" value="78">
      </label>
    </div>

    <div class="wout">
      <div class="wbig"><span id="ac-out-main">—</span><span class="wunit" id="ac-out-unit"></span></div>
      <div class="wsub" id="ac-out-sub">vs the most upright position</div>
      <div class="wcda">CdA <b id="ac-cda">0.231</b> m² · van Druenen 2023</div>
    </div>
  </div>

  <p class="wnote">
    Speed/power from standard cycling physics
    (<code>P = ½·ρ·CdA·v³ + C<sub>rr</sub>·m·g·v</code>), flat road, no wind,
    ρ = 1.225 kg/m³, C<sub>rr</sub> = 0.004. The <b>difference between positions</b>
    comes from measured CdA values in van Druenen &amp; Blocken (2023); absolute
    numbers are an illustrative estimate that shifts with your real rolling
    resistance, air density and drivetrain losses.
  </p>
</figure>
"""

_AERO_CALC_CSS = """
.aero-calc{margin:34px 0;border:1px solid var(--c-line);border-radius:var(--radius);
  background:#fff;box-shadow:var(--shadow);overflow:hidden}
.aero-calc .wcap{padding:18px 22px 0;font-family:var(--f-display);font-weight:800;
  text-transform:uppercase;font-size:17px;color:var(--c-navy);line-height:1.2}
.aero-calc .weyebrow{display:block;font-family:var(--f-body);font-size:11px;letter-spacing:.16em;
  color:var(--c-blue);margin-bottom:6px}
.aero-calc .wgrid{display:grid;grid-template-columns:1fr 260px;gap:22px;padding:20px 22px}
.aero-calc .wctl{display:block;margin-bottom:20px}
.aero-calc .wlab{display:flex;justify-content:space-between;align-items:baseline;
  font-size:13px;font-weight:600;color:var(--c-ink-soft);margin-bottom:8px}
.aero-calc .wlab b{font-weight:800;color:var(--c-navy);font-family:var(--f-mono);font-size:14px}
.aero-calc input[type=range]{width:100%;accent-color:var(--c-blue);cursor:pointer;height:22px}
.aero-calc .wscale{display:flex;justify-content:space-between;font-size:10px;
  color:var(--c-ink-faint);margin-top:2px;font-weight:600}
.aero-calc .wtabs{display:flex;gap:4px;padding:4px;background:var(--c-panel);border-radius:10px;margin-bottom:20px}
.aero-calc .wtabs button{flex:1;font-family:var(--f-body);font-weight:700;font-size:12.5px;
  color:var(--c-ink-soft);background:transparent;border:none;border-radius:7px;padding:9px 6px;cursor:pointer;transition:.15s}
.aero-calc .wtabs button[aria-selected=true]{background:#fff;color:var(--c-blue);box-shadow:var(--shadow)}
.aero-calc .wcollapsed{display:none}
.aero-calc .wout{background:linear-gradient(135deg,#14294B,#2F6DE0);color:#fff;border-radius:var(--radius);
  padding:22px 20px;display:flex;flex-direction:column;justify-content:center;text-align:center}
.aero-calc .wbig{font-family:var(--f-display);font-weight:900;font-style:italic;font-size:44px;line-height:1;letter-spacing:-.02em}
.aero-calc .wbig .wunit{font-size:17px;font-style:normal;font-weight:800;margin-left:6px;opacity:.85}
.aero-calc .wsub{font-size:13px;color:rgba(255,255,255,.82);margin-top:10px;font-weight:600}
.aero-calc .wcda{font-family:var(--f-mono);font-size:11px;color:rgba(255,255,255,.6);margin-top:14px;
  padding-top:12px;border-top:1px solid rgba(255,255,255,.18)}
.aero-calc .wcda b{color:#fff}
.aero-calc .wnote{font-size:12px;line-height:1.5;color:var(--c-ink-soft);
  background:var(--c-panel);margin:0;padding:14px 22px;border-top:1px solid var(--c-line)}
.aero-calc .wnote code{font-family:var(--f-mono);font-size:11px;background:#fff;padding:1px 5px;border-radius:4px;color:var(--c-navy)}
@media(max-width:640px){.aero-calc .wgrid{grid-template-columns:1fr}}
"""

_AERO_CALC_JS = """
(function(){
  var el=document.querySelector('.aero-calc[data-widget-ready]'); if(!el) return;
  // CdA per position — van Druenen & Blocken 2023 (single study, comparable).
  var POS=[{n:'Drops \\u00b7 high torso',cda:0.266},
           {n:'Drops \\u00b7 low torso',cda:0.231},
           {n:'Aero tuck (TT)',cda:0.213}];
  var RHO=1.225, CRR=0.004, G=9.81;
  var q=function(id){return el.querySelector(id);};
  var pos=q('#ac-pos'),power=q('#ac-power'),speed=q('#ac-speed'),mass=q('#ac-mass');
  var mode='power';
  function pForV(cda,v,m){return 0.5*RHO*cda*v*v*v + CRR*m*G*v;}          // watts, v in m/s
  function vForP(cda,P,m){var lo=0.1,hi=25,mid;for(var k=0;k<60;k++){mid=(lo+hi)/2;
    if(pForV(cda,mid,m)<P)lo=mid;else hi=mid;}return (lo+hi)/2;}          // m/s
  function render(){
    var i=+pos.value, m=+mass.value, cda=POS[i].cda, base=POS[0].cda;
    q('#ac-posname').textContent=POS[i].n;
    q('#ac-cda').textContent=cda.toFixed(3);
    q('#ac-mass-val').textContent=m;
    q('#ac-power-val').textContent=power.value;
    q('#ac-speed-val').textContent=speed.value;
    var main,unit,sub;
    if(mode==='power'){
      var P=+power.value;
      var v=vForP(cda,P,m)*3.6, v0=vForP(base,P,m)*3.6;
      main=v.toFixed(1); unit='km/h';
      var d=v-v0; sub=(i===0)?'the reference position':(d>=0?'+':'')+d.toFixed(1)+' km/h vs drops (high torso)';
    }else{
      var v2=(+speed.value)/3.6;
      var P1=pForV(cda,v2,m), P0=pForV(base,v2,m);
      main=Math.round(P1); unit='W';
      var dw=P1-P0, pct=P0?(100*dw/P0):0;
      sub=(i===0)?'the reference position':(dw>=0?'+':'')+Math.round(dw)+' W ('+(dw>=0?'+':'')+pct.toFixed(0)+'%) vs drops (high torso)';
    }
    q('#ac-out-main').textContent=main;
    q('#ac-out-unit').textContent=unit;
    q('#ac-out-sub').textContent=sub;
  }
  function setMode(mp){mode=mp;
    q('#ac-mode-power').setAttribute('aria-selected',mp==='power');
    q('#ac-mode-speed').setAttribute('aria-selected',mp==='speed');
    q('#ac-power-row').classList.toggle('wcollapsed',mp!=='power');
    q('#ac-speed-row').classList.toggle('wcollapsed',mp!=='speed');
    render();}
  [pos,power,speed,mass].forEach(function(s){s.addEventListener('input',render);});
  q('#ac-mode-power').addEventListener('click',function(){setMode('power');});
  q('#ac-mode-speed').addEventListener('click',function(){setMode('speed');});
  render();
})();
"""


# ---------------------------------------------------------------------------
# Crank-swap fit helper
# ---------------------------------------------------------------------------
# Pure geometry, no invented numbers: a shorter crank lowers the pedal's lowest
# point by the length difference, so the saddle should be RAISED by that same
# amount to keep the same leg extension. Reach also opens slightly, which the
# widget flags qualitatively. This directly serves the article's "re-set your
# fit after changing cranks" rule — the most common mistake when switching.
_CRANK_FIT = """
<figure class="crank-fit" data-widget-ready="1">
  <figcaption class="wcap">
    <span class="weyebrow">Interactive</span>
    Crank-swap fit helper — keep your leg extension when you go shorter
  </figcaption>
  <div class="wgrid">
    <div class="wcontrols">
      <label class="wctl">
        <span class="wlab">Current crank <b><span id="cf-old">172.5</span> mm</span></span>
        <input type="range" id="cf-old-r" min="160" max="180" step="2.5" value="172.5">
      </label>
      <label class="wctl">
        <span class="wlab">New crank <b><span id="cf-new">165</span> mm</span></span>
        <input type="range" id="cf-new-r" min="160" max="180" step="2.5" value="165">
      </label>
      <label class="wctl">
        <span class="wlab">Current saddle height <b><span id="cf-sh">730</span> mm</span></span>
        <input type="range" id="cf-sh-r" min="600" max="820" step="1" value="730">
      </label>
    </div>
    <div class="wout">
      <div class="wsmall">New saddle height</div>
      <div class="wbig"><span id="cf-out">737</span><span class="wunit">mm</span></div>
      <div class="wsub" id="cf-delta">raise the saddle 7 mm</div>
    </div>
  </div>
  <p class="wnote">
    Geometry only: a shorter crank lowers the bottom of the pedal stroke, so to
    keep the same leg extension you <b>raise the saddle by the crank difference</b>
    (and vice-versa). Your reach to the bars also opens a touch — re-check it. This
    is a starting point, not a substitute for a proper bike fit.
  </p>
</figure>
"""

_CRANK_FIT_CSS = """
.crank-fit{margin:34px 0;border:1px solid var(--c-line);border-radius:var(--radius);
  background:#fff;box-shadow:var(--shadow);overflow:hidden}
.crank-fit .wcap{padding:18px 22px 0;font-family:var(--f-display);font-weight:800;
  text-transform:uppercase;font-size:17px;color:var(--c-navy);line-height:1.2}
.crank-fit .weyebrow{display:block;font-family:var(--f-body);font-size:11px;letter-spacing:.16em;
  color:var(--c-blue);margin-bottom:6px}
.crank-fit .wgrid{display:grid;grid-template-columns:1fr 240px;gap:22px;padding:20px 22px}
.crank-fit .wctl{display:block;margin-bottom:20px}
.crank-fit .wlab{display:flex;justify-content:space-between;align-items:baseline;
  font-size:13px;font-weight:600;color:var(--c-ink-soft);margin-bottom:8px}
.crank-fit .wlab b{font-weight:800;color:var(--c-navy);font-family:var(--f-mono);font-size:14px}
.crank-fit input[type=range]{width:100%;accent-color:var(--c-blue);cursor:pointer;height:22px}
.crank-fit .wout{background:linear-gradient(135deg,#14294B,#2F6DE0);color:#fff;border-radius:var(--radius);
  padding:22px 20px;display:flex;flex-direction:column;justify-content:center;text-align:center}
.crank-fit .wsmall{font-size:12px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:rgba(255,255,255,.72)}
.crank-fit .wbig{font-family:var(--f-display);font-weight:900;font-style:italic;font-size:42px;line-height:1;letter-spacing:-.02em;margin-top:8px}
.crank-fit .wbig .wunit{font-size:16px;font-style:normal;font-weight:800;margin-left:6px;opacity:.85}
.crank-fit .wsub{font-size:13px;color:rgba(255,255,255,.85);margin-top:12px;font-weight:600}
.crank-fit .wnote{font-size:12px;line-height:1.5;color:var(--c-ink-soft);background:var(--c-panel);
  margin:0;padding:14px 22px;border-top:1px solid var(--c-line)}
@media(max-width:640px){.crank-fit .wgrid{grid-template-columns:1fr}}
"""

_CRANK_FIT_JS = """
(function(){
  var el=document.querySelector('.crank-fit[data-widget-ready]'); if(!el) return;
  var q=function(id){return el.querySelector(id);};
  var oldR=q('#cf-old-r'),newR=q('#cf-new-r'),shR=q('#cf-sh-r');
  function render(){
    var o=+oldR.value,n=+newR.value,sh=+shR.value;
    q('#cf-old').textContent=o.toFixed(1).replace('.0','');
    q('#cf-new').textContent=n.toFixed(1).replace('.0','');
    q('#cf-sh').textContent=sh;
    var diff=o-n;                 // shorter new crank (n<o) -> positive -> raise saddle
    var nsh=Math.round(sh+diff);
    q('#cf-out').textContent=nsh;
    var d=Math.round(diff*10)/10;
    var msg;
    if(Math.abs(d)<0.05) msg='no saddle change needed';
    else if(d>0) msg='raise the saddle '+Math.abs(d).toFixed(1).replace('.0','')+' mm';
    else msg='lower the saddle '+Math.abs(d).toFixed(1).replace('.0','')+' mm';
    q('#cf-delta').textContent=msg;
  }
  [oldR,newR,shR].forEach(function(s){s.addEventListener('input',render);});
  render();
})();
"""


# name -> {html, css, js}. The renderer inlines css/js once per page if the
# widget appears in the article body.
WIDGETS: dict[str, dict[str, str]] = {
    "aero-calculator": {
        "html": _AERO_CALC,
        "css": _AERO_CALC_CSS,
        "js": _AERO_CALC_JS,
    },
    "crank-fit": {
        "html": _CRANK_FIT,
        "css": _CRANK_FIT_CSS,
        "js": _CRANK_FIT_JS,
    },
}


def render_widgets(body_html: str) -> tuple[str, str, str]:
    """Replace ``<div data-widget="NAME"></div>`` placeholders with widget HTML.

    Returns ``(html, css, js)`` where css/js are the concatenated assets for
    every widget actually used (each included at most once).
    """
    used: list[str] = []
    for name in WIDGETS:
        needle = f'<div data-widget="{name}"></div>'
        if needle in body_html:
            body_html = body_html.replace(needle, WIDGETS[name]["html"])
            used.append(name)
    css = "".join(WIDGETS[n]["css"] for n in used)
    js = "".join(WIDGETS[n]["js"] for n in used)
    return body_html, css, js
