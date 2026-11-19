"""index.html carries no styling of its own: classes, tokens and data, nothing else.

On 2026-11-17 the SPA had 238 style="" attributes and 35 hard-coded colours that
bypassed tokens.css -- 80 of the attributes restated .hint's own text-align, and
two inline colours were dead under an !important. They moved to component
classes, a small u-* utility set on the --sp-* spacing scale, CSS custom
properties for values that come from data (--x, --w, --sw, --track), and
tok()/var(--token) for every colour the script draws with. These tests keep
them from drifting back.
"""
from __future__ import annotations

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "app" / "static"
HTML = (STATIC / "index.html").read_text(encoding="utf-8")
APP = (STATIC / "app.css").read_text(encoding="utf-8")
TOKENS = (STATIC / "tokens.css").read_text(encoding="utf-8")

# The one exception: the PDF opens in a blank window that has no stylesheet yet.
PRINT_WAIT = re.compile(r"^function printWaitHtml\(t\)\{.*$", re.M)
SANS_EXCEPTIONS = PRINT_WAIT.sub("", HTML)
# <meta name="theme-color"> must be a literal colour.
SANS_EXCEPTIONS = re.sub(r'<meta name="theme-color" content="#[0-9A-Fa-f]{6}">', "", SANS_EXCEPTIONS)

DATA_PROPS = {"--x", "--o", "--w", "--sw", "--track", "--used", "--pos"}   # the last two via setProperty; --o spreads timeline dots


def test_no_inline_style_but_data():
    """style="" may only hand data to CSS: custom properties the stylesheet reads."""
    bad = []
    for m in re.finditer(r'\sstyle="([^"]*)"', SANS_EXCEPTIONS):
        decls = [d.strip() for d in m.group(1).split(";") if d.strip()]
        for d in decls:
            name = d.split(":", 1)[0].strip()
            if name not in DATA_PROPS:
                bad.append(m.group(0)[:80])
                break
    assert not bad, f"inline styling in index.html (use a class or a u-* utility): {bad[:8]}"
    for prop in DATA_PROPS:
        assert f"var({prop}" in APP, f"{prop} is set in index.html but app.css never reads it"


def test_script_sets_geometry_not_looks():
    """JS may size and place things (canvas, popovers, progress); colour, visibility
    and type belong to classes -- toggle .hidden, not style.display."""
    allowed = {"width", "height", "left", "top", "maxWidth", "aspectRatio", "backgroundSize"}
    used = set(re.findall(r"\.style\.([a-zA-Z]+)\s*=(?!=)", HTML))
    assert used <= allowed, f"style properties set from JS: {sorted(used - allowed)}"
    assert "cssText" not in HTML


def test_no_colour_literals_in_the_spa():
    """Every colour comes from tokens.css: var(--token) in markup and SVG, tok() on canvas."""
    # the lookbehind skips element ids such as $('#adCdaF')
    hexes = re.findall(r"(?<!\$\(')#[0-9A-Fa-f]{6}\b|['\"]#[0-9A-Fa-f]{3}['\"]", SANS_EXCEPTIONS)
    assert not hexes, f"hex colours in index.html: {sorted(set(hexes))}"
    rgb = re.findall(r"rgba?\([^)]*\)", SANS_EXCEPTIONS)
    assert not rgb, f"rgb() colours in index.html: {sorted(set(rgb))}"


def test_every_token_the_spa_reads_exists():
    """A misspelt var(--token) fails silently -- the element just loses its colour."""
    defined = set(re.findall(r"(--[a-z0-9-]+)\s*:", TOKENS + APP))
    read = set(re.findall(r"var\((--[a-z0-9-]+)", HTML)) | set(re.findall(r"'(--[a-z0-9-]+)'", HTML))
    missing = sorted(read - defined - DATA_PROPS)
    assert not missing, f"index.html reads tokens nobody defines: {missing}"


def test_utilities_are_defined_and_on_the_scale():
    used = set(re.findall(r"\bu-[a-z0-9-]+\b", HTML))
    defined = set(re.findall(r"^\.(u-[a-z0-9-]+)\{", APP, re.M))
    assert used, "no utilities found -- did the class names change?"
    assert used <= defined, f"utilities used but not defined: {sorted(used - defined)}"
    assert not (defined - used), f"utilities defined but unused: {sorted(defined - used)}"
    for m in re.finditer(r"^\.u-(?:mt|mb|ml|mr|gap)-n?[1-8]\{([^}]*)\}", APP, re.M):
        assert re.search(r"var\(--sp-[1-8]\)", m.group(1)), m.group(0)
    assert "--sp-1:4px; --sp-2:8px; --sp-3:12px; --sp-4:16px; --sp-5:20px; --sp-6:24px; --sp-8:32px;" in TOKENS


def test_hidden_beats_every_utility():
    """.hidden must come after the utilities: an inline display:flex never beat it either."""
    util_end = max(m.end() for m in re.finditer(r"^\.u-[a-z0-9-]+\{", APP, re.M))
    hidden = [m.start() for m in re.finditer(r"^\.hidden\{display:none!important\}", APP, re.M)]
    assert hidden and hidden[-1] > util_end


def test_modifiers_come_after_their_base():
    """.x--y has the same weight as .x, so it only wins if it comes later. A modifier
    filed above its base silently loses (the sign-up eyebrow went grey that way)."""
    first = {}
    for m in re.finditer(r"^\.([a-z][a-z0-9-]*?)(--[a-z0-9-]+)?\{", APP, re.M):
        name, mod = m.group(1), m.group(2)
        if not mod:
            first.setdefault(name, m.start())
    late = []
    for m in re.finditer(r"^\.([a-z][a-z0-9-]*?)(--[a-z0-9-]+)\{", APP, re.M):
        base = first.get(m.group(1))
        if base is not None and base > m.start():
            late.append(m.group(0))
    assert not late, f"modifiers above their base rule: {late}"


def test_timeline_dots_that_would_overlap_are_spread():
    """History's strip: two scores filmed days apart shared a spot and one hid
    the other (audit 2026-11-18). They now spread by their own width."""
    fn = HTML[HTML.index("function histStripSpread"):][:900]
    assert "<6" in fn and "n-(k-1)/2" in fn
    assert "left:calc(var(--x) + var(--o,0) * 32px)" in APP


def test_progress_on_a_phone_folds_readings_but_keeps_the_charts():
    """Board ProgressPhone, inside Artur's ruling that the consulting charts
    stay: a phone shows a reading as a row that opens its full chart; Moving
    away keeps its chart open; wider screens are unchanged."""
    fn = HTML[HTML.index("function renderProgress"):][:12000]
    assert "phone&&cls!=='pg-reg'?row:card" in fn
    assert "trendChart2(x.series,x.opt,x.m.unit,{ts:x.tser,dec:FORM_DEC[x.m.key]})}</details>" in fn
    assert "function progSpark" in HTML


def test_audit_small_gaps():
    """Audit 2026-11-18, the small ones: History says when it is loading or did
    not load (never "0 analyses"), Compare's change can be copied, the CdA chip
    counts only like-for-like, old " -- " in plan text never reaches the screen."""
    assert "History did not load" in HTML and "data-histretry" in HTML
    assert "state.cloudSlow=true" in HTML and "300)" in HTML
    assert "Copy summary" in HTML and "state.cmpSummary=" in HTML
    fn = HTML[HTML.index("function cdaSinceChip"):][:600]
    assert "cdaLike(r,last)" in fn
    assert "${dash(esc(p.cue))}" in HTML and "${dash(esc(first.d))}" in HTML


def test_knee_speed_is_not_a_progress_trend():
    """Artur 2026-11-18: knee opening/closing speed (deg/s) reads as strange
    angles on Progress and mostly follows cadence and our filtering. It stays
    in Compare and the report, not in Progress."""
    fn = HTML[HTML.index("function renderProgress"):][:6000]
    assert "TREND_METRICS[sport].filter(m=>!m.noTrend)" in fn
    assert HTML.count("label:'Knee opening speed',unit:'°/s',noTrend:true") == 2


def test_the_kona_block_shows_our_range_beside_the_pros():
    """Artur 2026-11-18: the Kona block shows the optimal range too, on the
    same axis as the pro field, from the band the report grades against."""
    fn = HTML[HTML.index("function renderKona"):][:2500]
    assert "const opt=formBand('bike',k,e)" in fn and "konaStrip(v,f,opt)" in fn
    assert "Our range" in fn
    assert ".ks-opt{" in APP


def test_fix_first_reads_the_readings_when_the_report_says_nothing():
    """Artur 2026-11-18: advice showed for the bike only. A run with a reading
    out of range (or at the edge) but no written "Fix first" counted as clean,
    so the clean ride won. Readings now count, ranked fix > edge > clean."""
    fn = HTML[HTML.index("function dashFixTips"):][:2500]
    assert "level:'fix'" in fn and "level:'edge'" in fn and "is at the edge" in fn
    assert "const RANK={fix:2,edge:1,clean:0};" in HTML
