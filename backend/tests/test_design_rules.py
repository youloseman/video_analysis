"""The design system's acceptance table, as tests (brandbook-v3.html, section 13).

Each rule here was measured before it was written: ten hover lifts, eight ad-hoc
shadows, thirty-three radius literals, seven breakpoint widths, fifteen
keyframes on the landing. A generated interface is recognised by the absence of
decisions; these are the decisions, kept from drifting back.

Scope is what the pass covered: ``app.css`` and ``tokens.css`` for the app, and
the landing's motion. The landing keeps its own radius and shadow scale -- it is
a dark marketing page whose mockups sit on navy and need real depth -- and that
is declared in its own :root, which tokens.css explicitly allows.
"""
from __future__ import annotations

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "app" / "static"
APP = (STATIC / "app.css").read_text(encoding="utf-8")
LANDING = (STATIC / "landing.html").read_text(encoding="utf-8")
TOKENS = (STATIC / "tokens.css").read_text(encoding="utf-8")


def _rules_with(css: str, pattern: str) -> list[str]:
    return [m.group(0)[:80] for m in re.finditer(pattern, css)]


def test_hover_never_lifts_or_casts():
    """Hover is a colour step. The lift-with-shadow is the surest tell."""
    for name, css in (("app.css", APP), ("landing.html", LANDING)):
        lifts = _rules_with(css, r":hover\{[^}]*transform")
        assert not lifts, f"{name} lifts on hover: {lifts}"
        casts = _rules_with(css, r":hover\{[^}]*box-shadow")
        assert not casts, f"{name} adds a shadow on hover: {casts}"


def test_pressed_is_one_pixel_down():
    assert re.search(r"\.btn:active[^{]*\{transform:translateY\(1px\)\}", APP)


def test_shadows_come_from_the_two_tokens():
    """Outside @keyframes (a ring animation is not a shadow)."""
    css = re.sub(r"@keyframes[^{]*\{(?:[^{}]|\{[^{}]*\})*\}", "", APP)
    bad = [m.group(0)[:70] for m in re.finditer(r"box-shadow:(?!var\(--shadow|var\(--ring|none)[^;}]+", css)]
    assert not bad, f"ad-hoc shadows in app.css: {bad}"
    assert "--shadow:" in TOKENS and "--shadow-lg:" in TOKENS


def test_report_sections_carry_no_shadow():
    m = re.search(r"\n\.block\{[^}]*\}", APP)
    assert m and "box-shadow" not in m.group(0), "a report section is a rectangle on a page, not an object on a table"


def test_radii_come_from_tokens():
    """xs 4 / btn 8 / panel 10 / lg 14, pills 999, dots 50%."""
    for tok in ("--radius-xs:", "--radius-btn:", "--radius:", "--radius-lg:"):
        assert tok in TOKENS
    literals = sorted(set(re.findall(r"border-radius:[^;}]*?\b([1-9]\d?px)", APP)))
    assert not literals, f"px radius literals in app.css: {literals}"


def test_one_breakpoint_scale():
    """900 / 640 / 560, plus 1100 where the kinogram is wide enough to fit whole."""
    widths = sorted({int(w) for w in re.findall(r"@media ?\((?:max|min)-width:(\d+)px\)", APP)})
    assert set(widths) <= {560, 640, 900, 901, 1100}, widths


def test_transitions_are_named_and_short():
    assert "transition:all" not in APP, "transition:all animates layout by accident"
    assert "transition:.15s" not in APP
    durations = set(re.findall(r"transition:[^;}]*?(\.\d+s|\d+ms)", APP))
    # .3s: the analysis progress bar, the one movement that is allowed to be slow.
    assert durations <= {".15s", ".3s"}, durations


def test_the_landing_is_shown_at_rest():
    """No reveal-on-scroll, no burst, no count-up: the first frame is the page."""
    assert ".rv{opacity:0" not in LANDING and ".pop{scale:0" not in LANDING
    assert "data-count" not in LANDING
    assert "float-anim" not in LANDING and '<div class="orb' not in LANDING
    assert 'class="ticker"' not in LANDING


def test_landing_keyframes_show_process_not_decoration():
    """What stays: the recording dot, the live-overlay pulse, and the six frames
    of the how-it-works scanning demo. What left: fade/lift/grid/orb/float/
    marquee/ringspin -- animation that decorated arrival rather than showed
    something happening."""
    names = set(re.findall(r"@keyframes ([A-Za-z]+)", LANDING))
    assert names <= {"blink", "pulse", "tpScan", "tpRead", "tpOver", "tpDone", "tpRec", "tpCorners"}, names
