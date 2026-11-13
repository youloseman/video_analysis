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


RADIUS_ROLES = {"--radius-card": "16px", "--radius-panel": "12px", "--radius-ctl": "8px",
                "--radius-tag": "6px", "--radius-bar": "2px"}


def test_radii_come_from_tokens():
    """By role (the 2026-11-13 redesign): card 16, panel and picture 12,
    control 8, tag 6, the range bar's marker and the Kona strip 2; pills
    999, dots 50%."""
    for tok, val in RADIUS_ROLES.items():
        assert f"{tok}:{val}" in TOKENS, tok
    literals = sorted(set(re.findall(r"border-radius:[^;}]*?\b([1-9]\d?px)", APP)))
    assert not literals, f"px radius literals in app.css: {literals}"
    # The app speaks in roles. The old names stay in tokens.css for the
    # Academy and inline styles, mapped onto the roles.
    old = sorted(set(re.findall(r"var\((--radius(?:-xs|-lg|-btn|-cta)?)\)", APP)))
    assert not old, f"app.css reads an old radius name: {old}"


def _radius_of(selector: str) -> str:
    m = re.search(r"(?:^|\})" + re.escape(selector) + r"\{[^}]*border-radius:var\((--radius-[a-z]+)\)", APP, re.M)
    assert m, selector
    return m.group(1)


def test_each_kind_of_thing_wears_its_role():
    for sel in (".card", ".block", ".dcard", ".modal-card", ".tier"):
        assert _radius_of(sel) == "--radius-card", sel
    for sel in (".btn", ".navlink", ".seg", ".fld input", ".toast"):
        assert _radius_of(sel) == "--radius-ctl", sel
    for sel in (".videoframe", ".hthumb", ".banner", ".issue"):
        assert _radius_of(sel) == "--radius-panel", sel
    # The range bar and the Kona strip keep their hairline corners (Artur, 2026-11-11).
    for sel in (".mbar-mark", ".ks-range", ".ks-mid"):
        assert _radius_of(sel) == "--radius-bar", sel


def test_headings_and_controls_follow_the_scale():
    assert "--fs-h1:40px; --fs-h2:24px;" in TOKENS
    assert ":root{--fs-h1:28px;--fs-h2:22px}" in APP
    for sel in ("h2.sec", ".uphead"):
        assert re.search(re.escape(sel) + r"\{[^}]*font-size:var\(--fs-h1\)", APP), sel
    for sel in (".blockhead h3", ".dcard-head h2"):
        assert re.search(re.escape(sel) + r"\{[^}]*font-size:var\(--fs-h2\)", APP), sel
    assert re.search(r"\.btn\{[^}]*min-height:44px", APP)
    assert "@media(pointer:fine){ .btn{min-height:40px} }" not in APP


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
