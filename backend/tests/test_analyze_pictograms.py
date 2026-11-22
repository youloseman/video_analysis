"""Analyze step "What are we analyzing?" (Artur 2026-11-21): the tiles carry the
house pictograms (the chunky rider and runner the metric icons are built from),
a chosen navy tile turns its glyphs light, and on a phone the compact library
slot keeps the viewfinder's speed-gradient frame instead of a grey dash."""
from __future__ import annotations

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "app" / "static"
SPA = (STATIC / "index.html").read_text(encoding="utf-8")
APP = (STATIC / "app.css").read_text(encoding="utf-8")


def _tile(what: str) -> str:
    m = re.search(rf'<button type="button" data-what="{what}".*?</button>', SPA, re.S)
    assert m, what
    return m.group(0)


def test_the_sprite_has_both_figures():
    for k in ("pic-bike", "pic-run"):
        assert f'<symbol id="{k}" viewBox=' in SPA


def test_video_and_photo_tiles_show_the_figure():
    assert '<use href="#pic-run"/>' in _tile("run-video")
    assert '<use href="#pic-bike"/>' in _tile("bike-video")
    for what, fig in (("bike-photo", "pic-bike"), ("run-photo", "pic-run")):
        t = _tile(what)
        assert f'<use href="#{fig}"/>' in t and '<use href="#i-camera"/>' in t and 'class="pic-photo"' in t


def test_a_chosen_tile_turns_its_glyphs_light():
    # The bike icon once sat navy (--c-head) on the navy chosen tile and vanished.
    rule = re.search(r'\.modetabs\.what button\[aria-pressed="true"\] \.ic,\s*'
                     r'\.modetabs\.what button\[aria-pressed="true"\] \.pic\{([^}]*)\}', APP)
    assert rule and "color:var(--c-on-dark)" in rule.group(1)
    # the far limbs are knocked out of the tile's own ground, not white
    assert "--c-bg:var(--c-brand-navy)" in rule.group(1)


def test_figure_classes_are_defined_and_prefixed():
    used = set()
    for body in re.findall(r'<symbol id="(?:pic|pm|pmb)-[a-z0-9_]+"[^>]*>(.*?)</symbol>', SPA, re.S):
        for cls in re.findall(r'class="([^"]*)"', body):
            used.update(cls.split())
    assert used and all(c.startswith("pg-") for c in used)
    for c in used:
        assert re.search(rf"\.{re.escape(c)}\{{", APP), c


def test_phone_library_slot_has_a_gradient_frame():
    rule = re.search(r"\.cliprow:not\(:has\(\.camrow\.hidden\)\) \.drop\{([^}]*)\}", APP)
    assert rule
    body = rule.group(1)
    assert "dashed" not in body
    assert "var(--g-speed) border-box" in body and "border:2px solid transparent" in body
