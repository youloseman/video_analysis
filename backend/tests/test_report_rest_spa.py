"""The rest of the report as drawn on the Report board (2026-11-17): how the clip
was filmed beside Aero, aero's zone scale, and your data beside Coach Notes."""
from __future__ import annotations

import re
from pathlib import Path

from app.services.video_analysis.biomechanics.aero_estimator import AERO_ZONES

STATIC = Path(__file__).resolve().parents[1] / "app" / "static"
SPA = (STATIC / "index.html").read_text(encoding="utf-8")
CSS = (STATIC / "app.css").read_text(encoding="utf-8")


def test_the_zone_scale_is_the_estimators_own_zones_in_order():
    """The filled segment is a claim about the rider; it has to be the server's zone."""
    m = re.search(r"const AERO_ZONE_SHORT=\[(.*?)\];", SPA, re.S)
    assert m, "AERO_ZONE_SHORT missing"
    keys = re.findall(r"\['(\w+)','", m.group(1))
    assert keys == [z["key"] for z in AERO_ZONES]


def test_filming_sits_beside_aero():
    i = SPA.index('id="filmAero"')
    assert i < SPA.index('id="capRep"') < SPA.index('id="aeroBlock"')
    assert "capHost.innerHTML=renderCaptureReport(cr)" in SPA
    # open only where it has a partner on the row; alone it stays one line
    assert "function openFilmCard()" in SPA and "openFilmCard();" in SPA
    assert "#filmAero:has(.aero-hero) > *{flex-basis:100%}" in CSS


def test_your_data_sits_beside_coach_notes():
    i = SPA.index('id="rsEnd"')
    assert i < SPA.index('id="exportBlock"') < SPA.index('id="notesOffer"') < SPA.index('id="cmpNextV"')
    # one coral button on the screen: the export is not it
    assert 'class="btn btn-ghost btn-sm" id="exportCopy"' in SPA


def test_the_pairs_are_wide_screen_only():
    for sel in ("#filmAero{display:flex", "#rsEnd{display:flex"):
        i = CSS.index(sel)
        assert CSS.rfind("@media screen and (min-width:901px){", 0, i) > CSS.rfind("\n}", 0, i)
