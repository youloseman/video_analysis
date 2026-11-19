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


def test_a_photo_letter_is_the_photo_scorers_own_lines():
    """The photo is graded in words (Excellent / Good / Fair / Needs Work); the
    letter beside the word must not say more than the word does."""
    from app.services.video_analysis.photo_analyzer import _assign_photo_grade
    m = re.search(r"const scoreLetter=n=>(.*?);", SPA)
    assert m, "scoreLetter missing"
    cuts = [int(x) for x in re.findall(r"n>=(\d+)", m.group(1))]
    words = {"A": "Excellent", "B": "Good", "C": "Fair"}
    for at, letter in zip(cuts, "ABC"):
        assert _assign_photo_grade(at) == words[letter]
        assert _assign_photo_grade(at - 1) != words[letter]


def test_the_photo_header_says_what_was_shot_and_the_pedal_sits_with_the_score():
    assert 'id="pResultMeta"' in SPA and 'id="pActsTop"' in SPA
    card = SPA.index('class="card scorecard sc-photo"')
    assert card < SPA.index('id="pPhasePick"') < SPA.index('id="pAnglesBlock"')
    assert "function photoAngleRow(" in SPA and "function angleRow(" not in SPA


def test_photo_findings_sit_beside_the_photo_and_stay_paid():
    """Exact angles are a paid read: a locked photo hides Findings with the table."""
    assert SPA.index('id="pImage"') < SPA.index('id="pFindings"') < SPA.index('id="pWarnings"')
    assert "['pCoachBlock','pAnglesBlock','pFindings']" in SPA
    body = SPA[SPA.index("function renderPhotoFindings("):]
    assert "if(res.locked || sc.withheld || sc.overall_score==null)" in body[:600]
    assert "renderPhotoFindings(A, phase, res);" in SPA


def test_the_analysing_card_promises_only_what_a_reload_keeps():
    """A reload picks the job up again from #job=; leaving for good saves nothing,
    so the card must not say the report will be in History (the board did)."""
    assert "If the page reloads, the analysis picks up where it was</p>" in SPA
    assert "the report will be in History" not in SPA
    assert "function jobMeta(parts)" in SPA and "fchip" not in SPA
    assert SPA.count('<span class="sline"></span>') == 3


def test_a_failed_clip_says_what_to_film_and_how_to_report_it():
    assert 'id="errReport"' in SPA and "rep.classList.toggle('hidden',!tips)" in SPA
    assert "title:'We could not read this clip'" in SPA


def test_the_pdf_ring_colours_on_the_apps_bands():
    """83 was green in the app and amber on paper: one score, two verdicts."""
    from app.services.review_pdf import _ring, grade_label
    m = re.search(r"function scoreStatus\(n\)\{.*?n>=(\d+)\?'good':n>=(\d+)\?'watch'", SPA)
    assert m, "scoreStatus moved"
    good, watch = int(m.group(1)), int(m.group(2))
    assert "#2E9E5B" in _ring(good, "B") and "#2E9E5B" not in _ring(good - 1, "B")
    assert "#C97B10" in _ring(watch, "C") and "#C97B10" not in _ring(watch - 1, "D")
    assert grade_label(83, "B") == "B · Good"
    assert grade_label(93, "Excellent") == "A · Excellent"   # a photo grades in words
