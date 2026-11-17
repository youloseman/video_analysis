"""The result's first screen as drawn on the Result board (2026-11-17): what was
filmed under the title, the grade in a word with what the next one takes, one
trust strip, and Findings rows that read on one line."""
from __future__ import annotations

import re
from pathlib import Path

from app.services.video_analysis.biomechanics.technique_scorer import assign_grade

SPA = (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")


def test_next_grade_thresholds_match_the_scorer():
    """"A from 90" is a promise; it has to be the scorer's own line."""
    m = re.search(r"const GRADE_NEXT=\{([^}]*)\}", SPA)
    assert m, "GRADE_NEXT missing"
    pairs = re.findall(r"(\w):\['(\w)',(\d+)\]", m.group(1))
    assert {p[0] for p in pairs} == {"F", "D", "C", "B"}
    for cur, nxt, at in pairs:
        at = int(at)
        assert assign_grade(at) == nxt, f"{nxt} from {at}, but the scorer says {assign_grade(at)}"
        assert assign_grade(at - 1) == cur, f"{at - 1} should still be {cur}"


def test_every_grade_has_a_word():
    m = re.search(r"const GRADE_WORD=\{([^}]*)\}", SPA)
    assert m and set(re.findall(r"(\w):'", m.group(1))) == {"A", "B", "C", "D", "F"}


def test_header_says_what_was_filmed_and_holds_the_actions():
    assert 'id="resultMeta"' in SPA and 'id="resultActsTop"' in SPA
    assert "function placeResultActs()" in SPA
    # a coach's actions are long and stay in the card
    assert "RESULT_WIDE.matches && !hasCoach" in SPA


def test_a_held_grade_is_said_once():
    assert "A needs ${capped.join(', ')} in range" in SPA
    assert "Object.assign({},r.score_coverage,{capped_by:[]})" in SPA


def test_one_trust_strip_folds_the_framing_note():
    assert 'class="sc-trust t-${level}"' in SPA
    assert "state.fwInfo=" in SPA
    # high confidence and a well-framed clip: nothing to say in the strip
    assert "if(level==='high' && !fw)" in SPA


def test_the_report_folds_on_a_phone_only():
    """ReportPhone board: each section one line until opened; desktop and paper
    keep every section open, so the fold rule is phone-width and screen-only."""
    css = (Path(__file__).resolve().parents[1] / "app" / "static" / "app.css").read_text(encoding="utf-8")
    assert "@media screen and (max-width:640px){" in css
    i = css.index("#results .block.foldable:not(.is-open) > :not(.fold-head){display:none}")
    assert css.rfind("@media screen and (max-width:640px){", 0, i) > css.rfind("}\n}", 0, i) - 1
    assert "foldReport(r, sport, state.verdict);" in SPA
    # what to do opens by itself; navigating to a section opens it
    assert "(id==='fitBlock'&&!!(v&&v.diag))||(id==='planBlock'&&!!(v&&v.drill))" in SPA
    assert SPA.count("openSection(el)") >= 2
    assert "track('result_section_opened'" in SPA
