"""The saved report reads as the fresh one (2026-11-17)."""
from __future__ import annotations

from pathlib import Path

SPA = (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")


def test_the_saved_report_says_what_and_when_and_has_findings():
    assert 'id="hdHeadMeta"' in SPA and 'id="hdActsTop"' in SPA
    assert "function renderHistoryFindings(e)" in SPA and "renderHistoryFindings(e);" in SPA
    assert "formTiles(e.sport,[e])" in SPA[SPA.index("function renderHistoryFindings(e)"):]


def test_joint_angles_fold_and_open_for_print():
    assert 'id="hdAnglesFold"' in SPA
    assert "#hdAnglesFold:not([open])" in SPA


def test_a_coachs_review_is_one_column_with_what_the_athlete_gets():
    assert 'id="hdCoachWrap"' in SPA and "function renderCoachPreview()" in SPA
    place = SPA[SPA.index("function placeCoachNote(e){"):SPA.index("function goCoachNote(")]
    assert "main.append(rv)" in place and "main.append(acts)" in place
    # one filled button on a coach's clip: the one that sends
    assert "share.classList.toggle('btn-primary', !coach)" in place
