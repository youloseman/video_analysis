"""The dashboard as drawn on the Main board, short version (2026-11-17): a card
per sport with the latest frame and the readings, then one Fix first card."""
from __future__ import annotations

import re
from pathlib import Path

SPA = (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")


def _fn(name: str) -> str:
    m = re.search(r"function " + name + r"\(.*?\n}\n", SPA, re.S)
    assert m, f"{name} is gone"
    return m.group(0)


def test_one_fix_first_card_replaces_the_four_blocks():
    for gone in ('id="dashKino"', 'id="dashFocus"', 'id="dashPlan"', "function kinoCard(", "function renderDashPlan("):
        assert gone not in SPA
    assert 'id="dashFix"' in SPA and "renderDashFix(all);" in SPA


def test_the_sport_card_reads_the_same_rows_as_the_table():
    body = _fn("sportCard")
    assert "formTiles(sport, sameSide)" in body and "measStatus(m)" in body
    # a photo grades in words: the letter is the score's
    assert "GRADE_WORD[last.grade]?last.grade:scoreLetter(last.score)" in body


def test_a_bike_band_is_the_one_the_report_graded_against():
    """149 at the bottom of the stroke was in range on the report (non-drive
    margin) and out of range on the dashboard, which read the position table."""
    body = _fn("formBand")
    assert body.index("entry.trendOpt&&entry.trendOpt[key]") < body.index("bikePosRange(")


def test_the_lead_is_not_repeated_under_also_work_on():
    assert "leadT.includes(" in _fn("renderDashFix")


def test_then_now_and_kona_are_lines_on_home_and_pictures_on_progress():
    assert 'id="dashLinks"' in SPA and 'id="dashThen"' not in SPA and 'id="dashKona"' not in SPA
    assert 'id="progThen"' in SPA and 'id="progKona"' in SPA
    assert "renderThenNow(pool,'progThen',sport);" in SPA
    links = _fn("renderDashLinks")
    assert "data-goto" in links and "konaField()" in links


def test_all_measurements_fold_to_one_line():
    body = _fn("renderCurrentForm")
    assert '<details class="dcard dform"' in body and "All measurements" in body
    assert "state.formOpen=true" in SPA          # See all on a sport card opens it


def test_one_news_banner_and_one_coral_button():
    """The review news no longer carries a second filled button, and the swim
    news waits while a review is unread."""
    card = SPA[SPA.index("async function renderReviewCard("):SPA.index("/* ---- the athlete's copy ---- */")]
    assert "btn-primary" not in card
    assert "#dashReview .rev-card.is-unread" in _fn("renderDashSwim")
    assert "renderReviewCard().finally(renderDashSwim);" in SPA
