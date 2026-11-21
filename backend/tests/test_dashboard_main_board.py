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


def test_what_to_fix_is_a_checklist_per_sport_under_the_cards():
    """Artur 2026-11-19: the latest analyses on top, then what to fix and how,
    sport by sport (the one with something out of range first), each fix and
    drill a checkbox that strikes through and stays ticked."""
    assert SPA.index('id="dashSports"') < SPA.index('<div id="dashFix"></div>')
    body = _fn("renderDashFix")
    assert 'class="fx-cb" data-fx="${i.fx}"' in body and "(red(b)-red(a))" in body
    assert "lead?'Fix first':'To fix'" in body
    assert "const f=cb.dataset.fx==='drill'?'planDone':'fixDone';" in SPA and "persistEntryPatch(e);" in SPA
    assert "'fileName','fixDone']" in SPA          # a rebuilt entry keeps its ticks
    assert "const lead=($('#dashFix').dataset.sport)||'run';" in SPA   # cards follow the list


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


def test_the_card_and_what_to_fix_agree_and_line_up():
    """Review 2026-11-19: What to fix in two columns on a desk under the two
    cards, the cards' footers on one line, and a reading the report says to
    fix leads its sport card even when it is not scored (a clip's overstride)."""
    card = _fn("sportCard")
    assert "say:'to fix',named:true" in card and "st.say||DSC_WORD[st.cls]" in card
    css = (Path(__file__).resolve().parents[1] / "app" / "static" / "app.css").read_text(encoding="utf-8")
    assert ".dsc{grid-template-rows:auto 1fr auto}" in css
    assert ".fx-card{display:grid;grid-template-columns:repeat(2,minmax(0,1fr))" in css


def test_every_fix_says_how_and_history_holds_a_hundred():
    """Artur 2026-11-20: a fix with no way to do it ("Pelvic ratio is too
    high, range 2-4"); 117 analyses ran off the History strip; Then -> now
    spans Progress. Fit adjustments in the position's words, cues for the
    run; pelvic ratio is shown, never a fix."""
    assert "function fixHow(e, key, dir)" in SPA and "Raise the saddle about 5 mm" in SPA
    assert "aero?'Lower the pads" in SPA and "Take quicker steps at the same pace" in SPA
    assert "m.key==='pelvic_ratio'||(ungraded&&ungraded.has(m.key))" in SPA
    assert '<span class="fx-how"><b>How:</b>' in SPA
    assert "const dense=es.length>(PHONE_MQ.matches?8:16)" in SPA and "hstrip-pip" in SPA
    css = (Path(__file__).resolve().parents[1] / "app" / "static" / "app.css").read_text(encoding="utf-8")
    assert ".prog-vis{margin:0 0 24px;grid-template-columns:minmax(0,1fr)}" in css
    assert ".hstrip-dot.hstrip-pip{" in css


def test_progress_reads_one_way_of_filming_and_draws_clean_lines():
    """Artur 2026-11-20: the knee line ran 75 -> 145 "closer to the range"
    across photos (any point of the stroke) and clips (the bottom), and road
    and aero positions shared one line; markers on 46 points were noise; a
    reading at the band's edge read "0° to go"."""
    body = _fn("renderProgress")
    assert "(e.kind||'video')===(ref.kind||'video')&&(sport!=='bike'||(e.position||'')===(ref.position||''))" in body
    chart = _fn("trendChart2")
    assert "smoothPath(P)" in chart and "t2-hit" in chart and "!inBandShown(last,opt,dec)" in chart
    assert 'y="12">Range ' in chart
    assert "function smoothPath(P)" in SPA and "function inBandShown(v,opt,dec)" in SPA
    assert "psp-pt psp-last" in _fn("progSpark") and "smoothPath(" in _fn("progSpark")
    assert '<span class="dthen-tag">${lbl}</span>' in SPA
    css = (Path(__file__).resolve().parents[1] / "app" / "static" / "app.css").read_text(encoding="utf-8")
    assert ".pc-fold > summary::after{" in css and "body.authed .backhome-row{display:none}" in css
