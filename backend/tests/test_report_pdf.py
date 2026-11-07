"""The Full plan's PDF report: a designed A4 document per analysis.

Built from the committed /examples samples -- real measurements of real
footage -- so the document is checked against the shapes the analyzer
actually produces: a run with a training plan, a bike with a fit plan.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from app.api import me
from app.models.analysis import Analysis
from app.models.user import TIER_ENTHUSIAST, TIER_FULL
from app.services.report_pdf import markdown, plan_rows, render_report_html

EXAMPLES = Path(__file__).resolve().parents[1] / "content" / "examples"


def example(slug: str) -> dict:
    return json.loads((EXAMPLES / f"{slug}.json").read_text(encoding="utf-8"))


def test_the_coach_markdown_is_escaped_and_structured():
    html = markdown("## Fix first\n**Saddle** <script>x</script>\n\n- one\n- two")
    assert "<h3>Fix first</h3>" in html
    assert "<b>Saddle</b>" in html and "<script>" not in html
    assert "<ul><li>one</li><li>two</li></ul>" in html


def test_a_run_reads_its_numbers_from_the_plan():
    rows = plan_rows(example("run-treadmill"))
    by = {r["field"]: r for r in rows}
    assert by["cadence"]["status"] == "good" and by["cadence"]["band"] == (170.0, 190.0)
    assert by["overstride"]["status"] == "bad" and by["overstride"]["band"] == (0.0, 0.15)


def test_a_run_report_has_the_plan_and_the_numbers():
    html = render_report_html(example("run-treadmill"), {"athlete": "a@b.c"})
    assert "Technique Report" in html and "Fix these, in this order" in html
    assert "Quick-Feet Cadence Drill" in html
    assert "Every measurement, against its range" in html


def test_a_bike_report_turns_the_fit_plan_into_the_plan():
    html = render_report_html(example("bike-road"), {})
    assert "Saddle height" in html and "Lower saddle 5mm" in html
    # In the order the fit plan gives: saddle before the bars.
    assert html.index("Saddle height —") < html.index("Bar position —")


def _filmed(*checks: dict) -> dict:
    return {"sport_type": "bike", "camera_side": "right",
            "sport_specific_metrics": {"capture_report": {"checks": list(checks)}}}


CHAIN = {"id": "drive_side", "status": "warn", "measured": "drive side"}
SMALL = {"id": "framing", "status": "warn", "label": "How much of the frame you fill",
         "measured": "320 px tall (25% of the frame height)"}


def test_the_report_says_how_the_clip_was_filmed():
    html = render_report_html(_filmed(CHAIN, SMALL), {})
    assert "How this clip was filmed" in html and "reads about 8 to 10" not in html
    assert "Small in the frame: 320 px tall" in html and "right side · chainring side" in html
    # "Keep filming the same way" can't stand under "film the other side".
    assert "Keep filming the same way" not in html


def test_a_coachs_note_is_not_followed_by_nothing_to_report():
    """The coach has just said what to work on; "nothing is outside its range" argues with them."""
    clean = _filmed({"id": "framing", "status": "good"})
    assert "Nothing in this clip" in render_report_html(clean, {})
    html = render_report_html(clean, {"coach_note": "Drop the saddle 5 mm."})
    assert "Drop the saddle 5 mm." in html and "Nothing in this clip" not in html


def test_a_clean_clip_has_no_filming_note():
    assert "How this clip was filmed" not in render_report_html(_filmed({"id": "framing", "status": "good"}), {})


def test_a_coachs_bars_reach_the_report():
    html = render_report_html(_filmed(CHAIN, SMALL), {"capture_bars": {"small_px": 300, "drive_side": False}})
    assert "How this clip was filmed" not in html


def test_the_document_has_a_phone_layout_and_stays_a4_in_print():
    css = render_report_html(_filmed(), {})
    assert "@media screen and (max-width:720px)" in css
    assert "size:A4" in css


def test_nothing_from_the_result_is_written_unescaped():
    res = example("run-treadmill")
    res["detected_issues"] = [{"type": "x", "value": "<b>v</b>", "recommendation": "<img src=x onerror=1>"}]
    html = render_report_html(res, {"athlete": "<script>"})
    assert "<img src=x" not in html and "<script>" not in html


async def _stored(db, user, slug="run-treadmill"):
    row = Analysis(user_id=user.id, client_id="h1", job_id="j1", created_at_ms=1,
                   data={"id": "h1"}, result={**example(slug), "ai_recommendations": {"report": "## Fix first\nX"}})
    db.add(row)
    await db.commit()
    return row


async def test_full_gets_the_report(db, make_user):
    user = await make_user(tier=TIER_FULL)
    await _stored(db, user)
    resp = await me.analysis_report("h1", user, db)
    assert resp.status_code == 200 and b"Technique Report" in resp.body


async def test_the_report_is_a_full_plan_feature(db, make_user):
    user = await make_user(tier=TIER_ENTHUSIAST)
    await _stored(db, user)
    with pytest.raises(HTTPException) as exc:
        await me.analysis_report("h1", user, db)
    assert exc.value.status_code == 402


async def test_someone_elses_analysis_is_a_404(db, make_user):
    owner = await make_user(tier=TIER_FULL)
    other = await make_user(tier=TIER_FULL)
    await _stored(db, owner)
    with pytest.raises(HTTPException) as exc:
        await me.analysis_report("h1", other, db)
    assert exc.value.status_code == 404


def test_the_cover_shows_what_the_score_is_built_from():
    from app.services.report_pdf import breakdown_items

    items = breakdown_items(example("bike-road"))
    assert items and items[0]["name"] == "Knee at bottom" and items[0]["weight"] == 0.22
    html = render_report_html(example("bike-road"), {})
    assert "Score built from" in html


def test_a_photo_breakdown_comes_from_its_own_score():
    from app.services.report_pdf import breakdown_items

    res = {"sport": "bike", "score": {"overall_score": 80, "per_angle": {
        "knee": {"score": 90, "weight": 0.24}, "elbow": {"score": 41, "weight": 0.11}}}}
    items = breakdown_items(res)
    assert [i["name"] for i in items] == ["Knee", "Elbow"]
    assert "80" in render_report_html(res, {})


def test_the_numbers_are_drawn_like_the_app():
    html = render_report_html(example("bike-road"), {})
    assert "class='tile'" in html and "class='gr-band'" in html and "class='gr-mk'" in html
