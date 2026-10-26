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
