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
from sqlalchemy import select

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
    sample = example("bike-road")
    assert "Bar position —" in render_report_html(sample, {})
    # In the order the fit plan gives: saddle before the bars.
    saddle = {"component": "saddle_height", "status": "needs_adjustment", "metric_name": "knee_at_bdc",
              "action": "lower_saddle", "amount": "5mm", "reason": "Knee reads open.", "priority": 1}
    fit = {**sample["fit_plan"], "diagnostics": [saddle, *sample["fit_plan"]["diagnostics"]]}
    html = render_report_html({**sample, "fit_plan": fit}, {})
    assert "Lower saddle 5mm" in html
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


# --- the clip a report is read against

async def _clip(db, user, cid, at, **kw) -> Analysis:
    result = {"sport_type": "bike", "technique_score": kw.pop("score", 70), "camera_side": kw.pop("side", "left"),
              "cycling_position": kw.pop("position", "road_hoods"), "sport_specific_metrics": kw.pop("metrics", {})}
    row = Analysis(user_id=user.id, client_id=cid, created_at_ms=at, sport="bike", kind="video",
                   result=result, data=kw.pop("data", {}), **kw)
    db.add(row)
    await db.commit()
    return row


async def test_the_previous_clip_is_the_latest_like_for_like_one(db, make_user):
    user = await make_user(TIER_FULL)
    await _clip(db, user, "old", 1_000_000)
    await _clip(db, user, "prev", 2_000_000, score=71)
    now = await _clip(db, user, "now", 3_000_000)
    await _clip(db, user, "later", 4_000_000, score=99)
    prev = await me._previous_clip(db, now, None)
    assert prev["result"]["technique_score"] == 71 and prev["filmed"] == "1 Jan 1970"


async def test_a_different_side_or_position_is_no_before(db, make_user):
    """The two sides of a bike read the knee apart, so the gap would be the camera."""
    user = await make_user(TIER_FULL)
    await _clip(db, user, "side", 1_000_000, side="right")
    await _clip(db, user, "pos", 1_500_000, position="road_drops")
    now = await _clip(db, user, "now", 3_000_000)
    assert await me._previous_clip(db, now, None) is None


async def test_a_clip_with_doubtful_numbers_is_no_before(db, make_user):
    user = await make_user(TIER_FULL)
    bad = {"capture_report": {"checks": [{"id": "framing", "status": "bad", "label": "How much of the frame you fill",
                                          "measured": "200 px tall"}]}}
    await _clip(db, user, "bad", 1_000_000, metrics=bad)
    now = await _clip(db, user, "now", 3_000_000)
    assert await me._previous_clip(db, now, None) is None
    # A coach whose own size bar passes it can read against it.
    assert await me._previous_clip(db, now, {"small_px": 150}) is not None


async def test_the_date_filmed_decides_what_came_before(db, make_user):
    user = await make_user(TIER_FULL)
    await _clip(db, user, "uploaded_late", 5_000_000, data={"filmedAt": 1_000_000})
    now = await _clip(db, user, "now", 3_000_000)
    assert (await me._previous_clip(db, now, None)) is not None


async def test_a_preview_or_another_account_is_no_before(db, make_user):
    user = await make_user(TIER_FULL)
    other = await make_user(TIER_FULL)
    await _clip(db, user, "preview", 1_000_000, preview=True)
    await _clip(db, other, "theirs", 2_000_000)
    now = await _clip(db, user, "now", 3_000_000)
    assert await me._previous_clip(db, now, None) is None


async def test_the_report_route_reads_against_the_previous_clip(db, make_user):
    user = await make_user(TIER_FULL)
    bands = {"knee_at_bdc": {"lo": 135, "hi": 145}}
    await _clip(db, user, "prev", 1_000_000, metrics={"knee_at_bdc": 172.0})
    now = await _clip(db, user, "now", 3_000_000, metrics={"knee_at_bdc": 150.0}, score=80)
    for row in (now, (await db.execute(select(Analysis).where(Analysis.client_id == "prev"))).scalar_one()):
        row.result = {**row.result, "reference_bands": bands}
    await db.commit()
    html = await me.render_saved_report(db, reader=user, owner=user, row=now, athlete_id=None)
    assert "172° → 150°" in html and "70 → 80" in html


async def test_a_report_marked_before_the_seatpost_card_gets_it_and_a_new_coach_text(db, make_user, monkeypatch):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from test_report_pdf_verdicts import _far

    user = await make_user(TIER_FULL)
    res = {**_far(), "ai_recommendations": {"report": "**Fix first**\n- Move saddle forward 5mm"}}
    row = Analysis(user_id=user.id, client_id="far", created_at_ms=1, sport="bike", kind="video", result=res, data={})
    db.add(row)
    await db.commit()
    written = []

    def fake(*a, **k):
        written.append(k.get("fit_plan"))
        return {"report": "**Fix first**\n- A seatpost", "model": "fake"}

    import app.services.video_analysis.llm_recommendations as llm
    monkeypatch.setattr(llm, "generate_recommendations", fake)
    resp = await me.analysis_report("far", user, db)
    html = resp.body.decode()
    assert "A seatpost" in html and "Move saddle forward 5mm" not in html
    assert written and written[0]["diagnostics"][0]["action"] == "move_saddle_forward_far"


# --- the report as a file (services/pdf_render, Gotenberg)

class _FakeGotenberg:
    def __init__(self, status=200, body=b"%PDF-1.7 fake"):
        self.status, self.body, self.calls = status, body, []

    def install(self, monkeypatch):
        import httpx

        from app.services import pdf_render

        def handler(request):
            self.calls.append(request)
            return httpx.Response(self.status, content=self.body)

        real = httpx.AsyncClient
        monkeypatch.setattr(httpx, "AsyncClient",
                            lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
        monkeypatch.setattr(pdf_render, "service_url", lambda: "http://pdf.internal:3000")
        return self


async def test_the_report_downloads_as_a_file_named_after_the_athlete(db, make_user, monkeypatch):
    fake = _FakeGotenberg().install(monkeypatch)
    user = await make_user(tier=TIER_FULL)
    await _stored(db, user)
    resp = await me.analysis_report_pdf("h1", user, db)
    assert resp.media_type == "application/pdf" and resp.body.startswith(b"%PDF")
    assert "filename*=UTF-8''" in resp.headers["content-disposition"] and "Running" in resp.headers["content-disposition"]
    sent = fake.calls[0]
    assert str(sent.url) == "http://pdf.internal:3000/forms/chromium/convert/html"
    body = sent.content.decode("utf-8", "ignore")
    # A4, the page count in the footer, and the document's own footer row gone.
    assert 'name="paperWidth"' in body and "8.27" in body and 'class="pageNumber"' in body
    assert ".flow tfoot{display:none}" in body


async def test_no_pdf_service_is_a_503_and_the_app_prints_instead(db, make_user, monkeypatch):
    from app.services import pdf_render

    monkeypatch.setattr(pdf_render, "service_url", lambda: None)
    user = await make_user(tier=TIER_FULL)
    await _stored(db, user)
    with pytest.raises(HTTPException) as e:
        await me.analysis_report_pdf("h1", user, db)
    assert e.value.status_code == 503


async def test_a_failed_render_is_a_503_too(db, make_user, monkeypatch):
    _FakeGotenberg(status=500, body=b"boom").install(monkeypatch)
    user = await make_user(tier=TIER_FULL)
    await _stored(db, user)
    with pytest.raises(HTTPException) as e:
        await me.analysis_report_pdf("h1", user, db)
    assert e.value.status_code == 503


async def test_the_pdf_file_is_a_full_plan_feature_like_the_page(db, make_user, monkeypatch):
    _FakeGotenberg().install(monkeypatch)
    user = await make_user(tier=TIER_ENTHUSIAST)
    await _stored(db, user)
    with pytest.raises(HTTPException) as e:
        await me.analysis_report_pdf("h1", user, db)
    assert e.value.status_code == 402


def test_the_file_name_is_safe_everywhere():
    from app.services.pdf_render import file_name

    name = file_name('Maya: "Chen"/Cycling, 6 Nov')
    assert name.endswith(".pdf") and not any(ch in name for ch in '\/:*?"<>|') and "Cycling, 6 Nov" in name
    assert file_name("") == "Flapp report.pdf"
