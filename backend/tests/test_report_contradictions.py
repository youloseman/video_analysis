"""What the before/after bike-fit pair showed the report saying against itself.

Gona, 2026-11-28: a public "before fit" clip read 79/B with "Nothing to fix"
over a red helmet row the text called "on the edge"; the "after fit" clip
read 99/A Excellent with two borderline rows under "Needs attention" and a
saddle change below "Nothing to fix". A 2.9 s clip was told to keep the whole
body in the frame, and both ended on "focus on maintaining" a fine shoulder.
"""
from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from app.services import report_pdf
from app.services.video_analysis import concern_hints
from app.services.video_analysis import llm_recommendations as llm
from tests.test_coach_spa import SPA, fn

CLEAN = {
    "knee_at_bdc": 141.0, "knee_at_tdc": 66.0, "trunk_angle_avg": 22.0, "elbow_angle_avg": 90.0,
    "shoulder_angle_avg": 92.0, "hip_angle_avg": 50.0, "forearm_tilt_avg": 20.0, "pelvic_ratio": 3.0,
    "helmet_check": False,
}


# --------------------------------------------------------------------------
# The head: red in the table, a fix in the text
# --------------------------------------------------------------------------
def test_a_head_the_table_paints_red_is_a_material_problem():
    block = llm._materiality_block("bike", "triathlon", {**CLEAN, "helmet_above_back": 35.0})
    assert "MATERIAL PROBLEMS (only these" in block and "Helmet +35%" in block
    concern = next(c for c in llm._video_concerns("bike", "triathlon", {**CLEAN, "helmet_above_back": 35.0})
                   if c["field"] == "helmet_above_back")
    assert concern["material"] is True
    assert "outside the band" in concern_hints.worth_a_look([concern])


def test_a_head_just_past_the_free_band_stays_on_the_edge():
    block = llm._materiality_block("bike", "triathlon", {**CLEAN, "helmet_above_back": 25.0})
    assert "MATERIAL PROBLEMS: NONE" in block and "Helmet +25%" in block
    concern = next(c for c in llm._video_concerns("bike", "triathlon", {**CLEAN, "helmet_above_back": 25.0})
                   if c["field"] == "helmet_above_back")
    assert concern["material"] is False


# --------------------------------------------------------------------------
# The coach's text
# --------------------------------------------------------------------------
def test_a_next_session_that_only_says_keep_going_is_left_out():
    text = ("**Overall**\nGood.\n\n**Nothing to fix**\nHold on to it.\n\n"
            "**Next session**\nFocus on maintaining your established shoulder relaxation.\n")
    assert "Next session" not in concern_hints.drop_idle_next_session(text)
    real = text.replace("Focus on maintaining your established shoulder relaxation.",
                        "Drop the chin and shoulders 2 cm and hold it for the whole interval.")
    assert concern_hints.drop_idle_next_session(real) == real


def test_nothing_to_fix_over_a_checklist_prints_as_nothing_urgent():
    text = ("**Nothing to fix**\nHold on to it.\n\n**Worth a look**\n"
            "- **Knee angle at the bottom of the stroke**: 146° vs 138-145°, on the edge of the band.")
    out = report_pdf._coach_text(text, set())
    assert "**Nothing urgent**" in out and "Nothing to fix" not in out
    assert "Nothing to fix" in report_pdf._coach_text("**Nothing to fix**\nHold on to it.", set())


# --------------------------------------------------------------------------
# The PDF table and the film-again line
# --------------------------------------------------------------------------
def _row(status, field="knee_at_bdc", value=146.0):
    return {"field": field, "label": "Knee at the bottom", "value": value, "unit": "°",
            "band": (138.0, 145.0), "status": status, "scale": (132, 151)}


def test_a_borderline_row_is_on_the_edge_not_needing_attention():
    html = report_pdf.metrics_table([_row("warn"), _row("good", "trunk_angle_avg", 22.0)])
    assert "On the edge</b> · 1" in html and "Needs attention" not in html
    assert "Needs attention</b> · 1" in report_pdf.metrics_table([_row("bad", value=155.0)])


def test_a_short_clip_is_told_to_film_longer_a_lost_rider_to_stay_in_frame():
    short = [{"code": "duration", "text": "Little usable footage: 2.9 s. The numbers rest on very little movement."}]
    lost = [{"code": "duration", "text": "Little usable footage: 9.5 s long, but the athlete was only "
                                         "trackable in 3.7 s of it."}]
    assert "film 10-15 seconds" in report_pdf._reshoot(short)
    assert "keep the whole body in the frame" in report_pdf._reshoot(lost)


def test_the_client_table_has_no_pelvic_ratio():
    from tests.test_report_pdf import example

    res = example("bike-road")
    res.setdefault("sport_specific_metrics", {})["pelvic_ratio"] = 3.25
    html = report_pdf.render_report_html(res, {})
    assert "Pelvic ratio" not in html.split("Every measurement, against its range")[-1].split("Where the ranges")[0]


# --------------------------------------------------------------------------
# The coach's preview of what the athlete gets
# --------------------------------------------------------------------------
node = shutil.which("node")


@pytest.mark.skipif(not node, reason="node is not installed")
def test_the_preview_quotes_whole_sentences_of_the_overall():
    md = ("**Overall**\nYour triathlon aero position is well-balanced, sitting inside the target bands. "
          "The head is the one thing out.\nA third sentence that should not print because two is plenty.\n\n"
          "**What's working**\n- Trunk 22°.")
    script = fn("coachLead") + f"\nconsole.log(JSON.stringify(coachLead({json.dumps(md)})));"
    out = json.loads(subprocess.run([node, "-e", script], capture_output=True, text=True, check=True).stdout)
    assert out.startswith("Your triathlon aero position") and out.endswith("one thing out.")
    assert "Overall" not in out and "working" not in out and "…" not in out


def test_the_preview_names_the_red_row_shows_the_frame_and_asks_for_branding():
    src = fn("renderCoachPreview")
    assert "out?`<span>First thing to change</span><b>${esc(out.m.lbl)}</b>`" in src
    assert "cr-frame" in src and "fetchKeyframe(e)" in src
    assert "Add your ${COACH.name?'logo':'name & logo'}" in src and "openCoachProfile" in src
    assert "No email on file for" in src and "button reads" not in src
    assert "'On the edge',by.edge" in SPA
