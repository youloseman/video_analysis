"""A clean photo does not end the report.

Gabriel's photo (2026-10-13): 100/100 Excellent, points on the joints, every
scored angle in its band -- and four measures out of six, because the crank
was mid-stroke. The picture said "100/100 · Excellent" and the coach said
"Nothing to fix", so an athlete with a road helmet on a TT bike, an unverified
saddle height and a photo from the chainring side had nowhere to go. Now the
score says "upper body only" where it is, the helmet flag leads with the
advice, every photo ends with where to go next, and the coach is told to say
what the still could not grade.
"""
from __future__ import annotations

from pathlib import Path

from app.services import result_gating as RG
from app.services.video_analysis import photo_analyzer as PA
from app.services.video_analysis.llm_recommendations import _build_photo_prompt

from test_photo_pedal_phase import _measure

STATIC = Path(__file__).resolve().parents[1] / "app" / "static"
SPA = (STATIC / "index.html").read_text(encoding="utf-8")
CSS = (STATIC / "app.css").read_text(encoding="utf-8")


def test_an_upper_body_only_score_says_so():
    res = _measure(118.0)                         # mid-stroke: knee and hip unscored
    assert res["pedal_phase"] == "uncertain"
    assert res["score"]["partial"] == "upper body only"
    assert res["score_by_phase"]["uncertain"]["partial"] == "upper body only"
    assert "partial" not in res["score_by_phase"]["bdc"], "a marked frame is a full score"


def test_every_photo_ends_with_where_to_go_from_here():
    res = _measure(118.0)                         # right side = chainring toward the camera
    keys = [s["key"] for s in res["next_steps"]]
    assert keys[0] == "pedal", "the saddle-height reading comes first"
    assert "side" in keys and "clip" in keys and "pros" in keys
    assert all(s["title"] and s["text"] for s in res["next_steps"])
    # Marked at the bottom of the stroke there is no pedal step to take.
    steps = PA._photo_next_steps("bike", "tt_aero", "bdc", False, "left", {"value": 15.0, "check": False})
    assert [s["key"] for s in steps] == ["clip", "pros"]
    # A road helmet (no edge to read) is a step, said as advice.
    steps = PA._photo_next_steps("bike", "triathlon", "bdc", False, "left", {"value": 15.0, "check": True})
    assert steps[0]["key"] == "helmet" and "aero helmet" in steps[0]["text"]
    # A runner's photo has a clip to film and no bike steps.
    assert [s["key"] for s in PA._photo_next_steps("run", None, None, False, "left", None)] == ["clip"]


def test_the_helmet_flag_leads_with_the_advice():
    assert PA._HELMET_CHECK_WARNING.startswith("We couldn't read a helmet against your back")
    assert "cheapest gain" in PA._HELMET_CHECK_WARNING
    assert "drag it to the highest point of your back" in PA._HELMET_CHECK_WARNING


def test_the_steps_reach_a_free_reader():
    """How to get a better result is not behind the paywall (capture_report's rule)."""
    res = _measure(118.0)
    block = RG.quality_block(res)
    assert [s["key"] for s in block["next_steps"]] == [s["key"] for s in res["next_steps"]]


def test_the_coach_is_told_a_clean_photo_still_has_a_next_step():
    res = {"score": {"overall_score": 100, "grade": "Excellent"}, "cycling_position": "triathlon",
           "angles_with_context": {"trunk": {"value": 23.0, "label": "Trunk", "optimal_min": 20,
                                             "optimal_max": 25, "status": "optimal"}},
           "next_steps": [{"key": "pedal", "title": "Mark the pedal position", "text": "x"},
                          {"key": "clip", "title": "Film a 10-second clip", "text": "y"}]}
    prompt = _build_photo_prompt("bike", res)
    assert "Material problems: NONE" in prompt
    assert "could NOT grade" in prompt and "Mark the pedal position; Film a 10-second clip" in prompt
    assert "Do not end the report with nothing to do." in prompt


def test_the_page_shows_both():
    assert 'id="pPartial"' in SPA and "function setPhotoPartial" in SPA
    assert SPA.count("setPhotoPartial(sc);") == 2, "on render and on a phase change"
    assert 'id="pNextBlock"' in SPA and "renderPhotoNext(res.next_steps||(res.quality&&res.quality.next_steps)||[])" in SPA
    i = SPA.index("renderPhotoNext(res.next_steps")
    assert i < SPA.index("if(res.locked){", i), "drawn before the teaser returns, so a free reader sees it"
    assert ".score-partial{" in CSS and ".next-steps li{" in CSS
