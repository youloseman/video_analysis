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
    assert "side" not in keys and "clip" in keys and "pros" in keys
    assert all(s["title"] and s["text"] for s in res["next_steps"])
    # Marked at the bottom of the stroke there is no pedal step to take.
    steps = PA._photo_next_steps("bike", "tt_aero", "bdc", False, "left", {"value": 15.0, "check": False})
    assert [s["key"] for s in steps] == ["clip", "pros"]
    # Never a helmet step: its type cannot be told from the side, and riders
    # in aero helmets were told to put one on (2026-10-14).
    steps = PA._photo_next_steps("bike", "triathlon", "bdc", False, "left", {"value": 15.0, "check": True})
    assert "helmet" not in [s["key"] for s in steps]
    # A runner's photo has a clip to film and no bike steps.
    assert [s["key"] for s in PA._photo_next_steps("run", None, None, False, "left", None)] == ["clip"]


def test_the_helmet_flag_gives_no_helmet_advice():
    assert PA._HELMET_CHECK_WARNING.startswith("We couldn't find a clear edge where your back ends")
    assert "race" not in PA._HELMET_CHECK_WARNING and "road helmet" not in PA._HELMET_CHECK_WARNING
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
    assert 'id="pNextBlock"' in SPA and "renderNextSteps('pNextBlock','pNextList',res.next_steps||(res.quality&&res.quality.next_steps)||[])" in SPA
    i = SPA.index("renderNextSteps('pNextBlock'")
    assert i < SPA.index("if(res.locked){", i), "drawn before the teaser returns, so a free reader sees it"
    assert ".score-partial{" in CSS and ".next-steps li{" in CSS


def test_an_aero_photo_with_a_high_back_is_not_read_as_road_drops():
    """upload/Check 2026-11-07: three aero race photos with a 24-29 deg back were
    graded as road drops by the trunk alone, 13-32 points low."""
    from types import SimpleNamespace

    import numpy as np

    from app.services.video_analysis.photo_analyzer import _auto_detect_cycling_position

    def lm(x, y):
        return SimpleNamespace(x=x / 1000, y=y / 1000, z=0.0, visibility=0.99)

    pts = {i: lm(500, 500) for i in range(33)}
    # Facing right: shoulder ahead of the hip, elbow under it, forearm level on the pads.
    pts.update({0: lm(700, 330), 11: lm(620, 380), 12: lm(620, 380), 23: lm(400, 500), 24: lm(400, 500),
                13: lm(640, 520), 14: lm(640, 520), 15: lm(780, 500), 16: lm(780, 500),
                25: lm(560, 650), 26: lm(560, 650), 27: lm(470, 860), 28: lm(470, 860)})
    nl = [pts[i] for i in range(33)]
    image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    assert _auto_detect_cycling_position(27.0, nl, image) == "triathlon"
    assert _auto_detect_cycling_position(18.0, nl, image) == "tt_aero"
    # Without the arms it falls back to the trunk -- but never to the drops:
    # the grip is not readable from the side (2026-11-13).
    assert _auto_detect_cycling_position(27.0) == "road_hoods"
    assert _auto_detect_cycling_position(18.0) == "tt_aero"
    assert _auto_detect_cycling_position(50.0) == "casual"
