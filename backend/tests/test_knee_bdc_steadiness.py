"""A knee at the bottom of the stroke that will not hold still moves no saddle.

IMG_5577 (prod, 2026-11-09, before the leg labels were fixed): eight strokes
read 120-141 deg and the report said raise the saddle 43 mm (+-26).
"""
from __future__ import annotations

from app.services.video_analysis.biomechanics.cycling_positions import (
    knee_bdc_steadiness,
    knee_bdc_unsteady,
)
from app.services.video_analysis.biomechanics.technique_scorer import score_cycling
from app.services.video_analysis.capture_report import film_notes_of

BROKEN = [141.5, 121.2, 139.3, 132.1, 124.8, 139.1, 119.9, 135.9]   # IMG_5577, labels from MediaPipe
CLEAN = [149.0, 149.5, 151.7, 148.7, 151.8, 152.5, 148.3]            # IMG_5572
ONE_BAD = [144.3, 144.2, 143.7, 143.1, 144.7, 143.9, 143.4, 143.9, 140.7, 128.5]  # IMG_5571


def test_a_broken_clip_is_unsteady_a_clean_one_and_one_bad_stroke_are_not():
    assert knee_bdc_steadiness(BROKEN)["unsteady"] is True
    assert knee_bdc_steadiness(CLEAN)["unsteady"] is False
    one = knee_bdc_steadiness(ONE_BAD)
    assert one["unsteady"] is False and one["spread_deg"] < 1.5, "one wild stroke is not a broken clip"


def test_too_few_strokes_say_nothing():
    assert knee_bdc_steadiness([140.0, 120.0, 141.0]) is None


SUMMARY = {
    "frames_analyzed": 150, "near_side": "left", "left_knee_at_bdc": 128.0,
    "knee_at_bdc": 128.0, "left_knee_at_tdc": 70.0, "trunk_angle_avg": 22.0,
    "elbow_angle_avg": 90.0, "shoulder_angle_avg": 88.0, "pelvic_ratio": 2.5,
    "saddle_height_assessment": "too_low",
}


def _unsteady(summary):
    return {**summary, "diagnostics": {"bdc_tdc": {"steadiness": knee_bdc_steadiness(BROKEN)}}}


def test_an_unsteady_knee_leaves_the_score():
    steady = score_cycling(SUMMARY, {}, cycling_position="triathlon")
    shaky = score_cycling(_unsteady(SUMMARY), {}, cycling_position="triathlon")
    assert "knee_bdc" in steady["component_scores"] and "knee_bdc" not in shaky["component_scores"]
    assert "saddle_fit" not in steady["component_scores"], "the saddle is the knee, counted once"
    assert "knee_bdc" in shaky["coverage"]["excluded"]
    assert shaky["overall_score"] > steady["overall_score"]
    assert knee_bdc_unsteady(_unsteady(SUMMARY)) and not knee_bdc_unsteady(SUMMARY)


def test_no_saddle_card_and_a_film_again_note():
    from app.services.video_analysis.biomechanics.action_plan_builder import build_action_plan

    def saddle_cards(sm):
        plan = build_action_plan("triathlon", {}, sm, 80, "B", [])
        return [d for d in plan.diagnostics if d.component == "saddle_height"]

    assert saddle_cards(SUMMARY)
    assert not saddle_cards(_unsteady(SUMMARY))
    notes = film_notes_of({"sport_specific_metrics": _unsteady(SUMMARY), "cycling_position": "triathlon"})
    assert notes and notes[0]["code"] == "knee_unsteady" and notes[0]["level"] == "bad"
    assert "120-142" in notes[0]["text"]
