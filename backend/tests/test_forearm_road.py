"""Forearm tilt is an aero-bar measure, and off aero bars it is not scored.

Three hoods clips of one rider: two read -42 deg and were silently dropped
by the summary's plausibility bound, the third read -25, passed, and lost
five points to (5, 20) -- an aero band pasted onto the hoods. Pinned here:
on a road position the component is absent AND named in the coverage's
``excluded`` with the reason; on aero bars it is scored as before; and the
plan does not list it as "already in range" on the hoods.
"""
from __future__ import annotations

from app.services.video_analysis.biomechanics.action_plan_builder import build_action_plan
from app.services.video_analysis.biomechanics.technique_scorer import score_cycling

SUMMARY = {
    "frames_analyzed": 150, "knee_at_bdc": 140.0, "knee_at_tdc": 70.0,
    "trunk_angle_avg": 45.0, "elbow_angle_avg": 155.0, "shoulder_angle_avg": 90.0,
    "helmet_above_back": 15.0, "pelvic_ratio": 2.5, "saddle_height_assessment": "optimal",
    "forearm_tilt_avg": -25.0,
}


def test_the_hoods_are_scored_on_seven_measures_and_say_seven():
    """Forearm tilt and the helmet reading are aero-bar measures: off the
    bars both leave the rubric rather than counting as missing."""
    out = score_cycling(SUMMARY, {}, cycling_position="road_hoods")
    assert "forearm_tilt" not in out["component_scores"]
    assert "helmet_above_back" not in out["component_scores"]
    cov = out["coverage"]
    assert cov["measures_total"] == 7 and cov["measures_scored"] == 7
    assert cov["missing"] == [] and cov["weight_covered"] == 1.0
    assert out["overall_score"] >= 95
    drops = score_cycling(SUMMARY, {}, cycling_position="road_drops")
    assert "forearm_tilt" not in drops["component_scores"]
    # The lottery, pinned: a -25 forearm on the hoods used to cost points.
    assert out["overall_score"] == score_cycling(
        {**SUMMARY, "forearm_tilt_avg": 12.0}, {}, cycling_position="road_hoods",
    )["overall_score"]


def test_aero_bars_still_score_it():
    out = score_cycling({**SUMMARY, "forearm_tilt_avg": 12.0}, {}, cycling_position="triathlon")
    assert out["component_scores"]["forearm_tilt"] == 100.0
    assert out["coverage"]["measures_total"] == 9


def test_the_plan_does_not_praise_a_hoods_forearm():
    plan = build_action_plan(
        position="road_hoods", angle_statistics={}, sport_specific_metrics=SUMMARY,
        technique_score=90, letter_grade="A", detected_issues=[],
    )
    assert not any(g.get("metric") == "forearm_tilt" for g in plan.good_metrics)
    aero = build_action_plan(
        position="triathlon", angle_statistics={},
        sport_specific_metrics={**SUMMARY, "forearm_tilt_avg": 12.0},
        technique_score=90, letter_grade="A", detected_issues=[],
    )
    assert any(g.get("metric") == "forearm_tilt" for g in aero.good_metrics)


def test_the_overlay_draws_no_forearm_callout_off_aero_bars():
    from app.services.video_analysis.pipeline import VideoAnalysisPipeline

    p = VideoAnalysisPipeline()
    hoods = [c["key"] for c in p._get_angle_display_config("bike", {"camera_side": "left"}, None, cycling_position="road_hoods")]
    aero = [c["key"] for c in p._get_angle_display_config("bike", {"camera_side": "left"}, None, cycling_position="triathlon")]
    assert "left_forearm_tilt" not in hoods
    assert "left_forearm_tilt" in aero


def test_the_road_levers_name_road_parts():
    from app.services.video_analysis.biomechanics.fit_tradeoffs import rank_adjustments

    def row(label, value, lo, hi, status="needs_work"):
        return {"label": label, "value": value, "optimal_min": lo, "optimal_max": hi, "status": status}

    closed = {"shoulder": row("Shoulder", 70, 80, 100), "trunk": row("Trunk", 29, 40, 55),
              "forearm_tilt": row("Forearm", -57, 5, 20)}
    road = " ".join(r["label"] for r in rank_adjustments(closed, "road_hoods"))
    assert "raise the bars" in road
    assert "aerobar" not in road and "extensions" not in road and "pads" not in road
    aero = " ".join(r["label"] for r in rank_adjustments(closed, "triathlon"))
    assert "aerobar" in aero and "tilt the extensions" in aero


def _coach_with(monkeypatch, drafts):
    from app.services.video_analysis import llm_recommendations as L

    calls = []

    def fake(system, prompt, max_tokens, tag=None):
        calls.append(prompt)
        return {"text": drafts[min(len(calls) - 1, len(drafts) - 1)], "model": "m"}

    monkeypatch.setattr(L, "_complete", fake)
    return L, calls


def test_a_road_report_naming_aero_kit_is_sent_back(monkeypatch):
    L, calls = _coach_with(monkeypatch, ["Raise the aerobar stack 10mm.", "Raise the bars 10mm."])
    out = L._coach("p", [], tag="photo", road=True)
    assert len(calls) == 2 and out["report"] == "Raise the bars 10mm."


def test_a_road_report_that_insists_loses_the_sentence(monkeypatch):
    L, _ = _coach_with(monkeypatch, ["Your elbow is fine. Tilt the extensions up 10 deg. Hold it."])
    out = L._coach("p", [], tag="photo", road=True)
    assert "extensions" not in out["report"]
    assert "Your elbow is fine." in out["report"] and "Hold it." in out["report"]


def test_aero_positions_and_knee_extension_are_left_alone(monkeypatch):
    L, calls = _coach_with(monkeypatch, ["Raise the aerobar stack 10mm."])
    assert "aerobar" in L._coach("p", [], tag="photo", road=False)["report"]
    assert len(calls) == 1
    L, calls = _coach_with(monkeypatch, ["Knee extension at the bottom is good. Raising the bars costs aero."])
    assert "Knee extension" in L._coach("p", [], tag="photo", road=True)["report"]
    assert len(calls) == 1


def test_the_road_checklist_talks_about_the_stem():
    from app.services.video_analysis import concern_hints as H

    c = H.concern("shoulder_angle_avg", "Shoulder", 70.0, 80, 100, material=True)
    assert "arm support" in c["hint"]
    assert "stem" in H.for_road([c])[0]["hint"] and "arm support" not in H.for_road([c])[0]["hint"]
