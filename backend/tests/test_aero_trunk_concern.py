"""On aero bars a trunk flatter than its band: no points, amber, and the trade said.

Artur's call (2026-10-11), after the Kona 2026 field: six of fourteen pros rode
under the triathlon floor (20 deg), and they can. Most age-groupers cannot,
and the evidence says what it costs (Fintelman 2015: lower torso, lower maximal
power and worse physiology; a flat back worst of all). So the floor stays,
nobody loses points for going under it, and everyone is told the trade.

Before this the same reading got three verdicts: the photo forgave it 3x and
called it "aero gain", the fit plan listed it as a strength ("the goal, not a
fault"), and the clip took points for it at the full rate.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from app.services.video_analysis.biomechanics import technique_scorer as TS
from app.services.video_analysis.biomechanics.cycling_positions import (
    AERO_TRUNK_LOW_CONCERN,
    trunk_below_aero_floor,
)

HTML = (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")


def test_the_rule_is_aero_bars_only():
    assert trunk_below_aero_floor("triathlon", 16.0)
    assert trunk_below_aero_floor("tt_aero", 12.0)
    assert not trunk_below_aero_floor("tt_aero", 18.0), "inside the TT band"
    assert not trunk_below_aero_floor("road_drops", 20.0), "road keeps its symmetric rule"
    assert not trunk_below_aero_floor("triathlon", float("nan"))


def test_the_concern_names_the_trade_and_its_source():
    text = AERO_TRUNK_LOW_CONCERN.format(lo=20, hi=30)
    assert "20-30" in text and "costs nothing in the score" in text
    assert "pros" in text and "age-group" in text and "Fintelman 2015" in text


class TestTheClip:
    BASE = {"near_side": "left", "left_knee_at_bdc": 141.0, "knee_at_bdc": 141.0,
            "elbow_angle_avg": 90.0, "shoulder_angle_avg": 88.0, "frames_analyzed": 300,
            "forearm_tilt_avg": 15.0, "head_alignment_avg": 95.0}

    @staticmethod
    def _trunk(res):
        return (res.get("component_scores") or res.get("components") or {}).get("trunk_angle")

    def test_a_flat_aero_trunk_costs_no_points(self):
        low = TS.score_cycling({**self.BASE, "trunk_angle_avg": 14.0}, {}, cycling_position="triathlon")
        mid = TS.score_cycling({**self.BASE, "trunk_angle_avg": 24.0}, {}, cycling_position="triathlon")
        assert self._trunk(low) == 100.0
        assert low["overall_score"] == mid["overall_score"]

    def test_too_upright_still_costs(self):
        high = TS.score_cycling({**self.BASE, "trunk_angle_avg": 40.0}, {}, cycling_position="triathlon")
        assert self._trunk(high) < 100

    def test_road_low_trunk_still_costs(self):
        road = TS.score_cycling({**self.BASE, "trunk_angle_avg": 20.0}, {}, cycling_position="road_drops")
        assert self._trunk(road) < 100

    def test_severity_is_acceptable_not_a_warning(self):
        assert TS.classify_metric_severity("triathlon", "trunk_angle", 14.0) == "ACCEPTABLE"
        assert TS.classify_metric_severity("road_drops", "trunk_angle", 15.0) in ("WARNING", "CRITICAL")


class TestTheFitPlan:
    def test_it_states_the_concern_instead_of_calling_it_the_goal(self):
        from app.services.video_analysis.biomechanics.action_plan_builder import build_action_plan
        plan = build_action_plan(
            position="triathlon", angle_statistics={},
            sport_specific_metrics={"knee_at_bdc": 141.0, "trunk_angle_avg": 15.0, "hip_at_tdc": 55.0},
            technique_score=90, letter_grade="A", detected_issues=[],
        )
        trunk = next(g for g in plan.good_metrics if g["metric"] == "trunk_angle")
        notes = trunk.get("note") or ""
        assert "age-group" in notes and "the goal, not a fault" not in notes
        assert not any(d.metric_name == "trunk_angle" for d in plan.diagnostics), "no fix is prescribed"


class TestTheVideoCoach:
    def test_it_gets_the_same_instruction_as_the_photo_coach(self):
        from app.services.video_analysis.llm_recommendations import _bike_data_block
        low = _bike_data_block(88, "B", "triathlon", [], {"trunk_angle_avg": 15.0, "knee_at_bdc": 141.0})
        assert "BELOW its band on aero bars" in low and "Fintelman 2015" in low
        assert "do not advise raising the bars" in low
        ok = _bike_data_block(88, "B", "triathlon", [], {"trunk_angle_avg": 24.0, "knee_at_bdc": 141.0})
        road = _bike_data_block(88, "B", "road_drops", [], {"trunk_angle_avg": 20.0, "knee_at_bdc": 141.0})
        assert "BELOW its band" not in ok and "BELOW its band" not in road


class TestThePhoto:
    def test_amber_with_the_concern_and_no_points(self):
        np = pytest.importorskip("numpy")
        pytest.importorskip("cv2")
        from test_photo_pedal_phase import _bike_pose

        from app.services.video_analysis import photo_analyzer as PA
        assert PA._AERO_TOLERANCES["trunk"][0] == math.inf
        world, norm = _bike_pose(142.0)
        # flatten the near shoulder toward hip height: trunk ~13 deg
        for lst in (norm, world):
            lst[12].y = 0.33 if lst is norm else (0.33 - 0.5) * 2
        res = PA.analyze_from_pose(np.full((1280, 720, 3), 128, np.uint8), world, norm,
                                   "right", [], "bike", "triathlon")
        trunk = res["angles"]["trunk"]
        assert trunk < 20, f"the synthetic rider should sit under the floor, got {trunk}"
        row = res["angles_with_context"]["trunk"]
        assert row["status"] == "acceptable" and "age-group" in row["note"]
        assert res["score"]["per_angle"]["trunk"]["score"] == 100

    def test_the_coach_is_told_to_state_it_once_and_not_fix_it(self):
        from app.services.video_analysis.llm_recommendations import _build_photo_prompt
        res = {"cycling_position": "triathlon", "score": {"overall_score": 95, "grade": "Excellent"},
               "angles_with_context": {"trunk": {"value": 15.0, "label": "Trunk", "status": "acceptable",
                                                 "optimal_min": 20, "optimal_max": 30}}}
        prompt = _build_photo_prompt("bike", res)
        assert "costs no points" in prompt and "do not advise raising the bars" in prompt
        assert "Fintelman 2015" in prompt


class TestTheSurfaces:
    def test_the_overlay_never_paints_it_red(self):
        from app.services.video_analysis.pipeline import VideoAnalysisPipeline
        from app.services.video_analysis.video_visualizer import _status_for_cfg
        p = VideoAnalysisPipeline.__new__(VideoAnalysisPipeline)
        cfgs = p._get_angle_display_config("bike", {"camera_side": "left"}, cycling_position="triathlon")
        trunk = next(c for c in cfgs if c["key"] == "trunk_angle")
        assert trunk.get("closed_ok") is True
        assert _status_for_cfg(trunk, 10.0) == "warn"
        road = p._get_angle_display_config("bike", {"camera_side": "left"}, cycling_position="road_drops")
        assert not next(c for c in road if c["key"] == "trunk_angle").get("closed_ok")

    def test_the_report_tile_is_amber_with_a_line(self):
        assert "metric==='trunk_low_ok'" in HTML
        assert "trR.metric='trunk_low_ok'" in HTML
        assert "faster, if you can hold it and still make power" in HTML


class TestTheCoachIsNotToldToFixIt:
    """The materiality block decides what may become "Fix first". It graded
    symmetrically, so a flat aero trunk and a closed aero shoulder were
    "OUT OF RANGE" there while the data block told the coach not to fix them."""

    def test_aero_trunk_and_shoulder_are_borderline_not_material(self):
        from app.services.video_analysis.llm_recommendations import _metric_statuses
        st = _metric_statuses("bike", "triathlon", {"trunk_angle_avg": 12.0, "shoulder_angle_avg": 68.0})
        assert st["trunk_angle_avg"][0] == "warn" and st["shoulder_angle_avg"][0] == "warn"

    def test_road_keeps_them_material(self):
        from app.services.video_analysis.llm_recommendations import _metric_statuses
        st = _metric_statuses("bike", "road_drops", {"trunk_angle_avg": 18.0, "shoulder_angle_avg": 60.0})
        assert st["trunk_angle_avg"][0] == "bad" and st["shoulder_angle_avg"][0] == "bad"


def test_the_video_coach_reads_the_closed_hip_not_the_stroke_mean():
    """Leftover of the hip_at_tdc fix: the coach line still printed the mean
    against the closed band ("Hip angle: 85 (optimal 55-65)")."""
    from app.services.video_analysis.llm_recommendations import _bike_data_block
    block = _bike_data_block(88, "B", "road_hoods", [], {"hip_at_tdc": 52.0, "hip_angle_avg": 85.0})
    hip = next(line for line in block.splitlines() if "Hip" in line)
    assert "top of stroke" in hip and "52.0" in hip and "(optimal 55-70" in hip
    assert "85.0" in hip and "no band" in hip
