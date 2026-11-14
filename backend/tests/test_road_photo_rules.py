"""Road photo rules from the ten-photo hoods test (2026-11-13).

upload/landing/test_roadbike: ten race-fit riders on the hoods. The pose landed
well on all ten; the grading did not -- a red shoulder on eight (bent elbows
close the upper arm), the most race-like position scored worst, a lower rider
auto-filed under drops scored 100, "Excellent" beside a red row, and a helmet
warning on every photo for a number that is never graded on the road.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from app.services.video_analysis import photo_analyzer as PA  # noqa: E402
from app.services.video_analysis.biomechanics.cycling_positions import (  # noqa: E402
    ROAD_TRUNK_LOW_FORGIVENESS,
    trunk_below_road_floor,
    trunk_low_is_amber,
)


def _road_ranges():
    return PA._get_cycling_optimal_ranges("road_hoods", "tdc")


class TestTheScore:
    def test_a_low_road_trunk_costs_a_third(self):
        ranges = _road_ranges()
        base = {"knee": 70.0, "hip": 62.0, "shoulder": 90.0, "elbow": 155.0}
        low = PA._score_photo_angles({**base, "trunk": 31.0}, ranges, "bike",
                                     PA._BIKE_PHASE_WEIGHTS["tdc"], "road_hoods", "tdc")
        upright = PA._score_photo_angles({**base, "trunk": 64.0}, ranges, "bike",
                                         PA._BIKE_PHASE_WEIGHTS["tdc"], "road_hoods", "tdc")
        # 9 deg under the band costs what 3 deg over it costs
        assert low["per_angle"]["trunk"]["score"] == PA._score_single_angle(
            40 - 9 / ROAD_TRUNK_LOW_FORGIVENESS, 40, 55)
        assert upright["per_angle"]["trunk"]["score"] < low["per_angle"]["trunk"]["score"]

    def test_a_closed_road_shoulder_costs_a_third_and_holds_nothing(self):
        ranges = _road_ranges()
        r = PA._score_photo_angles(
            {"knee": 70.0, "hip": 62.0, "trunk": 42.0, "shoulder": 65.0, "elbow": 150.0},
            ranges, "bike", PA._BIKE_PHASE_WEIGHTS["tdc"], "road_hoods", "tdc")
        assert r["per_angle"]["shoulder"]["score"] == PA._score_single_angle(80 - 15 / 3, 80, 100)
        assert "red_rows" not in r and r["grade"] == "Excellent"

    def test_excellent_has_nothing_red(self):
        # Elbow 139 against 145-165: red (past the 3 deg margin). The rest is
        # clean, so the score would be 95 -- it stops at 89, as the clip's does
        # (c6759bd).
        ranges = _road_ranges()
        r = PA._score_photo_angles(
            {"knee": 70.0, "hip": 62.0, "trunk": 42.0, "shoulder": 90.0, "elbow": 139.0},
            ranges, "bike", PA._BIKE_PHASE_WEIGHTS["tdc"], "road_hoods", "tdc")
        assert r["red_rows"] == ["elbow"]
        assert r["overall_score"] == 89 and r["grade"] == "Good"

    def test_amber_rows_do_not_hold_it(self):
        ranges = _road_ranges()
        r = PA._score_photo_angles(
            {"knee": 70.0, "hip": 62.0, "trunk": 42.0, "shoulder": 90.0, "elbow": 143.0},
            ranges, "bike", PA._BIKE_PHASE_WEIGHTS["tdc"], "road_hoods", "tdc")
        assert "red_rows" not in r and r["overall_score"] >= 90

    def test_an_open_hip_costs_nothing_and_holds_nothing(self):
        ranges = _road_ranges()  # hip at TDC 55-70
        r = PA._score_photo_angles(
            {"knee": 70.0, "hip": 85.0, "trunk": 42.0, "shoulder": 90.0, "elbow": 155.0},
            ranges, "bike", PA._BIKE_PHASE_WEIGHTS["tdc"], "road_hoods", "tdc")
        assert "red_rows" not in r and r["overall_score"] == 100

    def test_upright_keeps_the_symmetric_rule(self):
        assert not trunk_below_road_floor("casual", 20.0)
        assert not trunk_low_is_amber("casual", 20.0)
        assert trunk_low_is_amber("road_hoods", 31.0) and trunk_low_is_amber("triathlon", 14.0)


class TestTheAutoPosition:
    def test_the_road_is_never_guessed_as_drops(self):
        for trunk in (25.0, 31.0, 36.0, 44.0):
            assert PA._auto_detect_cycling_position(trunk) in ("road_hoods", "tt_aero")
        assert PA._auto_detect_cycling_position(31.0) == "road_hoods"


class TestTheHelmetOffAeroBars:
    def _run(self, monkeypatch, position):
        from test_photo_pedal_phase import _bike_pose

        fake = {"value": 36.0, "check": True, "back_point": [0.5, 0.3],
                "head_point": [0.45, 0.22], "line": [[0.5, 0.3], [0.45, 0.3]],
                "moved": []}
        monkeypatch.setattr(PA, "helmet_above_back", lambda *a, **k: dict(fake))
        world, norm = _bike_pose(72.0)
        return PA.analyze_from_pose(
            np.full((1280, 720, 3), 128, np.uint8), world, norm, "right", [],
            "bike", position, silhouette_top=[0.2] * 720,
        )

    @pytest.mark.parametrize("position", ["road_hoods", "road_drops", "casual"])
    def test_road_photos_leave_it_out_of_every_surface(self, monkeypatch, position):
        res = self._run(monkeypatch, position)
        assert "helmet_above_back" not in res["angles"]
        assert "helmet_above_back" not in res["angles_with_context"]
        assert "helmet_points" not in res
        assert PA._HELMET_CHECK_WARNING not in res["warnings"]
        assert res["helmet_research"] == 36.0

    def test_aero_photos_keep_it(self, monkeypatch):
        res = self._run(monkeypatch, "triathlon")
        assert res["angles"]["helmet_above_back"] == 36.0
        assert "helmet_points" in res and "helmet_research" not in res
