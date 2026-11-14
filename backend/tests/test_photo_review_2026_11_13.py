"""Thirty pro photos run through the photo path (2026-11-13): what was fixed."""
from __future__ import annotations

from app.services.video_analysis import photo_analyzer as pa


def test_an_open_hip_at_the_top_costs_a_bike_photo_nothing():
    # Kona pros at the top of the stroke read hip 73-74 against 45-62 and
    # lost 21-23 points for it; the video and the table call it fine.
    ranges = pa._get_cycling_optimal_ranges("triathlon", "tdc")
    out = pa._score_photo_angles({"hip": 74.0, "knee": 72.0, "trunk": 22.0}, ranges, "bike",
                                 pa._BIKE_PHASE_WEIGHTS["tdc"], cycling_position="triathlon",
                                 pedal_phase="tdc")
    assert out["per_angle"]["hip"]["score"] == 100


def test_a_closed_hip_still_costs():
    ranges = pa._get_cycling_optimal_ranges("triathlon", "tdc")
    out = pa._score_photo_angles({"hip": 30.0, "knee": 72.0, "trunk": 22.0}, ranges, "bike",
                                 pa._BIKE_PHASE_WEIGHTS["tdc"], cycling_position="triathlon",
                                 pedal_phase="tdc")
    assert out["per_angle"]["hip"]["score"] < 100


def test_a_face_on_photo_is_told_it_is_face_on():
    assert "not_side_view" in pa._WITHHELD_MESSAGES
    assert "front or the back" in pa._WITHHELD_MESSAGES["not_side_view"]
    assert pa._FRONT_VIEW_WARNING == "The photo appears to be from the front or back"


def test_one_touchdown_cannot_sink_a_run_photo_to_fair():
    # The MSC race photo: everything green but a shin leaning back 30 deg at a
    # straight-knee touchdown read 60/Fair; a still cannot tell the touchdown
    # from the instant before it. The shin still costs, but no lower than 50.
    angles = {"knee": 177.8, "hip": 145.8, "ankle": 106.9, "trunk": 2.3}
    ranges = {"knee": (150.0, 175.0), "hip": (140.0, 180.0), "ankle": (80.0, 120.0),
              "trunk": (2.0, 10.0), "shin": pa._RUN_SHIN_BAND}
    out = pa._score_photo_angles({**angles, "shin": -29.6}, ranges, "run",
                                 {**pa.PHOTO_RUNNING_WEIGHTS, "shin": pa._RUN_SHIN_WEIGHT})
    assert out["per_angle"]["shin"]["score"] == pa._RUN_SHIN_FLOOR
    assert 70 <= out["overall_score"] < 90


def test_a_moderate_lean_back_is_scored_as_before():
    out = pa._score_photo_angles({"shin": -15.0}, {"shin": pa._RUN_SHIN_BAND}, "run",
                                 {"shin": pa._RUN_SHIN_WEIGHT})
    assert out["per_angle"]["shin"]["score"] == 75
