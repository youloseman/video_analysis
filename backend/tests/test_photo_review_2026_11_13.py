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
