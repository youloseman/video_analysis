"""Where the rider sits over the crank, from points the athlete placed.

The reading the Kona 2026 pros and age-groupers differed on (3.2 deg, ~4-5 cm).
What would hurt if it broke: the angle read on the wrong side of the rider,
a tilted phone read as a position, a free account getting a paid reading,
or the clip's hip taken from one frame instead of the median.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.services.video_analysis.biomechanics import hip_over_bb as HOB


def _pt(x, y, vis=1.0):
    return SimpleNamespace(x=x, y=y, z=0.0, visibility=vis)


# --- the geometry -----------------------------------------------------------

def test_the_angle_is_read_on_the_back_side_whichever_way_the_rider_faces():
    """Hip 5 % behind the crank, 80 % of the frame above it: ~86.4 deg."""
    right = HOB.hip_over_bb((0.45, 0.2), {"C": (0.5, 1.0)}, aspect=1.0, shoulder=(0.7, 0.2))
    left = HOB.hip_over_bb((0.55, 0.2), {"C": (0.5, 1.0)}, aspect=1.0, shoulder=(0.3, 0.2))
    assert right["facing"] == "right" and left["facing"] == "left"
    assert right["angle_deg"] == left["angle_deg"] == pytest.approx(86.4, abs=0.1)
    # hip AHEAD of the crank: past vertical
    ahead = HOB.hip_over_bb((0.55, 0.2), {"C": (0.5, 1.0)}, aspect=1.0, shoulder=(0.7, 0.2))
    assert ahead["angle_deg"] == pytest.approx(93.6, abs=0.1)


def test_the_aspect_ratio_turns_normalized_offsets_into_proportions():
    tall = HOB.hip_over_bb((0.45, 0.2), {"C": (0.5, 1.0)}, aspect=9 / 16, shoulder=(0.7, 0.2))
    assert tall["angle_deg"] == pytest.approx(88.0, abs=0.1)


def test_the_wheel_axles_level_a_tilted_phone():
    """The same bike shot with the phone rolled 3 deg: against the frame the
    hip reads 3 deg off; against the axles it reads the same as level."""
    import math

    def rot(p, deg, about=(0.5, 0.5)):
        a = math.radians(deg)
        dx, dy = p[0] - about[0], p[1] - about[1]
        return (about[0] + dx * math.cos(a) - dy * math.sin(a), about[1] + dx * math.sin(a) + dy * math.cos(a))

    hip, C, FA, RA, S = (0.45, 0.2), (0.5, 0.9), (0.9, 0.85), (0.1, 0.85), (0.7, 0.2)
    level = HOB.hip_over_bb(hip, {"C": C, "FA": FA, "RA": RA}, aspect=1.0, shoulder=S)
    tilted_pts = {k: rot(v, 3.0) for k, v in {"C": C, "FA": FA, "RA": RA}.items()}
    tilted = HOB.hip_over_bb(rot(hip, 3.0), tilted_pts, aspect=1.0, shoulder=rot(S, 3.0))
    frame_only = HOB.hip_over_bb(rot(hip, 3.0), {"C": tilted_pts["C"]}, aspect=1.0, shoulder=rot(S, 3.0))
    assert level["ref"] == "axles" and abs(level["tilt_deg"]) < 0.1
    assert tilted["angle_deg"] == pytest.approx(level["angle_deg"], abs=0.15)
    assert abs(tilted["tilt_deg"]) == pytest.approx(3.0, abs=0.1)
    assert abs(frame_only["angle_deg"] - level["angle_deg"]) == pytest.approx(3.0, abs=0.15)
    assert frame_only["ref"] == "frame"


def test_the_crank_gives_the_centimetres_and_the_default_is_labelled():
    # pedal 0.06 of the frame below the crank, hip 0.8 above: 172.5 mm cranks
    # put the hip 0.8/0.06*17.25 = 230 cm away -- not a human, so typical scale.
    far = HOB.hip_over_bb((0.47, 0.1), {"C": (0.5, 0.9), "P": (0.5, 0.96)}, aspect=1.0)
    assert far["scale_source"] == "typical" and far["cm_per_deg"] == HOB.CM_PER_DEG_TYPICAL
    # pedal 0.2 below, hip 0.8 above: 0.8/0.2*17.5 = 70 cm, a rider.
    ok = HOB.hip_over_bb((0.47, 0.0), {"C": (0.5, 0.8), "P": (0.5, 1.0)}, aspect=1.0,
                         crank_length_mm=175.0)
    assert ok["scale_source"] == "crank_given" and ok["crank_to_hip_cm"] == pytest.approx(70.1, abs=0.2)
    assert ok["cm_per_deg"] == pytest.approx(70.1 * 3.14159 / 180, abs=0.01)
    assert ok["gap_pro_cm"] == pytest.approx(ok["gap_pro_deg"] * ok["cm_per_deg"], abs=0.1)


def test_the_field_rides_along_and_is_a_field():
    out = HOB.hip_over_bb((0.45, 0.2), {"C": (0.5, 1.0)}, aspect=1.0, cycling_position="triathlon")
    assert out["field"]["pro"]["median"] == 88.0 and out["field"]["age_group"]["median"] == 85.1
    assert out["aero"] is True and out["repeatability_deg"] == 1.4
    assert out["in_age_group_middle_half"] and not out["in_pro_middle_half"]
    assert "band" not in json.dumps(out["field"])


# --- what the athlete sends -------------------------------------------------

@pytest.mark.parametrize("raw, msg", [
    ({}, "crank axle first"),
    ({"C": [0.5, 1.2]}, "outside the picture"),
    ({"C": [0.5, 0.9], "FA": [0.9, 0.85]}, "both wheel axles"),
    ({"C": [0.5, 0.9], "FA": [0.5, 0.85], "RA": [0.51, 0.85]}, "on top of each other"),
    ({"C": [0.5, 0.9], "X": [0.5, 0.85]}, "unknown point"),
    ({"C": "here"}, r"must be \[x, y\]"),
])
def test_bad_points_are_refused_with_a_reason(raw, msg):
    with pytest.raises(ValueError, match=msg):
        HOB.normalize_bike_points(raw)


def test_axles_far_from_level_are_refused():
    with pytest.raises(ValueError, match="level bike"):
        HOB.hip_over_bb((0.45, 0.2), {"C": (0.5, 0.9), "FA": (0.9, 0.3), "RA": (0.1, 0.9)},
                        aspect=1.0, shoulder=(0.7, 0.2))


# --- the clip's hip ---------------------------------------------------------

def test_the_clip_hip_is_the_median_with_corrections_applied():
    def frame(hx, hy, vis=1.0):
        lm = [_pt(0.0, 0.0)] * 33
        lm[24] = _pt(hx, hy, vis)
        lm[12] = _pt(hx + 0.2, hy)
        return {"normalized_landmarks": lm}

    frames = [frame(0.40, 0.30), frame(0.42, 0.31), frame(0.90, 0.90, vis=0.1), frame(0.44, 0.29)]
    hip, sh, n = HOB.median_hip(frames, "right", [{"landmark": 24, "dx": 0.01, "dy": 0.0}])
    assert n == 3
    assert hip == pytest.approx((0.43, 0.30))
    assert sh[0] == pytest.approx(0.62, abs=0.03)
    assert HOB.median_hip([], "right") == (None, None, 0)


# --- the photo path ---------------------------------------------------------

def test_a_photo_can_be_measured_for_the_crank_alone_and_the_picture_gets_the_line():
    from tests.test_photo_corrections import _photo_bytes, _pose_result

    from app.services import photo_corrections as PC

    photo = _photo_bytes()
    blob = PC.build_pose_blob(photo, _pose_result(photo))
    hip = blob["landmarks"][24]
    res = PC.recompute_photo(photo, "bike", "triathlon", blob, [],
                             bike_points={"C": [hip[0] + 0.03, min(0.99, hip[1] + 0.4)]})
    hob = res["hip_over_bb"]
    assert 60 < hob["angle_deg"] < 120 and hob["hip_source"] == "photo"
    assert res["bike_points"]["C"] == hob["points"]["C"]
    assert not res.get("corrections") and res["thumbnail_base64"].startswith("data:image/jpeg")
    with pytest.raises(ValueError, match="No adjustment"):
        PC.recompute_photo(photo, "bike", "triathlon", blob, [])
