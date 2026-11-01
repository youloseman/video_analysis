"""Pre-scan: what a clip shows, from a few frames, before the athlete says.

What would hurt if it broke: a run proposed as a ride (the wrong analyzer runs
on a paid clip), a bike proposed as aero when the rider is on the hoods (the
wrong bands grade it), or a guess where the frames disagree -- the app is
supposed to ask then. The frame numbers below are medians measured on real
clips in upload/ (2026-10-21).
"""
from __future__ import annotations

import io
from types import SimpleNamespace

import pytest

from app.services.prescan import classify, frame_features


def _f(trunk, fwd, down, elbow, forearm, back, drop=1.4):
    return {"trunk_deg": trunk, "wrist_fwd": fwd, "wrist_down": down, "elbow_deg": elbow,
            "forearm_deg": forearm, "ankle_back": back, "ankle_drop": drop, "hip_x": 3.0, "hip_y": 4.0}


RUN = _f(86, 0.2, 0.6, 90, -30, 1.0)          # IMG_3979-like: upright, foot far behind
AERO = _f(24, 0.75, 0.4, 93, 16, 0.15)        # IMG_4088-like: low, elbows bent, hands up
HOODS = _f(38, 0.7, 0.95, 155, -45, 0.05)     # IMG_4527-like: arms long, hands below elbow
UPRIGHT = _f(62, 0.6, 0.6, 150, -30, 0.1)     # a city bike: trunk high but hands on bars


@pytest.mark.parametrize("frame, sport, family", [
    (RUN, "run", None), (AERO, "bike", "aero"), (HOODS, "bike", "road"), (UPRIGHT, "bike", "upright"),
])
def test_six_frames_of_one_thing_are_read_as_that_thing(frame, sport, family):
    got = classify([dict(frame) for _ in range(6)])
    assert (got["sport"], got["family"]) == (sport, family)


def test_too_few_frames_or_frames_that_disagree_mean_ask():
    assert classify([RUN, RUN])["sport"] is None                       # a clip needs 3
    assert classify([RUN, RUN, RUN, AERO, AERO, AERO])["sport"] is None
    # an upside-down pose (trunk below the horizon) is no evidence at all
    flipped = dict(HOODS, trunk_deg=-60.0)
    assert classify([flipped] * 6) == {"sport": None, "family": None, "frames": 0}


def test_a_photo_of_a_runner_is_a_run():
    assert classify([RUN], photo=True)["sport"] == "run"
    # midstance: the swing foot only just behind the hip (treadmill p13: 0.32)
    assert classify([dict(RUN, ankle_back=0.32)], photo=True)["sport"] == "run"


def test_an_upright_rider_with_the_foot_forward_is_still_a_bike():
    # the two bike photos as upright as a runner: foot ahead of the hip
    assert classify([dict(UPRIGHT, trunk_deg=85.0, ankle_back=-0.17)], photo=True)["sport"] == "bike"
    assert classify([UPRIGHT], photo=True)["sport"] == "bike"


def test_a_photo_is_otherwise_a_bike_and_one_frame_is_enough():
    assert classify([AERO], photo=True)["family"] == "aero"
    no_arm = {k: v for k, v in AERO.items() if k not in ("elbow_deg", "forearm_deg", "wrist_fwd", "wrist_down")}
    assert classify([no_arm], photo=True) == {"sport": "bike", "family": None, "frames": 1,
                                              "measured": classify([no_arm], photo=True)["measured"]}


def test_features_come_out_in_trunk_lengths_whatever_the_frame_size():
    # A rider facing right, trunk at 45 degrees, forearm level on the pads.
    pts = {i: (0.5, 0.5) for i in range(33)}
    pts.update({0: (0.62, 0.30), 11: (0.55, 0.35), 12: (0.55, 0.35), 23: (0.45, 0.45), 24: (0.45, 0.45),
                13: (0.58, 0.45), 14: (0.58, 0.45), 15: (0.68, 0.45), 16: (0.68, 0.45),
                27: (0.47, 0.70), 28: (0.47, 0.70)})
    lms = [SimpleNamespace(x=x, y=y, visibility=0.9) for x, y in (pts[i] for i in range(33))]
    small, big = frame_features(lms, 640, 640), frame_features(lms, 1280, 1280)
    assert small["trunk_deg"] == pytest.approx(45, abs=0.5)
    assert small["forearm_deg"] == pytest.approx(0, abs=0.5)
    assert small["wrist_fwd"] == pytest.approx(big["wrist_fwd"])


async def test_the_endpoint_refuses_floods_and_big_frames(monkeypatch):
    from fastapi import HTTPException, UploadFile

    from app.api import prescan as api

    monkeypatch.setattr(api, "PER_IP", 2)
    req = SimpleNamespace(headers={}, client=SimpleNamespace(host="10.9.8.7"))
    big = UploadFile(io.BytesIO(b"x" * (api.MAX_FRAME_BYTES + 10)), filename="f.jpg")
    with pytest.raises(HTTPException) as e:
        await api.prescan(req, [big], "video")
    assert e.value.status_code == 413
    with pytest.raises(HTTPException) as e:
        await api.prescan(req, [UploadFile(io.BytesIO(b"x"), filename="f.jpg")] * 9, "video")
    assert e.value.status_code == 400
    with pytest.raises(HTTPException) as e:
        await api.prescan(req, [UploadFile(io.BytesIO(b"x"), filename="f.jpg")], "video")
    assert e.value.status_code == 429


# ---- GO / NO-GO on the filming (capture_check) ------------------------------
# Mike, bike fitter (2026-11-01): flag a badly filmed clip before it is
# analysed. What would hurt: a good clip told to re-film (a lost upload), or a
# clip the model cannot hold its legs on waved through as GO.

from app.services.prescan import capture_check  # noqa: E402


def _cap(frame, n=6, **kw):
    return capture_check([dict(frame) for _ in range(n)], **kw)


def test_a_rider_who_fills_the_frame_from_the_quiet_side_is_go():
    # 1080p landscape -> 1280x720 for the detector; 0.8 of it is 576 px.
    got = _cap(dict(AERO, facing=-1.0, height_frac=0.8), sport="bike", video_w=1920, video_h=1080)
    assert got["verdict"] == "go" and got["reasons"] == [] and got["subject_px"] == 576


def test_too_small_to_hold_the_legs_is_no_go_and_says_so_first():
    got = _cap(dict(AERO, facing=1.0, height_frac=0.35), sport="bike", video_w=1920, video_h=1080)
    assert got["verdict"] == "no_go"
    assert [r["code"] for r in got["reasons"]] == ["tiny", "drive_side"]


def test_small_but_workable_and_the_chainring_side_are_only_check():
    small = _cap(dict(RUN, facing=1.0, height_frac=0.5), sport="run", video_w=1920, video_h=1080)
    assert small["verdict"] == "check" and [r["code"] for r in small["reasons"]] == ["small"]
    drive = _cap(dict(AERO, facing=1.0, height_frac=0.8), sport="bike", video_w=1920, video_h=1080)
    assert drive["verdict"] == "check" and [r["code"] for r in drive["reasons"]] == ["drive_side"]


def test_the_size_is_judged_in_the_frame_the_detector_sees_not_the_camera():
    # A 4K clip is shrunk to 1280 on the long edge, so it buys nothing.
    four_k = _cap(dict(AERO, facing=-1.0, height_frac=0.5), sport="bike", video_w=3840, video_h=2160)
    hd = _cap(dict(AERO, facing=-1.0, height_frac=0.5), sport="bike", video_w=1280, video_h=720)
    assert four_k["subject_px"] == hd["subject_px"] == 360


def test_without_the_clip_size_or_on_a_photo_the_size_is_not_judged():
    assert _cap(dict(AERO, facing=-1.0, height_frac=0.2), sport="bike")["verdict"] == "go"
    photo = _cap(dict(AERO, facing=1.0, height_frac=0.2), n=1, sport="bike",
                 video_w=1920, video_h=1080, photo=True)
    assert [r["code"] for r in photo["reasons"]] == ["drive_side"]


def test_nobody_found_is_a_check_not_a_no_go():
    # Six 640 px frames can miss a rider the full analysis at 1280 finds.
    got = capture_check([dict(AERO)] * 2, sport=None, video_w=1920, video_h=1080)
    assert got["verdict"] == "check" and got["reasons"][0]["code"] == "not_found"


def test_frame_features_carry_the_facing_and_the_nose_to_ankle_span():
    pts = {i: (0.5, 0.5) for i in range(33)}
    pts.update({0: (0.62, 0.30), 11: (0.55, 0.35), 12: (0.55, 0.35), 23: (0.45, 0.45), 24: (0.45, 0.45),
                27: (0.47, 0.70), 28: (0.47, 0.80)})
    lms = [SimpleNamespace(x=x, y=y, visibility=0.9) for x, y in (pts[i] for i in range(33))]
    f = frame_features(lms, 1280, 720)
    assert f["facing"] == 1.0
    assert f["height_frac"] == pytest.approx(0.5)      # nose 0.30 to the LOWER ankle 0.80
