"""Re-measuring a photo after the athlete moves a joint.

Nothing is stored between the two requests, so the pose the model found
travels to the client signed and comes back with the same photo. The tests
pin what makes that trustworthy: a tampered pose is refused, a pose from
another photo is refused, the corrected report says what was moved and keeps
the automatic reading beside it, and a moved point actually reaches the
number it feeds.

Needs OpenCV (decode + thumbnail) but no MediaPipe: that is the point of the
endpoint, and the import skip mirrors that.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="photo re-measurement decodes and draws with OpenCV")

from types import SimpleNamespace  # noqa: E402

from app.services import photo_corrections as PC  # noqa: E402
from app.services.correction_limits import (  # noqa: E402
    daily_limit,
    reset,
    take_recompute,
)

W, H = 720, 1280


def _rider() -> tuple[list, list]:
    """A right-side TT rider near the bottom of the stroke, in normalized
    image coordinates (x over the width, y over the height, all inside the
    picture)."""
    pts = {
        24: (0.38, 0.375), 26: (0.50, 0.58), 28: (0.42, 0.76),
        30: (0.38, 0.775), 32: (0.50, 0.78),
        12: (0.72, 0.30), 14: (0.82, 0.36), 16: (0.92, 0.36), 8: (0.86, 0.27),
        0: (0.93, 0.28),
    }
    norm, world = [], []
    for j in range(33):
        right = j in (8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32)
        left = j in (7, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31)
        x, y = pts.get(j + 1, (0.88, 0.27)) if left else pts.get(j, (0.88, 0.27))
        x = x - (0.01 if left else 0.0)
        z = -0.1 if right else (0.1 if left else 0.0)
        vis = 0.6 if left else 1.0
        norm.append(SimpleNamespace(x=x, y=y, z=z, visibility=vis))
        world.append(SimpleNamespace(x=(x - 0.5) * 1.2, y=(y - 0.5) * 2, z=z, visibility=vis))
    return world, norm


def _photo_bytes(seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    img = rng.integers(40, 200, size=(H, W, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _pose_result(image_bytes: bytes) -> dict:
    world, norm = _rider()
    return {
        "world": world, "normalized": norm,
        "image": cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR),
        "camera_side": "right", "avg_visibility": 0.95, "warnings": [],
    }


@pytest.fixture(autouse=True)
def _fresh_daily_counter():
    reset()
    yield
    reset()


# --- the signed pose --------------------------------------------------------

def test_the_pose_round_trips_and_names_the_draggable_points():
    photo = _photo_bytes()
    blob = PC.build_pose_blob(photo, _pose_result(photo))
    assert blob["draggable"] == [8, 12, 14, 16, 24, 26, 28, 30, 32]
    assert len(blob["landmarks"]) == 33 and len(blob["world"]) == 33
    assert PC.verify_pose_blob(blob, photo) is blob


def test_a_tampered_pose_is_refused():
    photo = _photo_bytes()
    blob = PC.build_pose_blob(photo, _pose_result(photo))
    blob["landmarks"][24][1] -= 0.05          # "the hip was actually higher"
    with pytest.raises(ValueError, match="not issued by this server"):
        PC.verify_pose_blob(blob, photo)


def test_a_pose_from_another_photo_is_refused():
    photo = _photo_bytes(1)
    blob = PC.build_pose_blob(photo, _pose_result(photo))
    with pytest.raises(ValueError, match="different photo"):
        PC.verify_pose_blob(blob, _photo_bytes(2))


def test_a_pose_from_a_different_version_is_refused():
    photo = _photo_bytes()
    blob = PC.build_pose_blob(photo, _pose_result(photo))
    blob["v"] = PC.POSE_VERSION + 1
    with pytest.raises(ValueError, match="different version"):
        PC.verify_pose_blob(blob, photo)


def test_garbage_is_refused_before_anything_is_decoded():
    with pytest.raises(ValueError):
        PC.verify_pose_blob("nope", b"x")
    with pytest.raises(ValueError):
        PC.verify_pose_blob({"v": 1, "landmarks": [], "world": []}, b"x")


# --- the re-measurement -----------------------------------------------------

def _recompute(corrections, position="triathlon"):
    photo = _photo_bytes()
    blob = PC.build_pose_blob(photo, _pose_result(photo))
    return PC.recompute_photo(photo, "bike", position, blob, corrections), blob


def test_a_moved_hip_reaches_the_knee_angle_and_the_baseline_stays():
    """Lowering the hip point (toward the ankle) bends the knee more."""
    res, blob = _recompute([{"landmark": 24, "dx": 0.0, "dy": 0.05}])

    assert res["corrections"] == [{"landmark": 24, "dx": 0.0, "dy": 0.05}]
    assert res["angles"]["knee"] < res["baseline"]["angles"]["knee"] - 3
    assert res["baseline"]["score"]["overall_score"] is not None
    assert res["thumbnail_base64"].startswith("data:image/jpeg;base64,")
    assert res["pose"] is blob
    assert res["cycling_position"] == "triathlon"


def test_a_moved_ear_reaches_the_head_alignment_score():
    """The one bike reading that used to come off the world skeleton; a moved
    ear changing the picture and not the score was a bug waiting to be found."""
    base, _ = _recompute([{"landmark": 8, "dx": 0.0, "dy": -0.001}])
    moved, _ = _recompute([{"landmark": 8, "dx": 0.0, "dy": -0.08}])
    assert moved["angles"]["head_alignment"] != base["angles"]["head_alignment"]


def test_the_geometry_gets_a_say_on_a_single_photo():
    res, _ = _recompute([{"landmark": 26, "dx": 0.0, "dy": -0.12}])
    segments = {w["segment"] for w in res["plausibility_warnings"]}
    assert "hip-knee" in segments and "knee-ankle" in segments


def test_nothing_moved_is_not_a_re_measurement():
    with pytest.raises(ValueError, match="move a joint point first"):
        _recompute([])


def test_the_far_leg_cannot_be_adjusted():
    with pytest.raises(ValueError, match="right-side joints"):
        _recompute([{"landmark": 23, "dx": 0.01, "dy": 0.0}])


def test_running_photos_are_refused():
    photo = _photo_bytes()
    blob = PC.build_pose_blob(photo, _pose_result(photo))
    with pytest.raises(ValueError, match="cycling photos only"):
        PC.recompute_photo(photo, "run", None, blob, [{"landmark": 24, "dx": 0.01, "dy": 0}])


def test_the_corrected_pose_keeps_the_position_it_was_filed_under():
    """A moved shoulder must not re-file the rider under another position
    and silently change every band they are judged against."""
    res, _ = _recompute([{"landmark": 12, "dx": 0.0, "dy": 0.10}], position=None)
    assert res["baseline"]["cycling_position"] in {"tt_aero", "triathlon", "road_drops", "road_hoods", "casual"}
    assert res["cycling_position"] == res["baseline"]["cycling_position"]


def test_the_first_result_hands_the_pose_out_for_bike_only(monkeypatch):
    """`analyze_photo` attaches the signed pose so the client can come back."""
    from app.services.video_analysis import photo_analyzer as PA

    photo = _photo_bytes()
    monkeypatch.setattr(PA, "detect_pose_in_image", lambda b: _pose_result(b))
    res = PA.analyze_photo(photo, "bike", "triathlon")
    assert res["pose"]["token"] and res["pose"]["draggable"]
    assert res["corrections"] is None
    assert math.isfinite(res["angles"]["knee"])


# --- the daily cap ----------------------------------------------------------

def test_free_accounts_have_no_adjustments_at_all():
    assert daily_limit("starter") == 0
    assert take_recompute(1, "starter") == (False, 0, 0)


def test_the_cap_counts_a_rolling_day():
    limit = daily_limit("enthusiast")
    for i in range(limit):
        assert take_recompute(7, "enthusiast", now=1000.0 + i)[0]
    assert take_recompute(7, "enthusiast", now=2000.0) == (False, limit, limit)
    # A full day after the LAST of them, every stamp has aged out.
    allowed, used, _ = take_recompute(7, "enthusiast", now=1000.0 + limit + 24 * 3600)
    assert allowed and used == 1


def test_accounts_do_not_share_a_cap():
    limit = daily_limit("enthusiast")
    for i in range(limit):
        take_recompute(1, "enthusiast", now=1000.0 + i)
    assert take_recompute(2, "enthusiast", now=1000.0)[0]
