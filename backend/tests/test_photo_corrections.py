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


def _silhouette_top() -> list:
    """A rider's top edge, as head_silhouette.top_contour hands it over: a
    straight back from hip to shoulder a little above the joint line, then a
    helmet rising over it in front of the shoulder, nothing elsewhere."""
    from app.services.video_analysis.biomechanics.head_silhouette import CONTOUR_COLUMNS

    edge = []
    for j in range(CONTOUR_COLUMNS):
        x = (j + 0.5) / CONTOUR_COLUMNS
        if 0.34 <= x <= 0.76:
            edge.append(round(0.345 - (x - 0.38) * (0.075 / 0.34), 4))
        elif 0.76 < x <= 0.95:
            edge.append(0.22)
        else:
            edge.append(None)
    return edge


def test_the_helmet_reading_survives_a_correction():
    """The silhouette edge travels in the signed pose, so a re-measured photo
    keeps its helmet-above-back row; moving the shoulder moves the torso the
    reading is a share of, so the number follows the correction."""
    photo = _photo_bytes()
    pose = _pose_result(photo)
    pose["silhouette_top"] = _silhouette_top()
    blob = PC.build_pose_blob(photo, pose)
    assert blob["silhouette_top"] == pose["silhouette_top"]

    res = PC.recompute_photo(photo, "bike", "triathlon", blob,
                             [{"landmark": 12, "dx": 0.0, "dy": 0.03}])
    before = res["baseline"]["angles"]["helmet_above_back"]
    after = res["angles"]["helmet_above_back"]
    assert before > 0 and after != before
    row = res["angles_with_context"]["helmet_above_back"]
    if res["helmet_points"]["check"]:
        assert row["status"] == "phase_dependent"
        assert "helmet_above_back" not in res["score"]["per_angle"], "a flagged reading is not scored"
    else:
        assert row["status"] in ("optimal", "acceptable") and row["optimal_max"] == 23.0
        assert "helmet_above_back" in res["score"]["per_angle"]


def test_a_tampered_silhouette_is_refused():
    photo = _photo_bytes()
    pose = _pose_result(photo)
    pose["silhouette_top"] = _silhouette_top()
    blob = PC.build_pose_blob(photo, pose)
    blob["silhouette_top"] = [0.0 if v is not None else None for v in blob["silhouette_top"]]
    with pytest.raises(ValueError, match="not issued by this server"):
        PC.verify_pose_blob(blob, photo)


def test_a_pose_issued_before_the_silhouette_still_verifies():
    """Blobs from before the silhouette existed carry no such key; a rider
    mid-edit across the deploy must not be told to analyze again."""
    photo = _photo_bytes()
    blob = PC.build_pose_blob(photo, _pose_result(photo))
    assert "silhouette_top" not in blob
    PC.verify_pose_blob(blob, photo)
    res, _ = _recompute([{"landmark": 24, "dx": 0.0, "dy": 0.02}])
    assert "helmet_above_back" not in res["angles"]


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


def _helmet_blob():
    photo = _photo_bytes()
    pose = _pose_result(photo)
    pose["silhouette_top"] = _silhouette_top()
    return photo, PC.build_pose_blob(photo, pose, helmet_points=True)


def test_the_top_of_back_and_helmet_are_offered_for_dragging():
    _, blob = _helmet_blob()
    assert 100 in blob["draggable"] and 101 in blob["draggable"]
    photo = _photo_bytes()
    assert 100 not in PC.build_pose_blob(photo, _pose_result(photo))["draggable"]


def test_moving_the_top_of_back_moves_the_helmet_reading_and_nothing_else():
    """Artur, 2026-10-11: on a helmet tail lying flush on the back the
    silhouette cannot tell where the back ends, so the athlete places the
    point. It is not a joint: no angle, bone or landmark may move with it."""
    photo, blob = _helmet_blob()
    res = PC.recompute_photo(photo, "bike", "triathlon", blob,
                             [{"landmark": 100, "dx": 0.0, "dy": 0.02}])
    base = res["baseline"]["angles"]
    assert res["angles"]["helmet_above_back"] > base["helmet_above_back"], "lower back top, taller helmet"
    for k, v in base.items():
        if k != "helmet_above_back":
            assert res["angles"][k] == v, k
    hp = res["helmet_points"]
    assert hp["moved"] == [100] and hp["check"] is False
    assert hp["auto_back"] != hp["back"]
    assert res["plausibility_warnings"] == []


def test_a_point_that_is_neither_a_joint_nor_a_helmet_point_is_refused():
    photo, blob = _helmet_blob()
    with pytest.raises(ValueError, match="cannot be adjusted"):
        PC.recompute_photo(photo, "bike", "triathlon", blob, [{"landmark": 102, "dx": 0.0, "dy": 0.01}])


class TestTheHelmetPenalty:
    """Artur, 2026-10-11: penalise a head held too high. Free up to the
    point where twenty-nine of the thirty-three Kona pros sat (+23 % of torso), amber and never red above it, only on
    aero positions and only on a reading that can be trusted."""

    def _score(self, value, position="triathlon", check=False):
        from app.services.video_analysis import photo_analyzer as PA
        angles = {"trunk": 25.0, "shoulder": 90.0, "elbow": 90.0, "forearm_tilt": 15.0,
                  "helmet_above_back": value}
        ranges = PA._with_helmet_band(PA._get_cycling_optimal_ranges(position, "uncertain"),
                                      position in PA._AERO_POSITIONS and not check)
        return PA._score_photo_angles(angles, ranges, "bike", PA.PHOTO_CYCLING_WEIGHTS_UNCERTAIN,
                                      cycling_position=position, pedal_phase="uncertain")

    def test_up_to_the_highest_pro_costs_nothing(self):
        assert self._score(23.0)["per_angle"]["helmet_above_back"]["score"] == 100
        assert self._score(-5.0)["per_angle"]["helmet_above_back"]["score"] == 100

    def test_above_it_costs_a_little_and_never_more_than_amber(self):
        hi = self._score(33.0)
        assert hi["per_angle"]["helmet_above_back"]["score"] == 80
        assert self._score(80.0)["per_angle"]["helmet_above_back"]["score"] == 70, "floored"
        assert hi["overall_score"] < self._score(20.0)["overall_score"]

    def test_a_flagged_reading_and_a_road_position_are_not_scored(self):
        assert "helmet_above_back" not in self._score(40.0, check=True)["per_angle"]
        assert "helmet_above_back" not in self._score(40.0, position="road_hoods")["per_angle"]
