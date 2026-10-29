"""Dragging a joint point on a running photo (behind the run-photo read).

The run photo names its legs by position: the lowest foot is the stance leg
and is moved onto the camera-side indices (photo_analyzer._name_run_legs). The
signed pose handed to the client has to be THAT pose, or the editor offers the
leg in the air for dragging while the report measures the one on the ground.
"""
from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="photo re-measurement decodes and draws with OpenCV")

from app.services import photo_corrections as PC  # noqa: E402
from app.services.correction_limits import reset  # noqa: E402

W, H = 720, 1280

# A runner filmed from the right, moving right, landing on the leg the model
# labelled LEFT (far) -- the treadmill case: the label is on the wrong leg.
_STANCE = {25: (0.56, 0.66), 27: (0.61, 0.82), 29: (0.59, 0.84), 31: (0.65, 0.845)}
_SWING = {26: (0.43, 0.63), 28: (0.36, 0.71), 30: (0.34, 0.70), 32: (0.37, 0.735)}
_BODY = {
    0: (0.58, 0.21), 7: (0.54, 0.22), 8: (0.54, 0.22),
    11: (0.53, 0.30), 12: (0.53, 0.30), 13: (0.47, 0.40), 14: (0.47, 0.40),
    15: (0.55, 0.45), 16: (0.55, 0.45), 23: (0.50, 0.50), 24: (0.50, 0.50),
}


def _runner():
    pts = {**_BODY, **_STANCE, **_SWING}
    norm, world = [], []
    for j in range(33):
        x, y = pts.get(j, (0.56, 0.21))
        z = -0.1 if j % 2 == 0 else 0.1      # right = near
        norm.append(SimpleNamespace(x=x, y=y, z=z, visibility=0.95))
        world.append(SimpleNamespace(x=(x - 0.5) * 1.2, y=(y - 0.5) * 2, z=z, visibility=0.95))
    return world, norm


def _photo_bytes(seed: int = 3) -> bytes:
    rng = np.random.default_rng(seed)
    img = rng.integers(40, 200, size=(H, W, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _pose_result(image_bytes: bytes) -> dict:
    world, norm = _runner()
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


@pytest.fixture
def analyzed(monkeypatch):
    from app.services.video_analysis import photo_analyzer as PA

    monkeypatch.setattr(PA, "detect_pose_in_image", lambda b: _pose_result(b))
    photo = _photo_bytes()
    return photo, PA.analyze_photo(photo, "run", None, False, True)


def test_the_pose_handed_out_is_the_stance_leg_on_the_near_indices(analyzed):
    _photo, res = analyzed
    assert res["contact"]["legs"]["swapped"] is True
    pose = res["pose"]
    assert pose["sport"] == "run" and pose["camera_side"] == "right"
    # The near knee in the pose is where the STANCE knee is in the picture.
    assert pose["landmarks"][26][:2] == list(_STANCE[25])
    assert pose["landmarks"][25][:2] == list(_SWING[26])
    # No running angle reads the ear.
    assert pose["draggable"] == [12, 14, 16, 24, 26, 28, 30, 32]


def test_without_the_run_photo_read_there_is_no_pose(monkeypatch):
    from app.services.video_analysis import photo_analyzer as PA

    monkeypatch.setattr(PA, "detect_pose_in_image", lambda b: _pose_result(b))
    res = PA.analyze_photo(_photo_bytes(), "run", None, False, False)
    assert "pose" not in res


def test_the_baseline_is_the_reading_the_athlete_saw(analyzed):
    """Re-measuring from the handed-out pose reproduces the first result: the
    second leg-naming pass finds the stance leg already where it belongs."""
    photo, res = analyzed
    out = PC.recompute_photo(photo, "run", None, res["pose"],
                             [{"landmark": 32, "dx": 0.005, "dy": 0.0}])
    assert out["baseline"]["angles"] == res["angles"]
    assert out["baseline"]["score"]["overall_score"] == res["score"]["overall_score"]
    assert out["baseline"]["gait_phase"] == res["gait_phase"]
    assert out["baseline"]["contact"]["overstride_ratio"] == res["contact"]["overstride_ratio"]


def test_a_moved_stance_knee_reaches_the_knee_angle(analyzed):
    photo, res = analyzed
    out = PC.recompute_photo(photo, "run", None, res["pose"],
                             [{"landmark": 26, "dx": 0.03, "dy": 0.0}])
    assert out["corrections"] == [{"landmark": 26, "dx": 0.03, "dy": 0.0}]
    assert out["angles"]["knee"] < out["baseline"]["angles"]["knee"] - 3
    assert out["sport"] == "run" and out.get("cycling_position") is None
    assert out["thumbnail_base64"].startswith("data:image/")


def test_a_moved_ankle_moves_the_overstride(analyzed):
    photo, res = analyzed
    out = PC.recompute_photo(photo, "run", None, res["pose"],
                             [{"landmark": 28, "dx": -0.03, "dy": 0.0}])
    before = out["baseline"]["contact"]["overstride_ratio"]
    after = out["contact"]["overstride_ratio"]
    assert math.isfinite(before) and after < before


def test_a_pose_is_only_good_for_the_sport_it_was_issued_for(analyzed):
    photo, res = analyzed
    with pytest.raises(ValueError, match="different sport"):
        PC.recompute_photo(photo, "bike", "triathlon", res["pose"],
                           [{"landmark": 26, "dx": 0.01, "dy": 0.0}])
    bike = PC.build_pose_blob(photo, _pose_result(photo))
    with pytest.raises(ValueError, match="different sport"):
        PC.recompute_photo(photo, "run", None, bike, [{"landmark": 26, "dx": 0.01, "dy": 0.0}])


def test_the_sport_is_signed(analyzed):
    photo, res = analyzed
    pose = {k: v for k, v in res["pose"].items() if k != "sport"}
    with pytest.raises(ValueError, match="not issued by this server"):
        PC.verify_pose_blob(pose, photo)


def test_no_crank_on_a_running_photo(analyzed):
    photo, res = analyzed
    with pytest.raises(ValueError, match="cycling photos only"):
        PC.recompute_photo(photo, "run", None, res["pose"], [],
                           bike_points={"C": [0.5, 0.9]})


# --- the endpoint -----------------------------------------------------------

main = pytest.importorskip("app.main", reason="needs the analysis stack (mediapipe/opencv)")


def _client(tier):
    from fastapi.testclient import TestClient

    from app.core.security import get_current_user

    user = SimpleNamespace(id=424242, tier=tier, email=f"{tier}@example.com")
    main.app.dependency_overrides[get_current_user] = lambda: user
    return TestClient(main.app)


def _post(client, photo, pose, corrections):
    return client.post(
        "/analyze-photo/recompute",
        files={"photo": ("run.jpg", photo, "image/jpeg")},
        data={"sport": "run", "pose": json.dumps(pose), "coaching": "false",
              "corrections": json.dumps(corrections)},
    )


def test_the_endpoint_re_measures_a_run_for_whoever_gets_the_run_read(analyzed, monkeypatch):
    import dataclasses

    from app.core import config
    from app.core.security import get_current_user

    photo, res = analyzed
    monkeypatch.setattr(config, "settings", dataclasses.replace(config.settings, run_contact_metrics=False))
    try:
        r = _post(_client("full"), photo, res["pose"], [{"landmark": 26, "dx": 0.03, "dy": 0.0}])
        assert r.status_code == 400, r.text

        r = _post(_client("admin"), photo, res["pose"], [{"landmark": 26, "dx": 0.03, "dy": 0.0}])
        assert r.status_code == 200, r.text
        j = r.json()
        assert j["corrections"][0]["landmark"] == 26
        assert j["angles"]["knee"] < j["baseline"]["angles"]["knee"]
        assert j["contact"]["measured"] is True
    finally:
        main.app.dependency_overrides.pop(get_current_user, None)


# The far stance leg's own hip, where race photos put it: ahead of the near one
# (the photographer stands a little in front). Chicago 2026, images.jpg: from
# the near hip the landing read 0.32 overstride and a 117 deg knee; from its
# own hip 0.18 and 131.
_HIP_AHEAD = 0.53


@pytest.fixture
def far_hip_ahead(monkeypatch):
    from app.services.video_analysis import photo_analyzer as PA

    def pose(b):
        r = _pose_result(b)
        r["normalized"][23].x = _HIP_AHEAD
        r["world"][23].x = (_HIP_AHEAD - 0.5) * 1.2
        return r

    monkeypatch.setattr(PA, "detect_pose_in_image", pose)
    photo = _photo_bytes()
    return photo, PA.analyze_photo(photo, "run", None, False, True)


def _img(pt):
    return (pt[0] * W / H, pt[1])


def test_a_far_stance_leg_is_measured_from_its_own_hip(analyzed, far_hip_ahead):
    _photo, res = far_hip_ahead
    hip, knee, ankle = _img((_HIP_AHEAD, 0.50)), _img(_STANCE[25]), _img(_STANCE[27])
    leg = math.dist(hip, knee) + math.dist(knee, ankle)
    assert res["contact"]["overstride_ratio"] == pytest.approx(abs(ankle[0] - hip[0]) / leg, abs=1e-3)
    assert res["contact"]["overstride_ratio"] < analyzed[1]["contact"]["overstride_ratio"]
    # The trunk stays on the visible hip: the stance hip moving changes nothing.
    assert res["angles"]["trunk"] == analyzed[1]["angles"]["trunk"]
    # The hip angle is trunk against thigh, not shoulder -> hidden hip -> knee.
    sh, vis = _img(_BODY[12]), _img(_BODY[24])
    trunk = (sh[0] - vis[0], sh[1] - vis[1])
    thigh = (knee[0] - hip[0], knee[1] - hip[1])
    want = math.degrees(math.acos((trunk[0] * thigh[0] + trunk[1] * thigh[1])
                                  / (math.hypot(*trunk) * math.hypot(*thigh))))
    assert res["angles"]["hip"] == pytest.approx(want, abs=0.1)


def test_the_hidden_hip_is_the_one_handed_out_to_drag(far_hip_ahead):
    photo, res = far_hip_ahead
    pose = res["pose"]
    # The near-side "hip" the athlete drags is the stance leg's own; the
    # visible one is named for the drawing only.
    assert pose["landmarks"][24][0] == _HIP_AHEAD and pose["pelvis_hip"] == 23
    assert 23 not in pose["draggable"]
    out = PC.recompute_photo(photo, "run", None, pose, [{"landmark": 32, "dx": 0.005, "dy": 0.0}])
    assert out["baseline"]["angles"] == res["angles"]
    assert out["baseline"]["contact"]["overstride_ratio"] == res["contact"]["overstride_ratio"]
