"""A camera-motion estimator is only worth having if it recovers a KNOWN motion.

There is no clip in the repo filmed with a moving camera, so validating this
against real footage is not possible -- and guessing whether it works would be
worse than not shipping it. Instead the tests take a still scene and move it by
an exact amount, which is stricter than a handheld clip would be: the answer is
known to the pixel.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="needs the analysis stack (opencv)")

from app.services.video_analysis import camera_motion as cm  # noqa: E402

W, H = 640, 480
FRAMES = 24


def _scene(rng):
    """A textured background ORB can actually find corners in."""
    img = rng.integers(0, 60, size=(H * 2, W * 2, 3), dtype=np.uint8)
    for _ in range(400):
        x = int(rng.integers(10, W * 2 - 10))
        y = int(rng.integers(10, H * 2 - 10))
        s = int(rng.integers(4, 14))
        colour = tuple(int(c) for c in rng.integers(120, 255, size=3))
        cv2.rectangle(img, (x, y), (x + s, y + s), colour, -1)
    return img


def _landmarks(cx=0.80, cy=0.5):
    """A subject parked in the right-hand fifth, away from the crop window."""
    lm = []
    for i in range(33):
        lm.append(SimpleNamespace(
            x=cx + (i % 3) * 0.01, y=cy + (i % 5) * 0.02,
            z=0.0, visibility=0.9,
        ))
    return lm


def write_clip(path, offsets):
    """Pan a big textured scene by a known (dx, dy) per frame."""
    rng = np.random.default_rng(7)
    scene = _scene(rng)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H),
    )
    assert writer.isOpened()
    frame_data = []
    for i, (ox, oy) in enumerate(offsets):
        x0 = int(round(W // 2 + ox))
        y0 = int(round(H // 2 + oy))
        writer.write(scene[y0:y0 + H, x0:x0 + W].copy())
        frame_data.append({"frame_idx": i, "normalized_landmarks": _landmarks()})
    writer.release()
    return str(path), frame_data


def steady(n=FRAMES):
    return [(0.0, 0.0)] * n


def drifting(n=FRAMES, per_frame=2.0):
    return [(i * per_frame, 0.0) for i in range(n)]


def bouncing(n=FRAMES, amp=6.0, period=6):
    return [(0.0, amp * np.sin(2 * np.pi * i / period)) for i in range(n)]


# --- it recovers a known motion --------------------------------------------

def test_a_locked_off_camera_reads_as_barely_moving(tmp_path):
    path, fd = write_clip(tmp_path / "still.mp4", steady())
    out = cm.estimate_camera_motion(path, fd, hip_amplitude_norm=0.05)
    assert out is not None
    assert out["vertical_bounce_px"] < 2.0
    assert out["verdict"] == "good"


def test_a_pan_is_measured_in_the_right_axis_and_size(tmp_path):
    """24 frames at 2 px each is 46 px of pan, and none of it vertical."""
    path, fd = write_clip(tmp_path / "pan.mp4", drifting(per_frame=2.0))
    out = cm.estimate_camera_motion(path, fd, hip_amplitude_norm=0.05)
    assert out["pan_px"] == pytest.approx(46.0, abs=6.0)
    assert out["tilt_px"] < 5.0


def test_a_bouncing_camera_is_measured_close_to_its_real_amplitude(tmp_path):
    """A +/-6 px bob is 12 px peak to peak; the p05-p95 span reads a little
    under that, which is the price of not being set by one frame."""
    path, fd = write_clip(tmp_path / "bounce.mp4", bouncing(amp=6.0))
    out = cm.estimate_camera_motion(path, fd, hip_amplitude_norm=0.05)
    assert out["vertical_bounce_px"] == pytest.approx(12.0, rel=0.35)


def test_a_steady_drift_is_not_counted_as_bounce(tmp_path):
    """A slow tilt is a different fault from a shake, and only the shake
    lands on top of the athlete's own rise and fall."""
    path, fd = write_clip(
        tmp_path / "tilt.mp4", [(0.0, i * 2.0) for i in range(FRAMES)],
    )
    out = cm.estimate_camera_motion(path, fd, hip_amplitude_norm=0.05)
    assert out["tilt_px"] > 30.0
    assert out["vertical_bounce_px"] < 5.0


# --- the number that is actually reported ----------------------------------

def test_the_headline_is_the_share_of_the_hip_motion_not_a_pixel_count(tmp_path):
    """24 px of hip travel with a 12 px camera bob means half the measured
    oscillation could be the operator."""
    path, fd = write_clip(tmp_path / "share.mp4", bouncing(amp=6.0))
    out = cm.estimate_camera_motion(path, fd, hip_amplitude_norm=24.0 / H)
    assert out["vertical_share_of_hip_motion"] == pytest.approx(0.5, rel=0.4)
    assert out["verdict"] == "bad"


def test_the_same_camera_bob_matters_less_to_a_bigger_athlete(tmp_path):
    """The reason a bare pixel count would be useless."""
    path, fd = write_clip(tmp_path / "big.mp4", bouncing(amp=6.0))
    small = cm.estimate_camera_motion(path, fd, hip_amplitude_norm=20.0 / H)
    large = cm.estimate_camera_motion(path, fd, hip_amplitude_norm=200.0 / H)
    assert small["vertical_share_of_hip_motion"] > large["vertical_share_of_hip_motion"]
    assert large["verdict"] == "good"


def test_without_a_hip_amplitude_it_declines_to_judge(tmp_path):
    path, fd = write_clip(tmp_path / "nohip.mp4", bouncing())
    out = cm.estimate_camera_motion(path, fd, hip_amplitude_norm=None)
    assert out["vertical_share_of_hip_motion"] is None
    assert out["verdict"] == "unknown"


@pytest.mark.parametrize("share,expected", [
    (0.0, "good"), (cm.SHARE_WARN - 0.01, "good"), (cm.SHARE_WARN, "warn"),
    (cm.SHARE_BAD - 0.01, "warn"), (cm.SHARE_BAD, "bad"), (2.0, "bad"),
    (None, "unknown"),
])
def test_verdict_bands(share, expected):
    assert cm._verdict(share) == expected


# --- it must not measure the athlete ---------------------------------------

def test_the_subject_is_masked_out_of_the_feature_search():
    mask = cm._subject_mask(cv2, (H, W), _landmarks(cx=0.5, cy=0.5))
    assert mask is not None
    assert mask[int(H * 0.52), int(W * 0.52)] == 0     # on the subject
    assert mask[5, 5] == 255                           # background corner


def test_a_subject_filling_the_frame_leaves_nothing_to_measure_against():
    """Correctly refuses rather than estimating camera motion from a mask with
    no background left in it."""
    lm = [SimpleNamespace(x=0.02 + 0.96 * (i % 2), y=0.02 + 0.96 * ((i // 2) % 2),
                          z=0.0, visibility=0.9) for i in range(33)]
    assert cm._subject_mask(cv2, (H, W), lm) is None


def test_too_few_landmarks_means_no_mask_rather_than_a_wrong_one():
    assert cm._subject_mask(cv2, (H, W), []) is None
    assert cm._subject_mask(cv2, (H, W), None) is None


# --- degrading -------------------------------------------------------------

def test_an_unreadable_video_returns_nothing_rather_than_raising(tmp_path):
    _, fd = write_clip(tmp_path / "ok.mp4", steady())
    assert cm.estimate_camera_motion(str(tmp_path / "missing.mp4"), fd) is None


def test_a_clip_too_short_to_compare_returns_nothing(tmp_path):
    path, fd = write_clip(tmp_path / "short.mp4", steady(n=3))
    assert cm.estimate_camera_motion(path, fd[:2]) is None


def test_a_featureless_scene_says_it_could_not_read_rather_than_zero(tmp_path):
    """A blank wall gives ORB nothing. 'No motion detected' and 'could not
    detect motion' must not look the same."""
    writer = cv2.VideoWriter(
        str(tmp_path / "blank.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H),
    )
    for _ in range(FRAMES):
        writer.write(np.full((H, W, 3), 128, dtype=np.uint8))
    writer.release()
    fd = [{"frame_idx": i, "normalized_landmarks": _landmarks()} for i in range(FRAMES)]
    assert cm.estimate_camera_motion(str(tmp_path / "blank.mp4"), fd) is None


def test_the_result_is_json_serializable(tmp_path):
    import json

    path, fd = write_clip(tmp_path / "j.mp4", bouncing())
    json.dumps(cm.estimate_camera_motion(path, fd, hip_amplitude_norm=0.05))
