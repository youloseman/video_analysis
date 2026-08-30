"""The measuring half of the pipeline, run on frames alone.

Splitting ``run_analysis`` at the stabilizer gives a function that turns
landmark frames into a report with no detector in the loop. Three things are
worth pinning about it:

* it measures what the frames contain -- a synthetic rider built to a known
  geometry comes back with that geometry's knee angles;
* frames written to the store and read back produce the SAME report, to the
  digit, which is the whole premise of re-analysing them later;
* an athlete's correction applied to the frames changes the report in the
  direction the geometry says it must.

The clip itself is deliberately absent (a path that does not exist): the
pictures are lost, the numbers are not, and the result says which.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

from app.services.video_analysis.biomechanics.corrections import apply_corrections
from app.services.video_analysis.landmark_store import load_frames, save_frames
from app.services.video_analysis.runner import analyze_from_frames

# A portrait phone clip. Geometry below is authored in aspect-corrected units
# (u = x * aspect, v = y) and converted back, so lengths mean what they say.
W, H = 1080, 1920
ASPECT = W / H
FPS = 30.0
FRAMES_PER_REV = 40
N_FRAMES = 150

HIP = (0.30, 0.375)
BB = (0.28, 0.72)
CRANK = 0.055
THIGH, SHANK = 0.22, 0.21

# What the construction implies, by the law of cosines.
def _knee_angle_for(ankle_v: float) -> float:
    d = math.hypot(HIP[0] - BB[0], HIP[1] - ankle_v)
    cos_t = (THIGH**2 + SHANK**2 - d**2) / (2 * THIGH * SHANK)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_t))))


EXPECTED_BDC = _knee_angle_for(BB[1] + CRANK)   # ~137 deg, leg most extended
EXPECTED_TDC = _knee_angle_for(BB[1] - CRANK)   # ~85 deg, leg most flexed


def _knee(hip, ankle):
    """Two-link IK: the knee at THIGH from the hip, SHANK from the ankle, forward."""
    d = math.hypot(ankle[0] - hip[0], ankle[1] - hip[1])
    d = min(d, THIGH + SHANK - 1e-6)
    e = ((ankle[0] - hip[0]) / d, (ankle[1] - hip[1]) / d)
    cos_a = (THIGH**2 + d**2 - SHANK**2) / (2 * THIGH * d)
    a = math.acos(max(-1.0, min(1.0, cos_a)))
    n = (e[1], -e[0])
    if n[0] < 0:
        n = (-n[0], -n[1])
    return (
        hip[0] + THIGH * (e[0] * math.cos(a) + n[0] * math.sin(a)),
        hip[1] + THIGH * (e[1] * math.cos(a) + n[1] * math.sin(a)),
    )


def _lm(u, v, *, z, vis=1.0):
    return SimpleNamespace(x=u / ASPECT, y=v, z=z, visibility=vis)


def _rider_frame(i: int) -> dict:
    phi = 2 * math.pi * i / FRAMES_PER_REV
    pts: dict[int, tuple[float, float]] = {}
    for side, offset, phase in (("right", 0, 0.0), ("left", -1, math.pi)):
        ankle = (BB[0] + CRANK * math.sin(phi + phase), BB[1] - CRANK * math.cos(phi + phase))
        hip = (HIP[0] + 0.005 * offset, HIP[1])
        knee = _knee(hip, ankle)
        base = 24 if side == "right" else 23
        pts[base] = hip
        pts[base + 2] = knee
        pts[base + 4] = ankle
        pts[base + 6] = (ankle[0] - 0.02, ankle[1] + 0.015)   # heel
        pts[base + 8] = (ankle[0] + 0.05, ankle[1] + 0.02)    # toe
        sh = 12 if side == "right" else 11
        pts[sh] = (0.50 + 0.005 * offset, 0.30)
        pts[sh + 2] = (0.56, 0.36)
        pts[sh + 4] = (0.64, 0.36)
        pts[sh - 4] = (0.58, 0.27)                             # ear 8 / 7
    pts[0] = (0.63, 0.28)                                      # nose
    for j in (1, 2, 3, 4, 5, 6, 9, 10):
        pts[j] = (0.60, 0.27)
    for j in (17, 19, 21):
        pts[j] = (0.66, 0.36)
    for j in (18, 20, 22):
        pts[j] = (0.66, 0.36)

    norm, world = [], []
    for j in range(33):
        u, v = pts[j]
        right = j in (8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32)
        left = j in (7, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31)
        z = -0.1 if right else (0.1 if left else 0.0)
        vis = 1.0 if not left else 0.6
        norm.append(_lm(u, v, z=z, vis=vis))
        world.append(SimpleNamespace(
            x=(u - 0.3) * 2, y=(v - 0.5) * 2, z=z, visibility=vis,
        ))
    return {
        "world_landmarks": world,
        "normalized_landmarks": norm,
        "timestamp_ms": i / FPS * 1000.0,
        "frame_idx": i,
        "frame_width": W,
        "frame_height": H,
    }


def _frames() -> list[dict]:
    return [_rider_frame(i) for i in range(N_FRAMES)]


VIDEO_INFO = {"fps": FPS, "frame_count": float(N_FRAMES), "duration": N_FRAMES / FPS}


def _analyze(frames, **kw):
    return analyze_from_frames(
        frames, "/nowhere/this-clip-has-expired.mp4", "bike", "triathlon",
        video_info=VIDEO_INFO, recommendations=False, kinogram=False, **kw,
    )


def test_the_report_measures_the_geometry_the_frames_contain():
    res = _analyze(_frames())
    m = res["sport_specific_metrics"]

    assert res["status"] == "completed"
    assert res["frames_analyzed"] == N_FRAMES
    assert res["camera_side"] == "right"
    assert abs(m["knee_at_bdc"] - EXPECTED_BDC) < 2.5
    assert abs(m["knee_at_tdc"] - EXPECTED_TDC) < 2.5
    assert isinstance(res["technique_score"], int) and 0 <= res["technique_score"] <= 100


def test_losing_the_clip_costs_the_pictures_and_not_the_numbers():
    res = _analyze(_frames())
    assert res["keyframe_base64"] is None
    assert res["sport_specific_metrics"].get("keyframe_failed") is True
    assert res["overlay_video_path"] is None
    assert res["sport_specific_metrics"]["knee_at_bdc"] is not None


def test_stored_frames_give_the_same_report_to_the_digit(tmp_path):
    frames = _frames()
    live = _analyze(frames)

    save_frames(tmp_path / "landmarks.npz", _frames(), meta={"sport_type": "bike"})
    back, _ = load_frames(tmp_path / "landmarks.npz")
    stored = _analyze(back)

    for key in ("knee_at_bdc", "knee_at_tdc", "trunk_angle_avg", "hip_angle_avg",
                "elbow_angle_avg", "shoulder_angle_avg", "pelvic_ratio"):
        assert stored["sport_specific_metrics"][key] == live["sport_specific_metrics"][key], key
    assert stored["technique_score"] == live["technique_score"]
    assert stored["score_breakdown"] == live["score_breakdown"]


def test_a_correction_moves_the_report_the_way_the_geometry_says():
    """Lowering the hip point brings it closer to the ankle at the bottom of
    the stroke, so the knee reads more bent there. The sign is the check: a
    plumbing bug that dropped the correction would leave the number alone."""
    baseline = _analyze(_frames())["sport_specific_metrics"]["knee_at_bdc"]

    frames = _frames()
    apply_corrections(frames, [{"landmark": 24, "dx": 0.0, "dy": 0.05}], "bike")
    corrected = _analyze(frames, corrections=[{"landmark": 24, "dx": 0.0, "dy": 0.05}])

    assert corrected["sport_specific_metrics"]["knee_at_bdc"] < baseline - 3


def test_the_report_says_which_points_the_athlete_moved():
    corrections = [{"landmark": 24, "dx": 0.0, "dy": 0.05, "frame_idx": 12}]
    frames = _frames()
    apply_corrections(frames, corrections, "bike")
    res = _analyze(frames, corrections=corrections)
    assert res["corrections"] == corrections


def test_an_automatic_report_carries_no_corrections():
    assert _analyze(_frames())["corrections"] is None
