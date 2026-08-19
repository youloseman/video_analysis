"""Measuring the two things that decide whether a side-view clip is readable.

An athlete filmed from the far side of a track lands ~140 px tall in the frame
MediaPipe is handed, and at that size the legs are a few pixels apart for much
of the stride -- the tracker's left/right assignment becomes a coin flip. That
is a capture problem with a capture fix ("stand closer"), so it has to be
measured and named rather than blamed on lighting.

The near-side cues are the groundwork for deciding which leg faces the camera
from the image instead of from MediaPipe's depth guess. They decide nothing
yet; these tests pin their sign convention (positive = LEFT is nearer) and
their refusal to answer when the evidence is thin, so that when they are
promoted to a decision the promotion is the only thing that changes.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from app.services.video_analysis.biomechanics.tracking_diagnostics import (
    compute_framing,
    compute_near_side_cues,
)


def _lm(x=0.5, y=0.5, z=0.0, vis=0.95):
    return SimpleNamespace(x=x, y=y, z=z, visibility=vis)


def make_frames(
    n=60,
    *,
    near="left",
    subject_top=0.30,
    subject_bottom=0.90,
    vis_gap=0.25,
    depth_gap=0.06,
    length_gap=0.04,
    z_gap=0.05,
):
    """Frames where the ``near`` leg carries every near-side signature.

    The far leg is tracked less confidently, lands higher in the image
    (perspective), subtends fewer pixels, and sits further in z.
    """
    hip_y = (subject_top + subject_bottom) / 2
    frames = []
    for i in range(n):
        phase = 2 * math.pi * i / 20.0
        lms = [_lm() for _ in range(33)]
        lms[0] = _lm(0.5, subject_top)                    # nose
        lms[11] = _lm(0.5, subject_top + 0.08)
        lms[12] = _lm(0.5, subject_top + 0.08)
        lms[23] = _lm(0.49, hip_y)
        lms[24] = _lm(0.51, hip_y)

        swing = 0.05 * math.sin(phase)
        base = subject_bottom
        for side in ("left", "right"):
            is_near = side == near
            knee_i, ankle_i, heel_i, toe_i, hip_i = (
                (25, 27, 29, 31, 23) if side == "left" else (26, 28, 30, 32, 24)
            )
            vis = 0.98 if is_near else 0.98 - vis_gap
            # near leg lands lower in the image and is drawn slightly longer
            foot_y = base + (depth_gap if is_near else 0.0) - abs(swing)
            scale = 1.0 + (length_gap if is_near else 0.0)
            z = -z_gap if is_near else z_gap
            lms[knee_i] = _lm(0.5 + swing, hip_y + (foot_y - hip_y) * 0.5 * scale,
                              z=z, vis=vis)
            lms[ankle_i] = _lm(0.5 + swing, foot_y, z=z, vis=vis)
            lms[heel_i] = _lm(0.48 + swing, foot_y + 0.01, z=z, vis=vis)
            lms[toe_i] = _lm(0.53 + swing, foot_y + 0.01, z=z, vis=vis)

        world = [
            SimpleNamespace(x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility)
            for lm in lms
        ]
        frames.append({"normalized_landmarks": lms, "world_landmarks": world})
    return frames


# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------
def test_a_distant_athlete_is_reported_in_pixels_not_just_percent():
    """Reproduces the real case: a track clip whose runner measured 141 px of
    a 720 px frame (~20%). Well lit, sharp, and still unreadable -- the
    skeleton jumped between legs because at that size there is barely a leg's
    width between them."""
    frames = make_frames(subject_top=0.40, subject_bottom=0.56)
    out = compute_framing(frames, detect_height_px=720)
    assert out["subject_height_px"] == pytest.approx(141, abs=12)
    assert out["verdict"] == "tiny"


def test_a_well_framed_athlete_reads_ok():
    frames = make_frames(subject_top=0.08, subject_bottom=0.92)
    out = compute_framing(frames, detect_height_px=1080)
    assert out["verdict"] == "ok"
    assert out["subject_height_frac"] > 0.7


def test_framing_without_a_detector_size_still_reports_the_fraction():
    out = compute_framing(make_frames(), detect_height_px=None)
    assert out["subject_height_frac"] is not None
    assert out["subject_height_px"] is None
    assert out["verdict"] is None


def test_framing_on_an_empty_clip_says_nothing_rather_than_crashing():
    assert compute_framing([], detect_height_px=720)["verdict"] is None


# ---------------------------------------------------------------------------
# Near-side cues
# ---------------------------------------------------------------------------
def test_every_cue_points_at_the_near_leg_and_agrees():
    for near in ("left", "right"):
        out = compute_near_side_cues(make_frames(near=near))
        assert out["suggested_near_side"] == near, (near, out)
        assert out["conflict"] is False
        assert out["agreement"] >= 3, out


def test_the_sign_convention_is_positive_for_left():
    out = compute_near_side_cues(make_frames(near="left"))
    assert out["visibility_pp"] > 0
    assert out["contact_depth_pct"] > 0
    assert out["segment_length_pct"] > 0
    assert out["z_bias_pct"] > 0


def test_cues_abstain_when_the_two_legs_look_identical():
    """A symmetric clip carries no evidence either way. Saying "left" from a
    rounding error is worse than saying nothing."""
    out = compute_near_side_cues(
        make_frames(vis_gap=0.0, depth_gap=0.0, length_gap=0.0, z_gap=0.0)
    )
    assert out["suggested_near_side"] is None
    assert out["cues_voting"] == 0


def test_a_single_disagreeing_cue_does_not_flip_the_verdict():
    """MediaPipe's depth is one voter, not the judge: with the image cues
    pointing left, a z that says right loses and the conflict is recorded."""
    frames = make_frames(near="left", z_gap=-0.05)   # z now favours right
    out = compute_near_side_cues(frames)
    assert out["suggested_near_side"] == "left"
    assert out["conflict"] is True
    assert out["cue_votes"]["z_bias_pct"] == "right"


def test_missing_landmarks_do_not_crash_the_cues():
    frames = make_frames()
    for frame in frames[:20]:
        for idx in (27, 28, 31, 32):
            frame["normalized_landmarks"][idx] = SimpleNamespace(
                x=math.nan, y=math.nan, z=math.nan, visibility=0.0,
            )
    out = compute_near_side_cues(frames)      # must not raise
    assert "suggested_near_side" in out


def test_too_few_frames_produce_no_opinion():
    assert compute_near_side_cues(make_frames(n=3))["suggested_near_side"] is None


def test_broken_tracking_reports_nothing_rather_than_a_negative_height():
    """Seen for real: with most leg landmarks gated out the nose/ankle span
    collapsed and framing reported an athlete "-10 px tall". A measurement
    that cannot be trusted has to come back as None."""
    frames = make_frames()
    for frame in frames:
        lms = frame["normalized_landmarks"]
        for idx in (0, 11, 12):
            lms[idx] = SimpleNamespace(x=0.5, y=0.95, z=0.0, visibility=0.9)
        for idx in (27, 28):
            lms[idx] = SimpleNamespace(x=0.5, y=0.40, z=0.0, visibility=0.9)
    out = compute_framing(frames, detect_height_px=720)
    assert out["subject_height_frac"] is None
    assert out["subject_height_px"] is None
    assert out["verdict"] is None
