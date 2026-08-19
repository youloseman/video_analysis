"""The skeleton must not jump between the athlete's legs.

On backlit / silhouette clips MediaPipe re-assigns the left/right leg
indices freely -- the drawn skeleton visibly flips between the near and the
far leg, and every near-side leg metric (knee angle, gait phase, cadence,
overstride) silently reads a mix of both legs. Torso-level anti-flip can't
see it: hips and shoulders stay put. ``_fix_leg_swaps`` tracks each leg by
image-space continuity and swaps the indices back.

The synthetic gait here is honest about the hard part: the legs scissor,
so the ankles genuinely cross twice per stride -- the pass has to survive
real crossings without churning, and still catch injected identity swaps.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from app.services.video_analysis.biomechanics.landmark_stabilizer import (
    _fix_leg_swaps,
    stabilize_landmarks,
)

FPS = 60.0
STRIDE_HZ = 1.375          # 165 spm
N_FRAMES = 480             # 8 s


def _leg(phi: float) -> tuple[float, float, float, float]:
    """(ankle_x, ankle_y, knee_x, knee_y) for one leg at stride phase phi."""
    ankle_x = 0.50 + 0.12 * math.sin(phi)
    swing = max(0.0, math.sin(phi - math.pi / 2))
    ankle_y = 0.85 - 0.06 * swing ** 1.5
    knee_x = 0.50 + 0.07 * math.sin(phi + 0.6)
    knee_y = 0.68 - 0.02 * swing
    return ankle_x, ankle_y, knee_x, knee_y


def _lm(x: float = 0.5, y: float = 0.5, vis: float = 0.95) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=0.0, visibility=vis)


def make_gait_frames(n: int = N_FRAMES, noise: float = 0.002, seed: int = 7):
    """Scissor-gait frame dicts with ground truth left/right leg positions."""
    rng = np.random.default_rng(seed)
    frames = []
    truth = []  # (ankle_L, ankle_R) ground-truth positions per frame
    for i in range(n):
        t = i / FPS
        phi = 2 * math.pi * STRIDE_HZ * t
        lax, lay, lkx, lky = _leg(phi)
        rax, ray, rkx, rky = _leg(phi + math.pi)
        jit = lambda: float(rng.normal(0, noise))  # noqa: E731

        lms = [_lm() for _ in range(33)]
        lms[25] = _lm(lkx + jit(), lky + jit())
        lms[26] = _lm(rkx + jit(), rky + jit())
        lms[27] = _lm(lax + jit(), lay + jit())
        lms[28] = _lm(rax + jit(), ray + jit())
        lms[29] = _lm(lax - 0.02 + jit(), lay + 0.03 + jit())
        lms[30] = _lm(rax - 0.02 + jit(), ray + 0.03 + jit())
        lms[31] = _lm(lax + 0.04 + jit(), lay + 0.03 + jit())
        lms[32] = _lm(rax + 0.04 + jit(), ray + 0.03 + jit())

        world = [SimpleNamespace(x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility)
                 for lm in lms]
        frames.append({
            "normalized_landmarks": lms,
            "world_landmarks": world,
            "timestamp_ms": i / FPS * 1000.0,
        })
        truth.append(((lax, lay), (rax, ray)))
    return frames, truth


def swap_legs(frame: dict) -> None:
    for key in ("normalized_landmarks", "world_landmarks"):
        lms = frame[key]
        for li, ri in ((25, 26), (27, 28), (29, 30), (31, 32)):
            lms[li], lms[ri] = lms[ri], lms[li]


def mismatches(frames, truth, tol: float = 0.04) -> int:
    """Frames where landmark 27 is closer to the ground-truth RIGHT ankle."""
    bad = 0
    for frame, (tl, tr) in zip(frames, truth):
        la = frame["normalized_landmarks"][27]
        if math.dist((la.x, la.y), tl) > math.dist((la.x, la.y), tr) + tol:
            bad += 1
    return bad


# ---------------------------------------------------------------------------
# No false positives: a clean scissor gait must pass through untouched
# ---------------------------------------------------------------------------
def test_clean_gait_is_not_churned_by_the_pass():
    frames, truth = make_gait_frames()
    swaps, swap_pct, collapse_pct = _fix_leg_swaps(frames)
    assert swaps == 0, f"false swaps on clean gait: {swaps}"
    assert mismatches(frames, truth) == 0
    # Scissor crossings are brief -- collapse must stay a small fraction.
    assert collapse_pct < 12.0, collapse_pct


# ---------------------------------------------------------------------------
# Injected identity swaps get corrected
# ---------------------------------------------------------------------------
def test_contiguous_swap_streak_is_corrected():
    frames, truth = make_gait_frames()
    for i in range(120, 200):  # 80-frame streak of swapped identity
        swap_legs(frames[i])
    before = mismatches(frames, truth)
    assert before >= 75  # corruption is real

    swaps, swap_pct, _ = _fix_leg_swaps(frames)
    after = mismatches(frames, truth)
    assert after <= before * 0.1, (before, after)
    assert swaps >= 70
    assert swap_pct > 10.0


def test_scattered_single_frame_swaps_are_corrected():
    frames, truth = make_gait_frames()
    rng = np.random.default_rng(3)
    hit = sorted(rng.choice(range(30, 450), size=40, replace=False))
    for i in hit:
        swap_legs(frames[i])
    before = mismatches(frames, truth)
    assert before >= 30

    _fix_leg_swaps(frames)
    after = mismatches(frames, truth)
    assert after <= before * 0.2, (before, after)


def test_world_landmarks_are_swapped_in_lockstep():
    frames, truth = make_gait_frames()
    for i in range(150, 220):
        swap_legs(frames[i])
    _fix_leg_swaps(frames)
    for frame in frames:
        nl, wl = frame["normalized_landmarks"], frame["world_landmarks"]
        for idx in (25, 26, 27, 28, 29, 30, 31, 32):
            assert nl[idx].x == wl[idx].x and nl[idx].y == wl[idx].y


# ---------------------------------------------------------------------------
# Degenerate inputs: collapse and NaN gaps must not crash or churn
# ---------------------------------------------------------------------------
def test_leg_collapse_is_measured_not_corrected():
    frames, truth = make_gait_frames()
    for i in range(100, 340):  # half the clip: both legs on one trajectory
        nl = frames[i]["normalized_landmarks"]
        for li, ri in ((25, 26), (27, 28)):
            nl[ri] = SimpleNamespace(
                x=nl[li].x + 0.005, y=nl[li].y, z=0.0, visibility=0.95,
            )
    _, _, collapse_pct = _fix_leg_swaps(frames)
    assert collapse_pct >= 40.0, collapse_pct


def test_nan_gaps_do_not_poison_the_tail():
    """A visibility gap of ~half a leg cycle once locked in a persistent
    false swap for the whole tail: the tracker matched the post-gap frame
    against a stale pre-gap prediction, and the wrong outcome
    self-perpetuated. After a real gap the pass must re-seed (trust
    MediaPipe's labels) instead of guessing -- the WHOLE clip stays clean.
    """
    for gap_len in (22, 60):  # ~half a leg cycle @60fps, and a 1 s dropout
        frames, truth = make_gait_frames()
        for i in range(200, 200 + gap_len):
            nl = frames[i]["normalized_landmarks"]
            nl[27] = SimpleNamespace(x=math.nan, y=math.nan, z=0.0, visibility=0.0)
        swaps, _, _ = _fix_leg_swaps(frames)  # must not raise
        clean = [i for i in range(len(frames)) if not (200 <= i < 200 + gap_len)]
        bad = mismatches([frames[i] for i in clean], [truth[i] for i in clean])
        assert bad == 0, (gap_len, bad)


def test_hard_detection_gap_does_not_poison_the_tail():
    """Undetected frames never enter frame_results at all -- adjacent list
    entries can sit ~0.4 s apart with no NaN frames in between. The
    timestamp check must trigger the same re-seed.
    """
    frames, truth = make_gait_frames()
    del frames[200:222]
    del truth[200:222]
    swaps, _, _ = _fix_leg_swaps(frames)
    assert mismatches(frames, truth) == 0


# ---------------------------------------------------------------------------
# Wiring: stabilize_landmarks reports the metrics for run, none for bike
# ---------------------------------------------------------------------------
def test_stabilizer_context_carries_leg_metrics_for_run():
    frames, _ = make_gait_frames(n=120)
    ctx: dict = {}
    stabilize_landmarks(frames, "run", None, fps=FPS, context=ctx)
    assert ctx["leg_swap_pct"] is not None
    assert ctx["leg_collapse_pct"] is not None
    assert "flips_corrected" in ctx


def test_stabilizer_skips_leg_pass_for_bike():
    frames, _ = make_gait_frames(n=120)
    ctx: dict = {}
    stabilize_landmarks(frames, "bike", None, fps=FPS, context=ctx)
    assert ctx["leg_swap_pct"] is None
    assert ctx["leg_collapse_pct"] is None
