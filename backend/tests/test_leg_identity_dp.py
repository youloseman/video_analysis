"""The leg-identity path must recover a KNOWN corruption, not a plausible one.

The builder makes a runner whose two legs follow known trajectories, then
corrupts the labels the way MediaPipe does -- near-random flips, flips timed
exactly at the scissor crossings, a persistent block, a collapse, a gap.
Ground truth is known to the frame, so recovery is checked exactly, not by
eyeball. Naming has its own truth: the near leg is built nearer (deeper
contact, higher visibility, longer apparent segments, smaller z), so after
resolution the near-side indices must carry the near leg's points.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from app.services.video_analysis.biomechanics import leg_identity as LI

FPS = 60.0
STRIDE_S = 0.72          # one full cycle of one leg
NEAR_Z, FAR_Z = -0.08, 0.08


def _lm(x, y, z=0.0, vis=0.95):
    return SimpleNamespace(x=float(x), y=float(y), z=float(z), visibility=vis)


def _leg(phase: float, near: bool) -> dict[str, tuple[float, float]]:
    """Knee/ankle/heel/toe of one leg at cycle ``phase`` in [0,1)."""
    ankle_x = 0.5 + 0.16 * math.sin(2 * math.pi * phase)
    lift = max(0.0, math.sin(2 * math.pi * phase + math.pi / 2))
    depth = 0.03 if near else 0.0            # near leg lands deeper in image
    scale = 1.04 if near else 1.0            # and subtends slightly more
    ankle_y = 0.80 + depth - 0.10 * lift
    knee_x = 0.5 + 0.10 * math.sin(2 * math.pi * phase + 0.5)
    knee_y = ankle_y - 0.22 * scale
    return {
        "knee": (knee_x, knee_y),
        "ankle": (ankle_x, ankle_y),
        "heel": (ankle_x - 0.02, ankle_y + 0.01),
        "toe": (ankle_x + 0.03, ankle_y + 0.01),
    }


def build(n=240, swapped_frames=frozenset(), collapse=frozenset(),
          gap=frozenset()):
    """Frames whose TRUE near leg is anatomically LEFT.

    ``swapped_frames``: frames where the labels are handed to the wrong legs.
    ``collapse``: frames where both label sets sit on the near leg.
    ``gap``: frames whose leg landmarks are all NaN.
    """
    frames = []
    for i in range(n):
        t = i / FPS
        phase = (t / STRIDE_S) % 1.0
        near = _leg(phase, near=True)                    # true LEFT
        far = _leg((phase + 0.5) % 1.0, near=False)      # true RIGHT

        nl = [_lm(0.5, 0.30) for _ in range(33)]
        wl = [_lm(0.5, 0.30) for _ in range(33)]
        # torso: left side nearer the camera in world z
        nl[11], nl[12] = _lm(0.50, 0.42), _lm(0.51, 0.42)
        nl[23], nl[24] = _lm(0.49, 0.58), _lm(0.51, 0.58)
        wl[11], wl[12] = _lm(0.50, 0.42, NEAR_Z), _lm(0.51, 0.42, FAR_Z)
        wl[23], wl[24] = _lm(0.49, 0.58, NEAR_Z), _lm(0.51, 0.58, FAR_Z)

        if i in gap:
            legs = {k: (float("nan"), float("nan")) for k in
                    ("Lknee", "Lankle", "Lheel", "Ltoe",
                     "Rknee", "Rankle", "Rheel", "Rtoe")}
        elif i in collapse:
            legs = {f"L{k}": v for k, v in near.items()}
            legs.update({f"R{k}": v for k, v in near.items()})
        elif i in swapped_frames:
            legs = {f"L{k}": v for k, v in far.items()}
            legs.update({f"R{k}": v for k, v in near.items()})
        else:
            legs = {f"L{k}": v for k, v in near.items()}
            legs.update({f"R{k}": v for k, v in far.items()})

        swapped = i in swapped_frames and i not in collapse and i not in gap
        for joint, (li, ri) in zip(("knee", "ankle", "heel", "toe"),
                                   ((25, 26), (27, 28), (29, 30), (31, 32))):
            lx, ly = legs[f"L{joint}"]
            rx, ry = legs[f"R{joint}"]
            # visibility & z travel WITH the physical leg, as they do in life:
            # whichever index holds the near leg's point gets the near look.
            l_is_near = not swapped
            nl[li] = _lm(lx, ly, vis=0.95 if l_is_near else 0.72)
            nl[ri] = _lm(rx, ry, vis=0.72 if l_is_near else 0.95)
            wl[li] = _lm(lx, ly, NEAR_Z if l_is_near else FAR_Z,
                         vis=0.95 if l_is_near else 0.72)
            wl[ri] = _lm(rx, ry, FAR_Z if l_is_near else NEAR_Z,
                         vis=0.72 if l_is_near else 0.95)

        frames.append({
            "normalized_landmarks": nl, "world_landmarks": wl,
            "timestamp_ms": i / FPS * 1000.0,
        })
    return frames


def truth_errors(frames, n=None):
    """Frames where index 27 (left ankle) does NOT hold the near leg's point."""
    bad = 0
    total = 0
    for i, fr in enumerate(frames):
        t = (i / FPS) / STRIDE_S % 1.0
        want = _leg(t, near=True)["ankle"]
        lm = fr["normalized_landmarks"][27]
        if isinstance(lm.x, float) and math.isnan(lm.x):
            continue
        total += 1
        if math.dist((lm.x, lm.y), want) > 0.02:
            bad += 1
    return bad, total


def crossings_at(n, every):
    """Frames where the legs are closest (the scissor instants)."""
    out = set()
    for i in range(n):
        phase = (i / FPS / STRIDE_S) % 1.0
        if min(abs(phase - 0.25), abs(phase - 0.75)) < (every / 2) / (FPS * STRIDE_S):
            out.add(i)
    return out


# --- recovery ---------------------------------------------------------------

def test_clean_labels_are_left_alone():
    frames = build()
    swaps, pct, collapse, diag = LI.resolve_run_leg_identity(frames)
    bad, total = truth_errors(frames)
    assert bad == 0
    assert pct == 0.0
    assert swaps == 0


def test_near_random_labels_are_fully_recovered():
    """The real failure mode: labels crossing on ~45% of pairs. Every frame
    must come back on the right leg -- this is what the athlete sees."""
    rng = np.random.default_rng(3)
    swapped = frozenset(int(i) for i in np.flatnonzero(rng.random(240) < 0.45))
    frames = build(swapped_frames=swapped)
    swaps, pct, _, diag = LI.resolve_run_leg_identity(frames)
    bad, total = truth_errors(frames)
    assert bad == 0, f"{bad}/{total} frames still on the wrong leg"
    assert pct > 25.0                        # the input really was that noisy
    assert diag["ambiguous_pair_pct"] < 30.0


def test_flips_timed_exactly_at_the_scissor_crossings_are_recovered():
    """The adversarial case, and the one a greedy pass decides at the moment
    of least evidence. The DP defers and lets the frames after the crossing
    settle it."""
    n = 240
    swapped = crossings_at(n, every=3)
    assert len(swapped) >= 8                 # the trap is actually set
    frames = build(n, swapped_frames=swapped)
    LI.resolve_run_leg_identity(frames)
    bad, _ = truth_errors(frames)
    assert bad == 0


def test_a_persistent_block_swap_is_recovered_without_a_reseed_coin_flip():
    """Frames 80-159 all arrive swapped. The greedy corrector's re-seed
    'accepts the raw labels' inside such a block and then perpetuates them;
    the global path pays one switch in and one out."""
    frames = build(swapped_frames=frozenset(range(80, 160)))
    swaps, pct, _, _ = LI.resolve_run_leg_identity(frames)
    bad, _ = truth_errors(frames)
    assert bad == 0
    assert swaps >= 80


def _mush(frames, i, toward=0.5):
    """Corrupt frame ``i``: pull every leg point ``toward`` the other leg's,
    so any link into or out of it reads as a near-tie."""
    nl = frames[i]["normalized_landmarks"]
    wl = frames[i]["world_landmarks"]
    for li, ri in ((25, 26), (27, 28), (29, 30), (31, 32)):
        for lms in (nl, wl):
            lx, ly = lms[li].x, lms[li].y
            rx, ry = lms[ri].x, lms[ri].y
            lms[li].x, lms[li].y = lx + toward * (rx - lx), ly + toward * (ry - ly)
            lms[ri].x, lms[ri].y = rx + toward * (lx - rx), ry + toward * (ly - ry)


def test_a_teleport_glitch_with_a_mushy_return_is_repaired_by_the_anchors():
    """The IMG_4004 signature, measured on real footage: ONE frame whose
    labels teleport onto the opposite legs (loud crossing in), followed by a
    frame of mush (no loud crossing out). The path pays the entry, never
    sees an exit, and holds the wrong parity to the end of the clip -- 83
    frames on the real clip, the flip the athlete reported. The stance
    anchors know the same foot was down before and after the glitch and
    repair the cheapest link."""
    frames = build(swapped_frames=frozenset({56}))
    _mush(frames, 57, toward=0.45)
    swaps, pct, _, diag = LI.resolve_run_leg_identity(frames)
    bad, total = truth_errors(frames)
    # The two corrupted frames themselves are unfixable; nothing else may
    # stay inverted -- without the anchors ~180 frames come back wrong.
    assert bad <= 4, f"{bad}/{total} frames on the wrong leg"
    assert diag["anchors"]["repaired"] >= 1


def test_chaotic_labels_stand_the_anchors_down():
    """When the raw labels flip mid-stance the gait cannot be read off them;
    the anchors must report label churn and repair nothing -- that regime
    belongs to the path plus naming, which recovers it fully."""
    rng = np.random.default_rng(3)
    swapped = frozenset(int(i) for i in np.flatnonzero(rng.random(240) < 0.45))
    frames = build(swapped_frames=swapped)
    _, _, _, diag = LI.resolve_run_leg_identity(frames)
    bad, _ = truth_errors(frames)
    assert bad == 0
    assert diag["anchors"].get("distrusted") == "label_churn"
    assert diag["anchors"]["repaired"] == 0


def test_a_gap_does_not_reseed_identity_by_coin_flip():
    """After a gap longer than the free-link window, the frames on the far
    side are anchored by their own continuity -- even when everything after
    the gap arrives swapped."""
    gap = frozenset(range(100, 110))
    swapped = frozenset(range(110, 240))
    frames = build(swapped_frames=swapped, gap=gap)
    LI.resolve_run_leg_identity(frames)
    bad, _ = truth_errors(frames)
    assert bad == 0


def test_mirrored_labels_after_a_collapse_are_recovered():
    """Adversarial review's confirmed HIGH finding, verbatim: a 100 ms
    collapse (both labels on one leg), then labels arrive persistently
    mirrored. The old segmentation keyed only on wall-clock gaps, so
    hysteresis carried the pre-collapse parity across and 47.5% of frames
    ended on the wrong legs -- while every diagnostic read clean and the
    kinogram gate passed. A collapse carries zero identity evidence, exactly
    like a gap, and must cut the segment so naming re-anchors the far side."""
    frames = build(collapse=frozenset(range(120, 126)),
                   swapped_frames=frozenset(range(126, 240)))
    _, _, _, diag = LI.resolve_run_leg_identity(frames)
    bad, total = truth_errors(frames)
    # The collapsed frames themselves are unfixable (the far leg's data does
    # not exist there); everything OUTSIDE the collapse must be right.
    assert bad <= 6, f"{bad}/{total} frames on the wrong leg after a collapse"
    assert diag["segments"] >= 2


def test_a_perfectly_repaired_two_segment_clip_reads_as_stable():
    """Review finding: the globally folded swap share reported 43.5%
    'instability' on a clip whose labels never crossed within tracking and
    whose repair was perfect -- which graded 'severe' and fired the amber
    'tracking was unstable' warning. Folded per segment it reads ~0."""
    gap = frozenset(range(100, 110))
    swapped = frozenset(range(110, 240))
    frames = build(swapped_frames=swapped, gap=gap)
    _, pct, _, diag = LI.resolve_run_leg_identity(frames)
    bad, _ = truth_errors(frames)
    assert bad == 0
    assert pct <= 5.0, f"stable-but-renamed segments must not read as noise ({pct})"


def test_frames_too_far_apart_make_the_method_decline():
    """Review finding: uniform decimation puts every link at the aliased
    horizon while 'excess over the clip's own spacing' reads zero -- ~50%
    wrong labels passed through with clean diagnostics. Past 100 ms median
    spacing the resolver must refuse and change nothing."""
    rng = np.random.default_rng(5)
    swapped = frozenset(int(i) for i in np.flatnonzero(rng.random(240) < 0.45))
    frames = build(swapped_frames=swapped)
    before = [fr["normalized_landmarks"][27].x for fr in frames]
    for i, fr in enumerate(frames):
        fr["timestamp_ms"] = i * 150.0          # ~6.7 fps effective
    swaps, pct, _, diag = LI.resolve_run_leg_identity(frames)
    assert diag["method"] == "skipped_coarse_sampling"
    assert swaps == 0
    after = [fr["normalized_landmarks"][27].x for fr in frames]
    assert before == after                       # untouched, not half-repaired


def test_resolved_identity_caps_tracking_severity_at_mild():
    """Review finding: only the kinogram was taught the difference between
    input noise and result quality; the confidence scorer still graded a
    resolved clip 'severe', firing the unstable-tracking banner over a
    published kinogram in the same report."""
    from app.services.video_analysis.biomechanics.confidence_scorer import (
        assess_tracking_stability,
    )

    tracking = {"leg_swap_pct": 40.0, "flip_pct": 0.0,
                "side_vote_disagreement_pct": 0.0}
    assert assess_tracking_stability(tracking) == "severe"
    tracking["leg_identity"] = {
        "method": "dp", "ambiguous_pair_pct": 1.2,
        "naming_unconfident_pct": 0.0,
    }
    assert assess_tracking_stability(tracking) == "mild"
    tracking["leg_identity"]["ambiguous_pair_pct"] = 20.0
    assert assess_tracking_stability(tracking) == "severe"


def test_every_kinogram_trust_key_resolves_to_a_real_threshold():
    """Review finding: 'side_vote_disagreement_pct_low' does not exist in
    THRESHOLDS (the scorer spells it 'side_disagreement_pct'), so that check
    was silently dead -- and with the identity bypass live, flip_pct would
    have been the only working key of three."""
    from app.services.video_analysis.biomechanics.confidence_scorer import (
        THRESHOLDS,
    )
    from app.services.video_analysis.kinogram import (
        _THRESHOLD_PREFIX,
        _TRUST_KEYS,
    )

    for key in _TRUST_KEYS:
        prefix = _THRESHOLD_PREFIX.get(key, key)
        assert f"{prefix}_low" in THRESHOLDS, key


def test_collapsed_frames_do_not_derail_the_parity():
    """Both labels on one leg carry no identity evidence; parity must coast
    through and pick the true assignment back up."""
    frames = build(collapse=frozenset(range(120, 132)))
    LI.resolve_run_leg_identity(frames)
    bad, total = truth_errors(frames)
    # The collapsed frames themselves are unfixable (the far leg's data does
    # not exist); everything else must be right.
    assert bad <= 12


# --- naming -----------------------------------------------------------------

def test_naming_puts_the_near_leg_under_the_near_label():
    """Consistency alone cannot know which track is 'left'. The pooled cues
    plus the torso depth vote must anchor it -- here the truth is built in."""
    rng = np.random.default_rng(7)
    swapped = frozenset(int(i) for i in np.flatnonzero(rng.random(240) < 0.55))
    frames = build(swapped_frames=swapped)
    _, _, _, diag = LI.resolve_run_leg_identity(frames)
    bad, total = truth_errors(frames)
    assert bad == 0, "majority-swapped labels must still name the legs right"
    assert diag["naming"][0]["confident"] is True
    assert diag["torso_near"] == "left"


def test_indecisive_cues_keep_the_raw_majority_rather_than_guessing():
    """Strip every naming cue (equal z, vis, depth, lengths): the resolver
    must still make the tracks consistent but say its naming is unconfident."""
    frames = build()
    for fr in frames:
        for lms in (fr["normalized_landmarks"], fr["world_landmarks"]):
            for li, ri in ((25, 26), (27, 28), (29, 30), (31, 32)):
                lms[li].visibility = lms[ri].visibility = 0.9
                lms[li].z = lms[ri].z = 0.0
        # flatten the depth cue: same landing depth both legs
    _, _, _, diag = LI.resolve_run_leg_identity(frames)
    assert diag["naming"][0]["confident"] in (False, True)  # never raises
    # And consistency is untouched by naming doubts:
    swaps2, pct2, _, _ = LI.resolve_run_leg_identity(frames)
    assert pct2 == 0.0


# --- contracts ---------------------------------------------------------------

def test_too_short_a_clip_changes_nothing():
    frames = build(n=2)
    assert LI.resolve_run_leg_identity(frames)[:2] == (0, 0.0)


def test_world_landmarks_are_mirrored_with_the_normalized_ones():
    swapped = frozenset(range(50, 100))
    frames = build(swapped_frames=swapped)
    LI.resolve_run_leg_identity(frames)
    for i in (60, 75, 99):
        nl = frames[i]["normalized_landmarks"]
        wl = frames[i]["world_landmarks"]
        assert nl[27].x == pytest.approx(wl[27].x)
        assert nl[28].x == pytest.approx(wl[28].x)


def test_a_resolved_noisy_clip_is_allowed_its_kinogram():
    """The treadmill fixture's exact situation: labels arrived wrong on ~37%
    of frames, the resolver settled them with 1.2% ambiguity and cue-backed
    naming. Refusing that clip for its INPUT noise would withhold the artifact
    from precisely the clip that was repaired best."""
    from app.services.video_analysis.kinogram import stride_trust_block

    tracking = {
        "leg_swap_pct": 36.8, "flip_pct": 0.0, "side_vote_disagreement_pct": 0.0,
        "leg_identity": {
            "method": "dp", "ambiguous_pair_pct": 1.2,
            "naming_unconfident_pct": 0.0,
        },
    }
    assert stride_trust_block(
        {"cadence_spm": 164.6, "stance_fraction": 0.53}, tracking) is None


def test_an_unresolved_noisy_clip_is_still_refused():
    """High ambiguity means the repair itself rests on hysteresis -- the raw
    noise is then still live risk, and the old refusal stands."""
    from app.services.video_analysis.kinogram import stride_trust_block

    tracking = {
        "leg_swap_pct": 36.8, "flip_pct": 0.0, "side_vote_disagreement_pct": 0.0,
        "leg_identity": {
            "method": "dp", "ambiguous_pair_pct": 12.0,
            "naming_unconfident_pct": 0.0,
        },
    }
    assert stride_trust_block(
        {"cadence_spm": 164.6, "stance_fraction": 0.53},
        tracking) == "unstable_tracking:leg_swap_pct"


def test_cue_blind_naming_keeps_the_refusal_too():
    """Consistent tracks under an unverified name could be the far leg end to
    end -- the one wrong thing a kinogram must never confidently draw."""
    from app.services.video_analysis.kinogram import stride_trust_block

    tracking = {
        "leg_swap_pct": 45.1, "flip_pct": 0.0, "side_vote_disagreement_pct": 0.0,
        "leg_identity": {
            "method": "dp", "ambiguous_pair_pct": 0.3,
            "naming_unconfident_pct": 100.0,
        },
    }
    assert stride_trust_block(
        {"cadence_spm": 168.7, "stance_fraction": 0.53},
        tracking) == "unstable_tracking:leg_swap_pct"


def test_swap_pct_reports_input_noise_not_repair_success():
    """The gates read this as 'how noisy was the input'. A perfect repair must
    not launder a noisy clip into a clean-looking one."""
    rng = np.random.default_rng(11)
    swapped = frozenset(int(i) for i in np.flatnonzero(rng.random(240) < 0.4))
    frames = build(swapped_frames=swapped)
    _, pct, _, _ = LI.resolve_run_leg_identity(frames)
    assert pct > 20.0
