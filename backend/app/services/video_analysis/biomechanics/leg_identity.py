"""Whole-clip left/right leg identity for running.

MediaPipe assigns leg identity per frame from appearance, and on a side view
appearance barely distinguishes the legs: on the best-framed treadmill clip in
the fixture set its labels cross between frames on ~44% of pairs. Every
consumer inherits that -- the overlay visibly jumps between legs, and any
metric read from one named leg is spliced from both.

The greedy corrector this replaces (``_fix_leg_swaps``) walks the clip once,
deciding each pair against a velocity prediction. Three structural faults,
none fixable by tuning:

* **A wrong decision self-perpetuates.** Once the tracks latch onto the wrong
  legs, every following frame matches the wrong prediction best, until a gap
  forces a re-seed.
* **A re-seed is a coin flip.** It "accepts MediaPipe's labels" for the first
  post-gap frame -- on a clip whose labels are near-random, that is chance,
  and whichever way the coin lands then persists (see fault one).
* **Crossings are decided at the crossing.** The one instant where the
  evidence is weakest is the one instant a sequential pass must decide.

This module decides from the whole clip instead. Between two measured frames
the labels either stayed on their legs or crossed, and each link is decided by
which matching is cheaper, with hysteresis holding parity through near-ties.
(Stated honestly: with symmetric transition costs this "shortest path"
decomposes into independent per-link decisions -- review proved the
retroactive-repair story wrong -- so the real safety comes from the two rules
around the path: any link WITHOUT decisive evidence cuts the clip into
segments, and every segment is re-anchored by its own pooled naming cues.
A link is evidence-free when the frames are far apart in time OR when the
legs were collapsed onto one another between them; both now cut.) There are
no re-seeds and no coin flips: what cannot be decided from evidence is
handed to per-segment naming, and what naming cannot decide is REPORTED as
unconfident rather than guessed silently.

Identity has two halves, and the path only solves the first:

1. **Consistency** -- the same physical leg keeps the same label throughout.
   Decided by the DP from continuity alone.
2. **Naming** -- which physical leg is called "left". Continuity cannot know
   this (both namings cost the same), so it is decided per SEGMENT of
   continuous tracking: the leg the segment's pooled image cues place nearer
   the camera gets the label the torso's depth vote says is near. Pooling
   uses every frame of the segment, so no single mistracked stretch decides
   -- and on the one clip whose near side a human has confirmed (IMG_3979,
   right), this naming agrees.

Running only. The bike path keeps its blank-don't-correct gate
(``_gate_leg_identity_breaks``): there the far leg's landmarks are partly
invented (median far-knee visibility 0.47 on a clean trainer clip), and a
correction that must trust both legs has been measured making bike clips
worse. See the discussion on ``_fix_leg_swaps`` for those numbers.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# Landmark index pairs exchanged by one identity swap. Mirrors the greedy
# corrector's list (kept local -- this module is imported by the stabilizer,
# so importing back from it would be circular).
_SWAP_PAIRS = ((25, 26), (27, 28), (29, 30), (31, 32))

# Joints the path cost is measured on. Knees + ankles: the same set the greedy
# corrector trusted. Heels and toes follow the swap but are the noisiest
# points MediaPipe produces and would mostly add variance to the cost.
_COST_JOINTS = ((25, 26), (27, 28))

# Ankles closer than this are one blob, not two legs (same constant class as
# the stabilizer's LEG_COLLAPSE_DIST).
_COLLAPSE_DIST = 0.02

# Switch hysteresis, as a fraction of the clip's own median matched step. Big
# enough that a near-tie (legs coincident) holds the current parity; small
# enough that any real crossing -- whose evidence is on the order of the leg
# separation, tens of times a normal step -- pays it without hesitation.
# Known cost of the toll (measured, IMG_4004): a one-frame glitch whose ENTRY
# is loud but whose RETURN is merely decisive gets its return suppressed and
# the parity sticks wrong -- the stance anchors below exist to repair exactly
# that, from gait structure rather than from a cheaper toll. (A tie-band rule
# -- waive the toll on decisive links -- was tried first and repaired this
# clip while breaking the validated blurry fixture, whose noise clears the
# bar in isolated links more often than the toll lets through.)
_SWITCH_COST_STEPS = 3.0

# A pair whose stay/cross costs differ by less than this fraction of the step
# scale is counted ambiguous. The share of such pairs is the honest residual:
# the fraction of the clip where even a global decision rests on hysteresis
# rather than evidence.
_AMBIGUOUS_FRAC = 1.0

# The hysteresis exists because labels rarely flip WITHIN continuous tracking.
# Across a short gap that prior weakens -- MediaPipe re-detects from scratch
# on the far side -- so the switch cost decays with the gap's length.
_GAP_PRIOR_HALFLIFE_MS = 150.0

# And the position evidence itself decays FASTER than intuition says, because
# past a short horizon it turns actively misleading rather than merely weak.
# The legs run in antiphase, so once a gap approaches a quarter of the stride
# cycle, each label's old position is often CLOSER to the other leg's new one
# -- measured on a known-truth fixture, a 183 ms gap scored staying at 0.45
# against crossing at 0.72 when crossing was the truth. Aliasing, not noise.
_GAP_EVIDENCE_HALFLIFE_MS = 120.0

# Past this much silence the link is cut outright: the two sides become
# separate segments, each named on its own. A quarter cycle at a slow 150 spm
# stride is 200 ms; beyond that continuity is a trap, not a tiebreak. One
# global naming bit was tried first and failed the same fixture by flipping
# the correct half of the clip to repair the wrong half.
_SEGMENT_GAP_MS = 150.0

# Naming is only trusted when the pooled cue score clears the same bar the
# near-side veto uses; below it the raw-majority naming is kept and reported
# as unconfident.
_NAMING_MIN_SCORE = 1.0

# Frame spacing beyond which the whole method declines to run. Every link on a
# uniformly decimated clip sits at the aliased horizon the gap logic cuts, so
# "excess over the clip's own median spacing" reads zero exactly where the
# evidence is least trustworthy -- adversarial review demonstrated ~50% wrong
# labels passing through with every diagnostic reading clean. Below 10
# effective fps this returns the clip to the greedy path, unrepaired but also
# unlaundered.
_MAX_MEDIAN_SPACING_MS = 100.0

# Consecutive collapsed frames (both labels on one leg) that cut the identity
# chain. A collapse carries zero left/right evidence -- exactly like a time
# gap -- and MediaPipe's re-detection after one is the documented coin flip,
# so the far side must be re-anchored by its own segment's naming. Keying the
# cuts on wall-clock gaps alone left this hole: a 100 ms collapse followed by
# mirrored labels kept 47% of frames on the wrong legs while swap_pct read
# 0.0 (adversarial review, reproduced exactly).
_COLLAPSE_CUT_FRAMES = 2

# --- stance anchors -------------------------------------------------------
# Running guarantees a structure the link costs cannot see: during one stance
# the same physical foot stays down, and consecutive stances use opposite
# feet. That pins the parity DIFFERENCE between any two neighbouring stance
# midpoints: down-labels that alternate demand no net parity change between
# them; a repeated down-label demands exactly one. The DP satisfies these for
# free when its links are honest, but a detection glitch -- one frame whose
# labels teleport, loud on entry, mushy on the way back -- leaves it holding
# a wrong parity that no later link ever contradicts (IMG_4004: one such
# frame inverted the skeleton for 83 frames; a second inverted the tail).
# The anchors read the truth straight off the gait and repair the cheapest
# link in the violating stretch.

# A down-run must hold this many consecutive measured frames to count as a
# stance fragment (shorter runs are scissor noise).
_ANCHOR_MIN_RUN = 5

# One ankle is "down" when it sits lower than the other by this fraction of
# the clip's median ankle separation (floored for tiny subjects).
_ANCHOR_DOWN_MARGIN_FRAC = 0.04
_ANCHOR_DOWN_MARGIN_MIN = 0.008

# Fragments whose starts are closer than MERGE x the median stance spacing
# are the same physical stance, split by a glitch. Between MERGE and CHAIN
# they are consecutive stances and constrain each other; farther apart a
# stance went missing (collapse, occlusion) and no constraint is drawn.
_ANCHOR_MERGE_FRAC = 0.6
_ANCHOR_CHAIN_FRAC = 1.6

# When repairing a violated stretch, flipping at an existing crossing (undo)
# is preferred over inventing a crossing at a stay link: a held near-tie at a
# scissor scores almost the same margin as a barely-won crossing, and the
# smaller claim should win. Stay links pay this many steps extra.
_ANCHOR_STAY_PENALTY_STEPS = 0.5

# If more than this share of constraints is violated the anchors themselves
# are suspect (walking? mis-detected stances?) -- report, repair nothing.
_ANCHOR_MAX_VIOLATION_FRAC = 0.3

# Direct down-label flips (adjacent measured frames, both labels defined,
# different) per stance fragment. A physical handover passes through the
# undecided margin zone, so clean labels produce well under one direct flip
# per stance (0.31 on the clean 58 fps fixture); raw label chaos flips the
# down label mid-stance (1.28 on the blurry fixture whose shipped output is
# validated). Above this bar the anchors read gait off junk labels -- and the
# junk-label regime is exactly the one the DP-plus-naming path already
# handles -- so they stand down.
_ANCHOR_MAX_CHURN_PER_STANCE = 0.8


def _point(lms: Any, idx: int) -> tuple[float, float] | None:
    try:
        lm = lms[idx]
    except (IndexError, TypeError):
        return None
    x, y = getattr(lm, "x", None), getattr(lm, "y", None)
    if x is None or y is None:
        return None
    if (isinstance(x, float) and math.isnan(x)) or (
        isinstance(y, float) and math.isnan(y)
    ):
        return None
    return (float(x), float(y))


def _frame_joints(frame: dict[str, Any]) -> list[tuple[Any, Any]] | None:
    """((left_pt, right_pt) per cost joint) or None when nothing usable."""
    lms = frame.get("normalized_landmarks")
    if lms is None:
        return None
    out = []
    for li, ri in _COST_JOINTS:
        out.append((_point(lms, li), _point(lms, ri)))
    if all(lp is None or rp is None for lp, rp in out):
        return None
    return out


def _pair_costs(
    prev: list, cur: list,
) -> tuple[float, float, int]:
    """(stay, cross, joints_used) between two measured frames.

    ``stay``: left matched to left, right to right. ``cross``: labels
    exchanged between the frames. Only joints measured on BOTH frames and on
    BOTH legs enter, so a one-leg frame contributes nothing rather than a
    biased half-measurement.
    """
    stay = cross = 0.0
    used = 0
    for (pl, pr), (cl, cr) in zip(prev, cur):
        if None in (pl, pr, cl, cr):
            continue
        stay += math.dist(cl, pl) + math.dist(cr, pr)
        cross += math.dist(cl, pr) + math.dist(cr, pl)
        used += 1
    return stay, cross, used


def resolve_run_leg_identity(
    frame_results: list[dict[str, Any]],
) -> tuple[int, float, float, dict[str, Any]]:
    """Relabel legs in place; returns (swaps, swap_pct, collapse_pct, diag).

    The first three match ``_fix_leg_swaps``'s contract so the stabilizer's
    context keys keep their meaning. ``swap_pct`` is the share of measured
    frames whose labels arrived opposite to their segment's resolved naming,
    folded per segment so that a stable-but-renamed stretch reads ~0 --
    renaming is not instability. The raw crossing rate is ``crossings_pct``
    in ``diag``; the honest RESIDUAL-risk numbers for a gate to read are
    ``ambiguous_pair_pct`` (links decided by hysteresis, not evidence) and
    ``naming_unconfident_pct`` (frames whose segment naming had no cues
    behind it).
    """
    n = len(frame_results)
    empty_diag: dict[str, Any] = {"method": "dp", "frames": n}
    if n < 3:
        return 0, 0.0, 0.0, empty_diag

    joints = [_frame_joints(fr) for fr in frame_results]

    # Collapse share, same definition as the greedy pass: ankles too close to
    # be two legs, over frames where both ankles were measured. The flags are
    # kept per frame because a collapsed frame carries ZERO left/right
    # evidence -- both labels sit on one leg -- and has to be treated exactly
    # like a gap by everything downstream: excluded from the path, and, in a
    # long enough run, cutting the naming segment.
    collapsed = np.zeros(n, dtype=bool)
    collapse = measured = 0
    for idx, j in enumerate(joints):
        if j is None:
            continue
        la, ra = j[1] if len(j) > 1 else (None, None)
        if la is not None and ra is not None:
            measured += 1
            if math.dist(la, ra) < _COLLAPSE_DIST:
                collapse += 1
                collapsed[idx] = True
    collapse_pct = round(collapse / max(measured, 1) * 100, 1)

    # --- arcs -------------------------------------------------------------
    # arcs[k] connects measured frame k to measured frame k+1.
    measured_idx = [
        i for i, j in enumerate(joints) if j is not None and not collapsed[i]
    ]
    if len(measured_idx) < 3:
        return 0, 0.0, 0.0, empty_diag

    def _ts(i: int) -> float | None:
        ts = frame_results[i].get("timestamp_ms")
        return float(ts) if isinstance(ts, (int, float)) else None

    spacings = [
        b - a
        for a, b in zip(
            (_ts(i) for i in measured_idx), (_ts(i) for i in measured_idx[1:]),
        )
        if a is not None and b is not None and b > a
    ]
    median_spacing = float(np.median(spacings)) if spacings else 33.0
    if median_spacing > _MAX_MEDIAN_SPACING_MS:
        # Uniform decimation puts EVERY link at the aliased horizon while
        # "excess over the clip's own spacing" reads zero -- the regime where
        # the review measured ~50% wrong labels passing through with clean
        # diagnostics. Decline, and let the caller fall back to the greedy
        # path: unrepaired, but not laundered either.
        return 0, 0.0, collapse_pct, {
            "method": "skipped_coarse_sampling",
            "frames": n,
            "median_spacing_ms": round(median_spacing, 1),
        }

    arcs: list[tuple[int, float, float, float]] = []  # (frame, stay, cross, gap_ms)
    steps: list[float] = []
    prev_i = measured_idx[0]
    for i in measured_idx[1:]:
        stay, cross, used = _pair_costs(joints[prev_i], joints[i])
        ta, tb = _ts(prev_i), _ts(i)
        gap_ms = (
            tb - ta if ta is not None and tb is not None and tb > ta
            else (i - prev_i) * median_spacing
        )
        if used == 0:
            # Nothing matched across this link: no evidence either way, and
            # the prior still decays with its length.
            stay = cross = 0.0
        # A link that steps over a collapse is blind regardless of how little
        # wall-clock time passed: the frames in between carried no identity.
        blind = bool(collapsed[prev_i + 1:i].sum() >= _COLLAPSE_CUT_FRAMES)
        arcs.append((i, stay, cross, gap_ms, blind))
        if used > 0 and gap_ms <= 2 * median_spacing:
            steps.append(min(stay, cross))
        prev_i = i

    step_scale = float(np.median(steps)) if steps else 0.0
    if step_scale <= 1e-9:
        # A static clip has no evidence either way; leave labels alone.
        return 0, 0.0, collapse_pct, empty_diag
    switch_cost = _SWITCH_COST_STEPS * step_scale
    ambiguous_bar = _AMBIGUOUS_FRAC * step_scale

    # --- two-state shortest path -------------------------------------------
    # State: parity at the current measured frame (0 = keep raw labels,
    # 1 = swapped). Costs accumulate; backpointers recover the path.
    INF = float("inf")
    best = [0.0, 0.0]
    back: list[tuple[int, int]] = []
    margins: list[tuple[float, bool]] = []  # per arc: (decision margin, crossed)
    ambiguous = 0
    for _, stay, cross, gap_ms, blind in arcs:
        if blind:
            # Free in both directions: parity on the far side of a collapse is
            # anchored by its own segment's naming, never by a stale match.
            stay = cross = 0.0
            gap_ms = median_spacing + _SEGMENT_GAP_MS + 1.0
        excess = max(0.0, gap_ms - median_spacing)
        if excess > _SEGMENT_GAP_MS:
            # Antiphase aliasing: past this horizon the position evidence
            # points the wrong way as often as the right one. Cut the link --
            # both transitions free -- and let per-segment naming anchor the
            # far side instead of a stale, misleading match.
            stay = cross = hyst = 0.0
        else:
            w = math.exp(-excess / _GAP_EVIDENCE_HALFLIFE_MS)
            stay *= w
            cross *= w
            hyst = switch_cost * math.exp(-excess / _GAP_PRIOR_HALFLIFE_MS)
            if abs(stay - cross) < ambiguous_bar:
                ambiguous += 1
        nxt = [INF, INF]
        ptr = [0, 0]
        for cur_state in (0, 1):
            for prev_state in (0, 1):
                arc = stay if cur_state == prev_state else cross + hyst
                total = best[prev_state] + arc
                if total < nxt[cur_state]:
                    nxt[cur_state] = total
                    ptr[cur_state] = prev_state
        best = nxt
        back.append((ptr[0], ptr[1]))
        # With symmetric transition costs the path decomposes per link, so
        # this margin IS the confidence of the link's own decision.
        margins.append((abs(stay - (cross + hyst)), cross + hyst < stay))

    state = 0 if best[0] <= best[1] else 1
    parity_at = {measured_idx[-1]: state}
    for k in range(len(arcs) - 1, -1, -1):
        state = back[k][state]
        parity_at[measured_idx[k]] = state

    anchor_diag = _apply_stance_anchors(
        joints, measured_idx, parity_at, margins, _ts,
        median_spacing, step_scale,
    )

    # Unmeasured frames inherit the parity of the nearest previous measured
    # frame -- their landmarks are NaN anyway; this only keeps any stray
    # surviving points on a consistent side.
    parities = np.zeros(n, dtype=int)
    current = parity_at[measured_idx[0]]
    for i in range(n):
        current = parity_at.get(i, current)
        parities[i] = current

    crossings = sum(
        1
        for a, b in zip(measured_idx, measured_idx[1:])
        if parity_at[a] != parity_at[b]
    )

    # --- naming, per segment -------------------------------------------------
    # The path fixes consistency WITHIN stretches of continuous evidence; it
    # cannot know which track is "left", and across a long enough gap it
    # cannot even relate the parity on the two sides. So the clip is split at
    # those gaps and each segment is named on its own: the leg the segment's
    # pooled image cues place nearer the camera gets the label the torso's
    # depth vote says faces it. One global bit was tried first and failed a
    # known-truth fixture by flipping the correct half of a clip to repair
    # the wrong half.
    segments: list[tuple[int, int]] = []          # [start, end) in frame index
    seg_start = 0
    for i, _, _, gap_ms, blind in arcs:
        if blind or gap_ms - median_spacing > _SEGMENT_GAP_MS:
            segments.append((seg_start, i))
            seg_start = i
    segments.append((seg_start, n))

    torso_near = _torso_near_side(frame_results)
    seg_flips: list[dict[str, Any]] = []
    flip = np.zeros(n, dtype=int)
    unconfident_frames = 0
    for a, b in segments:
        info = _segment_naming(frame_results, parities, a, b, torso_near)
        if not info["confident"]:
            # No cue evidence either way: anchor to the segment's raw-label
            # MAJORITY rather than to whatever frame happened to come first.
            # Either naming gives equally consistent tracks; the majority one
            # rewrites the fewest frames, which is the least-surprise default
            # and keeps a cue-blind clip closest to what MediaPipe reported.
            seg_parity = parities[a:b]
            if seg_parity.size and float(seg_parity.mean()) > 0.5:
                info["flip"] = True
                info["anchor"] = "raw_majority"
            unconfident_frames += b - a
        seg_flips.append(info)
        if info["flip"]:
            flip[a:b] = 1

    swaps = 0
    for i, fr in enumerate(frame_results):
        if int(parities[i]) ^ int(flip[i]):
            for key in ("world_landmarks", "normalized_landmarks"):
                lms = fr.get(key)
                if lms is None:
                    continue
                for li, ri in _SWAP_PAIRS:
                    lms[li], lms[ri] = lms[ri], lms[li]
            swaps += 1

    # Input-noise figure for the gates, folded SYMMETRICALLY and PER SEGMENT:
    # within each naming segment, a stable-but-misnamed block reads ~0 --
    # renaming is not instability -- while genuinely alternating labels read
    # their true share. Folding globally was measured mis-reporting a
    # perfectly repaired two-segment clip as 43.5% "unstable" (and tripping
    # the severe tracking warning); folding per segment returns ~0 there.
    m = len(measured_idx)
    folded = 0
    for a, b in segments:
        seg_meas = [i for i in measured_idx if a <= i < b]
        relabelled = sum(
            1 for i in seg_meas if int(parities[i]) ^ int(flip[i])
        )
        folded += min(relabelled, len(seg_meas) - relabelled)
    swap_pct = round(folded / max(m, 1) * 100, 1)

    diag = {
        "method": "dp",
        "frames": n,
        "measured_frames": m,
        "crossings": crossings,
        "crossings_pct": round(crossings / max(len(arcs), 1) * 100, 1),
        "ambiguous_pair_pct": round(ambiguous / max(len(arcs), 1) * 100, 1),
        "step_scale": round(step_scale, 5),
        "segments": len(segments),
        "naming": seg_flips,
        "naming_unconfident_pct": round(unconfident_frames / n * 100, 1),
        # Frame spans whose naming rests on a raw-label majority rather than
        # cues. Consumers that draw ONE stride (the kinogram) must not pick it
        # from inside these.
        "unconfident_spans": [
            list(info["span"]) for info in seg_flips if not info["confident"]
        ],
        "relabelled_frames": swaps,
        "torso_near": torso_near,
        "anchors": anchor_diag,
    }
    logger.info(
        "LEG_IDENTITY_DP",
        **{k: v for k, v in diag.items() if k != "naming"},
    )
    return swaps, swap_pct, collapse_pct, diag


def _apply_stance_anchors(
    joints: list,
    measured_idx: list[int],
    parity_at: dict[int, int],
    margins: list[tuple[float, bool]],
    ts,
    median_spacing: float,
    step_scale: float,
) -> dict[str, Any]:
    """Repair the parity path against gait structure, in place.

    Down-runs of the RAW labels give stance fragments; fragments merge when
    their starts sit closer than a fraction of the clip's own stance spacing
    (one stance split by a glitch frame). Between neighbouring stances the
    required net parity change is fixed by whether the down-labels alternate.
    A violated stretch is repaired at its least-confident link -- undoing a
    barely-won crossing in preference to inventing one at a held near-tie.

    Mutates ``parity_at`` and returns a small diagnostic dict.
    """
    out: dict[str, Any] = {"stances": 0, "constraints": 0, "violated": 0,
                           "repaired": 0}
    m = len(measured_idx)
    if m < 3 or step_scale <= 0:
        return out

    # Median ankle separation sets the down margin's scale.
    seps = []
    for i in measured_idx:
        la, ra = joints[i][1]
        if la is not None and ra is not None:
            seps.append(math.dist(la, ra))
    if len(seps) < _ANCHOR_MIN_RUN:
        return out
    down_margin = max(
        _ANCHOR_DOWN_MARGIN_MIN,
        _ANCHOR_DOWN_MARGIN_FRAC * float(np.median(seps)),
    )

    # Down label per measured position: the ankle clearly lower in the image.
    def down(i: int) -> int | None:
        la, ra = joints[i][1]
        if la is None or ra is None:
            return None
        dy = la[1] - ra[1]
        if dy > down_margin:
            return 27
        if dy < -down_margin:
            return 28
        return None

    # Stance fragments: runs of one down label over consecutive measured
    # positions. (start_pos, end_pos) index into measured_idx.
    frags: list[tuple[int, int, int]] = []
    run_label: int | None = None
    run_start = 0
    for pos in range(m + 1):
        lab = down(measured_idx[pos]) if pos < m else None
        if lab != run_label:
            if run_label is not None and pos - run_start >= _ANCHOR_MIN_RUN:
                frags.append((run_start, pos - 1, run_label))
            run_label = lab
            run_start = pos
    if len(frags) < 3:
        return out

    # Down-label churn: the raw labels flipping while a foot is planted. Gait
    # can only be read off labels stable at the stance scale; past the bar
    # the anchors would repair the path against noise, so they stand down.
    churn = 0
    prev_lab: int | None = None
    for pos in range(m):
        lab = down(measured_idx[pos])
        if lab is not None:
            if prev_lab is not None and lab != prev_lab:
                churn += 1
            prev_lab = lab
        else:
            prev_lab = None
    out["churn_per_stance"] = round(churn / len(frags), 2)
    if churn > _ANCHOR_MAX_CHURN_PER_STANCE * len(frags):
        out["distrusted"] = "label_churn"
        return out

    def frag_ms(pos: int) -> float:
        t = ts(measured_idx[pos])
        return t if t is not None else measured_idx[pos] * median_spacing

    spacings = [
        frag_ms(b[0]) - frag_ms(a[0]) for a, b in zip(frags, frags[1:])
    ]
    step_ms = float(np.median(spacings))
    if not (100.0 <= step_ms <= 2000.0):
        return out

    # Merge same-label fragments split by a glitch; keep differing labels
    # apart (a label handed over mid-stance is itself a parity event).
    stances: list[tuple[int, int, int]] = []
    for frag in frags:
        if (
            stances
            and frag[2] == stances[-1][2]
            and frag_ms(frag[0]) - frag_ms(stances[-1][0])
            < _ANCHOR_MERGE_FRAC * step_ms
        ):
            stances[-1] = (stances[-1][0], frag[1], frag[2])
        else:
            stances.append(frag)
    out["stances"] = len(stances)
    if len(stances) < 2:
        return out

    # Anchor = the stance's midpoint (a measured position); constraint = the
    # net parity change required between neighbouring anchors.
    def down_ankle(pos: int) -> tuple[float, float] | None:
        la, ra = joints[measured_idx[pos]][1]
        lab = down(measured_idx[pos])
        return la if lab == 27 else ra if lab == 28 else None

    median_sep = float(np.median(seps))
    constraints: list[tuple[int, int, int]] = []  # (pos_a, pos_b, delta)
    for (s_a, e_a, lab_a), (s_b, e_b, lab_b) in zip(stances, stances[1:]):
        mid_a, mid_b = (s_a + e_a) // 2, (s_b + e_b) // 2
        gap = frag_ms(s_b) - frag_ms(s_a)
        if gap >= _ANCHOR_CHAIN_FRAC * step_ms:
            continue  # a stance went missing in between; no constraint
        if gap < _ANCHOR_MERGE_FRAC * step_ms:
            # Two down-runs closer than a stance can repeat, with DIFFERENT
            # labels (same-label splits were merged above). Same physical
            # foot -- its position continues across the split -- means the
            # label was handed over mid-stance: one parity event. A clear
            # position jump means the next stance simply landed early or the
            # first was clipped: ordinary alternation.
            pa, pb = down_ankle(e_a), down_ankle(s_b)
            if pa is not None and pb is not None and (
                math.dist(pa, pb) < 0.5 * median_sep
            ):
                delta = 1  # label handover on a planted foot
            else:
                delta = 0  # two stances, alternating feet
        else:
            delta = 1 if lab_a == lab_b else 0  # consecutive stances
        constraints.append((mid_a, mid_b, delta))
    out["constraints"] = len(constraints)
    if not constraints:
        return out

    def parity(pos: int) -> int:
        return parity_at[measured_idx[pos]]

    violated = [
        (a, b, d) for a, b, d in constraints if (parity(a) ^ parity(b)) != d
    ]
    out["violated"] = len(violated)
    if not violated:
        return out
    if len(violated) / len(constraints) > _ANCHOR_MAX_VIOLATION_FRAC:
        # The anchors themselves are suspect; do not repair from them.
        out["distrusted"] = True
        return out

    stay_penalty = _ANCHOR_STAY_PENALTY_STEPS * step_scale
    repaired = 0
    for a, b, delta in constraints:
        if (parity(a) ^ parity(b)) == delta:
            continue
        # Least-confident link in (a, b]; margins[k] belongs to the arc from
        # measured position k to k+1. Stay links pay the invention penalty.
        best_k = None
        best_cost = float("inf")
        for k in range(a, b):
            cost = margins[k][0] + (0.0 if margins[k][1] else stay_penalty)
            if cost < best_cost:
                best_cost = cost
                best_k = k
        if best_k is None:
            continue
        for pos in range(best_k + 1, m):
            parity_at[measured_idx[pos]] ^= 1
        repaired += 1
    out["repaired"] = repaired
    return out


def _torso_near_side(frame_results: list[dict[str, Any]]) -> str | None:
    """Which labelled side the torso's own depth puts nearer the camera.

    The same signal ``finalize_camera_side`` votes on later, pooled here over
    the whole clip. Legs are deliberately not consulted: their labels are the
    thing in question.
    """
    left_z: list[float] = []
    right_z: list[float] = []
    for fr in frame_results:
        wl = fr.get("world_landmarks")
        if wl is None:
            continue
        try:
            lz = (wl[11].z + wl[23].z) / 2.0
            rz = (wl[12].z + wl[24].z) / 2.0
        except (IndexError, TypeError, AttributeError):
            continue
        if not (math.isnan(lz) or math.isnan(rz)):
            left_z.append(lz)
            right_z.append(rz)
    if len(left_z) < 5:
        return None
    return "left" if float(np.median(left_z)) < float(np.median(right_z)) else "right"


def _segment_naming(
    frame_results: list[dict[str, Any]],
    parities: np.ndarray,
    start: int,
    end: int,
    torso_near: str | None,
) -> dict[str, Any]:
    """Naming decision for one segment: flip its parity, or leave it.

    Works on a RELABELLED VIEW (the segment's parities applied virtually) so
    the pooled cues describe consistent tracks rather than the raw mixture.
    Unconfident segments keep their raw-majority naming and say so.
    """
    out: dict[str, Any] = {
        "span": [int(start), int(end)], "flip": False, "confident": False,
    }
    if torso_near is None:
        out["reason"] = "torso_depth_unreadable"
        return out

    view: list[dict[str, Any]] = []
    for i in range(start, end):
        fr = frame_results[i]
        nl, wl = fr.get("normalized_landmarks"), fr.get("world_landmarks")
        if nl is None or wl is None:
            continue
        if parities[i]:
            nl = list(nl)
            wl = list(wl)
            for li, ri in _SWAP_PAIRS:
                nl[li], nl[ri] = nl[ri], nl[li]
                wl[li], wl[ri] = wl[ri], wl[li]
        view.append({"normalized_landmarks": nl, "world_landmarks": wl})
    if len(view) < 5:
        out["reason"] = "too_short"
        return out

    from app.services.video_analysis.biomechanics.tracking_diagnostics import (
        compute_near_side_cues,
    )

    cues = compute_near_side_cues(view)
    score = cues.get("near_side_score") or 0.0
    cue_near = cues.get("suggested_near_side")
    out["cue_near"] = cue_near
    out["cue_score"] = score
    if not cue_near or abs(score) < _NAMING_MIN_SCORE:
        out["reason"] = "cues_indecisive"
        return out
    out["confident"] = True
    # Cues say which LABELLED side currently reads as near. If that is not
    # the side the torso says faces the camera, this segment's near track is
    # wearing the far label: flip it.
    out["flip"] = cue_near != torso_near
    return out
