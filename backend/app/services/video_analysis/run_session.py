"""Two run clips, one runner: what they agree on, and what they cannot say.

WHY THIS IS NOT THE BIKE SESSION. On a bike the two clips share a rigid
object -- the crank, the saddle, the unchanged fit -- so their geometry can be
pooled against a common ruler. Two running clips share nothing but the
athlete. There is no scale to reconcile and nothing to reconstruct; each clip
simply measures the leg nearest its camera, plus a set of whole-body
quantities that both clips measure independently.

That difference is what makes a run pair worth having, and it is the opposite
of the bike's. Cadence, trunk lean, ground contact, flight time and vertical
oscillation are ONE quantity seen twice. Whatever gap appears between the two
clips on those is the method's own error, measured on this athlete, on this
day, with no assumption at all. It is the only honest yardstick available for
reading the per-leg numbers next to it.

MEASURED ON THE FIRST REAL PAIR (2026-08-29, IMG_4258 / IMG_4262). Those
whole-body metrics disagreed by 6% on cadence, 10% on flight time, 22% on
oscillation, 36% on ground contact and 68% on trunk lean -- while the two
legs' knee angles differed by 4% at the mean. In other words the error on
quantities that CANNOT differ was larger than the difference between the legs.
On that pair, a confident left-versus-right verdict would have been invented.

So: the session reports the agreement first and treats it as the error bar,
and it refuses to merge at all when either clip failed the leg-identity check
(see biomechanics/stride_consistency.py) -- merging a clip whose legs traded
places is merging the swap into the answer.
"""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Quantities BOTH clips measure independently: one runner, one number. Their
# spread across the two clips is this session's own error bar.
WHOLE_BODY_METRICS = (
    "cadence_spm",
    "trunk_lean_avg",
    "vertical_oscillation_m",
    "ground_contact_ms",
    "flight_time_ms",
    "stance_fraction",
)

# Measured on the leg nearest each camera, so the two clips describe DIFFERENT
# legs. The only place a real left/right difference could live -- and only if
# it clears the error bar above.
PER_LEG_METRICS = (
    "knee_min",
    "knee_max",
    "knee_mean",
    "knee_flexion_velocity_dps",
    "knee_extension_velocity_dps",
    "foot_strike_angle_deg",
    "overstride_ratio",
)

# A whole-body metric disagreeing by more than this between two clips of one
# runner means the two analyses are not describing the same run closely enough
# to pool. On the first real pair, trunk lean disagreed by 68%.
AGREEMENT_LIMIT_PCT = 25.0


def _side_of(result: dict[str, Any]) -> str | None:
    side = result.get("camera_side")
    return side if side in ("left", "right") else None


def _stability(result: dict[str, Any]) -> dict[str, Any] | None:
    """The clip's leg-identity stability, from where the runner files it.

    Inside ``sport_specific_metrics``, not at the top level -- runner.py does
    `summary["tracking_stability"] = ...`. A top-level lookup here looks right
    and silently finds nothing, which would let an unstable clip through the
    gate that exists to stop it.
    """
    summary = result.get("sport_specific_metrics") or {}
    tracking = summary.get("tracking_stability") or result.get("tracking_stability") or {}
    ident = tracking.get("leg_identity") or {}
    return ident.get("stability")


def _num(summary: dict[str, Any], key: str) -> float | None:
    v = summary.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def agreement(
    summary_a: dict[str, Any], summary_b: dict[str, Any],
) -> dict[str, Any]:
    """How far apart the two clips are on things that cannot differ.

    Reported as a percentage of the larger value so a 10 ms gap on contact
    time and a 1 degree gap on trunk lean are comparable.
    """
    gaps: dict[str, float] = {}
    for key in WHOLE_BODY_METRICS:
        a, b = _num(summary_a, key), _num(summary_b, key)
        if a is None or b is None:
            continue
        scale = max(abs(a), abs(b))
        if scale <= 1e-9:
            continue
        gaps[key] = round(abs(a - b) / scale * 100.0, 1)
    if not gaps:
        return {"agree": None, "gaps": {}, "worst": None, "worst_metric": None}
    worst_metric = max(gaps, key=lambda k: gaps[k])
    return {
        "agree": gaps[worst_metric] <= AGREEMENT_LIMIT_PCT,
        "gaps": gaps,
        "worst": gaps[worst_metric],
        "worst_metric": worst_metric,
        "limit_pct": AGREEMENT_LIMIT_PCT,
    }


def compare_legs(
    summary_left: dict[str, Any],
    summary_right: dict[str, Any],
    agree: dict[str, Any],
) -> list[dict[str, Any]]:
    """Left against right, each difference judged against this session's own
    measured error.

    The error bar is not a constant and not a guess: it is the largest gap the
    two clips showed on a quantity that CANNOT differ between them. A per-leg
    difference smaller than that is not something this pair can resolve, and
    saying so is the whole point -- the bike feature had to withdraw a tidy
    left/right number once for exactly this reason.
    """
    bar = agree.get("worst")
    out: list[dict[str, Any]] = []
    for key in PER_LEG_METRICS:
        a, b = _num(summary_left, key), _num(summary_right, key)
        if a is None or b is None:
            continue
        scale = max(abs(a), abs(b))
        diff_pct = (abs(a - b) / scale * 100.0) if scale > 1e-9 else 0.0
        out.append({
            "metric": key,
            "left": round(a, 2),
            "right": round(b, 2),
            "difference_pct": round(diff_pct, 1),
            # None when there is no error bar to judge against -- unknown, not
            # "significant by default".
            "readable": None if bar is None else bool(diff_pct > bar),
        })
    return out


def _leg_card(result: dict[str, Any]) -> dict[str, Any]:
    """One clip's contribution: its own leg, its own numbers, no score."""
    summary = result.get("sport_specific_metrics") or {}
    stab = _stability(result) or {}
    return {
        "camera_side": _side_of(result),
        "metrics": {k: _num(summary, k) for k in PER_LEG_METRICS
                    if _num(summary, k) is not None},
        "frames_analyzed": result.get("frames_analyzed"),
        "identity_instability": stab.get("instability"),
        "identity_unstable": bool(stab.get("unstable")),
        "keyframe_base64": result.get("keyframe_base64"),
        "has_keyframe": bool(result.get("keyframe_base64")),
    }


def build_run_session(
    result_a: dict[str, Any], result_b: dict[str, Any],
) -> dict[str, Any]:
    """Merge two completed single-side run analyses into one session verdict.

    Never raises on a pair that cannot be merged: the athlete filmed two clips
    and is owed an answer about them either way. What is withheld on a refusal
    is the merged verdict, not the measurements.
    """
    sides: dict[str, dict[str, Any]] = {}
    for r in (result_a, result_b):
        s = _side_of(r)
        if s:
            sides[s] = r
    if len(sides) != 2:
        return _refusal(result_a, result_b, "sides_not_identified")

    left, right = sides["left"], sides["right"]
    sum_l = left.get("sport_specific_metrics") or {}
    sum_r = right.get("sport_specific_metrics") or {}
    agree = agreement(sum_l, sum_r)

    # Gate one: a clip whose legs traded places cannot be pooled -- the swap
    # would be merged into the answer, and the per-leg numbers on that clip
    # are already spliced from both legs.
    unstable = [s for s, r in sides.items() if (_stability(r) or {}).get("unstable")]
    if unstable:
        return _refusal(left, right, "identity_unstable", agree=agree,
                        unstable_sides=unstable)

    # Gate two: two clips of one runner that disagree about a quantity neither
    # of them may have an opinion on are not describing the same run.
    if agree.get("agree") is False:
        return _refusal(left, right, "clips_disagree", agree=agree)

    # Which clip everything UNMERGEABLE comes from -- the per-leg metrics here,
    # and the score, angle table and findings the caller carries over.
    #
    # This used to be `sum_l if len(sum_l) >= len(sum_r) else sum_r`: the clip
    # with more KEYS won, so one optional field (a foot-strike angle that
    # happened to be measurable on one side) decided whose knee the session
    # reported. It was arbitrary in the literal sense -- nothing about it
    # tracked which clip was better.
    #
    # Frames analysed is a real proxy: more frames is more strides behind every
    # per-leg number. Ties go left, only so the choice is deterministic.
    base_side = "right" if (
        int(right.get("frames_analyzed") or 0) > int(left.get("frames_analyzed") or 0)
    ) else "left"
    base_summary = sum_r if base_side == "right" else sum_l

    merged = dict(base_summary)
    for key in WHOLE_BODY_METRICS:
        a, b = _num(sum_l, key), _num(sum_r, key)
        if a is not None and b is not None:
            merged[key] = (a + b) / 2.0
    merged["camera_side"] = "both"
    merged["camera_side_label"] = "Both sides"
    # Everything in this summary that is NOT in WHOLE_BODY_METRICS was measured
    # on one leg, and "both" would otherwise be the only label on it. The bike
    # session does not need this key because it genuinely reconstructs a
    # two-legged number; a run pair cannot, so it says whose leg this is
    # instead of implying it is nobody's in particular.
    merged["per_leg_source"] = base_side

    session = {
        "combined": True,
        "agreement": agree,
        "legs": {"left": _leg_card(left), "right": _leg_card(right)},
        "merged_whole_body": {k: round(merged[k], 2) for k in WHOLE_BODY_METRICS
                              if isinstance(merged.get(k), (int, float))},
        "leg_comparison": compare_legs(sum_l, sum_r, agree),
        # The caller must carry the score, angles and findings from THIS clip,
        # or the report shows one leg's score above another leg's numbers.
        "base_side": base_side,
    }
    logger.info(
        "RUN_SESSION", combined=True, worst_gap=agree.get("worst"), base_side=base_side,
    )
    return {"session": session, "merged_summary": merged, "base_side": base_side}


def _refusal(
    result_a: dict[str, Any],
    result_b: dict[str, Any],
    reason: str,
    agree: dict[str, Any] | None = None,
    unstable_sides: list[str] | None = None,
) -> dict[str, Any]:
    """A pair that cannot be merged, said plainly and with its evidence."""
    logger.info("RUN_SESSION", combined=False, reason=reason)
    # Even a refusal has to name the clip the report is built from: the caller
    # still shows a score and an angle table, and they still come from ONE of
    # these two clips. Same rule as the merge -- the one with more frames.
    base_side = "left"
    if int((result_b or {}).get("frames_analyzed") or 0) > int(
        (result_a or {}).get("frames_analyzed") or 0
    ):
        base_side = _side_of(result_b) or "left"
    else:
        base_side = _side_of(result_a) or "left"
    return {
        "session": {
            "combined": False,
            "reason": reason,
            "agreement": agree or {},
            "unstable_sides": unstable_sides or [],
            "legs": {
                s: _leg_card(r)
                for r in (result_a, result_b)
                if (s := _side_of(r))
            },
            "base_side": base_side,
        },
        "merged_summary": None,
        "base_side": base_side,
    }
