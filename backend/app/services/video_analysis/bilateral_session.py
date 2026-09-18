"""Assemble two single-side analyses into one two-sided session result.

The measurement work is in ``biomechanics.bilateral``; this module is the
product layer around it -- one score, one plan, one verdict, and an honest
account of what happened when the two clips could not be merged.

Shape of the returned result: a normal analysis result, so every renderer,
gate and history writer downstream keeps working, with the merged numbers
substituted and a ``bilateral`` block describing the merge. The per-side
results ride along under ``bilateral.sides`` -- they are what the rider filmed
and they stay inspectable -- but only ONE score is presented, because two
scores for one body is the confusion this feature was built to end.
"""

from __future__ import annotations

from typing import Any

import structlog

from app.services.video_analysis.biomechanics.bilateral import (
    SideGeometry,
    combine_sides,
    merge_summaries,
    merge_summaries_partial,
    midline_agreement,
)

logger = structlog.get_logger(__name__)

# Fields that describe ONE clip and would be a lie on a merged result.
_PER_SIDE_ONLY = (
    "keyframe_base64", "overlay_video_path", "kinogram_base64",
    "bilateral_geometry", "ai_recommendations",
    # One clip's frame record; the session plays two overlays. No timeline
    # on a pair until it carries one per side.
    "timeline",
)


def _side_of(result: dict[str, Any]) -> str | None:
    side = result.get("camera_side")
    return side if side in ("left", "right") else None


def _warnings_of(result: dict[str, Any]) -> list[str]:
    """Where the analysis actually keeps its quality warnings.

    Inside ``sport_specific_metrics`` -- see runner.py, which writes
    ``summary["quality_warnings"]``, and the SPA, which reads
    ``s.quality_warnings`` to fill the amber banner. A top-level key here
    looks right and is never rendered; the first version of this module wrote
    one and the warnings silently went nowhere.
    """
    summary = result.get("sport_specific_metrics") or {}
    return list(summary.get("quality_warnings") or result.get("quality_warnings") or [])


def _side_card(result: dict[str, Any]) -> dict[str, Any]:
    """What the session shows ABOUT one clip: no score, on purpose.

    ``knee_at_bdc`` is what this clip measured ON ITS OWN -- a fact about a
    clip, not a claim about a leg. A previous version printed a per-side value
    "reconciled" against the shared body, which looked much better (the two
    sides landed a degree apart) and was an artifact: that split is the scale
    choice restated, so it agreed by construction. See the note on
    ``bilateral.combine_sides``. The difference between these two numbers is
    the instrument, and the panel says so.
    """
    summary = result.get("sport_specific_metrics") or {}
    geom = result.get("bilateral_geometry") or {}
    return {
        "camera_side": _side_of(result),
        "knee_at_bdc": summary.get("knee_at_bdc"),
        "trunk_angle_avg": summary.get("trunk_angle_avg"),
        "frames_analyzed": result.get("frames_analyzed"),
        "revolutions": geom.get("revolutions"),
        "quality_warnings": _warnings_of(result)[:3],
        "keyframe_base64": result.get("keyframe_base64"),
        "has_keyframe": bool(result.get("keyframe_base64")),
        # The clip's own joint table. A side view measures the near leg, so
        # each clip of a session fills the half of the angle table the other
        # cannot see -- without this the report showed one clip's six rows and
        # asked the rider to "analyze the other side" he had just uploaded.
        "angle_statistics": _compact_angle_stats(result.get("angle_statistics")),
    }


# What the angle table reads per joint (see angleTableRows in the SPA): the
# percentiles it prints, the mean it grades, and the frame counts behind the
# "valid" column. Everything else on a stats entry is per-frame detail that
# would double the size of a session result for nothing the page shows.
_ANGLE_STAT_FIELDS = (
    "min", "mean", "max", "p05", "p95",
    "valid_frames", "nan_frames", "artefact_frames", "artefact_pct",
)


def _compact_angle_stats(stats: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(stats, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, entry in stats.items():
        if not isinstance(entry, dict):
            continue
        out[str(key)] = {f: entry.get(f) for f in _ANGLE_STAT_FIELDS if f in entry}
    return out


def build_pair_result(
    result_a: dict[str, Any],
    result_b: dict[str, Any],
    cycling_position: str | None = None,
    recommendations: bool = False,
) -> dict[str, Any]:
    """Merge two completed single-side bike analyses into one session result.

    Never raises on a merge that cannot be made: a refusal comes back as a
    result carrying both clips and the reason, because the rider filmed two
    clips and is owed an answer about them either way.
    """
    sides = {}
    for r in (result_a, result_b):
        s = _side_of(r)
        if s:
            sides[s] = r
    if len(sides) != 2:
        return _refusal(result_a, result_b, "sides_not_identified")

    left, right = sides["left"], sides["right"]
    # Computed before anything can refuse: it is the evidence a partial merge
    # stands on when the knee cannot be pooled.
    agreement = midline_agreement(
        left.get("sport_specific_metrics") or {},
        right.get("sport_specific_metrics") or {},
    )
    geom_left = SideGeometry.from_dict(left.get("bilateral_geometry"))
    geom_right = SideGeometry.from_dict(right.get("bilateral_geometry"))
    if geom_left is None or geom_right is None:
        # One clip was too disturbed to reduce -- most often the tracker never
        # held the ankle on the pedal path long enough. The knee cannot be
        # pooled; the rest of the rider may still be.
        return _without_pooled_knee(
            left, right, "geometry_unavailable", agreement,
            cycling_position, recommendations,
        )

    fit = combine_sides(geom_left, geom_right)
    if not fit.combined:
        return _without_pooled_knee(
            left, right, fit.reason or "not_combinable", agreement,
            cycling_position, recommendations, fit=fit,
        )

    merged_summary = merge_summaries(
        left.get("sport_specific_metrics") or {},
        right.get("sport_specific_metrics") or {},
        fit, cycling_position,
    )

    # Build on the clip with more revolutions behind it, so the keys a
    # renderer expects are all present and internally consistent.
    base = left if (geom_left.revolutions >= geom_right.revolutions) else right
    other = right if base is left else left
    result = {k: v for k, v in base.items() if k not in _PER_SIDE_ONLY}
    result["sport_specific_metrics"] = merged_summary
    result["camera_side"] = "both"
    result["frames_analyzed"] = (
        int(base.get("frames_analyzed") or 0) + int(other.get("frames_analyzed") or 0)
    )

    scoring = _rescore(merged_summary, base, cycling_position)
    result["technique_score"] = scoring.get("overall_score")
    result["letter_grade"] = scoring.get("letter_grade")
    result["score_breakdown"] = scoring.get("component_scores")
    if scoring.get("coverage") is not None:
        result["score_coverage"] = scoring.get("coverage")

    result["fit_plan"] = _rebuild_fit_plan(
        merged_summary, base, cycling_position, scoring,
    )

    # Both clips' warnings, deduplicated: a problem with either one is a
    # problem with the session. Written where the renderer looks for them --
    # inside the merged summary, not at the top level.
    seen, warnings = set(), []
    for r in (left, right):
        for w in _warnings_of(r):
            if w not in seen:
                seen.add(w)
                warnings.append(w)
    merged_summary["quality_warnings"] = warnings

    result["bilateral"] = {
        **fit.as_dict(),
        "agreement": agreement,
        "sides": [_side_card(left), _side_card(right)],
        "base_side": _side_of(base),
    }
    result["keyframe_base64"] = base.get("keyframe_base64")

    if recommendations:
        result["ai_recommendations"] = _coach(result, cycling_position)

    logger.info(
        "BILATERAL_SESSION", combined=True,
        score=result.get("technique_score"),
        knee=fit.knee_at_bdc and round(fit.knee_at_bdc, 1),
        agree=agreement.get("agree"),
    )
    return result


# Knee-pooling failures a partial merge may stand in for. Not `leg_mismatch`:
# that one says the two legs are too different to be one rider on one fit,
# and a session that agrees on the trunk but not on who is riding is not a
# session. Not the side-identity failures either -- there is no pair to be
# partial about.
_PARTIAL_OK_REASONS = frozenset({
    "geometry_unavailable", "no_shared_scale", "degenerate_geometry",
    "scale_mismatch",
})


def _without_pooled_knee(
    left: dict[str, Any],
    right: dict[str, Any],
    reason: str,
    agreement: dict[str, Any],
    cycling_position: str | None,
    recommendations: bool,
    fit: Any = None,
) -> dict[str, Any]:
    """The knee could not be pooled. Decide between a partial merge and none.

    A partial merge needs two things: a reason of the kind that says "the
    knee" rather than "not one rider", and the midline metrics AGREEING --
    that agreement is the whole basis for pooling them, and it is the number
    the panel prints as the session's own error bar. Without it the clips are
    refused for disagreeing, which is a truer reason than the knee's.
    """
    if reason not in _PARTIAL_OK_REASONS:
        return _refusal(left, right, reason, fit=fit, agreement=agreement)
    if agreement.get("agree") is False:
        return _refusal(left, right, "midline_disagree", fit=fit, agreement=agreement)
    if agreement.get("agree") is None:
        # Nothing both clips measured: no evidence to merge on.
        return _refusal(left, right, reason, fit=fit, agreement=agreement)
    knee, other = _knee_clip(left, right)
    if knee is None:
        return _refusal(left, right, reason, fit=fit, agreement=agreement)
    return _partial(knee, other, reason, agreement, cycling_position,
                    recommendations, fit=fit)


def _knee_clip(
    left: dict[str, Any], right: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Which clip's knee the session stands on: the one with a pedal circle,
    or with more revolutions behind it when both have one."""
    def _revs(r: dict[str, Any]) -> int:
        return int((r.get("bilateral_geometry") or {}).get("revolutions") or 0)

    def _has_knee(r: dict[str, Any]) -> bool:
        v = (r.get("sport_specific_metrics") or {}).get("knee_at_bdc")
        return isinstance(v, (int, float))

    ranked = sorted((r for r in (left, right) if _has_knee(r)),
                    key=_revs, reverse=True)
    if not ranked:
        return None, None
    knee = ranked[0]
    return knee, (right if knee is left else left)


def _partial(
    knee: dict[str, Any],
    other: dict[str, Any],
    reason: str,
    agreement: dict[str, Any],
    cycling_position: str | None,
    recommendations: bool,
    fit: Any = None,
) -> dict[str, Any]:
    """Merged on everything both clips see; the knee from one clip, said so."""
    knee_side, other_side = _side_of(knee), _side_of(other)
    merged_summary = merge_summaries_partial(
        knee.get("sport_specific_metrics") or {},
        other.get("sport_specific_metrics") or {},
    )
    result = {k: v for k, v in knee.items() if k not in _PER_SIDE_ONLY}
    result["sport_specific_metrics"] = merged_summary
    result["camera_side"] = "both"
    result["frames_analyzed"] = (
        int(knee.get("frames_analyzed") or 0) + int(other.get("frames_analyzed") or 0)
    )

    scoring = _rescore(merged_summary, knee, cycling_position)
    result["technique_score"] = scoring.get("overall_score")
    result["letter_grade"] = scoring.get("letter_grade")
    result["score_breakdown"] = scoring.get("component_scores")
    if scoring.get("coverage") is not None:
        result["score_coverage"] = scoring.get("coverage")
    result["fit_plan"] = _rebuild_fit_plan(merged_summary, knee, cycling_position, scoring)

    # Why the knee is one clip's, in the words the reason deserves. The
    # drive-side case is named as such: the runner already warns that clip
    # about the chainring, and the same fact explains why it has no circle.
    if reason == "geometry_unavailable":
        drive = other_side == "right"
        why = (
            f"the {other_side}-side clip's pedal circle could not be measured"
            + (" -- it was filmed from the drive side, where the chainring sits "
               "behind the ankle at the bottom of the stroke" if drive else "")
        )
    else:
        pct = None if fit is None else fit.as_dict().get("scale_chord_disagreement_pct")
        why = (
            "the two clips could not be put on one scale"
            + (f" (they disagree by {pct}% on a length that cannot change between "
               f"two clips of one bike)" if pct is not None else "")
        )
    lead = (
        f"Merged on everything both clips see: trunk, hip, shoulder and elbow "
        f"agree to {agreement.get('worst')}°. The knee at the bottom of the "
        f"stroke, and the saddle verdict built on it, come from the {knee_side}-side "
        f"clip alone, because {why}."
    )
    seen, warnings = set(), [lead]
    for r in (knee, other):
        for w in _warnings_of(r):
            if w not in seen:
                seen.add(w)
                warnings.append(w)
    merged_summary["quality_warnings"] = warnings

    geom = knee.get("bilateral_geometry") or {}
    result["bilateral"] = {
        "combined": False,
        "partial": True,
        "reason": reason,
        "agreement": agreement,
        "sides": [_side_card(r) for r in (knee, other)],
        "base_side": knee_side,
        "knee_side": knee_side,
        "knee_single": {
            "side": knee_side,
            "value": merged_summary.get("knee_at_bdc"),
            "revolutions": geom.get("revolutions"),
        },
        "scale_chord_disagreement_pct": (
            None if fit is None else fit.as_dict().get("scale_chord_disagreement_pct")
        ),
    }
    result["keyframe_base64"] = knee.get("keyframe_base64")
    if recommendations:
        result["ai_recommendations"] = _coach(result, cycling_position)
    logger.info(
        "BILATERAL_SESSION", combined=False, partial=True, reason=reason,
        knee_side=knee_side, score=result.get("technique_score"),
        agree_worst=agreement.get("worst"),
    )
    return result


def _refusal(
    result_a: dict[str, Any],
    result_b: dict[str, Any],
    reason: str,
    fit: Any = None,
    agreement: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A session that could not be merged, told plainly.

    The two analyses are both real and both stay readable; what is withheld is
    the single merged verdict, because that is the part we cannot stand behind.
    The score shown is the better-supported clip's own, clearly labelled as one
    side -- not a merged number wearing a merged number's authority.
    """
    base = result_a if (result_a.get("technique_score") is not None) else result_b
    result = {k: v for k, v in base.items() if k not in _PER_SIDE_ONLY}
    result["keyframe_base64"] = base.get("keyframe_base64")
    result["bilateral"] = {
        "combined": False,
        "reason": reason,
        "agreement": agreement or {},
        "sides": [_side_card(r) for r in (result_a, result_b) if _side_of(r)],
        "scale_chord_disagreement_pct": (
            None if fit is None
            else fit.as_dict().get("scale_chord_disagreement_pct")
        ),
        # Which clip the numbers above this panel actually came from. Without
        # it the metric table reads as the session's, and a rider who filmed
        # two sides believes he is looking at both.
        "metrics_side": _side_of(base),
    }
    # A refusal has to reach the loud channel, not only the panel underneath.
    # It changes what every number on the page MEANS -- from "your fit" to
    # "one side of your fit" -- and that is not a footnote.
    side = _side_of(base) or "one"
    summary = dict(result.get("sport_specific_metrics") or {})
    summary["quality_warnings"] = [
        f"The two clips could not be merged into one verdict, so every number "
        f"on this page was measured from the {side}-side clip alone. Both "
        f"clips were analysed and both are shown below.",
        *_warnings_of(base),
    ]
    result["sport_specific_metrics"] = summary
    logger.info("BILATERAL_SESSION", combined=False, reason=reason)
    return result


def _rescore(
    merged_summary: dict[str, Any],
    base: dict[str, Any],
    cycling_position: str | None,
) -> dict[str, Any]:
    """Score the merged ride. One body, one number."""
    try:
        from app.services.video_analysis.biomechanics.technique_scorer import (
            score_cycling,
        )
        return score_cycling(
            merged_summary, base.get("angle_statistics") or {}, cycling_position,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("BILATERAL_RESCORE_FAILED", err=str(e))
        return {}


def _rebuild_fit_plan(
    merged_summary: dict[str, Any],
    base: dict[str, Any],
    cycling_position: str | None,
    scoring: dict[str, Any],
) -> Any:
    """Re-derive the adjustments from the MERGED metrics.

    Carrying one clip's plan forward would hand the rider a saddle instruction
    computed from one leg while the report above it shows a merged number --
    the two halves of the product disagreeing again, one layer down.
    """
    try:
        from app.services.video_analysis.biomechanics.action_plan_builder import (
            action_plan_to_json,
            build_action_plan,
        )
        return action_plan_to_json(build_action_plan(
            position=cycling_position or "road_hoods",
            angle_statistics=base.get("angle_statistics") or {},
            sport_specific_metrics=merged_summary,
            technique_score=scoring.get("overall_score") or 0,
            letter_grade=scoring.get("letter_grade") or "--",
            detected_issues=base.get("detected_issues") or [],
            mobility_fit=base.get("mobility_fit"),
        ))
    except Exception as e:  # noqa: BLE001
        logger.warning("BILATERAL_FIT_PLAN_FAILED", err=str(e))
        return None


def _coach(result: dict[str, Any], cycling_position: str | None) -> Any:
    """Fresh coaching prose for the merged numbers.

    Regenerated rather than inherited: a report written about one leg, printed
    under a merged score, is the same contradiction in words instead of digits.
    """
    try:
        from app.services.video_analysis.llm_recommendations import (
            generate_recommendations,
        )
        return generate_recommendations(
            sport_type="bike",
            technique_score=result.get("technique_score"),
            letter_grade=result.get("letter_grade"),
            detected_issues=result.get("detected_issues") or [],
            sport_specific_metrics=result.get("sport_specific_metrics") or {},
            angle_statistics=result.get("angle_statistics") or {},
            cycling_position=cycling_position,
            # Hand it the merged adjustments, same as the single-clip path, so
            # the prose and the plan cannot disagree about which way to move.
            fit_plan=result.get("fit_plan"),
            focus=result.get("focus"),
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("BILATERAL_COACH_FAILED", err=str(e))
        return None
