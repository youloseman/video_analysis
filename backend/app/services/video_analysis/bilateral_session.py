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
    midline_agreement,
)

logger = structlog.get_logger(__name__)

# Fields that describe ONE clip and would be a lie on a merged result.
_PER_SIDE_ONLY = (
    "keyframe_base64", "overlay_video_path", "kinogram_base64",
    "bilateral_geometry", "ai_recommendations",
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


def _side_card(
    result: dict[str, Any], merged_knee: float | None = None,
) -> dict[str, Any]:
    """What the session shows ABOUT one clip: no score, on purpose.

    ``knee_at_bdc`` is the value AFTER the merge when there is one -- the leg
    re-measured against the shared body. Showing each clip's raw reading here
    instead was the first version's mistake and it undid the whole feature:
    the rider saw 153 and 143 side by side under a merged verdict and read the
    old contradiction back into it. The raw number is kept as
    ``knee_at_bdc_alone``, which is a different claim and is labelled as one.
    """
    summary = result.get("sport_specific_metrics") or {}
    geom = result.get("bilateral_geometry") or {}
    raw = summary.get("knee_at_bdc")
    return {
        "camera_side": _side_of(result),
        "knee_at_bdc": merged_knee if merged_knee is not None else raw,
        "knee_at_bdc_alone": raw,
        "merged": merged_knee is not None,
        "trunk_angle_avg": summary.get("trunk_angle_avg"),
        "frames_analyzed": result.get("frames_analyzed"),
        "revolutions": geom.get("revolutions"),
        "quality_warnings": _warnings_of(result)[:3],
        "keyframe_base64": result.get("keyframe_base64"),
        "has_keyframe": bool(result.get("keyframe_base64")),
    }


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
    geom_left = SideGeometry.from_dict(left.get("bilateral_geometry"))
    geom_right = SideGeometry.from_dict(right.get("bilateral_geometry"))
    if geom_left is None or geom_right is None:
        # One clip was too disturbed to reduce -- most often the tracker never
        # held the ankle on the pedal path long enough. Nothing to merge.
        return _refusal(left, right, "geometry_unavailable")

    fit = combine_sides(geom_left, geom_right)
    agreement = midline_agreement(
        left.get("sport_specific_metrics") or {},
        right.get("sport_specific_metrics") or {},
    )
    if not fit.combined:
        return _refusal(left, right, fit.reason or "not_combinable",
                        fit=fit, agreement=agreement)

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
        "sides": [_side_card(left, fit.per_side.get("left")),
                  _side_card(right, fit.per_side.get("right"))],
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
        "scale_disagreement_pct": (
            None if fit is None else fit.as_dict().get("scale_disagreement_pct")
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
