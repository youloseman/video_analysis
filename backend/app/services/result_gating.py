"""Free-tier teaser gating: trim a full analysis result down to what a starter
(or anonymous) caller is allowed to see.

Single source of truth for the paywall. The backend never ships locked content
to a free caller -- the client only ever renders a blurred *placeholder*, not
real data hidden with CSS. Paid tiers (enthusiast/full/admin) pass through
untouched.

The trimmed payload keeps enough to make the value obvious (score, grade, the
annotated keyframe with the skeleton) and replaces the paid parts with a
``locked`` marker the frontend turns into an upgrade prompt.
"""

from __future__ import annotations

from typing import Any

from app.models.user import User, TIER_STARTER

# ALLOWLIST (not denylist): only these keys survive for a free caller. An
# allowlist is the safe default -- a new paid field added later is withheld by
# default rather than leaking. Two result shapes share this module:
#   video (runner.run_analysis) and photo (photo_analyzer.analyze_photo).
_SAFE_KEYS = frozenset({
    # identity / status -- harmless, needed to render the card
    "status", "sport_type", "sport", "cycling_position",
    "cycling_position_label", "camera_side", "frames_analyzed",
    "processing_time_seconds",
    # headline score + grade (the hook that makes them want the detail)
    "technique_score", "letter_grade", "score",
    # the annotated keyframe -- rendered number-free + watermarked upstream
    "keyframe_base64", "thumbnail_base64",
    # whether the analyzer distrusted its own measurement (see _quality_block)
    "quality_gate_triggered",
})

# Fields lifted out of the (paid) ``sport_specific_metrics`` blob into a compact
# ``quality`` block that free callers DO get.
#
# This is not a paid feature being given away -- it is the caveat attached to the
# number we already show them. A free result is a score with no angles and no
# coaching to argue with, so if the analyzer only half-tracked the athlete, or
# the clip was shot from the wrong angle, the headline score is the ONLY thing
# they see and they have no way to tell it is unreliable. Withholding that turns
# a "we couldn't measure this properly" into a confident-looking verdict.
#
# Everything here is explicitly named: nothing is copied wholesale out of
# ``sport_specific_metrics``, so a paid field added to that dict later cannot
# ride along.
_QUALITY_KEYS = ("quality_warnings", "time_base_uncertain", "sampling_degraded")

# Where each result shape keeps its capture warnings. The video path files them
# under ``sport_specific_metrics["quality_warnings"]``; the photo path returns a
# top-level ``warnings`` list. Reading only the video location dropped every
# photo warning here -- and the client's "Clean side-view capture." all-clear is
# an ELSE branch, so a free caller was actively told the opposite of the truth.
_PHOTO_WARNINGS_KEY = "warnings"

# What an upgrade unlocks (shown by the frontend on the blurred sections).
_UNLOCKS = ["coaching", "angles", "issues", "ranges", "video", "second_phase"]


def is_free(user: User | None) -> bool:
    """Free = anonymous, or signed in on the starter tier."""
    return user is None or user.tier == TIER_STARTER


def gate_result_for_tier(result: dict[str, Any], user: User | None) -> dict[str, Any]:
    """Return the result the caller is allowed to see.

    Paid tiers get the full object unchanged. Free callers get score + grade +
    the (number-free, watermarked) keyframe, with paid fields removed and a
    ``locked`` block describing what an upgrade unlocks.
    """
    if not is_free(user):
        return result
    return gate_free_result(result)


def quality_block(result: dict[str, Any]) -> dict[str, Any]:
    """Build the caveat that travels with a free score.

    Reads from the paid ``sport_specific_metrics`` blob but copies only the
    named quality fields out of it -- never the blob itself. ``confidence`` is
    reduced to its level (``high``/``medium``/``low``); the per-factor
    breakdown behind it stays paid.

    ``warnings`` merges both result shapes (see ``_PHOTO_WARNINGS_KEY``) so the
    client has one place to look regardless of whether it rendered a clip or a
    still. A cycling photo also files its MEDICAL warnings there (closed hip ->
    iliac-artery risk); one of those strings quotes the measured trunk angle,
    which is otherwise a paid number. That is a deliberate trade: a paywall is
    not a reason to withhold "this position carries a documented injury risk".
    """
    metrics = result.get("sport_specific_metrics") or {}
    gate = metrics.get("quality_gate") or {}
    confidence = metrics.get("analysis_confidence") or {}

    block: dict[str, Any] = {
        "triggered": bool(result.get("quality_gate_triggered")),
        "reasons": list(gate.get("reasons") or []),
        "warnings": [str(w) for w in (result.get(_PHOTO_WARNINGS_KEY) or [])],
    }
    for key in _QUALITY_KEYS:
        value = metrics.get(key)
        if key == "quality_warnings":
            block["warnings"].extend(str(w) for w in (value or []))
        elif value:
            block[key] = value
    level = confidence.get("level")
    if level:
        block["confidence"] = level
    return block


def gate_free_result(result: dict[str, Any]) -> dict[str, Any]:
    """Trim a result to the free teaser payload (caller already known free).

    Keeps only allowlisted keys (works for both video + photo result shapes),
    forces the overlay video off, attaches the ``quality`` caveat block, and
    adds the ``locked`` paywall marker.
    """
    kept: dict[str, Any] = {k: v for k, v in result.items() if k in _SAFE_KEYS}
    kept["overlay_video_path"] = None  # never a video for free
    kept["quality"] = quality_block(result)
    kept["locked"] = {"reason": "starter", "unlocks": list(_UNLOCKS)}
    return kept
