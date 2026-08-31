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
    # Why there is no score, when there is no score. A free result IS the
    # number, so withholding it without the reason leaves an empty card that
    # reads as a broken page rather than as a deliberate refusal -- and the
    # one thing that reader most needs is the sentence explaining how to get
    # a clip we can measure. Same reasoning as _QUALITY_KEYS below.
    "score_withheld",
    # how much of the rubric that score was actually computed from. Same
    # reasoning as _QUALITY_KEYS below: the free result IS the number, so a
    # score built on five of nine measures must not read as a complete one.
    # It names measures, never their values -- no paid number rides along.
    "score_coverage",
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
#
# ``capture_report`` belongs here for the same reason and then some: it is the
# only part of a free result that tells the reader how to get a BETTER one. A
# paywall in front of "your clip was filmed too far away, here is the fix" would
# be charging admission to the instructions.
_QUALITY_KEYS = (
    "quality_warnings", "time_base_uncertain", "sampling_degraded",
    "capture_report",
)

# Where each result shape keeps its capture warnings. The video path files them
# under ``sport_specific_metrics["quality_warnings"]``; the photo path returns a
# top-level ``warnings`` list. Reading only the video location dropped every
# photo warning here -- and the client's "Clean side-view capture." all-clear is
# an ELSE branch, so a free caller was actively told the opposite of the truth.
_PHOTO_WARNINGS_KEY = "warnings"

# What an upgrade unlocks (shown by the frontend on the blurred sections).
_UNLOCKS = [
    "coaching", "angles", "issues", "ranges", "kinogram", "video",
    "second_phase", "export", "fit",
]

# --------------------------------------------------------------------------
# The preview: one report, once per account, that is worth reading.
#
# The teaser above shows a score and nothing else. That demonstrates the
# product *exists*; it does not demonstrate that the hidden part is worth $9,
# because the reader has no idea what is behind the blur. Somebody who has once
# seen the whole thing is deciding whether to get it back -- a much stronger
# position than deciding whether to gamble on it.
#
# So the first analysis on a free account keeps the two sections that explain
# the score in words -- what we found, and what a coach makes of it -- and holds
# back the two that are the ongoing product: the measurement table, and the
# drills that say what to actually do about it.
#
# ``detected_issues`` quote the measurement they fired on ("18.4 deg"), so
# showing findings does leak a handful of numbers. That is deliberate: a finding
# with the number stripped out reads as an accusation with no evidence, and the
# paid product is the *complete* table plus the plan, not the three numbers that
# happen to be broken.
_PREVIEW_EXTRA = ("detected_issues", "ai_recommendations")

# What is still behind the paywall after a preview. Note ``coaching`` and
# ``issues`` are absent -- they were just shown, and listing them would promise
# the reader something they already have.
_PREVIEW_UNLOCKS = [
    "plan", "fit", "angles", "ranges", "kinogram", "video", "second_phase",
    "export",
]

ACCESS_FULL = "full"
ACCESS_PREVIEW = "preview"
ACCESS_TEASER = "teaser"


def is_free(user: User | None) -> bool:
    """Free = anonymous, or signed in on the starter tier."""
    return user is None or user.tier == TIER_STARTER


def access_for(user: User | None, preview: bool = False) -> str:
    """How much of a result this caller may see.

    An anonymous caller never gets the preview, whatever the flag says. The
    preview is spent from an account -- there is nothing to charge it against
    here, so granting it would be an unlimited preview for anyone who simply
    does not sign in. Analysis requires an account today, so this is a guard on
    a door rather than a live path; it is also the exact door that would open
    quietly if anonymous analysis ever came back.
    """
    if user is None:
        return ACCESS_TEASER
    if not is_free(user):
        return ACCESS_FULL
    return ACCESS_PREVIEW if preview else ACCESS_TEASER


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


def _trim(
    result: dict[str, Any], keys: Any, reason: str, unlocks: list[str],
) -> dict[str, Any]:
    """Allowlist a result down to ``keys`` and attach the paywall markers."""
    kept: dict[str, Any] = {k: v for k, v in result.items() if k in keys}
    kept["overlay_video_path"] = None  # never a video on a free plan
    kept["quality"] = quality_block(result)
    kept["locked"] = {"reason": reason, "unlocks": list(unlocks)}
    return kept


def gate_free_result(result: dict[str, Any]) -> dict[str, Any]:
    """Trim a result to the free teaser payload (caller already known free).

    Keeps only allowlisted keys (works for both video + photo result shapes),
    forces the overlay video off, attaches the ``quality`` caveat block, and
    adds the ``locked`` paywall marker.
    """
    return _trim(result, _SAFE_KEYS, "starter", _UNLOCKS)


def gate_preview_result(result: dict[str, Any]) -> dict[str, Any]:
    """The one report a free account gets in full-enough form to judge us by.

    Same allowlist discipline as the teaser -- a paid field added later is
    withheld by default rather than joining the preview by accident -- with the
    two explanatory sections added on top.
    """
    kept = _trim(
        result, _SAFE_KEYS.union(_PREVIEW_EXTRA), "preview", _PREVIEW_UNLOCKS,
    )
    # Count what is behind the blur, so the upgrade prompt can say "3 drills and
    # 7 measured angles" instead of "more". The reader has just been shown what
    # our writing is worth; a vague promise is a strange thing to follow it with.
    plan = (result.get("training_plan") or {}).get("top_3_priorities") or []
    kept["locked"]["counts"] = {
        "plan": sum(1 for p in plan if isinstance(p, dict) and p.get("drill")),
        "angles": len(result.get("angle_statistics") or {}),
    }
    return kept


def access_for_stored(user: User | None, row: Any) -> str:
    """Access level for a saved analysis, which can be bought a la carte.

    Order matters: the tier is checked first, so subscribing opens everything
    the account ever ran -- including reports it looked at months ago through a
    teaser. That retroactive opening is most of why the full result is stored
    at all, and it is a better argument for a plan than any wording on the
    pricing page.
    """
    if user is not None and not is_free(user):
        return ACCESS_FULL
    if getattr(row, "unlocked_at_ms", None):
        return ACCESS_FULL
    if getattr(row, "preview", False):
        return ACCESS_PREVIEW
    return ACCESS_TEASER


def gate_for_access(result: dict[str, Any], access: str) -> dict[str, Any]:
    """Dispatch on the access level from ``access_for``."""
    if access == ACCESS_FULL:
        return result
    if access == ACCESS_PREVIEW:
        return gate_preview_result(result)
    return gate_free_result(result)
