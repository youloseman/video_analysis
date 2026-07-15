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
    "processing_time_seconds", "warnings",
    # headline score + grade (the hook that makes them want the detail)
    "technique_score", "letter_grade", "score",
    # the annotated keyframe -- rendered number-free + watermarked upstream
    "keyframe_base64", "thumbnail_base64",
})

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


def gate_free_result(result: dict[str, Any]) -> dict[str, Any]:
    """Trim a result to the free teaser payload (caller already known free).

    Keeps only allowlisted keys (works for both video + photo result shapes),
    forces the overlay video off, and attaches the ``locked`` paywall marker.
    """
    kept: dict[str, Any] = {k: v for k, v in result.items() if k in _SAFE_KEYS}
    kept["overlay_video_path"] = None  # never a video for free
    kept["locked"] = {"reason": "starter", "unlocks": list(_UNLOCKS)}
    return kept
