"""Swimming: a raw clip that a coach reviews by eye. No pose analysis.

Why no numbers: on an underwater freestyle frame (probe 2026-10-28) MediaPipe
heavy put the swimmer's head on the buttocks with 0.96-1.00 visibility --
confidently wrong, the one failure no gate of ours catches -- and rotating the
frame upright fixed the orientation but lost the legs and the extended arm.
Off-the-shelf models are trained on people standing on land. So a swim clip
is stored, never measured, and sold only as Coach Notes or an Expert Review
that Bogdan writes from the footage (Artur, 2026-10-28; same prices as run and
bike). Automation is a later project, from consented, labelled clips.

The clip lives exactly where a run or bike upload does (``uploads/<job>/
input.*``), and its history row carries the job id, so retention, the review
queue's player and the coach's brief find it with no special case.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.analysis import Analysis
from app.models.order import ORDER_REFUNDED, Order

SPORT_SWIM = "swim"

STROKES = {
    "freestyle": "Freestyle",
    "breaststroke": "Breaststroke",
    "backstroke": "Backstroke",
    "butterfly": "Butterfly",
}
POOLS = {
    "pool_25m": "25 m pool",
    "pool_50m": "50 m pool",
    "pool_25y": "25 yd pool",
    "endless": "Endless pool / flume",
}
VIEWS = {
    "underwater_side": "Underwater, from the side",
    "underwater_front": "Underwater, from the front",
    "deck_side": "From the deck, side-on",
}
DEFAULT_STROKE = "freestyle"
DEFAULT_POOL = "pool_25m"
DEFAULT_VIEW = "underwater_side"

# Clips stored without a review bought for them. A swim upload costs nothing
# and is never analysed, so without a cap it is free video hosting; three is
# room to re-film twice before buying.
MAX_UNORDERED_CLIPS = 3

QUESTION_MAX_CHARS = 200

# The same names in Russian, for the coach's brief.
STROKES_RU = {
    "freestyle": "кроль", "breaststroke": "брасс",
    "backstroke": "на спине", "butterfly": "баттерфляй",
}
POOLS_RU = {
    "pool_25m": "бассейн 25 м", "pool_50m": "бассейн 50 м",
    "pool_25y": "бассейн 25 ярдов", "endless": "бассейн с противотоком",
}
VIEWS_RU = {
    "underwater_side": "под водой, сбоку",
    "underwater_front": "под водой, спереди",
    "deck_side": "с бортика, сбоку",
}


def is_swim(sport: Any) -> bool:
    return sport == SPORT_SWIM


def pick(raw: str | None, allowed: dict[str, str], default: str) -> str:
    """A form choice, or the default when it is missing. An unknown value is
    the caller's mistake and is refused by the endpoint, not guessed here."""
    return (raw or "").strip() or default


def clean_question(raw: str | None) -> str | None:
    if not raw:
        return None
    text = " ".join(str(raw).split())
    return text[:QUESTION_MAX_CHARS] or None


def build_entry(
    *, client_id: str, at_ms: int, job_id: str, stroke: str, pool: str,
    view: str, question: str | None, consent_training: bool, filename: str | None,
) -> dict[str, Any]:
    """The history entry for one swim clip, in the shape the app stores.

    ``focus`` is the field every other entry uses for the athlete's question,
    so the review queue and the brief read it with no special case.
    """
    return {
        "id": client_id,
        "at": at_ms,
        "jobId": job_id,
        "sport": SPORT_SWIM,
        "kind": "video",
        "score": None,
        "focus": question,
        "swim": {
            "stroke": stroke,
            "pool": pool,
            "view": view,
            # Opt-in, unticked by default, and only ever for improving the
            # analysis: without it the clip is seen by the coach on this order
            # and nobody else (privacy policy).
            "consent_training": bool(consent_training),
            "filename": (filename or "")[:120] or None,
        },
    }


async def unordered_clips(db: AsyncSession, user_id: int) -> int:
    """Swim clips on this account with no live order attached."""
    ordered = (
        select(Order.analysis_client_id)
        .where(
            Order.user_id == user_id,
            Order.analysis_client_id.is_not(None),
            Order.status != ORDER_REFUNDED,
        )
    )
    return int(
        await db.scalar(
            select(func.count(Analysis.id)).where(
                Analysis.user_id == user_id,
                Analysis.sport == SPORT_SWIM,
                Analysis.client_id.not_in(ordered),
            )
        ) or 0
    )
