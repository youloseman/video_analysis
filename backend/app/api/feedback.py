"""Micro-feedback on analysis results: submit (anyone) + read (admin only).

Why this exists: the analysis pipeline is a black box to the athlete -- they
cannot verify a pose estimate or a 0-100 score. The one signal we cannot get any
other way is "this result does not match what I see", attached to the machine
context that produced it. That turns the guesswork out of tuning: the 👎 rate
sliced by ``quality_gate_triggered`` / confidence tier / sport says where the
pipeline actually misfires.

Deliberately a private channel, not a public board -- nothing here is exposed to
other users, and submitting creates no obligation to reply. See
``/admin/feedback`` + ``/admin/feedback/stats`` for the read side (admin tier).

Endpoints
    POST  /feedback                -> submit or amend a rating (auth optional)
    GET   /admin/feedback          -> list rows without keyframes (admin)
    GET   /admin/feedback/stats    -> aggregates for tuning (admin)
    GET   /admin/feedback/{id}     -> one row incl. its keyframe (admin)
    PATCH /admin/feedback/{id}     -> triage: status + private note (admin)
"""

from __future__ import annotations

import json
import re
import time
from collections import deque
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_session
from app.core.net import client_ip, ip_hash
from app.core.security import optional_user, require_admin
from app.models.feedback import (
    KIND_ANALYSIS,
    STATUS_NEW,
    VALID_STATUSES,
    Feedback,
)
from app.models.user import User

logger = structlog.get_logger()

router = APIRouter(tags=["feedback"])

MAX_MESSAGE_CHARS = 2000
MAX_NOTE_CHARS = 4000
MAX_TAGS = 8
MAX_TAG_CHARS = 32
# ~300 KB of base64 -- a 720px-wide annotated JPEG lands well under this.
MAX_KEYFRAME_CHARS = 420_000
# Serialized ``context`` cap. Oversized context is dropped, never a reason to
# reject the rating itself (the rating is the valuable part).
MAX_CONTEXT_CHARS = 8000
MAX_CONTEXT_KEYS = 48

# Abuse guard: submissions per client per rolling 24h. In-memory, single-instance
# (same tradeoff as the analysis limiter in app/main.py).
_LIMIT_PER_DAY = 40
_WINDOW_S = 24 * 3600
_hits: dict[str, deque] = {}

_TAG_RE = re.compile(r"[^a-z0-9_-]+")


def _over_limit(key: str) -> bool:
    now = time.time()
    dq = _hits.setdefault(key, deque())
    while dq and now - dq[0] > _WINDOW_S:
        dq.popleft()
    if len(dq) >= _LIMIT_PER_DAY:
        return True
    dq.append(now)
    return False


def _clean_tags(raw: list[Any] | None) -> list[str]:
    """Slugify to ``[a-z0-9_-]`` so aggregation stays clean, but keep unknown
    tags: whitelisting here would silently drop reasons added by a newer client
    against an older server."""
    out: list[str] = []
    for item in (raw or [])[:MAX_TAGS]:
        tag = _TAG_RE.sub("_", str(item).strip().lower()).strip("_")[:MAX_TAG_CHARS]
        if tag and tag not in out:
            out.append(tag)
    return out


def _clean_context(raw: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(raw, dict) or not raw:
        return {}
    ctx = dict(list(raw.items())[:MAX_CONTEXT_KEYS])
    try:
        if len(json.dumps(ctx)) > MAX_CONTEXT_CHARS:
            return {"_truncated": True}
    except (TypeError, ValueError):
        return {}
    return ctx


# --------------------------------------------------------------------------
# Submit
# --------------------------------------------------------------------------
class FeedbackIn(BaseModel):
    # +1 "looks right" / -1 "something's off" / None = comment with no verdict.
    rating: int | None = None
    tags: list[str] = Field(default_factory=list)
    message: str | None = None
    analysis_client_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    # Annotated keyframe (data URI). Sent only when the user ticks the consent
    # box in the widget -- the client omits it otherwise.
    keyframe: str | None = None


@router.post("/feedback", status_code=status.HTTP_201_CREATED)
async def submit_feedback(
    body: FeedbackIn,
    request: Request,
    user: User | None = Depends(optional_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Record one rating of an analysis result.

    Amends in place when the same author re-rates the same analysis (👍 then 👎,
    or adding a comment after the thumb) so one result yields one row instead of
    a trail of contradictory ones.
    """
    if body.rating not in (None, 1, -1):
        raise HTTPException(400, "rating must be 1, -1 or null")

    tags = _clean_tags(body.tags)
    message = (body.message or "").strip()[:MAX_MESSAGE_CHARS] or None
    if body.rating is None and not message and not tags:
        raise HTTPException(400, "nothing to record")

    ihash = ip_hash(client_ip(request))
    if _over_limit(f"u{user.id}" if user else ihash):
        raise HTTPException(429, "Too much feedback from this client today.")

    keyframe = body.keyframe or None
    if keyframe and (
        not keyframe.startswith("data:image/") or len(keyframe) > MAX_KEYFRAME_CHARS
    ):
        keyframe = None  # oversized / not an image -> drop it, keep the rating

    cid = (body.analysis_client_id or "").strip()[:64] or None
    row: Feedback | None = None
    if cid:
        owner = (
            Feedback.user_id == user.id if user else Feedback.ip_hash == ihash
        )
        row = (
            await db.execute(
                select(Feedback).where(Feedback.analysis_client_id == cid, owner)
            )
        ).scalars().first()

    ctx = _clean_context(body.context)
    now_ms = int(time.time() * 1000)
    if row is not None:
        row.created_at_ms = now_ms
        row.rating = body.rating
        row.tags = tags
        row.message = message
        row.context = ctx or row.context
        if keyframe:
            row.keyframe = keyframe
        row.status = STATUS_NEW  # an amended rating needs looking at again
    else:
        row = Feedback(
            created_at_ms=now_ms,
            user_id=user.id if user else None,
            email=user.email if user else None,
            ip_hash=ihash,
            kind=KIND_ANALYSIS,
            rating=body.rating,
            tags=tags,
            message=message,
            analysis_client_id=cid,
            context=ctx,
            keyframe=keyframe,
            status=STATUS_NEW,
        )
        db.add(row)
    await db.commit()
    await db.refresh(row)
    logger.info(
        "FEEDBACK", id=row.id, rating=row.rating, tags=tags,
        sport=ctx.get("sport"), kind=ctx.get("kind"), gated=ctx.get("gated"),
        score=ctx.get("score"), has_message=bool(message), frame=bool(keyframe),
        user_id=row.user_id,
    )
    return {"ok": True, "id": row.id}


# --------------------------------------------------------------------------
# Read (admin)
# --------------------------------------------------------------------------
# Columns for the list view: everything except the keyframe blob.
_LIST_COLS = (
    Feedback.id, Feedback.created_at_ms, Feedback.user_id, Feedback.email,
    Feedback.kind, Feedback.rating, Feedback.tags, Feedback.message,
    Feedback.analysis_client_id, Feedback.context, Feedback.status,
    Feedback.admin_note,
)


def _row_dict(r: Any, *, has_keyframe: bool | None = None) -> dict[str, Any]:
    out = {
        "id": r.id, "at": r.created_at_ms, "user_id": r.user_id,
        "email": r.email, "kind": r.kind, "rating": r.rating,
        "tags": r.tags or [], "message": r.message,
        "analysis_client_id": r.analysis_client_id, "context": r.context or {},
        "status": r.status, "admin_note": r.admin_note,
    }
    if has_keyframe is not None:
        out["has_keyframe"] = has_keyframe
    return out


@router.get("/admin/feedback")
async def list_feedback(
    status_filter: str | None = None,
    rating: int | None = None,
    limit: int = 100,
    offset: int = 0,
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Newest-first feedback, keyframes excluded (fetch one via the detail route)."""
    q = select(*_LIST_COLS, (Feedback.keyframe.isnot(None)).label("has_kf"))
    if status_filter in VALID_STATUSES:
        q = q.where(Feedback.status == status_filter)
    if rating in (1, -1):
        q = q.where(Feedback.rating == rating)
    q = q.order_by(Feedback.created_at_ms.desc()).limit(
        max(1, min(limit, 500))
    ).offset(max(0, offset))
    rows = (await db.execute(q)).all()
    total = (await db.execute(select(func.count(Feedback.id)))).scalar_one()
    return {
        "total": total,
        "items": [_row_dict(r, has_keyframe=bool(r.has_kf)) for r in rows],
    }


# Declared before /admin/feedback/{fid} so "stats" is not parsed as an id.
@router.get("/admin/feedback/stats")
async def feedback_stats(
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Aggregates that answer "where does the pipeline misfire?".

    The headline numbers are the negative rates sliced by the pipeline's own
    self-assessment (quality gate, confidence tier): if 👎 clusters on runs the
    gate already flagged, the gate works and the copy needs to be louder; if it
    clusters on clean, high-confidence runs, the scoring is wrong.
    """
    rows = (
        await db.execute(
            select(
                Feedback.rating, Feedback.tags, Feedback.context,
                Feedback.created_at_ms, Feedback.status,
                (Feedback.message.isnot(None)).label("has_message"),
            )
        )
    ).all()

    now_ms = int(time.time() * 1000)
    day = 24 * 3600 * 1000

    def _blank() -> dict[str, int]:
        return {"up": 0, "down": 0}

    def _bump(bucket: dict[str, dict[str, int]], key: str, rating: int | None) -> None:
        if rating not in (1, -1):
            return
        b = bucket.setdefault(str(key), _blank())
        b["up" if rating == 1 else "down"] += 1

    total = up = down = comments = new = 0
    last_7 = last_30 = 0
    by_sport: dict[str, dict[str, int]] = {}
    by_kind: dict[str, dict[str, int]] = {}
    by_gate: dict[str, dict[str, int]] = {}
    by_confidence: dict[str, dict[str, int]] = {}
    by_position: dict[str, dict[str, int]] = {}
    tag_counts: dict[str, int] = {}

    for r in rows:
        total += 1
        ctx = r.context or {}
        if r.rating == 1:
            up += 1
        elif r.rating == -1:
            down += 1
        if r.status == STATUS_NEW:
            new += 1
        age = now_ms - (r.created_at_ms or 0)
        if age <= 7 * day:
            last_7 += 1
        if age <= 30 * day:
            last_30 += 1
        for tag in r.tags or []:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        if r.has_message:
            comments += 1
        _bump(by_sport, ctx.get("sport") or "unknown", r.rating)
        _bump(by_kind, ctx.get("kind") or "unknown", r.rating)
        _bump(by_gate, "gated" if ctx.get("gated") else "clean", r.rating)
        _bump(by_confidence, ctx.get("confidence") or "unknown", r.rating)
        if (ctx.get("sport") or "") == "bike":
            _bump(by_position, ctx.get("position") or "unknown", r.rating)

    def _with_rate(bucket: dict[str, dict[str, int]]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for k, v in bucket.items():
            n = v["up"] + v["down"]
            out[k] = {**v, "n": n,
                      "down_rate": round(v["down"] / n, 3) if n else None}
        return out

    rated = up + down
    return {
        "total": total,
        "rated": rated,
        "up": up,
        "down": down,
        "down_rate": round(down / rated, 3) if rated else None,
        "with_message": comments,
        "untriaged": new,
        "last_7d": last_7,
        "last_30d": last_30,
        "top_tags": dict(
            sorted(tag_counts.items(), key=lambda kv: -kv[1])[:12]
        ),
        "by_sport": _with_rate(by_sport),
        "by_kind": _with_rate(by_kind),
        "by_quality_gate": _with_rate(by_gate),
        "by_confidence": _with_rate(by_confidence),
        "by_bike_position": _with_rate(by_position),
    }


@router.get("/admin/feedback/{fid}")
async def get_feedback(
    fid: int,
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """One row with its annotated keyframe -- the actual debugging view."""
    row = await db.get(Feedback, fid)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
    out = _row_dict(row)
    out["keyframe"] = row.keyframe
    return out


class TriageIn(BaseModel):
    status: str | None = None
    admin_note: str | None = None


@router.patch("/admin/feedback/{fid}")
async def triage_feedback(
    fid: int,
    body: TriageIn,
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    row = await db.get(Feedback, fid)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
    if body.status is not None:
        if body.status not in VALID_STATUSES:
            raise HTTPException(400, f"status must be one of {list(VALID_STATUSES)}")
        row.status = body.status
    if body.admin_note is not None:
        row.admin_note = body.admin_note.strip()[:MAX_NOTE_CHARS] or None
    await db.commit()
    return {"ok": True, "id": row.id, "status": row.status}
