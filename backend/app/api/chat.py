"""``/me/analyses/{client_id}/chat`` -- ask about this report.

See services/report_chat.py for what the model is given and held to. This is
the gate around it:

* **Who may ask.** A paid plan (report_chat.CHAT_LIMITS), reading a report it
  sees in full. Access is the report owner's, exactly as for the export: a
  coach reading a linked athlete's report sees it at the athlete's level, and
  asks on the coach's own allowance.
* **What a question costs.** One of the month's questions, counted only when
  an answer came back. A failed call stores nothing and charges nothing. A
  second, hidden ceiling holds what the month's answers cost us under a tenth
  of the plan's price (report_chat.CHAT_BUDGET_USD).
* **Nothing is analysed again.** The stored result is the context; the clip is
  never touched.
"""

from __future__ import annotations

import time
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_session
from app.core.security import get_current_user, require_admin
from app.models.analysis import Analysis
from app.models.cda import CdaMeasurement
from app.models.chat import MAX_QUESTION_CHARS, ROLE_ASSISTANT, ROLE_USER, ChatMessage
from app.models.profile import Profile
from app.models.user import User
from app.services import report_chat
from app.services.result_gating import ACCESS_FULL, access_for_stored

logger = structlog.get_logger()
router = APIRouter(prefix="/me", tags=["chat"])
admin_router = APIRouter(tags=["chat"])

OUT_OF_BUDGET = ("You have used this month's questions. They reset on the 1st; "
                 "for a judgement call, a Flapp coach can review this clip.")


class QuestionIn(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS * 2)

    @field_validator("question")
    @classmethod
    def _q(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Ask a question.")
        if len(v) > MAX_QUESTION_CHARS:
            raise ValueError(f"Keep a question under {MAX_QUESTION_CHARS} characters.")
        return v


def _quota(used: int, limit: int) -> dict[str, Any]:
    return {"used": used, "limit": limit, "resets_at": report_chat.resets_at()}


async def _readable(
    db: AsyncSession, user: User, client_id: str, athlete_id: int | None,
) -> tuple[User, Analysis]:
    """The report, if this reader may chat about it. 402 / 409 otherwise."""
    from app.api.me import _row_for_reader

    state = report_chat.chat_state(user)
    if state != "open":
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=(
                "Asking about a report is part of the Full plan." if state == "full"
                else "Asking about a report is part of the paid plans."
            ),
        )
    owner, row = await _row_for_reader(db, user, client_id, athlete_id)
    if access_for_stored(owner, row) != ACCESS_FULL:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Questions open once this report is unlocked.",
        )
    if not row.result:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This analysis was run before full reports were stored.",
        )
    return owner, row


@router.get("/analyses/{client_id}/chat")
async def get_chat(
    client_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
    athlete_id: int | None = None,
) -> dict[str, Any]:
    owner, row = await _readable(db, user, client_id, athlete_id)
    msgs = await report_chat.thread(db, user.id, owner.id, client_id)
    used = await report_chat.used_this_month(db, user.id)
    limit = report_chat.chat_limit(user.tier)
    # Out of dollars before out of questions: say so as "none left" rather
    # than offer a box that will refuse.
    if await report_chat.spent_this_month(db, user.id) >= report_chat.chat_budget(user.tier):
        used = max(used, limit)
    return {
        "messages": [report_chat.message_out(m) for m in msgs],
        "suggestions": report_chat.suggestions(
            row.result or {},
            has_cda=await _has_cda(db, row),
            has_previous=await _has_previous(db, owner, row),
        ),
        **_quota(used, limit),
    }


async def _has_cda(db: AsyncSession, row: Analysis) -> bool:
    if row.profile_id is None:
        return False
    n = await db.scalar(select(func.count(CdaMeasurement.id)).where(CdaMeasurement.profile_id == row.profile_id))
    return bool(n)


async def _has_previous(db: AsyncSession, owner: User, row: Analysis) -> bool:
    if row.profile_id is None:
        return False
    n = await db.scalar(select(func.count(Analysis.id)).where(
        Analysis.user_id == owner.id, Analysis.profile_id == row.profile_id,
        Analysis.client_id != row.client_id, Analysis.created_at_ms < row.created_at_ms,
    ))
    return bool(n)


async def _context(db: AsyncSession, owner: User, row: Analysis, reader: User) -> dict[str, str | None]:
    from app.api.cda import rows_by_profile
    from app.services.export import ai_export

    result = row.result or {}
    entry = row.data or {}
    sport = str(result.get("sport_type") or row.sport or "run")
    report_md = ai_export.build_markdown(
        result, job_id=row.client_id,
        coach_notes=entry.get("coachNote"),
        coach_name=", ".join(x for x in (entry.get("coachName"), entry.get("coachBio")) if x),
    )

    profile = None
    cda: list[dict[str, Any]] = []
    if row.profile_id is not None:
        p = await db.get(Profile, row.profile_id)
        if p is not None and p.user_id == owner.id:
            profile = {
                "name": p.name, "kind": p.kind, "crank_length_mm": p.crank_length_mm,
                "bike_family": p.bike_family, "bike_model": p.bike_model,
                "frame_size": p.frame_size, "bike_year": p.bike_year,
                "bike_generation": p.bike_generation,
            }
            if sport == "bike":
                cda = (await rows_by_profile(db, [p], reader.id)).get(p.id, [])

    previous: list[dict[str, Any]] = []
    if row.profile_id is not None:
        prev_rows = (
            await db.execute(
                select(Analysis.created_at_ms, Analysis.score)
                .where(
                    Analysis.user_id == owner.id,
                    Analysis.profile_id == row.profile_id,
                    Analysis.client_id != row.client_id,
                    Analysis.created_at_ms < row.created_at_ms,
                )
                .order_by(Analysis.created_at_ms.desc())
                .limit(report_chat.PREVIOUS_CLIPS)
            )
        ).all()
        previous = [{"at": at, "score": sc} for at, sc in reversed(prev_rows)]

    measured_at = getattr(owner, "mobility_measured_at", None)
    athlete_md = report_chat.athlete_block(
        sport=sport,
        height_cm=owner.height_cm,
        mobility={
            "hamstring_deg": owner.mobility_hamstring_deg,
            "hip_flexion_deg": owner.mobility_hip_flexion_deg,
            "measured_on": measured_at.date().isoformat() if measured_at else None,
        },
        profile=profile, cda=cda, previous=previous,
    )
    from app.services import parts

    parts_md = parts.chat_block((profile or {}).get("bike_generation")) if sport == "bike" else None
    if parts_md:
        athlete_md = athlete_md + "\n\n" + parts_md
    return {
        "report_md": report_md,
        "athlete_md": athlete_md,
        "withheld_md": report_chat.withheld_block(result),
        "sport": sport,
    }


@router.post("/analyses/{client_id}/chat")
async def ask_chat(
    client_id: str,
    body: QuestionIn,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
    athlete_id: int | None = None,
) -> dict[str, Any]:
    owner, row = await _readable(db, user, client_id, athlete_id)
    limit = report_chat.chat_limit(user.tier)
    used = await report_chat.used_this_month(db, user.id)
    if used >= limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"You have asked all {limit} questions this month. They reset on the 1st.",
        )

    ctx = await _context(db, owner, row, user)
    history = await report_chat.thread(db, user.id, owner.id, client_id)
    prompt = report_chat.build_prompt(
        report_md=ctx["report_md"] or "", athlete_md=ctx["athlete_md"] or "",
        withheld_md=ctx["withheld_md"], history=history, question=body.question,
    )
    spent = await report_chat.spent_this_month(db, user.id)
    if spent + report_chat.projected_cost_usd(prompt) > report_chat.chat_budget(user.tier):
        logger.warning("CHAT_BUDGET_REACHED", user_id=user.id, tier=user.tier, spent_usd=round(spent, 4))
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=OUT_OF_BUDGET)
    answer = await run_in_threadpool(report_chat.ask, prompt)
    if not answer:
        logger.warning("CHAT_NO_ANSWER", user_id=user.id, analysis=client_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The assistant did not answer. Your question was not counted; try again in a minute.",
        )

    now_ms = int(time.time() * 1000)
    rows = report_chat.new_rows(
        reader_id=user.id, owner_id=owner.id, client_id=client_id,
        question=body.question, answer=answer, now_ms=now_ms, prompt=prompt,
    )
    db.add_all(rows)
    await db.commit()
    logger.info(
        "CHAT_ANSWERED", user_id=user.id, analysis=client_id, model=answer.get("model"),
        prompt_chars=len(prompt), answer_chars=len(answer["text"]),
        tokens_in=rows[1].tokens_in, tokens_out=rows[1].tokens_out,
        cost_usd=round(report_chat.answer_cost(rows[1]), 5),
    )
    try:
        from app.services import analytics

        # Lengths and the turn number, not the text: the questions live in our
        # own table, which is where they are read from.
        analytics.capture_bg(analytics.person_id(user), "chat_question", {
            "sport": ctx["sport"], "turn": len(history) // 2 + 1,
            "question_chars": len(body.question), "model": answer.get("model"),
            "withheld": bool(ctx["withheld_md"]), "coach_reading": owner.id != user.id,
            "tier": user.tier,
        })
    except Exception:  # noqa: BLE001 -- analytics never fails a request
        pass
    return {
        "answer": report_chat.message_out(rows[1]),
        "question": report_chat.message_out(rows[0]),
        **_quota(used + 1, limit),
    }


# --------------------------------------------------------------------------
# Admin: what the chat costs, this month
# --------------------------------------------------------------------------
@admin_router.get("/admin/chat/usage")
async def chat_usage(
    _admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Every reader who asked this month: questions, tokens, estimated cost
    against their plan's ceiling, and the totals. Estimated: see report_chat."""
    from app.services.usage_limits import window_start_ms

    start = window_start_ms(report_chat.CHAT_WINDOW)
    rows = (await db.execute(
        select(ChatMessage, User.email, User.tier)
        .join(User, User.id == ChatMessage.user_id)
        .where(ChatMessage.created_at_ms >= start)
    )).all()
    per: dict[int, dict[str, Any]] = {}
    models: dict[str, int] = {}
    for m, email, tier in rows:
        u = per.setdefault(m.user_id, {"email": email, "tier": tier, "questions": 0, "reports": set(),
                                       "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0})
        if m.role == ROLE_USER:
            u["questions"] += 1
            u["reports"].add(m.analysis_client_id)
        elif m.role == ROLE_ASSISTANT:
            u["tokens_in"] += m.tokens_in if m.tokens_in is not None else report_chat.LEGACY_TOKENS_IN
            u["tokens_out"] += m.tokens_out if m.tokens_out is not None else report_chat.estimate_tokens(m.text)
            u["cost_usd"] += report_chat.answer_cost(m)
            models[m.model or "?"] = models.get(m.model or "?", 0) + 1
    users = []
    for u in per.values():
        budget = report_chat.chat_budget(u["tier"])
        limit = report_chat.chat_limit(u["tier"])
        users.append({
            "email": u["email"], "tier": u["tier"], "questions": u["questions"], "limit": limit,
            "reports": len(u["reports"]), "tokens_in": u["tokens_in"], "tokens_out": u["tokens_out"],
            "cost_usd": round(u["cost_usd"], 4), "budget_usd": budget,
            "budget_used": round(u["cost_usd"] / budget, 3) if budget else None,
        })
    users.sort(key=lambda x: -x["cost_usd"])
    return {
        "since_ms": start,
        "resets_at": report_chat.resets_at(),
        "rollout": report_chat.rollout(),
        "limits": {t: {"questions": report_chat.chat_limit(t), "budget_usd": b}
                   for t, b in report_chat.CHAT_BUDGET_USD.items()},
        "totals": {
            "askers": len(users),
            "questions": sum(u["questions"] for u in users),
            "tokens_in": sum(u["tokens_in"] for u in users),
            "tokens_out": sum(u["tokens_out"] for u in users),
            "cost_usd": round(sum(u["cost_usd"] for u in users), 4),
        },
        "models": models,
        "users": users,
    }
