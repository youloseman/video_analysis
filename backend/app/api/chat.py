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
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import SessionLocal, get_session
from app.core.security import get_current_user, require_admin
from app.models.analysis import Analysis
from app.models.cda import CdaMeasurement
from app.models.chat import MAX_QUESTION_CHARS, ROLE_ASSISTANT, ROLE_MEMORY, ROLE_USER, ChatMessage
from app.models.profile import Profile
from app.models.user import User
from app.services import chat_topics, report_chat
from app.services.result_gating import ACCESS_FULL, access_for_stored

logger = structlog.get_logger()
router = APIRouter(prefix="/me", tags=["chat"])
admin_router = APIRouter(tags=["chat"])

OUT_OF_BUDGET = ("You have used this month's questions. They reset on the 1st; "
                 "for a judgement call, a Flapp coach can review this clip.")


class QuestionIn(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS * 2)
    # Analytics only: typed, or one of the offered questions (and which).
    asked_from: Literal["typed", "suggestion"] = "typed"
    suggestion_i: int | None = Field(default=None, ge=0, le=9)

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
    if row.kind == "photo":
        # One frame, a differently shaped result (no technique_score, no
        # sport_type): the assistant read a scored photo as "withheld".
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Questions are answered about video reports.",
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
    mem = await report_chat.memory_of(db, user.id, owner.id, report_chat.memory_key(row.athlete_id))
    return {
        "messages": [report_chat.message_out(m) for m in msgs],
        "memory": mem.text if mem else None,
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
    dirs = report_chat.directions_block(result)
    if dirs:
        report_md = f"{report_md}\n\n{dirs}"
    return {
        "report_md": report_md,
        "athlete_md": athlete_md,
        "withheld_md": report_chat.withheld_block(result),
        "sport": sport,
    }


def _track(user: User, event: str, props: dict[str, Any]) -> None:
    try:
        from app.services import analytics

        analytics.capture_bg(analytics.person_id(user), event, props)
    except Exception:  # noqa: BLE001 -- analytics never fails a request
        pass


async def _update_memory(reader_id: int, owner_id: int, key: str, question: str, answer: str) -> None:
    """After the answer is out: the notes rewritten with this exchange. Its own
    session (the request's is closed by now); a failure only loses an update."""
    try:
        async with SessionLocal() as db:
            reader = await db.get(User, reader_id)
            if reader is None:
                return
            spent = await report_chat.spent_this_month(db, reader_id)
            if spent >= report_chat.chat_budget(reader.tier):
                return
            old = await report_chat.memory_of(db, reader_id, owner_id, key)
            got = await run_in_threadpool(report_chat.remember, old.text if old else None, question, answer)
            if not got or not got["text"] or (old and got["text"] == old.text):
                return
            m = ChatMessage(user_id=reader_id, owner_user_id=owner_id, analysis_client_id=key,
                            role=ROLE_MEMORY, text=got["text"], model=got["model"],
                            created_at_ms=int(time.time() * 1000),
                            tokens_in=got["tokens_in"], tokens_out=got["tokens_out"])
            db.add(m)
            await db.commit()
            _track(reader, "chat_memory_updated", {
                "chars": len(got["text"]), "first": old is None, "model": got["model"],
                "cost_usd": round(report_chat.answer_cost(m), 5),
            })
    except Exception as e:  # noqa: BLE001
        logger.warning("CHAT_MEMORY_FAILED", err=str(e)[:200])


@router.post("/analyses/{client_id}/chat")
async def ask_chat(
    client_id: str,
    body: QuestionIn,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
    athlete_id: int | None = None,
    # Injected by FastAPI; None when a test calls the route directly.
    background: BackgroundTasks = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    owner, row = await _readable(db, user, client_id, athlete_id)
    await _check_questions_left(user, db)
    ctx = await _context(db, owner, row, user)

    def build(history: list[ChatMessage], memory: str | None) -> str:
        return report_chat.build_prompt(
            report_md=ctx["report_md"] or "", athlete_md=ctx["athlete_md"] or "",
            withheld_md=ctx["withheld_md"], history=history, question=body.question, memory=memory,
        )

    return await _answer(
        db, user, owner, client_id, report_chat.memory_key(row.athlete_id), body, build,
        background, where="report", sport=ctx["sport"],
        position=(row.result or {}).get("cycling_position"), withheld=bool(ctx["withheld_md"]),
    )


async def _check_questions_left(user: User, db: AsyncSession) -> None:
    limit = report_chat.chat_limit(user.tier)
    used = await report_chat.used_this_month(db, user.id)
    if used >= limit:
        _track(user, "chat_limit_reached", {"kind": "questions", "tier": user.tier, "used": used, "limit": limit})
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"You have asked all {limit} questions this month. They reset on the 1st.",
        )


async def _answer(
    db: AsyncSession, user: User, owner: User, thread_key: str, mkey: str, body: QuestionIn,
    build: Any, background: BackgroundTasks | None, *, where: str, sport: str,
    position: str | None, withheld: bool,
) -> dict[str, Any]:
    """Everything after the context: the dollar ceiling, the call, the two
    rows, analytics and the memory update. Shared by every chat surface."""
    limit = report_chat.chat_limit(user.tier)
    used = await report_chat.used_this_month(db, user.id)
    topic = chat_topics.classify(body.question)
    history = await report_chat.thread(db, user.id, owner.id, thread_key)
    mem = await report_chat.memory_of(db, user.id, owner.id, mkey)
    prompt = build(history, mem.text if mem else None)
    spent = await report_chat.spent_this_month(db, user.id)
    budget = report_chat.chat_budget(user.tier)
    if spent + report_chat.projected_cost_usd(prompt) > budget:
        logger.warning("CHAT_BUDGET_REACHED", user_id=user.id, tier=user.tier, spent_usd=round(spent, 4))
        _track(user, "chat_limit_reached", {"kind": "budget", "tier": user.tier, "used": used, "limit": limit,
                                            "month_cost_usd": round(spent, 4), "budget_usd": budget, "where": where})
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=OUT_OF_BUDGET)
    answer = await run_in_threadpool(report_chat.ask, prompt)
    if not answer:
        logger.warning("CHAT_NO_ANSWER", user_id=user.id, analysis=thread_key)
        _track(user, "chat_failed", {"reason": "no_answer", "tier": user.tier, "topic": topic, "where": where})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The assistant did not answer. Your question was not counted; try again in a minute.",
        )

    now_ms = int(time.time() * 1000)
    rows = report_chat.new_rows(
        reader_id=user.id, owner_id=owner.id, client_id=thread_key,
        question=body.question, answer=answer, now_ms=now_ms, prompt=prompt,
    )
    db.add_all(rows)
    await db.commit()
    cost = report_chat.answer_cost(rows[1])
    logger.info(
        "CHAT_ANSWERED", user_id=user.id, analysis=thread_key, where=where, model=answer.get("model"),
        prompt_chars=len(prompt), answer_chars=len(answer["text"]),
        tokens_in=rows[1].tokens_in, tokens_out=rows[1].tokens_out, cost_usd=round(cost, 5),
    )
    # A topic, lengths and counts -- never the text: the questions live in our
    # own table (and on the admin screen), not in the analytics service.
    month_cost = spent + cost
    _track(user, "chat_question", {
        "where": where, "sport": sport, "position": position,
        "turn": len(history) // 2 + 1, "topic": topic, "lang": chat_topics.language(body.question),
        "asked_from": body.asked_from, "suggestion_i": body.suggestion_i,
        "question_chars": len(body.question), "answer_chars": len(answer["text"]),
        "model": answer.get("model"), "withheld": withheld,
        "coach_reading": owner.id != user.id, "tier": user.tier, "has_memory": mem is not None,
        "tokens_in": rows[1].tokens_in, "tokens_out": rows[1].tokens_out, "cost_usd": round(cost, 5),
        "used": used + 1, "limit": limit, "questions_left": max(0, limit - used - 1),
        "month_cost_usd": round(month_cost, 4), "budget_used": round(month_cost / budget, 3) if budget else None,
        "$set": {"chat_questions_month": used + 1, "chat_cost_month_usd": round(month_cost, 4),
                 "chat_budget_used": round(month_cost / budget, 3) if budget else None},
    })
    if background is not None and len(body.question) >= 12:
        background.add_task(_update_memory, user.id, owner.id, mkey, body.question, answer["text"])
    return {
        "answer": report_chat.message_out(rows[1]),
        "question": report_chat.message_out(rows[0]),
        **_quota(used + 1, limit),
    }


# --------------------------------------------------------------------------
# Progress: the same assistant over the series the Progress page shows
# --------------------------------------------------------------------------
MAX_PROGRESS_IDS = 12


class ProgressQuestionIn(QuestionIn):
    sport: Literal["bike", "run"]
    ids: list[str] = Field(min_length=2, max_length=MAX_PROGRESS_IDS)


async def _progress_rows(db: AsyncSession, user: User, sport: str, ids: list[str]) -> list[Analysis]:
    """The reader's own clips among ``ids``, readable in full, oldest first.
    A coach's linked athletes are not here: their clips belong to the athlete."""
    if report_chat.chat_state(user) != "open":
        raise HTTPException(status_code=status.HTTP_402_PAYMENT_REQUIRED,
                            detail="Asking about your progress is part of the paid plans.")
    got = (await db.execute(
        select(Analysis).where(Analysis.user_id == user.id, Analysis.client_id.in_(ids[:MAX_PROGRESS_IDS]))
        .order_by(Analysis.created_at_ms)
    )).scalars().all()
    rows = [r for r in got if r.result and r.kind != "photo"
            and str((r.result or {}).get("sport_type") or r.sport) == sport
            and access_for_stored(user, r) == ACCESS_FULL]
    if len(rows) < 2:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="Progress questions need two full reports of this sport.")
    return rows


def _day_of(r: Analysis) -> str:
    from datetime import datetime, timezone

    ms = (r.data or {}).get("filmedAt") or r.created_at_ms
    try:
        return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        return "?"


def _series(rows: list[Analysis]) -> dict[str, Any]:
    return report_chat.progress_series([(_day_of(r), r.result or {}) for r in rows])


@router.get("/progress/chat")
async def get_progress_chat(
    sport: Literal["bike", "run"],
    ids: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    rows = await _progress_rows(db, user, sport, [i for i in ids.split(",") if i])
    athlete_id = rows[-1].athlete_id
    key = report_chat.progress_key(sport, athlete_id)
    msgs = await report_chat.thread(db, user.id, user.id, key)
    used = await report_chat.used_this_month(db, user.id)
    limit = report_chat.chat_limit(user.tier)
    if await report_chat.spent_this_month(db, user.id) >= report_chat.chat_budget(user.tier):
        used = max(used, limit)
    mem = await report_chat.memory_of(db, user.id, user.id, report_chat.memory_key(athlete_id))
    return {
        "messages": [report_chat.message_out(m) for m in msgs],
        "memory": mem.text if mem else None,
        "suggestions": report_chat.progress_suggestions(_series(rows)),
        "clips": len(rows),
        **_quota(used, limit),
    }


@router.post("/progress/chat")
async def ask_progress_chat(
    body: ProgressQuestionIn,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
    background: BackgroundTasks = None,  # type: ignore[assignment]
) -> dict[str, Any]:
    rows = await _progress_rows(db, user, body.sport, body.ids)
    await _check_questions_left(user, db)
    latest = rows[-1]
    ctx = await _context(db, user, latest, user)
    series_md = report_chat.progress_block(_series(rows))

    def build(history: list[ChatMessage], memory: str | None) -> str:
        return report_chat.build_prompt(
            heading=report_chat.PROGRESS_RULES,
            report_md=series_md + "\n\n# THE LATEST REPORT IN FULL\n\n" + (ctx["report_md"] or ""),
            athlete_md=ctx["athlete_md"] or "", withheld_md=None, history=history,
            question=body.question, memory=memory,
        )

    return await _answer(
        db, user, user, report_chat.progress_key(body.sport, latest.athlete_id),
        report_chat.memory_key(latest.athlete_id), body, build, background,
        where="progress", sport=body.sport, position=(latest.result or {}).get("cycling_position"),
        withheld=False,
    )


@router.delete("/chat/memory")
async def forget_memory(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Forget everything the assistant kept about this reader's athletes and
    themselves (every memory row they own as reader). Threads stay."""
    from sqlalchemy import delete

    res = await db.execute(delete(ChatMessage).where(ChatMessage.user_id == user.id,
                                                     ChatMessage.role == ROLE_MEMORY))
    await db.commit()
    _track(user, "chat_memory_forgotten", {"rows": res.rowcount or 0})
    return {"forgotten": res.rowcount or 0}


def _where(key: str) -> str:
    return "progress" if key.startswith(report_chat.PROGRESS_KEY) else "report"


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
        elif m.role in (ROLE_ASSISTANT, ROLE_MEMORY):
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
    asked = sorted(((m, email) for m, email, _t in rows if m.role == ROLE_USER),
                   key=lambda x: -x[0].created_at_ms)
    # Each question's answer is written 1 ms after it (report_chat.new_rows):
    # read together on the admin screen, to check what the model says.
    answers = {(m.user_id, m.analysis_client_id, m.created_at_ms): m.text
               for m, _e, _t in rows if m.role == ROLE_ASSISTANT}
    models_by = {(m.user_id, m.analysis_client_id, m.created_at_ms): m.model
                 for m, _e, _t in rows if m.role == ROLE_ASSISTANT}
    topics: dict[str, int] = {}
    repeats: dict[str, int] = {}
    for m, _e in asked:
        t = chat_topics.classify(m.text)
        topics[t] = topics.get(t, 0) + 1
        k = " ".join(m.text.lower().split())
        repeats[k] = repeats.get(k, 0) + 1
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
        "topics": dict(sorted(topics.items(), key=lambda kv: -kv[1])),
        "top_questions": [{"text": k, "n": n} for k, n in
                          sorted(repeats.items(), key=lambda kv: -kv[1]) if n > 1][:10],
        "recent": [{"email": e, "text": m.text, "topic": chat_topics.classify(m.text), "at": m.created_at_ms,
                    "where": _where(m.analysis_client_id), "answer": answers.get((m.user_id, m.analysis_client_id, m.created_at_ms + 1)),
                    "model": models_by.get((m.user_id, m.analysis_client_id, m.created_at_ms + 1))}
                   for m, e in asked[:30]],
    }
