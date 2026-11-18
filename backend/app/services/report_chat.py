"""Ask about this report: a chat grounded in one saved analysis.

The athlete asks "why does it say lower the saddle?" or "how much would
dropping my head buy over 90 km?" and the answer comes from THEIR numbers,
not from the internet. Nothing is analysed again: the context is the same
document the AI export hands to an outside model (services/export/ai_export.py
-- angle convention, bands, what was not measured), plus what the account
knows about the athlete and the setup: height, mobility screens, the bike,
measured CdA, and the scores of earlier clips on the same setup.

Rules the model is held to, and why each exists:

* **Only this report.** What was not measured it does not know, and it says
  so. The export already lists what a side view cannot see.
* **No CdA from video.** Flapp estimates a relative drag ZONE from the trunk
  angle and a head lever; it does not measure CdA, and a measured CdA (from a
  Notio, a tunnel, a field protocol) is a different kind of number. The two
  are never blended, and a figure from one is never presented as the other.
* **No helmet advice.** Same rule as the coach (llm_recommendations): from a
  side view we cannot tell which helmet is on, and a client in a Giro aero
  helmet was once told to put one on.
* **A withheld score stays withheld.** When the clip could not be graded the
  only useful answer is how to film it again; reasoning about numbers the
  report refused to stand behind would put them back.
* **Minutes and watts are quoted, never computed.** The report carries the
  ranges, and measured CdA differences arrive already converted
  (services/cda_math.py); a model doing its own physics produces a second,
  different number.
* **"What next?" comes from the ladder.** The next field test is the next
  untested step of a fixed order, not an experiment invented per question.
* **Red rows are not fine.** The coach code refuses "Nothing to fix" over a
  red row; the chat bypasses that code, so the rule is restated here.

Cost: one call is ~4-6k tokens in, a few hundred out, on the cheapest Gemini
the chain runs (llm_recommendations._complete, OpenAI behind it) -- a fraction
of a cent. The monthly cap per plan is an abuse ceiling, not a cost control.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat import ROLE_ASSISTANT, ROLE_USER, ChatMessage
from app.models.user import TIER_ADMIN, TIER_COACH, TIER_ENTHUSIAST, TIER_FULL
from app.services import bike_catalog, cda_math
from app.services.usage_limits import next_reset, window_start_ms

# Questions a month, per plan. Free plans have no chat: a free report is a
# score with the rest held back, and a chat about held-back numbers would
# either leak them or answer nothing.
CHAT_LIMITS: dict[str, int] = {
    TIER_ENTHUSIAST: 50,
    TIER_FULL: 200,
    TIER_COACH: 500,
    TIER_ADMIN: 500,
}
CHAT_WINDOW = "month"

# Earlier messages of the thread replayed into each call (both roles).
HISTORY_MESSAGES = 8
ANSWER_MAX_TOKENS = 900
# Earlier clips on the same setup, by date and score: enough to say "you were
# 78, then 84", not a trend analysis.
PREVIOUS_CLIPS = 5


def chat_limit(tier: str | None) -> int:
    return CHAT_LIMITS.get(tier or "", 0)


ROLLOUTS = ("admin", "full", "all")


def rollout() -> str:
    """Who has the chat right now (settings.report_chat). Unknown = admin."""
    from app.core.config import settings

    v = getattr(settings, "report_chat", "admin")
    return v if v in ROLLOUTS else "admin"


def chat_state(user: object) -> str:
    """What the app shows this account under a report.

    ``open`` -- the chat. ``full`` / ``plans`` -- a line saying which plan has
    it (Full during the Full-only step, any paid plan once it is everyone's).
    ``""`` -- nothing: the chat is not launched for anyone this account could
    become, so there is nothing to sell yet.
    """
    tier = getattr(user, "tier", None)
    step = rollout()
    if tier == TIER_ADMIN:
        return "open"
    if step == "admin":
        return ""
    if step == "full":
        return "open" if tier in (TIER_FULL, TIER_COACH) else "full"
    return "open" if chat_limit(tier) > 0 else "plans"


async def used_this_month(db: AsyncSession, user_id: int) -> int:
    start = window_start_ms(CHAT_WINDOW)
    n = await db.scalar(
        select(func.count(ChatMessage.id)).where(
            ChatMessage.user_id == user_id,
            ChatMessage.role == ROLE_USER,
            ChatMessage.created_at_ms >= start,
        )
    )
    return int(n or 0)


def resets_at() -> str:
    return next_reset(CHAT_WINDOW).isoformat()


async def thread(
    db: AsyncSession, reader_id: int, owner_id: int, client_id: str,
) -> list[ChatMessage]:
    rows = (
        await db.execute(
            select(ChatMessage)
            .where(
                ChatMessage.user_id == reader_id,
                ChatMessage.owner_user_id == owner_id,
                ChatMessage.analysis_client_id == client_id,
            )
            .order_by(ChatMessage.created_at_ms, ChatMessage.id)
        )
    ).scalars().all()
    return list(rows)


def message_out(m: ChatMessage) -> dict[str, Any]:
    return {"role": m.role, "text": m.text, "model": m.model, "at": m.created_at_ms}


# ---------------------------------------------------------------------------
# The context
# ---------------------------------------------------------------------------

def withheld_block(result: dict[str, Any]) -> str | None:
    w = result.get("score_withheld")
    if not isinstance(w, dict) or not w:
        return None
    return "\n".join([
        "## SCORE WITHHELD",
        "",
        f"This clip was not graded. Reason: {w.get('reason') or 'unknown'}. "
        f"{w.get('detail') or ''}".strip(),
        "",
        "Do not interpret the measurements in this report and do not give fit or "
        "technique advice from them. Explain why the clip could not be graded and "
        "how to film it so it can be: that is the only useful answer here.",
    ])


def _day(ms: int | None) -> str:
    if not ms:
        return "unknown date"
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d")


def athlete_block(
    *,
    sport: str,
    height_cm: int | None = None,
    mobility: dict[str, Any] | None = None,
    profile: dict[str, Any] | None = None,
    cda: list[dict[str, Any]] | None = None,
    previous: list[dict[str, Any]] | None = None,
) -> str:
    """What the account knows that the clip does not. Every line says where it
    came from, and an empty section says it is empty -- a missing bike is a
    reason to ask, not a licence to assume one."""
    lines = ["## The athlete and the setup (from their account, not from the clip)", ""]
    lines.append(f"- Height: {height_cm} cm" if height_cm else "- Height: not given")

    mob = mobility or {}
    if mob.get("hamstring_deg") is not None or mob.get("hip_flexion_deg") is not None:
        bits = []
        if mob.get("hamstring_deg") is not None:
            bits.append(f"straight-leg raise {round(float(mob['hamstring_deg']))}°")
        if mob.get("hip_flexion_deg") is not None:
            bits.append(f"knee-to-chest hip flexion {round(float(mob['hip_flexion_deg']))}°")
        when = f" (screened {mob['measured_on']})" if mob.get("measured_on") else ""
        lines.append("- Mobility screens: " + ", ".join(bits) + when)
    else:
        lines.append("- Mobility screens: not done")

    p = profile or {}
    if sport == "bike":
        if p:
            bike = bike_catalog.describe(
                p.get("bike_family"), p.get("bike_model"), p.get("frame_size"), p.get("bike_year"),
            )
            lines.append(f"- Setup: \"{p.get('name')}\" ({p.get('kind') or 'bike'})")
            lines.append(f"- Bike: {bike}" if bike else "- Bike: make and model not given")
            lines.append(
                f"- Cranks: {p['crank_length_mm']:g} mm" if p.get("crank_length_mm")
                else "- Cranks: not given (the saddle estimate assumed 172.5 mm)"
            )
        else:
            lines.append("- Setup: this clip is not filed under a saved bike; make, model and cranks unknown")

        rows = cda or []
        if rows:
            lines += [
                "",
                "### Measured CdA on this setup (the athlete's own field or tunnel tests)",
                "",
                "These are MEASUREMENTS brought from outside Flapp. Flapp did not "
                "produce them and does not measure CdA from video. Two values may be "
                "compared only when source and protocol match and neither is a single "
                "ride; otherwise say they are not comparable.",
                "",
            ]
            for r in rows:
                note = f"; note: {r['note']}" if r.get("note") else ""
                tested = cda_math.VARIABLES.get(r.get("variable") or "", "")
                tested = f" [tested: {tested}]" if tested else ""
                lines.append(
                    f"- {r.get('measured_on')}: {r.get('value_m2')} m² "
                    f"({r.get('source')}, {r.get('protocol')}){tested}{note}"
                )
                d = r.get("vs_baseline")
                if d:
                    w = ", ".join(f"{v:+g} W at {k} km/h" for k, v in (d.get("watts") or {}).items())
                    lines.append(
                        f"  - vs baseline of {d.get('baseline_on')}: {d['delta_m2']:+.3f} m² "
                        f"({d['delta_pct']:+.1f}%) = {w}; {d['seconds']:+d} s over "
                        f"{d.get('seconds_basis')}"
                    )
                elif r.get("vs_baseline_missing") == "measured_differently":
                    lines.append("  - no baseline measured the same way: not comparable")
            ladder = cda_math.ladder_status(rows)
            nxt = next((s for s in ladder if s["next"]), None)
            done = [s["title"] for s in ladder if s["done"]]
            lines.append("")
            lines.append("Test ladder (Flapp's fixed order of what to field-test next):")
            lines.append("- Already tested: " + ("; ".join(done) if done else "nothing tagged yet"))
            lines.append(
                f"- Next on the ladder: {nxt['title']}. {nxt['why']}" if nxt
                else "- Every step of the ladder has a tagged test."
            )
        else:
            lines.append("- Measured CdA: none on record for this setup")

    prev = previous or []
    if prev:
        lines += ["", "### Earlier clips on the same setup (date, score)", ""]
        lines += [
            f"- {_day(x.get('at'))}: {x['score'] if x.get('score') is not None else 'not graded'}"
            for x in prev
        ]
    return "\n".join(lines)


SYSTEM = (
    "You are Flapp's report assistant: an experienced bike fitter and run coach "
    "answering an athlete's follow-up questions about ONE analysis of their own "
    "video. The report, the athlete's account details and the conversation so far "
    "are below. Answer from them.\n\n"
    "GROUNDING\n"
    "- Use only the numbers in the report and the account section. Never invent a "
    "measurement, an angle, a bike part or a test result. If the report does not "
    "contain what the question needs, say so plainly and say what would answer it "
    "(a clip filmed another way, a measurement, a fitter's tape).\n"
    "- The report's *Not measured* and *Missing context* sections list what a side "
    "view cannot see. Do not reason past them.\n"
    "- Joint angles are INTERNAL (180° = straight limb). Cycling trunk angle is "
    "from horizontal; running trunk lean from vertical. Never invert these.\n"
    "- A value a degree or two outside its band is inside the method's error and "
    "is not a finding. A row marked out of range or a detected issue IS one: never "
    "call it fine, and never answer \"nothing to fix\" while one stands.\n"
    "- Quote time and watt figures only as the report or the account section "
    "states them, with the speed and distance they were given for. Never compute "
    "your own: not from the estimate, not from two measured CdA values the "
    "account section did not already compare. A pair it marks not comparable "
    "stays not comparable.\n"
    "- Asked what to test next: answer from the test ladder in the account "
    "section, the next untested step first. Do not invent a different "
    "experiment.\n\n"
    "AERODYNAMICS\n"
    "- Flapp does NOT measure CdA from video. The aero zone in the report is a "
    "relative estimate from torso angle and head position. Never present it as a "
    "CdA, never give a CdA number for this rider unless it is one of their own "
    "measurements listed in the account section, and always say which kind of "
    "number you are quoting: an estimate from the video, or their measurement.\n"
    "- Never recommend buying, wearing or filming in a different helmet, and never "
    "name helmet models: from a side view we cannot tell which helmet is on.\n\n"
    "FIT ADVICE\n"
    "- Body angles on a bike are coupled. When you suggest a change, say what it "
    "costs in the same breath (opening the hip with more stack sits the rider up).\n"
    "- When the report has a ranked list of adjustment options, recommend from the "
    "top of it.\n"
    "- Give sizes of change only as the report gives them. Name a part (stem, "
    "extension, riser, pads, wedge) ONLY when it appears under 'Parts on record "
    "for this frame', and say its status (confirmed by the maker, several owners, "
    "one owner's report, does not fit). With no such list, or a part not on it, "
    "say compatibility is unknown for this frame and suggest checking with the "
    "maker. Never guess that a part fits.\n"
    "- Pain, numbness or injury: say the clip cannot assess it and suggest a "
    "fitter or a physio. No diagnosis.\n"
    "- If a human review would answer better (a judgement call, a position the "
    "numbers cannot settle), say that a Flapp coach can review this clip.\n\n"
    "STYLE\n"
    "- Answer in the language the athlete wrote in.\n"
    "- Short: two to six sentences, or a few bullets when listing steps. Lead with "
    "the answer. Plain words; keep the report's metric names (Knee BDC, Trunk "
    "angle) as they are.\n"
    "- Off-topic questions (not about this analysis, training for it, or the "
    "equipment in it): one line saying you only answer about this report.\n"
    "- Never use the em dash character (the long dash, U+2014): use a colon, "
    "comma or full stop instead."
)


def build_prompt(
    *,
    report_md: str,
    athlete_md: str,
    withheld_md: str | None,
    history: list[ChatMessage],
    question: str,
) -> str:
    parts = []
    if withheld_md:
        parts.append(withheld_md)
    parts += ["# THE REPORT", report_md, athlete_md]
    recent = history[-HISTORY_MESSAGES:]
    if recent:
        convo = "\n\n".join(
            f"{'ATHLETE' if m.role == ROLE_USER else 'YOU'}: {m.text}" for m in recent
        )
        parts.append(f"# CONVERSATION SO FAR\n\n{convo}")
    parts.append(f"# THE ATHLETE'S QUESTION\n\n{question}")
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


def ask(prompt: str) -> dict[str, Any] | None:
    """One call through the coaching chain. ``{"text", "model"}`` or None.

    Sync: the SDKs are blocking. The route runs it in a thread.
    """
    from app.services.video_analysis.llm_recommendations import _complete

    return _complete(SYSTEM, prompt, ANSWER_MAX_TOKENS, "chat")


def new_rows(
    *, reader_id: int, owner_id: int, client_id: str, question: str,
    answer: dict[str, Any], now_ms: int,
) -> list[ChatMessage]:
    """The question and its answer, written together or not at all."""
    return [
        ChatMessage(
            user_id=reader_id, owner_user_id=owner_id, analysis_client_id=client_id,
            role=ROLE_USER, text=question, model=None, created_at_ms=now_ms,
        ),
        ChatMessage(
            user_id=reader_id, owner_user_id=owner_id, analysis_client_id=client_id,
            role=ROLE_ASSISTANT, text=answer["text"], model=answer.get("model"),
            created_at_ms=now_ms + 1,
        ),
    ]
