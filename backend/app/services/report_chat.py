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

Cost, and the two limits on it (Artur, 2026-11-22: "the chat must never cost
more than the subscription brings in"):

* **Questions a month** (CHAT_LIMITS) -- what the athlete sees: "12 of 50
  left". One call is ~5-9k tokens in (the report, the account, the last few
  turns), a few hundred out: about $0.004 on Gemini 3.5 Flash-Lite.
* **Dollars a month** (CHAT_BUDGET_USD) -- the backstop the athlete never
  sees until it bites: a tenth of the plan's monthly price, priced per model
  (MODEL_PRICES) from the tokens each answer used. It exists for the day the
  chain falls back to a dearer model, a model turns up its thinking, or the
  context grows: the question count alone would not notice any of them.

Tokens are estimated from characters (CHARS_PER_TOKEN) with a margin on the
answer for the thinking a model may bill and not show (OUTPUT_MARGIN): the
chain returns text, not usage, and an estimate on the high side is the safe
one for a ceiling.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat import MEMORY_KEY, ROLE_ASSISTANT, ROLE_MEMORY, ROLE_USER, ChatMessage
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
# Older turns of the same thread go in as their questions only.
OLDER_QUESTIONS = 20
OLDER_QUESTION_CHARS = 200
ANSWER_MAX_TOKENS = 900
# Earlier clips on the same setup, by date and score: enough to say "you were
# 78, then 84", not a trend analysis.
PREVIOUS_CLIPS = 5


def chat_limit(tier: str | None) -> int:
    return CHAT_LIMITS.get(tier or "", 0)


# What a month of answers may cost us, per plan: a tenth of its monthly price
# (Enthusiast $9, Full $13, Coach $25). Typical use spends a fraction of it.
CHAT_BUDGET_USD: dict[str, float] = {
    TIER_ENTHUSIAST: 0.90,
    TIER_FULL: 1.30,
    TIER_COACH: 2.50,
    TIER_ADMIN: 5.00,
}

# USD per million tokens (input, output), first matching prefix wins. Public
# list prices, 2026-11-22: Gemini 3.5 Flash-Lite 0.30 / 2.50 (aggregators
# quoting ai.google.dev), gpt-5-mini 0.25 / 2.00. Anything unlisted is priced
# like a large model, so a surprise model can only make the ceiling tighter.
MODEL_PRICES: tuple[tuple[str, float, float], ...] = (
    ("gemini-3.5-flash-lite", 0.30, 2.50),
    ("gemini-2.5-flash-lite", 0.10, 0.40),
    ("gemini", 0.30, 2.50),
    ("gpt-5-mini", 0.25, 2.00),
    ("gpt-5-nano", 0.05, 0.40),
    ("fake", 0.30, 2.50),
)
UNKNOWN_MODEL_PRICE = (2.50, 10.00)
CHARS_PER_TOKEN = 3.5
OUTPUT_MARGIN = 1.5
# An answer stored before tokens were (2026-11-22): the average prompt.
LEGACY_TOKENS_IN = 7000


def chat_budget(tier: str | None) -> float:
    return CHAT_BUDGET_USD.get(tier or "", 0.0)


def estimate_tokens(text: str) -> int:
    return int(len(text or "") / CHARS_PER_TOKEN) + 1


def model_price(model: str | None) -> tuple[float, float]:
    m = (model or "").lower()
    for prefix, p_in, p_out in MODEL_PRICES:
        if m.startswith(prefix):
            return p_in, p_out
    return UNKNOWN_MODEL_PRICE


def cost_usd(model: str | None, tokens_in: int, tokens_out: int) -> float:
    p_in, p_out = model_price(model)
    return (tokens_in * p_in + tokens_out * OUTPUT_MARGIN * p_out) / 1_000_000


def projected_cost_usd(prompt: str) -> float:
    """What the next answer may cost at most: the prompt, a full-length
    answer, priced as the primary model or the dearer fallback, whichever is
    more."""
    t_in = estimate_tokens(SYSTEM) + estimate_tokens(prompt)
    return max(cost_usd(m, t_in, ANSWER_MAX_TOKENS) for m, _i, _o in MODEL_PRICES[:4])


def answer_cost(m: ChatMessage) -> float:
    t_in = m.tokens_in if m.tokens_in is not None else LEGACY_TOKENS_IN
    t_out = m.tokens_out if m.tokens_out is not None else estimate_tokens(m.text)
    return cost_usd(m.model, t_in, t_out)


async def spent_this_month(db: AsyncSession, user_id: int) -> float:
    """Dollars this reader's answers cost this month (estimated)."""
    rows = (
        await db.execute(
            select(ChatMessage).where(
                ChatMessage.user_id == user_id,
                ChatMessage.role.in_((ROLE_ASSISTANT, ROLE_MEMORY)),
                ChatMessage.created_at_ms >= window_start_ms(CHAT_WINDOW),
            )
        )
    ).scalars().all()
    return sum(answer_cost(m) for m in rows)


# --------------------------------------------------------------------------
# Questions worth asking, written from the report itself
# --------------------------------------------------------------------------
MAX_SUGGESTIONS = 4


def _fmt(v: float, unit: str) -> str:
    return f"{v:.0f}{unit}" if unit in ("°", "") and abs(v) >= 20 else f"{v:g}{unit}"


def suggestions(
    result: dict[str, Any], *, has_cda: bool = False, has_previous: bool = False,
) -> list[str]:
    """Up to four questions about THIS report: its worst readings first, then
    what the account adds (a measured CdA, earlier clips). Asked in English,
    answered in whatever language the athlete writes in."""
    sport = str(result.get("sport_type") or "run")
    pos = result.get("cycling_position")
    out: list[str] = []
    if result.get("score_withheld") or result.get("quality_gate_triggered"):
        out.append("Why could this clip not be scored, and how should I film the next one?")
    flagged: list[dict[str, Any]] = []
    try:
        if sport == "bike":
            from app.services.review_pdf import metric_rows

            rows = metric_rows(result)
        else:
            from app.services.report_pdf import plan_rows

            rows = plan_rows(result)
        rank = {"bad": 0, "warn": 1}
        flagged = sorted((r for r in rows if r.get("status") in rank and r.get("band")
                          and isinstance(r.get("value"), (int, float))),
                         key=lambda r: rank[r["status"]])
    except Exception:  # noqa: BLE001 -- a suggestion is never worth a failed request
        flagged = []
    def name(r: dict[str, Any]) -> str:
        # "Knee at the bottom · saddle height" -> "knee at the bottom"; an
        # acronym-led name (BDC, CdA) keeps its capitals.
        lbl = str(r.get("label") or "").split(" · ")[0].strip()
        return lbl[:1].lower() + lbl[1:] if lbl[1:2].islower() else lbl

    def reading(r: dict[str, Any]) -> str:
        unit = (r.get("unit") or "").strip()
        return _fmt(float(r["value"]), unit)

    if flagged and not out:
        r = flagged[0]
        unit = (r.get("unit") or "").strip()
        lo, hi = r["band"]
        out.append(f"My {name(r)} is {reading(r)} and the range is {lo:g}-{hi:g}{unit}. "
                   "What should I change first, and what does it cost me?")
    if len(flagged) > 1:
        out.append(f"Is my {name(flagged[1])} ({reading(flagged[1])}) worth fixing, or can it wait?")
    if sport == "bike":
        if pos in ("tt_aero", "triathlon"):
            out.append("How much time could my position be worth over my race, and where is it hiding?")
        elif pos in ("road_hoods", "road_drops"):
            out.append("Is this a racing or an endurance position, and what would you change for a long ride?")
        if has_cda:
            out.append("What did my CdA tests show, and what should I test next?")
    else:
        out.append("Which drill fits this report best, and how often should I do it?")
    if has_previous:
        out.append("What changed since my last clip on this setup?")
    if not flagged and not out[:1]:
        out.insert(0, "Everything reads inside its range. Where would you look for more speed?")
    out.append("How should I film the next clip to get the most from it?")
    seen: list[str] = []
    for q in out:
        if q not in seen:
            seen.append(q)
    return seen[:MAX_SUGGESTIONS]


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
    "costs in the same breath (opening the hip with more stack sits the rider up). "
    "Name only couplings that follow from the geometry: saddle height moves the "
    "knee and the hip; bar reach and stack move the elbow, the shoulder and the "
    "trunk. Do not claim a saddle change moves the elbow.\n"
    "- Directions, never reversed (angles are internal, 180 = straight): a LONGER "
    "reach (longer stem, bars further forward) straightens the elbow, opens the "
    "shoulder and lowers the trunk; a SHORTER reach bends the elbow more and "
    "raises the trunk. More stack (spacers, riser) raises the trunk and opens "
    "the hip; less stack does the opposite. A higher saddle straightens the knee "
    "at the bottom of the stroke; a lower saddle bends it. An elbow below its "
    "band is bent more than the target, so shortening the reach makes it worse.\n"
    "- Hoods and drops are road positions, not aero positions: do not call them "
    "aero or tell a road rider they are leaving an aero position.\n"
    "- Stiffness or discomfort on long rides is a fit question: answer it from "
    "the numbers (what in this position could load that area, what to try "
    "first, in small steps, well before an event), then say in one line that a "
    "fitter can check it in person. Only pain, numbness or an injury is the "
    "fitter-or-physio-only case below.\n"
    "- Never say a reading CAUSES a symptom (stiffness, discomfort, fatigue): "
    "the clip cannot tell. Say which readings could play a part, the trade-off "
    "of changing each (a longer reach also flattens the torso, which can load "
    "the lower back more), and that a fitter can find the cause in person.\n"
    "- When the report has a ranked list of adjustment options, recommend from the "
    "top of it.\n"
    "- Mention part compatibility ONLY when the athlete asks about buying, "
    "fitting or swapping a part; never add it to an answer that needs no part "
    "(a saddle height change needs none).\n"
    "- Give sizes of change only as the report gives them. Name a part (stem, "
    "extension, riser, pads, wedge) ONLY when it appears under 'Parts on record "
    "for this frame', and say its status (confirmed by the maker, several owners, "
    "one owner's report, does not fit). With no such list, or a part not on it, "
    "say compatibility is unknown for this frame and suggest checking with the "
    "maker. Never guess that a part fits.\n"
    "- Pain, numbness or injury: say the clip cannot assess it and suggest a "
    "fitter or a physio. No diagnosis. Do not add this line to answers where the "
    "athlete mentioned no pain.\n"
    "- If a human review would answer better (a judgement call, a position the "
    "numbers cannot settle), say that a Flapp coach can review this clip.\n\n"
    "STYLE\n"
    "- Answer in the language the athlete wrote in.\n"
    "- Short: two to six sentences, or a few bullets when listing steps. Lead with "
    "a direct answer to the question asked (yes or no, which way, what first), "
    "then the reason. Plain words; keep the report's metric names (Knee BDC, Trunk "
    "angle) as they are.\n"
    "- Off-topic questions (not about this analysis, training for it, or the "
    "equipment in it): one line saying you only answer about this report.\n"
    "- Never use the em dash character (the long dash, U+2014): use a colon, "
    "comma or full stop instead."
)


_ANSWER_LANG = {"ru": "Russian"}


def _lang(text: str) -> str:
    from app.services.chat_topics import language

    return language(text)


def build_prompt(
    *,
    report_md: str,
    athlete_md: str,
    withheld_md: str | None,
    history: list[ChatMessage],
    question: str,
    memory: str | None = None,
    heading: str = "# THE REPORT",
) -> str:
    parts = []
    if withheld_md:
        parts.append(withheld_md)
    parts += [heading, report_md, athlete_md]
    if memory:
        parts.append(
            "# WHAT THE ATHLETE TOLD YOU IN EARLIER CHATS\n\n"
            "Their own words, kept as notes. Use them for context (goals, races, "
            "limits, what they tried). They are not measurements: the report's "
            "numbers win, and you never quote a note as if the clip measured it.\n\n"
            + memory
        )
    older = [m for m in history[:-HISTORY_MESSAGES] if m.role == ROLE_USER] if len(history) > HISTORY_MESSAGES else []
    if older:
        # The thread's earlier turns, compressed to the questions: what the
        # athlete wanted to know is the part that carries meaning; the answers
        # came from this same report and can be given again from it.
        qs = "\n".join(f"- {m.text[:OLDER_QUESTION_CHARS]}" for m in older[-OLDER_QUESTIONS:])
        parts.append(f"# EARLIER QUESTIONS IN THIS THREAD (answered before)\n\n{qs}")
    recent = history[-HISTORY_MESSAGES:]
    if recent:
        convo = "\n\n".join(
            f"{'ATHLETE' if m.role == ROLE_USER else 'YOU'}: {m.text}" for m in recent
        )
        parts.append(f"# CONVERSATION SO FAR\n\n{convo}")
    parts.append(f"# THE ATHLETE'S QUESTION\n\n{question}")
    lang = _ANSWER_LANG.get(_lang(question))
    if lang:
        # Flash-Lite answered a Russian question in English with the rule only
        # in the system prompt (live check, 2026-11-22): said last, it holds.
        parts.append(f"Write your whole answer in {lang}.")
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


def ask(prompt: str) -> dict[str, Any] | None:
    """One call through the coaching chain. ``{"text", "model"}`` or None.

    Sync: the SDKs are blocking. The route runs it in a thread.
    """
    from app.services.video_analysis.llm_recommendations import _complete

    return _complete(SYSTEM, prompt, ANSWER_MAX_TOKENS, "chat")


# --------------------------------------------------------------------------
# Memory: what the athlete told the assistant, kept across reports
# --------------------------------------------------------------------------
# Each report keeps its whole thread (shown again when the athlete comes
# back). Across reports the assistant keeps NOTES, not transcripts: a few
# bullets of what the athlete said about themselves -- the race, the back
# that complains, the stem they cannot change -- rewritten by the cheapest
# model after each answer. A transcript would grow every call's prompt for
# good; the notes stay under MEMORY_MAX_CHARS whatever the history.
#
# Keyed by reader, report owner and athlete: an athlete's own notes; a coach
# keeps separate notes for each athlete they read, never one mixed pile.
MEMORY_MAX_CHARS = 700
MEMORY_ANSWER_TOKENS = 300
NO_CHANGE = "NO_CHANGE"

MEMORY_SYSTEM = (
    "You keep short notes about one athlete for a bike and run technique "
    "assistant. You get the current notes and the latest exchange. Return the "
    "updated notes: durable facts the ATHLETE stated about themselves -- goals, "
    "races and dates, distances, injuries or pain they mentioned, flexibility "
    "limits, kit they own or cannot change, what they tried and how it went, "
    "preferences about how they want answers.\n"
    "- Only what the athlete said. Never copy numbers from the report or the "
    "assistant's advice, unless the athlete said they acted on it, and never "
    "copy what the account already records (their bike, height, flexibility).\n"
    "- At most 8 bullets starting with '- ', under 600 characters, in English. "
    "Drop what a newer statement replaced.\n"
    "- If the exchange adds nothing durable, reply with exactly NO_CHANGE."
)


def memory_key(athlete_id: int | None) -> str:
    return MEMORY_KEY if athlete_id is None else f"{MEMORY_KEY}:a{athlete_id}"


async def memory_of(db: AsyncSession, reader_id: int, owner_id: int, key: str) -> ChatMessage | None:
    return (
        await db.execute(
            select(ChatMessage)
            .where(
                ChatMessage.user_id == reader_id,
                ChatMessage.owner_user_id == owner_id,
                ChatMessage.analysis_client_id == key,
                ChatMessage.role == ROLE_MEMORY,
            )
            .order_by(ChatMessage.created_at_ms.desc(), ChatMessage.id.desc())
            .limit(1)
        )
    ).scalars().first()


def memory_prompt(notes: str | None, question: str, answer: str) -> str:
    return (f"# CURRENT NOTES\n\n{notes or '(none yet)'}\n\n"
            f"# LATEST EXCHANGE\n\nATHLETE: {question}\n\nASSISTANT: {answer[:1500]}")


def clean_memory(text: str | None) -> str | None:
    """The notes as stored, or None for "nothing new"."""
    t = (text or "").strip()
    if not t or t.upper().startswith(NO_CHANGE):
        return None
    # Small models vary the bullet ("* ", "• ", "1. "): any of them is a note.
    lines = []
    for ln in t.splitlines():
        m = re.match(r"\s*(?:[-*\u2022]|\d+[.)])\s+(.+)", ln)
        if m:
            lines.append("- " + m.group(1).strip())
    if not lines:
        return None
    out = "\n".join(lines).replace("\u2014", ",")
    return out[:MEMORY_MAX_CHARS]


def remember(notes: str | None, question: str, answer: str) -> dict[str, Any] | None:
    """One cheap call: the notes rewritten with the latest exchange, or None."""
    from app.services.video_analysis.llm_recommendations import _complete

    prompt = memory_prompt(notes, question, answer)
    got = _complete(MEMORY_SYSTEM, prompt, MEMORY_ANSWER_TOKENS, "chat_memory")
    if not got:
        return None
    return {"text": clean_memory(got.get("text")), "model": got.get("model"),
            "tokens_in": estimate_tokens(MEMORY_SYSTEM) + estimate_tokens(prompt),
            "tokens_out": estimate_tokens(got.get("text") or "")}


# --------------------------------------------------------------------------
# Which way each flagged reading moves, worked out here, not by the model
# --------------------------------------------------------------------------
# Live check 2026-11-22: with the rule only in SYSTEM, Flash-Lite still told a
# rider with bent elbows that a SHORTER reach would straighten them (one answer
# in ten, a different one each run). The directions for this report's own
# flagged readings are now written out in its context, for the model to quote.
# Bike only: a run's fixes are drills, already in the report's plan.
_DIRECTIONS: dict[str, tuple[str, str]] = {
    # field: (below the band, above the band)
    "knee_at_bdc": ("the knee is more bent than the target at the bottom: raising the saddle straightens it, "
                    "lowering it bends it more",
                    "the knee is straighter than the target at the bottom: lowering the saddle bends it, "
                    "raising it straightens it further"),
    "knee_at_tdc": ("the knee is more closed than the target at the top: a higher saddle or shorter cranks open it",
                    "the knee is more open than the target at the top: a lower saddle closes it"),
    "hip_at_tdc": ("the hip is more closed than the target at the top: more stack, a shorter reach or a "
                   "higher saddle opens it; less stack or a longer reach closes it further",
                   "the hip is more open than the target: that is comfort, not a fault"),
    "trunk_angle_avg": ("the torso is lower than the band: more stack or a shorter reach raises it",
                        "the torso is more upright than the band: a longer or lower front (longer stem, fewer "
                        "spacers) lowers it; a shorter or higher front raises it further"),
    "shoulder_angle_avg": ("the shoulder is more closed than the band: a longer reach opens it; a shorter "
                           "reach closes it further",
                           "the shoulder is more open than the band: a shorter reach closes it; a longer reach "
                           "opens it further"),
    "elbow_angle_avg": ("the arms are more bent than the target: a LONGER reach straightens them; a SHORTER "
                        "reach bends them more",
                        "the arms are nearly straight: a SHORTER reach bends them; a longer reach straightens "
                        "them further"),
}


def directions_block(result: dict[str, Any]) -> str | None:
    """Which change moves each out-of-band bike reading which way, or None."""
    if str(result.get("sport_type") or "") != "bike":
        return None
    lines = []
    for r in _rows_for(result):
        f, v, band = r.get("field"), r.get("value"), r.get("band")
        if f not in _DIRECTIONS or r.get("status") not in ("bad", "warn") or not band or not isinstance(v, (int, float)):
            continue
        side = 0 if v < band[0] else 1 if v > band[1] else None
        if side is None:
            continue
        lines.append(f"- {r['label']} {v:.0f}{r.get('unit') or ''} (range {band[0]:g}-{band[1]:g}): {_DIRECTIONS[f][side]}.")
    if not lines:
        return None
    return ("## Which way each flagged reading moves\n\n"
            "Worked out from the bike's geometry. Quote these directions; never reverse them.\n\n"
            + "\n".join(lines))


# --------------------------------------------------------------------------
# Progress: the same assistant over a series of comparable clips
# --------------------------------------------------------------------------
# The Progress page already picks a like-for-like series (one sport, one
# camera side, one kind, one bike position); the chat is given exactly that
# series, so it can only compare what the page compares. One thread per sport
# (and per athlete, for a coach's own uploads), under a pseudo report id.
PROGRESS_KEY = "_progress"
PROGRESS_MAX_CLIPS = 8
# Below these a change is inside what 2D video measures twice on one rider.
NOISE_DEG = 3.0
NOISE_SCORE = 3.0

PROGRESS_RULES = (
    "# THIS IS A PROGRESS CHAT\n\n"
    "The athlete is asking about a series of their own clips, oldest first, all "
    "filmed the same way (same sport, camera side, kind and bike position). "
    "Rules on top of the usual ones:\n"
    f"- A change smaller than {NOISE_DEG:g} degrees, or {NOISE_SCORE:g} score points, is "
    "within what the method measures twice on the same rider: call it steady, "
    "not progress and not a decline.\n"
    "- Do not invent a cause. If the athlete did not say what they changed "
    "between clips, say what moved and ask what they changed.\n"
    "- Compare clips only through the table below; the latest report in full "
    "follows it for detail and for what to do next.\n"
)


def progress_key(sport: str, athlete_id: int | None) -> str:
    base = f"{PROGRESS_KEY}:{sport}"
    return base if athlete_id is None else f"{base}:a{athlete_id}"


def _rows_for(result: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        if str(result.get("sport_type") or "run") == "bike":
            from app.services.review_pdf import metric_rows

            return metric_rows(result)
        from app.services.report_pdf import plan_rows

        return plan_rows(result)
    except Exception:  # noqa: BLE001 -- one odd stored result never sinks the chat
        return []


_MARK = {"good": "in range", "warn": "near edge", "bad": "OUT"}


def progress_series(clips: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """{labels, clips:[{date, score, grade, values:{label:(v, unit, status)}}], bands}."""
    out: list[dict[str, Any]] = []
    labels: list[str] = []
    bands: dict[str, Any] = {}
    for day, res in clips[-PROGRESS_MAX_CLIPS:]:
        vals = {}
        for r in _rows_for(res):
            v = r.get("value")
            if not isinstance(v, (int, float)):
                continue
            lbl = str(r.get("label") or r.get("field") or "?")
            vals[lbl] = (float(v), (r.get("unit") or "").strip(), r.get("status"))
            if lbl not in labels:
                labels.append(lbl)
            if r.get("band"):
                bands[lbl] = r["band"]
        out.append({"date": day, "score": None if res.get("score_withheld") else res.get("technique_score"),
                    "grade": res.get("letter_grade"), "values": vals})
    return {"labels": labels, "clips": out, "bands": bands}


def progress_block(series: dict[str, Any]) -> str:
    clips = series["clips"]
    head = "| Reading | " + " | ".join(c["date"] for c in clips) + " | Range |"
    lines = [f"## Progress across {len(clips)} clips (oldest first)", "",
             "| Score | " + " | ".join("withheld" if c["score"] is None else f"{c['score']}" + (f" ({c['grade']})" if c.get("grade") else "")
                                      for c in clips) + " | |", "", head,
             "|" + "---|" * (len(clips) + 2)]
    for lbl in series["labels"]:
        cells = []
        unit = ""
        for c in clips:
            got = c["values"].get(lbl)
            if not got:
                cells.append("-")
                continue
            v, unit, st = got
            cells.append(f"{v:.0f}{unit}" + (f" {_MARK[st]}" if st in _MARK else "") if abs(v) >= 10
                         else f"{v:.2f}{unit}" + (f" {_MARK[st]}" if st in _MARK else ""))
        band = series["bands"].get(lbl)
        lines.append(f"| {lbl} | " + " | ".join(cells) + f" | {f'{band[0]:g}-{band[1]:g}{unit}' if band else 'not graded'} |")
    return "\n".join(lines)


def _human_day(iso: str) -> str:
    """2026-10-08 -> Oct 8 (the table keeps ISO for the model)."""
    try:
        d = datetime.strptime(iso, "%Y-%m-%d")
        return f"{d:%b} {d.day}"
    except ValueError:
        return iso


def progress_suggestions(series: dict[str, Any]) -> list[str]:
    clips = series["clips"]
    out: list[str] = []
    scored = [c for c in clips if c["score"] is not None]
    if len(scored) >= 2:
        a, b = scored[0], scored[-1]
        out.append(f"My score went from {a['score']} to {b['score']} since {_human_day(a['date'])}. "
                   "What changed the most?")
    moves = []
    for lbl in series["labels"]:
        pts = [c["values"][lbl] for c in clips if lbl in c["values"]]
        if len(pts) >= 2:
            moves.append((abs(pts[-1][0] - pts[0][0]), lbl, pts[0], pts[-1]))
    moves.sort(reverse=True)
    if moves:
        _d, lbl, first, last = moves[0]
        name = lbl.split(" · ")[0]
        name = name[:1].lower() + name[1:] if name[1:2].islower() else name
        out.append(f"My {name} moved from {first[0]:.0f}{first[1]} to {last[0]:.0f}{last[1]}. "
                   "Is that real, or measurement noise?")
    out.append("What should I work on next to keep improving?")
    out.append("Which of my changes is worth keeping, and which should I undo?")
    return out[:MAX_SUGGESTIONS]


def new_rows(
    *, reader_id: int, owner_id: int, client_id: str, question: str,
    answer: dict[str, Any], now_ms: int, prompt: str = "",
) -> list[ChatMessage]:
    """The question and its answer, written together or not at all. The
    answer carries what it cost in tokens (estimated: see the module notes)."""
    return [
        ChatMessage(
            user_id=reader_id, owner_user_id=owner_id, analysis_client_id=client_id,
            role=ROLE_USER, text=question, model=None, created_at_ms=now_ms,
        ),
        ChatMessage(
            user_id=reader_id, owner_user_id=owner_id, analysis_client_id=client_id,
            role=ROLE_ASSISTANT, text=answer["text"], model=answer.get("model"),
            created_at_ms=now_ms + 1,
            tokens_in=estimate_tokens(SYSTEM) + estimate_tokens(prompt or question),
            tokens_out=estimate_tokens(answer["text"]),
        ),
    ]
