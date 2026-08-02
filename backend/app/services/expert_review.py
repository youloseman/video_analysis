"""The Expert Review deliverable: one report template, three consumers.

The $29 Expert Review is a human reading the machine's output. That framing is
the whole design here, and it decides the shape of the template:

* The expert does NOT write from scratch. The automated analysis already
  produced angles, a score and AI coaching -- what a buyer is paying for is
  *judgment on top of it*: which of those numbers to trust, which single thing
  actually matters, and what to leave alone. Sections are worded as reactions to
  the analysis, and ``prefill`` seeds the one section that can be derived.
* It has to be fillable in ~20 minutes. At $29 a review that takes two hours is
  a hobby, not a product, so every section is short and several are lists.
* The clip itself is gone. Uploads expire with their job (``job_ttl_hours``,
  6h), so a review bought on Tuesday for Monday's ride can only ever see the
  STORED analysis entry and its annotated keyframe. The template says so out
  loud rather than pretending otherwise.

The section list is defined once, here, and drives all three renderings -- the
admin editor, the athlete's copy, and the sample shown on the pricing page --
so they cannot drift into disagreeing about what a report is.
"""

from __future__ import annotations

from typing import Any

REPORT_VERSION = 1

# Per-field caps. Generous enough not to be felt while writing, tight enough
# that a runaway paste cannot turn an order row into a megabyte.
MAX_PROSE_CHARS = 4000
MAX_LINE_CHARS = 300
MAX_PRIORITIES = 3
MAX_FIT_ROWS = 6
MAX_REVIEWER_CHARS = 80

# kind:
#   "prose"      -> one Markdown block
#   "priorities" -> ordered list of {title, why, change, check}
#   "fit"        -> table of {part, now, suggest, why, risk}   (cycling only)
SECTIONS: tuple[dict[str, Any], ...] = (
    {
        "key": "verdict",
        "title": "The verdict",
        "kind": "prose",
        "required": True,
        "hint": "Two or three sentences. The one thing they should walk away "
                "with. Write it last.",
    },
    {
        "key": "trust",
        "title": "How much to trust these numbers",
        "kind": "prose",
        "required": True,
        "hint": "Which measurements are solid and which to discount, and why. "
                "Pre-filled from the capture quality — edit it, don't just ship it.",
    },
    {
        "key": "working",
        "title": "What's already working",
        "kind": "prose",
        "required": True,
        "hint": "Name it explicitly. This is what stops them 'fixing' something "
                "that was fine.",
    },
    {
        "key": "priorities",
        "title": "Fix these, in this order",
        "kind": "priorities",
        "required": True,
        "hint": "One to three. Ranked. If everything is a priority, nothing is.",
    },
    {
        "key": "fit",
        "title": "Position changes",
        "kind": "fit",
        "required": False,
        "sports": ("bike",),
        "hint": "Only where you'd actually move something. Every row names its "
                "risk — a fit change always costs something somewhere.",
    },
    {
        "key": "plan",
        "title": "The next four weeks",
        "kind": "prose",
        "required": True,
        "hint": "Concrete: drills, sets, how often. Something they can put in a "
                "calendar.",
    },
    {
        "key": "dont_change",
        "title": "Don't change this",
        "kind": "prose",
        "required": False,
        "hint": "The anti-list. Especially useful when the AI coaching flagged "
                "something you disagree with.",
    },
    {
        "key": "retest",
        "title": "When to film again",
        "kind": "prose",
        "required": True,
        "hint": "A date and what to compare. A review with no re-test is a "
                "one-off opinion.",
    },
    {
        "key": "caveats",
        "title": "What this footage couldn't tell me",
        "kind": "prose",
        "required": False,
        "hint": "Honest limits: one plane, one speed, no force data, no history.",
    },
)

SECTION_BY_KEY = {s["key"]: s for s in SECTIONS}

PRIORITY_FIELDS = ("title", "why", "change", "check")
PRIORITY_LABELS = {
    "title": "What",
    "why": "Why it matters",
    "change": "What to change",
    "check": "How you'll know it worked",
}
FIT_FIELDS = ("part", "now", "suggest", "why", "risk")
FIT_LABELS = {
    "part": "Part", "now": "Now", "suggest": "Suggested",
    "why": "Why", "risk": "Trade-off",
}


def sections_for(sport: str | None) -> tuple[dict[str, Any], ...]:
    """The sections that apply to this sport (the fit table is cycling-only)."""
    return tuple(
        s for s in SECTIONS
        if not s.get("sports") or (sport or "run") in s["sports"]
    )


def blank_report(sport: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"version": REPORT_VERSION, "reviewer": ""}
    for s in sections_for(sport):
        out[s["key"]] = [] if s["kind"] in ("priorities", "fit") else ""
    return out


# --------------------------------------------------------------------------
# Prefill
# --------------------------------------------------------------------------
def _quality_sentences(entry: dict[str, Any]) -> list[str]:
    """Read the stored analysis for anything that qualifies its own numbers.

    Deliberately conservative: this is a DRAFT for a human to correct, so it
    states what the pipeline reported and stops. It never invents a judgement.

    Note ``entry["metrics"]`` is the client's *display* list of angle rows, not
    a metrics dict -- the quality flags live at the top level, saved alongside
    the score (see saveToHistory).
    """
    gated = bool(entry.get("gated"))
    confidence = entry.get("confidence")
    warnings = [w for w in (entry.get("warnings") or []) if w][:3]

    # Nothing was recorded at all. Entries saved before the flags were persisted
    # look exactly like clean ones, so calling them clean would invent an
    # all-clear the analyzer never gave -- the same failure the free report used
    # to have. Say what is actually true: we don't know.
    if "gated" not in entry and "confidence" not in entry and not warnings:
        return [
            "This analysis was saved before capture-quality flags were "
            "recorded, so there is nothing automated to go on here — judge the "
            "reliability from the frame yourself."
        ]

    lines: list[str] = []
    if gated:
        lines.append(
            "The analyzer flagged this capture as partial, so treat the score "
            "as a reference rather than a measurement."
        )
    if confidence and confidence != "high":
        lines.append(f"Automated confidence on this run was **{confidence}**.")
    lines += [f"- {w}" for w in warnings]
    if not lines:
        lines.append(
            "The analyzer reported a clean capture and did not qualify its own "
            "measurements."
        )
    return lines


def prefill(entry: dict[str, Any] | None) -> dict[str, Any]:
    """A starting draft for an order, given the athlete's stored analysis.

    Only ``trust`` is seeded -- it is the one section that is genuinely derived
    rather than judged. Seeding the rest would invite shipping a report nobody
    actually wrote.
    """
    sport = (entry or {}).get("sport") or "run"
    draft = blank_report(sport)
    if entry:
        draft["trust"] = "\n".join(_quality_sentences(entry))
    return draft


# --------------------------------------------------------------------------
# Normalisation (everything from the admin form goes through this)
# --------------------------------------------------------------------------
def _clip(v: Any, limit: int) -> str:
    return str(v or "").strip()[:limit]


def _rows(raw: Any, fields: tuple[str, ...], cap: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for item in (raw or [])[:cap]:
        if not isinstance(item, dict):
            continue
        row = {f: _clip(item.get(f), MAX_LINE_CHARS) for f in fields}
        if any(row.values()):          # drop rows the expert left entirely blank
            out.append(row)
    return out


def normalize_report(raw: Any, sport: str | None = None) -> dict[str, Any]:
    """Coerce whatever the editor posted into the stored shape.

    An allowlist over the section list: a key that is not a section cannot be
    smuggled into the order row, and a section that does not apply to this sport
    is dropped rather than stored and silently never rendered.
    """
    raw = raw if isinstance(raw, dict) else {}
    out: dict[str, Any] = {
        "version": REPORT_VERSION,
        "reviewer": _clip(raw.get("reviewer"), MAX_REVIEWER_CHARS),
    }
    for s in sections_for(sport):
        key, kind = s["key"], s["kind"]
        if kind == "priorities":
            out[key] = _rows(raw.get(key), PRIORITY_FIELDS, MAX_PRIORITIES)
        elif kind == "fit":
            out[key] = _rows(raw.get(key), FIT_FIELDS, MAX_FIT_ROWS)
        else:
            out[key] = _clip(raw.get(key), MAX_PROSE_CHARS)
    return out


def missing_required(report: Any, sport: str | None = None) -> list[str]:
    """Required sections still empty. Publishing is blocked on this being [].

    A half-written report reaching a paying customer is worse than a late one,
    and 'delivered' is the state the customer is shown -- so the gate lives on
    the server, not in the editor's markup.
    """
    report = report if isinstance(report, dict) else {}
    return [
        s["title"] for s in sections_for(sport)
        if s.get("required") and not report.get(s["key"])
    ]


def is_empty(report: Any) -> bool:
    """True when nothing has been written yet (an untouched draft)."""
    if not isinstance(report, dict):
        return True
    return not any(
        report.get(s["key"]) for s in SECTIONS
    ) and not report.get("reviewer")


# --------------------------------------------------------------------------
# "Your review is ready"
#
# The email carries the VERDICT and nothing else. Not the whole report: the
# point is to get the athlete back to the app where the priorities, the plan and
# the re-test live together — and a review pasted into an inbox is a review
# nobody re-reads in week three. But not a bare "you have a notification"
# either; the one paragraph the reviewer wrote as the takeaway is exactly what
# earns the click.
# --------------------------------------------------------------------------
def _first_paragraph(md: str, limit: int = 320) -> str:
    para = (md or "").strip().split("\n\n")[0].replace("\n", " ").strip()
    # Strip the Markdown emphasis markers; this is going into a mail client.
    para = para.replace("**", "").replace("*", "")
    return para if len(para) <= limit else para[: limit - 1].rstrip() + "…"


def _esc(s: str) -> str:
    return (
        str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def ready_email(report: dict[str, Any], app_url: str) -> tuple[str, str, str]:
    """(subject, text, html) telling an athlete their review has landed."""
    verdict = _first_paragraph((report or {}).get("verdict") or "")
    reviewer = (report or {}).get("reviewer") or "your reviewer"
    subject = "Your Flapp Expert Review is ready"
    text = (
        "Your Expert Review is ready.\n\n"
        f"{verdict}\n\n"
        f"— {reviewer}\n\n"
        f"Read the full review (priorities, the four-week plan and when to "
        f"film again):\n{app_url}\n"
    )
    html = (
        '<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;'
        'font-size:16px;line-height:1.6;color:#14294B;max-width:520px">'
        '<p style="font-size:13px;letter-spacing:.08em;text-transform:uppercase;'
        'color:#2F6DE0;margin:0 0 6px">Flapp · Expert Review</p>'
        '<h1 style="font-size:22px;margin:0 0 18px">Your review is ready</h1>'
        f'<p style="margin:0 0 16px">{_esc(verdict)}</p>'
        f'<p style="margin:0 0 24px;color:#5A6478">— {_esc(reviewer)}</p>'
        f'<p style="margin:0 0 24px"><a href="{_esc(app_url)}" '
        'style="background:#2F6DE0;color:#fff;text-decoration:none;'
        'padding:12px 20px;border-radius:8px;display:inline-block;'
        'font-weight:600">Read the full review</a></p>'
        '<p style="font-size:13px;color:#5A6478;margin:0">It has your ranked '
        'priorities, a four-week plan, and when to film again.</p></div>'
    )
    return subject, text, html


# --------------------------------------------------------------------------
# The sample shown on the pricing page
#
# The Expert Review button currently sells something invisible: nobody knows
# what $29 buys until after they have spent it. This is a real, complete report
# rendered by the same code as a real one -- if the template changes, the
# sample changes with it or the tests fail.
# --------------------------------------------------------------------------
SAMPLE_SPORT = "run"
SAMPLE_CONTEXT = {
    "sport": "run",
    "kind": "video",
    "score": 74,
    "grade": "C",
    "when": "a 10-second treadmill clip at easy pace",
}
SAMPLE_REPORT: dict[str, Any] = {
    "version": REPORT_VERSION,
    "reviewer": "Artur Podgornyi",
    "verdict": (
        "Your cadence is not the problem, and I'd ignore the app's suggestion to "
        "raise it. The problem is that you're landing with the foot well ahead of "
        "your hips and braking on every step — the low cadence is a *symptom* of "
        "that long reach, not the cause. Fix where the foot lands and the cadence "
        "will come up on its own."
    ),
    "trust": (
        "The joint angles here are solid — tracking was clean through the whole "
        "clip and the near-side leg is fully visible.\n\n"
        "Two numbers I'd treat loosely: **ground contact time**, because a "
        "treadmill belt flatters it by 5–10 ms, and **vertical oscillation**, "
        "which is measured at the hip and will read slightly high if the camera "
        "wasn't exactly level. Neither changes what I've written below."
    ),
    "working": (
        "Your trunk position is genuinely good — 8° of forward lean from the "
        "ankle, held steady across the whole clip. Most runners at your level "
        "either run stacked upright or break at the waist; you do neither.\n\n"
        "Arm carriage is relaxed and symmetric. Leave both alone."
    ),
    "priorities": [
        {
            "title": "Land closer to your hips",
            "why": "Your shin is ~12° past vertical at contact, which puts the "
                   "foot in front of your centre of mass. Every step applies a "
                   "small brake, and the load goes through a straight knee — "
                   "that's where the shin-splint history comes from.",
            "change": "Don't reach for the ground with the front leg. Think "
                      "about pulling the foot back underneath you just before "
                      "it lands. Cue: 'land under the hip'.",
            "check": "In the next clip the shin should be within ~5° of vertical "
                     "at contact, and the knee visibly soft rather than locked.",
        },
        {
            "title": "Let cadence rise on its own — to about 172",
            "why": "You're at 163. Chasing 180 with the current landing just "
                   "makes you shuffle. Once the reach is gone, a shorter step "
                   "raises turnover for free.",
            "change": "Run 4×2 min at 170–174 spm with a metronome, easy pace, "
                      "once a week. Don't hold it for a whole run yet.",
            "check": "170+ feels unremarkable rather than rushed.",
        },
        {
            "title": "Calf tolerance before you change anything else",
            "why": "Landing closer to the hips shifts load from the shin to the "
                   "calf and Achilles. Making that change on untrained calves is "
                   "the classic way to trade shin splints for Achilles pain.",
            "change": "Straight- and bent-knee calf raises, 3×15 each, twice a "
                      "week, starting the week BEFORE you touch your form.",
            "check": "3×15 bent-knee raises with no next-day soreness.",
        },
    ],
    "plan": (
        "**Weeks 1–2** — calf work twice a week (above). Keep running exactly as "
        "you do now; change nothing about your form yet.\n\n"
        "**Weeks 2–4** — add one cadence session a week (4×2 min at 170–174). In "
        "your easy runs, spend the first 5 minutes deliberately thinking 'land "
        "under the hip', then stop thinking about it.\n\n"
        "**Every run** — the last 30 seconds at a comfortably quick turnover. "
        "It's a small dose that teaches the pattern without a session for it."
    ),
    "dont_change": (
        "Don't change your shoes. Your current pair is doing nothing wrong, and "
        "changing footwear at the same time as a landing change means you won't "
        "know which one caused whatever happens next.\n\n"
        "Don't try to land on your forefoot. Your midfoot strike is fine — the "
        "issue is *where* the foot lands relative to your hips, not which part "
        "of it touches first."
    ),
    "retest": (
        "Film again in four weeks, same setup: same treadmill, same camera "
        "height, same easy pace. Compare shin angle at contact first, cadence "
        "second. Ignore the overall score — it moves for reasons that have "
        "nothing to do with what we're working on."
    ),
    "caveats": (
        "This is one plane, one speed, one clip. I can't see rotation, hip drop "
        "or anything about how you land when tired — and tired form is what "
        "actually injures people. If the shin pain comes back on long runs "
        "specifically, that's a different question and a fresh clip at the end "
        "of one would tell us more than this did."
    ),
}
