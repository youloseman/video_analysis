"""The analysis report as a branded A4 document -- the Full plan's PDF.

Artur (2026-10-26): "PDF report" on the Full card was the browser printing
the app's page. Full now gets a designed document in the Expert Review's
style (services/review_pdf.py): a cover with the score and the frame, the
coach's read, the plan in order, and every number against its range -- the
bike's against the Kona pro field too.

``render_report_html`` is pure: a stored result + a few facts in, one
self-contained HTML page out (images as data URIs). The browser turns it into
a PDF (Print > Save as PDF), so the server needs no Chromium and a phone can
save it the same way.
"""
from __future__ import annotations

import re
from typing import Any

from app.services.review_pdf import (
    _CSS,
    _STATUS_WORD,
    _e,
    _ring,
    _status,
    metric_rows,
)
from app.services.video_analysis.biomechanics.action_plan_builder import (
    FIT_STEP_NOTE,
    apply_hip_over_bb,
)
from app.services.video_analysis.biomechanics.cycling_positions import NON_DRIVE_KNEE_NOTE
from app.services.video_analysis.biomechanics.running_action_plan_builder import (
    _FAULT_DIRECTION,
)
from app.services.video_analysis.biomechanics.technique_scorer import (
    CYCLING_DERIVED_COMPONENTS,
    CYCLING_WEIGHTS,
    RUNNING_WEIGHTS,
    WORST_COMPONENT_PULL,
)
from app.services.video_analysis.capture_report import film_notes_of

_SPORT = {"bike": "Cycling", "run": "Running"}

# What the score is built from: the scorer's components, named as the app
# names them (SCORE_PARTS / PHOTO_PARTS in index.html).
_PART_NAMES = {
    "knee_bdc": "Knee at bottom", "trunk_angle": "Trunk", "knee_tdc": "Knee at top",
    "elbow_angle": "Elbow", "shoulder_angle": "Shoulder", "helmet_above_back": "Helmet above back",
    "pelvic_ratio": "Pelvis", "forearm_tilt": "Forearm", "saddle_fit": "Saddle height",
    "cadence": "Cadence", "trunk_lean": "Trunk lean", "overstride": "Overstride",
    "knee_contact": "Knee at contact", "phase_stability": "Stride stability",
    "elbow_swing": "Arm swing", "vertical_osc": "Vertical oscillation",
    "knee_swing": "Knee in swing", "waveform_similarity": "Stride shape",
    "knee": "Knee", "trunk": "Trunk", "shoulder": "Shoulder", "elbow": "Elbow",
    "hip": "Hip", "ankle": "Ankle",
}


def breakdown_items(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Each scored component with its score and weight, heaviest first.

    A clip carries ``score_breakdown`` (component -> score) and the weights
    live in the scorer; a photo carries both on ``score.per_angle``.
    """
    items: list[dict[str, Any]] = []
    parts = result.get("score_breakdown")
    if isinstance(parts, dict) and parts:
        sport = result.get("sport_type") or result.get("sport")
        weights = CYCLING_WEIGHTS if sport == "bike" else RUNNING_WEIGHTS
        for k, w in weights.items():
            v = parts.get(k)
            if isinstance(v, (int, float)):
                items.append({"name": _PART_NAMES.get(k, k), "score": float(v), "weight": w})
    else:
        per = (result.get("score") or {}).get("per_angle") if isinstance(result.get("score"), dict) else None
        for k, d in (per or {}).items():
            if isinstance(d, dict) and isinstance(d.get("score"), (int, float)):
                items.append({"name": _PART_NAMES.get(k, k.replace("_", " ").capitalize()),
                              "score": float(d["score"]), "weight": float(d.get("weight") or 0)})
    return sorted(items, key=lambda i: -i["weight"])


def _score_notes(result: dict[str, Any], score: Any) -> list[str]:
    """Why the table above does not simply average to the score: what the clip
    could not measure, and the pull toward the worst part (a bike's score is
    as good as its weakest point: technique_scorer.WORST_COMPONENT_PULL). Said
    only when the sum checks out, so the sentence can't be wrong about a score
    that was reached some other way."""
    parts = result.get("score_breakdown")
    if not (isinstance(parts, dict) and parts and isinstance(score, (int, float))):
        return []
    sport = result.get("sport_type") or result.get("sport")
    weights = CYCLING_WEIGHTS if sport == "bike" else RUNNING_WEIGHTS
    scored = {k: float(parts[k]) for k in weights if isinstance(parts.get(k), (int, float))}
    notes = []
    missing = [_PART_NAMES.get(k, k) for k in weights if k not in scored]
    if missing:
        notes.append(f"Not measured on this clip, so not counted: {', '.join(missing)}.")
    candidates = {k: v for k, v in scored.items() if k not in CYCLING_DERIVED_COMPONENTS}
    if sport == "bike" and candidates:
        mean = sum(v * weights[k] for k, v in scored.items()) / sum(weights[k] for k in scored)
        worst = min(candidates, key=lambda k: candidates[k])
        pulled = mean - WORST_COMPONENT_PULL * (mean - candidates[worst])
        if mean - score >= 1.5 and abs(pulled - score) <= 1.5:
            notes.append(f"These average {mean:.0f}. The weakest, {_PART_NAMES.get(worst, worst)} at "
                         f"{candidates[worst]:.0f}, pulls the score down to {score:.0f}.")
    return notes


def _bd_html(items: list[dict[str, Any]], notes: list[str] | None = None) -> str:
    """The cover's breakdown: one bar per component, coloured by its score.

    The weight shown is the share of what this clip counted, so the column adds
    to 100 and the sum can be done by hand: the scorer's own weights add to less,
    and it divides by the ones it used."""
    if not items:
        return ""
    total = sum(i["weight"] for i in items) or 1.0
    rows = "".join(
        f"<div class='bd-row'><span class='bd-n'>{_e(i['name'])}</span>"
        f"<span class='bd-b'><i style='width:{max(2, min(100, round(i['score'])))}%;background:"
        f"{'#3DBE74' if i['score'] >= 80 else '#E5A23A' if i['score'] >= 60 else '#F1553F'}'></i></span>"
        f"<span class='bd-v'>{round(i['score'])}</span><span class='bd-w'>{round(i['weight'] / total * 100)}%</span></div>"
        for i in items)
    note = "".join(f"<p class='bd-note'>{_e(n)}</p>" for n in notes or [])
    return (f"<div class='bd'><div class='bd-h'><span>Score built from</span>"
            f"<span>score · weight</span></div>{rows}{note}</div>")


# The range bar as the app draws it (gradientRangeBar in index.html): a hairline
# track, the green band, a marker at the value with the value above it, and the
# axis under it -- optimal +/- one warning margin +/- one more, widened to fit
# a value that falls outside.
_MARGIN_FLOOR = {"pelvic_ratio": 0.2, "pelvic_tilt": 0.2}


def _row_value(row: dict[str, Any]) -> str:
    """A number as the report prints it, the same on the tile and on a finding."""
    v, unit = row["value"], row.get("unit") or ""
    return f"{v:.0f}°" if unit == "°" else _num(v) + unit


def _grbar(row: dict[str, Any]) -> str:
    v, band, unit, pro = row["value"], row.get("band"), row.get("unit") or "", row.get("pro")
    if row.get("text"):
        return f"<div class='gr-text'><b>{_e(row['text'])}</b> {_e(row.get('note'))}</div>"
    if v is None:
        return "<div class='gr-none'>Not measured on this clip</div>"
    if not band:
        return f"<div class='gr-none'>{_e(_num(v))}{_e(unit)} · no reference range for this one</div>"
    lo, hi = band
    m = max((hi - lo) * 0.10, _MARGIN_FLOOR.get(row.get("field") or "", 3.0 if unit == "°" else (hi - lo) * 0.25 or 0.1))
    ax_min, ax_max = lo - 2 * m, hi + 2 * m
    pad = max((hi - lo) * 0.15, m)
    ax_min = min(ax_min, v - pad)
    ax_max = max(ax_max, v + pad)
    span = (ax_max - ax_min) or 1.0

    def pos(x: float) -> float:
        return max(0.0, min(100.0, (x - ax_min) / span * 100))

    # The colour is the row's verdict, not a second opinion on it: a verdict and
    # a colour worked out apart gave "In range" in red.
    z = row.get("status") if row.get("status") in ("good", "warn", "bad") else "good"
    mk = max(3.0, min(97.0, pos(v)))
    vtxt = _row_value(row)
    tick = (f"<span class='gr-pt' style='left:{pos(pro['median']):.1f}%'></span>"
            if pro and ax_min <= pro.get("median", ax_min - 1) <= ax_max else "")
    prol = (f"<div class='gr-pro'><span class='gr-pm'></span>Kona pros {_num(pro['median'])}{unit} "
            f"<small>· middle half {_num(pro['q1'])}–{_num(pro['q3'])}{unit}</small></div>") if pro else ""
    return (f"<div class='grbar'><div class='gr-chips'><span class='gr-chip {z}' style='left:{max(9, min(91, mk)):.1f}%'>{vtxt}</span></div>"
            f"<div class='gr-track'><span class='gr-band' style='left:{pos(lo):.1f}%;width:{max(1.0, pos(hi) - pos(lo)):.1f}%'></span>"
            f"<span class='gr-mk' style='left:{mk:.1f}%'></span>{tick}</div>"
            f"<div class='gr-scale'><span>{_num(ax_min)}</span><span class='gr-opt'>{_num(lo)}–{_num(hi)}{unit}</span>"
            f"<span>{_num(ax_max)}</span></div>{prol}</div>")


def _tile(row: dict[str, Any]) -> str:
    st = row.get("status") or "info"
    word = row.get("word") or _STATUS_WORD.get(st, "")
    return (f"<div class='tile'><div class='tile-h'><span class='tile-l'>{_e(row['label'])}</span>"
            f"<span class='chip {st}'>{_e(word)}</span></div>{_grbar(row)}</div>")


def _inline(s: str) -> str:
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    return re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", s)


def markdown(text: str) -> str:
    """The Markdown the coach writes: headings, bullet and numbered lists,
    paragraphs, **bold** and *italic*. Everything is escaped first."""
    out: list[str] = []
    para: list[str] = []
    items: list[str] = []

    def flush() -> None:
        if para:
            out.append("<p>" + _inline(" ".join(para)) + "</p>")
            para.clear()
        if items:
            out.append("<ul>" + "".join(f"<li>{_inline(i)}</li>" for i in items) + "</ul>")
            items.clear()

    for raw in str(text or "").splitlines():
        line = _e(raw.strip())
        if not line:
            flush()
            continue
        head = re.match(r"#{1,6}\s+(.*)", line)
        bullet = re.match(r"(?:[-*•]|\d+[.)])\s+(.*)", line)
        if head:
            flush()
            out.append(f"<h3>{_inline(head.group(1))}</h3>")
        elif bullet:
            if para:
                flush()
            items.append(bullet.group(1))
        else:
            if items:
                flush()
            para.append(line)
    flush()
    return "".join(out)


def _clean(text: Any) -> str:
    """Plan and finding text as the analyzers write it, in the report's words:
    the degree sign and the rounding of the tiles, no crank jargon."""
    s = re.sub(r"(-?\d+(?:\.\d+)?)\s*deg\b", lambda m: f"{float(m.group(1)):.0f}°", str(text or ""))
    s = re.sub(r"\bBDC\b", "the bottom of the stroke", s)
    s = re.sub(r"\bTDC\b", "the top of the stroke", s)
    return s.replace("about ≈", "about ")


def _status_of(severity: Any) -> str:
    return {"critical": "bad", "high": "bad", "warning": "warn", "moderate": "warn"}.get(
        str(severity or "").lower(), "warn")


def _range(item: dict[str, Any]) -> tuple[float, float] | None:
    r = item.get("range")
    if isinstance(r, (list, tuple)) and len(r) == 2:
        try:
            return float(r[0]), float(r[1])
        except (TypeError, ValueError):
            return None
    m = re.match(r"\s*(-?[\d.]+)\s*[-–]\s*(-?[\d.]+)", str(item.get("optimal_range") or ""))
    return (float(m.group(1)), float(m.group(2))) if m else None


# A run's numbers, under the names the cover's breakdown uses, with their units.
_RUN_ROW = {
    "cadence": ("Cadence", " spm"),
    "overstride": ("Overstride", ""),
    "vertical_osc": ("Vertical oscillation", " cm"),
    "trunk_lean": ("Trunk lean", "°"),
    "knee_contact": ("Knee at contact", "°"),
    "knee_swing": ("Knee in swing", "°"),
    "elbow_angle": ("Arm swing · elbow", "°"),
}


def _run_verdict(v: float, band: tuple[float, float] | None, metric: Any,
                 severity: Any, listed_good: bool) -> tuple[str, str | None]:
    """(status, word) for one row of a run, so the chip, the colour and the
    "already good" list can't disagree.

    The plan is lenient on purpose: a small miss is not a finding, and past the
    band on its harmless side is not a fault (_FAULT_DIRECTION). It isn't "in
    range" either, and a mark outside the green under those words reads as an error."""
    if band is None:
        return ("good" if listed_good else _status_of(severity)), None
    lo, hi = band
    if not listed_good:
        return ("bad" if _status_of(severity) == "bad" or _status(v, lo, hi) == "bad" else "warn"), None
    if lo <= v <= hi:
        return "good", None
    fault = _FAULT_DIRECTION.get(str(metric))
    if (fault == "high" and v < lo) or (fault == "low" and v > hi):
        return "good", "Better than range"
    return "warn", None


def plan_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    """The run's measurements, from the plan's own strong points and issues:
    each one already carries its value and the range it was judged against."""
    plan = result.get("training_plan") or {}
    rows: list[dict[str, Any]] = []
    for item, listed_good in [(i, True) for i in plan.get("strong_points") or []] + \
                             [(i, False) for i in plan.get("issues") or []]:
        v = item.get("value")
        metric = item.get("metric")
        # Categorical: the plan carries a placeholder 0.0 and a 0-0 "range".
        if not isinstance(v, (int, float)) or metric == "foot_strike":
            continue
        band = _range(item)
        named = _RUN_ROW.get(str(metric))
        label = named[0] if named else (
            item.get("label") or str(item.get("description") or metric or "").split(":")[0])
        lo, hi = band if band else (float(v) - 1, float(v) + 1)
        span = max(hi - lo, 1e-6)
        scale = (min(lo - span, float(v) - span * 0.2), max(hi + span, float(v) + span * 0.2))
        status, word = _run_verdict(float(v), band, metric, item.get("severity"), listed_good)
        rows.append({"label": label, "value": float(v), "unit": named[1] if named else "", "band": band,
                     "pro": None, "status": status, "word": word, "scale": scale, "field": metric})
    metrics = result.get("sport_specific_metrics") or {}
    strike = metrics.get("foot_strike")
    if rows and strike:
        made = "Estimated from 2D video. " if metrics.get("foot_strike_estimated") else ""
        rows.append({"label": "Foot strike", "value": None, "text": str(strike).capitalize() + ".",
                     "note": made + "Not graded: the landing pattern alone isn't a fault.",
                     "band": None, "pro": None, "status": "info", "word": "Not graded", "field": "foot_strike"})
    return rows


def _num(v: float) -> str:
    return f"{v:.0f}" if abs(v) >= 20 else f"{v:.2f}".rstrip("0").rstrip(".")


_COMPONENT = {"saddle_height": "Saddle height", "saddle_setback": "Saddle fore-aft", "saddle_fore_aft": "Saddle fore-aft",
              "bar_position": "Bar position", "bar_reach": "Bar reach", "cleats": "Cleats"}
_ACTION_TITLE = {"move_saddle_forward_far": "Move the saddle forward",
                 "reassess_after_fore_aft": "Set it after the fore-aft change",
                 "reassess_after_saddle_height": "Set it after the saddle height"}
# Same words as the app's FIT_EFFORT (index.html).
_EFFORT = {"home": "at home, 5 min", "depends": "home or bike shop", "shop": "bike shop"}


def _fit_priorities(fit: dict[str, Any]) -> list[dict[str, Any]]:
    """The fit plan's adjustments as plan cards, in the order to make them."""
    out = []
    for d in sorted(fit.get("diagnostics") or [], key=lambda d: d.get("priority") or 99):
        if d.get("status") != "needs_adjustment":
            continue
        action = _ACTION_TITLE.get(str(d.get("action")), str(d.get("action") or "").replace("_", " ").capitalize())
        amount = f" {d['amount']}" if d.get("amount") else ""
        size = re.search(r"(\d+)\s*mm", amount)
        if size and int(size.group(1)) > 5:
            amount += ", in 5 mm steps"
        comp = _COMPONENT.get(d.get("component"), str(d.get("component") or "").replace("_", " ").capitalize())
        drill = (action + amount).strip()
        effort = d.get("effort") or {}
        if drill and effort.get("level") in _EFFORT:
            drill += f" · {_EFFORT[effort['level']]}"
        why, do = _why_and_do(d, bool(size and int(size.group(1)) > 5))
        out.append({"label": comp, "drill": drill or None, "why": why, "instruction": do, "action": d.get("action"),
                    "article": d.get("article"),
                    "metric": d.get("metric_name")})
    return out


def _why_and_do(d: dict[str, Any], steps: bool) -> tuple[str, str | None]:
    """The fit plan's one sentence, split where it turns from the measurement to the
    change, in the builder's own position-aware words."""
    why, sep, do = str(d.get("reason") or "").rpartition(" -- ")
    if not sep:
        return do, None
    do = do[0].upper() + do[1:] if do else do
    if steps:
        do += " Work in 5 mm steps."
    note = (d.get("effort") or {}).get("note")
    return why + ".", f"{do} {note}" if note else do


# A saved run still carries the old drill: a march for knee lift under a cue about
# folding the leg. Same fix as the builder's.
_OLD_DRILL = {"High Knee March": {
    "drill": "Heel Flicks",
    "instruction": "Jog 3x20 m with short, quick steps, letting each heel flick up toward your glute.",
}}
# A finding about a number the plan already covers is the same advice twice.
_ISSUE_METRIC = {"insufficient_knee_drive": "knee_swing", "low_cadence": "cadence",
                 "excessive_forward_lean": "trunk_lean", "saddle_too_high": "knee_at_bdc",
                 "saddle_too_low": "knee_at_bdc", "trunk_too_aggressive": "trunk_angle",
                 "trunk_too_upright": "trunk_angle"}
# The app's measurement floor (TREND_METRICS.noise): what the pose model moves by on
# one clip run twice. Run metrics have no measured floor, so they count as changed
# only when their verdict does.
_ANGLE_NOISE = 3.0
_RANK = {"good": 0, "warn": 1, "bad": 2}
READ_EYEBROW = "The coach's read"


def _miss(row: dict[str, Any], v: float) -> float:
    lo, hi = row["band"]
    return max(lo - v, v - hi, 0.0)


def changes_since(rows: list[dict[str, Any]], before: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The numbers that moved between two like-for-like clips, by more than two clips
    of an unchanged position already differ by."""
    then = {r.get("field"): r for r in before if r.get("value") is not None}
    moved = []
    for row in rows:
        old = then.get(row.get("field"))
        if row.get("value") is None or old is None or not row.get("band"):
            continue
        v, was = float(row["value"]), float(old["value"])
        better = _RANK.get(row.get("status"), 0) - _RANK.get(old.get("status"), 0)
        if row.get("unit") == "°":
            if abs(v - was) < _ANGLE_NOISE:
                continue
        elif not better:
            continue
        gap = _miss(row, v) - _miss(row, was)
        if better:
            word, tone = ("Further out", "bad") if better > 0 else ("Back in range" if row.get("status") == "good" else "Closer", "good")
        elif gap:
            word, tone = ("Further out", "bad") if gap > 0 else ("Closer", "good")
        else:
            word, tone = "Still in range", "info"
        moved.append({"label": str(row["label"]).split(" · ")[0], "then": _row_value({**row, "value": was}),
                      "now": _row_value(row), "word": word, "tone": tone})
    return moved


# The coach's crank words, from text written before the prompt said not to.
_COACH_WORDS = [
    (re.compile(r"\s*\((?:BDC|TDC)\)"), ""),
    (re.compile(r"\bBDC knee angle\b"), "knee angle at the bottom of the stroke"),
    (re.compile(r"\bTDC knee angle\b"), "knee angle at the top of the stroke"),
    (re.compile(r"\b(?:at )?BDC\b"), "at the bottom of the stroke"),
    (re.compile(r"\b(?:at )?TDC\b"), "at the top of the stroke"),
    (re.compile(r"\bbottom of stroke\b"), "bottom of the stroke"),
    (re.compile(r"\btop of stroke\b"), "top of the stroke"),
]
# "Worth a look" lines name a metric the way the prompt does; the plan by field.
_LOOK_FIELD = [("knee angle at bottom", "knee_at_bdc"), ("knee angle at the bottom", "knee_at_bdc"),
               ("knee angle at top", "knee_at_tdc"), ("knee angle at the top", "knee_at_tdc"),
               ("hip angle at top", "hip_at_tdc"), ("hip angle at the top", "hip_at_tdc"),
               ("trunk angle", "trunk_angle"), ("elbow angle", "elbow_angle"),
               ("shoulder angle", "shoulder_angle"), ("cadence", "cadence"), ("trunk lean", "trunk_lean")]


def _coach_text(text: str, planned: set[str]) -> str:
    """The coach's text as the report prints it: crank jargon spelled out, and the
    checklist without the lines the plan below already says, so a fix isn't read
    four times in two sizes ("3-5 mm" there, "5 mm" here)."""
    for pat, rep in _COACH_WORDS:
        text = pat.sub(rep, text)
    out, in_look, kept = [], False, 0
    for line in text.splitlines():
        if re.match(r"\s*\**Worth a look\**\s*$", line):
            in_look, kept = True, 0
            out.append(line)
            continue
        if in_look and re.match(r"\s*\**(Next session|Overall|Fix first)", line):
            in_look = False
        if in_look and line.lstrip().startswith("- "):
            head = line.split("**")[1].lower() if line.count("**") >= 2 else ""
            field = next((f for k, f in _LOOK_FIELD if head.startswith(k)), None)
            if field and field in planned:
                continue
            kept += 1
        out.append(line)
    if not kept:
        out = [ln for ln in out if not re.match(r"\s*\**Worth a look\**\s*$", ln)]
    return "\n".join(out)


# The article behind a card, by the path the fit plan carries.
_ARTICLES = {"/academy/kona-2026-hip-over-crank": "Kona Pros Don't Sit Higher. They Sit Further Forward."}


def foot_line(meta: dict[str, Any], sport: str) -> str:
    """Whose report, on every page after the cover: athlete, sport, when filmed."""
    who = str(meta.get("athlete") or "").strip()
    return " · ".join(x for x in (who, _SPORT.get(sport, sport.title()),
                                  f"Filmed {meta['filmed']}" if meta.get("filmed") else "") if x)


def doc_title(result: dict[str, Any], meta: dict[str, Any]) -> str:
    sport = result.get("sport_type") or result.get("sport") or ""
    return _doc_title(meta, _SPORT.get(sport, sport.title()))


def _doc_title(meta: dict[str, Any], sport_name: str) -> str:
    """A browser names the saved file after this, so it says whose report it is. An
    account's email is left out: it isn't a name, and it would land in a file name."""
    who = str(meta.get("athlete") or "").strip()
    parts = [who if who and "@" not in who else "", sport_name, str(meta.get("filmed") or "")]
    return ", ".join(p for p in parts if p) or "Flapp report"


# The notes that make the whole clip doubtful, as against one number on it: a shaky
# camera costs vertical oscillation, the chainring side costs the knee.
_LOW_TRUST_CODES = {"framing", "duration", "position"}
_RESHOOT = {
    "duration": "keep the whole body in the frame for the whole clip",
    "position": "run the clip with the position actually ridden",
}


def _low_trust(film: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [n for n in film if n.get("level") == "bad" and n.get("code") in _LOW_TRUST_CODES]


def _reshoot(low: list[dict[str, Any]]) -> str:
    """What the next clip needs, from what was wrong with this one."""
    asks = []
    for n in low:
        if n["code"] == "framing":
            asks.append("film closer, so the body fills most of the frame"
                        if str(n.get("short")).startswith("small")
                        else "send the original file from the camera roll")
        else:
            asks.append(_RESHOOT[n["code"]])
    return "; ".join(asks)


def _fit_note(note: Any, changes: int) -> str:
    """The fit plan's sequence note. A saved report still carries the old "5mm
    only" beside "Lower saddle about 23 mm", and an order of one is no order."""
    s = str(note or "").replace("Each adjustment: 5mm only, ride 30 minutes before reassessing.", FIT_STEP_NOTE)
    return re.sub(r"Adjust in order: [^.]*\.\s*", "", s) if changes < 2 else s


# A finding about a number that also has a tile prints the tile's name and value:
# the analyzer's "Min knee" is the single lowest frame, the tile the stride's.
_ISSUE_ROW = {"insufficient_knee_drive": "knee_swing", "low_cadence": "cadence",
              "excessive_forward_lean": "trunk_lean"}


_EXTRA_CSS = """
/* A browser prints its date, title and the page's address in any page margin, and
   the box for it is ticked by default: every page of a coach's document carried its
   /r/ link. No margin, no room for them. A repeating table head and foot make the
   space instead; the cover bleeds to the edge and sits outside. */
@page{size:A4;margin:0}
@page:first{margin:0}
.flow{width:100%;border-collapse:collapse;break-before:page}
.flow td{padding:0}
.flow thead td{height:14mm}
.flow tfoot td{height:16mm;vertical-align:middle}
.run-foot{display:flex;justify-content:space-between;gap:12px;font-size:7.6pt;color:#6E7787;border-top:1px solid #DCE2EA;padding-top:5px}
.flow section:first-child{break-before:auto}
@media screen{
  .flow,.flow tbody,.flow tr,.flow td{display:block}
  .flow thead,.flow tfoot{display:none}
}
.screen{position:sticky;top:0;z-index:5;display:flex;gap:12px;align-items:center;justify-content:space-between;
  background:#14294B;color:#fff;margin:0 -15mm 0;padding:10px 15mm;font:600 10pt 'Manrope',sans-serif}
.screen button,.screen .dl{font:700 10pt 'Manrope',sans-serif;background:#F1553F;color:#fff;border:0;border-radius:6px;padding:10px 16px;cursor:pointer;text-decoration:none;white-space:nowrap}
@media print{.screen{display:none}}
/* A phone clip is tall. In a wide box it was 63% dark bars, so the frame takes the
   height it needs and the cards sit beside it. */
.lead{display:flex;gap:14px;align-items:flex-start;margin:14px 0 0}
.lead img{display:block;width:auto;max-width:100mm;max-height:150mm;border-radius:8px}
/* iPhone prints on US Letter with its own margins and ignores size:A4: the cover
   was two pages there. Sized for the smaller page. */
@media print{.lead img{max-height:108mm;max-width:80mm}}
.lead-side{flex:1;min-width:0}
.lead-side .glance{grid-template-columns:minmax(0,1fr);margin:0}
.lead-side .cap{margin:10px 0 0}
.glance{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:14px 0 0}
.glance .card{border-top:3px solid #2F6DE0}
.glance .card.fix{border-top-color:#F1553F}
.glance .card.ok{border-top-color:#2E9E5B}
.glance b{display:block;font:italic 800 11pt 'Archivo',sans-serif;color:#14294B;margin:2px 0 3px}
/* A heading stays with what it heads: break-after alone doesn't hold inside the
   table the pages flow in. */
.keep{break-inside:avoid}
.since{border:1px solid #DCE2EA;border-radius:8px;padding:10px 14px;margin:0 0 14px;break-inside:avoid}
.since-score{font:italic 800 13pt 'Archivo',sans-serif;color:#14294B;margin:2px 0 6px}
.since-row{display:grid;grid-template-columns:minmax(0,1fr) auto auto;gap:10px;align-items:center;padding:3px 0;border-top:1px solid #EEF1F5;font-size:9pt}
.since-v{font:500 8.6pt 'IBM Plex Mono',monospace;color:#14294B}
.since p:last-child{font-size:8pt;color:#6E7787;margin:6px 0 0}
.finding{display:grid;grid-template-columns:auto 1fr;gap:4px 12px;border:1px solid #DCE2EA;border-radius:8px;padding:10px 14px;margin:0 0 8px;break-inside:avoid}
.finding h3{margin:0}
.finding p{grid-column:1/-1;margin:0}
.finding .chip{justify-self:start}
.coachtext{border-left:3px solid #2F6DE0;padding:2px 0 2px 14px;margin:0 0 14px}
.coachbrand{display:flex;align-items:center;gap:12px;margin:0 0 6px}
.coachbrand img{height:44px;width:auto;max-width:150px;background:#fff;border-radius:4px;padding:3px}
.coachbrand b{display:block;font:600 14pt 'IBM Plex Sans',sans-serif;color:#fff}
.coachbrand small{display:block;font-size:8pt;color:#C5D3EA}
.coachnote{background:#F3F5F8;border-left:3px solid #14294B;border-radius:4px;padding:10px 14px;margin:0 0 14px;break-inside:avoid}
.coachnote p{margin:4px 0 0}
.coachtext h3{margin:12px 0 4px}
.coachtext h3:first-child{margin-top:0}
.ok-list li{margin:0 0 3px}
.hob{display:grid;grid-template-columns:auto 1fr;gap:16px;align-items:center;background:#F3F5F8;border-radius:8px;padding:12px 16px;margin:4px 0 12px;break-inside:avoid}
.prio a{color:#2459C2;font-weight:600;text-decoration:none}
.prio .url{display:block;font:7.6pt 'IBM Plex Mono',monospace;color:#6E7787}
.hob .big{font:italic 900 26pt/1 'Archivo',sans-serif;color:#14294B}
.cover-grid3{display:grid;grid-template-columns:auto minmax(0,1fr) minmax(0,1.15fr);gap:18px;align-items:center;margin:16px 0 4px}
.brand{font-size:17pt;gap:0}
.bd-h span{white-space:nowrap}
.bd{border-left:1px solid #2A4170;padding-left:16px}
.bd-h{display:flex;justify-content:space-between;gap:10px;font:500 7pt 'IBM Plex Mono',monospace;letter-spacing:.1em;text-transform:uppercase;color:#8EA6CF;margin:0 0 5px}
.bd-note{font-size:7.4pt;line-height:1.35;color:#C5D3EA;margin:6px 0 0}
.bd-row{display:grid;grid-template-columns:minmax(0,1fr) 14mm 7mm 8mm;gap:6px;align-items:center;font-size:8.4pt;line-height:1.3;padding:1.5px 0;color:#fff}
.bd-n{overflow-wrap:anywhere}
.bd-b{height:5px;background:#2A4170;border-radius:3px;overflow:hidden}
.bd-b i{display:block;height:100%;border-radius:3px}
.bd-v{font:600 8.4pt 'IBM Plex Mono',monospace;text-align:right}
.bd-w{font:8pt 'IBM Plex Mono',monospace;color:#8EA6CF;text-align:right}
.tiles{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:4px 0 12px}
.tile{border:1px solid #DCE2EA;border-radius:8px;padding:10px 12px 8px;break-inside:avoid}
/* Safari cuts a grid item across pages whatever break-inside says; an inline
   block it can't. */
@media print{.tiles{display:block;font-size:0}.tile{display:inline-block;vertical-align:top;width:calc(50% - 5px);margin:0 0 10px;font-size:9.6pt}.tile:nth-child(odd){margin-right:10px}}
.tile-h{display:flex;justify-content:space-between;align-items:flex-start;gap:8px}
.tile-l{font-weight:700;color:#14294B;font-size:9.6pt}
.grbar{margin-top:6px}
.gr-chips{position:relative;height:15px}
.gr-chip{position:absolute;transform:translateX(-50%);white-space:nowrap;font:600 9.4pt 'IBM Plex Mono',monospace}
.gr-chip.good{color:#1F7A45}.gr-chip.warn{color:#9A5E0B}.gr-chip.bad{color:#B8371F}
.gr-track{position:relative;height:10px;margin-top:2px}
.gr-track:before{content:"";position:absolute;left:0;right:0;top:4px;height:2px;background:#B9C2CF}
.gr-band{position:absolute;top:1px;height:8px;background:#DDEFE3;border:1px solid #2E9E5B;border-radius:2px;box-sizing:border-box}
.gr-mk{position:absolute;top:-3px;width:2px;height:16px;background:#14181F;transform:translateX(-1px)}
.gr-pt{position:absolute;top:-2px;width:2px;height:14px;background:#14294B;opacity:.55;transform:translateX(-1px)}
.gr-scale{display:flex;justify-content:space-between;font:7.6pt 'IBM Plex Mono',monospace;color:#6E7787;margin-top:3px}
.gr-scale .gr-opt{color:#1F7A45;font-weight:600}
.gr-pro{font:7.8pt 'IBM Plex Mono',monospace;color:#5C6675;margin-top:4px}
.gr-pro small{color:#8A93A1}
.gr-pm{display:inline-block;width:2px;height:9px;background:#14294B;opacity:.55;margin-right:5px;vertical-align:-1px}
.gr-none{font-size:8.4pt;color:#8A93A1;margin-top:8px}
.gr-text{font-size:8.6pt;color:#5C6675;margin-top:8px}
.gr-text b{color:#14294B;font-weight:700}
.facts dd{overflow-wrap:anywhere;min-width:0}
.foot{margin:18px 0 0;padding-top:10px;border-top:1px solid #DCE2EA;font-size:8pt;color:#6E7787;display:flex;justify-content:space-between}
@media print{.foot{display:none}}
.filmnote{background:#FFF3DC;border-left:3px solid #E5A23A;border-radius:4px;padding:10px 14px;margin:0 0 14px;break-inside:avoid}
.filmnote ul{margin:4px 0 0}
.filmnote li{margin:0 0 3px}
/* The document is A4 in millimetres, and a coach's link opens on a phone. Screen
   only: Save as PDF prints the A4 page from any device. */
@media screen and (max-width:720px){
  body{padding:0 16px;font-size:14px}
  .screen{margin:0 -16px;padding:10px 16px;font-size:13px}
  .screen button,.screen .dl{flex:none;font-size:13px}
  .cover-band{margin:0 -16px;padding:24px 16px 20px}
  .cover-band h1{font-size:30px}
  .cover-grid3{grid-template-columns:auto minmax(0,1fr);gap:14px}
  .ring{width:96px;height:96px}
  .facts{font-size:13px;gap:3px 10px}
  .facts dd{overflow-wrap:anywhere;min-width:0}
  .bd{grid-column:1/-1;border-left:0;padding-left:0;border-top:1px solid #2A4170;padding-top:12px}
  .bd-row{grid-template-columns:minmax(0,1fr) 72px 26px 32px;font-size:13px}
  .lead{flex-direction:column}
  .lead img{width:100%;max-width:100%;max-height:none}
  .glance,.tiles,.two{grid-template-columns:minmax(0,1fr)}
  .prio dl{grid-template-columns:minmax(0,1fr)}
  .prio dt{padding-top:6px}
  .legend,.foot{flex-wrap:wrap;gap:6px 14px}
  h2{font-size:24px}
  h3{font-size:16px}
}
"""


def _rows_of(result: dict[str, Any], sport: str) -> list[dict[str, Any]]:
    return metric_rows(result) if sport == "bike" else plan_rows(result)


def card_key(p: dict[str, Any]) -> str:
    """A plan card's name for the coach's edits: what it measures and what it asks."""
    return f"{p.get('metric') or p.get('label') or ''}|{p.get('action') or p.get('drill') or ''}"


def _apply_card_edits(prios: list[dict[str, Any]], cards: Any) -> list[dict[str, Any]]:
    """The plan as the coach left it: their order, their words, without the cards
    they hid. A card they never saw (the plan changed after they edited it) keeps
    Flapp's words and goes last."""
    if not isinstance(cards, list):
        return prios
    by_key = {p["key"]: p for p in prios}
    out = []
    for c in cards:
        p = by_key.pop(str((c or {}).get("key")), None) if isinstance(c, dict) else None
        if p is None or c.get("hidden"):
            continue
        p = dict(p)
        for field, slot in (("title", "drill"), ("why", "why"), ("do", "instruction")):
            if isinstance(c.get(field), str):
                p[slot] = c[field].strip() or None
        out.append(p)
    return out + list(by_key.values())


def render_report_html(result: dict[str, Any], meta: dict[str, Any],
                       out: dict[str, Any] | None = None) -> str:
    """The whole document. ``meta``: athlete, setup, filmed, prepared,
    position_label, keyframe and kinogram data URIs, and ``previous``:
    {filmed, result} of the last like-for-like clip, when there is one.

    A coach's report carries ``edits`` (result.coach_edits: their words over
    Flapp's), ``draft`` (not sent yet) and ``reviewed`` ("name · date"). ``out``,
    when given, gets the plan cards and the coach text as Flapp wrote them, for
    the coach's review screen."""
    sport = result.get("sport_type") or result.get("sport") or ""
    plan = result.get("training_plan") or {}
    # Filtered here as well as in the builder: a saved run still has it leading.
    prios = [{**p, **_OLD_DRILL.get(str(p.get("drill")), {})} for p in plan.get("top_3_priorities") or []
             if p.get("metric") != "foot_strike"]
    strong = plan.get("strong_points") or []
    # Read against the crank here too: a crank marked before this rule existed
    # left a plan saying 5 mm.
    hob = (result.get("sport_specific_metrics") or {}).get("hip_over_bb") or result.get("hip_over_bb")
    fit = apply_hip_over_bb(result.get("fit_plan") or {}, hob, result.get("cycling_position")) or {}
    if sport == "bike" and not prios:
        # A bike's plan is its fit plan: what to adjust, in order, by how much.
        prios = _fit_priorities(fit)
        strong = strong or [{"description": g.get("label") or g.get("metric")}
                            for g in fit.get("good_metrics") or []]
    issues = result.get("detected_issues") or []
    coach = ((result.get("ai_recommendations") or {}).get("report")
             or (result.get("ai_recommendations") or {}).get("summary") or "")
    rows = _rows_of(result, sport)
    sc = result.get("score") if isinstance(result.get("score"), dict) else {}
    score = result.get("technique_score") if result.get("technique_score") is not None else sc.get("overall_score")
    grade = result.get("letter_grade") or sc.get("grade")
    side = result.get("camera_side")
    position = meta.get("position_label") or str(result.get("cycling_position") or "").replace("_", " ")

    facts = [
        ("Athlete", meta.get("athlete")),
        ("Sport", " · ".join(x for x in (_SPORT.get(sport, sport.title()), position if sport == "bike" else "") if x)),
        ("Setup", meta.get("setup")),
        ("Filmed", meta.get("filmed")),
        ("Camera", ("both sides" if side == "both" else f"{side} side"
                    + (" · chainring side" if sport == "bike" and side == "right" else "")) if side else None),
        ("Prepared", str(meta.get("prepared") or "").removeprefix("Prepared ")),
    ]
    def head(eyebrow: str, title: str) -> str:
        return f"<div class='head'><div class='eyebrow'>{_e(eyebrow)}</div><h2>{_e(title)}</h2><div class='speed'></div></div>"

    # Read from the capture report here, so a report saved before these notes
    # existed carries them as well. ``capture_bars`` is the coach's own grid.
    film = film_notes_of(result, meta.get("capture_bars")) or []
    film_codes = {n.get("code") for n in film}
    low = _low_trust(film)
    # Advice worked out against the wrong ranges is worse than none: "raise the bars"
    # to a rider on aero bars. The numbers stay, under a note that says what they are.
    withheld = result.get("score_withheld") if isinstance(result.get("score_withheld"), dict) else {}
    wrong_ranges = "position" in film_codes or withheld.get("reason") == "position_mismatch"
    if wrong_ranges:
        prios, strong, issues = [], [], []
    for p in prios:
        p["key"] = card_key(p)
    base_prios = list(prios)
    edits = meta.get("edits") if isinstance(meta.get("edits"), dict) else {}
    prios = _apply_card_edits(prios, edits.get("cards"))
    if meta.get("reviewed"):
        facts.append(("Reviewed", meta["reviewed"]))
    if low:
        facts.append(("Reliability", "Low · " + ", ".join(str(n["short"]) for n in low)))
    facts_html = "".join(f"<dt>{_e(k)}</dt><dd>{_e(v)}</dd>" for k, v in facts if v)

    # Everything that counts or lists the numbers reads the same rows, so the
    # cover, the "leave these alone" line and the tiles can't disagree.
    by_field = {r.get("field"): r for r in rows}
    graded = [r for r in rows if r.get("value") is not None and r.get("band")]
    has_pro = any(r.get("pro") for r in rows)
    if rows:
        ok_names = [str(r["label"]).split(" · ")[0] for r in graded if r.get("status") == "good"]
    else:
        ok_names = [n for n in (str(s.get("description") or "").split(":")[0].split(" (")[0] for s in strong) if n]

    def prio_label(p: dict[str, Any]) -> str:
        named = _RUN_ROW.get(str(p.get("metric")))
        return named[0] if named else str(p.get("label") or "")

    if wrong_ranges:
        ok_names = []
    # Against the last clip of the same setup, side and position (me._previous_clip).
    # A doubtful clip gets none: what moved is the one thing it can't say.
    previous = meta.get("previous") if not low else None
    moved = changes_since(rows, _rows_of(previous["result"], sport)) if previous else []
    then_score = (previous or {}).get("result", {}).get("technique_score")
    show_scores = bool(moved) and isinstance(then_score, (int, float)) and isinstance(score, (int, float))
    since_card = (f"<div class='card'><div class='eyebrow'>Since {_e(previous['filmed'])}</div>"
                  f"<b>{f'{then_score:.0f} → {score:.0f}' if show_scores else 'No real change'}</b>"
                  + (f"{len(moved)} measurement{'s' if len(moved) != 1 else ''} moved" if moved
                     else "Nothing moved beyond the method's own error") + "</div>") if previous else ""
    first = prios[0] if prios else None
    glance = "".join([
        f"<div class='card fix'><div class='eyebrow'>Film again first</div><b>{_e(str(low[0]['short']).capitalize())}</b>"
        "Confirm on a better clip before changing anything</div>" if low else
        f"<div class='card fix'><div class='eyebrow'>Fix first</div><b>{_e(prio_label(first))}</b>{_e(_clean(first.get('cue') or first.get('drill') or first.get('why')))}</div>"
        if first else "",
        since_card,
        f"<div class='card ok'><div class='eyebrow'>Already working</div><b>{_e(ok_names[0])}</b>"
        f"{len(ok_names)} measurement{'s' if len(ok_names) != 1 else ''} where {'they' if len(ok_names) != 1 else 'it'} should be</div>" if ok_names else "",
        f"<div class='card'><div class='eyebrow'>Measured</div><b>{len(graded)} number{'s' if len(graded) != 1 else ''}</b>"
        f"each against its range{', with the Kona pro field' if has_pro else ''}</div>" if graded else "",
    ])

    # A coach's report carries the coach's mark where Flapp sat; Flapp moves to
    # a line under it, as the instrument rather than the author.
    coach_brand = meta.get("coach") or {}
    logo = coach_brand.get("logo") or ""
    brand = (
        "<div class='coachbrand'>"
        + (f"<img src='{_e(logo)}' alt=''>" if str(logo).startswith("data:image/") else "")
        + f"<div><b>{_e(coach_brand.get('name'))}</b>"
        + (f"<small>{_e(coach_brand.get('tagline'))}</small>" if coach_brand.get("tagline") else "")
        + "<small>Measured with Flapp · getflapp.com</small></div></div>"
    ) if (coach_brand.get("name") or logo) else '<div class="brand">Flapp<span class="dot">.</span></div>'
    note = str(meta.get("coach_note") or "").strip()
    note_by = meta.get("coach_note_by") or coach_brand.get("name") or "Your coach"
    note_html = (
        f"<div class='coachnote'><div class='eyebrow'>{_e(note_by)}’s notes</div>"
        + "".join(f"<p>{_e(p)}</p>" for p in note.split("\n") if p.strip())
        + "</div>"
    ) if note else ""
    film_html = (
        "<div class='filmnote'><div class='eyebrow'>How this clip was filmed</div><ul>"
        + "".join(f"<li>{_e(n['text'])}</li>" for n in film)
        + "</ul></div>"
    ) if film else ""

    cover = f"""
<section>
  <div class="screen"><span>{"Draft · only you see it until you approve and send it" if meta.get("draft") else "Your Flapp report · use Print, then “Save as PDF”"}</span>{f'<a class="dl" href="{_e(meta["pdf_url"])}">Download PDF</a>' if meta.get("pdf_url") else '<button onclick="window.print()">Save as PDF</button>'}</div>
  <div class="cover-band">
    {brand}
    <h1>Technique Report</h1>
    <div class="sub">{"What this clip suggests, and what to film next." if low else "What to fix, in what order — and the numbers behind it."}</div>
    <div class="cover-grid3">{_ring(score, grade)}<dl class="facts">{facts_html}</dl>{_bd_html(breakdown_items(result), _score_notes(result, score))}</div>
  </div>
  {f'<div class="lead"><img src="{meta["keyframe"]}" alt="A frame from the clip, with its angles drawn on it"><div class="lead-side"><div class="glance">{glance}</div><p class="cap">One moment from the clip. The report reads its numbers across the whole clip, so they can differ from these.</p></div></div>' if meta.get('keyframe') else (f'<div class="glance">{glance}</div>' if glance else '')}
</section>"""

    def finding(i: dict[str, Any]) -> str:
        row = by_field.get(_ISSUE_ROW.get(str(i.get("type"))))
        if row and row.get("value") is not None:
            title, value, status = str(row["label"]).split(" · ")[0], _row_value(row), row.get("status")
        else:
            title = str(i.get("type") or "").replace("_", " ").capitalize()
            value, status = _clean(i.get("value")), _status_of(i.get("severity"))
        return (f"<div class='finding'><h3>{_e(title)}</h3><span class='chip {status}'>{_e(value)}</span>"
                f"<p>{_e(_clean(i.get('recommendation')))}</p></div>")

    # A number outside its range that neither the plan nor a finding mentions was
    # just a red tile on the last page, however much it cost the score.
    planned = {str(p.get("metric") or "").removesuffix("_avg") for p in base_prios}
    named = " ".join(str(i.get("type") or "") for i in issues)
    who = "your coach" if coach_brand.get("name") else "a coach or fitter" if sport == "bike" else "a coach"

    def unplanned(r: dict[str, Any]) -> str:
        lo, hi = r["band"]
        return (f"<div class='finding'><h3>{_e(str(r['label']).split(' · ')[0])}</h3><span class='chip bad'>{_e(_row_value(r))}</span>"
                f"<p>Outside its range of {_e(_num(lo))}–{_e(_num(hi))}{_e(r.get('unit') or '')}. No single change follows from this "
                f"number alone, so it isn't in the plan. Worth going through with {who}.</p></div>")

    findings = "" if wrong_ranges else "".join([
        finding(i) for i in issues if _ISSUE_METRIC.get(str(i.get("type"))) not in planned] + [
        unplanned(r) for r in graded if r.get("status") == "bad"
        and str(r.get("field")).removesuffix("_avg") not in planned
        and str(r.get("field")).split("_")[0] not in named])
    since_rows = "".join(
        f"<div class='since-row'><span>{_e(m['label'])}</span><span class='since-v'>{_e(m['then'])} → {_e(m['now'])}</span>"
        f"<span class='chip {m['tone']}'>{_e(m['word'])}</span></div>" for m in moved)
    since_html = (
        f"<div class='since'><div class='eyebrow'>Since {_e(previous['filmed'])} · same side, same position</div>"
        + (f"<div class='since-score'>{then_score:.0f} → {score:.0f}</div>" if show_scores else "")
        + since_rows
        + ("<p>Two clips of an unchanged position differ by up to 3°, so smaller moves aren't listed.</p>" if moved
           else "<p>No measurement moved by more than two clips of an unchanged position differ by.</p>")
        + "</div>"
    ) if previous else ""
    flapp_text = _coach_text(coach, planned) if coach else ""
    shown = edits.get("coachText") if isinstance(edits.get("coachText"), str) else flapp_text
    if out is not None:
        out["coach"] = flapp_text
        out["cards"] = [{"key": p["key"], "label": prio_label(p), "title": _clean(p.get("drill")) or "",
                         "why": _clean(p.get("why")) or "", "do": _clean(p.get("instruction")) or ""}
                        for p in base_prios]
    read_body = "".join([
        note_html, film_html, since_html,
        f'<div class="coachtext">{markdown(shown)}</div>' if shown.strip() else "",
        ("<div class='eyebrow' style='margin-top:6px'>What the measurements flagged</div>" + findings) if findings else "",
    ])
    # With nothing to say the plan doesn't already say, the plan comes first.
    read_title = "What this clip suggests" if low else "What's off, and what to change first"
    read = f"<section>{head(READ_EYEBROW, read_title)}{read_body}</section>" if read_body else ""

    prio_cards = [
        f"<div class='prio'><div class='n'>{n}</div><div><h3>{_e(prio_label(p))}{' — ' + _e(_clean(p.get('drill'))) if p.get('drill') else ''}</h3><dl>"
        + "".join(f"<dt>{lbl}</dt><dd>{_e(_clean(p.get(k)))}</dd>" for k, lbl in
                  (("why", "Why"), ("instruction", "Do this"), ("cue", "Think")) if p.get(k))
        + (f"<dt>For</dt><dd>{_e(p.get('weeks'))} weeks</dd>" if p.get('weeks') and not low else "")
        + (f"<dt>Read</dt><dd><a href='https://getflapp.com{_e(p['article'])}'>{_e(_ARTICLES[p['article']])}</a> "
           f"<span class='url'>getflapp.com{_e(p['article'])}</span></dd>" if p.get("article") in _ARTICLES else "")
        + "</dl></div></div>"
        for n, p in enumerate(prios, 1)]
    prio_html = "".join(prio_cards)
    weeks = [p.get("weeks") for p in prios if isinstance(p.get("weeks"), (int, float))]
    when = (f"In {int(min(weeks))} week{'s' if min(weeks) != 1 else ''}" if weeks
            else "After each change" if sport == "bike" and prios else "")
    note = _fit_note(fit.get("fitting_sequence_note"), len(prios)) if sport == "bike" else ""
    # The same note that says how this clip was filmed decides how the next one
    # should be: "same side" under "film the other side" was two orders at once.
    closer = any(n.get("code") == "framing" and str(n.get("short")).startswith("small") for n in film)
    how = (f"same camera height, same side, same speed{', but closer' if closer else ''}. "
           "Flapp puts the two clips side by side and shows what moved.")
    retest = (f"<div class='retest'><div class='eyebrow' style='color:#2459C2'>Film again</div>"
              f"<p>{_e(when)}: {_e(how)}{' ' + _e(note) if note else ''}</p></div>") if when else ""
    if low:
        retest = (f"<div class='retest'><div class='eyebrow' style='color:#2459C2'>Film again first</div>"
                  f"<p>Before changing anything: {_e(_reshoot(low))}. Flapp puts the two clips side by side "
                  "and shows whether this holds.</p></div>")
    # What already works sits on the numbers page as one line: on its own it
    # was a block that spilled onto a page of its own.
    ok_html = ("<div class='working' style='margin:0 0 12px'><h3>Already good — leave these alone</h3>"
               f"<p>{_e(', '.join(ok_names))}.</p></div>") if ok_names else ""
    if not (read or prio_html or low):
        read = (f"<section>{head(READ_EYEBROW, 'Nothing to change')}<p class='lead'>Nothing in this clip is "
                "outside its range. Keep filming the same way and watch the trend.</p></section>")
    plan_sec = f"""
<section>
  <div class="keep">{head('The plan', ('What to try, once a better clip confirms it' if prio_html else 'Film this again first') if low else 'Fix these, in this order')}
  {retest if low else (prio_cards[0] if prio_cards else '')}</div>
  {''.join(prio_cards) if low else ''.join(prio_cards[1:])}
  {'' if low else retest}
</section>""" if (prio_html or low) else ""

    tiles = "".join(_tile(r) for r in rows)
    hob_html = ""
    if isinstance(hob, dict) and hob.get("angle_deg") is not None:
        if hob.get("aero") is False:
            # The field was read on aero bars; a road position sits further back
            # by design (biomechanics/hip_over_bb), so no comparison is drawn.
            text = ("Measured from the back, over the crank axle. The Kona 2026 field was read on "
                    "aero bars, so it is not a reference for a road position; use this number to "
                    "compare your own setups.")
        else:
            gap = hob.get("gap_pro_cm")
            where = (f"{abs(gap):.1f} cm {'ahead of' if (hob.get('gap_pro_deg') or 0) > 0 else 'behind'} the Kona pros' middle"
                     if isinstance(gap, (int, float)) else "")
            text = f"Measured from the back, over the crank axle. {where}. Pros 88°, age-groupers 85°."
        hob_html = (f"<div class='hob'><div class='big'>{_e(hob['angle_deg'])}°</div><div><h3>Hip over the crank</h3>"
                    f"<p style='margin:0'>{_e(text)}</p></div></div>")
    legend = ("<div class='legend'><span><i><svg width='26' height='10'><rect x='1' y='3' width='24' height='4' fill='#DDEFE3' stroke='#2E9E5B'/></svg></i>Healthy range</span>"
              "<span><i><svg width='6' height='12'><rect x='2' y='0' width='2' height='12' fill='#14181F'/></svg></i>You</span>"
              + ("<span><i><svg width='6' height='12'><rect x='2' y='0' width='2' height='12' fill='#14294B' opacity='.6'/></svg></i>Kona 2026 pros' median</span>" if has_pro else "")
              + "</div>")
    kino = meta.get("kinogram")
    # A fitter knows the knee range by heart: a wider one has to say why, here.
    knee_band = (result.get("reference_bands") or {}).get("knee_at_bdc") or {}
    knee_note = (f"<p class='lead' style='font-size:9.5pt'>{_e(NON_DRIVE_KNEE_NOTE)}</p>"
                 if NON_DRIVE_KNEE_NOTE in str(knee_band.get("source") or "") else "")
    numbers = f"""
<section>
  <div class="keep">{head('The numbers', 'Every measurement, against its range')}
  <p class="lead">Measured by Flapp from your clip. Ranges come from sports-science research{' and the Kona 2026 pro field' if has_pro else ''}.</p>
  {ok_html}
  {hob_html}
  {legend if rows else ''}{knee_note}</div>
  {f"<div class='tiles'>{tiles}</div>" if rows else ''}
  {f'<div class="keep"><div class="eyebrow" style="margin-top:12px">{"One stride, five positions" if sport == "run" else "One pedal revolution"}</div><figure class="fig"><img src="{kino}" alt="Kinogram"></figure></div>' if kino else ''}
  <div class="foot"><span>Flapp · getflapp.com</span><span>{_e(meta.get('prepared'))}</span></div>
</section>""" if (rows or kino) else ""

    run_foot = (f"<div class='run-foot'><span>{_e(foot_line(meta, sport))}</span>"
                "<span>Measured with Flapp · getflapp.com</span></div>")
    # Rendered to a file by the PDF service: its own footer carries the page
    # count in the bottom margin, so the document's footer row goes.
    pdf_css = ".flow tfoot{display:none}.foot{display:none}" if meta.get("pdf") else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{_e(_doc_title(meta, _SPORT.get(sport, sport.title())))}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link href="https://fonts.googleapis.com/css2?family=Archivo:ital,wght@0,700;0,800;1,800;1,900&family=Manrope:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@600&display=swap" rel="stylesheet">
<style>{_CSS}{_EXTRA_CSS}{pdf_css}</style></head><body>{cover}<table class="flow"><thead><tr><td></td></tr></thead><tbody><tr><td>{read}{plan_sec}{numbers}</td></tr></tbody><tfoot><tr><td>{run_foot}</td></tr></tfoot></table></body></html>"""
