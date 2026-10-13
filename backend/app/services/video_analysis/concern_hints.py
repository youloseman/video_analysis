"""Every reading outside its band, with what to check -- under every coach report.

The first paid Coach Notes order (2026-10-11) arrived on a report whose AI
coach had written that the knee at the top of the stroke (85, band 60-72) and
the pelvic ratio (4.9, band 2-4) sat outside their bands -- and then headed the
section "Nothing to fix". An athlete looking for room to improve was told there
was none, beside a table showing two red rows.

The model may choose what leads. It does not get to choose what the athlete
sees: this list is built from the same per-metric statuses as the results
table, after the model has written, so no prompt wording can talk it away.

A hint says what usually causes the reading and what to check or try first. It
is not a diagnosis -- 2D pose numbers are good to a few degrees -- and where the
honest answer is "this is often deliberate", it says so.
"""
from __future__ import annotations

import re
from typing import Any

# (field, side) -> what to check. Field names are the summary fields served by
# reference_bands(); photo keys are mapped onto them in photo_field().
_HINTS: dict[tuple[str, str], str] = {
    # --- bike -------------------------------------------------------------
    ("knee_at_bdc", "high"): (
        "The saddle may be a touch high. Lower it 3-5 mm, ride 30 minutes and "
        "re-film. Hips rocking on the saddle or toes pointing down hard at the "
        "bottom are the tell-tales."
    ),
    ("knee_at_bdc", "low"): (
        "The saddle may be low, which loads the front of the knee. Raise it "
        "3-5 mm at a time and re-film after each change."
    ),
    ("knee_at_tdc", "high"): (
        "The knee does not close much at the top. The usual causes are short "
        "cranks (often a deliberate choice on a TT bike) or a saddle on the high "
        "side. If the knee at the bottom also sits at the top of its band, try "
        "the saddle 3-5 mm lower first. On cranks of 165 mm or shorter this "
        "reading is expected and not a fault."
    ),
    ("knee_at_tdc", "low"): (
        "The knee closes tightly at the top of the stroke. Check the saddle is "
        "not low first; if saddle height is right, shorter cranks are the usual "
        "fix."
    ),
    ("hip_at_tdc", "low"): (
        "The hip closes hard at the top of the stroke, which costs power and "
        "can pinch. Moving the saddle forward, raising the front a little or "
        "shorter cranks all open it -- each changes something else, so one at "
        "a time."
    ),
    ("trunk_angle_avg", "low"): (
        "A very low torso is fast but hard to hold. Test it: can you breathe "
        "and hold race power in it for as long as your race lasts? If not, "
        "raise the front 5-10 mm."
    ),
    ("trunk_angle_avg", "high"): (
        "A tall torso costs speed. If you are comfortable, lower the front "
        "5-10 mm at a time and re-check that the hip angle stays in its band."
    ),
    ("shoulder_angle_avg", "low"): (
        "Elbows sit behind the shoulders. That is often a deliberate choice; if "
        "your neck or shoulders tire, move the arm support forward 1-2 cm."
    ),
    ("shoulder_angle_avg", "high"): (
        "The arms reach well forward, which loads the neck and shoulders. "
        "Bring the arm support or bars back 1-2 cm."
    ),
    ("elbow_angle_avg", "low"): (
        "Elbows are tightly bent, which usually means the reach is short. Check "
        "the reach before anything else."
    ),
    ("elbow_angle_avg", "high"): (
        "Arms are close to straight: the reach may be long, or you are locking "
        "out. Soften the elbows; if you cannot, shorten the reach."
    ),
    ("forearm_tilt_avg", "high"): (
        "The forearms tilt steeply up. Check it is legal under your race's "
        "rules, and that the shoulders stay relaxed rather than shrugged."
    ),
    ("forearm_tilt_avg", "low"): (
        "The forearms are close to flat. A few degrees of upward tilt often "
        "lets the head and shoulders drop. Optional, and worth testing."
    ),
    ("pelvic_ratio", "high"): (
        "The torso is low compared with how far the hip closes, so you get low "
        "mostly by rolling the pelvis forward or rounding the back. If the "
        "saddle nose is uncomfortable, a split-nose saddle or the nose 1-2 deg "
        "down usually helps."
    ),
    ("pelvic_ratio", "low"): (
        "The hip closes a lot for how low the torso is, which usually means the "
        "pelvis sits upright. Rolling it forward (saddle nose 1-2 deg down, or a "
        "split-nose saddle) opens the hip."
    ),
    ("helmet_above_back", "high"): (
        "The helmet sits higher above the back than 28 of the 31 Kona pros. "
        "Dropping the head a little cuts drag, as long as you can still see the "
        "road."
    ),
    # --- run --------------------------------------------------------------
    ("cadence_spm", "low"): (
        "Try about 5% more steps per minute with a metronome, in short blocks. "
        "It usually shortens an overstride too."
    ),
    ("cadence_spm", "high"): (
        "A high cadence is rarely a problem. It is only worth a look if your "
        "stride feels choppy at easy pace."
    ),
    ("trunk_lean_avg", "low"): (
        "You run very upright. A slight forward lean from the ankles, not the "
        "waist, helps."
    ),
    ("trunk_lean_avg", "high"): (
        "You lean well forward, usually from the waist. Run tall and lean from "
        "the ankles."
    ),
    ("vertical_oscillation_cm", "high"): (
        "You bounce: energy goes up instead of forward. Think 'run quiet', with "
        "slightly quicker steps."
    ),
    ("vertical_oscillation_cm", "low"): (
        "A very flat stride is usually fine; at easy pace it can mean "
        "shuffling."
    ),
    ("overstride_ratio", "high"): (
        "The foot lands well ahead of the body, which brakes each step. A "
        "slightly higher cadence and landing under the hips fix it."
    ),
    ("ground_contact_ms", "high"): (
        "Long ground contact. Short plyometrics (pogo hops, skipping) shorten "
        "it over a few weeks."
    ),
}

_DEFAULT_HINT = (
    "Re-film to confirm before changing anything. If it repeats, it is worth a "
    "closer look."
)

# Units printed after a value; pelvic ratio and overstride are ratios.
_UNITS = {
    "pelvic_ratio": "",
    "overstride_ratio": "",
    "cadence_spm": " spm",
    "vertical_oscillation_cm": " cm",
    "ground_contact_ms": " ms",
    "flight_time_ms": " ms",
    "helmet_above_back": "%",
}


def hint_for(field: str, side: str) -> str:
    return _HINTS.get((field, side), _DEFAULT_HINT)


def unit_for(field: str) -> str:
    return _UNITS.get(field, "°")


def photo_field(sport: str, key: str, entry: dict[str, Any]) -> str:
    """The summary field a photo row stands for, so it gets the same hint."""
    if sport != "bike":
        return key
    if key == "knee":
        return "knee_at_tdc" if entry.get("phase") == "tdc" else "knee_at_bdc"
    if key == "hip":
        return "hip_at_tdc"
    return {
        "trunk": "trunk_angle_avg", "elbow": "elbow_angle_avg",
        "shoulder": "shoulder_angle_avg", "forearm_tilt": "forearm_tilt_avg",
    }.get(key, key)


def concern(field: str, label: str, value: float, lo: float | None,
            hi: float | None, material: bool) -> dict[str, Any]:
    side = "high" if hi is not None and value > hi else "low"
    return {"field": field, "label": label, "value": value, "lo": lo, "hi": hi,
            "side": side, "material": material, "hint": hint_for(field, side)}


def _num(v: float) -> str:
    return f"{v:.1f}".rstrip("0").rstrip(".")


def _line(c: dict[str, Any]) -> str:
    unit = unit_for(c["field"])
    if c["lo"] is None and c["hi"] is not None:
        band = f"up to {_num(c['hi'])}{unit}"
    else:
        band = f"{_num(c['lo'])}-{_num(c['hi'])}{unit}"
    where = "outside the band" if c["material"] else "on the edge of the band"
    return (f"- **{c['label']}**: {_num(c['value'])}{unit} vs {band}, {where}. "
            f"{c['hint']}")


def _gap_line(g: dict[str, Any]) -> str:
    unit = unit_for(g["field"])
    where = "below" if g["side"] == "low" else "above"
    return (f"- **{g['label']}**: {_num(g['value'])}{unit}, inside its band but {where} "
            f"every one of the {g['n']} Kona pros ({_num(g['lo'])}-{_num(g['hi'])}{unit}, "
            f"median {_num(g['median'])}{unit}). Not a fault; the field is where the next gain "
            f"usually is. {hint_for(g['field'], g['side'])}")


def worth_a_look(concerns: list[dict[str, Any]], gaps: list[dict[str, Any]] = ()) -> str:
    """The checklist section: clearly-out readings first, edges, then the
    readings inside their band but outside the Kona field. '' when clean."""
    ordered = sorted(concerns, key=lambda c: not c["material"])
    seen = {c["field"] for c in concerns}
    lines = [_line(c) for c in ordered] + [_gap_line(g) for g in gaps if g["field"] not in seen]
    if not lines:
        return ""
    return "**Worth a look**\n" + "\n".join(lines)


# Framing verdicts the clip's tracking report files (tracking_stability.framing).
_SMALL_FRAMING = ("small", "tiny")


def video_next_steps(
    sport: str, cycling_position: str | None, summary: dict[str, Any],
    gaps: list[dict[str, Any]] = (),
) -> list[dict[str, str]]:
    """Where to go from this clip, whatever the score (the photo has its own).

    A clean clip used to end the report at "Nothing to fix". What the capture
    limited, what sits outside the pro field, and what to film next, in the
    order a fitter would take them.
    """
    from app.services.video_analysis.biomechanics.cycling_positions import AERO_POSITIONS

    steps: list[dict[str, str]] = []
    aero = sport == "bike" and cycling_position in AERO_POSITIONS
    framing = ((summary.get("tracking_stability") or {}).get("framing") or {}).get("verdict")
    if sport == "bike" and summary.get("camera_side") == "right":
        steps.append({"key": "side", "title": "Film from the non-drive side",
                      "text": "This clip is from the chainring side. The chainring sits behind the "
                              "ankle at the bottom of the stroke and disturbs the knee there, so the "
                              "saddle-height reading is the least reliable number on it. The other "
                              "side of the bike measures the leg cleanly."})
    if framing in _SMALL_FRAMING:
        steps.append({"key": "framing", "title": "Fill the frame",
                      "text": "You are small in the picture, and at that size the two legs are a few "
                              "pixels apart. Film closer, or zoom in so you span most of the height: "
                              "it does more for accuracy than any other change."})
    if aero and (summary.get("helmet_above_back") is None or summary.get("helmet_check")):
        steps.append({"key": "helmet", "title": "Film in your race helmet",
                      "text": "We could not read a helmet against your back on this clip, which "
                              "usually means a road helmet or none. On aero bars an aero helmet is the "
                              "cheapest gain there is, and with it on the head gets measured too."})
    if gaps:
        parts = []
        for g in gaps:
            unit = unit_for(g["field"])
            parts.append(f"{g['label'].lower()} {_num(g['value'])}{unit} against {_num(g['lo'])}-"
                         f"{_num(g['hi'])}{unit} for all {g['n']} of them")
        steps.append({"key": "pros", "title": "Where the Kona pros sit",
                      "text": "Inside the bands, but outside the whole pro field: " + "; ".join(parts)
                              + ". Not a fault, and not a target (the pros hold positions most "
                              "age-groupers cannot), but it is where your next gain most likely is."})
    steps.append({"key": "refilm", "title": "Change one thing, then film again",
                  "text": ("One adjustment at a time, 30 minutes on it, then the same side and the "
                           "same framing: History puts the two clips side by side."
                           if sport == "bike" else
                           "One cue at a time, a week on it, then the same framing: History puts "
                           "the two clips side by side.")})
    return steps


NOTHING_TO_FIX = re.compile(r"\*\*\s*Nothing to fix\s*\*\*", re.I)
_NEXT_SESSION = re.compile(r"\n\s*\*\*\s*Next session", re.I)
_NOTHING_SECTION = re.compile(
    r"\*\*\s*Nothing to fix\s*\*\*[\s\S]*?(?=\n\s*\*\*[^*\n]+\*\*|\Z)", re.I,
)


def replace_nothing_to_fix(report: str, concerns: list[dict[str, Any]]) -> str:
    """Last resort when the model insists: the section becomes the real list."""
    material = [c for c in concerns if c["material"]][:3]
    fix = "**Fix first**\n" + "\n".join(_line(c) for c in material)
    return _NOTHING_SECTION.sub(lambda _m: fix, report, count=1)


def with_worth_a_look(report: str, concerns: list[dict[str, Any]],
                      gaps: list[dict[str, Any]] = ()) -> str:
    """The checklist goes before Next session, so the single cue still ends it."""
    section = worth_a_look(concerns, gaps)
    if not section:
        return report
    m = _NEXT_SESSION.search(report)
    if not m:
        return report.rstrip() + "\n\n" + section
    return report[:m.start()].rstrip() + "\n\n" + section + "\n\n" + report[m.start():].lstrip("\n")
