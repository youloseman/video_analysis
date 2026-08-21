"""One analysis, written so another language model can read it correctly.

The athlete downloads this file (or copies it) and pastes it into ChatGPT,
Claude or Gemini to go further than the in-app coaching does. So it is written
FOR a model, not for a spreadsheet -- which is exactly why it is not "the result
JSON with a download button". Handed raw numbers, an outside model reconstructs
the missing context from priors, and the priors here are wrong:

* **Angle convention.** Every joint angle we report is INTERNAL: 180 deg is a
  straight limb. The running literature reports FLEXION from straight. A model
  told "knee 140 at midstance" reads 140 deg of flexion -- roughly the opposite
  posture -- and inverts the verdict. Trunk is the same trap twice over: running
  trunk lean is measured from VERTICAL (0 = upright), cycling trunk angle from
  HORIZONTAL (0 = flat/aero).
* **What is not in the file.** Side view sees one side of the body and one
  plane. Nothing here measures the far leg, pelvic drop, knee valgus, ground
  forces, running speed or footwear -- and a model that does not know that will
  happily diagnose them.
* **How much to trust each number.** Per-angle tracked-frame counts, the quality
  gate, the confidence tier and the sampling notice all decide whether a number
  is a measurement or an indication.
* **What the athlete has to supply.** Volume, injuries, goal race, the pace of
  the clip. The file ends by naming them, so the model asks instead of assuming.

The metric listing itself is the SAME block the in-app coach prompt is built
from (``llm_recommendations.build_metrics_block``): one source of truth for
labels, units, reference bands and the per-metric caveats we have already
argued out, rather than a second copy that drifts.

Free-tier results are already trimmed by ``result_gating`` before they reach
here, so the builder never has paid fields to leak -- but the export endpoint
also refuses free callers outright, because a teaser export is a worse product
than no export.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

# Bump when the document's SHAPE changes in a way a downstream consumer could
# notice (sections added/removed/renamed, JSON keys moved). Prose edits inside a
# section do not count.
EXPORT_SCHEMA = 1

PRODUCT = "Flapp"
PRODUCT_URL = "https://getflapp.com"

# Angle keys whose cycle statistics are meaningful on their own. Everything else
# sweeps through the movement cycle, so its mean is an artefact of how long the
# joint dwells near each extreme -- the phase-specific values in the metrics
# block are the ones to read. Same rule the coach prompt states.
_STABLE_ANGLES = frozenset({"trunk", "trunk_angle", "elbow", "left_elbow", "right_elbow"})


def _now_iso(dt: datetime | None = None) -> str:
    dt = dt or datetime.now(timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sport_of(result: dict[str, Any]) -> str:
    return str(result.get("sport_type") or result.get("sport") or "run")


def _position_label(position: str | None) -> str | None:
    if not position:
        return None
    try:
        from app.services.video_analysis.biomechanics.cycling_positions import (
            get_position_label,
        )
        return get_position_label(position)
    except Exception:  # noqa: BLE001 -- a label is never worth failing an export
        return position


def _yaml_str(value: Any) -> str:
    """Quote a scalar for the front matter. Single-quoted YAML needs '' doubled."""
    return "'" + str(value).replace("'", "''") + "'"


def _pct(valid: Any, nan: Any) -> str | None:
    try:
        v, n = int(valid or 0), int(nan or 0)
    except (TypeError, ValueError):
        return None
    total = v + n
    if total <= 0:
        return None
    return f"{round(v / total * 100)}% ({v}/{total})"


def _num(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{round(value, digits):g}"
    return str(value)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _front_matter(
    result: dict[str, Any], generated: str, job_id: str | None,
) -> str:
    sport = _sport_of(result)
    metrics = result.get("sport_specific_metrics") or {}
    sampling = metrics.get("sampling") or {}
    confidence = (metrics.get("analysis_confidence") or {}).get("level")
    score, grade = result.get("technique_score"), result.get("letter_grade")

    lines = [
        "---",
        f"flapp_export_schema: {EXPORT_SCHEMA}",
        f"generated: {generated}",
        f"source: {_yaml_str(f'{PRODUCT} side-view video analysis ({PRODUCT_URL})')}",
        f"sport: {sport}",
    ]
    if sport == "bike":
        label = _position_label(result.get("cycling_position"))
        if label:
            lines.append(f"cycling_position: {_yaml_str(label)}")
    if score is not None:
        lines.append(f"technique_score: {_yaml_str(str(score) + '/100 (' + str(grade or '--') + ')')}")
    if confidence:
        lines.append(f"analysis_confidence: {confidence}")
    if result.get("quality_gate_triggered"):
        lines.append("partial_analysis: true")
    if result.get("camera_side"):
        lines.append(
            f"camera_side: {_yaml_str(str(result['camera_side']) + ' (near side only)')}"
        )
    if job_id:
        lines.append(f"analysis_id: {_yaml_str(job_id)}")
    if result.get("frames_analyzed") is not None:
        lines.append(f"frames_analyzed: {result['frames_analyzed']}")
    if sampling.get("video_fps"):
        fps = f"{sampling['video_fps']} fps"
        if sampling.get("downsampled") and sampling.get("effective_fps"):
            fps += f" (analyzed at ~{sampling['effective_fps']} fps effective)"
        lines.append(f"capture: {_yaml_str(fps)}")
    lines += [
        "method: 'MediaPipe BlazePose (heavy), 2D sagittal plane, single camera'",
        "angle_convention: 'INTERNAL joint angles -- 180 deg = fully straight limb, "
        "NOT flexion from straight'",
        "---",
    ]
    return "\n".join(lines)


def _how_to_read(sport: str) -> str:
    trunk = (
        "Trunk lean is measured from VERTICAL: 0 deg = perfectly upright, larger "
        "= leaning further forward."
        if sport != "bike" else
        "Trunk angle is measured from HORIZONTAL: 0 deg = torso parallel to the "
        "ground (maximally aero), 90 deg = sitting bolt upright. A SMALLER number "
        "is the flatter, more aggressive position."
    )
    return "\n".join([
        "## How to read this file",
        "",
        "You are being handed the numeric output of one side-view video analysis. "
        "Read these five points before interpreting any number.",
        "",
        "1. **Joint angles are INTERNAL angles: 180 deg = the limb is straight.** "
        "The literature usually reports FLEXION measured from straight, so "
        "`flexion = 180 - value`. A knee of 140 deg here is 40 deg of flexion, "
        "not 140.",
        f"2. **{trunk}**",
        "3. **Optimal ranges are efficiency targets from the literature, not "
        "diagnoses.** These are 2D estimates from a phone camera; a degree or two "
        "outside a band is inside the method's own error and is not a finding. "
        "Only a value clearly outside its range, or an issue listed under "
        "*Detected issues*, is worth acting on.",
        "4. **Tracked-frame counts decide how much a number is worth.** Below "
        "~60% tracked, treat the value as indicative only and say so.",
        "5. **Read the *Not measured* and *Missing context* sections before "
        "prescribing anything.** Much of what a coach would want here is simply "
        "absent, and inventing it is the main way this file gets misused.",
    ])


def _metrics_section(result: dict[str, Any]) -> str:
    """The numbers, rendered by the same builder the in-app coach prompt uses."""
    from app.services.video_analysis.llm_recommendations import build_metrics_block

    block = build_metrics_block(
        _sport_of(result),
        result.get("technique_score"),
        result.get("letter_grade"),
        result.get("cycling_position"),
        result.get("detected_issues") or [],
        result.get("angle_statistics") or {},
        result.get("sport_specific_metrics") or {},
    )
    return f"## Measurements\n\n{block}"


def _angle_table(result: dict[str, Any]) -> str:
    stats = result.get("angle_statistics") or {}
    rows = []
    for name, s in stats.items():
        if not isinstance(s, dict):
            continue
        tracked = _pct(s.get("valid_frames"), s.get("nan_frames"))
        rows.append(
            f"| {name} | {_num(s.get('min'))} | {_num(s.get('mean'))} | "
            f"{_num(s.get('max'))} | {_num(s.get('range'))} | {tracked or 'n/a'} |"
        )
    if not rows:
        return ""
    unstable = sorted(k for k in stats if k not in _STABLE_ANGLES)
    note = ""
    if unstable:
        note = (
            "\n\nThese are statistics over the WHOLE movement cycle. For a joint "
            "that sweeps through the cycle (" + ", ".join(unstable[:6]) + ") the "
            "mean is an artefact of how long the joint dwells near each extreme "
            "and must not be verdicted on its own -- use the phase-specific "
            "values under *Measurements* instead. Only the near (camera-facing) "
            "side is measured; the far side is absent, not symmetrical."
        )
    return "\n".join([
        "## Joint angles over the cycle (degrees, internal convention)",
        "",
        "| angle | min | mean | max | range | tracked frames |",
        "| --- | --- | --- | --- | --- | --- |",
        *rows,
    ]) + note


def _issues(result: dict[str, Any]) -> str:
    issues = result.get("detected_issues") or []
    if not issues:
        return (
            "## Detected issues\n\n"
            "None. The rule-based checks flagged nothing on this clip -- treat "
            "that as \"no material problem found\", not as \"not checked\". Do not "
            "manufacture a problem out of a metric sitting slightly off its band."
        )
    lines = ["## Detected issues", "", "Rule-based checks, strongest signal first."]
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        kind = str(issue.get("type", "")).replace("_", " ") or "issue"
        severity = str(issue.get("severity", "")).lower()
        head = f"- **{kind}**" + (f" ({severity})" if severity else "")
        value = issue.get("value")
        if value:
            head += f" — measured: {value}"
        lines.append(head)
        if issue.get("recommendation"):
            lines.append(f"  - {issue['recommendation']}")
    return "\n".join(lines)


def _plan(result: dict[str, Any]) -> str:
    plan = result.get("training_plan") or {}
    priorities = plan.get("top_3_priorities") or []
    if not priorities:
        return ""
    lines = [
        "## Corrective plan already generated for this athlete",
        "",
        "Deterministic drill mapping from the detected issues. Build on it rather "
        "than replacing it, and say so if you disagree with a priority.",
    ]
    for item in priorities:
        if not isinstance(item, dict):
            continue
        label = item.get("label") or item.get("metric") or "priority"
        lines.append(f"- **{label}** — drill: {item.get('drill', 'n/a')}")
        if item.get("why"):
            lines.append(f"  - why: {item['why']}")
        if item.get("instruction"):
            lines.append(f"  - how: {item['instruction']}")
        if item.get("cue"):
            lines.append(f"  - cue: \"{item['cue']}\"")
        if item.get("weeks"):
            lines.append(f"  - expected to take ~{item['weeks']} weeks")
    return "\n".join(lines)


def _coach(result: dict[str, Any]) -> str:
    report = (result.get("ai_recommendations") or {}).get("report")
    if not report:
        return ""
    return "\n".join([
        f"## What the {PRODUCT} coach already told this athlete",
        "",
        "Do not simply restate this. Extend it, or disagree with it and say why.",
        "",
        str(report).strip(),
    ])


def _advanced(result: dict[str, Any]) -> str:
    bio = (result.get("sport_specific_metrics") or {}).get("biomechanics") or {}
    if not isinstance(bio, dict):
        return ""
    items = []
    stability = (bio.get("phase_portraits") or {}).get("overall_stability_score")
    if stability is not None:
        items.append(
            f"- Movement stability (phase-portrait, 0-100): {_num(stability)} — "
            "how repeatable the joint trajectories are cycle to cycle."
        )
    bsi = (bio.get("symmetry") or {}).get("bilateral_symmetry_index")
    if bsi is not None:
        items.append(
            f"- Bilateral symmetry index: {_num(bsi)} — from side-view data, so it "
            "compares near vs far side estimates and is the least reliable number "
            "in this file."
        )
    similarity = (bio.get("waveform") or {}).get("overall_similarity_score")
    if similarity is not None:
        items.append(
            f"- Waveform similarity to the reference pattern (0-100): "
            f"{_num(similarity)}."
        )
    if not items:
        return ""
    return "\n".join(["## Advanced biomechanics (derived, secondary)", "", *items])


def _quality(result: dict[str, Any]) -> str:
    metrics = result.get("sport_specific_metrics") or {}
    gated = result.get("quality") or {}          # free-tier shape
    gate = metrics.get("quality_gate") or {}
    confidence = metrics.get("analysis_confidence") or {}
    lines = ["## Data quality and how far to trust this", ""]

    level = confidence.get("level") or gated.get("confidence")
    if level:
        line = f"- Analysis confidence: **{level}**"
        if confidence.get("explanation"):
            line += f" — {confidence['explanation']}"
        lines.append(line)

    if result.get("quality_gate_triggered") or gated.get("triggered"):
        reasons = list(gate.get("reasons") or gated.get("reasons") or [])
        lines.append(
            "- **Partial analysis.** Capture quality was limited, so the score is "
            "a reference, not a measurement."
            + ("" if not reasons else " Reasons: " + "; ".join(str(r) for r in reasons))
        )

    # How the RECORDING limited the measurement, in the measured numbers. An
    # outside model reading joint angles has no way to know the athlete was 255
    # pixels tall and the pose model was swapping the legs -- and without that
    # it will happily diagnose technique from a signal that was never there.
    capture = metrics.get("capture_report") or gated.get("capture_report") or {}
    problems = [
        c for c in (capture.get("checks") or [])
        if c.get("status") in ("bad", "warn")
    ]
    if problems:
        lines.append(
            f"- **Capture: {capture.get('verdict', 'limited')}.** "
            + " ".join(
                f"{c.get('label')}: {c.get('measured')} (target {c.get('target')})."
                for c in problems
            )
        )

    warnings = list(metrics.get("quality_warnings") or gated.get("warnings") or [])
    warnings += [
        w for w in (metrics.get("analysis_warnings") or []) if w not in warnings
    ]
    for warning in warnings:
        lines.append(f"- {warning}")

    if metrics.get("time_base_uncertain") or gated.get("time_base_uncertain"):
        lines.append(
            "- **Time base unverified (looks like slow motion).** Cadence was "
            "unmeasurable, so ground contact and flight time were withheld rather "
            "than reported wrong. Joint angles are unaffected."
        )
    degraded = metrics.get("sampling_degraded") or gated.get("sampling_degraded")
    if degraded:
        lines.append(f"- **Long clip, sampled.** {degraded}")

    if len(lines) == 2:
        lines.append(
            "- No quality flags. Capture was clean and every metric below was "
            "measured normally."
        )
    return "\n".join(lines)


_NOT_MEASURED_COMMON = [
    "The far side of the body. A side view sees one side; the other is not "
    "symmetrical by assumption, it is simply absent.",
    "Anything in the frontal plane — pelvic drop, hip adduction, knee valgus, "
    "foot pronation. These need a rear-view clip and are not in this file.",
    "Forces, loading rates and torque. Nothing here is force-plate data.",
    "Anything about the athlete's body: mobility, strength, leg-length "
    "difference, injury history.",
]
_NOT_MEASURED = {
    "run": [
        "Running speed and the pace this clip was shot at. Ground contact, "
        "flight time and cadence are all pace-dependent, so a slow easy-run clip "
        "reads long on contact time legitimately.",
        "Footwear, surface and gradient.",
        "Fatigue state — late-run form differs from a fresh 10-second clip.",
    ],
    "bike": [
        "Actual bike measurements: saddle height, setback, stack, reach, crank "
        "length. Every fit conclusion here is inferred from body angles alone.",
        "Power, cadence, speed and how long the rider can actually hold this "
        "position.",
        "Cleat position, saddle comfort and any pain the rider reports.",
    ],
}


def _not_measured(sport: str) -> str:
    extra = _NOT_MEASURED.get(sport, _NOT_MEASURED["run"])
    items = [f"- {line}" for line in (*_NOT_MEASURED_COMMON, *extra)]
    return "\n".join([
        "## Not measured (do not infer these)",
        "",
        "One clip, one camera, one plane. If your answer depends on any of the "
        "following, say that it is out of scope for this data instead of "
        "estimating it:",
        "",
        *items,
    ])


_MISSING_CONTEXT = {
    "run": [
        "Weekly volume and how long they have been running.",
        "Injury history, current niggles, and where.",
        "Goal race, distance and date.",
        "The pace of this clip, and whether it was fresh or fatigued.",
        "Shoes, and whether anything changed recently.",
    ],
    "bike": [
        "Weekly hours, event type (road race, TT, triathlon, sportive) and goal date.",
        "Any pain or numbness — knees, lower back, hands, saddle contact.",
        "How long they can currently hold this position before sitting up.",
        "What has already been adjusted on the bike, and when.",
        "Hip and hamstring mobility, and any history of back or knee injury.",
    ],
}


def _missing_context(sport: str) -> str:
    items = [f"- {line}" for line in _MISSING_CONTEXT.get(sport, _MISSING_CONTEXT["run"])]
    return "\n".join([
        "## Missing context — ask before prescribing",
        "",
        "This analysis contains no training or medical context whatsoever. Ask "
        "for what you need rather than assuming it:",
        "",
        *items,
    ])


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_markdown(
    result: dict[str, Any],
    *,
    job_id: str | None = None,
    generated_at: datetime | None = None,
) -> str:
    """Render one analysis result as an AI-readable Markdown document."""
    sport = _sport_of(result)
    generated = _now_iso(generated_at)
    sections = [
        _front_matter(result, generated, job_id),
        f"# {PRODUCT} — {'cycling' if sport == 'bike' else 'running'} form analysis",
        _how_to_read(sport),
        _metrics_section(result),
        _angle_table(result),
        _issues(result),
        _plan(result),
        _coach(result),
        _advanced(result),
        _quality(result),
        _not_measured(sport),
        _missing_context(sport),
        f"---\n\nGenerated by {PRODUCT} ({PRODUCT_URL}) on {generated}. "
        "Side-view video analysis for runners and cyclists.",
    ]
    return "\n\n".join(s.strip() for s in sections if s and s.strip()) + "\n"


def build_json(
    result: dict[str, Any],
    *,
    job_id: str | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Structured twin of the Markdown, for people wiring their own agent.

    Carries the same interpretation rules as machine-readable fields, and drops
    the payloads no model can use: the annotated keyframe (a megabyte of base64)
    and the raw per-frame waveform series behind the advanced modules.
    """
    sport = _sport_of(result)
    # Shallow copy: the job's result dict is served again on every poll, so the
    # export must not mutate it while trimming.
    metrics = dict(result.get("sport_specific_metrics") or {})
    bio = metrics.pop("biomechanics", None)
    if isinstance(bio, dict) and bio:
        metrics["biomechanics_summary"] = {
            "effective_fps": bio.get("effective_fps"),
            "frame_count": bio.get("frame_count"),
            "stability_score": (bio.get("phase_portraits") or {}).get(
                "overall_stability_score"
            ),
            "bilateral_symmetry_index": (bio.get("symmetry") or {}).get(
                "bilateral_symmetry_index"
            ),
            "waveform_similarity_score": (bio.get("waveform") or {}).get(
                "overall_similarity_score"
            ),
        }

    reference: dict[str, Any] = {}
    try:
        if sport == "bike":
            from app.services.video_analysis.biomechanics.cycling_positions import (
                get_cycling_reference,
            )
            reference = {
                k: list(v)
                for k, v in get_cycling_reference(result.get("cycling_position")).items()
                if isinstance(v, (tuple, list)) and len(v) == 2
            }
        else:
            from app.services.video_analysis.biomechanics.sport_configs import (
                RUNNING_REFERENCE,
            )
            reference = {k: list(v) for k, v in RUNNING_REFERENCE.items()}
    except Exception:  # noqa: BLE001 -- bands are context, never a reason to fail
        reference = {}

    return {
        "flapp_export_schema": EXPORT_SCHEMA,
        "generated": _now_iso(generated_at),
        "source": {"product": PRODUCT, "url": PRODUCT_URL, "job_id": job_id},
        "conventions": {
            "joint_angles": (
                "INTERNAL angles in degrees; 180 = fully straight limb. "
                "flexion = 180 - value."
            ),
            "trunk": (
                "Cycling trunk_angle is measured from HORIZONTAL (0 = flat/aero, "
                "90 = upright)."
                if sport == "bike" else
                "Running trunk lean is measured from VERTICAL (0 = upright)."
            ),
            "measurement": (
                "2D sagittal-plane pose estimation (MediaPipe BlazePose heavy) "
                "from one side-view camera. Near (camera-facing) side only."
            ),
            "reference_ranges": (
                "Efficiency targets from the literature, not diagnoses. A value "
                "within ~2 degrees of a band is inside the method's own error."
            ),
            "not_measured": [*_NOT_MEASURED_COMMON, *_NOT_MEASURED.get(sport, [])],
            "missing_context": _MISSING_CONTEXT.get(sport, []),
        },
        "analysis": {
            "sport": sport,
            "cycling_position": result.get("cycling_position"),
            "cycling_position_label": _position_label(result.get("cycling_position")),
            "camera_side": result.get("camera_side"),
            "frames_analyzed": result.get("frames_analyzed"),
            "technique_score": result.get("technique_score"),
            "letter_grade": result.get("letter_grade"),
            "score_breakdown": result.get("score_breakdown"),
            "quality_gate_triggered": result.get("quality_gate_triggered"),
            "angle_statistics": result.get("angle_statistics"),
            "detected_issues": result.get("detected_issues"),
            "sport_specific_metrics": metrics,
            "training_plan": result.get("training_plan"),
            "coach_report": (result.get("ai_recommendations") or {}).get("report"),
        },
        "reference_ranges": reference,
    }


def export_filename(
    result: dict[str, Any],
    *,
    job_id: str | None = None,
    ext: str = "md",
    generated_at: datetime | None = None,
) -> str:
    """``flapp-run-2026-08-18-a1b2c3.md`` -- sortable, and unique per job."""
    day = _now_iso(generated_at)[:10]
    suffix = f"-{job_id}" if job_id else ""
    return f"flapp-{_sport_of(result)}-{day}{suffix}.{ext}"
