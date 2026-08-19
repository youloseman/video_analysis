"""Gemini-powered coaching recommendations from the biomechanics result.

Self-contained (this is NOT the full Motus ``llm_recommendations.py``, which is
coupled to a shared AI-client abstraction + per-sport action-plan builders +
swimming). It builds a compact, numbers-first prompt from the analysis result
and asks Gemini for concise, evidence-based coaching.

Degrades gracefully: returns ``None`` when no API key is configured, the SDK is
missing, or the call fails -- analysis never depends on the LLM.
"""

from __future__ import annotations

from typing import Any

import structlog

from app.core.config import settings
from app.services.video_analysis.biomechanics.fit_tradeoffs import (
    build_tradeoff_block,
)

logger = structlog.get_logger()

SYSTEM_PROMPT = (
    "You are an elite endurance-sports coach and bike-fitter with a "
    "biomechanics background. You are given side-view video pose-analysis data "
    "for ONE athlete (running or cycling). Write concise, specific, "
    "evidence-based feedback in Markdown with EXACTLY these sections:\n"
    "**Overall** — 1-2 sentences on the headline takeaway.\n"
    "**What's working** — 1-3 short bullets.\n"
    "**Fix first** — ONLY genuinely material problems: 1-3 of them, "
    "highest-impact first. For each: what it is, why it matters, and ONE "
    "concrete drill or position/fit change.\n"
    "**Next session** — a single cue to focus on.\n\n"
    "MATERIALITY FLOOR (this governs the Fix first section):\n"
    "- These numbers come from 2D pose estimation. A degree or two either way "
    "is inside the method's own error, so a metric sitting just outside its "
    "band is NOT a finding. Anything the data marks 'acceptable', "
    "'borderline' or otherwise within tolerance is off-limits as a fix -- "
    "never build advice or a drill around one. Mention it, at most, in What's "
    "working as \"on the edge of the band, worth an eye\".\n"
    "- Only a metric clearly outside its range (marked out of range / "
    "needs_work), or a rule-based detected issue, may become a Fix first item.\n"
    "- If nothing clears that floor, DO NOT invent a problem and do not pad "
    "the section to reach three. Rename the heading to **Nothing to fix** and "
    "use it to say what the athlete should hold on to, plus any sustainability "
    "caveat the data actually contains. Telling an athlete their position is "
    "solid, when it is, is the correct answer -- not a failure to find "
    "something.\n\n"
    "TRADE-OFFS (this governs every fix you prescribe):\n"
    "- Body angles on a bike are coupled: the contact points are the saddle, "
    "the bars and the pedals, so almost nothing moves alone. When the data "
    "includes a ranked list of adjustment options, RECOMMEND FROM THE TOP OF "
    "IT. The ranking already accounts for what each option disturbs; it is not "
    "a menu to pick from by preference.\n"
    "- State the cost of whatever you prescribe, in the same breath as the "
    "prescription. \"Raise the stack 10mm; this opens the hip but sits you up "
    "and costs you aerodynamically\" is a coaching instruction. \"Raise the "
    "stack 10mm\" is half of one.\n"
    "- NEVER praise a metric and then prescribe something that degrades it "
    "without saying so. If the only fix available damages something that is "
    "working, say that plainly and let the athlete decide -- that judgement is "
    "theirs, and a fit is a set of trade-offs, not a set of correct answers.\n\n"
    "Rules: address the athlete as \"you\". Reference their actual numbers vs "
    "the optimal ranges given. Be direct and practical, no fluff, no medical "
    "diagnoses. Up to 260 words; a clean analysis should be shorter."
)


def _client(genai: Any, types: Any) -> Any:
    """Gemini client with an explicit request timeout.

    Without ``timeout`` the SDK inherits the underlying HTTP client's default,
    which does not bound a server that accepts the connection and then stops
    responding. Every caller here runs on a threadpool thread, so a hung request
    does not just delay one report -- it holds a worker. ``HttpOptions.timeout``
    is in MILLISECONDS.
    """
    return genai.Client(
        api_key=settings.gemini_api_key,
        http_options=types.HttpOptions(
            timeout=int(settings.gemini_timeout_s * 1000),
        ),
    )


def _fmt(v: Any, unit: str = "") -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        v = round(v, 1)
    return f"{v}{unit}"


def _bike_data_block(
    score: Any, grade: Any, position: str | None,
    issues: list[dict], summary: dict,
) -> str:
    try:
        from app.services.video_analysis.biomechanics.cycling_positions import (
            get_cycling_reference,
            get_position_label,
        )
        ref = get_cycling_reference(position)
        pos_label = get_position_label(position) if position else "unknown"
    except Exception:
        ref, pos_label = {}, position or "unknown"

    def rng(key: str) -> str:
        r = ref.get(key)
        return f" (optimal {r[0]}-{r[1]})" if r else ""

    lines = [
        f"Sport: cycling | Position: {pos_label} | Technique score: {score}/100 ({grade})",
        f"- Knee angle at bottom of stroke (BDC): {_fmt(summary.get('knee_at_bdc'), '°')}{rng('knee_at_bdc')}",
        f"- Knee angle at top of stroke (TDC): {_fmt(summary.get('knee_at_tdc'), '°')}{rng('knee_at_tdc')}",
        # ``hip_angle_max`` is the name of the reference BAND; the analyzer
        # writes the measurement as ``hip_angle_avg``. Reading only the former
        # meant every cycling clip reported "Hip angle: n/a" to the coach --
        # silently hiding the one metric with a documented medical floor (hip
        # closure below 45 deg, iliac artery). Same fallback the scorer uses.
        f"- Hip angle: {_fmt(summary.get('hip_angle_max') or summary.get('hip_angle_avg'), '°')}"
        f"{rng('hip_angle_max')}",
        f"- Trunk angle: {_fmt(summary.get('trunk_angle_avg'), '°')}{rng('trunk_angle')}",
        f"- Elbow angle: {_fmt(summary.get('elbow_angle_avg'), '°')}{rng('elbow_angle')}",
        f"- Shoulder angle: {_fmt(summary.get('shoulder_angle_avg'), '°')}{rng('shoulder_angle')}",
        f"- Pelvic ratio: {_fmt(summary.get('pelvic_ratio'), 'x')}",
        f"- Head alignment score: {_fmt(summary.get('head_alignment_avg'))}",
        f"- Saddle height assessment: {summary.get('saddle_height_assessment', 'n/a')}",
    ]
    arch = summary.get("position_archetype")
    if isinstance(arch, dict) and arch.get("label"):
        lines.append(f"- Position archetype: {arch['label']} — {arch.get('description', '')}")

    # Relative aero read-out (torso-angle CdA zone). Never an absolute
    # CdA -- give the coach the zone + the flatten-delta so its advice
    # stays consistent with what the rider sees, and remind it the
    # number is relative.
    aero = summary.get("aero_estimate")
    if isinstance(aero, dict):
        nxt = aero.get("next_zone")
        if isinstance(nxt, dict):
            lines.append(
                f"- Aero (relative, torso-angle only): {aero.get('zone_label')} "
                f"(~{aero.get('drag_vs_upright_pct')}% of upright drag). Flattening "
                f"torso toward {nxt.get('target_trunk_angle')}° would cut drag "
                f"~{nxt.get('drag_reduction_pct')}% (~{nxt.get('approx_watts_saved')}W "
                f"@ {aero.get('reference_speed_kmh')}km/h). NOT an absolute CdA; only "
                f"suggest it if the rider can hold power and breathe there."
            )
        else:
            lines.append(
                f"- Aero (relative, torso-angle only): {aero.get('zone_label')} "
                f"(~{aero.get('drag_vs_upright_pct')}% of upright drag) — already "
                f"well-placed for this position; no flatter target advised."
            )
    return "\n".join(lines)


def _run_data_block(
    score: Any, grade: Any, issues: list[dict], summary: dict, angle_stats: dict,
) -> str:
    vosc = summary.get("vertical_oscillation_cm")
    if vosc is None and summary.get("vertical_oscillation_m") is not None:
        vosc = round(summary["vertical_oscillation_m"] * 100, 1)
    knee = angle_stats.get("knee", {}) if isinstance(angle_stats, dict) else {}

    # Foot strike: pattern + signed angle (heel-up positive) when measured.
    fs = summary.get("foot_strike")
    fs_angle = summary.get("foot_strike_angle_deg")
    # Rearfoot striking is normal for ~80-95% of runners and the "heel strike is
    # bad / forefoot is more economical" claims are refuted (Napier). Tell the
    # coach NOT to fault the strike pattern itself -- the brake is overstriding.
    _fs_caveat = (
        " (rearfoot striking is normal for most runners and NOT a fault by "
        "itself -- do not tell the runner to change their strike pattern; the "
        "braking issue is overstriding, i.e. the foot landing ahead of the hip)"
    )
    if fs and fs_angle is not None:
        foot_strike_line = (
            f"- Foot strike: {fs} ({fs_angle:+.0f} deg, heel-up positive; "
            f"estimated from 2D video){_fs_caveat}"
        )
    elif fs:
        foot_strike_line = f"- Foot strike: {fs} (estimated from 2D video){_fs_caveat}"
    else:
        foot_strike_line = "- Foot strike: n/a"

    # Overstride: hip->ankle-ahead ratio at contact (lower is better).
    overstride = summary.get("overstride_ratio")
    if overstride is not None:
        overstride_line = (
            f"- Overstride: {overstride:.2f} x leg length ahead of hip at "
            f"contact (target < 0.15; higher = braking/overstride, estimated)"
        )
    else:
        overstride_line = "- Overstride: n/a"

    lines = [
        f"Sport: running | Technique score: {score}/100 ({grade})",
        f"- Cadence: {_fmt(summary.get('cadence_spm'), ' spm')} (target ~170-185)",
        f"- Vertical oscillation: {_fmt(vosc, ' cm')} (lower is generally better)",
        f"- Trunk lean: {_fmt(summary.get('trunk_lean_avg'), '°')} (target ~5-10 forward)",
        f"- Ground contact time: {_fmt(summary.get('ground_contact_ms'), ' ms')}"
        f"{' (estimated from 2D video)' if summary.get('ground_contact_ms_estimated') else ''}",
        f"- Flight time: {_fmt(summary.get('flight_time_ms'), ' ms')} (aerial phase, target ~80-150"
        f"{', estimated from 2D video' if summary.get('flight_time_ms_estimated') else ''})",
        foot_strike_line,
        overstride_line,
        # Knee is gait-phase-dependent: map the cycle extremes to phases
        # (Novacheck 1998) and give the coach the PHASE-SPECIFIC optima, so it
        # grounds its verdict instead of inferring "locked knee at contact" from
        # a raw range. The 2D max can read a few degrees high when the leg
        # aligns with the camera, so it's flagged as an estimate.
        f"- Knee near initial contact (~cycle max): {_fmt(knee.get('max'), '°')} "
        f"(optimal 160-175°; above ~176° = over-extended/near-locked at contact, "
        f"which couples with overstriding; 2D estimate, can read slightly high)",
        f"- Knee peak swing flexion (~cycle min): {_fmt(knee.get('min'), '°')} "
        f"(optimal 80-100°; higher = insufficient knee drive)",
        "Note: do not verdict the mean/range of knee, hip or ankle directly -- "
        "they sweep through the stride; use only the phase-specific values above.",
    ]
    return "\n".join(lines)


def _issues_block(issues: list[dict]) -> str:
    if not issues:
        # The video prompt ships raw numbers with no per-metric status, so the
        # rule-based checks are the materiality signal here. "Nothing flagged"
        # has to be stated as a result, or the model reads the bare numbers and
        # manufactures a finding out of a metric sitting a degree off its band.
        return (
            "Detected issues: NONE flagged by the rule-based checks. Treat this "
            "as no material problem found. Only flag a metric here if it is "
            "CLEARLY outside the optimal range stated above -- not a degree or "
            "two off. If nothing is, use the **Nothing to fix** variant."
        )
    out = ["Detected issues (rule-based):"]
    for it in issues[:6]:
        t = str(it.get("type", "")).replace("_", " ")
        out.append(f"- {t}: {it.get('recommendation', '')} ({it.get('value', '')})")
    return "\n".join(out)


def build_metrics_block(
    sport_type: str, score: Any, grade: Any, position: str | None,
    issues: list[dict], angle_stats: dict, summary: dict,
) -> str:
    """The numbers-first metric listing, with units, bands and caveats.

    Public because the AI export (``services/export/ai_export.py``) hands the
    athlete the same block to paste into an outside model. Both consumers are
    talking to an LLM, and the caveats attached to these numbers -- the knee
    values are phase-specific, the strike pattern is not a fault, the aero
    read-out is relative -- are what stops one from misreading them. Two copies
    would mean fixing every such lesson twice.
    """
    if sport_type == "bike":
        return _bike_data_block(score, grade, position, issues, summary)
    return _run_data_block(score, grade, issues, summary, angle_stats)


def _build_prompt(
    sport_type: str, score: Any, grade: Any, position: str | None,
    issues: list[dict], angle_stats: dict, summary: dict,
) -> str:
    data = build_metrics_block(
        sport_type, score, grade, position, issues, angle_stats, summary,
    )
    # Without this note the coach confidently builds "Fix first" items on
    # numbers the same result page flags as unreliable (amber tracking
    # warning + LOW confidence badge) -- the report must not contradict
    # the quality verdict shown right above it.
    from app.services.video_analysis.biomechanics.confidence_scorer import (
        assess_tracking_stability,
    )
    severity = assess_tracking_stability(summary.get("tracking_stability"))
    caveat = ""
    if severity == "severe":
        caveat = (
            "\n\nNOTE: pose tracking was unstable on this clip (the left/right "
            "leg identity was repeatedly confused -- typical for backlit or "
            "silhouette footage). Cadence, stride and leg-angle numbers above "
            "are unreliable; do NOT build 'Fix first' items on them. Lead with "
            "advice to re-film with the light behind the camera, and keep any "
            "technique notes tentative."
        )
    elif severity == "mild":
        caveat = (
            "\n\nNOTE: pose tracking occasionally confused the left and right "
            "legs on this clip -- treat leg-based numbers (cadence, stride, "
            "knee angles) as approximate."
        )
    return (
        f"{data}{caveat}\n\n{_issues_block(issues)}\n\n"
        "Write the coaching feedback now, following the required section structure."
    )


def generate_recommendations(
    sport_type: str,
    technique_score: Any,
    letter_grade: Any,
    detected_issues: list[dict[str, Any]],
    angle_statistics: dict[str, Any],
    sport_specific_metrics: dict[str, Any],
    cycling_position: str | None = None,
) -> dict[str, Any] | None:
    """Return ``{"report": markdown, "model": name}`` or ``None`` (graceful)."""
    if not settings.gemini_api_key:
        logger.info("LLM_SKIP", reason="no_api_key")
        return None

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logger.warning("LLM_SKIP", reason="google-genai not installed")
        return None

    prompt = _build_prompt(
        sport_type, technique_score, letter_grade, cycling_position,
        detected_issues or [], angle_statistics or {}, sport_specific_metrics or {},
    )

    try:
        client = _client(genai, types)
        resp = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.3,
                max_output_tokens=1200,
                # gemini-2.5-flash spends "thinking" tokens by default, which
                # would eat the output budget and truncate the reply. This is a
                # short copywriting task from structured data -- no thinking
                # needed; disabling it gives the full budget to the answer.
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        text = (getattr(resp, "text", None) or "").strip()
        if not text:
            logger.warning("LLM_EMPTY", model=settings.gemini_model)
            return None
        logger.info("LLM_OK", model=settings.gemini_model, chars=len(text))
        return {"report": text, "model": settings.gemini_model}
    except Exception as e:  # noqa: BLE001
        logger.warning("LLM_FAILED", err=str(e))
        return None


PROGRESS_SYSTEM_PROMPT = (
    "You are an elite endurance-sports coach. You are given the SAME athlete's "
    "side-view form metrics from TWO analyses -- an earlier 'before' and a later "
    "'after' -- for the same sport. Write a short, encouraging but honest "
    "progress read in Markdown with EXACTLY these sections:\n"
    "**Progress** — 1-2 sentences on the headline change (did form improve?).\n"
    "**Improved** — 1-3 bullets naming metrics that moved toward optimal.\n"
    "**Still work on** — 1-2 bullets on what regressed or is still off-target.\n"
    "**Next** — one concrete cue for the next session.\n\n"
    "Rules: address the athlete as \"you\". Cite the actual before->after numbers. "
    "'Closer to the optimal range' = improvement, even if the raw number went "
    "down. Be direct, no fluff, no medical claims. 120-180 words total."
)


def _progress_data_block(sport: str, before: dict, after: dict) -> str:
    """Compact before/after table from the two frontend 'trends' maps.

    ``before``/``after`` each carry ``score``, ``trends`` (numeric metric map),
    and ``cat`` (categorical, e.g. foot_strike). Keeps only keys present in
    both so the model compares like with like.
    """
    labels = {
        "cadence": "Cadence (spm)", "vertical_oscillation": "Vertical oscillation (cm)",
        "trunk_lean": "Trunk lean (deg)", "ground_contact": "Ground contact (ms)",
        "flight_time": "Flight time (ms)", "overstride": "Overstride (x leg length)",
        "knee": "Knee angle (deg)", "hip": "Hip angle (deg)", "trunk": "Trunk angle (deg)",
        "elbow": "Elbow angle (deg)", "shoulder": "Shoulder angle (deg)",
        "pelvic_ratio": "Pelvic ratio",
    }
    bt = before.get("trends") or {}
    at = after.get("trends") or {}
    lines = [
        f"Sport: {'cycling' if sport == 'bike' else 'running'}",
        f"Technique score: {before.get('score')} -> {after.get('score')} (/100)",
    ]
    for key, label in labels.items():
        if key in bt and key in at:
            lines.append(f"- {label}: {bt[key]} -> {at[key]}")
    # Categorical (foot strike): only when present in both.
    bc = before.get("cat") or {}
    ac = after.get("cat") or {}
    for key in ("foot_strike",):
        if bc.get(key) and ac.get(key):
            lines.append(f"- Foot strike: {bc[key]} -> {ac[key]}")
    return "\n".join(lines)


def generate_progress_summary(
    sport: str, before: dict[str, Any], after: dict[str, Any],
) -> dict[str, Any] | None:
    """Return ``{"summary": markdown, "model": name}`` or ``None`` (graceful).

    Compares two of the athlete's analyses. Same Gemini path + graceful
    degradation contract as ``generate_recommendations``.
    """
    if not settings.gemini_api_key:
        logger.info("PROGRESS_LLM_SKIP", reason="no_api_key")
        return None
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logger.warning("PROGRESS_LLM_SKIP", reason="google-genai not installed")
        return None

    prompt = (
        f"{_progress_data_block(sport, before, after)}\n\n"
        "Write the progress read now, following the required section structure."
    )
    try:
        client = _client(genai, types)
        resp = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=PROGRESS_SYSTEM_PROMPT,
                temperature=0.3,
                max_output_tokens=900,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        text = (getattr(resp, "text", None) or "").strip()
        if not text:
            logger.warning("PROGRESS_LLM_EMPTY", model=settings.gemini_model)
            return None
        logger.info("PROGRESS_LLM_OK", model=settings.gemini_model, chars=len(text))
        return {"summary": text, "model": settings.gemini_model}
    except Exception as e:  # noqa: BLE001
        logger.warning("PROGRESS_LLM_FAILED", err=str(e))
        return None


# Not every "angle" in the photo result is an angle: head tuck is a 0-100 score
# and pelvic rotation is a ratio. Each entry is (suffix printed after every
# number, one-off clarification). Spelled out in the prompt so the coach text
# does not report them in degrees.
_PHOTO_METRIC_UNITS: dict[str, tuple[str, str]] = {
    "head_alignment": ("/100", " [a 0-100 tuck score, NOT degrees]"),
    "pelvic_ratio": ("x", " [a ratio, NOT degrees]"),
}


def _build_photo_prompt(sport: str, res: dict[str, Any]) -> str:
    sc = res.get("score", {}) or {}
    sport_label = "cycling" if sport == "bike" else sport
    lines = [
        f"Sport: {sport_label} | SINGLE STILL PHOTO (a snapshot -- no cadence "
        "or dynamics available).",
        f"Overall score: {sc.get('overall_score')}/100 ({sc.get('grade')})",
    ]
    if sport == "bike":
        lines.append(
            f"Position: {res.get('cycling_position_label') or res.get('cycling_position')} "
            f"| pedal phase in this photo: {res.get('pedal_phase')}"
        )
    lines.append("Joint angles (value vs optimal, status):")
    for k, v in (res.get("angles_with_context") or {}).items():
        # Units are explicit: two bike metrics are NOT degrees, and without the
        # unit the model narrates them as "96 degrees" / "3.94 degrees".
        unit, unit_note = _PHOTO_METRIC_UNITS.get(k, (" deg", ""))
        # Unresolved pedal-phase joints (ankle always; hip when the crank
        # position could not be determined): no band applies, so the coach has
        # nothing to judge against and must not invent a verdict.
        #
        # A hip WITH a resolved band is different -- it carries a real status
        # and is scored. Suppressing it here while the materiality block listed
        # it as a problem left the model with two contradictory instructions,
        # and it resolved them by dropping the hip and reaching for a metric it
        # had been told was fine.
        if v.get("status") == "phase_dependent":
            lines.append(
                f"- {v.get('label', k)}: {v.get('value')}{unit}{unit_note} — pedal-phase-dependent, "
                f"scored interactively by the rider. Do NOT critique this angle or "
                f"call it too open/closed."
            )
            continue
        phase_note = (
            f" [scored against the {v['phase'].upper()} band from the "
            f"auto-detected crank position; the rider can correct that]"
            if v.get("bands") and v.get("phase") in ("bdc", "tdc") else ""
        )
        # Aero trunk below the comfort band: a deliberate trade-off, not a
        # defect. Without this the coach "fixes" a world-class tuck by telling
        # the rider to raise the bars.
        status = v.get("status") or ""
        if status.startswith("aero_"):
            lines.append(
                f"- {v.get('label', k)}: {v.get('value')}{unit} "
                f"(comfort band {v.get('optimal_min')}-{v.get('optimal_max')}{unit}, "
                f"{status}) — BELOW the band on purpose: flatter is the "
                f"aerodynamic goal here. Do NOT list this as a fault, do NOT "
                f"advise raising the bars or opening the trunk. "
                + (
                    "It is aggressive enough to be worth one line about "
                    "sustainability -- breathing and holding it for the full "
                    "event -- and nothing more."
                    if status == "aero_extreme"
                    else "Treat it as a strength."
                )
            )
            continue
        lines.append(
            f"- {v.get('label', k)}: {v.get('value')}{unit} "
            f"(optimal {v.get('optimal_min')}-{v.get('optimal_max')}{unit}, "
            f"{v.get('status')}){unit_note}{phase_note}"
        )
    # Materiality is decided HERE, not by the model. The photo result already
    # carries a per-angle status, so the set of things that may be "fixed" is a
    # fact, not a judgement call -- state it plainly rather than hoping the
    # model applies the floor to a list of numbers.
    material, borderline = [], []
    for k, v in (res.get("angles_with_context") or {}).items():
        status, label = v.get("status") or "", v.get("label", k)
        if status == "needs_work":
            material.append(f"{label} ({v.get('value')})")
        elif status == "acceptable":
            borderline.append(f"{label} ({v.get('value')})")
    if material:
        lines.append(
            "Material problems (ONLY these may appear in Fix first): "
            + ", ".join(material)
        )
    else:
        lines.append(
            "Material problems: NONE. Every measured angle is in range, within "
            "tolerance, or a deliberate aero trade-off. Do not invent a fix -- "
            "use the **Nothing to fix** variant of that section."
        )
    if borderline:
        lines.append(
            "Just outside the band but WITHIN TOLERANCE (not problems, no "
            "drills, no fit changes for these): " + ", ".join(borderline)
        )

    # Which lever actually fixes what, and what it costs elsewhere. Computed,
    # not left to the model: joint angles on a bike are coupled through a
    # handful of contact points, and without this the coach praises a trunk
    # angle and then prescribes the one adjustment that undoes it.
    if sport == "bike":
        tradeoffs = build_tradeoff_block(
            res.get("angles_with_context") or {}, res.get("cycling_position"),
        )
        if tradeoffs:
            lines.append(tradeoffs)

    warns = res.get("warnings") or []
    if warns:
        lines.append("Capture notes: " + "; ".join(warns))
    return "\n".join(lines) + (
        "\n\nWrite the coaching feedback now, following the required section "
        "structure. Note this is a single still, so avoid advice that needs video."
    )


def generate_photo_recommendations(
    sport: str, photo_result: dict[str, Any],
) -> dict[str, Any] | None:
    """Coaching from a single-photo analysis. Returns {report, model} or None."""
    if not settings.gemini_api_key:
        logger.info("LLM_SKIP", reason="no_api_key")
        return None
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logger.warning("LLM_SKIP", reason="google-genai not installed")
        return None

    prompt = _build_photo_prompt(sport, photo_result)
    try:
        client = _client(genai, types)
        resp = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.3,
                max_output_tokens=1200,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        text = (getattr(resp, "text", None) or "").strip()
        if not text:
            return None
        logger.info("LLM_PHOTO_OK", model=settings.gemini_model, chars=len(text))
        return {"report": text, "model": settings.gemini_model}
    except Exception as e:  # noqa: BLE001
        logger.warning("LLM_PHOTO_FAILED", err=str(e))
        return None
