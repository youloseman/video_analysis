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

logger = structlog.get_logger()

SYSTEM_PROMPT = (
    "You are an elite endurance-sports coach and bike-fitter with a "
    "biomechanics background. You are given side-view video pose-analysis data "
    "for ONE athlete (running or cycling). Write concise, specific, "
    "evidence-based feedback in Markdown with EXACTLY these sections:\n"
    "**Overall** — 1-2 sentences on the headline takeaway.\n"
    "**What's working** — 1-3 short bullets.\n"
    "**Fix first** — the 1-3 highest-impact issues. For each: what it is, why "
    "it matters, and ONE concrete drill or position/fit change.\n"
    "**Next session** — a single cue to focus on.\n\n"
    "Rules: address the athlete as \"you\". Reference their actual numbers vs "
    "the optimal ranges given. Be direct and practical, no fluff, no medical "
    "diagnoses. 180-260 words total."
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
        f"- Hip angle: {_fmt(summary.get('hip_angle_max'), '°')}",
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
    if fs and fs_angle is not None:
        foot_strike_line = (
            f"- Foot strike: {fs} ({fs_angle:+.0f} deg, heel-up positive; "
            f"estimated from 2D video)"
        )
    elif fs:
        foot_strike_line = f"- Foot strike: {fs} (estimated from 2D video)"
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
        f"- Knee angle range: {_fmt(knee.get('min'), '°')} to {_fmt(knee.get('max'), '°')}",
    ]
    return "\n".join(lines)


def _issues_block(issues: list[dict]) -> str:
    if not issues:
        return "Detected issues: none flagged by the rule-based checks."
    out = ["Detected issues (rule-based):"]
    for it in issues[:6]:
        t = str(it.get("type", "")).replace("_", " ")
        out.append(f"- {t}: {it.get('recommendation', '')} ({it.get('value', '')})")
    return "\n".join(out)


def _build_prompt(
    sport_type: str, score: Any, grade: Any, position: str | None,
    issues: list[dict], angle_stats: dict, summary: dict,
) -> str:
    if sport_type == "bike":
        data = _bike_data_block(score, grade, position, issues, summary)
    else:
        data = _run_data_block(score, grade, issues, summary, angle_stats)
    return (
        f"{data}\n\n{_issues_block(issues)}\n\n"
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
        client = genai.Client(api_key=settings.gemini_api_key)
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
        client = genai.Client(api_key=settings.gemini_api_key)
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
        if v.get("status") == "phase_dependent":
            # No verdict from a still -- the value swings with crank position.
            lines.append(
                f"- {v.get('label', k)}: {v.get('value')} — pedal-phase-dependent, "
                f"NOT scored. Do NOT critique this angle or call it too open/closed."
            )
            continue
        lines.append(
            f"- {v.get('label', k)}: {v.get('value')} "
            f"(optimal {v.get('optimal_min')}-{v.get('optimal_max')}, {v.get('status')})"
        )
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
        client = genai.Client(api_key=settings.gemini_api_key)
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
