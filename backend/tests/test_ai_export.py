"""The AI export has to survive being read by a model that has never seen Flapp.

Two things are load-bearing and neither is obvious from the numbers themselves:

* The **angle convention**. Our joint angles are internal (180 deg = straight);
  the running literature reports flexion from straight. Ship a knee value with
  no convention and an outside model reads roughly the opposite posture. The
  same trap runs the other way on trunk: from VERTICAL for running, from
  HORIZONTAL for cycling.
* The **limits**. Side view, one plane, near side only, no forces, no pace. A
  file that does not say so invites confident advice about pelvic drop.

The rest of the suite avoids the mediapipe/opencv stack (see conftest), and so
does the builder -- everything it touches is pure Python, so it is exercised
directly here. The gate around the endpoint lives in
``test_ai_export_endpoint.py``, which does need that stack.
"""

from __future__ import annotations

import json

from app.services.export import ai_export


def _run_result() -> dict:
    return {
        "status": "completed",
        "sport_type": "run",
        "camera_side": "left",
        "frames_analyzed": 148,
        "technique_score": 74,
        "letter_grade": "C+",
        "quality_gate_triggered": False,
        "angle_statistics": {
            "knee": {
                "min": 62.4, "mean": 121.3, "max": 172.1, "range": 109.7,
                "valid_frames": 142, "nan_frames": 6,
            },
        },
        "detected_issues": [
            {"type": "low_cadence", "severity": "warning", "value": "164 spm",
             "recommendation": "Cadence is 164 spm. Target: 170-190 spm."},
        ],
        "ai_recommendations": {"report": "**Overall** — slow legs."},
        "sport_specific_metrics": {
            "cadence_spm": 164.2,
            "trunk_lean_avg": 6.2,
            "quality_warnings": ["Head position not reliably detected."],
            "analysis_confidence": {"level": "medium", "explanation": "Partial tracking."},
            "sampling": {"video_fps": 30.0, "effective_fps": 30.0, "downsampled": False},
        },
    }


def _bike_result() -> dict:
    return {
        "status": "completed",
        "sport_type": "bike",
        "cycling_position": "tt_aero",
        "camera_side": "right",
        "technique_score": 81,
        "letter_grade": "B",
        "quality_gate_triggered": True,
        "angle_statistics": {},
        "detected_issues": [],
        "sport_specific_metrics": {
            "knee_at_bdc": 141.0,
            "trunk_angle_avg": 22.0,
            "hip_angle_avg": 48.0,
            "quality_gate": {"triggered": True, "reasons": ["Hip tracked in 31% of frames."]},
        },
    }


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------

def test_the_angle_convention_is_stated_before_any_number():
    """An outside model reads 140 deg as flexion unless told otherwise, which
    inverts the verdict. So the convention must arrive before the metrics."""
    md = ai_export.build_markdown(_run_result())
    convention = md.index("180 deg = the limb is straight")
    assert convention < md.index("## Measurements")
    assert "flexion = 180 - value" in md


def test_running_and_cycling_trunk_reference_different_axes():
    """Running trunk lean is from VERTICAL, cycling trunk angle from HORIZONTAL.
    One sentence for both would be wrong for one of them."""
    run_md = ai_export.build_markdown(_run_result())
    bike_md = ai_export.build_markdown(_bike_result())
    assert "from VERTICAL" in run_md and "from HORIZONTAL" not in run_md
    assert "from HORIZONTAL" in bike_md and "from VERTICAL" not in bike_md


def test_the_limits_of_the_method_are_spelled_out():
    md = ai_export.build_markdown(_run_result())
    assert "## Not measured (do not infer these)" in md
    assert "frontal plane" in md            # pelvic drop / valgus are not in here
    assert "far side" in md.lower()
    assert "## Missing context — ask before prescribing" in md


def test_quality_flags_travel_with_the_numbers():
    """A partial analysis that exports as a clean set of numbers is the whole
    failure mode this section exists to prevent."""
    md = ai_export.build_markdown(_bike_result())
    assert "Partial analysis" in md
    assert "Hip tracked in 31% of frames." in md


def test_a_clean_run_says_so_rather_than_leaving_the_section_empty():
    result = _run_result()
    result["sport_specific_metrics"].pop("quality_warnings")
    result["sport_specific_metrics"].pop("analysis_confidence")
    md = ai_export.build_markdown(result)
    assert "No quality flags." in md


def test_no_findings_reads_as_a_result_not_as_a_gap():
    """"Nothing flagged" has to be stated, or a model invents a finding out of a
    metric sitting a degree off its band."""
    md = ai_export.build_markdown(_bike_result())
    assert "no material problem found" in md


def test_tracked_frame_counts_are_carried_per_angle():
    md = ai_export.build_markdown(_run_result())
    assert "96% (142/148)" in md            # 142 valid of 148 attempted


def test_the_coach_report_is_included_and_framed_as_prior_work():
    md = ai_export.build_markdown(_run_result())
    assert "**Overall** — slow legs." in md
    assert "Do not simply restate this." in md


def test_the_metric_block_is_the_one_the_coach_prompt_uses():
    """One source of truth for labels, units, bands and per-metric caveats."""
    from app.services.video_analysis.llm_recommendations import build_metrics_block

    result = _run_result()
    block = build_metrics_block(
        "run", 74, "C+", None, [], result["angle_statistics"],
        result["sport_specific_metrics"],
    )
    assert block in ai_export.build_markdown(result)


def test_a_gated_free_result_does_not_crash_the_builder():
    """The endpoint refuses free callers, but the builder must not depend on
    that: every paid section is simply absent from a gated payload."""
    from app.services.result_gating import gate_free_result

    md = ai_export.build_markdown(gate_free_result(_run_result()))
    assert "## Measurements" in md
    assert "Head position not reliably detected." in md   # caveats survive gating


def test_the_export_stays_small_enough_to_paste_into_a_chat():
    md = ai_export.build_markdown(_run_result())
    assert len(md) < 12_000, "an export nobody can paste is not an export"


# ---------------------------------------------------------------------------
# The JSON twin
# ---------------------------------------------------------------------------

def test_json_carries_the_same_rules_as_machine_readable_fields():
    payload = ai_export.build_json(_run_result(), job_id="abc123")
    assert payload["flapp_export_schema"] == ai_export.EXPORT_SCHEMA
    assert "180 = fully straight limb" in payload["conventions"]["joint_angles"]
    assert payload["conventions"]["not_measured"]
    assert payload["conventions"]["missing_context"]
    assert payload["reference_ranges"]["cadence_spm"] == [170, 190]
    json.dumps(payload)          # must be serializable as-is


def test_json_drops_the_payloads_no_model_can_use():
    """A keyframe is a megabyte of base64 and the raw waveform series are
    thousands of floats -- both would crowd out the actual analysis."""
    result = _run_result()
    result["keyframe_base64"] = "data:image/jpeg;base64," + "A" * 5000
    result["sport_specific_metrics"]["biomechanics"] = {
        "phase_portraits": {"overall_stability_score": 78.2, "series": [0.1] * 5000},
    }
    payload = ai_export.build_json(result)
    blob = json.dumps(payload)
    assert "A" * 100 not in blob
    assert "series" not in blob
    summary = payload["analysis"]["sport_specific_metrics"]["biomechanics_summary"]
    assert summary["stability_score"] == 78.2


def test_filenames_sort_by_date_and_are_unique_per_analysis():
    name = ai_export.export_filename(_run_result(), job_id="abc123")
    assert name.startswith("flapp-run-")
    assert name.endswith("-abc123.md")
