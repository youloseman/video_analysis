"""Advanced biomechanics pipeline orchestrator.

Runs all lab-grade analysis modules (Butterworth filter, phase portraits,
symmetry/CRP, coordination, waveform comparison) on collected frame data.
Each module is independently wrapped so a failure in one does not block others.
"""

from typing import Any

import structlog

from app.services.video_analysis.biomechanics.base_analyzer import SportAnalyzer

logger = structlog.get_logger(__name__)

# Minimum frames required for meaningful biomechanics analysis
MIN_FRAMES_FOR_ANALYSIS = 10


def run_advanced_biomechanics(
    analyzer: SportAnalyzer, sport_type: str
) -> dict[str, Any] | None:
    """Run all advanced biomechanics modules on the analyzer's collected data.

    Called after all frames are processed but before compute_summary().
    Butterworth filter mutates angle_history in-place; subsequent modules
    read the filtered data.

    Returns dict with module results, or None if insufficient data.
    """
    if len(analyzer.frame_results) < MIN_FRAMES_FOR_ANALYSIS:
        logger.info(
            "BIOMECHANICS_SKIP",
            reason="insufficient_frames",
            frame_count=len(analyzer.frame_results),
            min_required=MIN_FRAMES_FOR_ANALYSIS,
        )
        return None

    effective_fps = analyzer.get_effective_fps()
    camera_side = getattr(analyzer, "camera_side", None)
    logger.info(
        "BIOMECHANICS_START",
        sport=sport_type,
        frames=len(analyzer.frame_results),
        effective_fps=round(effective_fps, 1),
        camera_side=camera_side,
    )

    result: dict[str, Any] = {
        "effective_fps": round(effective_fps, 1),
        "frame_count": len(analyzer.frame_results),
        "camera_side": camera_side,
    }

    # Module 1: Butterworth filter (mutates angle_history in-place)
    try:
        from app.services.video_analysis.biomechanics.butterworth_filter import (
            apply_butterworth_filter,
        )

        filter_info = apply_butterworth_filter(
            analyzer.angle_history, effective_fps, sport_type
        )
        result["butterworth"] = filter_info
        logger.info("BIOMECHANICS_M1_OK", filtered_angles=len(filter_info.get("filtered", [])))
    except Exception as e:
        logger.warning("BIOMECHANICS_M1_FAIL", err=str(e))
        result["butterworth"] = None

    # Module 2: Phase portraits
    try:
        from app.services.video_analysis.biomechanics.phase_portrait import (
            compute_phase_portraits,
        )

        result["phase_portraits"] = compute_phase_portraits(
            analyzer.angle_history, analyzer.angle_timestamps, sport_type,
            camera_side=camera_side,
        )
        logger.info("BIOMECHANICS_M2_OK")
    except Exception as e:
        logger.warning("BIOMECHANICS_M2_FAIL", err=str(e))
        result["phase_portraits"] = None

    # Module 3: Symmetry / CRP
    try:
        from app.services.video_analysis.biomechanics.symmetry_analyzer import (
            compute_symmetry,
        )

        result["symmetry"] = compute_symmetry(
            analyzer.angle_history, analyzer.angle_timestamps, sport_type,
            camera_side=camera_side,
        )
        logger.info("BIOMECHANICS_M3_OK")
    except Exception as e:
        logger.warning("BIOMECHANICS_M3_FAIL", err=str(e))
        result["symmetry"] = None

    # Module 4: Coordination (angle-angle diagrams)
    try:
        from app.services.video_analysis.biomechanics.coordination_analyzer import (
            compute_coordination,
        )

        phase_data = result.get("phase_portraits")
        result["coordination"] = compute_coordination(
            analyzer.angle_history, analyzer.angle_timestamps, sport_type,
            phase_data=phase_data, camera_side=camera_side,
        )
        logger.info("BIOMECHANICS_M4_OK")
    except Exception as e:
        logger.warning("BIOMECHANICS_M4_FAIL", err=str(e))
        result["coordination"] = None

    # Module 5: Waveform comparison
    try:
        from app.services.video_analysis.biomechanics.waveform_comparator import (
            compute_waveform_comparison,
        )

        phase_data = result.get("phase_portraits")
        result["waveform"] = compute_waveform_comparison(
            analyzer.angle_history, analyzer.angle_timestamps, sport_type,
            phase_data=phase_data, camera_side=camera_side,
        )
        logger.info("BIOMECHANICS_M5_OK")
    except Exception as e:
        logger.warning("BIOMECHANICS_M5_FAIL", err=str(e))
        result["waveform"] = None

    logger.info("BIOMECHANICS_DONE", modules_ok=sum(
        1 for k in ["butterworth", "phase_portraits", "symmetry", "coordination", "waveform"]
        if result.get(k) is not None
    ))

    return result
