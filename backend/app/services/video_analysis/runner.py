"""Reusable side-view analysis service (running + cycling).

Extracted from the Milestone-1 CLI so both the CLI (scripts/analyze_local.py)
and the FastAPI service (app/main.py) share one proven code path. Mirrors the
Motus ``VideoAnalysisPipeline.process_video`` side-view path:

    build_detector -> extract frames (CLAHE + detect) -> stabilize_landmarks
    -> determine_locked_camera_side -> landmark quality
    -> analyze_frame loop -> finalize_camera_side -> run_advanced_biomechanics
    -> compute_summary / detect_issues / compute_angle_statistics
    -> quality gate -> score_analysis -> (optional) overlay video

Helper functions are copied verbatim from Motus ``pipeline.py`` -- copied,
not rewritten, to keep the numeric behaviour identical.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import structlog

from app.core.config import settings
from app.services.video_analysis.pipeline import SPORT_SAMPLE_RATES
from app.services.video_analysis.detectors import PoseDetector, build_detector
from app.services.video_analysis.biomechanics.advanced_pipeline import (
    run_advanced_biomechanics,
)
from app.services.video_analysis.biomechanics.cycling_analyzer import (
    CyclingAnalyzer,
    determine_locked_camera_side,
)
from app.services.video_analysis.biomechanics.landmark_stabilizer import (
    stabilize_landmarks,
)
from app.services.video_analysis.biomechanics.quality_gate import (
    evaluate_quality_gate,
)
from app.services.video_analysis.biomechanics.running_analyzer import (
    RunningAnalyzer,
)
from app.services.video_analysis.biomechanics.technique_scorer import (
    score_analysis,
)

logger = structlog.get_logger()

VALID_POSITIONS = {"road_hoods", "road_drops", "tt_aero", "triathlon", "casual"}
DEFAULT_BIKE_POSITION = "road_hoods"

# Cap the long edge of frames fed to MediaPipe. Pose landmarks come back
# normalized (0-1), so 720p-class input yields the same pose as 4K for a
# properly-framed athlete while running detection far faster on big phone
# clips. The overlay/keyframe re-read the ORIGINAL video, so output quality
# is unaffected.
DETECT_MAX_LONG_EDGE = 1280


# ===========================================================================
# Helpers copied verbatim from Motus pipeline.py (module-level, no DB/LLM).
# ===========================================================================

# Body regions by MediaPipe landmark indices.
UPPER_BODY_INDICES = [11, 12, 13, 14, 15, 16]  # shoulders, elbows, wrists
CORE_INDICES = [23, 24]                         # hips
HEAD_INDICES = [0]                              # nose
LOWER_BODY_BILATERAL = [25, 26, 27, 28]         # left+right knees+ankles
LOWER_BODY_NEAR_SIDE_LEFT = [25, 27]            # left knee + left ankle
LOWER_BODY_NEAR_SIDE_RIGHT = [26, 28]           # right knee + right ankle


def _compute_unknown_gait_phase_pct_run(analyzer: Any) -> float | None:
    """Fraction (0-100) of frames whose gait phase is ``unknown``.

    Returns ``None`` when no frame results are available so the gate can
    skip the criterion rather than treat the absence as a "0%" success.
    """
    frame_results = getattr(analyzer, "frame_results", None)
    if not frame_results:
        return None
    unknown = sum(
        1 for fr in frame_results
        if fr.extra_metrics.get("gait_phase") == "unknown"
    )
    total = max(len(frame_results), 1)
    return round(unknown / total * 100, 1)


def _compute_landmark_quality(
    raw_frame_data: list[dict],
    sport_type: str,
    camera_view: str | None = None,
    camera_side: str | None = None,
) -> dict[str, Any]:
    """Assess landmark detection quality by body region.

    For unilateral side-view sports (bike side-view, run side-view) the
    lower-body region is computed against the camera-side knee + ankle
    only (2 indices). For bilateral sports the 4-index AND-gate is used.
    """
    if not raw_frame_data:
        return {
            "overall_pct": 0,
            "regions": {},
            "upper_body_detection_ratio": 0.0,
            "lower_body_detection_ratio": 0.0,
            "lower_body_measurement_basis": "bilateral",
            "lower_body_indices_used": LOWER_BODY_BILATERAL,
            "confidence": "none",
        }

    is_side_view = camera_view in (None, "side")
    is_unilateral_sport = sport_type in ("bike", "run")

    if is_unilateral_sport and is_side_view and camera_side in ("left", "right"):
        if camera_side == "left":
            lower_body_indices = LOWER_BODY_NEAR_SIDE_LEFT
        else:
            lower_body_indices = LOWER_BODY_NEAR_SIDE_RIGHT
        measurement_basis = "unilateral"
    else:
        lower_body_indices = LOWER_BODY_BILATERAL
        measurement_basis = "bilateral"
        if is_unilateral_sport and is_side_view and camera_side is None:
            logger.warning(
                "LANDMARK_QUALITY_FALLBACK_BILATERAL",
                sport=sport_type,
                camera_view=camera_view,
                reason=(
                    "camera_side is None on unilateral sport -- "
                    "falling back to bilateral measurement"
                ),
            )

    threshold = 0.3  # minimum visibility to count as "detected"
    total = len(raw_frame_data)

    def region_pct(indices: list[int]) -> float:
        count = 0
        for frame in raw_frame_data:
            wl = frame["world_landmarks"]
            try:
                if all(wl[i].visibility >= threshold for i in indices):
                    count += 1
            except (IndexError, AttributeError):
                pass
        return round(count / total * 100, 1) if total > 0 else 0.0

    regions = {
        "upper_body": region_pct(UPPER_BODY_INDICES),
        "core": region_pct(CORE_INDICES),
        "lower_body": region_pct(lower_body_indices),
        "head": region_pct(HEAD_INDICES),
    }

    overall = round(sum(regions.values()) / len(regions), 1)

    if sport_type == "swim":
        critical = regions["upper_body"]
    elif sport_type == "bike":
        critical = (regions["upper_body"] + regions["lower_body"]) / 2
    else:
        critical = overall

    if critical >= 60:
        confidence = "high"
    elif critical >= 30:
        confidence = "medium"
    else:
        confidence = "low"

    upper_body_pct = regions.get("upper_body", 0.0)
    lower_body_pct = regions.get("lower_body", 0.0)
    lower_body_ratio = round(lower_body_pct / 100.0, 3)

    return {
        "overall_pct": overall,
        "regions": regions,
        "upper_body_detection_ratio": round(upper_body_pct / 100.0, 3),
        "lower_body_detection_ratio": lower_body_ratio,
        "lower_body_measurement_basis": measurement_basis,
        "lower_body_indices_used": lower_body_indices,
        "confidence": confidence,
    }


def _detect_skeleton_jumps(raw_frame_data: list[dict]) -> int:
    """Count frames where the skeleton likely jumped to a different person."""
    jump_count = 0
    prev_hip: tuple[float, float] | None = None
    JUMP_THRESHOLD = 0.3  # 30% of frame in normalized coords

    for frame in raw_frame_data:
        nl = frame["normalized_landmarks"]
        try:
            hip_x = (nl[23].x + nl[24].x) / 2
            hip_y = (nl[23].y + nl[24].y) / 2
        except (IndexError, AttributeError):
            prev_hip = None
            continue

        if prev_hip is not None:
            dx = abs(hip_x - prev_hip[0])
            dy = abs(hip_y - prev_hip[1])
            if dx > JUMP_THRESHOLD or dy > JUMP_THRESHOLD:
                jump_count += 1
        prev_hip = (hip_x, hip_y)

    return jump_count


def _build_quality_warnings(
    landmark_quality: dict, sport_type: str, skeleton_jumps: int,
) -> list[str]:
    """Generate user-facing quality warnings based on landmark detection."""
    warnings: list[str] = []
    regions = landmark_quality.get("regions", {})

    if landmark_quality["confidence"] == "low":
        warnings.append(
            "Low landmark detection quality. The body may be too far away "
            "or partially occluded. For best results, film from the side "
            "at 3-5 meters with the full body in frame."
        )

    if regions.get("head", 100) < 30:
        warnings.append(
            "Head position not reliably detected. "
            "Head position metric may be inaccurate."
        )

    if skeleton_jumps > 3:
        warnings.append(
            "Multiple people may be visible in the video. "
            "Analysis accuracy is reduced. Film with only one person in frame."
        )

    return warnings


def extract_frames(
    video_path: str, sport_type: str, fps: float, detector: PoseDetector,
) -> list[dict[str, Any]]:
    """Iterate the video, CLAHE-enhance each sampled frame, run detection.

    Copied from Motus ``pipeline._iterate_video_frames`` (side-view path).
    ``sample_rate`` for run/bike is 1; adaptive sampling raises it on long
    clips to stay under ``settings.max_analysis_frames``.
    """
    sample_rate = SPORT_SAMPLE_RATES.get(sport_type, 3)

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    video_fps = cap.get(cv2.CAP_PROP_FPS) or fps

    max_analysis_frames = settings.max_analysis_frames
    expected_frames = (
        total_video_frames / sample_rate if sample_rate > 0 else total_video_frames
    )
    if expected_frames > max_analysis_frames and total_video_frames > 0:
        sample_rate = max(sample_rate, int(total_video_frames / max_analysis_frames))
        logger.info(
            "ADAPTIVE_SAMPLING",
            sport=sport_type,
            original_expected=int(expected_frames),
            adjusted_sample_rate=sample_rate,
            target_frames=max_analysis_frames,
        )

    # CLAHE for adaptive contrast enhancement (per-frame, before MediaPipe).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    frame_idx = 0
    frame_results: list[dict[str, Any]] = []
    total_sampled = 0
    total_detected = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if frame_idx % sample_rate != 0:
            frame_idx += 1
            continue

        total_sampled += 1
        timestamp_ms = (frame_idx / video_fps) * 1000.0

        # Downscale oversized frames before detection (normalized landmarks =>
        # pose is unchanged at 720p, but 4K detection is much slower). CLAHE
        # then runs on the smaller frame too. Stored frame_width/height stay
        # the original resolution; the overlay re-reads the source video.
        longest = max(frame.shape[0], frame.shape[1])
        if longest > DETECT_MAX_LONG_EDGE:
            s = DETECT_MAX_LONG_EDGE / float(longest)
            frame = cv2.resize(
                frame,
                (int(round(frame.shape[1] * s)), int(round(frame.shape[0] * s))),
                interpolation=cv2.INTER_AREA,
            )

        # Enhance contrast with CLAHE before detection.
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detector_frame = detector.detect(rgb, int(timestamp_ms))

        if detector_frame is not None:
            total_detected += 1
            frame_results.append({
                "world_landmarks": detector_frame.world_landmarks,
                "normalized_landmarks": detector_frame.normalized_landmarks,
                "timestamp_ms": timestamp_ms,
                "frame_idx": frame_idx,
                "frame_width": frame_width,
                "frame_height": frame_height,
            })

        frame_idx += 1

    cap.release()

    logger.info(
        "FRAME_EXTRACTION",
        sport=sport_type,
        video_frames=frame_idx,
        video_fps=round(video_fps, 1),
        sample_rate=sample_rate,
        sampled=total_sampled,
        detected=total_detected,
        accepted=len(frame_results),
    )
    if len(frame_results) < 30:
        logger.warning(
            "LOW_FRAME_COUNT",
            accepted=len(frame_results),
            hint="Video may be too short. Recommend 5+ seconds.",
        )
    return frame_results


# ===========================================================================
# Driver
# ===========================================================================

def get_video_info(video_path: str) -> dict[str, float]:
    """FPS / frame-count / duration via cv2 (no ffprobe needed for MVP)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()
    duration = frame_count / fps if fps > 0 else 0.0
    return {"fps": float(fps), "frame_count": float(frame_count), "duration": duration}


def _json_safe(obj: Any) -> Any:
    """Recursively make a structure JSON-serializable and NaN-free.

    NaN convention: a missing/gated measurement is NaN upstream; here it
    becomes ``null`` (never 0) so downstream consumers can tell "no data"
    from a real zero.
    """
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (int, str)):
        return obj
    return str(obj)


def run_analysis(
    video_path: str, sport_type: str, cycling_position: str | None,
    overlay_path: str | None = None, recommendations: bool = True,
) -> dict[str, Any]:
    """Reproduce the proven Motus side-view path and return a result dict.

    If ``overlay_path`` is given, also render an annotated overlay video
    (skeleton + angle arcs/labels + score badge) to that path.
    """
    camera_angle = None   # side view only in Milestone 1
    camera_view = None    # None == side view (implicit default in Motus)

    # Step 1: video info (fps) via cv2.
    video_info = get_video_info(video_path)
    fps = video_info["fps"]
    logger.info("VIDEO_INFO", fps=round(fps, 1), **{
        "frames": int(video_info["frame_count"]),
        "duration_s": round(video_info["duration"], 1),
    })

    # Step 2: detector + frame extraction (blocking).
    detector = build_detector(sport_type, camera_angle)
    try:
        raw_frame_data = extract_frames(video_path, sport_type, fps, detector)
    finally:
        detector.close()

    if not raw_frame_data:
        return {
            "status": "failed",
            "error_message": (
                "No pose landmarks detected. Ensure the full body is visible "
                "and the pose model file is present in backend/models/."
            ),
        }

    # Step 2a: stabilize (visibility gate + flip fix + smooth).
    stabilizer_ctx: dict[str, Any] = {}
    stabilize_landmarks(
        raw_frame_data, sport_type, camera_angle, fps=fps,
        context=stabilizer_ctx, camera_view=camera_view,
    )

    # Step 2b: early camera-side lock (bike + run side-view are unilateral).
    early_camera_side, early_lock_meta = determine_locked_camera_side(raw_frame_data)
    camera_side_for_quality = (
        None if early_lock_meta.get("fallback") else early_camera_side
    )
    logger.info(
        "EARLY_CAMERA_SIDE_LOCK",
        side=early_camera_side,
        fallback=early_lock_meta.get("fallback"),
        fallback_reason=early_lock_meta.get("fallback_reason"),
    )

    # Step 2c: landmark quality + warnings.
    landmark_quality = _compute_landmark_quality(
        raw_frame_data, sport_type,
        camera_view=camera_view, camera_side=camera_side_for_quality,
    )
    skeleton_jumps = _detect_skeleton_jumps(raw_frame_data)
    quality_warnings = _build_quality_warnings(
        landmark_quality, sport_type, skeleton_jumps,
    )

    # Step 3: analyzer.
    is_bike = sport_type == "bike"
    if is_bike:
        analyzer: Any = CyclingAnalyzer(fps=fps, cycling_position=cycling_position)
        # Bike side-view never physically flips: lock the side up front so
        # analyze_frame sees a stable camera_side from the first call.
        locked_side = (
            early_camera_side
            if early_camera_side in ("left", "right")
            else (early_lock_meta["votes"][0]
                  if early_lock_meta.get("votes") else "left")
        )
        analyzer.camera_side = locked_side
        analyzer.camera_side_votes = [locked_side]
        if hasattr(analyzer, "_near_side"):
            analyzer._near_side = locked_side
    else:
        analyzer = RunningAnalyzer(fps=fps)

    # Step 3a: per-frame analysis (run/swim vote per frame; bike is locked).
    for fd in raw_frame_data:
        frame_result = analyzer.analyze_frame(
            fd["world_landmarks"], fd["normalized_landmarks"], fd["timestamp_ms"],
        )
        analyzer.add_frame_result(frame_result)
        if not is_bike:
            analyzer.camera_side_votes.append(
                analyzer.detect_camera_side(fd["world_landmarks"])
            )

    analyzer.finalize_camera_side()
    logger.info("CAMERA_SIDE", side=analyzer.camera_side)

    # Step 3b: advanced biomechanics (must not crash the run).
    biomechanics_data = None
    try:
        biomechanics_data = run_advanced_biomechanics(analyzer, sport_type)
    except Exception as e:  # noqa: BLE001
        logger.warning("ADVANCED_BIOMECHANICS_FAILED", err=str(e))

    # Step 3c: summaries.
    summary = analyzer.compute_summary()
    issues = analyzer.detect_issues()
    angle_stats = analyzer.compute_angle_statistics()
    if biomechanics_data:
        summary["biomechanics"] = biomechanics_data

    # Step 4: partial-analysis quality gate.
    max_valid_frames = max(
        (
            s.get("valid_frames", 0)
            for s in angle_stats.values()
            if isinstance(s, dict) and isinstance(s.get("valid_frames"), int)
        ),
        default=0,
    )
    if sport_type == "run":
        gate_unknown_pct = _compute_unknown_gait_phase_pct_run(analyzer)
        lower_body_ratio = landmark_quality.get("lower_body_detection_ratio")
        bdc_present = tdc_present = None
    else:  # bike side-view
        gate_unknown_pct = None
        lower_body_ratio = landmark_quality.get("lower_body_detection_ratio")
        bdc_present = summary.get("knee_at_bdc") is not None
        tdc_present = summary.get("knee_at_tdc") is not None

    quality_gate_result = evaluate_quality_gate(
        unknown_phase_pct=gate_unknown_pct,
        angle_statistics=angle_stats,
        valid_frames=max_valid_frames,
        frames_processed=len(raw_frame_data),
        camera_angle=camera_angle,
        upper_body_detection_ratio=None,
        sport=sport_type,
        camera_view=camera_view,
        lower_body_detection_ratio=lower_body_ratio,
        cycling_position=cycling_position if is_bike else None,
        bdc_present=bdc_present,
        tdc_present=tdc_present,
    )

    # Step 5: score. Unlike the full pipeline (which nulls the score in
    # partial mode) we always compute + surface it -- Milestone 1 is about
    # proving the core yields a number. The gate result is reported so a
    # low-quality clip is still flagged.
    scoring = score_analysis(
        sport_type, summary, angle_stats,
        cycling_position=cycling_position,
        landmark_quality=landmark_quality,
    )

    # Enrich the summary the same way the pipeline persists it.
    summary["camera_side"] = analyzer.camera_side
    summary["landmark_quality"] = landmark_quality
    summary["quality_warnings"] = quality_warnings
    summary["quality_gate"] = quality_gate_result

    # Step 6: annotated visuals. Always render a compact keyframe (a single
    # annotated frame for the history record); render the full overlay video
    # only when requested. Wrapped so a rendering failure never kills the result.
    overlay_video_path = None
    keyframe_base64 = None
    try:
        from app.services.video_analysis.video_visualizer import VideoVisualizer
        _base = Path(overlay_path) if overlay_path else Path(video_path)
        visualizer = VideoVisualizer(
            video_path=video_path,
            frame_data_list=raw_frame_data,
            analyzer=analyzer,
            sport_type=sport_type,
            cycling_position=cycling_position,
            output_dir=str(_base.resolve().parent),
            analysis_id=(_base.stem if overlay_path else "keyframe"),
            technique_score=(
                scoring["overall_score"]
                if scoring.get("overall_score") is not None else 0
            ),
            letter_grade=scoring.get("letter_grade") or "--",
            angle_stats=angle_stats,
            summary=summary,
        )
        keyframe_base64 = visualizer.render_keyframe()
        if overlay_path:
            overlay_video_path = visualizer.generate()
            logger.info("OVERLAY_DONE", path=overlay_video_path)
    except Exception as e:  # noqa: BLE001
        logger.warning("VISUALS_FAILED", err=str(e))

    # Step 7 (Milestone 5): LLM coaching recommendations. Skips gracefully when
    # no API key is configured or the call fails -- never blocks the result.
    ai_recommendations = None
    if recommendations:
        from app.services.video_analysis.llm_recommendations import (
            generate_recommendations,
        )
        ai_recommendations = generate_recommendations(
            sport_type=sport_type,
            technique_score=scoring.get("overall_score"),
            letter_grade=scoring.get("letter_grade"),
            detected_issues=issues,
            angle_statistics=angle_stats,
            sport_specific_metrics=summary,
            cycling_position=cycling_position if is_bike else None,
        )

    return {
        "status": "completed",
        "sport_type": sport_type,
        "cycling_position": cycling_position if is_bike else None,
        "camera_side": analyzer.camera_side,
        "frames_analyzed": len(raw_frame_data),
        "technique_score": scoring.get("overall_score"),
        "letter_grade": scoring.get("letter_grade"),
        "score_breakdown": scoring.get("component_scores"),
        "quality_gate_triggered": bool(quality_gate_result.get("triggered")),
        "overlay_video_path": overlay_video_path,
        "keyframe_base64": keyframe_base64,
        "ai_recommendations": ai_recommendations,
        "angle_statistics": angle_stats,
        "detected_issues": issues,
        "sport_specific_metrics": summary,
    }
