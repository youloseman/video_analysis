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
from app.services.video_analysis.biomechanics.confidence_scorer import (
    assess_tracking_stability,
    compute_analysis_confidence,
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
from app.services.video_analysis.biomechanics.sport_configs import reference_bands
from app.services.video_analysis.biomechanics.running_analyzer import (
    RunningAnalyzer,
)
from app.services.video_analysis.biomechanics.tracking_diagnostics import (
    compute_framing,
    compute_near_side_cues,
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

# How many times the NEAR leg's identity track may be re-seeded before the
# athlete is told. A re-seed means the leg was unidentifiable for long enough
# that the track had to accept whatever the model said next, so the series past
# that point may be describing the other leg -- unlike a blanked frame, which
# only costs a measurement. Bike only; see
# landmark_stabilizer._gate_leg_identity_breaks.
LEG_IDENTITY_RESEED_WARN = 3


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


def _tracked_ratio(
    detected: int, video_info: dict[str, Any], sampling_meta: dict[str, Any],
) -> float | None:
    """Share of the frames handed to the detector that had a pose in them.

    ``sampling_meta`` records the sampling STRIDE, not the number of frames it
    produced, so the denominator is reconstructed from the clip length and that
    stride. Returns None rather than a guess when either is missing -- a wrong
    ratio here would tell an athlete their clip was half empty when it was not.
    """
    try:
        total = float(video_info.get("frame_count") or 0)
        stride = float(sampling_meta.get("sample_rate") or 0)
    except (TypeError, ValueError):
        return None
    if total <= 0 or stride <= 0:
        return None
    sampled = total / stride
    if sampled < 1:
        return None
    return min(1.0, detected / sampled)


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


def _ordinal(n: int) -> str:
    """1 -> '1st', 2 -> '2nd', 4 -> '4th'. Used in the sampling notice."""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def extract_frames(
    video_path: str, sport_type: str, fps: float, detector: PoseDetector,
    meta: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Iterate the video, CLAHE-enhance each sampled frame, run detection.

    Copied from Motus ``pipeline._iterate_video_frames`` (side-view path).
    ``sample_rate`` for run/bike is 1; adaptive sampling raises it on long
    clips to stay under ``settings.max_analysis_frames``.

    ``meta`` (optional, filled in place) reports the sampling actually used.
    That matters downstream: raising the stride throws away frames, and the
    time-based metrics (cadence, ground contact, flight time) are measured in
    frames -- at stride 4 their resolution drops fourfold. The caller turns this
    into a user-facing caveat rather than reporting a decimated cadence as if it
    were measured at full rate.
    """
    base_sample_rate = SPORT_SAMPLE_RATES.get(sport_type, 3)
    sample_rate = base_sample_rate

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    # max(0, ...) for the same reason as get_video_info: a stream-written
    # container can report -1 frames, and -1 here would poison the sampling
    # maths and the duration in sampling_meta.
    total_video_frames = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
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
    if meta is not None:
        meta.update({
            "sample_rate": sample_rate,
            "base_sample_rate": base_sample_rate,
            "effective_fps": round(video_fps / sample_rate, 1) if sample_rate else None,
            "video_fps": round(video_fps, 1),
            "duration_s": (
                round(total_video_frames / video_fps, 1)
                if video_fps and total_video_frames else None
            ),
            # True once the clip was long enough to force a coarser stride than
            # this sport's baseline -- i.e. timing resolution was traded away.
            "downsampled": sample_rate > base_sample_rate,
        })
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
    # max(0, ...): a stream-written container (MediaRecorder's webm) has no
    # frame count in its header and some decoders report -1 for it. `or 0`
    # does not catch -1, and a negative count would ride into the capture
    # report as a negative duration. Zero is the honest value: every consumer
    # already treats it as "unknown" rather than as a measurement.
    frame_count = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
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
    hide_angle_values: bool = False, kinogram: bool = True,
    athlete_height_cm: float | None = None,
    focus: str | None = None,
    mobility_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reproduce the proven Motus side-view path and return a result dict.

    If ``overlay_path`` is given, also render an annotated overlay video
    (skeleton + angle arcs/labels + score badge) to that path.

    ``hide_angle_values`` (free-tier teaser): render the keyframe/overlay with
    the skeleton visible but the numeric angle labels masked + a watermark.

    ``kinogram`` (running only): compose the five ALTIS stride positions into
    one image. Costs a partial second decode of the video, and the gate strips
    it from a free response anyway, so callers serving a teaser pass False
    rather than render something nobody will see.

    ``athlete_height_cm`` (running only): the athlete's standing height, when
    they have told us. It is the one real-world length a side view ever gets,
    and it is what turns image fractions into honest centimetres -- see
    ``RunningAnalyzer._lock_body_scale``. Omitted, the analysis falls back to a
    population-average torso and says so in the result.

    ``mobility_profile`` (cycling only): the rider's off-bike range, from
    ``biomechanics.mobility``. Every fit window in the app is a statement about
    a bike; this is the only one that is a statement about the rider, and what
    it buys is the difference between "you could go lower" and "you could go
    lower if your hips went there".
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
    sampling_meta: dict[str, Any] = {}
    try:
        raw_frame_data = extract_frames(
            video_path, sport_type, fps, detector, meta=sampling_meta,
        )
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
    # The smoothing filters must see the rate the SAMPLED series actually
    # has: on long clips adaptive sampling raises the stride (e.g. every
    # 4th frame of a 60 fps clip -> a 15 Hz series). Feeding the raw video
    # fps in would make One Euro over-smooth (skeleton lags the athlete)
    # and would place the Butterworth cutoff relative to a Nyquist the
    # series doesn't have.
    stabilizer_ctx: dict[str, Any] = {}
    _stride = max(1, int(sampling_meta.get("sample_rate") or 1))
    stabilize_landmarks(
        raw_frame_data, sport_type, camera_angle, fps=fps / _stride,
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
        analyzer = RunningAnalyzer(fps=fps, height_cm=athlete_height_cm)
        # Vote the camera side over the WHOLE clip BEFORE measuring anything.
        # The vote itself is unchanged (same per-frame test, same majority and
        # tie rule as finalize_camera_side) -- it just happens up front now.
        # Previously the loop ran with a side seeded from the first 30 frames
        # and finalize_camera_side then overwrote it with this majority, so a
        # clip where the two disagreed reported angles measured off one leg
        # under a skeleton drawn on the other.
        analyzer.camera_side_votes = [
            analyzer.detect_camera_side(fd["world_landmarks"])
            for fd in raw_frame_data
        ]
        analyzer.finalize_camera_side()

    # Step 3a: per-frame analysis (bike and run both run with a locked side).
    for fd in raw_frame_data:
        frame_result = analyzer.analyze_frame(
            fd["world_landmarks"], fd["normalized_landmarks"], fd["timestamp_ms"],
        )
        analyzer.add_frame_result(frame_result)

    logger.info("CAMERA_SIDE", side=analyzer.camera_side)

    # Tracking stability: how firmly did the model hold left/right identity?
    # Backlit / silhouette clips make MediaPipe re-assign limbs frame to
    # frame; every signal below sits near 0 on a clean clip. Bike measures
    # none of them (side pre-locked, anti-flip and the leg pass are off),
    # so its dict carries None/0 values the scorer skips.
    side_disagreement_pct = None
    if not is_bike and analyzer.camera_side_votes:
        _votes = [v for v in analyzer.camera_side_votes if v in ("left", "right")]
        if _votes:
            _n_dis = sum(1 for v in _votes if v != analyzer.camera_side)
            side_disagreement_pct = round(_n_dis / len(_votes) * 100, 1)
    # Framing + near/far cues. DIAGNOSTIC ONLY for now: recorded and logged so
    # the cues can be checked against clips whose near side is known before
    # they are allowed to decide anything. Never let a failure here stop an
    # analysis that otherwise succeeded.
    framing: dict[str, Any] = {}
    near_cues: dict[str, Any] = {}
    try:
        _fw = raw_frame_data[0].get("frame_width") or 0
        _fh = raw_frame_data[0].get("frame_height") or 0
        _scale = (
            min(1.0, DETECT_MAX_LONG_EDGE / float(max(_fw, _fh)))
            if max(_fw, _fh) else 1.0
        )
        framing = compute_framing(raw_frame_data, detect_height_px=_fh * _scale)
        if not is_bike:
            near_cues = compute_near_side_cues(raw_frame_data)
        logger.info("CAPTURE_FRAMING", **framing)
        if near_cues:
            logger.info(
                "NEAR_SIDE_CUES", chosen=analyzer.camera_side,
                **{k: v for k, v in near_cues.items() if k != "cue_votes"},
            )
    except Exception as e:  # noqa: BLE001
        logger.warning("TRACKING_DIAGNOSTICS_FAILED", err=str(e))

    tracking_stability = {
        "side_vote_disagreement_pct": side_disagreement_pct,
        "flip_pct": stabilizer_ctx.get("flip_pct"),
        "flips_corrected": stabilizer_ctx.get("flips_corrected"),
        "leg_swap_pct": stabilizer_ctx.get("leg_swap_pct"),
        "leg_swaps_corrected": stabilizer_ctx.get("leg_swaps_corrected"),
        "leg_collapse_pct": stabilizer_ctx.get("leg_collapse_pct"),
        # Run only: how the whole-clip identity resolution went -- crossings
        # found, the share of links decided by hysteresis rather than
        # evidence, and whether each segment's naming had cues behind it.
        # This is the RESULT-quality signal; leg_swap_pct above stays the
        # INPUT-noise signal, and gates should read the one they mean.
        "leg_identity": stabilizer_ctx.get("leg_identity"),
        # Bike only: frames where a leg left its own track and was declined
        # rather than corrected. See landmark_stabilizer._gate_leg_identity_breaks.
        "leg_identity_gate": stabilizer_ctx.get("leg_identity_gate"),
        "framing": framing or None,
        "near_side_cues": near_cues or None,
    }
    logger.info(
        "TRACKING_STABILITY",
        **{k: v for k, v in tracking_stability.items()
           if k not in ("framing", "near_side_cues")},
    )
    # Read the NEAR leg only. The far leg is expected to be a mess on a side
    # view -- its median deviation is 3.5x the near leg's -- and it is neither
    # drawn nor measured, so warning on its numbers would be crying wolf about
    # a landmark nothing depends on.
    _gate = tracking_stability.get("leg_identity_gate") or {}
    _near_reseeds = (_gate.get("reseeds") or {}).get(early_camera_side or "", 0)
    if _near_reseeds >= LEG_IDENTITY_RESEED_WARN:
        # Blanked frames are the mechanism working: a few blinks cost a few
        # measurements. A RE-SEED is different -- it means identity was lost
        # for long enough that the track had to accept whatever the model said,
        # so from there the series may be describing the other leg.
        quality_warnings.append(
            "The pose model lost track of which leg is which several times on "
            "this clip, for long enough that we could not recover it. Frames we "
            "could not identify were left unmeasured rather than guessed, so "
            "some knee readings are based on less of the clip than usual. "
            "Filming from a little in front of or behind square-on, with the "
            "whole drivetrain visible, separates the two legs best."
        )
    if assess_tracking_stability(tracking_stability) == "severe":
        # This warning renders as an amber banner in the UI and travels to
        # the free tier too (quality_warnings are never gated) -- the
        # confidence factor alone would only reach paid results.
        #
        # State the SYMPTOM and offer causes. The first version asserted
        # backlight, which read as wrong to an athlete who had filmed a
        # cleanly-lit track -- there the cause was distance, not light.
        quality_warnings.append(
            "Skeleton tracking was unstable on this clip: the pose model "
            "could not reliably tell the left and right legs apart, so "
            "cadence, stride and angle metrics may be unreliable. The usual "
            "causes are the athlete being small in the frame (film closer or "
            "zoom in), the body reading as a dark silhouette (keep the light "
            "behind the camera), or a view almost exactly side-on (a few "
            "degrees of angle helps separate the legs)."
        )
    if framing.get("verdict") == "tiny":
        px = framing.get("subject_height_px")
        quality_warnings.append(
            f"The athlete is small in the frame (about {px} pixels tall in "
            "the image we analyze). At that size the two legs are only a few "
            "pixels apart for much of the stride, which is what makes the "
            "skeleton jump between them. Filling more of the frame -- filming "
            "closer, or zooming in so the runner spans most of the height -- "
            "does more for accuracy than any other change."
        )

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
        # How fast the near knee opens and closes. Already computed inside the
        # phase portrait and used only for a hull area until now; lifted here
        # so it reaches the report as a metric. Ungraded on purpose -- see
        # phase_portrait.summary_peak_velocities for why a band would be a
        # verdict on the athlete's pace rather than their technique.
        try:
            from app.services.video_analysis.biomechanics.phase_portrait import (
                summary_peak_velocities,
            )

            summary.update(summary_peak_velocities(
                biomechanics_data.get("phase_portraits"),
                sport_type, analyzer.camera_side,
            ))
        except Exception as e:  # noqa: BLE001 -- a metric must not break the run
            logger.warning("PEAK_VELOCITY_LIFT_FAILED", err=str(e))

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
    # Which leg faces the camera decides where two metrics are measured from.
    # Overstride and foot strike are sampled off the NEAR-side ankle at the
    # moment it lands; get the side wrong and both are read from a leg in
    # mid-swing, which does not produce a missing number, it produces a
    # confident wrong one. Measured on a treadmill clip whose near side the
    # cues could not agree on: overstride read 0.39-0.62 leg-lengths, well into
    # "overstriding" territory, from a runner whose foot lands under them.
    #
    # The cues already vote on this and already record when they conflict. That
    # verdict has been diagnostic-only, deliberately, until it had been checked
    # against clips whose near side is known -- and it still does not get to
    # DECIDE the side here. It only gets to say "then do not publish the two
    # numbers that hang on it", which is the same plausibility gating every
    # other metric in this file already has.
    # Withheld when the cues actively point at the OTHER leg, not merely when
    # they cannot agree. Balanced cues are no evidence of a problem, and
    # treating them as one silenced a clip whose side was right: IMG_3979,
    # confirmed near side RIGHT, where three of four cues said right and a
    # ballot-counting tie called it conflicted anyway.
    _suggested = (near_cues or {}).get("suggested_near_side")
    if not is_bike and _suggested and _suggested != analyzer.camera_side:
        withheld = [k for k in (
            "overstride_ratio", "overstride_ratio_estimated",
            "overstride_ratio_contacts", "foot_strike",
            "foot_strike_angle_deg", "foot_strike_estimated",
            "foot_strike_contacts",
        ) if summary.pop(k, None) is not None]
        if withheld:
            quality_warnings.append(
                "The depth, size and visibility cues point at the opposite leg "
                "from the one this analysis measured. Overstride and "
                "foot-strike pattern are read from the near-side foot as it "
                "lands, so they are withheld rather than reported from a leg "
                "that may be the far one. Filming a few degrees off square, "
                "rather than exactly side-on, separates the two legs."
            )
            logger.info(
                "NEAR_SIDE_DISAGREEMENT", chosen=analyzer.camera_side,
                cues_suggest=_suggested,
                score=(near_cues or {}).get("near_side_score"),
                withheld=withheld,
            )

    summary["landmark_quality"] = landmark_quality
    summary["quality_warnings"] = quality_warnings
    summary["tracking_stability"] = tracking_stability
    summary["quality_gate"] = quality_gate_result
    summary["sampling"] = sampling_meta
    # A long clip is analyzed end to end, but at a coarser stride -- so the
    # frame-counted metrics (cadence, ground contact, flight time) lose
    # resolution. Report that instead of presenting them as full-rate measures.
    if sampling_meta.get("downsampled"):
        summary["sampling_degraded"] = (
            f"This clip is {sampling_meta.get('duration_s')}s, so we analyzed "
            f"every {_ordinal(sampling_meta['sample_rate'])} frame "
            f"(~{sampling_meta.get('effective_fps')} fps effective). Cadence, "
            f"ground contact and flight time are measured in frames, so they "
            f"are coarser than usual."
        )

    # An ordered account of what the RECORDING cost the measurement, with the
    # numbers and the fix. Everything it reads is already computed above; what
    # it adds is priority and an action, which two sentences of prose fired at
    # the extreme of each scale could not carry. Built last so it can see the
    # summary's own verdicts (slow motion, decimation).
    # Did the person holding the phone move? Running only: the harm being
    # measured is contamination of vertical oscillation, which is a running
    # metric, and a rider on a trainer does not have one. Measured and
    # reported, never compensated for -- no clip yet seen actually has a
    # moving camera, so a compensator would be fixing an undemonstrated
    # problem. See camera_motion.py.
    camera_motion = None
    if sport_type == "run":
        try:
            from app.services.video_analysis.camera_motion import (
                estimate_camera_motion,
            )

            hip_y = getattr(analyzer, "norm_hip_y_history", None) or []
            hip_amp = None
            if len(hip_y) >= 10:
                _arr = np.array(hip_y, dtype=float)
                _arr = _arr[np.isfinite(_arr)]
                if _arr.size >= 10:
                    hip_amp = float(
                        np.percentile(_arr, 95) - np.percentile(_arr, 5)
                    )
            camera_motion = estimate_camera_motion(
                video_path, raw_frame_data, hip_amplitude_norm=hip_amp,
            )
            if camera_motion:
                summary["camera_motion"] = camera_motion
        except Exception as e:  # noqa: BLE001 -- a diagnostic, never fatal
            logger.warning("CAMERA_MOTION_FAILED", err=str(e))

    try:
        from app.services.video_analysis.capture_report import build_capture_report

        summary["capture_report"] = build_capture_report(
            sport_type=sport_type,
            duration_s=video_info.get("duration"),
            frame_width=raw_frame_data[0].get("frame_width"),
            frame_height=raw_frame_data[0].get("frame_height"),
            framing=framing or None,
            tracking_stability=tracking_stability,
            time_base_uncertain=summary.get("time_base_uncertain"),
            sampling_degraded=summary.get("sampling_degraded"),
            camera_motion=camera_motion,
            # Frames with a pose in them over frames looked at. A clip can run
            # ten seconds with the athlete findable in four of them, and every
            # stride metric is built from the four. The denominator is derived
            # rather than read: sampling_meta records the STRIDE, not how many
            # frames it ended up handing to the detector.
            tracked_ratio=_tracked_ratio(
                len(raw_frame_data), video_info, sampling_meta,
            ),
            # Slow motion stretches the picture, not the stride: at 8x, twelve
            # seconds of playback holds a second and a half of running.
            slow_motion_factor=summary.get("slow_motion_factor"),
        )
        logger.info(
            "CAPTURE_REPORT",
            verdict=summary["capture_report"]["verdict"],
            problems=summary["capture_report"]["problem_count"],
        )
    except Exception as e:  # noqa: BLE001 -- a caveat must never break the run
        logger.warning("CAPTURE_REPORT_FAILED", err=str(e))

    # Soft confidence tier (high/medium/low) -- informational, hides nothing.
    # Butterworth meta exists for bike + run side-view (swim above-water keeps
    # One Euro -> None, which the scorer handles). phase_diagnostics is
    # swim-only -> omitted here,
    # so Factor 5 stays dormant for run/bike. Warnings from the analyzer and the
    # quality gate are merged for scoring only (each keeps its own UI panel).
    try:
        merged_warnings = list(summary.get("analysis_warnings") or []) + list(quality_warnings)
        summary["analysis_confidence"] = compute_analysis_confidence(
            angle_statistics=angle_stats,
            frames_processed=len(raw_frame_data),
            butterworth_meta=stabilizer_ctx.get("butterworth_meta"),
            analysis_warnings=merged_warnings,
            landmark_quality=landmark_quality,
            tracking_stability=tracking_stability,
        )
    except Exception as e:  # noqa: BLE001 -- confidence must never break the run
        logger.warning("ANALYSIS_CONFIDENCE_FAILED", err=str(e))

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
            hide_angle_values=hide_angle_values,
        )
        keyframe_base64 = visualizer.render_keyframe()
        if overlay_path:
            overlay_video_path = visualizer.generate()
            logger.info("OVERLAY_DONE", path=overlay_video_path)
    except Exception as e:  # noqa: BLE001
        logger.warning("VISUALS_FAILED", err=str(e))

    # Step 6b: the kinogram -- the five ALTIS stride positions in one image.
    # Running only (the positions are defined off the gait cycle) and wrapped
    # like every other visual: an artifact worth having, never worth failing an
    # analysis for.
    kinogram_base64 = None
    kinogram_meta = None
    if kinogram and sport_type == "run":
        try:
            from app.services.video_analysis.kinogram import build_run_kinogram

            kinogram_base64, kinogram_meta = build_run_kinogram(
                video_path, analyzer, raw_frame_data,
                # The kinogram inherits the report's own verdict on this clip:
                # when the stride could not be timed or the legs could not be
                # told apart, there is nothing to draw five honest positions
                # from. See kinogram.stride_trust_block.
                summary=summary,
                tracking_stability=tracking_stability,
                technique_score=(
                    scoring["overall_score"]
                    if scoring.get("overall_score") is not None else 0
                ),
                letter_grade=scoring.get("letter_grade") or "--",
                hide_values=hide_angle_values,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("KINOGRAM_FAILED", err=str(e))

    # Step 6c: the rider's action plan. A runner's is drills to practise; a
    # rider's is adjustments to make to the bike -- component, current value,
    # target, how far to move it, what it costs. Same intent ("what do I do
    # now"), different shape, so it gets its own key and no renderer has to
    # guess which one it received.
    #
    # Built BEFORE the coaching call on purpose: the prose is handed these
    # adjustments so the two halves of the report cannot disagree about which
    # way the saddle should go.
    fit_plan = None
    mobility_fit = None
    if is_bike:
        try:
            from app.services.video_analysis.biomechanics.action_plan_builder import (
                action_plan_to_json,
                build_action_plan,
            )
            from app.services.video_analysis.biomechanics.mobility import (
                assess_position,
            )
            mobility_fit = assess_position(
                mobility_profile, cycling_position or "road_hoods",
            )
            fit_plan = action_plan_to_json(build_action_plan(
                position=cycling_position or "road_hoods",
                angle_statistics=angle_stats,
                sport_specific_metrics=summary,
                technique_score=scoring.get("overall_score") or 0,
                letter_grade=scoring.get("letter_grade") or "--",
                detected_issues=issues,
                mobility_fit=mobility_fit,
            ))
        except Exception as e:  # noqa: BLE001
            logger.warning("FIT_PLAN_FAILED", err=str(e))

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
            fit_plan=fit_plan,
            focus=focus,
        )

    # Step 8: structured training plan (running only, deterministic -- no LLM).
    # Maps detected issues to concrete corrective drills (with cue, weeks, and
    # an optional Academy link) so the result carries an actionable "what to do
    # next" plan, not just prose. Wrapped so a builder error never kills the
    # result.
    training_plan = None
    if sport_type == "run":
        try:
            from app.services.video_analysis.biomechanics.running_action_plan_builder import (
                build_running_action_plan,
                running_action_plan_to_json,
            )
            plan = build_running_action_plan({
                "score": scoring.get("overall_score") or 0,
                "summary": summary,
            })
            training_plan = running_action_plan_to_json(plan)
        except Exception as e:  # noqa: BLE001
            logger.warning("TRAINING_PLAN_FAILED", err=str(e))


    return {
        "status": "completed",
        "sport_type": sport_type,
        "cycling_position": cycling_position if is_bike else None,
        "camera_side": analyzer.camera_side,
        "frames_analyzed": len(raw_frame_data),
        "technique_score": scoring.get("overall_score"),
        "letter_grade": scoring.get("letter_grade"),
        "score_breakdown": scoring.get("component_scores"),
        # What the score was computed from. The weighted average renormalises
        # over whatever was measurable, so without this a number from five
        # measures is indistinguishable from one built on nine.
        "score_coverage": scoring.get("coverage"),
        "quality_gate_triggered": bool(quality_gate_result.get("triggered")),
        "overlay_video_path": overlay_video_path,
        "keyframe_base64": keyframe_base64,
        "kinogram_base64": kinogram_base64,
        "kinogram": kinogram_meta,
        "ai_recommendations": ai_recommendations,
        # Echoed back so the report can show the question above its answer --
        # a reply to something you cannot see yourself having asked is hard to
        # judge.
        "focus": focus,
        "training_plan": training_plan,
        "fit_plan": fit_plan,
        # Whether the position they rode is one their off-bike range supports.
        # None when they have not done the screens -- no data is not a verdict.
        "mobility_fit": mobility_fit,
        "angle_statistics": angle_stats,
        "detected_issues": issues,
        "sport_specific_metrics": summary,
        # The bands the report grades against, and who decided them. Served
        # rather than duplicated in the client: a target the athlete cannot
        # trace is a target they have to take on faith, and the client's own
        # copy of these ranges is exactly the arrangement that lets two halves
        # of the product disagree about what "good" is.
        "reference_bands": reference_bands(
            sport_type, cycling_position if is_bike else None,
        ),
    }
