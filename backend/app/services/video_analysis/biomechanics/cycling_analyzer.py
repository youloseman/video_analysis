"""Cycling bike fit analysis module.

Analyzes pedal stroke phases, knee angles at BDC/TDC, trunk position,
and saddle height. Adapted from MEDIAPIPE_PROJECT_ANALYSIS.md section 16.2.
"""

import math
from collections import Counter
from types import SimpleNamespace
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger()

from app.services.video_analysis.biomechanics.angle_calculator import (
    SPORT_LANDMARK_VISIBILITY,
    calculate_angle_2d,
    calculate_forearm_tilt_2d,
    calculate_head_alignment_2d,
    calculate_segment_to_vertical,
    calculate_segment_to_vertical_from_points,
)
from app.services.video_analysis.biomechanics.base_analyzer import SportAnalyzer
from app.services.video_analysis.biomechanics.cycling_positions import (
    get_cycling_reference,
    get_position_label,
)
from app.services.video_analysis.biomechanics.landmarks import FrameAnalysis

NAN = float("nan")

# Ankle (knee-ankle-foot interior) angle at BDC. The recreational
# reference curve (bike_reference.json) sits at ~109 deg (SD ~9) at the
# bottom of the stroke. A markedly larger angle means the foot is
# plantarflexed -- toe pointing down -- which lengthens the effective
# leg and can make knee_at_bdc read as correctly-set while the saddle
# is actually too high. We flag toe-down past ~mean+1.8*SD so the saddle
# assessment can be caveated rather than trusted as exact.
ANKLE_BDC_PLANTARFLEXION_DEG = 125.0
# Plausibility envelope for a sampled ankle-at-BDC value; readings
# outside this are tracking noise, not a real foot angle.
_ANKLE_BDC_BOUNDS = (60.0, 170.0)

# Indices used to assess camera-side reliability per frame (shoulders+hips).
_CAMERA_SIDE_INDICES = (11, 12, 23, 24)
# How many frames to consider for the quality-vote window.
CAMERA_SIDE_EARLY_WINDOW = 30
# How many top-quality frames to vote with.
CAMERA_SIDE_TOP_K = 5
# Minimum quality score for a frame to qualify as a voter.
CAMERA_SIDE_MIN_QUALITY = 0.3


def _frame_quality_score(frame_result: dict[str, Any]) -> float:
    """0..1 score of how trustworthy a frame's camera-side signal is.

    Higher when shoulders+hips have high MediaPipe visibility AND none
    of their (x, y, z) coordinates are NaN. Quality voting picks the
    highest-scoring frames within an early window so ``camera_side``
    isn't pinned by a glitched frame[0].
    """
    wl = frame_result.get("world_landmarks") or []
    if len(wl) < max(_CAMERA_SIDE_INDICES) + 1:
        return 0.0

    visibilities: list[float] = []
    nan_count = 0
    for idx in _CAMERA_SIDE_INDICES:
        lm = wl[idx]
        visibilities.append(float(getattr(lm, "visibility", 0.0)))
        for axis in ("x", "y", "z"):
            v = getattr(lm, axis, math.nan)
            if not isinstance(v, (int, float)) or math.isnan(v):
                nan_count += 1

    if not visibilities:
        return 0.0
    avg_vis = sum(visibilities) / len(visibilities)
    nan_penalty = nan_count / (len(_CAMERA_SIDE_INDICES) * 3)
    return max(0.0, avg_vis * (1.0 - nan_penalty))


def determine_locked_camera_side(
    frame_results: list[dict[str, Any]],
    k: int = CAMERA_SIDE_TOP_K,
    early_window: int = CAMERA_SIDE_EARLY_WINDOW,
) -> tuple[str, dict[str, Any]]:
    """Pick the camera-facing side from the top-K quality frames.

    More robust than voting on frame[0] alone because:
      - frame[0] can be a MediaPipe initialization glitch.
      - Z noise on a single frame can flip the result.
      - Multiple frames vote, majority wins (tie -> "left" alphabetic).

    Returns ``(side, meta)`` where ``meta`` carries the votes used,
    average quality, and a fallback flag for diagnostics.
    """
    candidates = frame_results[:early_window]
    scored: list[tuple[int, float]] = [
        (i, _frame_quality_score(fr)) for i, fr in enumerate(candidates)
    ]
    # Stable sort by descending quality, frame index breaks ties.
    scored.sort(key=lambda x: (-x[1], x[0]))
    top = [
        (candidates[i], score)
        for i, score in scored[:k]
        if score >= CAMERA_SIDE_MIN_QUALITY
    ]

    if not top:
        return "left", {
            "votes": [],
            "fallback": True,
            "fallback_reason": "no_quality_frames",
            "quality_frames_used": 0,
            "avg_quality": 0.0,
        }

    votes: list[str] = []
    for fr, _score in top:
        wl = fr["world_landmarks"]
        try:
            l_z = (wl[11].z + wl[23].z) / 2.0
            r_z = (wl[12].z + wl[24].z) / 2.0
        except (IndexError, AttributeError):
            continue
        if math.isnan(l_z) or math.isnan(r_z):
            continue
        votes.append("left" if l_z < r_z else "right")

    if not votes:
        return "left", {
            "votes": [],
            "fallback": True,
            "fallback_reason": "no_valid_z",
            "quality_frames_used": len(top),
            "avg_quality": round(
                sum(s for _, s in top) / max(len(top), 1), 3
            ),
        }

    counts = Counter(votes)
    # Sorted by count desc, then alphabetic (deterministic tiebreak).
    side = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
    return side, {
        "votes": votes,
        "fallback": False,
        "quality_frames_used": len(top),
        "avg_quality": round(sum(s for _, s in top) / len(top), 3),
    }


def _strip_z(lm: Any) -> SimpleNamespace:
    """Project a landmark onto the sagittal (x, y) plane.

    Defense-in-depth: today the four cycling angle functions ignore z,
    but a future change to any of them would silently leak depth noise
    into the bike pipeline if we passed raw 3D landmarks through. By
    pinning z to 0 here we guarantee the cycling computation is strictly
    2D regardless of upstream landmark contents.
    """
    return SimpleNamespace(
        x=lm.x,
        y=lm.y,
        z=0.0,
        visibility=getattr(lm, "visibility", 1.0),
    )


# Ear visibility floor below which the C7 blend falls back to pure
# shoulder. Helmet brims, sunglasses, and head turns can drop ear
# confidence well under the bike-wide 0.45 floor, in which case the
# blend would amplify noise rather than reduce systematic bias.
EAR_FALLBACK_VISIBILITY = 0.4

# Weighted blend used to approximate the C7 vertebra position from
# MediaPipe's acromion (shoulder) and ear landmarks. C7 sits along the
# spine between them, behind the acromion and below the ear. A 0.7/0.3
# weighting compromises both forward biases (acromion is forward of
# C7; ear is forward of C7 too, but less so when the head is tucked)
# and brings the resulting trunk-segment angle within ~1-3 deg of the
# bike-fitter standard, vs. ~3-7 deg with shoulder alone.
SHOULDER_C7_WEIGHT = 0.7


def _blend_landmarks(
    landmark_a: Any,
    landmark_b: Any,
    weight_a: float = SHOULDER_C7_WEIGHT,
) -> SimpleNamespace:
    """Weighted blend of two landmarks into a synthetic anatomical point.

    Used to approximate MediaPipe-unavailable points (e.g. C7 vertebra)
    from adjacent landmarks. Visibility is the minimum of inputs so a
    weak input drags the blend below downstream NaN guards instead of
    silently averaging in unreliable coordinates.
    """
    weight_b = 1.0 - weight_a
    return SimpleNamespace(
        x=landmark_a.x * weight_a + landmark_b.x * weight_b,
        y=landmark_a.y * weight_a + landmark_b.y * weight_b,
        z=0.0,
        visibility=min(
            getattr(landmark_a, "visibility", 1.0),
            getattr(landmark_b, "visibility", 1.0),
        ),
    )


class CyclingAnalyzer(SportAnalyzer):
    """Analyzer for cycling bike fit technique."""

    def __init__(self, fps: float = 30.0, cycling_position: str | None = None):
        super().__init__(sport_type="bike", fps=fps)
        self.cycling_position = cycling_position
        self.reference = get_cycling_reference(cycling_position)
        self.trunk_angles: list[float] = []
        self.left_knee_angles: list[float] = []
        self.right_knee_angles: list[float] = []
        # Per-side BDC/TDC estimation diagnostics (method + stroke count),
        # populated by _get_bdc_tdc_angles for surfacing in the summary.
        self._bdc_tdc_diag: dict[str, dict[str, Any]] = {}
        self.left_head_scores: list[float] = []
        self.right_head_scores: list[float] = []
        self._min_vis = SPORT_LANDMARK_VISIBILITY.get("bike", 0.45)
        self._analyzer_warnings: list[str] = []
        # Trunk-angle method bookkeeping: per-frame visibility samples and
        # the blended-vs-fallback split. compute_summary aggregates these
        # into summary["diagnostics"]["trunk_angle_method"] and emits a
        # one-shot BIKE_TRUNK_ANGLE_METHOD log so operators can audit how
        # often the C7 blend is in use vs falling back to pure shoulder.
        self._trunk_method_counts: dict[str, int] = {
            "blended_c7": 0,
            "shoulder_only_fallback": 0,
        }
        self._ear_visibilities: list[float] = []
        self._shoulder_visibilities: list[float] = []
        # Locked on the first analyze_frame call. Bike side-view doesn't
        # physically flip during a recording, so picking the camera-facing
        # side once and reusing it removes a class of mid-clip side-flip
        # bugs. compute_summary syncs self.camera_side to this value so
        # downstream consumers (visualizer, summary) see what was actually
        # computed even if pipeline's per-frame vote majority disagrees.
        self._near_side: str | None = None

    def analyze_frame(
        self, world_landmarks: Any, normalized_landmarks: Any, timestamp_ms: float,
    ) -> FrameAnalysis:
        """Analyze a single cycling frame.

        Strict 2D sagittal-plane analysis: world_landmarks are projected
        to (x, y) with z=0 before any angle call (see ``_strip_z``).
        Only the camera-facing side is computed; the far side's keys are
        emitted as NaN so the FrameAnalysis shape stays stable for any
        downstream consumer that iterates angle keys.
        """
        mv = self._min_vis

        # Lock the near-side decision on the first frame.
        if self._near_side is None:
            try:
                self._near_side = self.detect_camera_side(world_landmarks)
            except (IndexError, AttributeError):
                self._near_side = "left"
        near = self._near_side

        # Project to 2D so any downstream angle math sees z=0 even if a
        # function were ever changed to read z.
        wl = [_strip_z(lm) for lm in world_landmarks]

        # Compute only the near-side joint angles. Far-side keys stay
        # NaN -- the data is unreliable in side-view (occluded by torso)
        # and consuming it has historically been a source of noise.
        if near == "left":
            knee, knee_vis = calculate_angle_2d(wl, 23, 25, 27, mv)
            hip, hip_vis = calculate_angle_2d(wl, 11, 23, 25, mv)
            ankle, ankle_vis = calculate_angle_2d(wl, 25, 27, 31, mv)
            shoulder, shoulder_vis = calculate_angle_2d(wl, 13, 11, 23, mv)
            elbow, elbow_vis = calculate_angle_2d(wl, 11, 13, 15, mv)
            forearm_tilt, forearm_vis = calculate_forearm_tilt_2d(wl, 13, 15, mv)
            head, _head_vis = calculate_head_alignment_2d(wl, 11, 23, 7, min_visibility=mv)
            trunk_angle = self._compute_trunk_angle(wl, "left", mv)
            left_knee, right_knee = knee, NAN
            left_hip, right_hip = hip, NAN
            left_ankle, right_ankle = ankle, NAN
            left_shoulder, right_shoulder = shoulder, NAN
            left_elbow, right_elbow = elbow, NAN
            left_forearm_tilt, right_forearm_tilt = forearm_tilt, NAN
            left_head, right_head = head, NAN
            left_knee_vis, right_knee_vis = knee_vis, NAN
            left_hip_vis, right_hip_vis = hip_vis, NAN
            left_ankle_vis, right_ankle_vis = ankle_vis, NAN
            left_shoulder_vis, right_shoulder_vis = shoulder_vis, NAN
            left_elbow_vis, right_elbow_vis = elbow_vis, NAN
            left_forearm_vis, right_forearm_vis = forearm_vis, NAN
        else:
            knee, knee_vis = calculate_angle_2d(wl, 24, 26, 28, mv)
            hip, hip_vis = calculate_angle_2d(wl, 12, 24, 26, mv)
            ankle, ankle_vis = calculate_angle_2d(wl, 26, 28, 32, mv)
            shoulder, shoulder_vis = calculate_angle_2d(wl, 14, 12, 24, mv)
            elbow, elbow_vis = calculate_angle_2d(wl, 12, 14, 16, mv)
            forearm_tilt, forearm_vis = calculate_forearm_tilt_2d(wl, 14, 16, mv)
            head, _head_vis = calculate_head_alignment_2d(wl, 12, 24, 8, min_visibility=mv)
            trunk_angle = self._compute_trunk_angle(wl, "right", mv)
            left_knee, right_knee = NAN, knee
            left_hip, right_hip = NAN, hip
            left_ankle, right_ankle = NAN, ankle
            left_shoulder, right_shoulder = NAN, shoulder
            left_elbow, right_elbow = NAN, elbow
            left_forearm_tilt, right_forearm_tilt = NAN, forearm_tilt
            left_head, right_head = NAN, head
            left_knee_vis, right_knee_vis = NAN, knee_vis
            left_hip_vis, right_hip_vis = NAN, hip_vis
            left_ankle_vis, right_ankle_vis = NAN, ankle_vis
            left_shoulder_vis, right_shoulder_vis = NAN, shoulder_vis
            left_elbow_vis, right_elbow_vis = NAN, elbow_vis
            left_forearm_vis, right_forearm_vis = NAN, forearm_vis

        self.trunk_angles.append(trunk_angle)
        self.left_knee_angles.append(left_knee)
        self.right_knee_angles.append(right_knee)
        self.left_head_scores.append(left_head)
        self.right_head_scores.append(right_head)

        frame = FrameAnalysis(
            timestamp_ms=timestamp_ms,
            angles={
                "left_knee": left_knee,
                "right_knee": right_knee,
                "left_hip": left_hip,
                "right_hip": right_hip,
                "left_ankle": left_ankle,
                "right_ankle": right_ankle,
                "left_shoulder": left_shoulder,
                "right_shoulder": right_shoulder,
                "left_elbow": left_elbow,
                "right_elbow": right_elbow,
                "left_forearm_tilt": left_forearm_tilt,
                "right_forearm_tilt": right_forearm_tilt,
                "trunk_angle": trunk_angle,
            },
            visibility={
                "left_knee": left_knee_vis,
                "right_knee": right_knee_vis,
                "left_hip": left_hip_vis,
                "right_hip": right_hip_vis,
                "left_ankle": left_ankle_vis,
                "right_ankle": right_ankle_vis,
                "left_shoulder": left_shoulder_vis,
                "right_shoulder": right_shoulder_vis,
                "left_elbow": left_elbow_vis,
                "right_elbow": right_elbow_vis,
                "left_forearm_tilt": left_forearm_vis,
                "right_forearm_tilt": right_forearm_vis,
            },
            extra_metrics={
                "left_head_alignment": left_head,
                "right_head_alignment": right_head,
            },
        )
        return frame

    def _compute_trunk_angle(
        self, wl: list[Any], near: str, min_visibility: float,
    ) -> float:
        """Bike trunk angle via blended C7 approximation, with ear-fallback.

        Approximates the bike-fitter standard C7-vertebra -> hip vector
        by blending the MediaPipe acromion (idx 11/12) with the ear
        (idx 7/8). Acromion alone sits ~3-5 cm forward of C7 when the
        rider is leaning forward, which biases the shoulder->hip vector
        toward more horizontal -- understating trunk angle by 3-7 deg
        relative to fitter convention. The 0.7/0.3 blend halves that.

        Falls back to pure shoulder when ear visibility is below
        ``EAR_FALLBACK_VISIBILITY`` (helmet brims, head turns, sunglass
        glare). Returns NaN-via-downstream when min_visibility gates
        either input.

        Side effects: appends per-frame visibilities to the analyzer
        bookkeeping and increments the method counter; emits a one-shot
        BIKE_TRUNK_FALLBACK log on the first fallback frame.
        """
        if near == "left":
            sh_idx, ear_idx, hip_idx = 11, 7, 23
        else:
            sh_idx, ear_idx, hip_idx = 12, 8, 24

        shoulder = wl[sh_idx]
        ear = wl[ear_idx]
        hip = wl[hip_idx]

        sh_vis = float(getattr(shoulder, "visibility", 1.0))
        ear_vis = float(getattr(ear, "visibility", 1.0))
        self._shoulder_visibilities.append(sh_vis)
        self._ear_visibilities.append(ear_vis)

        if ear_vis >= EAR_FALLBACK_VISIBILITY:
            c7_approx = _blend_landmarks(
                shoulder, ear, weight_a=SHOULDER_C7_WEIGHT,
            )
            angle_from_vertical = calculate_segment_to_vertical_from_points(
                top=c7_approx, bottom=hip, min_visibility=min_visibility,
            )
            self._trunk_method_counts["blended_c7"] += 1
        else:
            angle_from_vertical = calculate_segment_to_vertical(
                wl, sh_idx, hip_idx, min_visibility,
            )
            self._trunk_method_counts["shoulder_only_fallback"] += 1
            # First fallback frame only -- avoid per-frame log spam on
            # clips where the helmet hides the ear for the entire run.
            if self._trunk_method_counts["shoulder_only_fallback"] == 1:
                logger.info(
                    "BIKE_TRUNK_FALLBACK",
                    reason="ear_visibility_below_threshold",
                    ear_visibility=round(ear_vis, 3),
                    threshold=EAR_FALLBACK_VISIBILITY,
                    near=near,
                )

        if math.isnan(angle_from_vertical):
            return float("nan")
        return 90.0 - angle_from_vertical

    def _bdc_tdc_from_peaks(self, arr: np.ndarray) -> dict[str, float] | None:
        """Per-revolution BDC/TDC from peak/valley detection.

        Each pedal revolution produces one knee-extension peak (BDC)
        and one knee-flexion valley (TDC). Detecting them with
        ``scipy.find_peaks`` and taking the median across revolutions
        is robust to outlier frames and -- unlike a fixed percentile --
        unbiased by how long the knee dwells near each extreme. It also
        yields stroke-to-stroke variability, which the percentile
        method cannot.

        NaN gaps are linearly interpolated only to keep the signal
        continuous for peak finding; detected peaks landing on an
        interpolated (originally-NaN) sample are discarded so reported
        values always come from real measurements.

        Returns ``None`` when fewer than two clean revolutions are
        detected, so the caller can fall back to the percentile method
        on short / irregular clips.
        """
        from scipy.signal import find_peaks

        n = len(arr)
        if n < 5:
            return None
        idx = np.arange(n)
        mask = ~np.isnan(arr)
        if int(mask.sum()) < 5:
            return None

        # Interpolate NaN gaps for a continuous signal.
        filled = arr.astype(np.float64).copy()
        filled[~mask] = np.interp(idx[~mask], idx[mask], arr[mask])

        valid_vals = arr[mask]
        rng = float(np.percentile(valid_vals, 95) - np.percentile(valid_vals, 5))
        # Prominence floor of 8 deg rejects noise wobble; a real pedal
        # stroke spans ~60-80 deg of knee ROM so this never suppresses a
        # genuine revolution.
        prominence = max(8.0, 0.25 * rng)
        # Minimum spacing ~0.4 s between same-type extremes caps the
        # implied cadence at ~150 rpm and prevents double-counting a
        # single revolution. Floor of 3 frames guards very low fps.
        distance = max(3, int(round(self.fps * 0.4)))

        bdc_idx, _ = find_peaks(filled, distance=distance, prominence=prominence)
        tdc_idx, _ = find_peaks(-filled, distance=distance, prominence=prominence)
        bdc_idx = [int(i) for i in bdc_idx if mask[i]]
        tdc_idx = [int(i) for i in tdc_idx if mask[i]]
        if len(bdc_idx) < 2 or len(tdc_idx) < 2:
            return None

        bdc_vals = arr[bdc_idx]
        tdc_vals = arr[tdc_idx]
        return {
            "bdc": float(np.median(bdc_vals)),
            "tdc": float(np.median(tdc_vals)),
            "bdc_std": float(np.std(bdc_vals)),
            "tdc_std": float(np.std(tdc_vals)),
            "n_strokes": float(min(len(bdc_idx), len(tdc_idx))),
            # Real (non-interpolated) frame indices of the BDC extrema,
            # so the caller can sample other joint angles (e.g. ankle)
            # at the bottom of the pedal stroke.
            "bdc_indices": bdc_idx,
        }

    def _get_bdc_tdc_angles(self) -> dict[str, float]:
        """Estimate BDC (max extension) and TDC (max flexion) knee angles.

        BDC = bottom dead center = maximum knee extension (highest angle)
        TDC = top dead center = maximum knee flexion (lowest angle)

        Primary method: per-revolution peak/valley detection (see
        :meth:`_bdc_tdc_from_peaks`). Falls back to the 95th/5th
        percentile when too few clean revolutions are detected, which
        preserves the previous behaviour on short / irregular clips.

        NaN-safe + physiological outlier protection (30-170 deg range).
        Records per-side estimation method, stroke count and BDC/TDC
        variability in ``self._bdc_tdc_diag`` for the summary.
        """
        result: dict[str, float] = {}
        self._bdc_tdc_diag = {}
        for side, angles in [("left", self.left_knee_angles), ("right", self.right_knee_angles)]:
            if len(angles) < 5:
                continue
            arr = np.array(angles, dtype=np.float64)
            valid = arr[~np.isnan(arr)]
            # Physiological bounds: knee angle 30-170 deg in cycling
            valid = valid[(valid >= 30) & (valid <= 170)]
            if len(valid) < 5:
                continue

            peaks = self._bdc_tdc_from_peaks(arr)
            if peaks is not None:
                bdc_val = peaks["bdc"]
                tdc_val = peaks["tdc"]
                method = "peaks"
            else:
                bdc_val = float(np.percentile(valid, 95))
                tdc_val = float(np.percentile(valid, 5))
                method = "percentile"

            # Sanity: TDC should be < 100 deg (knee flexed at top of pedal stroke)
            if tdc_val > 100:
                # Percentile path can retry a less-extreme percentile;
                # the peak median is already robust, so just flag it.
                if method == "percentile":
                    tdc_val = float(np.percentile(valid, 10))
                if tdc_val > 100:
                    logger.warning("TDC_OUTLIER", side=side, tdc=tdc_val, method=method)
                    self._record_knee_outlier_warning()
                    continue  # Data too corrupted for this side

            # Sanity: BDC should be > 110 deg (knee extended at bottom)
            if bdc_val < 110:
                logger.warning("BDC_OUTLIER", side=side, bdc=bdc_val, method=method)
                self._record_knee_outlier_warning()
                continue

            result[f"{side}_knee_at_bdc"] = bdc_val
            result[f"{side}_knee_at_tdc"] = tdc_val
            diag: dict[str, Any] = {"method": method}
            if peaks is not None:
                diag["pedal_strokes"] = int(peaks["n_strokes"])
                diag["bdc_variability_deg"] = round(peaks["bdc_std"], 1)
                diag["tdc_variability_deg"] = round(peaks["tdc_std"], 1)
                result[f"{side}_bdc_variability_deg"] = round(peaks["bdc_std"], 1)
                # Ankle (foot) angle at the bottom of the stroke. Sampled
                # at the same real BDC frames the knee peaks landed on, so
                # it reflects foot posture exactly when the leg is most
                # extended -- the point where toe-pointing masks saddle
                # height. Only available on the peak path (the percentile
                # fallback has no per-stroke indices).
                ankle_at_bdc = self._ankle_at_bdc(side, peaks["bdc_indices"])
                if ankle_at_bdc is not None:
                    diag["ankle_at_bdc"] = round(ankle_at_bdc, 1)
                    if ankle_at_bdc > ANKLE_BDC_PLANTARFLEXION_DEG:
                        diag["plantarflexion_at_bdc"] = True
                        result[f"{side}_plantarflexion_at_bdc"] = True
                        self._record_plantarflexion_warning()
            self._bdc_tdc_diag[side] = diag
        return result

    def _ankle_at_bdc(
        self, side: str, bdc_indices: list[int],
    ) -> float | None:
        """Median ankle interior angle across the BDC frames for a side.

        Reads the per-frame ankle series accumulated in ``angle_history``
        (aligned 1:1 with the knee series since both are appended once per
        frame in :meth:`analyze_frame`). Returns ``None`` when the ankle
        was never measured, every BDC sample is NaN, or the result falls
        outside :data:`_ANKLE_BDC_BOUNDS` (tracking noise, not a foot).
        """
        ankle_series = self.angle_history.get(f"{side}_ankle")
        if not ankle_series:
            return None
        n = len(ankle_series)
        samples = [
            ankle_series[i]
            for i in bdc_indices
            if 0 <= i < n and not math.isnan(ankle_series[i])
        ]
        if not samples:
            return None
        val = float(np.median(samples))
        lo, hi = _ANKLE_BDC_BOUNDS
        if val < lo or val > hi:
            return None
        return val

    def _record_plantarflexion_warning(self) -> None:
        """Append a deduplicated user-facing plantarflexion caveat."""
        msg = (
            "Your foot points down (toes low) at the bottom of the pedal "
            "stroke. This stretches the leg and can make the saddle look "
            "correctly set when it is actually too high -- re-check saddle "
            "height with a flatter foot before trusting the knee-angle reading."
        )
        if msg not in self._analyzer_warnings:
            self._analyzer_warnings.append(msg)

    def _record_knee_outlier_warning(self) -> None:
        """Append a deduplicated user-facing warning for BDC/TDC outliers."""
        msg = (
            "Knee angle range looks unusual for pedal stroke detection -- "
            "saddle height assessment may be less reliable. Make sure the full "
            "leg is visible and the camera is stable at hip height."
        )
        if msg not in self._analyzer_warnings:
            self._analyzer_warnings.append(msg)

    def _assess_saddle_height(self, knee_at_bdc: float) -> str:
        """Assess saddle height based on knee angle at BDC."""
        opt_min, opt_max = self.reference["knee_at_bdc"]
        if knee_at_bdc < opt_min - 5:
            return "too_low"
        elif knee_at_bdc > opt_max + 5:
            return "too_high"
        elif opt_min <= knee_at_bdc <= opt_max:
            return "optimal"
        else:
            return "acceptable"

    def _nan_safe_mean(self, values: list[float]) -> float | None:
        """Compute mean ignoring NaN values. Returns None if no valid values."""
        if not values:
            return None
        arr = np.array(values, dtype=np.float64)
        valid = arr[~np.isnan(arr)]
        return float(np.mean(valid)) if len(valid) > 0 else None

    # ------------------------------------------------------------------
    # Plausibility envelopes for compute_summary -- mirror the bounds in
    # action_plan_builder._IMPLAUSIBLE_BOUNDS so the analyzer never emits
    # a phantom-zero (or impossibly extreme) numeric metric. "Absent =
    # no card" is the convention shared with the swim pipeline.
    # ------------------------------------------------------------------
    _SUMMARY_BOUNDS: dict[str, tuple[float, float]] = {
        "knee_at_bdc":      (90.0, 175.0),
        "knee_at_tdc":      (30.0, 110.0),
        "trunk_angle_avg":  (5.0,  80.0),
        "elbow_angle_avg":  (60.0, 180.0),
        "shoulder_angle_avg": (50.0, 150.0),
        "hip_angle_avg":    (25.0, 100.0),
        "pelvic_ratio":     (1.0,  8.0),
        "forearm_tilt_avg": (-30.0, 45.0),
    }

    def _set_if_plausible(
        self, summary: dict[str, Any], key: str, value: float | None,
    ) -> None:
        """Write a metric to the summary only if it is plausible.

        ``None`` and out-of-envelope readings are dropped silently --
        downstream consumers (action plan builder, technique scorer)
        treat absence as "no measurement", which is what we want when
        the upstream computation failed or produced noise.
        """
        if value is None:
            return
        if isinstance(value, float) and np.isnan(value):
            return
        bounds = self._SUMMARY_BOUNDS.get(key)
        if bounds is not None:
            lo, hi = bounds
            if value < lo or value > hi:
                return
        summary[key] = round(float(value), 2)

    def compute_summary(self) -> dict[str, Any]:
        """Aggregate cycling metrics (bike fit report).

        Uses near-side (camera-facing) angles for primary metrics.
        Far-side data is unreliable in side-view cycling video.

        Numeric metrics are *omitted* from the summary when the
        upstream computation failed or the reading is biomechanically
        implausible -- that absence is the signal to the action plan
        builder and front-end to suppress the related card.
        """
        if not self.frame_results:
            return {}

        # Force camera_side to whatever analyze_frame actually computed.
        # The pipeline's per-frame vote may pick a different side from
        # the analyzer's first-frame lock; if so, the angle_history only
        # has values under self._near_side, so we must report that side
        # for the summary/visualizer to read non-NaN data.
        if self._near_side is not None and self.camera_side != self._near_side:
            logger.info(
                "BIKE_NEAR_SIDE_OVERRIDE",
                pipeline_vote=self.camera_side,
                analyzer_locked=self._near_side,
            )
            self.camera_side = self._near_side

        near = self.get_near_side_prefix()
        far = self.get_far_side_prefix()

        bdc_tdc = self._get_bdc_tdc_angles()

        # Trunk angle (NaN-safe). None if no valid frames.
        trunk_vals = [v for v in self.trunk_angles if not np.isnan(v)]
        trunk_avg: float | None = (
            float(np.mean(trunk_vals)) if trunk_vals else None
        )

        # Near-side elbow (primary metric); fall back to far side only
        # when the near side has no measurement.
        near_elbow_vals = self.angle_history.get(f"{near}_elbow", [])
        far_elbow_vals = self.angle_history.get(f"{far}_elbow", [])
        near_elbow_avg = self._nan_safe_mean(near_elbow_vals)
        far_elbow_avg = self._nan_safe_mean(far_elbow_vals)
        elbow_avg = near_elbow_avg if near_elbow_avg is not None else far_elbow_avg

        # Near-side shoulder angle
        near_shoulder_vals = self.angle_history.get(f"{near}_shoulder", [])
        shoulder_avg = self._nan_safe_mean(near_shoulder_vals)

        # Near-side forearm tilt
        near_forearm_vals = self.angle_history.get(f"{near}_forearm_tilt", [])
        forearm_tilt_avg = self._nan_safe_mean(near_forearm_vals)

        # Head alignment (from per-frame accumulator, not angle_history)
        head_scores = self.left_head_scores if near == "left" else self.right_head_scores
        head_alignment_avg = self._nan_safe_mean(head_scores)

        # Near-side hip angle
        near_hip_vals = self.angle_history.get(f"{near}_hip", [])
        hip_avg = self._nan_safe_mean(near_hip_vals)

        # Pelvic ratio (derived: mean hip / trunk avg). Both inputs must
        # be real measurements -- a phantom 0 in either would corrupt
        # the ratio.
        pelvic_ratio: float | None = None
        if hip_avg is not None and trunk_avg is not None and trunk_avg >= 5:
            pelvic_ratio = hip_avg / trunk_avg

        # Near-side BDC for saddle assessment (primary). _assess_saddle_height
        # already returns "insufficient_data" when the input is falsy.
        near_bdc = bdc_tdc.get(f"{near}_knee_at_bdc")
        far_bdc = bdc_tdc.get(f"{far}_knee_at_bdc")
        bdc_for_assessment = near_bdc if near_bdc else far_bdc
        saddle_assessment = (
            self._assess_saddle_height(bdc_for_assessment)
            if bdc_for_assessment else "insufficient_data"
        )

        # Position archetype detection -- needs all four angles. Skip
        # when any is missing rather than feeding 0.0 sentinels in (the
        # detector's own NaN/zero guard would reject anyway, but being
        # explicit here keeps the intent obvious).
        from app.services.video_analysis.biomechanics.cycling_positions import detect_position_archetype
        archetype = None
        if (
            shoulder_avg is not None
            and elbow_avg is not None
            and trunk_avg is not None
            and hip_avg is not None
        ):
            archetype = detect_position_archetype(
                shoulder_angle=shoulder_avg,
                elbow_angle=elbow_avg,
                trunk_angle=trunk_avg,
                hip_angle=hip_avg,
                cycling_position=self.cycling_position,
            )

        # Build summary -- numeric metrics are written only when both
        # present and within their plausibility envelope.
        summary: dict[str, Any] = {
            "saddle_height_assessment": saddle_assessment,
            "position_archetype": archetype,
            "frames_analyzed": len(self.frame_results),
            # Unilateral focus info
            "camera_side": self.camera_side,
            "near_side": near,
            "camera_side_label": near.capitalize() if near else "Left",
        }

        # Near-side knee BDC/TDC (backward-compat keys consumed by the
        # action plan builder and technique scorer).
        self._set_if_plausible(summary, "knee_at_bdc", bdc_tdc.get(f"{near}_knee_at_bdc"))
        self._set_if_plausible(summary, "knee_at_tdc", bdc_tdc.get(f"{near}_knee_at_tdc"))

        # BDC/TDC estimation diagnostics for the near (analysed) side:
        # which method produced the values, how many pedal revolutions
        # backed them, and the stroke-to-stroke BDC variability. High
        # variability (e.g. > 8 deg) flags an inconsistent pedal stroke
        # or noisy tracking, letting the frontend/LLM caveat the saddle
        # assessment instead of treating a single number as exact.
        near_diag = self._bdc_tdc_diag.get(near)
        if near_diag:
            bdc_tdc_diag: dict[str, Any] = {
                "method": near_diag.get("method"),
                "pedal_strokes": near_diag.get("pedal_strokes"),
                "bdc_variability_deg": near_diag.get("bdc_variability_deg"),
                "tdc_variability_deg": near_diag.get("tdc_variability_deg"),
            }
            # Foot posture at BDC -- present only when measurable on the
            # peak path. plantarflexion_at_bdc lowers confidence in the
            # saddle-height assessment (toe-down inflates knee extension).
            if "ankle_at_bdc" in near_diag:
                bdc_tdc_diag["ankle_at_bdc"] = near_diag["ankle_at_bdc"]
                # Also surface as a top-level metric so the card system
                # renders it like any other angle (Foot @ BDC) -- makes
                # the saddle-height foot signal observable, not a hidden
                # pass/fail flag.
                summary["ankle_at_bdc"] = near_diag["ankle_at_bdc"]
            if near_diag.get("plantarflexion_at_bdc"):
                bdc_tdc_diag["plantarflexion_at_bdc"] = True
            summary.setdefault("diagnostics", {})["bdc_tdc"] = bdc_tdc_diag
        self._set_if_plausible(summary, "trunk_angle_avg", trunk_avg)

        # Relative aero read-out from the (plausible) trunk angle. Only
        # surfaces for road/TT/tri positions; None otherwise. This is a
        # qualitative CdA *zone* + drag/watt delta, never an absolute
        # CdA -- see aero_estimator.py for the methodology and caveats.
        aero_trunk = summary.get("trunk_angle_avg")
        if aero_trunk is not None:
            from app.services.video_analysis.biomechanics.aero_estimator import (
                estimate_aero,
            )
            aero = estimate_aero(
                aero_trunk,
                self.cycling_position,
                optimal_trunk_band=self.reference.get("trunk_angle"),
            )
            if aero is not None:
                summary["aero_estimate"] = aero

        self._set_if_plausible(summary, "elbow_angle_avg", elbow_avg)
        self._set_if_plausible(summary, "shoulder_angle_avg", shoulder_avg)
        self._set_if_plausible(summary, "forearm_tilt_avg", forearm_tilt_avg)
        self._set_if_plausible(summary, "hip_angle_avg", hip_avg)
        self._set_if_plausible(summary, "pelvic_ratio", pelvic_ratio)

        # Head alignment is a 0-100 score -- structurally bounded, no
        # plausibility filter needed; only omit when no data at all.
        if head_alignment_avg is not None and not np.isnan(head_alignment_avg):
            summary["head_alignment_avg"] = round(float(head_alignment_avg), 2)

        if self._analyzer_warnings:
            existing = summary.get("analysis_warnings", [])
            summary["analysis_warnings"] = existing + list(self._analyzer_warnings)

        # Trunk-angle method diagnostics. Populated even when blended_c7
        # is the sole method so downstream consumers can rely on the
        # field's presence rather than its absence-as-default.
        blended_n = self._trunk_method_counts["blended_c7"]
        fallback_n = self._trunk_method_counts["shoulder_only_fallback"]
        total_n = blended_n + fallback_n
        if total_n > 0:
            primary_method = (
                "blended_c7" if blended_n >= fallback_n
                else "shoulder_only_fallback"
            )
            avg_ear_vis = (
                round(float(np.mean(self._ear_visibilities)), 3)
                if self._ear_visibilities else 0.0
            )
            avg_sh_vis = (
                round(float(np.mean(self._shoulder_visibilities)), 3)
                if self._shoulder_visibilities else 0.0
            )
            summary.setdefault("diagnostics", {})["trunk_angle_method"] = {
                "method": primary_method,
                "blended_frames": blended_n,
                "fallback_frames": fallback_n,
                "fallback_pct": round(fallback_n / total_n * 100, 1),
                "avg_ear_visibility": avg_ear_vis,
                "avg_shoulder_visibility": avg_sh_vis,
            }
            logger.info(
                "BIKE_TRUNK_ANGLE_METHOD",
                method=primary_method,
                blended_frames=blended_n,
                fallback_frames=fallback_n,
                fallback_pct=round(fallback_n / total_n * 100, 1),
                avg_ear_visibility=avg_ear_vis,
                avg_shoulder_visibility=avg_sh_vis,
            )

        return summary

    def detect_issues(self) -> list[dict[str, Any]]:
        """Detect cycling/bike fit issues using near-side angles only."""
        issues: list[dict[str, Any]] = []
        if not self.frame_results:
            return issues

        near = self.get_near_side_prefix()
        bdc_tdc = self._get_bdc_tdc_angles()

        # Saddle height issues (near-side only -- far side unreliable)
        bdc_key = f"{near}_knee_at_bdc"
        if bdc_key in bdc_tdc:
            bdc_val = bdc_tdc[bdc_key]
            opt_min, opt_max = self.reference["knee_at_bdc"]
            if bdc_val < opt_min - 5:
                issues.append({
                    "type": "saddle_too_low",
                    "severity": "warning",
                    "value": f"Knee at BDC: {bdc_val:.0f} deg",
                    "recommendation": f"Saddle may be too low. Knee at BDC is {bdc_val:.0f} deg (optimal: {opt_min}-{opt_max} deg).",
                })
            elif bdc_val > opt_max + 5:
                issues.append({
                    "type": "saddle_too_high",
                    "severity": "warning",
                    "value": f"Knee at BDC: {bdc_val:.0f} deg",
                    "recommendation": f"Saddle may be too high. Knee at BDC is {bdc_val:.0f} deg (optimal: {opt_min}-{opt_max} deg).",
                })

        # Trunk position (NaN-safe)
        trunk_vals = [v for v in self.trunk_angles if not np.isnan(v)]
        if trunk_vals:
            trunk_avg = float(np.mean(trunk_vals))
            opt_min, opt_max = self.reference["trunk_angle"]
            if trunk_avg < opt_min - 5:
                issues.append({
                    "type": "trunk_too_aggressive",
                    "severity": "info",
                    "value": f"{trunk_avg:.0f} deg",
                    "recommendation": f"Very aggressive trunk position ({trunk_avg:.0f} deg). May cause discomfort. Optimal for {get_position_label(self.cycling_position)}: {opt_min}-{opt_max} deg.",
                })
            elif trunk_avg > opt_max + 10:
                issues.append({
                    "type": "trunk_too_upright",
                    "severity": "info",
                    "value": f"{trunk_avg:.0f} deg",
                    "recommendation": f"Trunk is quite upright ({trunk_avg:.0f} deg). Less aerodynamic. Optimal for {get_position_label(self.cycling_position)}: {opt_min}-{opt_max} deg.",
                })

        return issues
