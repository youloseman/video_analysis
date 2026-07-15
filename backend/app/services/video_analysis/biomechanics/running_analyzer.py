"""Running gait analysis module.

Detects gait phases, computes cadence, vertical oscillation, and identifies
running technique issues. Adapted from MEDIAPIPE_PROJECT_ANALYSIS.md section 16.1.

Unilateral focus v4:
- Near-side only: computes ONLY camera-facing side angles
- Unprefixed keys: 'knee', 'hip', 'ankle', 'elbow', 'trunk' (no left_/right_)
- Cadence: single-leg peak detection x2 (one stride = two steps)
- No far-side data stored or displayed
"""

import math
from collections import deque
from enum import Enum
from typing import Any

import numpy as np
import structlog

from app.services.video_analysis.biomechanics.angle_calculator import (
    SPORT_LANDMARK_VISIBILITY,
    calculate_angle_2d,
    calculate_segment_to_vertical,
)
from app.services.video_analysis.biomechanics.base_analyzer import SportAnalyzer
from app.services.video_analysis.biomechanics.landmarks import FrameAnalysis
from app.services.video_analysis.biomechanics.sport_configs import RUNNING_REFERENCE

logger = structlog.get_logger()

# Near-side angle definitions per camera side
RUNNING_ANGLES: dict[str, dict[str, tuple[int, int, int]]] = {
    "left": {
        "knee":  (23, 25, 27),   # LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
        "hip":   (11, 23, 25),   # LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE
        "ankle": (25, 27, 29),   # LEFT_KNEE, LEFT_ANKLE, LEFT_HEEL
        "elbow": (11, 13, 15),   # LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST
    },
    "right": {
        "knee":  (24, 26, 28),   # RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
        "hip":   (12, 24, 26),   # RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE
        "ankle": (26, 28, 30),   # RIGHT_KNEE, RIGHT_ANKLE, RIGHT_HEEL
        "elbow": (12, 14, 16),   # RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST
    },
}

TRUNK_LANDMARKS: dict[str, tuple[int, int]] = {
    "left":  (11, 23),   # LEFT_SHOULDER, LEFT_HIP
    "right": (12, 24),   # RIGHT_SHOULDER, RIGHT_HIP
}


class GaitPhase(str, Enum):
    """Phases of the running gait cycle."""

    INITIAL_CONTACT = "initial_contact"
    LOADING_RESPONSE = "loading_response"
    MIDSTANCE = "midstance"
    TERMINAL_STANCE = "terminal_stance"
    PRE_SWING = "pre_swing"
    INITIAL_SWING = "initial_swing"
    MID_SWING = "mid_swing"
    TERMINAL_SWING = "terminal_swing"
    UNKNOWN = "unknown"


# Phases where the foot is on the ground (stance). Used to measure
# ground contact time (GCT) as the duration of contiguous stance runs.
# detect_gait_phase only emits a subset of these from its
# ground-contact branch today; loading_response is included for
# forward-compat with future detector revisions.
GROUND_CONTACT_PHASES: frozenset[str] = frozenset({
    GaitPhase.INITIAL_CONTACT.value,
    GaitPhase.LOADING_RESPONSE.value,
    GaitPhase.MIDSTANCE.value,
    GaitPhase.TERMINAL_STANCE.value,
    GaitPhase.PRE_SWING.value,
})


class RunningAnalyzer(SportAnalyzer):
    """Analyzer for running technique -- near-side only."""

    def __init__(self, fps: float = 30.0):
        super().__init__(sport_type="run", fps=fps)
        self.near_ankle_y_history: deque[float] = deque(maxlen=int(fps * 2))
        self.hip_center_y_history: deque[float] = deque(maxlen=int(fps * 2))
        self.prev_near_phase = GaitPhase.UNKNOWN
        self.trunk_lean_values: list[float] = []
        # Normalized-coordinate hip Y for vertical oscillation (image-space)
        self.norm_hip_y_history: list[float] = []
        self.norm_hip_y_timestamps: list[float] = []
        # Scale factor: normalized coords -> meters (estimated from body proportions)
        self._pixel_to_meter: float = 2.0  # default fallback
        self._body_scale_estimated = False
        self._analyzer_warnings: list[str] = []

    def _estimate_body_scale(self, nl: Any) -> None:
        """Estimate pixels-to-meters scale from body proportions in normalized coords.

        Uses shoulder-to-hip vertical distance. Average adult torso ~0.45m.
        """
        sh_y = (nl[11].y + nl[12].y) / 2  # shoulder center Y
        hp_y = (nl[23].y + nl[24].y) / 2  # hip center Y
        norm_torso = abs(hp_y - sh_y)

        if norm_torso > 0.03:
            self._pixel_to_meter = 0.45 / norm_torso
        else:
            self._pixel_to_meter = 2.0  # fallback

        self._body_scale_estimated = True
        logger.info(
            "BODY_SCALE_DEBUG",
            shoulder_y=f"{sh_y:.4f}",
            hip_y=f"{hp_y:.4f}",
            norm_torso=f"{norm_torso:.4f}",
            pixel_to_meter=f"{self._pixel_to_meter:.2f}",
        )

    def detect_gait_phase(
        self, knee_angle: float, ankle_y: float, hip_y: float,
        foot_y: float, ankle_y_velocity: float,
    ) -> GaitPhase:
        """Determine running gait phase from kinematic data."""
        is_ground_contact = ankle_y_velocity < 0.001 and foot_y > hip_y

        if is_ground_contact:
            if knee_angle > 160:
                return GaitPhase.INITIAL_CONTACT
            elif knee_angle > 140:
                return GaitPhase.MIDSTANCE
            elif knee_angle > 120:
                return GaitPhase.TERMINAL_STANCE
            else:
                return GaitPhase.PRE_SWING
        else:
            if knee_angle < 100:
                return GaitPhase.MID_SWING
            elif knee_angle < 130:
                return GaitPhase.INITIAL_SWING
            else:
                return GaitPhase.TERMINAL_SWING

    def _detect_steps_from_knee_angles(self) -> list[float]:
        """Detect strides using near-side knee angle oscillation pattern.

        Each valley (local minimum) in knee angle = max knee flexion in swing.
        Consecutive valleys of the same knee = one stride = 2 steps.
        """
        if not self.frame_results:
            return []

        # Unprefixed key -- near-side only
        knee_key = "knee"

        knee_angles: list[float] = []
        timestamps: list[float] = []

        for fr in self.frame_results:
            val = fr.angles.get(knee_key, 0)
            if val > 0:
                knee_angles.append(val)
                timestamps.append(fr.timestamp_ms)

        if len(knee_angles) < 5:
            return []

        # Find local minima using mean-crossing valley detection
        mean_angle = sum(knee_angles) / len(knee_angles)
        steps: list[float] = []
        in_valley = False
        valley_start_idx = 0

        for i in range(1, len(knee_angles) - 1):
            if knee_angles[i] < mean_angle and not in_valley:
                in_valley = True
                valley_start_idx = i
            elif knee_angles[i] >= mean_angle and in_valley:
                in_valley = False
                valley_region = knee_angles[valley_start_idx:i]
                min_idx = valley_start_idx + valley_region.index(min(valley_region))
                steps.append(timestamps[min_idx])

        logger.info(
            "CADENCE_KNEE_DEBUG",
            knee_key=knee_key,
            num_angles=len(knee_angles),
            mean_angle=f"{mean_angle:.1f}",
            num_steps=len(steps),
        )

        return steps

    def _cadence_from_ankle_position(self) -> float:
        """Fallback: count steps by tracking ankle Y-position oscillation.

        Uses normalized landmarks (image-space). When running, each ankle
        goes up (swing) and down (stance). Count zero-crossings of the
        deviation from mean = step transitions.
        """
        if not self.frame_results:
            return 0.0

        ankle_y_values: list[float] = []
        timestamps: list[float] = []

        for fr in self.frame_results:
            # Use per-frame normalized ankle Y stored in extra_metrics
            left_ankle_y = fr.extra_metrics.get("_norm_left_ankle_y", 0)
            right_ankle_y = fr.extra_metrics.get("_norm_right_ankle_y", 0)
            if left_ankle_y > 0 and right_ankle_y > 0:
                # Average both ankles - the combined signal has 2x step frequency
                avg_y = (left_ankle_y + right_ankle_y) / 2
                ankle_y_values.append(avg_y)
                timestamps.append(fr.timestamp_ms)

        if len(ankle_y_values) < 6:
            return 0.0

        mean_y = sum(ankle_y_values) / len(ankle_y_values)
        crossings = 0
        above = ankle_y_values[0] > mean_y

        for y in ankle_y_values[1:]:
            now_above = y > mean_y
            if now_above != above:
                crossings += 1
                above = now_above

        # Each step = 2 crossings (up + down) for one ankle
        # But we averaged both ankles, so each step produces 2 crossings
        num_steps = crossings / 2

        if num_steps < 1:
            return 0.0

        total_time_sec = (timestamps[-1] - timestamps[0]) / 1000.0
        if total_time_sec < 0.5:
            return 0.0

        cadence_spm = (num_steps / total_time_sec) * 60.0

        logger.info(
            "CADENCE_ANKLE_DEBUG",
            num_values=len(ankle_y_values),
            crossings=crossings,
            num_steps=f"{num_steps:.1f}",
            total_time_sec=f"{total_time_sec:.2f}",
            cadence_spm=f"{cadence_spm:.1f}",
        )

        # Sanity check: 100-240 spm for running
        if 100 < cadence_spm < 240:
            return round(cadence_spm, 1)

        return 0.0

    def _compute_cadence(self) -> float:
        """Compute cadence with 2 methods: knee angle valleys (primary), ankle position (fallback).

        Returns ``0.0`` when neither method produced a reliable
        cadence -- the caller (compute_summary) is responsible for
        omitting the value from the summary in that case so a
        phantom 0.0 doesn't reach downstream consumers as a
        measurement.
        """
        if self.frame_results:
            duration_ms = self.frame_results[-1].timestamp_ms - self.frame_results[0].timestamp_ms
            logger.info(
                "CADENCE_INPUT_DEBUG",
                total_frames=len(self.frame_results),
                video_duration_ms=round(duration_ms),
            )

        # Method 1: Knee angle oscillation (valleys = stride boundaries)
        steps = self._detect_steps_from_knee_angles()
        n_strides = len(steps)
        cadence = self._compute_cadence_from_strides(steps)
        if cadence > 0:
            logger.info("CADENCE_RESULT", method="knee_angles", cadence_spm=f"{cadence:.1f}")
            if n_strides < 4:
                self._record_warning(
                    f"Only {n_strides} strides detected -- cadence estimate may be "
                    f"imprecise. Longer clips (15+ seconds) give more reliable results."
                )
            return cadence

        # Method 2: Ankle Y-position oscillation (fallback)
        cadence = self._cadence_from_ankle_position()
        if cadence > 0:
            logger.info("CADENCE_RESULT", method="ankle_position", cadence_spm=f"{cadence:.1f}")
            return cadence

        logger.info("CADENCE_RESULT", method="none", cadence_spm="0.0")
        self._record_warning(
            "Could not reliably detect running cadence -- stride pattern may be "
            "irregular or the clip may be too short. Try a 15+ second clip with "
            "consistent running pace."
        )
        return 0.0

    def _record_warning(self, msg: str) -> None:
        """Append a deduplicated user-facing analyzer warning."""
        if msg not in self._analyzer_warnings:
            self._analyzer_warnings.append(msg)

    def _compute_cadence_from_strides(self, stride_timestamps: list[float]) -> float:
        """Compute cadence (steps/min) from single-leg stride timestamps.

        Each interval between consecutive timestamps = one stride of the near leg.
        One stride = 2 steps (near leg + far leg).
        Cadence = 120000 / avg_interval_ms (= 2 * 60000 / interval).
        """
        if len(stride_timestamps) < 2:
            return 0.0

        intervals: list[float] = []
        for i in range(1, len(stride_timestamps)):
            interval_ms = stride_timestamps[i] - stride_timestamps[i - 1]
            # At 160-200 spm, stride interval is 600-750ms
            if 300 < interval_ms < 1200:
                intervals.append(interval_ms)

        if not intervals:
            return 0.0

        avg_interval_ms = sum(intervals) / len(intervals)
        # Each interval = 1 stride = 2 steps -> cadence = 120000 / interval
        cadence = round(120000.0 / avg_interval_ms, 1)

        # Sanity check
        if cadence < 100 or cadence > 240:
            return 0.0

        return cadence

    def compute_vertical_oscillation(self) -> float:
        """Compute vertical oscillation from NORMALIZED (image-space) hip Y.

        World landmarks are body-centric (hip Y ~ 0 always), so we use
        normalized landmarks where actual vertical movement is visible.
        Scale to meters using estimated body proportions.
        """
        if len(self.norm_hip_y_history) < 5:
            logger.info("VOSC_DEBUG", status="not_enough_values", count=len(self.norm_hip_y_history))
            return 0.0

        y_values = self.norm_hip_y_history

        logger.info(
            "VOSC_DEBUG",
            count=len(y_values),
            y_min=f"{min(y_values):.5f}",
            y_max=f"{max(y_values):.5f}",
            y_range_norm=f"{max(y_values) - min(y_values):.5f}",
            pixel_to_meter=f"{self._pixel_to_meter:.2f}",
        )

        # Window size: ~3-4 frames per step at ~180 spm with ~10fps effective
        window = max(3, len(y_values) // 6)

        oscillations: list[float] = []
        for i in range(0, len(y_values) - window, max(1, window // 2)):
            chunk = y_values[i : i + window]
            osc_norm = max(chunk) - min(chunk)
            # Convert to meters
            osc_m = osc_norm * self._pixel_to_meter
            if osc_m > 0.003:  # filter noise (3mm threshold)
                oscillations.append(osc_m)

        if not oscillations:
            logger.info("VOSC_DEBUG", status="no_valid_windows", window=window)
            return 0.0

        avg_oscillation_m = sum(oscillations) / len(oscillations)

        logger.info(
            "VOSC_DEBUG",
            status="computed",
            num_windows=len(oscillations),
            avg_oscillation_m=f"{avg_oscillation_m:.4f}",
            avg_oscillation_cm=f"{avg_oscillation_m * 100:.1f}",
        )

        # Typical running oscillation: 0.04 - 0.13 meters (4-13 cm)
        return round(avg_oscillation_m, 4)

    def _median_frame_spacing_ms(self) -> float:
        """Median inter-frame interval in ms (temporal resolution).

        Robust to the adaptive downsampling the pipeline applies to
        long clips: GCT granularity is one frame spacing, so this
        drives the low-resolution caveat.
        """
        if len(self.frame_results) < 2:
            return 0.0
        deltas = [
            self.frame_results[i].timestamp_ms - self.frame_results[i - 1].timestamp_ms
            for i in range(1, len(self.frame_results))
            if self.frame_results[i].timestamp_ms > self.frame_results[i - 1].timestamp_ms
        ]
        if not deltas:
            return 0.0
        return float(np.median(deltas))

    def _compute_ground_contact_time(self) -> float:
        """Estimate ground contact time (GCT) in ms from gait phases.

        GCT is the duration of a single stance phase (foot-strike ->
        toe-off). We segment the per-frame gait-phase track into
        contiguous runs of stance phases and take the median run
        duration (median is robust to truncated runs at clip
        boundaries and to the occasional misclassified frame).

        Each stance run's duration is estimated as
        ``n_stance_frames * median_frame_spacing`` rather than
        ``last_ts - first_ts``: the latter systematically
        undercounts by ~one frame because contact begins ~half a
        frame before the first detected stance frame and ends ~half
        a frame after the last.

        Returns ``0.0`` when GCT cannot be measured (too few frames,
        no stance runs, or an implausible result). The caller is
        responsible for omitting a 0.0 from the summary rather than
        surfacing it as a measurement -- mirrors the cadence /
        vertical-oscillation "0.0 == no data" contract.

        NOTE: This is a 2D side-view estimate. Its resolution is one
        frame spacing (~33 ms at 30 fps), which is coarse relative to
        the 180-250 ms reference window, so compute_summary attaches
        a low-confidence caveat.
        """
        if len(self.frame_results) < 4:
            return 0.0

        spacing = self._median_frame_spacing_ms()
        if spacing <= 0:
            return 0.0

        # Collect contiguous stance-run lengths (in frames).
        run_frame_counts: list[int] = []
        current_run = 0
        for fr in self.frame_results:
            phase = fr.extra_metrics.get("gait_phase")
            if phase in GROUND_CONTACT_PHASES:
                current_run += 1
            else:
                if current_run > 0:
                    run_frame_counts.append(current_run)
                current_run = 0
        if current_run > 0:
            run_frame_counts.append(current_run)

        # Drop single-frame runs: a lone stance frame between swing
        # frames is almost always a misclassification, not a real
        # (sub-frame-spacing) contact.
        run_frame_counts = [n for n in run_frame_counts if n >= 2]
        if len(run_frame_counts) < 2:
            self._record_warning(
                "Ground contact time could not be measured reliably -- too "
                "few clean stance phases were detected. A steady side-view "
                "clip of 10+ seconds gives the best result."
            )
            return 0.0

        gct_ms = float(np.median(run_frame_counts)) * spacing

        # Plausibility gate: human running GCT is ~140-350 ms; outside
        # [100, 400] is misclassification or a non-running clip.
        if not (100.0 <= gct_ms <= 400.0):
            return 0.0

        # Low-resolution caveat: when effective fps < 25 (spacing > 40 ms)
        # the one-frame quantisation is a large fraction of GCT.
        if spacing > 40.0:
            self._record_warning(
                f"Ground contact time is a coarse estimate (~{spacing:.0f} ms "
                f"frame spacing). Film at 30+ fps and keep the clip under "
                f"30 seconds for finer resolution."
            )

        return round(gct_ms, 1)

    def _compute_flight_time(self) -> float:
        """Estimate flight time (aerial phase) in ms from gait phases.

        Flight time is the duration of a single swing phase where NEITHER
        foot is on the ground -- the complement of ground contact. In a 2D
        near-side view we can't see the far foot, so we approximate flight
        as the contiguous run of non-stance (swing) frames between two
        stance runs. This over-counts slightly (the far foot may still be
        loading), so the plausibility gate is deliberately generous.

        Same construction as ``_compute_ground_contact_time``: segment the
        per-frame gait-phase track into contiguous SWING runs, estimate each
        as ``n_swing_frames * median_frame_spacing``, take the median.

        Returns ``0.0`` when flight time cannot be measured -- the caller
        omits a 0.0 rather than surfacing it, per the "0.0 == no data"
        contract used by cadence / GCT.

        NOTE: 2D side-view estimate, resolution = one frame spacing. Coarse
        relative to the ~80-150 ms reference window, so compute_summary
        flags it as estimated (same as GCT).
        """
        if len(self.frame_results) < 4:
            return 0.0

        spacing = self._median_frame_spacing_ms()
        if spacing <= 0:
            return 0.0

        # Only count swing runs that sit BETWEEN two stance runs, so a
        # partial swing at either clip boundary (foot already/still in air
        # when recording starts/stops) doesn't skew the estimate. We track
        # whether a stance run has been seen before the current swing run.
        swing_frame_counts: list[int] = []
        current_swing = 0
        seen_stance = False
        pending_swing = 0
        for fr in self.frame_results:
            phase = fr.extra_metrics.get("gait_phase")
            is_stance = phase in GROUND_CONTACT_PHASES
            if is_stance:
                # A swing run that ended by hitting stance is bounded on both
                # sides (we had already seen a prior stance run) -> keep it.
                if pending_swing > 0 and seen_stance:
                    swing_frame_counts.append(pending_swing)
                pending_swing = 0
                seen_stance = True
            else:
                pending_swing += 1
        # A trailing swing run at the clip end is unbounded -> dropped.

        # Drop single-frame runs (misclassification, not a real aerial phase).
        swing_frame_counts = [n for n in swing_frame_counts if n >= 2]
        if len(swing_frame_counts) < 2:
            return 0.0

        flight_ms = float(np.median(swing_frame_counts)) * spacing

        # Plausibility gate: running flight time is ~40-250 ms. Outside this
        # is misclassification or walking (no flight phase at all).
        if not (40.0 <= flight_ms <= 250.0):
            return 0.0

        return round(flight_ms, 1)

    def _contact_frame_indices(self, min_run: int = 3) -> list[int]:
        """Indices of the frames where a real foot-strike begins.

        A foot-strike is the first frame of a *confirmed* stance run --
        one that lasts at least ``min_run`` frames -- immediately following
        a confirmed swing run. Debouncing both sides (not just requiring the
        next frame to be stance) prevents single-frame gait-phase flicker
        from registering as extra contacts; without it an 8 s clip yields
        30+ spurious "contacts" instead of ~1 per stride. Mirrors the
        stride-counter debounce in video_visualizer.

        Uses the same stance-phase set as GCT/flight so all three share one
        notion of "on the ground". Returns the frame indices in order.
        """
        n = len(self.frame_results)
        if n == 0:
            return []
        # Precompute confirmed stance runs (>= min_run contiguous stance frames).
        stance_flags = [
            fr.extra_metrics.get("gait_phase") in GROUND_CONTACT_PHASES
            for fr in self.frame_results
        ]
        idxs: list[int] = []
        i = 0
        while i < n:
            if stance_flags[i]:
                # Measure this stance run.
                j = i
                while j < n and stance_flags[j]:
                    j += 1
                if (j - i) >= min_run:
                    idxs.append(i)  # first frame of a confirmed stance run
                i = j
            else:
                i += 1
        return idxs

    def _compute_overstride_ratio(self) -> tuple[float, int]:
        """Estimate overstride at foot-strike from near-side world landmarks.

        Overstride = the foot landing ahead of the body's centre of mass.
        We proxy COM with the near-side hip and measure the horizontal
        (fore-aft) distance from hip to ankle at each foot-strike, then
        normalise by leg length (hip->ankle) to get a dimensionless ratio
        that is independent of body size and camera distance.

            ratio = |ankle_x - hip_x| / leg_length   (at contact)

        Magnitude, not sign: from a single side-view frame we can't robustly
        infer travel direction, but the *distance* the foot lands ahead of
        the hip is the overstride signal either way. A well-aligned foot-
        strike lands the ankle roughly under the hip (ratio ~0.0-0.15);
        ratio >~0.20 indicates the foot is reaching out ahead (overstride),
        which pairs with a braking force and a heel strike.

        Uses world_landmarks (metres, sagittal X = fore-aft for side-view).
        Requires hip + ankle visibility >= the run threshold; foot-strikes
        with an occluded ankle are skipped. Returns (median_ratio, n_used).
        Returns (0.0, 0) when it cannot be measured -- caller omits it.
        """
        contact_idxs = self._contact_frame_indices()
        if not contact_idxs:
            return 0.0, 0

        near = self.camera_side or "left"
        hip_idx = 23 if near == "left" else 24
        knee_idx = 25 if near == "left" else 26
        ankle_idx = 27 if near == "left" else 28
        vis_thresh = SPORT_LANDMARK_VISIBILITY.get("run", 0.7)

        ratios: list[float] = []
        for i in contact_idxs:
            wl = self.frame_results[i].extra_metrics.get("_world_landmarks")
            if wl is None:
                continue
            try:
                hip, knee, ankle = wl[hip_idx], wl[knee_idx], wl[ankle_idx]
            except (IndexError, TypeError):
                continue
            if min(
                getattr(hip, "visibility", 0.0),
                getattr(ankle, "visibility", 0.0),
            ) < vis_thresh:
                continue
            # Leg length via hip->knee->ankle (robust to a bent knee at contact).
            leg_len = (
                math.dist((hip.x, hip.y), (knee.x, knee.y))
                + math.dist((knee.x, knee.y), (ankle.x, ankle.y))
            )
            if leg_len < 1e-3:
                continue
            horiz = abs(ankle.x - hip.x)
            ratios.append(horiz / leg_len)

        if len(ratios) < 2:
            return 0.0, len(ratios)
        return round(float(np.median(ratios)), 3), len(ratios)

    def _compute_foot_strike(self) -> tuple[str | None, float, int]:
        """Classify foot-strike pattern (heel / mid / fore) at contact.

        At foot-strike, the vertical offset between heel and toe (foot
        index) reveals which part of the foot lands first:
          - heel below toe  -> heel strike  (heel.y > toe.y in image coords)
          - roughly level    -> midfoot strike
          - toe below heel   -> forefoot strike

        We measure the foot's angle to horizontal at each foot-strike and
        take the median (robust to a single mistracked frame). The angle is
        signed: positive = toe-up (heel strike), negative = toe-down
        (forefoot). |angle| < ~8 deg = midfoot.

        Uses NORMALIZED landmarks (image plane) -- foot orientation is a 2D
        image-plane quantity and the normalized foot points are what the
        overlay/other image-space metrics use. Foot landmarks (heel, toe)
        are the least reliable side-on, so this requires both visible above
        the run threshold and returns (None, nan, n) when too few clean
        contacts exist. Returns (pattern, median_angle_deg, n_used).
        """
        contact_idxs = self._contact_frame_indices()
        if not contact_idxs:
            return None, float("nan"), 0

        near = self.camera_side or "left"
        heel_idx = 29 if near == "left" else 30
        toe_idx = 31 if near == "left" else 32
        vis_thresh = SPORT_LANDMARK_VISIBILITY.get("run", 0.7)

        angles: list[float] = []
        for i in contact_idxs:
            nl = self.frame_results[i].extra_metrics.get("_norm_landmarks")
            if nl is None:
                continue
            try:
                heel, toe = nl[heel_idx], nl[toe_idx]
            except (IndexError, TypeError):
                continue
            if min(
                getattr(heel, "visibility", 0.0),
                getattr(toe, "visibility", 0.0),
            ) < vis_thresh:
                continue
            dx = toe.x - heel.x
            dy = toe.y - heel.y  # image Y increases downward
            if abs(dx) < 1e-6:
                continue
            # Signed angle to horizontal: +ve = toe ABOVE heel (heel strike),
            # -ve = toe BELOW heel (forefoot). -dy so up is positive.
            angle = math.degrees(math.atan2(-dy, abs(dx)))
            angles.append(angle)

        if len(angles) < 2:
            return None, float("nan"), len(angles)

        med = float(np.median(angles))
        if med > 8.0:
            pattern = "heel"
        elif med < -8.0:
            pattern = "forefoot"
        else:
            pattern = "midfoot"
        return pattern, round(med, 1), len(angles)

    def analyze_frame(
        self, world_landmarks: Any, normalized_landmarks: Any, timestamp_ms: float,
    ) -> FrameAnalysis:
        """Analyze a single running frame -- NEAR SIDE ONLY.

        Computes only camera-facing side angles with unprefixed keys:
        'knee', 'hip', 'ankle', 'elbow', 'trunk'
        """
        wl = world_landmarks
        nl = normalized_landmarks

        # Estimate body scale on first frame
        if not self._body_scale_estimated:
            self._estimate_body_scale(nl)

        # Determine near side (set by pipeline before frame processing)
        near = self.camera_side or "left"

        # Near-side joint angles (unprefixed keys)
        angle_defs = RUNNING_ANGLES[near]
        angles: dict[str, float] = {}
        visibility: dict[str, float] = {}

        for joint_name, (idx_a, idx_b, idx_c) in angle_defs.items():
            angle_val, vis = calculate_angle_2d(wl, idx_a, idx_b, idx_c)
            angles[joint_name] = angle_val
            visibility[joint_name] = vis

        # Trunk lean: near-side shoulder + hip (world landmarks)
        sh_idx, hp_idx = TRUNK_LANDMARKS[near]
        trunk_val = calculate_segment_to_vertical(wl, sh_idx, hp_idx)
        angles["trunk"] = trunk_val
        self.trunk_lean_values.append(trunk_val)

        # Collect NORMALIZED hip Y for vertical oscillation (image-space)
        norm_hip_y = (nl[23].y + nl[24].y) / 2
        self.norm_hip_y_history.append(norm_hip_y)
        self.norm_hip_y_timestamps.append(timestamp_ms)

        # Gait phase detection (near-side only)
        near_ankle_idx = 27 if near == "left" else 28
        near_foot_idx = 31 if near == "left" else 32
        near_ankle_y = nl[near_ankle_idx].y
        hip_center_y = (nl[23].y + nl[24].y) / 2

        self.near_ankle_y_history.append(near_ankle_y)
        self.hip_center_y_history.append(hip_center_y)

        near_vel = 0.0
        if len(self.near_ankle_y_history) > 1:
            near_vel = self.near_ankle_y_history[-1] - self.near_ankle_y_history[-2]

        near_phase = self.detect_gait_phase(
            angles["knee"], near_ankle_y, hip_center_y, nl[near_foot_idx].y, near_vel,
        )
        self.prev_near_phase = near_phase

        frame = FrameAnalysis(
            timestamp_ms=timestamp_ms,
            angles=angles,
            visibility=visibility,
            extra_metrics={
                "gait_phase": near_phase.value,
                # Store normalized ankle Y for fallback cadence detection
                "_norm_left_ankle_y": nl[27].y,
                "_norm_right_ankle_y": nl[28].y,
                # References to the frame's landmark arrays (not copies -- these
                # objects already live in the pipeline's raw_frame_data). Used by
                # foot-strike (normalized, image-plane) and overstride (world,
                # metric) at foot-strike frames. Kept per-frame so those metrics
                # can sample only the contact frames after the fact.
                "_world_landmarks": wl,
                "_norm_landmarks": nl,
            },
        )
        return frame

    def compute_summary(self) -> dict[str, Any]:
        """Aggregate running metrics from near-side only angles.

        Plausibility-gated: cadence outside [80, 220] spm and
        vertical oscillation outside [1, 25] cm are treated as
        "no measurement" -- they are omitted from summary entirely
        rather than emitted as 0.0. Without this, a phantom 0.0
        cadence from a short clip flowed through to the AI Coach
        as the #1 priority "critical low cadence" finding.
        Trunk-lean is only emitted when at least one valid trunk
        sample existed.
        """
        if not self.frame_results:
            return {}

        # Cadence: primary (knee oscillation) + fallback (ankle position)
        cadence = self._compute_cadence()

        # Vertical oscillation: normalized landmarks scaled to meters
        vert_osc = self.compute_vertical_oscillation()

        # Ground contact time (2D side-view estimate, frame-resolution-limited)
        gct_ms = self._compute_ground_contact_time()

        # Flight time / aerial phase (same construction as GCT, swing runs)
        flight_ms = self._compute_flight_time()

        # Overstride + foot-strike, sampled at foot-strike frames.
        overstride_ratio, overstride_n = self._compute_overstride_ratio()
        foot_strike, foot_strike_angle, foot_strike_n = self._compute_foot_strike()

        # Prefer Butterworth-filtered data (angle_history mutated in-place by filter)
        filtered_trunk = self.angle_history.get("trunk", self.trunk_lean_values)
        trunk_arr = np.array(filtered_trunk) if filtered_trunk else np.array([])
        valid_trunk = trunk_arr[~np.isnan(trunk_arr)] if len(trunk_arr) > 0 else trunk_arr
        # None when no valid samples -- absent metric > phantom 0.
        trunk_lean_avg: float | None = (
            float(np.mean(valid_trunk)) if len(valid_trunk) > 0 else None
        )

        # Knee stats from angle_history (unprefixed)
        knee_vals = self.angle_history.get("knee", [])
        knee_arr = np.array(knee_vals) if knee_vals else np.array([])
        valid_knee = knee_arr[~np.isnan(knee_arr)] if len(knee_arr) > 0 else knee_arr

        # Elbow stats
        elbow_vals = self.angle_history.get("elbow", [])
        elbow_arr = np.array(elbow_vals) if elbow_vals else np.array([])
        valid_elbow = elbow_arr[~np.isnan(elbow_arr)] if len(elbow_arr) > 0 else elbow_arr

        near = self.get_near_side_prefix()
        logger.info(
            "RUNNING_SUMMARY",
            cadence_spm=f"{cadence:.1f}",
            vertical_oscillation_m=f"{vert_osc:.4f}",
            trunk_lean_avg=(
                f"{trunk_lean_avg:.1f}" if trunk_lean_avg is not None else "n/a"
            ),
            frames_analyzed=len(self.frame_results),
            camera_side=self.camera_side,
            near_side=near,
        )

        summary: dict[str, Any] = {
            "knee_mean": round(float(np.mean(valid_knee)), 1) if len(valid_knee) > 0 else None,
            "knee_min": round(float(np.min(valid_knee)), 1) if len(valid_knee) > 0 else None,
            "knee_max": round(float(np.max(valid_knee)), 1) if len(valid_knee) > 0 else None,
            "elbow_mean": round(float(np.mean(valid_elbow)), 1) if len(valid_elbow) > 0 else None,
            "frames_analyzed": len(self.frame_results),
            "camera_side": self.camera_side,
            "near_side": near,
            "camera_side_label": near.capitalize() if near else "Left",
        }

        # Plausibility gates. Cadence and vertical oscillation
        # have a "0.0 = no measurement" failure mode; trunk_lean
        # has a "no valid samples = None" path. In each case,
        # absence is more honest than a sentinel that downstream
        # graders read as a measurement.
        if 80.0 <= cadence <= 220.0:
            summary["cadence_spm"] = cadence

        # vert_osc is stored in meters; range 0.01-0.25 m = 1-25 cm.
        if 0.01 <= vert_osc <= 0.25:
            summary["vertical_oscillation_m"] = vert_osc

        # GCT: _compute_ground_contact_time already plausibility-gated
        # to [100, 400] ms and returns 0.0 for "no measurement".
        # Flagged low-confidence: it's a coarse 2D estimate, not a
        # force-plate / IMU reading -- the frontend should label it
        # as estimated.
        if gct_ms > 0:
            summary["ground_contact_ms"] = gct_ms
            summary["ground_contact_ms_estimated"] = True

        # Flight time: gated to [40, 250] ms in _compute_flight_time; 0.0 =
        # no measurement. Same "estimated" caveat as GCT (coarse 2D estimate).
        if flight_ms > 0:
            summary["flight_time_ms"] = flight_ms
            summary["flight_time_ms_estimated"] = True

        # Overstride: dimensionless hip->ankle-ahead ratio at foot-strike.
        # Requires >= 2 clean contacts (returns 0.0/0 otherwise). 2D estimate.
        if overstride_ratio > 0 and overstride_n >= 2:
            summary["overstride_ratio"] = overstride_ratio
            summary["overstride_estimated"] = True
            summary["overstride_contacts"] = overstride_n

        # Foot-strike pattern (heel/midfoot/forefoot) at contact. Only emit
        # when a pattern was classified from >= 2 clean contacts.
        if foot_strike is not None and foot_strike_n >= 2:
            summary["foot_strike"] = foot_strike
            summary["foot_strike_angle_deg"] = foot_strike_angle
            summary["foot_strike_estimated"] = True
            summary["foot_strike_contacts"] = foot_strike_n

        if trunk_lean_avg is not None:
            summary["trunk_lean_avg"] = round(trunk_lean_avg, 1)

        if self._analyzer_warnings:
            existing = summary.get("analysis_warnings", [])
            summary["analysis_warnings"] = existing + list(self._analyzer_warnings)

        return summary

    def detect_issues(self) -> list[dict[str, Any]]:
        """Detect running technique issues from near-side data only."""
        issues: list[dict[str, Any]] = []
        if not self.frame_results:
            return issues

        # Excessive trunk lean (prefer Butterworth-filtered data)
        filtered_trunk = self.angle_history.get("trunk", self.trunk_lean_values)
        if filtered_trunk:
            trunk_arr = np.array(filtered_trunk)
            valid_trunk = trunk_arr[~np.isnan(trunk_arr)]
            avg_trunk = float(np.mean(valid_trunk)) if len(valid_trunk) > 0 else 0.0
            if avg_trunk > 12:
                issues.append({
                    "type": "excessive_forward_lean",
                    "severity": "warning",
                    "value": f"{avg_trunk:.1f} deg",
                    "recommendation": f"Trunk lean is {avg_trunk:.0f} deg. Optimal range: {RUNNING_REFERENCE['trunk_lean'][0]}-{RUNNING_REFERENCE['trunk_lean'][1]} deg.",
                })

        # Overstriding: foot landing well ahead of the hip at contact. A
        # ratio above ~0.20 (foot-ahead distance > 20% of leg length) is the
        # threshold; a co-occurring heel strike reinforces it in the message.
        overstride_ratio, overstride_n = self._compute_overstride_ratio()
        if overstride_ratio > 0.20 and overstride_n >= 2:
            foot_strike, _angle, _n = self._compute_foot_strike()
            heel_note = (
                " with a heel strike" if foot_strike == "heel" else ""
            )
            issues.append({
                "type": "overstriding",
                "severity": "warning",
                "value": f"{overstride_ratio:.2f} x leg length ahead of hip",
                "recommendation": (
                    f"The foot is landing ~{overstride_ratio:.0%} of a leg "
                    f"length ahead of the hip at contact{heel_note}. This "
                    f"brakes each step and raises impact load. Increasing "
                    f"cadence toward {RUNNING_REFERENCE['cadence_spm'][0]}-"
                    f"{RUNNING_REFERENCE['cadence_spm'][1]} spm and landing "
                    f"with the foot closer under the hip reduces overstride."
                ),
            })

        # Low cadence
        cadence = self._compute_cadence()
        if cadence > 0 and cadence < 165:
            issues.append({
                "type": "low_cadence",
                "severity": "warning",
                "value": f"{cadence:.0f} spm",
                "recommendation": f"Cadence is {cadence:.0f} spm. Target: {RUNNING_REFERENCE['cadence_spm'][0]}-{RUNNING_REFERENCE['cadence_spm'][1]} spm.",
            })

        # Prolonged ground contact -- usually a downstream symptom of
        # low cadence / overstriding, so the recommendation points back
        # to cadence rather than prescribing a separate drill.
        gct_ms = self._compute_ground_contact_time()
        gct_min, gct_max = RUNNING_REFERENCE["ground_contact_ms"]
        if gct_ms > 0 and gct_ms > gct_max + 40:  # +~1 frame tolerance
            issues.append({
                "type": "prolonged_ground_contact",
                "severity": "info",
                "value": f"{gct_ms:.0f} ms (estimated)",
                "recommendation": (
                    f"Ground contact time is ~{gct_ms:.0f} ms (target "
                    f"{gct_min}-{gct_max} ms). This is an estimate from 2D "
                    f"video. Increasing cadence toward "
                    f"{RUNNING_REFERENCE['cadence_spm'][0]}-"
                    f"{RUNNING_REFERENCE['cadence_spm'][1]} spm typically "
                    f"shortens contact time."
                ),
            })

        # Insufficient knee drive (swing phase) -- unprefixed key
        knee_vals = self.angle_history.get("knee", [])
        if knee_vals:
            knee_arr = np.array(knee_vals)
            valid_knees = knee_arr[~np.isnan(knee_arr)]
            if len(valid_knees) > 0:
                min_knee = float(np.min(valid_knees))
                if min_knee > 110:
                    issues.append({
                        "type": "insufficient_knee_drive",
                        "severity": "info",
                        "value": f"Min knee = {min_knee:.0f} deg",
                        "recommendation": "Increase knee drive in swing phase. Target minimum knee angle: 80-100 deg.",
                    })

        return issues
