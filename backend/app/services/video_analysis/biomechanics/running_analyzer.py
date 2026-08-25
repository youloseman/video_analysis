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
    calculate_forward_sign,
    calculate_shank_foot_angle_2d,
    calculate_signed_segment_to_vertical,
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
        "elbow": (11, 13, 15),   # LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST
    },
    "right": {
        "knee":  (24, 26, 28),   # RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE
        "hip":   (12, 24, 26),   # RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE
        "elbow": (12, 14, 16),   # RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST
    },
}

# Ankle is NOT a vertex triplet: it is the angle between the shank axis
# (ankle -> knee) and the foot axis (heel -> toe). Neutral = 90 deg,
# midstance dorsiflexion ~70, toe-off plantarflexion ~115. The axis form
# is robust to BlazePose drifting the ankle landmark up the shin on bulky
# running shoes (the old knee-ankle-HEEL vertex read ~25-40 deg high and
# moved the wrong way with dorsiflexion). Tuple: (knee, ankle, heel, toe).
RUNNING_ANKLE_LANDMARKS: dict[str, tuple[int, int, int, int]] = {
    "left":  (25, 27, 29, 31),
    "right": (26, 28, 30, 32),
}

TRUNK_LANDMARKS: dict[str, tuple[int, int]] = {
    "left":  (11, 23),   # LEFT_SHOULDER, LEFT_HIP
    "right": (12, 24),   # RIGHT_SHOULDER, RIGHT_HIP
}

# Shoulder-to-hip length as a fraction of standing height. Winter,
# *Biomechanics and Motor Control of Human Movement* (4th ed., 2009), Table
# 4.1: shoulder height 0.818 H, hip (greater trochanter) height 0.530 H. The
# MediaPipe shoulder and hip landmarks sit close enough to those two levels for
# the difference to be the segment we measure between them.
TORSO_FRACTION_OF_HEIGHT = 0.288

# The stand-in when the athlete has not told us their height. Note what it
# implies: 0.45 / 0.288 means a person about 156 cm tall. It is kept at the
# historical value rather than raised to a fitter average so that adding a
# height CHANGES a reading and omitting one does not.
DEFAULT_TORSO_M = 0.45

# Heights outside this are a typo or a unit mix-up (feet, inches, metres), not
# an athlete. Treated as "not told" rather than trusted.
PLAUSIBLE_HEIGHT_CM = (120.0, 230.0)

# --- whole-clip gait phase (see RunningAnalyzer.recompute_gait_phases) ------
# Fewer frames than this and there is no cycle to recognise stance against.
GAIT_MIN_FRAMES = 30
# Smoothing before differencing, in seconds rather than frames so it means the
# same thing at 30 fps, 60 fps and on a slow-motion clip.
GAIT_SMOOTH_S = 0.05
# How far up from its lowest point a foot may be and still count as planted,
# as a fraction of the ankle's own vertical range.
#
# Calibrated on `upload/IMG_3979.MOV`, a 10 s treadmill clip whose cadence is
# independently known (164.6 spm from the swap-immune spectral estimator, so
# a 362 ms step). Cadence comes out at 166 spm for every value in 0.55-0.80 --
# the timing does not depend on this at all -- and the band only sets how long
# contact lasts:
#
#     0.55 -> contact 155 ms, flight 207     0.70 -> contact 207 ms, flight 155
#     0.60 -> contact 190 ms, flight 173     0.75 -> contact 224 ms, flight 138
#     0.65 -> contact 207 ms, flight 155     0.80 -> cadence breaks to 174
#
# 0.65 sits mid-plateau and lands contact at 207 ms against a published
# 180-270 for endurance running (Folland 2017), with flight at 155.
GAIT_LOW_BAND = 0.65

# How near its deepest the near ankle must be for a footfall to count as that
# leg's. Generous on purpose: the cost of rejecting a real near-leg contact is
# one fewer sample, and the cost of accepting the far leg's is a fabricated
# overstride reading.
NEAR_CONTACT_BAND = 0.45


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average with edge padding, so no samples are lost."""
    if window <= 1 or values.size < 3:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window, window), mode="edge")
    return np.convolve(padded, kernel, mode="same")[window:-window]


def _contiguous_runs(flags: np.ndarray, min_run: int) -> list[tuple[int, int]]:
    """Inclusive ``(first, last)`` of every run of True at least ``min_run`` long."""
    runs: list[tuple[int, int]] = []
    n = len(flags)
    i = 0
    while i < n:
        if flags[i]:
            j = i
            while j < n and flags[j]:
                j += 1
            if j - i >= min_run:
                runs.append((i, j - 1))
            i = j
        else:
            i += 1
    return runs


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

    def __init__(self, fps: float = 30.0, height_cm: float | None = None):
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
        self._torso_norm_samples: list[float] = []
        # The athlete's standing height, when they have told us. This is the
        # ONLY real-world length a side view ever gets; everything else it
        # measures is a fraction of the picture. See _lock_body_scale.
        self.height_cm = (
            float(height_cm)
            if height_cm and PLAUSIBLE_HEIGHT_CM[0] <= height_cm <= PLAUSIBLE_HEIGHT_CM[1]
            else None
        )
        self._body_scale_source = "unknown"
        # Set once _compute_cadence recognises a slow-motion clip; every
        # time-derived metric divides by it (see _median_frame_spacing_ms).
        self._slowmo_factor: int | None = None
        self._analyzer_warnings: list[str] = []

    def _estimate_body_scale(self, nl: Any) -> None:
        """Collect a torso-length sample; the scale locks to the median.

        Uses shoulder-to-hip vertical distance (average adult torso ~0.45m).
        The old first-frame-only estimate let one bad opening detection --
        common on backlit clips, where the first frames are the least
        trustworthy -- poison every meter-denominated metric for the whole
        clip: a half-size torso read doubles vertical oscillation. Sampling
        every valid frame and taking the median (locked lazily by the first
        consumer) is insensitive to any bad stretch.
        """
        sh_y = (nl[11].y + nl[12].y) / 2  # shoulder center Y
        hp_y = (nl[23].y + nl[24].y) / 2  # hip center Y
        norm_torso = abs(hp_y - sh_y)

        if not math.isnan(norm_torso) and norm_torso > 0.03:
            self._torso_norm_samples.append(norm_torso)

    def _lock_body_scale(self) -> None:
        """Fix ``_pixel_to_meter`` to the median of collected torso samples.

        Idempotent; also called lazily by the first consumer in case the
        clip produced fewer than ``BODY_SCALE_MAX_SAMPLES`` valid frames.

        This is the single place a side view becomes centimetres. Everything
        the camera reports is a fraction of the picture; one known real length
        turns all of it into metres, and this picks which length to believe.

        With a stated height, the torso is modelled from it: Winter's
        anthropometric table puts the shoulder at 0.818 of standing height and
        the hip at 0.530, so the segment between them is ``0.288 H``. Without
        one, the old population constant stands in -- and it is worth knowing
        what that constant assumes: 0.45 m of torso is a person about 156 cm
        tall. For a 180 cm runner the real segment is nearer 0.52 m, so the
        assumption under-reports every centimetre reading by roughly 13%, which
        is most of the width of the vertical-oscillation band it is graded in.
        """
        if self._body_scale_estimated:
            return
        if self._torso_norm_samples:
            med = float(np.median(self._torso_norm_samples))
            if self.height_cm:
                torso_m = TORSO_FRACTION_OF_HEIGHT * (self.height_cm / 100.0)
                self._body_scale_source = "athlete_height"
            else:
                torso_m = DEFAULT_TORSO_M
                self._body_scale_source = "population_average"
            self._pixel_to_meter = torso_m / med
        else:
            self._pixel_to_meter = 2.0  # fallback
            self._body_scale_source = "no_samples"
        self._body_scale_estimated = True
        logger.info(
            "BODY_SCALE_DEBUG",
            samples=len(self._torso_norm_samples),
            pixel_to_meter=f"{self._pixel_to_meter:.2f}",
            source=self._body_scale_source,
            height_cm=self.height_cm,
        )

    def recompute_gait_phases(self) -> dict[str, Any] | None:
        """Redecide stance and swing for every frame, from the whole clip.

        Replaces what :meth:`detect_gait_phase` decided frame by frame while
        the clip was still streaming past. That version was wrong in three
        ways, and the third one is why this is a separate pass rather than a
        patch:

        1. ``ankle_y_velocity < 0.001`` accepts every NEGATIVE velocity, so an
           ankle travelling upward -- the whole heel-recovery phase -- counted
           as ground contact. Measured on a clean clip: 95% of the cycle
           called stance, against 30-40% for real running.
        2. The threshold is an absolute per-frame distance, so it means
           different things at 30 and 60 fps, and nothing at all on a
           slow-motion clip where every per-frame velocity is divided by eight.
        3. Stance is only recognisable against the cycle it sits in, and a
           streaming detector cannot see one. The rolling history it had was
           two seconds -- shorter than a single stride on an 8x slow clip.

        What replaces it is two conditions on the LOWER of the two ankles,
        which is a deliberate choice: it never asks which leg it is looking at.
        Left/right identity is the least reliable thing MediaPipe produces on a
        side view -- 43% of frames needed correcting on the treadmill fixture --
        and every phase decision made from a labelled leg inherits that.

        * **Travelling backward relative to the hips.** A planted foot does not
          move; the hips pass over it, so hip-relative it slides straight back
          at the speed of travel. On a treadmill, at belt speed -- the same
          thing. Swing returns it forward over a longer slice of the cycle.
        * **Low in its own vertical range.** The backward test alone cannot see
          flight, where a foot may still be drifting backward with nothing
          under it. The planted foot is the one at the bottom of its travel.

        Both signals are hip-relative, so a drifting camera and a bobbing body
        cancel out, and both are read off the whole clip, so frame rate and
        slow motion stop mattering.

        Returns a small diagnostic dict, or None when the clip cannot support
        the decision -- in which case the streamed phases are left alone.
        """
        n = len(self.frame_results)
        if n < GAIT_MIN_FRAMES:
            return None

        fps = self.get_effective_fps()
        if not fps or fps <= 0:
            return None

        lower_x = np.full(n, np.nan)
        lower_y = np.full(n, np.nan)
        hip_x = np.full(n, np.nan)
        hip_y = np.full(n, np.nan)
        forward_votes: list[float] = []
        for i, fr in enumerate(self.frame_results):
            nl = fr.extra_metrics.get("_norm_landmarks")
            if nl is None:
                continue
            try:
                la, ra = nl[27], nl[28]
                hips_x = (nl[23].x + nl[24].x) / 2.0
                hips_y = (nl[23].y + nl[24].y) / 2.0
            except (IndexError, TypeError, AttributeError):
                continue
            if any(v is None or math.isnan(v) for v in (la.y, ra.y, hips_x, hips_y)):
                continue
            # Image y grows downward, so the LOWER foot is the larger y.
            near_ground = la if la.y >= ra.y else ra
            if near_ground.x is None or math.isnan(near_ground.x):
                continue
            lower_x[i], lower_y[i] = near_ground.x, near_ground.y
            hip_x[i], hip_y[i] = hips_x, hips_y
            sign = calculate_forward_sign(nl)
            if sign:
                forward_votes.append(sign)

        valid = np.isfinite(lower_x) & np.isfinite(hip_x)
        if valid.sum() < GAIT_MIN_FRAMES or not forward_votes:
            return None
        forward = 1.0 if sum(forward_votes) > 0 else -1.0

        idx = np.arange(n)
        fore_aft = np.interp(idx, idx[valid], (lower_x - hip_x)[valid]) * forward
        height = np.interp(idx, idx[valid], (lower_y - hip_y)[valid])

        window = max(3, int(round(GAIT_SMOOTH_S * fps)) | 1)
        fore_aft = _moving_average(fore_aft, window)
        height = _moving_average(height, window)

        going_back = np.gradient(fore_aft) < 0
        low, high = np.percentile(height, 5), np.percentile(height, 95)
        span = high - low
        if span <= 1e-6:
            return None
        planted = height >= high - span * GAIT_LOW_BAND

        any_foot_down = going_back & planted

        # That is "SOME foot is on the ground", and it alternates feet. Every
        # consumer of gait_phase -- ground contact, the frames overstride and
        # foot-strike sample, the kinogram's positions -- means the NEAR foot,
        # so the far foot's contacts have to come out. Sampling the near-side
        # ankle at the moment the FAR foot lands measures a leg in mid-swing:
        # overstride read 0.63 leg-lengths that way, against 0.23 before.
        #
        # Contacts alternate by construction, so which parity is the near foot
        # is ONE decision for the whole clip rather than a per-frame label
        # lookup. That matters: per-frame left/right identity is the least
        # reliable thing here (43% of frames needed correcting on the fixture),
        # while a single vote pooled over every frame of every contact is not
        # meaningfully wrong.
        # These phases describe "A foot is on the ground", not "the near foot
        # is". Ground contact and flight time want exactly that -- they are
        # properties of a footfall, whichever leg made it, and reading them
        # from one labelled leg is what made them hostage to left/right
        # identity. The metrics that genuinely need the near leg (overstride,
        # foot strike, the kinogram's positions) filter these contacts
        # themselves -- see :meth:`_contact_frame_indices`.
        runs = _contiguous_runs(any_foot_down, max(2, int(round(0.04 * fps))))
        for i, fr in enumerate(self.frame_results):
            knee = fr.angles.get("knee", 0.0) or 0.0
            fr.extra_metrics["gait_phase"] = (
                self._stance_phase(knee) if any_foot_down[i]
                else self._swing_phase(knee)
            ).value
            fr.extra_metrics["_near_foot_depth"] = float(
                self._near_ankle_depth(fr)
            )

        return {
            "method": "lower_ankle_fore_aft",
            "stance_fraction": round(float(any_foot_down.mean()), 3),
            "contacts_detected": len(runs),
            "forward_sign": forward,
            "smooth_window_frames": window,
        }

    def _near_ankle_depth(self, frame: Any) -> float:
        """How far below the hips the near-side ankle hangs, or NaN.

        A planted foot sits at the bottom of its travel; a foot in mid-swing
        does not. That difference is what tells one leg's footfall from the
        other's without trusting a left/right label -- the labels being the
        least reliable thing on a side view.
        """
        nl = frame.extra_metrics.get("_norm_landmarks")
        if nl is None:
            return float("nan")
        idx = 27 if (self.camera_side or "left") == "left" else 28
        try:
            depth = nl[idx].y - (nl[23].y + nl[24].y) / 2.0
        except (IndexError, TypeError, AttributeError):
            return float("nan")
        return float(depth) if depth is not None else float("nan")

    @staticmethod
    def _stance_phase(knee_angle: float) -> GaitPhase:
        if knee_angle > 160:
            return GaitPhase.INITIAL_CONTACT
        if knee_angle > 140:
            return GaitPhase.MIDSTANCE
        if knee_angle > 120:
            return GaitPhase.TERMINAL_STANCE
        return GaitPhase.PRE_SWING

    @staticmethod
    def _swing_phase(knee_angle: float) -> GaitPhase:
        if knee_angle < 100:
            return GaitPhase.MID_SWING
        if knee_angle < 130:
            return GaitPhase.INITIAL_SWING
        return GaitPhase.TERMINAL_SWING

    def detect_gait_phase(
        self, knee_angle: float, ankle_y: float, hip_y: float,
        foot_y: float, ankle_y_velocity: float,
    ) -> GaitPhase:
        """Provisional per-frame phase, overwritten by recompute_gait_phases.

        Kept because the streaming loop needs *something* in the field before
        the clip has been seen whole, and because a clip too short for the
        whole-clip pass falls back to it.

        The comparison is on the ABSOLUTE velocity. It used to be signed, which
        accepted every ankle travelling upward and so counted the whole
        heel-recovery phase as ground contact -- the defect that put 95% of a
        cycle in stance. Reading a single frame it still cannot do better than
        "roughly stationary and below the hips", and the threshold is still an
        absolute per-frame distance that means different things at different
        frame rates; that is why the whole-clip pass exists and why this only
        survives as its fallback.
        """
        is_ground_contact = abs(ankle_y_velocity) < 0.001 and foot_y > hip_y

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

    def _ankle_spectrum(self) -> "tuple[Any, Any, float] | None":
        """Spectrum of the mean-of-both-ankles Y trace: ``(spec, freqs, dt)``.

        The averaged left+right ankle signal bounces once per STEP (the
        anti-phase per-leg components cancel), so the dominant frequency in
        the 1.4-4.0 Hz band is steps-per-second. Averaging both ankles makes
        the signal immune to left/right identity swaps -- exactly the
        corruption that fragments the knee-valley method on silhouette
        clips. The signal is resampled onto a regular grid (NaN-dropped
        frames leave gaps), Hann-windowed, zero-padded 4x, and the peak bin
        is refined parabolically: ~1-2 spm resolution on an 8 s clip.

        Returns spm, or 0.0 when the clip is too short, the signal too
        flat, or no band peak clearly dominates.
        """
        ys: list[float] = []
        ts: list[float] = []
        for fr in self.frame_results:
            ly = fr.extra_metrics.get("_norm_left_ankle_y", 0)
            ry = fr.extra_metrics.get("_norm_right_ankle_y", 0)
            if ly > 0 and ry > 0:
                ys.append((ly + ry) / 2.0)
                ts.append(fr.timestamp_ms)
        if len(ys) < 16:
            return None
        t = (np.asarray(ts, dtype=float) - ts[0]) / 1000.0
        duration_s = float(t[-1])
        if duration_s < 2.0:
            return None

        # Regular resampling grid at the median frame spacing.
        dt = float(np.median(np.diff(t)))
        if dt <= 0:
            return None
        grid = np.arange(0.0, duration_s, dt)
        if len(grid) < 16:
            return None
        sig = np.interp(grid, t, np.asarray(ys, dtype=float))

        # Detrend (slow camera pan / drift) + de-mean.
        x = np.arange(len(sig), dtype=float)
        sig = sig - np.polyval(np.polyfit(x, sig, 1), x)
        if float(np.std(sig)) < 1e-5:
            return None

        n = len(sig)
        padded = 4 * n
        spec = np.abs(np.fft.rfft(sig * np.hanning(n), n=padded))
        freqs = np.fft.rfftfreq(padded, d=dt)
        return spec, freqs, dt

    @staticmethod
    def _peak_spm_in_band(
        spec: Any, freqs: Any, lo_hz: float, hi_hz: float,
    ) -> float:
        """Dominant frequency in a band, as steps per minute (0.0 if none)."""
        band = np.where((freqs >= lo_hz) & (freqs <= hi_hz))[0]
        if len(band) < 3:
            return 0.0
        peak = band[int(np.argmax(spec[band]))]
        # A real step rhythm towers over the band floor; a flat or noisy
        # spectrum means there is no rhythm to read -- refuse to answer.
        if spec[peak] < 3.0 * float(np.median(spec[band])):
            return 0.0
        # Asymmetric gaits leak energy at the per-leg stride frequency (half
        # the step rate). If the double-frequency line carries comparable
        # energy, the picked peak IS that subharmonic -- promote to 2x so a
        # limp cannot read as half cadence. Symmetric runs measure ~10% here.
        twice = 2 * peak
        if (
            twice < len(spec)
            and freqs[twice] <= hi_hz
            and spec[twice] >= 0.5 * spec[peak]
        ):
            peak = twice

        # Parabolic peak refinement between bins.
        delta = 0.0
        if 0 < peak < len(spec) - 1:
            a, b, c = float(spec[peak - 1]), float(spec[peak]), float(spec[peak + 1])
            denom = a - 2.0 * b + c
            if abs(denom) > 1e-12:
                delta = max(-0.5, min(0.5, 0.5 * (a - c) / denom))
        bin_hz = float(freqs[1] - freqs[0])
        return (float(freqs[peak]) + delta * bin_hz) * 60.0

    def _cadence_spectral(self) -> float:
        """Spectral cadence at normal playback speed (0.0 if unreadable)."""
        spectrum = self._ankle_spectrum()
        if spectrum is None:
            return 0.0
        spec, freqs, dt = spectrum
        # Cap the band at 85% of Nyquist so a heavily downsampled long clip
        # cannot alias a sprint cadence into it.
        band_top = min(4.0, 0.85 * (0.5 / dt))
        if band_top <= 1.5:
            return 0.0
        spm = self._peak_spm_in_band(spec, freqs, 1.4, band_top)
        if spm <= 0:
            return 0.0
        logger.info("CADENCE_SPECTRAL_DEBUG", cadence_spm=f"{spm:.1f}")
        # Only accept a rhythm fast enough to actually be running. Below
        # REAL_RUN_SPM_MIN a reading is equally consistent with a shuffle and
        # with 2x slow motion of a normal stride, and nothing in the ankle
        # trace separates the two -- so it is refused rather than doubled.
        if self.REAL_RUN_SPM_MIN <= spm <= 240.0:
            return round(spm, 1)
        return 0.0

    # Phone slow motion is 120 or 240 fps played back at 30: 4x and 8x. 2x is
    # deliberately absent -- it maps a normal stride onto ~85 spm, which is
    # also a plausible (if slow) real rhythm.
    SLOW_MOTION_FACTORS = (4, 8)
    # A clip slower than this is not someone running at normal speed.
    REAL_RUN_SPM_MIN = 140.0
    # Where an inferred cadence has to land to be believable.
    INFERRED_SPM_RANGE = (150.0, 205.0)

    def _cadence_from_slow_motion(self) -> tuple[float, int | None]:
        """Recover cadence from a clip shot in slow motion.

        Running cadence is tightly bounded, so a rhythm far below it can be
        read backwards: if the ankles bounce at 22 spm, only one standard
        slow-motion factor puts that back inside a human range -- 8x gives
        176 spm, while 4x would mean 88, which nobody runs at. When exactly
        one factor fits, the answer is determined rather than guessed; when
        none or several fit, this returns nothing.

        Returns ``(cadence_spm, factor)`` or ``(0.0, None)``.
        """
        spectrum = self._ankle_spectrum()
        if spectrum is None:
            return 0.0, None
        spec, freqs, _dt = spectrum
        slow_spm = self._peak_spm_in_band(spec, freqs, 0.15, 1.4)
        if slow_spm <= 0:
            return 0.0, None
        lo, hi = self.INFERRED_SPM_RANGE
        fits = [k for k in self.SLOW_MOTION_FACTORS if lo <= slow_spm * k <= hi]
        if len(fits) != 1:
            return 0.0, None
        factor = fits[0]
        logger.info(
            "CADENCE_SLOW_MOTION",
            measured_spm=f"{slow_spm:.1f}", factor=factor,
            cadence_spm=f"{slow_spm * factor:.1f}",
        )
        return round(slow_spm * factor, 1), factor

    # Knee-valley vs spectral estimates farther apart than this fraction
    # mean the valley train is fragmented (tracking noise) -- the
    # swap-immune spectral reading wins.
    CADENCE_CROSSCHECK_TOLERANCE = 0.18

    def _compute_cadence(self) -> float:
        """Compute cadence: knee-angle valleys cross-checked against a
        spectral estimate, with ankle-crossing counting as last resort.

        The knee-valley method reads only the near-side knee, so left/right
        identity swaps fragment its stride train into spurious half- and
        1.5-period intervals that still pass the per-interval gate --
        biasing cadence low while looking plausible (a corrupted clip once
        read 124 spm on a ~165 spm run). The spectral method reads the
        averaged both-ankles bounce, which survives those swaps; when the
        two disagree by more than ``CADENCE_CROSSCHECK_TOLERANCE`` the
        spectral value is used.

        Returns ``0.0`` when no method produced a reliable cadence -- the
        caller (compute_summary) omits the value in that case so a phantom
        0.0 doesn't reach downstream consumers as a measurement.
        """
        if self.frame_results:
            duration_ms = self.frame_results[-1].timestamp_ms - self.frame_results[0].timestamp_ms
            logger.info(
                "CADENCE_INPUT_DEBUG",
                total_frames=len(self.frame_results),
                video_duration_ms=round(duration_ms),
            )

        # Method 1: knee angle oscillation (valleys = stride boundaries)
        steps = self._detect_steps_from_knee_angles()
        n_strides = len(steps)
        knee_cadence = self._compute_cadence_from_strides(steps)

        # Method 2: spectral (swap-immune cross-check / replacement)
        spectral = self._cadence_spectral()

        if knee_cadence > 0 and spectral > 0:
            rel_diff = abs(knee_cadence - spectral) / spectral
            if rel_diff > self.CADENCE_CROSSCHECK_TOLERANCE:
                logger.info(
                    "CADENCE_RESULT", method="spectral_override",
                    knee_spm=f"{knee_cadence:.1f}",
                    cadence_spm=f"{spectral:.1f}",
                    rel_diff=f"{rel_diff:.2f}",
                )
                return spectral
            logger.info(
                "CADENCE_RESULT", method="knee_angles",
                cadence_spm=f"{knee_cadence:.1f}",
                spectral_agrees=f"{spectral:.1f}",
            )
            if n_strides < 4:
                self._record_warning(
                    f"Only {n_strides} strides detected -- cadence estimate may be "
                    f"imprecise. Longer clips (15+ seconds) give more reliable results."
                )
            return knee_cadence

        if knee_cadence > 0:
            logger.info("CADENCE_RESULT", method="knee_angles", cadence_spm=f"{knee_cadence:.1f}")
            if n_strides < 4:
                self._record_warning(
                    f"Only {n_strides} strides detected -- cadence estimate may be "
                    f"imprecise. Longer clips (15+ seconds) give more reliable results."
                )
            return knee_cadence

        if spectral > 0:
            logger.info("CADENCE_RESULT", method="spectral", cadence_spm=f"{spectral:.1f}")
            return spectral

        # Method 3: ankle Y mean-crossing counting (legacy fallback)
        cadence = self._cadence_from_ankle_position()
        if cadence > 0:
            logger.info("CADENCE_RESULT", method="ankle_position", cadence_spm=f"{cadence:.1f}")
            return cadence

        # Method 4: the clip is slow motion. Recover the real cadence from the
        # slowed rhythm, and record the factor -- every other time-derived
        # metric (ground contact, flight) is stretched by the same amount and
        # reads it back through _median_frame_spacing_ms.
        slowmo_cadence, factor = self._cadence_from_slow_motion()
        if slowmo_cadence > 0 and factor:
            self._slowmo_factor = factor
            logger.info(
                "CADENCE_RESULT", method="slow_motion",
                factor=factor, cadence_spm=f"{slowmo_cadence:.1f}",
            )
            self._record_warning(
                f"This clip looks like {factor}x slow motion. Cadence, ground "
                f"contact and flight time were rescaled to real time on that "
                f"basis -- if the clip was filmed at normal speed, treat them "
                f"as wrong and re-upload a normal-speed video."
            )
            return slowmo_cadence

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

        # Short clip may not have hit BODY_SCALE_MAX_SAMPLES -- lock on
        # whatever was collected (idempotent).
        self._lock_body_scale()

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
        spacing = float(np.median(deltas))
        # A slow-motion clip's timestamps run k times too slow, so every
        # duration derived from them (ground contact, flight) is k times too
        # long. Correcting the spacing fixes all of them at one point --
        # and the plausibility gates downstream then act as a second check
        # on whether the inferred factor was right.
        if self._slowmo_factor:
            spacing /= float(self._slowmo_factor)
        return spacing

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

    def stance_runs(self, min_run: int = 3) -> list[tuple[int, int]]:
        """``(first_frame, last_frame)`` of every *confirmed* ground contact.

        A run is confirmed when it lasts at least ``min_run`` contiguous
        stance frames. Debouncing (not just requiring the next frame to be
        stance) prevents single-frame gait-phase flicker from registering as
        extra contacts; without it an 8 s clip yields 30+ spurious "contacts"
        instead of ~1 per stride. Mirrors the stride-counter debounce in
        video_visualizer.

        Both ends are inclusive: ``first`` is the foot-strike frame and
        ``last`` is the final frame before the foot leaves the ground
        (toe-off). Uses the same stance-phase set as GCT/flight so every
        consumer shares one notion of "on the ground".
        """
        n = len(self.frame_results)
        if n == 0:
            return []
        stance_flags = [
            fr.extra_metrics.get("gait_phase") in GROUND_CONTACT_PHASES
            for fr in self.frame_results
        ]
        runs: list[tuple[int, int]] = []
        i = 0
        while i < n:
            if stance_flags[i]:
                j = i
                while j < n and stance_flags[j]:
                    j += 1
                if (j - i) >= min_run:
                    runs.append((i, j - 1))
                i = j
            else:
                i += 1
        return runs

    def _contact_frame_indices(self, min_run: int = 3) -> list[int]:
        """Frames where the NEAR foot begins a ground contact.

        :meth:`stance_runs` reports every footfall, alternating legs, because
        that is what ground-contact and flight time are about. Overstride, foot
        strike and the kinogram are not: they read the near-side landmarks, and
        sampling those while the FAR foot lands measures a leg in mid-swing.
        Measured on the treadmill fixture, that mistake put overstride at 0.62
        leg-lengths against 0.23 -- a fabricated fault, confidently reported.

        So contacts are kept only where the near ankle is actually down. The
        test is its depth below the hips, compared against the deepest it gets
        anywhere in the clip: a planted foot is near the bottom of its travel,
        a swinging one is not. Depth rather than a left/right label, because
        the labels are the unreliable part.
        """
        runs = self.stance_runs(min_run)
        if not runs:
            return []
        depths = np.array([
            fr.extra_metrics.get("_near_foot_depth", float("nan"))
            for fr in self.frame_results
        ], dtype=float)
        if not np.isfinite(depths).any():
            self._contacts_unfiltered = True
            return [start for start, _ in runs]

        deepest = float(np.nanpercentile(depths, 95))
        shallowest = float(np.nanpercentile(depths, 5))
        span = deepest - shallowest
        if span <= 1e-6:
            self._contacts_unfiltered = True
            return [start for start, _ in runs]
        floor = deepest - span * NEAR_CONTACT_BAND

        kept: list[int] = []
        for start, end in runs:
            window = depths[start:end + 1]
            window = window[np.isfinite(window)]
            if window.size and float(np.median(window)) >= floor:
                kept.append(start)
        # Every contact rejected means the near side could not be told apart
        # at all; fall back to all of them rather than silently measuring
        # nothing -- but SAY SO. The docstring above records what an
        # unfiltered mix does to overstride (0.62 vs 0.23, a fabricated
        # fault); the flag is what lets the summary withhold the metrics
        # sampled at these contacts instead of publishing that.
        if not kept:
            self._contacts_unfiltered = True
        return kept or [start for start, _ in runs]

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

        # Ankle: shank axis vs foot axis (see RUNNING_ANKLE_LANDMARKS).
        k_i, a_i, h_i, t_i = RUNNING_ANKLE_LANDMARKS[near]
        ankle_val, ankle_vis = calculate_shank_foot_angle_2d(wl, k_i, a_i, h_i, t_i)
        angles["ankle"] = ankle_val
        visibility["ankle"] = ankle_vis

        # Trunk lean: near-side shoulder + hip (world landmarks), SIGNED by the
        # direction of travel -- positive leaning forward, negative leaning
        # back. Unsigned, a torso 6 deg BEHIND vertical read "6.0" and sat
        # comfortably inside a band written for forward lean, so the one trunk
        # posture that is unambiguously a fault scored as optimal.
        #
        # NaN when the feet are not visible enough to fix the direction; the
        # average below already drops NaN samples, so a few such frames cost
        # nothing and a whole clip of them yields None rather than a number
        # whose sign nobody can vouch for.
        sh_idx, hp_idx = TRUNK_LANDMARKS[near]
        trunk_val = calculate_signed_segment_to_vertical(
            wl, sh_idx, hp_idx, calculate_forward_sign(wl),
        )
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

        # Redecide stance/swing now that the whole clip is in hand. Everything
        # below that counts frames -- ground contact, flight, the contact
        # frames overstride and foot-strike sample -- reads the phases this
        # writes, so it has to run before any of them.
        self._gait_meta = None
        try:
            self._gait_meta = self.recompute_gait_phases()
        except Exception as e:  # noqa: BLE001 -- fall back to the streamed phases
            logger.warning("GAIT_RECOMPUTE_FAILED", err=str(e))
        logger.info(
            "GAIT_PHASES",
            method=(self._gait_meta or {}).get("method", "streamed_fallback"),
            stance_fraction=(self._gait_meta or {}).get("stance_fraction"),
        )

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
            # Robust stand-ins for the cycle extremes. The raw min/max are
            # single-sample statistics over hundreds of frames: one mistracked
            # frame sets them, always in the alarming direction, and both feed
            # graded advice ("knee locked at 179 deg at contact"). The 5th/95th
            # percentiles sit at the same place on a clean signal -- the knee
            # dwells near each extreme for several frames per stride -- while
            # ignoring a lone outlier.
            "knee_min": (
                round(float(np.percentile(valid_knee, 5)), 1)
                if len(valid_knee) > 0 else None
            ),
            "knee_max": (
                round(float(np.percentile(valid_knee, 95)), 1)
                if len(valid_knee) > 0 else None
            ),
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
        if self._slowmo_factor:
            # Say so wherever these numbers travel: they are real-time values
            # reconstructed from a slowed clip, not measured off the timeline.
            summary["slow_motion_factor"] = self._slowmo_factor
            summary["time_base_inferred"] = True

        # What share of the clip the near foot was called "on the ground".
        # Running spends 30-40% of a cycle in stance (less when sprinting);
        # walking is over half. So this is a cheap, physiological check on
        # whether the phase machinery produced anything believable at all --
        # and unlike ground-contact time it does not depend on the timebase,
        # so it survives a slow-motion clip that every frame-counted metric
        # gets wrong. Emitted always, including when it is damning.
        if self.frame_results:
            stance = sum(
                1 for fr in self.frame_results
                if fr.extra_metrics.get("gait_phase") in GROUND_CONTACT_PHASES
            )
            summary["stance_fraction"] = round(stance / len(self.frame_results), 3)

        # vert_osc is stored in meters; range 0.01-0.25 m = 1-25 cm.
        if 0.01 <= vert_osc <= 0.25:
            summary["vertical_oscillation_m"] = vert_osc
            # Which real length the centimetres came from. A reading scaled off
            # a population-average torso is graded against a band it can sit
            # outside by a body's worth of proportion, so the difference has to
            # travel with the number rather than stay in the logs.
            summary["body_scale_source"] = self._body_scale_source
            if self.height_cm:
                summary["athlete_height_cm"] = round(self.height_cm)

        # Time-base / slow-motion guard. Cadence, GCT and flight all read time
        # off the container FPS. A clean running clip yields a cadence in
        # ~150-190 spm; when cadence is UNMEASURABLE but GCT/flight computed,
        # the real-time base is unverified -- almost always slow-motion (the
        # motion is stretched into a normal-fps file, so every absolute time
        # metric is inflated). Withhold the (confident-but-unverifiable) time
        # metrics rather than surface a wrong "prolonged ground contact", and
        # flag it so detect_issues skips the GCT verdict. Angles are unaffected.
        cadence_ok = 80.0 <= cadence <= 220.0
        self._time_base_uncertain = (not cadence_ok) and (gct_ms > 0 or flight_ms > 0)
        if self._time_base_uncertain:
            summary["time_base_uncertain"] = True
            self._record_warning(
                "Cadence couldn't be measured, so ground-contact and flight time "
                "are withheld. This usually means the clip is slow-motion (or the "
                "pace is very slow) -- time-based metrics need a normal-speed video "
                "to be accurate. Your joint angles are unaffected."
            )
            gct_ms = 0.0
            flight_ms = 0.0

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
        contacts_unfiltered = getattr(self, "_contacts_unfiltered", False)
        if contacts_unfiltered:
            # Both metrics below are sampled AT the contact frames; with the
            # near-side filter fallen back to every footfall, half of those
            # frames are the far leg in mid-swing. Absence over fabrication.
            summary["contacts_unfiltered"] = True
        if overstride_ratio > 0 and overstride_n >= 2 and not contacts_unfiltered:
            summary["overstride_ratio"] = overstride_ratio
            summary["overstride_estimated"] = True
            summary["overstride_contacts"] = overstride_n

        # Foot-strike pattern (heel/midfoot/forefoot) at contact. Only emit
        # when a pattern was classified from >= 2 clean contacts.
        if foot_strike is not None and foot_strike_n >= 2 and not contacts_unfiltered:
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

        # Low cadence. One number, one verdict: the scorer already refuses to
        # grade a cadence whose time base was INFERRED (slow-mo multiplier)
        # or is uncertain -- a confident "low cadence" warning from the same
        # guessed number would be the report disagreeing with itself.
        cadence = self._compute_cadence()
        if (
            cadence > 0 and cadence < 165
            and not self._slowmo_factor
            and not getattr(self, "_time_base_uncertain", False)
        ):
            issues.append({
                "type": "low_cadence",
                "severity": "warning",
                "value": f"{cadence:.0f} spm",
                "recommendation": f"Cadence is {cadence:.0f} spm. Target: {RUNNING_REFERENCE['cadence_spm'][0]}-{RUNNING_REFERENCE['cadence_spm'][1]} spm.",
            })

        # Prolonged ground contact -- usually a downstream symptom of
        # low cadence / overstriding, so the recommendation points back
        # to cadence rather than prescribing a separate drill. Skip entirely
        # when the time-base is uncertain (slow-mo): GCT is unverifiable there,
        # so a "prolonged contact" verdict would be a confident wrong claim.
        gct_ms = self._compute_ground_contact_time()
        gct_min, gct_max = RUNNING_REFERENCE["ground_contact_ms"]
        if (not getattr(self, "_time_base_uncertain", False)
                and gct_ms > 0 and gct_ms > gct_max + 40):  # +~1 frame tolerance
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
