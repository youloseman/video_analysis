"""Overlay video generator -- skeleton + angle annotations on every frame.

Reads the original video, draws skeleton bones and angle labels per frame
using the pre-computed analysis data, then re-encodes to web-safe H.264 MP4
via ffmpeg.
"""

import bisect
import math
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import structlog

from app.services.video_analysis import overlay_style
from app.services.video_analysis.biomechanics.base_analyzer import SportAnalyzer
from app.services.video_analysis.pipeline import (
    ARC_TRIPLETS,
    MIN_OVERLAY_VISIBILITY,
    SPORT_SAMPLE_RATES,
    VideoAnalysisPipeline,
    _draw_dashed_line,
    build_skeleton_geometry,
    landmarks_to_pixels,
)

# Phase colors (BGR for OpenCV). Warm = propulsive, cool = setup, gray = unknown.
SWIM_PHASE_COLORS: dict[str, tuple[int, int, int]] = {
    "entry":    (200, 200, 100),   # cyan-ish
    "catch":    (50, 200, 50),     # green
    "pull":     (50, 150, 255),    # orange
    "push":     (50, 50, 255),     # red
    "recovery": (180, 130, 80),    # muted blue
    "unknown":  (128, 128, 128),   # gray
}
SWIM_PHASE_LABELS: dict[str, str] = {
    "entry": "ENTRY", "catch": "CATCH", "pull": "PULL",
    "push": "PUSH", "recovery": "RECOVERY", "unknown": "UNKNOWN",
}

# Running gait phases (BGR for OpenCV). Warm = stance (foot on ground),
# cool = swing (foot in air), gray = unknown. The 8 raw GaitPhase values
# collapse into a stance/swing colour family so the timeline and legend
# stay readable while still surfacing the finer phase in the corner badge.
RUN_PHASE_COLORS: dict[str, tuple[int, int, int]] = {
    # Stance (warm)
    "initial_contact":  (50, 200, 255),   # amber
    "loading_response": (50, 170, 255),   # orange
    "midstance":        (50, 120, 240),   # deep orange
    "terminal_stance":  (60, 90, 235),    # red-orange
    "pre_swing":        (60, 60, 230),    # red
    # Swing (cool)
    "initial_swing":    (230, 170, 60),   # blue
    "mid_swing":        (210, 140, 50),   # deep blue
    "terminal_swing":   (200, 190, 90),   # teal
    "unknown":          (128, 128, 128),  # gray
}
RUN_PHASE_LABELS: dict[str, str] = {
    "initial_contact": "CONTACT", "loading_response": "LOADING",
    "midstance": "MIDSTANCE", "terminal_stance": "TOE-OFF PREP",
    "pre_swing": "TOE-OFF", "initial_swing": "SWING (early)",
    "mid_swing": "SWING (mid)", "terminal_swing": "SWING (late)",
    "unknown": "UNKNOWN",
}
# Phases where the foot is on the ground -- used to count strides (each
# new stance run after a swing = one stride of the near leg).
_RUN_STANCE_PHASES = frozenset({
    "initial_contact", "loading_response", "midstance",
    "terminal_stance", "pre_swing",
})

# Optional brand watermark, burned into the top-right of every overlay frame.
# Disabled by default for this standalone app. To brand the output, drop a
# transparent PNG at the path below and set WATERMARK_ENABLED = True. The Motus
# watermark asset is intentionally NOT bundled here.
WATERMARK_ENABLED = False
WATERMARK_OPACITY = 0.9                       # global blend strength on top of the PNG's own alpha
WATERMARK_PATH = Path(__file__).parent / "assets" / "watermark.png"
WATERMARK_HEIGHT_FRAC = 0.10                  # target mark height as a fraction of video height
WATERMARK_MIN_H = 40
WATERMARK_MAX_H = 110
# Cache of the BGRA watermark resized per target height (videos in a batch
# share a size, so this is effectively a single resize).
_watermark_cache: dict[int, "np.ndarray | None"] = {}

logger = structlog.get_logger()

# --- display crop ----------------------------------------------------------
#
# An athlete filling a tenth of the frame gets an overlay whose chips are wider
# than they are tall, over a picture that is mostly sky and empty ground. The
# numbers are right and the artifact is unreadable, which for the one thing an
# athlete actually shows people is most of its value gone.
#
# So the overlay is FRAMED for display. Three rules, each with a reason:
#
# * It happens BEFORE the annotation is drawn, not after. Chip sizes scale with
#   the frame they are drawn on, so cropping afterwards would keep them huge and
#   then cut them off; cropping first makes them proportional to the athlete.
# * It follows HORIZONTALLY ONLY. Vertical oscillation is the rise and fall of
#   the hips in the frame -- a window that chased them vertically would remove
#   the very thing the number underneath the video is reporting, and the video
#   would visibly disagree with the report.
# * It changes nothing measured. Every landmark, angle and metric is computed
#   before this runs, on the whole frame. This is framing, not analysis -- which
#   is what makes it safe, and is exactly why the same idea was rejected for the
#   analysis path, where it moved vertical oscillation by 36%.
CROP_ENABLED = True
# Only bother when the athlete is genuinely lost in the frame. Above this share
# of the height the overlay is already legible and cropping would just throw
# away context the athlete filmed on purpose.
CROP_TRIGGER_FRAC = 0.45
# Height of the window as a multiple of the athlete's full-clip vertical
# extent -- which already includes their bounce, since it is measured across
# every frame. The margin is what keeps a swinging arm or trailing foot inside.
CROP_HEIGHT_MULT = 1.9
# Seconds the window takes to follow. Long enough that it drifts rather than
# jerks, short enough not to lag a runner crossing the frame.
CROP_FOLLOW_S = 0.7


class VideoVisualizer:
    """Generates an overlay video with skeleton + biomechanical annotations."""

    def __init__(
        self,
        video_path: str,
        frame_data_list: list[dict[str, Any]],
        analyzer: SportAnalyzer,
        sport_type: str,
        cycling_position: str | None,
        output_dir: str,
        analysis_id: int,
        technique_score: int,
        letter_grade: str,
        angle_stats: dict[str, Any] | None = None,
        summary: dict[str, Any] | None = None,
        hide_angle_values: bool = False,
        annotation_level: str = "material",
    ):
        self.video_path = video_path
        self.frame_data_list = frame_data_list
        self.analyzer = analyzer
        self.sport_type = sport_type
        self.cycling_position = cycling_position
        self.output_dir = output_dir
        self.analysis_id = analysis_id
        self.technique_score = technique_score
        self.letter_grade = letter_grade
        self.angle_stats = angle_stats or {}
        self.summary = summary or {}
        # Teaser mode (free tier): draw the skeleton + arcs + callout markers,
        # but replace the numeric angle value with a lock glyph. The athlete
        # sees the tech works and where the joints are measured -- the numbers
        # themselves are the paid unlock. When on, we also burn a text
        # watermark so free output can't be passed off as a full report.
        self.hide_angle_values = hide_angle_values
        self.teaser_watermark = hide_angle_values

        # Build frame index mapping: video_frame_idx -> analyzed_frame_index
        self.sample_rate = SPORT_SAMPLE_RATES.get(sport_type, 1)
        self._frame_index_map: dict[int, int] = {}
        for i, fd in enumerate(frame_data_list):
            self._frame_index_map[fd["frame_idx"]] = i
        # Sorted video-frame indices of analyzed frames, for bisecting the
        # neighbours of an un-analyzed frame (adaptive sampling stride > 1).
        self._analyzed_video_indices: list[int] = sorted(self._frame_index_map)

        # Camera side for near/far skeleton coloring
        self.camera_side = self.summary.get("camera_side") if sport_type in ("run", "bike") else None

        # Pre-build label display config (reuse pipeline logic)
        pipeline = VideoAnalysisPipeline()
        self.label_configs = pipeline._get_angle_display_config(
            sport_type, summary, None,
            cycling_position=cycling_position,
        )
        # Which joints get a numbered callout. "material" annotates only the
        # ones this clip actually has something to say about; "all" is the old
        # behaviour, every joint on every frame. Default is material because a
        # frame carrying six chips is read as decoration -- the table below the
        # video is where every number lives, and it is not competing for the
        # athlete's attention with their own footage.
        self.annotation_level = annotation_level
        self._material_keys = self._pick_material_keys()

        # Phase overlay (swim + run): pre-compute per-frame phase sequence and
        # a cycle counter (swim = strokes, run = strides). Sport-specific config
        # (which extra_metrics key holds the phase, the colour/label maps, and
        # what boundary starts a new cycle) is selected here so the per-frame
        # draw path stays sport-agnostic.
        self._phase_sequence: list[str] = []
        self._cycle_numbers: list[int] = []   # stroke# (swim) or stride# (run)
        self._total_cycles: int = 0
        self._timeline_cache: np.ndarray | None = None
        self._phase_colors: dict[str, tuple[int, int, int]] = {}
        self._phase_labels: dict[str, str] = {}
        self._phase_legend_order: list[str] = []
        self._cycle_noun: str = "Cycle"

        if sport_type == "swim":
            self._phase_colors = SWIM_PHASE_COLORS
            self._phase_labels = SWIM_PHASE_LABELS
            self._phase_legend_order = ["entry", "catch", "pull", "push", "recovery"]
            self._cycle_noun = "Stroke"
        elif sport_type == "run":
            self._phase_colors = RUN_PHASE_COLORS
            self._phase_labels = RUN_PHASE_LABELS
            # Legend uses coarse stance/swing families, not all 8 raw phases.
            self._phase_legend_order = ["midstance", "pre_swing", "mid_swing"]
            self._cycle_noun = "Stride"

        if sport_type in ("swim", "run") and hasattr(analyzer, "frame_results"):
            phase_key = "stroke_phase" if sport_type == "swim" else "gait_phase"
            cycle_count = 0
            prev_phase = ""
            prev_in_stance = False
            # Debounce the stance/swing state for stride counting: a state must
            # persist for MIN_RUN_STATE_FRAMES before it counts, so single-frame
            # phase flicker (a lone stance frame in a swing run) doesn't inflate
            # the stride count. Without this an 8 s clip counts 100+ "strides".
            MIN_RUN_STATE_FRAMES = 3
            stance_streak = swing_streak = 0
            for fr in analyzer.frame_results:
                phase = fr.extra_metrics.get(phase_key, "unknown")
                self._phase_sequence.append(phase)
                if sport_type == "swim":
                    # New stroke on entry into the "entry" phase.
                    if prev_phase != "entry" and phase == "entry":
                        cycle_count += 1
                else:
                    # Debounced swing->stance transition = one stride (foot lands).
                    raw_stance = phase in _RUN_STANCE_PHASES
                    if raw_stance:
                        stance_streak += 1
                        swing_streak = 0
                    else:
                        swing_streak += 1
                        stance_streak = 0
                    if not prev_in_stance and stance_streak >= MIN_RUN_STATE_FRAMES:
                        cycle_count += 1
                        prev_in_stance = True
                    elif prev_in_stance and swing_streak >= MIN_RUN_STATE_FRAMES:
                        prev_in_stance = False
                self._cycle_numbers.append(cycle_count)
                prev_phase = phase
            self._total_cycles = cycle_count

        # Arc triplets for this sport
        self.arc_triplets = ARC_TRIPLETS.get(sport_type, {})

    def _plan_display_crop(
        self, width: int, height: int, fps: float,
    ) -> dict[str, Any] | None:
        """A window that follows the athlete sideways, fixed vertically.

        Returns None when the clip does not need it -- an athlete already
        filling the frame, or too few landmarks to place a window honestly.
        See the module notes above for why this is display-only.
        """
        if not CROP_ENABLED or width <= 0 or height <= 0:
            return None

        xs_mid: list[float] = []
        tops: list[float] = []
        bottoms: list[float] = []
        for fd in self.frame_data_list:
            lms = fd.get("normalized_landmarks")
            if not lms:
                continue
            xs = [lm.x for lm in lms
                  if (getattr(lm, "visibility", 0) or 0) >= 0.3
                  and lm.x is not None and not math.isnan(lm.x)]
            ys = [lm.y for lm in lms
                  if (getattr(lm, "visibility", 0) or 0) >= 0.3
                  and lm.y is not None and not math.isnan(lm.y)]
            if len(xs) < 4 or len(ys) < 4:
                continue
            xs_mid.append((min(xs) + max(xs)) / 2.0)
            tops.append(min(ys))
            bottoms.append(max(ys))
        if len(xs_mid) < 5:
            return None

        # The athlete's full-clip vertical extent -- their bounce is already
        # inside it, because it is measured over every frame.
        top, bottom = min(tops), max(bottoms)
        extent = bottom - top
        if extent <= 0.01 or extent >= CROP_TRIGGER_FRAC:
            return None

        crop_h = min(1.0, extent * CROP_HEIGHT_MULT)
        crop_h_px = max(64, int(round(crop_h * height)))
        crop_w_px = min(width, max(64, int(round(crop_h_px * width / height))))
        if crop_h_px >= height and crop_w_px >= width:
            return None

        # Fixed vertically, centred on the athlete's whole-clip band.
        centre_y = (top + bottom) / 2.0
        y0 = int(round(centre_y * height - crop_h_px / 2))
        y0 = max(0, min(y0, height - crop_h_px))

        # Horizontally: one x per ANALYZED frame, smoothed, then read per video
        # frame through the same nearest-analyzed mapping the skeleton uses.
        span = max(3, int(round(CROP_FOLLOW_S * fps / max(1, self.sample_rate))))
        arr = np.array(xs_mid, dtype=np.float64)
        kernel = np.ones(span) / span
        smooth = np.convolve(
            np.pad(arr, (span, span), mode="edge"), kernel, mode="same",
        )[span:-span]

        xs_px: list[int] = []
        for cx in smooth:
            x0 = int(round(cx * width - crop_w_px / 2))
            xs_px.append(max(0, min(x0, width - crop_w_px)))

        logger.info(
            "OVERLAY_CROP",
            athlete_frac=round(extent, 3),
            crop=f"{crop_w_px}x{crop_h_px}",
            source=f"{width}x{height}",
            athlete_now=round(extent * height / crop_h_px, 2),
        )
        return {"xs": xs_px, "y0": y0, "w": crop_w_px, "h": crop_h_px}

    def generate(self) -> str | None:
        """Generate overlay video. Returns path to MP4 or None on failure."""
        if not self.frame_data_list:
            logger.warning("No frame data -- skipping overlay video")
            return None

        import cv2

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error("Cannot open video for overlay generation", path=self.video_path)
            return None

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Frame the athlete for display before anything is drawn (see the
        # module notes). Everything below writes at the CROP's size when one
        # applies, so the annotation is scaled to the athlete rather than to
        # the sky above them.
        crop = None
        try:
            crop = self._plan_display_crop(width, height, fps)
        except Exception as e:  # noqa: BLE001 -- framing must never lose a render
            logger.warning("OVERLAY_CROP_FAILED", err=str(e))
        draw_w = crop["w"] if crop else width
        draw_h = crop["h"] if crop else height

        # Output paths
        os.makedirs(self.output_dir, exist_ok=True)
        final_mp4 = os.path.join(self.output_dir, f"{self.analysis_id}.mp4")

        # With ffmpeg: write a temp AVI (XVID) then re-encode to web-safe H.264.
        # Without ffmpeg (common on dev machines): write the MP4 directly via
        # OpenCV's mp4v muxer -- plays in VLC/most players. Install ffmpeg and
        # re-run for browser-safe H.264 + faststart.
        self._use_ffmpeg = shutil.which("ffmpeg") is not None

        # What the encode will do to every drawn pixel (1.0 without ffmpeg,
        # where the mp4v writer's output IS the delivered clip). The skeleton
        # stroke is chosen against this rather than against draw_w: a display
        # crop makes the drawing surface SMALLER while the encoder shrinks it
        # further, and scaling the stroke to the surface let both effects land
        # on the athlete as a hairline.
        out_scale = 1.0
        if self._use_ffmpeg and draw_w:
            out_scale = min(1.0, self._even_target_dims(draw_w, draw_h)[0] / draw_w)
        if self._use_ffmpeg:
            temp_avi = os.path.join(self.output_dir, f"{self.analysis_id}_temp.avi")
            writer = cv2.VideoWriter(temp_avi, cv2.VideoWriter_fourcc(*"XVID"), fps, (draw_w, draw_h))
            writer_target = temp_avi
        else:
            logger.warning("ffmpeg not found -- writing MP4 directly via OpenCV (mp4v)")
            temp_avi = None
            writer = cv2.VideoWriter(final_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (draw_w, draw_h))
            writer_target = final_mp4

        if not writer.isOpened():
            logger.error("Cannot create VideoWriter", path=writer_target)
            cap.release()
            return None

        # Phase overlay (swim + run): cache timeline bar once (saves ~0.5 ms
        # per frame vs rebuilding).
        if self._phase_sequence:
            self._timeline_cache = self._build_phase_timeline(draw_w)
            self._overlay_fps = fps

        # Track the last resolvable pose (frames past the final analyzed one
        # keep drawing it, so the overlay doesn't vanish on the tail).
        last_analyzed_idx: int | None = None
        last_landmarks: list[Any] | None = None
        video_frame_idx = 0
        frames_written = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Rotation/orientation metadata can make the decoded frame size
            # differ from the reported CAP_PROP dims. The writer needs an exact
            # (width, height) match or it silently drops frames and produces an
            # empty/corrupt file -- resize the odd frame to keep the video valid.
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))

            # Resolve the pose for this video frame (exact hit, or an
            # interpolated skeleton between two analyzed frames).
            analyzed_idx, landmarks = self._pose_for_video_frame(video_frame_idx)
            if analyzed_idx is not None:
                last_analyzed_idx = analyzed_idx
                last_landmarks = landmarks

            # Frame for display BEFORE drawing, so the annotation is scaled to
            # the athlete rather than to the whole picture. The window follows
            # sideways and is fixed vertically -- see the module notes.
            crop_offset = (0, 0)
            if crop:
                _ai = last_analyzed_idx if last_analyzed_idx is not None else 0
                _x0 = crop["xs"][min(_ai, len(crop["xs"]) - 1)]
                _y0 = crop["y0"]
                frame = frame[_y0:_y0 + crop["h"], _x0:_x0 + crop["w"]].copy()
                crop_offset = (_x0, _y0)

            # Draw overlay if we have landmark data. The chip layer is painted
            # with PIL and returns a NEW array -- rebind rather than assume
            # in-place mutation.
            if last_analyzed_idx is not None:
                frame = self._draw_frame_overlay(
                    cv2, frame, last_analyzed_idx, draw_w, draw_h,
                    landmarks_override=last_landmarks,
                    source_dims=(width, height), crop_offset=crop_offset,
                    out_scale=out_scale,
                )

            # Brand watermark on every frame (even un-analyzed ones)
            if WATERMARK_ENABLED:
                self._draw_watermark(cv2, frame, draw_w, draw_h)

            writer.write(frame)
            frames_written += 1
            video_frame_idx += 1

        cap.release()
        writer.release()

        if frames_written == 0:
            logger.warning("No frames written to overlay video")
            if temp_avi:
                self._cleanup_file(temp_avi)
            return None

        # Re-encode with ffmpeg to web-safe H.264 MP4 (only when ffmpeg exists;
        # otherwise final_mp4 was written directly by the mp4v writer above).
        # The re-encode also downscales to ~720p -- overlays are for phone
        # viewing, so a smaller clip downloads faster and encodes quicker.
        if self._use_ffmpeg:
            # The AVI on disk is whatever was DRAWN -- the crop when one
            # applied -- so the encode target has to follow it, or ffmpeg is
            # handed the source aspect and stretches the result.
            out_w, out_h = self._even_target_dims(draw_w, draw_h)
            success = self._reencode_to_mp4(temp_avi, final_mp4, out_w, out_h)
            self._cleanup_file(temp_avi)
            if not success:
                return None

        logger.info(
            "Overlay video generated",
            analysis_id=self.analysis_id,
            frames=frames_written,
            path=final_mp4,
        )
        return final_mp4

    def render_keyframe(self, max_width: int = 720, quality: int = 82) -> str | None:
        """Render ONE representative annotated frame as a small JPEG data URI.

        Used for the history thumbnail -- a single frame with skeleton + angle
        labels + score badge, so we can keep a visual record without storing the
        whole overlay video. Returns None on any failure (never blocks analysis).
        """
        if not self.frame_data_list:
            return None
        import base64

        import cv2

        best_idx = self._pick_keyframe_idx()

        try:
            cap = cv2.VideoCapture(self.video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_data_list[best_idx]["frame_idx"])
            ok, frame = cap.read()
            cap.release()
            if not ok or frame is None:
                return None
            h, w = frame.shape[:2]
            # Drawn full-size, delivered at max_width -- tell the renderer, or
            # the stroke it picks is for a frame nobody will see at this size.
            frame = self._draw_frame_overlay(
                cv2, frame, best_idx, w, h,
                out_scale=min(1.0, max_width / w) if w else 1.0,
            )
            if w > max_width:
                nh = int(round(h * max_width / w))
                frame = cv2.resize(frame, (max_width, nh), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                return None
            return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
        except Exception as e:  # noqa: BLE001
            logger.warning("KEYFRAME_FAILED", err=str(e))
            return None

    def _pick_keyframe_idx(self) -> int:
        """The frame the keyframe still is drawn from.

        Readable means DRAWABLE first: this still leads the results page, and
        a frame whose near leg was identity-gated (blanked or display-filled
        from a prediction) renders with the leg missing or misplaced --
        visibility alone cannot see that, because the gate deliberately
        leaves visibility untouched. So a candidate must carry a finite,
        ungated near-side leg before visibility gets a vote.
        """
        n = len(self.frame_data_list)
        cand = sorted({int(n * p) for p in
                       (0.30 + 0.02 * k for k in range(21))})
        cand = [i for i in cand if 0 <= i < n] or [n // 2]

        side_leg = {
            "left": (23, 25, 27, 29, 31), "right": (24, 26, 28, 30, 32),
        }.get(self.camera_side or "", (23, 24, 25, 26, 27, 28))

        def leg_drawable(fd: dict) -> bool:
            if fd.get("leg_gate_filled"):
                return False
            for idx in side_leg:
                lm = fd["normalized_landmarks"][idx]
                x = getattr(lm, "x", None)
                if x is None or (isinstance(x, float) and math.isnan(x)):
                    return False
            return True

        drawable = [i for i in cand if leg_drawable(self.frame_data_list[i])]
        pool = drawable or cand

        if self.sport_type == "bike":
            # Prefer a BDC-phase frame: leg extended, foot planted on the
            # pedal -- the classic fit-photo pose, and the phase where the
            # foot cluster tracks best (the kinogram's tiles, which the
            # athlete called perfectly placed, are picked at these events;
            # mid-stroke and TDC frames are where the foot drifts).
            ankle_idx = {"left": 27, "right": 28}.get(self.camera_side or "")
            if ankle_idx is not None:
                ys = []
                for i in pool:
                    lm = self.frame_data_list[i]["normalized_landmarks"][ankle_idx]
                    y = getattr(lm, "y", None)
                    if y is not None and not (
                        isinstance(y, float) and math.isnan(y)
                    ):
                        ys.append((y, i))
                if len(ys) >= 5:
                    ys.sort(reverse=True)          # image y grows downward
                    pool = [i for _, i in ys[:max(3, len(ys) // 5)]]

        best_idx, best_vis = pool[0], -1.0
        for i in pool:
            lms = self.frame_data_list[i]["normalized_landmarks"]
            vis = sum(
                getattr(lm, "visibility", 0.5) for lm in lms
            ) / max(1, len(lms))
            if vis > best_vis:
                best_vis, best_idx = vis, i
        return best_idx

    def _pose_for_video_frame(
        self, video_frame_idx: int,
    ) -> tuple[int | None, list[Any] | None]:
        """Resolve the pose to draw on a given video frame.

        Returns ``(analyzed_idx, landmarks)``: ``analyzed_idx`` is the
        temporally nearest analyzed frame (its angles/phase drive the labels)
        and ``landmarks`` are the positions the skeleton is drawn at.

        At stride 1 every video frame is an exact hit. When adaptive sampling
        raised the stride, frames between two analyzed ones get a linearly
        interpolated skeleton -- holding the previous pose instead leaves the
        skeleton visibly trailing the athlete (worst on real-speed 60 fps
        clips, where one stride step is a lot of motion).
        """
        idxs = self._analyzed_video_indices
        if not idxs:
            return None, None
        pos = bisect.bisect_right(idxs, video_frame_idx) - 1
        if pos < 0:
            return None, None  # before the first analyzed frame

        v0 = idxs[pos]
        a0 = self._frame_index_map[v0]
        if v0 == video_frame_idx or pos + 1 >= len(idxs):
            # Exact hit, or past the last analyzed frame (hold it).
            return a0, self.frame_data_list[a0]["normalized_landmarks"]

        v1 = idxs[pos + 1]
        a1 = self._frame_index_map[v1]
        t = (video_frame_idx - v0) / float(v1 - v0)
        nearest_idx = a0 if t < 0.5 else a1

        def _finite(v: Any) -> bool:
            return isinstance(v, (int, float)) and not (
                isinstance(v, float) and math.isnan(v)
            )

        lms0 = self.frame_data_list[a0]["normalized_landmarks"]
        lms1 = self.frame_data_list[a1]["normalized_landmarks"]
        blended: list[Any] = []
        for lm0, lm1 in zip(lms0, lms1):
            if not (_finite(lm0.x) and _finite(lm0.y)
                    and _finite(lm1.x) and _finite(lm1.y)):
                # A gated (NaN) end: snap to the temporally closer pose --
                # blending with NaN would erase the landmark for the whole gap.
                blended.append(lm0 if t < 0.5 else lm1)
                continue
            vis0 = getattr(lm0, "visibility", 1.0)
            vis1 = getattr(lm1, "visibility", 1.0)
            blended.append(SimpleNamespace(
                x=lm0.x + (lm1.x - lm0.x) * t,
                y=lm0.y + (lm1.y - lm0.y) * t,
                z=lm0.z if t < 0.5 else lm1.z,
                # min(): a landmark unreliable at either end stays hidden for
                # the whole gap rather than popping in halfway through.
                visibility=min(
                    vis0 if vis0 is not None else 1.0,
                    vis1 if vis1 is not None else 1.0,
                ),
            ))
        return nearest_idx, blended

    def _draw_frame_overlay(
        self, cv2_mod: Any, frame: Any, analyzed_idx: int, width: int, height: int,
        landmarks_override: list[Any] | None = None,
        source_dims: tuple[int, int] | None = None,
        crop_offset: tuple[int, int] = (0, 0),
        out_scale: float = 1.0,
    ) -> Any:
        """Draw skeleton + angle labels on a single frame.

        Returns the annotated frame. The chip/text layer is painted with PIL, so
        a NEW array comes back -- callers must use the return value rather than
        relying on in-place mutation.

        ``width``/``height`` are the SURFACE being drawn on, which is what every
        chip position and type size is scaled against. When that surface is a
        crop of the source (see the display-crop notes), ``source_dims`` and
        ``crop_offset`` say where it came from, so the landmarks -- which are
        normalized against the WHOLE frame -- land in the right place on it.

        ``out_scale`` is the resize this render will be put through downstream
        (the ffmpeg encode, the keyframe's shrink to ``max_width``). The
        skeleton stroke is pre-divided by it so the delivered weight is the same
        whichever path drew it -- see ``overlay_style.skeleton_weights``.

        ``landmarks_override`` lets the video path draw the skeleton at
        interpolated positions (between two analyzed frames) while the labels,
        angles and phase still come from ``analyzed_idx``.
        """
        normalized_lms = (
            landmarks_override
            if landmarks_override is not None
            else self.frame_data_list[analyzed_idx]["normalized_landmarks"]
        )

        # Convert to pixel coordinates with visibility (NaN-safe -- see helper).
        # Normalized against the SOURCE frame, then shifted into the surface
        # being drawn on, which is the same thing when nothing was cropped.
        src_w, src_h = source_dims or (width, height)
        pixel_coords = landmarks_to_pixels(
            normalized_lms, src_w, src_h, offset=crop_offset,
        )

        # -- 1. Skeleton bones (near-side only for bike/run) --
        _segments, _dots = build_skeleton_geometry(
            pixel_coords, self.sport_type, self.camera_side,
        )

        # Neon skeleton (shared style). The glow runs on the clip as well as on
        # the still: it is what makes the skeleton read as neon rather than as
        # hairlines, and building it small (see overlay_style) brought the cost
        # to a few ms a frame. The still and the video are the same picture of
        # the same rider -- them looking like two different products was a bug.
        _sk = max(1.0, min(2.2, width / 900))
        _line_w, _dot_r = overlay_style.skeleton_weights(
            width, height, out_scale=out_scale,
        )
        overlay_style.draw_glow_skeleton(
            cv2_mod, frame, _segments, _dots,
            glow=True, line_w=_line_w, dot_r=_dot_r,
        )
        chips = overlay_style.ChipLayer(frame)

        # Header first so it reserves the top strip (chips get nudged clear of it).
        _sport_labels = {"run": "RUN", "bike": "BIKE", "swim": "SWIM"}
        _score = self.technique_score
        _hdr_status = (
            "good" if _score >= 75 else "warn" if _score >= 60 else "bad"
        )
        _title = (
            "AERODYNAMIC PROFILE"
            if (self.sport_type == "bike"
                and self.cycling_position in ("tt_aero", "triathlon"))
            else ("CYCLING PROFILE" if self.sport_type == "bike" else "RUNNING PROFILE")
        )
        _pad = int(max(10, height * 0.018))
        chips.header(
            (_pad, _pad), _sport_labels.get(self.sport_type, self.sport_type.upper()),
            f"{_score}/100", self.letter_grade, _hdr_status,
            right_text=_title, frame_w=width, scale=_sk,
        )

        # -- 2. Angle arcs + labels --
        if len(pixel_coords) > 25:
            # Get per-frame angle values from analyzer's angle_history
            frame_angles = self._get_frame_angles(analyzed_idx)

            # Body size reference for adaptive scaling
            s11x, s11y, _ = pixel_coords[11]
            h23x, h23y, _ = pixel_coords[23]
            body_height_px = abs(s11y - h23y)
            offset_px = max(70, int(body_height_px * 0.5))

            offset_vectors = {
                "left":       (-offset_px, 0),
                "right":      (offset_px, 0),
                "up":         (0, -offset_px),
                "down":       (0, offset_px),
                "up-left":    (-offset_px, -int(offset_px * 0.7)),
                "up-right":   (offset_px, -int(offset_px * 0.7)),
                "down-left":  (-offset_px, int(offset_px * 0.7)),
                "down-right": (offset_px, int(offset_px * 0.7)),
            }

            for cfg in self.label_configs:
                # Quiet joints keep their skeleton but lose the arc and the
                # callout -- see _pick_material_keys.
                if not self._annotates(str(cfg["key"])):
                    continue
                # Use per-frame angle value (not mean)
                angle_val = frame_angles.get(cfg["key"])
                if angle_val is None or np.isnan(angle_val):
                    continue

                lm_idx = cfg["idx"]
                if lm_idx >= len(pixel_coords):
                    continue

                _, _, lm_vis = pixel_coords[lm_idx]
                if lm_vis < MIN_OVERLAY_VISIBILITY:
                    continue

                opt_min, opt_max = cfg["optimal"]
                status = overlay_style.status_for(angle_val, opt_min, opt_max)
                status_rgb = overlay_style.STATUS_COLORS.get(
                    status, overlay_style.INK_SOFT,
                )
                # arc still wants BGR
                color = (status_rgb[2], status_rgb[1], status_rgb[0])

                # Draw angle arc
                triplet = self.arc_triplets.get(cfg["key"])
                if triplet:
                    VideoAnalysisPipeline._draw_angle_arc(
                        cv2_mod, frame, pixel_coords, *triplet,
                        color, body_height_px,
                    )

                # Leader line + neon chip
                jx, jy, _ = pixel_coords[lm_idx]
                dx, dy = offset_vectors.get(cfg["offset_dir"], (offset_px, 0))
                align = "right" if dx < 0 else "left"
                lx = max(5, min(width - 5, jx + dx))
                ly = max(20, min(height - 20, jy + dy))

                # Teaser: mask the number (skeleton + which joint stays visible,
                # the measured value is locked behind an upgrade).
                if self.hide_angle_values:
                    value_txt, status = "LOCKED", "muted"
                    status_rgb = overlay_style.INK_SOFT
                else:
                    value_txt = f"{angle_val:.0f}°"

                overlay_style.draw_leader(cv2_mod, frame, (jx, jy), (lx, ly), status_rgb)
                chips.metric_chip((lx, ly), str(cfg["name"]).upper(), value_txt,
                                  status, scale=_sk, align=align)

        # -- 2b. Head alignment + pelvic ratio overlays (bike only) --
        if self.sport_type == "bike" and len(pixel_coords) > 25:
            near = self.summary.get("near_side", "left")
            s11x, s11y, _ = pixel_coords[11]
            h23x, h23y, _ = pixel_coords[23]
            bh_px = abs(s11y - h23y)

            # Per-frame head alignment from extra_metrics
            fr = (
                self.analyzer.frame_results[analyzed_idx]
                if analyzed_idx < len(self.analyzer.frame_results)
                else None
            )
            if fr and fr.extra_metrics and bh_px > 40:
                head_key = f"{near}_head_alignment"
                head_score = fr.extra_metrics.get(head_key)
                if head_score is not None and not np.isnan(head_score) and head_score > 0:
                    if near == "left":
                        sh_i, hp_i, ear_i = 11, 23, 7
                    else:
                        sh_i, hp_i, ear_i = 12, 24, 8

                    shx, shy, shv = pixel_coords[sh_i]
                    hpx, hpy, hpv = pixel_coords[hp_i]
                    eax, eay, eav = pixel_coords[ear_i]

                    if shv >= MIN_OVERLAY_VISIBILITY and hpv >= MIN_OVERLAY_VISIBILITY:
                        _draw_dashed_line(cv2_mod, frame, (hpx, hpy), (shx, shy), (180, 180, 255), 1)
                        dx_ext = shx - hpx
                        dy_ext = shy - hpy
                        ext_x = shx + int(dx_ext * 0.5)
                        ext_y = shy + int(dy_ext * 0.5)
                        _draw_dashed_line(cv2_mod, frame, (shx, shy), (ext_x, ext_y), (180, 180, 255), 1)

                        if eav >= MIN_OVERLAY_VISIBILITY and self._annotates("head_alignment"):
                            if self.hide_angle_values:
                                h_status, h_text = "muted", "LOCKED"
                            else:
                                h_status = (
                                    "good" if head_score >= 75
                                    else ("warn" if head_score >= 50 else "bad")
                                )
                                h_text = f"{head_score:.0f}/100"
                            hlx = max(5, min(width - 5, eax + int(28 * _sk)))
                            hly = max(20, min(height - 20, eay - int(26 * _sk)))
                            overlay_style.draw_leader(
                                cv2_mod, frame, (eax, eay), (hlx, hly),
                                overlay_style.STATUS_COLORS.get(h_status, overlay_style.INK_SOFT),
                            )
                            chips.metric_chip((hlx, hly), "HEAD POSITION", h_text,
                                              h_status, scale=_sk)

            # Pelvic ratio (use summary average)
            pelvic = self.summary.get("pelvic_ratio", 0)
            if pelvic > 0 and bh_px > 40 and self._annotates("pelvic_ratio"):
                from app.services.video_analysis.biomechanics.cycling_positions import get_cycling_reference
                ref = get_cycling_reference(self.cycling_position)
                p_min, p_max = ref["pelvic_ratio"]
                if self.hide_angle_values:
                    p_status, p_text = "muted", "LOCKED"
                else:
                    # ratio, not degrees -> small margin floor
                    p_status = overlay_style.status_for(
                        pelvic, p_min, p_max, min_margin=0.3,
                    )
                    p_text = f"{pelvic:.1f}x"
                hp_i2 = 23 if near == "left" else 24
                hpx2, hpy2, _ = pixel_coords[hp_i2]
                off_px = max(50, int(bh_px * 0.35))
                plx = max(5, min(width - 5, hpx2 - int(off_px * 0.5)))
                ply = max(20, min(height - 20, hpy2 + int(off_px * 0.6)))
                overlay_style.draw_leader(
                    cv2_mod, frame, (hpx2, hpy2), (plx, ply),
                    overlay_style.STATUS_COLORS.get(p_status, overlay_style.INK_SOFT),
                )
                chips.metric_chip((plx, ply), "PELVIC TILT", p_text, p_status,
                                  scale=_sk, align="right")

        # -- 3. (header is staged on `chips` up top and painted in the flush below) --

        # -- 4. Phase overlay (swim + run): phase label, cycle counter,
        #       timeline, legend. Driven by sport-specific config set in __init__.
        #       The phase label + counter are meaningful on a single frame; the
        #       timeline bar + legend need a moving marker, so they only draw on
        #       the video (guarded by _timeline_cache, built in generate()).
        if self._phase_sequence and analyzed_idx < len(self._phase_sequence):
            idx = analyzed_idx
            phase = self._phase_sequence[idx]
            cycle = self._cycle_numbers[idx]
            self._draw_phase_label(cv2_mod, frame, phase, width)
            self._draw_cycle_counter(cv2_mod, frame, cycle, width)
            self._draw_phase_timeline(cv2_mod, frame, idx, width, height)
            if self._timeline_cache is not None and \
                    analyzed_idx < int(getattr(self, "_overlay_fps", 30) * 2):
                self._draw_phase_legend(cv2_mod, frame, width, height)

        # -- 5. Branding / free-tier teaser --
        chips.brand(
            (width - _pad, height - _pad), "FLAPP",
            "FREE" if self.teaser_watermark else "", scale=_sk,
        )

        # -- 6. Paint the header + all chips in a single PIL pass --
        return chips.flush()

    # Share of a clip's readable frames that must sit outside the reference
    # band before a joint earns a callout of its own. Well under half: a fault
    # that shows up in a quarter of the stride is still a fault, but a joint
    # that only grazes the band on a handful of frames is noise.
    _MATERIAL_FRAME_SHARE = 0.25
    # Always annotated. These carry the score in both sports -- knee extension
    # is the headline number of a bike fit and trunk lean of a running stride --
    # so their absence would read as "not measured" rather than as "fine".
    _HEADLINE_KEYS = ("knee", "trunk")
    # Ceiling on joint callouts per frame. Four labelled joints over a moving
    # body is about what a person can read; past that the frame is decoration.
    _MAX_CALLOUTS = 4

    def _pick_material_keys(self) -> set[str]:
        """Joints whose readings this clip actually has something to say about.

        Decided once, over the whole clip, rather than per frame: a chip that
        blinks in and out as a value crosses its band mid-stride is harder to
        read than one that never appears at all.
        """
        keys = {str(cfg["key"]) for cfg in self.label_configs}
        if self.annotation_level == "all":
            return keys

        def is_headline(key: str) -> bool:
            # Match on whole words, so "trunk" catches both the bike's
            # trunk_angle and the run's trunk_lean, and "knee" catches
            # left_knee / right_knee without also catching a hypothetical
            # kneecap_something.
            parts = key.split("_")
            return any(h in parts for h in self._HEADLINE_KEYS)

        material = {k for k in keys if is_headline(k)}
        scored: list[tuple[float, float, str]] = []
        for cfg in self.label_configs:
            key = str(cfg["key"])
            if key in material:
                continue
            values = self.analyzer.angle_history.get(key) or []
            opt_min, opt_max = cfg["optimal"]
            readable = flagged = bad = 0
            for val in values:
                if val is None or np.isnan(val):
                    continue
                readable += 1
                status = overlay_style.status_for(val, opt_min, opt_max)
                if status != "good":
                    flagged += 1
                if status == "bad":
                    bad += 1
            if readable and flagged / readable >= self._MATERIAL_FRAME_SHARE:
                scored.append((bad / readable, flagged / readable, key))

        # Worst first, so the cap drops the mildest rather than whichever the
        # config happened to list last.
        for _, _, key in sorted(scored, reverse=True):
            if len(material) >= self._MAX_CALLOUTS:
                break
            material.add(key)

        # A sport whose config names neither a knee nor a trunk would otherwise
        # render bare.
        if not material:
            material = keys

        # The two bike callouts that aren't joint angles carry their own
        # clip-level verdicts, so they are judged on those rather than on a
        # frame count.
        if self.sport_type == "bike":
            head_avg = self.summary.get("head_alignment_avg")
            if isinstance(head_avg, (int, float)) and not np.isnan(head_avg) and head_avg < 75:
                material.add("head_alignment")
            pelvic = self.summary.get("pelvic_ratio")
            if isinstance(pelvic, (int, float)) and pelvic > 0:
                from app.services.video_analysis.biomechanics.cycling_positions import (
                    get_cycling_reference,
                )
                p_min, p_max = get_cycling_reference(self.cycling_position)["pelvic_ratio"]
                if overlay_style.status_for(pelvic, p_min, p_max, min_margin=0.3) != "good":
                    material.add("pelvic_ratio")
        return material

    def _annotates(self, key: str) -> bool:
        return self.annotation_level == "all" or key in self._material_keys

    def _get_frame_angles(self, analyzed_idx: int) -> dict[str, float]:
        """Get angle values for a specific analyzed frame index."""
        result: dict[str, float] = {}
        for angle_name, values in self.analyzer.angle_history.items():
            if analyzed_idx < len(values):
                result[angle_name] = values[analyzed_idx]
        # Also include trunk_lean / trunk_angle from frame_results
        if analyzed_idx < len(self.analyzer.frame_results):
            fr = self.analyzer.frame_results[analyzed_idx]
            if "trunk_angle" in fr.angles:
                result["trunk_angle"] = fr.angles["trunk_angle"]
            if "trunk_lean" in fr.angles:
                result["trunk_lean"] = fr.angles["trunk_lean"]
        return result

    @staticmethod
    def _get_watermark(cv2_mod: Any, target_h: int) -> "np.ndarray | None":
        """Load the brand watermark (BGRA), resized to target_h. Cached.

        Returns None if the asset is missing or unreadable -- watermarking
        is best-effort and must never break video generation.
        """
        if target_h in _watermark_cache:
            return _watermark_cache[target_h]

        mark: "np.ndarray | None" = None
        try:
            raw = cv2_mod.imread(str(WATERMARK_PATH), cv2_mod.IMREAD_UNCHANGED)
            if raw is not None:
                # Ensure a 4-channel BGRA image.
                if raw.ndim == 3 and raw.shape[2] == 3:
                    raw = cv2_mod.cvtColor(raw, cv2_mod.COLOR_BGR2BGRA)
                if raw.ndim == 3 and raw.shape[2] == 4:
                    h, w = raw.shape[:2]
                    scale = target_h / float(h)
                    target_w = max(1, int(round(w * scale)))
                    mark = cv2_mod.resize(
                        raw, (target_w, target_h), interpolation=cv2_mod.INTER_AREA,
                    )
            else:
                logger.warning("Watermark asset not found", path=str(WATERMARK_PATH))
        except Exception as e:
            logger.warning("Watermark load failed", err=str(e))

        _watermark_cache[target_h] = mark
        return mark

    def _draw_text_watermark(
        self, cv2_mod: Any, frame: Any, width: int, height: int,
    ) -> None:
        """Burn a small 'FLAPP · FREE' text mark bottom-right.

        Asset-free (unlike the PNG lockup) so it always renders. Used for the
        free-tier teaser so its output is visibly a free sample.
        """
        text = "FLAPP - FREE"
        font = cv2_mod.FONT_HERSHEY_SIMPLEX
        scale = max(0.4, min(0.7, width / 1400))
        thick = 1 if scale < 0.55 else 2
        (tw, th), _ = cv2_mod.getTextSize(text, font, scale, thick)
        pad = int(max(8, height * 0.015))
        x = width - tw - pad
        y = height - pad
        if x < 0 or y - th < 0:
            return
        # Subtle drop shadow for legibility on any background, then the mark.
        cv2_mod.putText(frame, text, (x + 1, y + 1), font, scale, (0, 0, 0), thick + 1, cv2_mod.LINE_AA)
        cv2_mod.putText(frame, text, (x, y), font, scale, (255, 255, 255), thick, cv2_mod.LINE_AA)

    def _draw_watermark(
        self, cv2_mod: Any, frame: Any, width: int, height: int,
    ) -> None:
        """Composite the (optional) brand lockup into the top-right corner.

        Alpha-blends the rasterized logo (with its own transparency) onto
        the frame, scaled relative to the video so it reads on phone clips
        and 4K alike. No-op if the asset can't be loaded or won't fit.
        """
        target_h = int(min(WATERMARK_MAX_H, max(WATERMARK_MIN_H, height * WATERMARK_HEIGHT_FRAC)))
        mark = VideoVisualizer._get_watermark(cv2_mod, target_h)
        if mark is None:
            return

        mh, mw = mark.shape[:2]
        pad = int(max(10, height * 0.02))
        x = width - mw - pad
        y = pad
        if x < 0 or y < 0 or x + mw > width or y + mh > height:
            return  # frame too small for the mark at this size

        roi = frame[y:y + mh, x:x + mw].astype(np.float32)
        mark_bgr = mark[:, :, :3].astype(np.float32)
        alpha = (mark[:, :, 3:4].astype(np.float32) / 255.0) * WATERMARK_OPACITY
        blended = mark_bgr * alpha + roi * (1.0 - alpha)
        frame[y:y + mh, x:x + mw] = blended.astype(np.uint8)

    # --- Phase overlay helpers (swim + run) ---

    def _draw_phase_label(
        self, cv2_mod: Any, frame: Any, phase: str, width: int,
    ) -> None:
        """Large colored badge in the top-right corner showing the current phase.

        Shares the top strip with the score header, which is drawn from the
        left. On a narrow frame -- which the display crop makes routine -- the
        two want the same pixels, and this one used a fixed type size, so
        "MIDSTANCE" ended up printed through the header as "IDSTANCE". It now
        scales with the frame and drops below the header rather than into it.
        """
        color = self._phase_colors.get(phase, self._phase_colors.get("unknown", (128, 128, 128)))
        label = self._phase_labels.get(phase, "UNKNOWN")
        font = cv2_mod.FONT_HERSHEY_SIMPLEX
        scale = max(0.42, min(0.7, width / 900 * 0.7))
        thick = 2 if scale > 0.55 else 1
        (tw, th), _ = cv2_mod.getTextSize(label, font, scale, thick)
        pad = 8
        x = max(2, width - tw - pad * 2 - 10)
        # The header owns the left of the strip. When the badge would reach
        # into it, take the row underneath instead of overlapping.
        y = 10 if x > width * 0.45 else 10 + self._header_band_px(frame.shape[0])
        self._phase_badge_bottom = y + th + pad * 2
        overlay = frame.copy()
        cv2_mod.rectangle(overlay, (x, y), (x + tw + pad * 2, y + th + pad * 2), color, -1)
        cv2_mod.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2_mod.putText(
            frame, label, (x + pad, y + th + pad),
            font, scale, (255, 255, 255), thick, cv2_mod.LINE_AA,
        )

    @staticmethod
    def _header_band_px(height: int) -> int:
        """Vertical room the score header occupies, mirroring ChipLayer.header.

        Duplicated arithmetic rather than a shared constant because the header
        is staged through the chip layer and painted later -- there is nothing
        to ask at the time the badge needs to know.
        """
        return int(max(10, height * 0.018)) + 43

    def _draw_cycle_counter(
        self, cv2_mod: Any, frame: Any, cycle_num: int, width: int,
    ) -> None:
        """Small 'Stroke/Stride N of M' text below the phase label."""
        if self._total_cycles < 1:
            return
        text = f"{self._cycle_noun} {cycle_num} of {self._total_cycles}"
        font = cv2_mod.FONT_HERSHEY_SIMPLEX
        scale = max(0.32, min(0.45, width / 900 * 0.45))
        thick = 1
        (tw, _th), _ = cv2_mod.getTextSize(text, font, scale, thick)
        x = max(2, width - tw - 18 - 10)
        # Sits under whatever row the phase badge ended up on.
        y = getattr(self, "_phase_badge_bottom", 45) + 12
        cv2_mod.putText(
            frame, text, (x, y),
            font, scale, (200, 200, 200), thick, cv2_mod.LINE_AA,
        )

    def _build_phase_timeline(self, width: int) -> np.ndarray:
        """Pre-render the phase timeline bar once (called from generate)."""
        margin = 12
        bar_w = width - 2 * margin
        bar_h = 28
        bar = np.zeros((bar_h, bar_w, 3), dtype=np.uint8)
        n = len(self._phase_sequence)
        if n == 0:
            return bar
        for i, phase in enumerate(self._phase_sequence):
            x0 = int(i * bar_w / n)
            x1 = int((i + 1) * bar_w / n)
            color = self._phase_colors.get(phase, self._phase_colors.get("unknown", (128, 128, 128)))
            bar[:, x0:x1] = color
        return bar

    def _draw_phase_timeline(
        self, cv2_mod: Any, frame: Any, analyzed_idx: int, width: int, height: int,
    ) -> None:
        """Blit the cached timeline bar at the bottom + draw a current-time marker."""
        if self._timeline_cache is None:
            return
        margin = 12
        bar_h = self._timeline_cache.shape[0]
        bar_w = self._timeline_cache.shape[1]
        y0 = height - bar_h - margin
        y1 = y0 + bar_h
        x0 = margin
        x1 = x0 + bar_w
        if y0 < 0 or x1 > width:
            return
        # Alpha-blend the bar (70% bar, 30% original)
        roi = frame[y0:y1, x0:x1]
        cv2_mod.addWeighted(self._timeline_cache, 0.7, roi, 0.3, 0, roi)
        frame[y0:y1, x0:x1] = roi
        # Border
        cv2_mod.rectangle(frame, (x0, y0), (x1, y1), (60, 60, 60), 1)
        # Marker
        n = len(self._phase_sequence)
        if n > 0:
            marker_x = x0 + int(analyzed_idx * bar_w / n)
            cv2_mod.line(frame, (marker_x, y0 - 2), (marker_x, y1 + 2), (255, 255, 255), 2)

    # Legend swatch labels. For run the timeline collapses 8 phases into a
    # stance/swing read, so the legend uses coarse family labels rather than
    # the finer per-phase badge text.
    _LEGEND_LABEL_OVERRIDE: dict[str, str] = {
        "midstance": "STANCE", "pre_swing": "TOE-OFF", "mid_swing": "SWING",
    }

    def _draw_phase_legend(
        self, cv2_mod: Any, frame: Any, width: int, height: int,
    ) -> None:
        """Compact phase legend above the timeline bar. Only shown for first 2 seconds."""
        phases = self._phase_legend_order
        if not phases:
            return
        font = cv2_mod.FONT_HERSHEY_SIMPLEX
        scale = 0.35
        thick = 1
        swatch = 10
        gap = 6
        x = 14
        y = height - 28 - 12 - 22  # above the timeline bar
        for phase in phases:
            color = self._phase_colors[phase]
            label = self._LEGEND_LABEL_OVERRIDE.get(phase, self._phase_labels[phase])
            cv2_mod.rectangle(frame, (x, y), (x + swatch, y + swatch), color, -1)
            cv2_mod.putText(
                frame, label, (x + swatch + 3, y + swatch - 1),
                font, scale, (180, 180, 180), thick, cv2_mod.LINE_AA,
            )
            (tw, _), _ = cv2_mod.getTextSize(label, font, scale, thick)
            x += swatch + 3 + tw + gap

    @staticmethod
    def _even_target_dims(
        width: int, height: int, max_long: int = 1280,
    ) -> tuple[int, int]:
        """Downscale (never upscale) so the long edge is <= max_long, both even.

        Overlay clips are for phone viewing, not archival -- capping at ~720p
        keeps the download small and the ffmpeg re-encode fast. The forced-even
        dimensions also satisfy libx264 + yuv420p: odd width/height (common in
        landscape clips from editors/social apps, e.g. 1918x1078) would abort
        the encode and silently drop the overlay.
        """
        if width <= 0 or height <= 0:
            return max(2, width), max(2, height)
        scale = min(1.0, max_long / float(max(width, height)))
        w = int(round(width * scale)) & ~1   # round down to even
        h = int(round(height * scale)) & ~1
        return max(2, w), max(2, h)

    def _reencode_to_mp4(
        self, input_path: str, output_path: str, out_w: int, out_h: int,
    ) -> bool:
        """Re-encode AVI to web-safe H.264 MP4 (capped to out_w x out_h)."""
        cmd = [
            "ffmpeg",
            "-i", input_path,
            # Concrete even target dims (see _even_target_dims): caps the clip to
            # ~720p for a small, fast download and guarantees the even dimensions
            # libx264 + yuv420p needs -- odd dims would abort the encode.
            "-vf", f"scale={out_w}:{out_h}",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",  # no audio
            "-y",   # overwrite
            output_path,
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 min max
            )
            if result.returncode != 0:
                logger.error(
                    "ffmpeg re-encode failed",
                    returncode=result.returncode,
                    stderr=result.stderr[:500],
                )
                return False
            return True
        except subprocess.TimeoutExpired:
            logger.error("ffmpeg timed out after 300s")
            return False
        except Exception as e:
            logger.error("ffmpeg execution failed",err=str(e))
            return False

    @staticmethod
    def _cleanup_file(path: str) -> None:
        """Remove a file, ignoring errors."""
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            logger.debug("Temp file cleanup failed", path=path, exc_info=True)
