"""Overlay video generator -- skeleton + angle annotations on every frame.

Reads the original video, draws skeleton bones and angle labels per frame
using the pre-computed analysis data, then re-encodes to web-safe H.264 MP4
via ffmpeg.
"""

import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from app.services.video_analysis.biomechanics.base_analyzer import SportAnalyzer

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
from app.services.video_analysis.pipeline import (
    ARC_TRIPLETS,
    LEFT_SIDE_LANDMARKS,
    MIDLINE_LANDMARKS,
    MIN_OVERLAY_VISIBILITY,
    POSE_CONNECTIONS,
    RIGHT_SIDE_LANDMARKS,
    SPORT_SAMPLE_RATES,
    VideoAnalysisPipeline,
    _draw_dashed_line,
    _is_near_side_landmark,
)

logger = structlog.get_logger()


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

        # Camera side for near/far skeleton coloring
        self.camera_side = self.summary.get("camera_side") if sport_type in ("run", "bike") else None

        # Pre-build label display config (reuse pipeline logic)
        pipeline = VideoAnalysisPipeline()
        self.label_configs = pipeline._get_angle_display_config(
            sport_type, summary, None,
            cycling_position=cycling_position,
        )

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
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Output paths
        os.makedirs(self.output_dir, exist_ok=True)
        final_mp4 = os.path.join(self.output_dir, f"{self.analysis_id}.mp4")

        # With ffmpeg: write a temp AVI (XVID) then re-encode to web-safe H.264.
        # Without ffmpeg (common on dev machines): write the MP4 directly via
        # OpenCV's mp4v muxer -- plays in VLC/most players. Install ffmpeg and
        # re-run for browser-safe H.264 + faststart.
        self._use_ffmpeg = shutil.which("ffmpeg") is not None
        if self._use_ffmpeg:
            temp_avi = os.path.join(self.output_dir, f"{self.analysis_id}_temp.avi")
            writer = cv2.VideoWriter(temp_avi, cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))
            writer_target = temp_avi
        else:
            logger.warning("ffmpeg not found -- writing MP4 directly via OpenCV (mp4v)")
            temp_avi = None
            writer = cv2.VideoWriter(final_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            writer_target = final_mp4

        if not writer.isOpened():
            logger.error("Cannot create VideoWriter", path=writer_target)
            cap.release()
            return None

        # Phase overlay (swim + run): cache timeline bar once (saves ~0.5 ms
        # per frame vs rebuilding).
        if self._phase_sequence:
            self._timeline_cache = self._build_phase_timeline(width)
            self._overlay_fps = fps

        # Track nearest analyzed frame for nearest-frame-hold
        last_analyzed_idx: int | None = None
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

            # Find the nearest analyzed frame for this video frame
            analyzed_idx = self._get_nearest_analyzed_frame(video_frame_idx)
            if analyzed_idx is not None:
                last_analyzed_idx = analyzed_idx

            # Draw overlay if we have landmark data
            if last_analyzed_idx is not None:
                self._draw_frame_overlay(cv2, frame, last_analyzed_idx, width, height)

            # Brand watermark on every frame (even un-analyzed ones)
            if WATERMARK_ENABLED:
                self._draw_watermark(cv2, frame, width, height)

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
            out_w, out_h = self._even_target_dims(width, height)
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

        # Pick the most readable frame among a few central candidates.
        n = len(self.frame_data_list)
        cand = sorted({int(n * p) for p in (0.5, 0.4, 0.6, 0.35, 0.65)})
        cand = [i for i in cand if 0 <= i < n] or [n // 2]
        best_idx, best_vis = cand[0], -1.0
        for i in cand:
            lms = self.frame_data_list[i]["normalized_landmarks"]
            vis = sum(getattr(lm, "visibility", 0.5) for lm in lms) / max(1, len(lms))
            if vis > best_vis:
                best_vis, best_idx = vis, i

        try:
            cap = cv2.VideoCapture(self.video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_data_list[best_idx]["frame_idx"])
            ok, frame = cap.read()
            cap.release()
            if not ok or frame is None:
                return None
            h, w = frame.shape[:2]
            self._draw_frame_overlay(cv2, frame, best_idx, w, h)
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

    def _get_nearest_analyzed_frame(self, video_frame_idx: int) -> int | None:
        """Map a video frame index to the nearest analyzed frame index.

        For sample_rate=1 (run/bike): direct 1:1 mapping.
        For sample_rate=3 (swim): nearest-frame-hold -- use the closest
        analyzed frame at or before this video frame.
        """
        # Direct hit
        if video_frame_idx in self._frame_index_map:
            return self._frame_index_map[video_frame_idx]

        # Find nearest analyzed frame at or before this index
        best_idx = None
        best_video_idx = -1
        for fd_video_idx, analyzed_idx in self._frame_index_map.items():
            if fd_video_idx <= video_frame_idx and fd_video_idx > best_video_idx:
                best_video_idx = fd_video_idx
                best_idx = analyzed_idx

        return best_idx

    def _draw_frame_overlay(
        self, cv2_mod: Any, frame: Any, analyzed_idx: int, width: int, height: int,
    ) -> None:
        """Draw skeleton + angle labels on a single frame."""
        fd = self.frame_data_list[analyzed_idx]
        normalized_lms = fd["normalized_landmarks"]

        # Convert to pixel coordinates with visibility.
        # Gated landmarks carry NaN x/y (see landmark_stabilizer); we mark them
        # with visibility 0.0 so the existing MIN_OVERLAY_VISIBILITY filters in
        # the drawing loops below skip them without ever calling int(NaN).
        pixel_coords: list[tuple[int, int, float]] = []
        for lm in normalized_lms:
            vis = getattr(lm, "visibility", 1.0)
            if vis is None:
                vis = 1.0
            lx, ly = lm.x, lm.y
            if (
                lx is None or ly is None
                or (isinstance(lx, float) and math.isnan(lx))
                or (isinstance(ly, float) and math.isnan(ly))
            ):
                pixel_coords.append((-1, -1, 0.0))
                continue
            pixel_coords.append((int(lx * width), int(ly * height), float(vis)))

        # -- 1. Skeleton bones (near-side only for bike/run) --
        near_color = (180, 255, 180)   # light green (BGR)
        near_dot = (0, 255, 200)       # cyan-green

        # For bike, the near-side filter is unconditional even if
        # camera_side is somehow falsy (defaults to "left"). This is
        # belt-and-suspenders against cross-body "spider-web" bones
        # (11-12 shoulder pair, 23-24 hip pair) drawing when the side
        # has not been determined.
        effective_side = self.camera_side
        if self.sport_type == "bike" and not effective_side:
            effective_side = "left"
        side_filter_active = bool(effective_side)

        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx >= len(pixel_coords) or end_idx >= len(pixel_coords):
                continue
            sx, sy, sv = pixel_coords[start_idx]
            ex, ey, ev = pixel_coords[end_idx]
            if sv < MIN_OVERLAY_VISIBILITY or ev < MIN_OVERLAY_VISIBILITY:
                continue
            both_near = (
                _is_near_side_landmark(start_idx, effective_side)
                and _is_near_side_landmark(end_idx, effective_side)
            )
            # Skip far-side bones entirely for side-view sports
            if side_filter_active and not both_near:
                continue
            cv2_mod.line(frame, (sx, sy), (ex, ey), near_color, 1, cv2_mod.LINE_AA)

        for i, (px, py, vis) in enumerate(pixel_coords):
            if vis < MIN_OVERLAY_VISIBILITY:
                continue
            # Skip far-side dots for side-view sports
            if side_filter_active and not _is_near_side_landmark(i, effective_side):
                continue
            cv2_mod.circle(frame, (px, py), 2, near_dot, -1, cv2_mod.LINE_AA)

        # -- 2. Angle arcs + labels --
        if len(pixel_coords) > 25:
            # Get per-frame angle values from analyzer's angle_history
            frame_angles = self._get_frame_angles(analyzed_idx)

            # Body size reference for adaptive scaling
            s11x, s11y, _ = pixel_coords[11]
            h23x, h23y, _ = pixel_coords[23]
            body_height_px = abs(s11y - h23y)
            font_scale = max(0.35, min(0.55, body_height_px / 400))
            thickness_t = 1 if font_scale < 0.45 else 2
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

            font = cv2_mod.FONT_HERSHEY_SIMPLEX

            for cfg in self.label_configs:
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
                if opt_min <= angle_val <= opt_max:
                    color = (0, 220, 0)       # GREEN
                elif abs(angle_val - opt_min) < 15 or abs(angle_val - opt_max) < 15:
                    color = (0, 200, 255)     # ORANGE (BGR)
                else:
                    color = (0, 0, 255)       # RED (BGR)

                # Draw angle arc
                triplet = self.arc_triplets.get(cfg["key"])
                if triplet:
                    VideoAnalysisPipeline._draw_angle_arc(
                        cv2_mod, frame, pixel_coords, *triplet,
                        color, body_height_px,
                    )

                # Callout line + label
                jx, jy, _ = pixel_coords[lm_idx]
                dx, dy = offset_vectors.get(cfg["offset_dir"], (offset_px, 0))
                lx = max(5, min(width - 130, jx + dx))
                ly = max(20, min(height - 10, jy + dy))

                cv2_mod.line(frame, (jx, jy), (lx, ly), (200, 200, 200), 1, cv2_mod.LINE_AA)
                cv2_mod.circle(frame, (jx, jy), 3, color, -1, cv2_mod.LINE_AA)

                # Teaser: mask the number (skeleton + which joint stays visible,
                # the measured value is locked behind an upgrade).
                if self.hide_angle_values:
                    text = f"{cfg['name']} [locked]"
                    color = (150, 150, 150)  # gray, de-emphasized (BGR)
                else:
                    text = f"{cfg['name']} {angle_val:.0f}"
                (tw, th_t), _ = cv2_mod.getTextSize(text, font, font_scale, thickness_t)
                pad = 3
                cv2_mod.rectangle(
                    frame,
                    (lx - pad, ly - th_t - pad),
                    (lx + tw + pad, ly + pad),
                    (0, 0, 0), -1,
                )
                cv2_mod.putText(
                    frame, text, (lx, ly),
                    font, font_scale, color, thickness_t, cv2_mod.LINE_AA,
                )

        # -- 2b. Head alignment + pelvic ratio overlays (bike only) --
        if self.sport_type == "bike" and len(pixel_coords) > 25:
            near = self.summary.get("near_side", "left")
            s11x, s11y, _ = pixel_coords[11]
            h23x, h23y, _ = pixel_coords[23]
            bh_px = abs(s11y - h23y)
            small_scale = max(0.28, min(0.45, bh_px / 500))
            font_hl = cv2_mod.FONT_HERSHEY_SIMPLEX

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

                        if eav >= MIN_OVERLAY_VISIBILITY:
                            if self.hide_angle_values:
                                h_color = (150, 150, 150)
                                h_text = "Head [locked]"
                            else:
                                h_color = (0, 220, 0) if head_score >= 75 else ((0, 200, 255) if head_score >= 50 else (0, 0, 255))
                                h_text = f"Head {head_score:.0f}"
                            (htw, hth), _ = cv2_mod.getTextSize(h_text, font_hl, small_scale, 1)
                            hlx = max(5, min(width - htw - 5, eax + 10))
                            hly = max(hth + 5, min(height - 5, eay - 10))
                            cv2_mod.rectangle(frame, (hlx - 2, hly - hth - 2), (hlx + htw + 2, hly + 2), (0, 0, 0), -1)
                            cv2_mod.putText(frame, h_text, (hlx, hly), font_hl, small_scale, h_color, 1, cv2_mod.LINE_AA)

            # Pelvic ratio (use summary average)
            pelvic = self.summary.get("pelvic_ratio", 0)
            if pelvic > 0 and bh_px > 40:
                from app.services.video_analysis.biomechanics.cycling_positions import get_cycling_reference
                ref = get_cycling_reference(self.cycling_position)
                p_min, p_max = ref["pelvic_ratio"]
                if self.hide_angle_values:
                    p_color = (150, 150, 150)
                    p_text = "Pelvic [locked]"
                else:
                    if p_min <= pelvic <= p_max:
                        p_color = (0, 220, 0)
                    elif abs(pelvic - p_min) < 0.5 or abs(pelvic - p_max) < 0.5:
                        p_color = (0, 200, 255)
                    else:
                        p_color = (0, 0, 255)
                    p_text = f"Pelvic {pelvic:.1f}x"
                (ptw, pth), _ = cv2_mod.getTextSize(p_text, font_hl, small_scale, 1)
                hp_i2 = 23 if near == "left" else 24
                hpx2, hpy2, _ = pixel_coords[hp_i2]
                off_px = max(50, int(bh_px * 0.35))
                plx = max(5, min(width - ptw - 5, hpx2 + int(off_px * 0.3)))
                ply = max(pth + 5, min(height - 5, hpy2 + int(off_px * 0.6)))
                cv2_mod.rectangle(frame, (plx - 2, ply - pth - 2), (plx + ptw + 2, ply + 2), (0, 0, 0), -1)
                cv2_mod.putText(frame, p_text, (plx, ply), font_hl, small_scale, p_color, 1, cv2_mod.LINE_AA)

        # -- 3. Info badge (top-left corner) --
        self._draw_info_badge(cv2_mod, frame, width)

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

        # -- 5. Free-tier teaser watermark (burned into every rendered frame) --
        if self.teaser_watermark:
            self._draw_text_watermark(cv2_mod, frame, width, height)

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

    def _draw_info_badge(self, cv2_mod: Any, frame: Any, width: int) -> None:
        """Draw a small info badge in the top-left corner."""
        sport_labels = {"run": "RUN", "bike": "BIKE", "swim": "SWIM"}
        sport_text = sport_labels.get(self.sport_type, self.sport_type.upper())
        badge_text = f"{sport_text}  {self.technique_score}/100  {self.letter_grade}"

        font = cv2_mod.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), _ = cv2_mod.getTextSize(badge_text, font, font_scale, thickness)

        pad = 8
        x, y = 10, 10
        cv2_mod.rectangle(
            frame,
            (x, y),
            (x + tw + pad * 2, y + th + pad * 2),
            (0, 0, 0), -1,
        )
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2_mod.rectangle(
            overlay,
            (x, y),
            (x + tw + pad * 2, y + th + pad * 2),
            (0, 0, 0), -1,
        )
        cv2_mod.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2_mod.putText(
            frame, badge_text,
            (x + pad, y + th + pad),
            font, font_scale, (0, 255, 255), thickness, cv2_mod.LINE_AA,
        )

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
        """Large colored badge in the top-right corner showing the current phase."""
        color = self._phase_colors.get(phase, self._phase_colors.get("unknown", (128, 128, 128)))
        label = self._phase_labels.get(phase, "UNKNOWN")
        font = cv2_mod.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thick = 2
        (tw, th), _ = cv2_mod.getTextSize(label, font, scale, thick)
        pad = 8
        x = width - tw - pad * 2 - 10
        y = 10
        overlay = frame.copy()
        cv2_mod.rectangle(overlay, (x, y), (x + tw + pad * 2, y + th + pad * 2), color, -1)
        cv2_mod.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2_mod.putText(
            frame, label, (x + pad, y + th + pad),
            font, scale, (255, 255, 255), thick, cv2_mod.LINE_AA,
        )

    def _draw_cycle_counter(
        self, cv2_mod: Any, frame: Any, cycle_num: int, width: int,
    ) -> None:
        """Small 'Stroke/Stride N of M' text below the phase label."""
        if self._total_cycles < 1:
            return
        text = f"{self._cycle_noun} {cycle_num} of {self._total_cycles}"
        font = cv2_mod.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        thick = 1
        (tw, _th), _ = cv2_mod.getTextSize(text, font, scale, thick)
        x = width - tw - 18 - 10
        y = 55
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
