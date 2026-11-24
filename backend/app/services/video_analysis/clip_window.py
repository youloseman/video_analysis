"""A long clip is cut to its best stretch before it is measured (2026-11-24).

Result stats on 30 days of stored analyses: a quarter of bike clips ran past
1.5x the frame budget, and every one of them was then read at every second
frame across its whole length. That keeps the count of frames and throws away
the thing the bike report is built on: at 15 fps the bottom of the pedal stroke
gets one frame a revolution or none. The guide asks for 10-15 s; asking the
athlete to trim on a phone is friction before the upload, and the budget is in
frames, so a 60 fps clip is thinned past 11 s whatever the guide said.

So the clip is cut instead: a quick look at the whole of it (a few frames a
second through the prescan's pose model), the best stretch by "somebody is in
shot, big, in the clip's usual position", and only that stretch is analysed,
at full rate. The cut replaces the clip IN PLACE, frame-accurate, before
anything reads it: the overlay, kinogram, player and the joint editor's
re-measure then all see one file and need no offset.

Opt-in per caller (``run_analysis(trim_long=True)``; the jobs pass it for the
sports in VA_CLIP_TRIM_SPORTS, bike by default):
local scripts and the golden run analyse clips nobody may overwrite.
Never raises; on any doubt the clip is left as it was and is thinned as before.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from statistics import median
from typing import Any

import structlog

logger = structlog.get_logger()

# The window stays at stride 1: the extractor's nearest stride reaches 2 at
# 1.5x the budget, and 1.25x leaves room for a frame count that runs long.
WINDOW_BUDGET_FRAC = 1.25
# ... and inside the capture guide, under capture_report.DURATION_LONG_S (20 s).
MAX_WINDOW_S = 15.0
# Only clips that thinning drops to ~15 fps. A 60 fps clip thinned is read at
# 30, which is the rate the run path is tuned for: cut to a window it would be
# read at 60, where run leg identity and contact timing break (flapp-60fps-run;
# measured again 2026-11-24 on the HD treadmill set: contact 133 -> 167 ms on
# one clip, phase stability 95 -> 82 on another). High-speed and slow-motion
# containers are left alone for the same reason.
MAX_FPS = 35.0
SCAN_MIN, SCAN_MAX = 12, 40
SCAN_LONG_EDGE = 640
EDGE_S = 1.0            # the first and last second: getting on, reaching for the phone
TRUNK_OFF_DEG = 12.0    # a sample this far from the clip's usual trunk is another position
MIN_FOUND = 3
CUT_TIMEOUT_S = 180

_VF = ("scale='min(1920,iw)':'min(1920,ih)':force_original_aspect_ratio=decrease,"
       "scale=trunc(iw/2)*2:trunc(ih/2)*2")


def window_frames(frame_count: int, fps: float, budget: int) -> int | None:
    """Frames to keep, or None when the clip is read at full rate anyway."""
    if fps <= 0 or fps > MAX_FPS or frame_count <= 0 or budget <= 0:
        return None
    if frame_count / budget + 0.5 < 2:      # the extractor's stride stays 1
        return None
    return int(min(budget * WINDOW_BUDGET_FRAC, MAX_WINDOW_S * fps))


def sample_scores(feats: list[dict[str, float] | None]) -> list[float]:
    """0 for a sample without a body; else up to 1 for a big rider in the
    clip's usual position (the median trunk of the samples that found one)."""
    trunks = [f["trunk_deg"] for f in feats if f and "trunk_deg" in f]
    usual = median(trunks) if trunks else None
    out = []
    for f in feats:
        if not f:
            out.append(0.0)
            continue
        size = min(float(f.get("height_frac") or 0.3), 0.85) / 0.85
        s = 0.5 + 0.5 * size
        if usual is not None and abs(f.get("trunk_deg", usual) - usual) > TRUNK_OFF_DEG:
            s *= 0.5
        out.append(s)
    return out


def pick_start(times: list[float], scores: list[float], duration: float, win_s: float) -> float:
    """Start of the window with the best mean score; ties go to the window
    centred on where the athlete was seen (the clip's middle on a trainer)."""
    lo, hi = 0.0, max(0.0, duration - win_s)
    if hi - lo >= 2 * EDGE_S:
        lo, hi = lo + EDGE_S, hi - EDGE_S
    total = sum(scores)
    seen = sum(t * s for t, s in zip(times, scores)) / total if total > 0 else duration / 2
    mid = min(hi, max(lo, seen - win_s / 2))
    best, best_key = lo, None
    steps = max(1, int((hi - lo) / 0.25))
    for k in range(steps + 1):
        st = lo + (hi - lo) * k / steps
        inside = [s for t, s in zip(times, scores) if st <= t <= st + win_s]
        if not inside:
            continue
        key = (round(sum(inside) / len(inside), 3), -abs(st - mid))
        if best_key is None or key > best_key:
            best, best_key = st, key
    return best


def _scan(video_path: str, fps: float, frame_count: int) -> tuple[list[float], list[dict | None], float]:
    """Pose on a few frames spread over the clip: their times in seconds, their
    features (None without a body), and the clip's duration as decoded.

    Times are the decoder's own timestamps, not index / fps: a clip trimmed in
    the iPhone's Photos app keeps an edit list, and its header then says 707
    frames at 27.9 fps for what decodes as 655 at 29.97 (IMG_4091)."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

    from app.services.prescan import _model_path, frame_features

    path = _model_path()
    if not path:
        return [], [], 0.0
    n = max(SCAN_MIN, min(SCAN_MAX, int(frame_count / fps * 2)))
    targets = sorted({min(frame_count - 1, int((i + 0.5) * frame_count / n)) for i in range(n)})
    times: list[float] = []
    feats: list[dict | None] = []
    last_s = 0.0
    opts = PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=path),
                                 running_mode=RunningMode.IMAGE, num_poses=1)
    cap = cv2.VideoCapture(video_path)
    try:
        with PoseLandmarker.create_from_options(opts) as lm:
            idx, ti = 0, 0
            # to the end of the stream: the header's frame count can run long
            while True:
                if not cap.grab():
                    break
                last_s = (cap.get(cv2.CAP_PROP_POS_MSEC) or idx * 1000 / fps) / 1000
                if ti < len(targets) and idx >= targets[ti]:
                    ok, img = cap.retrieve()
                    ti += 1
                    if ok and img is not None:
                        h, w = img.shape[:2]
                        k = SCAN_LONG_EDGE / max(h, w)
                        if k < 1:
                            img = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
                        res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                 data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                        f = (frame_features(res.pose_landmarks[0], img.shape[1], img.shape[0])
                             if res.pose_landmarks else None)
                        times.append(last_s)
                        feats.append(f)
                idx += 1
    finally:
        cap.release()
    return times, feats, last_s + 1 / fps


def _cut(src: Path, start_s: float, frames: int) -> Path | None:
    """Frame-accurate re-encode of the window to a temp file beside the clip.
    Not a stream copy: that starts on a keyframe and leaves an edit list the
    browser honours and OpenCV may not, so the player and the frame indices
    would disagree by up to a GOP."""
    tmp = src.with_name(f"{src.stem}.window.tmp")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-noautorotate", "-ss", f"{start_s:.3f}", "-i", str(src),
         "-map", "0:v:0", "-frames:v", str(frames), "-vf", _VF, "-fps_mode", "passthrough", "-an",
         "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-movflags", "+faststart",
         "-f", "mp4", str(tmp)],
        check=True, capture_output=True, timeout=CUT_TIMEOUT_S,
    )
    return tmp


def trim_long_clip(video_path: str | Path, budget: int | None = None) -> dict[str, Any] | None:
    """Cut a long clip to its best window, in place. What was kept, or None
    when the clip was left as it was."""
    import cv2

    from app.core.config import settings

    p = Path(video_path)
    tmp: Path | None = None
    try:
        if not p.is_file() or shutil.which("ffmpeg") is None:
            return None
        cap = cv2.VideoCapture(str(p))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        frame_count = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        cap.release()
        keep = window_frames(frame_count, fps, budget or settings.max_analysis_frames)
        if not keep:
            return None
        times, feats, duration = _scan(str(p), fps, frame_count)
        found = sum(1 for f in feats if f)
        if found < MIN_FOUND:
            logger.info("CLIP_TRIM_SKIPPED", reason="nobody found", found=found, samples=len(feats))
            return None
        win_s = keep / fps
        if duration <= win_s + 1.0:
            # the header ran long (an edit list): the clip is not long after all
            logger.info("CLIP_TRIM_SKIPPED", reason="short as decoded", duration_s=round(duration, 1))
            return None
        start = pick_start(times, sample_scores(feats), duration, win_s)
        tmp = _cut(p, start, keep)
        cap = cv2.VideoCapture(str(tmp))
        got = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        fps_out = float(cap.get(cv2.CAP_PROP_FPS) or fps)   # a re-encoded file's header is honest
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        if not ok or got < keep * 0.9 or got >= frame_count:
            logger.warning("CLIP_TRIM_KEPT_ORIGINAL", frames_in=frame_count, frames_out=got, want=keep)
            tmp.unlink(missing_ok=True)
            return None
        os.replace(tmp, p)
        out = {
            "from_s": round(start, 1), "to_s": round(start + got / fps_out, 1),
            "source_s": round(duration, 1), "source_frames": frame_count, "frames": got,
            "scan": {"samples": len(feats), "found": found},
        }
        logger.info("CLIP_TRIMMED", **{k: v for k, v in out.items() if k != "scan"}, found=found)
        return out
    except Exception as e:  # noqa: BLE001 -- trimming is an improvement, never a failure
        logger.warning("CLIP_TRIM_FAILED", path=str(p), err=str(e))
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
        return None


def trimmed_note(t: dict[str, Any] | None) -> str | None:
    """A few words for the result's meta line: which part of the clip was
    measured ("0:08-0:23 of a 34 s clip")."""
    if not t:
        return None

    def mmss(s: float) -> str:
        s = int(round(s))
        return f"{s // 60}:{s % 60:02d}"

    return f"{mmss(t['from_s'])}-{mmss(t['to_s'])} of a {t['source_s']:.0f} s clip"
