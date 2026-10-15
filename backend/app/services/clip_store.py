"""Stored clips at 1080p once their analysis is done.

Clips are kept for weeks (services/retention.py) for an Expert Review, Coach
Notes and the joint editor, and they were kept as uploaded: a 10-second 4K
iPhone clip is 60 MB. On 2026-10-15 the 4.8 GB volume was 65 % full and
filling ~0.7 GB a day. Re-encoded to 1080p H.264 (CRF 23) the same clips
measured 58 -> 1.1 MB, 62 -> 1.6 MB, 7 -> 0.9 MB -- enough to watch and to
redraw on, a fortieth of the room. Artur chose 1080p over 720p for how it
looks when a coach watches it.

Nothing the analysis stored may stop lining up with the new file:
- every frame is kept (passthrough timing), so a stored frame index still
  names the same instant -- checked on two iPhone clips, frame for frame;
- the rotation flag is carried over rather than baked in (-noautorotate),
  so OpenCV turns the new file exactly as it turned the original, and
  normalized landmarks still land on the joints;
- the frame count is compared after encoding, and the original is kept on
  any doubt. Audio is dropped: nothing reads it.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import structlog

logger = structlog.get_logger()

# Below this a clip is already small; re-encoding buys little and costs a run.
MIN_BYTES = 8 * 2**20
LONG_EDGE = 1920
ENCODE_TIMEOUT_S = 600

_VF = (f"scale='min({LONG_EDGE},iw)':'min({LONG_EDGE},ih)':force_original_aspect_ratio=decrease,"
       "scale=trunc(iw/2)*2:trunc(ih/2)*2")


def _frame_count(path: Path) -> int:
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()


def compress_clip(path: str | Path) -> Path | None:
    """Re-encode a stored clip to 1080p H.264 in place; the new path, or None
    when it was left as it was. Never raises."""
    p = Path(path)
    tmp = p.with_name(f"{p.stem}.compressing.mp4")
    try:
        if not p.is_file() or p.stat().st_size < MIN_BYTES or shutil.which("ffmpeg") is None:
            return None
        before = p.stat().st_size
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-noautorotate", "-i", str(p),
             "-map", "0:v:0", "-vf", _VF, "-fps_mode", "passthrough", "-an",
             "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
             "-movflags", "+faststart", str(tmp)],
            check=True, capture_output=True, timeout=ENCODE_TIMEOUT_S,
        )
        after = tmp.stat().st_size
        frames_in, frames_out = _frame_count(p), _frame_count(tmp)
        if frames_in <= 0 or frames_in != frames_out or after >= before:
            logger.warning("CLIP_COMPRESS_KEPT_ORIGINAL", path=str(p), frames_in=frames_in,
                           frames_out=frames_out, before=before, after=after)
            tmp.unlink(missing_ok=True)
            return None
        final = p.with_suffix(".mp4")
        os.replace(tmp, final)
        if final != p:
            p.unlink(missing_ok=True)
        logger.info("CLIP_COMPRESSED", path=str(final), mb_before=round(before / 2**20, 1),
                    mb_after=round(after / 2**20, 1), frames=frames_out)
        return final
    except Exception as e:  # noqa: BLE001 -- a stored clip must never break a finished job
        logger.warning("CLIP_COMPRESS_FAILED", path=str(p), err=str(e))
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return None
