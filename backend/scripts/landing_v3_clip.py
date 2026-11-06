"""Put the product's own overlay video on black, for static/landing-v3.html.

Input is the overlay the analysis really returned for upload/IMG_4149.MOV,
rendered at 1920 long edge instead of the app's 1280 (score, angles and skeleton
are burnt in by the pipeline and are not touched). Only the room goes. The camera
is on a tripod, so the bike is a static mask taken from the median frame; the
rider comes from the pose model's person mask; the chips are found by tint.

Usage (the backend venv has MediaPipe):
    python scripts/landing_v3_clip.py <overlay.mp4> <out_dir>

Writes tt-dark.mp4, tt-dark-poster.webp and knee-bottom.webp. All geometry below
is in the 720x1280 frame the clip was first measured on and scales with the input.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

MODEL = Path(__file__).resolve().parents[1] / "models" / "pose_landmarker_heavy.task"
ROOM_LEVEL = 0.03       # what is left of the room
BIKE_TOP = 462          # above this line only the rider is kept: dispensers live there
SKIRTING = (905, 990)   # the dark skirting board behind the wheel
WHEEL = (175, 840, 197, 158)   # centre x, y, outer radius, inner radius of the tyre
TRAINER = (540, 975, 720, 1095)
CRANK = (440, 800, 640, 985)
FADE = (1010, 1110)     # the floor goes to black between these rows


def _bike_mask(median: np.ndarray, f: float) -> np.ndarray:
    """What of the median frame is bike: dark or cool, in the right place."""
    h, w = median.shape[:2]
    hsv = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)
    v, hue = hsv[..., 2], hsv[..., 0]
    keep = ((v < 95) | ((v < 125) & (hue >= 40) & (hue <= 125))).astype(np.uint8)
    keep[:int(BIKE_TOP * f)] = 0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.hypot(xx - WHEEL[0] * f, yy - WHEEL[1] * f)
    tyre = r <= WHEEL[2] * f
    inner = r < WHEEL[3] * f
    # Inside the wheel only thin things survive: the spokes, not the skirting
    # board and the floor seen through them.
    size = max(5, int(9 * f)) | 1
    thin = keep - cv2.morphologyEx(keep, cv2.MORPH_OPEN, np.ones((size, size), np.uint8))
    hub = r < 26 * f
    keep = np.where(inner & ~hub, thin, keep)
    # The skirting board and the floor: gone, but for the tyre, the crank and
    # the trainer standing on them.
    x0, y0, x1, y1 = [int(c * f) for c in TRAINER]
    trainer = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
    floor = yy >= SKIRTING[0] * f
    keep[floor & ~tyre & ~trainer] = 0
    # The wood and the skirting board inside the trainer's box: warm, not black.
    warm = (hue >= 5) & (hue <= 32) & (hsv[..., 1] >= 40) & (v >= 24)
    keep[trainer & warm] = 0
    # Specks of wood grain and wall.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(keep, 8)
    small = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] < 60 * f * f)
    keep[np.isin(labels, small)] = 0
    return keep.astype(np.float32)


def _moving(bgr: np.ndarray, median: np.ndarray, f: float) -> np.ndarray:
    """The crank and the chain turn: dark, and not what the median holds."""
    v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 2]
    diff = cv2.absdiff(bgr, median).max(axis=2)
    out = np.zeros(v.shape, np.float32)
    x0, y0, x1, y1 = [int(c * f) for c in CRANK]
    out[y0:y1, x0:x1] = ((v[y0:y1, x0:x1] < 70) & (diff[y0:y1, x0:x1] > 22)).astype(np.float32)
    return out


def _not_wall(bgr: np.ndarray, f: float) -> np.ndarray:
    """The person mask runs wide around the legs. Below the saddle, pale and
    unsaturated is the wall behind them; the white shoes are kept by the fade
    of this rule to the foot's height."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    wall = (hsv[..., 1] < 62) & (hsv[..., 2] > 105)
    wall[:int(BIKE_TOP * f)] = False
    wall[int(860 * f):] = False
    return (~wall).astype(np.float32)


TINTS = ((213, 235, 217), (247, 234, 200), (246, 217, 214))   # overlay_style's chip grounds, RGB


def _chips(bgr: np.ndarray, f: float) -> np.ndarray:
    """The pipeline's chips, found by their exact tint and their shape."""
    out = np.zeros(bgr.shape[:2], np.float32)
    k = max(5, int(9 * f)) | 1
    for rgb in TINTS:
        near = (np.abs(bgr.astype(np.int16) - np.array(rgb[::-1], np.int16)).max(axis=2) < 24).astype(np.uint8)
        near = cv2.morphologyEx(near, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(near, 8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if 40 * f <= w <= 240 * f and 16 * f <= h <= 52 * f and area / (w * h) > 0.6:
                cv2.rectangle(out, (x - 2, y - 2), (x + w + 2, y + h + 2), 1.0, -1)
    return out


def _green(bgr: np.ndarray, f: float) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    g = (hsv[..., 0] > 42) & (hsv[..., 0] < 85) & (hsv[..., 1] > 110) & (hsv[..., 2] > 130)
    k = max(3, int(5 * f)) | 1
    return cv2.dilate(g.astype(np.uint8), np.ones((k, k), np.uint8)).astype(np.float32)


def _plates(h: int, w: int) -> np.ndarray:
    """Score, title and watermark: fixed on every frame of this render."""
    out = np.zeros((h, w), np.float32)
    for x0, y0, x1, y1 in ((.031, .018, .271, .044), (.72, .022, .992, .039), (.876, .968, .982, .988)):
        out[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = 1.0
    return out


def _knee(lm) -> float:
    a, b, c = lm[23], lm[25], lm[27]  # the near leg of this clip, facing left
    ba, bc = np.array([a.x - b.x, a.y - b.y]), np.array([c.x - b.x, c.y - b.y])
    return float(np.degrees(np.arccos(np.clip(ba @ bc / (np.linalg.norm(ba) * np.linalg.norm(bc)), -1, 1))))


def main() -> None:
    src, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f = w / 720.0
    samples = []
    for i in np.linspace(0, total - 1, 31).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            samples.append(frame)
    median = np.median(np.stack(samples), axis=0).astype(np.uint8)
    bike = _bike_mask(median, f)
    plates = _plates(h, w)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fade = np.ones((h, 1), np.float32)
    y0, y1 = int(FADE[0] * f), int(FADE[1] * f)
    fade[y0:y1, 0] = np.linspace(1, 0, y1 - y0)
    fade[y1:] = 0

    pose = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)),
        running_mode=RunningMode.VIDEO, output_segmentation_masks=True))
    writer = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{w}x{h}", "-r", "30", "-i", "-", "-an", "-c:v", "libx264", "-crf", "21",
         "-preset", "slow", "-profile:v", "high", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         str(out_dir / "tt-dark.mp4")], stdin=subprocess.PIPE)
    prev = None
    i = 0
    best = (-1.0, None)
    poster = None
    step = 2 if round(fps) >= 50 else 1
    kernel = np.ones((3, 3), np.uint8)
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        i += 1
        if (i - 1) % step:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        r = pose.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int((i - 1) * 1000 / fps))
        person = r.segmentation_masks[0].numpy_view().squeeze() if r.segmentation_masks else np.zeros((h, w), np.float32)
        person = cv2.erode((person > 0.5).astype(np.uint8), kernel).astype(np.float32)
        person = person * _not_wall(bgr, f)
        body = np.maximum.reduce([person, bike, _moving(bgr, median, f)]) * fade[:, 0][:, None]
        printed = np.maximum.reduce([_chips(bgr, f), _green(bgr, f), plates])
        keep = cv2.GaussianBlur(np.maximum(body, printed), (0, 0), 1.2 * f)
        if prev is not None:
            keep = 0.7 * keep + 0.3 * prev   # no flicker at the edges
        prev = keep
        m = keep[..., None]
        lifted = 255.0 * (bgr.astype(np.float32) / 255.0) ** 0.82  # the tyres, on black
        out = np.clip(lifted * m + bgr.astype(np.float32) * (1 - m) * ROOM_LEVEL, 0, 255).astype(np.uint8)
        if r.pose_landmarks:
            k = _knee(r.pose_landmarks[0])
            if k > best[0]:
                best = (k, out.copy())
        if poster is None and i > 60:
            poster = out.copy()
        writer.stdin.write(out.tobytes())
    writer.stdin.close()
    writer.wait()
    cv2.imwrite(str(out_dir / "tt-dark-poster.webp"), poster, [cv2.IMWRITE_WEBP_QUALITY, 80])
    # The bottom of the stroke, cropped to where the beat looks: hip to foot.
    still = best[1][int(360 * f):int(1120 * f)]
    cv2.imwrite(str(out_dir / "knee-bottom.webp"), still, [cv2.IMWRITE_WEBP_QUALITY, 84])
    print("size", w, h, "best knee", round(best[0], 1), "frames", i)


if __name__ == "__main__":
    main()
