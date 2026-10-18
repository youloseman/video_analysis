"""Blur faces and sponsor marks on the race photos the landing shows.

The photos are of identifiable athletes in sponsor kit. The landing is an
advert, so the face and every brand mark are blurred before the photo is run
through the analyzer -- the overlay is then drawn sharp on top of the blurred
photo, and the numbers are the ones measured on what the page shows.

Regions are ellipses (cx, cy, rx, ry, angle_deg) in source pixels, placed by
hand against each photo. `--preview` draws them as outlines instead of blurring.

The pose is found on the ORIGINAL photo and the overlay is then drawn on the
blurred copy: a blurred face or chest moves MediaPipe's shoulder and head points
(measured on one photo: knee 162 deg sharp, 166 deg blurred), and the published
numbers should be the athlete's, not the blur's. Same pixels, same geometry --
only the marked regions differ.

Usage:
    python scripts/anonymize_landing_photos.py <src_dir> <out_dir> [--preview]
    python scripts/anonymize_landing_photos.py <src_dir> <out_dir> --annotate
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# Strong blur for faces: nothing of the face should survive.
FACE_SIGMA = 22
# Lighter blur for marks: enough that no word or logo can be read.
MARK_SIGMA = 13

REGIONS: dict[str, dict[str, list[tuple[int, int, int, int, int]]]] = {
    "IMG_5145": {
        "face": [(360, 232, 70, 84, 0)],
        "mark": [
            (310, 150, 42, 18, 0),     # cap, front text
            (480, 172, 22, 16, 0),     # cap, side text
            (520, 322, 30, 28, 0),     # shoulder logo
            (400, 381, 34, 14, 0),     # chest wordmark 1
            (400, 431, 36, 16, 0),     # chest wordmark 2
            (403, 478, 26, 20, 0),     # chest logo
            (593, 404, 26, 26, -60),   # sleeve tag
            (533, 520, 30, 60, -15),   # chest vertical wordmark
            (445, 890, 34, 95, 25),    # shorts wordmark
            (378, 968, 10, 10, 0),     # shorts small mark
            (346, 1310, 14, 34, 0),    # sock wordmark
            (262, 1452, 46, 22, -5),   # shoe wordmark
        ],
    },
    "IMG_5146": {
        "face": [(598, 680, 30, 38, 0)],
        "mark": [
            (590, 636, 28, 14, 0),     # cap logo
            (556, 772, 24, 50, 0),     # chest logos
            (537, 965, 22, 42, -20),   # shorts wordmark
            (560, 1013, 32, 16, -20),  # shorts small wordmark
            (490, 875, 30, 18, 0),     # race belt
            (248, 1110, 30, 50, 15),   # far shoe
            (620, 1232, 52, 26, 0),    # near shoe
        ],
        # The Instagram mute button burnt into the screenshot.
        "inpaint": [(1098, 1305, 50, 50, 0)],
    },
    "IMG_5148": {
        "face": [(432, 352, 42, 52, 0)],
        "mark": [
            (440, 305, 62, 30, -10),   # cap
            (470, 470, 40, 55, 0),     # singlet wordmark
            (500, 520, 30, 40, 0),     # singlet logos
            (540, 615, 45, 35, 0),     # belt
            (603, 495, 18, 18, 0),     # watch face
            (475, 1085, 75, 30, 0),    # near shoe
            (850, 895, 35, 65, 30),    # far shoe
        ],
    },
}


def _mask(shape: tuple[int, int], regions, feather: int) -> np.ndarray:
    m = np.zeros(shape, np.float32)
    for cx, cy, rx, ry, ang in regions:
        cv2.ellipse(m, (cx, cy), (rx, ry), ang, 0, 360, 1.0, -1)
    if feather:
        m = cv2.GaussianBlur(m, (0, 0), feather)
    return m[..., None]


def anonymize(img: np.ndarray, spec: dict) -> np.ndarray:
    out = img.astype(np.float32)
    h, w = img.shape[:2]
    if spec.get("inpaint"):
        m = (_mask((h, w), spec["inpaint"], 0)[..., 0] > 0).astype(np.uint8) * 255
        out = cv2.inpaint(out.astype(np.uint8), m, 9, cv2.INPAINT_TELEA).astype(np.float32)
    for key, sigma in (("mark", MARK_SIGMA), ("face", FACE_SIGMA)):
        if not spec.get(key):
            continue
        blurred = cv2.GaussianBlur(out, (0, 0), sigma)
        m = _mask((h, w), spec[key], 6)
        out = out * (1 - m) + blurred * m
    return np.clip(out, 0, 255).astype(np.uint8)


def preview(img: np.ndarray, spec: dict) -> np.ndarray:
    out = img.copy()
    colours = {"face": (0, 0, 255), "mark": (255, 0, 255), "inpaint": (0, 200, 255)}
    for key, regions in spec.items():
        for cx, cy, rx, ry, ang in regions:
            cv2.ellipse(out, (cx, cy), (rx, ry), ang, 0, 360, colours[key], 2)
    return out


def annotate(original_png: bytes, blurred: np.ndarray, sport: str = "run") -> dict:
    """Measure on the original, draw on the blurred copy (see module doc)."""
    backend = Path(__file__).resolve().parents[1]
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    from app.services.video_analysis.photo_analyzer import (
        analyze_from_pose,
        detect_pose_in_image,
    )

    pose = detect_pose_in_image(original_png)
    h, w = pose["image"].shape[:2]
    if blurred.shape[:2] != (h, w):
        blurred = cv2.resize(blurred, (w, h), interpolation=cv2.INTER_AREA)
    return analyze_from_pose(
        blurred, pose["world"], pose["normalized"], pose["camera_side"],
        list(pose["warnings"]), sport, None,
        silhouette_top=pose.get("silhouette_top"),
    )


def main() -> int:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    is_preview = "--preview" in sys.argv
    is_annotate = "--annotate" in sys.argv
    dst.mkdir(parents=True, exist_ok=True)
    for name, spec in REGIONS.items():
        raw = (src / f"{name}.PNG").read_bytes()
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if is_preview:
            cv2.imwrite(str(dst / f"{name}_preview.png"), preview(img, spec))
            continue
        clean = anonymize(img, spec)
        cv2.imwrite(str(dst / f"{name}.png"), clean)
        if is_annotate:
            import base64
            import json

            res = annotate(raw, clean)
            thumb = res.pop("thumbnail_base64").split(",", 1)[-1]
            (dst / f"{name}_overlay.jpg").write_bytes(base64.b64decode(thumb))
            res.pop("pose", None)
            (dst / f"{name}.json").write_text(json.dumps(res, indent=1, default=str))
            print(name, res.get("score", {}).get("overall"), res.get("angles"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
