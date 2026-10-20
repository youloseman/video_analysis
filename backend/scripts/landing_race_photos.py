"""Build the landing page's race-photo media from the edited photographs.

The three race photos on the landing are not published as taken. Each was
edited (Gemini image editing, 2026-10-18) to replace the athlete's face with a
fictional one and to remove every sponsor mark, race number and tattoo; one
also got a different background. The pose was checked against the original:
joints moved under 1.1 % of body height on two photos and under 2.5 % on the
third, which is within what the pose model itself moves when the same image is
shifted three pixels or re-saved as JPEG (angles up to ~6 deg on a single
still). The page says the photos are edited.

The analysis runs on the EDITED image -- the numbers printed on the overlay
are measured on exactly the picture the visitor sees.

Usage:
    python scripts/landing_race_photos.py <edited_dir> <media_dir>

<edited_dir> holds IMG_5145_edit.*, IMG_5146_edit.*, IMG_5148_edit.* (any
image format); <media_dir> is app/static/media.
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

from PIL import Image

# The originals were 1178-1179px wide.
SOURCE_WIDTH = 1180

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def _find(src: Path, stem: str) -> Path:
    hits = sorted(p for p in src.iterdir() if p.stem == stem)
    if not hits:
        raise SystemExit(f"missing {stem}.* in {src}")
    return hits[0]


def _source(path: Path) -> Image.Image:
    """The edit at the original photo's width. The image editor returns ~1920px;
    the overlay's labels are sized for a phone photo, and at 1920 they print at
    two thirds of the size they have on every other image on the page."""
    img = Image.open(path).convert("RGB")
    if img.width > SOURCE_WIDTH:
        img = img.resize((SOURCE_WIDTH, round(img.height * SOURCE_WIDTH / img.width)), Image.LANCZOS)
    return img


def _overlay(path: Path) -> tuple[Image.Image, dict]:
    from app.services.video_analysis.photo_analyzer import analyze_photo

    buf = io.BytesIO()
    _source(path).save(buf, "PNG")
    res = analyze_photo(buf.getvalue(), "run")
    thumb = res.pop("thumbnail_base64").split(",", 1)[-1]
    img = Image.open(io.BytesIO(base64.b64decode(thumb))).convert("RGB")
    return img, res


def _save(img: Image.Image, out: Path, width: int, quality: int = 80) -> None:
    if img.width > width:
        img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
    img.save(out, "WEBP", quality=quality, method=6)
    print(out.name, img.size)


def _crop_frac(img: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    w, h = img.size
    return img.crop((round(box[0] * w), round(box[1] * h), round(box[2] * w), round(box[3] * h)))


def main() -> int:
    src, media = Path(sys.argv[1]), Path(sys.argv[2])

    # How it works: the photo that went in, and what came back.
    p = _find(src, "IMG_5148_edit")
    over, res = _overlay(p)
    _save(_source(p), media / "race-frame-raw.webp", 820)
    _save(over, media / "race-frame-overlay.webp", 820)
    print("  5148", res["score"]["overall_score"], res["angles"])

    # Running card: 4:3-ish crop around the runner, labels included.
    p = _find(src, "IMG_5146_edit")
    over, res = _overlay(p)
    _save(_crop_frac(over, (0.068, 0.43, 0.848, 0.955)), media / "race-run-card.webp", 800)
    print("  5146", res["score"]["overall_score"], res["angles"])

    # Closing band.
    p = _find(src, "IMG_5145_edit")
    over, res = _overlay(p)
    _save(over, media / "race-run-overlay.webp", 900)
    print("  5145", res["score"]["overall_score"], res["angles"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
