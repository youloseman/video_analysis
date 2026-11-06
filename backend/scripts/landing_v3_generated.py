"""Turn Artur's generated images (upload/landing) into the media of landing-v3.html.

Nothing here claims to be a photograph of a real athlete. The runners are
generated images, analysed by the app's own photo analyzer, so the overlay on
them is the real render and the numbers are measured on exactly the picture
shown; the page says so under each.

Usage (backend venv):
    python scripts/landing_v3_generated.py <repo_root> <report_screenshot.png>
"""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

ANALYSIS_WIDTH = 1180   # the overlay's labels are sized for a phone photo

COVER = "Gemini_Generated_Image_ikx8g3ikx8g3ikx8.jpg"   # text baked into its left side
PAIR = "Gemini_Generated_Image_pzaf53pzaf53pzaf.jpg"    # two frames, facing each other
ROAD = "Gemini_Generated_Image_cpcjsacpcjsacpcj (1).jpg"
PHONE = "Gemini_Generated_Image_en5loten5loten5l.jpg"   # green screen
SWIM = "swim.PNG"
TRIPOD = "Gemini_Generated_Image_fjzkwifjzkwifjzk.jpg"  # green screen, phone square to the camera


def analyse(img: Image.Image) -> tuple[Image.Image, dict]:
    from app.services.video_analysis.photo_analyzer import analyze_photo

    if img.width > ANALYSIS_WIDTH:
        img = img.resize((ANALYSIS_WIDTH, round(img.height * ANALYSIS_WIDTH / img.width)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    res = analyze_photo(buf.getvalue(), "run")
    thumb = res.pop("thumbnail_base64").split(",", 1)[-1]
    return Image.open(io.BytesIO(base64.b64decode(thumb))).convert("RGB"), res


def save(img: Image.Image, out: Path, width: int, quality: int = 80) -> None:
    if img.width > width:
        img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
    img.save(out, "WEBP", quality=quality, method=6)
    print(out.name, img.size, out.stat().st_size // 1024, "KB")


def report(name: str, res: dict) -> None:
    keep = {k: res.get(k) for k in ("angles", "warnings")}
    score = (res.get("score") or {}).get("overall_score")
    print(name, "score", score, json.dumps(keep, ensure_ascii=False, default=str)[:600])


def _screen_quad(mask: np.ndarray) -> np.ndarray:
    """The screen's true corners: a line through each straight edge, intersected.
    The rounded corners make the extreme points fall short of them."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float32)
    s, d = pts.sum(axis=1), pts[:, 0] - pts[:, 1]
    rough = np.float32([pts[s.argmin()], pts[d.argmax()], pts[s.argmax()], pts[d.argmin()]])
    lines = []
    for i in range(4):
        a, b = rough[i], rough[(i + 1) % 4]
        ab = b - a
        length = float(np.linalg.norm(ab))
        t = ((pts - a) @ ab) / (length * length)
        dist = np.abs((pts[:, 0] - a[0]) * ab[1] - (pts[:, 1] - a[1]) * ab[0]) / length
        side = pts[(t > 0.15) & (t < 0.85) & (dist < 40)]
        vx, vy, x0, y0 = cv2.fitLine(side, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((float(vx), float(vy), float(x0), float(y0)))
    corners = []
    for i in range(4):
        (vx1, vy1, x1, y1), (vx2, vy2, x2, y2) = lines[i - 1], lines[i]
        t = np.linalg.solve(np.array([[vx1, -vx2], [vy1, -vy2]]), np.array([x2 - x1, y2 - y1]))[0]
        corners.append((x1 + t * vx1, y1 + t * vy1))
    return np.float32(corners)


def phone(src: Path, screen: Path, out: Path) -> None:
    """The real report on the generated phone: warp into the green, cut the black."""
    bgr = cv2.imread(str(src))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    green = ((hsv[..., 0] > 40) & (hsv[..., 0] < 85) & (hsv[..., 1] > 120) & (hsv[..., 2] > 120)).astype(np.uint8)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(green, 8)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == big).astype(np.uint8)
    quad = _screen_quad(mask)
    ui = cv2.imread(str(screen))
    h, w = ui.shape[:2]
    m = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), quad)
    warped = cv2.warpPerspective(ui, m, (bgr.shape[1], bgr.shape[0]), flags=cv2.INTER_AREA)
    soft = cv2.GaussianBlur(cv2.dilate(mask, np.ones((5, 5), np.uint8)).astype(np.float32), (0, 0), 1.2)[..., None]
    comp = warped * soft + bgr * (1 - soft)
    # Whatever is not phone goes transparent, so it sits on any ground.
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    body = cv2.morphologyEx((grey > 14).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(body, 8)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    alpha = cv2.GaussianBlur((labels == big).astype(np.float32), (0, 0), 1.2)
    rgba = np.dstack([comp[..., ::-1], alpha * 255]).clip(0, 255).astype(np.uint8)
    x, y, bw, bh = stats[big, :4]
    img = Image.fromarray(rgba, "RGBA").crop((x, y, x + bw, y + bh))
    img = img.resize((900, round(img.height * 900 / img.width)), Image.LANCZOS)
    img.save(out, "WEBP", quality=84, method=6)
    print(out.name, img.size, out.stat().st_size // 1024, "KB")


def tripod(src: Path, poster: Path, out: Path) -> None:
    """The real clip's poster on the phone in the garage. The page plays the clip
    itself in the same rectangle; this still is what shows until it starts. Prints
    the rectangle and the notch as percentages of the image for the page's CSS."""
    bgr = cv2.imread(str(src))
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    green = ((hsv[..., 0] > 40) & (hsv[..., 0] < 85) & (hsv[..., 1] > 110) & (hsv[..., 2] > 90)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(green, 8)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh = stats[big, :4]
    mask = cv2.dilate((labels == big).astype(np.uint8), np.ones((3, 3), np.uint8))
    still = cv2.imread(str(poster))
    ph, pw = still.shape[:2]
    scale = max(bw / pw, bh / ph)
    still = cv2.resize(still, (round(pw * scale), round(ph * scale)), interpolation=cv2.INTER_AREA)
    ox, oy = (still.shape[1] - bw) // 2, (still.shape[0] - bh) // 2
    canvas = bgr.copy()
    canvas[y:y + bh, x:x + bw] = still[oy:oy + bh, ox:ox + bw]
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), .8)[..., None]
    comp = canvas * soft + bgr * (1 - soft)
    img = Image.fromarray(comp[..., ::-1].clip(0, 255).astype(np.uint8))
    img.save(out, "WEBP", quality=84, method=6)
    print(out.name, img.size, out.stat().st_size // 1024, "KB")
    print("screen %", round(x / w * 100, 2), round(y / h * 100, 2), round(bw / w * 100, 2), round(bh / h * 100, 2))
    top = mask[y:y + round(bh * .08), x + bw // 4:x + bw * 3 // 4]
    ys, xs = np.nonzero(top == 0)
    if len(xs):
        print("notch %", round((x + bw // 4 + xs.min()) / w * 100, 2), round(y / h * 100, 2),
              round((xs.max() - xs.min() + 1) / w * 100, 2), round((ys.max() + 1) / h * 100, 2))


def main() -> None:
    root, screen = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
    src = root / "upload/landing"
    out = root / "backend/app/static/media/v3"

    cover = Image.open(src / COVER).convert("RGB")
    crop = cover.crop((round(cover.width * 0.46), 0, cover.width, cover.height))
    over, res = analyse(crop)
    report("cover", res)
    # The banner and the watermark go: both end up half lost in the fades. The
    # analyzer's score on a generated runner was never the point of this picture.
    over = over.crop((0, round(over.height * 0.065), over.width, round(over.height * 0.95)))
    save(over, out / "gen-run-ov.webp", 1180)

    pair = Image.open(src / PAIR).convert("RGB")
    w, h = pair.size
    bar = w // 2
    a = pair.crop((0, 0, bar - round(w * 0.006), h))
    # The second frame faces the other way; the comparison needs one direction.
    b = pair.crop((bar + round(w * 0.006), 0, w, h)).transpose(Image.FLIP_LEFT_RIGHT)
    # Shown raw, as an illustration. The photo analyzer reads the trailing leg on
    # both and scores them 100 and 95, so its overlay cannot carry this comparison.
    save(a, out / "gen-pair-reach.webp", 1100, 80)
    save(b, out / "gen-pair-under.webp", 1100, 80)

    road = Image.open(src / ROAD).convert("RGB")
    save(road, out / "gen-road.webp", 1536, 76)

    swim = Image.open(src / SWIM).convert("RGB")
    swim = swim.crop((0, 0, swim.width - 4, swim.height - 4))   # a screenshot's hairline edge
    save(swim, out / "gen-swim.webp", 1100, 82)

    phone(src / PHONE, screen, out / "gen-phone.webp")
    tripod(src / TRIPOD, out / "tt-dark-poster.webp", out / "gen-tripod.webp")


if __name__ == "__main__":
    main()
