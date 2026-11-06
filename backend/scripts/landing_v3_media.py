"""Build the media of the scroll-driven landing (static/landing-v3.html).

Two of Scott Flathouse's Kona 2026 photographs, each as a registered pair: the
raw frame, and the same frame as the app's own renderer drew it (the study's
figure in upload/kona-2026-overlays/credited, photo_analyzer's overlay on the
hand-checked joints). The page draws the skeleton on the raw frame as the
visitor scrolls, then fades to the real overlay, so what ends up on screen is
exactly what the app prints. The figure's baked credit band is cut off; the
page prints the credit in HTML under the frame.

Usage:
    python scripts/landing_v3_media.py <repo_root>

Prints the joints as 0..1000 of the frame's width, which is the viewBox of the page's
SVG, and where the hero's chip sits.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# name: (study entry, the figure rendered from it, also write the raw frame?)
FRAMES = {
    "hero": (17, "position-04_forearm-21_torso-21.jpg", True),
    "bike": (16, "position-09_forearm-24_torso-26.jpg", False),
}
WIDTHS = (2000, 1200)


def joints(root: Path, n: int) -> dict[str, list[int]]:
    """The hand-checked joints as 0..1000 of the square frame: the SVG viewBox."""
    study = json.loads((root / "backend/scripts/field_study/kona2026.json").read_text("utf-8"))
    entry = next(e for e in study["entries"] if e["n"] == n)
    fixed = json.loads(
        (root / f"upload/field-study/kona2026/corrections/p{n:02d}.json").read_text("utf-8"))
    side = Image.open(root / entry["photo"]).width
    ox, oy, s = *entry["origin"], entry["scale"]
    # The editor's points are the crop's scaled pixels shifted by the origin
    # in FULL pixels, so the origin comes off before the scale does.
    return {k: [round((ox + (x - ox) / s) * 1000 / side), round((oy + (y - oy) / s) * 1000 / side)]
            for k, (x, y) in fixed["points_photo_px"].items()}


KEEP = 0.95   # the foot of the frame goes: the figure's watermark, half lost in any fade


def chip(raw: Image.Image, fig: Image.Image, box: tuple[float, float, float, float], out: Path) -> None:
    """Cut one chip out of the figure, for the place where the photo fades to black
    under it. Found as what the figure added to the photo inside ``box``."""
    w, h = fig.size
    x0, y0, x1, y1 = round(box[0] * w), round(box[1] * h), round(box[2] * w), round(box[3] * h)
    a = np.asarray(raw.crop((x0, y0, x1, y1)).convert("L"), dtype=np.float32)
    b = np.asarray(fig.crop((x0, y0, x1, y1)).convert("L"), dtype=np.float32)
    ys, xs = np.nonzero((b > 205) & (np.abs(a - b) > 45))
    # The solid card: rows and columns that are mostly chip ground.
    mask = np.zeros(a.shape, bool)
    mask[ys, xs] = True
    rows = np.nonzero(mask.mean(axis=1) > 0.12)[0]
    cols = np.nonzero(mask[rows.min():rows.max() + 1].mean(axis=0) > 0.3)[0]
    pad = 3
    bx = (x0 + cols.min() - pad, y0 + rows.min() - pad, x0 + cols.max() + pad + 1, y0 + rows.max() + pad + 1)
    fig.crop(bx).save(out, "WEBP", quality=88, method=6)
    print(out.name, "left%", round(bx[0] / w * 100, 2), "top%", round(bx[1] / h * 100, 2),
          "width%", round((bx[2] - bx[0]) / w * 100, 2), out.stat().st_size // 1024, "KB")


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    out = root / "backend/app/static/media/v3"
    out.mkdir(parents=True, exist_ok=True)
    study = json.loads((root / "backend/scripts/field_study/kona2026.json").read_text("utf-8"))
    chips = {"hero": (0.0, 0.05, 0.4, 0.14)}
    for name, (n, figure_name, with_raw) in FRAMES.items():
        entry = next(e for e in study["entries"] if e["n"] == n)
        figure = Image.open(root / "upload/kona-2026-overlays/credited" / figure_name).convert("RGB")
        # The figure is the whole photo, scaled, plus field._credit_band's strip.
        fig = figure.crop((0, 0, figure.width, figure.height - max(36, round(figure.width * 0.047))))
        raw = Image.open(root / entry["photo"]).convert("RGB").resize(fig.size, Image.LANCZOS)
        h = round(fig.height * KEEP)
        fig, raw = fig.crop((0, 0, fig.width, h)), raw.crop((0, 0, fig.width, h))
        for width in WIDTHS:
            size = (width, round(width * KEEP))
            target = out / f"kona-{name}-ov-{width}.webp"
            fig.resize(size, Image.LANCZOS).save(target, quality=80, method=6)
            print(target.name, size, target.stat().st_size // 1024, "KB")
            if with_raw:
                target = out / f"kona-{name}-raw-{width}.webp"
                raw.resize(size, Image.LANCZOS).save(target, quality=76, method=6)
                print(target.name, size, target.stat().st_size // 1024, "KB")
        if name in chips:
            chip(raw, fig, chips[name], out / f"kona-{name}-chip.webp")
        print(name, json.dumps(joints(root, n)))


if __name__ == "__main__":
    main()
