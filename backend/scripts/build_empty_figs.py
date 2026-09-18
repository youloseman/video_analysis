"""Empty-state figures: docs/design-assets/empty-*.jpg -> static/media/empty-*.webp

The sources are Nano Banana line icons drawn to the "How to film it" spec
(docs/CLAUDE_DESIGN_BRIEF.md, Prompt 16). They come out hairline and in three
different blues, so: thicken the strokes (a MinFilter dilates ink on white),
crop to the ink with a margin, square, 512 px, lift white to alpha, and set
every blue to the token accent. Navy and the green ticks are left alone.

    python scripts/build_empty_figs.py [stroke]    # stroke: MinFilter size for the hairline set, default 13
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "docs" / "design-assets"
OUT = ROOT / "backend" / "app" / "static" / "media"
# name -> MinFilter size. Most sources came out hairline; both-sides was
# drawn at the reference stroke already and is left alone (1 = no dilation).
NAMES = {"empty-first": 13, "empty-withheld": 13, "empty-nothing-to-fix": 13, "empty-trends": 13,
         "empty-both-sides": 1}
# The step-03 tiles ("How was it filmed?"): one phone or two, a runner or a
# rider between them. Drawn at the reference stroke; kept WIDE (2.4:1) so
# they read at 48 px tall in a tile rather than shrinking into a square.
WIDE = {"film-one-run": 1, "film-both-run": 1, "film-one-bike": 1, "film-both-bike": 1}
WIDE_ASPECT, WIDE_W = 2.4, 720
ACCENT = (36 / 255, 87 / 255, 197 / 255)  # --c-blue


def unwhite(img: Image.Image) -> Image.Image:
    a = np.asarray(img.convert("RGB")).astype(np.float32) / 255
    alpha = np.clip((1 - a.min(axis=2)) * 1.15, 0, 1)  # bite into the AA halo a little
    al = alpha[..., None]
    s = np.where(al > 0.02, (a - (1 - al)) / np.maximum(al, 1e-3), 0)
    s = np.clip(s, 0, 1)
    r, g, b = s[..., 0], s[..., 1], s[..., 2]
    blue = (b > r + 0.12) & (b > g + 0.05) & (b > 0.45)
    s[blue] = ACCENT
    return Image.fromarray((np.dstack([s, alpha]) * 255 + 0.5).astype(np.uint8), "RGBA")


def _ink_box(im: Image.Image) -> tuple[int, int, int, int]:
    g = np.asarray(im.convert("L"))
    ys, xs = np.where(g < 235)
    return xs.min(), ys.min(), xs.max(), ys.max()


def build(name: str, thick: int, wide: bool = False) -> None:
    im = Image.open(SRC / f"{name}.jpg").convert("RGB")
    if thick > 1:
        im = im.filter(ImageFilter.MinFilter(thick))
    x0, y0, x1, y1 = _ink_box(im)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    if wide:
        # Crop to the ink with a margin, then pad to the common aspect so the
        # one-phone and two-phone figures sit the same size in their tiles.
        h = int((y1 - y0) * 1.16)
        w = max(int((x1 - x0) * 1.10), int(h * WIDE_ASPECT))
        h = int(w / WIDE_ASPECT)
        # The box can run past the source's edge (the one-phone figures sit
        # on the left); crop only what exists and place it -- PIL fills an
        # out-of-bounds crop with black.
        bx0, by0 = cx - w // 2, cy - h // 2
        sx0, sy0 = max(0, bx0), max(0, by0)
        sx1, sy1 = min(im.width, bx0 + w), min(im.height, by0 + h)
        canvas = Image.new("RGB", (w, h), "white")
        canvas.paste(im.crop((sx0, sy0, sx1, sy1)), (sx0 - bx0, sy0 - by0))
        out = canvas.resize((WIDE_W, int(WIDE_W / WIDE_ASPECT)), Image.LANCZOS)
    else:
        side = int(max(y1 - y0, x1 - x0) * 1.16)
        canvas = Image.new("RGB", (side, side), "white")
        canvas.paste(im.crop((cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2)), (0, 0))
        out = canvas.resize((512, 512), Image.LANCZOS)
    rgba = unwhite(out)
    rgba.save(OUT / f"{name}.webp", "WEBP", quality=92, method=6)
    print(name, rgba.size, (OUT / f"{name}.webp").stat().st_size, "bytes")


if __name__ == "__main__":
    override = int(sys.argv[1]) if len(sys.argv) > 1 else None
    for n, thick in NAMES.items():
        build(n, override if override is not None and thick > 1 else thick)
    for n, thick in WIDE.items():
        build(n, thick, wide=True)
