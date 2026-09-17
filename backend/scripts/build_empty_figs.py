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


def build(name: str, thick: int) -> None:
    im = Image.open(SRC / f"{name}.jpg").convert("RGB")
    if thick > 1:
        im = im.filter(ImageFilter.MinFilter(thick))
    g = np.asarray(im.convert("L"))
    ys, xs = np.where(g < 235)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    side = int(max(y1 - y0, x1 - x0) * 1.16)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    sq = Image.new("RGB", (side, side), "white")
    sq.paste(im.crop((cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2)), (0, 0))
    rgba = unwhite(sq.resize((512, 512), Image.LANCZOS))
    rgba.save(OUT / f"{name}.webp", "WEBP", quality=92, method=6)
    print(name, (OUT / f"{name}.webp").stat().st_size, "bytes")


if __name__ == "__main__":
    override = int(sys.argv[1]) if len(sys.argv) > 1 else None
    for n, thick in NAMES.items():
        build(n, override if override is not None and thick > 1 else thick)
