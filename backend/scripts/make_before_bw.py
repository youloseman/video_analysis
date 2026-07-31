"""Desaturate the photo in the landing page's "before" frame while keeping the
analysis overlay in colour.

    python scripts/make_before_bw.py [source.png]

The overlay Flapp burns in (skeleton lines, angle values) is drawn in flat,
vivid colour; the photograph behind it — sky, road, field, kit, skin — never
reaches the same saturation at the same brightness. So a soft mask on
saturation x value separates the two without any hand-drawn regions: the photo
goes greyscale, the red 35 degrees and 1.4x that flag the rider's problems stay
red. A plain CSS filter cannot do this, because the labels are burnt into the
photo and greyscale would take them with it.

The source is the full-size render straight out of Flapp; it lives outside the
repo (see .gitignore: compare/) because it is a multi-MB original. Only the
compressed result is version-controlled. A preview at full resolution is written
next to the source for eyeballing the mask.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else _ROOT / "compare" / "flapp-analysis (5).png"
DST = _ROOT / "backend" / "app" / "static" / "media" / "before.webp"
PREVIEW = SRC.with_name(SRC.stem + "-bw-preview.png")

# Saturation / value ramps: below the low end a pixel is treated as photograph,
# above the high end as overlay, and in between it cross-fades — which is what
# keeps anti-aliased glyph edges from turning into jagged colour fringes.
S0, S1 = 0.54, 0.70
V0, V1 = 0.45, 0.60
CROP_TOP = 0.085          # drops the score band Flapp bakes into the render
OUT_WIDTH = 1600


def ramp(x: np.ndarray, a: float, b: float) -> np.ndarray:
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)                       # smoothstep


def main() -> None:
    im = Image.open(SRC).convert("RGB")
    w, h = im.size
    im = im.crop((0, int(h * CROP_TOP), w, h))
    rgb = np.asarray(im).astype(np.float32) / 255.0

    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)

    alpha = (ramp(sat, S0, S1) * ramp(mx, V0, V1))[..., None]
    grey = (rgb * np.array([0.299, 0.587, 0.114], dtype=np.float32)).sum(axis=2, keepdims=True)
    out = grey * (1 - alpha) + rgb * alpha

    print(f"pixels kept in colour: {float(alpha.mean()) * 100:.2f}%")

    res = Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))
    res.save(PREVIEW)
    # int(), not round(): "after.webp" was scaled the same way, and the slider
    # overlays the two frames pixel-for-pixel (CSS aspect-ratio 1600/974).
    res.resize((OUT_WIDTH, int(res.height * OUT_WIDTH / res.width)), Image.LANCZOS).save(
        DST, "WEBP", quality=86, method=6,
    )
    print(f"wrote {DST} ({round(os.path.getsize(DST) / 1024)} KB)")


if __name__ == "__main__":
    main()
