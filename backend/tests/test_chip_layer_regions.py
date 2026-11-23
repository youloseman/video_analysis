"""The chip layer composites only where it has ink, and the frame is the same.

Converting the whole 1080x1920 frame BGR -> RGBA -> RGB -> BGR for a few chips
was ~10 s of a 25 s overlay render. flush() now blends only the bands that
hold ink; where the layer is transparent alpha_composite leaves the base
unchanged, so the output must be byte-identical to the full-frame blend.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.services.video_analysis import overlay_style
from app.services.video_analysis.overlay_style import ChipLayer, _ink_boxes


def _full_frame_flush(layer: ChipLayer) -> np.ndarray:
    """The previous implementation, verbatim in effect."""
    img = Image.fromarray(np.ascontiguousarray(layer._frame[:, :, ::-1]))
    base = img.convert("RGBA")
    ink = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ink)
    for kind, args, _ in layer._ops:
        getattr(ChipLayer, f"_render_{kind}")(d, *args)
    out = Image.alpha_composite(base, ink).convert("RGB")
    return np.array(out)[:, :, ::-1]


def _staged(frame: np.ndarray) -> ChipLayer:
    layer = ChipLayer(frame)
    h, w = frame.shape[:2]
    layer.header((10, 10), "BIKE", "77/100", "Good", "good", frame_w=w)
    layer.metric_chip((40, h // 3), "KNEE BDC", "141°", "good")
    layer.metric_chip((w - 20, h // 3 + 4), "TRUNK ANGLE", "9°", "warn", align="right")
    layer.metric_chip((w // 2, h // 2), "HIP", "52°", "bad", scale=1.4)
    layer.caption((w - 10, h - 60), "Bottom dead centre", align="right")
    layer.brand((w - 160, h - 30), "Flapp", "getflapp.com")
    return layer


@pytest.mark.parametrize("shape", [(1920, 1080), (1436, 802), (720, 1280), (481, 271)])
def test_region_flush_is_byte_identical_to_the_full_frame_blend(shape):
    rng = np.random.default_rng(7)
    frame = rng.integers(0, 256, size=(*shape, 3), dtype=np.uint8)
    want = _full_frame_flush(_staged(frame.copy()))
    got = _staged(frame.copy()).flush()
    assert got.shape == want.shape and got.dtype == want.dtype
    assert np.array_equal(got, want)


def test_flush_leaves_the_callers_frame_alone():
    frame = np.full((400, 300, 3), 90, dtype=np.uint8)
    layer = _staged(frame)
    layer.flush()
    assert (frame == 90).all()


def test_ink_boxes_cover_every_inked_pixel_and_split_far_bands():
    a = np.zeros((300, 200), dtype=np.uint8)
    a[5:20, 10:190] = 255      # header band
    a[280:295, 150:198] = 40   # wordmark, far below
    a[22:25, 50:60] = 1        # within the gap: merges with the header
    boxes = _ink_boxes(a)
    assert len(boxes) == 2
    covered = np.zeros_like(a, dtype=bool)
    for x1, y1, x2, y2 in boxes:
        covered[y1:y2, x1:x2] = True
    assert covered[a > 0].all()
    assert _ink_boxes(np.zeros((10, 10), dtype=np.uint8)) == []


def test_module_still_exports_what_callers_use():
    assert hasattr(overlay_style, "ChipLayer")
