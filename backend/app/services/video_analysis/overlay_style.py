"""Shared visual style for the analysis overlay (photo keyframe + video).

One place for the "Aerodynamic profile" look: neon skeleton with a soft glow,
dark rounded label chips with a TTF value, leader lines, and the header bar.

Both renderers (``video_visualizer._draw_frame_overlay`` and
``photo_analyzer._generate_photo_thumbnail``) call into here so the two paths
cannot drift apart visually.

Conventions
-----------
* OpenCV frames are **BGR** numpy arrays; PIL works in **RGB**. Helpers take and
  return BGR frames and convert internally, so callers never deal with it.
* Colors in this module are declared **RGB** (design-friendly) and converted with
  :func:`_bgr` where OpenCV needs them.
* Text uses a bundled TTF (``assets/fonts``) -- the deploy image
  (python:3.11-slim) ships no system fonts, so a bundled file is the only way
  PIL text renders in production.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# --- assets ---------------------------------------------------------------
_FONT_DIR = Path(__file__).resolve().parent.parent.parent / "assets" / "fonts"
FONT_REGULAR = _FONT_DIR / "DejaVuSans.ttf"
FONT_BOLD = _FONT_DIR / "DejaVuSans-Bold.ttf"

# --- palette (RGB) --------------------------------------------------------
NEON = (61, 255, 110)          # skeleton / in-range value
NEON_DIM = (30, 190, 80)       # glow underlay
AMBER = (255, 168, 46)         # warning value
ROSE = (255, 91, 77)           # out-of-range value
CHIP_BG = (26, 30, 36)         # dark chip fill
CHIP_EDGE = (86, 94, 105)      # chip hairline
INK = (232, 238, 245)          # chip label text
INK_SOFT = (150, 160, 172)     # secondary text
LEADER = (196, 204, 214)       # leader line

STATUS_COLORS = {"good": NEON, "warn": AMBER, "bad": ROSE, "muted": INK_SOFT}


def _bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """RGB -> BGR for OpenCV calls."""
    return (rgb[2], rgb[1], rgb[0])


def status_for(
    value: float, opt_min: float, opt_max: float, *, min_margin: float = 3.0,
) -> str:
    """Zone for a measured value: good | warn | bad.

    Warning margin = max(10% of the optimal width, ``min_margin``) -- the same
    rule the backend classifier and the web UI use, so the overlay colour agrees
    with the score and the coach notes.

    ``min_margin`` defaults to 3.0, which suits degrees. Pass a smaller floor for
    non-degree metrics (e.g. a pelvic ratio of 2.0-4.0, where a 3.0 floor would
    swallow the whole scale and call everything "warn").
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "muted"
    margin = max((opt_max - opt_min) * 0.10, min_margin)
    if opt_min <= value <= opt_max:
        return "good"
    if (opt_min - margin) <= value <= (opt_max + margin):
        return "warn"
    return "bad"


@lru_cache(maxsize=32)
def _font(bold: bool, size: int) -> Any:
    """Load (and cache) a bundled TTF at a given size.

    Falls back to PIL's bitmap default if the asset is missing so a packaging
    slip degrades the look instead of crashing the render.
    """
    from PIL import ImageFont

    path = FONT_BOLD if bold else FONT_REGULAR
    try:
        return ImageFont.truetype(str(path), size)
    except Exception as e:  # noqa: BLE001
        logger.warning("OVERLAY_FONT_FALLBACK", path=str(path), err=str(e))
        return ImageFont.load_default()


def text_size(text: str, bold: bool, size: int) -> tuple[int, int]:
    """Pixel (w, h) of ``text`` in the bundled font."""
    f = _font(bold, size)
    box = f.getbbox(text)
    return (box[2] - box[0], box[3] - box[1])


# --- skeleton -------------------------------------------------------------

def draw_glow_skeleton(
    cv2_mod: Any,
    frame: Any,
    segments: list[tuple[tuple[int, int], tuple[int, int]]],
    dots: list[tuple[int, int]],
    *,
    glow: bool = True,
    line_w: int = 2,
    dot_r: int = 4,
) -> None:
    """Draw a neon skeleton with an optional soft glow, in place.

    The glow is a blurred copy of the bones screened back over the frame -- one
    blur for the whole skeleton, not per bone, so it stays affordable. Pass
    ``glow=False`` on the per-frame video path if the cost is not worth it.
    """
    if not segments and not dots:
        return

    if glow:
        layer = np.zeros_like(frame)
        for (a, b) in segments:
            cv2_mod.line(layer, a, b, _bgr(NEON_DIM), line_w + 5, cv2_mod.LINE_AA)
        for p in dots:
            cv2_mod.circle(layer, p, dot_r + 3, _bgr(NEON_DIM), -1, cv2_mod.LINE_AA)
        layer = cv2_mod.GaussianBlur(layer, (0, 0), sigmaX=6, sigmaY=6)
        # screen-ish blend: keep the brighter of frame/glow so it never darkens
        cv2_mod.max(frame, layer, dst=frame)

    for (a, b) in segments:
        cv2_mod.line(frame, a, b, _bgr(NEON), line_w, cv2_mod.LINE_AA)
    for p in dots:
        cv2_mod.circle(frame, p, dot_r, _bgr(NEON), -1, cv2_mod.LINE_AA)
        cv2_mod.circle(frame, p, dot_r, _bgr((250, 255, 250)), 1, cv2_mod.LINE_AA)


def draw_leader(
    cv2_mod: Any, frame: Any,
    joint: tuple[int, int], anchor: tuple[int, int],
    color_rgb: tuple[int, int, int],
) -> None:
    """Thin leader line from a label chip to its joint, with an anchor ring."""
    cv2_mod.line(frame, joint, anchor, _bgr(LEADER), 1, cv2_mod.LINE_AA)
    cv2_mod.circle(frame, joint, 5, _bgr(color_rgb), 1, cv2_mod.LINE_AA)
    cv2_mod.circle(frame, joint, 2, _bgr(color_rgb), -1, cv2_mod.LINE_AA)


# --- PIL chip layer -------------------------------------------------------

class ChipLayer:
    """Batches every chip/text draw into ONE PIL round-trip per frame.

    Converting BGR->PIL->BGR is the expensive part, so callers stage all chips
    then :meth:`flush` once. Usage::

        layer = ChipLayer(frame)
        layer.metric_chip((x, y), "TRUNK ANGLE", "9", "good")
        layer.header((10, 10), "BIKE", "77/100", "Good", "good")
        frame = layer.flush()
    """

    def __init__(self, frame: Any):
        self._frame = frame
        self._ops: list[tuple[str, tuple, dict]] = []
        self._taken: list[tuple[int, int, int, int]] = []   # placed chip rects

    # -- staging ---------------------------------------------------------
    @staticmethod
    def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 3) -> bool:
        return not (
            a[2] + pad <= b[0] or a[0] >= b[2] + pad
            or a[3] + pad <= b[1] or a[1] >= b[3] + pad
        )

    def _place(self, rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """Nudge a chip vertically until it clears every chip already staged.

        Tries alternating up/down offsets so a crowded joint (e.g. head vs trunk)
        doesn't end up with one chip painted over another's value.
        """
        h = rect[3] - rect[1]
        step = h + 4
        for k in range(0, 7):
            for sign in ((0,) if k == 0 else (-1, 1)):
                cand = (rect[0], rect[1] + sign * step * k,
                        rect[2], rect[3] + sign * step * k)
                if cand[1] < 2:
                    continue
                if not any(self._overlaps(cand, t) for t in self._taken):
                    self._taken.append(cand)
                    return cand
        self._taken.append(rect)
        return rect

    def metric_chip(
        self, anchor: tuple[int, int], label: str, value: str, status: str,
        *, scale: float = 1.0, align: str = "left",
    ) -> tuple[int, int, int, int]:
        """Stage a dark rounded chip: LABEL + big coloured value.

        ``anchor`` is the chip's left-middle (or right-middle when
        ``align='right'``). The chip is nudged vertically if it would collide
        with one already staged. Returns the final rect (x1, y1, x2, y2).
        """
        lab_s = max(9, int(13 * scale))
        val_s = max(13, int(22 * scale))
        pad_x, pad_y, gap = int(10 * scale), int(7 * scale), int(8 * scale)

        lw, lh = text_size(label, False, lab_s)
        vw, vh = text_size(value, True, val_s)
        w = pad_x * 2 + lw + gap + vw
        h = pad_y * 2 + max(lh, vh)

        x = anchor[0] - w if align == "right" else anchor[0]
        y = anchor[1] - h // 2
        rect = self._place((x, y, x + w, y + h))
        self._ops.append(("chip", (rect, label, value, status, lab_s, val_s, pad_x, gap), {}))
        return rect

    def header(
        self, at: tuple[int, int], sport: str, score: str, grade: str, status: str,
        *, right_text: str | None = None, frame_w: int = 0, scale: float = 1.0,
    ) -> None:
        """Stage the top header: 'BIKE: 77/100 · Good' (+ optional right title).

        Stage this BEFORE the metric chips: it reserves its band across the top
        of the frame so no chip is placed underneath it.
        """
        h = int(21 * scale) + int(9 * scale) * 2 + 4
        band = (0, at[1] - 2, frame_w or at[0] + 400, at[1] + h + 2)
        self._taken.append(band)   # keep chips out of the header strip
        self._ops.append(("header", (at, sport, score, grade, status, right_text, frame_w, scale), {}))

    def brand(self, at: tuple[int, int], main: str, sub: str, *, scale: float = 1.0) -> None:
        """Stage the bottom-right wordmark."""
        self._ops.append(("brand", (at, main, sub, scale), {}))

    # -- render ----------------------------------------------------------
    def flush(self) -> Any:
        """Render every staged op and return the new BGR frame."""
        if not self._ops:
            return self._frame

        from PIL import Image, ImageDraw

        img = Image.fromarray(self._frame[:, :, ::-1])  # BGR -> RGB
        base = img.convert("RGBA")
        layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)

        for kind, args, _ in self._ops:
            if kind == "chip":
                self._render_chip(d, *args)
            elif kind == "header":
                self._render_header(d, *args)
            elif kind == "brand":
                self._render_brand(d, *args)

        out = Image.alpha_composite(base, layer).convert("RGB")
        return np.array(out)[:, :, ::-1]  # RGB -> BGR

    # -- painters --------------------------------------------------------
    @staticmethod
    def _render_chip(d, rect, label, value, status, lab_s, val_s, pad_x, gap) -> None:
        x1, y1, x2, y2 = rect
        radius = max(6, (y2 - y1) // 3)
        d.rounded_rectangle(rect, radius=radius, fill=CHIP_BG + (225,), outline=CHIP_EDGE + (140,), width=1)

        lab_f, val_f = _font(False, lab_s), _font(True, val_s)
        cy = (y1 + y2) // 2
        lx = x1 + pad_x
        d.text((lx, cy), label, font=lab_f, fill=INK + (255,), anchor="lm")
        lw = lab_f.getbbox(label)[2] - lab_f.getbbox(label)[0]
        d.text((lx + lw + gap, cy), value, font=val_f,
               fill=STATUS_COLORS.get(status, INK) + (255,), anchor="lm")

    @staticmethod
    def _render_header(d, at, sport, score, grade, status, right_text, frame_w, scale) -> None:
        s_lab = max(13, int(21 * scale))
        s_val = max(13, int(21 * scale))
        pad_x, pad_y = int(14 * scale), int(9 * scale)

        lab = f"{sport}: "
        tail = f" · {grade}" if grade else ""
        lab_f, val_f = _font(True, s_lab), _font(True, s_val)
        lw = lab_f.getbbox(lab)[2] - lab_f.getbbox(lab)[0]
        vw = val_f.getbbox(score)[2] - val_f.getbbox(score)[0]
        tw = lab_f.getbbox(tail)[2] - lab_f.getbbox(tail)[0] if tail else 0
        h = pad_y * 2 + s_val + 4
        w = pad_x * 2 + lw + vw + tw

        x, y = at
        d.rounded_rectangle((x, y, x + w, y + h), radius=max(8, h // 3),
                            fill=CHIP_BG + (225,), outline=CHIP_EDGE + (140,), width=1)
        cy = y + h // 2
        d.text((x + pad_x, cy), lab, font=lab_f, fill=INK + (255,), anchor="lm")
        d.text((x + pad_x + lw, cy), score, font=val_f,
               fill=STATUS_COLORS.get(status, NEON) + (255,), anchor="lm")
        if tail:
            d.text((x + pad_x + lw + vw, cy), tail, font=lab_f,
                   fill=STATUS_COLORS.get(status, NEON) + (255,), anchor="lm")

        if right_text and frame_w:
            rf = _font(True, max(11, int(17 * scale)))
            d.text((frame_w - int(18 * scale), cy), right_text, font=rf,
                   fill=INK_SOFT + (215,), anchor="rm")

    @staticmethod
    def _render_brand(d, at, main, sub, scale) -> None:
        mf = _font(True, max(12, int(19 * scale)))
        sf = _font(False, max(8, int(10 * scale)))
        x, y = at
        d.text((x, y), main, font=mf, fill=(255, 255, 255, 225), anchor="rs")
        if sub:
            d.text((x, y + int(12 * scale)), sub, font=sf, fill=INK_SOFT + (190,), anchor="rs")


__all__ = [
    "NEON", "AMBER", "ROSE", "STATUS_COLORS",
    "status_for", "text_size", "draw_glow_skeleton", "draw_leader", "ChipLayer",
    "FONT_REGULAR", "FONT_BOLD",
]
