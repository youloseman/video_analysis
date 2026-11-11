"""Angle icons for the report's range table (report_pdf.angle_icon).

The rider is Artur's Nano Banana drawing (2026-11-11), recoloured grey. The
measured segments and the angle are drawn on top here, in exact geometry: the
model's own highlighted versions traced clothing outlines instead of the lines
between joints and drew a near-full circle for a 20 degree torso.

    python scripts/build_angle_icons.py            # -> app/static/media/angle-icons/
"""
import glob
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path('E:/PROJECTS/Video_analysis')
SRC = sorted(glob.glob(str(ROOT / 'docs/design-assets/angle-icons/Gemini_Generated_Image_ieg1h3*.jpg')))[0]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'backend/app/static/media/angle-icons'
CHECK = '--check' in sys.argv

# Joint points on the base drawing (2400 x 1792 px).
P = {
    "sh": (1360, 472), "el": (1504, 688), "wr": (1840, 640), "hip": (936, 680),
    "kn": (1128, 984), "an": (1016, 1392), "kn_t": (1368, 760), "an_t": (1176, 1136),
    "back": (1200, 304), "helm": (1600, 170),
}
GREY = (195, 202, 212)
BLUE = (36, 89, 194)
CORAL = (241, 85, 63)


def base() -> Image.Image:
    im = Image.open(SRC).convert('L')
    # dark line art -> grey lines on white (alpha from darkness)
    a = im.point(lambda v: max(0, min(255, int((235 - v) * 255 / 180))))
    rgba = Image.new('RGBA', im.size, GREY + (0,))
    rgba.putalpha(a)
    return rgba


def seg(d, a, b, w=36, col=BLUE):
    d.line((P[a], P[b]), fill=col + (255,), width=w)
    for p in (P[a], P[b]):
        r = w * 0.5
        d.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=col + (255,))


def dashed(d, p, length, w=18):
    x, y = p
    on = 46
    while length > 0:
        d.line((x, y, x + min(on, length), y), fill=BLUE + (255,), width=w)
        x += on * 2
        length -= on * 2


def wedge(d, v, a, b, r=150):
    """Filled sector at vertex v between the directions to points a and b,
    the smaller of the two angles."""
    ang = lambda p: math.degrees(math.atan2(p[1] - v[1], p[0] - v[0]))  # noqa: E731
    t1, t2 = ang(a), ang(b)
    diff = (t2 - t1) % 360
    small = min(diff, 360 - diff)
    if small < 30:   # a narrow angle needs a longer wedge to be seen at icon size
        r = int(r * 2.2)
    start, end = (t1, t2) if diff <= 180 else (t2, t1)
    box = (v[0] - r, v[1] - r, v[0] + r, v[1] + r)
    d.pieslice(box, start, end if end > start else end + 360, fill=CORAL + (215,))
    d.arc(box, start, end if end > start else end + 360, fill=CORAL + (255,), width=12)


def icon(name: str, segs, vertex=None, level=None):
    im = base()
    lay = Image.new('RGBA', im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    if level:
        dashed(d, P[level], 520)
    for a, b in segs:
        seg(d, a, b)
    if name == "helmet_above_back":
        dashed(d, P["back"], 560)
        x, y0, y1 = P["helm"][0], P["back"][1], P["helm"][1]
        d.line((x, y0, x, y1), fill=CORAL + (255,), width=22)
        for y in (y0, y1):
            d.line((x - 50, y, x + 50, y), fill=CORAL + (255,), width=22)
    if vertex:
        v, a, b = vertex
        va = P[v]
        wedge(d, va, P[a] if a in P else a, P[b] if b in P else b)
    im = Image.alpha_composite(im, lay)
    if CHECK:
        dc = ImageDraw.Draw(im)
        for k, (x, y) in P.items():
            dc.ellipse((x - 14, y - 14, x + 14, y + 14), outline=(255, 0, 0, 255), width=5)
            dc.text((x + 18, y - 10), k, fill=(255, 0, 0, 255))
    # crop to the drawing, keep 4:3
    box = (480, 130, 2240, 1450)
    im = im.crop(box).resize((288, 216), Image.LANCZOS)
    OUT.mkdir(parents=True, exist_ok=True)
    im.quantize(colors=48, method=Image.Quantize.FASTOCTREE).save(OUT / f'{name}.png', optimize=True)


def far(p, q, k=1.0):
    return (p[0] + (q[0] - p[0]) * k, p[1] + (q[1] - p[1]) * k)


H = lambda k: (P[k][0] + 400, P[k][1])  # noqa: E731 -- a point straight ahead
ICONS = {
    "knee_at_bdc": ([("hip", "kn"), ("kn", "an")], ("kn", "hip", "an"), None),
    "knee_at_tdc": ([("hip", "kn_t"), ("kn_t", "an_t")], ("kn_t", "hip", "an_t"), None),
    "hip_at_tdc": ([("sh", "hip"), ("hip", "kn_t")], ("hip", "sh", "kn_t"), None),
    "trunk_angle_avg": ([("hip", "sh")], ("hip", "sh", H("hip")), "hip"),
    "shoulder_angle_avg": ([("hip", "sh"), ("sh", "el")], ("sh", "hip", "el"), None),
    "elbow_angle_avg": ([("sh", "el"), ("el", "wr")], ("el", "sh", "wr"), None),
    "forearm_tilt_avg": ([("el", "wr")], ("el", "wr", H("el")), "el"),
    "pelvic_ratio": ([("sh", "hip"), ("hip", "kn")], ("hip", "sh", "kn"), None),
    "helmet_above_back": ([], None, None),
}

for name, (segs, vertex, level) in ICONS.items():
    icon(name, segs, vertex, level)
print('wrote', len(ICONS), 'icons to', OUT)
