"""Flapp bike metric icons, v4: Artur's Nano Banana pictogram rider (nb/sheet-v1.jpg)
redrawn as clean vector on real road-bike proportions, one rider for all nine
metrics. The measured joint gets the logo dot plus an arc that spans the actual
angle; dot, arc and helmet take the row's status colour through --fi-dot.

    python rider_icons.py   -> v4/bike-icons.svg (sprite) + v4/sheet.html
"""
from __future__ import annotations

import math
from pathlib import Path

P = tuple[float, float]
HERE = Path(__file__).parent


def f(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


def path(pts, cls) -> str:
    return f'<path class="{cls}" d="M' + " L".join(f"{f(x)} {f(y)}" for x, y in pts) + '"/>'


def circ(c: P, r: float, cls: str) -> str:
    return f'<circle class="{cls}" cx="{f(c[0])}" cy="{f(c[1])}" r="{f(r)}"/>'


def add(a: P, b: P, k=1.0) -> P:
    return (a[0] + k * b[0], a[1] + k * b[1])


def polar(c: P, r: float, deg: float) -> P:
    t = math.radians(deg)
    return (c[0] + r * math.cos(t), c[1] + r * math.sin(t))


def joint(a: P, b: P, l1: float, l2: float, bend: int) -> P:
    """Middle joint of a two-bone limb a->b; bend=+1 puts it left of a->b (knee forward)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    d = min(math.hypot(dx, dy), l1 + l2 - 1e-6)
    x = (l1 * l1 - l2 * l2 + d * d) / (2 * d)
    h = math.sqrt(max(0.0, l1 * l1 - x * x))
    ux, uy = dx / d, dy / d
    return (a[0] + ux * x + bend * uy * h, a[1] + uy * x - bend * ux * h)


def arc(v: P, a: P, b: P, r: float, cls="fa") -> str:
    """Arc around v spanning the smaller angle between v->a and v->b."""
    t1 = math.atan2(a[1] - v[1], a[0] - v[0])
    t2 = math.atan2(b[1] - v[1], b[0] - v[0])
    d = (t2 - t1 + math.pi) % (2 * math.pi) - math.pi
    s = (v[0] + r * math.cos(t1), v[1] + r * math.sin(t1))
    e = (v[0] + r * math.cos(t1 + d), v[1] + r * math.sin(t1 + d))
    return (f'<path class="{cls}" d="M{f(s[0])} {f(s[1])} A{f(r)} {f(r)} 0 0 '
            f'{1 if d > 0 else 0} {f(e[0])} {f(e[1])}"/>')


def dash(a: P, b: P) -> str:
    return path((a, b), "fr")


def dot(p: P, r=8.2) -> str:
    return circ(p, r, "fj")


# ---- geometry: a road bike in a 96 box, 1 unit ~ 2.1 cm (700c wheel = 32 units) ----
RW, FW, BB = (25.0, 72.0), (65.0, 72.0), (42.0, 74.0)
WHEEL, CRANK = 13.5, 8.0
SEAT_TOP, HEAD_TOP, HEAD_BOT = (34.0, 47.0), (60.0, 48.0), (61.5, 55.0)
SADDLE = ((27.5, 43.8), (36.0, 43.8))
HOOD = (63.5, 45.0)
HIP = (31.0, 38.0)
TRUNK_DEG, TRUNK_LEN = 34.0, 26.0
SH = polar(HIP, TRUNK_LEN, -TRUNK_DEG)
HEAD, HEAD_R = (SH[0] + 10.8, SH[1] - 9.4), 7.0
THIGH = SHIN = 23.6
UPPER = FORE = 15.0
EL = joint(SH, HOOD, UPPER, FORE, -1)

BIKE = "".join([
    circ(RW, WHEEL, "fw"), circ(FW, WHEEL, "fw"),
    path((RW, BB, SEAT_TOP, RW), "ff"),                      # chainstay, seat tube, seat stay
    path((SEAT_TOP, HEAD_TOP), "ff"), path((BB, HEAD_BOT), "ff"),
    path((HEAD_TOP, HEAD_BOT, FW), "ff"),                    # head tube + fork
    path((HEAD_TOP, HOOD), "ff"),                            # stem; the hand hides the bar
    path((SEAT_TOP, (32, 43.8)), "ff"), path(SADDLE, "ff fs"),
])


def blob(*parts) -> str:
    """Filled union of tapered capsules: parts are ((x, y), r) pairs chained a->b->c."""
    out = []
    for (a, ra), (b, rb) in zip(parts, parts[1:]):
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        th = math.atan2(b[1] - a[1], b[0] - a[0])
        al = math.acos(max(-1.0, min(1.0, (ra - rb) / d)))
        q = [polar(a, ra, math.degrees(th + al)), polar(b, rb, math.degrees(th + al)),
             polar(b, rb, math.degrees(th - al)), polar(a, ra, math.degrees(th - al))]
        out.append('<path d="M' + " L".join(f"{f(x)} {f(y)}" for x, y in q) + 'Z"/>')
    out += [f'<circle cx="{f(c[0])}" cy="{f(c[1])}" r="{f(r)}"/>' for c, r in parts]
    return "".join(out)


def rider(theta: float = 0.0, helmet: bool = False) -> tuple[str, dict]:
    """theta: near crank angle, 0 = forward (3 o'clock), 90 = bottom, -90 = top."""
    pn, pf = polar(BB, CRANK, theta), polar(BB, CRANK, theta + 180)
    kn, kf = joint(HIP, pn, THIGH, SHIN, 1), joint(HIP, pf, THIGH, SHIN, 1)
    # Same build as the runner: tapered limbs, a back that rounds over the hip,
    # a fist on the bar, a shoe on the pedal.
    up = (-math.sin(math.radians(TRUNK_DEG)), -math.cos(math.radians(TRUNK_DEG)))
    back = add(polar(SH, 7.0, 180 - TRUNK_DEG), up, 1.4)            # the back rounds over
    foot = lambda pd: blob((pd, 2.6), (add(pd, (5.6, 0.8)), 1.9))  # noqa: E731
    far = blob((HIP, 4.4), (kf, 3.4), (pf, 2.5)) + foot(pf)
    body = (blob((HIP, 5.6), (back, 6.6)) + circ(add(SH, up, 0.4), 5.4, "")
            + blob((HIP, 4.8), (kn, 3.6), (pn, 2.6)) + foot(pn)
            + blob((SH, 3.3), (EL, 2.8), (HOOD, 2.5)) + circ(HOOD, 3.1, "") + circ(HEAD, HEAD_R, ""))
    s = ('<g class="fbk">' + BIKE + path((BB, pn), "ff") + "</g>"
         + f'<g class="fkf">{far}</g><g class="fbody fo">{far}</g><g class="fbody">{body}</g>')
    if helmet:
        hx, hy = HEAD
        s += (f'<path class="fhm" d="M{f(hx - 8.6)} {f(hy + 1.4)} A8.6 8.6 0 0 1 {f(hx + 8.6)} {f(hy + 1.4)}'
              f' L{f(hx + 9.6)} {f(hy + 2.6)} L{f(hx + 4)} {f(hy + 2.2)} L{f(hx - 6.5)} {f(hy + 3)} L{f(hx - 12)} {f(hy + 5.2)} Z"/>')
    return s, {"hip": HIP, "sh": SH, "el": EL, "ha": HOOD, "kn": kn, "pd": pn, "head": HEAD}


ICONS: dict[str, tuple[str, str]] = {}


PARTS: dict[str, tuple[str, str]] = {}   # key -> (the figure, the marks drawn over it)


def icon(key, label, theta=0.0, marks=lambda p: "", helmet=False):
    s, p = rider(theta, helmet)
    ICONS[key] = (label, s + marks(p))
    PARTS[key] = (s, marks(p))


AR = 15.5
icon("knee_bdc", "Knee · bottom", 90, lambda p: arc(p["kn"], p["hip"], p["pd"], AR) + dot(p["kn"]))
icon("knee_tdc", "Knee · top", -90, lambda p: arc(p["kn"], p["hip"], p["pd"], AR) + dot(p["kn"]))
icon("hip_tdc", "Hip · top of stroke", -90, lambda p: arc(p["hip"], p["sh"], p["kn"], AR) + dot(p["hip"]))
icon("trunk", "Trunk angle", 0, lambda p: dash(p["hip"], (p["hip"][0] + 30, p["hip"][1]))
     + arc(p["hip"], p["sh"], (60, p["hip"][1]), 20) + dot(p["hip"]))
icon("shoulder", "Shoulder angle", 0, lambda p: arc(p["sh"], p["hip"], p["el"], AR) + dot(p["sh"]))
icon("elbow", "Elbow angle", 0, lambda p: arc(p["el"], p["sh"], p["ha"], 13) + dot(p["el"]))
icon("forearm", "Forearm tilt", 0, lambda p: dash(p["el"], (p["el"][0] + 24, p["el"][1]))
     + arc(p["el"], p["ha"], (90, p["el"][1]), 16) + dot(p["el"]))
icon("helmet", "Helmet above back", 0, helmet=True)
icon("hip_crank", "Hip over crank", 0, lambda p: dash(BB, (BB[0], 26))
     + path(((p["hip"][0] + 4, p["hip"][1]), (BB[0], p["hip"][1])), "fm") + dot(p["hip"]))

# ---- the runner: same body, same strokes; side view at touchdown ----------------
G = 86.0                                   # ground line
R_HIP, LEAN, R_TRUNK = (41.0, 43.0), 11.0, 24.0
R_THIGH = R_SHIN = 19.5
R_UPPER, R_FORE = 12.5, 11.5


def arm(sh: P, swing: float, fore_up: float) -> tuple[P, P]:
    """Upper arm `swing` deg from straight down (+ = forward), forearm bent 90 deg upward."""
    el = polar(sh, R_UPPER, 90 - swing)
    return el, polar(el, R_FORE, 90 - swing - fore_up)


# a trainer in foot space: u forward along the sole from the heel's ground point, v up
# M/L/Q commands; a low-cut trainer: rounded toe box, sloping instep, soft heel counter
SHOE = (("M", (1.0, 0.0)), ("L", (8.8, 0.0)), ("Q", (11.8, 0.0), (11.4, 1.7)),
        ("Q", (11.0, 3.0), (8.4, 3.5)), ("Q", (5.6, 4.0), (4.4, 5.4)), ("L", (3.9, 7.3)),
        ("L", (0.5, 7.4)), ("Q", (-1.2, 3.4), (1.0, 0.0)))
ANKLE_UV = (2.1, 6.2)
GT = G - 1.7                               # top of the ground line


def shoe(heel: P, pitch: float, cls: str = "") -> tuple[str, P, P]:
    """Shoe whose heel touches `heel`, sole tilted `pitch` deg (toe up > 0). Returns svg, ankle, toe."""
    c, s_ = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    k = 1.15
    to = lambda u, v: (heel[0] + k * (u * c - v * s_), heel[1] - k * (u * s_ + v * c))  # noqa: E731
    d = " ".join(cmd + " ".join(f"{f(x)} {f(y)}" for x, y in (to(*q) for q in qs)) for cmd, *qs in SHOE) + "Z"
    attr = f' class="{cls}"' if cls else ""
    return f'<path{attr} d="{d}" stroke-width="1.6"/>', to(*ANKLE_UV), to(9.8, 0)


def runner(heel: P = (44.5, GT), pitch: float = 0.0, lean: float = LEAN, shoe_cls: str = "") -> tuple[str, dict]:
    sh = polar(R_HIP, R_TRUNK, -90 + lean)
    head = (sh[0] + 4.4, sh[1] - 11.4)
    near_shoe, ank, toe = shoe(heel, pitch, shoe_cls)
    kn = joint(R_HIP, ank, R_THIGH, R_SHIN, 1)
    fk = (31.5, 59.5)                                         # rear knee: the leg kicked up behind
    ne, nh = arm(sh, 40, 95)                                  # near arm forward: reads clear of the trunk
    fe, fh = arm(sh, -45, 100)                                # far arm back, pale
    far_shoe, fa2, _ = shoe((15.8, 47.0), -125)
    far = (blob((R_HIP, 4.4), (fk, 3.4), (fa2, 2.5)) + far_shoe
           + blob((sh, 3.2), (fe, 2.7), (fh, 2.4)) + circ(fh, 3.0, ""))
    chest = polar(polar(sh, 6.0, 90 + lean), 1.6, lean)       # chest ahead of the spine line
    body = (blob((R_HIP, 5.2), (chest, 6.6)) + circ(polar(sh, 1.5, 90 + lean), 5.6, "")
            + blob((R_HIP, 4.8), (kn, 3.6), (ank, 2.6)) + near_shoe
            + blob((sh, 3.3), (ne, 2.8), (nh, 2.5)) + circ(nh, 3.1, "") + circ(head, HEAD_R, ""))
    s = ('<g class="fbk">' + path(((8, G), (88, G)), "ff fg") + "</g>"
         + f'<g class="fkf">{far}</g><g class="fbody fo">{far}</g><g class="fbody">{body}</g>')
    return s, {"hip": R_HIP, "sh": sh, "kn": kn, "an": ank, "toe": toe, "heel": heel, "head": head}


def run_icon(key, label, marks, **kw):
    s, p = runner(**kw)
    ICONS[key] = (label, s + marks(p))
    PARTS[key] = (s, marks(p))


run_icon("cadence", "Cadence", lambda p: "".join(
    path(((x0, y), (x0 + ln, y)), "fm") for x0, y, ln in ((10, 30, 13), (6, 40, 17), (11, 50, 12))))
run_icon("overstride", "Overstride", lambda p: dash(p["hip"], (p["hip"][0], G - 3))
         + path(((p["hip"][0], G + 0.2), (p["heel"][0], G + 0.2)), "fm fbar"),
         heel=(58.5, GT), pitch=20)
run_icon("footstrike", "Foot strike", lambda p: dash(p["heel"], (p["heel"][0] + 18, p["heel"][1]))
         + arc(p["heel"], p["toe"], (90, p["heel"][1]), 19),
         heel=(46.0, GT), pitch=22, shoe_cls="fshoe")
run_icon("lean", "Trunk lean", lambda p: dash(p["hip"], (p["hip"][0], 6))
         + arc(p["hip"], (p["hip"][0], 0), p["sh"], 21) + dot(p["hip"]), lean=19.0)
run_icon("vosc", "Vertical oscillation", lambda p: path(((30, 4), (30, 22)), "fm")
         + path(((26, 8), (30, 4), (34, 8)), "fm") + path(((26, 18), (30, 22), (34, 18)), "fm"))
run_icon("contact", "Ground contact", lambda p: path(((p["heel"][0] - 1.5, G + 0.2), (p["toe"][0] + 1.5, G + 0.2)), "fm fbar"))
run_icon("shin", "Shin at touchdown", lambda p: dash(p["an"], (p["an"][0], 50))
         + arc(p["an"], (p["an"][0], 0), p["kn"], 16) + dot(p["an"], 6.6), heel=(49.0, GT))
run_icon("knee_run", "Knee · contact", lambda p: arc(p["kn"], p["hip"], p["an"], AR) + dot(p["kn"]))

CSS = """
.fi{width:40px;height:40px;flex:none;fill:none;stroke-linecap:round;stroke-linejoin:round;color:var(--c-head,var(--c-ink))}
.fbk{opacity:.3}
.fw{stroke:currentColor;stroke-width:3.6}
.ff{stroke:currentColor;stroke-width:2.4}
.fs{stroke-width:3.2}
.fg{stroke-width:3.4}
.fbar{stroke-width:6.5}
.fbody{fill:currentColor;stroke:currentColor;stroke-width:0}
.fkf{fill:var(--c-bg,#fff);stroke:var(--c-bg,#fff);stroke-width:0}
.fshoe{fill:var(--fi-dot,var(--c-coral));stroke:var(--fi-dot,var(--c-coral))}
.ft{stroke:currentColor;stroke-width:12.5}
.fl{stroke:currentColor;stroke-width:6.8}
.fa2{stroke-width:6}
.fk{stroke:var(--c-bg,#fff)}
.fo{opacity:.35}
.fh{fill:currentColor;stroke:none}
.fr{stroke:currentColor;stroke-width:2.8;stroke-dasharray:.1 4.6;opacity:.75}
.fa{stroke:var(--fi-dot,var(--c-coral));stroke-width:4.4}
.fm{stroke:var(--fi-dot,var(--c-coral));stroke-width:4.4}
.fj{fill:var(--fi-dot,var(--c-coral));stroke:var(--c-bg,#fff);stroke-width:2.8}
.fhm{fill:var(--fi-dot,var(--c-coral));stroke:var(--c-bg,#fff);stroke-width:2;paint-order:stroke}
"""


# crop windows (cx, cy, size) for row sizes: the measured joint fills the icon
ZOOM = {
    "knee_bdc": (42, 56, 64), "knee_tdc": (42, 52, 64), "hip_tdc": (42, 46, 64),
    "trunk": (46, 32, 60), "shoulder": (48, 30, 60), "elbow": (50, 32, 60),
    "forearm": (52, 34, 60), "helmet": (48, 30, 60), "hip_crank": (42, 52, 64),
}


def symbols(zoom: bool = False) -> str:
    out = []
    for k, (_, body) in ICONS.items():
        if zoom:
            if k not in ZOOM:
                continue
            cx, cy, w = ZOOM[k]
            out.append(f'<symbol id="fz-{k}" viewBox="{f(cx - w / 2)} {f(cy - w / 2)} {f(w)} {f(w)}">{body}</symbol>')
        else:
            out.append(f'<symbol id="fi-{k}" viewBox="2 0.5 87.5 87.5">{body}</symbol>')
    return "\n".join(out)


if __name__ == "__main__":
    out = HERE / "v4"
    out.mkdir(exist_ok=True)
    (out / "bike-icons.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" style="display:none"><style>{CSS}</style>{symbols()}{symbols(True)}</svg>\n',
        encoding="utf-8")
    print("ok", len(ICONS))


# ---- app export: the two plain figures for the Analyze tiles, classes prefixed ----
def _pg_classes(svg: str) -> str:
    import re
    return re.sub(r'class="([^"]*)"', lambda m: 'class="' + " ".join("pg-" + c for c in m.group(1).split()) + '"', svg)


def _tenth(svg: str) -> str:
    import re
    return re.sub(r"-?\d+\.\d+", lambda m: f"{float(m.group(0)):.1f}".rstrip("0").rstrip("."), svg)


def app_sprite() -> str:
    """The SPA's symbols: the two plain figures (pic-*), each distinct pose once
    (pmb-*), and every metric (pm-*) as its pose plus its own marks."""
    vb = 'viewBox="2 0.5 87.5 87.5"'
    bases: dict[str, str] = {}
    out = []
    for k, (fig, marks) in PARTS.items():
        bid = bases.setdefault(fig, f"pmb-{len(bases) + 1}")
        out.append(f'  <symbol id="pm-{k}" {vb}><use href="#{bid}"/>{_pg_classes(marks)}</symbol>')
    head = [f'  <symbol id="{bid}" {vb}>{_pg_classes(fig)}</symbol>' for fig, bid in bases.items()]
    plain = [f'  <symbol id="pic-bike" {vb}><use href="#{bases[PARTS["trunk"][0]]}"/></symbol>',
             f'  <symbol id="pic-run" {vb}><use href="#{bases[PARTS["cadence"][0]]}"/></symbol>']
    return _tenth("\n".join(o.replace(' class=""', "") for o in head + plain + out) + "\n")


def app_css() -> str:
    """Every class the symbols use, prefixed; the status colour is --pm-dot."""
    import re
    css = re.sub(r"\.([a-z][a-z0-9]*)", lambda m: ".pic" if m.group(1) == "fi" else ".pg-" + m.group(1), CSS.strip())
    css = css.replace("--fi-dot", "--pm-dot").replace(",#fff)", ")")
    used = set(re.findall(r"pg-[a-z0-9]+", app_sprite()))
    return "\n".join(ln for ln in css.splitlines() if ln.split("{")[0].lstrip(".") in used)
