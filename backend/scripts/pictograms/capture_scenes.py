"""Capture guide scenes for the Analyze screen: six "do / don't" pairs drawn
in the house style. Each scene is inline SVG (200 x 120): the left half is the
way to film it, the right half the common mistake, each with a status badge.
The athlete is the sprite's own figure (<use href="#pic-run|#pic-bike">), one
per sport, and the [data-sport] rule on .capguide hides the other.

    python capture_scenes.py  -> prints the <ol class="cg-body"> markup
"""
from __future__ import annotations

W, H, HALF = 200, 120, 104      # the right panel starts at x=104


def fig(x: float, y: float, s: float, extra: str = "") -> str:
    """Both sport figures, square side s, top-left at (x, y)."""
    return "".join(f'<use href="#pic-{sp}" class="fig-{sp}{extra}" x="{x:g}" y="{y:g}" width="{s:g}" height="{s:g}"/>'
                   for sp in ("run", "bike"))


def figs(run: tuple, bike: tuple, extra: str = "") -> str:
    """Each sport's figure at its own place and size (x, y, side)."""
    return "".join(f'<use href="#pic-{sp}" class="fig-{sp}{extra}" x="{x:g}" y="{y:g}" width="{s:g}" height="{s:g}"/>'
                   for sp, (x, y, s) in (("run", run), ("bike", bike)))


def badge(ox: float, ok: bool) -> str:
    cx, cy = ox + 86, 11
    mark = (f'M{cx - 4:g} {cy:g} L{cx - 1:g} {cy + 3:g} L{cx + 4.5:g} {cy - 3.5:g}' if ok
            else f'M{cx - 3.5:g} {cy - 3.5:g} L{cx + 3.5:g} {cy + 3.5:g} M{cx + 3.5:g} {cy - 3.5:g} L{cx - 3.5:g} {cy + 3.5:g}')
    return (f'<circle class="{"cgs-ok" if ok else "cgs-no"}" cx="{cx:g}" cy="{cy:g}" r="9"/>'
            f'<path class="cgs-mk" d="{mark}"/>')


def phone(x: float, y: float, w: float, h: float, inside: str = "") -> str:
    """A phone; `inside` is drawn in the phone's own coordinates and clipped to
    its screen (a nested <svg> is a viewport, so nothing spills past the glass)."""
    return (f'<rect class="cgs-ph" x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="7"/>'
            f'<rect class="cgs-scr" x="{x + 3:g}" y="{y + 3:g}" width="{w - 6:g}" height="{h - 6:g}" rx="4"/>'
            f'<svg x="{x + 3:g}" y="{y + 3:g}" width="{w - 6:g}" height="{h - 6:g}" viewBox="{x + 3:g} {y + 3:g} {w - 6:g} {h - 6:g}">{inside}</svg>')


def camera(cx: float, cy: float, facing: float = 1) -> str:
    """A camera body centred at (cx, cy), lens toward +x (facing 1) or -x."""
    return (f'<rect class="cgs-cam" x="{cx - 8:g}" y="{cy - 5.5:g}" width="16" height="11" rx="2.5"/>'
            f'<rect class="cgs-cam" x="{cx - 4:g}" y="{cy - 8:g}" width="6" height="3" rx="1"/>'
            f'<circle class="cgs-lens" cx="{cx + 2 * facing:g}" cy="{cy:g}" r="3"/>')


def tripod(cx: float, cy: float, ground: float) -> str:
    return (f'<path class="cgs-gr" d="M{cx:g} {cy + 5:g} L{cx - 7:g} {ground:g} M{cx:g} {cy + 5:g} L{cx + 7:g} {ground:g}'
            f' M{cx:g} {cy + 5:g} L{cx:g} {ground:g}"/>')


def ground(ox: float, y: float = 106) -> str:
    return f'<path class="cgs-gr" d="M{ox + 4:g} {y:g} L{ox + 92:g} {y:g}"/>'


def dash(a, b) -> str:
    return f'<path class="cgs-d" d="M{a[0]:g} {a[1]:g} L{b[0]:g} {b[1]:g}"/>'


def svg(left: str, right: str) -> str:
    return (f'<svg class="cgs" viewBox="0 0 {W} {H}" aria-hidden="true">'
            f'<g>{left}{badge(0, True)}</g><path class="cgs-sep" d="M100 8 L100 112"/><g>{right}{badge(HALF, False)}</g></svg>')


R = HALF
def sun(cx: float, cy: float, r: float) -> str:
    """The sun: a disc with short spikes round it."""
    import math
    ticks = " ".join(f"M{cx + (r + 3) * math.cos(a):.1f} {cy + (r + 3) * math.sin(a):.1f} "
                     f"L{cx + (r + 6) * math.cos(a):.1f} {cy + (r + 6) * math.sin(a):.1f}"
                     for a in (i * math.pi / 4 for i in range(8)))
    return f'<path class="cgs-ray" d="{ticks}"/><circle class="cgs-sun" cx="{cx:g}" cy="{cy:g}" r="{r:g}"/>'


def rays(src: tuple, dst: tuple, near: float, length: float, spread: float = 9) -> str:
    """Three parallel beams from the sun toward `dst`, each with an arrowhead:
    which way the light travels is the whole point of the picture."""
    import math
    dx, dy = dst[0] - src[0], dst[1] - src[1]
    n = math.hypot(dx, dy)
    ux, uy = dx / n, dy / n
    px, py = -uy, ux
    out = []
    for k in (-spread, 0, spread):
        sx, sy = src[0] + ux * near + px * k, src[1] + uy * near + py * k
        ex, ey = sx + ux * length, sy + uy * length
        head = []
        for t in (0.5, -0.5):
            c, s_ = math.cos(math.pi - t), math.sin(math.pi - t)
            hx, hy = ux * c - uy * s_, ux * s_ + uy * c
            head.append(f"M{ex:.1f} {ey:.1f} L{ex + hx * 5:.1f} {ey + hy * 5:.1f}")
        out.append(f"M{sx:.1f} {sy:.1f} L{ex:.1f} {ey:.1f} " + " ".join(head))
    return f'<path class="cgs-ray" d="{" ".join(out)}"/>'


SCENES = [
    # 1. Fill the frame: the athlete two thirds of the screen, not a speck
    ("Fill the frame",
     phone(27, 14, 42, 92, figs((20, 31, 56), (29, 42, 38))),
     phone(R + 27, 14, 42, 92, figs((R + 41, 70, 14), (R + 42, 74, 12)))),
    # 2. Portrait, not landscape
    ("Hold the phone upright",
     phone(27, 14, 42, 92, figs((20, 31, 56), (29, 42, 38))),
     phone(R + 3, 39, 90, 42, figs((R + 31, 43, 34), (R + 31, 43, 34)))),
    # 3. Side on: the camera square to the line the athlete runs along (from above)
    ("Film from the side",
     '<path class="cgs-gr" d="M6 36 L88 36 M82 31 L88 36 L82 41"/>' + '<circle class="cgs-head" cx="48" cy="36" r="6"/>'
     + dash((48, 44), (48, 82)) + camera(48, 90, 1),
     f'<path class="cgs-gr" d="M{R + 6} 36 L{R + 88} 36 M{R + 82} 31 L{R + 88} 36 L{R + 82} 41"/>'
     + f'<circle class="cgs-head" cx="{R + 40}" cy="36" r="6"/>' + dash((R + 46, 42), (R + 78, 80)) + camera(R + 82, 88, -1)),
    # 4. The camera at hip height, not looking down
    ("Camera at hip height",
     ground(0) + fig(30, 42, 64) + tripod(14, 70, 106) + camera(14, 70) + dash((24, 70), (58, 70)),
     ground(R) + fig(R + 30, 42, 64) + tripod(R + 14, 22, 106) + camera(R + 14, 22) + dash((R + 24, 26), (R + 56, 66))),
    # 5. Stand back and zoom in: a narrow view from 6-8 m, not a wide lens up close
    ("Stand back, zoom in",
     ground(0) + fig(54, 62, 44) + camera(10, 84) + dash((18, 82), (56, 62)) + dash((18, 86), (56, 106))
     + '<text class="cgs-t" x="32" y="118" text-anchor="middle">6–8 m</text>',
     ground(R) + fig(R + 54, 62, 44) + camera(R + 40, 84) + dash((R + 48, 80), (R + 94, 30)) + dash((R + 48, 88), (R + 94, 118))
     + f'<text class="cgs-t" x="{R + 36}" y="118" text-anchor="middle">1 m</text>'),
    # 6. Light behind the camera, not behind the athlete: beams show which way
    #    the light goes -- onto the athlete, or past them into the lens
    ("Light behind the camera",
     ground(0) + sun(12, 20, 7) + rays((12, 20), (66, 76), 13, 24) + fig(40, 50, 56) + camera(16, 86),
     ground(R) + f'<circle class="cgs-glow" cx="{R + 74}" cy="40" r="22"/>' + sun(R + 74, 40, 9)
     + fig(R + 30, 50, 56, " cgs-sil") + rays((R + 74, 40), (R + 12, 86), 16, 44) + camera(R + 12, 86)),
]


def markup() -> str:
    items = []
    for title, left, right in SCENES:
        items.append(f'          <li class="cg-step">\n            <div class="cg-figs"><span class="cg-frame cg-scene">'
                     f'{svg(left, right)}</span></div>\n            <p><b>{title}</b></p>\n          </li>')
    return '        <ol class="cg-body">\n' + "\n".join(items) + "\n        </ol>"


CSS = """/* The scenes (docs/design-assets/pictograms/capture_scenes.py): each pair
   is the way to film it and the usual mistake, the badge says which. */
.cg-frame.cg-scene{height:auto;padding:8px;border-color:var(--c-line)}
.cg-body:has(.cg-scene){grid-template-columns:repeat(auto-fit,minmax(min(280px,100%),1fr))}
.cgs{display:block;width:100%;max-width:300px;height:auto;aspect-ratio:200/120;fill:none;stroke-linecap:round;stroke-linejoin:round;color:var(--c-head)}
.cgs-sep{stroke:var(--c-line);stroke-width:1}
.cgs-ph{stroke:currentColor;stroke-width:2.4}
.cgs-scr{fill:var(--c-panel)}
.cgs-gr{stroke:currentColor;stroke-width:2;opacity:.3}
.cgs-d{stroke:currentColor;stroke-width:1.6;stroke-dasharray:2 4;opacity:.7}
.cgs-cam{fill:currentColor}
.cgs-lens{fill:var(--c-bg)}
.cgs-head{fill:currentColor}
.cgs-ok{fill:var(--m-ok)} .cgs-no{fill:var(--m-out)}
.cgs-mk{stroke:var(--c-bg);stroke-width:2.2}
.cgs-t{fill:currentColor;stroke:none;font:600 12px var(--f-mono)}
.cgs-sun{fill:var(--c-warn)}
.cgs-ray{stroke:var(--c-warn);stroke-width:2}
.cgs-glow{fill:var(--c-warn);opacity:.25}
.cgs-sil{opacity:.55}"""

if __name__ == "__main__":
    print(markup())
