"""Rasterise the brand SVGs into the iOS asset catalogue.

    python scripts/render-brand.py          (from mobile/)

Needs Playwright with Chromium (the backend's screenshot rig has it). The
SVGs in brand/ are the source of truth; the PNGs in ios/ are build output
that happens to be committed because Xcode reads them. check-native.mjs
refuses the Capacitor placeholders, which `cap add` would put back.
"""
from pathlib import Path
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent.parent
BRAND, XC = HERE / "brand", HERE / "ios/App/App/Assets.xcassets"
ICONS = {
    "app-icon-master.svg": XC / "AppIcon.appiconset/AppIcon-512@2x.png",
    "app-icon-dark.svg": XC / "AppIcon.appiconset/AppIcon-dark.png",
    "app-icon-tinted.svg": XC / "AppIcon.appiconset/AppIcon-tinted.png",
}
SPLASH = [XC / f"Splash.imageset/splash-2732x2732{s}.png" for s in ("", "-1", "-2")]


def shoot(pg, html, size, out):
    pg.set_viewport_size({"width": size, "height": size})
    pg.set_content(html)
    pg.wait_for_timeout(150)
    pg.screenshot(path=str(out), clip={"x": 0, "y": 0, "width": size, "height": size})
    print("wrote", out.relative_to(HERE))


with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(device_scale_factor=1)
    for svg, out in ICONS.items():
        s = (BRAND / svg).read_text(encoding="utf-8").replace('width="1024" height="1024"', 'width="1024" height="1024"', 1)
        shoot(pg, f"<body style='margin:0'>{s}</body>", 1024, out)
    mark = (BRAND / "splash-mark.svg").read_text(encoding="utf-8").replace('width="1024" height="1024"', 'width="900" height="900"', 1)
    html = ("<body style='margin:0;background:#14294B'><div style='width:2732px;height:2732px;"
            f"display:flex;align-items:center;justify-content:center'>{mark}</div></body>")
    shoot(pg, html, 2732, SPLASH[0])
    for extra in SPLASH[1:]:
        extra.write_bytes(SPLASH[0].read_bytes())
    b.close()
