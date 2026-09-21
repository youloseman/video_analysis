#!/usr/bin/env python3
"""Rasterise every brand asset from the SVG masters in docs/design-assets/logo.

The masters were drawn in Claude Design (2026-09-21): the "F" with two motion
trails and the coral joint dot. Everything a browser, a phone home screen or a
store listing needs is a raster at a fixed size, and those are generated here
so a change to the mark is one edit and one run.

    python backend/scripts/build_brand.py

Needs Pillow and Playwright with Chromium (`pip install playwright pillow &&
playwright install chromium`). Chromium is the rasteriser because it renders
the SVGs exactly as browsers will, fonts included -- the OG image sets the
wordmark in Archivo from Google Fonts, so that one needs the network.

What comes out, and why each size:

  backend/app/static/favicon.svg               the 32 px master, as is
  backend/app/static/favicon.ico               16 + 32 + 48, for /favicon.ico
  backend/app/static/icons/icon-{192,512}.png  PWA "any" (full square, no
                                               corners: the OS rounds them)
  backend/app/static/icons/icon-maskable-*.png PWA "maskable": mark inside the
                                               central 66 % safe circle
  backend/app/static/icons/apple-touch-icon.png 180 px, opaque, square
  backend/app/static/og-image.png              1200 x 630 social preview
  mobile/ios/.../AppIcon-512@2x.png            1024, opaque, no alpha -- App
                                               Store Connect rejects alpha
  mobile/ios/.../AppIcon-dark.png              iOS 18 dark appearance
  mobile/ios/.../AppIcon-tinted.png            iOS 18 tinted (greyscale, alpha)
  mobile/android/.../mipmap-*/ic_launcher*.png legacy square + round, and the
                                               adaptive foreground at 108 dp
  docs/design-assets/logo/play-store-512.png   Google Play listing icon
  docs/design-assets/logo/avatar-800.png       social profile avatar
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
MASTERS = ROOT / "docs" / "design-assets" / "logo"
STATIC = ROOT / "backend" / "app" / "static"
ICONS = STATIC / "icons"
IOS_SET = ROOT / "mobile" / "ios" / "App" / "App" / "Assets.xcassets" / "AppIcon.appiconset"
ANDROID_RES = ROOT / "mobile" / "android" / "app" / "src" / "main" / "res"

NAVY = "#14294B"

# Android launcher sizes: legacy icons at 48 dp, adaptive foreground at 108 dp.
ANDROID_DENSITIES = {"mdpi": 1.0, "hdpi": 1.5, "xhdpi": 2.0, "xxhdpi": 3.0, "xxxhdpi": 4.0}


def _svg_data_uri(path: Path) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(path.read_bytes()).decode()


def _render(page, svg: Path, size: int, out: Path, *, transparent: bool) -> None:
    """Draw one SVG at exactly ``size`` px square and save it as PNG."""
    bg = "transparent" if transparent else NAVY
    page.set_viewport_size({"width": size, "height": size})
    page.set_content(
        f'<html><body style="margin:0;background:{bg}">'
        f'<img src="{_svg_data_uri(svg)}" style="display:block;width:{size}px;height:{size}px">'
        f"</body></html>"
    )
    page.wait_for_timeout(60)
    out.parent.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(out), omit_background=transparent, clip={"x": 0, "y": 0, "width": size, "height": size})
    if not transparent:
        # Belt and braces: the store rejects an alpha channel, so drop it.
        Image.open(out).convert("RGB").save(out, optimize=True)


def _circle_crop(src: Path, out: Path) -> None:
    im = Image.open(src).convert("RGBA")
    mask = Image.new("L", im.size, 0)
    ImageDraw.Draw(mask).ellipse((0, 0, im.width - 1, im.height - 1), fill=255)
    im.putalpha(mask)
    im.save(out, optimize=True)


OG_HTML = """<!doctype html><html><head>
<link href="https://fonts.googleapis.com/css2?family=Archivo:ital,wght@1,900&family=IBM+Plex+Sans:wght@400;600&display=swap" rel="stylesheet">
<style>
  html,body{margin:0;width:1200px;height:630px;background:%(navy)s;overflow:hidden}
  .wrap{position:relative;width:1200px;height:630px;font-family:"IBM Plex Sans",system-ui,sans-serif}
  /* icon-foreground keeps the mark inside its 66 %% safe zone, so it is drawn
     larger and pulled up-left to land where the tile version sat. */
  .mark{position:absolute;left:24px;top:-4px;width:450px;height:450px}
  .word{position:absolute;left:96px;top:378px;display:flex;align-items:flex-end;gap:10px;
        font-family:Archivo,sans-serif;font-style:italic;font-weight:900;font-size:96px;line-height:1;
        letter-spacing:-.02em;color:#fff}
  .word i{display:block;width:22px;height:22px;border-radius:50%%;background:#F1553F;transform:translateY(-52px)}
  .tag{position:absolute;left:96px;top:504px;font-size:34px;color:#C6CCD6}
  .url{position:absolute;right:96px;bottom:52px;font-size:26px;font-weight:600;color:#8A929E}
</style></head><body><div class="wrap">
  <img class="mark" src="%(mark)s">
  <div class="word">Flapp<i></i></div>
  <div class="tag">The triathlete&rsquo;s honest technique lab</div>
  <div class="url">getflapp.com</div>
</div></body></html>"""


def build_og(page, out: Path) -> None:
    page.set_viewport_size({"width": 1200, "height": 630})
    # icon-foreground is the white F with the CORAL dot on a transparent ground;
    # mark-mono-white would put a white dot there.
    page.set_content(OG_HTML % {"navy": NAVY, "mark": _svg_data_uri(MASTERS / "icon-foreground.svg")})
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(400)
    page.screenshot(path=str(out), clip={"x": 0, "y": 0, "width": 1200, "height": 630})
    Image.open(out).convert("RGB").save(out, optimize=True)


def write_ios_contents() -> None:
    (IOS_SET / "Contents.json").write_text(json.dumps({
        "images": [
            {"filename": "AppIcon-512@2x.png", "idiom": "universal", "platform": "ios", "size": "1024x1024"},
            {"appearances": [{"appearance": "luminosity", "value": "dark"}],
             "filename": "AppIcon-dark.png", "idiom": "universal", "platform": "ios", "size": "1024x1024"},
            {"appearances": [{"appearance": "luminosity", "value": "tinted"}],
             "filename": "AppIcon-tinted.png", "idiom": "universal", "platform": "ios", "size": "1024x1024"},
        ],
        "info": {"author": "xcode", "version": 1},
    }, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright is not installed in this interpreter: pip install playwright && playwright install chromium",
              file=sys.stderr)
        return 2

    written: list[Path] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(device_scale_factor=1)

        def png(svg: str, size: int, out: Path, transparent: bool = False) -> Path:
            _render(page, MASTERS / f"{svg}.svg", size, out, transparent=transparent)
            written.append(out)
            return out

        # --- web ---------------------------------------------------------
        fav = STATIC / "favicon.svg"
        fav.write_bytes((MASTERS / "favicon.svg").read_bytes())
        written.append(fav)
        ico_frames = [png("favicon", s, STATIC / f"_fav{s}.png") for s in (16, 32, 48)]
        Image.open(ico_frames[-1]).save(
            STATIC / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)],
            append_images=[Image.open(f) for f in ico_frames[:-1]],
        )
        for f in ico_frames:
            f.unlink()
            written.remove(f)
        written.append(STATIC / "favicon.ico")

        for s in (192, 512):
            png("app-icon-master", s, ICONS / f"icon-{s}.png")
            png("icon-maskable", s, ICONS / f"icon-maskable-{s}.png")
        png("app-icon-master", 180, ICONS / "apple-touch-icon.png")

        build_og(page, STATIC / "og-image.png")
        written.append(STATIC / "og-image.png")

        # --- iOS ---------------------------------------------------------
        png("app-icon-master", 1024, IOS_SET / "AppIcon-512@2x.png")
        png("app-icon-dark", 1024, IOS_SET / "AppIcon-dark.png")
        png("app-icon-tinted", 1024, IOS_SET / "AppIcon-tinted.png", transparent=True)
        write_ios_contents()
        written.append(IOS_SET / "Contents.json")

        # --- Android -----------------------------------------------------
        for name, scale in ANDROID_DENSITIES.items():
            d = ANDROID_RES / f"mipmap-{name}"
            square = png("app-icon-master", round(48 * scale), d / "ic_launcher.png")
            _circle_crop(square, d / "ic_launcher_round.png")
            written.append(d / "ic_launcher_round.png")
            png("icon-foreground", round(108 * scale), d / "ic_launcher_foreground.png", transparent=True)
        bg_xml = ANDROID_RES / "values" / "ic_launcher_background.xml"
        bg_xml.write_text(
            '<?xml version="1.0" encoding="utf-8"?>\n<resources>\n'
            f'    <color name="ic_launcher_background">{NAVY}</color>\n</resources>\n',
            encoding="utf-8",
        )
        written.append(bg_xml)

        # --- stores / social ---------------------------------------------
        png("icon-maskable", 512, MASTERS / "play-store-512.png")
        png("avatar", 800, MASTERS / "avatar-800.png")

        browser.close()

    for w in written:
        print(f"  {w.relative_to(ROOT)}  {w.stat().st_size:>7} B")
    print(f"{len(written)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
