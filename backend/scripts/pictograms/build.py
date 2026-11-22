"""Build the pictogram sprite the PDF report inlines.

    python scripts/pictograms/build.py   -> app/static/media/pictograms.svg

The same symbols are in index.html's sprite (the SPA); the PDF has no page to
borrow them from, so it embeds this file once and points <use> at it. The
colours are tokens; report_pdf sets them for paper.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rider_icons as ri  # noqa: E402

OUT = HERE.parents[1] / "app" / "static" / "media" / "pictograms.svg"

if __name__ == "__main__":
    css = (".pic,.pm{fill:none;stroke-linecap:round;stroke-linejoin:round;color:var(--c-head)}\n" + ri.app_css())
    body = (f'<svg xmlns="http://www.w3.org/2000/svg" width="0" height="0" class="pm-sprite">'
            f"<style>{css}</style>\n{ri.app_sprite()}</svg>\n")
    OUT.write_bytes(body.encode("utf-8"))
    print("wrote", OUT, len(body) // 1024, "KB")
