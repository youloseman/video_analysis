#!/usr/bin/env python3
"""Print an Expert Review as Flapp's branded A4 PDF (services/review_pdf.py).

    python backend/scripts/review_pdf.py --result result.json --report report.json \
        --kinogram kino.jpg --keyframe kf.jpg --athlete "Name" --out review.pdf

``result.json`` is a stored analysis result; ``report.json`` the reviewer's
report (the editor's sections, or {"report": {...}}). Needs Playwright with
Chromium; fonts load from Google Fonts, so it needs the network.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.expert_review import public_report  # noqa: E402
from app.services.review_pdf import render_review_html  # noqa: E402


def _uri(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--result", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--kinogram")
    ap.add_argument("--keyframe")
    ap.add_argument("--athlete", default="")
    ap.add_argument("--filmed", default="")
    ap.add_argument("--date", default=date.today().strftime("%d %b %Y"))
    ap.add_argument("--position-label", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--html", help="also write the HTML here")
    a = ap.parse_args()

    result = json.loads(Path(a.result).read_text(encoding="utf-8"))
    report = json.loads(Path(a.report).read_text(encoding="utf-8"))
    report = public_report(report.get("report", report))
    page = render_review_html(report, result, {
        "athlete": a.athlete, "filmed": a.filmed, "date": a.date,
        "position_label": a.position_label,
        "kinogram": _uri(a.kinogram), "keyframe": _uri(a.keyframe),
    })
    if a.html:
        Path(a.html).write_text(page, encoding="utf-8")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        tab = browser.new_page()
        tab.set_content(page, wait_until="networkidle")
        tab.pdf(path=a.out, format="A4", print_background=True, prefer_css_page_size=True,
                display_header_footer=True, header_template="<span></span>",
                footer_template=("<div style='width:100%;font:7px IBM Plex Mono,monospace;color:#6E7787;"
                                 "padding:0 15mm;display:flex;justify-content:space-between'>"
                                 "<span>Flapp · Expert Review</span><span><span class='pageNumber'></span>"
                                 " / <span class='totalPages'></span></span></div>"))
        browser.close()
    print(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
