"""The Expert Review as a branded A4 document, for the athlete to keep as a PDF.

Artur (2026-10-14): the review the athlete pays $39 for should arrive as a
designed PDF in Flapp's own style -- the coach's text with the evidence it
rests on: the kinogram, the frame the angles were read on, and every
measured angle against its range and the Kona pro field.

``render_review_html`` is pure: report + stored result + a few facts in, one
self-contained HTML page out (images as data URIs). Turning it into a PDF is
a print of that page (scripts/review_pdf.py, Chromium); nothing here needs a
browser, so the same page can be served, previewed or tested.

Brand: brandbook-v3 -- navy #14294B, coral #F1553F, blue #2F6DE0, panels
#F3F5F8; Archivo italic for display, Manrope for text, IBM Plex Mono for
labels; status green / amber / coral as in the app.
"""
from __future__ import annotations

import html
import math
import re
from typing import Any

# field, label, unit, scale (lo, hi) for the bar
_BIKE_ROWS: tuple[tuple[str, str, str, tuple[float, float]], ...] = (
    ("knee_at_bdc", "Knee at the bottom · saddle height", "°", (120, 170)),
    ("knee_at_tdc", "Knee at the top", "°", (40, 100)),
    ("hip_at_tdc", "Hip at the top", "°", (30, 80)),
    ("trunk_angle_avg", "Torso from horizontal", "°", (0, 50)),
    ("shoulder_angle_avg", "Shoulder", "°", (60, 120)),
    ("elbow_angle_avg", "Elbow", "°", (60, 130)),
    ("forearm_tilt_avg", "Forearm tilt", "°", (-10, 40)),
    ("pelvic_ratio", "Pelvis rocking (ratio)", "", (1, 6)),
    ("helmet_above_back", "Helmet above the back", "%", (0, 40)),
)
_STATUS_WORD = {"good": "In range", "warn": "At the edge", "bad": "Outside", "info": "Measured", "na": "Not measured"}


def _e(s: Any) -> str:
    return html.escape(str(s or ""))


def _md(text: str) -> str:
    """The little Markdown a review uses: paragraphs, **bold**, *italic*."""
    out = []
    for para in re.split(r"\n\s*\n", str(text or "").strip()):
        if not para.strip():
            continue
        p = _e(para.strip()).replace("\n", "<br>")
        p = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", p)
        p = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", p)
        out.append(f"<p>{p}</p>")
    return "".join(out)


def _status(value: float, lo: float | None, hi: float | None) -> str:
    if lo is None or hi is None:
        return "info"
    if lo <= value <= hi:
        return "good"
    margin = max((hi - lo) * 0.10, 3.0 if hi - lo > 6 else 0.3)
    return "warn" if lo - margin <= value <= hi + margin else "bad"


def _bar(value: float | None, band: tuple[float, float] | None, pro: dict | None,
         scale: tuple[float, float], status: str) -> str:
    """Track, the band (green), the pro middle half (navy, median tick), you (dot)."""
    w, h = 300, 34
    lo_s, hi_s = scale

    def x(v: float) -> float:
        return round(8 + (min(max(v, lo_s), hi_s) - lo_s) / (hi_s - lo_s) * (w - 16), 1)

    parts = [f'<line x1="8" y1="20" x2="{w - 8}" y2="20" stroke="#DCE2EA" stroke-width="2"/>']
    if band:
        parts.append(f'<rect x="{x(band[0])}" y="14" width="{max(2, x(band[1]) - x(band[0]))}" height="12" '
                     f'rx="3" fill="#E6F4EC" stroke="#2E9E5B" stroke-width="1"/>')
    if pro:
        parts.append(f'<line x1="{x(pro["q1"])}" y1="7" x2="{x(pro["q3"])}" y2="7" stroke="#14294B" '
                     f'stroke-width="3" stroke-linecap="round"/>'
                     f'<line x1="{x(pro["median"])}" y1="3" x2="{x(pro["median"])}" y2="11" stroke="#14294B" stroke-width="2"/>')
    if value is not None:
        col = {"good": "#2E9E5B", "warn": "#C97B10", "bad": "#F1553F"}.get(status, "#2F6DE0")
        parts.append(f'<circle cx="{x(value)}" cy="20" r="6.5" fill="{col}" stroke="#fff" stroke-width="2"/>')
    return f'<svg class="bar" viewBox="0 0 {w} {h}" width="{w}" height="{h}" aria-hidden="true">{"".join(parts)}</svg>'


def metric_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Every measured bike angle with its band, the pro field and a status."""
    s = result.get("sport_specific_metrics") or {}
    bands = result.get("reference_bands") or {}
    pro = ((result.get("pro_field") or {}).get("metrics")) or {}
    rows = []
    for field, label, unit, scale in _BIKE_ROWS:
        v = s.get(field)
        b = bands.get(field)
        band = (float(b["lo"]), float(b["hi"])) if isinstance(b, dict) and b.get("lo") is not None else None
        if field == "helmet_above_back":
            band = (0.0, 23.0)
            if s.get("helmet_check") or not isinstance(v, (int, float)):
                rows.append({"field": field, "label": label, "value": None, "unit": unit, "band": band,
                             "pro": pro.get(field), "status": "na", "scale": scale})
                continue
        if not isinstance(v, (int, float)) or (isinstance(v, float) and math.isnan(v)):
            continue
        st = _status(float(v), *(band or (None, None)))
        rows.append({"field": field, "label": label, "value": float(v), "unit": unit, "band": band,
                     "pro": pro.get(field), "status": st, "scale": scale})
    return rows


def _band_txt(row: dict[str, Any]) -> str:
    if not row.get("band"):
        return "—"
    lo, hi = row["band"]
    if row["field"] == "helmet_above_back":
        return f"up to {hi:g}%"
    return f"{lo:g}–{hi:g}{row['unit']}"


def _fmt(v: float | None, unit: str) -> str:
    if v is None:
        return "—"
    return f"{v:.1f}{unit}" if unit == "" else f"{v:.0f}{unit}"


_LOGO = ('<svg viewBox="0 0 32 32" width="30" height="30" aria-hidden="true"><path d="M5 6.5 L18.5 6.5 '
         'L18.5 11 L9.5 11 L9.5 14.4 L16.5 14.4 L16.5 18.6 L9.5 18.6 L9.5 26 L5 26 Z" fill="#FFFFFF" '
         'transform="translate(5.05 0) skewX(-11)"/><circle cx="19.5" cy="8" r="3.2" fill="#F1553F" '
         'transform="translate(5.05 0) skewX(-11)"/></svg>')

_CSS = """
@page{size:A4;margin:0}
*{box-sizing:border-box}
html,body{margin:0;padding:0}
body{font:9.6pt/1.55 'Manrope','Segoe UI',sans-serif;color:#1E2530;-webkit-print-color-adjust:exact;print-color-adjust:exact}
.page{width:210mm;min-height:297mm;padding:14mm 15mm 18mm;position:relative;page-break-after:always;break-after:page}
.page:last-child{page-break-after:auto;break-after:auto}
.mono{font-family:'IBM Plex Mono',monospace}
.eyebrow{font:500 7.6pt 'IBM Plex Mono',monospace;letter-spacing:.12em;text-transform:uppercase;color:#6E7787;margin:0 0 6px}
h1,h2,h3{font-family:'Archivo','Arial Black',sans-serif;font-style:italic;font-weight:800;color:#14294B;margin:0;letter-spacing:-.01em}
h2{font-size:19pt;margin:0 0 4px}
h3{font-size:11.5pt;margin:0 0 4px}
.speed{height:3px;width:58px;background:linear-gradient(90deg,#2F6DE0,#F1553F);border-radius:2px;margin:6px 0 14px}
p{margin:0 0 8px}
/* cover */
.cover-band{background:#14294B;color:#fff;margin:-14mm -15mm 0;padding:12mm 15mm 10mm;position:relative;overflow:hidden}
.cover-band:after{content:"";position:absolute;left:0;right:0;bottom:0;height:4px;background:linear-gradient(90deg,#2F6DE0,#F1553F)}
.brand{display:flex;align-items:center;gap:8px;font:italic 900 15pt 'Archivo',sans-serif;letter-spacing:-.01em}
.brand .dot{color:#F1553F}
.cover-band h1{color:#fff;font-size:30pt;margin:16px 0 4px}
.cover-band .sub{color:#B9CDF2;font-size:10pt}
.cover-grid{display:grid;grid-template-columns:auto 1fr;gap:18px;align-items:center;margin:16px 0 4px}
.ring{width:118px;height:118px}
.facts{display:grid;grid-template-columns:auto 1fr;gap:3px 14px;font-size:9.4pt}
.facts dt{font:500 7.6pt 'IBM Plex Mono',monospace;letter-spacing:.1em;text-transform:uppercase;color:#6E7787;padding-top:2px}
.facts dd{margin:0;color:#1E2530;font-weight:600}
.verdict{background:#F3F5F8;border-left:4px solid #F1553F;padding:12px 16px;margin:14px 0 14px;font-size:11pt;line-height:1.55}
.verdict p:last-child{margin:0}
.fig{margin:0 0 6px}
.fig img{display:block;width:100%;border-radius:6px}
.cap{font-size:8pt;color:#6E7787;margin:4px 0 0}
.glance{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:14px 0 0}
.tile{background:#F3F5F8;border-radius:8px;padding:12px 14px;border-top:3px solid #DCE2EA}
.tile.good{border-top-color:#2E9E5B}.tile.bad{border-top-color:#F1553F}.tile.blue{border-top-color:#2F6DE0}
.tile .big{font:italic 900 24pt/1.1 'Archivo',sans-serif;color:#14294B}
.tile .big.txt{font-size:13pt;line-height:1.25;margin-top:4px}
.tile .big small{font-size:12pt;color:#5C6675}
.tile .lbl{font-size:9pt;color:#1E2530;font-weight:600;margin-top:2px}
/* numbers */
table.nums{width:100%;border-collapse:collapse;margin:4px 0 10px}
.nums th{font:500 7.2pt 'IBM Plex Mono',monospace;letter-spacing:.08em;text-transform:uppercase;color:#6E7787;text-align:left;padding:0 6px 6px;border-bottom:1px solid #DCE2EA}
.nums td{padding:6px;border-bottom:1px solid #EEF1F5;vertical-align:middle}
.nums td.v{font:600 10.5pt 'IBM Plex Mono',monospace;white-space:nowrap}
.nums td.band,.nums td.pro{font:9pt 'IBM Plex Mono',monospace;color:#5C6675;white-space:nowrap}
.chip{display:inline-block;font:600 7.4pt 'IBM Plex Mono',monospace;letter-spacing:.04em;padding:2px 8px;border-radius:999px;white-space:nowrap}
.chip.good{background:#E6F4EC;color:#1F7A45}.chip.warn{background:#FFF3DC;color:#9A5E0B}
.chip.bad{background:#FDE8E4;color:#B8371F}.chip.na,.chip.info{background:#F3F5F8;color:#5C6675}
.legend{display:flex;gap:16px;font-size:8pt;color:#5C6675;margin:0 0 6px;align-items:center}
.legend i{display:inline-block;vertical-align:middle;margin-right:5px}
.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.card{background:#F3F5F8;border-radius:8px;padding:12px 14px}
.card p:last-child{margin:0}
.kf{display:grid;grid-template-columns:62mm 1fr;gap:14px;align-items:start}
.kf img{width:100%;border-radius:6px;display:block}
/* priorities */
.prio{display:grid;grid-template-columns:34px 1fr;gap:12px;border:1px solid #DCE2EA;border-radius:8px;padding:12px 14px;margin:0 0 10px;break-inside:avoid}
.prio .n{font:italic 900 22pt/1 'Archivo',sans-serif;color:#F1553F}
.prio dl{display:grid;grid-template-columns:30mm 1fr;gap:4px 10px;margin:6px 0 0}
.prio dt{font:500 7.2pt 'IBM Plex Mono',monospace;letter-spacing:.08em;text-transform:uppercase;color:#6E7787;padding-top:2px}
.prio dd{margin:0}
table.fit{width:100%;border-collapse:collapse;font-size:8.8pt;margin:4px 0 12px}
.fit th{font:500 7.2pt 'IBM Plex Mono',monospace;letter-spacing:.08em;text-transform:uppercase;color:#fff;background:#14294B;text-align:left;padding:6px 8px}
.fit td{padding:7px 8px;border-bottom:1px solid #DCE2EA;vertical-align:top}
.fit td:first-child{font-weight:700;color:#14294B}
/* plan */
.weeks{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:4px 0 14px}
.week{background:#F3F5F8;border-radius:8px;padding:10px 11px;border-top:3px solid #2F6DE0;font-size:8.8pt}
.week b.w{display:block;font:italic 800 10.5pt 'Archivo',sans-serif;color:#14294B;margin-bottom:4px}
.retest{border:1.5px dashed #2F6DE0;border-radius:8px;padding:12px 14px;margin:0 0 12px;background:#EAF1FC}
.sign{display:flex;align-items:center;gap:12px;margin-top:16px;padding-top:12px;border-top:1px solid #DCE2EA}
.sign .who{font:italic 800 12pt 'Archivo',sans-serif;color:#14294B}
.sign .bio{font-size:8.6pt;color:#5C6675}
.sec{margin:0 0 14px;break-inside:avoid}
"""


def _ring(score: Any, grade: Any) -> str:
    try:
        sc = max(0.0, min(100.0, float(score)))
    except (TypeError, ValueError):
        return ""
    col = "#2E9E5B" if sc >= 85 else "#C97B10" if sc >= 70 else "#F1553F"
    r, c = 52, 2 * math.pi * 52
    return (f'<svg class="ring" viewBox="0 0 120 120"><circle cx="60" cy="60" r="{r}" fill="none" stroke="#2A4170" stroke-width="9"/>'
            f'<circle cx="60" cy="60" r="{r}" fill="none" stroke="{col}" stroke-width="9" stroke-linecap="round" '
            f'stroke-dasharray="{c * sc / 100:.1f} {c:.1f}" transform="rotate(-90 60 60)"/>'
            f'<text x="60" y="62" text-anchor="middle" font-family="Archivo" font-style="italic" font-weight="900" font-size="34" fill="#fff">{int(round(sc))}</text>'
            f'<text x="60" y="84" text-anchor="middle" font-family="IBM Plex Mono" font-size="11" fill="#B9CDF2">/100 · {_e(grade)}</text></svg>')


def _weeks(plan: str) -> str:
    """'**Week 1** — ...' paragraphs as four cards; anything else as prose."""
    chunks = re.split(r"\*\*\s*(Week\s*\d[^*]*)\*\*\s*[—:-]?\s*", str(plan or ""))
    if len(chunks) < 3:
        return f'<div class="card">{_md(plan)}</div>'
    cards = []
    for i in range(1, len(chunks) - 1, 2):
        body = chunks[i + 1].strip()
        cards.append(f'<div class="week"><b class="w">{_e(chunks[i].strip())}</b>{_md(body[:1].upper() + body[1:])}</div>')
    return f'<div class="weeks">{"".join(cards)}</div>'


def _glance(rows: list[dict[str, Any]], priorities: list[dict[str, Any]]) -> str:
    """Three tiles under the kinogram: how many angles sit in range, which
    are off, and the first change -- the review in one line of sight."""
    graded = [x for x in rows if x["status"] in ("good", "warn", "bad")]
    if not graded:
        return ""
    off = [x for x in graded if x["status"] != "good"]
    first = (priorities[0].get("title") if priorities else "") or ""
    tiles = [
        ("In range", f"{len(graded) - len(off)}<small> of {len(graded)}</small>", "angles inside their range", "good"),
        ("Worth a look", str(len(off)), ", ".join(x["label"].split(" · ")[0] for x in off) or "nothing outside", "bad" if off else "good"),
    ]
    out = [f"<div class='tile {c}'><div class='eyebrow'>{_e(k)}</div><div class='big'>{v}</div><div class='lbl'>{_e(lbl)}</div></div>"
           for k, v, lbl, c in tiles]
    if first:
        out.append(f"<div class='tile blue'><div class='eyebrow'>Change first</div><div class='big txt'>{_e(first)}</div></div>")
    return "<div class='glance'>" + "".join(out) + "</div>"


def render_review_html(report: dict[str, Any], result: dict[str, Any], meta: dict[str, Any]) -> str:
    """The whole document. ``meta``: athlete, date, kinogram / keyframe data URIs,
    and optionally ``position_label`` and ``filmed``."""
    r = report or {}
    rows = metric_rows(result)
    side = result.get("camera_side")
    facts = [
        ("Athlete", meta.get("athlete")),
        ("Position", meta.get("position_label") or str(result.get("cycling_position") or "").replace("_", " ")),
        ("Filmed", meta.get("filmed")),
        ("Camera", f"{side} side" + (" · chainring side" if side == "right" else "") if side else None),
        ("Reviewed by", r.get("reviewer")),
        ("Delivered", meta.get("date")),
    ]
    facts_html = "".join(f"<dt>{_e(k)}</dt><dd>{_e(v)}</dd>" for k, v in facts if v)

    num_rows = "".join(
        f"<tr><td>{_e(row['label'])}</td><td class='v'>{_fmt(row['value'], row['unit'])}</td>"
        f"<td>{_bar(row['value'], row['band'], row['pro'], row['scale'], row['status'])}</td>"
        f"<td class='band'>{_band_txt(row)}</td>"
        f"<td class='pro'>{(_fmt(row['pro']['median'], row['unit']) if row['pro'] else '—')}</td>"
        f"<td><span class='chip {row['status']}'>{_STATUS_WORD[row['status']]}</span></td></tr>"
        for row in rows)

    prios = "".join(
        f"<div class='prio'><div class='n'>{i}</div><div><h3>{_e(p.get('title'))}</h3><dl>"
        + "".join(f"<dt>{lbl}</dt><dd>{_e(p.get(k))}</dd>" for k, lbl in
                  (("why", "Why"), ("change", "What to change"), ("check", "How you'll know")) if p.get(k))
        + "</dl></div></div>"
        for i, p in enumerate(r.get("priorities") or [], 1))
    fit = r.get("fit") or []
    fit_html = ("<table class='fit'><tr><th>Part</th><th>Now</th><th>Change</th><th>Why</th><th>Trade-off</th></tr>"
                + "".join(f"<tr><td>{_e(f.get('part'))}</td><td>{_e(f.get('now'))}</td><td>{_e(f.get('suggest'))}</td>"
                          f"<td>{_e(f.get('why'))}</td><td>{_e(f.get('risk'))}</td></tr>" for f in fit)
                + "</table>") if fit else ""

    kino = meta.get("kinogram")
    kf = meta.get("keyframe")
    sec = lambda t, body, eyebrow="": (f"<div class='sec'>{f'<div class=eyebrow>{_e(eyebrow)}</div>' if eyebrow else ''}"  # noqa: E731
                                       f"<h3>{_e(t)}</h3>{body}</div>") if body else ""

    page1 = f"""
<section class="page">
  <div class="cover-band">
    <div class="brand">{_LOGO}<span>Flapp<span class="dot">.</span></span></div>
    <h1>Expert Review</h1>
    <div class="sub">Your position, read by a coach — with the numbers it rests on.</div>
    <div class="cover-grid">{_ring(result.get('technique_score'), result.get('letter_grade'))}<dl class="facts" style="color:#fff">{facts_html.replace('<dd>', '<dd style="color:#fff">')}</dl></div>
  </div>
  <div class="eyebrow" style="margin-top:16px">The verdict</div>
  <div class="verdict">{_md(r.get('verdict'))}</div>
  {f'<div class="eyebrow">One pedal revolution · top, bottom and 45° either side</div><figure class="fig"><img src="{kino}" alt="Kinogram"></figure>' if kino else ''}
  {_glance(rows, r.get('priorities') or [])}
</section>"""

    page2 = f"""
<section class="page">
  <div class="eyebrow">Your numbers</div><h2>Where you sit</h2><div class="speed"></div>
  <div class="legend"><span><i><svg width="26" height="10"><rect x="1" y="1" width="24" height="8" rx="2" fill="#E6F4EC" stroke="#2E9E5B"/></svg></i>Flapp range</span>
    <span><i><svg width="26" height="10"><line x1="2" y1="5" x2="24" y2="5" stroke="#14294B" stroke-width="3" stroke-linecap="round"/></svg></i>Kona 2026 pros, middle half · tick = median</span>
    <span><i><svg width="12" height="12"><circle cx="6" cy="6" r="5" fill="#2F6DE0"/></svg></i>You</span></div>
  <table class="nums"><tr><th>Angle</th><th>You</th><th></th><th>Range</th><th>Pros</th><th></th></tr>{num_rows}</table>
  <div class="kf">
    {f'<div><img src="{kf}" alt="The frame the angles were read on"><p class="cap">The frame your angles were read on.</p></div>' if kf else '<div></div>'}
    <div>{sec('How much to trust these numbers', _md(r.get('trust')))}{sec("What's already working", _md(r.get('working')))}</div>
  </div>
</section>"""

    page3 = f"""
<section class="page">
  <div class="eyebrow">The plan</div><h2>Fix these, in this order</h2><div class="speed"></div>
  {prios}
  {sec('Position changes', fit_html) if fit_html else ''}
  {sec("Don't change this", _md(r.get('dont_change')))}
</section>"""

    page4 = f"""
<section class="page">
  <div class="eyebrow">Next four weeks</div><h2>What to do, week by week</h2><div class="speed"></div>
  {_weeks(r.get('plan'))}
  <div class="retest"><div class="eyebrow" style="color:#2459C2">Film again</div>{_md(r.get('retest'))}</div>
  {sec("What this footage couldn't tell me", _md(r.get('caveats')))}
  <div class="sign">{_LOGO.replace('#FFFFFF', '#14294B')}<div><div class="who">{_e(r.get('reviewer'))}</div>
    <div class="bio">{_e(r.get('reviewer_bio'))}{' · ' if r.get('reviewer_bio') else ''}Flapp Expert Review · getflapp.com</div></div></div>
</section>"""

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Flapp Expert Review</title>
<link href="https://fonts.googleapis.com/css2?family=Archivo:ital,wght@0,700;0,800;1,800;1,900&family=Manrope:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>{_CSS}</style></head><body>{page1}{page2}{page3}{page4}</body></html>"""
