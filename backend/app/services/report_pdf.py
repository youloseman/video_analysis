"""The analysis report as a branded A4 document -- the Full plan's PDF.

Artur (2026-10-26): "PDF report" on the Full card was the browser printing
the app's page. Full now gets a designed document in the Expert Review's
style (services/review_pdf.py): a cover with the score and the frame, the
coach's read, the plan in order, and every number against its range -- the
bike's against the Kona pro field too.

``render_report_html`` is pure: a stored result + a few facts in, one
self-contained HTML page out (images as data URIs). The browser turns it into
a PDF (Print > Save as PDF), so the server needs no Chromium and a phone can
save it the same way.
"""
from __future__ import annotations

import re
from typing import Any

from app.services.review_pdf import (
    _CSS,
    _LOGO,
    _STATUS_WORD,
    _bar,
    _band_txt,
    _e,
    _fmt,
    _ring,
    metric_rows,
)

_SPORT = {"bike": "Cycling", "run": "Running"}


def _inline(s: str) -> str:
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    return re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", s)


def markdown(text: str) -> str:
    """The Markdown the coach writes: headings, bullet and numbered lists,
    paragraphs, **bold** and *italic*. Everything is escaped first."""
    out: list[str] = []
    para: list[str] = []
    items: list[str] = []

    def flush() -> None:
        if para:
            out.append("<p>" + _inline(" ".join(para)) + "</p>")
            para.clear()
        if items:
            out.append("<ul>" + "".join(f"<li>{_inline(i)}</li>" for i in items) + "</ul>")
            items.clear()

    for raw in str(text or "").splitlines():
        line = _e(raw.strip())
        if not line:
            flush()
            continue
        head = re.match(r"#{1,6}\s+(.*)", line)
        bullet = re.match(r"(?:[-*•]|\d+[.)])\s+(.*)", line)
        if head:
            flush()
            out.append(f"<h3>{_inline(head.group(1))}</h3>")
        elif bullet:
            if para:
                flush()
            items.append(bullet.group(1))
        else:
            if items:
                flush()
            para.append(line)
    flush()
    return "".join(out)


def _status_of(severity: Any) -> str:
    return {"critical": "bad", "high": "bad", "warning": "warn", "moderate": "warn"}.get(
        str(severity or "").lower(), "warn")


def _range(item: dict[str, Any]) -> tuple[float, float] | None:
    r = item.get("range")
    if isinstance(r, (list, tuple)) and len(r) == 2:
        try:
            return float(r[0]), float(r[1])
        except (TypeError, ValueError):
            return None
    m = re.match(r"\s*(-?[\d.]+)\s*[-–]\s*(-?[\d.]+)", str(item.get("optimal_range") or ""))
    return (float(m.group(1)), float(m.group(2))) if m else None


def plan_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    """The run's measurements, from the plan's own strong points and issues:
    each one already carries its value and the range it was judged against."""
    plan = result.get("training_plan") or {}
    rows: list[dict[str, Any]] = []
    for item, status in [(i, "good") for i in plan.get("strong_points") or []] + \
                        [(i, _status_of(i.get("severity"))) for i in plan.get("issues") or []]:
        v = item.get("value")
        if not isinstance(v, (int, float)):
            continue
        band = _range(item)
        label = item.get("label") or str(item.get("description") or item.get("metric") or "").split(":")[0]
        lo, hi = band if band else (float(v) - 1, float(v) + 1)
        span = max(hi - lo, 1e-6)
        scale = (min(lo - span, float(v) - span * 0.2), max(hi + span, float(v) + span * 0.2))
        rows.append({"label": label, "value": float(v), "unit": "", "band": band, "pro": None,
                     "status": status, "scale": scale, "field": item.get("metric")})
    return rows


def _num(v: float) -> str:
    return f"{v:.0f}" if abs(v) >= 20 else f"{v:.2f}".rstrip("0").rstrip(".")


_COMPONENT = {"saddle_height": "Saddle height", "saddle_setback": "Saddle fore-aft",
              "bar_position": "Bar position", "bar_reach": "Bar reach", "cleats": "Cleats"}


def _fit_priorities(fit: dict[str, Any]) -> list[dict[str, Any]]:
    """The fit plan's adjustments as plan cards, in the order to make them."""
    out = []
    for d in sorted(fit.get("diagnostics") or [], key=lambda d: d.get("priority") or 99):
        if d.get("status") != "needs_adjustment":
            continue
        action = str(d.get("action") or "").replace("_", " ").capitalize()
        amount = f" {d['amount']}" if d.get("amount") else ""
        comp = _COMPONENT.get(d.get("component"), str(d.get("component") or "").replace("_", " ").capitalize())
        out.append({"label": comp, "drill": (action + amount).strip() or None, "why": d.get("reason")})
    return out


_EXTRA_CSS = """
.screen{position:sticky;top:0;z-index:5;display:flex;gap:12px;align-items:center;justify-content:space-between;
  background:#14294B;color:#fff;margin:0 -15mm 0;padding:10px 15mm;font:600 10pt 'Manrope',sans-serif}
.screen button{font:700 10pt 'Manrope',sans-serif;background:#F1553F;color:#fff;border:0;border-radius:6px;padding:10px 16px;cursor:pointer}
@media print{.screen{display:none}}
.cover-img{margin:14px 0 0}
.cover-img img{display:block;width:100%;max-height:118mm;object-fit:contain;background:#0F1A2E;border-radius:8px}
.glance{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:14px 0 0}
.glance .card{border-top:3px solid #2F6DE0}
.glance .card.fix{border-top-color:#F1553F}
.glance .card.ok{border-top-color:#2E9E5B}
.glance b{display:block;font:italic 800 11pt 'Archivo',sans-serif;color:#14294B;margin:2px 0 3px}
.page{break-before:page}
.finding{display:grid;grid-template-columns:auto 1fr;gap:4px 12px;border:1px solid #DCE2EA;border-radius:8px;padding:10px 14px;margin:0 0 8px;break-inside:avoid}
.finding h3{margin:0}
.finding p{grid-column:1/-1;margin:0}
.coachtext{border-left:3px solid #2F6DE0;padding:2px 0 2px 14px;margin:0 0 14px}
.coachtext h3{margin:12px 0 4px}
.coachtext h3:first-child{margin-top:0}
.ok-list li{margin:0 0 3px}
.hob{display:grid;grid-template-columns:auto 1fr;gap:16px;align-items:center;background:#F3F5F8;border-radius:8px;padding:12px 16px;margin:4px 0 12px;break-inside:avoid}
.hob .big{font:italic 900 26pt/1 'Archivo',sans-serif;color:#14294B}
.foot{margin:18px 0 0;padding-top:10px;border-top:1px solid #DCE2EA;font-size:8pt;color:#6E7787;display:flex;justify-content:space-between}
"""


def render_report_html(result: dict[str, Any], meta: dict[str, Any]) -> str:
    """The whole document. ``meta``: athlete, setup, filmed, prepared,
    position_label, keyframe and kinogram data URIs."""
    sport = result.get("sport_type") or result.get("sport") or ""
    plan = result.get("training_plan") or {}
    prios = plan.get("top_3_priorities") or []
    strong = plan.get("strong_points") or []
    fit = result.get("fit_plan") or {}
    if sport == "bike" and not prios:
        # A bike's plan is its fit plan: what to adjust, in order, by how much.
        prios = _fit_priorities(fit)
        strong = strong or [{"description": g.get("label") or g.get("metric")}
                            for g in fit.get("good_metrics") or []]
    issues = result.get("detected_issues") or []
    coach = ((result.get("ai_recommendations") or {}).get("report")
             or (result.get("ai_recommendations") or {}).get("summary") or "")
    rows = metric_rows(result) if sport == "bike" else plan_rows(result)
    side = result.get("camera_side")
    position = meta.get("position_label") or str(result.get("cycling_position") or "").replace("_", " ")

    facts = [
        ("Athlete", meta.get("athlete")),
        ("Sport", " · ".join(x for x in (_SPORT.get(sport, sport.title()), position if sport == "bike" else "") if x)),
        ("Setup", meta.get("setup")),
        ("Filmed", meta.get("filmed")),
        ("Camera", f"{side} side" if side else None),
        ("Prepared", meta.get("prepared")),
    ]
    facts_html = "".join(f"<dt>{_e(k)}</dt><dd>{_e(v)}</dd>" for k, v in facts if v)

    def head(eyebrow: str, title: str) -> str:
        return f"<div class='head'><div class='eyebrow'>{_e(eyebrow)}</div><h2>{_e(title)}</h2><div class='speed'></div></div>"

    first = prios[0] if prios else None
    glance = "".join([
        f"<div class='card fix'><div class='eyebrow'>Fix first</div><b>{_e(first.get('label'))}</b>{_e(first.get('cue') or first.get('drill') or first.get('why'))}</div>"
        if first else "",
        f"<div class='card ok'><div class='eyebrow'>Already working</div><b>{_e(str(strong[0].get('description') or '').split(':')[0])}</b>"
        f"{_e(len(strong))} measurement{'s' if len(strong) != 1 else ''} in range</div>" if strong else "",
        f"<div class='card'><div class='eyebrow'>Measured</div><b>{len(rows)} numbers</b>"
        f"each against its range{', with the Kona pro field' if sport == 'bike' else ''}</div>" if rows else "",
    ])

    cover = f"""
<section>
  <div class="screen"><span>Your Flapp report · use Print, then “Save as PDF”</span><button onclick="window.print()">Save as PDF</button></div>
  <div class="cover-band">
    <div class="brand">{_LOGO}<span>Flapp<span class="dot">.</span></span></div>
    <h1>Technique Report</h1>
    <div class="sub">What to fix, in what order — and the numbers behind it.</div>
    <div class="cover-grid">{_ring(result.get('technique_score'), result.get('letter_grade'))}<dl class="facts">{facts_html}</dl></div>
  </div>
  {f'<figure class="cover-img"><img src="{meta["keyframe"]}" alt="Your frame, with the angles drawn on it"><p class="cap">The frame your angles were read on.</p></figure>' if meta.get('keyframe') else ''}
  {f'<div class="glance">{glance}</div>' if glance else ''}
</section>"""

    findings = "".join(
        f"<div class='finding'><h3>{_e(str(i.get('type') or '').replace('_', ' ').capitalize())}</h3>"
        f"<span class='chip {_status_of(i.get('severity'))}'>{_e(i.get('value'))}</span>"
        f"<p>{_e(i.get('recommendation'))}</p></div>"
        for i in issues)
    read = f"""
<section class="page">
  {head("The coach's read", "What's off, and what to change first")}
  {f'<div class="coachtext">{markdown(coach)}</div>' if coach else ''}
  {("<div class='eyebrow' style='margin-top:6px'>What the measurements flagged</div>" + findings) if findings else ''}
  {'' if (coach or findings) else '<p class="lead">Nothing in this clip is outside its range. Keep filming the same way and watch the trend.</p>'}
</section>"""

    prio_html = "".join(
        f"<div class='prio'><div class='n'>{n}</div><div><h3>{_e(p.get('label'))}{' — ' + _e(p.get('drill')) if p.get('drill') else ''}</h3><dl>"
        + "".join(f"<dt>{lbl}</dt><dd>{_e(p.get(k))}</dd>" for k, lbl in
                  (("why", "Why"), ("instruction", "Do this"), ("cue", "Think")) if p.get(k))
        + (f"<dt>For</dt><dd>{_e(p.get('weeks'))} weeks</dd>" if p.get('weeks') else "")
        + "</dl></div></div>"
        for n, p in enumerate(prios, 1))
    weeks = [p.get("weeks") for p in prios if isinstance(p.get("weeks"), (int, float))]
    when = (f"In {int(min(weeks))} week{'s' if min(weeks) != 1 else ''}" if weeks
            else "After each change" if sport == "bike" and prios else "")
    note = fit.get("fitting_sequence_note") if sport == "bike" else ""
    retest = (f"<div class='retest'><div class='eyebrow' style='color:#2459C2'>Film again</div>"
              f"<p>{_e(when)}: same camera height, same side, same speed. "
              f"Flapp puts the two clips side by side and shows what moved.{' ' + _e(note) if note else ''}</p></div>") if when else ""
    # What already works sits on the numbers page as one line: on its own it
    # was a block that spilled onto a page of its own.
    ok_names = [str(s.get("description") or "").split(":")[0].split(" (")[0] for s in strong[:8]]
    ok_html = ("<div class='working' style='margin:0 0 12px'><h3>Already in range — leave these alone</h3>"
               f"<p>{_e(', '.join(n for n in ok_names if n))}.</p></div>") if any(ok_names) else ""
    plan_sec = f"""
<section>
  {head('The plan', 'Fix these, in this order')}
  {prio_html}
  {retest}
</section>""" if prio_html else ""

    num_rows = "".join(
        f"<tr><td>{_e(r['label'])}</td><td class='v'>{_fmt(r['value'], r['unit']) if r['unit'] or r['value'] is None else _num(r['value'])}</td>"
        f"<td>{_bar(r['value'], r['band'], r['pro'], r['scale'], r['status'])}</td>"
        f"<td class='band'>{_band_txt(r) if r.get('band') else '—'}</td>"
        + (f"<td class='pro'>{(_fmt(r['pro']['median'], r['unit']) if r.get('pro') else '—')}</td>" if sport == "bike" else "")
        + f"<td><span class='chip {r['status']}'>{_STATUS_WORD.get(r['status'], '')}</span></td></tr>"
        for r in rows)
    hob = (result.get("sport_specific_metrics") or {}).get("hip_over_bb") or result.get("hip_over_bb")
    hob_html = ""
    if isinstance(hob, dict) and hob.get("angle_deg") is not None:
        gap = hob.get("gap_pro_cm")
        where = (f"{abs(gap):.1f} cm {'ahead of' if (hob.get('gap_pro_deg') or 0) > 0 else 'behind'} the Kona pros' middle"
                 if isinstance(gap, (int, float)) else "")
        hob_html = (f"<div class='hob'><div class='big'>{_e(hob['angle_deg'])}°</div><div><h3>Hip over the crank</h3>"
                    f"<p style='margin:0'>Measured from the back, over the crank axle. {_e(where)}. Pros 88°, age-groupers 85°.</p></div></div>")
    legend = ("<div class='legend'><span><i><svg width='26' height='10'><rect x='1' y='1' width='24' height='8' rx='2' fill='#E6F4EC' stroke='#2E9E5B'/></svg></i>Flapp range</span>"
              + ("<span><i><svg width='26' height='10'><line x1='2' y1='5' x2='24' y2='5' stroke='#14294B' stroke-width='3' stroke-linecap='round'/></svg></i>Kona 2026 pros, middle half</span>" if sport == "bike" else "")
              + "<span><i><svg width='12' height='12'><circle cx='6' cy='6' r='5' fill='#2F6DE0'/></svg></i>You</span></div>")
    kino = meta.get("kinogram")
    numbers = f"""
<section class="page">
  {head('The numbers', 'Every measurement, against its range')}
  <p class="lead">Measured by Flapp from your clip. Ranges come from sports-science research{' and the Kona 2026 pro field' if sport == 'bike' else ''}.</p>
  {ok_html}
  {hob_html}
  {legend if rows else ''}
  {f"<table class='nums'><tr><th>Measure</th><th>You</th><th></th><th>Range</th>{'<th>Pros</th>' if sport == 'bike' else ''}<th></th></tr>{num_rows}</table>" if rows else ''}
  {f'<div class="eyebrow" style="margin-top:12px">{"One stride, five positions" if sport == "run" else "One pedal revolution"}</div><figure class="fig"><img src="{kino}" alt="Kinogram"></figure>' if kino else ''}
  <div class="foot"><span>Flapp · getflapp.com</span><span>{_e(meta.get('prepared'))}</span></div>
</section>""" if (rows or kino) else ""

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Flapp report</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link href="https://fonts.googleapis.com/css2?family=Archivo:ital,wght@0,700;0,800;1,800;1,900&family=Manrope:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>{_CSS}{_EXTRA_CSS}</style></head><body>{cover}{read}{plan_sec}{numbers}</body></html>"""
