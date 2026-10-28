#!/usr/bin/env python
"""The standing Flapp analytics report, straight from PostHog.

One command, always the same sections, so two runs a month apart are
comparable line by line. Everything here is a read -- it never writes to
PostHog.

    python backend/scripts/analytics_report.py                # all time
    python backend/scripts/analytics_report.py --days 30      # last 30 days
    python backend/scripts/analytics_report.py --md > r.md    # markdown out
    python backend/scripts/analytics_report.py --html > r.html  # the email, to look at
    python backend/scripts/analytics_report.py --email        # send it to ADMIN_EMAIL

Every run opens with "Last 7 days": the seven whole Pacific days before today
against the seven before those. Whole days on purpose -- the Monday email then
covers Mon-Sun and the Friday one Fri-Thu, and neither ends on a half-day that
reads as a collapse. Then the standing sections over --days (default all
time), the eighth being the pricing funnel. .github/workflows/analytics-report.yml sends it Monday and Friday.

Credentials, in order: POSTHOG_PERSONAL_KEY / POSTHOG_PROJECT_ID from the
environment, else mobile/.env (git-ignored). The personal key is a read key
(``phx_...``) and is NOT the ``phc_...`` ingestion key the browser ships.
Sending needs EMAIL_API_KEY (Resend), EMAIL_FROM and ADMIN_EMAIL -- the same
names the app reads on Railway.

Two caveats print with every report because they change how the numbers should
be read, and a number without them invites a wrong decision:
  * ad-blockers eat requests to posthog.com and the reverse proxy is not up,
    so every count here is a FLOOR, not a measurement;
  * our own browsing is in the data -- nothing filters the team out.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import UTC, date, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HOST = os.environ.get("POSTHOG_API_HOST", "https://us.posthog.com")

# Display prices, minor units -> dollars. Mirrors services/pricing.py, which is
# the source of truth; purchase events carry no amount, so revenue has to be
# reconstructed from the plan name. If a price moves there, move it here.
# "expert" is the list price: a member buys it at $29 (Full) or $35
# (Enthusiast) under the same plan name, so an expert sale may read high.
PRICE_USD = {
    "unlock": 4, "notes": 12, "expert": 39, "expert_recheck": 27,
    "enthusiast_monthly": 9, "enthusiast_yearly": 69,
    "full_monthly": 13, "full_yearly": 99,
}

PV = "'$pageview'"
FUNNEL = [
    ("visitor", "$pageview"), ("opened app", "view_opened"),
    ("signed up", "signed_up"), ("started analysis", "analysis_started"),
    ("completed", "analysis_completed"), ("saw paywall", "paywall_shown"),
    ("checkout", "checkout_started"), ("paid", "purchase_completed"),
]
FUNNEL_SELECT = ", ".join(f"uniqIf(person_id, event='{e}')" for _, e in FUNNEL)

TZ = "America/Vancouver"
DAY = f"toDate(toTimeZone(timestamp, '{TZ}'))"
TODAY = f"toDate(toTimeZone(now(), '{TZ}'))"
THIS = f"{DAY} >= {TODAY} - 7 AND {DAY} < {TODAY}"
PREV = f"{DAY} >= {TODAY} - 14 AND {DAY} < {TODAY} - 7"
# The coarse timestamp bound is what lets PostHog skip partitions; the day
# bounds are what the report means.
RECENT = "timestamp >= now() - INTERVAL 16 DAY"
WEEK = f"{RECENT} AND {THIS}"
FORTNIGHT = f"{RECENT} AND {DAY} >= {TODAY} - 14 AND {DAY} < {TODAY}"

# Referrers that are us: our own pages, and the return from checkout.
OWN = ("getflapp.com", "checkout.stripe.com")

FLOOR_NOTE = ("Floor, not a measurement: ad-blockers drop posthog.com and the "
              "proxy is not up. Our own browsing is included.")

_creds_cache: tuple[str, str] | None = None


def _creds() -> tuple[str, str]:
    global _creds_cache
    if _creds_cache:
        return _creds_cache
    key = os.environ.get("POSTHOG_PERSONAL_KEY")
    pid = os.environ.get("POSTHOG_PROJECT_ID")
    if not (key and pid):
        env = REPO / "mobile" / ".env"
        if env.exists():
            cfg = {}
            for line in env.read_text(encoding="utf-8-sig").splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.split("=", 1)
                    cfg[k.strip()] = v.strip().strip('"').strip("'")
            key = key or cfg.get("POSTHOG_PERSONAL_KEY")
            pid = pid or cfg.get("POSTHOG_PROJECT_ID")
    if not key or not pid:
        sys.exit("No PostHog credentials. Set POSTHOG_PERSONAL_KEY and "
                 "POSTHOG_PROJECT_ID, or put them in mobile/.env")
    _creds_cache = (key, pid)
    return _creds_cache


def q(hogql: str) -> list:
    key, pid = _creds()
    req = urllib.request.Request(
        f"{HOST}/api/projects/{pid}/query/",
        data=json.dumps({"query": {"kind": "HogQLQuery", "query": hogql}}).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read().decode()).get("results", [])
    except urllib.error.HTTPError as e:
        sys.exit(f"PostHog HTTP {e.code}: {e.read().decode()[:400]}")


# --------------------------------------------------------------------------
# The report is built once as a list of blocks and rendered three ways, so
# the terminal, the markdown and the email can never disagree on a number.
# --------------------------------------------------------------------------
BLOCKS: list[tuple] = []


def part(t: str) -> None:
    BLOCKS.append(("part", t))


def head(t: str) -> None:
    BLOCKS.append(("head", t))


def label(t: str) -> None:
    BLOCKS.append(("label", t))


def note(t: str) -> None:
    BLOCKS.append(("note", t))


def bullets(items: list[str]) -> None:
    BLOCKS.append(("bullets", items))


def table(cols: list, rows: list, limit: int | None = None, bar: int | None = None) -> None:
    """``bar`` is the column the email draws a bar for; text ignores it."""
    BLOCKS.append(("table", cols, rows[:limit] if limit else rows, bar))


def pct(part, whole) -> str:
    return "-" if not whole else f"{part / whole * 100:.1f}%"


def delta(now, before) -> str:
    if not before:
        return "new" if now else "-"
    return f"{(now - before) / before * 100:+.0f}%"


def money(plan_counts) -> tuple[int, int]:
    """(revenue in dollars, purchases) from (plan, count) pairs."""
    rev = sum((PRICE_USD.get(p) or 0) * n for p, n in plan_counts)
    return rev, sum(n for _, n in plan_counts)


def _day(v) -> str:
    return date.fromisoformat(str(v)[:10]).strftime("%a %d %b")


def _when(v) -> str:
    return datetime.fromisoformat(str(v)).strftime("%a %d %b %H:%M")


def _cell(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


# -- renderers -------------------------------------------------------------
def render_text() -> str:
    out = []
    for b in BLOCKS:
        if b[0] == "part":
            out.append(f"\n{'#' * 64}\n{b[1].upper()}\n{'#' * 64}")
        elif b[0] == "head":
            out.append(f"\n{'=' * 64}\n{b[1]}\n{'=' * 64}")
        elif b[0] == "label":
            out.append(f"\n  -- {b[1]} --")
        elif b[0] == "note":
            out.append(f"\n  NB: {b[1]}")
        elif b[0] == "bullets":
            out.extend(f"  * {i}" for i in b[1])
        else:
            _, cols, rows, _bar = b
            if not rows:
                out.append("  (no data)")
                continue
            cells = [[_cell(v) for v in r] for r in rows]
            w = [max([len(cols[i])] + [len(r[i]) for r in cells]) for i in range(len(cols))]
            out.append("  " + "  ".join(c.ljust(w[i]) for i, c in enumerate(cols)))
            out.extend("  " + "  ".join(v.ljust(w[i]) for i, v in enumerate(r)) for r in cells)
    return "\n".join(out) + "\n"


def render_md() -> str:
    out = []
    for b in BLOCKS:
        if b[0] == "part":
            out.append(f"\n# {b[1]}\n")
        elif b[0] == "head":
            out.append(f"\n## {b[1]}\n")
        elif b[0] == "label":
            out.append(f"\n**{b[1]}**\n")
        elif b[0] == "note":
            out.append(f"\n_{b[1]}_\n")
        elif b[0] == "bullets":
            out.extend(f"- {i}" for i in b[1])
        else:
            _, cols, rows, _bar = b
            if not rows:
                out.append("_no data_")
                continue
            out.append("| " + " | ".join(cols) + " |")
            out.append("|" + "|".join(["---"] * len(cols)) + "|")
            out.extend("| " + " | ".join(_cell(v) for v in r) + " |" for r in rows)
    return "\n".join(out) + "\n"


# Email HTML: inline styles only (Gmail strips <style>), the app's palette.
NAVY, BLUE, GREY, LINE = "#14294B", "#2F6DE0", "#5A6478", "#E3E7EE"
UP, DOWN = "#1F8A4C", "#C2412D"
_DELTA = re.compile(r"^[+-]\d+%$")


# Per-cell markup is kept to the minimum: Gmail clips a message past ~102 KB,
# and repeated inline styles were 90 KB of a 92 KB body. Colour, nowrap and the
# row rule come from the table and the <tr>; padding and alignment are
# the old attributes, which every client still honours.
def _td(v, first: bool) -> str:
    s = _cell(v)
    attr = "" if first else ' align="right"'
    if _DELTA.match(s):
        attr += f' style="color:{UP if s[0] == "+" else DOWN}"'
    return f"<td{attr}>{html.escape(s)}</td>"


def render_html(title: str, subtitle: str) -> str:
    o = [f'<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;'
         f'font-size:14px;line-height:1.5;color:{NAVY};max-width:680px">'
         f'<p style="font-size:12px;letter-spacing:.08em;text-transform:uppercase;'
         f'color:{BLUE};margin:0 0 4px">Flapp · PostHog</p>'
         f'<h1 style="font-size:22px;margin:0 0 4px">{html.escape(title)}</h1>'
         f'<p style="margin:0 0 8px;color:{GREY}">{html.escape(subtitle)}</p>']
    for b in BLOCKS:
        if b[0] == "part":
            o.append(f'<h2 style="font-size:19px;margin:32px 0 4px;padding-top:16px;'
                     f'border-top:2px solid {NAVY}">{html.escape(b[1])}</h2>')
        elif b[0] == "head":
            o.append(f'<h3 style="font-size:16px;margin:22px 0 6px">{html.escape(b[1])}</h3>')
        elif b[0] == "label":
            o.append(f'<p style="font-size:11px;letter-spacing:.06em;text-transform:uppercase;'
                     f'color:{GREY};margin:14px 0 4px">{html.escape(b[1])}</p>')
        elif b[0] == "note":
            o.append(f'<p style="font-size:12px;color:{GREY};margin:8px 0">'
                     f'{html.escape(b[1])}</p>')
        elif b[0] == "bullets":
            o.append('<ul style="margin:8px 0;padding-left:20px">'
                     + "".join(f'<li style="margin:3px 0">{html.escape(i)}</li>' for i in b[1])
                     + "</ul>")
        else:
            _, cols, rows, bar = b
            if not rows:
                o.append(f'<p style="color:{GREY};font-size:13px">no data</p>')
                continue
            top = max((r[bar] or 0 for r in rows), default=0) if bar is not None else 0
            o.append(f'<table cellpadding="5" style="border-collapse:collapse;font-size:13px;color:{NAVY};'
                     'white-space:nowrap;margin:4px 0 6px"><tr>'
                     + "".join(f'<th align="{"left" if i == 0 else "right"}" style="'
                               f'color:{GREY};font-weight:600;border-bottom:1px solid {NAVY};'
                               f'white-space:nowrap">{html.escape(c)}</th>'
                               for i, c in enumerate(cols))
                     + ("<th></th>" if top else "") + "</tr>")
            for r in rows:
                cells = "".join(_td(v, i == 0) for i, v in enumerate(r))
                if top:
                    w = max(1, round((r[bar] or 0) / top * 120))
                    cells += (f'<td><div style="width:{w}px;'
                              f'height:8px;background:{BLUE}"></div></td>')
                o.append(f'<tr style="border-bottom:1px solid {LINE}">{cells}</tr>')
            o.append("</table>")
    o.append("</div>")
    return "".join(o)


def send_email(subject: str, text: str, body_html: str, to: str) -> None:
    key, sender = os.environ.get("EMAIL_API_KEY"), os.environ.get("EMAIL_FROM")
    if not (key and sender and to):
        sys.exit("Cannot send: EMAIL_API_KEY, EMAIL_FROM and ADMIN_EMAIL (or --to) "
                 "must all be set")
    req = urllib.request.Request(
        "https://api.resend.com/emails",
        data=json.dumps({"from": sender, "to": [to], "subject": subject,
                         "text": text, "html": body_html}).encode(),
        # Resend sits behind Cloudflare, which answers the stdlib's default
        # agent with a 1010 bot challenge -- see services/notify.py.
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "User-Agent": "Flapp/1.0 (+https://getflapp.com)",
                 "Accept": "application/json"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            print(f"sent to {to}: {r.status}", file=sys.stderr)
    except urllib.error.HTTPError as e:
        sys.exit(f"Resend HTTP {e.code}: {e.read().decode()[:400]}")


# --------------------------------------------------------------------------
# Last 7 days
# --------------------------------------------------------------------------
def last_week() -> dict:
    """The detailed week. Returns the numbers the email's subject line needs."""
    lo, hi = q(f"SELECT {TODAY} - 7, {TODAY} - 1")[0]
    span = f"{_day(lo)} – {_day(hi)}"
    part(f"Last 7 days · {span}")

    wk = f"if({DAY} >= {TODAY} - 7, 'this', 'prev')"
    by = {r[0]: r[1:] for r in q(f"SELECT {wk} AS w, {FUNNEL_SELECT} "
                                 f"FROM events WHERE {FORTNIGHT} GROUP BY w")}
    cur, prev = by.get("this", [0] * len(FUNNEL)), by.get("prev", [0] * len(FUNNEL))
    plans = {"this": [], "prev": []}
    for w, plan, n in q(f"SELECT {wk}, properties.plan, count() FROM events "
                        f"WHERE event='purchase_completed' AND {FORTNIGHT} GROUP BY 1, 2"):
        plans[w].append((plan, n))
    rev, buys = money(plans["this"])
    rev0, buys0 = money(plans["prev"])

    src = q(f"""SELECT properties.$referring_domain AS s,
                 uniqIf(person_id, {THIS}), uniqIf(person_id, {PREV})
                FROM events WHERE event={PV} AND {FORTNIGHT}
                GROUP BY s ORDER BY 2 DESC, 3 DESC""")
    pages = q(f"""SELECT properties.$pathname AS p,
                   uniqIf(person_id, {THIS}), uniqIf(person_id, {PREV})
                  FROM events WHERE event={PV} AND {FORTNIGHT}
                  GROUP BY p ORDER BY 2 DESC, 3 DESC""")
    failed = q(f"SELECT uniqIf(person_id, event='analysis_failed') FROM events WHERE {WEEK}")[0][0]

    head("At a glance")
    pw, co = cur[5], cur[6]
    lines = [
        f"Visitors {cur[0]:,} ({delta(cur[0], prev[0])} on the 7 days before: {prev[0]:,})",
        f"Signed up {cur[2]} ({delta(cur[2], prev[2])}), started an analysis {cur[3]}, "
        f"paid {cur[7]} people",
        f"Revenue ${rev} from {buys} purchases (the 7 days before: ${rev0} from {buys0})",
        f"Paywall → checkout {pct(co, pw)} of {pw} people "
        f"(the 7 days before: {pct(prev[6], prev[5])})",
    ]
    # "$direct" is always first and says nothing about where to post next.
    ref = [r for r in src if r[1] and r[0] and r[0] != "$direct" and not r[0].endswith(OWN)]
    if ref and cur[0]:
        lines.append(f"Top referrer: {ref[0][0]} — {ref[0][1]:,} visitors "
                     f"({pct(ref[0][1], cur[0])} of the week)")
    if pages and cur[0]:
        lines.append(f"Top page: {pages[0][0]} — {pages[0][1]:,} visitors")
    if cur[3]:
        lines.append(f"Analysis failures hit {failed} of {cur[3]} people who started one "
                     f"({pct(failed, cur[3])})")
    bullets(lines)
    note(FLOOR_NOTE)

    head("Day by day")
    days = q(f"""SELECT {DAY} AS d, uniqIf(person_id, event={PV}),
                 uniqIf(person_id, event='view_opened'),
                 uniqIf(person_id, event='signed_up'),
                 uniqIf(person_id, event='analysis_started'),
                 uniqIf(person_id, event='paywall_shown'),
                 uniqIf(person_id, event='checkout_started'),
                 countIf(event='purchase_completed')
                FROM events WHERE {WEEK} GROUP BY d ORDER BY d""")
    day_plans: dict[str, list] = {}
    for d, plan, n in q(f"SELECT {DAY}, properties.plan, count() FROM events "
                        f"WHERE event='purchase_completed' AND {WEEK} GROUP BY 1, 2"):
        day_plans.setdefault(str(d), []).append((plan, n))
    table(["day (Pacific)", "visitors", "opened app", "signed up", "analysed",
           "paywall", "checkout", "purchases", "revenue"],
          [[_day(r[0]), *r[1:], f"${money(day_plans.get(str(r[0]), []))[0]}"] for r in days],
          bar=1)

    head("Funnel, against the 7 days before")
    table(["stage", "people", "of previous", "7 days before", "change"],
          [[FUNNEL[i][0], cur[i], pct(cur[i], cur[i - 1]) if i else "100%",
            prev[i], delta(cur[i], prev[i])] for i in range(len(FUNNEL))])
    # The preview card got its own checkout button in 820d698; plan_buy says
    # which card a person saw. Read the plan button against the old 2.8%.
    card = ("multiIf(properties.plan_buy = true, 'plan button', "
            "properties.sellable = true, '$4 unlock', 'price list only')")
    label("Which paywall card, and who went on to checkout this week")
    table(["card", "saw it", "then checkout", "rate"],
          [[c, n, k, pct(k, n)] for c, n, k in q(
              f"""SELECT {card} AS c, uniq(person_id),
                   uniqIf(person_id, person_id IN (
                     SELECT person_id FROM events
                     WHERE event='checkout_started' AND {WEEK}))
                  FROM events WHERE event='paywall_shown' AND {WEEK}
                  GROUP BY c ORDER BY 2 DESC""")])

    pricing(WEEK, "Pricing funnel")

    head("Where they came from")
    label("Sources")
    table(["source", "visitors", "7 days before", "change"],
          [[s or "$direct", a, b, delta(a, b)] for s, a, b in src], 10)
    label("Pages")
    table(["page", "visitors", "7 days before", "change"],
          [[p, a, b, delta(a, b)] for p, a, b in pages], 10)
    label("Sign-ups by the source that first brought them")
    table(["first source", "signed up"],
          q(f"""SELECT person.properties.$initial_referring_domain, uniq(person_id)
                FROM events WHERE event='signed_up' AND {WEEK}
                GROUP BY 1 ORDER BY 2 DESC"""), 8)
    label("Landing goal ('What brings you here?')")
    table(["goal", "landed", "signed up", "paid"],
          q(f"""SELECT properties.goal, uniq(person_id),
                 uniqIf(person_id, person_id IN (
                   SELECT person_id FROM events WHERE event='signed_up' AND {WEEK})),
                 uniqIf(person_id, person_id IN (
                   SELECT person_id FROM events WHERE event='purchase_completed' AND {WEEK}))
                FROM events WHERE event='entry_goal_landed' AND {WEEK}
                GROUP BY 1 ORDER BY 2 DESC"""))
    label("Countries")
    table(["country", "visitors", "7 days before", "change"],
          [[c, a, b, delta(a, b)] for c, a, b in q(
              f"""SELECT properties.$geoip_country_name,
                   uniqIf(person_id, {THIS}), uniqIf(person_id, {PREV})
                  FROM events WHERE event={PV} AND {FORTNIGHT}
                  GROUP BY 1 ORDER BY 2 DESC, 3 DESC""")], 10)
    label("Devices")
    table(["device", "visitors"],
          q(f"""SELECT properties.$device_type, uniq(person_id)
                FROM events WHERE event={PV} AND {WEEK} GROUP BY 1 ORDER BY 2 DESC"""))

    head("Purchases")
    table(["when (Pacific)", "plan", "price", "first source", "country", "goal"],
          [[_when(t), plan, f"${PRICE_USD[plan]}" if plan in PRICE_USD else "?", s, c, g]
           for t, plan, s, c, g in q(
               f"""SELECT toTimeZone(timestamp, '{TZ}'), properties.plan,
                    person.properties.$initial_referring_domain,
                    person.properties.$initial_geoip_country_name,
                    person.properties.entry_goal
                   FROM events WHERE event='purchase_completed' AND {WEEK}
                   ORDER BY timestamp""")])

    head("Product")
    table(["sport", "mode", "runs", "people"],
          q(f"""SELECT properties.sport, properties.mode, count(), uniq(person_id)
                FROM events WHERE event='analysis_started' AND {WEEK}
                GROUP BY 1, 2 ORDER BY 3 DESC"""))
    g = q(f"""SELECT countIf(properties.gated = true), count()
              FROM events WHERE event='analysis_completed' AND {WEEK}""")[0]
    note(f"Quality gate fired on {g[0]} of {g[1]} completed runs ({pct(g[0], g[1])}).")
    label("What the athlete saw fail (analysis_failed)")
    table(["reason", "times", "people"],
          q(f"""SELECT properties.reason, count(), uniq(person_id)
                FROM events WHERE event='analysis_failed' AND {WEEK}
                GROUP BY 1 ORDER BY 2 DESC"""), 8)
    # Anonymous server-side records (services/failure_log.py), so times only.
    label("Server crashes (analysis_error)")
    table(["error", "times"],
          [[f"{t}: {(e or '')[:90]}", n] for t, e, n in q(
              f"""SELECT properties.err_type, properties.err, count()
                  FROM events WHERE event='analysis_error' AND {WEEK}
                  GROUP BY 1, 2 ORDER BY 3 DESC""")], 6)
    label("Refused uploads (analysis_rejected)")
    table(["reason", "times"],
          [[(r or "")[:90], n] for r, n in q(
              f"""SELECT properties.reason, count()
                  FROM events WHERE event='analysis_rejected' AND {WEEK}
                  GROUP BY 1 ORDER BY 2 DESC""")], 6)
    return {"span": span, "visitors": cur[0], "paid": cur[7], "revenue": rev}


# --------------------------------------------------------------------------
# Pricing funnel
#
# Prices are seen in two places: the #pricing section of the landing page and
# the app's pricing screen. Neither is on the way to every sale -- the paywall
# card sells the $4 unlock and (since 820d698) a plan without a detour -- so
# this is ORDERED per person: a checkout counts here only if it came after
# that person first saw prices. Built in Python from one row per person,
# because "after" is what a uniq() funnel cannot say.
# --------------------------------------------------------------------------
LANDING_PRICES = "(event='landing_section_viewed' AND properties.section='pricing')"
APP_PRICES = "(event='view_opened' AND properties.view='pricing')"


def pricing(where: str, title: str) -> None:
    head(title)
    steps = {"lp": LANDING_PRICES, "ap": APP_PRICES, "pw": "event='paywall_shown'",
             "co": "event='checkout_started'", "pd": "event='purchase_completed'"}
    cols = ", ".join(f"if(countIf({c}) > 0, minIf(timestamp, {c}), NULL)" for c in steps.values())
    # Timestamps come back as ISO strings in one format, so they compare as text.
    # The explicit LIMIT matters: HogQL returns 100 rows unless told otherwise,
    # and one row per person passes that within a week.
    people = [{"id": r[0], **dict(zip(steps, r[1:]))} for r in q(
        f"""SELECT person_id, {cols} FROM events
            WHERE {where} AND ({LANDING_PRICES} OR {APP_PRICES} OR event IN
              ('paywall_shown', 'checkout_started', 'purchase_completed'))
            GROUP BY person_id
            HAVING countIf({LANDING_PRICES} OR {APP_PRICES} OR event IN
              ('checkout_started', 'purchase_completed')) > 0
            LIMIT 10000""")]
    visitors = q(f"SELECT uniq(person_id) FROM events WHERE event={PV} AND {where}")[0][0]
    for p in people:
        seen = [t for t in (p["lp"], p["ap"]) if t]
        p["pr"] = min(seen) if seen else None
        p["co_after"] = bool(p["pr"] and p["co"] and p["co"] >= p["pr"])
        p["pd_after"] = bool(p["co_after"] and p["pd"] and p["pd"] >= p["pr"])

    saw = [p for p in people if p["pr"]]
    co = sum(p["co_after"] for p in saw)
    pd = sum(p["pd_after"] for p in saw)
    table(["stage", "people", "of previous", "of visitors"],
          [["visitor", visitors, "100%", "100%"],
           ["saw prices", len(saw), pct(len(saw), visitors), pct(len(saw), visitors)],
           ["then checkout", co, pct(co, len(saw)), pct(co, visitors)],
           ["then paid", pd, pct(pd, co), pct(pd, visitors)]])

    def row(name, group):
        c, d = sum(p["co_after"] for p in group), sum(p["pd_after"] for p in group)
        return [name, len(group), c, pct(c, len(group)), d, pct(d, len(group))]

    label("Where they saw prices")
    table(["where", "people", "then checkout", "rate", "then paid", "rate"],
          [row("landing section only", [p for p in saw if p["lp"] and not p["ap"]]),
           row("app pricing screen only", [p for p in saw if p["ap"] and not p["lp"]]),
           row("both", [p for p in saw if p["lp"] and p["ap"]])])
    app = [p for p in saw if p["ap"]]
    label("How the app pricing screen was reached")
    table(["first", "people", "then checkout", "rate", "then paid", "rate"],
          [row("after a paywall", [p for p in app if p["pw"] and p["pw"] <= p["ap"]]),
           row("without a paywall (menu, link)",
               [p for p in app if not (p["pw"] and p["pw"] <= p["ap"])])])

    # Every sale, by whether its buyer had seen prices before paying.
    path = {p["id"]: p for p in people}
    by: dict[str, list] = {"via prices": [], "straight from a paywall card": []}
    for pid, plan, t in q(f"""SELECT person_id, properties.plan, timestamp FROM events
                              WHERE event='purchase_completed' AND {where}"""):
        p = path.get(pid)
        via = bool(p and p["pr"] and p["pr"] <= t)
        by["via prices" if via else "straight from a paywall card"].append((plan, 1))
    label("Every purchase, by path")
    table(["path", "purchases", "revenue"],
          [[k, money(v)[1], f"${money(v)[0]}"] for k, v in by.items()])


# --------------------------------------------------------------------------
# The standing sections
# --------------------------------------------------------------------------
def standing(days: int | None) -> None:
    where = f"timestamp >= now() - INTERVAL {days} DAY" if days else "1=1"
    part(f"Last {days} days" if days else "All time")

    # 1. Data health -------------------------------------------------------
    head("1. Data health")
    table(["events", "first", "last", "people"],
          q(f"SELECT count(), min(timestamp), max(timestamp), uniq(person_id) "
            f"FROM events WHERE {where}"))
    note(FLOOR_NOTE)

    # 2. Reach -------------------------------------------------------------
    head("2. Reach")
    table(["week", "visitors", "pageviews", "signups", "analysed", "purchases"],
          q(f"""SELECT toStartOfWeek(timestamp) AS week,
                 uniqIf(person_id, event={PV}) AS visitors,
                 countIf(event={PV}) AS pageviews,
                 uniqIf(person_id, event='signed_up') AS signups,
                 uniqIf(person_id, event='analysis_started') AS analysed,
                 countIf(event='purchase_completed') AS purchases
                FROM events WHERE {where} GROUP BY week ORDER BY week"""), bar=1)

    # 3. Acquisition -------------------------------------------------------
    head("3. Acquisition")
    table(["source", "visitors", "views"],
          q(f"""SELECT properties.$referring_domain, uniq(person_id), count()
                FROM events WHERE event={PV} AND {where}
                GROUP BY 1 ORDER BY 2 DESC"""), 12)
    table(["page", "visitors", "views"],
          q(f"""SELECT properties.$pathname, uniq(person_id), count()
                FROM events WHERE event={PV} AND {where}
                GROUP BY 1 ORDER BY 2 DESC"""), 12)

    # 4. Geography ---------------------------------------------------------
    head("4. Geography")
    table(["country", "visitors"],
          q(f"""SELECT properties.$geoip_country_name, uniq(person_id)
                FROM events WHERE event={PV} AND {where}
                GROUP BY 1 ORDER BY 2 DESC"""), 15)
    table(["device", "visitors"],
          q(f"""SELECT properties.$device_type, uniq(person_id)
                FROM events WHERE event={PV} AND {where}
                GROUP BY 1 ORDER BY 2 DESC"""))

    # 5. Funnel ------------------------------------------------------------
    head("5. Funnel")
    f = q(f"SELECT {FUNNEL_SELECT} FROM events WHERE {where}")[0]
    top = f[0] or 1
    table(["stage", "people", "of visitors", "of previous"],
          [[FUNNEL[i][0], f[i], pct(f[i], top), pct(f[i], f[i - 1]) if i else "100%"]
           for i in range(len(f))])

    # 6. Revenue -----------------------------------------------------------
    head("6. Revenue")
    rows = q(f"""SELECT properties.plan, count(), uniq(person_id)
                 FROM events WHERE event='purchase_completed' AND {where}
                 GROUP BY 1 ORDER BY 2 DESC""")
    out = []
    for plan, cnt, ppl in rows:
        price = PRICE_USD.get(plan)
        out.append([plan, cnt, ppl, f"${price}" if price else "?", f"${(price or 0) * cnt}"])
    total, n = money([(r[0], r[1]) for r in rows])
    out.append(["TOTAL", n, "", "", f"${total}"])
    table(["plan", "purchases", "buyers", "price", "revenue"], out)
    if f[0]:
        note(f"Revenue per visitor ${total / f[0]:.2f}   |   "
             f"visitor → paid {pct(f[7], f[0])}")

    # 7. Product -----------------------------------------------------------
    head("7. Product")
    table(["sport", "mode", "runs", "people"],
          q(f"""SELECT properties.sport, properties.mode, count(), uniq(person_id)
                FROM events WHERE event='analysis_started' AND {where}
                GROUP BY 1,2 ORDER BY 3 DESC"""))
    g = q(f"""SELECT countIf(properties.gated = true), count()
              FROM events WHERE event='analysis_completed' AND {where}""")[0]
    note(f"Quality gate fired on {g[0]} of {g[1]} completed runs ({pct(g[0], g[1])}).")
    table(["failure", "times", "people"],
          q(f"""SELECT properties.reason, count(), uniq(person_id)
                FROM events WHERE event='analysis_failed' AND {where}
                GROUP BY 1 ORDER BY 2 DESC"""), 8)

    # 8. Pricing -----------------------------------------------------------
    pricing(where, "8. Pricing funnel")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, help="window for the standing sections; omit for all time")
    out = ap.add_mutually_exclusive_group()
    out.add_argument("--md", action="store_true", help="markdown output")
    out.add_argument("--html", action="store_true", help="the email body, to stdout")
    out.add_argument("--email", action="store_true", help="send it (Resend)")
    ap.add_argument("--to", default=os.environ.get("ADMIN_EMAIL"), help="default: ADMIN_EMAIL")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")   # a Windows console is cp1252

    week = last_week()
    standing(a.days)

    title = f"Flapp analytics · {week['span']}"
    subtitle = (f"The last 7 whole days against the 7 before, then "
                f"{f'the last {a.days} days' if a.days else 'all time'}. "
                f"Generated {datetime.now(UTC).strftime('%a %d %b %Y %H:%M')} UTC.")
    if a.email:
        subject = (f"Flapp · {week['span']}: {week['visitors']:,} visitors, "
                   f"{week['paid']} paid, ${week['revenue']}")
        send_email(subject, f"{title}\n{subtitle}\n{render_text()}",
                   render_html(title, subtitle), a.to)
    elif a.html:
        print(render_html(title, subtitle))
    elif a.md:
        print(f"# {title}\n\n{subtitle}\n{render_md()}")
    else:
        print(f"\n{title.upper()}\n{subtitle}\n{render_text()}")


if __name__ == "__main__":
    main()
