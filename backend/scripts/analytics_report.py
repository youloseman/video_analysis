#!/usr/bin/env python
"""The standing Flapp analytics report, straight from PostHog.

One command, always the same seven sections, so two runs a month apart are
comparable line by line. Everything here is a read -- it never writes to
PostHog.

    python backend/scripts/analytics_report.py                # all time
    python backend/scripts/analytics_report.py --days 30      # last 30 days
    python backend/scripts/analytics_report.py --md > r.md    # markdown out

Credentials, in order: POSTHOG_PERSONAL_KEY / POSTHOG_PROJECT_ID from the
environment, else mobile/.env (git-ignored). The personal key is a read key
(``phx_...``) and is NOT the ``phc_...`` ingestion key the browser ships.

Two caveats print with every report because they change how the numbers should
be read, and a number without them invites a wrong decision:
  * ad-blockers eat requests to posthog.com and the reverse proxy is not up,
    so every count here is a FLOOR, not a measurement;
  * our own browsing is in the data -- nothing filters the team out.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HOST = os.environ.get("POSTHOG_API_HOST", "https://us.posthog.com")

# Display prices, minor units -> dollars. Mirrors services/pricing.py, which is
# the source of truth; purchase events carry no amount, so revenue has to be
# reconstructed from the plan name. If a price moves there, move it here.
PRICE_USD = {
    "unlock": 4, "notes": 12, "expert": 39, "expert_recheck": 27,
    "enthusiast_monthly": 9, "enthusiast_yearly": 69, "full_yearly": 99,
}


def _creds() -> tuple[str, str]:
    key = os.environ.get("POSTHOG_PERSONAL_KEY")
    pid = os.environ.get("POSTHOG_PROJECT_ID")
    if key and pid:
        return key, pid
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
    return key, pid


KEY, PID = _creds()
MD = False


def q(hogql: str) -> list:
    req = urllib.request.Request(
        f"{HOST}/api/projects/{PID}/query/",
        data=json.dumps({"query": {"kind": "HogQLQuery", "query": hogql}}).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read().decode()).get("results", [])
    except urllib.error.HTTPError as e:
        sys.exit(f"PostHog HTTP {e.code}: {e.read().decode()[:400]}")


def head(t: str) -> None:
    print(f"\n## {t}\n" if MD else f"\n{'=' * 64}\n{t}\n{'=' * 64}")


def table(cols: list, rows: list, limit: int | None = None) -> None:
    rows = rows[:limit] if limit else rows
    if not rows:
        print("_no data_" if MD else "  (no data)")
        return
    if MD:
        print("| " + " | ".join(cols) + " |")
        print("|" + "|".join(["---"] * len(cols)) + "|")
        for r in rows:
            print("| " + " | ".join("-" if v is None else str(v) for v in r) + " |")
    else:
        w = [max([len(cols[i])] + [len(str(r[i])) for r in rows])
             for i in range(len(cols))]
        print("  " + "  ".join(c.ljust(w[i]) for i, c in enumerate(cols)))
        for r in rows:
            print("  " + "  ".join(("-" if v is None else str(v)).ljust(w[i])
                                   for i, v in enumerate(r)))


def pct(part, whole) -> str:
    return "-" if not whole else f"{part / whole * 100:.1f}%"


def main() -> None:
    global MD
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, help="window; omit for all time")
    ap.add_argument("--md", action="store_true", help="markdown output")
    a = ap.parse_args()
    MD = a.md
    where = f"timestamp >= now() - INTERVAL {a.days} DAY" if a.days else "1=1"
    PV = "'$pageview'"
    span = f"last {a.days} days" if a.days else "all time"

    print(f"# Flapp analytics - {span}" if MD else f"\nFLAPP ANALYTICS - {span}")

    # 1. Data health -------------------------------------------------------
    head("1. Data health")
    table(["events", "first", "last", "people"],
          q(f"SELECT count(), min(timestamp), max(timestamp), uniq(person_id) "
            f"FROM events WHERE {where}"))
    print("\n_Floor, not a measurement: ad-blockers drop posthog.com and the "
          "proxy is not up. Our own browsing is included._" if MD else
          "\n  NB: ad-blockers drop posthog.com (no proxy) -> counts are a FLOOR.\n"
          "      Our own browsing is NOT filtered out.")

    # 2. Reach -------------------------------------------------------------
    head("2. Reach")
    table(["week", "visitors", "pageviews", "signups", "analysed", "purchases"],
          q(f"""SELECT toStartOfWeek(timestamp) AS week,
                 uniqIf(person_id, event={PV}) AS visitors,
                 countIf(event={PV}) AS pageviews,
                 uniqIf(person_id, event='signed_up') AS signups,
                 uniqIf(person_id, event='analysis_started') AS analysed,
                 countIf(event='purchase_completed') AS purchases
                FROM events WHERE {where} GROUP BY week ORDER BY week"""))

    # 3. Acquisition -------------------------------------------------------
    head("3. Acquisition")
    table(["source", "visitors", "views"],
          q(f"""SELECT properties.$referring_domain, uniq(person_id), count()
                FROM events WHERE event={PV} AND {where}
                GROUP BY 1 ORDER BY 2 DESC"""), 12)
    print()
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
    print()
    table(["device", "visitors"],
          q(f"""SELECT properties.$device_type, uniq(person_id)
                FROM events WHERE event={PV} AND {where}
                GROUP BY 1 ORDER BY 2 DESC"""))

    # 5. Funnel ------------------------------------------------------------
    head("5. Funnel")
    f = q(f"""SELECT uniqIf(person_id, event={PV}),
               uniqIf(person_id, event='view_opened'),
               uniqIf(person_id, event='signed_up'),
               uniqIf(person_id, event='analysis_started'),
               uniqIf(person_id, event='analysis_completed'),
               uniqIf(person_id, event='paywall_shown'),
               uniqIf(person_id, event='checkout_started'),
               uniqIf(person_id, event='purchase_completed')
              FROM events WHERE {where}""")[0]
    names = ["visitor", "opened app", "signed up", "started analysis",
             "completed", "saw paywall", "checkout", "paid"]
    top = f[0] or 1
    table(["stage", "people", "of visitors", "of previous"],
          [[names[i], f[i], pct(f[i], top), pct(f[i], f[i - 1]) if i else "100%"]
           for i in range(len(f))])

    # 6. Revenue -----------------------------------------------------------
    head("6. Revenue")
    rows = q(f"""SELECT properties.plan, count(), uniq(person_id)
                 FROM events WHERE event='purchase_completed' AND {where}
                 GROUP BY 1 ORDER BY 2 DESC""")
    out, total, n = [], 0, 0
    for plan, cnt, ppl in rows:
        price = PRICE_USD.get(plan)
        rev = (price or 0) * cnt
        total += rev
        n += cnt
        out.append([plan, cnt, ppl, f"${price}" if price else "?", f"${rev}"])
    out.append(["TOTAL", n, "", "", f"${total}"])
    table(["plan", "purchases", "buyers", "price", "revenue"], out)
    if f[0]:
        print(f"\n  revenue per visitor ${total / f[0]:.2f}"
              f"   |   visitor->paid {pct(f[7], f[0])}")

    # 7. Product -----------------------------------------------------------
    head("7. Product")
    table(["sport", "mode", "runs", "people"],
          q(f"""SELECT properties.sport, properties.mode, count(), uniq(person_id)
                FROM events WHERE event='analysis_started' AND {where}
                GROUP BY 1,2 ORDER BY 3 DESC"""))
    print()
    g = q(f"""SELECT countIf(properties.gated = true), count()
              FROM events WHERE event='analysis_completed' AND {where}""")[0]
    print(f"  quality gate fired on {g[0]} of {g[1]} completed runs ({pct(g[0], g[1])})")
    print()
    table(["failure", "times", "people"],
          q(f"""SELECT properties.reason, count(), uniq(person_id)
                FROM events WHERE event='analysis_failed' AND {where}
                GROUP BY 1 ORDER BY 2 DESC"""), 8)


if __name__ == "__main__":
    main()
