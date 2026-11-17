"""The redesign's before/after numbers, from PostHog. Reads only.

Run once before a screen ships and again two weeks after; the same queries
over two windows are the only honest way to say a redesign helped.

    python scripts/redesign_kpis.py --since 2026-10-20 --until 2026-11-17 --save docs/redesign/baseline-2026-11-17.json

Every rate is per PERSON, not per event. Credentials as analytics_report.py
(env or mobile/.env). Counts are a floor: ad-blockers drop posthog.com.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analytics_report import FAILED, TZ, q  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def kpis(since: date, until: date) -> dict:
    """Window = [since, until) in Pacific days."""
    day = f"toDate(toTimeZone(timestamp, '{TZ}'))"
    w = (f"timestamp >= toDateTime('{since - timedelta(days=1)}') AND "
         f"{day} >= toDate('{since}') AND {day} < toDate('{until}')")

    def people(event_sql: str) -> int:
        return q(f"SELECT uniq(person_id) FROM events WHERE ({event_sql}) AND {w}")[0][0]

    def ev(name: str) -> str:
        return f"event='{name}'"

    done = people(ev("analysis_completed"))
    opened = people(ev("view_opened"))
    started = people(ev("analysis_started"))
    out: dict = {"window": {"since": str(since), "until": str(until), "days": (until - since).days},
                 "people": {"opened_app": opened, "signed_up": people(ev("signed_up")),
                            "started_analysis": started, "completed_analysis": done,
                            "saw_paywall": people(ev("paywall_shown")),
                            "checkout": people(ev("checkout_started"))}}

    def share(event_sql: str) -> dict:
        n = people(event_sql)
        return {"people": n, "of_completed": round(n / done, 4) if done else None}

    # After the result: what people do with it.
    out["after_result"] = {
        "pdf": share("event IN ('report_pdf_opened','report_pdf_downloaded')"),
        "share_card": share(ev("share_card_built")),
        "player": share(ev("player_used")),
        "feedback": share(ev("result_feedback")),
        "chat": share(ev("chat_opened")),
        "compare": share(ev("compare_opened")),
        "ai_export": share(ev("ai_export")),
    }
    # How far down the report people read (event exists since 2026-11-17).
    rows = q(f"""SELECT properties.section, uniq(person_id) FROM events
                 WHERE event='result_section_seen' AND {w} GROUP BY 1 ORDER BY 2 DESC LIMIT 50""")
    out["sections_seen"] = {str(k): {"people": n, "of_completed": round(n / done, 4) if done else None}
                            for k, n in rows}

    p = out["people"]
    out["rates"] = {
        "signup_of_opened": round(p["signed_up"] / opened, 4) if opened else None,
        "completed_of_started": round(done / started, 4) if started else None,
        "paywall_to_checkout": round(p["checkout"] / p["saw_paywall"], 4) if p["saw_paywall"] else None,
        "failed_of_started": round(people(FAILED) / started, 4) if started else None,
    }

    # Minutes from first opening the app to the first finished analysis, for
    # people whose first app open falls in the window.
    r = q(f"""SELECT quantile(0.5)(m), quantile(0.75)(m), count() FROM (
                SELECT person_id,
                  dateDiff('minute', minIf(timestamp, event='view_opened'),
                                     minIf(timestamp, event='analysis_completed')) AS m
                FROM events WHERE event IN ('view_opened','analysis_completed') AND {w}
                GROUP BY person_id
                HAVING countIf(event='analysis_completed') > 0 AND countIf(event='view_opened') > 0 AND m >= 0)""")[0]
    out["time_to_first_result_min"] = {"median": r[0], "p75": r[1], "people": r[2]}

    # Came back within 14 days of their first finished analysis (any event on a
    # later day), for first analyses at least 14 days before the window ends.
    cut = until - timedelta(days=14)
    if cut > since:
        r = q(f"""SELECT countIf(back), count() FROM (
                    SELECT f.person_id AS pid,
                      countIf(e.timestamp > f.first + INTERVAL 1 DAY AND e.timestamp <= f.first + INTERVAL 14 DAY) > 0 AS back
                    FROM (SELECT person_id, min(timestamp) AS first FROM events
                          WHERE event='analysis_completed' AND {w} GROUP BY person_id
                          HAVING toDate(toTimeZone(first, '{TZ}')) < toDate('{cut}')) AS f
                    JOIN events AS e ON e.person_id = f.person_id
                    WHERE e.timestamp >= toDateTime('{since - timedelta(days=1)}')
                    GROUP BY f.person_id) LIMIT 1""")[0]
        out["return_14d"] = {"returned": r[0], "people": r[1], "rate": round(r[0] / r[1], 4) if r[1] else None}
    else:
        out["return_14d"] = None

    out["device_of_completed"] = {str(k): n for k, n in q(
        f"""SELECT properties.$device_type, uniq(person_id) FROM events
            WHERE event='analysis_completed' AND {w} GROUP BY 1 ORDER BY 2 DESC LIMIT 10""")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", required=True, type=date.fromisoformat)
    ap.add_argument("--until", required=True, type=date.fromisoformat, help="exclusive")
    ap.add_argument("--save", help="write the JSON here (relative to the repo root)")
    a = ap.parse_args()
    res = kpis(a.since, a.until)
    text = json.dumps(res, indent=2, ensure_ascii=False)
    print(text)
    if a.save:
        path = REPO / a.save
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
