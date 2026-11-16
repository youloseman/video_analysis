"""Check the parts-compatibility data (app/data/parts/frames).

    python scripts/check_parts_db.py            # schema + age
    python scripts/check_parts_db.py --links    # also open every URL (slow)

Exit 1 when a file is broken, a record is older than parts.STALE_DAYS, or
(with --links) a URL answers 404/410. Run monthly by
.github/workflows/parts-db-check.yml, which posts the output as an issue.

A 403/429 is reported but not failed: several makers block automated
requests, and a block is not a dead page. Never work around one.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services import parts  # noqa: E402

UA = "Mozilla/5.0 (compatible; FlappPartsCheck/1.0)"


def check_url(url: str) -> tuple[str, int | None]:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return "ok", r.status
    except urllib.error.HTTPError as e:
        if e.code == 405:  # HEAD refused: ask properly
            try:
                with urllib.request.urlopen(
                    urllib.request.Request(url, headers={"User-Agent": UA}), timeout=15,
                ) as r:
                    return "ok", r.status
            except urllib.error.HTTPError as e2:
                return ("dead" if e2.code in (404, 410) else "blocked"), e2.code
            except Exception:  # noqa: BLE001
                return "error", None
        return ("dead" if e.code in (404, 410) else "blocked"), e.code
    except Exception:  # noqa: BLE001 -- timeouts, TLS, DNS
        return "error", None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--links", action="store_true")
    args = ap.parse_args()

    failed = False
    files = sorted(parts.DATA_DIR.glob("*.json"))
    today = date.today()
    print(f"{len(files)} frame files in {parts.DATA_DIR}")
    for path in files:
        try:
            frame = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as e:
            print(f"BROKEN  {path.name}: {e}")
            failed = True
            continue
        errs = parts.problems(frame, path.name)
        for e in errs:
            print(f"SCHEMA  {e}")
        failed |= bool(errs)
        n = len(frame.get("parts") or [])
        by = {s: sum(1 for p in frame.get("parts") or [] if p.get("status") == s) for s in parts.STATUSES}
        flag = "STALE " if parts.stale(frame, today) else "ok    "
        failed |= flag.startswith("STALE")
        print(f"{flag}  {path.name}: checked {frame.get('checked')}, {n} parts "
              + ", ".join(f"{k} {v}" for k, v in by.items() if v))
        if args.links:
            urls = {e["url"] for p in frame.get("parts") or [] for e in p.get("evidence") or []}
            urls |= {p["url"] for p in frame.get("parts") or [] if p.get("url")}
            for url in sorted(urls):
                state, code = check_url(url)
                if state != "ok":
                    print(f"  {state.upper():7} {code or '-'}  {url}")
                failed |= state == "dead"
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
