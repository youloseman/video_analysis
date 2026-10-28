#!/usr/bin/env python
"""How the overstride band (0-0.15 leg lengths) treats real runners. Reads only.

On 2026-10-28 three Chicago race touchdowns, Kiptum's among them, all came
back red on overstride. Before the band moves, this prints what the clips
people actually uploaded measure: the distribution, the share outside the
band, and the same split by cadence (the speed proxy a clip carries) and by
foot strike. Photos read behind the run-photo flag are listed too.

Run it on the box that can reach the database:

    railway ssh --service video-analysis "python scripts/run_overstride_stats.py"
"""

from __future__ import annotations

import asyncio
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402

from app.core.db import SessionLocal  # noqa: E402
from app.models.analysis import Analysis  # noqa: E402

BAND_HI = 0.15


def _q(xs: list[float]) -> str:
    if not xs:
        return "n=0"
    xs = sorted(xs)
    pct = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]  # noqa: E731
    out = sum(1 for x in xs if x > BAND_HI)
    return (f"n={len(xs)}  p10 {pct(.1):.2f}  p25 {pct(.25):.2f}  median {statistics.median(xs):.2f}  "
            f"p75 {pct(.75):.2f}  p90 {pct(.9):.2f}  >{BAND_HI}: {out} ({out / len(xs):.0%})")


async def run() -> int:
    async with SessionLocal() as db:
        rows = (await db.execute(
            select(Analysis.id, Analysis.kind, Analysis.result).where(Analysis.sport == "run")
        )).all()
    video: list[tuple[float, float | None, str | None, bool]] = []
    photo: list[tuple[float, float | None, str | None]] = []
    for _id, kind, res in rows:
        res = res or {}
        if kind == "photo":
            c = res.get("contact") or {}
            if isinstance(c.get("overstride_ratio"), (int, float)):
                photo.append((c["overstride_ratio"], c.get("shin_angle"), c.get("foot_strike")))
            continue
        s = res.get("summary") or {}
        o = s.get("overstride_ratio")
        if not isinstance(o, (int, float)):
            continue
        video.append((float(o), s.get("cadence_spm"), s.get("foot_strike"),
                      bool(s.get("overstride_ratio_estimated"))))

    print(f"run analyses: {len(rows)}; video with overstride: {len(video)}; photo touchdowns: {len(photo)}")
    print("\nVIDEO all:        ", _q([v[0] for v in video]))
    print("VIDEO measured:   ", _q([v[0] for v in video if not v[3]]))
    for lo, hi in ((0, 160), (160, 170), (170, 180), (180, 300)):
        print(f"VIDEO cadence {lo}-{hi}:".ljust(20),
              _q([v[0] for v in video if isinstance(v[1], (int, float)) and lo <= v[1] < hi]))
    for fs in ("heel", "midfoot", "forefoot"):
        print(f"VIDEO {fs}:".ljust(20), _q([v[0] for v in video if v[2] == fs]))
    print("\nPHOTO touchdowns: ", _q([p[0] for p in photo]))
    for o, shin, fs in sorted(photo):
        print(f"  overstride {o:.2f}  shin {shin if shin is not None else '-'}  strike {fs or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
