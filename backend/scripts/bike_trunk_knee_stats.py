#!/usr/bin/env python
"""Bike clips on production: the knee near the saddle threshold, and the trunk
without the ear blend. Reads only; prints counts and quantiles, never a user.

Two questions from the hip-convention study (2026-11-17):

1. The pose model's hip sits forward of a fitter's trochanter marker, which
   opens the knee at the bottom by 5-12 deg against the convention the 138-145
   band is written in. How many stored clips sit where that decides the
   verdict (145-151 "acceptable", above 151 "too high")?
2. The trunk was read off a shoulder/ear blend until that day. Each report
   stores its points, so the plain shoulder-hip line is recomputed per clip:
   how far the number moves, and how many change band status.

Run it on the box that can reach the database:

    railway ssh --service video-analysis "python scripts/bike_trunk_knee_stats.py"
"""
from __future__ import annotations

import asyncio
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402

from app.core.db import SessionLocal  # noqa: E402
from app.models.analysis import Analysis  # noqa: E402
from app.services.video_analysis.biomechanics.cycling_positions import get_cycling_reference  # noqa: E402
from app.services.video_analysis.biomechanics.kona_field import KONA_PRO_FIELD  # noqa: E402

AERO = {"tt_aero", "triathlon"}
RTM_SINCE_MS = 1794009600000  # 2026-11-07 00:00 UTC


def q(xs, nd=1):
    if not xs:
        return "n=0"
    xs = sorted(xs)
    p = lambda f: xs[min(len(xs) - 1, int(f * len(xs)))]  # noqa: E731
    return (f"n={len(xs)} min {xs[0]:.{nd}f} p10 {p(.1):.{nd}f} p25 {p(.25):.{nd}f} "
            f"med {statistics.median(xs):.{nd}f} p75 {p(.75):.{nd}f} p90 {p(.9):.{nd}f} max {xs[-1]:.{nd}f}")


def trunk_pair(tl):
    """(old blended, new shoulder-hip) mean trunk angle from stored points."""
    try:
        side = tl.get("camera_side")
        w, h = float(tl["width"]), float(tl["height"])
        ear_i, sh_i, hip_i = ("7", "11", "23") if side == "left" else ("8", "12", "24")
        pts = tl["points"]
        ear, sh, hip = pts.get(ear_i), pts.get(sh_i), pts.get(hip_i)
        if not sh or not hip:
            return None
        old, new = [], []
        for i in range(min(len(sh), len(hip))):
            s, hp = sh[i], hip[i]
            if not s or not hp:
                continue
            sx, sy, hx, hy = s[0] * w, s[1] * h, hp[0] * w, hp[1] * h

            def ang(tx, ty):
                dx, dy = abs(tx - hx), hy - ty
                return math.degrees(math.atan2(dy, dx))
            n = ang(sx, sy)
            e = ear[i] if ear and i < len(ear) else None
            if e and (len(e) < 3 or e[2] is None or e[2] >= 0.4):
                o = ang(0.7 * sx + 0.3 * e[0] * w, 0.7 * sy + 0.3 * e[1] * h)
            else:
                o = n
            old.append(o)
            new.append(n)
        if len(new) < 10:
            return None
        return statistics.fmean(old), statistics.fmean(new)
    except Exception:
        return None


def status(v, band):
    return "low" if v < band[0] else "high" if v > band[1] else "in"


async def run() -> int:
    async with SessionLocal() as db:
        rows = (await db.execute(
            select(Analysis.kind, Analysis.created_at_ms, Analysis.result).where(Analysis.sport == "bike")
        )).all()
    n_all = len(rows)
    clips = []
    for kind, ms, res in rows:
        if kind == "photo" or not isinstance(res, dict):
            continue
        s = res.get("sport_specific_metrics") or res.get("summary") or {}
        if not s:
            continue
        clips.append((ms, res, s))
    print(f"bike rows {n_all}, clips with a stored result {len(clips)}")
    det = Counter((s.get("pose_detector") or "none") for _, _, s in clips)
    print("detector:", dict(det))
    pos = Counter((r.get("cycling_position") or "none") for _, r, _ in clips)
    print("position:", dict(pos))

    for label, sel in (("RTMPose clips", lambda s: str(s.get("pose_detector") or "").endswith("rtmpose_m")),
                       ("MediaPipe-only clips", lambda s: not str(s.get("pose_detector") or "").endswith("rtmpose_m"))):
        for grp, in_grp in (("aero", lambda p: p in AERO), ("road/other", lambda p: p not in AERO)):
            sub = [(r, s) for _, r, s in clips if sel(s) and in_grp(r.get("cycling_position"))]
            knees = [float(s["knee_at_bdc"]) for _, s in sub if isinstance(s.get("knee_at_bdc"), (int, float))]
            print(f"\n== {label}, {grp}: clips {len(sub)}")
            print("  knee_at_bdc:", q(knees))
            if knees:
                b = Counter("<138" if k < 138 else "138-145" if k <= 145 else "145-148" if k <= 148
                            else "148-151" if k <= 151 else "151-155" if k <= 155 else ">155" for k in knees)
                print("  knee buckets:", {k: b.get(k, 0) for k in ("<138", "138-145", "145-148", "148-151", "151-155", ">155")})
            print("  saddle verdict:", dict(Counter((s.get("saddle_height_assessment") or "none") for _, s in sub)))
            print("  saddle direction:", dict(Counter(((s.get("saddle_estimate") or {}).get("direction") or "none") for _, s in sub)))
            caps = Counter()
            for r, _ in sub:
                for c in ((r.get("score_coverage") or {}).get("capped_by") or []):
                    caps[c] += 1
            print("  score capped by:", dict(caps))
            tdc = [float(s["knee_at_tdc"]) for _, s in sub if isinstance(s.get("knee_at_tdc"), (int, float))]
            print("  knee_at_tdc:", q(tdc))

    # Trunk: stored value, and what the plain shoulder-hip line reads.
    print("\n== TRUNK without the ear blend (recomputed from stored points)")
    out = []
    methods = Counter()
    for ms, r, s in clips:
        m = ((s.get("diagnostics") or {}).get("trunk_angle_method") or {}).get("method") or "none"
        methods[m] += 1
        t = s.get("trunk_angle_avg")
        pair = trunk_pair(r.get("timeline") or {})
        if not isinstance(t, (int, float)) or pair is None:
            continue
        out.append((r.get("cycling_position"), float(t), pair[1] - pair[0], pair[0], pair[1]))
    print("  stored method:", dict(methods), "| clips with points:", len(out))
    print("  check, stored trunk minus recomputed old:", q([t - o for _, t, _, o, _ in out]))
    for grp, in_grp in (("aero", lambda p: p in AERO), ("road/other", lambda p: p not in AERO)):
        sub = [x for x in out if in_grp(x[0])]
        d = [x[2] for x in sub]
        print(f"  {grp}: shift new-old {q(d)}")
        print(f"     |shift|>=1: {sum(abs(x) >= 1 for x in d)}, >=2: {sum(abs(x) >= 2 for x in d)}, "
              f">=3: {sum(abs(x) >= 3 for x in d)}; up {sum(x > 0.5 for x in d)}, down {sum(x < -0.5 for x in d)}")
        flips = Counter()
        gap_old = gap_new = 0
        for p, t, dlt, _, _ in sub:
            try:
                band = get_cycling_reference(p)["trunk_angle"]
            except Exception:
                band = None
            if band:
                a, b = status(t, band), status(t + dlt, band)
                flips[f"{a}->{b}"] += 1
            if p in AERO:
                f = KONA_PRO_FIELD["trunk_angle_avg"]
                gap_old += t < f["lo"]
                gap_new += (t + dlt) < f["lo"]
        print("     band status old->new:", dict(flips))
        if grp == "aero":
            print(f"     below the whole pro field (12.2): old {gap_old}, new {gap_new}")
            print("     stored trunk:", q([x[1] for x in sub]), "| new:", q([x[1] + x[2] for x in sub]))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
