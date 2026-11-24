#!/usr/bin/env python
"""What stored results say about the analyzer: where athletes moved our joint
points, and where the analysis broke. Reads only; prints counts and quantiles,
never a user, an email, a clip or a comment.

Two questions (Artur, 2026-11-23):

1. **Model error, from the athletes' own corrections.** A paid athlete can drag
   a joint the model put in the wrong place; the offset is stored with the
   result, beside the automatic reading it replaced (``baseline``). Across all
   of them: which joints get moved, which way, how far (in % of the thigh), and
   what that did to the numbers the report grades.
2. **Where analyses fail.** Score withheld and why, the capture report's checks
   (framing, camera view, drive side, motion, duration...), frame rate and
   resolution, and how bad result feedback lines up with each.

Staff, admin and test accounts are left out. Run it where the database is:

    railway ssh --service video-analysis "python scripts/result_quality_stats.py"
    railway ssh --service video-analysis "python scripts/result_quality_stats.py --days 60"
"""
from __future__ import annotations

import argparse
import asyncio
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402

from app.core.db import SessionLocal  # noqa: E402
from app.models.analysis import Analysis  # noqa: E402
from app.models.feedback import Feedback  # noqa: E402
from app.models.user import TIER_ADMIN, User  # noqa: E402
from app.services.newsletter import is_test_address  # noqa: E402

AERO = {"tt_aero", "triathlon"}
THIGH_CM = 43.0   # adult femur ~0.245 x 175 cm: only to put a % of thigh into cm
JOINT = {7: "ear", 8: "ear", 11: "shoulder", 12: "shoulder", 13: "elbow", 14: "elbow",
         15: "wrist", 16: "wrist", 23: "hip", 24: "hip", 25: "knee", 26: "knee",
         27: "ankle", 28: "ankle", 29: "heel", 30: "heel", 31: "toe", 32: "toe"}
GRADED = ("knee_at_bdc", "knee_at_tdc", "trunk_angle_avg", "hip_angle_avg", "elbow_angle_avg",
          "shoulder_angle_avg", "forearm_tilt_avg")


def q(xs, nd=1):
    xs = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]
    if not xs:
        return "n=0"
    xs = sorted(xs)
    p = lambda f: xs[min(len(xs) - 1, int(f * len(xs)))]  # noqa: E731
    return (f"n={len(xs)} p10 {p(.1):.{nd}f} p25 {p(.25):.{nd}f} med {statistics.median(xs):.{nd}f} "
            f"p75 {p(.75):.{nd}f} p90 {p(.9):.{nd}f}")


def pct(a, b):
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else f"{a}/0"


def summary_of(res: dict) -> dict:
    return res.get("sport_specific_metrics") or res.get("summary") or {}


def thigh_px(res: dict, side: str) -> float | None:
    """Median near-side hip-knee length in pixels, from the stored timeline."""
    tl = res.get("timeline") or {}
    try:
        w, h = float(tl["width"]), float(tl["height"])
        hip_i, knee_i = ("23", "25") if side == "left" else ("24", "26")
        hip, knee = tl["points"].get(hip_i), tl["points"].get(knee_i)
        ds = [math.hypot((a[0] - b[0]) * w, (a[1] - b[1]) * h)
              for a, b in zip(hip or [], knee or []) if a and b]
        return statistics.median(ds) if len(ds) >= 5 else None
    except (KeyError, TypeError, ValueError):
        return None


def frame_wh(res: dict) -> tuple[float, float] | None:
    tl = res.get("timeline") or {}
    try:
        return float(tl["width"]), float(tl["height"])
    except (KeyError, TypeError, ValueError):
        return None


def family(res: dict) -> str:
    p = res.get("cycling_position") or "none"
    return "aero" if p in AERO else ("road/other" if p != "none" else "unknown")


def corrections_section(rows) -> None:
    print("\n" + "=" * 78 + "\n1. ATHLETES' CORRECTIONS: where the model put the joint wrong\n" + "=" * 78)
    corrected = [(k, r, u) for k, r, u in rows if r.get("corrections")]
    by_kind = Counter(k for k, _, _ in rows)
    by_kind_c = Counter(k for k, _, _ in corrected)
    for kind in sorted(by_kind):
        print(f"  {kind}: corrected {pct(by_kind_c.get(kind, 0), by_kind[kind])} results, "
              f"{len({u for k, _, u in corrected if k == kind})} athletes")
    if not corrected:
        return
    # one entry per (result, joint): dx toward the rider's FRONT, dy DOWN, both in % of thigh
    per_joint = defaultdict(lambda: {"fwd": [], "down": [], "dist": [], "fwd_f": [], "down_f": []})
    det_moved = Counter()
    for kind, res, _ in corrected:
        side = res.get("camera_side") or (summary_of(res).get("camera_side")) or "left"
        front = -1.0 if side == "left" else 1.0     # filming the left side, the rider faces image-left
        wh = frame_wh(res)
        th = thigh_px(res, side)
        det = str(summary_of(res).get("pose_detector") or "none")
        for c in res.get("corrections") or []:
            name = JOINT.get(c.get("landmark"), f"lm{c.get('landmark')}")
            dx, dy = float(c.get("dx") or 0), float(c.get("dy") or 0)
            j = per_joint[(kind, name)]
            j["fwd_f"].append(100 * dx * front)          # % of frame width
            j["down_f"].append(100 * dy)                 # % of frame height
            if wh and th:
                fx, fy = dx * wh[0] * front, dy * wh[1]
                j["fwd"].append(100 * fx / th)
                j["down"].append(100 * fy / th)
                j["dist"].append(100 * math.hypot(fx, fy) / th)
            det_moved[(det.endswith("rtmpose_m"), name)] += 1
    print("\n  Offset the athlete applied, per joint (+fwd = toward the rider's front, +down = lower).")
    print("  In % of thigh length where the clip stored its skeleton; ~cm at a 43 cm thigh.")
    for (kind, name), j in sorted(per_joint.items(), key=lambda kv: (-len(kv[1]["fwd_f"]), kv[0])):
        n = len(j["fwd_f"])
        print(f"\n  [{kind}] {name}: moved in {n} results")
        if j["dist"]:
            print(f"     distance %thigh  {q(j['dist'])}   ~cm med {statistics.median(j['dist']) * THIGH_CM / 100:.1f}")
            print(f"     forward  %thigh  {q(j['fwd'])}")
            print(f"     down     %thigh  {q(j['down'])}")
        else:
            print(f"     forward %frame-w {q(j['fwd_f'])}")
            print(f"     down    %frame-h {q(j['down_f'])}")
        print(f"     direction: fwd {sum(x > 0.3 for x in j['fwd_f'])}, back {sum(x < -0.3 for x in j['fwd_f'])}, "
              f"down {sum(x > 0.3 for x in j['down_f'])}, up {sum(x < -0.3 for x in j['down_f'])}")
    print("\n  Which detector the moved joints came from (RTMPose since 2026-11-07):")
    for (rtm, name), n in sorted(det_moved.items(), key=lambda kv: -kv[1]):
        print(f"     {'RTMPose ' if rtm else 'MediaPipe'} {name}: {n}")

    # What the correction did to the graded numbers: corrected minus automatic.
    print("\n  Effect on the report: corrected minus automatic (baseline), video results")
    deltas = defaultdict(list)
    fams = defaultdict(lambda: defaultdict(list))
    verdict = Counter()
    score_d = []
    for kind, res, _ in corrected:
        base = (res.get("baseline") or {})
        bm = base.get("metrics") or {}
        s = summary_of(res)
        for k in GRADED:
            a, b = bm.get(k), s.get(k)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                deltas[k].append(b - a)
                fams[family(res)][k].append(b - a)
        a, b = bm.get("saddle_height_assessment"), s.get("saddle_height_assessment")
        if a and b:
            verdict[f"{a} -> {b}"] += 1
        a, b = base.get("technique_score"), res.get("technique_score")
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            score_d.append(b - a)
    for k, xs in deltas.items():
        print(f"     {k:22s} {q(xs)}   |d|>=3: {sum(abs(x) >= 3 for x in xs)}")
    for fam, d in fams.items():
        kb = d.get("knee_at_bdc") or []
        if kb:
            print(f"     knee_at_bdc on {fam}: {q(kb)}")
    print(f"     technique score: {q(score_d, 0)}")
    print(f"     saddle verdict changed by the athlete: {dict(verdict.most_common(10))}")


def failures_section(rows, fb_by_client: dict) -> None:
    print("\n" + "=" * 78 + "\n2. WHERE ANALYSES FAIL\n" + "=" * 78)
    groups = defaultdict(list)
    for kind, res, _, cid, sport in rows:
        groups[(sport or res.get("sport_type") or "?", kind)].append((res, cid))
    for (sport, kind), items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        n = len(items)
        print(f"\n--- {sport} / {kind}: {n} results")
        wh = Counter(((r.get("score_withheld") or {}).get("reason") or "scored") for r, _ in items)
        print("  score:", {k: pct(v, n) for k, v in wh.most_common()})
        reasons = Counter()
        for r, _ in items:
            for x in ((summary_of(r).get("quality_gate") or {}).get("reasons") or []):
                reasons[str(x)[:90]] += 1
        if reasons:
            print("  quality-gate reasons:")
            for k, v in reasons.most_common(8):
                print(f"     {v:4d}  {k}")
        cr = [summary_of(r).get("capture_report") or {} for r, _ in items]
        print("  capture verdict:", dict(Counter(c.get("verdict") or "none" for c in cr).most_common()))
        checks = defaultdict(Counter)
        for c in cr:
            for ch in c.get("checks") or []:
                checks[ch.get("id") or ch.get("label") or "?"][ch.get("status") or "?"] += 1
        for cid_, st in sorted(checks.items(), key=lambda kv: -(kv[1].get("bad", 0) + kv[1].get("warn", 0))):
            tot = sum(st.values())
            print(f"     check {cid_:16s} bad {pct(st.get('bad', 0), tot):>14s}  warn {pct(st.get('warn', 0), tot):>14s}")
        heads = Counter((c.get("headline") or "")[:90] for c in cr if c.get("headline"))
        if heads:
            print("  most common 'refilm' advice:")
            for k, v in heads.most_common(5):
                print(f"     {v:4d}  {k}")
        # numbers about the recording
        fps, eff, short_side, dur, slow, down = [], [], [], [], [], 0
        for r, _ in items:
            s = summary_of(r)
            smp = s.get("sampling") or {}
            if isinstance(smp.get("video_fps"), (int, float)):
                fps.append(float(smp["video_fps"]))
            if isinstance(smp.get("effective_fps"), (int, float)):
                eff.append(float(smp["effective_fps"]))
            down += bool(smp.get("downsampled"))
            w = frame_wh(r)
            if w:
                short_side.append(min(w))
            if isinstance(smp.get("duration_s"), (int, float)):
                dur.append(float(smp["duration_s"]))
            sm = s.get("slow_motion_factor")
            if isinstance(sm, (int, float)):
                slow.append(float(sm))
        if fps:
            print(f"  video fps: {q(fps, 0)}   analysed fps: {q(eff, 0)}   downsampled: {pct(down, len(items))}")
        if short_side:
            print(f"  frame short side px (analysis frame): {q(short_side, 0)}")
        if dur:
            print(f"  duration s: {q(dur)}")
        if slow:
            print(f"  slow-motion factor: {q(slow)}")
        if sport == "bike":
            print("  camera side:", dict(Counter((r.get("camera_side") or summary_of(r).get("camera_side") or "?")
                                                for r, _ in items)))
            print("  position:", dict(Counter((r.get("cycling_position") or "none") for r, _ in items).most_common(8)))
        print("  detector:", dict(Counter(str(summary_of(r).get("pose_detector") or "none") for r, _ in items)))
        # feedback against the capture verdict
        rated = [(r, fb_by_client[cid]) for r, cid in items if cid in fb_by_client]
        if rated:
            print(f"  result feedback on {len(rated)} of these:")
            by_v = defaultdict(list)
            for r, fbs in rated:
                v = (summary_of(r).get("capture_report") or {}).get("verdict") or "none"
                for f in fbs:
                    if isinstance(f["rating"], (int, float)):
                        by_v[v].append(f["rating"])
            for v, rs in by_v.items():
                low = sum(x <= 2 for x in rs)
                print(f"     capture {v:8s} ratings {q(rs, 1)}   rated <=2: {pct(low, len(rs))}")
            tags = Counter(t for _, fbs in rated for f in fbs for t in (f["tags"] or []))
            print("     tags:", dict(tags.most_common(12)))


async def run(days: int | None, include_staff: bool = False) -> int:
    since = int((time.time() - days * 86400) * 1000) if days else 0
    async with SessionLocal() as db:
        users = {uid: (tier, email) for uid, tier, email in
                 (await db.execute(select(User.id, User.tier, User.email))).all()}
        rows = (await db.execute(
            select(Analysis.user_id, Analysis.kind, Analysis.client_id, Analysis.result, Analysis.sport)
            .where(Analysis.created_at_ms >= since)
        )).all()
        fbs = (await db.execute(
            select(Feedback.analysis_client_id, Feedback.rating, Feedback.tags, Feedback.user_id)
            .where(Feedback.created_at_ms >= since)
        )).all()

    def staff(uid) -> bool:
        if include_staff:
            return False
        tier, email = users.get(uid, (None, None))
        return tier == TIER_ADMIN or is_test_address(email)

    kept = [(uid, kind or "video", cid, res, sport or res.get("sport_type")) for uid, kind, cid, res, sport in rows
            if isinstance(res, dict) and not staff(uid)]
    print(f"results {len(rows)}, with a stored result and not staff/test {len(kept)}"
          + (f", last {days} days" if days else ", all time"))
    print(f"athletes: {len({u for u, *_ in kept})}")
    fb_by_client = defaultdict(list)
    for cid, rating, tags, uid in fbs:
        if cid and not staff(uid):
            fb_by_client[cid].append({"rating": rating, "tags": tags})

    bike = [(kind, res, uid) for uid, kind, cid, res, sport in kept if sport == "bike"]
    corrections_section(bike)
    failures_section([(kind, res, uid, cid, sport) for uid, kind, cid, res, sport in kept], fb_by_client)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=None, help="only results from the last N days")
    ap.add_argument("--include-staff", action="store_true",
                    help="local smoke test only: keep admin/test accounts")
    a = ap.parse_args()
    sys.exit(asyncio.run(run(a.days, a.include_staff)))
