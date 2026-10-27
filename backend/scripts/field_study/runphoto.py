#!/usr/bin/env python3
"""Running from one photo: what the product reads, against points checked by hand.

The running photo analysis (photo_analyzer, behind RUN_CONTACT_METRICS) has
been checked on two real frames and thirteen treadmill photos. This is the
tool that measures how often it is right -- the running counterpart of
``hipknee.py``, and the editor the NYC article would be measured with.

    python backend/scripts/field_study/runphoto.py build     # detect + editor page
    python backend/scripts/field_study/runphoto.py editor    # editor page only
    python backend/scripts/field_study/runphoto.py compare   # product vs hand

``build`` reads every image in the SOURCES folders (a subfolder of
``upload/run_photos`` is a group: ``upload/run_photos/chicago/*.jpg`` ->
group "chicago"), skips byte-identical copies, and keeps photo numbers stable
across reruns. For each new photo it stores two things: the points the editor
starts from -- the near arm, the hip and the STANCE leg, picked by position the
way the product picks it (``_name_run_legs``) -- and what the product itself
reported on the photo (phase, angles, contact block), so ``compare`` never has
to run the model again.

The editor saves to its db collection ``points`` (doc ``pNN``: ``moved`` =
crop-pixel offset per key, ``phase`` = the instant Artur sees, ``group``,
``checked``). Pull those docs into ``<work>/points/pNN.json`` with the
ArtifactData tool before ``compare``.

Keys: S E W -- near shoulder, elbow, wrist; H -- near hip; K A L F -- knee,
ankle, heel and toe of the leg on (or about to land on) the ground.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from field import LEGS, NEAR, ROOT  # noqa: E402

STUDY = HERE / "runphoto2026.json"
WORK = ROOT / "upload" / "field-study" / "runphoto2026"
# (folder, group) -- a group of None takes the subfolder's name.
SOURCES = [
    (ROOT / "upload" / "photo_run", "treadmill"),
    (ROOT / "upload" / "run_photos", None),
]
MAX_W = 1100
KEYS = ("S", "E", "W", "H", "K", "A", "L", "F")
PHASES = ("initial_contact", "midstance", "toe_off", "swing", "flight", "unclear")
# Altman & Davis 2012, as photo_analyzer uses them.
STRIKE_HEEL_ABOVE, STRIKE_FORE_BELOW = 8.1, -1.6


def load() -> dict:
    if STUDY.exists():
        return json.loads(STUDY.read_text(encoding="utf-8"))
    return {"name": "runphoto2026", "editor_url": None, "work": WORK.relative_to(ROOT).as_posix(),
            "duplicates": [], "entries": []}


def save(study: dict) -> None:
    STUDY.write_text(json.dumps(study, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


# ---- the readings, from points in pixels (the editor computes the same) ----

def _ang(a, b, c) -> float:
    v1, v2 = (a[0] - b[0], a[1] - b[1]), (c[0] - b[0], c[1] - b[1])
    n = math.hypot(*v1) * math.hypot(*v2)
    return math.degrees(math.acos(max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / n)))) if n else math.nan


def readings(P: dict, facing: int) -> dict:
    """What a frame says, from S E W H K A L F in pixels.

    The same quantities, conventions and thresholds as photo_analyzer's
    running path, on the image plane: overstride |ankle_x - hip_x| over
    hip->knee->ankle, shin from vertical (+ = knee ahead of the ankle), trunk
    from vertical (+ = shoulder ahead of the hip), foot heel->toe against the
    horizontal (+ = toe up) and its Altman & Davis category.
    """
    S, E, W, H, K, A, L, F = (P[k] for k in KEYS)
    leg = math.dist(H, K) + math.dist(K, A)
    along = facing * (F[0] - L[0])
    foot = math.degrees(math.atan2(L[1] - F[1], along)) if along > 0 else math.nan
    strike = (None if math.isnan(foot) else "heel" if foot > STRIKE_HEEL_ABOVE
              else "forefoot" if foot < STRIKE_FORE_BELOW else "midfoot")
    return {
        "knee": _ang(H, K, A), "hip": _ang(S, H, K), "elbow": _ang(S, E, W),
        "trunk": math.degrees(math.atan2(facing * (S[0] - H[0]), H[1] - S[1])),
        "shin": math.degrees(math.atan2(facing * (K[0] - A[0]), A[1] - K[1])),
        "overstride": abs(A[0] - H[0]) / leg if leg else math.nan,
        "foot": foot, "strike": strike,
    }


# ---- build ----

def _detect(path: Path):
    """The product's own reading of the photo, and its stance-leg points."""
    import cv2
    import numpy as np
    from app.services.video_analysis import photo_analyzer as PA

    raw = path.read_bytes()
    try:
        r = PA.analyze_photo(raw, "run", contact_metrics=True)
        p = PA.detect_pose_in_image(raw)
    except ValueError:
        return None
    nl, _, legs = PA._name_run_legs(p["normalized"], p["world"], p["image"], p["camera_side"])
    side = p["camera_side"]
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    Hh, Ww = img.shape[:2]
    ids = {**NEAR[side], **LEGS[side]}
    px = {k: (nl[ids[k]].x * Ww, nl[ids[k]].y * Hh) for k in KEYS}
    allpts = [(lm.x * Ww, lm.y * Hh) for lm in nl]
    contact = r.get("contact") or {}
    auto = {
        "phase": r.get("gait_phase"),
        "knee": (r.get("angles") or {}).get("knee"), "hip": (r.get("angles") or {}).get("hip"),
        "elbow": (r.get("angles") or {}).get("elbow"), "trunk": (r.get("angles") or {}).get("trunk"),
        "overstride": contact.get("overstride_ratio"), "shin": contact.get("shin_angle"),
        "foot": contact.get("foot_angle_to_ground_deg"), "strike": contact.get("foot_strike"),
        "score": (r.get("score") or {}).get("overall_score"),
        "withheld": (r.get("score") or {}).get("withheld"),
        "stance": legs.get("stance"), "legs_clear": legs.get("clear"),
    }
    facing = 1 if px["F"][0] >= px["L"][0] else -1
    return side, facing, px, allpts, (Ww, Hh), auto


def _clean(v):
    return None if isinstance(v, float) and math.isnan(v) else (round(v, 2) if isinstance(v, float) else v)


def _add(path: Path, group: str, n: int, found) -> dict:
    from PIL import Image

    side, facing, px, allpts, (Ww, Hh), auto = found
    xs, ys = [p[0] for p in allpts], [p[1] for p in allpts]
    body = max(ys) - min(ys)
    pad = 0.15 * body
    x0, x1 = max(0, int(min(xs) - pad)), min(Ww, int(max(xs) + pad))
    y0, y1 = max(0, int(min(ys) - pad)), min(Hh, int(max(ys) + pad))
    if Ww <= 1600 and Hh <= 1600:      # a small frame is shown whole
        x0, y0, x1, y1 = 0, 0, Ww, Hh
    scale = min(1.0, MAX_W / (x1 - x0))
    img = Image.open(path).convert("RGB").crop((x0, y0, x1, y1))
    if scale < 1.0:
        img = img.resize((round((x1 - x0) * scale), round((y1 - y0) * scale)), Image.LANCZOS)
    (WORK / "img").mkdir(parents=True, exist_ok=True)
    img.save(WORK / "img" / f"p{n:02d}.webp", "WEBP", quality=88, method=6)
    model = {k: [round((x - x0) * scale, 1), round((y - y0) * scale, 1)] for k, (x, y) in px.items()}
    # The editor's formulas on the model's own points must give the product's
    # numbers back; if they do not, a convention differs and compare() would
    # be measuring that difference instead of the detector.
    mine = readings(model, facing)
    for k in ("knee", "elbow"):
        if auto.get(k) is not None and abs(mine[k] - auto[k]) > 1.5:
            print(f"  #{n} note: {k} on model points {mine[k]:.1f} vs product {auto[k]:.1f}")
    print(f"  #{n} {path.name}: {side}, phase {auto['phase']}, stance {auto['stance']}")
    return {"n": n, "photo": path.relative_to(ROOT).as_posix(), "group": group, "side": side,
            "facing": facing, "w": img.width, "h": img.height, "origin": [x0, y0],
            "scale": round(scale, 6), "model": model,
            "auto": {k: _clean(v) for k, v in auto.items()}}


def _sources():
    for folder, group in SOURCES:
        if not folder.exists():
            continue
        if group:
            yield from ((p, group) for p in sorted(folder.iterdir()))
        else:
            for sub in sorted(d for d in folder.iterdir() if d.is_dir()):
                yield from ((p, sub.name) for p in sorted(sub.iterdir()))


def build(study: dict) -> None:
    known = {e["photo"] for e in study["entries"]}
    seen = {hashlib.md5((ROOT / e["photo"]).read_bytes()).hexdigest(): e["photo"] for e in study["entries"]}
    n = max((e["n"] for e in study["entries"]), default=0)
    for path, group in _sources():
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp") or not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel in known:
            continue
        h = hashlib.md5(path.read_bytes()).hexdigest()
        if h in seen:
            if not any(d["photo"] == rel for d in study["duplicates"]):
                study["duplicates"].append({"photo": rel, "same_as": seen[h]})
            continue
        seen[h] = rel
        found = _detect(path)
        if found is None:
            print(f"  {path.name}: NO POSE -- not added")
            continue
        n += 1
        study["entries"].append(_add(path, group, n, found))
        save(study)
    print(f"{len(study['entries'])} photos, {len(study['duplicates'])} duplicates")


def editor(study: dict) -> Path:
    data = [{"n": e["n"], "photo": e["photo"], "img": f"img/p{e['n']:02d}.webp", "w": e["w"],
             "h": e["h"], "group": e["group"], "facing": e["facing"], "model": e["model"],
             "auto": e["auto"]} for e in study["entries"]]
    page = (HERE / "runphoto_editor.html").read_text(encoding="utf-8")
    out = WORK / "editor.html"
    WORK.mkdir(parents=True, exist_ok=True)
    out.write_text(page.replace("__DATA__", json.dumps(data, separators=(",", ":"))), encoding="utf-8")
    print(f"{out} ({len(data)} photos)")
    return out


# ---- compare ----

def compare(study: dict) -> dict:
    """Product (auto) against the hand-checked points, per reading.

    Angles and the ratio are compared on every checked photo; the contact
    block (overstride, shin, foot) only where Artur marked the frame as the
    foot landing -- it is defined nowhere else -- and the phase on all.
    """
    pdir = WORK / "points"
    rows = []
    for e in study["entries"]:
        f = pdir / f"p{e['n']:02d}.json"
        if not f.exists():
            continue
        rec = json.loads(f.read_text(encoding="utf-8"))
        if not rec.get("checked") or rec.get("group") == "excluded":
            continue
        mv = rec.get("moved") or {}
        P = {k: [e["model"][k][0] + mv.get(k, (0, 0))[0], e["model"][k][1] + mv.get(k, (0, 0))[1]] for k in KEYS}
        rows.append((e, rec.get("phase") or "unclear", readings(P, e["facing"])))
    out: dict = {"n": len(rows), "phase": {}, "errors": {}, "strike": {}}
    for _, phase, _ in rows:
        out["phase"].setdefault(phase, {})
    for e, phase, _ in rows:
        got = e["auto"].get("phase") or "uncertain"
        out["phase"][phase][got] = out["phase"][phase].get(got, 0) + 1
    contact = {"overstride", "shin", "foot"}
    for key in ("knee", "hip", "elbow", "trunk", "overstride", "shin", "foot"):
        diffs = [abs(e["auto"][key] - truth[key]) for e, phase, truth in rows
                 if e["auto"].get(key) is not None and truth[key] is not None
                 and not math.isnan(truth[key]) and (key not in contact or phase == "initial_contact")]
        if diffs:
            diffs.sort()
            out["errors"][key] = {"n": len(diffs), "median": round(statistics.median(diffs), 3),
                                  "p75": round(diffs[(3 * len(diffs)) // 4], 3),
                                  "max": round(diffs[-1], 3)}
    for e, phase, truth in rows:
        if phase == "initial_contact" and e["auto"].get("strike") and truth["strike"]:
            k = "agree" if e["auto"]["strike"] == truth["strike"] else f"{truth['strike']}->{e['auto']['strike']}"
            out["strike"][k] = out["strike"].get(k, 0) + 1
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("build", "editor", "compare"))
    args = ap.parse_args()
    study = load()
    if args.cmd == "build":
        build(study)
        editor(study)
    elif args.cmd == "editor":
        editor(study)
    else:
        compare(study)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
