#!/usr/bin/env python3
"""Field studies: pro bike positions measured from race photographs.

The first is Kona 2026 (``kona2026.json``, the Academy article
kona-2026-pro-bike-positions). A study file lists every photo with the points
the model found, in the coordinates of the crop the point editor shows. Artur
checks the points in the editor artifact (``editor_url``); its database keeps
his offsets, one document per photo (collection ``corrections``, doc ``pNN``,
field ``moved`` = crop-pixel offset from the model's point per key). Pull them
into ``<work>/corrections/`` with the ArtifactData tool before ``rebuild``.

    python backend/scripts/field_study/field.py add upload/<photo>.jpg [--label NAME] [--race RACE]
    python backend/scripts/field_study/field.py editor
    python backend/scripts/field_study/field.py rebuild

``add`` appends a photo (it gets the next editor number, so the editor's saved
corrections keep pointing at the right photo) and rebuilds the editor page.
``editor`` writes ``<work>/editor.html`` + ``<work>/img/`` -- publish those to
``editor_url`` with the Artifact tool (files: img/pNN.webp). ``rebuild``
measures every photo with the checked points and writes the article figures,
the social exports, the field chart and ``<work>/rebuilt.json`` (stats +
captions) for the article text.

Each photo carries a group ("pro" / "age_group", or "excluded" -- a duplicate
rider, an oblique camera, a frame the model cannot read) and a category ("men" /
"women"), set in the editor (stored with its corrections) or given to ``add``.
The article is built from the pros only; rebuilt.json also breaks every metric
down by group and category, for the pro-vs-age-group and men-vs-women
comparisons once there are enough photos.

Keys per photo: S E W H are the near-side shoulder, elbow, wrist and hip;
B and T are the top of the back and the top of the helmet the helmet reading
is taken between (head_silhouette.py). Needs the backend requirements with
MediaPipe >= 1.0 -- the silhouette is unavailable on older builds -- and
Playwright with Chromium for the field chart.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import math
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))

KEYS = ("S", "E", "W", "H")
NEAR = {"right": {"S": 12, "E": 14, "W": 16, "H": 24}, "left": {"S": 11, "E": 13, "W": 15, "H": 23}}
SHOWN = ("trunk", "shoulder", "elbow", "forearm_tilt", "helmet_above_back")
CROP_MAX_W = 900  # editor crops are downscaled to this; ``scale`` records by how much


def load(name: str) -> dict:
    return json.loads((HERE / f"{name}.json").read_text(encoding="utf-8"))


def save(study: dict) -> None:
    (HERE / f"{study['name']}.json").write_text(
        json.dumps(study, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def work_dir(study: dict) -> Path:
    d = ROOT / study["work"]
    (d / "img").mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- add / editor

def _detect(photo: Path):
    from app.services.video_analysis import photo_analyzer as PA
    p = PA.detect_pose_in_image(photo.read_bytes())
    if not p.get("silhouette_top"):
        raise SystemExit("no silhouette: run with MediaPipe >= 1.0 (backend/requirements.txt)")
    return p


def add(study: dict, photo: str, label: str | None, race: str,
        group: str = "pro", category: str | None = None) -> dict:
    from PIL import Image
    from app.services.video_analysis.biomechanics.head_silhouette import helmet_above_back

    path = ROOT / photo
    rel = path.resolve().relative_to(ROOT).as_posix()
    if any(e["photo"] == rel for e in study["entries"]):
        raise SystemExit(f"{rel} is already in the study")
    p = _detect(path)
    nl, side = p["normalized"], p["camera_side"]
    H, W = p["image"].shape[:2]
    ids = NEAR[side]
    px = {k: (nl[i].x * W, nl[i].y * H) for k, i in ids.items()}
    torso = math.dist(px["S"], px["H"])
    helmet = helmet_above_back(p["silhouette_top"], (nl[ids["S"]].x, nl[ids["S"]].y),
                               (nl[ids["H"]].x, nl[ids["H"]].y), (nl[0].x, nl[0].y), W / H)
    if helmet:
        px["B"] = (helmet["back_point"][0] * W, helmet["back_point"][1] * H)
        px["T"] = (helmet["head_point"][0] * W, helmet["head_point"][1] * H)
    # the crop: the upper body with room for the helmet, in torso lengths
    xs = [v[0] for v in px.values()]
    ys = [v[1] for v in px.values()]
    facing = 1 if px["S"][0] > px["H"][0] else -1
    back, front = 0.5 * torso, 0.7 * torso
    x0 = max(0, int(min(xs) - (back if facing > 0 else front)))
    x1 = min(W, int(max(xs) + (front if facing > 0 else back)))
    y0, y1 = max(0, int(min(ys) - 0.6 * torso)), min(H, int(max(ys) + 0.4 * torso))
    scale = min(1.0, CROP_MAX_W / (x1 - x0))
    n = max(e["n"] for e in study["entries"]) + 1
    img = Image.open(path).convert("RGB").crop((x0, y0, x1, y1))
    if scale < 1.0:
        img = img.resize((round((x1 - x0) * scale), round((y1 - y0) * scale)), Image.LANCZOS)
    img.save(work_dir(study) / "img" / f"p{n:02d}.webp", "WEBP", quality=86, method=6)
    model = {k: [round((x - x0) * scale, 1), round((y - y0) * scale, 1)] for k, (x, y) in px.items()}
    entry = {"n": n, "key": Path(rel).stem, "photo": rel, "race": race, "side": side,
             "group": group, "category": category,
             "w": img.width, "h": img.height, "origin": [x0, y0], "scale": round(scale, 6),
             "model": model, "start": dict(model)}
    if label:
        entry["label"] = label
    study["entries"].append(entry)
    save(study)
    print(f"added {rel} as editor position {n} ({side} side, helmet "
          f"{helmet['value'] if helmet else 'n/a'}%{' CHECK' if helmet and helmet.get('check') else ''})")
    return entry


def editor(study: dict) -> Path:
    """The point editor page, data embedded; crops already in <work>/img."""
    data = [{"n": e["n"], "src_index": e["key"], "photo": e["photo"], "label": e.get("label"),
             "img": f"img/p{e['n']:02d}.webp", "w": e["w"], "h": e["h"], "origin": e["origin"],
             "model": e["model"], "start": e["start"],
             "group": e.get("group", "pro"), "category": e.get("category")} for e in study["entries"]]
    page = (HERE / "editor_template.html").read_text(encoding="utf-8")
    page = page.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    out = work_dir(study) / "editor.html"
    out.write_text(page, encoding="utf-8")
    missing = [d["img"] for d in data if not (out.parent / d["img"]).exists()]
    print(f"{out} ({len(data)} photos){'; MISSING crops: ' + ', '.join(missing) if missing else ''}")
    return out


# ---------------------------------------------------------------- rebuild

def _records(study: dict) -> dict[int, dict]:
    """The editor's saved documents, by editor number."""
    out = {}
    for f in glob.glob(str(work_dir(study) / "corrections" / "*.json")):
        r = json.loads(Path(f).read_text(encoding="utf-8"))
        out[int(r["n"])] = r
    return out


# Face landmarks in MediaPipe's pose model: nose, eyes, ears, mouth corners.
FACE = tuple(range(11))


def _blur_face(img, nl, scale_px: float):
    """Pixelate the face, feathered at the edge. Returns a new image.

    Age-groupers are private people: a pro's face is the reason a race photo is
    news, an age-grouper's is not, and they did not ask to appear in an article
    about bike positions. Every age-group render is blurred before anything is
    drawn on it, so no copy with a face exists outside the source photo.

    The region is the face landmarks' box, padded by a share of the torso
    length (``scale_px``): on a side view the near ear and the nose can sit a
    few pixels apart, and a box that tight would leave the eye and the mouth.
    """
    import cv2
    import numpy as np

    H, W = img.shape[:2]
    xs = [nl[i].x * W for i in FACE]
    ys = [nl[i].y * H for i in FACE]
    pad = max(0.18 * scale_px, 0.6 * max(max(xs) - min(xs), max(ys) - min(ys)))
    x0, x1 = int(max(0, min(xs) - pad)), int(min(W, max(xs) + pad))
    y0, y1 = int(max(0, min(ys) - pad)), int(min(H, max(ys) + pad))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return img
    out = img.copy()
    roi = out[y0:y1, x0:x1]
    blocks = max(4, (x1 - x0) // 14)
    small = cv2.resize(roi, (blocks, max(4, blocks * (y1 - y0) // (x1 - x0))),
                       interpolation=cv2.INTER_AREA)
    pix = cv2.GaussianBlur(cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST),
                           (0, 0), (x1 - x0) / 20)
    mask = np.zeros((y1 - y0, x1 - x0), np.float32)
    cv2.ellipse(mask, ((x1 - x0) // 2, (y1 - y0) // 2), ((x1 - x0) // 2, (y1 - y0) // 2),
                0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), (x1 - x0) / 16)[..., None]
    out[y0:y1, x0:x1] = (pix * mask + roi * (1 - mask)).astype(img.dtype)
    return out


def _measure(entry: dict, moved: dict, render: bool, blur_face: bool = False):
    import cv2
    from app.services.video_analysis import photo_analyzer as PA
    from app.services.video_analysis.biomechanics import head_silhouette as HS

    p = _detect(ROOT / entry["photo"])
    side = p["camera_side"]
    ids = NEAR[side]
    H, W = p["image"].shape[:2]
    scale = entry.get("scale", 1.0)
    nl = [SimpleNamespace(x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility) for lm in p["normalized"]]
    corrected = []
    for k in KEYS:
        # Every point in a study is checked by hand -- moved or confirmed where
        # it was -- so it is seen, whatever the model's confidence in its own
        # guess. Only raising the moved ones dropped angles as "insufficient
        # visibility" twice: a misread video frame (editor #19), then a wrist
        # Artur left where the model had put it, unsure (#23: elbow and forearm
        # went NaN and took the pro medians with them).
        nl[ids[k]].visibility = 1.0
        dx, dy = moved.get(k, (0.0, 0.0))
        if math.hypot(dx, dy) >= 0.5:
            nl[ids[k]].x += dx / scale / W
            nl[ids[k]].y += dy / scale / H
            corrected.append(ids[k])
    ox, oy = entry["origin"]

    def placed(key):
        mx, my = entry["model"][key]
        dx, dy = moved.get(key, (0.0, 0.0))
        return ((mx + dx) / scale + ox) / W, ((my + dy) / scale + oy) / H

    auto = HS.helmet_above_back(p["silhouette_top"], (nl[ids["S"]].x, nl[ids["S"]].y),
                                (nl[ids["H"]].x, nl[ids["H"]].y), (nl[0].x, nl[0].y), W / H)
    offsets, helmet = None, None
    if auto and "B" in entry["model"]:
        # Every point in a study is checked by hand, so the hand placement is
        # the reading (and is scored), whether or not Artur had to move it.
        B, T = placed("B"), placed("T")
        offsets = {HS.BACK_TOP_POINT: (B[0] - auto["back_point"][0], B[1] - auto["back_point"][1]),
                   HS.HELMET_TOP_POINT: (T[0] - auto["head_point"][0], T[1] - auto["head_point"][1])}
        helmet = HS.apply_point_corrections(auto, offsets)
        helmet["moved"] = [pid for pid, key in ((HS.BACK_TOP_POINT, "B"), (HS.HELMET_TOP_POINT, "T"))
                           if math.hypot(*moved.get(key, (0.0, 0.0))) >= 0.5]
    r = PA.analyze_from_pose(p["image"], p["world"], nl, side, [], "bike", "triathlon",
                             render_thumbnail=False, silhouette_top=p["silhouette_top"],
                             helmet_offsets=offsets)
    angles = {k: r["angles"].get(k) for k in SHOWN}
    rows = {k: r["angles_with_context"][k]["status"] for k in SHOWN if k in r["angles_with_context"]}
    # A reading the person checking the points judged unrepresentative of the
    # position -- editor #40, a rider caught ducking the head under the back
    # line for an instant -- leaves the numbers and the figure; the rest of
    # the photo still counts.
    for k in entry.get("skip", ()):
        angles[k] = None
        rows.pop(k, None)
    if "helmet_above_back" in entry.get("skip", ()):
        helmet = None
    out = {"key": entry["key"], "n": entry["n"], "race": entry["race"], "label": entry.get("label"),
           "caption_note": entry.get("caption_note"),
           "angles": angles, "rows": rows, "corrected": sorted(corrected),
           "moved_cm": [round(math.hypot(*moved[k]) / scale / math.dist(
               (nl[ids["S"]].x * W, nl[ids["S"]].y * H), (nl[ids["H"]].x * W, nl[ids["H"]].y * H)) * 50, 2)
               for k in KEYS if k in moved and math.hypot(*moved[k]) >= 0.5],
           "helmet_moved": bool(helmet and helmet["moved"])}
    if render:
        if blur_face:
            torso = math.dist((nl[ids["S"]].x * W, nl[ids["S"]].y * H),
                              (nl[ids["H"]].x * W, nl[ids["H"]].y * H))
            p["image"] = _blur_face(p["image"], nl, torso)
        ranges = PA._with_helmet_band(PA._get_cycling_optimal_ranges("triathlon", "uncertain"), True)
        # No score_by_phase when the app withholds the score: the near leg was
        # not found (editor #30: a disc wheel behind it). The study measures
        # the upper body only, so the photo still counts; it just has no score.
        score = (r.get("score_by_phase") or {}).get("uncertain") or r["score"]
        out["jpg"] = PA._generate_photo_thumbnail(
            cv2, p["image"], nl, {k: v for k, v in angles.items() if k not in entry.get("skip", ())},
            score, side, "bike",
            PA._build_arc_triplets("bike", side), cycling_position="triathlon",
            optimal_ranges=ranges, row_status=rows, show_header=False,
            corrected=sorted(corrected), helmet=helmet)
        full = PA.analyze_from_pose(p["image"], p["world"], nl, side, [], "bike", "triathlon",
                                    silhouette_top=p["silhouette_top"], helmet_offsets=offsets)
        full["pedal_phase"] = "uncertain"
        full["score"] = (full.get("score_by_phase") or {}).get("uncertain") or full["score"]
        for k in ("knee", "hip"):
            if k not in full["angles_with_context"]:
                continue
            full["angles_with_context"][k].update(status="phase_dependent", optimal_min=None,
                                                  optimal_max=None, unscored_reason="pedal position")
        full["warnings"] = []
        full.pop("pose", None)
        out["report"] = full
    return out


_NUMBER_WORDS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                 "fourteen fifteen sixteen seventeen eighteen nineteen twenty").split()


def _number_word(n: int) -> str:
    """'twenty-one' -- the chart title spells its count like the article does."""
    if n < len(_NUMBER_WORDS):
        return _NUMBER_WORDS[n]
    tens = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty", 7: "seventy",
            8: "eighty", 9: "ninety"}
    if n < 100:
        return tens[n // 10] + (f"-{_NUMBER_WORDS[n % 10]}" if n % 10 else "")
    return str(n)


def _frozen(entry: dict, moved: dict) -> dict:
    """A photo whose source file is gone, from the measurement already made.

    Editor #10's original left the upload folder after its points had been
    checked and its figure published. The angles and the rendered figure are
    kept in the study (``frozen``), so the position still counts; the
    centimetre size of each hand correction is recomputed from the stored
    model points, which are in the same crop pixels as the offsets.
    """
    from PIL import Image

    f = entry["frozen"]
    scale = entry.get("scale", 1.0)
    pt = {k: (entry["model"][k][0] + moved.get(k, (0, 0))[0],
              entry["model"][k][1] + moved.get(k, (0, 0))[1]) for k in KEYS}
    torso = math.dist(pt["S"], pt["H"]) / scale
    buf = io.BytesIO()
    Image.open(ROOT / f["image"]).convert("RGB").save(buf, "JPEG", quality=95)
    return {"key": entry["key"], "n": entry["n"], "race": entry["race"], "label": entry.get("label"),
            "caption_note": entry.get("caption_note"), "angles": dict(f["angles"]),
            "rows": dict(f["rows"]), "corrected": [],
            "moved_cm": [round(math.hypot(*moved[k]) / scale / torso * 50, 2)
                         for k in KEYS if k in moved and math.hypot(*moved[k]) >= 0.5],
            "helmet_moved": any(math.hypot(*moved.get(k, (0, 0))) >= 0.5 for k in ("B", "T")),
            "jpg": buf.getvalue(), "report": None}


def rebuild(study: dict) -> None:
    import numpy as np
    from PIL import Image

    work = work_dir(study)
    recs = _records(study)
    measured = []
    for e in study["entries"]:
        rec = recs.get(e["n"], {})
        if (rec.get("group") or e.get("group")) == "excluded":
            continue
        group = rec.get("group") or e.get("group", "pro")
        if not (ROOT / e["photo"]).exists() and e.get("frozen"):
            d = _frozen(e, rec.get("moved") or {})
        else:
            d = _measure(e, rec.get("moved") or {}, True, blur_face=(group == "age_group"))
        d["group"] = group
        d["figure"] = e.get("figure", True)
        d["category"] = rec.get("category") or e.get("category")
        if d["group"] != "excluded":
            measured.append(d)
    groups = _group_stats(measured)
    # The article, its figures and the social exports are the pros only.
    items = sorted((d for d in measured if d["group"] == "pro"), key=lambda d: d["angles"]["forearm_tilt"])
    media = ROOT / study["media"]
    social = ROOT / study["social"]
    social.mkdir(parents=True, exist_ok=True)

    def webp(jpg, path, q):
        im = Image.open(io.BytesIO(jpg)).convert("RGB")
        if im.width > 1200:
            im = im.resize((1200, int(1200 * im.height / im.width)), Image.LANCZOS)
        im.save(path, "WEBP", quality=q, method=6)

    for old in glob.glob(str(media / "position-*.webp")) + glob.glob(str(social / "position-*.jpg")):
        os.remove(old)
    count = len(items)
    # Every pro counts in the numbers; only the ones whose photo is good enough
    # to print get a figure. Broadcast screen grabs measure fine once the points
    # are checked by hand, but they are too soft and too small to publish.
    shown = [d for d in items if d["figure"]]
    captions = []
    for n, d in enumerate(shown, 1):
        a = d["angles"]
        webp(d["jpg"], media / f"position-{n:02d}.webp", 82)
        tag = f"_{d['key']}" if d["race"] != "Kona 2026" else ""
        Image.open(io.BytesIO(d["jpg"])).convert("RGB").save(
            social / f"position-{n:02d}{tag}_forearm-{a['forearm_tilt']:.0f}_torso-{a['trunk']:.0f}.jpg",
            "JPEG", quality=93)
        extra = f" {d['caption_note']}" if d.get("caption_note") else ""
        helmet = (f", helmet {a['helmet_above_back']:+.0f}% above the back"
                  if a.get("helmet_above_back") is not None else "")
        captions.append(
            f"![Position {n} of {len(shown)}: forearm tilt {a['forearm_tilt']:.0f}°, torso {a['trunk']:.0f}°, "
            f"shoulder {a['shoulder']:.0f}°, elbow {a['elbow']:.0f}°{helmet}.{extra}]"
            f"(/media/academy/{Path(study['media']).name}/position-{n:02d}.webp)")
    # Age-groupers: their own folder, faces blurred (_blur_face), never into
    # the article media until there is an age-group article to put them in.
    ag_dir = social / "age-group"
    ag_dir.mkdir(exist_ok=True)
    for old in glob.glob(str(ag_dir / "*.jpg")):
        os.remove(old)
    for d in measured:
        if d["group"] == "age_group":
            a = d["angles"]
            Image.open(io.BytesIO(d["jpg"])).convert("RGB").save(
                ag_dir / (f"editor-{d['n']:02d}_{d['category'] or 'untagged'}"
                          f"_forearm-{a['forearm_tilt']:.0f}_torso-{a['trunk']:.0f}.jpg"),
                "JPEG", quality=93)
    for tag, key in study["riders"].items():
        d = next(x for x in items if x["key"] == key)
        webp(d["jpg"], media / f"rider-{tag}.webp", 84)
    (work / "reports.json").write_text(json.dumps(
        {t.upper(): next(x for x in items if x["key"] == k)["report"] for t, k in study["riders"].items()},
        default=str), encoding="utf-8")

    def stats(k):
        v = np.array([d["angles"][k] for d in items if d["angles"].get(k) is not None])
        q1, q3 = np.percentile(v, [25, 75])
        return {"median": float(np.median(v)), "q1": float(q1), "q3": float(q3),
                "min": float(v.min()), "max": float(v.max()), "n": int(v.size)}

    ft = np.array([d["angles"]["forearm_tilt"] for d in items])
    el = np.array([d["angles"]["elbow"] for d in items])
    cm = [c for d in items for c in d["moved_cm"]]
    out = {
        "count": count,
        "stats": {k: stats(k) for k in SHOWN},
        "r_elbow_forearm": float(np.corrcoef(ft, el)[0, 1]),
        "counts": {
            "forearm_over_25": int((ft > 25).sum()), "forearm_over_30": int((ft > 30).sum()),
            "forearm_min": float(ft.min()),
            "trunk_under_20": sum(d["angles"]["trunk"] < 20 for d in items),
            "shoulder_under_80": sum(d["angles"]["shoulder"] < 80 for d in items),
            "elbow_under_75": sum(d["angles"]["elbow"] < 75 for d in items),
            "joint_points_moved": len(cm), "joint_points": 4 * count,
            "joint_move_median_cm": float(np.median(cm)) if cm else 0.0,
            "joint_move_max_cm": max(cm) if cm else 0.0,
            "helmet_back_points_moved": sum(d["helmet_moved"] for d in items),
        },
        "shown": len(shown),
        "positions": [{"n": n, "key": d["key"], "editor_n": d["n"], "category": d["category"],
                       "figure": (shown.index(d) + 1) if d in shown else None,
                       **{k: d["angles"][k] for k in SHOWN}, "rows": d["rows"]}
                      for n, d in enumerate(items, 1)],
        "groups": groups,
        "untagged": [d["n"] for d in measured if not d["category"]],
        "captions": captions,
    }
    (work / "rebuilt.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    _field_chart(study, sorted(items, key=lambda d: d["n"]), media, social)
    for k, s in out["stats"].items():
        print(f"{k:18s} median {s['median']:.1f}  middle {s['q1']:.2f}-{s['q3']:.2f}  "
              f"range {s['min']:.1f}-{s['max']:.1f}")
    print(json.dumps(out["counts"]))
    print(f"r(elbow, forearm) = {out['r_elbow_forearm']:.3f}; {work / 'rebuilt.json'}")


def _group_stats(measured: list) -> dict:
    """Median and middle half of every metric per group, per category, and
    per group x category -- only where a cell has photos in it."""
    import numpy as np

    cells: dict[str, list] = {}
    for d in measured:
        for name in (d["group"], f"{d['group']}/{d['category'] or 'untagged'}"):
            cells.setdefault(name, []).append(d)
    out = {}
    for name, ds in sorted(cells.items()):
        row = {"n": len(ds)}
        for k in SHOWN:
            v = np.array([d["angles"][k] for d in ds if d["angles"].get(k) is not None])
            if v.size:
                q1, q3 = np.percentile(v, [25, 75])
                row[k] = {"median": round(float(np.median(v)), 1), "q1": round(float(q1), 1),
                          "q3": round(float(q3), 1)}
        out[name] = row
    return out


def _field_chart(study: dict, items: list, media: Path, social: Path) -> None:
    from PIL import Image
    from playwright.sync_api import sync_playwright

    s = (HERE / "field_chart.html").read_text(encoding="utf-8")
    for k in ("trunk", "shoulder", "elbow", "forearm_tilt"):
        vals = ",".join(f"{d['angles'][k]:.1f}" for d in items)
        s, n = re.subn(r"(%s:\s*\{[^}]*?v:\[)[^\]]*(\])" % k, lambda m: m.group(1) + vals + m.group(2),
                       s, flags=re.S)
        assert n == 1, k
    word = _number_word(len(items))
    s = re.sub(r"<h2>\w+ pro positions", f"<h2>{word.capitalize()} pro positions", s)
    page = work_dir(study) / "field_chart.html"
    page.write_text(s, encoding="utf-8")
    png = page.with_suffix(".png")
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1100, "height": 900}, device_scale_factor=2)
        pg.goto(page.as_uri())
        pg.wait_for_timeout(600)
        pg.query_selector("#viz").screenshot(path=str(png))
        b.close()
    im = Image.open(png).convert("RGB")
    im.resize((1600, round(1600 * im.height / im.width)), Image.LANCZOS).save(
        media / "field.webp", "WEBP", quality=88, method=6)
    im.save(social / "field-chart.png")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--study", default="kona2026")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("photo")
    a.add_argument("--label")
    a.add_argument("--race", default="Kona 2026")
    a.add_argument("--group", default="pro", choices=("pro", "age_group"))
    a.add_argument("--category", choices=("men", "women"))
    sub.add_parser("editor")
    sub.add_parser("rebuild")
    args = ap.parse_args()
    study = load(args.study)
    if args.cmd == "add":
        add(study, args.photo, args.label, args.race, args.group, args.category)
        editor(study)
    elif args.cmd == "editor":
        editor(study)
    else:
        rebuild(study)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
