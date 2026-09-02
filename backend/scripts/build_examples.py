#!/usr/bin/env python
"""Generate the public sample reports for /examples from real footage.

    python scripts/build_examples.py            # rebuild every sample
    python scripts/build_examples.py --only run-treadmill

Why fixtures rather than a live analysis: the page is public, cached and
crawled, and running MediaPipe on a crawler's request would be absurd. So each
sample is analysed ONCE here, trimmed to what the page shows, and committed --
the numbers on the page are then a real measurement of real footage that
anybody can check against, rather than a designer's placeholder.

The clips are gitignored (they are Artur's), so this only runs where they are.
The OUTPUT is committed: a small JSON per sample plus one keyframe image.

The coaching prose is deliberately absent when no GEMINI_API_KEY is set. The
sample is built from the deterministic half -- the measurements, the reference
bands, the findings and the plan -- which is the half that distinguishes this
product anyway. A sample report full of invented coaching would be the one
thing on the page that is not a real measurement.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

CONTENT = BACKEND / "content" / "examples"
MEDIA = BACKEND / "app" / "static" / "media" / "examples"

# Four samples, chosen so the page is not four victory laps. A rider whose fit
# needs work sells the fit report better than a perfect one does.
SAMPLES: dict[str, dict] = {
    "run-treadmill": {
        # Was IMG_4004. Its keyframe put the near leg on the treadmill deck --
        # correct leg IDENTITY (the stride check reads 0.000 there) but the
        # landmarks themselves sitting beside the athlete, next to people
        # walking behind him. Nothing in the pipeline catches that: the bone
        # lengths are normal, the point sits on its own trajectory, and the
        # identity metric only asks WHICH leg a label is on, never whether the
        # leg is on the runner. So the sample moved to a clip whose overlay is
        # actually on the athlete, which is the one thing this page is for.
        "clip": "upload/IMG_3979.MOV",
        "sport": "run", "position": None, "mode": "video",
        "title": "Running, treadmill, side view",
        "blurb": "A 10-second treadmill clip. Every stride metric measured: "
                 "cadence, ground contact, flight time, vertical oscillation "
                 "and overstride, plus the five kinogram positions.",
    },
    "bike-road": {
        "clip": "upload/IMG_4091.MOV",
        "sport": "bike", "position": "road_drops", "mode": "video",
        "title": "Road bike on a trainer",
        "blurb": "A road position with real fit findings -- what to change, "
                 "in fitting order, with the measurement each one fired on.",
    },
    "run-photo": {
        "clip": "upload/IMG_4258.MOV", "frame": 0.45,
        "sport": "run", "position": None, "mode": "photo",
        "title": "Running, from a single photo",
        "blurb": "One still frame. Fewer metrics than a clip -- no cadence, no "
                 "contact time -- and the report says which ones are missing "
                 "rather than estimating them.",
    },
    "bike-aero": {
        "clip": "upload/IMG_4148.MOV", "frame": 0.45,
        "sport": "bike", "position": "tt_aero", "mode": "photo",
        "title": "Aero position, from a single photo",
        "blurb": "A TT position judged against the aero fit window rather than "
                 "the road one -- a flat back is the goal here, not a fault.",
    },
}

# Everything the page renders. Anything not on this list is dropped, which is
# both a size decision and a privacy one: the fixtures are committed.
KEEP = (
    "sport_type", "cycling_position", "cycling_position_label", "camera_side",
    "frames_analyzed", "technique_score", "letter_grade", "score_withheld",
    "score_breakdown", "score_coverage", "quality_gate_triggered",
    "detected_issues", "angle_statistics", "sport_specific_metrics",
    "training_plan", "fit_plan", "reference_bands",
)

# Summary keys worth showing. The full blob carries per-frame diagnostics that
# mean nothing to a reader and would triple the fixture.
KEEP_SUMMARY = (
    "cadence_spm", "ground_contact_ms", "flight_time_ms", "overstride_ratio",
    "vertical_oscillation_m", "trunk_lean_avg", "foot_strike", "stance_fraction",
    "knee_at_bdc", "knee_at_tdc", "trunk_angle_avg", "hip_angle_avg",
    "elbow_angle_avg", "shoulder_angle_avg", "head_alignment_avg",
    "pelvic_ratio", "forearm_tilt_avg", "saddle_height_assessment",
    "pedaling_style", "aero_estimate", "analysis_confidence", "capture_report",
    "quality_warnings", "camera_view", "athlete_height_cm",
)


# The page is public and image-heavy by nature, so the keyframes are re-encoded
# rather than dumped straight out of the analysis. A 477 KB JPEG of a keyframe
# is fine inside a report the athlete asked for and wrong on a marketing page
# that has to load before anyone has decided to care.
MAX_IMAGE_WIDTH = 1100
WEBP_QUALITY = 78


def _write_image(b64: str, path: Path) -> str | None:
    """Re-encode a data-URI image as a sized webp and return its public URL."""
    if not b64:
        return None
    import io

    from PIL import Image

    payload = base64.b64decode(b64.split(",", 1)[-1])
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(io.BytesIO(payload)).convert("RGB")
    if img.width > MAX_IMAGE_WIDTH:
        h = round(img.height * MAX_IMAGE_WIDTH / img.width)
        img = img.resize((MAX_IMAGE_WIDTH, h), Image.LANCZOS)
    out = path.with_suffix(".webp")
    img.save(out, "WEBP", quality=WEBP_QUALITY, method=6)
    # Drop a stale JPEG from an earlier build so the directory does not carry
    # two copies of every sample.
    if path.suffix != ".webp" and path.exists():
        path.unlink()
    return f"/media/examples/{out.name}"


def build(slug: str, spec: dict) -> dict | None:
    from app.services.video_analysis.photo_analyzer import analyze_photo
    from app.services.video_analysis.runner import run_analysis

    clip = BACKEND.parent / spec["clip"]
    if not clip.exists():
        print(f"  SKIP {slug}: {spec['clip']} is not on this machine")
        return None

    if spec["mode"] == "photo":
        import cv2

        cap = cv2.VideoCapture(str(clip))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * spec.get("frame", 0.45)))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print(f"  SKIP {slug}: could not read a frame")
            return None
        result = analyze_photo(
            cv2.imencode(".jpg", frame)[1].tobytes(),
            spec["sport"], spec["position"],
        )
    else:
        result = run_analysis(
            str(clip), spec["sport"], spec["position"],
            # No coaching: without a key it would be absent anyway, and an
            # invented one would be the only thing on the page that is not a
            # real measurement.
            recommendations=False,
            kinogram=True,
        )

    out = {k: result[k] for k in KEEP if k in result}
    summary = result.get("sport_specific_metrics") or {}
    out["sport_specific_metrics"] = {
        k: summary[k] for k in KEEP_SUMMARY if k in summary
    }
    # Photo results carry their own shape: the score is nested under
    # ``score``, and the reference ranges ride along with each angle in
    # ``angles_with_context`` rather than in a separate reference_bands block.
    for k in ("angles", "angles_with_context", "score", "gait_phase",
              "pedal_phase", "warnings", "cycling_position_label", "sport"):
        if k in result:
            out[k] = result[k]

    out["slug"] = slug
    out["title"] = spec["title"]
    out["blurb"] = spec["blurb"]
    out["mode"] = spec["mode"]
    out["source_clip"] = Path(spec["clip"]).name

    MEDIA.mkdir(parents=True, exist_ok=True)
    out["keyframe_url"] = _write_image(
        result.get("keyframe_base64") or result.get("thumbnail_base64") or "",
        MEDIA / f"{slug}.jpg",
    )
    out["kinogram_url"] = _write_image(
        result.get("kinogram_base64") or "", MEDIA / f"{slug}-kinogram.jpg",
    )

    CONTENT.mkdir(parents=True, exist_ok=True)
    target = CONTENT / f"{slug}.json"
    target.write_text(
        json.dumps(out, indent=1, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    score = out.get("technique_score", out.get("overall_score"))
    print(f"  {slug}: {score}{out.get('letter_grade') or out.get('grade') or ''} "
          f"-> {target.relative_to(BACKEND)} ({target.stat().st_size // 1024} KB)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", help="rebuild a single sample by slug")
    args = ap.parse_args()

    names = [args.only] if args.only else list(SAMPLES)
    for slug in names:
        spec = SAMPLES.get(slug)
        if spec is None:
            print(f"unknown sample: {slug}; have {list(SAMPLES)}")
            return 2
        print(f"\n=== {slug} ({spec['clip']})")
        build(slug, spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
