# Video Analysis

Standalone technique-analysis app for **running** (side view) and **cycling
position** (side view), extracted from the Motus platform. Computer-vision
pose estimation (MediaPipe BlazePose) → biomechanics → technique score.

> Status: **Milestone 1 complete** — the analysis core runs autonomously and
> produces a numeric result (angles, issues, metrics, 0–100 score + grade)
> from a local video file. No API / DB / storage / LLM / overlay yet — those
> are later milestones.

## Layout

```
backend/
├── app/
│   ├── core/config.py                     # minimal settings (core reads no settings.*)
│   └── services/video_analysis/
│       ├── detectors/                     # MediaPipe pose detector (abstracted)
│       ├── biomechanics/                  # analyzers, filters, scoring, quality gate
│       └── pipeline.py                    # stub: shared constants only
├── models/                                # pose_landmarker_heavy.task goes here (git-ignored)
├── scripts/analyze_local.py               # CLI driver (Milestone 1)
└── requirements.txt
```

## Quickstart

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate   |   *nix: source .venv/bin/activate
pip install -r requirements.txt

# One-time: download the pose model into backend/models/
#   pose_landmarker_heavy.task  (see backend/models/README.md for the URL)

# Analyze a local clip (side view):
python scripts/analyze_local.py <path/to/run.mp4>  run
python scripts/analyze_local.py <path/to/bike.mp4> bike --position road_hoods
```

Cycling positions: `road_hoods` (default) · `road_drops` · `tt_aero` ·
`triathlon` · `casual`.

Output is JSON: `technique_score`, `letter_grade`, `angle_statistics`,
`detected_issues`, `sport_specific_metrics`. Missing measurements are `null`
(never `0`) — a landmark that was not reliably detected is NaN upstream and
serialized as `null`.

## Roadmap (post-M1)

- **M2** — annotated overlay video (skeleton + angles + score per frame)
- **M3** — FastAPI service (upload → analyze → JSON + overlay)
- **M4** — storage (local dev, S3/R2 in prod) + deploy to Railway
- **M5** — LLM coaching recommendations from the metrics
- **M6** — web frontend (upload UI + results + overlay player)
- **later** — rear-view running, swimming (re-add the trimmed analyzers)

## Provenance

The biomechanics/detector core is copied (not rewritten) from Motus
(`CoachPowerBoost`) and trimmed to the running-side + cycling-side path.
Swimming and rear-view analyzers were excluded for this milestone.
