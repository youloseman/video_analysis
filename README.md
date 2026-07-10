# Video Analysis

Standalone technique-analysis app for **running** (side view) and **cycling
position** (side view), extracted from the Motus platform. Computer-vision
pose estimation (MediaPipe BlazePose) → biomechanics → technique score.

> Status: **Milestones 1–3 complete** — the analysis core runs autonomously
> (angles, issues, metrics, 0–100 score + grade), renders an annotated
> **overlay video** (skeleton + angle labels + score per frame), and is exposed
> over a **FastAPI service** (upload → poll → JSON + overlay). No DB / cloud
> storage / LLM yet — those are later milestones.

## Layout

```
backend/
├── app/
│   ├── main.py                            # FastAPI service (M3)
│   ├── core/config.py                     # minimal settings
│   └── services/video_analysis/
│       ├── detectors/                     # MediaPipe pose detector (abstracted)
│       ├── biomechanics/                  # analyzers, filters, scoring, quality gate
│       ├── runner.py                      # shared analysis service (CLI + API call this)
│       ├── video_visualizer.py            # overlay renderer (M2)
│       └── pipeline.py                    # shared constants + overlay-draw helpers
├── models/                                # pose_landmarker_heavy.task goes here (git-ignored)
├── scripts/analyze_local.py               # thin CLI over runner.run_analysis
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

# Also render an annotated overlay video (skeleton + angles + score):
python scripts/analyze_local.py <path/to/run.mp4> run --overlay
#   -> writes <path/to/run>_overlay.mp4  (or pass an explicit path: --overlay out.mp4)
```

**ffmpeg (optional):** if `ffmpeg` is on `PATH`, overlays are re-encoded to
web-safe H.264; otherwise they are written directly via OpenCV (`mp4v`), which
plays in VLC/most players. Install ffmpeg for browser-embeddable output.

Cycling positions: `road_hoods` (default) · `road_drops` · `tt_aero` ·
`triathlon` · `casual`.

Output is JSON: `technique_score`, `letter_grade`, `angle_statistics`,
`detected_issues`, `sport_specific_metrics`. Missing measurements are `null`
(never `0`) — a landmark that was not reliably detected is NaN upstream and
serialized as `null`.

## API (Milestone 3)

Run the service:

```bash
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
# interactive docs at http://localhost:8000/docs
```

Analyze a clip (async job — upload, poll, fetch):

```bash
# 1) upload -> {"job_id": "...", "poll_url": "/jobs/<id>"}
curl -s -X POST http://localhost:8000/analyze \
  -F "video=@bike.mp4" -F "sport=bike" -F "position=triathlon" -F "overlay=true"

# 2) poll until status == "completed" (analysis is ~30-60s)
curl -s http://localhost:8000/jobs/<job_id>

# 3) download the annotated overlay
curl -s http://localhost:8000/jobs/<job_id>/overlay -o overlay.mp4
```

`GET /health` reports liveness + whether the pose model is installed. Job state
is in-memory (single-worker MVP — not persisted across restarts); uploads +
overlays are stored under `backend/uploads/` (git-ignored).

## Roadmap

- **M1** — ✅ standalone analysis core (run + bike, side view)
- **M2** — ✅ annotated overlay video (skeleton + angles + score per frame)
- **M3** — ✅ FastAPI service (upload → poll → JSON + overlay; in-memory jobs)
- **M4** — persistence (job store + S3/R2 storage) + deploy to Railway
- **M5** — LLM coaching recommendations from the metrics
- **M6** — web frontend (upload UI + results + overlay player)
- **later** — rear-view running, swimming (re-add the trimmed analyzers)

## Provenance

The biomechanics/detector core is copied (not rewritten) from Motus
(`CoachPowerBoost`) and trimmed to the running-side + cycling-side path.
Swimming and rear-view analyzers were excluded for this milestone.
