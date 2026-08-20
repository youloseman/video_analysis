# Deploy to Railway (M4a)

> **Current deployment:** https://video-analysis-production-1f54.up.railway.app
> (Railway project `video-analysis`, workspace *youloseman's Projects*).
>
> **Merging to `main` deploys automatically** — via GitHub Actions
> (`.github/workflows/deploy.yml`), not via Railway. The service is still not
> wired to the GitHub repo; the workflow runs the same `railway up --ci` a human
> used to run by hand, after the test suite passes, and then checks `/health`.
>
> This needs the `RAILWAY_TOKEN` repository secret to exist. Without it the
> deploy job fails loudly, which is the point: before the workflow existed the
> symptom of a dead deploy path was silence. On 2026-07-30 three pull requests
> merged to `main` and none of them shipped; production was serving a snapshot
> taken before the first of them was even committed.

The API ships as a Docker image (`Dockerfile` at the repo root). The pose model
is downloaded **at build time** and baked in, and `ffmpeg` is installed so
overlays come out as web-safe H.264. Config is in `railway.json` (Dockerfile
builder + `/health` check + single replica).

## Option A — GitHub Actions (what runs today)

`.github/workflows/deploy.yml`, on every push to `main` and on manual dispatch:
tests → `railway up --ci --service video-analysis` → poll `/health` until 200.

**One-time setup.** Railway → project `video-analysis` → **Settings → Tokens** →
create a **project** token for the production environment (not an account token,
which would reach every project in the workspace). Then GitHub → **Settings →
Secrets and variables → Actions** → new repository secret named `RAILWAY_TOKEN`.

The test gate currently skips `tests/test_gzip_middleware.py` — those 7 tests
fail on `main`, and gating on them would block every deploy. Fix them and delete
the `--ignore` line in the workflow.

To deploy without pushing (e.g. re-run a failed deploy): Actions → **Deploy** →
**Run workflow**.

## Option B — Railway's own GitHub integration (NOT connected)

Native builds triggered by Railway itself, no Actions minutes, no token in
GitHub. Would replace Option A. Needs the Railway **web dashboard**; there is no
CLI equivalent.

1. Railway → project `video-analysis` → service `video-analysis` →
   **Settings → Source** → connect **`youloseman/video_analysis`**, branch
   `main`. Railway reads the `Dockerfile` and `railway.json`.
2. Push something to `main` and confirm a build actually starts. Do not assume:
   the symptom of a dead connection is silence, not an error.

How to tell whether a given deployment came from GitHub or from someone's
laptop — GitHub builds carry the commit, CLI uploads do not:

```bash
railway deployment list --json   # CLI uploads have an empty commitMessage
```

If you connect it, delete `.github/workflows/deploy.yml` — two things deploying
the same service on the same push is how you get racing builds.

## Option C — Railway CLI (manual, still works)

The escape hatch when Actions is down or you need to ship something that is not
on `main`. From the repo root, logged in as top.raider90@gmail.com:

```bash
railway up --ci -m "what changed"   # upload + build the Dockerfile on Railway
```

Two things to know:

* **It uploads the working directory, not a git commit.** Check `git status`
  first — whatever is on disk is what ships, including uncommitted edits and
  whatever branch you happen to be on.
* The build context is filtered by `.dockerignore`. Patterns without `**/` match
  only the repo root, which is why `.venv/` alone did not exclude
  `backend/.venv` — the venv the README tells you to create. Context should be
  a few MB; if an upload suddenly takes minutes, that is the thing to check.

`railway link` first if the project is not already linked.

## Smoke-test the live service

```bash
BASE=https://<your-domain>
curl -s $BASE/health          # {"status":"ok","model_present":true,...}

# upload -> poll -> download overlay
curl -s -X POST $BASE/analyze -F "video=@bike.mp4" -F "sport=bike" -F "position=triathlon"
curl -s $BASE/jobs/<job_id>
curl -s $BASE/jobs/<job_id>/overlay -o overlay.mp4
```

## Tuning (env vars)

| Var | Default | What it does |
|-----|---------|--------------|
| `VA_MAX_CONCURRENT_ANALYSES` | `2` | How many clips are analyzed at once. This is the OOM knob — raise it only alongside container memory. |
| `VA_MAX_QUEUED_ANALYSES` | `8` | Backlog depth. Past this, `/analyze` returns 503 + `Retry-After` instead of accepting an upload nobody will wait for. |
| `VA_JOB_TTL_HOURS` | `6` | How long a finished job stays pollable before it and its upload directory are deleted. `0` disables the sweeper (debugging only — the disk then grows unbounded). |
| `VA_JOB_SWEEP_INTERVAL_S` | `600` | How often the reaper runs. |
| `VA_RATE_LIMIT_PER_DAY` | `3` | Anonymous per-IP daily analyses. `0` disables. |
| `GEMINI_TIMEOUT_S` | `20` | Hard ceiling on one Gemini request. Coaching is optional by design — a timeout degrades to "no coaching", never to a failed analysis. |
| `VA_CORS_ORIGINS` | *(unset)* | Comma-separated allowed origins. **Unset in production = no cross-origin access at all** (the SPA is same-origin, so it needs none); unset locally = `*`. Set to `*` to force the old permissive behaviour. |

## Caveats (current MVP)

- **Memory:** MediaPipe "heavy" + 1080p60 clips are RAM-hungry. Concurrency is
  capped at `VA_MAX_CONCURRENT_ANALYSES` (default 2) precisely so the container
  cannot start more of them than it can hold; if it still OOM-restarts, lower
  that before bumping the service memory in Railway (Settings → Resources).
- **Single instance only.** The job store AND the per-IP rate limiter are
  in-memory, and the uploads directory is a volume attached to this one service.
  Do **not** scale replicas or workers > 1 — a poll could hit a replica that
  never saw the job, the rate limit would not be shared, and only one instance
  can mount the volume. Restarts drop in-flight jobs and reset the rate
  counters; stored footage now survives them.
- **A volume is required.** Railway → service → Settings → Volumes → mount at
  `/data`, then set `VA_UPLOADS_DIR=/data/uploads`. Without it the container
  filesystem is ephemeral and every clip disappears on the next deploy — which
  silently breaks two promises at once: the privacy policy says footage is kept
  for a fixed period, and an Expert Review is sold on a human watching the clip.
  Size it for roughly 25 MB per analysis (a phone clip plus its overlay) times
  30 days of volume.
- **Two different lifetimes, deliberately.** `VA_JOB_TTL_HOURS` now governs only
  the in-memory job store and the grace window for an upload nobody has saved to
  history yet. How long a *stored* clip lives is decided per analysis by
  `app/services/retention.py` — 7 days free, 30 paid, and indefinitely while an
  Expert Review is bought and undelivered. Do not set the TTL to `0` in prod: it
  disables both sweeps, and the uploads directory then grows without limit.
- **Access control:** `/analyze` is open (rate-limited to
  `VA_RATE_LIMIT_PER_DAY` per client IP), but **reading a job back is not** —
  `/jobs/{id}` and `/jobs/{id}/overlay` require either the capability token
  issued at upload (`?t=`) or the account that created the job. Unauthorized
  reads answer 404, not 403, so the surface stays undiscoverable. Free-tier
  analyses no longer make a Gemini call at all.
- **First build is slow** (downloads the ML stack + model); later builds reuse
  Docker layer cache.
