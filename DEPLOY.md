# Deploy to Railway (M4a)

> **Current deployment:** https://video-analysis-production-1f54.up.railway.app
> (Railway project `video-analysis`, workspace *youloseman's Projects*). Every
> `railway up` / GitHub push redeploys it.

The API ships as a Docker image (`Dockerfile` at the repo root). The pose model
is downloaded **at build time** and baked in, and `ffmpeg` is installed so
overlays come out as web-safe H.264. Config is in `railway.json` (Dockerfile
builder + `/health` check + single replica).

## Option A — GitHub integration (recommended, auto-deploys on push)

1. Go to <https://railway.app> → **New Project** → **Deploy from GitHub repo**.
2. Pick **`youloseman/video_analysis`**. Railway detects the `Dockerfile` and
   `railway.json` and starts a build (~3-5 min the first time — it installs
   MediaPipe/OpenCV/SciPy and fetches the 30 MB model).
3. When it's live, open **Settings → Networking → Generate Domain** to get a
   public URL, then hit `https://<your-domain>/health` and `/docs`.

Every `git push` to `main` redeploys automatically.

## Option B — Railway CLI (you're already logged in as top.raider90@gmail.com)

From the repo root:

```bash
railway init          # create a project (interactive: name + workspace)
railway up            # upload + build the Dockerfile on Railway
railway domain        # generate a public URL
```

`railway link` instead of `init` if the project already exists.

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
  in-memory, and uploaded files + overlays live on the container's ephemeral
  disk. Do **not** scale replicas or workers > 1 until M4b (external job store +
  object storage) — a poll could otherwise hit a replica that never saw the job,
  and the rate limit wouldn't be shared. Restarts drop in-flight jobs, stored
  overlays, and reset the rate counters.
- **Storage is reaped, not permanent.** Jobs and their upload directories are
  deleted `VA_JOB_TTL_HOURS` after they were accepted, and directories left
  behind by a previous process are cleared at startup. This is what backs the
  retention promise in the privacy policy — do not set the TTL to `0` in prod.
- **Access control:** `/analyze` is open (rate-limited to
  `VA_RATE_LIMIT_PER_DAY` per client IP), but **reading a job back is not** —
  `/jobs/{id}` and `/jobs/{id}/overlay` require either the capability token
  issued at upload (`?t=`) or the account that created the job. Unauthorized
  reads answer 404, not 403, so the surface stays undiscoverable. Free-tier
  analyses no longer make a Gemini call at all.
- **First build is slow** (downloads the ML stack + model); later builds reuse
  Docker layer cache.
