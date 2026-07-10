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

## Caveats (current MVP)

- **Memory:** MediaPipe "heavy" + 1080p60 clips are RAM-hungry. If the container
  OOM-restarts, bump the service memory in Railway (Settings → Resources).
- **Single instance only.** The job store is in-memory and the uploaded files +
  overlays live on the container's ephemeral disk. Do **not** scale replicas or
  workers > 1 until M4b (external job store + object storage) — a poll could
  otherwise hit a replica that never saw the job. Restarts also drop in-flight
  jobs and stored overlays.
- **First build is slow** (downloads the ML stack + model); later builds reuse
  Docker layer cache.
