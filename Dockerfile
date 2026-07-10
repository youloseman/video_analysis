# syntax=docker/dockerfile:1
# Container image for the Video Technique Analysis API (Milestone 4a).
# Builds on Railway (or any Docker host). The pose model is baked in at build
# time so the running container has no external download dependency.

FROM python:3.11-slim

# System deps:
#   ffmpeg          -> re-encode overlays to web-safe H.264 (else OpenCV mp4v)
#   libgl1/libglib  -> runtime libs for opencv-contrib-python (pulled by mediapipe)
#   curl            -> fetch the pose model at build time
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps first for layer caching.
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Bake the MediaPipe pose model into the image (~30 MB, git-ignored in the repo).
RUN mkdir -p models \
    && curl -fSL -o models/pose_landmarker_heavy.task \
       "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"

# Application code (config.py resolves BACKEND_DIR to /app -> /app/models, /app/uploads).
COPY backend/app ./app
COPY backend/scripts ./scripts

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Railway injects $PORT. Single worker on purpose: the M3 job store is in-memory,
# so multiple workers/replicas would not share job state (fixed in M4b).
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
