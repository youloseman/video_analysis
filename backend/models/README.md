# Pose model

Place the MediaPipe BlazePose model here:

```
backend/models/pose_landmarker_heavy.task
```

The detector (`app/services/video_analysis/detectors/mediapipe_detector.py`)
searches this directory first via its `_MODEL_SEARCH_PATHS`. If the file is
missing, the MediaPipe **Tasks API** cannot initialise and the detector falls
back to the Legacy Solutions API (available only in older mediapipe builds) --
on mediapipe 0.10.35 that fallback is unavailable, so the model file is
effectively **required**.

Download (Google MediaPipe model card, "Pose landmarker (heavy)"):

```
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
```

This file is intentionally git-ignored (it is large, ~30 MB).
