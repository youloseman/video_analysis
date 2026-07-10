# Pose Detectors

Every video analysis runs through a pose detector that implements
[`base.PoseDetector`](base.py). The pipeline does not import MediaPipe
(or any other detector library) directly; it calls `build_detector()`
and then uses the abstract interface.

## Current detectors

- [`MediaPipePoseDetector`](mediapipe_detector.py) — Google MediaPipe
  BlazePose. Tries the Tasks API (`PoseLandmarker`, VIDEO mode) first;
  on initialisation failure it falls back to Legacy Solutions
  (`mp.solutions.pose.Pose`). Reports `name` as `mediapipe_tasks` or
  `mediapipe_legacy`. Used for all sports. Confidence thresholds live
  in `DEFAULT_CONFIDENCE_TABLE` inside that file.

## Adding a new detector

1. **Inherit from `PoseDetector`.** Implement `__init__`, `detect`,
   `close`, and the `name` / `config` properties.
2. **Map to BlazePose-33.** If your native format is different
   (COCO-17, OpenPose-25, ...), translate into the 33-landmark indexing
   documented in [`base.py`](base.py). Missing landmarks use
   `visibility=0` and NaN coordinates — never zeros, which pollute
   averages downstream.
3. **Normalise coordinates.** `x` and `y` in `[0, 1]`, origin
   top-left, `x` right, `y` down. Relative depth `z` is a qualitative
   signal only (used for flip-fix side detection); standardise its
   sign if possible.
4. **Provide world landmarks.** Metres, pose-centre origin. If the
   native model has no 3D output, fill every `world_landmark` with
   NaN coordinates and visibility 0 — downstream Z-dependent metrics
   are already NaN-safe.
5. **Branch in the factory.** Add the new detector to
   `build_detector()` in [`__init__.py`](__init__.py). Select via
   sport/camera-angle or a feature flag.
6. **Document accuracy.** Add a short paragraph here describing
   strengths and weaknesses per sport and camera angle (where you
   tested, failure modes observed).

## Testing a new detector

1. Run `pytest backend/tests/unit/test_detector_interface.py -v` —
   the contract tests must pass for any `PoseDetector`.
2. Pick a representative set of videos (swim_above, swim_under, run,
   bike). Run the full pipeline through each.
3. Compare `unknown_phase_pct` (swim), `valid_frames / frames_processed`
   ratios, and `nan_pct` across `angle_statistics` against the
   MediaPipe baseline. Sign-off threshold: within 5 percentage points,
   or better.
4. Inspect the generated overlay videos visually — every swap has
   surprised us at least once.
5. Run the full backend test suite — all existing tests must pass
   unchanged. No "update the fixture" — if a test drifts, you've
   changed downstream behaviour.

## Design invariants (never break)

- **33 landmarks per frame, BlazePose indexing.** Shorter lists get
  padded with NaN/zero-visibility; longer lists get truncated.
- **Normalised coords in `[0, 1]`**, origin top-left.
- **Visibility in `[0, 1]`**, MediaPipe semantics.
- **Missing data = NaN coords + visibility=0.** Never zero
  coordinates — zeros silently pollute averages.
- **Detectors are stateless across videos.** Instantiate per analysis,
  call `close()` after. No process-wide singletons.
- **`close()` is idempotent and exception-swallowing.** A failing
  cleanup should never mask a detection result or raise into the
  caller.
- **The pipeline is not allowed to import MediaPipe (or any detector
  library) directly.** Anything new goes through `PoseDetector`.

## Diagnostic surface

Every analysis attaches `summary.diagnostics.detector`:

```json
{
  "name": "mediapipe_tasks",
  "sport": "swim",
  "camera_angle": "under_water",
  "config": {
    "min_detection_confidence": 0.3,
    "min_presence_confidence": 0.3,
    "min_tracking_confidence": 0.3
  }
}
```

This is the infrastructure for scientific A/B testing of detector
swaps. Do not remove it.
