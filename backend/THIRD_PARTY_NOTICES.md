# Third-party notices

Flapp's analysis server uses the following third-party software and models.
None of them is redistributed to users: they run on the server only.

## MediaPipe (pose landmarker)

- Google LLC, Apache License 2.0 — https://github.com/google-ai-edge/mediapipe
- Model: `pose_landmarker_heavy.task`, downloaded at image build time.

## RTMPose / rtmlib (pose refinement)

- `app/services/video_analysis/detectors/rtmpose_refiner.py` reimplements the
  top-down preprocessing (box padding, affine crop, normalisation) and the
  SimCC decoding of **rtmlib** by Tau-J, Apache License 2.0 —
  https://github.com/Tau-J/rtmlib. Changes: rewritten on onnxruntime and
  OpenCV without the rtmlib package; scores combined as the mean of the two
  axes, as in rtmlib's 2D path.
- RTMPose is part of **MMPose** by OpenMMLab, Apache License 2.0 —
  https://github.com/open-mmlab/mmpose.
- Model: `rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605`
  (ONNX, SHA-256 `26f3a19e61304a600dfb82d1001d41d24343b89fc70a33ffc84657e0b0bf2ecf`),
  downloaded at image build time from download.openmmlab.com. It was trained
  on the Body7 datasets (AI Challenger, MS COCO, CrowdPose, MPII, sub-JHMDB,
  Halpe, PoseTrack18), each under its own terms.

## ONNX Runtime

- Microsoft Corporation, MIT License — https://github.com/microsoft/onnxruntime

---

Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0
