"""Re-measuring a photo after the athlete moves a joint point.

The photo path stores nothing: the picture is analysed on the request that
carries it and forgotten. A correction therefore has to bring everything back
with it -- the photo (still in the athlete's browser) and the pose the model
found (handed to the client with the first result). The pose comes back
SIGNED, so the baseline a correction is applied to is the one this server
produced, not something the client typed: the report says "adjusted by the
athlete" about exactly the delta they made, on exactly the pose we detected.
The signature also binds the pose to the photo by its hash, so a pose cannot
be re-used on a different picture.

Stateless on purpose. A server-side cache of detected poses would do the same
job with a TTL and a sweeper, and it would forget on every deploy; an HMAC
over the blob costs nothing and remembers forever.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from types import SimpleNamespace
from typing import Any

from app.core.config import settings
from app.services.video_analysis.biomechanics.head_silhouette import (
    CONTOUR_COLUMNS,
    HELMET_POINT_NAMES,
)
from app.services.video_analysis.biomechanics.corrections import (
    DRAGGABLE_LANDMARKS,
    apply_corrections,
    check_plausibility,
    normalize_corrections,
)

POSE_VERSION = 1
_SIGNED_KEYS = ("v", "image_sha256", "camera_side", "landmarks", "world")
_N = 33


def _r(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, 6) if math.isfinite(f) else None


def _pack(landmarks: Any) -> list[list[float | None]]:
    return [
        [_r(getattr(lm, "x", None)), _r(getattr(lm, "y", None)),
         _r(getattr(lm, "z", 0.0)), _r(getattr(lm, "visibility", 1.0))]
        for lm in landmarks
    ]


def _unpack(rows: list[list[Any]]) -> list[SimpleNamespace]:
    out = []
    for row in rows:
        x, y, z, vis = (list(row) + [None] * 4)[:4]
        out.append(SimpleNamespace(
            x=math.nan if x is None else float(x),
            y=math.nan if y is None else float(y),
            z=0.0 if z is None else float(z),
            visibility=1.0 if vis is None else float(vis),
        ))
    return out


def _canonical(blob: dict[str, Any]) -> bytes:
    # The silhouette edge and the sport are signed when the blob carries them.
    # Blobs issued before they existed have no such keys and verify exactly as
    # they did (a blob without a sport is a bike pose).
    keys = _SIGNED_KEYS + tuple(k for k in ("silhouette_top", "sport") if k in blob)
    return json.dumps(
        {k: blob.get(k) for k in keys},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()


def _sign(blob: dict[str, Any]) -> str:
    return hmac.new(
        settings.jwt_secret.encode(), _canonical(blob), hashlib.sha256,
    ).hexdigest()


def image_sha256(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()


def build_pose_blob(
    image_bytes: bytes, pose_result: dict[str, Any], *, helmet_points: bool = False,
    sport: str = "bike",
) -> dict[str, Any]:
    """What the client keeps so it can ask for a re-measurement later.

    ``helmet_points``: the photo carries a helmet reading, so its top-of-back
    and top-of-helmet points are offered for dragging too.

    ``sport="run"``: ``pose_result`` carries the legs already NAMED by position
    (photo_analyzer._name_run_legs) -- the stance leg on the camera-side
    indices -- so the points the athlete drags are the leg the report
    measures, not whichever leg the model labelled near, its hip included.
    The ear is not offered: no running angle reads it.
    """
    side = pose_result["camera_side"]
    draggable = list(DRAGGABLE_LANDMARKS.get(side, ()))
    if sport == "run":
        draggable = [i for i in draggable if i not in (7, 8)]
    blob: dict[str, Any] = {
        "v": POSE_VERSION,
        "image_sha256": image_sha256(image_bytes),
        "camera_side": side,
        "landmarks": _pack(pose_result["normalized"]),
        "world": _pack(pose_result["world"]),
        "warnings": [str(w) for w in (pose_result.get("warnings") or [])],
        # Which points the editor may offer -- decided here, not in the client.
        "draggable": draggable + (list(HELMET_POINT_NAMES) if helmet_points else []),
    }
    if sport != "bike":
        blob["sport"] = sport
    if sport == "run":
        from app.services.video_analysis.photo_analyzer import _run_pelvis_hip

        # A far stance leg took its own hip across (the point the athlete
        # drags as "hip"); the visible hip the trunk is read from is then the
        # other index. For the editor's drawing only -- like ``draggable``,
        # not signed: the re-measure finds it again from the signed depths.
        pelvis = _run_pelvis_hip(_unpack(blob["world"]), side)
        if pelvis != (23 if side == "left" else 24):
            blob["pelvis_hip"] = pelvis
    # The silhouette's top edge: the helmet-above-back reading needs it again
    # after a correction, and there is no second model pass to get it from.
    if pose_result.get("silhouette_top"):
        blob["silhouette_top"] = list(pose_result["silhouette_top"])
    blob["token"] = _sign(blob)
    return blob


def verify_pose_blob(blob: Any, image_bytes: bytes) -> dict[str, Any]:
    """The blob as this server issued it for this photo, or ``ValueError``."""
    if not isinstance(blob, dict):
        raise ValueError("pose must be the object the analysis returned")
    if blob.get("v") != POSE_VERSION:
        raise ValueError("this pose was produced by a different version -- analyze the photo again")
    for key in ("landmarks", "world"):
        rows = blob.get(key)
        if not isinstance(rows, list) or len(rows) != _N:
            raise ValueError("pose landmarks are malformed -- analyze the photo again")
    if blob.get("camera_side") not in ("left", "right"):
        raise ValueError("pose camera side is malformed -- analyze the photo again")
    if blob.get("sport", "bike") not in ("bike", "run"):
        raise ValueError("pose sport is malformed -- analyze the photo again")
    if "silhouette_top" in blob:
        edge = blob["silhouette_top"]
        if not isinstance(edge, list) or len(edge) != CONTOUR_COLUMNS or not all(
            v is None or (isinstance(v, (int, float)) and 0.0 <= v <= 1.0) for v in edge
        ):
            raise ValueError("pose silhouette is malformed -- analyze the photo again")
    token = blob.get("token")
    if not isinstance(token, str) or not hmac.compare_digest(token, _sign(blob)):
        raise ValueError("this pose was not issued by this server for this photo")
    if blob.get("image_sha256") != image_sha256(image_bytes):
        raise ValueError("this pose belongs to a different photo -- upload the one it was measured on")
    return blob


def recompute_photo(
    image_bytes: bytes,
    sport: str,
    cycling_position: str | None,
    pose_blob: Any,
    corrections_raw: list[dict[str, Any]] | None,
    *,
    hide_angle_values: bool = False,
    bike_points: dict[str, Any] | None = None,
    crank_length_mm: float | None = None,
) -> dict[str, Any]:
    """Measure the photo again with the athlete's corrections applied.

    Blocking (decode + thumbnail); the endpoint runs it in a thread. Returns
    the same shape as ``analyze_photo`` plus ``baseline`` (the automatic
    angles and score, kept so the adjustment is visible as a delta),
    ``plausibility_warnings`` and the verified ``pose`` blob handed back.
    Raises ``ValueError`` with a message meant for the athlete.

    ``bike_points``: the crank axle (and, if given, the pedal and wheel
    axles) the athlete placed -- see biomechanics/hip_over_bb. A photo can
    be re-measured for that alone, with no joint moved.

    ``sport="run"``: a running photo read the way the run-photo flag reads it
    (``contact_metrics``: image-plane angles, the stance leg named by
    position, the touchdown block). The caller decides who gets that; the
    pose has to have been issued for a run.
    """
    from app.services.video_analysis.biomechanics.hip_over_bb import (
        hip_over_bb,
        normalize_bike_points,
    )
    from app.services.video_analysis.photo_analyzer import (
        analyze_from_pose,
        decode_photo,
    )

    if sport not in ("bike", "run"):
        raise ValueError(
            "Adjusting joint points is available for cycling and running photos only."
        )
    blob = verify_pose_blob(pose_blob, image_bytes)
    if blob.get("sport", "bike") != sport:
        raise ValueError("this pose was measured for a different sport -- analyze the photo again")
    run = sport == "run"
    if run and bike_points:
        raise ValueError("The crank axle is marked on cycling photos only.")
    side = blob["camera_side"]
    corrections = normalize_corrections(
        corrections_raw, side, extra_points=None if run else HELMET_POINT_NAMES,
    )
    points = normalize_bike_points(bike_points) if bike_points else None
    if not corrections and not points:
        raise ValueError("No adjustment to apply -- move a joint point first.")
    # The top of the back and of the helmet are not skeleton joints: they move
    # the helmet reading only, never a landmark or a bone length.
    joints = [c for c in corrections if c["landmark"] not in HELMET_POINT_NAMES]
    helmet_offsets = {c["landmark"]: (c["dx"], c["dy"])
                      for c in corrections if c["landmark"] in HELMET_POINT_NAMES}

    image = decode_photo(image_bytes)
    h, w = image.shape[:2]
    warnings = list(blob.get("warnings") or [])

    # Baseline: the automatic reading, measured on the pose as detected.
    silhouette_top = blob.get("silhouette_top")
    baseline = analyze_from_pose(
        image, _unpack(blob["world"]), _unpack(blob["landmarks"]), side, warnings,
        sport, cycling_position, hide_angle_values=hide_angle_values,
        render_thumbnail=False, silhouette_top=silhouette_top, contact_metrics=run,
    )

    frame = {
        "normalized_landmarks": _unpack(blob["landmarks"]),
        "world_landmarks": _unpack(blob["world"]),
        "frame_width": w, "frame_height": h,
    }
    plausibility = check_plausibility(
        [frame], joints, side, aspect=(w / h) if h else 1.0, min_samples=1,
    ) if joints else []
    apply_corrections([frame], joints, sport, image_plane_angles=True)

    result = analyze_from_pose(
        image, frame["world_landmarks"], frame["normalized_landmarks"], side,
        warnings, sport,
        # Judge the corrected pose against the position the automatic one was
        # filed under: a moved shoulder must not silently re-file the rider
        # from "triathlon" to "casual" and change every band under them.
        cycling_position or baseline.get("cycling_position"),
        hide_angle_values=hide_angle_values, corrections=corrections,
        silhouette_top=silhouette_top, helmet_offsets=helmet_offsets,
        contact_metrics=run,
    )
    result["baseline"] = {
        "angles": baseline.get("angles"),
        "score": baseline.get("score"),
        "cycling_position": baseline.get("cycling_position"),
    }
    if run:
        # A moved foot can move the stride instant and the touchdown readings,
        # not only the angles.
        result["baseline"]["gait_phase"] = baseline.get("gait_phase")
        result["baseline"]["contact"] = baseline.get("contact")
    result["plausibility_warnings"] = plausibility
    result["pose"] = blob
    if points:
        nl = frame["normalized_landmarks"]
        hip_i, sh_i = (23, 11) if side == "left" else (24, 12)
        hip = (float(nl[hip_i].x), float(nl[hip_i].y))
        if not all(math.isfinite(v) for v in hip):
            raise ValueError("The hip was not found on this photo, so the crank "
                             "cannot be measured against it.")
        sh = (float(nl[sh_i].x), float(nl[sh_i].y))
        result["hip_over_bb"] = hip_over_bb(
            hip, points, aspect=(w / h) if h else 1.0,
            shoulder=sh if all(math.isfinite(v) for v in sh) else None,
            crank_length_mm=crank_length_mm, hip_source="photo",
            cycling_position=result.get("cycling_position") or cycling_position,
        )
        result["bike_points"] = result["hip_over_bb"]["points"]
        result["thumbnail_base64"] = _with_hip_over_bb(
            result.get("thumbnail_base64"), result["hip_over_bb"],
        )
    return result


def _with_hip_over_bb(thumb: Any, hob: dict[str, Any]) -> Any:
    """The annotated photo with the crank line drawn on, or unchanged."""
    if not isinstance(thumb, str) or not thumb.startswith("data:image/"):
        return thumb
    try:
        import base64

        import cv2
        import numpy as np

        from app.services.video_analysis import overlay_style

        _head, b64 = thumb.split(",", 1)
        img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return thumb
        th, tw = img.shape[:2]
        bb = (int(round(hob["points"]["C"][0] * tw)), int(round(hob["points"]["C"][1] * th)))
        hip = (int(round(hob["hip"][0] * tw)), int(round(hob["hip"][1] * th)))
        overlay_style.draw_hip_over_bb(
            cv2, img, bb, hip, float(hob["angle_deg"]),
            1 if hob.get("facing") == "right" else -1, scale=max(1.0, min(2.4, tw / 900)),
        )
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            return thumb
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
    except Exception:  # noqa: BLE001 -- the picture is a courtesy; the number is the product
        return thumb


__all__ = [
    "POSE_VERSION",
    "build_pose_blob",
    "image_sha256",
    "recompute_photo",
    "verify_pose_blob",
]
