"""Hip over the bottom bracket: where the rider sits over the crank.

The angle of the line from the crank axle up to the hip joint, measured from
the horizontal on the rider's back side. It is the one number the Kona 2026
pros and age-groupers differed on in our field study (hipknee2026, 31 pros
and 68 age-groupers, every point placed by hand): pros 88.0 deg, age-groupers
85.1, a 3.1 deg gap [2.1, 4.1] that holds in men and women, about 4-5 cm
further forward over the crank. Knee and hip angles did not differ.

The crank axle is not a landmark the pose model finds, and no pose-derived
guess of it is good enough (crank_phase_dead): the athlete places it, with
the pedal axle and the two wheel axles if they want more. Why the axles: a
2 deg tilt of the phone is a 2 deg error on a 3 deg effect, and on a trainer
with the front wheel on its block the axle line IS level. Given, it is the
horizontal; not given, the frame edge is, and the result says so.

On a clip the hip is the median of every frame's near-side hip: the pelvis
rocks through the stroke, and one frame is one phase. On a photo it is that
frame's hip. Both are measured against the same field, which was read from
single dead-centre frames; within-rider repeatability in the study was
+/-1.4 deg and the output carries that.

The field is an AERO field: every rider in it sat on extensions, where the
hip is further forward by design (a 78-80 deg seat tube against a road
bike's 72-74). A rider on the hoods measured against it reads "5 deg behind
the pros" for sitting on a road bike (2026-11-08). So on a road or upright
position the comparison fields are None: the angle stands on its own, for
the rider's own setups over time. A road field needs its own study.
"""
from __future__ import annotations

import math
from statistics import median
from typing import Any

# Pro / age-group fields: median, middle half, full range (final_stats.json,
# 2026-10-24: Simmonds #6/#7 and blurry #102 excluded; 16 of the frames are Scott
# Flathouse's full-resolution originals, the rest store previews). A FIELD, never a band.
FIELD: dict[str, dict[str, float]] = {
    "pro": {"median": 88.0, "q1": 86.4, "q3": 89.7, "lo": 84.8, "hi": 94.6, "n": 31},
    "age_group": {"median": 85.1, "q1": 83.4, "q3": 87.0, "lo": 79.0, "hi": 90.6, "n": 68},
}
DIFF_DEG = {"hl": 3.1, "ci": [2.1, 4.1]}
REPEATABILITY_DEG = 1.4
# Crank-to-hip is about 83 cm on an adult tri bike (the study's image ruler):
# one degree of this angle is about 1.45 cm of hip travel. Used only when no
# crank length gives a real scale, and labelled as typical.
CM_PER_DEG_TYPICAL = 1.45
DEFAULT_CRANK_MM = 172.5
SOURCE = ("Flapp field study, Ironman World Championship 2026: 31 professional and "
          "68 age-group riders, hip and crank axle placed by hand on race photographs")
ARTICLE = "/academy/kona-2026-hip-over-crank"

# What the athlete places, in normalized image coordinates.
POINT_KEYS = ("C", "P", "FA", "RA", "SN", "ST")
POINT_NAMES = {"C": "crank axle", "P": "pedal axle", "FA": "front axle", "RA": "rear axle",
               "SN": "saddle nose", "ST": "saddle back"}


def normalize_bike_points(raw: Any) -> dict[str, tuple[float, float]]:
    """Validate what the client sent: ``{"C": [x, y], ...}`` in 0..1.

    ``C`` is required; the others optional. Raises ``ValueError`` with a
    message meant for the athlete.
    """
    if not isinstance(raw, dict):
        raise ValueError("bike points must be an object of named points")
    out: dict[str, tuple[float, float]] = {}
    for key, val in raw.items():
        if key not in POINT_KEYS:
            raise ValueError(f"unknown point '{key}'; expected one of {', '.join(POINT_KEYS)}")
        if (not isinstance(val, (list, tuple)) or len(val) != 2
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                           and math.isfinite(v) for v in val)):
            raise ValueError(f"{POINT_NAMES[key]} must be [x, y]")
        x, y = float(val[0]), float(val[1])
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"{POINT_NAMES[key]} is outside the picture")
        out[key] = (round(x, 5), round(y, 5))
    if "C" not in out:
        raise ValueError("Mark the crank axle first -- the measurement starts there.")
    if ("FA" in out) != ("RA" in out):
        raise ValueError("Mark both wheel axles, or neither.")
    if "FA" in out and math.dist(out["FA"], out["RA"]) < 0.05:
        raise ValueError("The two wheel axles sit on top of each other.")
    if "P" in out and math.dist(out["P"], out["C"]) < 0.005:
        raise ValueError("The pedal axle sits on the crank axle.")
    if ("SN" in out) != ("ST" in out):
        raise ValueError("Mark both ends of the saddle, or neither.")
    if "SN" in out and math.dist(out["SN"], out["ST"]) < 0.02:
        raise ValueError("The two saddle points sit on top of each other.")
    return out


# ---------------------------------------------------------------------------
# Saddle tilt: the line along the top of the saddle against the wheel axles
# ---------------------------------------------------------------------------

# Fitter guidance, not a measured band (no study has measured where riders do
# best): level, at most 2-3 deg nose down, and nose down past that is usually
# a saddle that is too high -- the rider tips the nose to get off the front.
# Burt, Bike Fit 2014, and his saddle interview on MyVeloFit: heavy tilt compensates
# for something else (a high saddle, long cranks, a big drop). Fitters'
# practice (research/myvelofit) says the same. The line runs nose to back, which on a saddle with a kicked-up tail reads a
# little more nose down than its flat middle, so the "nose down" call starts
# a degree past the fitters' 3.
TILT_NOSE_DOWN_DEG = -4.0
TILT_NOSE_UP_DEG = 2.0
# One tap lands within about 0.4 % of the picture; two of them over a saddle
# a tenth of the picture long are a couple of degrees.
_TAP_ERR = 0.004
TILT_SOURCE = ("Fitter guidance, not a measured range (Burt, Bike Fit 2014): start level, at most "
               "2-3 deg nose down; more than that usually makes up for something else, most often "
               "a saddle that is too high")


def saddle_tilt(points: dict[str, tuple[float, float]], *, aspect: float,
                knee_bdc_high: bool = False) -> dict[str, Any] | None:
    """The saddle's nose-to-back line in degrees, + nose up, - nose down.

    Levelled on the wheel axles when both are marked: that line is the bike's
    own horizontal whatever the trainer or the phone does. Without them it is
    the frame edge, and the result says so. None when the saddle is not marked.
    """
    if "SN" not in points or "ST" not in points:
        return None
    nx, ny = points["SN"]
    tx, ty = points["ST"]
    # Up is positive from here on; the saddle runs back -> nose.
    sx, sy = (nx - tx) * aspect, ty - ny
    length = math.hypot(sx, sy)
    if length < 1e-6:
        raise ValueError("The two saddle points sit on top of each other.")
    if "FA" in points and "RA" in points:
        gx = (points["FA"][0] - points["RA"][0]) * aspect
        gy = points["RA"][1] - points["FA"][1]
        ref = "axles"
    else:
        # Forward is wherever the nose points.
        gx, gy = (1.0 if sx >= 0 else -1.0), 0.0
        ref = "frame"
    g = math.hypot(gx, gy)
    gx, gy = gx / g, gy / g
    ux, uy = (-gy, gx) if gx >= 0 else (gy, -gx)  # the normal that points up
    deg = math.degrees(math.atan2(sx * ux + sy * uy, sx * gx + sy * gy))
    if abs(deg) > 25.0:
        raise ValueError("That saddle line is steeper than any saddle sits "
                         "-- check the nose and the back points.")
    pm = max(1.0, round(math.degrees(math.atan(_TAP_ERR * math.sqrt(2) / length)), 0))
    if deg <= TILT_NOSE_DOWN_DEG:
        verdict = "nose_down"
    elif deg >= TILT_NOSE_UP_DEG:
        verdict = "nose_up"
    else:
        verdict = "level"
    return {
        "deg": round(deg, 1),
        "plus_minus_deg": pm,
        "ref": ref,
        "verdict": verdict,
        "nose_down_from_deg": TILT_NOSE_DOWN_DEG,
        "nose_up_from_deg": TILT_NOSE_UP_DEG,
        "note": _tilt_note(deg, verdict, knee_bdc_high),
        "source": TILT_SOURCE,
    }


def _tilt_note(deg: float, verdict: str, knee_bdc_high: bool) -> str:
    d = f"{abs(deg):.0f}"
    if verdict == "nose_down" and knee_bdc_high:
        return (f"Nose down {d}°, and the bottom of the stroke already says the saddle is "
                "high: the nose went down to get off the front. Lower the saddle first, then "
                "level it and film again")
    if verdict == "nose_down":
        return (f"Nose down {d}°. Fitters start level and stop at 2-3° nose down; more than "
                "that usually makes up for something else, most often a saddle that is too "
                "high. Check the height, then level the middle of the saddle")
    if verdict == "nose_up":
        return (f"Nose up {d}°. No saddle is shaped to be ridden nose up: it rolls the pelvis "
                "back and loads the front. Level the middle of the saddle")
    return "Level, inside what fitters aim for"


def report_says_knee_high(result: dict[str, Any] | None) -> bool:
    """Whether the report already says the knee opens too far at the bottom.

    A video plan says it as "lower the saddle"; a photo says it on the knee row
    of a frame caught at the bottom of the stroke.
    """
    if not isinstance(result, dict):
        return False
    for d in (result.get("fit_plan") or {}).get("diagnostics") or []:
        if isinstance(d, dict) and d.get("action") == "lower_saddle":
            return True
    knee = (result.get("angles_with_context") or {}).get("knee") or {}
    hi, val = knee.get("optimal_max"), knee.get("value")
    return (result.get("pedal_phase") == "bdc" and knee.get("status") == "needs_work"
            and isinstance(hi, (int, float)) and isinstance(val, (int, float)) and val > hi)


def _facing(shoulder_x: float | None, hip_x: float) -> int:
    """+1 when the rider faces image-right (shoulder ahead of the hip)."""
    if shoulder_x is None or not math.isfinite(shoulder_x):
        return 1
    return 1 if shoulder_x >= hip_x else -1


def hip_over_bb(
    hip: tuple[float, float],
    points: dict[str, tuple[float, float]],
    *,
    aspect: float,
    shoulder: tuple[float, float] | None = None,
    crank_length_mm: float | None = None,
    hip_source: str = "photo",
    frames: int | None = None,
    cycling_position: str | None = None,
    knee_bdc_high: bool = False,
) -> dict[str, Any]:
    """The angle, its horizontal, its scale and its place in the field.

    ``aspect`` = frame width / height, so normalized offsets become true
    proportions. ``shoulder`` decides which way the rider faces; without it
    image-right is assumed.
    """
    hx, hy = hip
    cx, cy = points["C"]
    facing = _facing(shoulder[0] if shoulder else None, hx)
    # Up is positive from here on.
    vx, vy = (hx - cx) * aspect, cy - hy
    if math.hypot(vx, vy) < 1e-6:
        raise ValueError("The hip and the crank axle sit on top of each other.")
    theta_v = math.degrees(math.atan2(vy, vx))
    ref = "frame"
    tilt = 0.0
    if "FA" in points and "RA" in points:
        gx, gy = (points["FA"][0] - points["RA"][0]) * aspect, points["RA"][1] - points["FA"][1]
        # The axles are level on the bike; whichever way it points, the ground
        # direction runs with the rider.
        theta_g = math.degrees(math.atan2(gy, gx))
        forward = 0.0 if facing > 0 else 180.0
        tilt = (theta_g - forward + 180.0) % 360.0 - 180.0
        if abs(tilt) > 25.0:
            raise ValueError("The wheel axles are not where a level bike would put them "
                             "-- check the two axle points.")
        ref = "axles"
    else:
        theta_g = 0.0 if facing > 0 else 180.0
    angle = abs(180.0 - ((theta_v - theta_g) % 360.0))

    # Scale: the crank arm, when the pedal axle is marked; else typical.
    dist_norm = math.hypot(vx, vy)
    cm_per_deg = CM_PER_DEG_TYPICAL
    scale_source = "typical"
    crank_to_hip_cm = None
    if "P" in points:
        px, py = points["P"]
        crank_norm = math.hypot((px - cx) * aspect, cy - py)
        mm = float(crank_length_mm) if crank_length_mm else DEFAULT_CRANK_MM
        if crank_norm > 1e-6:
            crank_to_hip_cm = dist_norm / crank_norm * mm / 10.0
            # A hip 40-110 cm from the crank is a human on a bicycle; outside
            # that the pedal point is not on the pedal.
            if 40.0 <= crank_to_hip_cm <= 110.0:
                cm_per_deg = round(crank_to_hip_cm * math.pi / 180.0, 3)
                scale_source = "crank_given" if crank_length_mm else "crank_default"
            else:
                crank_to_hip_cm = None

    pro, ag = FIELD["pro"], FIELD["age_group"]
    aero = cycling_position in ("tt_aero", "triathlon")
    out: dict[str, Any] = {
        "angle_deg": round(angle, 1),
        "ref": ref,
        "tilt_deg": round(tilt, 1),
        "facing": "right" if facing > 0 else "left",
        "hip_source": hip_source,
        "frames": frames,
        "hip": [round(hx, 5), round(hy, 5)],
        "points": {k: [v[0], v[1]] for k, v in points.items()},
        "cm_per_deg": cm_per_deg,
        "scale_source": scale_source,
        "crank_to_hip_cm": None if crank_to_hip_cm is None else round(crank_to_hip_cm, 1),
        # The comparison is only made where the field applies (see the module
        # docstring); the field itself rides along so a reader can say why not.
        "gap_pro_deg": round(angle - pro["median"], 1) if aero else None,
        "gap_pro_cm": round((angle - pro["median"]) * cm_per_deg, 1) if aero else None,
        "gap_age_group_deg": round(angle - ag["median"], 1) if aero else None,
        "in_pro_middle_half": (pro["q1"] <= angle <= pro["q3"]) if aero else None,
        "in_age_group_middle_half": (ag["q1"] <= angle <= ag["q3"]) if aero else None,
        "field": {k: dict(v) for k, v in FIELD.items()},
        "diff": dict(DIFF_DEG),
        "repeatability_deg": REPEATABILITY_DEG,
        "aero": aero,
        "source": SOURCE,
        "article": ARTICLE,
        # The saddle line, when the athlete marked it: same picture, same
        # horizontal (hip_over_bb and saddle_tilt both level on the axles).
        "saddle_tilt": saddle_tilt(points, aspect=aspect, knee_bdc_high=knee_bdc_high),
    }
    return out


def median_hip(frames: list[dict[str, Any]], camera_side: str,
               corrections: list[dict[str, Any]] | None = None,
               ) -> tuple[tuple[float, float] | None, tuple[float, float] | None, int]:
    """The near-side hip and shoulder over a clip: per-axis medians of every
    frame that has them, with the corrections in force applied. Returns
    ``(hip, shoulder, n_frames)``; ``hip`` is None when nothing was tracked.
    """
    hip_i, sh_i = (23, 11) if camera_side == "left" else (24, 12)
    off = {int(c["landmark"]): (float(c.get("dx") or 0.0), float(c.get("dy") or 0.0))
           for c in (corrections or []) if isinstance(c, dict) and "landmark" in c}
    hx, hy, sx, sy = [], [], [], []
    for f in frames:
        lm = f.get("normalized_landmarks") or []
        if len(lm) <= max(hip_i, sh_i):
            continue
        h = lm[hip_i]
        x, y = float(getattr(h, "x", math.nan)), float(getattr(h, "y", math.nan))
        if math.isfinite(x) and math.isfinite(y) and float(getattr(h, "visibility", 1.0)) >= 0.3:
            hx.append(x + off.get(hip_i, (0.0, 0.0))[0])
            hy.append(y + off.get(hip_i, (0.0, 0.0))[1])
        s = lm[sh_i]
        x, y = float(getattr(s, "x", math.nan)), float(getattr(s, "y", math.nan))
        if math.isfinite(x) and math.isfinite(y):
            sx.append(x + off.get(sh_i, (0.0, 0.0))[0])
            sy.append(y + off.get(sh_i, (0.0, 0.0))[1])
    if not hx:
        return None, None, 0
    sh = (median(sx), median(sy)) if sx else None
    return (median(hx), median(hy)), sh, len(hx)


__all__ = [
    "ARTICLE", "CM_PER_DEG_TYPICAL", "DEFAULT_CRANK_MM", "DIFF_DEG", "FIELD",
    "POINT_KEYS", "POINT_NAMES", "REPEATABILITY_DEG", "SOURCE",
    "hip_over_bb", "median_hip", "normalize_bike_points", "report_says_knee_high", "saddle_tilt",
]
