"""Which joints the overlay calls out.

The decision is made once per clip (never per frame, or chips would blink as a
value crosses its band mid-stride), and it is deliberately conservative: the
joints the score leans on are always labelled, everything else has to earn the
space by actually sitting outside its reference band.
"""
from __future__ import annotations

from types import SimpleNamespace

from app.services.video_analysis.video_visualizer import VideoVisualizer


def _visualizer(label_configs, angle_history, *, level="material",
                sport="bike", summary=None):
    """A visualizer with only the fields the decision reads.

    Built without __init__ on purpose: the real one opens a video, builds a
    pose pipeline and walks every frame, none of which this decision touches.
    """
    vis = object.__new__(VideoVisualizer)
    vis.label_configs = label_configs
    vis.analyzer = SimpleNamespace(angle_history=angle_history)
    vis.annotation_level = level
    vis.sport_type = sport
    vis.summary = summary or {}
    vis.cycling_position = "road_hoods"
    # The real __init__ caches this right after building the label configs;
    # _annotates() reads it.
    vis._material_keys = vis._pick_material_keys()
    return vis


def _cfg(key, optimal):
    return {"key": key, "optimal": optimal}


def test_quiet_joints_lose_their_callout():
    """A joint inside its band all clip long is not worth a label."""
    configs = [_cfg("right_knee", (138, 145)), _cfg("trunk_angle", (20, 30)),
               _cfg("right_elbow", (75, 110))]
    history = {
        "right_knee": [100.0] * 20,      # headline, always labelled
        "trunk_angle": [25.0] * 20,      # headline, always labelled
        "right_elbow": [90.0] * 20,      # squarely in band -> quiet
    }
    vis = _visualizer(configs, history)
    keys = vis._pick_material_keys()
    assert "right_elbow" not in keys
    assert vis._annotates("right_knee") and vis._annotates("trunk_angle")
    assert not vis._annotates("right_elbow")


def test_a_joint_outside_its_band_earns_a_callout():
    configs = [_cfg("right_knee", (138, 145)), _cfg("trunk_angle", (20, 30)),
               _cfg("right_hip", (30, 80))]
    history = {
        "right_knee": [140.0] * 20,
        "trunk_angle": [25.0] * 20,
        "right_hip": [110.0] * 20,       # far outside -> flagged
    }
    keys = _visualizer(configs, history)._pick_material_keys()
    assert "right_hip" in keys


def test_grazing_the_band_is_not_a_finding():
    """Outside the band on a handful of frames is noise, not a fault."""
    configs = [_cfg("right_knee", (138, 145)), _cfg("right_elbow", (75, 110))]
    # 10% of frames just outside: under the 25% share the rule asks for.
    history = {"right_knee": [140.0] * 20, "right_elbow": [90.0] * 18 + [130.0] * 2}
    assert "right_elbow" not in _visualizer(configs, history)._pick_material_keys()


def test_callouts_are_capped_worst_first():
    """Past four labels the frame is decoration -- keep the worst offenders."""
    configs = [
        _cfg("right_knee", (138, 145)), _cfg("trunk_angle", (20, 30)),
        _cfg("right_hip", (30, 80)), _cfg("right_shoulder", (80, 105)),
        _cfg("right_elbow", (75, 110)), _cfg("right_forearm_tilt", (5, 25)),
    ]
    history = {
        "right_knee": [140.0] * 20,
        "trunk_angle": [25.0] * 20,
        "right_hip": [200.0] * 20,          # worst
        "right_shoulder": [160.0] * 20,     # next worst
        "right_elbow": [112.0] * 20,        # only just outside
        "right_forearm_tilt": [26.5] * 20,  # only just outside
    }
    keys = _visualizer(configs, history)._pick_material_keys()
    assert len(keys) == VideoVisualizer._MAX_CALLOUTS
    assert {"right_knee", "trunk_angle", "right_hip", "right_shoulder"} == keys


def test_all_level_keeps_every_joint():
    configs = [_cfg("right_knee", (138, 145)), _cfg("right_elbow", (75, 110))]
    history = {"right_knee": [140.0] * 20, "right_elbow": [90.0] * 20}
    vis = _visualizer(configs, history, level="all")
    assert vis._pick_material_keys() == {"right_knee", "right_elbow"}
    assert vis._annotates("right_elbow")


def test_bike_extras_follow_their_own_clip_level_verdict():
    """Head position and pelvic tilt are judged on the summary, not on frames."""
    configs = [_cfg("right_knee", (138, 145))]
    history = {"right_knee": [140.0] * 20}
    quiet = _visualizer(configs, history,
                        summary={"head_alignment_avg": 88.0, "pelvic_ratio": 0})
    assert not quiet._annotates("head_alignment")
    assert not quiet._annotates("pelvic_ratio")

    flagged = _visualizer(configs, history,
                          summary={"head_alignment_avg": 40.0, "pelvic_ratio": 0})
    assert flagged._annotates("head_alignment")


def test_a_clean_clip_still_shows_the_headline_joints():
    """Nothing flagged must not mean a bare frame."""
    configs = [_cfg("left_knee", (138, 145)), _cfg("trunk_angle", (20, 30)),
               _cfg("left_elbow", (75, 110))]
    history = {"left_knee": [141.0] * 20, "trunk_angle": [25.0] * 20,
               "left_elbow": [90.0] * 20}
    keys = _visualizer(configs, history)._pick_material_keys()
    assert keys == {"left_knee", "trunk_angle"}
