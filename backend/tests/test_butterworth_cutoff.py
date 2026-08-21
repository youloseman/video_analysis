"""Residual analysis has to work, and it has to not be wired to the steering.

Two halves. First: the method (Winter 2009) is only worth having if it actually
separates a clean signal from a noisy one, so the tests build both with the
noise level known and check the chosen cutoff moves the right way.

Second, and just as important: it currently runs as a REPORTED DIAGNOSTIC and
must not touch what the filter does. Flapp low-passes the landmark coordinates
at 6-8 Hz before any angle exists, so by this stage there is no noise floor left
for the method to find and it saturates at its ceiling -- acting on that would
silently turn the second smoothing stage off and move every graded percentile.
So there are tests here whose whole job is to pin the operating cutoff to the
sport constant.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.services.video_analysis.biomechanics import butterworth_filter as bw

FPS = 50.0


def signal(hz=1.4, seconds=6.0, noise=0.0, fps=FPS, seed=0):
    """A joint-angle-ish sinusoid with its first harmonic, plus white noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1.0 / fps)
    clean = 120.0 + 35.0 * np.sin(2 * np.pi * hz * t) + 8.0 * np.sin(
        2 * np.pi * 2 * hz * t
    )
    return clean + rng.normal(0.0, noise, size=t.size)


# --- Durbin-Watson ---------------------------------------------------------

def test_white_noise_scores_near_two():
    rng = np.random.default_rng(1)
    assert bw.durbin_watson(rng.normal(size=4000)) == pytest.approx(2.0, abs=0.1)


def test_a_smooth_series_is_strongly_autocorrelated():
    t = np.linspace(0, 10, 1000)
    assert bw.durbin_watson(np.sin(t)) < 0.5


def test_a_degenerate_series_has_no_answer_rather_than_a_flattering_one():
    assert np.isnan(bw.durbin_watson(np.zeros(50)))
    assert np.isnan(bw.durbin_watson(np.array([1.0, 2.0])))


# --- the sweep itself ------------------------------------------------------

def test_a_noisier_signal_gets_a_lower_cutoff():
    """The whole point: more noise on the same movement should be filtered
    harder. If this does not hold, the method is not measuring anything."""
    bounds = bw.SPORT_CUTOFF_BOUNDS["run"]
    quiet, _ = bw.select_cutoff(signal(noise=0.3), FPS, bounds)
    loud, _ = bw.select_cutoff(signal(noise=6.0), FPS, bounds)
    assert quiet is not None and loud is not None
    assert loud < quiet


def test_the_chosen_cutoff_keeps_the_movement_it_was_given():
    """A 1.4 Hz stride with a 2.8 Hz harmonic must survive: the filter is
    supposed to remove the noise, not the athlete."""
    data = signal(hz=1.4, noise=3.0)
    fc, _ = bw.select_cutoff(data, FPS, bw.SPORT_CUTOFF_BOUNDS["run"])
    assert fc > 2.8

    from scipy.signal import butter, sosfiltfilt
    sos = butter(bw.FILTER_ORDER, fc / (FPS / 2), btype="low", output="sos")
    filtered = sosfiltfilt(sos, data)
    clean = signal(hz=1.4, noise=0.0)
    # Closer to the truth than the noisy input was.
    assert np.std(filtered - clean) < np.std(data - clean)


def test_the_sweep_never_leaves_its_band():
    for noise in (0.1, 2.0, 20.0):
        fc, _ = bw.select_cutoff(signal(noise=noise), FPS,
                                 bw.SPORT_CUTOFF_BOUNDS["run"])
        lo, hi = bw.SPORT_CUTOFF_BOUNDS["run"]
        assert lo <= fc <= min(hi, bw.NYQUIST_SAFETY * FPS / 2)


def test_a_low_frame_rate_pulls_the_ceiling_down_to_stay_off_nyquist():
    fps = 12.0
    fc, _ = bw.select_cutoff(signal(noise=0.5, fps=fps), fps,
                             bw.SPORT_CUTOFF_BOUNDS["run"])
    assert fc <= bw.NYQUIST_SAFETY * fps / 2


def test_a_flat_channel_has_no_answer_instead_of_a_confident_one():
    fc, score = bw.select_cutoff(np.full(200, 90.0), FPS,
                                 bw.SPORT_CUTOFF_BOUNDS["run"])
    assert fc is None and score == float("inf")


def test_a_series_too_short_to_filter_has_no_answer():
    fc, _ = bw.select_cutoff(np.arange(5.0), FPS, bw.SPORT_CUTOFF_BOUNDS["run"])
    assert fc is None


# --- applied to a whole angle_history --------------------------------------

def history(**channels):
    return {k: list(v) for k, v in channels.items()}


def test_the_operating_cutoff_is_still_the_sport_constant():
    """The guard on the whole change: residual analysis reports, it does not
    steer. Acting on it here would turn the second smoothing stage off."""
    h = history(knee=signal(hz=1.4, noise=2.0, seed=1))
    info = bw.apply_butterworth_filter(h, FPS, "run")
    assert info["cutoff_hz"] == pytest.approx(bw.SPORT_CUTOFFS["run"])
    assert info["suggested_applied"] is False


def test_the_suggestion_is_reported_next_to_the_cutoff_that_was_used():
    h = history(
        knee=signal(hz=1.4, noise=2.0, seed=1),
        trunk=110.0 + 2.0 * np.sin(np.linspace(0, 6 * np.pi, int(6 * FPS))),
    )
    info = bw.apply_butterworth_filter(h, FPS, "run")
    assert set(info["suggested_cutoffs"]) == {"knee", "trunk"}
    assert info["suggested_method"] == "winter_residual_analysis"


def test_channels_are_suggested_their_own_cutoffs():
    """A knee and a trunk do not carry the same frequencies, which is the whole
    reason one constant for both is worth questioning."""
    h = history(
        knee=signal(hz=1.4, noise=2.0, seed=1),
        trunk=110.0 + 2.0 * np.sin(np.linspace(0, 6 * np.pi, int(6 * FPS))),
    )
    info = bw.apply_butterworth_filter(h, FPS, "run")
    assert info["suggested_cutoffs"]["knee"] != info["suggested_cutoffs"]["trunk"]


def test_a_short_channel_gets_no_suggestion_rather_than_a_shaky_one():
    n = bw.AUTO_CUTOFF_MIN_SAMPLES - 1
    h = history(knee=signal(seconds=n / FPS, noise=1.0)[:n])
    info = bw.apply_butterworth_filter(h, FPS, "run")
    assert "knee" not in info["suggested_cutoffs"]
    assert "knee" in info["filtered"]            # still filtered, at the constant
    assert info["cutoff_hz"] == pytest.approx(bw.SPORT_CUTOFFS["run"])


def test_a_flat_channel_is_still_filtered_and_simply_has_no_suggestion():
    h = history(trunk=np.full(300, 95.0))
    info = bw.apply_butterworth_filter(h, FPS, "run")
    assert "trunk" not in info["suggested_cutoffs"]
    assert "trunk" in info["filtered"]


def test_left_and_right_of_a_joint_are_suggested_the_lower_cutoff():
    """Filtering one side harder than the other would show up downstream as an
    asymmetry the athlete does not have -- so the numbers reported are the ones
    that would actually be usable."""
    h = history(
        left_knee=signal(hz=1.4, noise=0.5, seed=2),
        right_knee=signal(hz=1.4, noise=8.0, seed=3),
    )
    info = bw.apply_butterworth_filter(h, FPS, "bike")
    s = info["suggested_cutoffs"]
    assert s["left_knee"] == s["right_knee"]


def test_pairing_does_not_drag_an_unrelated_joint_along():
    h = history(
        left_knee=signal(hz=1.4, noise=0.5, seed=4),
        right_knee=signal(hz=1.4, noise=8.0, seed=5),
        left_elbow=signal(hz=1.4, noise=0.5, seed=6),
    )
    info = bw.apply_butterworth_filter(h, FPS, "bike")
    s = info["suggested_cutoffs"]
    assert s["left_elbow"] != s["left_knee"]


def test_a_failing_diagnostic_never_costs_the_filtering():
    """The suggestion is a nice-to-have; the smoothing is the product."""
    h = history(knee=signal(hz=1.4, noise=2.0, seed=12))
    real = bw.suggest_cutoffs
    bw.suggest_cutoffs = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        info = bw.apply_butterworth_filter(h, FPS, "run")
    finally:
        bw.suggest_cutoffs = real
    assert info["filtered"] == ["knee"]
    assert info["suggested_cutoffs"] == {}


def test_the_history_is_mutated_in_place_and_actually_smoothed():
    raw = signal(hz=1.4, noise=6.0, seed=7)
    h = history(knee=raw)
    values = h["knee"]
    bw.apply_butterworth_filter(h, FPS, "run")
    assert values is h["knee"]                        # same list object
    assert np.std(np.diff(values)) < np.std(np.diff(raw))


def test_nan_gaps_survive_the_new_cutoff_path():
    data = signal(hz=1.4, noise=1.0, seed=8)
    data[100:120] = np.nan                            # a 20-frame dropout
    h = history(knee=data)
    bw.apply_butterworth_filter(h, FPS, "run")
    out = np.array(h["knee"])
    assert np.isnan(out[105:115]).all()               # centre stays missing
    assert np.isfinite(out[:90]).all()


def test_a_channel_too_short_to_filter_at_all_is_skipped_not_invented():
    h = history(knee=[100.0] * (bw.MIN_SAMPLES - 1))
    info = bw.apply_butterworth_filter(h, FPS, "run")
    assert info["filtered"] == [] and info["skipped"] == ["knee"]


def test_an_impossible_frame_rate_is_refused():
    info = bw.apply_butterworth_filter(history(knee=signal()), 0.0, "run")
    assert info["reason"] == "invalid_fps"


# --- what the rest of the app reads ----------------------------------------

def test_the_headline_cutoff_is_still_a_single_number_for_older_readers():
    """The report footer prints one figure. The diagnostic must not turn that
    into None or a list without anyone noticing."""
    h = history(knee=signal(noise=2.0, seed=9), trunk=signal(noise=0.5, seed=10))
    info = bw.apply_butterworth_filter(h, FPS, "run")
    assert isinstance(info["cutoff_hz"], float)


def test_the_meta_is_json_serializable():
    import json

    h = history(knee=signal(noise=2.0, seed=11))
    json.dumps(bw.apply_butterworth_filter(h, FPS, "run"))


@pytest.mark.parametrize("sport", ["run", "bike", "swim"])
def test_every_sport_has_a_band_and_a_fallback_inside_it(sport):
    lo, hi = bw.SPORT_CUTOFF_BOUNDS[sport]
    assert lo < hi
    assert lo <= bw.SPORT_CUTOFFS[sport] <= hi
