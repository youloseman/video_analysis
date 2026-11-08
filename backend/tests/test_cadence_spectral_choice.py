"""When the knee count and the spectrum agree, the spectrum is reported.

Eye-counted on five clips (2026-11-07) the spectrum missed by <= 1.5 spm on
both detectors; the knee count missed by up to 20 (MediaPipe) and moved
181.5 -> 189.5 between two machines on one clip (RTMPose).
"""

from app.services.video_analysis.biomechanics.running_analyzer import RunningAnalyzer


def _analyzer(knee_spm: float, spectral_spm: float, monkeypatch) -> RunningAnalyzer:
    an = RunningAnalyzer(fps=60.0)
    monkeypatch.setattr(an, "_detect_slow_motion", lambda: None)
    monkeypatch.setattr(an, "_detect_steps_from_knee_angles", lambda: list(range(8)))
    monkeypatch.setattr(an, "_compute_cadence_from_strides", lambda steps: knee_spm)
    monkeypatch.setattr(an, "_cadence_spectral", lambda: spectral_spm)
    return an


def test_agreeing_methods_report_the_spectrum(monkeypatch):
    assert _analyzer(189.5, 180.9, monkeypatch)._compute_cadence() == 180.9


def test_disagreeing_methods_still_report_the_spectrum(monkeypatch):
    assert _analyzer(124.0, 165.0, monkeypatch)._compute_cadence() == 165.0


def test_knee_count_alone_when_the_spectrum_is_unreadable(monkeypatch):
    assert _analyzer(172.0, 0.0, monkeypatch)._compute_cadence() == 172.0
