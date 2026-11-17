"""scripts/analyze_local.py measures with production's detectors by default.

It ran MediaPipe alone while production refines bike and run with RTMPose, so
a local result was not the one a user gets: one time-trial clip read 61/C
"lower the saddle 30 mm" locally and 89/B "no change" on production settings
(2026-11-17), and a day of analysis started from the wrong number.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from app.core import config

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_local.py"


@pytest.fixture
def cli(monkeypatch):
    saved = config.settings
    spec = importlib.util.spec_from_file_location("analyze_local_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    yield mod
    config.settings = saved


def test_the_default_is_what_production_runs(cli, monkeypatch):
    monkeypatch.delenv("VA_RTMPOSE_SPORTS", raising=False)
    assert cli.use_detector("env") == ("bike", "run")
    assert config.settings.rtmpose_sports == ("bike", "run")


def test_mediapipe_alone_must_be_asked_for(cli, monkeypatch):
    monkeypatch.delenv("VA_RTMPOSE_SPORTS", raising=False)
    assert cli.use_detector("mediapipe") == ()
    assert config.settings.rtmpose_sports == ()
    assert cli.use_detector("prod") == ("bike", "run")


def test_an_explicit_flag_beats_the_environment(cli, monkeypatch):
    monkeypatch.setenv("VA_RTMPOSE_SPORTS", "")
    assert cli.use_detector("prod") == ("bike", "run")


def test_the_result_names_its_detector(cli):
    prod = {"sport_specific_metrics": {"pose_detector": "mediapipe_tasks+rtmpose_m"}}
    alone = {"sport_specific_metrics": {"pose_detector": "mediapipe_tasks"}}
    assert cli.detector_note(prod, "bike", ("bike", "run")) == \
        "Detector: mediapipe_tasks+rtmpose_m (as production)."
    asked_but_missing = cli.detector_note(alone, "bike", ("bike", "run"))
    assert asked_but_missing.startswith("WARNING") and "did not run" in asked_but_missing
    not_asked = cli.detector_note(alone, "bike", ())
    assert not_asked.startswith("WARNING") and "NOT what a user gets" in not_asked
    assert cli.detector_note({}, "run", ()).startswith("WARNING: detector unknown")
