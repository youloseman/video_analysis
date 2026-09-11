"""Coaching survives one provider failing -- and says so when it does not.

Two real failures sit behind this file.

1. **The tuning knob is model-specific and undiscoverable.** Measured against
   the live API on 2026-09-11: ``gemini-2.5-flash`` accepts
   ``thinking_budget=0``; ``gemini-3.5-flash-lite`` rejects that exact argument
   with a bare ``400 INVALID_ARGUMENT`` and wants ``thinking_level="minimal"``;
   ``gemini-3.8-flash`` takes the budget or ``thinking_level="low"`` but 400s on
   "minimal". So switching the model by environment variable alone -- which is
   how the model is meant to be switched -- would have made every coaching call
   fail, and the failure is silent: the athlete sees a report with no coach
   block, which is what a clean analysis looks like too.

2. **A fallback at the same vendor is not a fallback.** What takes coaching
   down is an outage, a suspended key, or a model retired out from under us,
   and each of those takes the whole vendor with it. Hence a chain that ends at
   somebody else's API, and a counter for the case where even that fails.
"""
from __future__ import annotations

import dataclasses

import pytest

from app.services.video_analysis import llm_recommendations as llm


@pytest.fixture(autouse=True)
def _clear_failures():
    llm._FAILURES.clear()
    yield
    llm._FAILURES.clear()


def _configure(monkeypatch, **overrides):
    """Point the module at a made-up chain (frozen dataclass -> replace)."""
    fields = {
        "gemini_api_key": "gem-key",
        "gemini_model": "gemini-3.5-flash-lite",
        "openai_api_key": "oa-key",
        "openai_model": "gpt-5-mini",
        "openai_reasoning_effort": "low",
    }
    fields.update(overrides)
    base = dataclasses.replace(llm.settings, **fields)
    monkeypatch.setattr(llm, "settings", base)
    return base


def _fakes(monkeypatch, gemini, openai):
    """Install fake providers; each is called ``(model, system, prompt, max, tuning)``."""
    calls: list[tuple[str, str, dict]] = []

    def wrap(provider, behaviour):
        def _call(model, system, prompt, max_tokens, tuning):
            calls.append((provider, model, tuning))
            if isinstance(behaviour, Exception):
                raise behaviour
            if callable(behaviour):
                return behaviour(tuning)
            return behaviour
        return _call

    monkeypatch.setattr(llm, "_call_gemini", wrap("gemini", gemini))
    monkeypatch.setattr(llm, "_call_openai", wrap("openai", openai))
    return calls


# --- 1. the knob ----------------------------------------------------------

@pytest.mark.parametrize(
    ("model", "expected"),
    [
        # The live-API results quoted in the docstring, one case each.
        ("gemini-3.5-flash-lite", {"thinking_level": "minimal"}),
        ("gemini-3.8-flash", {"thinking_level": "low"}),
        ("gemini-2.5-flash", {"thinking_budget": 0}),
    ],
)
def test_thinking_knob_matches_the_model_generation(model, expected):
    assert llm._gemini_tuning(model) == expected


def test_a_rejected_knob_costs_a_retry_not_the_coaching(monkeypatch):
    """The exact 400 that switching to flash-lite would have caused."""
    _configure(monkeypatch)

    def gemini(tuning):
        if tuning:
            raise RuntimeError("400 INVALID_ARGUMENT")
        return "coached anyway"

    calls = _fakes(monkeypatch, gemini, "openai should not be reached")
    got = llm._complete("sys", "prompt", 100, tag="video")

    assert got == {"text": "coached anyway", "model": "gemini-3.5-flash-lite"}
    # Same model twice -- tuned, then bare -- and the fallback left alone.
    assert [c[0] for c in calls] == ["gemini", "gemini"]
    assert calls[0][2] == {"thinking_level": "minimal"}
    assert calls[1][2] == {}
    assert llm.llm_failures_24h() == 0


def test_no_bare_retry_when_there_was_no_tuning_to_drop(monkeypatch):
    """A byte-identical second attempt buys nothing and costs a call."""
    _configure(monkeypatch, gemini_api_key=None)
    calls = _fakes(monkeypatch, "unused", RuntimeError("down"))

    assert llm._complete("sys", "prompt", 100, tag="video") is None
    # OpenAI carries tuning (reasoning effort), so it DOES retry bare: twice.
    assert [c[0] for c in calls] == ["openai", "openai"]
    assert calls[0][2] == {"reasoning": {"effort": "low"}}
    assert calls[1][2] == {}


# --- 2. the fallback ------------------------------------------------------

def test_the_second_provider_answers_when_the_first_is_down(monkeypatch):
    _configure(monkeypatch)
    calls = _fakes(monkeypatch, RuntimeError("503 upstream"), "fallback coaching")

    got = llm._complete("sys", "prompt", 100, tag="video")

    assert got == {"text": "fallback coaching", "model": "gpt-5-mini"}
    assert [c[0] for c in calls] == ["gemini", "gemini", "openai"]
    assert llm.llm_failures_24h() == 0


def test_an_empty_reply_counts_as_a_failure_and_falls_through(monkeypatch):
    """A 200 with no text is a failed call -- it just does not raise."""
    _configure(monkeypatch)
    _fakes(monkeypatch, "   ", "fallback coaching")

    got = llm._complete("sys", "prompt", 100, tag="photo")
    assert got["model"] == "gpt-5-mini"


def test_a_missing_sdk_skips_that_provider_without_retrying(monkeypatch):
    _configure(monkeypatch)
    calls = _fakes(monkeypatch, ImportError("no google-genai"), "fallback coaching")

    assert llm._complete("sys", "prompt", 100, tag="video")["model"] == "gpt-5-mini"
    # Not retryable and not the model's fault: one shot, then the next provider.
    assert [c[0] for c in calls] == ["gemini", "openai"]


# --- 3. when it really is gone --------------------------------------------

def test_exhausting_the_chain_is_counted(monkeypatch):
    _configure(monkeypatch)
    _fakes(monkeypatch, RuntimeError("down"), RuntimeError("also down"))

    assert llm._complete("sys", "prompt", 100, tag="video") is None
    assert llm.llm_failures_24h() == 1


def test_no_key_configured_is_not_a_failure(monkeypatch):
    """Coaching switched off is a deliberate configuration, not an outage."""
    _configure(monkeypatch, gemini_api_key=None, openai_api_key=None)
    calls = _fakes(monkeypatch, "unused", "unused")

    assert llm._complete("sys", "prompt", 100, tag="video") is None
    assert calls == []
    assert llm.llm_failures_24h() == 0


def test_failures_older_than_a_day_drop_out_of_the_count():
    llm._FAILURES.append(1.0)                      # 1970
    llm._FAILURES.append(__import__("time").time())
    assert llm.llm_failures_24h() == 1


# --- 4. the public callers keep their contract ----------------------------

def test_public_helpers_pass_the_model_name_through(monkeypatch):
    """The report names the model that wrote it; the UI prints that."""
    _configure(monkeypatch)
    _fakes(monkeypatch, "**Overall** — fine.", "unused")

    rec = llm.generate_recommendations("run", 88, "B", [], {}, {})
    assert rec == {"report": "**Overall** — fine.", "model": "gemini-3.5-flash-lite"}

    prog = llm.generate_progress_summary("run", {"score": 80}, {"score": 88})
    assert prog == {"summary": "**Overall** — fine.", "model": "gemini-3.5-flash-lite"}


def test_public_helpers_return_none_when_everything_fails(monkeypatch):
    """The contract the whole app leans on: coaching degrades, never raises."""
    _configure(monkeypatch)
    _fakes(monkeypatch, RuntimeError("down"), RuntimeError("down"))

    assert llm.generate_recommendations("run", 88, "B", [], {}, {}) is None
    assert llm.generate_photo_recommendations("bike", {"score": {}}) is None
    assert llm.generate_progress_summary("run", {}, {}) is None
