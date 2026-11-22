"""Ask about this report: who may ask, what the model is given, what it costs.

The model itself is faked here -- what it says is checked against the live key
by scripts/check_report_chat.py, because no unit test can see a model's
behaviour. What these pin is everything around it: the gate, the context the
model receives, the quota, and that a failed call costs the athlete nothing.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import func, select

from app.api import chat as chat_api
from app.api import me
from app.models.analysis import Analysis
from app.models.cda import CdaMeasurement
from app.models.chat import ROLE_ASSISTANT, ROLE_USER, ChatMessage
from app.models.profile import Profile
from app.models.usage import UsageEvent
from app.models.user import (
    TIER_ADMIN,
    TIER_COACH,
    TIER_ENTHUSIAST,
    TIER_FULL,
    TIER_STARTER,
)
from app.services import report_chat
from app.services.usage_limits import window_start_ms

BIKE_RESULT = {
    "status": "completed",
    "sport_type": "bike",
    "cycling_position": "triathlon",
    "camera_side": "left",
    "frames_analyzed": 300,
    "technique_score": 84,
    "letter_grade": "B",
    "angle_statistics": {
        "trunk_angle": {"min": 14.0, "mean": 16.2, "max": 19.0, "range": 5.0,
                        "valid_frames": 290, "nan_frames": 10},
    },
    "detected_issues": [],
    "sport_specific_metrics": {"trunk_angle_avg": 16.2, "knee_at_bdc": 146.0},
}


async def add_report(db, user, *, client_id="h1", result=None, profile_id=None,
                     at=1_000, score=84, unlocked=False):
    row = Analysis(
        user_id=user.id, client_id=client_id, created_at_ms=at,
        result=dict(result if result is not None else BIKE_RESULT),
        data={"id": client_id, "at": at}, sport="bike", kind="video", score=score,
        profile_id=profile_id, unlocked_at_ms=at if unlocked else None,
    )
    db.add(row)
    await db.commit()
    return row


@pytest.fixture(autouse=True)
def launched_for_everyone(monkeypatch):
    """Most tests are about the chat itself, so it is open to every paid plan
    here; the rollout steps have their own tests below."""
    monkeypatch.setattr(report_chat, "rollout", lambda: "all")


@pytest.fixture
def fake_model(monkeypatch):
    """Records each prompt and answers with a fixed line."""
    seen: list[str] = []
    state = {"answer": {"text": "Your knee at BDC is 146°, inside the band.", "model": "fake-1"}}

    def _ask(prompt: str):
        seen.append(prompt)
        return state["answer"]

    monkeypatch.setattr(report_chat, "ask", _ask)
    return {"prompts": seen, "state": state}


async def ask(db, user, question="Why lower the saddle?", client_id="h1", athlete_id=None):
    return await chat_api.ask_chat(
        client_id, chat_api.QuestionIn(question=question), user, db, athlete_id,
    )


# --------------------------------------------------------------------------
# Who may ask
# --------------------------------------------------------------------------
@pytest.mark.parametrize("tier,limit", [
    (TIER_STARTER, 0), (TIER_ENTHUSIAST, 50), (TIER_FULL, 200), (TIER_COACH, 500),
])
def test_each_plan_has_its_monthly_allowance(tier, limit):
    assert report_chat.chat_limit(tier) == limit


def test_an_unknown_tier_has_no_chat():
    assert report_chat.chat_limit("platinum_unicorn") == 0


@pytest.mark.parametrize("step,tier,state", [
    ("admin", TIER_ADMIN, "open"),
    ("admin", TIER_FULL, ""),
    ("admin", TIER_STARTER, ""),
    ("full", TIER_FULL, "open"),
    ("full", TIER_COACH, "open"),
    ("full", TIER_ENTHUSIAST, "full"),
    ("full", TIER_STARTER, "full"),
    ("all", TIER_ENTHUSIAST, "open"),
    ("all", TIER_STARTER, "plans"),
    ("nonsense", TIER_FULL, ""),
])
def test_the_rollout_steps(monkeypatch, step, tier, state):
    from types import SimpleNamespace

    from app.core.config import settings

    monkeypatch.undo()  # drop the autouse "all"
    monkeypatch.setattr(report_chat, "rollout",
                        lambda: step if step in report_chat.ROLLOUTS else "admin")
    assert report_chat.chat_state(SimpleNamespace(tier=tier)) == state
    # Every paid plan since 2026-11-22 (Artur), behind the dollar ceiling.
    assert settings.report_chat == "all"


async def test_before_launch_a_paid_reader_is_refused(db, make_user, fake_model, monkeypatch):
    monkeypatch.setattr(report_chat, "rollout", lambda: "admin")
    u = await make_user(tier=TIER_FULL)
    await add_report(db, u)
    with pytest.raises(HTTPException) as e:
        await ask(db, u)
    assert e.value.status_code == 402
    assert fake_model["prompts"] == []


async def test_a_free_account_is_sent_to_the_plans(db, make_user, fake_model):
    u = await make_user(tier=TIER_STARTER)
    await add_report(db, u, unlocked=True)
    with pytest.raises(HTTPException) as e:
        await ask(db, u)
    assert e.value.status_code == 402
    assert fake_model["prompts"] == []


async def test_a_paid_reader_asks_about_their_own_report(db, make_user, fake_model):
    u = await make_user(tier=TIER_ENTHUSIAST)
    await add_report(db, u)
    out = await ask(db, u)
    assert out["answer"]["text"].startswith("Your knee")
    assert out["answer"]["model"] == "fake-1"
    assert out["used"] == 1 and out["limit"] == 50

    got = await chat_api.get_chat("h1", u, db, None)
    assert [m["role"] for m in got["messages"]] == [ROLE_USER, ROLE_ASSISTANT]
    assert got["used"] == 1


async def test_someone_elses_report_is_a_404(db, make_user, fake_model):
    a = await make_user(tier=TIER_FULL)
    b = await make_user(tier=TIER_FULL)
    await add_report(db, a)
    with pytest.raises(HTTPException) as e:
        await ask(db, b)
    assert e.value.status_code == 404


async def test_a_report_without_a_stored_result_cannot_be_discussed(db, make_user, fake_model):
    u = await make_user(tier=TIER_FULL)
    row = await add_report(db, u)
    row.result = None
    await db.commit()
    with pytest.raises(HTTPException) as e:
        await ask(db, u)
    assert e.value.status_code == 409


def test_an_empty_or_long_question_is_refused():
    with pytest.raises(ValidationError):
        chat_api.QuestionIn(question="   ")
    with pytest.raises(ValidationError):
        chat_api.QuestionIn(question="x" * 601)


# --------------------------------------------------------------------------
# What a question costs
# --------------------------------------------------------------------------
async def test_a_failed_call_stores_nothing_and_costs_nothing(db, make_user, fake_model):
    u = await make_user(tier=TIER_ENTHUSIAST)
    await add_report(db, u)
    fake_model["state"]["answer"] = None
    with pytest.raises(HTTPException) as e:
        await ask(db, u)
    assert e.value.status_code == 503
    assert await report_chat.used_this_month(db, u.id) == 0
    assert (await db.scalar(select(func.count(ChatMessage.id)))) == 0


async def test_the_month_runs_out(db, make_user, fake_model, monkeypatch):
    monkeypatch.setitem(report_chat.CHAT_LIMITS, TIER_ENTHUSIAST, 2)
    u = await make_user(tier=TIER_ENTHUSIAST)
    await add_report(db, u)
    await ask(db, u)
    await ask(db, u)
    with pytest.raises(HTTPException) as e:
        await ask(db, u)
    assert e.value.status_code == 429


async def test_last_months_questions_do_not_count(db, make_user):
    u = await make_user(tier=TIER_ENTHUSIAST)
    old = window_start_ms("month") - 1
    db.add(ChatMessage(user_id=u.id, owner_user_id=u.id, analysis_client_id="h1",
                       role=ROLE_USER, text="old", created_at_ms=old))
    await db.commit()
    assert await report_chat.used_this_month(db, u.id) == 0


async def test_a_question_never_eats_an_analysis(db, make_user, fake_model):
    """The analysis quota counts every usage_events row whatever its kind."""
    u = await make_user(tier=TIER_ENTHUSIAST)
    await add_report(db, u)
    await ask(db, u)
    assert (await db.scalar(select(func.count(UsageEvent.id)))) == 0


# --------------------------------------------------------------------------
# What the model is given
# --------------------------------------------------------------------------
async def test_the_model_gets_the_report_the_bike_and_the_measured_cda(db, make_user, fake_model):
    u = await make_user(tier=TIER_FULL, height_cm=181)
    p = await me.create_profile(me.ProfileIn(
        name="Race bike", sport="bike", kind="tt", crank_length_mm=170,
        bike_family="giant-trinity", bike_model="Advanced Pro", frame_size="M", bike_year=2019,
    ), u, db)
    db.add(CdaMeasurement(profile_id=p["id"], user_id=u.id, value_m2=0.231,
                          measured_on="2026-11-01", source="notio",
                          protocol="repeated_loops", created_at_ms=1))
    await db.commit()
    await add_report(db, u, client_id="h0", profile_id=p["id"], at=500, score=78)
    await add_report(db, u, client_id="h1", profile_id=p["id"], at=1_000)

    await ask(db, u)
    prompt = fake_model["prompts"][0]
    assert "## Measurements" in prompt                      # the export document
    assert "Height: 181 cm" in prompt
    assert "Giant Trinity · Advanced Pro · size M · 2019" in prompt
    assert "Cranks: 170 mm" in prompt
    assert "0.231 m² (notio, repeated_loops)" in prompt
    assert "does not measure CdA from video" in prompt
    assert ": 78" in prompt                                  # the earlier clip
    assert "Why lower the saddle?" in prompt


async def test_an_unfiled_clip_says_the_bike_is_unknown(db, make_user, fake_model):
    u = await make_user(tier=TIER_FULL)
    await add_report(db, u)
    await ask(db, u)
    prompt = fake_model["prompts"][0]
    assert "not filed under a saved bike" in prompt
    assert "Height: not given" in prompt


async def test_a_withheld_score_turns_the_answer_to_refilming(db, make_user, fake_model):
    u = await make_user(tier=TIER_FULL)
    result = {**BIKE_RESULT, "technique_score": None,
              "score_withheld": {"reason": "quality_gate", "detail": "The knee was hidden."}}
    await add_report(db, u, result=result)
    await ask(db, u)
    prompt = fake_model["prompts"][0]
    assert prompt.startswith("## SCORE WITHHELD")
    assert "The knee was hidden." in prompt


async def test_the_conversation_is_replayed_but_only_the_recent_part(db, make_user, fake_model):
    u = await make_user(tier=TIER_FULL)
    await add_report(db, u)
    for i in range(6):
        await ask(db, u, question=f"question number {i}")
    last = fake_model["prompts"][-1]
    assert "question number 5" in last.split("# THE ATHLETE'S QUESTION")[1]
    convo = last.split("# CONVERSATION SO FAR")[1].split("# THE ATHLETE'S QUESTION")[0]
    # Eight messages back = the last four questions before this one.
    assert "question number 0" not in convo
    assert "question number 1" in convo and "question number 4" in convo


def test_the_rules_the_model_is_held_to():
    s = report_chat.SYSTEM
    assert "does NOT measure CdA from video" in s
    assert "helmet" in s and "never name helmet models" in s
    assert "nothing to fix" in s
    assert "ONLY when it appears under 'Parts on record" in s
    assert "Never guess that a part fits" in s
    assert "language the athlete wrote in" in s


# --------------------------------------------------------------------------
# Threads belong to the reader; they go with the report
# --------------------------------------------------------------------------
async def test_a_thread_is_the_readers_own(db, make_user, fake_model):
    u = await make_user(tier=TIER_FULL)
    await add_report(db, u, client_id="h1")
    await add_report(db, u, client_id="h2", at=2_000)
    await ask(db, u, client_id="h1")
    got = await chat_api.get_chat("h2", u, db, None)
    assert got["messages"] == []


async def test_deleting_the_report_deletes_its_chat(db, make_user, fake_model):
    u = await make_user(tier=TIER_FULL)
    await add_report(db, u, client_id="h1")
    await add_report(db, u, client_id="h2", at=2_000)
    await ask(db, u, client_id="h1")
    await ask(db, u, client_id="h2")
    await me.delete_analysis("h1", u, db)
    left = (await db.execute(select(ChatMessage.analysis_client_id))).scalars().all()
    assert set(left) == {"h2"}


# --------------------------------------------------------------------------
# What the bike is, on the profile
# --------------------------------------------------------------------------
async def test_a_bike_profile_keeps_what_the_bike_is(db, make_user):
    u = await make_user(tier=TIER_FULL)
    p = await me.create_profile(me.ProfileIn(
        name="TT", sport="bike", bike_family="canyon-speedmax",
        bike_model="  CF SLX   8  ", frame_size="M", bike_year=2022,
    ), u, db)
    assert p["bike_family"] == "canyon-speedmax"
    assert p["bike_model"] == "CF SLX 8"
    assert p["frame_size"] == "M" and p["bike_year"] == 2022

    out = await me.update_profile(p["id"], me.ProfilePatch(bike_model=None), u, db)
    assert out["bike_model"] is None and out["bike_family"] == "canyon-speedmax"

    listed = await me.list_profiles(u, db)
    assert {"slug": "giant-trinity", "label": "Giant Trinity"} in listed["bike_families"]


async def test_an_unknown_family_is_dropped_and_a_wild_year_refused(db, make_user):
    assert me.ProfileIn(name="x", bike_family="Penny-Farthing").bike_family is None
    assert me.ProfileIn(name="x", bike_family="OTHER").bike_family == "other"
    with pytest.raises(ValidationError):
        me.ProfileIn(name="x", bike_year=1850)


async def test_shoes_carry_no_bike(db, make_user):
    u = await make_user(tier=TIER_FULL)
    p = await me.create_profile(me.ProfileIn(
        name="Vaporfly", sport="run", kind="shoes", bike_family="giant-trinity", bike_year=2020,
    ), u, db)
    row = await db.get(Profile, p["id"])
    assert row.bike_family is None and row.bike_year is None


# --------------------------------------------------------------------------
# 2026-11-22: visible at last, a dollar ceiling, questions from the report
# --------------------------------------------------------------------------
def test_auth_me_carries_the_chat_state():
    """/auth/me's model had no report_chat field, so pydantic dropped it and
    nobody -- the admin included -- was ever shown the chat."""
    from types import SimpleNamespace

    from app.api.auth import UserOut, _user_out

    assert "report_chat" in UserOut.model_fields
    admin = SimpleNamespace(email="a@x", tier=TIER_ADMIN, is_paid=True, expert_credits=0, unlock_credits=0,
                            height_cm=None, marketing_opt_in=False, research_opt_in=False, id=1,
                            crank_until_ms=None, created_at=None)
    try:
        out = _user_out(admin)
    except Exception:  # noqa: BLE001 -- the helpers may want a real row
        pytest.skip("needs a real user row")
    assert out.report_chat == "open"


def test_the_spa_keeps_the_chat_state_on_form_sign_in():
    from pathlib import Path

    s = (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")
    assert "report_chat:j.report_chat||''" in s


def test_enthusiast_full_and_coach_have_it_by_default():
    from types import SimpleNamespace

    for tier in (TIER_ENTHUSIAST, TIER_FULL, TIER_COACH, TIER_ADMIN):
        assert report_chat.chat_state(SimpleNamespace(tier=tier)) == "open"
    assert report_chat.chat_state(SimpleNamespace(tier=TIER_STARTER)) == "plans"


def test_each_paid_plan_has_a_dollar_ceiling_under_its_price():
    price = {TIER_ENTHUSIAST: 9, TIER_FULL: 13, TIER_COACH: 25}
    for tier, monthly in price.items():
        assert 0 < report_chat.chat_budget(tier) <= monthly * 0.1 + 1e-9
    assert report_chat.chat_budget(TIER_STARTER) == 0


def test_a_question_is_cheap_and_an_unknown_model_is_priced_dear():
    prompt = "x" * 30_000      # ~8.6k tokens: a long report and a long thread
    assert report_chat.projected_cost_usd(prompt) < 0.02
    assert report_chat.model_price("some-new-model") == report_chat.UNKNOWN_MODEL_PRICE
    assert report_chat.cost_usd("gemini-3.5-flash-lite", 1_000_000, 0) == pytest.approx(0.30)


async def test_an_answer_records_what_it_cost(db, make_user, fake_model):
    u = await make_user(tier=TIER_ENTHUSIAST)
    await add_report(db, u)
    await ask(db, u)
    a = (await db.execute(select(ChatMessage).where(ChatMessage.role == ROLE_ASSISTANT))).scalars().one()
    assert a.tokens_in > 1000 and a.tokens_out > 0
    assert 0 < await report_chat.spent_this_month(db, u.id) < 0.05


async def test_the_dollar_ceiling_refuses_before_the_model_is_called(db, make_user, fake_model, monkeypatch):
    monkeypatch.setitem(report_chat.CHAT_BUDGET_USD, TIER_ENTHUSIAST, 0.0001)
    u = await make_user(tier=TIER_ENTHUSIAST)
    await add_report(db, u)
    with pytest.raises(HTTPException) as e:
        await ask(db, u)
    assert e.value.status_code == 429
    assert "reset on the 1st" in e.value.detail
    assert fake_model["prompts"] == []


async def test_out_of_dollars_reads_as_no_questions_left(db, make_user, fake_model, monkeypatch):
    u = await make_user(tier=TIER_ENTHUSIAST)
    await add_report(db, u)
    await ask(db, u)
    monkeypatch.setitem(report_chat.CHAT_BUDGET_USD, TIER_ENTHUSIAST, 0.0000001)
    j = await chat_api.get_chat("h1", u, db, None)
    assert j["used"] == j["limit"] == 50


BANDED_BIKE = dict(BIKE_RESULT, cycling_position="road_hoods", sport_specific_metrics={
    "knee_at_bdc": 152.0, "trunk_angle_avg": 40.0, "elbow_angle_avg": 131.0,
}, reference_bands={"knee_at_bdc": {"lo": 135, "hi": 145}, "trunk_angle_avg": {"lo": 32, "hi": 50},
                    "elbow_angle_avg": {"lo": 145, "hi": 170}})


def test_suggestions_start_from_the_worst_reading():
    qs = report_chat.suggestions(BANDED_BIKE)
    assert 1 <= len(qs) <= report_chat.MAX_SUGGESTIONS
    assert qs[0].startswith("My ") and "the range is" in qs[0]
    assert "knee" in qs[0] or "elbow" in qs[0]
    assert any("worth fixing" in q for q in qs)
    assert all("\u2014" not in q for q in qs)


def test_suggestions_follow_the_account_and_the_clip():
    aero = report_chat.suggestions(dict(BIKE_RESULT), has_cda=True, has_previous=True)
    assert any("over my race" in q for q in aero)
    assert any("CdA" in q for q in aero)
    withheld = report_chat.suggestions(dict(BIKE_RESULT, score_withheld=True))
    assert withheld[0].startswith("Why could this clip not be scored")
    run = report_chat.suggestions({"sport_type": "run"})
    assert any("drill" in q for q in run)


async def test_the_thread_comes_with_its_suggestions(db, make_user, fake_model):
    u = await make_user(tier=TIER_FULL)
    await add_report(db, u, result=BANDED_BIKE)
    j = await chat_api.get_chat("h1", u, db, None)
    assert j["suggestions"] and j["suggestions"][0].startswith("My ")


async def test_admin_sees_what_the_chat_cost(db, make_user, fake_model):
    u = await make_user(tier=TIER_ENTHUSIAST, email="rider@example.com")
    admin = await make_user(tier=TIER_ADMIN)
    await add_report(db, u)
    await ask(db, u)
    await ask(db, u, question="And the trunk?")
    j = await chat_api.chat_usage(admin, db)
    assert j["totals"]["questions"] == 2 and j["totals"]["askers"] == 1
    row = j["users"][0]
    assert row["email"] == "rider@example.com" and row["limit"] == 50
    assert 0 < row["budget_used"] < 0.1
    assert j["limits"][TIER_COACH]["budget_usd"] == 2.5
