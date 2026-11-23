"""The "your analysis is ready" mail, and the ways it must not misfire.

The whole justification for this email is that it only reaches somebody who
left before their result appeared. An analysis takes under a minute, so a mail
to someone still watching the spinner is pure noise -- and the second copy of
that noise is what gets a sender marked as spam.
"""
from __future__ import annotations

import app.main as main
from app.main import _schedule_ready_mail, unsubscribe_token
from app.services.notify import analysis_ready_email


def _completed_job(**over):
    job = {"status": "completed", "owner_user_id": 7,
           "result": {"sport_type": "run", "technique_score": 82, "letter_grade": "B"}}
    job.update(over)
    return job


def _run_timer_now(monkeypatch, sent: list):
    """Run the scheduled check inline instead of 45 seconds later."""
    class Immediate:
        def __init__(self, _delay, fn):
            self.fn = fn
            self.daemon = True

        def start(self):
            self.fn()

    monkeypatch.setattr(main.threading, "Timer", Immediate)
    monkeypatch.setattr(main, "_send_ready_mail",
                        lambda job_id, job: sent.append(job_id))
    # _schedule_ready_mail hands the coroutine to the server loop; with the
    # stub above it has already run, so there is nothing to hand over.
    monkeypatch.setattr(main, "_submit_to_server_loop", lambda coro, what, **ctx: True)


def test_no_mail_when_the_athlete_is_watching(monkeypatch):
    sent: list = []
    _run_timer_now(monkeypatch, sent)
    main.JOBS["j-seen"] = _completed_job(seen=True)
    _schedule_ready_mail("j-seen")
    assert sent == [], "somebody who polled for the result must not be emailed"


def test_mail_when_nobody_came_for_it(monkeypatch):
    sent: list = []
    _run_timer_now(monkeypatch, sent)
    main.JOBS["j-left"] = _completed_job()
    _schedule_ready_mail("j-left")
    assert sent == ["j-left"]


def test_never_twice_for_the_same_job(monkeypatch):
    sent: list = []
    _run_timer_now(monkeypatch, sent)
    main.JOBS["j-once"] = _completed_job()
    _schedule_ready_mail("j-once")
    _schedule_ready_mail("j-once")
    assert sent == ["j-once"]


def test_a_vanished_job_is_not_an_error(monkeypatch):
    sent: list = []
    _run_timer_now(monkeypatch, sent)
    _schedule_ready_mail("j-gone")      # never registered / already swept
    assert sent == []


def test_the_unsubscribe_link_is_signed_per_account():
    a, b = unsubscribe_token(7), unsubscribe_token(8)
    assert a != b
    assert unsubscribe_token(7) == a          # stable across calls (and restarts)
    assert len(a) == 32


def test_unsubscribing_answers_the_same_either_way():
    """A wrong token changes nothing -- and says nothing either way."""
    from fastapi.testclient import TestClient
    with TestClient(main.app) as client:
        good = client.get("/unsubscribe?u=1&t=" + unsubscribe_token(1))
        bad = client.get("/unsubscribe?u=1&t=deadbeef")
    assert good.status_code == 200 and bad.status_code == 200
    # Same page for both: otherwise the endpoint tells a stranger whether an
    # address has an account here.
    assert good.text == bad.text


def test_the_mail_says_what_it_is_and_how_to_stop():
    subject, text, html = analysis_ready_email(
        "bike", 88, "B", "https://x/app#job=abc", "https://x/unsubscribe?u=1&t=z",
    )
    assert "ride" in subject and "88/100" in subject     # rider, not runner
    assert "https://x/app#job=abc" in text
    assert "https://x/unsubscribe?u=1&t=z" in text
    # The HTML copy carries both too, with the ampersand escaped as it must be
    # inside an attribute -- an unescaped one is how a link ends up truncated
    # at the query string in some clients.
    assert "https://x/app#job=abc" in html
    assert "https://x/unsubscribe?u=1&amp;t=z" in html


def test_a_run_is_called_a_run():
    subject, _t, _h = analysis_ready_email("run", None, None, "u", "un")
    assert "run analysis" in subject
    assert "ready to read" in subject       # no score, no fake number


def test_the_mail_runs_on_the_server_loop_not_a_new_one(monkeypatch):
    """The timer fires on its own thread; the coroutine must still run on the
    loop that owns the database pool. asyncio.run there opened a second loop
    and took pooled asyncpg connections across loops."""
    import asyncio
    import threading

    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    ran_on: list = []
    finished = threading.Event()

    async def probe():
        ran_on.append(asyncio.get_running_loop())
        finished.set()

    try:
        monkeypatch.setattr(main, "SERVER_LOOP", loop)
        worker = threading.Thread(target=lambda: main._submit_to_server_loop(probe(), "TEST"))
        worker.start()
        worker.join()
        assert finished.wait(5)
        assert ran_on == [loop]
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(5)
        loop.close()


def test_without_a_server_loop_the_mail_is_dropped_not_run_elsewhere(monkeypatch):
    monkeypatch.setattr(main, "SERVER_LOOP", None)
    ran: list = []

    async def probe():
        ran.append(1)

    assert main._submit_to_server_loop(probe(), "TEST") is False
    assert ran == []


def test_ready_mail_no_longer_opens_its_own_loop():
    import inspect

    assert "asyncio.run(" not in inspect.getsource(main._schedule_ready_mail)
