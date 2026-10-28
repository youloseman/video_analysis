"""The coach's Russian brief for a Coach Notes order.

Pinned: numbers are read against the served bands and land in a fixed order
with the right direction; the "what to look at" list follows the numbers that
are out; an automatic saddle step bigger than 10 mm is flagged to the coach;
the deadline is the evening before the athlete's day; athlete text is
escaped; and the endpoint serves only Coach Notes orders, to admins.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.api import billing, me
from app.models.order import ORDER_PAID, Order
from app.services import coach_brief as cb

# Wed 28 Oct 2026 14:02 UTC -- 07:02 in Vancouver.
PAID_MS = 1793196120000

BIKE = {
    "sport_type": "bike", "cycling_position": "triathlon", "camera_side": "right",
    "technique_score": 82, "letter_grade": "B", "focus": None,
    "sport_specific_metrics": {
        "knee_at_bdc": 150.8, "knee_at_tdc": 77.01, "hip_at_tdc": 45.79, "trunk_angle_avg": 14.78,
        "shoulder_angle_avg": 94.82, "elbow_angle_avg": 86.61, "forearm_tilt_avg": 23.79, "pelvic_ratio": 4.5,
        "helmet_above_back": 3.7, "ankle_at_bdc": 122.6, "hip_at_bdc": 84.6,
        "sampling": {"duration_s": 11.6},
        "saddle_estimate": {"drive_side": True},
    },
    "reference_bands": {
        "hip_at_tdc": {"lo": 45, "hi": 62}, "knee_at_bdc": {"lo": 138, "hi": 145},
        "knee_at_tdc": {"lo": 60, "hi": 72}, "pelvic_ratio": {"lo": 2.0, "hi": 4.0},
        "elbow_angle_avg": {"lo": 75, "hi": 110}, "trunk_angle_avg": {"lo": 20, "hi": 25},
        "forearm_tilt_avg": {"lo": 5, "hi": 30}, "shoulder_angle_avg": {"lo": 80, "hi": 105},
    },
    "pro_field": {"metrics": {"trunk_angle_avg": {"median": 20.2},
                              "helmet_above_back": {"median": 18.4, "lo": 6.7, "hi": 33.0}}},
    "fit_plan": {"diagnostics": [{"component": "saddle_height", "amount": "≈23 mm (±12)"}]},
}


def _render(result=BIKE, entry=None, clip=b"mp4", media=None):
    return cb.render_brief_html(order_id=14, paid_ms=PAID_MS, result=result,
                                entry=entry or {"kind": "video", "report": "**Overall** — fine <b>"},
                                media=media or {"keyframe": "data:image/jpeg;base64,AAA", "kinogram": None},
                                clip=clip)


def test_rows_come_in_a_fixed_order_with_their_direction():
    rows = cb._rows(BIKE)
    assert [r["key"] for r in rows][:3] == ["knee_at_bdc", "knee_at_tdc", "hip_at_tdc"]
    by = {r["key"]: r for r in rows}
    assert (by["knee_at_bdc"]["tag"], by["knee_at_bdc"]["cls"]) == ("выше нормы", "bad")
    assert by["knee_at_tdc"]["tag"] == "выше нормы"
    assert by["hip_at_tdc"]["tag"] == "на нижней границе"
    assert by["trunk_angle_avg"]["tag"] == "ниже нормы"
    assert by["elbow_angle_avg"]["tag"] == "ок"
    assert by["helmet_above_back"]["band"] == "профи 7–33%"
    assert by["pelvic_ratio"]["value"] == "4,5"


def test_the_look_list_follows_the_numbers_that_are_out():
    items = cb._look_items(BIKE, cb._rows(BIKE))
    assert "Колено внизу" in items[0] and "5–10 мм" in items[0]
    assert not any("Локоть" in i for i in items)
    assert items[-1].startswith("Всё, чего цифры не видят")


def test_a_big_automatic_saddle_step_is_flagged_and_a_small_one_is_not():
    assert "≈23 mm" in cb._saddle_flag(BIKE)
    small = {**BIKE, "fit_plan": {"diagnostics": [{"component": "saddle_height", "amount": "≈5 mm"}]}}
    assert cb._saddle_flag(small) is None


def test_the_coach_is_due_the_evening_before_the_athlete():
    coach, client = cb._deadline(PAID_MS)
    assert client == "субботы 31 октября"
    assert coach == "пятницы 30 октября, вечер"


def test_the_page_is_whole_and_escapes_what_it_did_not_write():
    html = _render(result={**BIKE, "focus": "<script>x</script> seat height"})
    assert html.startswith("<!doctype html>") and html.rstrip().endswith("</html>")
    assert "заказ №14" in html and "flapp-coach-order14-draft" in html
    assert "<script>x</script>" not in html and "&lt;script&gt;x" in html
    assert "<b>Overall</b>" in html and "fine &lt;b&gt;" in html
    assert "data:video/mp4;base64," in html and "Со стороны звёзд" in html
    assert "{{" not in html


def test_a_run_order_renders_with_run_labels():
    run = {"sport_type": "run", "technique_score": 70,
           "sport_specific_metrics": {"cadence_spm": 158.0, "overstride_ratio": 0.31},
           "reference_bands": {"cadence_spm": {"lo": 170, "hi": 190}, "overstride_ratio": {"lo": 0, "hi": 0.2}}}
    html = _render(result=run, clip=None, entry={"kind": "photo"})
    assert "техника бега" in html and "Каденс" in html and "158 шаг/мин" in html
    assert "<video" not in html and "Видео в файл не попало" not in html


def test_a_missing_clip_on_a_video_order_says_so():
    assert "Видео в файл не попало" in _render(clip=None)


def test_the_clip_falls_back_to_the_original_without_ffmpeg(tmp_path, monkeypatch):
    clip = tmp_path / "input.mp4"
    clip.write_bytes(b"x" * 100)
    monkeypatch.setattr(cb.shutil, "which", lambda name: None)
    assert cb.clip_for_brief(clip) == b"x" * 100
    assert cb.clip_for_brief(tmp_path / "gone.mp4") is None


# --------------------------------------------------------------------------
# the endpoint
# --------------------------------------------------------------------------

async def _order(db, user, plan="notes", client_id="h1"):
    await me._upsert(db, user, {"id": client_id, "at": 1, "jobId": "j1", "sport": "bike", "kind": "video"})
    now = billing._now_ms()
    order = Order(user_id=user.id, stripe_session_id=f"cs_{plan}", plan=plan, status=ORDER_PAID,
                  created_at_ms=now, updated_at_ms=now, analysis_client_id=client_id)
    db.add(order)
    await db.commit()
    return order


async def test_the_brief_downloads_as_a_russian_html_file(db, make_user):
    user = await make_user()
    order = await _order(db, user)
    resp = await billing.admin_order_brief(order.id, user, db)
    assert resp.media_type == "text/html"
    assert "attachment" in resp.headers["content-disposition"]
    assert f"заказ №{order.id}".encode() in resp.body


async def test_the_brief_is_only_for_coach_notes(db, make_user):
    user = await make_user()
    order = await _order(db, user, plan="expert")
    with pytest.raises(HTTPException) as exc:
        await billing.admin_order_brief(order.id, user, db)
    assert exc.value.status_code == 400
    with pytest.raises(HTTPException) as exc:
        await billing.admin_order_brief(99999, user, db)
    assert exc.value.status_code == 404
