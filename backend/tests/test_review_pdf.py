"""The Expert Review PDF page: the coach's text with the numbers under it.

What would hurt if it broke: a measured angle missing from the table, a
helmet the analyzer could not read shown as a number, or the coach's text
reaching the page as live HTML.
"""

from __future__ import annotations

from app.services.review_pdf import metric_rows, render_review_html

RESULT = {
    "sport_type": "bike",
    "cycling_position": "tt_aero",
    "technique_score": 67,
    "letter_grade": "C",
    "camera_side": "right",
    "sport_specific_metrics": {
        "knee_at_bdc": 155.4, "knee_at_tdc": 70.7, "trunk_angle_avg": 19.2,
        "forearm_tilt_avg": 12.9, "helmet_above_back": 13.3, "helmet_check": True,
    },
    "reference_bands": {
        "knee_at_bdc": {"lo": 138, "hi": 145}, "knee_at_tdc": {"lo": 60, "hi": 72},
        "trunk_angle_avg": {"lo": 15, "hi": 25}, "forearm_tilt_avg": {"lo": 5, "hi": 30},
    },
    "pro_field": {"metrics": {"trunk_angle_avg": {"median": 20, "q1": 17, "q3": 23}}},
}

REPORT = {
    "verdict": "Your saddle is **too high**.",
    "seen": "**The leg runs out at the bottom.** The toes drop.",
    "priorities": [{"title": "Lower the saddle 10 mm", "why": "Knee 155°", "change": "-10 mm", "check": "145–150°"}],
    "fit": [{"part": "Saddle height", "now": "155°", "suggest": "-10 mm", "why": "x", "risk": "y"}],
    "plan": "**Week 1** — saddle down.\n\n**Week 2** — film again.",
    "retest": "Film the left side.",
    "reviewer": "Bogdan <script>",
}


def test_rows_grade_against_the_band_and_skip_an_unreadable_helmet():
    rows = {r["field"]: r for r in metric_rows(RESULT)}
    assert rows["knee_at_bdc"]["status"] == "bad"
    assert rows["knee_at_tdc"]["status"] == "good"
    assert rows["trunk_angle_avg"]["pro"]["median"] == 20
    # helmet_check set: the reading is not trusted, so no number is shown
    assert rows["helmet_above_back"]["value"] is None
    assert rows["helmet_above_back"]["status"] == "na"
    # not measured at all -> no row, rather than an empty one
    assert "elbow_angle_avg" not in rows


def test_page_carries_the_report_figures_and_escapes_text():
    page = render_review_html(REPORT, RESULT, {
        "athlete": "A", "date": "14 Oct 2026",
        "kinogram": "data:image/jpeg;base64,AAA", "keyframe": "data:image/jpeg;base64,BBB",
    })
    assert "<b>too high</b>" in page
    # the coach first, the numbers after
    assert page.index("What I see") < page.index("Fix these") < page.index("The numbers behind it")
    assert "data:image/jpeg;base64,AAA" in page and "data:image/jpeg;base64,BBB" in page
    assert "Lower the saddle 10 mm" in page
    assert page.count("class=\"week\"") == 2 and "Saddle down." in page
    assert "Bogdan &lt;script&gt;" in page and "Bogdan <script>" not in page
    assert "Not measured" in page


def test_page_renders_without_figures_or_fit():
    page = render_review_html({"verdict": "Fine."}, RESULT, {})
    assert "Kinogram" not in page and "class='fit'" not in page


def test_capture_warnings_close_the_numbers_page():
    page = render_review_html({"verdict": "v"}, RESULT | {"warnings": ["Filmed from the drive side"]}, {})
    assert page.index("The numbers behind it") < page.index("Filmed from the drive side")
