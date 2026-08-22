"""The built-in camera, checked against the shipped SPA.

String checks, same reasoning as ``test_print_report.py``: the things that
break here are a guard, an attribute, or a constant, and no Python test would
otherwise ever look at them. A live recording can only be exercised on a real
device -- these pin down everything short of that.

The failure modes worth pinning are the quiet ones. A record button inside the
drop label would ALSO open the file picker. A camera stream left running after
the modal closes keeps the phone's camera light on. A recording that bypasses
setFile() skips the size limit and the preflight checks that every picked file
goes through. None of these would error -- they would just ship wrong.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.main import ALLOWED_SUFFIXES

SPA = Path(__file__).resolve().parents[1] / "app" / "static" / "index.html"


@pytest.fixture(scope="module")
def html() -> str:
    return SPA.read_text(encoding="utf-8")


def _between(html: str, start: str, end: str) -> str:
    i = html.index(start)
    return html[i:html.index(end, i)]


# --------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------

def test_the_record_button_is_outside_the_drop_label(html: str):
    """The dropzone is a <label for="file">. A button INSIDE it would open the
    file picker on every click as well as the camera -- two dialogs for one
    tap. The button must live after the label closes."""
    drop = _between(html, '<label class="drop" id="drop"', "</label>")
    assert 'id="camOpen"' not in drop
    assert 'id="camOpen"' in html


def test_the_modal_exists_with_guides_countdown_and_clock(html: str):
    modal = _between(html, 'id="camModal"', "<div class=\"hero\">")
    for needle in ('id="camView"', "cam-guides", 'id="camCount"',
                   'id="camRec"', 'id="camStart"', 'id="camStop"'):
        assert needle in modal, f"camera modal is missing {needle}"


def test_the_preview_video_is_muted_and_inline(html: str):
    """Without playsinline, iOS Safari fullscreens the preview and the framing
    guides -- the entire point of the feature -- go with it."""
    m = re.search(r'<video id="camView"[^>]*>', html)
    assert m and "playsinline" in m.group(0) and "muted" in m.group(0)


# --------------------------------------------------------------------------
# behaviour that must not regress
# --------------------------------------------------------------------------

def test_feature_detection_guards_every_api_it_uses(html: str):
    block = _between(html, "function camSupported(", "}")
    for api in ("mediaDevices", "getUserMedia", "MediaRecorder", "isSecureContext"):
        assert api in block, f"camSupported() does not check {api}"


def test_no_microphone_is_requested(html: str):
    """The analysis never uses sound. Asking for audio is one more permission
    prompt and one more reason to say no -- and a recording with a mic track
    would be the only place the product ever captures a voice."""
    block = _between(html, "async function openCam(", "function closeCam(")
    assert "audio:false" in block.replace(" ", "")
    assert "audio:true" not in block.replace(" ", "")


def test_closing_stops_the_camera_tracks(html: str):
    """A stream that outlives the modal keeps the camera light on -- the
    universal signal for 'this site is watching you'."""
    block = _between(html, "function closeCam(", "function camStartFlow(")
    assert "getTracks().forEach(t=>t.stop())" in block
    assert "srcObject=null" in block


def test_a_stream_granted_after_close_is_also_stopped(html: str):
    """getUserMedia resolves AFTER the permission prompt; the user can close
    the modal while the prompt is still up. The late stream must be stopped,
    not attached to a hidden video."""
    block = _between(html, "async function openCam(", "function closeCam(")
    assert "closed while we waited" in block


def test_the_recording_goes_through_setfile(html: str):
    """setFile() is where the size limit, the preview and the preflight checks
    live. A recording that skips it is a file the rest of the app never got to
    inspect."""
    block = _between(html, "function camKeep(", "$('#camOpen')")
    assert "setFile(file)" in block


def test_countdown_and_autostop_constants(html: str):
    """15 s is the product's own clip guidance; the countdown exists because
    the athlete has to get from the phone into the frame."""
    assert "CAM_MAX_S=15" in html
    assert "CAM_COUNTDOWN_S=5" in html
    block = _between(html, "function camRecord(", "function camKeep(")
    assert "CAM_MAX_S*1000" in block, "the auto-stop no longer uses the constant"


def test_recorded_extensions_are_on_the_server_allowlist(html: str):
    """The client names the file .mp4 or .webm from the recorder's mime. If
    either ever leaves ALLOWED_SUFFIXES, recordings upload and are refused."""
    block = _between(html, "function camKeep(", "$('#camOpen')")
    for ext in re.findall(r"'(mp4|webm)'", block):
        assert f".{ext}" in ALLOWED_SUFFIXES


def test_the_mime_ladder_covers_safari_and_chrome(html: str):
    """Safari records mp4 and refuses webm; Chrome/Firefox the reverse. A
    ladder missing either family silently loses that browser."""
    block = _between(html, "function camMime(", "}")
    assert "video/mp4" in block
    assert "video/webm" in block


def test_the_row_hides_for_photos_and_unsupported_browsers(html: str):
    block = _between(html, "function renderCamRow(", "}")
    assert "'photo'" in block
    assert "camSupported()" in block
    assert "camMime()" in block


def test_mode_switch_rerenders_the_row(html: str):
    """Switching video->photo->video must hide and restore the button; a row
    rendered once at load would offer recording on the photo tab."""
    seg = _between(html, "$('#modeSeg').addEventListener", "/* ---- sport tiles")
    assert "renderCamRow()" in seg


def test_errors_name_the_fallback(html: str):
    """Every camera failure ends the same way: film with the camera app and
    upload. An error that only says what broke strands the user; one that
    names the fallback loses nobody."""
    block = _between(html, "async function openCam(", "function closeCam(")
    for _err in ("NotAllowedError", "NotFoundError"):
        assert _err in block
    assert block.count("upload the clip") >= 2


# --------------------------------------------------------------------------
# the backend half: a stream-written container reports no length
# --------------------------------------------------------------------------

def test_negative_frame_count_reads_as_unknown_not_negative(monkeypatch):
    """MediaRecorder's webm is written as a stream: no frame count in the
    header, and some decoders report -1 for it. `int(...) or 0` keeps -1
    (it is truthy), and a negative count would ride into the capture report
    as a negative duration. Zero is the honest value -- every consumer
    already treats it as "could not be read"."""
    from app.services.video_analysis import runner

    class _Cap:
        def __init__(self, *_): pass
        def isOpened(self): return True
        def get(self, prop):
            import cv2
            return {cv2.CAP_PROP_FPS: 30.0, cv2.CAP_PROP_FRAME_COUNT: -1.0}.get(prop, 0.0)
        def release(self): pass

    monkeypatch.setattr(runner.cv2, "VideoCapture", _Cap)
    info = runner.get_video_info("whatever.webm")
    assert info["frame_count"] == 0.0
    assert info["duration"] == 0.0
