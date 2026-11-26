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
CSS = SPA.with_name("app.css")


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


def test_the_viewfinder_has_guides_level_countdown_and_clock(html: str):
    modal = _between(html, 'id="camModal"', "<div class=\"hero\">")
    for needle in ('id="camView"', "cam-guide", 'id="camLevel"', 'id="camCount"',
                   'id="camRec"', 'id="camStart"', 'id="camStop"',
                   'id="camTimer"', 'id="camFlip"', 'id="camPick"'):
        assert needle in modal, f"camera modal is missing {needle}"


def test_the_viewfinder_is_the_whole_screen(html: str):
    """Artur 2026-11-25: the phone stands three or four metres away; in a
    small box inside a white card the athlete cannot find themselves. No card,
    no title on screen, and the picture is shown WHOLE -- a cropped preview
    would promise a frame the recording does not have."""
    modal = _between(html, 'id="camModal"', "<div class=\"hero\">")
    assert '<div class="modal modal-cam hidden" id="camModal"' in html and "modal-card" not in modal
    assert "Film it right here" not in html
    css = CSS.read_text(encoding="utf-8")
    assert ".modal-cam{padding:0;" in css
    assert re.search(r"\.cam-stage video\{[^}]*object-fit:contain", css)


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
    assert "CAM_COUNTDOWN_S=5" in html          # the default until the athlete picks
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
# the viewfinder (2026-11-25): guide lines, level, timer, the other camera
# --------------------------------------------------------------------------

def test_the_countdown_is_the_athletes_choice(html: str):
    """Five fixed seconds fit nobody: a friend filming just stands there, and
    alone on a trainer five is not enough to climb on and clip in."""
    assert "const CAM_TIMERS=[0,5,10]" in html
    modal = _between(html, 'id="camModal"', "<div class=\"hero\">")
    assert [int(t) for t in re.findall(r'data-t="(\d+)"', modal)] == [0, 5, 10]
    block = _between(html, "function camStartFlow(", "function camRecord(")
    assert "let n=camTimerS;" in block and "if(!n){ camRecord(); return; }" in block
    assert "a second tap calls it off" in block, "a running countdown can no longer be cancelled"
    assert "localStorage.setItem(CAM_TIMER_KEY" in html


def test_the_take_is_heard_and_no_microphone_is_opened(html: str):
    """From the saddle the screen is unreadable: the countdown and both ends
    of the take are beeps. Output only -- the sound context is made on the
    Start tap, and nothing here asks for a microphone."""
    block = _between(html, "function camStartFlow(", "function camRecord(")
    assert "camAudioWake()" in block and "camBeep(" in block
    rec = _between(html, "function camRecord(", "function camKeep(")
    assert rec.count("camBeep(") >= 3          # rolling, and two notes at the end
    assert "audio:true" not in html.replace(" ", "")


def test_swapping_cameras_stops_the_old_one_first(html: str):
    """A phone will not run both cameras at once: ask for the second while
    the first is live and the request fails (or the first goes black)."""
    block = _between(html, "async function camStream(", "async function camFlip(")
    assert block.index("getTracks().forEach(t=>t.stop())") < block.index("getUserMedia")
    flip = _between(html, "async function camFlip(", "function closeCam(")
    assert "await camStream(was)" in flip, "a failed swap must come back to the camera that worked"


def test_only_the_preview_is_mirrored(html: str):
    """The front camera's preview is a mirror because that is how people move
    in front of one. The RECORDING must stay as the camera saw it: the
    analysis reads which way the rider faces."""
    assert "classList.toggle('mirror', front)" in html
    assert ".cam-stage.mirror video{transform:scaleX(-1)}" in CSS.read_text(encoding="utf-8")
    rec = _between(html, "function camRecord(", "function camKeep(")
    assert "new MediaRecorder(camState.stream" in rec


def test_two_guide_lines_and_no_silhouette(html: str):
    """Artur 2026-11-25, second pass: the rider and runner outlines went; the
    athlete fits between two hairlines, head under one and feet over the
    other. They are drawn on the filmed picture's own rectangle and take the
    verdict's colour."""
    modal = _between(html, 'id="camModal"', "<div class=\"hero\">")
    frame = _between(modal, 'id="camFrame"', 'id="camLevel"')
    assert re.findall(r'<span class="cam-guide[^"]*">(\w+)</span>', frame) == ["Head", "Feet"]
    assert "cam-st-" not in html and "camGhost" not in html
    css = CSS.read_text(encoding="utf-8")
    # 20% / 80% since the same evening: at 12 / 88 filling the lines meant
    # standing too close.
    assert re.search(r"\.cam-guide\{[^}]*top:20%[^}]*color:var\(--cam-tone\)", css)
    assert ".cam-guide--feet{top:80%;" in css


def test_zoom_is_the_cameras_own_and_asks_for_nothing(html: str):
    """Zoom is set on the track, so the clip is zoomed and not only the
    preview. It is offered where the browser exposes it and never requested
    in getUserMedia: on desktop that would raise a second permission prompt
    ("move your camera") for a button most webcams cannot honour."""
    modal = _between(html, 'id="camModal"', "<div class=\"hero\">")
    assert 'class="cam-btn cam-zoom hidden" id="camZoom"' in modal, "the zoom button must start hidden"
    read = _between(html, "function camZoomRead(", "function camZoomX(")
    assert "getCapabilities" in read and "caps.zoom.max>caps.zoom.min" in read
    assert "applyConstraints({advanced:[{zoom:z.val}]})" in _between(html, "function camZoomSet(", "function camZoomStep(")
    stream = _between(html, "async function camStream(", "async function camFlip(")
    assert "zoom:true" not in stream.replace(" ", "")
    assert "camZoomRead()" in stream, "a swapped camera has its own zoom range"


def test_motion_access_is_asked_inside_the_tap(html: str):
    """iOS grants DeviceOrientation only from a user gesture. Asked after the
    camera's await, the level silently never appears on any iPhone."""
    block = _between(html, "async function openCam(", "async function camStream(")
    assert block.index("camLevelAsk()") < block.index("await ")
    ask = _between(html, "function camLevelAsk(", "function camLevelRead(")
    assert "requestPermission" in ask
    stop = _between(html, "function camLevelStop(", "/* ---- Live GO")
    assert "removeEventListener('deviceorientation',camLevelRead)" in stop


def test_the_level_is_green_within_two_degrees(html: str):
    assert "const CAM_LEVEL={okDeg:2, offDeg:3," in html
    paint = _between(html, "function camLevelPaint(", "function camLevelStop(")
    assert "camLevel.ok?CAM_LEVEL.offDeg:CAM_LEVEL.okDeg" in paint      # in at 2, out at 3: no blinking
    shown = _between(html, "function camLiveShown(", "function camLivePaint(")
    assert "Level the phone" in shown, "a good distance on a tilted phone must not read GO"


def test_a_verdict_is_a_command(html: str):
    """Artur 2026-11-25: not an abstract NO-GO. The athlete is told which way
    to move the phone."""
    block = _between(html, "function camLiveRead(", "function camLiveShown(")
    for text in ("Move closer", "Step back", "Tilt up", "Tilt down", "A bit closer", "Good distance"):
        assert text in block, text
    assert "NO-GO" not in block.split("*/")[-1], "the pill shows the command, not the grade"


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
