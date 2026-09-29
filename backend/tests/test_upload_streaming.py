"""Uploads reach the disk without going through memory first.

``save_upload`` replaced ``dest.write_bytes(await upload.read())`` on both video
endpoints. The old line had three faults that only appear on the clips people
actually film -- a 30-second 4K phone clip is around 150 MB:

* the whole upload became one ``bytes`` object, and the queue admits eight;
* the size cap was checked *after* that, so it bounded what was stored rather
  than what was buffered -- the one thing a cap is for;
* the write was synchronous inside an ``async def``, stopping the event loop
  for its duration.

What is tested here is the part that can go wrong quietly: the cap, and the
cleanup behind a refusal. A helper that leaves half a clip on the volume every
time somebody uploads something too big fills the disk without ever raising,
which is the same shape of failure as the storage bugs next door.

No mediapipe import: ``core.jobs`` is split out precisely so this is testable
without the analysis stack (see its module docstring).
"""

from __future__ import annotations

import io

import pytest

from app.core.jobs import UPLOAD_CHUNK_BYTES, UploadTooLarge, save_upload


# --------------------------------------------------------------------------
# the happy path
# --------------------------------------------------------------------------

def test_the_bytes_arrive_intact(tmp_path):
    payload = bytes(range(256)) * 40
    dest = tmp_path / "input.mp4"
    written = save_upload(io.BytesIO(payload), dest, max_bytes=1_000_000)
    assert written == len(payload)
    assert dest.read_bytes() == payload


def test_a_clip_larger_than_one_chunk_is_not_truncated(tmp_path):
    """The loop is the whole point; an off-by-one in it would land as a clip
    that decodes to a shorter video, which nothing downstream would call an
    error."""
    payload = b"\x00\x01\x02\x03" * UPLOAD_CHUNK_BYTES  # ~4 MB, several chunks
    dest = tmp_path / "input.mov"
    assert save_upload(io.BytesIO(payload), dest, max_bytes=len(payload)) == len(payload)
    assert dest.read_bytes() == payload


def test_exactly_the_cap_is_allowed(tmp_path):
    """The cap is a limit, not a threshold one byte below itself."""
    payload = b"x" * 5000
    dest = tmp_path / "input.mp4"
    assert save_upload(io.BytesIO(payload), dest, max_bytes=5000) == 5000
    assert dest.exists()


def test_an_empty_upload_reports_zero_rather_than_raising(tmp_path):
    """The caller turns this into "empty upload"; it is a 400, not a 413, and
    the two must not be decided in the same place."""
    dest = tmp_path / "input.mp4"
    assert save_upload(io.BytesIO(b""), dest, max_bytes=1000) == 0


def test_a_handle_mid_stream_is_rewound_first(tmp_path):
    """Whatever the framework did with the handle before passing it over, the
    upload is the whole file. Reading from wherever it was left would store a
    clip with its opening seconds missing."""
    src = io.BytesIO(b"abcdefghij")
    src.read(4)
    dest = tmp_path / "input.mp4"
    save_upload(src, dest, max_bytes=1000)
    assert dest.read_bytes() == b"abcdefghij"


# --------------------------------------------------------------------------
# the cap
# --------------------------------------------------------------------------

def test_one_byte_over_the_cap_is_refused(tmp_path):
    dest = tmp_path / "input.mp4"
    with pytest.raises(UploadTooLarge):
        save_upload(io.BytesIO(b"x" * 5001), dest, max_bytes=5000)


def test_a_refused_upload_leaves_nothing_on_disk(tmp_path):
    """The failure that costs money rather than a request: a partial file per
    rejected upload, on a volume that is a hard ceiling, deleted by nothing --
    no live job claims it and the sweeper's grace window has to expire first."""
    dest = tmp_path / "input.mp4"
    with pytest.raises(UploadTooLarge):
        save_upload(io.BytesIO(b"x" * 9_000_000), dest, max_bytes=1000)
    assert not dest.exists()
    assert list(tmp_path.iterdir()) == []


def test_the_cap_stops_the_read_rather_than_measuring_afterwards(tmp_path):
    """The point of the rewrite. A 200 MB body must not be pulled through
    memory to discover that it is 200 MB, so the refusal has to arrive within a
    chunk or two of the limit -- not after the source has been drained."""
    huge = 500 * UPLOAD_CHUNK_BYTES

    class _Counting(io.RawIOBase):
        """A source that is enormous and remembers how much was taken."""

        def __init__(self) -> None:
            self.served = 0

        def read(self, size: int = -1) -> bytes:
            if self.served >= huge:
                return b""
            n = min(size if size and size > 0 else UPLOAD_CHUNK_BYTES,
                    huge - self.served)
            self.served += n
            return b"\x00" * n

    src = _Counting()
    with pytest.raises(UploadTooLarge):
        save_upload(src, tmp_path / "input.mp4", max_bytes=2 * UPLOAD_CHUNK_BYTES)
    # Read past the cap, but by chunks -- not the whole 500 MB.
    assert src.served <= 4 * UPLOAD_CHUNK_BYTES


def test_a_write_failure_also_cleans_up(tmp_path):
    """A full volume raises OSError mid-copy. Same rule as the cap: whatever
    reached the disk goes with the exception."""
    dest = tmp_path / "input.mp4"

    class _Breaking(io.RawIOBase):
        def __init__(self) -> None:
            self.calls = 0

        def read(self, size: int = -1) -> bytes:
            self.calls += 1
            if self.calls > 2:
                raise OSError("no space left on device")
            return b"y" * 1024

    with pytest.raises(OSError, match="no space left"):
        save_upload(_Breaking(), dest, max_bytes=10_000_000)
    assert not dest.exists()
