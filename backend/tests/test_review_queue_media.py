"""The review queue shows the reviewer the clip and the pictures.

On the first real Coach Notes order (2026-10-12) the queue showed a black
player and an empty frame: the clip URL is admin-only and a <video src>
request cannot carry the bearer token (401), and the frame was read from the
history entry -- the free reader's copy, with keyframes long since moved to
their own column. The reviewer is not the free reader: the order now carries
the annotated frame WITH its numbers and the kinogram, and the page fetches
the clip with the token.
"""
from __future__ import annotations

from app.api.billing import _analysis_media
from app.models.analysis import Analysis

from conftest import read_spa

KF_FULL = "data:image/jpeg;base64,FULL"
KF_FREE = "data:image/jpeg;base64,FREE"
KINO = "data:image/jpeg;base64,KINO"


async def test_the_reviewer_gets_the_frame_with_numbers_and_the_kinogram(db, make_user):
    user = await make_user()
    db.add(Analysis(user_id=user.id, client_id="a1", created_at_ms=1, sport="bike", kind="video",
                    score=78, keyframe=KF_FREE, data={"id": "a1", "locked": True},
                    result={"keyframe_base64": KF_FULL, "kinogram_base64": KINO}))
    await db.commit()
    media = await _analysis_media(db, user.id, "a1")
    assert media == {"keyframe": KF_FULL, "kinogram": KINO}


async def test_an_older_analysis_falls_back_to_its_stored_frame(db, make_user):
    user = await make_user()
    db.add(Analysis(user_id=user.id, client_id="a2", created_at_ms=1, sport="run", kind="video",
                    score=70, keyframe=KF_FREE, data={"id": "a2"}, result=None))
    await db.commit()
    assert await _analysis_media(db, user.id, "a2") == {"keyframe": KF_FREE, "kinogram": None}
    assert await _analysis_media(db, user.id, None) == {"keyframe": None, "kinogram": None}


def test_the_player_is_fed_with_the_token_not_a_bare_src():
    spa = read_spa()
    assert 'src="/admin/orders/${d.id}/clip"' not in spa
    i = spa.index("async function exLoadClip(")
    body = spa[i:spa.index("\n}", i)]
    assert "headers:authHeaders()" in body and "createObjectURL" in body
