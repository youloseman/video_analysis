"""Verification harness for the cabinet work.

Mounts the REAL auth + me routers and the REAL index.html against the real DB
layer, but skips app.main so the CV pipeline (cv2/mediapipe, not installed on
this machine) never gets imported. Everything under test -- the thin list, the
keyframe endpoint, the upsert keyframe-preservation -- is exercised as shipped.
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.api import auth as auth_routes
from app.api import me as me_routes
from app.core.db import init_db

STATIC = Path(__file__).resolve().parent / "app" / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(auth_routes.router)
app.include_router(me_routes.router)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
