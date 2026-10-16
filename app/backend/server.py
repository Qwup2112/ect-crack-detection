"""Web backend: serves the UI and streams the chat as Server-Sent Events.

Run:  .venv\\Scripts\\python -m app.backend.server      (or run_app.bat)
Listens on 127.0.0.1 only. Port: ECT_PORT (default 8765).
"""
import os

# This PC runs close to its memory commit limit: OpenBLAS with one thread per core then fails
# to allocate at import. Two threads is plenty for ~150 x 100 point scans. Must be set before
# numpy is imported anywhere.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

# No API: the agent must run on this machine's Claude Code login (the harness), never on a
# pay-per-token API key. Drop any key from the environment the bundled CLI would inherit.
for _k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"):
    os.environ.pop(_k, None)

import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agent import images, memory
from .agent.session import MODEL, ChatSession
from .ect import detect, scan
from .ect.paths import APP_DIR, ensure_state_dirs

FRONTEND = APP_DIR / "frontend"
PORT = int(os.environ.get("ECT_PORT", "8765"))
SESSIONS: dict[str, ChatSession] = {}


@asynccontextmanager
async def lifespan(app):
    ensure_state_dirs()
    await asyncio.to_thread(scan.catalog)        # warm the scan catalogue
    yield
    for s in list(SESSIONS.values()):
        await s.close()


app = FastAPI(title="ECT crack chat", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


class ChatIn(BaseModel):
    message: str
    session_id: str | None = None


def sse(kind, data):
    return f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


@app.get("/api/health")
def health():
    cat = scan.catalog()
    return {"ok": True, "model": MODEL, "yolo_weights": detect.yolo_available(),
            "scans": len(cat), "readable": sum(e["status"] == "ok" for e in cat)}


@app.post("/api/chat")
async def chat(body: ChatIn):
    text = body.message.strip()
    if not text:
        raise HTTPException(400, "empty message")
    sess = SESSIONS.get(body.session_id or "")
    if sess is None:
        sess = ChatSession(body.session_id)
        SESSIONS[sess.id] = sess

    async def stream():
        yield sse("session", {"id": sess.id})
        async for kind, data in sess.ask(text):
            yield sse(kind, data)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/session/{sid}/reset")
async def reset(sid: str):
    s = SESSIONS.pop(sid, None)
    if s:
        await s.close()
    return {"ok": True}


@app.get("/api/session/{sid}/images")
def session_images(sid: str):
    return images.for_session(sid)


@app.get("/api/images/{iid}.png")
def image(iid: str):
    p = images.path_of(iid)
    if p is None:
        raise HTTPException(404)
    return FileResponse(p, media_type="image/png")


@app.get("/api/scans")
def scans():
    return scan.catalog()


@app.get("/api/memory")
def get_memory():
    return memory.all_items()


@app.delete("/api/memory/{mid}")
def del_memory(mid: str):
    return {"deleted": memory.forget(mid)}


def main():
    import sys
    import uvicorn
    for stream in (sys.stdout, sys.stderr):         # logs go to a file / cp1252 console
        stream.reconfigure(encoding="utf-8", errors="replace")
    # loop="asyncio" without reload: on Windows that is the Proactor loop, which the Agent
    # SDK needs to start the bundled Claude Code process.
    uvicorn.run(app, host="127.0.0.1", port=PORT, loop="asyncio", log_level="warning")


if __name__ == "__main__":
    main()
