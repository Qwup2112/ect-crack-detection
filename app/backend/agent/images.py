"""Images the tools draw: saved under app/state/images/<id>.png with a JSON sidecar, served
to the right-hand panel at /api/images/<id>.png. A figure with numbers behind it (see
ect/interactive.py) also has <id>.data.gz, served at /api/images/<id>/data."""
import gzip
import json
import time
import uuid

from ..ect.paths import IMAGES_DIR, ensure_state_dirs


def save(png_bytes, title, caption="", scan=None, tool=None, session=None, data=None):
    ensure_state_dirs()
    iid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    (IMAGES_DIR / f"{iid}.png").write_bytes(png_bytes)
    meta = {"id": iid, "url": f"/api/images/{iid}.png", "title": title, "caption": caption,
            "data_url": f"/api/images/{iid}/data" if data else None,
            "scan": scan, "tool": tool, "session": session, "created": time.time()}
    if data:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        (IMAGES_DIR / f"{iid}.data.gz").write_bytes(gzip.compress(body, compresslevel=1))
    (IMAGES_DIR / f"{iid}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return meta


def path_of(iid):
    p = IMAGES_DIR / f"{iid}.png"
    return p if p.exists() and p.parent == IMAGES_DIR else None


def data_path_of(iid):
    p = IMAGES_DIR / f"{iid}.data.gz"
    return p if p.exists() and p.parent == IMAGES_DIR else None


def for_session(session):
    out = []
    for j in sorted(IMAGES_DIR.glob("*.json")):
        m = json.loads(j.read_text(encoding="utf-8"))
        if m.get("session") == session:
            out.append(m)
    return out
