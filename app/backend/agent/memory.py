"""Long-term memory, kept across chats in app/state/memory.json.

Three layers of memory in this app:
  1. agent_home/CLAUDE.md - fixed knowledge about the dataset and the rules (loaded by the
     harness at every session start).
  2. this file           - what the agent learns while chatting: the user's preferences
     ("always use magma"), facts ("the 10 kHz pair is faint"), notes. Written by the
     remember tool, read by recall, and a short digest is injected into every prompt by
     the UserPromptSubmit hook.
  3. the SDK session     - the running conversation itself.
"""
import json
import threading
import time

from ..ect.paths import MEMORY_FILE, ensure_state_dirs

KINDS = ("preference", "fact", "note")
_lock = threading.Lock()


def _load():
    if MEMORY_FILE.exists():
        return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    return {"items": [], "recent_scans": [], "next_id": 1}


def _save(d):
    ensure_state_dirs()
    tmp = MEMORY_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(MEMORY_FILE)


def add(text, kind="note"):
    text = " ".join(str(text).split())
    if not text:
        raise ValueError("empty memory")
    kind = kind if kind in KINDS else "note"
    with _lock:
        d = _load()
        for it in d["items"]:
            if it["text"].lower() == text.lower():
                return it
        it = {"id": f"m{d['next_id']}", "kind": kind, "text": text,
              "created": time.strftime("%Y-%m-%d %H:%M")}
        d["next_id"] += 1
        d["items"].append(it)
        _save(d)
        return it


def forget(mid):
    with _lock:
        d = _load()
        before = len(d["items"])
        d["items"] = [i for i in d["items"] if i["id"] != mid]
        _save(d)
        return len(d["items"]) < before


def search(query=""):
    words = [w for w in str(query).lower().split() if w]
    items = _load()["items"]
    return [i for i in items if all(w in i["text"].lower() for w in words)] if words else items


def touch_scan(sid):
    with _lock:
        d = _load()
        d["recent_scans"] = [sid] + [s for s in d["recent_scans"] if s != sid][:9]
        _save(d)


def digest(max_items=15):
    """Short text block for the prompt: preferences first, then recent facts/notes."""
    d = _load()
    prefs = [i for i in d["items"] if i["kind"] == "preference"]
    other = [i for i in d["items"] if i["kind"] != "preference"]
    room = max(0, max_items - len(prefs))
    chosen = prefs[-max_items:] + (other[-room:] if room else [])
    lines = [f"- [{i['id']}, {i['kind']}] {i['text']}" for i in chosen]
    if d["recent_scans"]:
        lines.append("- recent scans: " + ", ".join(d["recent_scans"][:5]))
    return "\n".join(lines)


def all_items():
    return _load()
