"""Claude Code development hook (PostToolUse on Edit|Write, see .claude/settings.json).

After a change to a .py file in app/backend/ect, run the smoke test in this process (one
Python, not two - the dev PC is short of memory). Exit 2 on failure: Claude Code then shows
the failure to Claude, which fixes it before moving on. Any other file: exit 0 at once.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ECT = ROOT / "app" / "backend" / "ect"


def main():
    for stream in (sys.stdout, sys.stderr):         # non-ASCII text (σ, –, °) on a cp1252 console
        stream.reconfigure(encoding="utf-8", errors="replace")
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0
    fp = Path((payload.get("tool_input") or {}).get("file_path") or "")
    if fp.suffix != ".py" or not fp.resolve().is_relative_to(ECT):
        return 0
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    sys.path.insert(0, str(ROOT))
    sys.argv = [sys.argv[0]]                    # smoke test without --yolo: fast
    try:
        from app.tests import smoke_ect
        smoke_ect.main()
    except BaseException as e:                  # AssertionError, ImportError, SyntaxError...
        print(f"Smoke test failed after editing {fp.name}: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
