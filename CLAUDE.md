# ECT Crack Chat - developer notes

A local desktop app: chat on the left, figures on the right. The user asks in Vietnamese to
draw, crop borders, boost contrast or detect cracks in ECT scan files (TDMS); a Claude agent
answers and its tools draw the figures. The previous Tkinter app is gone (git tag `legacy-app`).

## Layout

- `data/` - raw TDMS scans. **Read-only** (denied in .claude/settings.json).
- `results/labels/` - DRAFT crack labels; only a person edits them (denied too).
- `results/model/crack_yolo.pt` - trained YOLO11n weights (gitignored, keep the file).
- `scripts/` - the lab's own code (dataloader.py: reshape_raster, LPF; plot_all_1Hall_heatmaps.py).
  Reference only; the app re-implements what it needs with scipy.
- `app/backend/ect/` - processing library, pure functions: scan.py (read, serpentine fix,
  catalogue, name resolution), process.py (lpf, crop, detrend, enhance), detect.py (YOLO,
  threshold, measure, labels), render.py (matplotlib PNGs in mm).
- `app/backend/agent/` - the chat agent on the Claude Agent SDK: tools.py (MCP tools), hooks.py
  (PreToolUse/PostToolUse/UserPromptSubmit), memory.py, session.py, system_prompt.md.
- `app/agent_home/` - the agent's working folder: CLAUDE.md (its fixed knowledge) and
  `.claude/skills/` (cat-vien, bat-tong, nhan-dien-vet-nut, ve-hinh). Those are the agent's
  instructions, not yours.
- `app/backend/server.py` - FastAPI on 127.0.0.1:8765, SSE chat stream.
- `app/frontend/` - temporary UI; `docs/google_ai_studio_prompt.md` is the brief for the real one.
- `app/state/` - runtime images, memory.json, logs (gitignored).

## Run and test

- Python env: `.venv` (created with `--system-site-packages`, so torch/ultralytics come from the
  user's Python 3.14). Always use `.venv/Scripts/python`.
- App: `run_app.bat`, or `.venv/Scripts/python -m app.backend.server`.
- Smoke test: `.venv/Scripts/python -m app.tests.smoke_ect [--yolo]`. A PostToolUse hook runs it
  after every edit under `app/backend/ect/`; fix a failure before moving on.
- Expected values: cambien_4lop_2 R 1.7 V 300 kHz -> z_top5 10.35; P 0.3 V 20 kHz 145x95 -> 9.17.

## Rules

- **No API.** The agent runs on this PC's Claude Code login through the SDK's bundled CLI.
  Never add an ANTHROPIC_API_KEY path or the `anthropic` client; server.py strips API keys.
- **Memory is tight on this PC** (2 GB pagefile, commit limit ~17.7 GB, often nearly full).
  Keep `OPENBLAS_NUM_THREADS=2`; don't start extra Python processes; torch may fail to load
  ("paging file is too small") - YOLO then falls back to the threshold detector by design.
- Run the server without `--reload`: the SDK needs the Proactor event loop on Windows.
- Anti-sawtooth is the lab's method: reshape_raster + direction alignment + LPF (Hamming FIR,
  order 24, cutoff 0.3, pad 50, delay removed) down each column + median 3; bilinear display.
- Coordinates: boxes and mm values are in the ORIGINAL grid frame (before crop) everywhere.
- The frontend contract (SSE events: session, delta, text, tool, image, done, error and the
  /api routes) is shared with docs/google_ai_studio_prompt.md - change both together.
- UI text and agent replies are Vietnamese; code, comments and agent instructions are English.
