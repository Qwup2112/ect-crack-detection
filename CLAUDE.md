# ECT Crack Chat - developer notes

A local desktop app: chat on the left, figures on the right. The user asks in Vietnamese to
draw, crop borders, boost contrast or detect cracks in ECT scan files (TDMS); a Claude agent
answers and its tools draw the figures. The previous Tkinter app is gone (git tag `legacy-app`).

## Layout

- `data/` - raw TDMS scans. **Read-only** (denied in .claude/settings.json).
- `results/labels/` - DRAFT crack labels; only a person edits them (denied too).
- `results/model/crack_yolo.pt` - trained YOLO11n weights (gitignored, keep the file).
  `crack_yolo.onnx` is its export, what the app actually runs (onnxruntime, ~50 MB instead of
  torch's 1+ GB, which made detection crash this PC). After retraining, re-export and verify with
  `python -m app.tools.export_onnx` (needs torch; the CUDA torch in the user's Python often cannot
  load here, so use the CPU-only venv `D:\Claude\ect-onnx-export\venv\Scripts\python`). It must
  print OK: identical boxes to PyTorch on every readable scan. The export is dynamic-shape because
  ultralytics predicts with rect letterboxing (150 x 101 -> 320 x 224), not a square.
- `scripts/` - the lab's own code (dataloader.py: reshape_raster, LPF; plot_all_1Hall_heatmaps.py).
  Reference only; the app re-implements what it needs with scipy.
- `app/backend/ect/` - processing library, pure functions: scan.py (read, serpentine fix,
  catalogue, name resolution), process.py (lpf, crop, detrend, enhance), detect.py (YOLO,
  threshold, measure, labels), render.py (matplotlib PNGs in mm).
- `app/backend/agent/` - the chat agent on the Claude Agent SDK: tools.py (MCP tools), hooks.py
  (PreToolUse/PostToolUse/UserPromptSubmit), memory.py, session.py, system_prompt.md,
  python_tool.py (run_python: the agent's own code in a child process; an audit hook limits
  writes to app/state/work/<chat>/ and blocks subprocesses, network and registry writes;
  ECT_PYTHON_TOOL=0 turns it off), sysmem.py (free commit memory).
- `app/agent_home/` - the agent's working folder: CLAUDE.md (its fixed knowledge) and
  `.claude/skills/` (phan-tich-tu-dong = do the whole pipeline without asking, cat-vien, bat-tong,
  nhan-dien-vet-nut, ve-hinh, phan-tich-python). Those are the agent's instructions, not yours;
  a new skill must also be added to SKILLS in session.py.
- `app/tools/` - developer scripts (export_onnx.py).
- `app/backend/server.py` - FastAPI, SSE chat stream. Normally started by app/desktop.py on a
  free 127.0.0.1 port inside the desktop program (standalone: port 8765, for debugging).
- `app/desktop.py` - the program: pywebview window (WebView2), splash, js_api for opening /
  saving images. The user wants a desktop program, never a browser link.
- `app/frontend/index.html` - the user's Google AI Studio design, connected to the backend. Its
  original "local simulation" (fake heatmaps and invented numbers) was removed on purpose; never
  bring back simulated results.
- `app/state/` - runtime images, memory.json, logs, run_python work folders (gitignored).

## Run and test

- Python env: `.venv` (created with `--system-site-packages`, so torch/ultralytics come from the
  user's Python 3.14). Always use `.venv/Scripts/python`.
- App: "ECT Crack Chat.lnk" / `run_app.bat` (= `.venv/Scripts/pythonw -m app.desktop`, log in
  app/state/desktop.log). Backend alone for debugging: `.venv/Scripts/python -m app.backend.server`.
- Smoke test: `.venv/Scripts/python -m app.tests.smoke_ect [--yolo]`. A PostToolUse hook runs it
  after every edit under `app/backend/ect/`; fix a failure before moving on.
- Expected values: cambien_4lop_2 R 1.7 V 300 kHz -> z_top5 10.35; P 0.3 V 20 kHz 145x95 -> 9.17.

## Rules

- **No API.** The agent runs on this PC's Claude Code login through the SDK's bundled CLI.
  Never add an ANTHROPIC_API_KEY path or the `anthropic` client; server.py strips API keys.
- **Memory is tight on this PC** (2 GB pagefile, commit limit ~17.7 GB, often nearly full).
  Keep `OPENBLAS_NUM_THREADS=2`; don't start extra Python processes (run_python's child is the
  exception: it checks free commit first and exits after each run). Never import torch in the
  app: the CUDA build cannot even load here ("paging file is too small"), and loading it during
  a detection once starved the CLI to death. YOLO runs on onnxruntime; the threshold detector is
  the fallback if that fails.
  Each chat's bundled Claude Code CLI commits ~0.5 GB; with under ~1 GB free commit it dies at
  start-up with exit 0xC0000409 or 0x80000003 and an empty stderr - that is memory, not code.
  So server.py keeps one live CLI (idle chats resume via `resume=`), and session.py starts it
  with CREATE_NO_WINDOW (under pythonw it would otherwise get its own terminal window).
- Run the server without `--reload`: the SDK needs the Proactor event loop on Windows.
- Anti-sawtooth is the lab's method: reshape_raster + direction alignment + LPF (Hamming FIR,
  order 24, cutoff 0.3, pad 50, delay removed) down each column + median 3; bilinear display.
- Coordinates: boxes and mm values are in the ORIGINAL grid frame (before crop) everywhere.
- The frontend contract (SSE events: session, delta, text, tool, image, done, error and the
  /api routes) is shared with docs/google_ai_studio_prompt.md - change both together.
- UI text and agent replies are Vietnamese; code, comments and agent instructions are English.
