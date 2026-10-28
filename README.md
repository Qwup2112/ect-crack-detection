# ECT Crack Chat

A chat app for analysing eddy-current testing (ECT) scans stored in TDMS files. The chat and the
answers are on the left, the result figures on the right. You ask questions or give commands,
for example "Draw the heatmap of file R 1.7V 300k", "Crop the border", "Boost the contrast",
"Detect cracks", "Compare R and P". The assistant calls the processing tools and draws the
results on the right. It answers in English; Vietnamese commands are understood too.

The brain is the **Claude Agent SDK**, the Claude Code harness packaged as a library. It runs on
the Claude Code account logged in on this PC: **no API key is needed or used**.

## Running it

Install once (on the D: drive, in the project folder):

```bat
python -m venv --system-site-packages .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Then double-click **`ECT Crack Chat`** (the shortcut with the icon in the project folder) or
`run_app.bat`. It is a program with its own window (WebView2, built into Windows 11): no browser,
no link. The backend runs hidden inside the program; closing the window stops everything.
To put a shortcut on the Desktop:
`powershell -ExecutionPolicy Bypass -File app\assets\make_shortcut.ps1 -Dest "$env:USERPROFILE\Desktop"`.

The PC must be logged in to Claude Code (the `claude` command works). Log: `app/state/desktop.log`.

## What the assistant can do

| Tool | Job |
|---|---|
| `list_scans`, `scan_info` | Browse the catalogue of 110 files (101 readable), parameters, warnings |
| `draw_scan` | Draw: raw, filtered, border cropped, amplitude/phase map, background removed, dA/dx derivative, z-map, 6-panel report, YOLO input image |
| `crop_border` | Automatic border crop (measured per edge, usually asymmetric) or fixed, showing the cropped bands |
| `enhance_contrast` | Contrast boost: stretch, background removal, local normalisation, gamma, strong |
| `detect_cracks` | Crack detection with YOLO (runs on ONNX, ~0.4 s, light on memory; falls back to the threshold detector if it cannot run), scored against the labels |
| `measure_crack` | Length, width, angle and SNR of one crack |
| `compare_scans` | Compare 2–8 files, ranked by z_top5 |
| `profile_line` | 1D profile along X or Y |
| `run_python` | **In-depth analysis**: the assistant writes and runs its own Python (statistics over many files, trends over frequency, FFT, histograms, any plot) |
| `remember`, `recall`, `forget` | Long-term memory (preferences, things learned) |

The assistant **does the whole job itself**: for "analyse file X", "detect cracks" or "send the
final result" it crops the border, boosts the contrast, detects, measures and reports in one
answer, without stopping to ask at every step.

**What `run_python` may do:** the code runs in a separate process (at most 180 s) and can write
files only in `app/state/work/<chat>/`; `data/` and `results/labels/` are read-only; it cannot
start other programs, use the network or write the registry. Every piece of code is logged in
`app/state/logs/tools.jsonl`. To turn the tool off, set the environment variable
`ECT_PYTHON_TOOL=0`.

The maps have no sawtooth thanks to the lab's method (`scripts/dataloader.py`): reverse the odd
lines of the zig-zag scan (`reshape_raster`), align the forward and return lines, low-pass filter
(`LPF`, Hamming FIR of order 24) down each column, then display with bilinear interpolation.
Measured on 3 files, 20–30% of the sawtooth is left compared with reversing the lines only.

## Harness: memory, tools, hooks, skills

- **Memory**: `app/agent_home/CLAUDE.md` (fixed knowledge about the dataset) +
  `app/state/memory.json` (what the assistant learns while chatting, through `remember`) + the
  current chat session.
- **Tools**: an in-process MCP server (`app/backend/agent/tools.py`) with only the tools above;
  Claude Code's Bash, file reading/writing and so on are off. Free-form analysis goes through
  `run_python` (guarded).
- **Hooks** (`app/backend/agent/hooks.py`): PreToolUse blocks unknown tools and turns short scan
  names into exact ids (or refuses with the candidates when ambiguous); PostToolUse reports the
  status to the UI and writes the log; UserPromptSubmit adds the memory to every question.
- **Skills** (`app/agent_home/.claude/skills/`): teach the assistant the **automatic end-to-end
  analysis** (`auto-analysis`), **border cropping** (`border-crop`), **contrast boost**
  (`contrast`), **crack detection** (`crack-detection`), **which figure to draw** (`drawing`)
  and **in-depth analysis with Python** (`python-analysis`).

When developing with Claude Code: `CLAUDE.md` at the project root, `.claude/settings.json` blocks
writes to `data/` and `results/labels/`, and a hook runs the smoke test after every edit to
`app/backend/ect/`.

## Interface

`app/frontend/index.html` is a design made with Google AI Studio
(prompt: [`docs/google_ai_studio_prompt.md`](docs/google_ai_studio_prompt.md)), connected to the
backend: the answer text streams in, each tool shows its status, figures appear on the right,
"Open original" opens the image in the Windows image viewer, "Save image" opens a save dialog.
The AI Studio version's "simulation" (fake images and invented numbers when offline) was removed:
every figure and number comes from the real tools.

## Notes on the data

- The labels in `results/labels/` are a **draft**, not reviewed by a person. Every score against
  them is provisional.
- The YOLO model learned only on the 15-crack calibration coupon (3 × 5 grid); it still fires on
  noise images.
- 1 empty file, 10 truncated files and 18 files with the old naming convention: the tools read
  them and report a warning.

## Layout

```
data/                 original TDMS files (read-only)
results/labels/       draft labels;  results/model/  YOLO weights
scripts/              the lab's code (dataloader.py, plot_all_1Hall_heatmaps.py)
app/backend/ect/      processing: read, filter, border crop, contrast, detection, drawing
app/backend/agent/    the assistant: tools, hooks, memory, chat session, system prompt
app/agent_home/       the assistant's CLAUDE.md + skills
app/backend/server.py FastAPI backend (runs hidden inside the program)
app/desktop.py        the desktop program: WebView2 window + backend start-up
app/frontend/         the UI (Google AI Studio design, connected to the backend)
app/assets/           icon, shortcut script
app/state/            images, memory, logs at run time (not in git)
docs/                 the Google AI Studio prompt
```

The old program (Tkinter app) can be restored with `git checkout legacy-app`.
