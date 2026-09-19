# Design outline: a specialized agent for ECT crack-detection heatmaps

## 1. Data context (verified by code, not assumed)

- Source format: `.tdms` (National Instruments), produced by a 2D raster eddy-current (ECT) scanner.
- Structure of every file:
  - Group **`Freq_Sampling_SizeX_SizeY`** — 1 channel, 4 values: `[frequency Hz, sampling, sizeX, sizeY]`.
  - Group **`Waveform`** — 1 channel, a flat array of length `sizeX × sizeY` (raster: X fast, Y slow) reshaped to `(sizeY, sizeX)`.
- Filenames encode the measurement parameters. **This dataset uses five different naming conventions**, which matters more than it looks:

  | Pattern | Example |
  |---|---|
  | standard, with plate size | `4lop_R_amp_1.3V_fre_200k_lf_1mm_145x95mm` |
  | standard, no plate size | `4lop_R_amp_4V_fre_10k_lf_1mm` |
  | `freq` / capital `K` / `liffoff` (a typo) | `R_amp_3V_freq_20K_liffoff_1mm` |
  | `_tron` suffix (different defect shape) | `R_amp_4V_fre_100k_lf_1mm_tron` |
  | template files, no parameters | `R_default`, `Default_Differential` |

- `sensor` is **R** (amplitude channel, ~0.01–0.02 scale) or **P** (phase channel, tens), plus `PECT` / `Differential` / `Feedback`.
- Files ending in `_default.tdms` are usually **empty** (metadata only, no `Waveform`) — configuration/calibration templates. They must be skipped explicitly, never silently counted as zero.
- The specimen is a calibration coupon with **15 artificial cracks** in a 3 × 5 grid at varying depths, visible on the heatmap as periodic dark/bright column pairs.

## 2. Goal

Automate the whole chain — **read TDMS → rebuild the 2D scan image → preprocess → produce the six required views → locate cracks → build a sensor comparison table → analyze the entire dataset → have an LLM interpret the numbers** — so nobody has to rewrite analysis code for every new file.

Governing principles, inherited from the `tdms-analysis` skill: **code computes the numbers, the model only interprets them**; **"not enough data" beats a confident wrong number**; every anomaly (empty file, sample-count mismatch, missing plate size) must appear in the log rather than be silently smoothed over.

## 3. The six required views

| # | Name | Source | Design note |
|---|---|---|---|
| 1 | **Raw (before filter)** | `Waveform` reshaped, untouched | The baseline. If filtering or cropping destroys real signal, this is where it shows. |
| 2 | **Filtered** | Raw through a median/gaussian filter (configurable kernel) | Median by default — it does not blunt crack edges the way a heavy gaussian does. |
| 3 | **Cropped** | Filtered, N px trimmed per side | Mandatory: filters always create border artifacts (criterion A2). |
| 4 | **Amplitude heatmap** | Cropped, perceptual colormap (`magma`/`viridis`, **never jet** — criterion E1) | The figure people actually read. Fix `vmin`/`vmax` when comparing images (E2). |
| 5 | **X-Derivative** (`dA/dx`) | `np.gradient(cropped, axis=X)` | X is the fast scan axis, where crack edges show as a sign reversal. |
| 6 | **X-Derivative Inverse** (`-dA/dx`) | Sign-flipped (5) | A crack is a **dipole**: the leading edge gives one sign, the trailing edge the other (which one depends on sensor and frequency). The inverted view makes polarity comparable across sensors/frequencies and exposes the edge the original view buries in the diverging colormap's white midpoint. |

Panels 5 and 6 use a **diverging colormap symmetric about zero** (`RdBu_r` + `TwoSlopeNorm(vcenter=0)`), because positive and negative mean physically different things here (entering vs leaving edge), not two ends of one continuum.

## 4. Crack localization + border-noise cropping (`detect`)

### Automatic border cropping

A fixed N-px crop is wrong in practice: the noise band at the start of a scan line is much thicker than at the end. Measured on the R 200 kHz file: **top 14 px, bottom 3 px, left 7 px, right 4 px** — an even 5 px crop both misses noise at the top and discards good data at the bottom.

`--margin auto` (the default) measures the border from **two** signals, because border noise appears in two different ways:

| Signal | How it shows | If you measure only this one |
|---|---|---|
| **Level** — row/column median clearly offset | The bright band at the start of a scan line | You miss the streaks where the probe turns around |
| **Spread** — abnormally large dispersion | Left/right noise streaks with a normal median | You miss the bright start-of-scan band |

Stopping rule: walk inward **from each edge** while the robust z-score (median/MAD — *not* mean/std, since cracks and artifacts are themselves the extreme values that would drag the scale) stays above threshold, and **stop at the first normal row/column**. Only the contiguous border band is trimmed; real signal in the middle is never touched. Trimming is capped at 15% per side, falling back to a fixed 5 px crop with a warning if the whole image looks noisy.

### Locating cracks

`detrend → robust z-score → threshold |z| → merge fragments → bounding boxes`

Three steps decide the outcome and are easy to skip:

- **Pick the right detrend direction (`--detrend`, default `row`).** A row detrend only removes trends along Y. Sensor P in this dataset has a bright background on **both left and right** — a trend along X that a row detrend never touches, leaving the mask dominated by border bands. Measured: P @200 kHz gives **4 fragmented candidates** with `row` but **7 correctly placed ones** with `both`. Rule: look at the z-score map; detrend along whichever axis the background slopes.
- **Merge fragments (`--merge-gap`, default 2).** One crack's signal usually breaks into several horizontal streaks; labelling directly counts it as many candidates (measured: **35 candidates before merging, 13 after**). Merging dilates the mask, then computes each box from the **original** pixels so sizes are not inflated. Note: crack rows on this coupon are only ~5 px apart, so a merge gap of 3 fuses two rows into one — hence the default of 2, not 3.
- **Flag edge-touching boxes (`touches_edge`).** Artifacts surviving the crop always sit against the edge. The tool **flags** them (orange, dashed) rather than silently dropping them, because a genuine crack near the plate edge also touches the edge. `--drop-edge-boxes` removes them (measured: 13 → 9).

Outputs: `crack_detect_report.png` (heatmap with located cracks, z-score map, threshold mask), `auto_boxes.json`, `auto_boxes_flat.json` (**usable directly as `--boxes` for `compare` or for the `tdms-analysis` skill's `scan_metrics.py`**), `auto_boxes.csv`.

> **Coordinates in `auto_boxes.json` are relative to the CROPPED image, not the original.** Add `crop.left` / `crop.top` to convert. The `crop` block is written into the JSON precisely so this offset can be applied automatically — feeding cropped coordinates into a full-image measurement misaligns every label by exactly the border width, which is the classic fault that produces recall 0 and gets blamed on the model.

### Limits to state when reporting

This is a **threshold** detector, not a machine-learning model. It marks "abnormal deviation from the background" — whether that is a crack, a plate edge, or a scan artifact must be checked against the specimen drawing. Also, an ECT crack signature is a **dipole**, so a bounding box may straddle the bright/dark pair rather than sit on the crack centre.

## 5. Sensor comparison table (`compare`)

Answers criteria B2/B3 of the skill — *which frequency/sensor gives the best signal, and is combining channels worth it?* — with numbers rather than impressions.

Columns: `sensor, amp_V, freq_kHz, lf_mm, size_px, mean, std, rms, dynamic_range, std_after_row_detrend, dx_std`, plus **`CNR` and `CNR_abs`** when a box file is supplied.

### Why there are two CNR columns

An ECT crack signature is a **dipole**: a high-amplitude lobe beside a low-amplitude one. When a box covers both lobes, the mean inside the box ≈ the background mean because the lobes cancel, so **the signed CNR collapses toward zero even when the signal is obvious**. That trap can lead a reader to conclude "there is no signal here" and abandon a perfectly good measurement setup.

`CNR_abs` (computed on absolute deviation) is immune. Measured on this dataset:

| file | signed CNR | CNR_abs |
|---|---|---|
| R 4V 200 kHz | 0.85 | **1.94** |
| R 1.3V 200 kHz | 0.71 | **1.75** |

Reading rule: **`CNR_abs` >> `CNR` is the fingerprint of a dipole signature.** Read `CNR_abs`, and remember that a solid box label misdescribes the phenomenon if it is later used to train a model (criterion B4).

Ranking by `CNR_abs` also **reverses** the ranking by `dx_std`: `dx_std` puts sensor P thousands of times above R, while `CNR_abs` puts R above P. The reason is that `dx_std` only measures overall "roughness" — sensor P has a large value scale and correspondingly large noise, so it looks high-contrast without actually separating cracks from background. **That is exactly why `dx_std` must not be used to choose a sensor.**

## 6. Whole-dataset analysis (`sweep`)

Recursively scans every `.tdms`, runs the identical pipeline on each, and emits one table. Measured: **110 files / 10.9 MB in ~9 seconds**.

```bash
python tools/crack_heatmap/tdms_crack_scan.py sweep . --out sweep_all.csv
```

**The ranking metric is `z_top5`** (mean of the five highest z scores), not `dx_std`. This is the decisive property: z is already normalized by each image's own background noise (median/MAD), so it **compares fairly between sensor R (~0.01 scale) and sensor P (~30 scale)**, whereas `dx_std` merely rewards a large value scale. Ranking by `z_top5` agrees with ranking by `CNR_abs` (R above P) and disagrees with `dx_std` — three independent measures corroborating each other.

Results on the current dataset:

| folder | sensor | best configuration | z_top5 |
|---|---|---|---|
| cambien_4lop_2 | R | 4 V, 200 kHz | **15.61** |
| cam_4lop_1 | R | 4 V, 160 kHz | 14.18 |
| data | R | 4 V, 200 kHz | 8.35 |
| cambien_4lop_2 | P | 0.19 V, 20 kHz | 8.67 |
| cam_4lop_1 | P | 4 V, 30 kHz | 6.28 |
| data | P | 4 V, 20 kHz | 6.22 |

### Three data faults only the full sweep exposed

1. **18 files use a different naming convention** (`freq_20K` instead of `fre_20k`, `liffoff` instead of `lf`, a `_tron` suffix). The original regex did not match them, so they **silently lost every measurement parameter**. The regex now covers all variants: 102/110 files parse, and the remaining 8 are genuinely the `*_default` templates.
2. **`data/R_amp_5V_freq_100K_liffoff_1mm.tdms` is empty** — 230 bytes where its siblings are ~109 KB, with no `Waveform` group: that measurement was never written. Its internal metadata also says 30 kHz while the filename says 100K.
3. **10 files are truncated**, worst being the `P/R_amp_5V_freq_160K` pair, each missing 22 scan rows in **both channels** — a sign that scan was aborted.

> For truncated files the code **drops the incomplete scan rows** rather than padding with NaN. Padding creates all-NaN rows, which make that row's median/MAD NaN and propagate into both detrending and the z-score (this exact bug produced `RuntimeWarning: All-NaN slice`). Truncated files are still analyzed, but carry the label `ok (TRUNCATED: lost N scan rows)` — neither silently skipped nor silently treated as normal.

## 7. Backend architecture

```
tools/crack_heatmap/
├── tdms_crack_scan.py          analysis backend (CLI + importable functions)
│   ├── parse_name()             filename -> measurement parameters (5 naming variants)
│   ├── load_scan()              TDMS -> 2D grid; warns on truncation/missing size
│   ├── apply_filter()           median | gaussian | none
│   ├── detrend()                row | col | both | poly | none  (criterion A1)
│   ├── _robust_z()              median/MAD z-score
│   ├── auto_border_margin()     measure + trim the noisy border (level + spread)
│   ├── detect_cracks()          z-score -> threshold -> merge -> boxes + score
│   ├── analyze_one()            one file -> one row of numbers (no figures)
│   ├── build_report_figure()    the 6 panels      } shared by CLI and GUI
│   ├── build_detect_figure()    the 3 panels      }
│   └── cmd_report / cmd_detect / cmd_sweep / cmd_compare
├── crack_scan_app.py           Tkinter desktop application
└── ai_advisor.py               the Claude API layer
```

CLI (installed and tested against the real data):

```bash
python tools/crack_heatmap/tdms_crack_scan.py report  DATA.tdms
python tools/crack_heatmap/tdms_crack_scan.py detect  DATA.tdms --drop-edge-boxes
python tools/crack_heatmap/tdms_crack_scan.py sweep   . --out sweep_all.csv
python tools/crack_heatmap/tdms_crack_scan.py compare "DATA/*.tdms" --boxes auto_boxes.json
```

Typical sequence when no labels exist yet: `detect` (produces `auto_boxes.json`) → `compare --boxes auto_boxes.json` to get a **real CNR** instead of the relative `dx_std` index.

## 8. Desktop application

`tools/crack_heatmap/crack_scan_app.py`, launched by double-clicking [`run_app.bat`](run_app.bat) or `python tools/crack_heatmap/crack_scan_app.py`. Tkinter ships with Python, so nothing beyond the backend libraries needs installing.

| Area | Contents |
|---|---|
| Left | `.tdms` files found recursively, with **Sensor / Amp / Freq / Plate** columns read from the filename. Files that do not match the convention show `-` but still open |
| Top right | Parameters: Filter, Kernel, Border crop (`auto` or px), Detrend, \|z\| threshold, Merge gap, Min area, Drop edge-touching boxes |
| Middle | Four tabs: **Figures** (embedded matplotlib with zoom/pan/save), **Results table**, **AI assistant**, **Log** |
| Bottom | Four action buttons + save figure / table CSV / boxes JSON |

The four actions map to the four CLI commands: *1 – Draw the 6 figures* = `report`, *2 – Locate cracks* = `detect`, *3 – Compare sensors* = `compare` (Ctrl/Shift to multi-select), *4 – Analyze ALL data* = `sweep` (with a progress bar).

### Two implementation traps worth remembering

- **Never read a Tk variable from a worker thread.** Heavy work runs on a background thread so the UI stays responsive, but calling `tk.StringVar.get()` off the main thread raises `RuntimeError: main thread is not in main loop`. `_params()` snapshots every parameter into a plain dict **on the main thread** before the worker starts. This only shows up at runtime, never by reading the code.
- **Pack the save-button bar before the canvas.** The canvas uses `expand=True`, so any widget packed after it is pushed off the bottom of the window.

### What the app does not do

The app's comparison table has no **CNR / CNR_abs** column (that needs a label file). For a real CNR, run `detect` to produce `auto_boxes.json`, then use the CLI: `compare --boxes auto_boxes.json`. The in-app table ranks by `dx_std` and states in the log that this is only a relative index.

## 9. The AI layer (Claude API)

`tools/crack_heatmap/ai_advisor.py` plus the **AI assistant** tab.

| | |
|---|---|
| **Input** | Numbers **computed by code** — a single measurement (conditions, crop, statistics, candidate list), a group of files (comparison table), or **the whole dataset** (101-row table + faulty/truncated file list + best-configuration table) — with the figure PNG and the user's question |
| **Output** | An interpretation, streamed chunk by chunk into the chat |
| **Model** | `claude-opus-5`, adaptive thinking, streaming via `client.messages.stream()` |

**The decisive design boundary: the model never computes the numbers.** It receives values the backend computed over the full data and does interpretation only. This is deliberate — a language model eyeballing statistics off raw arrays is an uncontrolled source of error, and the reader is making engineering decisions from the answer. The system prompt adds four constraints, each drawn from a trap actually hit in this project:

1. Never invent a number; if a quantity is missing, name the command that would produce it.
2. Never call the `detect` boxes ground truth, and never use them to compute a model's accuracy — that scores a detector against itself.
3. Diagnose in order: background → border artifacts → CNR → threshold (each step far cheaper than the next).
4. Always end with a "Limits" section.

The prompt also pre-teaches the **dipole** trap: when `CNR_abs` is far above the signed `CNR`, read `CNR_abs` and say so.

For the whole-dataset mode the context is about **4,100 tokens** (~US$0.02 per question at Opus 5 input rates). The table is sent **complete, all 101 rows**, together with the pre-computed aggregates — deliberately not asking the model to average the table itself and report the result as a measurement. The default question also varies by context (`scan` / `compare` / `sweep`): the whole-dataset mode asks about frequency trends, consistency across measurement sessions, broken measurements, and what to re-measure. Whole-dataset mode **sends no image** — the context is a hundred-row table, not one figure.

**API key:** prefer the `ANTHROPIC_API_KEY` environment variable (`setx ANTHROPIC_API_KEY "sk-ant-…"`). The "API key…" button keeps the key **in session memory only, never on disk** — this data folder gets zipped and shared, so no secret belongs in it.

API errors are translated into a sentence that says what to do (bad key / rate limit / no network / server error), caught as a chain from most specific to most general rather than one blanket `except`.

## 10. Deliberate limits

- **There is no trained AI crack model** (the probability heatmaps in `ai_heatmaps/*.png` came from earlier work). `detect` is a **statistical threshold** detector: it produces no probability map, learns nothing from data, and cannot separate a crack from an artifact of the same amplitude. A probability map requires training or loading a model, then evaluating it with `scan_metrics.py eval` from the `tdms-analysis` skill.
- **`auto_boxes.json` holds candidates, not ground truth.** Using it for relative sensor/frequency comparison is valid; using it as training labels or to publish an accuracy figure is not, because the detector would be grading itself. Real labels must still be digitized once from the specimen drawing.
- **Measured sensitivity** on the 15-crack coupon (3 rows × 5 columns): sensor R @200 kHz resolves rows 1–2 clearly, row 3 only as weak fragments — matching the fact that row 3 was also missed by the earlier AI model (see `ai_heatmaps/*.png`). This is a **physical limit of the measurement** at that frequency, not a flaw in the detection algorithm.
- **Calibration** (ADC scaling, `unit_string`) is not applied because these files declare none; the code reports amplitudes as `a.u.` rather than assuming a unit.
