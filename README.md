# ECT Crack Detection — TDMS scan analysis

Tools for analysing 2D raster eddy-current-testing (ECT) scans stored as National
Instruments `.tdms` files, locating crack candidates, and comparing measurement
configurations across a whole dataset.

The specimen is a calibration coupon with 15 artificial cracks in a 3 × 5 grid at
varying depths. Scans vary by sensor channel (R = amplitude, P = phase), excitation
amplitude, and frequency.

## Requirements

Python 3.9 or newer. On Linux the desktop app also needs Tk (`sudo apt install python3-tk`);
on Windows and macOS it ships with Python. The command-line tools do not need Tk.

## Quick start

```bash
pip install -r requirements.txt   # or: pip install npTDMS numpy scipy matplotlib pandas anthropic

# One file -> the 6 standard views
python tools/crack_heatmap/tdms_crack_scan.py report cambien_4lop_2/4lop_R_amp_1.3V_fre_200k_lf_1mm_145x95mm.tdms

# Locate cracks + trim the noisy scan border automatically
python tools/crack_heatmap/tdms_crack_scan.py detect <file>.tdms --drop-edge-boxes

# Analyze every .tdms in the tree (110 files in ~9 s)
python tools/crack_heatmap/tdms_crack_scan.py sweep . --out sweep_all.csv

# Compare sensors/frequencies, with a real CNR when labels are supplied
python tools/crack_heatmap/tdms_crack_scan.py compare "DATA/*.tdms" --boxes auto_boxes.json
```

Or run the desktop application — double-click **`run_app.bat`**, or:

```bash
python tools/crack_heatmap/crack_scan_app.py
```

## What the tools produce

**`report`** — six panels: Raw (before filter), Filtered, Cropped, Amplitude heatmap,
X-Derivative (`dA/dx`), and X-Derivative Inverse (`-dA/dx`). The inverse view matters
because an ECT crack signature is a *dipole*: one edge gives a positive lobe, the other
negative, and which is which depends on sensor and frequency.

**`detect`** — measures the noisy scan border and trims it *asymmetrically* (measured on
the R 200 kHz file: top 14 px, bottom 3 px, left 7 px, right 4 px — an even 5 px crop both
misses noise at the top and discards good data at the bottom), then locates crack
candidates by robust z-score, merging fragments that belong to the same crack.

**`sweep`** — runs the identical pipeline over every file and ranks by `z_top5`.

**`compare`** — sensor/frequency table with a real CNR when a box file is supplied.

## Two results worth knowing before you read any table

**Rank on `z_top5` or `CNR_abs`, never on `dx_std`.** `z_top5` is normalized by each
image's own background noise (median/MAD), so it compares fairly between sensor R
(~0.01 scale) and sensor P (~30 scale). `dx_std` merely rewards a large value scale — on
this dataset it ranks P far above R, while both `z_top5` and `CNR_abs` rank R above P.

**Read `CNR_abs`, not the signed `CNR`, when they disagree.** Because the crack signature
is a dipole, a box covering both lobes averages to the background and the signed CNR
collapses toward zero even when the signal is obvious (measured: R 4V 200 kHz gives signed
CNR 0.85 but `CNR_abs` 1.94). `CNR_abs >> CNR` is the fingerprint of a dipole.

## Best configuration found

| folder | sensor | best configuration | z_top5 |
|---|---|---|---|
| cambien_4lop_2 | R | 4 V, 200 kHz | **15.61** |
| cam_4lop_1 | R | 4 V, 160 kHz | 14.18 |
| data | R | 4 V, 200 kHz | 8.35 |
| cambien_4lop_2 | P | 0.19 V, 20 kHz | 8.67 |

## Data faults found by the full sweep

- **18 files use a different naming convention** (`freq_20K` vs `fre_20k`, `liffoff` vs
  `lf`, a `_tron` suffix). Any parser that does not cover all five variants silently loses
  their measurement parameters.
- **`data/R_amp_5V_freq_100K_liffoff_1mm.tdms` is empty** — 230 bytes where its siblings
  are ~109 KB, with no `Waveform` group: that measurement was never written. Its internal
  metadata also says 30 kHz while the filename says 100K.
- **10 files are truncated**, worst being the `P/R_amp_5V_freq_160K` pair, each missing 22
  scan rows in *both* channels. These are analysed on their complete rows only and
  labelled `ok (TRUNCATED: lost N scan rows)` — never silently padded or dropped.

## AI assistant

The desktop app has an **AI assistant** tab that sends the *computed* numbers plus the
figure to `claude-opus-5` and streams back an interpretation. The model never derives
statistics from raw data itself — it only interprets values the backend computed over the
full dataset.

Set a key first:

```bash
setx ANTHROPIC_API_KEY "sk-ant-..."
```

The key is read from the environment, or can be entered in the app for the current session
only. It is never written to disk.

## Limits

`detect` is a **statistical threshold detector**, not a machine-learning model. It marks
"abnormal deviation from the background" — whether that is a crack, a plate edge, or a scan
artifact must be confirmed against the specimen drawing. The boxes it produces are
*candidates*, valid for relative sensor comparison but **not** ground truth: using them to
score a detector means grading the detector against itself.

See [`AGENT_DESIGN_crack_heatmap.md`](AGENT_DESIGN_crack_heatmap.md) for the full design
rationale.

## Layout

```
tools/crack_heatmap/
  tdms_crack_scan.py   analysis backend (CLI + importable functions)
  crack_scan_app.py    Tkinter desktop application
  ai_advisor.py        Claude API layer
cambien_4lop_2/        scan session 2 (.tdms + generated plots)
cam_4lop_1/            scan session 1
data/                  earlier session, mixed naming conventions
```
