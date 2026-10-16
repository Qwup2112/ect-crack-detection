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
python tools/crack_heatmap/tdms_crack_scan.py report data/cambien_4lop_2/4lop_R_amp_1.3V_fre_200k_lf_1mm_145x95mm.tdms

# Locate cracks + trim the noisy scan border automatically
python tools/crack_heatmap/tdms_crack_scan.py detect data/<session>/<file>.tdms --drop-edge-boxes

# Analyze every .tdms under data/ -> results/sweep_all.csv
python tools/crack_heatmap/tdms_crack_scan.py sweep

# Compare sensors/frequencies, with a real CNR when labels are supplied
python tools/crack_heatmap/tdms_crack_scan.py compare "data/cambien_4lop_2/*.tdms" --boxes results/cambien_4lop_2/<file>_results/auto_boxes.json
```

Or run the desktop application — double-click **`run_app.bat`**, or:

```bash
python tools/crack_heatmap/crack_scan_app.py
```

The app opens on `data/` with the scans grouped by session. For the selected file:
**Draw figures** (or double-click), **R + P side by side**, **Locate cracks**; for the whole
dataset: **Analyze & rank all files** (results in the *Results* tab, with CSV/JSON export).
Processing settings sit folded under **Settings** as a one-line summary; *Defaults* restores
the tested pipeline. Figures save from the disk icon on the plot toolbar (into `results/`).

In the **Figures** tab each map opens on its own, full size; the view bar (or the ← / →
keys) switches maps. On the map on screen:

- **Profile 1D** — a row at a given *Y* or a column at a given *X* (mm): the map with the
  cut line, and the signal along it.
- **Point value** — the value at (*X*, *Y*) in mm, marked on the map.
  Positions off the map are rejected with the valid range, never clamped to the edge.

**R + P side by side** shows the same map for the R (amplitude) and P (phase) file of one
measurement side by side, whichever of the two is selected. Both are cropped with ONE shared
set of margins (the larger per side), so the same X, Y in mm is the same spot on the plate
in both; profile and point value then give R and P together. Each keeps its own colour
scale, since amplitude and phase are not comparable. All 51 R/P pairs share a grid size.

### Labelling cracks and the YOLO model

**LABEL row (Figures tab)** — turn on *✎ Draw boxes*, then on any single map: drag to add
a box, click a box to select it and see its analysis (size in mm, value range, CNR and
CNR_abs against the surrounding ring, dipole flag), right-click or Delete to remove it.
*Auto-fill* loads the threshold detector's interior boxes as a starting point — it misses
cracks touching the crop edge and adds false ones, so correct them. *Copy to session* copies
the boxes to every unlabelled scan of the same session with the same grid (same plate
position: all frequencies, amplitudes, both sensors). Labels are saved as you go in
`results/labels/<session>/<file>.json`, in original grid pixels, so they stay right on the
cropped and uncropped maps alike. The file list shows each scan's label count.

Labels are drawn only while *✎ Draw boxes* is on; drawn over every detection they made the
map unreadable. **Detect with model** instead colours its boxes against the labels: green =
matches a label, red = false alarm, pink dotted = a label the model missed.

**Click a detected box to measure the crack** (Locate cracks and Detect with model maps). The
size is that of the crack *signature*, not of the box: a tilted plane through the box border
is removed as background, the peak is taken in the box centre, and the pixels above half of
it (FWHM) that join the peak are measured along their principal axes: **length** (long axis),
**width** (short axis), orientation, area, centre, peak and SNR. The two axes are drawn on
the map, in cyan, or in orange with a warning when the signature is weak (SNR < 3) or fills
most of the box. When it runs past the box edge the length is shown as a minimum (`≥`). The
width is the width of the *signal*, probe footprint included (about 6 mm here), so a real
crack is far narrower; depth needs a calibration and is not estimated.

**Locate cracks** drops every candidate touching the crop edge (plate edge / scan border
artifacts); only interior candidates are drawn and listed.

**Train crack model (YOLO)** fine-tunes a COCO-pretrained YOLO11n (`pip install
ultralytics`) on the labelled scans. It trains on the NVIDIA GPU when PyTorch is a CUDA
build (`python -c "import torch; print(torch.cuda.is_available())"` prints `True`; else
install one from pytorch.org), otherwise on the CPU; `set CRACK_DEVICE=cpu` forces the CPU.
80 epochs take about 1.5 minutes on an RTX 3050 laptop GPU. The model sees a 3-channel image: detrended signal,
d/dx, d/dy. Validation holds out whole *sessions*: scans of one session share plate
placements and, after *Copy to session*, the same boxes, so a random split would reward
memorising positions (grid size is no safe key: truncated files share a placement, and one
grid size held two placements). With one session labelled the score is reported as optimistic.
The app asks which run: **Yes** = train on ALL labelled scans (the final model, for
detecting; its score is on training data), **No** = hold out a session (an honest score).

**Detect with model** draws the model's boxes on EVERY map of the scan: the six standard
views (Raw, Filtered, Cropped, Amplitude, dA/dx, -dA/dx) and the model's own input
(detrended signal), each shifted by its crop offset; a box inside a cropped-off border is
left off that map. It counts found / false / missed against the labels (IoU ≥ 0.3), flagging
scans that were in the training set. Boxes below confidence 0.5 or touching the scan border
(the plate edge) are dropped: on the held-out session 0.5 still finds 450/450 cracks, while
on pure-noise images it cuts the false boxes from 65 to 17 per image.
Command line: `python tools/crack_heatmap/crack_model.py export | train [--all] | predict <file>`.

Labels must come from a person: a model trained on unedited Auto-fill boxes only learns to
copy the threshold detector, and scoring it against them grades the detector against itself.

**Current labels are DRAFT.** The 56 files in `results/labels/` (cam_4lop_1 and
cambien_4lop_2, 15 boxes each) were produced by fitting the 3 × 5 coupon layout to the R scan
of each measurement; review them in the app before trusting any score. Unlabelled: the
cam_4lop_1 4 V 10 kHz pair (faint signal, a different grid) and `early_session`.

2-fold cross-validation, one whole session held out per fold, 80 epochs, IoU ≥ 0.3, scored
against the draft labels:

| held-out session | scans | YOLO found / false / missed | YOLO P / R | threshold detector P / R |
|---|---|---|---|---|
| cam_4lop_1 | 26 | 390 / 3 / 0 | 0.99 / 1.00 | 0.46 / 0.30 |
| cambien_4lop_2 | 30 | 450 / 15 / 0 | 0.97 / 1.00 | 0.45 / 0.35 |

(Measured at confidence 0.25 with border boxes kept. With the current defaults, confidence
0.5 and border boxes dropped, the cambien_4lop_2 fold gives 450 / 0 / 0.)

`results/model/crack_yolo.pt` is the final model, trained on all 56 labelled scans. Run on
every readable scan: 15 cracks on each of the 58 cam_4lop_1 / cambien_4lop_2 files
(including the unlabelled 10 kHz pair, whose faint dipoles it finds), and 3-13 (median 8) on
the 43 `early_session` files, a placement it never saw. Early_session is unlabelled, so those
counts are not checked; the one inspected (R 4 V 60 kHz) shows 8 boxes on the 8 visible
dipoles. The model still fires on random texture (17 boxes at 0.5 on noise), because every
training image contains cracks: a scan with NO crack would give false boxes. Both sessions
use the same coupon, so this shows the model generalises across placements, sensors and
frequencies, not to other specimens; and the draft boxes are layout-shaped, which favours
YOLO over the detector's free-shaped boxes in the IoU match.

### Raw dataset, detected automatically

**Detect all → raw dataset** (or `python tools/crack_heatmap/crack_dataset.py`) runs the
model on every scan and writes a separate dataset to `results/dataset_raw/`:

| path | content |
|---|---|
| `images/<session>/<file>.png` | raw map (viridis, 2–98 % scale), 4 px per scan point |
| `arrays/<session>/<file>.npy` | the raw values behind the image |
| `labels/<session>/<file>.txt` | the model's boxes, YOLO format `0 cx cy w h conf` |
| `annotated/<session>/<file>.png` | raw map with the boxes, confidence and mm axes |
| `detections.csv` | one row per crack: position, box size, length, width, SNR, conf, match with a label |
| `files.csv` | one row per scan: cracks found, mean confidence, found/false/missed vs labels |

"Raw" means unfiltered, uncropped and not detrended, but with the serpentine lines
un-mirrored and the scan-direction lag fixed (otherwise each crack is a zig-zag). The model
still sees its usual processed input; its boxes are in scan pixels and land on the raw map
1:1. On the raw map the cracks are faint next to the plate's background slope, which is why
the model and the length/width measurement work on the processed signal. Whole run: 101
scans, 1176 cracks, about a minute on the GPU; the 9 template/empty files are listed as
skipped. These labels are the model's output, **not ground truth**: do not train on them
unchecked.

Colour scales span the 2nd–98th percentile (±99th for derivative/z maps), so a few extreme
pixels cannot wash out the cracks; arrow ends on the colorbar mean values beyond it.

## What the tools produce

**`report`** — six panels: Raw (before filter), Filtered, Cropped, Amplitude heatmap,
X-Derivative (`dA/dx`), and X-Derivative Inverse (`-dA/dx`). The inverse view matters
because an ECT crack signature is a *dipole*: one edge gives a positive lobe, the other
negative, and which is which depends on sensor and frequency.

**`detect`** — measures the noisy scan border and trims it *asymmetrically* (measured on
the R 1.3 V 200 kHz file: top 14 px, bottom 3 px, left 2 px, right 7 px — an even 5 px crop
both misses noise at the top and discards good data at the bottom), then locates crack
candidates by robust z-score, merging fragments that belong to the same crack.

**`sweep`** — runs the identical pipeline over every file and ranks by `z_top5`, computed
from *interior* candidates only (`n_inner`). Edge-touching boxes are excluded from the
ranking: on the `early_session/P_*` scans the unremoved plate edge merges into one image-sized box
with |z| up to 26, which otherwise tops the table.

## Preprocessing

- **Serpentine raster.** The scanner runs back and forth, so odd scan lines are stored
  right-to-left. `load_scan` un-mirrors them (same mapping as `dataloader.reshape_raster`);
  without it every other row is flipped and each crack shows as a zig-zag. Verified on all
  101 readable files.
- **Scan-direction fix (the "sawtooth").** Return lines lag the probe by a fixed number of
  pixels (cam_4lop_1: 2 px on 26/28 files) and carry a small level offset. `load_scan`
  picks the shift that minimizes the row-to-row zig-zag and matches the levels: 50-60% of
  the sawtooth removed. The shift is stored as `row_shift_px`.
- **Smoothing (`--filter lpf`, the default)** runs `dataloader.LPF` (Hamming FIR, cutoff 0.3)
  down each column, then a 3 px median: sawtooth down to 0.33x of the original on the R
  200 kHz scan, with no loss of detected cracks over the whole dataset. Figures are drawn
  with bilinear interpolation, as in `plot_all_1Hall_heatmaps.py`.
- **Background removal (`--detrend bg`, the default)** uses
  `dataloader.remove_background_trend` — subtract an 8 px Gaussian blur. It is the only
  option that gives one box per crack on the 3 × 5 grid; the old `row` default merged
  cracks sharing a column into one tall box. Also available from `dataloader.py`:
  `--detrend surface` (`detrend_2d_surface`), `--detrend linear` (`detrend_1d`) and
  `--filter wavelet` (`wavelet_denoise`, no gain over median on this data). Only these
  three import `dataloader.py`, which needs scikit-learn, h5py and PyWavelets. The default
  `lpf` and `bg` are the same dataloader functions written out with scipy alone (identical
  output, checked on three scans), because importing `dataloader.py` loads scikit-learn,
  and Windows **Smart App Control** blocks scikit-learn's compiled files on this PC
  (`DLL load failed … An Application Control policy has blocked this file`).
- The time-domain functions in `dataloader.py` / `plot_all_1Hall_heatmaps.py`
  (`LPF_2D`, `read_scan_tdms_feedback`, slice/LOI/peak features) do **not** apply: every
  file here stores one value per pixel, with no waveform to filter.

**`compare`** — sensor/frequency table with a real CNR when a box file is supplied.

## Two results worth knowing before you read any table

**Rank on `z_top5` or `CNR_abs`, never on `dx_std`.** `z_top5` is normalized by each
image's own background noise (median/MAD), so it compares fairly between sensor R
(~0.01 scale) and sensor P (~30 scale). `dx_std` merely rewards a large value scale — on
this dataset it ranks P far above R, while both `z_top5` and `CNR_abs` rank R above P.

**Read `CNR_abs`, not the signed `CNR`, when they disagree.** Because the crack signature
is a dipole, a box covering both lobes averages to the background and the signed CNR
collapses toward zero even when the signal is obvious (measured: R 4V 200 kHz gives signed
CNR 1.60 but `CNR_abs` 2.63). `CNR_abs >> CNR` is the fingerprint of a dipole.

## Best configuration found

Default pipeline (direction fix, `lpf` filter, auto crop, `bg` detrend), interior candidates only:

| folder | sensor | best configuration | z_top5 |
|---|---|---|---|
| cambien_4lop_2 | R | 1.7 V, 300 kHz | 10.35 |
| cambien_4lop_2 | P | 0.3 V, 20 kHz | 9.17 |
| cam_4lop_1 | R | 4 V, 250 kHz | 8.27 |
| cam_4lop_1 | P | 4 V, 40 kHz | 7.21 |
| early_session | R | 4 V, 60 kHz | 3.67 |
| early_session | P | 3 V, 30 kHz | 2.85 |

These are not comparable with the values in earlier versions of this file (15.61 etc.),
which were measured on mirrored rows and included edge artifacts in the ranking.

## Data faults found by the full sweep

- **18 files use a different naming convention** (`freq_20K` vs `fre_20k`, `liffoff` vs
  `lf`, a `_tron` suffix). Any parser that does not cover all five variants silently loses
  their measurement parameters.
- **`data/early_session/R_amp_5V_freq_100K_liffoff_1mm.tdms` is empty** — 230 bytes where its siblings
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

See [`docs/AGENT_DESIGN_crack_heatmap.md`](docs/AGENT_DESIGN_crack_heatmap.md) for the full design
rationale.

## Layout

Raw measurements and generated output are kept apart: `data/` holds only `.tdms` scans,
and everything the tools produce goes to the matching folder under `results/`
(`detect`/`report` write `results/<session>/<file>_results/`).

```
data/                     raw scans only (.tdms)
  cam_4lop_1/             scan session 1
  cambien_4lop_2/         scan session 2
  early_session/          earliest session, mixed naming conventions
results/                  generated output, same session folders as data/
  sweep_all.csv           whole-dataset table from `sweep`
  cam_4lop_1/             heatmaps/, plot/, plot_1_P/, plot_1_R/, plots/, ai_heatmaps/
  cambien_4lop_2/         <file>_results/, plots/, ai_heatmaps/
scripts/
  dataloader.py           lab filters/loaders (used by the `bg`/`surface`/`wavelet` options)
  plot_all_1Hall_heatmaps.py  1Hall waveform heatmaps (server paths, waveform-format files)
tools/crack_heatmap/
  tdms_crack_scan.py      analysis backend (CLI + importable functions)
  crack_scan_app.py       Tkinter desktop application
  ai_advisor.py           Claude API layer
  crack_labels.py         hand labels, crack measurement, model input image
  crack_model.py          YOLO training / detection
  crack_dataset.py        raw-map dataset of every scan, detected by the model
docs/                     design notes
```
