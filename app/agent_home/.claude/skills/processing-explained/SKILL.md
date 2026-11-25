---
name: processing-explained
description: Explaining the processing in detail, with measured benchmark numbers - how a scan becomes a 2D or 3D map, the order of the steps, denoising, border crop, background removal, detection. Use when the user asks how or why a step works, what a method does, which option is better, for the benchmark, or for a 3D view (Vietnamese "giải thích", "tại sao", "như thế nào", "trình tự", "các bước", "benchmark", "so sánh phương pháp", "vẽ 3D", "nổi khối").
---

# Processing explained — how and why, with evidence

## How to answer

1. **Show first.** Draw the one figure that shows the step (table below), then explain it.
2. **Explain the step**: what goes in, what comes out, why it is needed, its parameters.
3. **Back the choice with the benchmark** below: the default, what the alternatives gave, and
   what that means. Quote numbers exactly as written here; do not invent others.
4. **Say the caveats** that apply (end of this file), once, briefly.
5. Stay on the step(s) asked about. Offer at most one next step (e.g. the 3D view, the next step
   of the sequence).

| the user asks about | draw |
|---|---|
| the whole sequence, "how does a scan become a heatmap" ("trình tự", "các bước") | `draw_scan view=steps` |
| filtering ("lọc nhiễu") | `draw_scan view=raw`, then `view=filtered` |
| the border ("cắt viền") | `crop_border mode=auto` |
| background removal ("trừ nền") | `draw_scan view=detrended` |
| the crack dipole | `draw_scan view=dx` |
| 3D, "surface", "relief" ("vẽ 3D", "nổi khối") | `draw_scan plot=3d`, view `detrended` or `dx` (see below) |
| what YOLO sees | `draw_scan view=model_input` |
| detection | `detect_cracks` (skill crack-detection) |
| the benchmark charts or full tables | `run_python` (below) |

Use the scan the user is working on; with none, use cambien_4lop_2 R 1.7 V 300 kHz (strong
signal, 15 labelled cracks).

## The sequence, step by step (app defaults)

1. **Read the TDMS file.** Group `Freq_Sampling_SizeX_SizeY` holds [frequency Hz, sampling,
   points per line X, number of lines Y]; group `Waveform` holds X·Y values, one per scan point,
   in recording order. A truncated file keeps only its complete lines (never padded).
2. **Build the grid.** The probe scans zig-zag: line 1 left→right, line 2 right→left, ... Cut
   into lines, every 2nd line is backwards, so cracks smear into a comb. Reverse those lines (the
   lab's `reshape_raster`). The probe also lags when it turns: shift the return lines by the lag
   (searched in −5…+5 px) that minimises the row-to-row zig-zag, and match their level by medians.
3. **Filter (lpf).** Hamming-window FIR low-pass, order 24, cutoff 0.3 of Nyquist, run down
   every column (along Y, where the 2-line zig-zag lives), edges padded 50 points, group delay
   removed; then a 3×3 median for spikes. Along X only the median acts, so crack edges stay sharp.
4. **Crop the border (auto).** Per edge, walk inward while a row/column is abnormal in level
   (median) or spread (std) as a robust z-score > 3; stop at the first normal line; at most 15 %
   per side. Measured on the filtered map. Asymmetric on purpose (e.g. top 10, bottom 6, left 5,
   right 12 px on cambien_4lop_2 R 1.7 V 300 kHz).
5. **Remove the background.** Subtract an 8 px Gaussian blur: wider than a crack signature
   (~16 px boxes), narrower than the crack spacing (25 px), so plate-scale trends go and each
   crack stays one blob.
6. **Derived maps.** dx = d/dx of the cropped map (a crack shows as a +/− dipole); zmap = robust
   z-score (median/MAD) of the background-removed map.
7. **Detect.** YOLO11n (fine-tuned on the labelled scans, run on onnxruntime) reads its own
   3-channel image of the *whole* grid: R = lpf + background removed, G = d/dx, B = d/dy, each
   robustly scaled; letterboxed to 320 px; confidence 0.5; boxes on the scan border dropped. The
   fallback is the threshold detector: |robust z| > 2.5 on the cropped, background-removed map,
   fragments within 2 px joined, boxes touching the crop edge dropped. Boxes are in original
   grid pixels everywhere.
8. **Draw.** Axes in mm (pitch = plate size from the filename / number of points); colours
   clipped at the 2nd–98th percentile (±99th |value| for the diverging dx / zmap maps) so a few
   spikes cannot wash out the cracks; bilinear display. 2D heatmap or 3D surface.

## Benchmark evidence

From `python -m app.tools.benchmark`, run 2026-10-28: 103 readable scans, 55 of them labelled
(cam_4lop_1 and cambien_4lop_2, 825 DRAFT-labelled cracks). Scores against labels: a box is found
when it overlaps a label at IoU ≥ 0.3; "centre recall" counts a crack as seen when any box centre
falls inside its labelled box.

**Grid** (zig-zag left, all scans, relative to the recorded order):
recorded 100 % → every 2nd line reversed 26 % → + lag fix 12 %.
- Lag found: cam_4lop_1 +2 px on 25 of 28 scans; cambien_4lop_2 mostly 0 or +2 px (−1…+2);
  early_session spread from −5 to +5 px.
- 20 early_session scans sit on the +5 px search limit. With a ±8 px search, 19 of them pick +6 px
  and their zig-zag drops to a median 70 % (best 50 %) of the ±5 px result; no labelled scan
  changes. The app still searches ±5 px: say so if the user asks about early_session quality.
- The threshold detector cannot tell the grids apart on this coupon (F1 0.42 recorded vs 0.41
  aligned): the cracks sit in evenly spaced columns (x = 25, 50, 75, 100, 125 px on a 155 px
  grid), so a crack on a backwards line lands ~3 px from another crack column. Show the smear
  with `view=steps` (panel 3 vs 4) instead.

**Denoising** (zig-zag left / crack peak kept / threshold-detector F1):

| filter | zig-zag left | crack peak kept | F1 |
|---|---|---|---|
| none | 100 % | 100 % | 0.42 |
| median 3×3 | 56 % | 92 % | 0.41 |
| Gaussian σ 1.5 | 16 % | 78 % | 0.38 |
| FIR down the columns only | 21 % | 96 % | 0.43 |
| FIR + median 3 (default lpf) | 22 % | 91 % | 0.41 |

Meaning: filtering down the columns removes ~80 % of the zig-zag and keeps 91–96 % of the crack
peak; the Gaussian removes slightly more zig-zag but costs 22 % of the peak because it also blurs
along X. No filter changes threshold-detector F1 much (0.38–0.43). The filter's job is a clean map
without stripes, and it is the input YOLO was trained on - do not claim it improves detection
scores.

**Border crop** (on the lpf map):

| crop | area removed (median / worst) | labelled cracks cut | threshold F1 |
|---|---|---|---|
| none | 0 % | 0 | 0.20 |
| fixed 5 px | 17 % / 20 % | 0 | 0.39 |
| fixed 10 px | 32 % / 38 % | 0 | 0.37 |
| auto, z > 2 | 27 % / 45 % | 5 | 0.38 |
| auto, z > 3 (default) | 21 % / 43 % | 0 | 0.41 |
| auto, z > 4 | 17 % / 41 % | 0 | 0.40 |

Meaning: cropping doubles threshold F1 (0.20 → 0.41), because border streaks look like cracks.
Auto z > 3 scores best and cuts no labelled crack; z > 2 cuts 5. YOLO is not affected by the crop
(it reads the whole grid and drops border boxes itself).

**Order and background removal** (threshold-detector F1):
- Filter then crop (default) 0.41 vs crop then filter 0.41: the order hardly matters; cropping
  first removes slightly less border (median 23 vs 26 px over the four sides).
- Filter × crop: the crop decides (no crop 0.15–0.23, fixed 5 px 0.34–0.39, auto 0.38–0.42 for
  any filter).
- Background removal: 8 px Gaussian (default) 0.41; rows then columns 0.29; row medians 0.25;
  none 0.15; column medians 0.13. Likely reason: each row and column crosses several cracks (3 × 5
  layout), so a row or column median partly follows the cracks instead of the plate.

**Detection** (all 55 labelled scans):

| detector | precision | recall | F1 | mean IoU | centre recall | ms/scan |
|---|---|---|---|---|---|---|
| YOLO conf 0.25 / 0.5 (default) / 0.7 | 1.00 | 1.00 | 1.00 | 0.90 | 1.00 | ~13 |
| threshold \|z\| > 2 | 0.49 | 0.27 | 0.35 | 0.56 | 0.45 | ~4 |
| threshold \|z\| > 2.5 (fallback) | 0.60 | 0.31 | 0.41 | 0.57 | 0.46 | ~4 |
| threshold \|z\| > 3 | 0.64 | 0.30 | 0.41 | 0.58 | 0.43 | ~4 |

Meaning: YOLO finds all 825 labelled cracks with no false box at any of the three thresholds -
but it was trained on these same scans, so this is how well it fits them, not how it would do on
a new plate. On the 48 unlabelled scans it gives a median 11 boxes per scan (max 15) at 0.5, 13 at
0.25, 8 at 0.7 - no ground truth there. The threshold detector is a model-free cross-check: its
boxes cover only the strongest part of a signature, so they often miss the IoU 0.3 bar even when
they sit on the crack.

**2D vs 3D drawing** (median of 8 scans): 2D ~210–230 ms, 3D ~230–255 ms per figure, both
~200 KB. Use them for different things:
- 2D heatmap: exact positions, box edges, mm measurements - the default for analysis.
- 3D surface (height and colour = value): shows at a glance how strong and how sharp each crack
  is compared with the others, the background slope, and the +/− dipole (on `dx`). Peaks hide
  what is behind them and perspective distorts positions: use it to explain, never to measure.
  Best on `detrended` or `dx`; on `amplitude` the background slope dominates.

## Showing the benchmark itself

The full report, tables and charts are in `results/benchmark/` (REPORT.md, benchmark.json, one
CSV per step, figures/A_grid.png, B_denoise.png, C_crop.png, D_pipeline.png, E_detect.png). Show a
chart with run_python:

```python
bench = sc.DATA_DIR.parent / "results" / "benchmark"
img = plt.imread(bench / "figures" / "C_crop.png")
fig, ax = plt.subplots(figsize=(12, 3.2)); ax.imshow(img); ax.axis("off")
show(fig, "Benchmark: border crop", "results/benchmark/figures/C_crop.png")
```

Read numbers from `benchmark.json` the same way (`json.loads(...)`) when the user asks for one
not listed here. Re-running the benchmark is a developer command
(`.venv\Scripts\python -m app.tools.benchmark`, ~40 s); run_python cannot run it (it writes
outside the work folder) - tell the user how.

## Caveats

- The labels are DRAFT (fitted to the 3 × 5 coupon layout, not reviewed): every score against
  them is provisional.
- YOLO was trained on all labelled scans; its perfect scores are a fit, not a validation.
- The threshold detector is crude (recall ~0.3); use it to compare pipeline variants with each
  other, not as the app's detection quality.
- Times were measured on this PC and only compare the variants with each other.
