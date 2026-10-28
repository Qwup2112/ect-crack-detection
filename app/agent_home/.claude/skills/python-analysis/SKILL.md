---
name: python-analysis
description: In-depth analysis with Python - how to use run_python for analysis the fixed tools cannot do (statistics over many scans, trends over frequency or amplitude, FFT, histograms, correlations, custom filters and plots). Load before the first run_python call of a chat.
---

# In-depth analysis with Python — run_python

Use the fixed tools for the standard figures (draw_scan, crop_border, enhance_contrast,
detect_cracks, measure_crack, compare_scans, profile_line). Write Python when the question
needs a number or a figure none of them produces. Never guess a number you could compute.

## What is preloaded

- `np`, `scipy`, `ndimage`, `signal`, `stats`, `plt` (matplotlib, Agg), and the app's modules
  `sc` (scan), `process`, `detect`, `render`.
- `load(scan) -> (grid, meta)`: raw grid (rows = Y, columns = X), `meta` has `id`, `session`,
  `sensor` ('R' amplitude / 'P' phase), `amp` and `freq` as strings (`float(meta["freq"])` kHz),
  `pitch_mm`, `size_x`, `size_y`, `warnings`.
- `stages(scan, crop=None) -> dict`: `raw`, `filtered` (lab LPF), `cropped`, `detrended`, `dx`,
  `zmap` (robust z), `margins`, `offset` = (left, top) of the cropped stages in raw px.
  crop=None uses the chat's crop choice for that scan, else 'auto'.
- `yolo(scan, conf=0.5)`, `threshold(scan, z=2.5)` -> boxes `{x0, y0, x1, y1, conf|score}` in
  ORIGINAL grid px; `labels(scan)` -> draft label boxes; `detect.iou`, `detect.match_labels`,
  `detect.measure_crack(img, box, pitch)`, `detect.cnr(img, box)`, `detect.z_top5(boxes)`.
- `catalog()` -> list of `{id, session, sensor, amp, freq, size_x, size_y, status}`; use
  only `status == "ok"` (or starting with "truncated", with a warning).
- `show(fig, title, caption)` -> the figure appears on the right panel. Titles and axis labels
  in English.
- `chat` -> this chat's state (crop / enhance choices, last scan, last detections).
- The working directory is the chat's work folder: save intermediate files there
  (`np.save("x.npy", a)`), they stay for the rest of the chat.

## Rules

- Coordinates: mm = px * `meta["pitch_mm"]`. Positions you report are in the original grid frame.
- R and P have different scales (~0.025 vs ~-30): compare sensors with robust z or z_top5,
  never raw values.
- Print a compact result (at most ~30 lines; a short table is fine) and report only numbers
  that were printed. Plots: axes in mm, a colour bar on maps (`extent=` with pitch), labelled axes.
- One run_python call per question when possible. On an error, fix the code and retry, at most
  twice; then tell the user what failed.
- Files can be written only in the work folder; data/ and results/labels/ are read-only, and
  network and subprocesses are blocked. Never try to work around the guard.
- If the tool reports too little memory, say so and continue with the fixed tools.

## Patterns

Trend over frequency for one session and sensor:

```python
rows = []
for e in catalog():
    if e["session"] == "cambien_4lop_2" and e.get("sensor") == "R" and e["status"] == "ok":
        st = stages(e["id"])
        b, _ = detect.threshold_detect(st["cropped"], st["offset"])
        rows.append((float(e["freq"]), float(e["amp"]), detect.z_top5(b), len(yolo(e["id"]))))
rows.sort()
for r in rows: print(r)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot([r[0] for r in rows], [r[2] for r in rows], "o-")
ax.set_xlabel("Frequency (kHz)"); ax.set_ylabel("z_top5")
show(fig, "z_top5 versus frequency - cambien_4lop_2 R", "crack contrast against background noise")
```

A map in mm with the detected boxes:

```python
grid, meta = load("R 1.7V 300k"); st = stages(meta["id"]); p = meta["pitch_mm"]
fig, ax = plt.subplots(figsize=(8, 5))
render.draw_map(ax, st["detrended"], p, st["offset"], "magma", title="Background removed")
render.draw_boxes(ax, yolo(meta["id"]), p)
show(fig, "Background-removed map + YOLO boxes")
```
