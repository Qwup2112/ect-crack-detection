---
name: auto-analysis
description: Automatic analysis - run the whole ECT job yourself (pick the scan, crop the border, boost contrast, detect, measure, report) without stopping to ask between steps. Use when the user gives a goal rather than one step - "analyse file X", "process everything", "do it automatically", "final result", "detect cracks" on a scan not prepared yet, "which file is best", "draw a random file and find the cracks" (Vietnamese "phân tích file X", "xử lý hết", "kết quả sau cùng").
---

# Automatic analysis — doing the whole job

The user wants the result, not a menu of next steps. When a request implies several steps, do
all of them in the same answer and report once at the end. Ask only when the scan itself is
ambiguous (the PreToolUse hook then lists the candidates) or the request contradicts itself.

## Pipeline

1. **Scan.** Use the one the user named or the last scan of the chat. "Random" / "any file":
   pick a readable file yourself (list_scans) and say which. Read scan_info when the file may be
   truncated or has no plate size.
2. **Border.** crop_border mode=auto (skip it if this chat already cropped the scan). Look at
   the preview: if a bright band or the plate edge is still inside on one side, run it once more
   with z_thresh 2.5 (crops more). Never accept a crop that cuts into the 3 x 5 crack grid.
3. **Contrast** (for viewing only). Clear map: skip. A background slope hides cracks: detrend.
   Cracks still faint (deep row, low frequency, P sensor): strong. Decide from the preview, at
   most two tries, and keep the better one.
4. **Detect.** detect_cracks method=yolo. Then check it yourself:
   - the coupon has 15 cracks (3 rows x 5 columns): count the boxes per row and column;
   - boxes off the grid, or conf < 0.6, are suspect - say so;
   - fewer than 12 or more than 18 boxes: also run method=threshold and compare the two.
5. **Measure.** measure_crack on the strongest crack and on up to two suspicious ones, unless the
   user only asked where the cracks are.
6. **Beyond the tools** (trends over frequency or amplitude, statistics, a custom figure): use
   run_python (skill python-analysis) instead of estimating by eye.
7. **Report**, short:
   - the scan, and what was done (crop per side, contrast method);
   - N cracks, then `#n: X, Y mm, conf` (one line each, grouped by row when all 15 are there);
   - measurements, with their caveats (width includes the ~6 mm probe footprint);
   - warnings: draft labels, scan in YOLO's training set, threshold fallback, truncated file;
   - what the last figure on the right shows, in one sentence;
   - at most one optional next step, as an offer ("If you like, I can ..."), never a question
     that the user must answer before getting the result.

## Budget

A full run is 4-8 tool calls. Do not redraw a view you already drew. If the same tool fails
twice, stop, report what worked and what failed.

## Several scans

"Which file is best", "compare the frequencies": rank with compare_scans (z_top5, up to 8 at a
time), or run_python over the catalogue for more files, then run the pipeline above on the winner.
