"""Benchmark of the processing steps: what each choice in the pipeline does to the result.

    .venv\\Scripts\\python -m app.tools.benchmark [--quick]

Each step is run with its alternatives, the app's default marked:
  A. grid     - recording cut into lines / every 2nd line reversed / + return-line lag fixed
  B. denoise  - none, median 3x3, Gaussian, FIR down the columns, FIR + median (lpf, default)
  C. crop     - none, fixed 5 / 10 px, auto with z 2 / 3 (default) / 4
  D. order    - filter x crop combinations, crop before filter, other background removals
  E. detect   - YOLO at confidence 0.25 / 0.5 (default) / 0.7, threshold |z| 2 / 2.5 / 3
  F. drawing  - 2D heatmap vs 3D surface: time and file size
Ground truth is the DRAFT hand labels (results/labels/). The YOLO model was trained on all
labelled scans, so its scores show how well it fits them, not how it would do on a new plate.
Writes results/benchmark/: REPORT.md, benchmark.json, one CSV per step, figures/.
"""
import argparse
import csv
import datetime
import json
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np                                            # noqa: E402

from app.backend.ect import detect, process, render, scan as sc   # noqa: E402  (render selects Agg)
from app.backend.ect.paths import RESULTS_DIR, PROJECT_ROOT       # noqa: E402
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.patches import Rectangle                           # noqa: E402
from matplotlib.ticker import PercentFormatter                     # noqa: E402

OUT = RESULTS_DIR / "benchmark"
FIG = OUT / "figures"
REF_SCAN = "cambien_4lop_2 R 1.7V 300k"        # the smoke test's scan: strong, 15 labelled cracks
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"    # categorical slots 1-3 (dataviz palette)
GREY = "#c3c2b7"
INK, MUTED = render.INK, render.MUTED

GRID_STEPS = [("recorded", "cut into lines (every 2nd backwards)"),
              ("unmirrored", "every 2nd line reversed"),
              ("aligned", "+ lag fixed, search ±5 px (default)"),
              ("aligned8", "+ lag fixed, search ±8 px")]
FILTERS = [("none", "none", 3), ("median 3x3", "median", 3), ("Gaussian σ 1.5", "gaussian", 3),
           ("FIR down columns", "lpf", 1), ("FIR + median 3 (default)", "lpf", 3)]
CROPS = [("none", "none", 3.0), ("fixed 5 px", 5, 3.0), ("fixed 10 px", 10, 3.0),
         ("auto, z > 2", "auto", 2.0), ("auto, z > 3 (default)", "auto", 3.0), ("auto, z > 4", "auto", 4.0)]
COMBO_FILTERS = [("none", "none", 3), ("median", "median", 3), ("Gaussian", "gaussian", 3), ("lpf", "lpf", 3)]
COMBO_CROPS = [("none", "none"), ("fixed 5 px", 5), ("auto z>3", "auto")]
DETRENDS = [("bg: minus 8 px Gaussian (default)", "bg"), ("row medians", "row"), ("column medians", "col"),
            ("rows, then columns", "both"), ("none", "none")]
DETECTORS = [("YOLO conf 0.25", "yolo", 0.25), ("YOLO conf 0.5 (default)", "yolo", 0.5),
             ("YOLO conf 0.7", "yolo", 0.7), ("threshold |z| > 2", "thr", 2.0),
             ("threshold |z| > 2.5 (fallback)", "thr", 2.5), ("threshold |z| > 3", "thr", 3.0)]


# ---------------------------------------------------------------- data and scores

def load_all(quick):
    items = []
    for e in sc.catalog():
        if e["status"].startswith(("empty", "unreadable")):
            continue
        try:
            rs = sc.raster_steps(sc.scan_path(e["id"]))
        except ValueError:
            continue
        labelled = detect.label_file(e["id"]).exists()
        items.append({"id": e["id"], "session": e["session"], "sensor": e.get("sensor"), "rs": rs,
                      "grid": rs["aligned"], "labelled": labelled,
                      "labels": detect.load_labels(e["id"])[0] if labelled else []})
    if quick:                                   # a few scans of each kind, for testing the script
        keep, seen = [], {}
        for it in items:
            k = (it["session"], it["labelled"])
            if seen.get(k, 0) < 3 or sc.resolve(REF_SCAN) == it["id"]:
                keep.append(it)
                seen[k] = seen.get(k, 0) + 1
        items = keep
    return items


def counts(pred, labels):
    """found / false / missed against the labels (IoU >= 0.3), the summed IoU of the found
    boxes, and the labels hit by any box centre (a looser 'was it seen at all')."""
    pred = [dict(b) for b in pred]
    found, false, missed = detect.match_labels(pred, labels)
    iou = sum(max(detect.iou(p, lb) for lb in labels) for p in pred if p["match"])
    centres = [((p["x0"] + p["x1"]) / 2, (p["y0"] + p["y1"]) / 2) for p in pred]
    hit = sum(any(lb["x0"] <= x <= lb["x1"] and lb["y0"] <= y <= lb["y1"] for x, y in centres) for lb in labels)
    return np.array([found, false, len(missed), iou, hit])


def prf(c):
    tp, fp, fn, iou, hit = c
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return {"found": int(tp), "false": int(fp), "missed": int(fn), "precision": round(p, 3),
            "recall": round(r, 3), "f1": round(f1, 3), "mean_iou": round(iou / tp, 3) if tp else None,
            "recall_centre": round(hit / (tp + fn), 3) if tp + fn else None}


def threshold(img_f, crop="auto", crop_z=3.0, z=2.5, detrend="bg"):
    m = process.margins_for(img_f, crop, crop_z)
    boxes, _ = detect.threshold_detect(process.apply_crop(img_f, m), (m["left"], m["top"]), z,
                                       detrend_method=detrend)
    return boxes, m


def kept_fraction(b, m, shape):
    """Share of a label box inside the region a crop keeps."""
    H, W = shape
    ix = max(0, min(b["x1"], W - m["right"]) - max(b["x0"], m["left"]))
    iy = max(0, min(b["y1"], H - m["bottom"]) - max(b["y0"], m["top"]))
    return ix * iy / max(1, (b["x1"] - b["x0"]) * (b["y1"] - b["y0"]))


def med(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return round(float(np.median(v)), 3) if v else None


def crack_peak(det, labels):
    """Median over the labelled boxes of the largest |value| inside the box."""
    return float(np.median([np.abs(det[b["y0"]:b["y1"], b["x0"]:b["x1"]]).max() for b in labels]))


# ---------------------------------------------------------------- A. grid

def bench_grid(items):
    rows, totals = [], {k: np.zeros(5) for k, _ in GRID_STEPS}
    for it in items:
        rs = dict(it["rs"])
        rs["aligned8"], shift8, _ = sc.fix_scan_direction(rs["unmirrored"], max_shift=8)
        base = sc.sawtooth(rs["recorded"]) or 1.0
        row = {"scan": it["id"], "session": it["session"], "lag_shift_px": rs["shift_px"], "lag_shift_px_8": shift8,
               "zigzag_8_vs_5": round(sc.sawtooth(rs["aligned8"]) / (sc.sawtooth(rs["aligned"]) or 1.0), 4)}
        for k, _ in GRID_STEPS:
            row[f"zigzag_{k}"] = round(sc.sawtooth(rs[k]) / base, 4)
            if it["labelled"]:                                      # the rest of the default pipeline
                c = counts(threshold(process.denoise(rs[k], "lpf"))[0], it["labels"])
                totals[k] += c
                row[f"thr_found_{k}"], row[f"thr_false_{k}"] = int(c[0]), int(c[1])
        rows.append(row)
    summary = [{"variant": name, "default": k == "aligned",
                "zigzag_vs_recorded": med([r[f"zigzag_{k}"] for r in rows]),
                "threshold": prf(totals[k])} for k, name in GRID_STEPS]
    shifts = {}
    for r in rows:
        s = shifts.setdefault(r["session"], {})
        s[str(r["lag_shift_px"])] = s.get(str(r["lag_shift_px"]), 0) + 1
    mirror, widths = [], []                  # where a backwards line puts a crack vs the nearest crack
    for it in items:
        if it["labels"]:
            xs = np.array([(b["x0"] + b["x1"]) / 2 for b in it["labels"]])
            W = it["grid"].shape[1]
            mirror.append(float(np.median([np.abs(xs - (W - x)).min() for x in xs])))
            widths.append(float(np.median([b["x1"] - b["x0"] for b in it["labels"]])))
    changed = [r for r in rows if r["lag_shift_px_8"] != r["lag_shift_px"]]
    wider = {"scans_changed": len(changed), "sessions": sorted({r["session"] for r in changed}),
             "labelled_changed": sum(r["scan"] in {it["id"] for it in items if it["labelled"]} for r in changed),
             "new_shifts": sorted({r["lag_shift_px_8"] for r in changed}),
             "zigzag_vs_5px_median": med([r["zigzag_8_vs_5"] for r in changed]),
             "zigzag_vs_5px_best": round(min((r["zigzag_8_vs_5"] for r in changed), default=1.0), 3)}
    return rows, {"variants": summary, "lag_shift_by_session": shifts, "wider_search": wider,
                  "mirror_to_crack_px": med(mirror), "label_width_px": med(widths)}


# ---------------------------------------------------------------- B. denoise

def bench_denoise(items):
    rows, totals = [], {name: np.zeros(5) for name, _, _ in FILTERS}
    for it in items:
        grid = it["grid"]
        base_zz = sc.sawtooth(grid) or 1.0
        ref = {}
        for name, method, k in FILTERS:
            t0 = time.perf_counter()
            f = process.denoise(grid, method, kernel=k)
            ms = (time.perf_counter() - t0) * 1000
            row = {"scan": it["id"], "session": it["session"], "filter": name, "ms": round(ms, 2),
                   "zigzag_left": round(sc.sawtooth(f) / base_zz, 4)}
            if it["labels"]:
                peak = ref.setdefault("peak", crack_peak(process.detrend(grid, "bg"), it["labels"]))
                boxes, _ = threshold(f)
                c = counts(boxes, it["labels"])
                totals[name] += c
                row.update(peak_kept=round(crack_peak(process.detrend(f, "bg"), it["labels"]) / peak, 4),
                           thr_found=int(c[0]), thr_false=int(c[1]), thr_missed=int(c[2]))
            rows.append(row)
    summary = []
    for name, _, _ in FILTERS:
        rs = [r for r in rows if r["filter"] == name]
        summary.append({"variant": name, "default": "default" in name, "ms_per_scan": med([r["ms"] for r in rs]),
                        "zigzag_left": med([r["zigzag_left"] for r in rs]),
                        "peak_kept": med([r["peak_kept"] for r in rs if "peak_kept" in r]),
                        "threshold": prf(totals[name])})
    return rows, summary


# ---------------------------------------------------------------- C. crop

def bench_crop(items):
    rows, totals = [], {name: np.zeros(5) for name, _, _ in CROPS}
    for it in items:
        f = process.denoise(it["grid"], "lpf")
        H, W = f.shape
        for name, crop, cz in CROPS:
            m = process.margins_for(f, crop, cz)
            kept = (H - m["top"] - m["bottom"]) * (W - m["left"] - m["right"])
            row = {"scan": it["id"], "session": it["session"], "crop": name, **m,
                   "area_removed": round(1 - kept / (H * W), 4)}
            if it["labelled"]:
                boxes, _ = threshold(f, crop, cz)
                c = counts(boxes, it["labels"])
                totals[name] += c
                row.update(labels_cut=sum(kept_fraction(b, m, f.shape) < 0.5 for b in it["labels"]),
                           thr_found=int(c[0]), thr_false=int(c[1]), thr_missed=int(c[2]))
            rows.append(row)
    summary = []
    for name, _, _ in CROPS:
        rs = [r for r in rows if r["crop"] == name]
        lab = [r for r in rs if "labels_cut" in r]
        summary.append({"variant": name, "default": "default" in name,
                        "area_removed": med([r["area_removed"] for r in rs]),
                        "max_area_removed": round(max(r["area_removed"] for r in rs), 3),
                        "labels_cut": int(sum(r["labels_cut"] for r in lab)),
                        "labels_total": int(sum(len(it["labels"]) for it in items if it["labelled"])),
                        "threshold": prf(totals[name])})
    return rows, summary


# ---------------------------------------------------------------- D. order and combinations

def bench_pipeline(items):
    lab = [it for it in items if it["labelled"]]
    combos = {}
    for fname, method, k in COMBO_FILTERS:
        for cname, crop in COMBO_CROPS:
            c = sum((counts(threshold(process.denoise(it["grid"], method, kernel=k), crop)[0], it["labels"])
                     for it in lab), np.zeros(5))
            combos[f"{fname} | {cname}"] = {"filter": fname, "crop": cname, **prf(c)}
    order = {}
    c_default, c_first, removed = np.zeros(5), np.zeros(5), {"default": [], "crop_first": []}
    for it in lab:
        g = it["grid"]
        boxes, m = threshold(process.denoise(g, "lpf"))
        c_default += counts(boxes, it["labels"])
        removed["default"].append(sum(m.values()))
        m2 = process.margins_for(g, "auto")                         # border measured on the raw grid
        f2 = process.denoise(process.apply_crop(g, m2), "lpf")
        boxes2, _ = detect.threshold_detect(f2, (m2["left"], m2["top"]))
        c_first += counts(boxes2, it["labels"])
        removed["crop_first"].append(sum(m2.values()))
    order["filter, then crop (default)"] = {**prf(c_default), "crop_px_sum": med(removed["default"])}
    order["crop, then filter"] = {**prf(c_first), "crop_px_sum": med(removed["crop_first"])}
    detrends = {}
    for name, method in DETRENDS:
        c = sum((counts(threshold(process.denoise(it["grid"], "lpf"), detrend=method)[0], it["labels"])
                 for it in lab), np.zeros(5))
        detrends[name] = prf(c)
    return {"filter_x_crop": combos, "order": order, "background_removal": detrends}


# ---------------------------------------------------------------- E. detect

def bench_detect(items):
    yolo_ok = detect.yolo_available()
    if yolo_ok:
        detect.yolo_detect(items[0]["grid"])                        # load the model before timing
    rows, totals = [], {}
    for it in items:
        f = process.denoise(it["grid"], "lpf")
        for name, kind, p in DETECTORS:
            if kind == "yolo" and not yolo_ok:
                continue
            t0 = time.perf_counter()
            boxes = detect.yolo_detect(it["grid"], p) if kind == "yolo" else threshold(f, z=p)[0]
            row = {"scan": it["id"], "session": it["session"], "detector": name,
                   "ms": round((time.perf_counter() - t0) * 1000, 1), "boxes": len(boxes)}
            if it["labelled"]:
                c = counts(boxes, it["labels"])
                for key in (it["session"], "all"):
                    totals[(name, key)] = totals.get((name, key), np.zeros(5)) + c
                row.update(found=int(c[0]), false=int(c[1]), missed=int(c[2]))
            rows.append(row)
    sessions = sorted({it["session"] for it in items if it["labelled"]}) + ["all"]
    summary = []
    for name, kind, _ in DETECTORS:
        rs = [r for r in rows if r["detector"] == name]
        if not rs:
            continue
        unl = [r["boxes"] for r in rs if "found" not in r]
        summary.append({"variant": name, "default": "default" in name, "kind": kind,
                        "ms_per_scan": med([r["ms"] for r in rs]),
                        "by_session": {s: prf(totals[(name, s)]) for s in sessions if (name, s) in totals},
                        "unlabelled_scans": len(unl), "unlabelled_boxes_median": med(unl),
                        "unlabelled_boxes_max": max(unl) if unl else None})
    return rows, {"yolo_available": yolo_ok, "yolo_backend": detect.yolo_backend() if yolo_ok else None,
                  "detectors": summary}


# ---------------------------------------------------------------- F. drawing

def bench_drawing(items):
    lab = [it for it in items if it["labels"]]
    sample = lab[::max(1, len(lab) // 8)][:8]
    rows = []
    for it in sample:
        grid, meta = sc.load_scan(sc.scan_path(it["id"]))
        st = process.pipeline(grid)
        for view in ("amplitude", "dx"):
            img, off, title, cmap, div = render.view_spec(st, view, it["sensor"])
            for plot in ("2d", "3d"):
                t0 = time.perf_counter()
                fn = render.surface if plot == "3d" else render.single
                data = fn(img, meta["pitch_mm"], off, cmap=cmap, diverging=div, title=title, boxes=it["labels"])
                rows.append({"scan": it["id"], "view": view, "plot": plot,
                             "ms": round((time.perf_counter() - t0) * 1000), "kb": round(len(data) / 1024)})
    summary = [{"view": v, "plot": p, "ms_per_figure": med([r["ms"] for r in rows if r["view"] == v and r["plot"] == p]),
                "kb_per_figure": med([r["kb"] for r in rows if r["view"] == v and r["plot"] == p])}
               for v in ("amplitude", "dx") for p in ("2d", "3d")]
    return rows, {"scans": len(sample), "figures": summary}


def example_figures():
    sid = sc.resolve(REF_SCAN)
    grid, meta = sc.load_scan(sc.scan_path(sid))
    st = process.pipeline(grid)
    boxes = detect.yolo_detect(grid) if detect.yolo_available() else threshold_boxes(st)
    p, sensor = meta["pitch_mm"], meta.get("sensor")
    img, off, title, cmap, div = render.view_spec(st, "detrended", sensor)
    out = {"example_2d.png": render.single(img, p, off, title + " (2D)", cmap, div, "amplitude (a.u.)", boxes=boxes),
           "example_3d.png": render.surface(img, p, off, cmap, div, title + " (3D)", "amplitude (a.u.)", boxes=boxes)}
    img, off, title, cmap, div = render.view_spec(st, "dx", sensor)
    out["example_3d_dx.png"] = render.surface(img, p, off, cmap, div, title + " (3D)", "amplitude (a.u.)")
    out["example_steps.png"] = render.steps(sc.raster_steps(sc.scan_path(sid)), p, st, boxes,
                                            suptitle=f"{sid}: from scan to heatmap")
    for name, data in out.items():
        (FIG / name).write_bytes(data)
    return sid, list(out)


def threshold_boxes(st):
    return detect.threshold_detect(st["cropped"], st["offset"])[0]


# ---------------------------------------------------------------- charts

def _style(ax, title, xlabel):
    ax.set_title(title, color=INK, loc="left", fontsize=10.5)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.grid(axis="x", color="#e6e4dc", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0, colors=INK)


def bars(ax, rows, key, title, xlabel, fmt="{:.2f}", get=None):
    """One horizontal bar per variant; the app's default in blue, alternatives grey."""
    names = [r["variant"] for r in rows]
    vals = [(get(r) if get else r[key]) for r in rows]
    vals = [np.nan if v is None else v for v in vals]
    y = np.arange(len(rows))
    ax.barh(y, vals, height=0.62, color=[BLUE if r["default"] else GREY for r in rows], edgecolor="white", lw=2)
    top = np.nanmax(vals) if np.isfinite(np.nanmax(vals)) else 1
    for yi, v in zip(y, vals):
        if np.isfinite(v):
            ax.text(v + top * 0.02, yi, fmt.format(v), va="center", fontsize=9, color=INK)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, top * 1.22 or 1)
    if "%" in fmt:
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    _style(ax, title, xlabel)


def fig_grid(summary):
    rows = summary["variants"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 2.8))
    bars(axs[0], rows, "zigzag_vs_recorded", "Row-to-row zig-zag left (all scans, median)",
         "share of the zig-zag in the recorded order", "{:.0%}")
    bars(axs[1], rows, "f1", "Threshold detector F1 after the rest of the pipeline\n(labelled scans)",
         "F1 against the labels", get=lambda r: r["threshold"]["f1"])
    for ax in axs[1:]:
        ax.set_yticklabels([])
    fig.tight_layout()
    fig.savefig(FIG / "A_grid.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_denoise(rows):
    sid = sc.resolve(REF_SCAN)
    grid, meta = sc.load_scan(sc.scan_path(sid))
    labels, _ = detect.load_labels(sid)
    b = labels[len(labels) // 2]
    cy, cx = (b["y0"] + b["y1"]) // 2, (b["x0"] + b["x1"]) // 2
    y0, x0 = int(np.clip(cy - 18, 0, grid.shape[0] - 36)), int(np.clip(cx - 28, 0, grid.shape[1] - 56))
    fig = plt.figure(figsize=(16, 6.8))
    gs = fig.add_gridspec(2, 15, height_ratios=[1, 1.05], hspace=0.4, wspace=1.2)
    maps = [process.denoise(grid, m, kernel=k)[y0:y0 + 36, x0:x0 + 56] for _, m, k in FILTERS]
    lim = process.percentile_limits(maps[-1], 1.0)
    for i, ((name, _, _), img) in enumerate(zip(FILTERS, maps)):
        ax = fig.add_subplot(gs[0, 3 * i:3 * i + 3])
        render.draw_map(ax, img, meta["pitch_mm"], (x0, y0), "viridis", title=name, limits=lim, colorbar=False)
        if i:
            ax.set_ylabel("")
    panels = [("zigzag_left", "Zig-zag left (all scans)", "share of the unfiltered zig-zag", "{:.0%}", None),
              ("peak_kept", "Crack peak kept (labelled)", "share of the unfiltered peak", "{:.0%}", None),
              ("f1", "Threshold detector F1 (labelled)", "F1 against the labels", "{:.2f}",
               lambda r: r["threshold"]["f1"])]
    for i, (key, title, xlabel, fmt, get) in enumerate(panels):
        ax = fig.add_subplot(gs[1, 5 * i:5 * i + 5])
        bars(ax, rows, key, title, xlabel, fmt, get)
        if i:
            ax.set_yticklabels([])
    fig.suptitle(f"Denoising - top: one crack area of {sid}, same colour scale; bottom: all scans",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.savefig(FIG / "B_denoise.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_crop(rows):
    sid = sc.resolve(REF_SCAN)
    grid, meta = sc.load_scan(sc.scan_path(sid))
    f = process.denoise(grid, "lpf")
    p = meta["pitch_mm"]
    fig = plt.figure(figsize=(17, 4.2))
    left, right = fig.subfigures(1, 2, width_ratios=[1, 2.7], wspace=0.02)
    ax = left.subplots()
    render.draw_map(ax, f, p, cmap="gray", title=f"Kept region on {sid.split('/')[0]} R 1.7 V 300 kHz",
                    colorbar=False)
    H, W = f.shape
    shown = [("auto, z > 3 (default)", "auto", 3.0, BLUE, "-"), ("fixed 10 px", 10, 3.0, ORANGE, "--"),
             ("auto, z > 2", "auto", 2.0, AQUA, ":")]
    for name, crop, cz, color, ls in shown:
        m = process.margins_for(f, crop, cz)
        ax.add_patch(Rectangle((m["left"] * p, m["top"] * p), (W - m["left"] - m["right"]) * p,
                               (H - m["top"] - m["bottom"]) * p, fill=False, ec=color, lw=2, ls=ls, label=name))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False, fontsize=9)
    panels = [("area_removed", "Area removed (median)", "share of the scan", "{:.0%}", None),
              ("labels_cut", "Labelled cracks cut off", "cracks (of all labels)", "{:.0f}", None),
              ("f1", "Threshold detector F1", "F1 against the labels", "{:.2f}", lambda r: r["threshold"]["f1"])]
    axs = right.subplots(1, 3, gridspec_kw={"wspace": 0.12})
    right.subplots_adjust(left=0.17, bottom=0.2)
    for i, (ax, (key, title, xlabel, fmt, get)) in enumerate(zip(axs, panels)):
        bars(ax, rows, key, title, xlabel, fmt, get)
        if i:
            ax.set_yticklabels([])
    fig.savefig(FIG / "C_crop.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_pipeline(res):
    fnames, cnames = [f[0] for f in COMBO_FILTERS], [c[0] for c in COMBO_CROPS]
    f1 = np.array([[res["filter_x_crop"][f"{f} | {c}"]["f1"] for c in cnames] for f in fnames])
    fig, axs = plt.subplots(1, 2, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1, 1.3], "wspace": 0.9})
    ax = axs[0]
    ax.imshow(f1, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(fnames)):
        for j in range(len(cnames)):
            ax.text(j, i, f"{f1[i, j]:.2f}", ha="center", va="center", fontsize=10,
                    color="white" if f1[i, j] > 0.6 else INK)
    ax.set_xticks(range(len(cnames)), cnames)
    ax.set_yticks(range(len(fnames)), fnames)
    ax.set_xlabel("border crop", color=MUTED)
    ax.set_ylabel("filter", color=MUTED)
    ax.set_title("Threshold detector F1: filter x crop\n(default = lpf | auto z>3)", color=INK, loc="left", fontsize=10.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    rows = [{"variant": k, "default": "default" in k, "f1": v["f1"]} for k, v in res["order"].items()]
    rows += [{"variant": "detrend: " + k, "default": False, "f1": v["f1"]} for k, v in res["background_removal"].items()
             if "default" not in k]
    bars(axs[1], rows, "f1", "Order of steps and background removal\n(threshold detector, labelled scans)",
         "F1 against the labels")
    fig.savefig(FIG / "D_pipeline.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_detect(summary):
    dets = summary["detectors"]
    sessions = [s for s in dets[0]["by_session"] if s != "all"]
    fig, axs = plt.subplots(1, len(sessions), figsize=(6.2 * len(sessions), 3.4), sharey=True)
    axs = np.atleast_1d(axs)
    y = np.arange(len(dets))
    for ax, s in zip(axs, sessions):
        for key, color, marker, dy in (("precision", BLUE, "o", -0.18), ("recall", ORANGE, "s", 0),
                                       ("f1", AQUA, "D", 0.18)):
            vals = [d["by_session"][s][key] for d in dets]
            ax.scatter(vals, y + dy, s=46, color=color, marker=marker, edgecolor="white", lw=1.2, label=key, zorder=3)
        n = dets[0]["by_session"][s]
        ax.set_xlim(0, 1.05)
        ax.set_yticks(y, [d["variant"] for d in dets])
        _style(ax, f"{s} ({n['found'] + n['missed']} labelled cracks)", "score (0-1)")
    axs[0].invert_yaxis()                               # once: the axes share Y
    axs[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
    fig.suptitle("Detection against the DRAFT labels (YOLO was trained on these scans)", x=0.01, ha="left",
                 color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "E_detect.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- report

def write_csv(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / name, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)


def pct(v):
    return "–" if v is None else f"{v:.0%}"


def num(v, d=2):
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |" for r in rows]
    return "\n".join(out)


def bold_default(name, is_default):
    return f"**{name}**" if is_default else name


def write_report(res):
    g, dn, cr, pl, de, dr = (res[k] for k in ("grid", "denoise", "crop", "pipeline", "detect", "drawing"))
    best_combo = max(pl["filter_x_crop"].values(), key=lambda v: v["f1"])
    yolo = next((d for d in de["detectors"] if d["kind"] == "yolo" and d["default"]), None)
    thr = next(d for d in de["detectors"] if "2.5" in d["variant"])
    lines = [
        "# ECT processing benchmark",
        "",
        f"Generated {res['generated']} by `python -m app.tools.benchmark` on {res['scans']} readable scans "
        f"({res['labelled_scans']} with labels, {res['labels']} labelled cracks). Every step is run with its "
        "alternatives; the app's default is in **bold**.",
        "",
        "Caveats: the labels are DRAFT (drawn by eye, not checked against the plate drawing). The YOLO model was "
        "trained on all labelled scans, so its scores measure how well it fits them, not how it would do on a "
        "new plate. The threshold detector has no training and is scored on the same scans.",
        "",
        "## A. From the recording to the grid",
        "",
        "The probe scans the plate zig-zag: line 1 left to right, line 2 right to left, and so on. The TDMS "
        "file stores one long list. Cut into lines, every 2nd line is backwards, so the cracks smear into a "
        "comb. Reversing those lines (the lab's `reshape_raster`) fixes the order; the probe also lags a few "
        "points when it turns, so the return lines are shifted by the lag that minimises the zig-zag, and "
        "their level is matched by medians.",
        "",
        table(["grid", "zig-zag left", "threshold P", "R", "F1", "recall, centre in box"],
              [[bold_default(v["variant"], v["default"]), pct(v["zigzag_vs_recorded"]),
                num(v["threshold"]["precision"]), num(v["threshold"]["recall"]), num(v["threshold"]["f1"]),
                num(v["threshold"]["recall_centre"])] for v in g["variants"]]),
        "",
        "The detector columns run the rest of the default pipeline (lpf, auto crop, threshold |z| > 2.5) on each grid.",
        "",
        "Lag shift chosen per session (px: number of scans): "
        + "; ".join(f"{s}: " + ", ".join(f"{k} px x{n}" for k, n in sorted(d.items(), key=lambda kv: int(kv[0])))
                    for s, d in g["lag_shift_by_session"].items()) + ".",
        "The lag fix picks the shift with the least zig-zag, so the zig-zag drop of that step is by construction; "
        "the detector scores are the independent check - but here they cannot separate the grids: the cracks of "
        "this coupon sit in evenly spaced columns, so a crack on a backwards line lands a median "
        f"{num(g['mirror_to_crack_px'], 0)} px from another crack column (labelled boxes are {num(g['label_width_px'], 0)} "
        "px wide) and still falls inside a box. On a plate with an irregular layout the recorded order would put "
        "the signal in the wrong place; the step-by-step figure (panels 3 and 4) shows the smear.",
        "",
        f"Search limit: with ±8 px instead of the app's ±5 px, {g['wider_search']['scans_changed']} scans pick another "
        f"shift ({', '.join(g['wider_search']['sessions']) or 'none'}; new shift "
        f"{', '.join(f'{s:+d}' for s in g['wider_search']['new_shifts']) or '-'} px; "
        f"{g['wider_search']['labelled_changed']} of them labelled) and their zig-zag drops to a median "
        f"{pct(g['wider_search']['zigzag_vs_5px_median'])} (best {pct(g['wider_search']['zigzag_vs_5px_best'])}) of the "
        "±5 px result: those scans hit the ±5 px limit, their real lag is larger.",
        "",
        "![grid](figures/A_grid.png)",
        "",
        "## B. Denoising",
        "",
        "What is left is the row-to-row zig-zag (the comb) and pixel noise. The comb lives along Y, so the "
        "lab filters down each column: a Hamming FIR low-pass (order 24, cutoff 0.3 of Nyquist, edges padded "
        "50 points, group delay removed), then a 3 x 3 median for spikes.",
        "",
        table(["filter", "zig-zag left", "crack peak kept", "threshold P", "R", "F1", "recall, centre in box",
               "ms/scan"],
              [[bold_default(v["variant"], v["default"]), pct(v["zigzag_left"]), pct(v["peak_kept"]),
                num(v["threshold"]["precision"]), num(v["threshold"]["recall"]), num(v["threshold"]["f1"]),
                num(v["threshold"]["recall_centre"]), num(v["ms_per_scan"], 1)] for v in dn]),
        "",
        "Crack peak kept: the largest |value| inside each labelled box on the background-removed map, relative "
        "to no filter. A background noise level is not reported: on this coupon the labelled boxes and the "
        "border cover about three quarters of the grid, so what is left between the cracks is mostly their "
        "halos, not noise. The zig-zag is a 2-line pattern, finer than a crack, which is why a filter down the "
        "columns can remove most of it and keep most of the crack peak.",
        "",
        "![denoise](figures/B_denoise.png)",
        "",
        "## C. Border crop",
        "",
        "The scan border carries probe turn-around streaks and a bright band where lines start. The auto crop "
        "walks inward from each edge while a row/column is abnormal in level (median) or spread (std), as a "
        "robust z-score above the threshold, capped at 15 % per side, so each edge gets its own margin.",
        "",
        table(["crop", "area removed (median)", "worst scan", "labelled cracks cut", "threshold P", "R", "F1"],
              [[bold_default(v["variant"], v["default"]), pct(v["area_removed"]), pct(v["max_area_removed"]),
                f"{v['labels_cut']} / {v['labels_total']}", num(v["threshold"]["precision"]),
                num(v["threshold"]["recall"]), num(v["threshold"]["f1"])] for v in cr]),
        "",
        "A crack counts as cut when less than half of its labelled box stays inside the kept region. YOLO is not "
        "affected by the crop: it reads the whole grid and drops boxes on the scan border itself.",
        "",
        "![crop](figures/C_crop.png)",
        "",
        "## D. Order of the steps and background removal",
        "",
        "Threshold detector F1 for every filter x crop pair:",
        "",
        table(["filter \\ crop"] + [c[0] for c in COMBO_CROPS],
              [[f[0]] + [num(pl["filter_x_crop"][f"{f[0]} | {c[0]}"]["f1"]) for c in COMBO_CROPS]
               for f in COMBO_FILTERS]),
        "",
        f"Best pair: {best_combo['filter']} | {best_combo['crop']} (F1 {best_combo['f1']:.2f}).",
        "",
        table(["order", "P", "R", "F1", "border removed (px, sum of 4 sides, median)"],
              [[bold_default(k, "default" in k), num(v["precision"]), num(v["recall"]), num(v["f1"]),
                num(v["crop_px_sum"], 0)] for k, v in pl["order"].items()]),
        "",
        f"Crop first, then filter: F1 {num(pl['order']['crop, then filter']['f1'])} vs "
        f"{num(pl['order']['filter, then crop (default)']['f1'])} for the default order; the border is then "
        "measured on the unfiltered grid and the FIR runs on the cropped map, so its edge padding starts at the "
        "crop line instead of at the scan edge.",
        "",
        table(["background removal", "P", "R", "F1"],
              [[bold_default(k, "default" in k), num(v["precision"]), num(v["recall"]), num(v["f1"])]
               for k, v in pl["background_removal"].items()]),
        "",
        "![pipeline](figures/D_pipeline.png)",
        "",
        "## E. Detection",
        "",
        "YOLO11n reads a 3-channel image of the whole grid (R = lpf + background removed, G = d/dx, B = d/dy), "
        "letterboxed to 320 px, on onnxruntime. The threshold detector marks |robust z| above the threshold on "
        "the background-removed, cropped map and joins fragments within 2 px. A box matches a label at IoU >= 0.3.",
        "",
        table(["detector", "session", "found", "false", "missed", "precision", "recall", "F1", "mean IoU",
               "recall, centre in box"],
              [[bold_default(d["variant"], d["default"]), s, v["found"], v["false"], v["missed"],
                num(v["precision"]), num(v["recall"]), num(v["f1"]), num(v["mean_iou"]), num(v["recall_centre"])]
               for d in de["detectors"] for s, v in d["by_session"].items()]),
        "",
        "The last column counts a crack as seen when any box centre falls inside its labelled box, whatever the "
        "box size: threshold boxes cover the strongest part of the signature only and often overlap a label by "
        "less than IoU 0.3.",
        "",
        table(["detector", "ms per scan", "unlabelled scans", "boxes per unlabelled scan (median / max)"],
              [[bold_default(d["variant"], d["default"]), num(d["ms_per_scan"], 1), d["unlabelled_scans"],
                f"{num(d['unlabelled_boxes_median'], 0)} / {d['unlabelled_boxes_max']}"] for d in de["detectors"]]),
        "",
        "Boxes on unlabelled scans have no ground truth: many of those scans show the same coupon, so boxes there "
        "are not all false.",
        "",
        "![detect](figures/E_detect.png)",
        "",
        "## F. Drawing: 2D heatmap vs 3D surface",
        "",
        f"Render time and PNG size, median of {dr['scans']} labelled scans with their label boxes drawn:",
        "",
        table(["view", "plot", "ms per figure", "KB per figure"],
              [[f["view"], f["plot"].upper(), num(f["ms_per_figure"], 0), num(f["kb_per_figure"], 0)]
               for f in dr["figures"]]),
        "",
        "Both are drawn in mm with the same percentile clipping. The 2D heatmap shows positions and box edges "
        "exactly and is what measurements use. The 3D surface (height = value) shows how strong and how sharp "
        "each crack signal is relative to the others and the shape of the background, but peaks hide what is "
        "behind them and the perspective makes positions hard to read: use it to explain, not to measure.",
        "",
        "![2D](figures/example_2d.png)",
        "![3D](figures/example_3d.png)",
        "![3D dx](figures/example_3d_dx.png)",
        "![steps](figures/example_steps.png)",
        "",
        "## Summary",
        "",
    ]
    if yolo:
        a = yolo["by_session"]["all"]
        lines.append(f"- Detection: YOLO (conf 0.5) found {a['found']} of {a['found'] + a['missed']} labelled cracks "
                     f"with {a['false']} false boxes (F1 {a['f1']:.2f}, fitted on these scans); the threshold "
                     f"fallback reaches F1 {thr['by_session']['all']['f1']:.2f}.")
    bg = pl["background_removal"]
    lines += [
        f"- Grid: reversing every 2nd line + the lag fix leaves {pct(g['variants'][2]['zigzag_vs_recorded'])} of the "
        f"recorded zig-zag (threshold F1 {num(g['variants'][0]['threshold']['f1'])} -> "
        f"{num(g['variants'][2]['threshold']['f1'])}: the evenly spaced crack columns hide the difference from "
        "the detector). A ±8 px lag search would improve "
        f"{g['wider_search']['scans_changed']} unlabelled scans and change no labelled one.",
        f"- Denoising: the default lpf removes {pct(1 - dn[-1]['zigzag_left'])} of the remaining zig-zag and keeps "
        f"{pct(dn[-1]['peak_kept'])} of the crack peak (Gaussian: {pct(dn[2]['peak_kept'])}); threshold F1 "
        f"{num(dn[-1]['threshold']['f1'])} vs {num(dn[0]['threshold']['f1'])} unfiltered.",
        f"- Border crop: the default auto crop removes a median {pct(cr[4]['area_removed'])} of the scan and cuts "
        f"{cr[4]['labels_cut']} labelled cracks; threshold F1 {num(cr[4]['threshold']['f1'])} vs "
        f"{num(cr[0]['threshold']['f1'])} without a crop.",
        f"- Background removal: F1 {num(bg[DETRENDS[0][0]]['f1'])} with the 8 px Gaussian vs "
        f"{num(bg['none']['f1'])} without and {num(max(v['f1'] for k, v in bg.items() if k != DETRENDS[0][0]))} "
        "for the best row/column variant.",
        "",
        "## Metrics",
        "",
        "- zig-zag: mean |second difference down Y| without the 12 outer columns (`scan.sawtooth`).",
        "- crack peak kept: median over the labelled boxes of the largest |value| inside the box on the "
        "background-removed map, divided by the same without a filter.",
        "- found / false / missed: greedy one-to-one matching, most confident box first, IoU >= 0.3 "
        "(`detect.match_labels`); precision = found / (found + false), recall = found / (found + missed).",
        "- recall, centre in box: share of labelled cracks with at least one box centre inside their box.",
        "- Times are on this PC (2 BLAS threads) and only compare the variants with each other.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="a few scans per session, to test the script")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    items = load_all(args.quick)
    lab = [it for it in items if it["labelled"]]
    print(f"{len(items)} scans, {len(lab)} labelled ({sum(len(it['labels']) for it in lab)} cracks)")
    res = {"generated": datetime.date.today().isoformat(), "scans": len(items), "labelled_scans": len(lab),
           "labels": sum(len(it["labels"]) for it in lab), "quick": args.quick}
    steps = [("grid", bench_grid, "A_grid.csv"), ("denoise", bench_denoise, "B_denoise.csv"),
             ("crop", bench_crop, "C_crop.csv"), ("detect", bench_detect, "E_detect.csv"),
             ("drawing", bench_drawing, "F_drawing.csv")]
    for key, fn, csv_name in steps:
        t = time.time()
        rows, res[key] = fn(items)
        write_csv(csv_name, rows)
        print(f"  {key}: {time.time() - t:.1f}s")
    t = time.time()
    res["pipeline"] = bench_pipeline(items)
    print(f"  pipeline: {time.time() - t:.1f}s")
    res["examples"] = example_figures()[1]
    fig_grid(res["grid"])
    fig_denoise(res["denoise"])
    fig_crop(res["crop"])
    fig_pipeline(res["pipeline"])
    if res["detect"]["detectors"]:
        fig_detect(res["detect"])
    (OUT / "benchmark.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    write_report(res)
    print(f"wrote {OUT.relative_to(PROJECT_ROOT)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
