#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crack_dataset.py - A separate dataset of the RAW maps of every scan, detected
automatically by the trained YOLO model (crack_model.py).

  results/dataset_raw/
    images/<session>/<stem>.png      raw map (viridis, 2-98 % scale), 4 px per scan point
    arrays/<session>/<stem>.npy      raw values (float), the exact data behind the image
    labels/<session>/<stem>.txt      the model's boxes, YOLO format: 0 cx cy w h conf
    annotated/<session>/<stem>.png   raw map with the boxes, confidence and mm axes
    detections.csv                   one row per crack: position, box, length, width, conf
    files.csv                        one row per scan: cracks found, status
    README.txt

"Raw" = the map as load_scan returns it: serpentine lines un-mirrored and the scan-direction
lag fixed (without that each crack is a zig-zag), but NOT filtered, cropped or detrended.
The model itself still sees its usual input (crack_labels.model_image); the boxes are in
scan pixels, so they sit on the raw map 1:1. Crack length/width are measured on the
detrended map (crack_labels.measure_crack), where the background slope is removed.

The labels here are the MODEL's detections, not ground truth: do not train on them
unchecked (the model would learn its own mistakes).

  python crack_dataset.py            # all scans under data/
  python crack_dataset.py --conf 0.6
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tdms_crack_scan as backend
import crack_labels
import crack_model

OUT_DIR = backend.RESULTS_DIR / "dataset_raw"
UPSCALE = 4          # scans are ~150 x 100 points: 4 px per point to be viewable
MARKER = "README.txt"


def _raw_png(grid):
    lo, hi = backend.robust_limits(grid)
    norm = np.clip((grid - lo) / (hi - lo if hi > lo else 1.0), 0, 1)
    rgb = (colormaps["viridis"](np.nan_to_num(norm))[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb).resize((grid.shape[1] * UPSCALE, grid.shape[0] * UPSCALE),
                                       Image.NEAREST)


def _annotated(grid, boxes, meta, title, path):
    """Raw map + boxes, drawn like the app's Raw view. A bare Figure (no pyplot), so it
    is safe from the app's worker thread."""
    fig = Figure(figsize=(9, 6), facecolor=backend.SURFACE, layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    fig.suptitle(title, fontsize=10, color=backend.INK)
    view = {"img": grid, "title": f"Raw ({backend.signal_name(meta.get('sensor'))})",
            "cmap": "viridis", "cbar": "raw value (a.u.)", "boxes": boxes}
    backend._draw_view(ax, view, meta["pitch_mm"], scale=1.2, wrap=80)
    fig.savefig(path, dpi=90)


def scan_files(data_dir=None):
    return sorted(Path(data_dir or backend.DATA_DIR).rglob("*.tdms"))


def build(out_dir=OUT_DIR, conf=crack_model.CONF, files=None, log=print, progress=None):
    """Export + detect every scan. Returns (files DataFrame, detections DataFrame)."""
    if not crack_model.WEIGHTS.exists():
        raise FileNotFoundError("No trained model yet - train the crack model first.")
    out_dir = Path(out_dir)
    if out_dir.exists():
        if not (out_dir / MARKER).exists():
            raise FileExistsError(f"{out_dir} exists and is not a dataset this tool wrote; "
                                  "choose another folder.")
        shutil.rmtree(out_dir)              # regenerate: no stale files of an older model
    files = list(files or scan_files())
    file_rows, det_rows = [], []
    for i, p in enumerate(files, start=1):
        session = p.parent.name
        row = {"session": session, "file": p.name, "cracks": 0, "mean_conf": None,
               "labelled": False, "found": None, "false": None, "missed": None}
        try:
            boxes, grid = crack_model.predict(p, conf=conf)
            _, meta = backend.load_scan(p)
        except Exception as e:                  # template / empty files: no scan inside
            row["status"] = f"skipped: {e}"
            file_rows.append(row)
            if progress:
                progress(i, len(files))
            continue
        pitch = meta["pitch_mm"]
        H, W = grid.shape
        flat = backend.detrend(backend.apply_filter(grid, backend.DEFAULT_FILTER, 3),
                               backend.DEFAULT_DETREND)
        labels = crack_labels.load_labels(p)
        if labels:
            hit, free = crack_model.match_to_labels(boxes, labels)
            for b, h in zip(boxes, hit):
                b["match"] = h
            row.update(labelled=True, found=sum(hit), false=len(hit) - sum(hit),
                       missed=len(free))

        for sub in ("images", "arrays", "labels", "annotated"):
            (out_dir / sub / session).mkdir(parents=True, exist_ok=True)
        _raw_png(grid).save(out_dir / "images" / session / f"{p.stem}.png")
        np.save(out_dir / "arrays" / session / f"{p.stem}.npy", grid)
        (out_dir / "labels" / session / f"{p.stem}.txt").write_text("\n".join(
            f"0 {(b['x0'] + b['x1']) / 2 / W:.6f} {(b['y0'] + b['y1']) / 2 / H:.6f} "
            f"{(b['x1'] - b['x0']) / W:.6f} {(b['y1'] - b['y0']) / H:.6f} {b['conf']:.3f}"
            for b in boxes))
        title = (f"{session}/{p.name}   sensor={meta.get('sensor', '?')}  "
                 f"amp={meta.get('amp', '?')}V  freq={meta.get('freq', '?')}kHz\n"
                 f"{len(boxes)} cracks detected by the model (conf ≥ {conf})")
        _annotated(grid, boxes, meta, title, out_dir / "annotated" / session / f"{p.stem}.png")

        for k, b in enumerate(sorted(boxes, key=lambda b: (b["y0"], b["x0"])), start=1):
            m = crack_labels.measure_crack(flat, b, pitch) or {}
            det_rows.append({
                "session": session, "file": p.name, "sensor": meta.get("sensor"),
                "amp_V": meta.get("amp"), "freq_kHz": meta.get("freq"), "crack": k,
                "conf": round(b["conf"], 3),
                "x0_px": b["x0"], "y0_px": b["y0"], "x1_px": b["x1"], "y1_px": b["y1"],
                "x_mm": round((b["x0"] + b["x1"]) / 2 * pitch, 1),
                "y_mm": round((b["y0"] + b["y1"]) / 2 * pitch, 1),
                "box_w_mm": round((b["x1"] - b["x0"]) * pitch, 1),
                "box_h_mm": round((b["y1"] - b["y0"]) * pitch, 1),
                "length_mm": round(m["length_mm"], 1) if m else None,
                "length_is_minimum": m.get("spills"),
                "width_mm": round(m["width_mm"], 1) if m else None,
                "along": m.get("along"), "snr": round(m["snr"], 1) if m else None,
                "size_reliable": m.get("reliable"),
                "matches_label": b.get("match"),
            })
        row.update(cracks=len(boxes), status="ok",
                   mean_conf=round(float(np.mean([b["conf"] for b in boxes])), 3)
                   if boxes else None)
        file_rows.append(row)
        if progress:
            progress(i, len(files))

    out_dir.mkdir(parents=True, exist_ok=True)
    files_df, det_df = pd.DataFrame(file_rows), pd.DataFrame(det_rows)
    files_df.to_csv(out_dir / "files.csv", index=False)
    det_df.to_csv(out_dir / "detections.csv", index=False)
    (out_dir / MARKER).write_text(
        (__doc__.split("  python crack_dataset.py")[0]).strip() + "\n\n"
        f"Model: {crack_model.WEIGHTS}\nConfidence cut: {conf}\n"
        f"Scans: {int((files_df['status'] == 'ok').sum())} detected, "
        f"{int((files_df['status'] != 'ok').sum())} skipped; {len(det_df)} cracks.\n",
        encoding="utf-8")
    ok = files_df[files_df["status"] == "ok"]
    log(f"Raw dataset: {len(ok)} scans, {len(det_df)} cracks -> {out_dir}")
    for s, g in ok.groupby("session"):
        log(f"  {s:<16} {len(g):>3} scans, cracks per scan "
            f"min {g['cracks'].min()} median {int(g['cracks'].median())} max {g['cracks'].max()}")
    lab = ok[ok["labelled"]]
    if len(lab):
        log(f"  vs labels ({len(lab)} labelled scans): {int(lab['found'].sum())} found, "
            f"{int(lab['false'].sum())} false, {int(lab['missed'].sum())} missed "
            "(scans the model was trained on - not a fair test)")
    skipped = files_df[files_df["status"] != "ok"]
    if len(skipped):
        log(f"  skipped {len(skipped)} file(s) with no scan in them: "
            + ", ".join(skipped["file"]))
    return files_df, det_df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--conf", type=float, default=crack_model.CONF)
    a = ap.parse_args()
    build(a.out, a.conf,
          progress=lambda i, n: print(f"\r  {i}/{n}", end="", flush=True) or
          (print() if i == n else None))


if __name__ == "__main__":
    main()
