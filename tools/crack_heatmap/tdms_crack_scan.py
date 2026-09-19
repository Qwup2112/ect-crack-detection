#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tdms_crack_scan.py - Backend for reading TDMS 2D scan images (R/P/PECT eddy-current)
and producing the standard crack-detection figures, plus sensor comparison tables.

TDMS structure of this dataset (verified by code, not assumed):
    Group "Freq_Sampling_SizeX_SizeY" -> 1 channel, 4 values [freq_Hz, sampling, sizeX, sizeY]
    Group "Waveform"                  -> 1 channel, flat array sizeX*sizeY (raster: X fast, Y slow)
Filenames encode the measurement parameters:
    <sample>_<sensor>_amp_<V>V_fre_<kHz>k_lf_<mm>mm_<W>x<H>mm.tdms
    sensor: R (amplitude), P (phase), or PECT/Differential/Feedback
    ("_default" files are usually empty templates)

FOUR COMMANDS
  report   one file   -> 6 panels: Raw before filter, Filtered, Cropped,
                        Amplitude heatmap, X-Derivative, X-Derivative Inverse
  detect   one file   -> automatic border-noise crop + crack localization,
                        writes auto_boxes.json for use as --boxes in compare
  sweep    a tree     -> analyze EVERY .tdms found, one row per file, ranked
  compare  many files -> sensor/frequency comparison table (real CNR with --boxes,
                        otherwise descriptive stats + derivative contrast)

EXAMPLES
  python tdms_crack_scan.py report DATA/4lop_R_amp_1.3V_fre_200k_lf_1mm_145x95mm.tdms
  python tdms_crack_scan.py detect DATA/4lop_P_amp_1.3V_fre_200k_lf_1mm_145x95mm.tdms
  python tdms_crack_scan.py sweep . --out sweep_all.csv
  python tdms_crack_scan.py compare "DATA/*.tdms" --boxes auto_boxes.json --out sensor_compare.csv

INSTALL: pip install npTDMS numpy scipy matplotlib pandas
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from nptdms import TdmsFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import matplotlib.patches as mpatches

try:
    from scipy.ndimage import (median_filter, gaussian_filter, label,
                               find_objects, binary_dilation)
except ImportError:
    median_filter = gaussian_filter = label = find_objects = binary_dilation = None

INK, MUTED, SURFACE = "#0b0b0b", "#6b6a64", "#fcfcfb"

# This dataset uses 5 different naming conventions across measurement sessions:
#   4lop_R_amp_1.3V_fre_200k_lf_1mm_145x95mm   (standard, with plate size)
#   4lop_R_amp_4V_fre_10k_lf_1mm               (no plate size)
#   R_amp_3V_freq_20K_liffoff_1mm              ('freq', capital 'K', 'liffoff' - a typo)
#   R_amp_4V_fre_100k_lf_1mm_tron              ('_tron' suffix: different defect shape)
#   R_default / Default_Differential           (template files, no parameters)
# If the regex does not cover these variants, 18 files silently lose their parameters.
NAME_RE = re.compile(
    r"(?P<sensor>[A-Za-z]+)_amp_(?P<amp>[\d.]+)V"
    r"_(?:fre|freq)_(?P<freq>[\d.]+)[kK]"
    r"_(?:lf|lif+off)_(?P<lf>[\d.]+)mm"
    r"(?:_(?P<w>\d+)x(?P<h>\d+)mm)?"
    r"(?P<variant>_tron)?"
)


# ---------------------------------------------------------------- reading

def parse_name(path):
    """Extract sensor/amp/freq/lift-off/plate size from the filename. {} if no match."""
    m = NAME_RE.search(Path(path).stem)
    return m.groupdict() if m else {}


def load_scan(path):
    """
    Read a TDMS file -> 2D grid (Y rows, X columns) + metadata.
    Raises a clear error instead of returning wrong data when a group,
    channel, or sample count is missing.
    """
    tdms = TdmsFile.read(path)
    group_names = {g.name for g in tdms.groups()}
    if "Freq_Sampling_SizeX_SizeY" not in group_names or "Waveform" not in group_names:
        raise ValueError(
            f"{Path(path).name}: missing group 'Freq_Sampling_SizeX_SizeY' or 'Waveform' "
            "(empty/template file, e.g. the '_default' files) - skipping."
        )
    meta = np.asarray(tdms["Freq_Sampling_SizeX_SizeY"].channels()[0][:], dtype=float)
    if meta.size < 4:
        raise ValueError(f"{Path(path).name}: metadata channel is short (need 4, got {meta.size}).")
    freq_hz, sampling, size_x, size_y = meta[:4]
    size_x, size_y = int(round(size_x)), int(round(size_y))

    raw = np.asarray(tdms["Waveform"].channels()[0][:], dtype=float)
    expected = size_x * size_y
    truncated_rows = 0
    if raw.size != expected:
        # The recording was cut short. Do NOT pad with NaN to reach the declared size:
        # a padded row is all-NaN, which makes that row's median/MAD NaN and propagates
        # into both detrending and the z-score. Drop the INCOMPLETE rows instead and
        # state exactly how many were lost.
        full_rows = int(raw.size // size_x)
        truncated_rows = size_y - full_rows
        print(f"  [WARNING] {Path(path).name}: sample count ({raw.size}) != sizeX*sizeY "
              f"({size_x}x{size_y}={expected}). File is truncated - dropping "
              f"{truncated_rows} incomplete scan row(s), analyzing {full_rows}/{size_y} rows.",
              file=sys.stderr)
        if full_rows < 3:
            raise ValueError(
                f"{Path(path).name}: only {full_rows} complete scan rows - too few to analyze.")
        raw = raw[:full_rows * size_x]
        size_y = full_rows

    grid = raw.reshape(size_y, size_x)

    fname_info = parse_name(path)
    pitch = 1.0  # mm/pixel - default; stated as an ASSUMPTION when the filename lacks a size
    if fname_info.get("w") and fname_info.get("h"):
        w_mm, h_mm = float(fname_info["w"]), float(fname_info["h"])
        if size_x and size_y:
            pitch_x = w_mm / size_x
            pitch_y = h_mm / size_y
            if abs(pitch_x - pitch_y) / max(pitch_x, pitch_y) > 0.15:
                print(f"  [WARNING] {Path(path).name}: grid pitch differs a lot between X "
                      f"({pitch_x:.3f} mm/px) and Y ({pitch_y:.3f} mm/px) - "
                      "pixels may not be square.", file=sys.stderr)
            pitch = (pitch_x + pitch_y) / 2.0

    return grid, {
        "path": str(path), "freq_hz": freq_hz, "sampling": sampling,
        "size_x": size_x, "size_y": size_y, "pitch_mm": pitch,
        "truncated_rows": truncated_rows,
        **fname_info,
    }


# ---------------------------------------------------------------- image processing

def apply_filter(img, method="median", kernel=3):
    """Denoise. 'none' keeps the data as-is - used as the reference against raw."""
    if method == "none" or kernel <= 1:
        return img.copy()
    if median_filter is None:
        print("  [WARNING] scipy is not installed, skipping the filter step.", file=sys.stderr)
        return img.copy()
    if method == "median":
        return median_filter(img, size=kernel, mode="nearest")
    if method == "gaussian":
        return gaussian_filter(img, sigma=kernel / 2.0, mode="nearest")
    raise ValueError(f"invalid filter method: {method}")


def crop_margin(img, margin):
    """Trim the border so filter/smoothing edge artifacts are excluded (criterion A2)."""
    if margin <= 0:
        return img
    H, W = img.shape
    if margin * 2 >= min(H, W):
        raise ValueError(f"margin={margin} is too large for a {H}x{W} image")
    return img[margin:H - margin, margin:W - margin]


def x_derivative(img):
    """Derivative along X (columns) - the fast scan axis, where crack edges show best."""
    return np.gradient(img, axis=1)


def detrend_row(img):
    """Subtract the per-row median - removes amplitude decay along the scan axis (A1)."""
    return img - np.nanmedian(img, axis=1, keepdims=True)


def detrend(img, method="row"):
    """
    Remove the background trend. Only the right direction actually helps:
      row  - subtract per-row median (removes decay along Y, the usual case)
      col  - subtract per-column median (removes a trend along X, e.g. sensor P has
             a bright background on both left and right that a row detrend never touches)
      both - remove in both directions
      poly - subtract a 2nd-order polynomial fitted along Y
      none - leave unchanged
    """
    if method == "none":
        return img.copy()
    if method == "row":
        return detrend_row(img)
    if method == "col":
        return img - np.nanmedian(img, axis=0, keepdims=True)
    if method == "both":
        return detrend(detrend(img, "row"), "col")
    if method == "poly":
        y = np.arange(img.shape[0], dtype=float)
        prof = np.nanmedian(img, axis=1)
        ok = np.isfinite(prof)
        if ok.sum() < 3:
            return detrend_row(img)
        coef = np.polyfit(y[ok], prof[ok], 2)
        return img - np.polyval(coef, y)[:, None]
    raise ValueError(f"invalid detrend method: {method}")


def _robust_z(a):
    """z-score from median/MAD rather than mean/std: cracks and border artifacts are
    themselves extreme values, so mean/std would let them drag the scale toward
    themselves and hide what we are trying to measure."""
    med = np.nanmedian(a)
    mad = np.nanmedian(np.abs(a - med)) * 1.4826
    if not np.isfinite(mad) or mad == 0:
        return np.zeros_like(a, dtype=float)
    return (a - med) / mad


def auto_border_margin(img, z_thresh=3.0, max_frac=0.15):
    """
    Measure the noisy border automatically, then trim it.

    Border noise shows up in two different ways, so both must be measured:
      - LEVEL: a row/column whose median is clearly offset (e.g. the bright band at
        the start of a scan line),
      - SPREAD: a row/column with abnormally large dispersion (e.g. the streaks where
        the probe turns around) while its median still looks normal - measuring level
        alone would miss these entirely.
    Compute a robust z-score for both profiles, then walk inward FROM EACH EDGE while
    the deviation stays large, stopping at the first "normal" row/column. That way only
    the contiguous border band is trimmed and real signal in the middle is never touched.

    Trimming is capped at max_frac per side, so a fully noisy image is not wiped out.
    """
    H, W = img.shape
    flat = detrend_row(img)  # remove the scan-axis trend before measuring spread
    row_z = np.maximum(np.abs(_robust_z(np.nanmedian(img, axis=1))),
                       np.abs(_robust_z(np.nanstd(flat, axis=1))))
    col_z = np.maximum(np.abs(_robust_z(np.nanmedian(img, axis=0))),
                       np.abs(_robust_z(np.nanstd(flat, axis=0))))

    def run_length(vals, limit):
        n = 0
        for v in vals[:limit]:
            if not np.isfinite(v) or v <= z_thresh:
                break
            n += 1
        return n

    max_row, max_col = int(H * max_frac), int(W * max_frac)
    m = {
        "top": run_length(row_z, max_row),
        "bottom": run_length(row_z[::-1], max_row),
        "left": run_length(col_z, max_col),
        "right": run_length(col_z[::-1], max_col),
    }

    out = img[m["top"]:H - m["bottom"], m["left"]:W - m["right"]]
    if out.shape[0] < 10 or out.shape[1] < 10:
        print("  [WARNING] auto-crop would remove almost the whole image - "
              "falling back to a fixed 5px crop.", file=sys.stderr)
        m = {"top": 5, "bottom": 5, "left": 5, "right": 5}
        out = crop_margin(img, 5)
    return out, m


def detect_cracks(img, z_thresh=2.5, min_area=6, edge=0, merge_gap=3, detrend_method="row"):
    """
    Locate crack candidates without hand-drawn labels.

    Pipeline: remove the background trend (scan amplitude decays strongly along the scan
    axis; without removing it, any threshold is dominated by that trend) -> robust
    z-score -> threshold on |z| -> MERGE nearby fragments -> bounding box per cluster.

    The merge step is essential: one crack's signal usually breaks into several separate
    horizontal streaks (the signal oscillates row to row), so labelling directly would
    count a single crack as dozens of candidates. Merging dilates the mask by merge_gap
    px so fragments closer than that join up, but each bounding box is still computed
    from the ORIGINAL pixels, so box sizes are not inflated.

    Returns (boxes, zmap, mask). Each box carries 'score' = the largest |z| in its
    cluster, for ranking confidence. This is a THRESHOLD detector, not a machine-learning
    model: it only says "this spot deviates from the background abnormally". Confirming
    that a spot is really a crack still requires the specimen drawing.
    """
    if label is None:
        raise RuntimeError("scipy is required for connected-component labelling: pip install scipy")

    flat = detrend(img, detrend_method)
    zmap = _robust_z(flat)
    mask = np.abs(zmap) > z_thresh
    if edge > 0:
        keep = np.zeros_like(mask)
        keep[edge:-edge, edge:-edge] = True
        mask &= keep

    if merge_gap > 0:
        k = 2 * merge_gap + 1
        grouped = binary_dilation(mask, structure=np.ones((k, k), dtype=bool))
    else:
        grouped = mask

    labeled, n = label(grouped)
    boxes = []
    for i in range(1, n + 1):
        comp = (labeled == i) & mask      # real pixels only, not the dilated halo
        ys, xs = np.nonzero(comp)
        if ys.size == 0:
            continue
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        if (y1 - y0) * (x1 - x0) < min_area:
            continue
        boxes.append({
            "id": "", "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "score": round(float(np.abs(zmap[comp]).max()), 2),
            "area_px": int((y1 - y0) * (x1 - x0)),
            "n_px": int(ys.size),
            # touching the image edge -> most likely a scan artifact that survived the
            # border crop; flag it rather than silently dropping it, because a genuine
            # crack near the plate edge also touches the edge
            "touches_edge": bool(x0 == 0 or y0 == 0 or x1 == img.shape[1] or y1 == img.shape[0]),
        })
    boxes.sort(key=lambda b: (b["y0"], b["x0"]))
    for i, b in enumerate(boxes, start=1):
        b["id"] = f"crack_{i}"
    return boxes, zmap, mask


# ---------------------------------------------------------------- figures

def _panel(ax, img, title, cmap, norm=None, vmin=None, vmax=None, pitch=1.0, cbar_label=""):
    H, W = img.shape
    extent = [0, W * pitch, H * pitch, 0]
    im = ax.imshow(img, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
                    extent=extent, aspect="equal", interpolation="nearest")
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=6)
    ax.set_xlabel("X (mm)", color=MUTED, fontsize=8)
    ax.set_ylabel("Y (mm)", color=MUTED, fontsize=8)
    ax.tick_params(colors=MUTED, labelsize=7)
    for s in ax.spines.values():
        s.set_color("#c3c2b7")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=7, colors=MUTED)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=8, color=MUTED)


def resolve_crop(filtered, margin_arg, auto_z=3.0):
    """margin='auto' -> measure the noisy border; margin=<n> -> fixed crop."""
    if str(margin_arg).lower() == "auto":
        cropped, m = auto_border_margin(filtered, z_thresh=auto_z)
        label_txt = (f"auto: top {m['top']} bottom {m['bottom']} "
                     f"left {m['left']} right {m['right']} px")
    else:
        n = int(margin_arg)
        cropped = crop_margin(filtered, n)
        m = {"top": n, "bottom": n, "left": n, "right": n}
        label_txt = f"fixed {n}px per side"
    return cropped, m, label_txt


def build_report_figure(raw, filtered, cropped, meta, crop_label, filter_label, title):
    """Build the 6-panel figure. Shared by the CLI and the desktop app."""
    dx = x_derivative(cropped)
    dx_abs = float(np.nanmax(np.abs(dx))) or 1.0
    diverge_norm = TwoSlopeNorm(vcenter=0.0, vmin=-dx_abs, vmax=dx_abs)
    pitch = meta["pitch_mm"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), facecolor=SURFACE)
    fig.suptitle(title, fontsize=12, color=INK, y=0.99)

    _panel(axes[0, 0], raw, "1. Raw (before filter)", "viridis", pitch=pitch,
           cbar_label="Amplitude (a.u.)")
    _panel(axes[0, 1], filtered, f"2. Filtered ({filter_label})", "viridis",
           pitch=pitch, cbar_label="Amplitude (a.u.)")
    _panel(axes[0, 2], cropped, f"3. Cropped ({crop_label})", "viridis",
           pitch=pitch, cbar_label="Amplitude (a.u.)")
    _panel(axes[1, 0], cropped, "4. Amplitude heatmap (filtered + cropped)", "magma",
           pitch=pitch, cbar_label="Amplitude (a.u.)")
    _panel(axes[1, 1], dx, "5. X-Derivative (dA/dx)", "RdBu_r", norm=diverge_norm,
           pitch=pitch, cbar_label="dA/dx")
    _panel(axes[1, 2], -dx, "6. X-Derivative Inverse (-dA/dx)", "RdBu_r", norm=diverge_norm,
           pitch=pitch, cbar_label="-dA/dx")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig, dx, dx_abs


def build_detect_figure(cropped, zmap, mask, boxes, meta, title, z_thresh, detrend_label):
    """Build the 3-panel detection figure. Shared by the CLI and the desktop app."""
    pitch = meta["pitch_mm"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), facecolor=SURFACE)
    fig.suptitle(title, fontsize=11, color=INK, y=1.0)

    _panel(axes[0], cropped, "1. Cropped heatmap + located cracks", "magma",
           pitch=pitch, cbar_label="Amplitude (a.u.)")
    for b in boxes:
        axes[0].add_patch(mpatches.Rectangle(
            (b["x0"] * pitch, b["y0"] * pitch),
            (b["x1"] - b["x0"]) * pitch, (b["y1"] - b["y0"]) * pitch,
            fill=False, linewidth=1.2,
            edgecolor="#ff9d3a" if b["touches_edge"] else "#39ff88",
            linestyle="--" if b["touches_edge"] else "-"))

    zlim = float(np.nanmax(np.abs(zmap))) or 1.0
    _panel(axes[1], zmap, f"2. z-score map (detrend: {detrend_label})", "RdBu_r",
           norm=TwoSlopeNorm(vcenter=0.0, vmin=-zlim, vmax=zlim),
           pitch=pitch, cbar_label="z (robust)")
    _panel(axes[2], mask.astype(float), f"3. Mask |z| > {z_thresh}", "gray_r",
           pitch=pitch, cbar_label="1 = candidate")

    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return fig


def cmd_report(args):
    grid, meta = load_scan(args.tdms)
    raw = grid
    filtered = apply_filter(raw, args.filter, args.kernel)
    cropped, margins, crop_label = resolve_crop(filtered, args.margin, args.auto_margin_z)

    title = (f"Crack scan report - {Path(args.tdms).name}\n"
             f"sensor={meta.get('sensor','?')}  amp={meta.get('amp','?')}V  "
             f"freq={meta.get('freq','?')}kHz  grid={meta['size_x']}x{meta['size_y']}px")
    fig, dx, dx_abs = build_report_figure(
        raw, filtered, cropped, meta, crop_label, f"{args.filter}, k={args.kernel}", title)

    outdir = Path(args.tdms).parent / f"{Path(args.tdms).stem}_results"
    outdir.mkdir(exist_ok=True)
    fig_path = outdir / "crack_scan_report.png"
    fig.savefig(fig_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)

    np.save(outdir / "cropped_filtered.npy", cropped)
    np.save(outdir / "x_derivative.npy", dx)

    stats = {
        "file": Path(args.tdms).name, **meta,
        "raw_min": float(np.nanmin(raw)), "raw_max": float(np.nanmax(raw)),
        "raw_std": float(np.nanstd(raw)),
        "cropped_std": float(np.nanstd(cropped)),
        "dx_absmax": dx_abs,
    }
    pd.DataFrame([stats]).to_csv(outdir / "scan_stats.csv", index=False, encoding="utf-8-sig")

    print(f"Saved: {fig_path}")
    print(f"  cropped_filtered.npy, x_derivative.npy, scan_stats.csv  (in {outdir})")
    print(f"\n  Original grid : {meta['size_x']}x{meta['size_y']} px "
          f"(~{meta['pitch_mm']:.3f} mm/px; ASSUMED when the filename has no plate size)")
    print(f"  Border crop   : {crop_label}  ->  {cropped.shape[1]}x{cropped.shape[0]} px left")
    print(f"  Raw amplitude : min {stats['raw_min']:.4g}  max {stats['raw_max']:.4g}  "
          f"std {stats['raw_std']:.4g}")
    print(f"  Std after filter+crop: {stats['cropped_std']:.4g}")
    print("\n  LIMITS: this command only draws figures, it does not locate cracks."
          " Use the 'detect' command for positions + auto_boxes.json.")


# ---------------------------------------------------------------- detect

def cmd_detect(args):
    grid, meta = load_scan(args.tdms)
    filtered = apply_filter(grid, args.filter, args.kernel)
    cropped, margins, crop_label = resolve_crop(filtered, args.margin, args.auto_margin_z)

    boxes, zmap, mask = detect_cracks(cropped, z_thresh=args.z_thresh,
                                      min_area=args.min_area, edge=args.edge,
                                      merge_gap=args.merge_gap,
                                      detrend_method=args.detrend)

    n_edge = sum(b["touches_edge"] for b in boxes)
    if args.drop_edge_boxes:
        boxes = [b for b in boxes if not b["touches_edge"]]
        for i, b in enumerate(boxes, start=1):
            b["id"] = f"crack_{i}"

    pitch = meta["pitch_mm"]
    outdir = Path(args.tdms).parent / f"{Path(args.tdms).stem}_results"
    outdir.mkdir(exist_ok=True)

    payload = {
        "source": Path(args.tdms).name,
        "crop": margins,
        "note": ("Coordinates are pixels of the CROPPED image, not the original. "
                 "Add crop.left/crop.top to convert to original-image coordinates."),
        "z_thresh": args.z_thresh,
        "boxes": boxes,
    }
    (outdir / "auto_boxes.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # flat version, usable directly as --boxes for compare / scan_metrics.py
    (outdir / "auto_boxes_flat.json").write_text(
        json.dumps([{k: b[k] for k in ("id", "x0", "y0", "x1", "y1")} for b in boxes],
                   ensure_ascii=False, indent=2), encoding="utf-8")

    title = (f"Crack localization - {Path(args.tdms).name}\n"
             f"sensor={meta.get('sensor','?')}  freq={meta.get('freq','?')}kHz  "
             f"crop: {crop_label}  |  threshold |z|>{args.z_thresh}  |  "
             f"{len(boxes)} candidates")
    fig = build_detect_figure(cropped, zmap, mask, boxes, meta, title,
                              args.z_thresh, args.detrend)
    fig_path = outdir / "crack_detect_report.png"
    fig.savefig(fig_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)

    print(f"Border crop: {crop_label}  ->  {cropped.shape[1]}x{cropped.shape[0]} px "
          f"(original {meta['size_x']}x{meta['size_y']})")
    print(f"Found {len(boxes)} crack candidates (threshold |z| > {args.z_thresh}, "
          f"minimum area {args.min_area} px)\n")

    if boxes:
        df = pd.DataFrame(boxes)
        df["x_mm"] = (df["x0"] * pitch).round(1)
        df["y_mm"] = (df["y0"] * pitch).round(1)
        df["w_px"] = df["x1"] - df["x0"]
        df["h_px"] = df["y1"] - df["y0"]
        cols = ["id", "x0", "y0", "x1", "y1", "w_px", "h_px", "x_mm", "y_mm",
                "area_px", "n_px", "score", "touches_edge"]
        print(df[cols].to_string(index=False))
        df[cols].to_csv(outdir / "auto_boxes.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved: {fig_path}")
    print(f"  auto_boxes.json / auto_boxes_flat.json / auto_boxes.csv  (in {outdir})")

    if n_edge:
        verb = "Dropped" if args.drop_edge_boxes else "Kept"
        print(f"\n{verb} {n_edge} candidate(s) TOUCHING THE IMAGE EDGE (orange dashed in the"
              " figure). Scan borders often produce artifacts, so most of these are false"
              " alarms; but a genuine crack near the plate edge also touches the edge, so"
              " look at the figure before deciding."
              + ("" if args.drop_edge_boxes else " Add --drop-edge-boxes to remove them."))

    if not boxes:
        print("\nNO candidate passed the threshold. Two possibilities, check in this order:")
        print("  1. Threshold too high -> lower --z-thresh (e.g. 2.0) and re-run.")
        print("  2. Signal too weak    -> run 'compare' to see this sensor's contrast;"
              " if CNR < 1 the problem is in the measurement, not the threshold.")
    elif len(boxes) > 60:
        print(f"\nWARNING: {len(boxes)} candidates is far more than the number of real cracks"
              " usually present - the threshold is too low or background noise remains."
              " Raise --z-thresh or --min-area and re-run.")

    print("\nLIMITS: this is a THRESHOLD detector (z-score after detrending), not a machine"
          " learning model. It marks 'abnormal deviation from the background' - whether that"
          " is a crack, a plate edge, or a scan artifact must be checked against the"
          " specimen drawing before drawing conclusions.")


# ---------------------------------------------------------------- sweep

def analyze_one(path, filter_method="median", kernel=3, margin="auto",
                detrend_method="row", z_thresh=2.5, min_area=6, merge_gap=2,
                auto_margin_z=3.0):
    """
    Run the whole pipeline on ONE file and return a single row of numbers (no figures).
    Shared by the 'sweep' command (CLI) and the "Analyze all data" button (GUI).

    The detectability metric here uses the z SCALE (median/MAD), not dx_std: z is already
    normalized by each image's own background noise, so it compares fairly between
    sensor R (~0.01 scale) and sensor P (~30 scale). dx_std does not - it simply rewards
    whichever file has the larger value scale, which ranks them backwards versus real CNR.
    """
    p = Path(path)
    row = {"folder": p.parent.name, "file": p.name, "status": "ok"}
    info = parse_name(p) or {}
    for k in ("sensor", "amp", "freq", "lf"):
        row[k] = info.get(k)
    row["variant"] = (info.get("variant") or "").lstrip("_")

    try:
        grid, meta = load_scan(p)
    except ValueError as e:
        row["status"] = ("skipped: too few complete scan rows" if "complete scan rows" in str(e)
                         else "skipped: empty/template file (no Waveform)")
        return row
    except Exception as e:
        row["status"] = f"read error: {type(e).__name__}"
        return row

    if meta.get("truncated_rows"):
        row["status"] = f"ok (TRUNCATED: lost {meta['truncated_rows']} scan rows)"

    filtered = apply_filter(grid, filter_method, kernel)
    cropped, m, _ = resolve_crop(filtered, margin, auto_margin_z)
    boxes, zmap, _ = detect_cracks(cropped, z_thresh=z_thresh, min_area=min_area,
                                   merge_gap=merge_gap, detrend_method=detrend_method)
    flat = detrend(filtered, detrend_method)
    scores = sorted((b["score"] for b in boxes), reverse=True)

    row.update({
        "size_px": f"{meta['size_x']}x{meta['size_y']}",
        "crop_top": m["top"], "crop_bottom": m["bottom"],
        "crop_left": m["left"], "crop_right": m["right"],
        "cropped_px": f"{cropped.shape[1]}x{cropped.shape[0]}",
        "n_boxes": len(boxes),
        "n_strong": sum(1 for s in scores if s >= 4.0),
        "n_edge_boxes": sum(1 for b in boxes if b["touches_edge"]),
        "z_max": round(scores[0], 2) if scores else 0.0,
        "z_top5": round(float(np.mean(scores[:5])), 2) if scores else 0.0,
        "std_raw": float(f"{np.nanstd(grid):.4g}"),
        "dynamic_range": float(f"{np.nanmax(grid) - np.nanmin(grid):.4g}"),
        "std_detrended": float(f"{np.nanstd(flat):.4g}"),
    })
    return row


def sweep_summary(df):
    """Summarize a sweep table -> (ok, skipped, top15, best_per_group). Shared CLI + GUI."""
    # Truncated files are still analyzable (status 'ok (TRUNCATED: ...)'), so match with
    # startswith rather than equality - otherwise they get misfiled as skipped.
    is_ok = df["status"].str.startswith("ok")
    ok = df[is_ok].copy()
    skipped = df[~is_ok]
    if ok.empty:
        return ok, skipped, ok, ok
    for col in ("freq", "amp"):
        ok[col] = pd.to_numeric(ok[col], errors="coerce")
    top = ok.nlargest(15, "z_top5")
    best = (ok.sort_values("z_top5", ascending=False)
              .groupby(["folder", "sensor"], as_index=False).first())
    return ok, skipped, top, best


def cmd_sweep(args):
    paths = []
    for r in (args.paths or ["."]):
        rp = Path(r)
        paths.extend(sorted(rp.rglob("*.tdms")) if rp.is_dir()
                     else sorted(Path().glob(r)))
    paths = list(dict.fromkeys(paths))
    if not paths:
        sys.exit(f"No .tdms files found in: {args.paths}")

    print(f"Scanning {len(paths)} files...\n")
    rows = []
    for i, p in enumerate(paths, start=1):
        row = analyze_one(p, filter_method=args.filter, kernel=args.kernel,
                          margin=args.margin, detrend_method=args.detrend,
                          z_thresh=args.z_thresh, min_area=args.min_area,
                          merge_gap=args.merge_gap)
        rows.append(row)
        print(f"  [{i}/{len(paths)}] {row['file'][:58]:<60} {row['status']}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    ok, skipped, top, best = sweep_summary(df)

    print(f"\n{'='*76}\nAnalyzed {len(ok)}/{len(df)} files"
          + (f", skipped {len(skipped)}" if len(skipped) else ""))
    for _, r in skipped.iterrows():
        print(f"  - {r['file']}: {r['status']}")
    if ok.empty:
        return

    cols = ["folder", "sensor", "amp", "freq", "variant", "n_boxes", "n_strong",
            "z_max", "z_top5"]
    print(f"\n{'='*76}\nTOP 15 BY z_top5 (signal separation from background, normalized)")
    print(top[cols].to_string(index=False))

    print(f"\n{'='*76}\nBEST CONFIGURATION PER (folder, sensor)")
    print(best[["folder", "sensor", "amp", "freq", "z_top5", "n_strong"]]
          .to_string(index=False))

    print(f"\nFull table saved: {args.out}")
    print("\nNOTE: z_top5 measures DEVIATION from each image's own background, not CNR")
    print("against real cracks. It answers 'does this image contain standout structure',")
    print("NOT 'is that structure actually a crack'. Confirming that needs real labels.")


# ---------------------------------------------------------------- compare

def cnr_for_boxes(img, boxes, edge=5, ring=5):
    """Per-ROI CNR - same formula as scan_metrics.py so both tools agree."""
    H, W = img.shape
    valid = np.zeros_like(img, dtype=bool)
    if edge > 0:
        valid[edge:H - edge, edge:W - edge] = True
    else:
        valid[:] = True
    all_boxes = np.zeros_like(img, dtype=bool)
    for b in boxes:
        all_boxes[b["y0"]:b["y1"], b["x0"]:b["x1"]] = True
    vals = []
    for b in boxes:
        x0, y0, x1, y1 = b["x0"], b["y0"], b["x1"], b["y1"]
        if not (0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H):
            continue
        roi = img[y0:y1, x0:x1]
        oy0, oy1 = max(0, y0 - ring), min(H, y1 + ring)
        ox0, ox1 = max(0, x0 - ring), min(W, x1 + ring)
        outer = np.zeros_like(img, dtype=bool)
        outer[oy0:oy1, ox0:ox1] = True
        outer[y0:y1, x0:x1] = False
        ring_px = img[outer & valid & ~all_boxes]
        roi = roi[np.isfinite(roi)]
        ring_px = ring_px[np.isfinite(ring_px)]
        if roi.size == 0 or ring_px.size < 10:
            continue
        sd = ring_px.std()
        vals.append(abs(roi.mean() - ring_px.mean()) / sd if sd > 0 else np.inf)
    return float(np.mean(vals)) if vals else float("nan")


def load_boxes_for_compare(path):
    """
    Read a box file and CONVERT IT TO ORIGINAL-IMAGE COORDINATES.

    'detect' writes coordinates relative to the CROPPED image, while 'compare' measures
    on the full image. Feeding them in unchanged would offset every label by exactly the
    border width that was cropped - the classic "misaligned labels" fault that produces
    recall 0 and gets blamed on the model (criterion C2). auto_boxes.json carries a
    'crop' block precisely so the offset can be added back here.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "boxes" in data:
        crop = data.get("crop") or {}
        dx, dy = int(crop.get("left", 0)), int(crop.get("top", 0))
        boxes = [dict(b) for b in data["boxes"]]
        if dx or dy:
            for b in boxes:
                b["x0"] += dx; b["x1"] += dx
                b["y0"] += dy; b["y1"] += dy
            print(f"  Converted {len(boxes)} boxes to original-image coordinates "
                  f"(crop offset: +{dx} px in X, +{dy} px in Y).")
        return boxes
    print("  [NOTE] Box file is a flat list - ASSUMING coordinates are in ORIGINAL-image"
          " space. If it came from 'detect', use auto_boxes.json instead (the full version"
          " with the 'crop' block) so the offset is applied automatically.", file=sys.stderr)
    return data


def cmd_compare(args):
    paths = sorted(set(sum((glob.glob(p) for p in args.tdms), [])))
    if not paths:
        sys.exit(f"No files matched: {args.tdms}")

    boxes = None
    if args.boxes:
        boxes = load_boxes_for_compare(args.boxes)

    rows = []
    for p in paths:
        try:
            grid, meta = load_scan(p)
        except ValueError as e:
            print(f"  [SKIPPED] {e}", file=sys.stderr)
            continue
        filtered = apply_filter(grid, args.filter, args.kernel)
        flat = detrend_row(filtered)
        dx = x_derivative(filtered)

        row = {
            "file": Path(p).name,
            "sensor": meta.get("sensor", "?"),
            "amp_V": meta.get("amp", ""),
            "freq_kHz": meta.get("freq", ""),
            "lf_mm": meta.get("lf", ""),
            "size_px": f"{meta['size_x']}x{meta['size_y']}",
            "mean": float(np.nanmean(grid)),
            "std": float(np.nanstd(grid)),
            "rms": float(np.sqrt(np.nanmean(grid ** 2))),
            "dynamic_range": float(np.nanmax(grid) - np.nanmin(grid)),
            "std_after_row_detrend": float(np.nanstd(flat)),
            "dx_std": float(np.nanstd(dx)),
        }
        if boxes:
            # CNR on signed values AND on absolute deviation.
            # An ECT crack signature is a DIPOLE (a bright lobe beside a dark lobe): on
            # signed values the two lobes cancel, so the box mean lands near the background
            # and CNR collapses toward 0 even when the signal is obvious. CNR_abs does not
            # suffer from this. CNR_abs >> CNR is exactly the fingerprint of a dipole.
            row["CNR"] = cnr_for_boxes(flat, boxes, edge=args.edge)
            row["CNR_abs"] = cnr_for_boxes(np.abs(flat), boxes, edge=args.edge)
        rows.append(row)

    if not rows:
        sys.exit("No file could be read successfully.")

    df = pd.DataFrame(rows)
    sort_col = "CNR_abs" if boxes else "dx_std"
    df = df.sort_values(sort_col, ascending=False, na_position="last")

    out_csv = Path(args.out)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"Compared {len(df)} files  (sorted by {sort_col}, descending)\n")
    print(df.to_string(index=False))
    print(f"\nTable saved: {out_csv}")

    if boxes:
        best = df.iloc[0]
        print(f"\nBest sensor/condition by CNR_abs: {best['file']} "
              f"(CNR_abs={best['CNR_abs']:.2f}, signed CNR={best['CNR']:.2f})")

        dipole = df[df["CNR_abs"] > 1.5 * df["CNR"].clip(lower=1e-9)]
        if len(dipole):
            print(f"\n{len(dipole)}/{len(df)} files have CNR_abs far above signed CNR"
                  " -> the crack signature is a DIPOLE (bright lobe beside dark lobe).")
            print("  For this signal shape, do NOT read the signed CNR: the two lobes cancel"
                  " inside the box so it is always artificially low. Read CNR_abs - and note"
                  " that a solid box label misdescribes the phenomenon if you go on to train"
                  " a model on it (criterion B4).")

        if df["CNR_abs"].max() < 1:
            print("\nWARNING: no condition reaches CNR_abs >= 1 - the signal is below the noise"
                  " in EVERY configuration tried. The problem is in the measurement"
                  " (frequency, channel, scan resolution), not in the model (criteria B1/B2).")
        elif df["CNR_abs"].max() < 3:
            print("\nNOTE: the highest CNR_abs is still under 3 - the signal is weak but real."
                  " Improve preprocessing / channel combination before changing the model.")
    else:
        print("\nNOTE: without --boxes this is a RELATIVE index (dx_std), not a true CNR."
              " Supply a boxes.json (same format as scan_metrics.py) to get a real CNR.")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("report", help="one file -> the 6 standard panels + numbers")
    r.add_argument("tdms", help="path to a .tdms file")
    r.add_argument("--filter", default="median", choices=["median", "gaussian", "none"])
    r.add_argument("--kernel", type=int, default=3)
    r.add_argument("--margin", default="auto",
                   help="'auto' (measure the noisy border) or a fixed pixel count")
    r.add_argument("--auto-margin-z", type=float, default=3.0,
                   help="z-score above which an edge row/column counts as noise")
    r.set_defaults(func=cmd_report)

    d = sub.add_parser("detect", help="one file -> border crop + crack localization")
    d.add_argument("tdms", help="path to a .tdms file")
    d.add_argument("--filter", default="median", choices=["median", "gaussian", "none"])
    d.add_argument("--kernel", type=int, default=3)
    d.add_argument("--margin", default="auto",
                   help="'auto' (measure the noisy border) or a fixed pixel count")
    d.add_argument("--auto-margin-z", type=float, default=3.0)
    d.add_argument("--z-thresh", type=float, default=2.5,
                   help="|z| above which a pixel counts as a crack candidate")
    d.add_argument("--min-area", type=int, default=6,
                   help="minimum box area in px - filters out isolated specks")
    d.add_argument("--edge", type=int, default=0,
                   help="drop a further N px at the border after the auto-crop")
    d.add_argument("--merge-gap", type=int, default=2,
                   help="merge fragments closer than 2N px into one crack (0 = no merging)")
    d.add_argument("--drop-edge-boxes", action="store_true",
                   help="remove candidates touching the image edge instead of just flagging")
    d.add_argument("--detrend", default="row", choices=["row", "col", "both", "poly", "none"],
                   help="detrend direction before thresholding ('both' when the background "
                        "slopes along both axes)")
    d.set_defaults(func=cmd_detect)

    s = sub.add_parser("sweep", help="analyze an ENTIRE tree -> table + ranking")
    s.add_argument("paths", nargs="*", default=["."],
                   help="directories (searched recursively) or glob patterns; default '.'")
    s.add_argument("--filter", default="median", choices=["median", "gaussian", "none"])
    s.add_argument("--kernel", type=int, default=3)
    s.add_argument("--margin", default="auto")
    s.add_argument("--detrend", default="row",
                   choices=["row", "col", "both", "poly", "none"])
    s.add_argument("--z-thresh", type=float, default=2.5)
    s.add_argument("--min-area", type=int, default=6)
    s.add_argument("--merge-gap", type=int, default=2)
    s.add_argument("--out", default="sweep_all.csv")
    s.set_defaults(func=cmd_sweep)

    c = sub.add_parser("compare", help="many files -> sensor/frequency comparison table")
    c.add_argument("tdms", nargs="+", help="paths or glob patterns, e.g. 'DATA/*.tdms'")
    c.add_argument("--filter", default="median", choices=["median", "gaussian", "none"])
    c.add_argument("--kernel", type=int, default=3)
    c.add_argument("--edge", type=int, default=5)
    c.add_argument("--boxes", default=None, help="JSON label boxes (scan_metrics.py format)")
    c.add_argument("--out", default="sensor_compare.csv")
    c.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
