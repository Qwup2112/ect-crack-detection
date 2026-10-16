#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tdms_crack_scan.py - Backend for reading TDMS 2D scan images (R/P/PECT eddy-current)
and producing the standard crack-detection figures, plus sensor comparison tables.

TDMS structure of this dataset (verified by code, not assumed):
    Group "Freq_Sampling_SizeX_SizeY" -> 1 channel, 4 values [freq_Hz, sampling, sizeX, sizeY]
    Group "Waveform"                  -> 1 channel, flat array sizeX*sizeY (serpentine raster: X fast,
                                         Y slow, odd lines recorded right-to-left)
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
  python tdms_crack_scan.py report data/cambien_4lop_2/4lop_R_amp_1.3V_fre_200k_lf_1mm_145x95mm.tdms
  python tdms_crack_scan.py detect data/cambien_4lop_2/4lop_P_amp_1.3V_fre_200k_lf_1mm_145x95mm.tdms
  python tdms_crack_scan.py sweep            (all of data/ -> results/sweep_all.csv)
  python tdms_crack_scan.py compare "data/cambien_4lop_2/*.tdms" --boxes auto_boxes.json --out sensor_compare.csv

INSTALL: pip install npTDMS numpy scipy matplotlib pandas
"""

import argparse
import glob
import json
import re
import sys
import textwrap
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

# Project layout: data/<session>/*.tdms (raw scans only), results/<session>/ (generated
# output), scripts/ (lab scripts incl. dataloader.py), tools/crack_heatmap/ (this file).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# dataloader.py is imported lazily: at module level it pulls in scikit-learn, h5py and
# PyWavelets, and the default pipeline must keep working on a machine without them.
_DATALOADER_DIR = PROJECT_ROOT / "scripts"


def result_dir(tdms):
    """Output folder for one scan: results/<session>/<stem>_results for a scan under data/,
    so generated files never mix with raw data; next to the scan for a file elsewhere."""
    p = Path(tdms).resolve()
    try:
        out = RESULTS_DIR / p.parent.relative_to(DATA_DIR) / f"{p.stem}_results"
    except ValueError:
        out = p.parent / f"{p.stem}_results"
    out.mkdir(parents=True, exist_ok=True)
    return out

FILTERS = ["lpf", "median", "gaussian", "wavelet", "none"]
DETRENDS = ["row", "col", "both", "poly", "bg", "surface", "linear", "none"]
LPF_CUTOFF = 0.3
# 'bg' is the only option that gives one box per crack on the 3x5 coupon (compared visually
# against row/surface on the best R and P scans); 'row' merges cracks sharing a column.
# 'lpf' removes the remaining sawtooth. Both are the lab's dataloader.py functions, written
# out here with scipy alone (_lpf_columns, _bg_detrend): importing dataloader.py also loads
# scikit-learn, whose compiled files Windows Smart App Control blocks on this PC ("An
# Application Control policy has blocked this file") - the default pipeline must not depend
# on it. Only the optional surface / linear / wavelet options still import dataloader.py.
DEFAULT_FILTER = "lpf" if median_filter is not None else "median"
DEFAULT_DETREND = "bg" if gaussian_filter is not None else "row"


def _lpf_columns(img, cutoff=LPF_CUTOFF, order=24):
    """dataloader.LPF_1D applied down every column (along Y): Hamming-window FIR, the ends
    padded with 50 copies of the edge value, group delay (order/2) removed. Same numbers as
    dataloader.LPF, vectorised over columns."""
    from scipy.signal import firwin, lfilter
    b = firwin(order + 1, cutoff, window="hamming", pass_zero="lowpass")
    x = np.pad(np.asarray(img, dtype=float), ((50, 50), (0, 0)), mode="edge")
    y = lfilter(b, 1.0, x, axis=0)
    return y[50 + order // 2: y.shape[0] - 50 + order // 2]


def _bg_detrend(img, sigma=8):
    """dataloader.remove_background_trend(img, 'gaussian', sigma_trend=8, sigma_noise=0):
    subtract a heavy Gaussian blur - wider than a crack, narrower than the crack spacing."""
    img = np.asarray(img, dtype=np.float32)
    return (img - gaussian_filter(img, sigma=sigma)).astype(float)


def _dataloader():
    if str(_DATALOADER_DIR) not in sys.path:
        sys.path.insert(0, str(_DATALOADER_DIR))
    try:
        import dataloader
    except ImportError as e:
        blocked = "Application Control" in str(e) or "DLL load failed" in str(e)
        raise RuntimeError(
            f"this option uses dataloader.py ({_DATALOADER_DIR}), which needs "
            f"'{e.name}'"
            + (" - Windows blocked that library's file (Smart App Control). Use the default "
               "'lpf' filter and 'bg' detrend, which do not need it." if blocked else
               ": pip install scikit-learn h5py PyWavelets")) from e
    return dataloader


def preload():
    """Import the compiled libraries the default pipeline uses, on the CALLING thread. The
    app calls this on its main thread at start-up: imported first from a worker thread,
    scipy came up half-initialised on Python 3.14 and the first Draw figures failed.
    Returns an error message, or None."""
    try:
        import scipy.ndimage, scipy.signal  # noqa: E401,F401
    except Exception as e:                  # the app still runs; the option reports it
        return str(e)
    return None

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


def load_scan(path, serpentine=True, fix_direction=True):
    """
    Read a TDMS file -> 2D grid (Y rows, X columns) + metadata.
    serpentine=True un-mirrors the odd scan lines (verified on all 101 readable files in
    this dataset: row-to-row jumps drop 3-8x). Pass False only for a unidirectional scanner.
    fix_direction=True then aligns the two scan directions (see fix_scan_direction).
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
    if serpentine:
        # The scanner runs back and forth (boustrophedon): odd lines are recorded right-to-left.
        # Same mapping as dataloader.reshape_raster. Without this every other row is mirrored,
        # which shows up as a zig-zag / row-to-row oscillation across every crack.
        grid[1::2] = grid[1::2, ::-1]
    row_shift, row_offset = 0, 0.0
    if serpentine and fix_direction:
        grid, row_shift, row_offset = fix_scan_direction(grid)

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
        "row_shift_px": row_shift, "row_offset": row_offset,
        **fname_info,
    }


def _sawtooth(img, trim=12):
    """Row-to-row zig-zag energy: mean |2nd difference along Y|, ignoring the scan-turn edges."""
    f = img[:, trim:-trim] if img.shape[1] > 3 * trim else img
    return float(np.mean(np.abs(f[2:] - 2 * f[1:-1] + f[:-2])))


def fix_scan_direction(grid, max_shift=5):
    """
    Align the right-to-left lines with the left-to-right lines (the "sawtooth" / comb).

    Two faults between the two scan directions, both measured on this dataset:
      - POSITION: the probe lags when it reverses, so return lines sit a few px off
        (cam_4lop_1: exactly 2 px on 26/28 files; early_session: mostly 5 px);
      - LEVEL: the return lines carry a small constant offset.
    The shift is chosen as the one that minimizes the sawtooth itself (tested -max..+max);
    the level is matched by medians. Together they remove 50-60% of the sawtooth.
    Shifted-in columns are filled with the edge value, never wrapped around.
    """
    best = None
    for k in range(-max_shift, max_shift + 1):
        out = grid.copy()
        odd = np.roll(grid[1::2], k, axis=1)
        if k > 0:
            odd[:, :k] = odd[:, [k]]
        elif k < 0:
            odd[:, k:] = odd[:, [k - 1]]
        offset = float(np.median(odd) - np.median(grid[0::2]))
        out[1::2] = odd - offset
        s = _sawtooth(out)
        if best is None or s < best[0]:
            best = (s, out, k, offset)
    return best[1], best[2], best[3]


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
    if method == "lpf":
        # dataloader.LPF (Hamming FIR, group delay removed; see _lpf_columns) down every column, i.e. along Y -
        # the axis where the remaining row-to-row sawtooth lives - then a median of size
        # `kernel`. Cutoff 0.3: sawtooth down to 0.33x on the R 200 kHz scan while the
        # ~15 px tall crack lobes, including the faint deep row, keep their shape.
        smooth = _lpf_columns(img)
        return median_filter(smooth, size=kernel, mode="nearest")
    if method == "wavelet":
        # dataloader.wavelet_denoise: soft-threshold db4 detail coefficients. kernel sets the
        # threshold (x noise sigma). waverec2 pads odd sizes by one, so crop back.
        out = _dataloader().wavelet_denoise(img, wavelet="db4", level=2, threshold_ratio=kernel)
        return out[:img.shape[0], :img.shape[1]]
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
      bg   - dataloader.remove_background_trend: subtract a heavy Gaussian blur (sigma 8 px,
             wider than a crack, narrower than the crack spacing) - removes 2D background
      surface - dataloader.detrend_2d_surface: subtract a fitted 3rd-order 2D surface
      linear  - dataloader.detrend_1d: remove a straight-line trend along rows, then columns
      none - leave unchanged
    """
    if method == "bg":
        return _bg_detrend(img)
    if method == "surface":
        return _dataloader().detrend_2d_surface(img, degree=3)
    if method == "linear":
        return _dataloader().detrend_1d(img)
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

# Colour limits from percentiles, not min/max. With min/max a handful of extreme pixels
# (the bright scan-edge band, one derivative spike) stretch the scale and every crack is
# squeezed into a few pale shades: on the R 200 kHz scan 89% of the dA/dx pixels stayed
# below a quarter of full colour. Pixels beyond the limits show at full colour and the
# colorbar gets arrow ends to say so.
# Axis names on every map (both are scan positions in mm).
X_LABEL = "X"
Y_LABEL = "Y"

CLIP_PCT = 2.0        # sequential maps: colour spans the 2nd..98th percentile
CLIP_PCT_DIV = 99.0   # diverging maps: +/- the 99th percentile of |value|


def robust_limits(img, pct=CLIP_PCT):
    lo, hi = np.nanpercentile(img, [pct, 100 - pct])
    if not hi > lo:                      # flat or binary image: fall back to the full range
        lo, hi = np.nanmin(img), np.nanmax(img)
    return float(lo), float(hi)


def diverging_norm(img, pct=CLIP_PCT_DIV):
    lim = float(np.nanpercentile(np.abs(img), pct)) or float(np.nanmax(np.abs(img))) or 1.0
    return TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim)


def _panel(ax, img, title, cmap, norm=None, vmin=None, vmax=None, pitch=1.0, cbar_label="",
           scale=1.0, wrap=26, robust=True):
    """One heatmap + colorbar. scale enlarges the text for a single full-size view.
    robust=True sets the colour limits from percentiles (see CLIP_PCT)."""
    H, W = img.shape
    extend = "neither"
    if robust:
        if norm is None and vmin is None and vmax is None:
            vmin, vmax = robust_limits(img)
            if np.nanmin(img) < vmin or np.nanmax(img) > vmax:
                extend = "both"
        elif norm is not None and np.nanmax(np.abs(img)) > norm.vmax:
            extend = "both"
    extent = [0, W * pitch, H * pitch, 0]
    im = ax.imshow(img, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
                    extent=extent, aspect="equal", interpolation="bilinear")
    # wrap long titles: a left-aligned title wider than its panel runs into the colorbar
    ax.set_title(textwrap.fill(title, wrap), loc="left", fontsize=10 * scale, color=INK, pad=6)
    ax.set_xlabel(f"{X_LABEL} (mm)", color=MUTED, fontsize=8 * scale)
    ax.set_ylabel(f"{Y_LABEL} (mm)", color=MUTED, fontsize=8 * scale)
    ax.tick_params(colors=MUTED, labelsize=7 * scale)
    for s in ax.spines.values():
        s.set_color("#c3c2b7")
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03, extend=extend)
    cb.ax.tick_params(labelsize=7 * scale, colors=MUTED)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=8 * scale, color=MUTED)


def _draw_view(ax, v, pitch, scale=1.0, wrap=26):
    """Draw one view spec (see report_views / detect_views), including any crack boxes."""
    _panel(ax, v["img"], v["title"], v["cmap"], norm=v.get("norm"), pitch=pitch,
           cbar_label=v.get("cbar", ""), scale=scale, wrap=wrap,
           robust=v.get("robust", True))
    for b in v.get("boxes", ()):
        if b.get("touches_edge"):
            continue                  # the plate edge / scan border, not a crack
        # green = crack found; red = the model's box matches none of the labels
        color = "#ff4d4d" if b.get("match") is False else "#39ff88"
        ax.add_patch(mpatches.Rectangle(
            (b["x0"] * pitch, b["y0"] * pitch),
            (b["x1"] - b["x0"]) * pitch, (b["y1"] - b["y0"]) * pitch,
            fill=False, linewidth=0.7 * scale, edgecolor=color))
        if "conf" in b:
            ax.text(b["x0"] * pitch, b["y0"] * pitch, f"{b['conf']:.2f}", color=color,
                    fontsize=6 * scale, va="bottom", ha="left")
    for b in v.get("missed", ()):     # labels no detection matched: only these are drawn
        ax.add_patch(mpatches.Rectangle(
            (b["x0"] * pitch, b["y0"] * pitch),
            (b["x1"] - b["x0"]) * pitch, (b["y1"] - b["y0"]) * pitch,
            fill=False, linewidth=0.7 * scale, edgecolor="#ff4fd8", linestyle=":"))


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


def signal_name(sensor):
    """What a map's values are: sensor P records the PHASE, R (and others) the amplitude."""
    return "Phase" if sensor == "P" else "Amplitude"


def report_views(raw, filtered, cropped, crop_label, filter_label, sensor=None,
                 margins=None):
    """
    The 6 standard views as specs {key, title, img, cmap, norm, cbar}. Drawn all together
    by build_report_figure, or one at a time and full size by build_view_figure (the app
    shows each view on its own so the image is large enough to read).
    Each view carries "offset" = (left, top): where its pixel (0, 0) sits in the original
    grid (0 for the uncropped raw/filtered maps, the crop margins for the others).
    Returns (views, dx, dx_abs).
    """
    dx = x_derivative(cropped)
    dx_abs = float(np.nanmax(np.abs(dx))) or 1.0
    diverge_norm = diverging_norm(dx)
    word = signal_name(sensor)
    amp = f"{word} (a.u.)"
    views = [
        {"key": "raw", "title": "1. Raw (before filter)", "img": raw,
         "cmap": "viridis", "cbar": amp},
        {"key": "filtered", "title": f"2. Filtered ({filter_label})", "img": filtered,
         "cmap": "viridis", "cbar": amp},
        {"key": "cropped", "title": f"3. Cropped ({crop_label})", "img": cropped,
         "cmap": "viridis", "cbar": amp},
        {"key": "amplitude", "title": f"4. {word} heatmap (filtered + cropped)",
         "img": cropped, "cmap": "magma", "cbar": amp},
        {"key": "dx", "title": "5. X-Derivative (dA/dx)", "img": dx,
         "cmap": "RdBu_r", "norm": diverge_norm, "cbar": "dA/dx"},
        {"key": "dx_inv", "title": "6. X-Derivative Inverse (-dA/dx)", "img": -dx,
         "cmap": "RdBu_r", "norm": diverge_norm, "cbar": "-dA/dx"},
    ]
    off = (margins["left"], margins["top"]) if margins else (0, 0)
    for v in views:
        v["offset"] = (0, 0) if v["key"] in ("raw", "filtered") else off
    return views, dx, dx_abs


def shift_boxes(boxes, offset, shape):
    """Boxes in original grid pixels -> the pixels of a view whose (0, 0) sits at `offset`
    = (left, top) in the original grid, clipped to the view `shape`. Boxes entirely off
    the view (inside a cropped border) are left out. Other keys (conf, match) are kept."""
    ox, oy = offset
    H, W = shape
    out = []
    for b in boxes:
        x0, y0 = max(0, b["x0"] - ox), max(0, b["y0"] - oy)
        x1, y1 = min(W, b["x1"] - ox), min(H, b["y1"] - oy)
        if x1 > x0 and y1 > y0:
            out.append({**b, "x0": x0, "y0": y0, "x1": x1, "y1": y1})
    return out


def detect_views(cropped, zmap, mask, boxes, z_thresh, detrend_label, margins=None):
    """The 3 detection views as specs (see report_views)."""
    off = (margins["left"], margins["top"]) if margins else (0, 0)
    views = [
        {"key": "boxes", "title": "1. Cropped heatmap + located cracks", "img": cropped,
         "cmap": "magma", "cbar": "Amplitude (a.u.)", "boxes": boxes},
        {"key": "zmap", "title": f"2. z-score map (detrend: {detrend_label})", "img": zmap,
         "cmap": "RdBu_r", "norm": diverging_norm(zmap),
         "cbar": "z (robust)"},
        {"key": "mask", "title": f"3. Mask |z| > {z_thresh}", "img": mask.astype(float),
         "cmap": "gray_r", "cbar": "1 = candidate", "robust": False},
    ]
    for v in views:
        v["offset"] = off
    return views


def build_view_figure(view, meta, title, point=None):
    """ONE view, full size, with larger text - for looking at a single map closely.
    point=(x_mm, y_mm) marks a looked-up point with a crosshair."""
    fig, ax = plt.subplots(figsize=(11, 7), facecolor=SURFACE, layout="constrained")
    fig.suptitle(title, fontsize=12, color=INK)
    _draw_view(ax, view, meta["pitch_mm"], scale=1.5, wrap=80)
    if point is not None:
        _mark_point(ax, point)
    return fig


def _to_index(pos_mm, pitch, n, name):
    """mm -> pixel index. Pixel i covers [i*pitch, (i+1)*pitch) (see the imshow extent).
    A position off the map is an error, not silently clamped to the edge pixel."""
    size = n * pitch
    if not 0 <= pos_mm <= size:
        raise ValueError(f"{name} = {pos_mm:g} mm is outside this map (0 to {size:.1f} mm).")
    return int(min(np.floor(pos_mm / pitch), n - 1))


def value_at(view, meta, x_mm, y_mm):
    """Value of a view at a point given in mm. Returns (value, col, row, x_mm, y_mm), the
    last two being the centre of the pixel actually read."""
    img, pitch = view["img"], meta["pitch_mm"]
    col = _to_index(x_mm, pitch, img.shape[1], X_LABEL)
    row = _to_index(y_mm, pitch, img.shape[0], Y_LABEL)
    return float(img[row, col]), col, row, (col + 0.5) * pitch, (row + 0.5) * pitch


def profile(view, meta, along, pos_mm):
    """
    1D cut through a view. along='x': the row at Y=pos_mm, signal as a function of X;
    along='y': the column at X=pos_mm, signal as a function of Y.
    Returns (positions_mm, values, index, pos_mm_of_that_pixel_centre).
    """
    img, pitch = view["img"], meta["pitch_mm"]
    if along == "x":
        i = _to_index(pos_mm, pitch, img.shape[0], Y_LABEL)
        vals = img[i, :]
    elif along == "y":
        i = _to_index(pos_mm, pitch, img.shape[1], X_LABEL)
        vals = img[:, i]
    else:
        raise ValueError(f"along must be 'x' or 'y', got {along!r}")
    pos = (np.arange(vals.size) + 0.5) * pitch
    return pos, vals, i, (i + 0.5) * pitch


def _mark_point(ax, point):
    ax.plot(*point, marker="+", markersize=22, markeredgewidth=2.5, color="#00e5ff")
    ax.plot(*point, marker="o", markersize=10, fillstyle="none",
            markeredgewidth=1.5, color="#00e5ff")


def _draw_profile_axes(ax_map, ax_p, view, meta, along, pos_mm, scale=1.2):
    """Map with the cut line (ax_map) + the 1D signal along that line (ax_p)."""
    pos, vals, i, at = profile(view, meta, along, pos_mm)
    pitch = meta["pitch_mm"]
    _draw_view(ax_map, view, pitch, scale=scale, wrap=80)
    line = dict(color="#00e5ff", linewidth=2, linestyle="--")
    if along == "x":
        ax_map.axhline(at, **line)
        cut, xlab = f"row {i} at {Y_LABEL} = {at:.1f} mm", X_LABEL
    else:
        ax_map.axvline(at, **line)
        cut, xlab = f"column {i} at {X_LABEL} = {at:.1f} mm", Y_LABEL

    ax_p.plot(pos, vals, color=INK, linewidth=1.6)
    if view.get("norm") is not None:          # derivative / z maps: show the zero line
        ax_p.axhline(0, color=MUTED, linewidth=0.8)
    ax_p.set_xlim(pos[0] - 0.5 * pitch, pos[-1] + 0.5 * pitch)
    ax_p.set_title(f"Profile - {cut}", loc="left", fontsize=12, color=INK)
    ax_p.set_xlabel(f"{xlab} (mm)", color=MUTED, fontsize=11)
    ax_p.set_ylabel(view.get("cbar") or "value", color=MUTED, fontsize=11)
    ax_p.tick_params(colors=MUTED, labelsize=10)
    ax_p.grid(True, color="#e4e3dc", linewidth=0.7)
    for sp in ax_p.spines.values():
        sp.set_color("#c3c2b7")
    return pos, vals


def build_profile_figure(view, meta, title, along, pos_mm):
    """Map with the cut line marked (top) + the 1D signal along that line (bottom)."""
    fig, (ax_map, ax_p) = plt.subplots(2, 1, figsize=(11, 8.5), facecolor=SURFACE,
                                       layout="constrained", height_ratios=[1.35, 1])
    fig.suptitle(title, fontsize=12, color=INK)
    pos, vals = _draw_profile_axes(ax_map, ax_p, view, meta, along, pos_mm)
    return fig, pos, vals


# ---------------------------------------------------------------- R + P pairs

def pair_partner(path):
    """The other sensor's file of the same measurement (R <-> P, same amp/freq/lift-off/
    size, same folder), or None. All 51 R/P pairs in this dataset share one grid size."""
    p = Path(path)
    m = NAME_RE.search(p.stem)
    if not m or m.group("sensor") not in ("R", "P"):
        return None
    a, b = m.span("sensor")
    other = "P" if m.group("sensor") == "R" else "R"
    q = p.with_name(p.stem[:a] + other + p.stem[b:] + p.suffix)
    return q if q.exists() else None


def shared_crop(filtered_list, margin_arg, auto_z=3.0):
    """
    Crop several scans of the SAME measurement with ONE set of margins. Cropping R and P
    separately would give each its own origin, so the same (X, Y) in mm would point at two
    different spots on the plate. With 'auto', each side takes the larger of the measured
    noisy borders, so both images are clean.
    """
    if str(margin_arg).lower() == "auto":
        found = [auto_border_margin(f, z_thresh=auto_z)[1] for f in filtered_list]
        m = {k: max(x[k] for x in found) for k in ("top", "bottom", "left", "right")}
        label = (f"auto, shared R+P: top {m['top']} bottom {m['bottom']} "
                 f"left {m['left']} right {m['right']} px")
    else:
        n = int(margin_arg)
        m = {"top": n, "bottom": n, "left": n, "right": n}
        label = f"fixed {n}px per side"
    crops = []
    for f in filtered_list:
        H, W = f.shape
        if m["top"] + m["bottom"] >= H or m["left"] + m["right"] >= W:
            raise ValueError(f"crop {m} is too large for a {H}x{W} image")
        crops.append(f[m["top"]:H - m["bottom"], m["left"]:W - m["right"]])
    return crops, m, label


def _tagged(view, sensor):
    v = dict(view)
    v["title"] = f"{sensor}  -  {view['title']}"
    return v


def build_pair_figure(view_r, view_p, meta, title, point=None):
    """The same map for R (left) and P (right), side by side. Each keeps its own colour
    scale: R is an amplitude and P a phase, so their values are not comparable."""
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5), facecolor=SURFACE, layout="constrained")
    fig.suptitle(title, fontsize=12, color=INK)
    for ax, v, sensor in zip(axes, (view_r, view_p), "RP"):
        _draw_view(ax, _tagged(v, sensor), meta["pitch_mm"], scale=1.25, wrap=70)
        if point is not None:
            _mark_point(ax, point)
    return fig


def build_pair_profile_figure(view_r, view_p, meta, title, along, pos_mm):
    """R and P maps with the same cut line (top) and their profiles (bottom)."""
    fig, axes = plt.subplots(2, 2, figsize=(17, 9), facecolor=SURFACE,
                             layout="constrained", height_ratios=[1.3, 1])
    fig.suptitle(title, fontsize=12, color=INK)
    out = []
    for col, (v, sensor) in enumerate(zip((view_r, view_p), "RP")):
        pos, vals = _draw_profile_axes(axes[0, col], axes[1, col], _tagged(v, sensor),
                                       meta, along, pos_mm, scale=1.1)
        out.append((sensor, pos, vals))
    return fig, out


def build_report_figure(raw, filtered, cropped, meta, crop_label, filter_label, title):
    """Build the 6-panel overview figure. Shared by the CLI and the desktop app."""
    views, dx, dx_abs = report_views(raw, filtered, cropped, crop_label, filter_label,
                                     meta.get("sensor"))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), facecolor=SURFACE,
                             layout="constrained")
    fig.suptitle(title, fontsize=12, color=INK)
    for ax, v in zip(axes.flat, views):
        _draw_view(ax, v, meta["pitch_mm"])
    return fig, dx, dx_abs


def build_detect_figure(cropped, zmap, mask, boxes, meta, title, z_thresh, detrend_label):
    """Build the 3-panel detection overview figure. Shared by the CLI and the desktop app."""
    views = detect_views(cropped, zmap, mask, boxes, z_thresh, detrend_label)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), facecolor=SURFACE,
                             layout="constrained")
    fig.suptitle(title, fontsize=11, color=INK)
    for ax, v in zip(axes, views):
        _draw_view(ax, v, meta["pitch_mm"])
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

    outdir = result_dir(args.tdms)
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
    outdir = result_dir(args.tdms)

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
        print(f"\n{verb} {n_edge} candidate(s) TOUCHING THE IMAGE EDGE (listed in the box"
              " files, not drawn). Scan borders often produce artifacts, so most of these"
              " are false alarms."
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

def analyze_one(path, filter_method=DEFAULT_FILTER, kernel=3, margin="auto",
                detrend_method=DEFAULT_DETREND, z_thresh=2.5, min_area=6, merge_gap=2,
                auto_margin_z=3.0, fix_direction=True):
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
        grid, meta = load_scan(p, fix_direction=fix_direction)
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
    # Rank on interior candidates only. An edge-touching box is usually the plate edge the
    # crop did not fully remove; merged, it becomes one image-sized box with a huge |z| that
    # would otherwise top the whole ranking (seen on the data/P_* scans).
    scores = sorted((b["score"] for b in boxes if not b["touches_edge"]), reverse=True)

    row.update({
        "size_px": f"{meta['size_x']}x{meta['size_y']}",
        "row_shift_px": meta["row_shift_px"],
        "crop_top": m["top"], "crop_bottom": m["bottom"],
        "crop_left": m["left"], "crop_right": m["right"],
        "cropped_px": f"{cropped.shape[1]}x{cropped.shape[0]}",
        "n_boxes": len(boxes),
        "n_inner": len(scores),
        "n_strong": sum(1 for s in scores if s >= 4.0),
        "n_edge_boxes": len(boxes) - len(scores),
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
    for r in (args.paths or [str(DATA_DIR)]):
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

    cols = ["folder", "sensor", "amp", "freq", "variant", "n_inner", "n_strong",
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
    r.add_argument("--filter", default=DEFAULT_FILTER, choices=FILTERS)
    r.add_argument("--kernel", type=int, default=3)
    r.add_argument("--margin", default="auto",
                   help="'auto' (measure the noisy border) or a fixed pixel count")
    r.add_argument("--auto-margin-z", type=float, default=3.0,
                   help="z-score above which an edge row/column counts as noise")
    r.set_defaults(func=cmd_report)

    d = sub.add_parser("detect", help="one file -> border crop + crack localization")
    d.add_argument("tdms", help="path to a .tdms file")
    d.add_argument("--filter", default=DEFAULT_FILTER, choices=FILTERS)
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
    d.add_argument("--detrend", default=DEFAULT_DETREND, choices=DETRENDS,
                   help="detrend direction before thresholding ('both' when the background "
                        "slopes along both axes)")
    d.set_defaults(func=cmd_detect)

    s = sub.add_parser("sweep", help="analyze an ENTIRE tree -> table + ranking")
    s.add_argument("paths", nargs="*", default=[str(DATA_DIR)],
                   help="directories (searched recursively) or glob patterns; default data/")
    s.add_argument("--filter", default=DEFAULT_FILTER, choices=FILTERS)
    s.add_argument("--kernel", type=int, default=3)
    s.add_argument("--margin", default="auto")
    s.add_argument("--detrend", default=DEFAULT_DETREND, choices=DETRENDS)
    s.add_argument("--z-thresh", type=float, default=2.5)
    s.add_argument("--min-area", type=int, default=6)
    s.add_argument("--merge-gap", type=int, default=2)
    s.add_argument("--out", default=str(RESULTS_DIR / "sweep_all.csv"))
    s.set_defaults(func=cmd_sweep)

    c = sub.add_parser("compare", help="many files -> sensor/frequency comparison table")
    c.add_argument("tdms", nargs="+", help="paths or glob patterns, e.g. 'data/cambien_4lop_2/*.tdms'")
    c.add_argument("--filter", default=DEFAULT_FILTER, choices=FILTERS)
    c.add_argument("--kernel", type=int, default=3)
    c.add_argument("--edge", type=int, default=5)
    c.add_argument("--boxes", default=None, help="JSON label boxes (scan_metrics.py format)")
    c.add_argument("--out", default="sensor_compare.csv")
    c.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
