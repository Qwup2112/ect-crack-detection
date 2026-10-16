#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crack_labels.py - Hand-drawn crack labels: storage, region analysis, and the image a
model sees. Shared by the desktop app (drawing boxes with the mouse) and crack_model.py
(training / running YOLO).

Labels are the GROUND TRUTH a model is trained and scored against, so they must come from
a person looking at the scan. Boxes from 'Locate cracks' may be loaded as a starting point
to correct, but training on them unedited only teaches the model to copy the z-score
threshold detector (grading the detector against itself).

Storage: results/labels/<session>/<file stem>.json, boxes in ORIGINAL grid pixels (the grid
load_scan returns, before any border crop), so they stay valid whatever crop or view is
used to draw them.
"""

import json
from pathlib import Path

import numpy as np

import tdms_crack_scan as backend

LABELS_DIR = backend.RESULTS_DIR / "labels"


# ---------------------------------------------------------------- storage

def label_path(tdms):
    p = Path(tdms).resolve()
    try:
        rel = p.parent.relative_to(backend.DATA_DIR)
    except ValueError:
        rel = Path("_other")
    return LABELS_DIR / rel / f"{p.stem}.json"


def load_labels(tdms):
    """Boxes [{x0, y0, x1, y1}] in original grid pixels; [] when the file is unlabelled."""
    lp = label_path(tdms)
    if not lp.exists():
        return []
    return json.loads(lp.read_text(encoding="utf-8"))["boxes"]


def label_note(tdms):
    """The file's note, e.g. 'DRAFT ...' for labels nobody has reviewed yet; '' if none."""
    lp = label_path(tdms)
    return json.loads(lp.read_text(encoding="utf-8")).get("note", "") if lp.exists() else ""


def save_labels(tdms, boxes, grid_shape, note=None):
    """Write the boxes. Saving from the app passes no note, so a DRAFT note disappears as
    soon as a person edits that scan's labels."""
    lp = label_path(tdms)
    if not boxes:
        if lp.exists():
            lp.unlink()          # no boxes left: the file is unlabelled again
        return lp
    lp.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": str(Path(tdms).resolve().relative_to(backend.PROJECT_ROOT)),
        "grid_hw": list(grid_shape),
        "coords": "original grid pixels (before border crop), x = column, y = row",
        "boxes": [{k: int(b[k]) for k in ("x0", "y0", "x1", "y1")} for b in boxes],
    }
    if note:
        payload["note"] = note
    lp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return lp


def labelled_files():
    """(tdms path, boxes, grid_hw) for every labelled scan."""
    out = []
    for lp in sorted(LABELS_DIR.rglob("*.json")):
        d = json.loads(lp.read_text(encoding="utf-8"))
        src = backend.PROJECT_ROOT / d["source"]
        if src.exists() and d["boxes"]:
            out.append((src, d["boxes"], tuple(d["grid_hw"])))
    return out


def same_geometry_files(tdms, grid_shape):
    """Other scans of the same session with the same grid - the same plate position, so
    the same crack boxes (all frequencies, amplitudes and both sensors)."""
    p = Path(tdms).resolve()
    out = []
    for q in sorted(p.parent.glob("*.tdms")):
        if q == p:
            continue
        try:
            g, _ = backend.load_scan(q)
        except Exception:
            continue
        if g.shape == tuple(grid_shape):
            out.append(q)
    return out


# ---------------------------------------------------------------- analysis

def region_stats(img, box, pitch, ring=5):
    """
    Describe one box on one map: size, value range, and contrast against the ring of
    pixels around it. CNR on signed values collapses for a dipole (bright lobe beside a
    dark lobe cancel out); CNR_abs does not - CNR_abs >> CNR is the dipole fingerprint.
    `box` is in the pixel coordinates of `img`.
    """
    x0, y0, x1, y1 = box["x0"], box["y0"], box["x1"], box["y1"]
    roi = img[y0:y1, x0:x1]
    flat = img - np.nanmedian(img)
    cnr = backend.cnr_for_boxes(flat, [box], edge=0, ring=ring)
    cnr_abs = backend.cnr_for_boxes(np.abs(flat), [box], edge=0, ring=ring)
    return {
        "w_mm": (x1 - x0) * pitch, "h_mm": (y1 - y0) * pitch,
        "min": float(np.nanmin(roi)), "max": float(np.nanmax(roi)),
        "mean": float(np.nanmean(roi)),
        "CNR": cnr, "CNR_abs": cnr_abs,
        "dipole": bool(np.isfinite(cnr_abs) and cnr_abs > 1.5 * max(cnr, 1e-9)),
    }


def measure_crack(img, box, pitch, level=0.5, core=0.5):
    """
    Size of the crack SIGNATURE inside a box - not of the box. The background is the median
    of the box's border pixels, and a tilted plane fitted to them is removed, so a slope
    across the box (the plate edge near the outer cracks) is not taken for the crack. The
    peak is searched in the central `core` fraction of the box only - a box is centred on
    its crack, while the plate edge or a neighbour sits at its border. Pixels reaching
    `level` x that peak (full width at half maximum) and joined to it (both dipole lobes,
    after bridging 1-px gaps) are measured along their principal axes: length = long axis,
    width = short axis.

    The width is the width of the SIGNAL, which includes the probe footprint: a real crack
    is usually far narrower than this. Depth cannot be read off a map without a calibration;
    the peak deviation is only a relative depth indicator between cracks of one scan.
    `box` is in the pixel coordinates of `img`; lengths come back in mm, the centre and the
    axis end points in the map's mm coordinates.
    """
    from scipy import ndimage
    H, W = img.shape
    x0, y0 = max(0, box["x0"]), max(0, box["y0"])
    x1, y1 = min(W, box["x1"]), min(H, box["y1"])
    roi = np.asarray(img[y0:y1, x0:x1], dtype=float)
    h, w = roi.shape
    if h < 3 or w < 3:
        return None
    yy, xx = np.mgrid[0:h, 0:w]
    border = np.zeros((h, w), bool)
    border[[0, -1], :] = border[:, [0, -1]] = True
    ok = border & np.isfinite(roi)
    A = np.column_stack([np.ones(ok.sum()), xx[ok], yy[ok]])
    coef = np.linalg.lstsq(A, roi[ok], rcond=None)[0]
    dev = np.nan_to_num(np.abs(roi - (coef[0] + coef[1] * xx + coef[2] * yy)))
    cy0, cy1 = int(h * (1 - core) / 2), int(np.ceil(h * (1 + core) / 2))
    cx0, cx1 = int(w * (1 - core) / 2), int(np.ceil(w * (1 + core) / 2))
    centre_dev = dev[cy0:cy1, cx0:cx1]
    peak = float(centre_dev.max()) if centre_dev.size else 0.0
    if peak <= 0:
        return None
    mask = ndimage.binary_closing(dev >= level * peak, structure=np.ones((3, 3)))
    mask |= dev >= level * peak                 # closing can erode at the region border
    lab, _ = ndimage.label(mask)
    py, px = np.unravel_index(int(np.argmax(centre_dev)), centre_dev.shape)
    py, px = py + cy0, px + cx0
    comp = lab == lab[py, px]
    # how far to trust the size: a weak peak (against the border noise), or a signature
    # filling most of its box, has no clear outline to measure. A signature reaching the
    # box border runs on outside it: its length is then a LOWER bound, not unreliable.
    resid = dev[border]
    noise = 1.4826 * float(np.median(resid)) or 1e-12
    snr = peak / noise
    fill = float(comp.sum()) / comp.size
    spills = bool(comp[border].any())
    reliable = snr >= 3 and fill < 0.4
    ys, xs = np.nonzero(comp)
    pts = np.column_stack([(xs + x0 + 0.5) * pitch, (ys + y0 + 0.5) * pitch])
    centre = pts.mean(axis=0)
    if len(pts) >= 3:
        evals, evecs = np.linalg.eigh(np.cov((pts - centre).T))
        major, minor = evecs[:, 1], evecs[:, 0]
    else:
        major, minor = np.array([0.0, 1.0]), np.array([1.0, 0.0])
    pm, pn = (pts - centre) @ major, (pts - centre) @ minor
    length = float(np.ptp(pm) + pitch)          # + one pixel: extents are pixel centres
    width = float(np.ptp(pn) + pitch)
    ang = float(np.degrees(np.arctan2(abs(major[0]), abs(major[1]))))   # 0 = along Y
    return {
        "length_mm": length, "width_mm": width,
        "area_mm2": float(len(pts) * pitch * pitch),
        "angle_deg": ang, "along": "Y" if ang < 45 else "X",
        "centre_mm": (float(centre[0]), float(centre[1])),
        "length_ends": (centre + major * pm.min(), centre + major * pm.max()),
        "width_ends": (centre + minor * pn.min(), centre + minor * pn.max()),
        "peak": peak, "level": level,
        "snr": snr, "fill": fill, "spills": spills, "reliable": reliable,
    }


# ---------------------------------------------------------------- model input

def _norm(a, pct=1.0, signed=False):
    """Robust scaling to 0..1: percentile limits, so a few extreme pixels do not flatten
    the rest (same idea as the display colour scales)."""
    if signed:
        lim = float(np.nanpercentile(np.abs(a), 100 - pct)) or 1.0
        out = 0.5 + 0.5 * a / lim
    else:
        lo, hi = np.nanpercentile(a, [pct, 100 - pct])
        out = (a - lo) / (hi - lo if hi > lo else 1.0)
    return np.nan_to_num(np.clip(out, 0, 1))


def model_image(grid):
    """
    The 3-channel image the model sees, same size as the original grid (so label pixels
    map 1:1): the tested preprocessing (lpf filter + bg detrend), then
      R = detrended signal, G = d/dx, B = d/dy, each robustly scaled to 0..255.
    The derivatives carry the crack edges (the ECT dipole); the detrended map the body.
    """
    f = backend.apply_filter(grid, backend.DEFAULT_FILTER, 3)
    flat = backend.detrend(f, backend.DEFAULT_DETREND)
    ch = [_norm(flat), _norm(np.gradient(f, axis=1), signed=True),
          _norm(np.gradient(f, axis=0), signed=True)]
    return (np.dstack(ch) * 255).astype(np.uint8)
