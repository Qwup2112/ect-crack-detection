"""Crack detection ("nhận diện vết nứt") and measurement.

Two detectors, used together:
  yolo      - YOLO11n fine-tuned on the labelled scans (results/model/crack_yolo.pt).
              Finds all 15 cracks per scan on both labelled sessions; still fires on pure
              noise (it never saw a crack-free scan), so cross-check weak boxes.
  threshold - robust z-score on the background-removed map, fragments merged. Model-free:
              says "this deviates from the background", nothing more.
Boxes are always returned in ORIGINAL grid pixels (before any crop), the same frame as the
hand labels in results/labels/.
"""
import json
import os
import threading
from pathlib import Path

import numpy as np
from scipy import ndimage

from . import process
from .paths import LABELS_DIR, DATA_DIR, WEIGHTS, SPLIT_FILE, PROJECT_ROOT

YOLO_IMGSZ = 320          # scans are ~150 x 100 points, letterboxed to 320
YOLO_CONF = 0.5           # 0.5 still finds 450/450 held-out cracks; cuts noise boxes 65 -> 17


# ---------------------------------------------------------------- labels

def label_file(sid):
    return LABELS_DIR / f"{sid}.json"


def load_labels(sid):
    """(boxes, note). Boxes in original grid pixels; [] when the scan has no labels."""
    lp = label_file(sid)
    if not lp.exists():
        return [], ""
    d = json.loads(lp.read_text(encoding="utf-8"))
    return d.get("boxes", []), d.get("note", "")


def in_training_set(sid):
    if not SPLIT_FILE.exists():
        return False
    rel = (Path("data") / f"{sid}.tdms").as_posix()
    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    return any(Path(p).as_posix() == rel for p in split.get("train", []))


def iou(a, b):
    ix = max(0, min(a["x1"], b["x1"]) - max(a["x0"], b["x0"]))
    iy = max(0, min(a["y1"], b["y1"]) - max(a["y0"], b["y0"]))
    inter = ix * iy
    area = lambda r: (r["x1"] - r["x0"]) * (r["y1"] - r["y0"])
    return inter / (area(a) + area(b) - inter) if inter else 0.0


def match_labels(pred, labels, thr=0.3):
    """Greedy one-to-one match, most confident first. Sets pred[i]['match'] and returns
    (found, false, missed_label_indices)."""
    free = list(range(len(labels)))
    order = sorted(range(len(pred)), key=lambda j: -pred[j].get("conf", pred[j].get("score", 0)))
    for j in order:
        best = max(free, key=lambda i: iou(pred[j], labels[i]), default=None)
        ok = best is not None and iou(pred[j], labels[best]) >= thr
        pred[j]["match"] = bool(ok)
        if ok:
            free.remove(best)
    found = sum(p["match"] for p in pred)
    return found, len(pred) - found, free


# ---------------------------------------------------------------- threshold detector

def threshold_detect(cropped, offset, z_thresh=2.5, min_area=6, merge_gap=2, detrend_method="bg"):
    """Robust |z| > z_thresh on the background-removed map, fragments within merge_gap px
    joined (one crack's signal breaks into row streaks), boxes from the original pixels.
    Boxes touching the crop edge are dropped (plate edge / scan border). Returns boxes in
    ORIGINAL grid pixels, each with 'score' = max |z|."""
    z = process.robust_z(process.detrend(cropped, detrend_method))
    mask = np.abs(z) > z_thresh
    k = 2 * merge_gap + 1
    grouped = ndimage.binary_dilation(mask, np.ones((k, k), bool)) if merge_gap > 0 else mask
    lab, n = ndimage.label(grouped)
    H, W = cropped.shape
    ox, oy = offset
    boxes = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero((lab == i) & mask)
        if ys.size == 0:
            continue
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        if (y1 - y0) * (x1 - x0) < min_area:
            continue
        if x0 == 0 or y0 == 0 or x1 == W or y1 == H:
            continue
        boxes.append({"x0": x0 + ox, "y0": y0 + oy, "x1": x1 + ox, "y1": y1 + oy,
                      "score": round(float(np.abs(z[ys, xs]).max()), 2)})
    boxes.sort(key=lambda b: (b["y0"], b["x0"]))
    return boxes, z


def z_top5(boxes):
    s = sorted((b["score"] for b in boxes), reverse=True)[:5]
    return round(float(np.mean(s)), 2) if s else 0.0


# ---------------------------------------------------------------- YOLO

def _norm01(a, pct=1.0, signed=False):
    if signed:
        lim = float(np.nanpercentile(np.abs(a), 100 - pct)) or 1.0
        out = 0.5 + 0.5 * a / lim
    else:
        lo, hi = np.nanpercentile(a, [pct, 100 - pct])
        out = (a - lo) / (hi - lo if hi > lo else 1.0)
    return np.nan_to_num(np.clip(out, 0, 1))


def model_image(grid):
    """The 3-channel image the YOLO model was trained on, same size as the grid:
    R = lpf + bg-detrended signal, G = d/dx, B = d/dy, each robustly scaled to 0..255."""
    f = process.denoise(grid, "lpf")
    flat = process.detrend(f, "bg")
    ch = [_norm01(flat), _norm01(np.gradient(f, axis=1), signed=True),
          _norm01(np.gradient(f, axis=0), signed=True)]
    return (np.dstack(ch) * 255).astype(np.uint8)


_yolo_lock = threading.Lock()
_yolo = {}


def _device():
    forced = os.environ.get("CRACK_DEVICE")
    if forced:
        return forced
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def yolo_available():
    return WEIGHTS.exists()


def yolo_detect(grid, conf=YOLO_CONF, drop_edge=True):
    """Boxes [{x0,y0,x1,y1,conf}] in original grid pixels. Boxes on the scan border are the
    plate edge, not cracks, and are dropped unless drop_edge=False."""
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"no trained model at {WEIGHTS.relative_to(PROJECT_ROOT)}")
    with _yolo_lock:
        key = WEIGHTS.stat().st_mtime
        if _yolo.get("key") != key:
            from ultralytics import YOLO
            _yolo.update(key=key, model=YOLO(str(WEIGHTS)), device=_device())
        img = model_image(grid)[..., ::-1]           # RGB -> BGR, what the trainer read
        r = _yolo["model"].predict(img, imgsz=YOLO_IMGSZ, conf=conf, device=_yolo["device"],
                                   verbose=False)[0]
    H, W = grid.shape
    out = []
    for (x0, y0, x1, y1), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
        b = {"x0": int(max(0, round(x0))), "y0": int(max(0, round(y0))),
             "x1": int(min(W, round(x1))), "y1": int(min(H, round(y1))), "conf": round(float(c), 3)}
        if drop_edge and (b["x0"] <= 0 or b["y0"] <= 0 or b["x1"] >= W or b["y1"] >= H):
            continue
        out.append(b)
    out.sort(key=lambda b: (b["y0"], b["x0"]))
    return out


# ---------------------------------------------------------------- measurement

def cnr(img, box, ring=5):
    """Contrast-to-noise of a box against the ring of pixels around it."""
    H, W = img.shape
    x0, y0, x1, y1 = box["x0"], box["y0"], box["x1"], box["y1"]
    roi = img[y0:y1, x0:x1]
    outer = np.zeros_like(img, bool)
    outer[max(0, y0 - ring):min(H, y1 + ring), max(0, x0 - ring):min(W, x1 + ring)] = True
    outer[y0:y1, x0:x1] = False
    ring_px = img[outer]
    ring_px, roi = ring_px[np.isfinite(ring_px)], roi[np.isfinite(roi)]
    if roi.size == 0 or ring_px.size < 10 or ring_px.std() == 0:
        return float("nan")
    return float(abs(roi.mean() - ring_px.mean()) / ring_px.std())


def measure_crack(img, box, pitch, level=0.5, core=0.5):
    """Size of the crack SIGNATURE inside a box (not of the box): a tilted plane through the
    box border is removed as background, the peak is taken in the box centre, the pixels
    above level x peak joined to it are measured along their principal axes.
    Width includes the probe footprint (~6 mm here); depth needs a calibration.
    `box` and the returned mm coordinates are in the pixel frame of `img`."""
    H, W = img.shape
    x0, y0, x1, y1 = max(0, box["x0"]), max(0, box["y0"]), min(W, box["x1"]), min(H, box["y1"])
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
    centre = dev[cy0:cy1, cx0:cx1]
    peak = float(centre.max()) if centre.size else 0.0
    if peak <= 0:
        return None
    mask = ndimage.binary_closing(dev >= level * peak, np.ones((3, 3))) | (dev >= level * peak)
    lab, _ = ndimage.label(mask)
    py, px = np.unravel_index(int(np.argmax(centre)), centre.shape)
    comp = lab == lab[py + cy0, px + cx0]
    noise = 1.4826 * float(np.median(dev[border])) or 1e-12
    snr, fill = peak / noise, float(comp.sum()) / comp.size
    ys, xs = np.nonzero(comp)
    pts = np.column_stack([(xs + x0 + 0.5) * pitch, (ys + y0 + 0.5) * pitch])
    c = pts.mean(axis=0)
    if len(pts) >= 3:
        _, vecs = np.linalg.eigh(np.cov((pts - c).T))
        major, minor = vecs[:, 1], vecs[:, 0]
    else:
        major, minor = np.array([0.0, 1.0]), np.array([1.0, 0.0])
    pm, pn = (pts - c) @ major, (pts - c) @ minor
    ang = float(np.degrees(np.arctan2(abs(major[0]), abs(major[1]))))
    return {
        "length_mm": round(float(np.ptp(pm) + pitch), 2), "width_mm": round(float(np.ptp(pn) + pitch), 2),
        "area_mm2": round(float(len(pts) * pitch * pitch), 1), "angle_deg": round(ang, 1),
        "along": "Y" if ang < 45 else "X", "centre_mm": [round(float(c[0]), 2), round(float(c[1]), 2)],
        "length_ends": [(c + major * pm.min()).tolist(), (c + major * pm.max()).tolist()],
        "width_ends": [(c + minor * pn.min()).tolist(), (c + minor * pn.max()).tolist()],
        "snr": round(snr, 1), "spills_out_of_box": bool(comp[border].any()),
        "reliable": bool(snr >= 3 and fill < 0.4),
    }


def describe_box(b, pitch):
    """Centre and size of a box in mm (original grid frame)."""
    return {"x_mm": round((b["x0"] + b["x1"]) / 2 * pitch, 1), "y_mm": round((b["y0"] + b["y1"]) / 2 * pitch, 1),
            "w_mm": round((b["x1"] - b["x0"]) * pitch, 1), "h_mm": round((b["y1"] - b["y0"]) * pitch, 1)}
