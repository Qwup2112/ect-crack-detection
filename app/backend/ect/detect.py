"""Crack detection ("nhận diện vết nứt") and measurement.

Two detectors, used together:
  yolo      - YOLO11n fine-tuned on the labelled scans (results/model/crack_yolo.pt).
              Finds all 15 cracks per scan on both labelled sessions; still fires on pure
              noise (it never saw a crack-free scan), so cross-check weak boxes.
              Runs on onnxruntime (crack_yolo.onnx, ~50 MB of memory) when the export is
              there; PyTorch + ultralytics (well over 1 GB) is only the fallback.
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
from .paths import LABELS_DIR, DATA_DIR, WEIGHTS, ONNX_WEIGHTS, SPLIT_FILE, PROJECT_ROOT

YOLO_IMGSZ = 320          # scans are ~150 x 100 points, letterboxed to 320
YOLO_CONF = 0.5           # 0.5 still finds 450/450 held-out cracks; cuts noise boxes 65 -> 17
YOLO_IOU = 0.7            # NMS overlap, ultralytics' predict default
YOLO_MAX_DET = 300


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
_yolo = {}                # backend -> (weights mtime, loaded model)


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
    return WEIGHTS.exists() or ONNX_WEIGHTS.exists()


def onnx_current():
    """The ONNX export exists and is not older than the .pt it was made from."""
    return ONNX_WEIGHTS.exists() and (not WEIGHTS.exists()
                                      or ONNX_WEIGHTS.stat().st_mtime >= WEIGHTS.stat().st_mtime)


def yolo_backend():
    return "onnx" if onnx_current() else "torch"


def _resize_linear(img, size_wh):
    try:
        import cv2                                  # what ultralytics resizes with
        return cv2.resize(img, size_wh, interpolation=cv2.INTER_LINEAR)
    except ImportError:
        from PIL import Image
        return np.asarray(Image.fromarray(img).resize(size_wh, Image.BILINEAR))


def _letterbox(img, size=YOLO_IMGSZ, stride=32):
    """ultralytics predict's LetterBox (rect=True -> auto): fit the long side to `size`, pad
    the short side with 114 only up to a multiple of `stride` (150 x 101 -> 320 x 224)."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    if (nw, nh) != (w, h):
        img = _resize_linear(img, (nw, nh))
    dw, dh = np.mod(size - nw, stride) / 2, np.mod(size - nh, stride) / 2
    top, bottom, left, right = round(dh - 0.1), round(dh + 0.1), round(dw - 0.1), round(dw + 0.1)
    out = np.full((nh + top + bottom, nw + left + right, 3), 114, np.uint8)
    out[top:top + nh, left:left + nw] = img
    return out


def _nms(xyxy, scores, iou_thr):
    order, keep = np.argsort(-scores), []
    while order.size:
        i, rest = order[0], order[1:]
        keep.append(i)
        ix = np.clip(np.minimum(xyxy[i, 2], xyxy[rest, 2]) - np.maximum(xyxy[i, 0], xyxy[rest, 0]), 0, None)
        iy = np.clip(np.minimum(xyxy[i, 3], xyxy[rest, 3]) - np.maximum(xyxy[i, 1], xyxy[rest, 1]), 0, None)
        inter = ix * iy
        area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
        order = rest[inter / (area(xyxy[i]) + area(xyxy[rest]) - inter + 1e-9) <= iou_thr]
    return np.array(keep, int)


def _onnx_session():
    key = ONNX_WEIGHTS.stat().st_mtime
    if _yolo.get("onnx", (None,))[0] != key:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = 2, 1
        so.enable_cpu_mem_arena = False             # give memory back between calls: this PC is tight
        _yolo["onnx"] = (key, ort.InferenceSession(str(ONNX_WEIGHTS), so, providers=["CPUExecutionProvider"]))
    return _yolo["onnx"][1]


def _yolo_onnx(img_rgb, conf):
    """[(x0, y0, x1, y1, conf)] in the pixels of img_rgb, same steps as ultralytics predict."""
    h0, w0 = img_rgb.shape[:2]
    x = _letterbox(img_rgb).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    h1, w1 = x.shape[2:]
    sess = _onnx_session()
    pred = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    if pred.shape[-1] == 6:                         # end-to-end head (NMS inside): x0 y0 x1 y1 conf cls
        xyxy, scores = pred[:, :4].astype(float), pred[:, 4].astype(float)
        keep = np.nonzero(scores > conf)[0][:YOLO_MAX_DET]
    else:                                           # (4 + classes, anchors): cx cy w h scores...
        pred = pred.T
        scores = pred[:, 4:].max(axis=1).astype(float)
        cand = scores > conf
        cx, cy, bw, bh = pred[cand, :4].astype(float).T
        xyxy, scores = np.column_stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]), scores[cand]
        keep = _nms(xyxy, scores, YOLO_IOU)[:YOLO_MAX_DET] if scores.size else np.zeros(0, int)
    xyxy, scores = xyxy[keep], scores[keep]
    gain = min(h1 / h0, w1 / w0)                    # ultralytics ops.scale_boxes
    pad_x = round((w1 - round(w0 * gain)) / 2 - 0.1)
    pad_y = round((h1 - round(h0 * gain)) / 2 - 0.1)
    xyxy = (xyxy - [pad_x, pad_y, pad_x, pad_y]) / gain
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w0)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h0)
    return [(*b, s) for b, s in zip(xyxy.tolist(), scores.tolist())]


def _yolo_torch(img_rgb, conf):
    key = WEIGHTS.stat().st_mtime
    if _yolo.get("torch", (None,))[0] != key:
        from ultralytics import YOLO
        _yolo["torch"] = (key, (YOLO(str(WEIGHTS)), _device()))
    model, device = _yolo["torch"][1]
    r = model.predict(img_rgb[..., ::-1], imgsz=YOLO_IMGSZ, conf=conf, device=device,  # BGR, what the trainer read
                      verbose=False)[0]
    return [(*b, c) for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]


def yolo_detect(grid, conf=YOLO_CONF, drop_edge=True, backend=None):
    """Boxes [{x0,y0,x1,y1,conf}] in original grid pixels. Boxes on the scan border are the
    plate edge, not cracks, and are dropped unless drop_edge=False. backend: 'onnx' or
    'torch'; default onnx when the export is current."""
    backend = backend or yolo_backend()
    if backend == "onnx" and not ONNX_WEIGHTS.exists():
        raise FileNotFoundError(f"no ONNX model at {ONNX_WEIGHTS.relative_to(PROJECT_ROOT)}")
    if backend == "torch" and not WEIGHTS.exists():
        raise FileNotFoundError(f"no trained model at {WEIGHTS.relative_to(PROJECT_ROOT)}")
    with _yolo_lock:
        img = model_image(grid)
        raw = _yolo_onnx(img, conf) if backend == "onnx" else _yolo_torch(img, conf)
    H, W = grid.shape
    out = []
    for x0, y0, x1, y1, c in raw:
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
