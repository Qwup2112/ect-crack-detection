#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crack_model.py - YOLO crack detector trained on the hand-drawn labels (crack_labels.py).

  export   labelled scans -> YOLO dataset (results/model/dataset)
  train    fine-tune a pretrained YOLO11n on it -> results/model/crack_yolo.pt
  predict  run the model on one scan

WHY THE SPLIT IS BY SESSION. Scans of one session share plate placements - and, after
"Copy to session", the very same boxes. A random split by file would put near-identical
scans (other frequency, other sensor) on both sides: the model could score high by
remembering WHERE the cracks are, not by recognising them. Grid size is not a safe key
either: truncated files have a smaller grid but the same placement, and one grid size can
hold two placements. So validation holds out whole SESSIONS; with only one session
labelled there is no honest validation and the numbers are reported as optimistic.

Input image: crack_labels.model_image (detrended signal, d/dx, d/dy as R, G, B), at the
original grid size so label pixels map 1:1. Trained on the GPU when PyTorch has CUDA,
else on the CPU (CRACK_DEVICE=cpu forces the CPU).

  python crack_model.py export
  python crack_model.py train --epochs 80
  python crack_model.py predict data/cambien_4lop_2/<file>.tdms

REQUIRES: pip install ultralytics  (pulls in PyTorch)
"""

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tdms_crack_scan as backend
import crack_labels

MODEL_DIR = backend.RESULTS_DIR / "model"
DATASET_DIR = MODEL_DIR / "dataset"
WEIGHTS = MODEL_DIR / "crack_yolo.pt"
SPLIT_FILE = MODEL_DIR / "split.json"
IMGSZ = 320          # scans are ~150x100 px; letterboxed up to 320 for YOLO
BASE_MODEL = "yolo11n.pt"
# confidence cut: on the held-out session 0.25 and 0.5 both find 450/450 cracks, but on
# images of pure noise 0.5 gives 17 boxes instead of 65 (0.6 starts missing cracks)
CONF = 0.5


def _yolo():
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError("The model needs ultralytics: pip install ultralytics") from e
    return YOLO


def device():
    """The first CUDA GPU when PyTorch can use one, else the CPU. CRACK_DEVICE overrides it
    (e.g. CRACK_DEVICE=cpu, or =1 for a second GPU)."""
    forced = os.environ.get("CRACK_DEVICE")
    if forced:
        return forced
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def device_name(dev=None):
    dev = dev or device()
    if dev == "cpu":
        return "CPU"
    import torch
    return f"GPU {dev}: {torch.cuda.get_device_name(int(dev.split(',')[0]))}"


def group_key(src, grid_hw=None):
    """The unit a validation split must not cut through: the measurement session."""
    return Path(src).parent.name


# ---------------------------------------------------------------- dataset

def export_dataset(val_frac=0.25, seed=0, log=print, val_sessions=None, use_all=False):
    """use_all=True: EVERY labelled scan goes to training - the final model, once the
    held-out runs have measured how well it generalises. There is then nothing unseen
    left to validate on, so the validation score is the training score (optimistic)."""
    items = crack_labels.labelled_files()
    if len(items) < 2:
        raise ValueError(f"Only {len(items)} labelled scan(s). Label at least a few scans "
                         "(Figures tab -> LABEL -> Draw boxes) before training.")
    groups = {}
    for src, boxes, hw in items:
        groups.setdefault(group_key(src, hw), []).append((src, boxes, hw))

    if use_all:
        val_groups, honest = [], False
    elif val_sessions:
        unknown = set(val_sessions) - set(groups)
        if unknown or len(set(val_sessions)) >= len(groups):
            raise ValueError(f"val sessions {sorted(val_sessions)} must be labelled sessions "
                             f"and leave one for training; labelled: {sorted(groups)}")
        val_groups, honest = list(val_sessions), True
    elif len(groups) >= 2:
        # hold out the smallest session(s): plate placements the model never saw
        order = sorted(groups, key=lambda g: len(groups[g]))
        val_groups, n_val = [], 0
        for g in order:
            if n_val >= val_frac * len(items) or len(val_groups) == len(groups) - 1:
                break
            val_groups.append(g)
            n_val += len(groups[g])
        honest = True
    else:
        val_groups, honest = [], False
    rng = random.Random(seed)

    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    split = {"train": [], "val": [], "val_groups": val_groups, "honest_validation": honest,
             "all_data": use_all}
    for g, members in groups.items():
        if use_all:
            sides = ["train"] * len(members)
        elif honest:
            sides = ["val" if g in val_groups else "train"] * len(members)
        else:
            idx = list(range(len(members)))
            rng.shuffle(idx)
            k = max(1, int(round(val_frac * len(members))))
            sides = ["val" if i in idx[:k] else "train" for i in range(len(members))]
        for (src, boxes, hw), side in zip(members, sides):
            grid, _ = backend.load_scan(src)
            if grid.shape != tuple(hw):
                log(f"  [SKIPPED] {src.name}: grid {grid.shape} != labelled {hw}")
                continue
            img = crack_labels.model_image(grid)
            H, W = grid.shape
            # short numbered names: the scan names are long and Windows caps paths at
            # 260 characters; split.json maps each number back to its scan
            name = f"{len(split['train']) + len(split['val']):04d}"
            (DATASET_DIR / "images" / side).mkdir(parents=True, exist_ok=True)
            (DATASET_DIR / "labels" / side).mkdir(parents=True, exist_ok=True)
            Image.fromarray(img).save(DATASET_DIR / "images" / side / f"{name}.png")
            lines = [f"0 {(b['x0'] + b['x1']) / 2 / W:.6f} {(b['y0'] + b['y1']) / 2 / H:.6f} "
                     f"{(b['x1'] - b['x0']) / W:.6f} {(b['y1'] - b['y0']) / H:.6f}"
                     for b in boxes]
            (DATASET_DIR / "labels" / side / f"{name}.txt").write_text("\n".join(lines))
            rel = str(Path(src).resolve().relative_to(backend.PROJECT_ROOT))
            split[side].append(rel)
            split.setdefault("images", {})[name] = rel

    (DATASET_DIR / "data.yaml").write_text(
        f"path: {DATASET_DIR.as_posix()}\ntrain: images/train\n"
        f"val: images/{'train' if use_all else 'val'}\n"
        "names:\n  0: crack\n", encoding="utf-8")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_FILE.write_text(json.dumps(split, indent=2), encoding="utf-8")
    log(f"Dataset: {len(split['train'])} train / {len(split['val'])} val scans, "
        f"{sum(len(b) for _, b, _ in items)} boxes, {len(groups)} group(s).")
    if use_all:
        log("  ALL labelled scans used for training (final model). The validation score "
            "below is on training data - see the held-out runs for an honest one.")
    elif honest:
        log(f"  Validation = held-out group(s): {', '.join(val_groups)}")
    else:
        log("  WARNING: every labelled scan comes from ONE session. Validation scans are "
            "other frequencies/sensors of the same placements, so the scores will be "
            "OPTIMISTIC (the model can memorise positions). Label a second session for an "
            "honest test.")
    return split


# ---------------------------------------------------------------- training

def train(epochs=80, log=print, on_epoch=None, val_sessions=None, use_all=False):
    # ultralytics downloads the base weights and, on a GPU, yolo26n.pt for its AMP check
    # into the CURRENT folder (its weights_dir setting is relative): run from results/model
    # so they land there, not in the project root
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(MODEL_DIR)
    try:
        return _train(epochs, log, on_epoch, val_sessions, use_all)
    finally:
        os.chdir(cwd)


def _train(epochs, log, on_epoch, val_sessions, use_all):
    split = export_dataset(log=log, val_sessions=val_sessions, use_all=use_all)
    if not split["val"] and not use_all:
        raise ValueError("No validation scans - label more scans.")
    YOLO = _yolo()
    dev = device()
    log(f"Training on {device_name(dev)}")
    model = YOLO(BASE_MODEL)                  # COCO-pretrained, fine-tuned here
    if on_epoch:
        model.add_callback("on_train_epoch_end",
                           lambda tr: on_epoch(tr.epoch + 1, tr.epochs))
    res = model.train(
        data=str(DATASET_DIR / "data.yaml"), epochs=epochs, imgsz=IMGSZ,
        batch=16 if dev != "cpu" else 8, device=dev,
        # cache="ram": decoding the PNGs every epoch starved the GPU (505 s -> 82 s for
        # 80 epochs on an RTX 3050). workers=0: DataLoader worker processes fail to
        # spawn on Windows here, and with the cache they are not needed.
        workers=0, cache="ram",
        project=str(MODEL_DIR), name="run", exist_ok=True,
        patience=max(20, epochs // 3), plots=False, verbose=False,
        # the channels are signal/derivatives, not colours: no colour jitter, no rotation
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, fliplr=0.5, flipud=0.5, mosaic=0.5)
    best = Path(res.save_dir) / "weights" / "best.pt"
    shutil.copy(best, WEIGHTS)
    m = res.results_dict
    metrics = {k.split("/")[1].replace("(B)", ""): round(float(v), 3)
               for k, v in m.items() if k.startswith("metrics/")}
    metrics["honest_validation"] = split["honest_validation"]
    metrics["all_data"] = use_all
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log(f"Model saved: {WEIGHTS}")
    log(f"Validation: precision {metrics.get('precision')}  recall {metrics.get('recall')}  "
        f"mAP50 {metrics.get('mAP50')}  mAP50-95 {metrics.get('mAP50-95')}"
        + ("   (on TRAINING data)" if use_all else
           "" if split["honest_validation"] else "   (OPTIMISTIC - same plate position)"))
    return metrics


# ---------------------------------------------------------------- inference

_cached = {}


def predict(tdms, conf=CONF, drop_edge=True):
    """Boxes [{x0, y0, x1, y1, conf}] in original grid pixels."""
    if not WEIGHTS.exists():
        raise FileNotFoundError("No trained model yet - press 'Train crack model' first.")
    key = WEIGHTS.stat().st_mtime
    if _cached.get("key") != key:
        _cached.update(key=key, model=_yolo()(str(WEIGHTS)))
    grid, _ = backend.load_scan(tdms)
    img = crack_labels.model_image(grid)[..., ::-1]   # RGB -> BGR: what the trainer read
    r = _cached["model"].predict(img, imgsz=IMGSZ, conf=conf, device=device(), verbose=False)[0]
    H, W = grid.shape
    out = []
    for (x0, y0, x1, y1), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
        b = {"x0": int(max(0, round(x0))), "y0": int(max(0, round(y0))),
             "x1": int(min(W, round(x1))), "y1": int(min(H, round(y1))), "conf": float(c)}
        # a box on the scan border is the plate edge, not a crack (same rule as Locate
        # cracks); seen on early_session, where the plate edge is inside the scan
        if drop_edge and (b["x0"] <= 0 or b["y0"] <= 0 or b["x1"] >= W or b["y1"] >= H):
            continue
        out.append(b)
    return out, grid


def in_training_set(tdms):
    if not SPLIT_FILE.exists():
        return False
    rel = str(Path(tdms).resolve().relative_to(backend.PROJECT_ROOT))
    return rel in json.loads(SPLIT_FILE.read_text(encoding="utf-8"))["train"]


def _iou(a, b):
    ix = max(0, min(a["x1"], b["x1"]) - max(a["x0"], b["x0"]))
    iy = max(0, min(a["y1"], b["y1"]) - max(a["y0"], b["y0"]))
    inter = ix * iy
    area = lambda r: (r["x1"] - r["x0"]) * (r["y1"] - r["y0"])
    return inter / (area(a) + area(b) - inter) if inter else 0.0


def match_to_labels(pred, labels, iou=0.3):
    """Greedy one-to-one matching (highest confidence first) -> (matched flag per
    prediction, indices of the labels no prediction matched)."""
    free = list(range(len(labels)))
    hit = [False] * len(pred)
    for j in sorted(range(len(pred)), key=lambda j: -pred[j]["conf"]):
        best = max(free, key=lambda i: _iou(pred[j], labels[i]), default=None)
        if best is not None and _iou(pred[j], labels[best]) >= iou:
            free.remove(best)
            hit[j] = True
    return hit, free


def score_against_labels(pred, labels, iou=0.3):
    """-> (tp, fp, fn)."""
    hit, missed = match_to_labels(pred, labels, iou)
    return sum(hit), len(pred) - sum(hit), len(missed)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("export")
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=80)
    t.add_argument("--val-session", action="append",
                   help="session folder to hold out for validation (repeatable); "
                        "default: the smallest labelled session")
    t.add_argument("--all", action="store_true",
                   help="train on EVERY labelled scan (final model, no held-out test)")
    pr = sub.add_parser("predict")
    pr.add_argument("tdms")
    pr.add_argument("--conf", type=float, default=CONF)
    a = ap.parse_args()
    if a.cmd == "export":
        export_dataset()
    elif a.cmd == "train":
        train(a.epochs, val_sessions=a.val_session, use_all=a.all)
    else:
        boxes, _ = predict(a.tdms, a.conf)
        for b in boxes:
            print(b)
        labels = crack_labels.load_labels(a.tdms)
        if labels:
            tp, fp, fn = score_against_labels(boxes, labels)
            print(f"vs labels: {tp} found, {fp} false, {fn} missed"
                  + ("   (scan was in the TRAINING set)" if in_training_set(a.tdms) else ""))


if __name__ == "__main__":
    main()
