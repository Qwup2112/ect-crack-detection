"""Export the crack model to ONNX and prove the app's onnxruntime path gives the same boxes.

The app runs YOLO on onnxruntime (~50 MB) instead of PyTorch (well over 1 GB, which this PC
often cannot spare). Re-run this after every retraining of results/model/crack_yolo.pt:

    .venv\\Scripts\\python -m app.tools.export_onnx

Needs torch + ultralytics + onnx once, for the export and the comparison. Exit code 0 = the
ONNX model matches PyTorch on every readable scan.
"""
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")    # CPU only: CUDA start-up alone costs ~1 GB
os.environ.setdefault("CRACK_DEVICE", "cpu")

from app.backend.ect import detect, scan            # noqa: E402
from app.backend.ect.paths import ONNX_WEIGHTS, WEIGHTS  # noqa: E402

CONF = 0.25          # compare below the app's 0.5 so borderline boxes are checked too
IOU_SAME = 0.9       # a box counts as the same box above this overlap
CONF_TOL = 0.02


def export():
    from ultralytics import YOLO
    # dynamic: predict letterboxes to a rectangle (150 x 101 points -> 320 x 224), not a square
    out = YOLO(str(WEIGHTS)).export(format="onnx", imgsz=detect.YOLO_IMGSZ, dynamic=True, simplify=False,
                                    half=False, nms=False, device="cpu", verbose=False)
    if os.path.abspath(out) != os.path.abspath(ONNX_WEIGHTS):
        os.replace(out, ONNX_WEIGHTS)
    print(f"exported {ONNX_WEIGHTS} ({ONNX_WEIGHTS.stat().st_size / 1e6:.1f} MB)")


def compare():
    bad, n_boxes, worst = [], 0, 0.0
    ok = [e for e in scan.catalog() if e["status"] == "ok" or e["status"].startswith("truncated")]
    for e in ok:
        grid, _ = scan.load_scan(scan.scan_path(e["id"]))
        a = detect.yolo_detect(grid, CONF, drop_edge=False, backend="torch")
        b = detect.yolo_detect(grid, CONF, drop_edge=False, backend="onnx")
        n_boxes += len(a)
        free = list(range(len(b)))
        for p in a:
            j = max(free, key=lambda k: detect.iou(p, b[k]), default=None)
            if j is None or detect.iou(p, b[j]) < IOU_SAME:
                if p["conf"] > CONF + CONF_TOL:      # a box right at the threshold may flip: not a fault
                    bad.append((e["id"], "missing in onnx", p))
                continue
            worst = max(worst, abs(p["conf"] - b[j]["conf"]))
            free.remove(j)
        bad += [(e["id"], "extra in onnx", b[k]) for k in free if b[k]["conf"] > CONF + CONF_TOL]
    print(f"{len(ok)} scans, {n_boxes} PyTorch boxes at conf {CONF}; largest confidence difference {worst:.4f}")
    for row in bad[:20]:
        print("  MISMATCH", *row)
    return not bad and worst <= CONF_TOL


def main():
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding="utf-8", errors="replace")
    t0 = time.time()
    if "--compare-only" not in sys.argv:
        export()
    good = compare()
    print(("OK" if good else "ONNX DIFFERS FROM PYTORCH") + f" in {time.time() - t0:.0f}s")
    sys.exit(0 if good else 1)


if __name__ == "__main__":
    main()
