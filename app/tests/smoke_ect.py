"""Fast end-to-end check of the ect package on real scans: read, crop, detect, draw.
Run:  .venv\\Scripts\\python -m app.tests.smoke_ect [--yolo]
Exit code 0 = pass. The dev hook runs it (without --yolo) after edits to app/backend/ect.
"""
import sys
import time

from app.backend.ect import scan, process, detect, render

# Two scans whose behaviour is known from the legacy app (best R scan, a P scan).
CASES = ["cambien_4lop_2 R 1.7V 300k", "cambien_4lop_2 P 0.3V 20k 145x95"]


def main():
    for stream in (sys.stdout, sys.stderr):         # non-ASCII text (σ, –, °) on a cp1252 console
        stream.reconfigure(encoding="utf-8", errors="replace")
    t0 = time.time()
    use_yolo = "--yolo" in sys.argv
    cat = scan.catalog()
    ok = [e for e in cat if e["status"] == "ok"]
    assert len(cat) >= 100 and len(ok) >= 80, f"catalogue looks wrong: {len(cat)} files, {len(ok)} ok"
    for q in CASES:
        sid = scan.resolve(q)
        grid, meta = scan.load_scan(scan.scan_path(sid))
        st = process.pipeline(grid)
        m = st["margins"]
        assert st["cropped"].shape[0] < grid.shape[0] and any(m.values()), f"{sid}: no border crop"
        boxes, _ = detect.threshold_detect(st["cropped"], st["offset"])
        assert 5 <= len(boxes) <= 40, f"{sid}: {len(boxes)} threshold boxes"
        img, off, title, cmap, div = render.view_spec(st, "amplitude", meta.get("sensor"))
        data = render.single(img, meta["pitch_mm"], off, title, cmap, div, boxes=boxes)
        assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 10_000
        line = f"{sid}: crop {process.describe_margins(m)}, {len(boxes)} candidates, z_top5 {detect.z_top5(boxes)}"
        if use_yolo:
            yb = detect.yolo_detect(grid)
            labels, _ = detect.load_labels(sid)
            found, false, missed = detect.match_labels(yb, labels)
            line += f", YOLO {len(yb)} boxes (found {found} / false {false} / missed {len(missed)})"
        print(line)
    print(f"OK in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"SMOKE TEST FAILED: {e}")
        sys.exit(1)
