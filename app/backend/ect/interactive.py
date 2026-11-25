"""The numbers behind a figure, for the interactive viewer (app/frontend/viewer.js).

A figure is {v, mode, subtitle, ncols, panels, cmaps, box_colors, profile}. Every map panel
carries the whole grid of values it shows (float32: never clipped, resampled or smoothed) in the
same frame as the PNG, so the number shown for a point is the number the tools computed, not a
colour read back from a picture. Arrays travel as base64; nothing here is drawn."""
import base64

import matplotlib
import numpy as np

from . import render

VERSION = 1


def _b64(arr, dtype):
    return base64.b64encode(np.ascontiguousarray(arr, dtype=dtype).tobytes()).decode("ascii")


def _grid(arr, offset=(0, 0)):
    arr = np.asarray(arr)
    return {"w": int(arr.shape[1]), "h": int(arr.shape[0]), "ox": int(offset[0]), "oy": int(offset[1]),
            "f32": _b64(arr, "<f4")}


def _px(box):
    return {k: int(box[k]) for k in ("x0", "y0", "x1", "y1")}


def boxes_json(boxes, missed=()):
    """Boxes in ORIGINAL grid pixels, numbered like the PNG and the detection list."""
    out = []
    for i, b in enumerate(boxes or [], start=1):
        kind = "plain" if "match" not in b else ("found" if b["match"] else "false")
        item = {"n": i, "kind": kind, **_px(b)}
        for k in ("conf", "score"):
            if k in b:
                item[k] = float(b[k])
        out.append(item)
    out += [{"n": None, "kind": "missed", **_px(b)} for b in missed]
    return out


def readout(name, arr, offset=(0, 0)):
    """An extra value reported at every point (e.g. the raw reading under a filtered map)."""
    return {"name": name, **_grid(arr, offset)}


def point(st, pitch, x_mm, y_mm):
    """The values at one position, exactly as the viewer reads them: the scan point that covers it,
    no interpolation. Cropped stages (detrended, dx, z) are None outside the kept region."""
    col, row = render.cell_of(x_mm, pitch), render.cell_of(y_mm, pitch)
    h, w = st["raw"].shape
    if not (0 <= col < w and 0 <= row < h):
        raise ValueError(f"({x_mm}, {y_mm}) mm is outside the scan: X 0-{w * pitch:.1f}, Y 0-{h * pitch:.1f} mm")
    ox, oy = st["offset"]
    out = {"col": col, "row": row, "x_mm": round((col + 0.5) * pitch, 2), "y_mm": round((row + 0.5) * pitch, 2),
           "raw": float(st["raw"][row, col]), "filtered": float(st["filtered"][row, col])}
    ch, cw = st["cropped"].shape
    inside = 0 <= col - ox < cw and 0 <= row - oy < ch
    for name, stage in (("detrended", "detrended"), ("dx", "dx"), ("z", "zmap")):
        out[name] = float(st[stage][row - oy, col - ox]) if inside else None
    return out


def map_panel(img, pitch, offset=(0, 0), title="", cmap="magma", diverging=False, label="", limits=None,
              boxes=None, missed=(), shade=(), rect=None, lines=(), window=None, readouts=(), link=None,
              name=None, clip_pct=2.0):
    """One map. Colours run over the same limits as the PNG (percentiles); the values do not.
    shade / rect / lines / window are in mm; panels with the same link key share the crosshair
    and the zoom; name = short name in the readout."""
    lo, hi = render.color_limits(img, diverging, limits, clip_pct)
    return {"type": "map", "title": title, "name": name or title, "pitch": float(pitch), **_grid(img, offset),
            "lo": float(lo), "hi": float(hi), "cmap": cmap, "label": label, "link": link,
            "boxes": boxes_json(boxes, missed), "readouts": list(readouts),
            "shade": [[float(v) for v in s] for s in shade],
            "rect": [float(v) for v in rect] if rect else None,
            "lines": [{"pts": [[float(x), float(y)] for x, y in ln["pts"]], "color": ln["color"],
                       "label": ln.get("label", "")} for ln in lines],
            "window": [float(v) for v in window] if window else None}


def rgb_panel(img, pitch, title="", channels=("R", "G", "B"), offset=(0, 0)):
    """An RGB image (the YOLO input); the readout gives the three channel values 0-255."""
    H, W = img.shape[:2]
    return {"type": "rgb", "title": title, "name": title, "pitch": float(pitch), "w": int(W), "h": int(H),
            "ox": int(offset[0]), "oy": int(offset[1]), "u8": _b64(img[..., :3], "u1"),
            "channels": list(channels), "boxes": [], "link": None}


def line_panel(y, title="", xlabel="", ylabel="", vlines=(), notes=()):
    """A 1D series against its index. notes = [(x, text)] along the top."""
    y = np.asarray(y, dtype=float).ravel()
    return {"type": "line", "title": title, "name": title, "xlabel": xlabel, "ylabel": ylabel, "n": int(y.size),
            "f32": _b64(y, "<f4"), "vlines": [float(v) for v in vlines],
            "notes": [[float(x), str(t)] for x, t in notes], "boxes": [], "link": None}


def figure(panels, subtitle="", mode="2d", ncols=2, profile=None):
    """profile = {"along": "X"|"Y", "pos": mm}: open the cut through the first map."""
    cmaps = {}
    for p in panels:
        name = p.get("cmap")
        if name and name not in cmaps:
            cmaps[name] = _b64(matplotlib.colormaps[name](np.arange(256), bytes=True)[:, :3], "u1")
    return {"v": VERSION, "mode": mode, "subtitle": subtitle, "ncols": int(min(ncols, len(panels))),
            "panels": panels, "cmaps": cmaps, "box_colors": render.BOX_COLORS, "profile": profile}


def steps(rs, pitch, st, boxes, box_note="", subtitle=""):
    """render.steps as data (without its schematic of the scan path): the recorded list, then the
    four maps. rs = scan.raster_steps(), st = process.pipeline()."""
    ny, nx = rs["recorded"].shape
    lines = min(3, ny)
    arrows = ["line 1 \u2192", "line 2 \u2190", "line 3 \u2192"]
    maps = [(rs["recorded"], "Cut into lines: every 2nd line is still backwards", "recorded", None),
            (rs["aligned"], f"Every 2nd line reversed, probe lag fixed ({rs['shift_px']:+d} px)", "aligned", "grid"),
            (st["filtered"], "Low-pass filter down each column + median", "filtered", "grid")]
    panels = [line_panel(rs["sequence"][:lines * nx], f"The file is one long list ({rs['sequence'].size} values; "
                         f"first {lines} scan lines)", "sample number in the file", "value",
                         vlines=[k * nx for k in range(1, lines)],
                         notes=[(k * nx + nx / 2, arrows[k]) for k in range(lines)])]
    panels += [map_panel(img, pitch, title=t, cmap="viridis", name=n, link=lk) for img, t, n, lk in maps]
    panels.append(map_panel(st["cropped"], pitch, st["offset"], "Border cropped, "
                            f"{len(boxes)} cracks found{': ' + box_note if box_note else ''}",
                            "magma", name="cropped", link="grid", boxes=boxes))
    return figure(panels, subtitle, ncols=3)
