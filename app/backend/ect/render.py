"""Figures as PNG bytes. Every map is drawn in mm (scan position), with its colour scale
clipped at percentiles so a few extreme pixels cannot wash out the cracks."""
import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import TwoSlopeNorm

from . import process

INK, MUTED = "#1b1f24", "#5f6b76"
BOX_COLORS = {"found": "#2fbf71", "false": "#ff4d4f", "missed": "#ff6ad5", "plain": "#22d3ee"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": "#c9d1d9", "figure.facecolor": "white"})

VIEWS = {
    # key: (title, cmap, diverging, stage)
    "raw": ("Thô (chưa lọc)", "viridis", False, "raw"),
    "filtered": ("Đã lọc nhiễu", "viridis", False, "filtered"),
    "cropped": ("Đã cắt viền", "viridis", False, "cropped"),
    "amplitude": ("Bản đồ {signal}", "magma", False, "cropped"),
    "detrended": ("Đã trừ nền", "magma", False, "detrended"),
    "dx": ("Đạo hàm theo X (dA/dx)", "RdBu_r", True, "dx"),
    "dx_inv": ("Đạo hàm nghịch (−dA/dx)", "RdBu_r", True, "dx"),
    "zmap": ("Bản đồ z (độ lệch so với nền)", "RdBu_r", True, "zmap"),
}


def signal_name(sensor):
    return "pha" if sensor == "P" else "biên độ"


def png(fig, dpi=130):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def preview(png_bytes, max_px=640):
    """Small copy of a figure for the model to look at (keeps the context light)."""
    from PIL import Image
    im = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    im.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def view_spec(stages, view, sensor=None):
    """(img, offset, title, cmap, diverging) for one named view of a pipeline() result."""
    if view not in VIEWS:
        raise ValueError(f"view must be one of {sorted(VIEWS)}")
    title, cmap, div, stage = VIEWS[view]
    img = stages[stage]
    if view == "dx_inv":
        img = -img
    offset = (0, 0) if stage in ("raw", "filtered") else stages["offset"]
    return img, offset, title.format(signal=signal_name(sensor)), cmap, div


def _extent(shape, offset, pitch):
    H, W = shape
    ox, oy = offset
    return [ox * pitch, (ox + W) * pitch, (oy + H) * pitch, oy * pitch]


def draw_map(ax, img, pitch, offset=(0, 0), cmap="magma", diverging=False, title="",
             cbar_label="", clip_pct=2.0, colorbar=True, limits=None):
    """One map in mm, colour clipped at percentiles (±99th |value| when diverging)."""
    if diverging:
        lim = float(np.nanpercentile(np.abs(img), 99)) or float(np.nanmax(np.abs(img))) or 1.0
        kw = {"norm": TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim)}
    else:
        lo, hi = limits or process.percentile_limits(img, clip_pct)
        kw = {"vmin": lo, "vmax": hi}
    im = ax.imshow(img, cmap=cmap, extent=_extent(img.shape, offset, pitch),
                   interpolation="bilinear", aspect="equal", **kw)
    ax.set_title(title, color=INK, loc="left")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    if colorbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, extend="both")
        cb.set_label(cbar_label, color=MUTED)
        cb.ax.tick_params(colors=MUTED)
    return im


def draw_boxes(ax, boxes, pitch, numbered=True, missed=()):
    """Boxes in ORIGINAL grid pixels drawn on a map in mm. A box with 'match' is green
    (matches a label) or red (false alarm); labels in `missed` are dotted pink."""
    for i, b in enumerate(boxes, start=1):
        kind = "plain" if "match" not in b else ("found" if b["match"] else "false")
        col = BOX_COLORS[kind]
        ax.add_patch(patches.Rectangle((b["x0"] * pitch, b["y0"] * pitch), (b["x1"] - b["x0"]) * pitch,
                                       (b["y1"] - b["y0"]) * pitch, fill=False, lw=1.6, ec=col))
        if numbered:
            tag = f"{i}" + (f" {b['conf']:.2f}" if "conf" in b else "")
            ax.text(b["x0"] * pitch, b["y0"] * pitch - 0.6, tag, color=col, fontsize=8,
                    fontweight="bold", va="bottom")
    for b in missed:
        ax.add_patch(patches.Rectangle((b["x0"] * pitch, b["y0"] * pitch), (b["x1"] - b["x0"]) * pitch,
                                       (b["y1"] - b["y0"]) * pitch, fill=False, lw=1.4,
                                       ec=BOX_COLORS["missed"], ls=":"))


def single(img, pitch, offset, title, cmap, diverging, cbar_label="", boxes=None, missed=(),
           suptitle=None, limits=None):
    H, W = img.shape
    fig, ax = plt.subplots(figsize=(7.2, 7.2 * H / W + 0.9))
    draw_map(ax, img, pitch, offset, cmap, diverging, title, cbar_label, limits=limits)
    if boxes or missed:
        draw_boxes(ax, boxes or [], pitch, missed=missed)
    if suptitle:
        fig.suptitle(suptitle, x=0.01, ha="left", fontsize=9, color=MUTED)
    return png(fig)


def panels(items, ncols=2, suptitle=None, width=6.0):
    """items: dicts with img, pitch, offset, title, cmap, diverging, [cbar, boxes, missed, line]."""
    n = len(items)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    h, w = items[0]["img"].shape
    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, width * h / w * nrows + 0.8),
                             squeeze=False)
    for ax, it in zip(axes.flat, items):
        draw_map(ax, it["img"], it["pitch"], it.get("offset", (0, 0)), it.get("cmap", "magma"),
                 it.get("diverging", False), it.get("title", ""), it.get("cbar", ""),
                 limits=it.get("limits"))
        if it.get("boxes") or it.get("missed"):
            draw_boxes(ax, it.get("boxes") or [], it["pitch"], missed=it.get("missed", ()))
        if it.get("rect"):
            x0, y0, x1, y1 = it["rect"]
            ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=1.8,
                                           ec="#22d3ee", ls="--"))
        for band in it.get("shade", []):
            ax.add_patch(patches.Rectangle(band[:2], band[2], band[3], fc="#ff4d4f", alpha=0.28, lw=0))
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, x=0.01, ha="left", fontsize=10, color=MUTED)
    fig.tight_layout()
    return png(fig)


def rgb(img, pitch, title, offset=(0, 0)):
    """An RGB image (e.g. the 3-channel YOLO input) in mm."""
    H, W = img.shape[:2]
    fig, ax = plt.subplots(figsize=(7.2, 7.2 * H / W + 0.6))
    ax.imshow(img, extent=_extent((H, W), offset, pitch), interpolation="nearest")
    ax.set_title(title, color=INK, loc="left")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    return png(fig)


def crop_bands(shape, m, pitch):
    """Shaded rectangles (x, y, w, h in mm) for the border bands a crop removes."""
    H, W = shape
    b = []
    if m["top"]:
        b.append((0, 0, W * pitch, m["top"] * pitch))
    if m["bottom"]:
        b.append((0, (H - m["bottom"]) * pitch, W * pitch, m["bottom"] * pitch))
    if m["left"]:
        b.append((0, 0, m["left"] * pitch, H * pitch))
    if m["right"]:
        b.append(((W - m["right"]) * pitch, 0, m["right"] * pitch, H * pitch))
    return b


def profile_figure(img, pitch, offset, along, pos_mm, title, cmap, diverging):
    """Map with the cut line, and the values along it."""
    H, W = img.shape
    ox, oy = offset
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.2, 1]})
    draw_map(a1, img, pitch, offset, cmap, diverging, title, colorbar=False)
    if along == "X":                                  # a row at Y = pos
        r = int(round(pos_mm / pitch)) - oy
        if not 0 <= r < H:
            raise ValueError(f"Y = {pos_mm} mm is outside {oy * pitch:.1f}–{(oy + H) * pitch:.1f} mm")
        x = (np.arange(W) + ox + 0.5) * pitch
        a1.axhline(pos_mm, color="#22d3ee", lw=1.5)
        a2.plot(x, img[r], color=INK, lw=1.3)
        a2.set_xlabel("X (mm)")
    else:                                             # a column at X = pos
        c = int(round(pos_mm / pitch)) - ox
        if not 0 <= c < W:
            raise ValueError(f"X = {pos_mm} mm is outside {ox * pitch:.1f}–{(ox + W) * pitch:.1f} mm")
        y = (np.arange(H) + oy + 0.5) * pitch
        a1.axvline(pos_mm, color="#22d3ee", lw=1.5)
        a2.plot(y, img[:, c], color=INK, lw=1.3)
        a2.set_xlabel("Y (mm)")
    a2.set_title(f"Lát cắt {'ngang' if along == 'X' else 'dọc'} tại {'Y' if along == 'X' else 'X'} = {pos_mm:g} mm",
                 loc="left", color=INK)
    a2.grid(alpha=0.3)
    fig.tight_layout()
    return png(fig)


def measure_figure(img, pitch, offset, box, m, title):
    """Zoom on one box with the measured length (cyan) and width (orange) axes."""
    ox, oy = offset
    pad = 6
    H, W = img.shape
    x0, y0 = max(0, box["x0"] - ox - pad), max(0, box["y0"] - oy - pad)
    x1, y1 = min(W, box["x1"] - ox + pad), min(H, box["y1"] - oy + pad)
    sub = img[y0:y1, x0:x1]
    fig, ax = plt.subplots(figsize=(5.5, 5.5 * sub.shape[0] / sub.shape[1] + 0.8))
    draw_map(ax, sub, pitch, (x0 + ox, y0 + oy), "magma", False, title)
    draw_boxes(ax, [box], pitch, numbered=False)
    if m:
        col = "#22d3ee" if m["reliable"] else "#ffa940"
        (a, b), (c, d) = m["length_ends"], m["width_ends"]
        ax.plot([a[0], b[0]], [a[1], b[1]], color=col, lw=2.2)
        ax.plot([c[0], d[0]], [c[1], d[1]], color="#ffa940", lw=2.2)
    return png(fig)
