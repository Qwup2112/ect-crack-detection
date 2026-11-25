"""Figures as PNG bytes. Every map is drawn in mm (scan position), with its colour scale
clipped at percentiles so a few extreme pixels cannot wash out the cracks."""
import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize, TwoSlopeNorm

from . import process

INK, MUTED = "#1b1f24", "#5f6b76"
BOX_COLORS = {"found": "#2fbf71", "false": "#ff4d4f", "missed": "#ff6ad5", "plain": "#22d3ee"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": "#c9d1d9", "figure.facecolor": "white"})

VIEWS = {
    # key: (title, cmap, diverging, stage)
    "raw": ("Raw (unfiltered)", "viridis", False, "raw"),
    "filtered": ("Filtered", "viridis", False, "filtered"),
    "cropped": ("Border cropped", "viridis", False, "cropped"),
    "amplitude": ("{signal} map", "magma", False, "cropped"),
    "detrended": ("Background removed", "magma", False, "detrended"),
    "dx": ("X derivative (dA/dx)", "RdBu_r", True, "dx"),
    "dx_inv": ("Inverted X derivative (−dA/dx)", "RdBu_r", True, "dx"),
    "zmap": ("z-map (deviation from background)", "RdBu_r", True, "zmap"),
}


def signal_name(sensor):
    return "Phase" if sensor == "P" else "Amplitude"


def quantity(view, sensor):
    """What the numbers of a view are, with the unit (colour-bar label, readouts)."""
    name, stage = signal_name(sensor).lower(), VIEWS[view][3]
    if stage == "zmap":
        return "z (robust sigmas)"
    if stage == "dx":
        return f"{'-' if view == 'dx_inv' else ''}d({name})/dx (a.u. per point)"
    return f"{name} (a.u.)"


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


def color_limits(img, diverging, limits=None, clip_pct=2.0):
    """The values at the two ends of the colour scale."""
    if diverging:
        lim = float(np.nanpercentile(np.abs(img), 99)) or float(np.nanmax(np.abs(img))) or 1.0
        return -lim, lim
    return limits or process.percentile_limits(img, clip_pct)


def draw_map(ax, img, pitch, offset=(0, 0), cmap="magma", diverging=False, title="",
             cbar_label="", clip_pct=2.0, colorbar=True, limits=None):
    """One map in mm, colour clipped at percentiles (±99th |value| when diverging)."""
    lo, hi = color_limits(img, diverging, limits, clip_pct)
    kw = {"norm": TwoSlopeNorm(vcenter=0.0, vmin=lo, vmax=hi)} if diverging else {"vmin": lo, "vmax": hi}
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


def cell_of(pos_mm, pitch):
    """Index of the scan point that covers a position: point i spans [i, i + 1) * pitch mm."""
    return int(np.floor(pos_mm / pitch + 1e-9))


def profile_figure(img, pitch, offset, along, pos_mm, title, cmap, diverging):
    """Map with the cut line, and the values along it."""
    H, W = img.shape
    ox, oy = offset
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.2, 1]})
    draw_map(a1, img, pitch, offset, cmap, diverging, title, colorbar=False)
    if along == "X":                                  # a row at Y = pos
        r = cell_of(pos_mm, pitch) - oy
        if not 0 <= r < H:
            raise ValueError(f"Y = {pos_mm} mm is outside {oy * pitch:.1f}–{(oy + H) * pitch:.1f} mm")
        x = (np.arange(W) + ox + 0.5) * pitch
        a1.axhline(pos_mm, color="#22d3ee", lw=1.5)
        a2.plot(x, img[r], color=INK, lw=1.3)
        a2.set_xlabel("X (mm)")
    else:                                             # a column at X = pos
        c = cell_of(pos_mm, pitch) - ox
        if not 0 <= c < W:
            raise ValueError(f"X = {pos_mm} mm is outside {ox * pitch:.1f}–{(ox + W) * pitch:.1f} mm")
        y = (np.arange(H) + oy + 0.5) * pitch
        a1.axvline(pos_mm, color="#22d3ee", lw=1.5)
        a2.plot(y, img[:, c], color=INK, lw=1.3)
        a2.set_xlabel("Y (mm)")
    a2.set_title(f"{'Horizontal' if along == 'X' else 'Vertical'} profile at {'Y' if along == 'X' else 'X'} = {pos_mm:g} mm",
                 loc="left", color=INK)
    a2.grid(alpha=0.3)
    fig.tight_layout()
    return png(fig)


def surface(img, pitch, offset=(0, 0), cmap="magma", diverging=False, title="", zlabel="",
            boxes=None, suptitle=None, limits=None, elev=50, azim=-62):
    """The map as a 3D surface: height and colour = value, X/Y in mm. Values are clipped like
    the 2D maps (percentiles), so a few spikes cannot flatten the cracks. Boxes (ORIGINAL grid
    pixels) are traced on the surface."""
    H, W = img.shape
    ox, oy = offset
    lo, hi = color_limits(img, diverging, limits)
    z = np.clip(np.nan_to_num(np.asarray(img, dtype=float), nan=lo), lo, hi)
    X, Y = np.meshgrid((np.arange(W) + ox + 0.5) * pitch, (np.arange(H) + oy + 0.5) * pitch)
    fig = plt.figure(figsize=(8.6, 6.4))
    ax = fig.add_subplot(projection="3d")
    ax.computed_zorder = False                        # box outlines stay visible over the peaks
    norm = TwoSlopeNorm(vcenter=0.0, vmin=lo, vmax=hi) if diverging else Normalize(lo, hi)
    surf = ax.plot_surface(X, Y, z, cmap=cmap, norm=norm, rstride=1, cstride=1, linewidth=0,
                           antialiased=False, zorder=1)
    for b in boxes or []:                             # outline each box along the surface
        x0, x1 = max(b["x0"] - ox, 0), min(b["x1"] - ox, W) - 1
        y0, y1 = max(b["y0"] - oy, 0), min(b["y1"] - oy, H) - 1
        if x1 <= x0 or y1 <= y0:
            continue
        ring = ([(y0, x) for x in range(x0, x1 + 1)] + [(y, x1) for y in range(y0, y1 + 1)]
                + [(y1, x) for x in range(x1, x0 - 1, -1)] + [(y, x0) for y in range(y1, y0 - 1, -1)])
        ys, xs = np.array(ring).T
        ax.plot(X[ys, xs], Y[ys, xs], z[ys, xs] + 0.02 * (hi - lo), color=BOX_COLORS["plain"], lw=1.4,
                zorder=5)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlim(lo, hi)                               # the value's name is on the colour bar
    ax.invert_yaxis()                                 # Y grows towards the viewer, like the 2D maps
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((W, H, 0.35 * max(W, H)))      # X and Y in proportion, height readable
    ax.tick_params(colors=MUTED, labelsize=8)
    cb = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, extend="both")
    cb.set_label(zlabel, color=MUTED)
    ax.set_title(title, color=INK, loc="left")
    if suptitle:
        fig.suptitle(suptitle, x=0.01, ha="left", fontsize=9, color=MUTED)
    return png(fig)


def steps(rs, pitch, st, boxes, box_note="", suptitle=None):
    """How one scan becomes the heatmap, in six panels: the zig-zag scan path, the recorded
    list, the list cut into lines, every 2nd line reversed + lag fixed, the filter, the cropped
    map with detections. rs = scan.raster_steps(), st = process.pipeline()."""
    seq, recorded, aligned = rs["sequence"], rs["recorded"], rs["aligned"]
    ny, nx = recorded.shape
    fig = plt.figure(figsize=(18.5, 9.6))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.42)

    ax = fig.add_subplot(gs[0, 0])                    # 1. the scan path (schematic)
    cols, rows = 9, 5
    path = [(c, r) for r in range(rows) for c in (range(cols) if r % 2 == 0 else range(cols - 1, -1, -1))]
    px, py = zip(*path)
    ax.plot(px, py, "-", color="#c87d55", lw=1.6, zorder=1)
    ax.scatter(px, py, s=22, color=INK, zorder=2)
    for r in range(rows):
        a, b = (cols - 2.3, cols - 1.3) if r % 2 == 0 else (1.3, 0.3)
        ax.annotate("", (b, r), (a, r), arrowprops={"arrowstyle": "-|>", "color": "#c87d55", "lw": 1.6})
    ax.text(0, -0.55, "start", ha="center", fontsize=9, color=MUTED)
    ax.set_xlim(-0.8, cols - 0.2)
    ax.set_ylim(rows - 0.4, -1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("X: fast axis (the probe sweeps a line)")
    ax.set_ylabel("Y: slow axis (one step after each line)")
    ax.set_title(f"1. The probe scans the plate zig-zag\n({nx} x {ny} points, ~{pitch:.1f} mm apart; "
                 "one value per point)", color=INK, loc="left")

    ax = fig.add_subplot(gs[0, 1])                    # 2. what the file stores
    n = min(3, ny) * nx
    ax.plot(np.arange(n), seq[:n], color=INK, lw=0.9)
    for k in range(1, min(3, ny)):
        ax.axvline(k * nx, color="#c87d55", ls="--", lw=1)
    for k, label in enumerate(["line 1  →", "line 2  ←", "line 3  →"][:min(3, ny)]):
        ax.text(k * nx + nx / 2, ax.get_ylim()[1], label, ha="center", va="bottom", fontsize=9, color="#c87d55")
    ax.set_xlabel("sample number in the file")
    ax.set_ylabel("value")
    ax.set_title(f"2. The TDMS file is one long list\n({seq.size} values; first 3 scan lines shown)",
                 color=INK, loc="left", pad=16)
    ax.grid(alpha=0.3)

    draw_map(fig.add_subplot(gs[0, 2]), recorded, pitch, cmap="viridis",
             title="3. Cut the list into lines - but every 2nd\nline was recorded backwards: cracks smear")
    draw_map(fig.add_subplot(gs[1, 0]), aligned, pitch, cmap="viridis",
             title=f"4. Reverse every 2nd line + fix the probe lag\n({rs['shift_px']:+d} px, reshape_raster): "
                   "cracks line up")
    draw_map(fig.add_subplot(gs[1, 1]), st["filtered"], pitch, cmap="viridis",
             title="5. Low-pass filter down each column\n(FIR Hamming, order 24) + median: no comb")
    ax = fig.add_subplot(gs[1, 2])
    draw_map(ax, st["cropped"], pitch, st["offset"], "magma",
             title=f"6. Crop the noisy border, draw in mm\n({len(boxes)} cracks found{': ' + box_note if box_note else ''})")
    draw_boxes(ax, boxes, pitch, numbered=False)
    if suptitle:
        fig.suptitle(suptitle, x=0.01, ha="left", fontsize=13, color=INK)
    return png(fig, dpi=110)


def measure_window(shape, offset, box, pad=6):
    """(x0, y0, x1, y1) of a box plus a margin, in pixels of the map."""
    H, W = shape
    ox, oy = offset
    return (max(0, box["x0"] - ox - pad), max(0, box["y0"] - oy - pad),
            min(W, box["x1"] - ox + pad), min(H, box["y1"] - oy + pad))


def measure_lines(m):
    """The measured length and width axes in mm: [{pts, color, label}]."""
    if not m:
        return []
    return [{"pts": m["length_ends"], "color": "#22d3ee" if m["reliable"] else "#ffa940",
             "label": f"length {m['length_mm']} mm"},
            {"pts": m["width_ends"], "color": "#ffa940", "label": f"width {m['width_mm']} mm"}]


def measure_figure(img, pitch, offset, box, m, title):
    """Zoom on one box with the measured length (cyan) and width (orange) axes."""
    ox, oy = offset
    x0, y0, x1, y1 = measure_window(img.shape, offset, box)
    sub = img[y0:y1, x0:x1]
    fig, ax = plt.subplots(figsize=(5.5, 5.5 * sub.shape[0] / sub.shape[1] + 0.8))
    draw_map(ax, sub, pitch, (x0 + ox, y0 + oy), "magma", False, title)
    draw_boxes(ax, [box], pitch, numbered=False)
    for ln in measure_lines(m):
        a, b = ln["pts"]
        ax.plot([a[0], b[0]], [a[1], b[1]], color=ln["color"], lw=2.2)
    return png(fig)
