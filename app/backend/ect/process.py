"""Signal processing: denoising, background removal, border cropping ("cắt viền") and
contrast enhancement ("bật tông"). Every function takes and returns plain 2D arrays.

Defaults are the pipeline tested on the whole dataset in the legacy app:
    lpf filter (FIR down each column + 3 px median)  -> removes the scan sawtooth
    auto border crop (asymmetric, measured per edge)  -> removes the noisy scan border
    bg detrend (subtract an 8 px Gaussian blur)       -> one blob per crack on the 3x5 coupon
"""
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import firwin, lfilter

FILTERS = ("lpf", "median", "gaussian", "none")
DETRENDS = ("bg", "row", "col", "both", "none")
ENHANCE = ("none", "stretch", "detrend", "local", "gamma", "strong")


# ---------------------------------------------------------------- denoise / detrend

def _lpf_columns(img, cutoff=0.3, order=24):
    """Hamming FIR low-pass down every column (along Y, where the sawtooth lives),
    edges padded, group delay removed - the lab's dataloader.LPF, vectorised."""
    b = firwin(order + 1, cutoff, window="hamming", pass_zero="lowpass")
    x = np.pad(np.asarray(img, dtype=float), ((50, 50), (0, 0)), mode="edge")
    y = lfilter(b, 1.0, x, axis=0)
    return y[50 + order // 2: y.shape[0] - 50 + order // 2]


def denoise(img, method="lpf", kernel=3):
    if method == "none":
        return np.array(img, dtype=float)
    if method == "lpf":
        return median_filter(_lpf_columns(img), size=kernel, mode="nearest")
    if method == "median":
        return median_filter(img, size=kernel, mode="nearest")
    if method == "gaussian":
        return gaussian_filter(img, sigma=kernel / 2.0, mode="nearest")
    raise ValueError(f"filter must be one of {FILTERS}")


def detrend(img, method="bg", sigma=8):
    """Remove the plate background. bg = subtract a heavy Gaussian blur (wider than a crack,
    narrower than the crack spacing); row/col = subtract per-row/column medians."""
    img = np.asarray(img, dtype=float)
    if method == "bg":
        return img - gaussian_filter(img, sigma=sigma)
    if method == "row":
        return img - np.nanmedian(img, axis=1, keepdims=True)
    if method == "col":
        return img - np.nanmedian(img, axis=0, keepdims=True)
    if method == "both":
        return detrend(detrend(img, "row"), "col")
    if method == "none":
        return img.copy()
    raise ValueError(f"detrend must be one of {DETRENDS}")


def robust_z(a):
    """z-score from median/MAD: cracks and edge artifacts are extreme values themselves,
    so mean/std would let them inflate the scale and hide what we look for."""
    med = np.nanmedian(a)
    mad = np.nanmedian(np.abs(a - med)) * 1.4826
    if not np.isfinite(mad) or mad == 0:
        return np.zeros_like(a, dtype=float)
    return (a - med) / mad


# ---------------------------------------------------------------- border crop

def auto_border_margins(img, z_thresh=3.0, max_frac=0.15):
    """Measure the noisy border per edge. A border row/column is abnormal in LEVEL (offset
    median, e.g. the bright band where a line starts) or in SPREAD (probe turn-around
    streaks with a normal median). Walk inward from each edge while either robust z stays
    above z_thresh; stop at the first normal line. Capped at max_frac per side.
    Returns {'top','bottom','left','right'} in pixels - usually asymmetric."""
    H, W = img.shape
    flat = detrend(img, "row")
    row_z = np.maximum(np.abs(robust_z(np.nanmedian(img, axis=1))),
                       np.abs(robust_z(np.nanstd(flat, axis=1))))
    col_z = np.maximum(np.abs(robust_z(np.nanmedian(img, axis=0))),
                       np.abs(robust_z(np.nanstd(flat, axis=0))))

    def run(vals, limit):
        n = 0
        for v in vals[:limit]:
            if not np.isfinite(v) or v <= z_thresh:
                break
            n += 1
        return n

    mr, mc = int(H * max_frac), int(W * max_frac)
    m = {"top": run(row_z, mr), "bottom": run(row_z[::-1], mr),
         "left": run(col_z, mc), "right": run(col_z[::-1], mc)}
    if H - m["top"] - m["bottom"] < 10 or W - m["left"] - m["right"] < 10:
        m = dict.fromkeys(m, 5)                     # nearly all noise: fixed 5 px instead
    return m


def fixed_margins(n):
    return dict.fromkeys(("top", "bottom", "left", "right"), int(n))


def apply_crop(img, m):
    H, W = img.shape
    if m["top"] + m["bottom"] >= H - 2 or m["left"] + m["right"] >= W - 2:
        raise ValueError(f"crop {m} would remove the whole {H}x{W} image")
    return img[m["top"]:H - m["bottom"], m["left"]:W - m["right"]]


def margins_for(img, crop="auto", z_thresh=3.0):
    """crop = 'auto' | 'none' | int (fixed px per side) | dict of four margins."""
    if isinstance(crop, dict):
        return {k: int(crop.get(k, 0)) for k in ("top", "bottom", "left", "right")}
    if crop in (None, "none", 0, "0"):
        return fixed_margins(0)
    if str(crop).lower() == "auto":
        return auto_border_margins(img, z_thresh=z_thresh)
    return fixed_margins(int(crop))


def describe_margins(m):
    if not any(m.values()):
        return "không cắt viền"
    return f"trên {m['top']}, dưới {m['bottom']}, trái {m['left']}, phải {m['right']} px"


# ---------------------------------------------------------------- contrast ("bật tông")

def percentile_limits(img, pct=2.0):
    lo, hi = np.nanpercentile(img, [pct, 100 - pct])
    if not hi > lo:
        lo, hi = np.nanmin(img), np.nanmax(img)
    return float(lo), float(hi)


def stretch(img, pct=2.0):
    lo, hi = percentile_limits(img, pct)
    return np.clip((img - lo) / ((hi - lo) or 1.0), 0, 1)


def local_normalize(img, sigma=8):
    """Local contrast: (x - local mean) / local std, both Gaussian-weighted. Faint cracks in
    a dim region get the same contrast as strong ones in a bright region."""
    mu = gaussian_filter(img, sigma)
    var = gaussian_filter((img - mu) ** 2, sigma)
    return (img - mu) / np.sqrt(var + 1e-12 * (np.nanmax(np.abs(img)) or 1.0) ** 2)


def enhance(img, method="stretch", pct=2.0, gamma=0.7, sigma=8):
    """Returns (enhanced image, short description). Display-oriented: the result is for
    seeing cracks, not for measuring values."""
    if method == "none":
        return img, "không tăng tương phản"
    if method == "stretch":
        return stretch(img, pct), f"kéo giãn tương phản theo bách phân vị {pct:g}–{100 - pct:g}%"
    if method == "detrend":
        return stretch(detrend(img, "bg", sigma), pct), f"trừ nền (Gaussian σ={sigma} px) rồi kéo giãn"
    if method == "local":
        return stretch(local_normalize(img, sigma), pct), f"chuẩn hóa tương phản cục bộ (σ={sigma} px)"
    if method == "gamma":
        return stretch(img, pct) ** gamma, f"kéo giãn rồi gamma {gamma:g}"
    if method == "strong":
        z = local_normalize(detrend(img, "bg", sigma), sigma / 2)
        return stretch(z, 1.0), "mạnh: trừ nền + chuẩn hóa cục bộ + kéo giãn 1–99%"
    raise ValueError(f"enhance must be one of {ENHANCE}")


# ---------------------------------------------------------------- full pipeline

def pipeline(grid, filter="lpf", crop="auto", detrend_method="bg", crop_z=3.0):
    """Every intermediate stage, so any of them can be drawn:
    raw, filtered, cropped, detrended, dx (d/dx of cropped), zmap (robust z of detrended),
    margins (crop in px, relative to the raw grid), offset (left, top) of the cropped stages."""
    filtered = denoise(grid, filter)
    m = margins_for(filtered, crop, crop_z)
    cropped = apply_crop(filtered, m)
    flat = detrend(cropped, detrend_method)
    return {
        "raw": np.asarray(grid, dtype=float), "filtered": filtered, "cropped": cropped,
        "detrended": flat, "dx": np.gradient(cropped, axis=1), "zmap": robust_z(flat),
        "margins": m, "offset": (m["left"], m["top"]),
        "filter": filter, "detrend": detrend_method,
    }
