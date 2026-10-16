"""Reading TDMS raster scans and finding them by name.

TDMS layout of this dataset (verified on all 101 readable files):
    group "Freq_Sampling_SizeX_SizeY" -> 1 channel, [freq_Hz, sampling, sizeX, sizeY]
    group "Waveform"                  -> 1 channel, sizeX*sizeY values, serpentine raster
                                         (X fast, Y slow, odd lines recorded right-to-left)
Filenames carry the measurement parameters in five naming variants (see NAME_RE).
"""
import re
import threading
from pathlib import Path

import numpy as np
from nptdms import TdmsFile

from .paths import DATA_DIR

META_GROUP, WAVE_GROUP = "Freq_Sampling_SizeX_SizeY", "Waveform"

# 4lop_R_amp_1.3V_fre_200k_lf_1mm_145x95mm | 4lop_R_amp_4V_fre_10k_lf_1mm |
# R_amp_3V_freq_20K_liffoff_1mm | R_amp_4V_fre_100k_lf_1mm_tron | R_default (templates)
NAME_RE = re.compile(
    r"(?P<sensor>[A-Za-z]+)_amp_(?P<amp>[\d.]+)V"
    r"_(?:fre|freq)_(?P<freq>[\d.]+)[kK]"
    r"_(?:lf|lif+off)_(?P<lf>[\d.]+)mm"
    r"(?:_(?P<w>\d+)x(?P<h>\d+)mm)?"
    r"(?P<variant>_tron)?"
)


def parse_name(path):
    m = NAME_RE.search(Path(path).stem)
    return {k: v for k, v in m.groupdict().items() if v} if m else {}


def scan_id(path):
    """Stable short id: path under data/ without the extension, forward slashes."""
    p = Path(path).resolve()
    try:
        return p.relative_to(DATA_DIR.resolve()).with_suffix("").as_posix()
    except ValueError:
        return p.with_suffix("").name


def scan_path(sid):
    p = (DATA_DIR / f"{sid}.tdms").resolve()
    if not p.is_relative_to(DATA_DIR.resolve()):
        raise ValueError(f"'{sid}' is outside the data folder")
    return p


# ---------------------------------------------------------------- reading

def _sawtooth(img, trim=12):
    f = img[:, trim:-trim] if img.shape[1] > 3 * trim else img
    return float(np.mean(np.abs(f[2:] - 2 * f[1:-1] + f[:-2])))


def fix_scan_direction(grid, max_shift=5):
    """Align the right-to-left lines with the left-to-right ones. The probe lags when it
    reverses (cam_4lop_1: 2 px on 26/28 files) and return lines carry a level offset; the
    shift minimising the row-to-row zig-zag is kept and levels are matched by medians."""
    best = None
    for k in range(-max_shift, max_shift + 1):
        out = grid.copy()
        odd = np.roll(grid[1::2], k, axis=1)
        if k > 0:
            odd[:, :k] = odd[:, [k]]
        elif k < 0:
            odd[:, k:] = odd[:, [k - 1]]
        offset = float(np.median(odd) - np.median(grid[0::2]))
        out[1::2] = odd - offset
        s = _sawtooth(out)
        if best is None or s < best[0]:
            best = (s, out, k, offset)
    return best[1], best[2], best[3]


def load_scan(path):
    """TDMS file -> (grid[Y, X] float, meta). Truncated recordings keep their complete rows
    only (never padded); meta['warnings'] says what was dropped."""
    path = Path(path)
    tdms = TdmsFile.read(path)
    groups = {g.name for g in tdms.groups()}
    if META_GROUP not in groups or WAVE_GROUP not in groups:
        raise ValueError(f"{path.name}: no '{WAVE_GROUP}' data - an empty or template file")
    meta = np.asarray(tdms[META_GROUP].channels()[0][:], dtype=float)
    if meta.size < 4:
        raise ValueError(f"{path.name}: metadata channel has {meta.size} values, need 4")
    freq_hz, sampling, size_x, size_y = meta[:4]
    size_x, size_y = int(round(size_x)), int(round(size_y))
    raw = np.asarray(tdms[WAVE_GROUP].channels()[0][:], dtype=float)
    warnings, truncated = [], 0
    if raw.size != size_x * size_y:
        full = int(raw.size // size_x)
        truncated = size_y - full
        if full < 3:
            raise ValueError(f"{path.name}: only {full} complete scan rows")
        warnings.append(f"file truncated: {truncated} scan row(s) missing, using {full}/{size_y}")
        raw, size_y = raw[:full * size_x], full
    grid = raw.reshape(size_y, size_x)
    grid[1::2] = grid[1::2, ::-1]                      # serpentine: un-mirror odd lines
    grid, shift, offset = fix_scan_direction(grid)

    info = parse_name(path)
    pitch = 1.0
    if info.get("w") and info.get("h"):
        pitch = (float(info["w"]) / size_x + float(info["h"]) / size_y) / 2
    else:
        warnings.append("plate size not in the filename: assuming 1 mm per scan point")
    return grid, {
        "id": scan_id(path), "path": str(path), "session": path.parent.name,
        "freq_hz": float(freq_hz), "size_x": size_x, "size_y": size_y,
        "pitch_mm": float(pitch), "truncated_rows": truncated,
        "row_shift_px": int(shift), "warnings": warnings, **info,
    }


# ---------------------------------------------------------------- catalogue

_catalog_lock = threading.Lock()
_catalog = {"key": None, "items": []}


def _quick_meta(path):
    """Grid size and status without reading the waveform values."""
    entry = {"id": scan_id(path), "session": path.parent.name, **parse_name(path)}
    try:
        with TdmsFile.open(path) as f:
            names = {g.name for g in f.groups()}
            if WAVE_GROUP not in names or META_GROUP not in names:
                entry["status"] = "empty"
                return entry
            m = f[META_GROUP].channels()[0][:]
            sx, sy = int(round(m[2])), int(round(m[3]))
            n = len(f[WAVE_GROUP].channels()[0])
        entry.update(size_x=sx, size_y=sy, freq_hz=float(m[0]))
        entry["status"] = "ok" if n == sx * sy else f"truncated ({sy - n // sx} rows missing)"
    except Exception as e:                                 # unreadable file: say so, keep going
        entry["status"] = f"unreadable: {e}"
    return entry


def catalog():
    """Every .tdms under data/, with parameters and status. Cached until a file changes."""
    files = sorted(DATA_DIR.rglob("*.tdms"))
    key = tuple((str(p), p.stat().st_mtime) for p in files)
    with _catalog_lock:
        if _catalog["key"] != key:
            _catalog.update(key=key, items=[_quick_meta(p) for p in files])
        return list(_catalog["items"])


def _tokens(entry):
    t = {entry["id"].lower(), entry["session"].lower(), Path(entry["id"]).name.lower()}
    if entry.get("sensor"):
        t.add(entry["sensor"].lower())
    if entry.get("amp"):
        t |= {entry["amp"], f"{entry['amp']}v"}
    if entry.get("freq"):
        t |= {entry["freq"], f"{entry['freq']}k", f"{entry['freq']}khz"}
    if entry.get("w"):
        t.add(f"{entry['w']}x{entry['h']}")
    if entry.get("variant"):
        t.add("tron")
    return t


def _norm_query(q):
    q = q.lower().replace(".tdms", "").replace("\\", "/")
    q = re.sub(r"(\d)\s*k\s*hz\b", r"\1k", q)
    q = re.sub(r"(\d)\s*v\b", r"\1v", q)
    return [t for t in re.split(r"[\s,;]+", q) if t]


def find_scans(query, readable_only=True):
    """All catalogue entries matching every word of `query` (an id, a filename, or words
    like 'R 1.7V 300k cambien'). Words of 3+ letters also match inside the filename."""
    items = [e for e in catalog() if not readable_only or not e["status"].startswith(("empty", "unreadable"))]
    words = _norm_query(query)
    exact = [e for e in items if e["id"].lower() in (query.lower().strip(), "/".join(words))]
    if exact:
        return exact
    # A number with or without a unit ("40k", "4v", "1.7", "155x100") must equal a parameter:
    # as a substring "40k" would also pick 140k and "4v" would pick 1.4v.
    numeric = re.compile(r"[\d.]+(k|khz|v|mm)?|\d+x\d+")
    out = []
    for e in items:
        toks, stem = _tokens(e), e["id"].lower()
        if all(w in toks or (not numeric.fullmatch(w) and len(w) >= 3 and w in stem) for w in words):
            out.append(e)
    return out


def resolve(query):
    """One scan id for `query`, or ValueError naming the candidates."""
    hits = find_scans(query)
    if len(hits) > 1 and "tron" not in query.lower():
        plain = [h for h in hits if not h.get("variant")]   # '_tron' only when asked for
        hits = plain if len(plain) == 1 else hits
    if len(hits) == 1:
        return hits[0]["id"]
    if not hits:
        raise ValueError(f"no readable scan matches '{query}'. Use list_scans to see the names.")
    names = ", ".join(h["id"] for h in hits[:12])
    raise ValueError(f"'{query}' matches {len(hits)} scans - be more specific: {names}"
                     + (" ..." if len(hits) > 12 else ""))
