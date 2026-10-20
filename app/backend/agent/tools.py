"""The agent's tools (an in-process MCP server named "ect").

Each tool computes with app.backend.ect in a worker thread, saves any figure to the image
store, pushes it to the right-hand panel through the session's event queue, and returns a
text summary plus a small preview of the figure so the model can see what it drew.
run_python (python_tool.py) runs the agent's own analysis code in a guarded child process.
"""
import asyncio
import base64
import json
import threading
from functools import lru_cache

import numpy as np
from claude_agent_sdk import create_sdk_mcp_server, tool

from ..ect import detect, process, render, scan as sc
from . import images, memory, python_tool
from .sysmem import free_commit_mb

RENDER_LOCK = threading.Lock()          # pyplot is not thread-safe: one figure at a time
MAX_COMPARE = 8
MAX_PREVIEWS = 3                         # figure previews returned to the model per tool call

LABELS = {                               # shown in the chat while a tool runs
    "list_scans": "Tra danh mục file quét",
    "scan_info": "Đọc thông tin file quét",
    "draw_scan": "Vẽ bản đồ",
    "crop_border": "Cắt viền",
    "enhance_contrast": "Bật tông (tăng tương phản)",
    "detect_cracks": "Nhận diện vết nứt",
    "measure_crack": "Đo vết nứt",
    "compare_scans": "So sánh các file quét",
    "profile_line": "Vẽ lát cắt 1D",
    "remember": "Ghi nhớ",
    "recall": "Tra bộ nhớ",
    "forget": "Xóa khỏi bộ nhớ",
    "run_python": "Phân tích bằng Python",
}

VIEW_KEYS = sorted(render.VIEWS) + ["report", "model_input"]


# ---------------------------------------------------------------- cached data

@lru_cache(maxsize=32)
def _load(sid, mtime):
    return sc.load_scan(sc.scan_path(sid))


def load(sid):
    return _load(sid, sc.scan_path(sid).stat().st_mtime)


@lru_cache(maxsize=32)
def _stages(sid, mtime, crop_key):
    grid, _ = load(sid)
    crop = json.loads(crop_key)
    return process.pipeline(grid, crop=crop)


def stages(sid, crop="auto"):
    return _stages(sid, sc.scan_path(sid).stat().st_mtime, json.dumps(crop, sort_keys=True))


def parse_crop(v):
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return int(v)
    v = str(v).strip().lower()
    return v if v in ("auto", "none") else int(v)


def scan_label(meta):
    parts = [meta.get("sensor", "?")]
    if meta.get("amp"):
        parts.append(f"{meta['amp']} V")
    if meta.get("freq"):
        parts.append(f"{meta['freq']} kHz")
    return f"{meta['session']} · " + " ".join(parts)


# ---------------------------------------------------------------- result helpers

def _text(t):
    return {"type": "text", "text": t}


def _error(msg):
    return {"content": [_text(f"Lỗi: {msg}")], "is_error": True}


class Ctx:
    """What a tool needs from its chat session: an event channel and per-chat state."""

    def __init__(self, session_id, emit, state):
        self.session_id, self.emit, self.state = session_id, emit, state

    def crop_for(self, sid, given):
        c = parse_crop(given)
        return c if c is not None else self.state.setdefault("crop", {}).get(sid, "auto")


async def _run(ctx, name, fn, args):
    """Compute in a thread, publish the figure (if any), build the tool result."""
    try:
        out = await asyncio.to_thread(fn, ctx, args)
    except (ValueError, FileNotFoundError, KeyError) as e:
        return _error(str(e))
    except Exception as e:                      # a bug or a resource problem: report, don't crash the chat
        import traceback
        from ..ect.paths import LOG_DIR, ensure_state_dirs
        ensure_state_dirs()
        with open(LOG_DIR / "tool_errors.log", "a", encoding="utf-8") as f:
            f.write(f"--- {name} {json.dumps(args, ensure_ascii=False)}\n{traceback.format_exc()}\n")
        return _error(f"{type(e).__name__}: {e}")
    figs = out.get("figures") or []
    if out.get("png"):
        figs = [{"png": out["png"], "title": out.get("title", LABELS[name]), "caption": out.get("caption", "")}]
    shown, previews = [], []
    for f in figs:
        meta = images.save(f["png"], f["title"], f.get("caption", ""), scan=out.get("scan"), tool=name,
                           session=ctx.session_id)
        ctx.state["last_image"] = meta["id"]
        await ctx.emit("image", meta)
        shown.append(f"{meta['title']} (id {meta['id']})")
        if len(previews) < MAX_PREVIEWS:
            prev = await asyncio.to_thread(render.preview, f["png"])
            previews.append({"type": "image", "data": base64.b64encode(prev).decode(), "mimeType": "image/png"})
    content = [_text(out["text"] + (f"\n[Hình đã hiển thị ở khung bên phải: {'; '.join(shown)}]" if shown else ""))]
    content += previews
    if out.get("scan"):
        memory.touch_scan(out["scan"])
        ctx.state["last_scan"] = out["scan"]
    return {"content": content}


# ---------------------------------------------------------------- implementations

def _list_scans(ctx, a):
    q = (a.get("query") or "").strip()
    items = sc.find_scans(q, readable_only=False) if q else sc.catalog()
    rows = []
    for e in items[:80]:
        n_lab = len(detect.load_labels(e["id"])[0]) if not e["status"].startswith(("empty", "unreadable")) else 0
        size = f"{e['size_x']}x{e['size_y']}" if "size_x" in e else "-"
        rows.append(f"{e['id']} | {e.get('sensor', '-')} | {e.get('amp', '-')} V | {e.get('freq', '-')} kHz | "
                    f"{size} | {e['status']} | {n_lab} nhãn")
    by_session = {}
    for e in sc.catalog():
        s = by_session.setdefault(e["session"], [0, 0])
        s[0] += 1
        s[1] += e["status"] == "ok"
    head = "Phiên đo: " + "; ".join(f"{k}: {v[1]}/{v[0]} file đọc đủ" for k, v in by_session.items())
    more = f"\n... và {len(items) - 80} file nữa, hãy lọc hẹp hơn." if len(items) > 80 else ""
    return {"text": f"{head}\n{len(items)} file khớp '{q or '*'}':\nid | cảm biến | biên độ kích thích | tần số | lưới | trạng thái | nhãn\n"
                    + "\n".join(rows) + more}


def _scan_info(ctx, a):
    sid = sc.resolve(a["scan"])
    grid, meta = load(sid)
    st = stages(sid, ctx.crop_for(sid, None))
    labels, note = detect.load_labels(sid)
    p = meta["pitch_mm"]
    info = {
        "id": sid, "mô tả": scan_label(meta), "lưới (điểm)": f"{meta['size_x']} x {meta['size_y']}",
        "kích thước (mm)": f"{meta['size_x'] * p:.1f} x {meta['size_y'] * p:.1f}",
        "bước lưới (mm/điểm)": round(p, 3), "tần số trong file (Hz)": meta["freq_hz"],
        "lệch chiều quét đã sửa (px)": meta["row_shift_px"],
        "giá trị thô min/median/max": [round(float(x), 4) for x in np.nanpercentile(grid, [0, 50, 100])],
        "cắt viền đang dùng": process.describe_margins(st["margins"]),
        "nhãn": f"{len(labels)} hộp" + (f" ({note})" if note else "") if labels else "chưa có nhãn",
        "trong tập huấn luyện YOLO": detect.in_training_set(sid),
        "cảnh báo": meta["warnings"] or "không",
    }
    return {"text": json.dumps(info, ensure_ascii=False, indent=1), "scan": sid}


def _boxes_for(ctx, sid, grid, st, method, conf=None, z=None):
    """Detection boxes (original grid px) + a note about what was used."""
    note = ""
    if method in ("yolo", "both"):
        try:
            boxes = detect.yolo_detect(grid, conf or detect.YOLO_CONF)
            note = f"YOLO (ngưỡng tin cậy {conf or detect.YOLO_CONF})"
        except Exception as e:
            boxes, _ = detect.threshold_detect(st["cropped"], st["offset"], z or 2.5)
            note = f"YOLO không chạy được ({type(e).__name__}: {str(e)[:120]}) - dùng bộ dò ngưỡng |z| > {z or 2.5}"
            method = "threshold"
    else:
        boxes, _ = detect.threshold_detect(st["cropped"], st["offset"], z or 2.5)
        note = f"bộ dò ngưỡng |z| > {z or 2.5} trên bản đồ đã trừ nền"
    return boxes, note, method


def _draw_scan(ctx, a):
    sid = sc.resolve(a["scan"])
    grid, meta = load(sid)
    crop = ctx.crop_for(sid, a.get("crop"))
    st = stages(sid, crop)
    view = a.get("view") or "amplitude"
    pitch, sensor = meta["pitch_mm"], meta.get("sensor")
    enh = a.get("enhance") or ctx.state.setdefault("enhance", {}).get(sid, "none")
    head = f"{scan_label(meta)} — "
    with RENDER_LOCK:
        if view == "report":
            items = []
            for k in ("raw", "filtered", "cropped", "amplitude", "dx", "dx_inv"):
                img, off, t, cm, div = render.view_spec(st, k, sensor)
                items.append({"img": img, "pitch": pitch, "offset": off, "title": t, "cmap": cm, "diverging": div})
            data = render.panels(items, ncols=3, suptitle=head + "6 bản đồ chuẩn", width=5.0)
            title = head + "báo cáo 6 hình"
        elif view == "model_input":
            data = render.rgb(detect.model_image(grid), pitch, "Ảnh đầu vào YOLO: R = tín hiệu đã trừ nền, G = d/dx, B = d/dy")
            title = head + "ảnh đầu vào mô hình"
        else:
            img, off, t, cm, div = render.view_spec(st, view, sensor)
            limits, cbar = None, ("pha (a.u.)" if sensor == "P" else "biên độ (a.u.)")
            if enh != "none" and not div:
                img, edesc = process.enhance(img, enh)
                limits, cbar, t = (0.0, 1.0), "tương phản (0–1)", f"{t} · {edesc}"
            if a.get("cmap"):
                cm = a["cmap"]
            boxes, missed, extra = None, (), ""
            overlay = a.get("overlay") or "none"
            if overlay == "labels":
                labels, note = detect.load_labels(sid)
                boxes = [dict(b) for b in labels]
                extra = f" · {len(labels)} nhãn" + (" (NHÁP)" if "DRAFT" in note.upper() else "")
            elif overlay in ("yolo", "threshold"):
                boxes, bnote, _ = _boxes_for(ctx, sid, grid, st, overlay)
                extra = f" · {len(boxes)} hộp ({bnote})"
            data = render.single(img, pitch, off, a.get("title") or t, cm, div, cbar, boxes=boxes,
                                 suptitle=head + f"cắt viền: {process.describe_margins(st['margins'])}{extra}",
                                 limits=limits)
            title = head + (a.get("title") or t)
    return {"text": f"Đã vẽ '{view}' của {sid}. Cắt viền: {process.describe_margins(st['margins'])}. "
                    f"Tăng tương phản: {enh}.",
            "png": data, "title": title, "scan": sid,
            "caption": f"{sid} · view {view} · crop {crop} · enhance {enh}"}


def _crop_border(ctx, a):
    sid = sc.resolve(a["scan"])
    grid, meta = load(sid)
    pitch = meta["pitch_mm"]
    mode = a.get("mode") or "auto"
    z = float(a.get("z_thresh") or 3.0)
    filtered = stages(sid, "none")["filtered"]
    if mode == "fixed":
        n = int(a.get("margin", 5))
        crop = n
        m = process.fixed_margins(n)
    else:
        m = process.auto_border_margins(filtered, z_thresh=z)
        crop = "auto" if z == 3.0 else dict(m)
    cropped = process.apply_crop(filtered, m)
    ctx.state.setdefault("crop", {})[sid] = crop
    H, W = filtered.shape
    keep = (m["left"] * pitch, m["top"] * pitch, (W - m["right"]) * pitch, (H - m["bottom"]) * pitch)
    with RENDER_LOCK:
        data = render.panels([
            {"img": filtered, "pitch": pitch, "title": "Trước: vùng đỏ là viền bị cắt", "cmap": "viridis",
             "shade": render.crop_bands(filtered.shape, m, pitch), "rect": keep},
            {"img": cropped, "pitch": pitch, "offset": (m["left"], m["top"]), "title": "Sau khi cắt viền",
             "cmap": "magma"},
        ], ncols=2, suptitle=f"{scan_label(meta)} — cắt viền {'tự động' if mode != 'fixed' else 'cố định'}: "
                             f"{process.describe_margins(m)}")
    mm = {k: round(v * pitch, 1) for k, v in m.items()}
    return {"text": f"Cắt viền {sid}: {process.describe_margins(m)} (mm: {mm}). Lưới còn "
                    f"{cropped.shape[1]} x {cropped.shape[0]} điểm. Từ giờ trong cuộc chat này, các hình của "
                    f"file này dùng cách cắt đó.",
            "png": data, "title": f"{scan_label(meta)} — cắt viền", "scan": sid,
            "caption": f"margins px {m}"}


def _enhance_contrast(ctx, a):
    sid = sc.resolve(a["scan"])
    grid, meta = load(sid)
    method = a.get("method") or "strong"
    st = stages(sid, ctx.crop_for(sid, a.get("crop")))
    img = st["cropped"]
    enh, desc = process.enhance(img, method)
    ctx.state.setdefault("enhance", {})[sid] = method
    p, off = meta["pitch_mm"], st["offset"]
    with RENDER_LOCK:
        data = render.panels([
            {"img": img, "pitch": p, "offset": off, "title": "Trước (đã lọc + cắt viền)", "cmap": "magma"},
            {"img": enh, "pitch": p, "offset": off, "title": f"Sau: {desc}", "cmap": a.get("cmap") or "magma",
             "limits": (0.0, 1.0)},
        ], ncols=2, suptitle=f"{scan_label(meta)} — bật tông")
    return {"text": f"Bật tông {sid} bằng '{method}': {desc}. Ghi nhớ cho file này trong cuộc chat: các "
                    f"bản đồ biên độ sau sẽ dùng cùng cách tăng tương phản (trừ khi bạn yêu cầu khác).",
            "png": data, "title": f"{scan_label(meta)} — bật tông ({method})", "scan": sid,
            "caption": desc}


def _detect_cracks(ctx, a):
    sid = sc.resolve(a["scan"])
    grid, meta = load(sid)
    st = stages(sid, ctx.crop_for(sid, a.get("crop")))
    pitch = meta["pitch_mm"]
    boxes, note, used = _boxes_for(ctx, sid, grid, st, a.get("method") or "yolo", a.get("conf"), a.get("z_thresh"))
    labels, lnote = detect.load_labels(sid)
    missed, score = [], ""
    if labels:
        found, false, miss = detect.match_labels(boxes, labels)
        missed = [labels[i] for i in miss]
        score = (f"So với {len(labels)} nhãn: tìm đúng {found}, báo sai {false}, bỏ sót {len(miss)} (IoU ≥ 0,3)."
                 + (" Nhãn là bản NHÁP chưa duyệt." if "DRAFT" in lnote.upper() else "")
                 + (" File này nằm trong tập huấn luyện YOLO nên điểm lạc quan." if used == "yolo" and detect.in_training_set(sid) else ""))
    ctx.state.setdefault("detections", {})[sid] = boxes
    view = a.get("view") or "amplitude"
    img, off, t, cm, div = render.view_spec(st, view if view in render.VIEWS else "amplitude", meta.get("sensor"))
    limits = None
    enh = ctx.state.setdefault("enhance", {}).get(sid, "none")
    if enh != "none" and not div:
        img, _ = process.enhance(img, enh)
        limits = (0.0, 1.0)
    with RENDER_LOCK:
        data = render.single(img, pitch, off, f"{t} · {len(boxes)} vết nứt", cm, div, boxes=boxes, missed=missed,
                             suptitle=f"{scan_label(meta)} — {note}", limits=limits)
    rows = []
    for i, b in enumerate(boxes, 1):
        d = detect.describe_box(b, pitch)
        s = f"conf {b['conf']:.2f}" if "conf" in b else f"|z| {b['score']}"
        m = "" if "match" not in b else (" ✓ khớp nhãn" if b["match"] else " ✗ không khớp nhãn")
        rows.append(f"#{i}: X {d['x_mm']} mm, Y {d['y_mm']} mm, khung {d['w_mm']} x {d['h_mm']} mm, {s}{m}")
    return {"text": f"Nhận diện {sid} bằng {note}: {len(boxes)} vết nứt.\n" + "\n".join(rows)
                    + (f"\n{score}" if score else "") + "\nTọa độ theo mm của lưới gốc (trước khi cắt viền).",
            "png": data, "title": f"{scan_label(meta)} — {len(boxes)} vết nứt", "scan": sid, "caption": note}


def _measure_crack(ctx, a):
    sid = sc.resolve(a.get("scan") or ctx.state.get("last_scan") or "")
    grid, meta = load(sid)
    pitch = meta["pitch_mm"]
    st = stages(sid, ctx.crop_for(sid, None))
    boxes = ctx.state.setdefault("detections", {}).get(sid)
    if not boxes:
        boxes, _, _ = _boxes_for(ctx, sid, grid, st, "yolo")
        ctx.state["detections"][sid] = boxes
    if not boxes:
        raise ValueError("không tìm thấy vết nứt nào để đo")
    if a.get("crack"):
        i = int(a["crack"])
        if not 1 <= i <= len(boxes):
            raise ValueError(f"chỉ có vết nứt #1..#{len(boxes)}")
        box = boxes[i - 1]
    elif a.get("x_mm") is not None and a.get("y_mm") is not None:
        x, y = float(a["x_mm"]) / pitch, float(a["y_mm"]) / pitch
        i, box = min(enumerate(boxes, 1), key=lambda t: ((t[1]["x0"] + t[1]["x1"]) / 2 - x) ** 2
                     + ((t[1]["y0"] + t[1]["y1"]) / 2 - y) ** 2)
    else:
        raise ValueError("cần số thứ tự vết nứt (crack) hoặc tọa độ x_mm, y_mm")
    img = st["filtered"]
    m = detect.measure_crack(img, box, pitch)
    if not m:
        raise ValueError("vùng này quá nhỏ hoặc không có tín hiệu để đo")
    with RENDER_LOCK:
        data = render.measure_figure(img, pitch, (0, 0), box, m, f"Vết nứt #{i}: dài {m['length_mm']} mm, rộng {m['width_mm']} mm")
    flags = []
    if not m["reliable"]:
        flags.append("tín hiệu yếu hoặc chiếm gần hết khung: số đo không chắc chắn")
    if m["spills_out_of_box"]:
        flags.append("tín hiệu tràn ra ngoài khung: chiều dài là giá trị tối thiểu")
    return {"text": f"Vết nứt #{i} của {sid}: dài {m['length_mm']} mm, rộng {m['width_mm']} mm "
                    f"(độ rộng tín hiệu, gồm cả vết đầu dò ~6 mm, vết nứt thật hẹp hơn nhiều), góc {m['angle_deg']}° "
                    f"(chạy dọc {m['along']}), tâm ({m['centre_mm'][0]}, {m['centre_mm'][1]}) mm, SNR {m['snr']}. "
                    "Độ sâu không ước lượng được nếu chưa hiệu chuẩn." + ("\nLưu ý: " + "; ".join(flags) if flags else ""),
            "png": data, "title": f"{scan_label(meta)} — đo vết nứt #{i}", "scan": sid,
            "caption": "trục cyan = chiều dài, cam = chiều rộng"}


def _compare_scans(ctx, a):
    ids = [sc.resolve(s) for s in a["scans"]]
    if not 2 <= len(ids) <= MAX_COMPARE:
        raise ValueError(f"so sánh từ 2 đến {MAX_COMPARE} file")
    view = a.get("view") or "amplitude"
    enh = a.get("enhance") or "none"
    items, rows = [], []
    for sid in ids:
        grid, meta = load(sid)
        st = stages(sid, ctx.crop_for(sid, a.get("crop")))
        img, off, t, cm, div = render.view_spec(st, view, meta.get("sensor"))
        lim = None
        if enh != "none" and not div:
            img, _ = process.enhance(img, enh)
            lim = (0.0, 1.0)
        boxes, _ = detect.threshold_detect(st["cropped"], st["offset"])
        items.append({"img": img, "pitch": meta["pitch_mm"], "offset": off, "cmap": cm, "diverging": div,
                      "title": scan_label(meta), "limits": lim})
        rows.append((detect.z_top5(boxes), len(boxes), sid, scan_label(meta)))
    with RENDER_LOCK:
        data = render.panels(items, ncols=2 if len(items) <= 4 else 3, width=5.0,
                             suptitle=f"So sánh {len(items)} file — {view}")
    rows.sort(reverse=True)
    table = "\n".join(f"{k}. {r[3]} ({r[2]}): z_top5 {r[0]}, {r[1]} ứng viên" for k, r in enumerate(rows, 1))
    return {"text": "Xếp hạng theo z_top5 (độ nổi bật của 5 ứng viên mạnh nhất so với nhiễu nền của chính ảnh "
                    "đó, so sánh công bằng giữa R và P):\n" + table,
            "png": data, "title": f"So sánh {len(items)} file quét", "scan": ids[0],
            "caption": ", ".join(ids)}


def _profile_line(ctx, a):
    sid = sc.resolve(a["scan"])
    grid, meta = load(sid)
    st = stages(sid, ctx.crop_for(sid, a.get("crop")))
    view = a.get("view") or "amplitude"
    img, off, t, cm, div = render.view_spec(st, view, meta.get("sensor"))
    along = (a.get("along") or "X").upper()
    with RENDER_LOCK:
        data = render.profile_figure(img, meta["pitch_mm"], off, along, float(a["position_mm"]), t, cm, div)
    return {"text": f"Lát cắt {'ngang (theo X)' if along == 'X' else 'dọc (theo Y)'} tại "
                    f"{'Y' if along == 'X' else 'X'} = {a['position_mm']} mm trên '{view}' của {sid}.",
            "png": data, "title": f"{scan_label(meta)} — lát cắt", "scan": sid, "caption": view}


def _run_python(ctx, a):
    free = free_commit_mb()
    if free is not None and free < python_tool.MIN_FREE_MB:
        raise ValueError(f"máy chỉ còn {free} MB bộ nhớ trống, chưa đủ để chạy Python (cần ~{python_tool.MIN_FREE_MB} MB). "
                         "Hãy dùng các công cụ có sẵn, hoặc nhờ người dùng đóng bớt chương trình.")
    chat = {k: ctx.state.get(k) for k in ("crop", "enhance", "last_scan", "detections")}
    out, figs, secs, timed_out = python_tool.run(a["code"], ctx.session_id, chat)
    work = python_tool.work_dir(ctx.session_id)
    figures = [{"png": (work / f["file"]).read_bytes(), "title": f["title"], "caption": f.get("caption", "")}
               for f in figs if (work / f["file"]).exists()]
    head = f"run_python ({secs:.1f} s{', HẾT GIỜ' if timed_out else ''}). Thư mục làm việc: {work}\n"
    return {"text": head + out, "figures": figures, "scan": ctx.state.get("last_scan")}


def _remember(ctx, a):
    it = memory.add(a["text"], a.get("kind") or "note")
    return {"text": f"Đã ghi nhớ [{it['id']}, {it['kind']}]: {it['text']}"}


def _recall(ctx, a):
    items = memory.search(a.get("query") or "")
    if not items:
        return {"text": "Bộ nhớ chưa có gì khớp."}
    return {"text": "\n".join(f"[{i['id']}, {i['kind']}, {i['created']}] {i['text']}" for i in items)}


def _forget(ctx, a):
    return {"text": "Đã xóa." if memory.forget(a["id"]) else f"Không có mục {a['id']}."}


# ---------------------------------------------------------------- tool definitions

S = {"type": "string"}
SCAN = {"type": "string", "description": "Scan id from list_scans, or words that pick exactly one scan, "
                                         "e.g. 'cambien_4lop_2 R 1.7V 300k'."}
CROP = {"type": "string", "description": "'auto' (measured asymmetric border, default), 'none', or a "
                                         "number of pixels per side. Omit to keep this chat's choice for the scan."}
ENH = {"type": "string", "enum": list(process.ENHANCE)}

SPECS = [
    ("list_scans", "List the dataset's TDMS scans (id, sensor R=amplitude/P=phase, excitation amplitude, "
     "frequency, grid size, status, number of labels). Optional query filters by words.",
     {"type": "object", "properties": {"query": S}}, _list_scans),
    ("scan_info", "Metadata and quick facts for one scan: size in mm, pitch, warnings (truncated file, "
     "assumed pitch), current border crop, labels, whether YOLO trained on it.",
     {"type": "object", "properties": {"scan": SCAN}, "required": ["scan"]}, _scan_info),
    ("draw_scan", "Draw one view of a scan on the right-hand panel ('vẽ'). Views: raw, filtered, cropped, "
     "amplitude (heatmap of the filtered+cropped map), detrended (background removed), dx / dx_inv (X "
     "derivative, shows the crack dipole), zmap, report (the 6 standard views), model_input (what YOLO sees). "
     "overlay draws boxes: yolo, threshold or labels.",
     {"type": "object", "properties": {
         "scan": SCAN, "view": {"type": "string", "enum": VIEW_KEYS}, "crop": CROP, "enhance": ENH,
         "overlay": {"type": "string", "enum": ["none", "yolo", "threshold", "labels"]},
         "cmap": {"type": "string", "description": "matplotlib colormap name, e.g. magma, viridis, gray, RdBu_r"},
         "title": S}, "required": ["scan"]}, _draw_scan),
    ("crop_border", "Crop the noisy scan border ('cắt viền') and show before/after with the removed bands "
     "shaded. mode auto measures each edge separately (usually asymmetric); fixed crops `margin` px per side. "
     "The choice is kept for this scan for the rest of the chat.",
     {"type": "object", "properties": {
         "scan": SCAN, "mode": {"type": "string", "enum": ["auto", "fixed"]},
         "margin": {"type": "integer", "minimum": 0, "maximum": 40},
         "z_thresh": {"type": "number", "description": "auto mode sensitivity, default 3 (lower = crop more)"}},
      "required": ["scan"]}, _crop_border),
    ("enhance_contrast", "Boost contrast so cracks stand out ('bật tông') and show before/after. Methods: "
     "stretch (percentile 2-98%), detrend (remove plate background), local (local contrast normalisation, "
     "for faint or uneven scans), gamma, strong (background removal + local normalisation, for weak scans). "
     "Kept for this scan for the rest of the chat.",
     {"type": "object", "properties": {"scan": SCAN, "method": ENH, "crop": CROP, "cmap": S},
      "required": ["scan"]}, _enhance_contrast),
    ("detect_cracks", "Detect cracks ('nhận diện vết nứt') and draw them. method yolo (trained model, default; "
     "falls back to threshold if it cannot run), threshold (robust z-score, model-free), or both. Lists each "
     "crack with centre and size in mm and, when the scan has labels, found/false/missed against them.",
     {"type": "object", "properties": {
         "scan": SCAN, "method": {"type": "string", "enum": ["yolo", "threshold", "both"]},
         "conf": {"type": "number", "minimum": 0.05, "maximum": 0.95},
         "z_thresh": {"type": "number", "minimum": 1.5, "maximum": 8},
         "view": {"type": "string", "enum": sorted(render.VIEWS)}, "crop": CROP},
      "required": ["scan"]}, _detect_cracks),
    ("measure_crack", "Measure one detected crack's signature: length, width, angle, centre, SNR, with a "
     "zoomed figure. Pick it by number from the last detect_cracks list, or by a position in mm.",
     {"type": "object", "properties": {"scan": SCAN, "crack": {"type": "integer", "minimum": 1},
                                       "x_mm": {"type": "number"}, "y_mm": {"type": "number"}}}, _measure_crack),
    ("compare_scans", f"Draw 2-{MAX_COMPARE} scans side by side and rank them by z_top5 (crack contrast "
     "against each scan's own noise - fair across R and P sensors).",
     {"type": "object", "properties": {"scans": {"type": "array", "items": S, "minItems": 2, "maxItems": MAX_COMPARE},
                                       "view": {"type": "string", "enum": sorted(render.VIEWS)},
                                       "crop": CROP, "enhance": ENH}, "required": ["scans"]}, _compare_scans),
    ("profile_line", "1D cut through a map: values along X at a given Y (along='X') or along Y at a given X.",
     {"type": "object", "properties": {"scan": SCAN, "along": {"type": "string", "enum": ["X", "Y"]},
                                       "position_mm": {"type": "number"},
                                       "view": {"type": "string", "enum": sorted(render.VIEWS)}, "crop": CROP},
      "required": ["scan", "along", "position_mm"]}, _profile_line),
    ("remember", "Save something to long-term memory (kept across chats): a user preference, a fact learned "
     "about the data, or a note.",
     {"type": "object", "properties": {"text": S, "kind": {"type": "string", "enum": list(memory.KINDS)}},
      "required": ["text"]}, _remember),
    ("recall", "Search long-term memory. Empty query lists everything.",
     {"type": "object", "properties": {"query": S}}, _recall),
    ("forget", "Delete one memory item by id (e.g. m3).",
     {"type": "object", "properties": {"id": S}, "required": ["id"]}, _forget),
]

if python_tool.ENABLED:
    SPECS.append(
        ("run_python", "Run your own Python for analysis the other tools cannot do: statistics over many scans, "
         "custom filters, FFT, histograms, correlations, any custom plot. Runs in a separate process (timeout "
         f"{python_tool.TIMEOUT_S} s, ~5 s start-up). Preloaded: np, scipy, ndimage, signal, stats, plt, the ect "
         "modules sc / process / detect / render, and helpers: load(scan) -> (grid, meta); stages(scan, crop=None) "
         "-> dict raw/filtered/cropped/detrended/dx/zmap/margins/offset (crop None = this chat's choice); "
         "yolo(scan, conf=0.5) and threshold(scan, z=2.5) -> boxes {x0,y0,x1,y1,...} in original grid px; "
         "labels(scan); catalog(); show(fig, title, caption) puts a matplotlib figure on the right panel "
         "(Vietnamese title). print() what you need to read back. The working directory is this chat's work "
         "folder: files can be written only there (kept for the chat); data/ and labels are read-only; no "
         "network, no subprocesses. Load skill phan-tich-python before the first use.",
         {"type": "object", "properties": {
             "code": {"type": "string", "description": "Python source to run"},
             "purpose": {"type": "string", "description": "one short Vietnamese line shown to the user, e.g. "
                                                          "'so sánh độ nổi bật vết nứt theo tần số'"}},
          "required": ["code", "purpose"]}, _run_python))

TOOL_NAMES = [s[0] for s in SPECS]
SERVER_NAME = "ect"


def build_server(ctx):
    """The MCP server for one chat session (tools close over that session's context)."""
    def make(name, desc, schema, fn):
        @tool(name, desc, schema)
        async def handler(args):
            return await _run(ctx, name, fn, args)
        return handler
    return create_sdk_mcp_server(SERVER_NAME, tools=[make(*s) for s in SPECS])
