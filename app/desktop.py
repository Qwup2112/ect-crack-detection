"""ECT Crack Chat as a desktop program: one native window, no browser, no address bar.

The window (WebView2 through pywebview) opens at once with a start-up screen; the backend is
started inside this same process on a free 127.0.0.1 port that the user never sees, then the
window switches to the app. Closing the window stops everything.

Launch: "ECT Crack Chat.lnk" or run_app.bat (both run `.venv\\Scripts\\pythonw -m app.desktop`).
The chat agent runs on this PC's Claude Code login - no API key (see app/backend/server.py).
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "app" / "state"
ICON = ROOT / "app" / "assets" / "ect.ico"
STATE.mkdir(parents=True, exist_ok=True)
LOG = STATE / "desktop.log"

# pythonw has no console (sys.stdout is None): everything goes to app/state/desktop.log.
_log = open(LOG, "a", encoding="utf-8", buffering=1)
sys.stdout = sys.stderr = _log

import json          # noqa: E402
import shutil        # noqa: E402
import socket        # noqa: E402
import threading     # noqa: E402
import time          # noqa: E402
import traceback     # noqa: E402
import urllib.request  # noqa: E402

import webview       # noqa: E402

BG = "#0a0d12"
SPLASH = f"""<!doctype html><html lang="vi"><head><meta charset="utf-8"><style>
html,body{{margin:0;height:100%;background:{BG};color:#94a3b8;font:15px "Segoe UI",system-ui,sans-serif;
display:flex;align-items:center;justify-content:center}} .box{{text-align:center}}
svg{{animation:spin 2.4s linear infinite}} @keyframes spin{{to{{transform:rotate(360deg)}}}}
h1{{color:#f1f5f9;font-size:18px;font-weight:600;margin:18px 0 6px}}</style></head><body><div class="box">
<svg width="64" height="64" viewBox="0 0 36 36"><circle cx="18" cy="18" r="14" fill="none" stroke="#c87d55"
stroke-width="2.2" stroke-dasharray="70 18"/><circle cx="18" cy="18" r="9.5" fill="none" stroke="#d7825e"
stroke-width="2.2"/><circle cx="18" cy="18" r="5" fill="none" stroke="#e5a489" stroke-width="2.2"/></svg>
<h1>ECT Crack Chat</h1><div>Đang khởi động bộ xử lý…</div></div></body></html>"""


def error_page(detail):
    esc = (detail or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"""<!doctype html><html lang="vi"><head><meta charset="utf-8"><style>
body{{margin:0;padding:28px;background:{BG};color:#e2e8f0;font:14px "Segoe UI",system-ui,sans-serif}}
h1{{color:#fca5a5;font-size:18px}} pre{{background:#10151d;border:1px solid #242e3f;padding:12px;
white-space:pre-wrap;font-size:12px;color:#cbd5e1;max-height:60vh;overflow:auto}}</style></head><body>
<h1>Không khởi động được bộ xử lý</h1>
<p>Thường do máy thiếu bộ nhớ (hãy đóng bớt ứng dụng hoặc khởi động lại máy), hoặc thiếu thư viện
trong .venv. Chi tiết (cũng có trong app/state/desktop.log):</p><pre>{esc}</pre></body></html>"""


class Api:
    """Exposed to the page as window.pywebview.api: file actions a web page cannot do."""

    def __init__(self):
        self._window = None

    def _image(self, iid):
        from app.backend.agent import images
        p = images.path_of(str(iid))
        if p is None:
            return None, {}
        meta_file = p.with_suffix(".json")
        meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
        return p, meta

    def open_image(self, iid):
        """Open the full-size PNG in the Windows image viewer."""
        p, _ = self._image(iid)
        if p is None:
            return {"error": "không tìm thấy ảnh"}
        os.startfile(p)
        return {"opened": str(p)}

    def save_image(self, iid):
        """Save the PNG where the user chooses (Save As dialog)."""
        p, meta = self._image(iid)
        if p is None:
            return {"error": "không tìm thấy ảnh"}
        title = meta.get("title") or p.stem
        name = "".join(c if c.isalnum() or c in " ._-" else "_" for c in title).strip()[:80] or p.stem
        res = self._window.create_file_dialog(webview.FileDialog.SAVE, directory=str(Path.home() / "Pictures"),
                                              save_filename=name + ".png", file_types=("PNG (*.png)",))
        if not res:
            return {"cancelled": True}
        dest = res if isinstance(res, str) else res[0]
        if not dest.lower().endswith(".png"):
            dest += ".png"
        shutil.copyfile(p, dest)
        return {"saved": dest}


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _healthy(port, timeout):
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.4)
    return False


def main():
    print(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} start")
    # Import the backend (numpy, scipy, matplotlib) on the MAIN thread: on Python 3.14, scipy
    # imported first from a worker thread came up half-initialised in the legacy app.
    try:
        import uvicorn
        from app.backend import server as backend
        boot_error = None
    except BaseException:
        boot_error = traceback.format_exc()
        print(boot_error)

    api = Api()
    window = webview.create_window("ECT Crack Chat", html=SPLASH if not boot_error else error_page(boot_error),
                                   js_api=api, width=1500, height=920, min_size=(1000, 640),
                                   background_color=BG, text_select=True)
    api._window = window
    state = {}

    def boot():
        if boot_error:
            return
        try:
            port = _free_port()
            config = uvicorn.Config(backend.app, host="127.0.0.1", port=port, loop="asyncio",
                                    log_level="warning")
            server = uvicorn.Server(config)
            state["server"] = server
            t = threading.Thread(target=server.run, name="backend", daemon=True)
            t.start()
            state["thread"] = t
            if _healthy(port, 90):
                print(f"backend ready on 127.0.0.1:{port}")
                window.load_url(f"http://127.0.0.1:{port}/")
            else:
                window.load_html(error_page("Bộ xử lý không trả lời sau 90 giây.\n\n" + LOG.read_text(
                    encoding="utf-8", errors="replace")[-3000:]))
        except BaseException:
            detail = traceback.format_exc()
            print(detail)
            window.load_html(error_page(detail))

    webview.settings["ALLOW_DOWNLOADS"] = True
    webview.start(boot, gui="edgechromium", private_mode=False, storage_path=str(STATE / "webview"),
                  icon=str(ICON) if ICON.exists() else None)

    server = state.get("server")                 # window closed: stop the backend (and its agents)
    if server is not None:
        server.should_exit = True
        if state.get("thread"):
            state["thread"].join(timeout=20)
    print(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} closed")


if __name__ == "__main__":
    main()
