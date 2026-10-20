"""run_python: lets the agent write and run its own analysis code ("phân tích sâu").

The fixed tools cover the usual requests; this one is for everything else (statistics over
many scans, custom filters, FFT, histograms, any plot). The code runs in a CHILD process, so a
bug, a memory blow-up or an endless loop cannot take the app down, and it is killed after a
timeout. The child preloads numpy / scipy / matplotlib and app.backend.ect, then installs an
audit hook that refuses:
  - writing, deleting or renaming files outside the chat's work folder (app/state/work/<chat>),
    so data/ and results/labels/ stay read-only;
  - starting programs, network access and registry writes.
The guard stops mistakes, not a determined attacker (the code comes from the agent, which
works for the user). Every run is logged with its code in app/state/logs/tools.jsonl.

Disable the tool with ECT_PYTHON_TOOL=0.

Parent side: run(). Child side: this file executed as a script (main()).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WORK_ROOT = ROOT / "app" / "state" / "work"
TIMEOUT_S = 180
MAX_OUTPUT = 12_000
MAX_FIGURES = 6
ENABLED = os.environ.get("ECT_PYTHON_TOOL", "1") != "0"
MIN_FREE_MB = 400                     # a child with numpy/scipy/matplotlib commits ~200-300 MB


# ---------------------------------------------------------------- parent side

def work_dir(session_id):
    d = WORK_ROOT / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _python_exe():
    exe = Path(sys.executable)
    console = exe.with_name("python.exe")         # under pythonw: same interpreter, real stdout
    return str(console if console.exists() else exe)


def run(code, session_id, chat_state, timeout=TIMEOUT_S):
    """Run `code` in a child. Returns (output text, [figure dicts], seconds, timed_out)."""
    work = work_dir(session_id)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    code_file = work / f"_run_{stamp}.py"
    code_file.write_text(code, encoding="utf-8")
    state_file = work / "_chat_state.json"
    state_file.write_text(json.dumps(chat_state, ensure_ascii=False, default=str), encoding="utf-8")
    fig_file = work / "_figures.json"
    fig_file.unlink(missing_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2", PYTHONIOENCODING="utf-8",
               MPLBACKEND="Agg", TMP=str(work), TEMP=str(work), CUDA_VISIBLE_DEVICES="",
               PYTHONDONTWRITEBYTECODE="1")
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        env.pop(k, None)
    t0 = time.time()
    timed_out = False
    try:
        p = subprocess.run([_python_exe(), "-X", "utf8", __file__, str(work), str(code_file), str(state_file)],
                           cwd=str(work), env=env, capture_output=True, timeout=timeout,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        out = (p.stdout or b"").decode("utf-8", "replace") + (p.stderr or b"").decode("utf-8", "replace")
        if p.returncode not in (0, 1):
            out += f"\n[tiến trình Python kết thúc với mã {p.returncode:#x} - thường do thiếu bộ nhớ]"
    except subprocess.TimeoutExpired as e:
        timed_out = True
        out = ((e.stdout or b"").decode("utf-8", "replace") + (e.stderr or b"").decode("utf-8", "replace")
               + f"\n[DỪNG: quá {timeout} s]")
    figs = []
    if fig_file.exists():
        figs = json.loads(fig_file.read_text(encoding="utf-8"))[:MAX_FIGURES]
    out = out.strip() or "(không in ra gì - dùng print() để xem kết quả)"
    if len(out) > MAX_OUTPUT:
        out = out[:MAX_OUTPUT // 2] + f"\n... [cắt bớt {len(out) - MAX_OUTPUT} ký tự] ...\n" + out[-MAX_OUTPUT // 2:]
    return out, figs, time.time() - t0, timed_out


# ---------------------------------------------------------------- child side

WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
BLOCKED_PREFIXES = ("subprocess.", "os.system", "os.exec", "os.spawn", "os.posix_spawn", "os.startfile",
                    "os.kill", "socket.", "urllib.", "http.client.", "ftplib.", "smtplib.", "webbrowser.",
                    "winreg.CreateKey", "winreg.DeleteKey", "winreg.DeleteValue", "winreg.SetValue",
                    "winreg.SaveKey", "winreg.LoadKey", "_winapi.CreateProcess", "ctypes.dlopen")
PATH_EVENTS = {"os.remove": (0,), "os.rmdir": (0,), "os.rename": (0, 1), "os.replace": (0, 1),
               "os.truncate": (0,), "os.chmod": (0,), "os.chown": (0,), "os.utime": (0,), "os.mkdir": (0,),
               "os.symlink": (0, 1), "os.link": (0, 1), "shutil.rmtree": (0,), "shutil.copyfile": (1,),
               "shutil.copytree": (1,), "shutil.move": (0, 1)}


def _install_guard(work):
    work = os.path.normcase(os.path.realpath(work))

    def inside(p):
        if isinstance(p, int):                    # an already-open file descriptor
            return True
        if isinstance(p, bytes):
            p = os.fsdecode(p)
        full = os.path.normcase(os.path.realpath(os.fspath(p)))
        return full == work or full.startswith(work + os.sep)

    def hook(event, args):
        if event == "open":
            path, mode, flags = (list(args) + [None, None, None])[:3]
            writes = (isinstance(mode, str) and any(c in mode for c in "wax+")) or \
                     (mode is None and isinstance(flags, int) and flags & WRITE_FLAGS)
            if writes and path is not None and not inside(path):
                raise PermissionError(f"chỉ được ghi file trong thư mục làm việc ({work}); không ghi được {path}")
        elif event in PATH_EVENTS:
            for i in PATH_EVENTS[event]:
                if i < len(args) and args[i] is not None and not inside(args[i]):
                    raise PermissionError(f"{event} ngoài thư mục làm việc bị chặn: {args[i]}")
        elif event.startswith(BLOCKED_PREFIXES) and event not in ("socket.gethostname",):
            raise PermissionError(f"'{event}' bị chặn trong run_python (không chạy chương trình khác, không mạng)")

    sys.addaudithook(hook)


def main():
    work, code_file, state_file = sys.argv[1:4]
    sys.path.insert(0, str(ROOT))
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding="utf-8", errors="replace")

    import numpy as np
    import scipy
    from scipy import ndimage, signal, stats
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from app.backend.ect import detect, process, render, scan as sc

    chat = json.loads(Path(state_file).read_text(encoding="utf-8"))
    figures = []
    fig_file = Path(work) / "_figures.json"

    def load(scan):
        """(grid, meta) of a scan; meta has id, sensor, amp, freq, pitch_mm, size_x/y, warnings."""
        sid = sc.resolve(scan)
        grid, meta = sc.load_scan(sc.scan_path(sid))
        return grid, dict(meta, id=sid)

    def stages(scan, crop=None):
        """process.pipeline of a scan: raw, filtered, cropped, detrended, dx, zmap, margins, offset.
        crop=None uses this chat's crop choice for the scan (else 'auto')."""
        grid, meta = load(scan)
        if crop is None:
            crop = (chat.get("crop") or {}).get(meta["id"], "auto")
        return process.pipeline(grid, crop=crop)

    def yolo(scan, conf=detect.YOLO_CONF):
        """YOLO boxes [{x0,y0,x1,y1,conf}] in original grid px (onnxruntime)."""
        return detect.yolo_detect(load(scan)[0], conf)

    def threshold(scan, z=2.5, crop=None):
        st = stages(scan, crop)
        return detect.threshold_detect(st["cropped"], st["offset"], z)[0]

    def labels(scan):
        return detect.load_labels(sc.resolve(scan))[0]

    def catalog():
        return sc.catalog()

    def show(fig=None, title="", caption=""):
        """Put a matplotlib figure on the right-hand panel of the app."""
        fig = fig or plt.gcf()
        name = f"fig_{time.strftime('%H%M%S')}_{len(figures) + 1}.png"
        fig.savefig(Path(work) / name, dpi=130, bbox_inches="tight")
        plt.close(fig)
        figures.append({"file": name, "title": title or f"Hình {len(figures) + 1}", "caption": caption})
        fig_file.write_text(json.dumps(figures, ensure_ascii=False), encoding="utf-8")

    ns = {"np": np, "scipy": scipy, "ndimage": ndimage, "signal": signal, "stats": stats, "plt": plt,
          "sc": sc, "process": process, "detect": detect, "render": render, "chat": chat,
          "load": load, "stages": stages, "yolo": yolo, "threshold": threshold, "labels": labels,
          "catalog": catalog, "show": show, "WORK": Path(work), "__name__": "__main__"}
    code = Path(code_file).read_text(encoding="utf-8")
    os.chdir(work)
    _install_guard(work)
    try:
        exec(compile(code, "<run_python>", "exec"), ns)
    except SystemExit:
        pass
    except BaseException:
        import traceback
        tb = traceback.format_exc()
        if 'File "<run_python>"' in tb:           # hide this runner's own frames
            tb = tb[tb.find('File "<run_python>"'):]
        print("LỖI khi chạy:\n" + tb, file=sys.stderr)
        sys.exit(1)
    if plt.get_fignums() and not figures:         # drew but forgot show(): show the last figure
        show(title="Hình từ run_python")


if __name__ == "__main__":
    main()
