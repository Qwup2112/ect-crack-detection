#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crack_scan_app.py - Desktop application for ECT scan analysis and crack detection.

A Tkinter GUI wrapped around the tdms_crack_scan.py backend: pick a folder, browse
.tdms files, adjust parameters, view figures in the window, ask Claude about the
results, and export.

RUN:  python crack_scan_app.py
      (or double-click run_app.bat in the project root)

REQUIRES: pip install npTDMS numpy scipy matplotlib pandas anthropic
"""

import queue
import sys
import threading
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt

import tdms_crack_scan as backend
import ai_advisor


PAD = 8


class CrackScanApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ECT Scan Analysis - Crack Detection")
        self.geometry("1500x900")
        self.minsize(1100, 700)

        self.folder = None
        self.files = []          # list of Path
        self.current_fig = None
        self.current_boxes = []
        self.current_meta = None
        self.msg_queue = queue.Queue()

        # AI layer state
        self.api_key = None          # session memory only, never written to disk
        self.ai_history = []         # multi-turn conversation
        self.ai_context = None       # numbers from the most recent analysis
        self.ai_context_kind = None  # "scan" | "compare" | "sweep"
        self.ai_busy = False

        self._build_ui()
        self.after(100, self._drain_queue)

    # ------------------------------------------------------------ interface

    def _build_ui(self):
        toolbar = ttk.Frame(self, padding=(PAD, PAD, PAD, 0))
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="Choose folder...",
                   command=self.choose_folder).pack(side="left")
        self.folder_label = ttk.Label(toolbar, text="No folder selected",
                                      foreground="#666")
        self.folder_label.pack(side="left", padx=PAD)

        main = ttk.PanedWindow(self, orient="horizontal")
        main.pack(fill="both", expand=True, padx=PAD, pady=PAD)

        main.add(self._build_left(main), weight=1)
        main.add(self._build_right(main), weight=3)

        self.status = ttk.Label(self, text="Ready.", relief="sunken",
                                anchor="w", padding=(PAD, 4))
        self.status.pack(fill="x", side="bottom")

    def _build_left(self, parent):
        frame = ttk.Frame(parent)

        ttk.Label(frame, text="Scan files (.tdms)",
                  font=("", 10, "bold")).pack(anchor="w")
        ttk.Label(frame, text="Hold Ctrl/Shift to select several files for comparison.",
                  foreground="#666", wraplength=300).pack(anchor="w", pady=(0, 4))

        cols = ("sensor", "amp", "freq", "size")
        self.tree = ttk.Treeview(frame, columns=cols, show="tree headings",
                                 selectmode="extended", height=20)
        self.tree.heading("#0", text="File name")
        self.tree.column("#0", width=230, stretch=True)
        for c, txt, w in (("sensor", "Sensor", 60), ("amp", "Amp(V)", 60),
                          ("freq", "Freq(kHz)", 70), ("size", "Plate", 80)):
            self.tree.heading(c, text=txt)
            self.tree.column(c, width=w, anchor="center", stretch=False)
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<Double-1>", lambda e: self.run_report())

        sb = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.place(in_=self.tree, relx=1.0, relheight=1.0, anchor="ne")

        btns = ttk.Frame(frame)
        btns.pack(fill="x", pady=(PAD, 0))
        ttk.Button(btns, text="1 - Draw the 6 figures",
                   command=self.run_report).pack(fill="x", pady=2)
        ttk.Button(btns, text="2 - Locate cracks",
                   command=self.run_detect).pack(fill="x", pady=2)
        ttk.Button(btns, text="3 - Compare sensors (several files)",
                   command=self.run_compare).pack(fill="x", pady=2)
        ttk.Separator(btns, orient="horizontal").pack(fill="x", pady=4)
        ttk.Button(btns, text="4 - Analyze ALL data",
                   command=self.run_sweep).pack(fill="x", pady=2)
        self.sweep_bar = ttk.Progressbar(btns, mode="determinate")
        self.sweep_bar.pack(fill="x", pady=(2, 0))
        return frame

    def _build_right(self, parent):
        frame = ttk.Frame(parent)
        self._build_params(frame).pack(fill="x", pady=(0, PAD))

        self.nb = ttk.Notebook(frame)
        self.nb.pack(fill="both", expand=True)

        self.plot_tab = ttk.Frame(self.nb)
        self.nb.add(self.plot_tab, text="Figures")
        self.plot_placeholder = ttk.Label(
            self.plot_tab, foreground="#666", justify="center",
            text="\n\nSelect a file on the left, then press "
                 "“Draw the 6 figures” or “Locate cracks”.")
        self.plot_placeholder.pack(expand=True)

        table_tab = ttk.Frame(self.nb)
        self.nb.add(table_tab, text="Results table")
        self.table = ttk.Treeview(table_tab, show="headings", height=12)
        self.table.pack(fill="both", expand=True)
        tsb = ttk.Scrollbar(table_tab, orient="horizontal", command=self.table.xview)
        self.table.configure(xscrollcommand=tsb.set)
        tsb.pack(fill="x")

        self.nb.add(self._build_ai_tab(self.nb), text="AI assistant")

        log_tab = ttk.Frame(self.nb)
        self.nb.add(log_tab, text="Log")
        self.log = tk.Text(log_tab, wrap="word", height=10, state="disabled",
                           font=("Consolas", 9))
        self.log.pack(fill="both", expand=True)
        return frame

    def _build_ai_tab(self, parent):
        frame = ttk.Frame(parent, padding=PAD)

        bar = ttk.Frame(frame)
        bar.pack(fill="x")
        ttk.Button(bar, text="Explain current results",
                   command=self.ai_explain).pack(side="left")
        ttk.Button(bar, text="New conversation",
                   command=self.ai_reset).pack(side="left", padx=4)
        ttk.Button(bar, text="API key...",
                   command=self.ai_set_key).pack(side="left")
        self.ai_status = ttk.Label(bar, foreground="#666", text="")
        self.ai_status.pack(side="left", padx=PAD)

        self.ai_out = tk.Text(frame, wrap="word", state="disabled",
                              font=("Segoe UI", 10), padx=8, pady=8)
        self.ai_out.pack(fill="both", expand=True, pady=(PAD, 4))
        self.ai_out.tag_configure("you", foreground="#1a5fb4",
                                  font=("Segoe UI", 10, "bold"))
        self.ai_out.tag_configure("ai", foreground="#5a5a5a",
                                  font=("Segoe UI", 10, "bold"))
        self.ai_out.tag_configure("err", foreground="#a01b1b")

        ask = ttk.Frame(frame)
        ask.pack(fill="x")
        self.ai_entry = ttk.Entry(ask)
        self.ai_entry.pack(side="left", fill="x", expand=True)
        self.ai_entry.bind("<Return>", lambda e: self.ai_ask())
        ttk.Button(ask, text="Send", command=self.ai_ask).pack(side="left", padx=(4, 0))

        ttk.Label(frame, foreground="#666", wraplength=900, justify="left",
                  text="The assistant only interprets numbers this application computed; "
                       "it never derives statistics from the raw data itself. Run an "
                       "analysis first, then press “Explain current results” and "
                       "ask follow-up questions in the box above."
                  ).pack(anchor="w", pady=(4, 0))
        return frame

    def _build_params(self, parent):
        box = ttk.LabelFrame(parent, text="Parameters", padding=PAD)

        pre = ttk.Frame(box)
        pre.pack(fill="x")
        ttk.Label(pre, text="Preprocessing:", font=("", 9, "bold")).pack(side="left")

        self.v_filter = tk.StringVar(value="median")
        self.v_kernel = tk.IntVar(value=3)
        self.v_margin = tk.StringVar(value="auto")
        self.v_detrend = tk.StringVar(value="row")

        self._combo(pre, "Filter", self.v_filter, ["median", "gaussian", "none"], 10)
        self._spin(pre, "Kernel", self.v_kernel, 1, 15)
        self._entry(pre, "Border crop", self.v_margin, 7,
                    "'auto' measures the noisy border; or enter a pixel count")
        self._combo(pre, "Detrend", self.v_detrend,
                    ["row", "col", "both", "poly", "none"], 7)

        det = ttk.Frame(box)
        det.pack(fill="x", pady=(6, 0))
        ttk.Label(det, text="Detection:", font=("", 9, "bold")).pack(side="left")

        self.v_zthresh = tk.DoubleVar(value=2.5)
        self.v_merge = tk.IntVar(value=2)
        self.v_minarea = tk.IntVar(value=6)
        self.v_dropedge = tk.BooleanVar(value=False)

        self._spin(det, "|z| threshold", self.v_zthresh, 0.5, 10.0, inc=0.5, width=6)
        self._spin(det, "Merge gap", self.v_merge, 0, 10)
        self._spin(det, "Min area", self.v_minarea, 1, 200, width=6)
        ttk.Checkbutton(det, text="Drop edge-touching boxes",
                        variable=self.v_dropedge).pack(side="left", padx=(10, 0))

        ttk.Label(box, foreground="#666", wraplength=900, justify="left",
                  text="Tip: if one crack splits into several boxes, raise “Merge gap”; "
                       "if two cracks fuse into one, lower it (crack rows on this coupon are "
                       "only ~5px apart). If the background slopes along X (sensor P), set "
                       "Detrend = both."
                  ).pack(anchor="w", pady=(6, 0))
        return box

    def _combo(self, parent, label, var, values, width):
        ttk.Label(parent, text=label).pack(side="left", padx=(10, 2))
        c = ttk.Combobox(parent, textvariable=var, values=values, width=width,
                         state="readonly")
        c.pack(side="left")
        return c

    def _spin(self, parent, label, var, lo, hi, inc=1, width=5):
        ttk.Label(parent, text=label).pack(side="left", padx=(10, 2))
        s = ttk.Spinbox(parent, textvariable=var, from_=lo, to=hi, increment=inc,
                        width=width)
        s.pack(side="left")
        return s

    def _entry(self, parent, label, var, width, tip=""):
        ttk.Label(parent, text=label).pack(side="left", padx=(10, 2))
        e = ttk.Entry(parent, textvariable=var, width=width)
        e.pack(side="left")
        if tip:
            self._tooltip(e, tip)
        return e

    def _tooltip(self, widget, text):
        tip = {"win": None}

        def show(_):
            if tip["win"]:
                return
            x = widget.winfo_rootx() + 20
            y = widget.winfo_rooty() + widget.winfo_height() + 4
            win = tk.Toplevel(widget)
            win.wm_overrideredirect(True)
            win.wm_geometry(f"+{x}+{y}")
            tk.Label(win, text=text, background="#ffffe0", relief="solid",
                     borderwidth=1, padx=4, pady=2, wraplength=300).pack()
            tip["win"] = win

        def hide(_):
            if tip["win"]:
                tip["win"].destroy()
                tip["win"] = None

        widget.bind("<Enter>", show)
        widget.bind("<Leave>", hide)

    # ------------------------------------------------------------ data

    def choose_folder(self):
        folder = filedialog.askdirectory(title="Choose the folder containing .tdms files")
        if not folder:
            return
        self.folder = Path(folder)
        self.load_files()

    def load_files(self):
        self.folder_label.config(text=str(self.folder))
        self.tree.delete(*self.tree.get_children())
        self.files = sorted(self.folder.rglob("*.tdms"))
        skipped = 0
        for p in self.files:
            info = backend.parse_name(p)
            self.tree.insert("", "end", text=p.name, values=(
                info.get("sensor", "-"), info.get("amp", "-"),
                info.get("freq", "-"),
                f"{info.get('w','?')}x{info.get('h','?')}mm" if info.get("w") else "-",
            ))
            if not info:
                skipped += 1
        self._log(f"Loaded {len(self.files)} .tdms files from {self.folder}")
        if skipped:
            self._log(f"  ({skipped} file(s) do not match the naming convention - "
                      "still readable, but their measurement parameters cannot be "
                      "taken from the name)")
        self._status(f"{len(self.files)} files.")

    def _selected_paths(self):
        idxs = [self.tree.index(i) for i in self.tree.selection()]
        return [self.files[i] for i in idxs]

    def _one_selected(self):
        paths = self._selected_paths()
        if not paths:
            messagebox.showinfo("No file selected",
                                "Select a file in the list on the left.")
            return None
        return paths[0]

    # ------------------------------------------------------------ running tasks

    def _params(self):
        """
        Snapshot every parameter into a plain Python dict.

        MUST be called on the main thread: Tk variables (.get()) cannot be read from a
        worker thread - Python raises 'main thread is not in main loop'. The worker only
        ever receives this snapshot and never touches a widget.
        """
        return {
            "filter": self.v_filter.get(), "kernel": self.v_kernel.get(),
            "margin": self.v_margin.get(), "detrend": self.v_detrend.get(),
            "z_thresh": self.v_zthresh.get(), "merge": self.v_merge.get(),
            "min_area": self.v_minarea.get(), "drop_edge": self.v_dropedge.get(),
        }

    def _run_async(self, fn, done, on_error=None):
        self._status("Working...")

        def worker():
            try:
                result = fn()
                self.msg_queue.put(("ok", done, result))
            except Exception as e:
                payload = (e, traceback.format_exc())
                if on_error:
                    # errors with their own handling (e.g. the AI tab writes into the chat)
                    self.msg_queue.put(("note", on_error, payload))
                else:
                    self.msg_queue.put(("err", None, payload))

        threading.Thread(target=worker, daemon=True).start()

    def _drain_queue(self):
        try:
            while True:
                kind, done, payload = self.msg_queue.get_nowait()
                if kind == "ok":
                    done(payload)
                    self._status("Done.")
                elif kind == "chunk":
                    # a text chunk streamed from the AI thread - write it straight out
                    self._ai_write(payload[0], payload[1])
                elif kind == "note":
                    done(payload)
                else:
                    err, tb = payload
                    self._log("ERROR: " + tb)
                    self._status("Something failed - see the Log tab.")
                    messagebox.showerror("Error", str(err))
        except queue.Empty:
            pass
        self.after(50, self._drain_queue)

    def _prepare(self, path, prm):
        """Read + filter + crop. Shared by both single-file tasks."""
        grid, meta = backend.load_scan(path)
        filtered = backend.apply_filter(grid, prm["filter"], prm["kernel"])
        cropped, margins, crop_label = backend.resolve_crop(filtered, prm["margin"])
        return grid, filtered, cropped, meta, margins, crop_label

    def run_report(self):
        path = self._one_selected()
        if not path:
            return
        prm = self._params()

        def work():
            grid, filtered, cropped, meta, margins, crop_label = self._prepare(path, prm)
            title = (f"Crack scan report - {path.name}\n"
                     f"sensor={meta.get('sensor','?')}  amp={meta.get('amp','?')}V  "
                     f"freq={meta.get('freq','?')}kHz  "
                     f"grid={meta['size_x']}x{meta['size_y']}px")
            fig, dx, _ = backend.build_report_figure(
                grid, filtered, cropped, meta, crop_label,
                f"{prm['filter']}, k={prm['kernel']}", title)
            return fig, path, meta, crop_label, cropped, grid, margins

        def done(res):
            fig, path, meta, crop_label, cropped, grid, margins = res
            self._show_fig(fig)
            self.current_meta = meta
            self.ai_context = ai_advisor.build_context(
                meta, prm, margins=margins, cropped_shape=cropped.shape,
                stats={
                    "minimum amplitude": f"{np.nanmin(grid):.6g}",
                    "maximum amplitude": f"{np.nanmax(grid):.6g}",
                    "std (original image)": f"{np.nanstd(grid):.6g}",
                    "std (after filter + crop)": f"{np.nanstd(cropped):.6g}",
                    "dynamic range": f"{np.nanmax(grid) - np.nanmin(grid):.6g}",
                })
            self.ai_context_kind = "scan"
            self._log(f"\n[6 figures] {path.name}")
            self._log(f"  Original grid : {meta['size_x']}x{meta['size_y']} px "
                      f"(~{meta['pitch_mm']:.3f} mm/px)")
            self._log(f"  Border crop   : {crop_label} -> "
                      f"{cropped.shape[1]}x{cropped.shape[0]} px left")
            self._log(f"  Amplitude     : min {np.nanmin(grid):.4g}  "
                      f"max {np.nanmax(grid):.4g}  std {np.nanstd(grid):.4g}")

        self._run_async(work, done)

    def run_detect(self):
        path = self._one_selected()
        if not path:
            return
        prm = self._params()

        def work():
            grid, filtered, cropped, meta, margins, crop_label = self._prepare(path, prm)
            boxes, zmap, mask = backend.detect_cracks(
                cropped, z_thresh=prm["z_thresh"], min_area=prm["min_area"],
                merge_gap=prm["merge"], detrend_method=prm["detrend"])
            n_edge = sum(b["touches_edge"] for b in boxes)
            if prm["drop_edge"]:
                boxes = [b for b in boxes if not b["touches_edge"]]
                for i, b in enumerate(boxes, start=1):
                    b["id"] = f"crack_{i}"
            title = (f"Crack localization - {path.name}\n"
                     f"sensor={meta.get('sensor','?')}  "
                     f"freq={meta.get('freq','?')}kHz  crop: {crop_label}  |  "
                     f"threshold |z|>{prm['z_thresh']}  |  {len(boxes)} candidates")
            fig = backend.build_detect_figure(
                cropped, zmap, mask, boxes, meta, title,
                prm["z_thresh"], prm["detrend"])
            return fig, boxes, n_edge, path, meta, margins, crop_label, cropped, prm

        def done(res):
            fig, boxes, n_edge, path, meta, margins, crop_label, cropped, prm = res
            self._show_fig(fig)
            self.current_boxes = boxes
            self.current_meta = meta
            self.current_crop = margins
            self.current_path = path
            self.ai_context = ai_advisor.build_context(
                meta, prm, margins=margins, cropped_shape=cropped.shape, boxes=boxes,
                stats={
                    "std (after filter + crop)": f"{np.nanstd(cropped):.6g}",
                    "candidates touching the image edge": n_edge,
                })
            self.ai_context_kind = "scan"

            pitch = meta["pitch_mm"]
            rows = [{
                "id": b["id"], "x0": b["x0"], "y0": b["y0"], "x1": b["x1"], "y1": b["y1"],
                "w_px": b["x1"] - b["x0"], "h_px": b["y1"] - b["y0"],
                "x_mm": round(b["x0"] * pitch, 1), "y_mm": round(b["y0"] * pitch, 1),
                "z_score": b["score"], "touches_edge": "yes" if b["touches_edge"] else "",
            } for b in boxes]
            self._fill_table(pd.DataFrame(rows))

            self._log(f"\n[Detection] {path.name}")
            self._log(f"  Border crop : {crop_label} -> "
                      f"{cropped.shape[1]}x{cropped.shape[0]} px left")
            self._log(f"  Found {len(boxes)} candidates "
                      f"(threshold |z|>{prm['z_thresh']}, detrend {prm['detrend']})")
            if n_edge:
                verb = "Dropped" if prm["drop_edge"] else "Kept"
                self._log(f"  {verb} {n_edge} candidate(s) touching the image edge "
                          "(orange dashed in the figure).")
            if not boxes:
                self._log("  No candidates. Lower the |z| threshold, or run "
                          "Compare sensors to check whether this sensor has any contrast.")
            elif len(boxes) > 60:
                self._log("  WARNING: too many candidates - the threshold is too low "
                          "or background noise remains.")
            self._log("  Coordinates are relative to the CROPPED image. Add "
                      f"+{margins['left']} (X) / +{margins['top']} (Y) for the original.")
            self.nb.select(1)

        self._run_async(work, done)

    def run_compare(self):
        paths = self._selected_paths()
        if len(paths) < 2:
            messagebox.showinfo(
                "Select at least 2 files",
                "Hold Ctrl or Shift to select several files, then press again.")
            return
        prm = self._params()

        def work():
            rows, skipped = [], []
            for p in paths:
                try:
                    grid, meta = backend.load_scan(p)
                except ValueError as e:
                    skipped.append(str(e))
                    continue
                filtered = backend.apply_filter(grid, prm["filter"], prm["kernel"])
                flat = backend.detrend(filtered, prm["detrend"])
                dx = backend.x_derivative(filtered)
                rows.append({
                    "file": p.name, "sensor": meta.get("sensor", "?"),
                    "amp_V": meta.get("amp", ""), "freq_kHz": meta.get("freq", ""),
                    "grid": f"{meta['size_x']}x{meta['size_y']}",
                    "mean": round(float(np.nanmean(grid)), 6),
                    "std": round(float(np.nanstd(grid)), 6),
                    "dynamic_range": round(float(np.nanmax(grid) - np.nanmin(grid)), 6),
                    "std_detrended": round(float(np.nanstd(flat)), 6),
                    "dx_std": round(float(np.nanstd(dx)), 6),
                })
            return pd.DataFrame(rows), skipped

        def done(res):
            df, skipped = res
            if df.empty:
                self._log("No file could be read.")
                return
            df = df.sort_values("dx_std", ascending=False)
            self._fill_table(df)
            self.ai_context = ai_advisor.build_context(
                {"path": f"{len(df)} files in {self.folder}"}, prm,
                compare_rows=df.to_dict("records"))
            self.ai_context_kind = "compare"
            self._log(f"\n[Compare] {len(df)} files, sorted by dx_std descending.")
            for s in skipped:
                self._log(f"  [SKIPPED] {s}")
            self._log("  NOTE: dx_std is only a RELATIVE index, not a CNR. It rewards a "
                      "large value scale and can rank sensors the opposite way to a real "
                      "CNR. For a real CNR, run the compare command with --boxes on the "
                      "command line.")
            self.nb.select(1)

        self._run_async(work, done)

    def run_sweep(self):
        if not self.files:
            messagebox.showinfo("No files", "Choose a data folder first.")
            return
        prm = self._params()
        paths = list(self.files)
        n = len(paths)
        self.sweep_bar.config(maximum=n, value=0)
        self._log(f"\n[All data] Scanning {n} files...")

        def work():
            rows = []
            for i, p in enumerate(paths, start=1):
                rows.append(backend.analyze_one(
                    p, filter_method=prm["filter"], kernel=prm["kernel"],
                    margin=prm["margin"], detrend_method=prm["detrend"],
                    z_thresh=prm["z_thresh"], min_area=prm["min_area"],
                    merge_gap=prm["merge"]))
                self.msg_queue.put(("note", self._sweep_tick, (i, n)))
            return pd.DataFrame(rows)

        def done(df):
            self.sweep_bar.config(value=n)
            ok, skipped, top, best = backend.sweep_summary(df)
            self.sweep_df = df

            show = ["folder", "file", "sensor", "amp", "freq", "variant",
                    "n_boxes", "n_strong", "n_edge_boxes", "z_max", "z_top5", "status"]
            show = [c for c in show if c in df.columns]
            if not ok.empty:
                ordered = ok.sort_values("z_top5", ascending=False)
                self._fill_table(pd.concat([ordered, skipped])[show])
            else:
                self._fill_table(df[show])

            self._log(f"  Analyzed {len(ok)}/{len(df)} files"
                      + (f", skipped {len(skipped)}" if len(skipped) else ""))
            for _, r in skipped.iterrows():
                self._log(f"    [SKIPPED] {r['folder']}/{r['file']}: {r['status']}")
            if not best.empty:
                self._log("  Best configuration per (folder, sensor):")
                for _, r in best.iterrows():
                    self._log(f"    {r['folder']:<16} {r['sensor']}  "
                              f"amp={r['amp']}V freq={r['freq']}kHz  "
                              f"z_top5={r['z_top5']}  ({r['n_strong']} strong boxes)")
            self._log("  NOTE: z_top5 measures deviation from each image's own background, "
                      "not CNR against real cracks.")

            if not ok.empty:
                self.ai_context = ai_advisor.build_sweep_context(
                    df, ok, skipped, best, prm, root=str(self.folder))
                self.ai_context_kind = "sweep"
            self.nb.select(1)

        self._run_async(work, done)

    def _sweep_tick(self, payload):
        i, n = payload
        self.sweep_bar.config(value=i)
        self._status(f"Scanning {i}/{n} files...")

    # ------------------------------------------------------------ AI layer

    def _ai_write(self, text, tag=None):
        self.ai_out.config(state="normal")
        self.ai_out.insert("end", text, tag or ())
        self.ai_out.see("end")
        self.ai_out.config(state="disabled")

    def ai_set_key(self):
        from tkinter import simpledialog
        key = simpledialog.askstring(
            "Anthropic API key",
            "Paste your API key (sk-ant-...).\n\n"
            "The key is kept in memory for this session only and is NEVER written to disk.\n"
            "To avoid retyping it, set the ANTHROPIC_API_KEY environment variable:\n"
            '    setx ANTHROPIC_API_KEY "sk-ant-..."   (then reopen the terminal)',
            show="*", parent=self)
        if key:
            self.api_key = key.strip()
            self.ai_status.config(text="Key accepted (this session only).")

    def ai_reset(self):
        self.ai_history = []
        self.ai_out.config(state="normal")
        self.ai_out.delete("1.0", "end")
        self.ai_out.config(state="disabled")
        self.ai_status.config(text="Conversation cleared.")

    def _ai_guard(self):
        if self.ai_busy:
            messagebox.showinfo("Busy", "The assistant is still replying - wait for it.")
            return False
        if not (self.api_key or ai_advisor.has_credentials()):
            messagebox.showinfo(
                "No API key",
                "No API key found.\n\n"
                "Option 1: set the ANTHROPIC_API_KEY environment variable and restart.\n"
                "Option 2: press “API key...” to enter one for this session.")
            return False
        return True

    def _ai_send(self, question, image_png=None, context_text=None, header=None):
        """Send one turn and stream the reply into the widget. Runs on a worker thread."""
        if not self._ai_guard():
            return
        self.ai_busy = True
        self.nb.select(2)
        if header:
            self._ai_write(header, "you")
        self._ai_write("\n\nClaude: ", "ai")
        self.ai_status.config(text="Asking Claude...")

        key = self.api_key
        history = self.ai_history

        def work():
            client = ai_advisor.make_client(key)
            ai_advisor.stream_reply(
                client, history,
                on_text=lambda t: self.msg_queue.put(("chunk", None, (t, None))),
                on_thinking=None,
                image_png=image_png, context_text=context_text, question=question)
            return None

        def done(_):
            self.ai_busy = False
            self.ai_status.config(text="Done.")
            self._ai_write("\n\n")

        def failed(payload):
            self.ai_busy = False
            err, tb = payload
            self._ai_write("\n[Error] " + ai_advisor.friendly_error(err) + "\n\n", "err")
            self.ai_status.config(text="Failed - see the message above.")
            self._log("AI ERROR: " + tb)

        self._run_async(work, done, on_error=failed)

    def ai_explain(self):
        if self.ai_context is None:
            messagebox.showinfo(
                "No results yet",
                "Run “Draw the 6 figures” or “Locate cracks” first, "
                "then come back here.")
            return
        ctx = self.ai_context
        kind = self.ai_context_kind
        # A whole-dataset question carries no image: the context is a hundred-row table.
        fig = None if kind == "sweep" else self.current_fig
        png = ai_advisor.figure_to_png_bytes(fig) if fig is not None else None
        first = not self.ai_history

        if kind == "sweep":
            q = ("Analyse this entire measurement sweep. Specifically: (1) which sensor and "
                 "frequency range give the best contrast, and is there a clear trend with "
                 "frequency or with excitation amplitude; (2) are the results consistent "
                 "across the folders (separate measurement sessions), and where do they "
                 "disagree; (3) which measurements look broken or anomalous and why; "
                 "(4) which configurations should be measured again or added.")
        elif kind == "compare":
            q = ("Compare these files: which configuration gives the best contrast and why? "
                 "Say clearly which metric is trustworthy and which is misleading.")
        else:
            q = ("Analyse this scan: is the signal strong enough for crack detection, is the "
                 "preprocessing sound, and are the candidates found trustworthy? If a "
                 "parameter needs changing, say which one and in which direction.")

        self._ai_send(question=q,
                      image_png=png if first else None,
                      context_text=ctx if first else None,
                      header="You: explain the current results")

    def ai_ask(self):
        q = self.ai_entry.get().strip()
        if not q:
            return
        self.ai_entry.delete(0, "end")
        first = not self.ai_history
        png = None
        ctx = None
        if first and self.ai_context:
            ctx = self.ai_context
            if self.ai_context_kind != "sweep" and self.current_fig is not None:
                png = ai_advisor.figure_to_png_bytes(self.current_fig)
        self._ai_send(question=q, image_png=png, context_text=ctx,
                      header=f"You: {q}")

    # ------------------------------------------------------------ display

    def _show_fig(self, fig):
        for w in self.plot_tab.winfo_children():
            w.destroy()
        if self.current_fig is not None:
            plt.close(self.current_fig)
        self.current_fig = fig

        # The button bar must be packed BEFORE the canvas: the canvas uses expand=True,
        # so anything packed after it gets pushed off the bottom of the window.
        bar = ttk.Frame(self.plot_tab, padding=(0, 4))
        bar.pack(side="bottom", fill="x")
        ttk.Button(bar, text="Save figure...", command=self._save_fig).pack(side="left", padx=2)
        ttk.Button(bar, text="Save table CSV...", command=self._save_table).pack(side="left", padx=2)
        ttk.Button(bar, text="Save boxes JSON...", command=self._save_boxes).pack(side="left", padx=2)

        canvas = FigureCanvasTkAgg(fig, master=self.plot_tab)
        canvas.draw()
        NavigationToolbar2Tk(canvas, self.plot_tab).update()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.nb.select(0)

    def _fill_table(self, df):
        self._table_df = df
        self.table.delete(*self.table.get_children())
        self.table["columns"] = list(df.columns)
        for c in df.columns:
            self.table.heading(c, text=c)
            self.table.column(c, width=max(70, min(190, 9 * len(str(c)) + 40)),
                              anchor="center")
        for _, r in df.iterrows():
            self.table.insert("", "end", values=list(r))

    def _save_fig(self):
        if self.current_fig is None:
            return
        p = filedialog.asksaveasfilename(defaultextension=".png",
                                         filetypes=[("PNG", "*.png")])
        if p:
            self.current_fig.savefig(p, dpi=150, facecolor=backend.SURFACE)
            self._log(f"Figure saved: {p}")

    def _save_table(self):
        df = getattr(self, "_table_df", None)
        if df is None or df.empty:
            messagebox.showinfo("No table", "Run Detection or Compare first.")
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv",
                                         filetypes=[("CSV", "*.csv")])
        if p:
            df.to_csv(p, index=False, encoding="utf-8-sig")
            self._log(f"Table saved: {p}")

    def _save_boxes(self):
        if not self.current_boxes:
            messagebox.showinfo("No boxes", "Run Locate cracks first.")
            return
        p = filedialog.asksaveasfilename(defaultextension=".json",
                                         filetypes=[("JSON", "*.json")])
        if not p:
            return
        import json
        payload = {
            "source": self.current_path.name,
            "crop": self.current_crop,
            "note": ("Coordinates are relative to the CROPPED image. Add crop.left / "
                     "crop.top to convert to original-image coordinates."),
            "z_thresh": self.v_zthresh.get(),
            "boxes": self.current_boxes,
        }
        Path(p).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                           encoding="utf-8")
        self._log(f"Boxes saved: {p}")

    def _log(self, text):
        self.log.config(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def _status(self, text):
        self.status.config(text=text)


if __name__ == "__main__":
    CrackScanApp().mainloop()
