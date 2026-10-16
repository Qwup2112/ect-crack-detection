#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crack_scan_app.py - Desktop application for ECT scan analysis and crack detection.

A Tkinter GUI wrapped around the tdms_crack_scan.py backend: browse the .tdms scans
(grouped by measurement session), view each map full size, compare R and P, locate
cracks, rank the whole dataset, and ask Claude about the results.

RUN:  python crack_scan_app.py
      (or double-click run_app.bat in the project root)

REQUIRES: pip install npTDMS numpy scipy matplotlib pandas anthropic
"""

import gc
import os
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
from matplotlib.widgets import RectangleSelector
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib as mpl

import tdms_crack_scan as backend
import ai_advisor
import crack_labels


PAD = 8
FONT = "Segoe UI"
MUTED = "#6b6a64"
ACCENT = "#0b5394"
DIR_PREFIX = "dir:"   # tree item ids of session folders; files use their full path

# The figure toolbar's save button (the disk icon) replaces a separate "Save figure" button:
# point it at results/ and save at print resolution.
mpl.rcParams["savefig.directory"] = str(backend.RESULTS_DIR)
mpl.rcParams["savefig.dpi"] = 150


class CrackScanApp(tk.Tk):
    # profile direction: label shown -> axis the signal runs along
    PROFILE_MODES = {f"row at {backend.Y_LABEL}": "x", f"column at {backend.X_LABEL}": "y"}

    def __init__(self):
        super().__init__()
        self.title("ECT Scan Analysis - Crack Detection")
        self.geometry("1500x900")
        self.minsize(1100, 700)

        self.folder = None
        self.files = []          # list of Path
        self.current_fig = None
        self.views = []            # the maps of the last run, shown one at a time
        self.view_title = ""
        self.view_meta = None
        self.view_overview = None  # () -> the all-in-one figure
        self.pair_views = None     # P views aligned with self.views in R + P pair mode
        self.ai_fig_fn = None      # () -> the figure sent to the AI assistant
        self.v_view = tk.StringVar()
        self.v_prof_label = tk.StringVar(value=next(iter(self.PROFILE_MODES)))
        self.v_prof_pos = tk.StringVar(value="35")
        self.v_px = tk.StringVar(value="20")
        self.v_py = tk.StringVar(value="35")
        self.current_boxes = []
        # hand labelling (ground truth for the model), boxes in ORIGINAL grid pixels
        self.v_label_mode = tk.BooleanVar(value=False)
        self.label_file = None      # scan the labels belong to
        self.label_grid = None      # its grid shape (rows, cols)
        self.labels = []
        self.sel_label = None       # index of the selected box
        self._selector = None       # keeps the RectangleSelector alive
        self._label_patches = []
        self._measure_marks = []     # overlay of the crack measured by a click
        self.score_text = ""         # model-vs-labels line, kept while boxes are measured
        self.current_meta = None
        self.msg_queue = queue.Queue()

        # AI layer state
        self.api_key = None          # session memory only, never written to disk
        self.ai_history = []         # multi-turn conversation
        self.ai_context = None       # numbers from the most recent analysis
        self.ai_context_kind = None  # "scan" | "sweep"
        self.ai_busy = False

        # Compiled libraries are imported HERE, on the main thread: imported first by a
        # worker, scipy came up half-initialised and the first Draw figures failed.
        preload_error = backend.preload()
        # Tk objects must be freed on the main thread. The garbage collector runs on
        # whichever thread happens to allocate, so a worker could free a figure's Tk
        # images ("main thread is not in main loop", or a Tcl crash). Automatic collection
        # is off; _drain_queue collects every 2 s on the main thread instead.
        gc.disable()
        self._gc_ticks = 0

        self._build_ui()
        if preload_error:
            self._log("WARNING: " + preload_error)
        self.after(100, self._drain_queue)
        if backend.DATA_DIR.is_dir():            # open straight on the dataset
            self.folder = backend.DATA_DIR
            self.load_files()

    # ------------------------------------------------------------ interface

    def _build_ui(self):
        self._style()

        header = ttk.Frame(self, padding=(PAD * 2, PAD, PAD * 2, PAD))
        header.pack(fill="x")
        ttk.Label(header, text="ECT Crack Scan", style="Title.TLabel").pack(side="left")
        ttk.Button(header, text="Change folder...",
                   command=self.choose_folder).pack(side="right")
        self.folder_label = ttk.Label(header, text="No folder", foreground=MUTED)
        self.folder_label.pack(side="right", padx=PAD)
        ttk.Separator(self).pack(fill="x")

        self.status_bar = ttk.Frame(self, padding=(PAD, 3))
        self.status_bar.pack(fill="x", side="bottom")
        self.status = ttk.Label(self.status_bar, text="Ready.", foreground=MUTED)
        self.status.pack(side="left")
        self.sweep_bar = ttk.Progressbar(self.status_bar, mode="determinate", length=220)
        ttk.Separator(self).pack(fill="x", side="bottom")

        main = ttk.PanedWindow(self, orient="horizontal")
        main.pack(fill="both", expand=True, padx=PAD, pady=PAD)
        main.add(self._build_left(main), weight=1)
        main.add(self._build_right(main), weight=4)

    def _style(self):
        st = ttk.Style(self)
        if "vista" in st.theme_names():
            st.theme_use("vista")
        st.configure(".", font=(FONT, 10))
        st.configure("Title.TLabel", font=(FONT, 14, "bold"), foreground="#1f1f1f")
        st.configure("Section.TLabel", font=(FONT, 9, "bold"), foreground=MUTED)
        st.configure("Hint.TLabel", font=(FONT, 9), foreground=MUTED)
        st.configure("Result.TLabel", font=(FONT, 10, "bold"), foreground=ACCENT)
        st.configure("Primary.TButton", font=(FONT, 10, "bold"), padding=(8, 6))
        st.configure("TButton", padding=(6, 4))
        st.configure("Treeview", rowheight=24)
        st.configure("Treeview.Heading", font=(FONT, 9, "bold"))
        st.configure("TNotebook.Tab", padding=(14, 5))
        st.configure("Toolbutton", padding=(10, 4))

    def _build_left(self, parent):
        frame = ttk.Frame(parent, padding=(0, 0, PAD, 0))

        ttk.Label(frame, text="SCAN FILES", style="Section.TLabel").pack(anchor="w")
        box = ttk.Frame(frame)
        box.pack(fill="both", expand=True, pady=(4, 0))
        cols = ("sensor", "amp", "freq", "labels")
        self.tree = ttk.Treeview(box, columns=cols, show="tree headings", selectmode="extended")
        self.tree.heading("#0", text="Session / file")
        self.tree.column("#0", width=260, stretch=True)
        for c, txt, w in (("sensor", "Sensor", 56), ("amp", "Amp (V)", 62),
                          ("freq", "Freq (kHz)", 76), ("labels", "Labels", 56)):
            self.tree.heading(c, text=txt)
            self.tree.column(c, width=w, anchor="center", stretch=False)
        self.tree.tag_configure("session", font=(FONT, 10, "bold"))
        self.tree.tag_configure("template", foreground="#9a9a9a")
        sb = ttk.Scrollbar(box, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.tree.pack(side="left", fill="both", expand=True)
        self.tree.bind("<Double-1>", self._on_double_click)
        ttk.Label(frame, text="Double-click a file to draw it.",
                  style="Hint.TLabel").pack(anchor="w", pady=(4, 0))

        one = ttk.LabelFrame(frame, text=" Selected file ", padding=PAD)
        one.pack(fill="x", pady=(PAD, 0))
        ttk.Button(one, text="Draw figures", style="Primary.TButton",
                   command=self.run_report).pack(fill="x", pady=2)
        ttk.Button(one, text="R + P side by side",
                   command=self.run_pair).pack(fill="x", pady=2)
        ttk.Button(one, text="Locate cracks",
                   command=self.run_detect).pack(fill="x", pady=2)
        ttk.Button(one, text="Detect with model (YOLO)",
                   command=self.run_model_detect).pack(fill="x", pady=2)

        allf = ttk.LabelFrame(frame, text=" Whole dataset ", padding=PAD)
        allf.pack(fill="x", pady=(PAD, 0))
        ttk.Button(allf, text="Analyze & rank all files",
                   command=self.run_sweep).pack(fill="x", pady=2)
        ttk.Button(allf, text="Train crack model (YOLO)",
                   command=self.run_train).pack(fill="x", pady=2)
        ttk.Button(allf, text="Detect all → raw dataset",
                   command=self.run_raw_dataset).pack(fill="x", pady=2)
        return frame

    def _build_right(self, parent):
        frame = ttk.Frame(parent, padding=(PAD, 0, 0, 0))
        self._build_params(frame).pack(fill="x", pady=(0, PAD))

        self.nb = ttk.Notebook(frame)
        typing = (tk.Entry, ttk.Entry, tk.Text, ttk.Spinbox, ttk.Combobox)
        for key, step in (("<Left>", -1), ("<Right>", 1)):
            self.bind_all(key, lambda e, s=step: None if isinstance(e.widget, typing)
                          else self._step_view(s))
        self.bind_all("<Delete>", lambda e: None if isinstance(e.widget, typing)
                      else self._label_delete())
        self.nb.pack(fill="both", expand=True)

        self.plot_tab = ttk.Frame(self.nb, padding=(0, PAD, 0, 0))
        self.nb.add(self.plot_tab, text="Figures")
        ttk.Label(self.plot_tab, style="Hint.TLabel", justify="center",
                  text="Select a file on the left, then press \u201cDraw figures\u201d.\n"
                       "Each map opens on its own, full size.").pack(expand=True)

        table_tab = ttk.Frame(self.nb, padding=(0, PAD, 0, 0))
        self.nb.add(table_tab, text="Results")
        tbar = ttk.Frame(table_tab)
        tbar.pack(side="bottom", fill="x", pady=(PAD, 0))
        ttk.Button(tbar, text="Save table (CSV)...", command=self._save_table).pack(side="left")
        ttk.Button(tbar, text="Save crack boxes (JSON)...",
                   command=self._save_boxes).pack(side="left", padx=PAD)
        self.table_note = ttk.Label(tbar, style="Hint.TLabel", text="")
        self.table_note.pack(side="left", padx=PAD)
        tbox = ttk.Frame(table_tab)
        tbox.pack(fill="both", expand=True)
        self.table = ttk.Treeview(tbox, show="headings")
        vsb = ttk.Scrollbar(tbox, orient="vertical", command=self.table.yview)
        hsb = ttk.Scrollbar(tbox, orient="horizontal", command=self.table.xview)
        self.table.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self.table.pack(fill="both", expand=True)

        self.nb.add(self._build_ai_tab(self.nb), text="AI assistant")

        log_tab = ttk.Frame(self.nb, padding=(0, PAD, 0, 0))
        self.nb.add(log_tab, text="Log")
        self.log = tk.Text(log_tab, wrap="word", height=10, state="disabled",
                           font=("Consolas", 9), relief="flat", padx=6, pady=6)
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
        """Processing settings, folded into one summary line: the defaults are the tested
        pipeline, so most runs never need to open this."""
        outer = ttk.Frame(parent)
        head = ttk.Frame(outer)
        head.pack(fill="x")
        self.adv_btn = ttk.Button(head, text="\u25b8 Settings", width=12,
                                  command=self._toggle_settings)
        self.adv_btn.pack(side="left")
        self.settings_summary = ttk.Label(head, style="Hint.TLabel")
        self.settings_summary.pack(side="left", padx=PAD)

        self.v_filter = tk.StringVar(value=backend.DEFAULT_FILTER)
        self.v_kernel = tk.IntVar(value=3)
        self.v_margin = tk.StringVar(value="auto")
        self.v_detrend = tk.StringVar(value=backend.DEFAULT_DETREND)
        self.v_fixdir = tk.BooleanVar(value=True)
        self.v_zthresh = tk.DoubleVar(value=2.5)
        self.v_merge = tk.IntVar(value=2)
        self.v_minarea = tk.IntVar(value=6)

        self.settings_box = box = ttk.Frame(outer, padding=(0, PAD, 0, 0))
        pre = ttk.Frame(box)
        pre.pack(fill="x")
        ttk.Label(pre, text="Preprocessing", style="Section.TLabel", width=14).pack(side="left")
        self._combo(pre, "Filter", self.v_filter, backend.FILTERS, 9)
        self._spin(pre, "Kernel", self.v_kernel, 1, 15)
        self._entry(pre, "Border crop", self.v_margin, 6,
                    "'auto' measures the noisy border; or enter a pixel count")
        self._combo(pre, "Detrend", self.v_detrend, backend.DETRENDS, 8)
        ttk.Checkbutton(pre, text="Fix scan direction",
                        variable=self.v_fixdir).pack(side="left", padx=(12, 0))

        det = ttk.Frame(box)
        det.pack(fill="x", pady=(6, 0))
        ttk.Label(det, text="Detection", style="Section.TLabel", width=14).pack(side="left")
        self._spin(det, "|z| threshold", self.v_zthresh, 0.5, 10.0, inc=0.5, width=5)
        self._spin(det, "Merge gap", self.v_merge, 0, 10)
        self._spin(det, "Min area", self.v_minarea, 1, 200, width=5)
        ttk.Button(det, text="Defaults", command=self._reset_settings).pack(side="right")

        for v in (self.v_filter, self.v_kernel, self.v_margin, self.v_detrend, self.v_fixdir,
                  self.v_zthresh, self.v_merge, self.v_minarea):
            v.trace_add("write", lambda *_: self._update_summary())
        self._update_summary()
        return outer

    def _toggle_settings(self):
        if self.settings_box.winfo_ismapped():
            self.settings_box.pack_forget()
            self.adv_btn.config(text="\u25b8 Settings")
        else:
            self.settings_box.pack(fill="x")
            self.adv_btn.config(text="\u25be Settings")

    def _reset_settings(self):
        for var, val in ((self.v_filter, backend.DEFAULT_FILTER), (self.v_kernel, 3),
                         (self.v_margin, "auto"), (self.v_detrend, backend.DEFAULT_DETREND),
                         (self.v_fixdir, True), (self.v_zthresh, 2.5), (self.v_merge, 2),
                         (self.v_minarea, 6)):
            var.set(val)

    def _update_summary(self):
        try:
            parts = [f"filter {self.v_filter.get()} {self.v_kernel.get()}",
                     f"detrend {self.v_detrend.get()}", f"crop {self.v_margin.get()}",
                     "direction fix " + ("on" if self.v_fixdir.get() else "off"),
                     f"|z| > {self.v_zthresh.get():g}"]
        except tk.TclError:          # a spinbox is being edited and is momentarily empty
            return
        self.settings_summary.config(text="   \u00b7   ".join(parts))

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
        folder = filedialog.askdirectory(title="Choose the folder containing .tdms files",
                                         initialdir=backend.DATA_DIR)
        if not folder:
            return
        self.folder = Path(folder)
        self.load_files()

    def load_files(self):
        """List the scans grouped by measurement session (sub-folder)."""
        self.folder_label.config(text=str(self.folder))
        self.tree.delete(*self.tree.get_children())
        self.files = sorted(self.folder.rglob("*.tdms"))
        sessions = {}
        for p in self.files:
            rel = p.parent.relative_to(self.folder)
            key = str(rel) if str(rel) != "." else self.folder.name
            if key not in sessions:
                sessions[key] = self.tree.insert(
                    "", "end", iid=DIR_PREFIX + key, text=key, open=True, tags=("session",))
            info = backend.parse_name(p)
            self.tree.insert(sessions[key], "end", iid=str(p), text=p.stem,
                             values=(info.get("sensor", ""), info.get("amp", ""),
                                     info.get("freq", ""),
                                     len(crack_labels.load_labels(p)) or ""),
                             tags=() if info else ("template",))
        for key, iid in sessions.items():
            n = len(self.tree.get_children(iid))
            self.tree.item(iid, text=f"{key}   ({n})")
        self._log(f"Loaded {len(self.files)} .tdms files in {len(sessions)} session(s) "
                  f"from {self.folder}")
        self._status(f"{len(self.files)} files in {len(sessions)} sessions. "
                     "Grey = template files without scan data.")

    def _on_double_click(self, event):
        iid = self.tree.identify_row(event.y)
        if iid and not iid.startswith(DIR_PREFIX):
            self.run_report()

    def _selected_paths(self):
        return [Path(i) for i in self.tree.selection() if not i.startswith(DIR_PREFIX)]

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
            "fix_dir": self.v_fixdir.get(),
            "z_thresh": self.v_zthresh.get(), "merge": self.v_merge.get(),
            "min_area": self.v_minarea.get(),
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
                    try:
                        done(payload)
                    except Exception as e:
                        # an exception here must not escape: it would stop this loop from
                        # rescheduling and leave the app stuck on 'Working...' for good
                        self._log("ERROR: " + traceback.format_exc())
                        self._status("Something failed - see the Log tab.")
                        messagebox.showerror("Error", str(e))
                        continue
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
        self._gc_ticks += 1
        if self._gc_ticks >= 40:                 # every 2 s, on the main thread
            self._gc_ticks = 0
            gc.collect()
        self.after(50, self._drain_queue)

    def _prepare(self, path, prm):
        """Read + filter + crop. Shared by both single-file tasks."""
        grid, meta = backend.load_scan(path, fix_direction=prm["fix_dir"])
        filtered = backend.apply_filter(grid, prm["filter"], prm["kernel"])
        cropped, margins, crop_label = backend.resolve_crop(filtered, prm["margin"])
        return grid, filtered, cropped, meta, margins, crop_label

    def run_pair(self):
        """R and P of the same measurement side by side, one map type at a time."""
        path = self._one_selected()
        if not path:
            return
        partner = backend.pair_partner(path)
        if partner is None:
            messagebox.showinfo("No R/P pair", f"{path.name} has no matching R/P file "
                                "(same amp, freq, lift-off, size) in the same folder.")
            return
        r_path, p_path = ((path, partner) if backend.parse_name(path)["sensor"] == "R"
                          else (partner, path))
        prm = self._params()

        def work():
            grids, filts, metas = [], [], []
            for q in (r_path, p_path):
                g, m = backend.load_scan(q, fix_direction=prm["fix_dir"])
                grids.append(g)
                metas.append(m)
                filts.append(backend.apply_filter(g, prm["filter"], prm["kernel"]))
            if grids[0].shape != grids[1].shape:
                raise ValueError(f"R and P grids differ ({grids[0].shape} vs {grids[1].shape})"
                                 " - one file is truncated differently; compare them singly.")
            crops, margins, crop_label = backend.shared_crop(filts, prm["margin"])
            flabel = f"{prm['filter']}, k={prm['kernel']}"
            views_r = backend.report_views(grids[0], filts[0], crops[0], crop_label, flabel, "R")[0]
            views_p = backend.report_views(grids[1], filts[1], crops[1], crop_label, flabel, "P")[0]
            m = metas[0]
            title = (f"R + P pair - {r_path.parent.name}  amp={m.get('amp','?')}V  "
                     f"freq={m.get('freq','?')}kHz  lf={m.get('lf','?')}mm\n"
                     f"R: {r_path.name}    P: {p_path.name}")
            return views_r, views_p, title, metas, margins, crop_label, crops

        def done(res):
            views_r, views_p, title, metas, margins, crop_label, crops = res
            self._set_label_file(None, None)
            self._show_views(views_r, None, title, metas[0], default="amplitude",
                             pair_views=views_p)
            self.current_meta = metas[0]
            self.ai_context = ai_advisor.build_context(
                metas[0], prm, margins=margins, cropped_shape=crops[0].shape,
                stats={
                    "pair": f"R = {r_path.name}, P = {p_path.name}",
                    "std R (after filter + crop)": f"{np.nanstd(crops[0]):.6g}",
                    "std P (after filter + crop)": f"{np.nanstd(crops[1]):.6g}",
                })
            self.ai_context_kind = "scan"
            self._log(f"\n[R + P pair] R: {r_path.name}\n             P: {p_path.name}")
            self._log(f"  Shared crop : {crop_label} -> "
                      f"{crops[0].shape[1]}x{crops[0].shape[0]} px left (same for both)")
            if prm["fix_dir"]:
                self._log(f"  Scan direction: R shifted {metas[0]['row_shift_px']:+d} px, "
                          f"P shifted {metas[1]['row_shift_px']:+d} px")

        self._run_async(work, done)

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
            filter_label = f"{prm['filter']}, k={prm['kernel']}"
            views, _, _ = backend.report_views(grid, filtered, cropped, crop_label, filter_label,
                                               meta.get("sensor"), margins)
            overview = lambda: backend.build_report_figure(
                grid, filtered, cropped, meta, crop_label, filter_label, title)[0]
            return views, overview, title, path, meta, crop_label, cropped, grid, margins

        def done(res):
            views, overview, title, path, meta, crop_label, cropped, grid, margins = res
            self._set_label_file(path, grid.shape)
            self._show_views(views, overview, title, meta, default="amplitude")
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
            self._log(f"\n[Figures] {path.name}")
            self._log(f"  Original grid : {meta['size_x']}x{meta['size_y']} px "
                      f"(~{meta['pitch_mm']:.3f} mm/px)")
            if prm["fix_dir"]:
                self._log(f"  Scan direction: return lines shifted {meta['row_shift_px']:+d} px "
                          f"and level-matched (sawtooth fix)")
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
            # boxes touching the crop edge are the plate edge / scan border, not cracks:
            # they are always dropped (drawn dashed they only cluttered the map)
            n_edge = sum(b["touches_edge"] for b in boxes)
            boxes = [b for b in boxes if not b["touches_edge"]]
            for i, b in enumerate(boxes, start=1):
                b["id"] = f"crack_{i}"
            title = (f"Crack localization - {path.name}\n"
                     f"sensor={meta.get('sensor','?')}  "
                     f"freq={meta.get('freq','?')}kHz  crop: {crop_label}  |  "
                     f"threshold |z|>{prm['z_thresh']}  |  {len(boxes)} candidates")
            views = backend.detect_views(cropped, zmap, mask, boxes,
                                         prm["z_thresh"], prm["detrend"], margins)
            overview = lambda: backend.build_detect_figure(
                cropped, zmap, mask, boxes, meta, title, prm["z_thresh"], prm["detrend"])
            return (views, overview, title, boxes, n_edge, path, meta, margins, crop_label,
                    cropped, prm, grid.shape)

        def done(res):
            (views, overview, title, boxes, n_edge, path, meta, margins, crop_label,
             cropped, prm, grid_shape) = res
            self._set_label_file(path, grid_shape)
            self._show_views(views, overview, title, meta, default="boxes")
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
            self.table_note.config(text=f"{len(boxes)} crack candidates - {path.name}")

            self._log(f"\n[Detection] {path.name}")
            self._log(f"  Border crop : {crop_label} -> "
                      f"{cropped.shape[1]}x{cropped.shape[0]} px left")
            self._log(f"  Found {len(boxes)} candidates "
                      f"(threshold |z|>{prm['z_thresh']}, detrend {prm['detrend']})")
            if n_edge:
                self._log(f"  Dropped {n_edge} candidate(s) touching the image edge.")
            if not boxes:
                self._log("  No candidates. Lower the |z| threshold, or run "
                          "Compare sensors to check whether this sensor has any contrast.")
            elif len(boxes) > 60:
                self._log("  WARNING: too many candidates - the threshold is too low "
                          "or background noise remains.")
            self._log("  Coordinates are relative to the CROPPED image. Add "
                      f"+{margins['left']} (X) / +{margins['top']} (Y) for the original.")
            # stay on the Figures tab (the located cracks, full size); the box list is in
            # the Results table tab next to it

        self._run_async(work, done)

    def run_sweep(self):
        if not self.files:
            messagebox.showinfo("No files", "Choose a data folder first.")
            return
        prm = self._params()
        paths = list(self.files)
        n = len(paths)
        self.sweep_bar.config(maximum=n, value=0)
        self.sweep_bar.pack(side="right")
        self._log(f"\n[All data] Scanning {n} files...")

        def work():
            rows = []
            for i, p in enumerate(paths, start=1):
                rows.append(backend.analyze_one(
                    p, filter_method=prm["filter"], kernel=prm["kernel"],
                    margin=prm["margin"], detrend_method=prm["detrend"],
                    z_thresh=prm["z_thresh"], min_area=prm["min_area"],
                    merge_gap=prm["merge"], fix_direction=prm["fix_dir"]))
                self.msg_queue.put(("note", self._sweep_tick, (i, n)))
            return pd.DataFrame(rows)

        def done(df):
            self.sweep_bar.pack_forget()
            ok, skipped, top, best = backend.sweep_summary(df)
            self.sweep_df = df

            show = ["folder", "file", "sensor", "amp", "freq", "variant",
                    "n_inner", "n_strong", "n_edge_boxes", "z_max", "z_top5", "status"]
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
            self.table_note.config(text=f"{len(ok)} files ranked by z_top5 (best first)")

            if not ok.empty:
                self.ai_context = ai_advisor.build_sweep_context(
                    df, ok, skipped, best, prm, root=str(self.folder))
                self.ai_context_kind = "sweep"
            self.nb.select(1)

        self._run_async(work, done)

    # ------------------------------------------------------------ YOLO model

    def run_train(self):
        import crack_model
        items = crack_labels.labelled_files()
        if len(items) < 2:
            messagebox.showinfo(
                "Not enough labels",
                f"{len(items)} labelled scan(s). Draw crack boxes on a few scans first "
                "(Figures tab \u2192 LABEL \u2192 \u270e Draw boxes), then train.")
            return
        groups = {crack_model.group_key(s, hw) for s, _, hw in items}
        answer = messagebox.askyesnocancel(
            "Train crack model",
            f"{len(items)} labelled scans, {sum(len(b) for _, b, _ in items)} boxes, "
            f"{len(groups)} session(s).\n\n"
            "YES  →  train on ALL of them: the final model, for detecting. Nothing is "
            "held back, so its score is on training data.\n\n"
            "NO  →  hold out a whole session and train on the rest: an honest score "
            "on scans the model never saw.\n\n"
            f"Runs on the {crack_model.device_name()} in the background"
            + (" - 1-2 minutes." if crack_model.device() != "cpu" else " - several minutes.")
            + ("" if len(groups) > 1 else
               "\n\nOnly ONE session is labelled: a held-out score will be optimistic."))
        if answer is None:
            return
        use_all = bool(answer)
        self._log("\n[Model] Training on " + ("ALL labelled scans..." if use_all else
                                              "all but one held-out session..."))
        self.sweep_bar.config(maximum=1, value=0)
        self.sweep_bar.pack(side="right")

        def log(msg):
            self.msg_queue.put(("note", lambda m: self._log("  " + m), msg))

        def on_epoch(i, n):
            self.msg_queue.put(("note", self._train_tick, (i, n)))

        def done(metrics):
            self.sweep_bar.pack_forget()
            self._status("Model trained.")

        def failed(payload):
            self.sweep_bar.pack_forget()
            err, tb = payload
            self._log("[Model] ERROR: " + tb)
            self._status("Training failed - see the Log tab.")
            messagebox.showerror("Training failed", str(err))

        if not self._model_libs_ready():
            self.sweep_bar.pack_forget()
            return
        self._run_async(lambda: crack_model.train(log=log, on_epoch=on_epoch,
                                                  use_all=use_all), done,
                        on_error=failed)
        self.nb.select(3)

    def run_raw_dataset(self):
        """Every scan's RAW map, detected by the model, into its own dataset folder."""
        import crack_model
        import crack_dataset
        if not crack_model.WEIGHTS.exists():
            messagebox.showinfo("No model", "Train the crack model first.")
            return
        paths = list(self.files) or crack_dataset.scan_files()
        out = crack_dataset.OUT_DIR
        if not messagebox.askyesno(
                "Detect all → raw dataset",
                f"Run the model on the raw maps of all {len(paths)} files and write the "
                f"dataset to\n{out}\n\n(images, raw arrays, YOLO labels, annotated maps, "
                "detections.csv with crack length/width).\n\nAn older dataset there is "
                "replaced. About a minute."):
            return
        self.sweep_bar.config(maximum=len(paths), value=0)
        self.sweep_bar.pack(side="right")
        self._log(f"\n[Raw dataset] Detecting {len(paths)} files...")

        def log(msg):
            self.msg_queue.put(("note", lambda m: self._log("  " + m), msg))

        def progress(i, n):
            self.msg_queue.put(("note", self._sweep_tick, (i, n)))

        def done(res):
            files_df, det_df = res
            self.sweep_bar.pack_forget()
            ok = files_df[files_df["status"] == "ok"]
            self._fill_table(files_df[["session", "file", "cracks", "mean_conf", "found",
                                       "false", "missed", "status"]])
            self.table_note.config(text=f"{len(ok)} scans, {len(det_df)} cracks detected "
                                        f"- per crack: {out / 'detections.csv'}")
            self.nb.select(1)
            self._status("Raw dataset written.")
            if messagebox.askyesno("Raw dataset", f"{len(ok)} scans, {len(det_df)} cracks."
                                   f"\n\nOpen the folder?\n{out}"):
                os.startfile(out)

        if not self._model_libs_ready():
            self.sweep_bar.pack_forget()
            return
        self._run_async(lambda: crack_dataset.build(files=paths, log=log, progress=progress),
                        done)

    def _train_tick(self, payload):
        i, n = payload
        self.sweep_bar.config(maximum=n, value=i)
        self._status(f"Training epoch {i}/{n}...")

    def _model_libs_ready(self):
        """Import PyTorch / ultralytics on the MAIN thread before a worker uses them (same
        reason as backend.preload). Done once, on the first model action; a few seconds."""
        import crack_model
        if getattr(self, "_model_libs", False):
            return True
        self._status("Loading the model libraries (PyTorch) - first time only...")
        self.update_idletasks()
        try:
            crack_model._yolo()
            import torch  # noqa: F401
        except Exception as e:
            self._status("Model libraries missing.")
            messagebox.showerror("Model libraries", str(e))
            return False
        self._model_libs = True
        return True

    def run_model_detect(self):
        import crack_model
        path = self._one_selected()
        if not path:
            return
        if not crack_model.WEIGHTS.exists():
            messagebox.showinfo("No model", "Train the model first "
                                "(\u201cTrain crack model\u201d, after labelling some scans).")
            return
        if not self._model_libs_ready():
            return
        prm = self._params()

        def work():
            boxes, _ = crack_model.predict(path)
            grid, filtered, cropped, meta, margins, crop_label = self._prepare(path, prm)
            flat = backend.detrend(filtered, backend.DEFAULT_DETREND)
            labels = crack_labels.load_labels(path)
            missed = []
            if labels:
                # colour each box by whether it matches a label, and draw only the labels
                # the model missed: labels over every box made the map unreadable
                hit, free = crack_model.match_to_labels(boxes, labels)
                for b, h in zip(boxes, hit):
                    b["match"] = h
                missed = [labels[i] for i in free]
            # the model's boxes on EVERY map: the 6 standard views (raw ... -dA/dx) and
            # the model's own input. Boxes are in original grid pixels; each view is
            # shifted by its crop offset and keeps only the boxes that fall on it.
            views, _, _ = backend.report_views(
                grid, filtered, cropped, crop_label, f"{prm['filter']}, k={prm['kernel']}",
                meta.get("sensor"), margins)
            views.append({"key": "model", "title": "7. Model input (detrended signal)",
                          "img": flat, "cmap": "magma", "cbar": "detrended signal",
                          "offset": (0, 0)})
            for v in views:
                v["boxes"] = backend.shift_boxes(boxes, v["offset"], v["img"].shape)
                v["missed"] = backend.shift_boxes(missed, v["offset"], v["img"].shape)
            title = f"Model detection - {path.name}\n{len(boxes)} boxes"
            if labels:
                title += "   (green found \u00b7 red false \u00b7 pink dotted missed)"
            return views, title, meta, boxes, grid.shape

        def done(res):
            views, title, meta, boxes, shape = res
            self._set_label_file(path, shape)
            self._show_views(views, None, title, meta, default="amplitude")
            self._log(f"\n[Model] {path.name}: {len(boxes)} boxes "
                      + ", ".join(f"{b['conf']:.2f}" for b in boxes))
            self.score_text = ""
            if self.labels:
                tp, fp, fn = crack_model.score_against_labels(boxes, self.labels)
                note = ("   (this scan was in the TRAINING set - not a fair test)"
                        if crack_model.in_training_set(path) else "")
                self._log(f"  vs your {len(self.labels)} labels (IoU \u2265 0.3): "
                          f"{tp} found, {fp} false, {fn} missed{note}")
                self.score_text = (f"Model vs labels: {tp} found \u00b7 {fp} false "
                                   f"\u00b7 {fn} missed{note}")
            self.label_stats.config(text=(self.score_text + "   \u2014   " if self.score_text
                                          else "") + "click a box to measure the crack")

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
                "Run \u201cDraw figures\u201d or \u201cLocate cracks\u201d first, "
                "then come back here.")
            return
        ctx = self.ai_context
        kind = self.ai_context_kind
        # A whole-dataset question carries no image: the context is a hundred-row table.
        png = None
        if kind != "sweep" and self.ai_fig_fn is not None:
            fig = self.ai_fig_fn()
            png = ai_advisor.figure_to_png_bytes(fig)
            plt.close(fig)
        first = not self.ai_history

        if kind == "sweep":
            q = ("Analyse this entire measurement sweep. Specifically: (1) which sensor and "
                 "frequency range give the best contrast, and is there a clear trend with "
                 "frequency or with excitation amplitude; (2) are the results consistent "
                 "across the folders (separate measurement sessions), and where do they "
                 "disagree; (3) which measurements look broken or anomalous and why; "
                 "(4) which configurations should be measured again or added.")
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
            if self.ai_context_kind != "sweep" and self.ai_fig_fn is not None:
                fig = self.ai_fig_fn()
                png = ai_advisor.figure_to_png_bytes(fig)
                plt.close(fig)
        self._ai_send(question=q, image_png=png, context_text=ctx,
                      header=f"You: {q}")

    # ------------------------------------------------------------ display

    def _show_views(self, views, overview, title, meta, default, pair_views=None):
        """
        Show the maps of one run ONE AT A TIME, full size, with a bar to pick which
        (Left/Right arrows step through). `overview` - all maps in one figure - is only
        sent to the AI assistant, never shown: six small maps were too hard to read.
        """
        self.views, self.view_overview = views, overview
        self.view_title, self.view_meta = title, meta
        self.pair_views = pair_views
        self.score_text = ""
        self.ai_fig_fn = overview or (lambda: self._figure_for("amplitude"))
        for w in self.plot_tab.winfo_children():
            w.destroy()

        top = ttk.Frame(self.plot_tab)
        top.pack(side="top", fill="x")
        for v in views:
            ttk.Radiobutton(top, text=v["title"].split(" (")[0], value=v["key"],
                            variable=self.v_view, style="Toolbutton",
                            command=self._draw_current_view).pack(side="left", padx=(0, 2))
        ttk.Label(top, text="\u2190 \u2192 to switch", style="Hint.TLabel").pack(side="right")

        tools = ttk.Frame(self.plot_tab, padding=(0, 6))
        tools.pack(side="top", fill="x")
        prof = ttk.Frame(tools)
        prof.pack(side="left")
        ttk.Label(prof, text="PROFILE", style="Section.TLabel").pack(side="left", padx=(0, 6))
        ttk.Combobox(prof, state="readonly", width=12, textvariable=self.v_prof_label,
                     values=list(self.PROFILE_MODES)).pack(side="left")
        ttk.Label(prof, text="=").pack(side="left", padx=4)
        e = ttk.Entry(prof, textvariable=self.v_prof_pos, width=6)
        e.pack(side="left")
        e.bind("<Return>", lambda _: self._draw_profile())
        ttk.Label(prof, text="mm").pack(side="left", padx=(3, 6))
        ttk.Button(prof, text="Draw", command=self._draw_profile).pack(side="left")

        ttk.Separator(tools, orient="vertical").pack(side="left", fill="y", padx=14)
        pt = ttk.Frame(tools)
        pt.pack(side="left")
        ttk.Label(pt, text="POINT", style="Section.TLabel").pack(side="left", padx=(0, 6))
        for label, var in ((backend.X_LABEL, self.v_px), (backend.Y_LABEL, self.v_py)):
            ttk.Label(pt, text=label).pack(side="left", padx=(4, 3))
            e = ttk.Entry(pt, textvariable=var, width=6)
            e.pack(side="left")
            e.bind("<Return>", lambda _: self._lookup_point())
        ttk.Label(pt, text="mm").pack(side="left", padx=(3, 6))
        ttk.Button(pt, text="Get value", command=self._lookup_point).pack(side="left")
        self.point_result = ttk.Label(tools, text="", style="Result.TLabel")
        self.point_result.pack(side="left", padx=12)

        lab = ttk.Frame(self.plot_tab, padding=(0, 0, 0, 6))
        lab.pack(side="top", fill="x")
        ttk.Label(lab, text="LABEL", style="Section.TLabel").pack(side="left", padx=(0, 6))
        ttk.Checkbutton(lab, text="\u270e Draw boxes", style="Toolbutton",
                        variable=self.v_label_mode,
                        command=self._on_label_mode).pack(side="left")
        for text, cmd in (("Auto-fill", self._label_autofill),
                          ("Copy to session", self._label_copy),
                          ("Delete", self._label_delete),
                          ("Clear", self._label_clear)):
            ttk.Button(lab, text=text, command=cmd).pack(side="left", padx=(4, 0))
        self.label_info = ttk.Label(lab, style="Hint.TLabel")
        self.label_info.pack(side="left", padx=10)
        self.label_stats = ttk.Label(self.plot_tab, style="Result.TLabel")
        self.label_stats.pack(side="top", anchor="w", pady=(0, 4))

        ttk.Separator(self.plot_tab).pack(fill="x")
        self.fig_holder = ttk.Frame(self.plot_tab)
        self.fig_holder.pack(fill="both", expand=True)

        if self.v_view.get() not in self._view_keys():
            self.v_view.set(default)       # keep the user's choice across runs when it exists
        self._draw_current_view()
        self.nb.select(0)

    def _view_keys(self):
        return [v["key"] for v in self.views]

    def _pair_of(self, view):
        return next(v for v in self.pair_views if v["key"] == view["key"])

    def _figure_for(self, key, point=None):
        view = next(v for v in self.views if v["key"] == key)
        if self.pair_views:
            return backend.build_pair_figure(view, self._pair_of(view), self.view_meta,
                                             self.view_title, point=point)
        return backend.build_view_figure(view, self.view_meta, self.view_title, point=point)

    def _draw_current_view(self):
        self.point_result.config(text="")
        self.label_stats.config(text=self.score_text)
        self._embed_fig(self._figure_for(self.v_view.get()))
        if (self.v_label_mode.get() and self.sel_label is not None
                and self._label_view() is not None):
            self._show_label_stats()

    def _single_view(self):
        """The view on screen, or None when nothing has been drawn yet."""
        if not self.views:
            return None
        return next(v for v in self.views if v["key"] == self.v_view.get())

    def _read_mm(self, var, name):
        try:
            return float(var.get().replace(",", "."))
        except ValueError:
            messagebox.showerror("Not a number", f"{name}: enter a position in mm, e.g. 35")
            return None

    def _draw_profile(self):
        view = self._single_view()
        if view is None:
            return
        along = self.PROFILE_MODES[self.v_prof_label.get()]
        pos = self._read_mm(self.v_prof_pos,
                            backend.Y_LABEL if along == "x" else backend.X_LABEL)
        if pos is None:
            return
        try:
            if self.pair_views:
                fig, curves = backend.build_pair_profile_figure(
                    view, self._pair_of(view), self.view_meta, self.view_title, along, pos)
            else:
                fig, xs, vals = backend.build_profile_figure(view, self.view_meta,
                                                             self.view_title, along, pos)
                curves = [("", xs, vals)]
        except ValueError as e:
            messagebox.showerror("Outside the map", str(e))
            return
        self._embed_fig(fig)
        axis = backend.X_LABEL if along == "x" else backend.Y_LABEL
        self._log(f"\n[Profile] {view['title']} - along {axis}, cut at {pos:g} mm")
        for sensor, xs, vals in curves:
            self._log(f"  {sensor or ' '} {xs.size} points, min {np.nanmin(vals):.4g}  "
                      f"max {np.nanmax(vals):.4g} (at {xs[int(np.nanargmax(vals))]:.1f} mm)")

    def _lookup_point(self):
        view = self._single_view()
        if view is None:
            return
        x = self._read_mm(self.v_px, backend.X_LABEL)
        y = self._read_mm(self.v_py, backend.Y_LABEL)
        if x is None or y is None:
            return
        try:
            val, col, row, xc, yc = backend.value_at(view, self.view_meta, x, y)
            val_p = (backend.value_at(self._pair_of(view), self.view_meta, x, y)[0]
                     if self.pair_views else None)
        except ValueError as e:
            messagebox.showerror("Outside the map", str(e))
            return
        self._embed_fig(self._figure_for(view["key"], point=(xc, yc)))
        unit = view.get("cbar", "")
        if val_p is None:
            text = f"= {val:.6g} {unit}"
        else:
            text = f"R = {val:.6g}   P = {val_p:.6g}"   # R amplitude, P phase: no shared unit
        self.point_result.config(text=f"{text}   (pixel col {col}, row {row})")
        self._log(f"[Point] {view['title']} at {backend.X_LABEL}={x:g}, "
                  f"{backend.Y_LABEL}={y:g} mm -> {text} (pixel col {col}, row {row})")

    def _step_view(self, step):
        if not self.views or self.nb.index("current") != 0:
            return
        keys = self._view_keys()
        cur = self.v_view.get()
        i = keys.index(cur) if cur in keys else 0
        self.v_view.set(keys[(i + step) % len(keys)])
        self._draw_current_view()

    def _embed_fig(self, fig):
        for w in self.fig_holder.winfo_children():
            w.destroy()
        if self.current_fig is not None:
            plt.close(self.current_fig)
        self.current_fig = fig
        canvas = FigureCanvasTkAgg(fig, master=self.fig_holder)
        self._canvas = canvas
        self._attach_labels(fig)
        canvas.draw()
        NavigationToolbar2Tk(canvas, self.fig_holder).update()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # ------------------------------------------------------------ hand labelling

    def _set_label_file(self, path, grid_shape):
        self.label_file, self.label_grid = path, grid_shape
        self.labels = crack_labels.load_labels(path) if path else []
        self.sel_label = None
        if path is None:
            self.v_label_mode.set(False)

    def _label_view(self):
        """(view, ax) when the screen shows ONE map of ONE labelled-able file, else None."""
        if self.label_file is None or self.pair_views or not self.views:
            return None
        fig = self.current_fig
        axes = [a for a in fig.axes if a.get_label() != "<colorbar>"] if fig else []
        if len(axes) != 1:           # e.g. a profile figure (map + curve)
            return None
        return self._single_view(), axes[0]

    def _map_view(self):
        """(view, ax) when the screen shows ONE map (any file, not R + P), else None."""
        if self.pair_views or not self.views or self.current_fig is None:
            return None
        axes = [a for a in self.current_fig.axes if a.get_label() != "<colorbar>"]
        return (self._single_view(), axes[0]) if len(axes) == 1 else None

    def _attach_labels(self, fig):
        self._selector = None
        self._label_patches = []
        self._measure_marks = []
        if not self.v_label_mode.get():
            # viewing: labels stay hidden (drawn over the detections they made the map
            # unreadable); a click on a detected box measures the crack inside it
            if self._map_view() is not None:
                fig.canvas.mpl_connect("button_press_event", self._on_measure_click)
            if self.label_file is not None and not self.pair_views:
                n = len(self.labels)
                self.label_info.config(
                    text=f"{n} label{'s' if n != 1 else ''} (hidden - turn on "
                         "\u270e Draw boxes to see and edit them)" if n else "")
            return
        lv = self._label_view()
        if lv is None:
            if self.pair_views:
                self.label_info.config(text="labelling works on one file - use Draw figures")
            return
        view, ax = lv
        self._draw_label_patches(ax, view)
        n = len(self.labels)
        self.label_info.config(
            text=f"{n} label{'s' if n != 1 else ''} on {self.label_file.stem}"
                 "   \u2014 drag to add, click to select, right-click to delete")
        self._selector = RectangleSelector(
            ax, self._on_box_drawn, useblit=True, button=[1], minspanx=0, minspany=0,
            spancoords="data", interactive=False,
            props=dict(edgecolor="#ff4fd8", facecolor="#ff4fd8", alpha=0.15, fill=True))
        fig.canvas.mpl_connect("button_press_event", self._on_label_click)

    def _to_view_px(self, view, b):
        ox, oy = view.get("offset", (0, 0))
        return {"x0": b["x0"] - ox, "y0": b["y0"] - oy, "x1": b["x1"] - ox, "y1": b["y1"] - oy}

    def _draw_label_patches(self, ax, view):
        for p in self._label_patches + self._measure_marks:
            p.remove()
        self._label_patches, self._measure_marks = [], []
        pitch = self.view_meta["pitch_mm"]
        for i, b in enumerate(self.labels):
            v = self._to_view_px(view, b)
            sel = i == self.sel_label
            r = mpatches.Rectangle((v["x0"] * pitch, v["y0"] * pitch),
                                   (v["x1"] - v["x0"]) * pitch, (v["y1"] - v["y0"]) * pitch,
                                   fill=False, edgecolor="#ff4fd8",
                                   linewidth=1.5 if sel else 0.7, linestyle="-")
            ax.add_patch(r)
            t = ax.text(v["x0"] * pitch, v["y1"] * pitch, f"L{i + 1}", color="#ff4fd8",
                        fontsize=7, va="top")
            self._label_patches += [r, t]
        if self.sel_label is not None and self.sel_label < len(self.labels):
            b = self._to_view_px(view, self.labels[self.sel_label])
            self._mark_measure(ax, view, b)

    # ------------------------------------------------------------ crack measurement

    def _measure(self, view, b):
        """Measure the crack in box `b` (view pixels) on the map on screen -> dict or None."""
        H, W = view["img"].shape
        b = {"x0": max(0, b["x0"]), "y0": max(0, b["y0"]),
             "x1": min(W, b["x1"]), "y1": min(H, b["y1"])}
        if b["x1"] - b["x0"] < 1 or b["y1"] - b["y0"] < 1:
            return None
        return crack_labels.measure_crack(view["img"], b, self.view_meta["pitch_mm"])

    def _mark_measure(self, ax, view, b):
        """Draw the measured length (solid) and width (short) axes of the crack in `b`."""
        for m in self._measure_marks:
            m.remove()
        self._measure_marks = []
        m = self._measure(view, b)
        if m is None:
            return None
        color = "#00e5ff" if m["reliable"] else "#ffb347"
        for (p, q), lw in ((m["length_ends"], 1.4), (m["width_ends"], 1.0)):
            ln, = ax.plot([p[0], q[0]], [p[1], q[1]], color=color, linewidth=lw,
                          solid_capstyle="butt")
            self._measure_marks.append(ln)
        return m

    @staticmethod
    def _measure_text(m):
        ge = "≥ " if m["spills"] else ""
        return (f"crack length {ge}{m['length_mm']:.1f} mm · width {m['width_mm']:.1f} mm "
                f"(along {m['along']}, {m['angle_deg']:.0f}°) · "
                f"area {m['area_mm2']:.1f} mm² · "
                f"centre {backend.X_LABEL} {m['centre_mm'][0]:.1f}, "
                f"{backend.Y_LABEL} {m['centre_mm'][1]:.1f} mm · peak {m['peak']:.3g} "
                f"(SNR {m['snr']:.1f})"
                + ("" if m["reliable"] else
                   "\n⚠ weak or unclear signature - the size is only a rough estimate")
                + ("\n↗ the signature runs past the box - the crack is at least this long"
                   if m["spills"] and m["reliable"] else ""))

    def _on_measure_click(self, event):
        mv = self._map_view()
        if (mv is None or event.button != 1 or event.xdata is None
                or event.inaxes is not mv[1] or self.v_label_mode.get()):
            return
        view, ax = mv
        if ax.get_navigate_mode():                  # zoom / pan tool active
            return
        pitch = self.view_meta["pitch_mm"]
        cx, cy = event.xdata / pitch, event.ydata / pitch
        hits = [b for b in view.get("boxes", ())
                if b["x0"] <= cx <= b["x1"] and b["y0"] <= cy <= b["y1"]]
        if not hits:
            return
        b = min(hits, key=lambda b: (b["x1"] - b["x0"]) * (b["y1"] - b["y0"]))
        m = self._mark_measure(ax, view, b)
        if m is None:
            return
        sel = ax.add_patch(mpatches.Rectangle(
            (b["x0"] * pitch, b["y0"] * pitch), (b["x1"] - b["x0"]) * pitch,
            (b["y1"] - b["y0"]) * pitch, fill=False, edgecolor="white", linewidth=1.2))
        self._measure_marks.append(sel)
        self._canvas.draw_idle()
        conf = f"conf {b['conf']:.2f} · " if "conf" in b else ""
        text = conf + self._measure_text(m)
        self.label_stats.config(text=text)
        self._log(f"[Crack] {view['title'].split(' (')[0]}: {text}")

    def _refresh_labels(self):
        lv = self._label_view()
        if lv is None:
            return
        view, ax = lv
        self._draw_label_patches(ax, view)
        self._canvas.draw_idle()
        crack_labels.save_labels(self.label_file, self.labels, self.label_grid)
        n = len(self.labels)
        if self.tree.exists(str(self.label_file)):
            self.tree.set(str(self.label_file), "labels", n or "")
        self.label_info.config(
            text=f"{n} label{'s' if n != 1 else ''} on {self.label_file.stem}   "
                 "\u2014 drag to add, click to select, right-click to delete")
        if self.sel_label is None:
            self.label_stats.config(text="")

    def _on_label_mode(self):
        if self.v_label_mode.get() and self._label_view() is None:
            messagebox.showinfo(
                "Labelling", "Open one file with \u201cDraw figures\u201d or "
                "\u201cLocate cracks\u201d and pick a map first (not R + P, not a profile).")
            self.v_label_mode.set(False)
            return
        self._draw_current_view()

    def _on_box_drawn(self, press, release):
        view, _ = self._label_view()
        pitch = self.view_meta["pitch_mm"]
        H, W = view["img"].shape
        xa, xb = sorted((press.xdata, release.xdata))
        ya, yb = sorted((press.ydata, release.ydata))
        x0, x1 = int(np.floor(xa / pitch)), int(np.ceil(xb / pitch))
        y0, y1 = int(np.floor(ya / pitch)), int(np.ceil(yb / pitch))
        x0, x1 = max(0, x0), min(W, x1)
        y0, y1 = max(0, y0), min(H, y1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return                     # a click, not a box: handled by _on_label_click
        ox, oy = view.get("offset", (0, 0))
        self.labels.append({"x0": x0 + ox, "y0": y0 + oy, "x1": x1 + ox, "y1": y1 + oy})
        self.sel_label = len(self.labels) - 1
        self._refresh_labels()
        self._show_label_stats()

    def _label_at(self, event):
        view, _ = self._label_view()
        pitch = self.view_meta["pitch_mm"]
        ox, oy = view.get("offset", (0, 0))
        cx, cy = event.xdata / pitch + ox, event.ydata / pitch + oy
        hits = [i for i, b in enumerate(self.labels)
                if b["x0"] <= cx <= b["x1"] and b["y0"] <= cy <= b["y1"]]
        # the smallest box under the cursor, so a box inside another stays reachable
        return min(hits, key=lambda i: (self.labels[i]["x1"] - self.labels[i]["x0"]) *
                   (self.labels[i]["y1"] - self.labels[i]["y0"])) if hits else None

    def _on_label_click(self, event):
        if event.inaxes is None or self._label_view() is None or event.xdata is None:
            return
        if event.inaxes is not self._label_view()[1]:
            return
        i = self._label_at(event)
        if event.button == 3:          # right click: delete the box under the cursor
            if i is not None:
                self.sel_label = i
                self._label_delete()
            return
        if event.button == 1 and i is not None:
            self.sel_label = i
            self._refresh_labels()
            self._show_label_stats()

    def _show_label_stats(self):
        if self.sel_label is None:
            return
        view, _ = self._label_view()
        b = self._to_view_px(view, self.labels[self.sel_label])
        H, W = view["img"].shape
        if b["x1"] <= 0 or b["y1"] <= 0 or b["x0"] >= W or b["y0"] >= H:
            self.label_stats.config(text=f"L{self.sel_label + 1}: outside this cropped map "
                                         "- switch to Raw or Filtered to see it")
            return
        b = {"x0": max(0, b["x0"]), "y0": max(0, b["y0"]),
             "x1": min(W, b["x1"]), "y1": min(H, b["y1"])}
        st = crack_labels.region_stats(view["img"], b, self.view_meta["pitch_mm"])
        cnr = "n/a" if not np.isfinite(st["CNR"]) else f"{st['CNR']:.2f}"
        cnr_abs = "n/a" if not np.isfinite(st["CNR_abs"]) else f"{st['CNR_abs']:.2f}"
        text = (f"L{self.sel_label + 1}  on {view['title'].split(' (')[0]}:  "
                f"{st['w_mm']:.1f} \u00d7 {st['h_mm']:.1f} mm   \u00b7   "
                f"min {st['min']:.4g}  max {st['max']:.4g}   \u00b7   "
                f"CNR {cnr}   CNR_abs {cnr_abs}"
                + ("   \u00b7   dipole signature" if st["dipole"] else ""))
        m = self._measure(view, b)
        if m is not None:
            text += "\n" + self._measure_text(m)
        self.label_stats.config(text=text)
        self._log("[Label] " + text)

    def _label_delete(self):
        # labels are hidden outside Draw-boxes mode: never delete one the user cannot see
        if (not self.v_label_mode.get() or self.sel_label is None
                or self._label_view() is None):
            return
        del self.labels[self.sel_label]
        self.sel_label = None
        self._refresh_labels()

    def _label_clear(self):
        if not self.labels or self._label_view() is None:
            return
        if messagebox.askyesno("Clear labels",
                               f"Delete all {len(self.labels)} labels of {self.label_file.name}?"):
            self.labels, self.sel_label = [], None
            self._refresh_labels()

    def _label_autofill(self):
        """Start from the threshold detector's interior boxes - to CORRECT, not to trust."""
        if self._label_view() is None:
            messagebox.showinfo("Labelling", "Open one file and one map first.")
            return
        prm = self._params()
        _, _, cropped, _, margins, _ = self._prepare(self.label_file, prm)
        boxes, _, _ = backend.detect_cracks(cropped, z_thresh=prm["z_thresh"],
                                            min_area=prm["min_area"], merge_gap=prm["merge"],
                                            detrend_method=prm["detrend"])
        new = [{"x0": b["x0"] + margins["left"], "y0": b["y0"] + margins["top"],
                "x1": b["x1"] + margins["left"], "y1": b["y1"] + margins["top"]}
               for b in boxes if not b["touches_edge"]]
        # skip boxes already covered by a label: pressing Auto-fill twice doubled every box
        import crack_model
        new = [b for b in new
               if not any(crack_model._iou(b, old) >= 0.3 for old in self.labels)]
        if not new:
            self._log(f"[Label] Auto-fill: every detector box on {self.label_file.name} "
                      "already has a label - nothing added.")
            return
        self.labels += new
        self.sel_label = None
        self._refresh_labels()
        self._log(f"[Label] Auto-fill added {len(new)} boxes from the threshold detector to "
                  f"{self.label_file.name}. Check each one: delete false ones, add missed "
                  "cracks, fix sizes - unedited they only teach a model to copy the detector.")

    def _label_copy(self):
        """Copy these boxes to the scans of the same session with the same grid: same plate
        position, so the same cracks (every frequency and amplitude, both sensors)."""
        if not self.labels or self._label_view() is None:
            messagebox.showinfo("Copy labels", "Draw some labels first.")
            return
        targets = crack_labels.same_geometry_files(self.label_file, self.label_grid)
        free = [q for q in targets if not crack_labels.load_labels(q)]
        if not free:
            messagebox.showinfo("Copy labels", "No unlabelled scan with the same session and "
                                "grid size.")
            return
        if not messagebox.askyesno(
                "Copy labels",
                f"Copy these {len(self.labels)} boxes to {len(free)} unlabelled scan(s) of "
                f"{self.label_file.parent.name} with the same {self.label_grid[1]}x"
                f"{self.label_grid[0]} grid?\n\n"
                f"({len(targets) - len(free)} already labelled scan(s) are left untouched.)\n"
                "Only valid if the plate sat in the same place for every scan - check a few "
                "of them afterwards."):
            return
        for q in free:
            crack_labels.save_labels(q, self.labels, self.label_grid)
            if self.tree.exists(str(q)):
                self.tree.set(str(q), "labels", len(self.labels))
        self._log(f"[Label] Copied {len(self.labels)} boxes to {len(free)} scans: "
                  + ", ".join(q.stem for q in free))

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

    def _save_table(self):
        df = getattr(self, "_table_df", None)
        if df is None or df.empty:
            messagebox.showinfo("No table", "Run \u201cLocate cracks\u201d or "
                                "\u201cAnalyze & rank all files\u201d first.")
            return
        p = filedialog.asksaveasfilename(initialdir=backend.RESULTS_DIR, defaultextension=".csv",
                                         filetypes=[("CSV", "*.csv")])
        if p:
            df.to_csv(p, index=False, encoding="utf-8-sig")
            self._log(f"Table saved: {p}")

    def _save_boxes(self):
        if not self.current_boxes:
            messagebox.showinfo("No boxes", "Run Locate cracks first.")
            return
        p = filedialog.asksaveasfilename(initialdir=backend.RESULTS_DIR, defaultextension=".json",
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
