"""
IQ_beta/iq_dialog.py

"Open I/Q File (Beta)" dialog: pick a Vaisala I/Q file, choose which
moments to render (with custom output names), set the azimuth resolution,
optionally enable GRF ground-clutter filtering, then Render.

Deliberately doesn't import anything from gurt_gui.py — the color
constants below just mirror its DLG_* palette so this dialog matches the
rest of the app, without creating a dependency in that direction. Rendered
volumes go back into the running app via install_iq_volume().
"""

import os
import threading

import tkinter as tk
from tkinter import filedialog, messagebox

from .products import PRODUCT_GROUPS
from .iq_engine import load_iq_volume, is_koun_file, IQLoadError

# Mirrors gurt_gui.py's DLG_* theme constants.
DLG_BG    = "#e6e6e6"
DLG_FG    = "black"
DLG_ENTRY = "#ffffff"
DLG_ABG   = "#cfcfcf"
DLG_HINT  = "#205080"
DLG_HINT2 = "#555555"
WARN_BG   = "#fff3cd"
WARN_FG   = "#7a5b00"


class IQBetaDialog(tk.Toplevel):
    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self._path = None
        self._row_vars = {}   # product key -> (BooleanVar checked, StringVar name)

        self.title("Open I/Q File (Beta)")
        self.configure(bg=DLG_BG)
        self.resizable(False, True)
        self.grab_set()
        self.transient(app)

        hdr = tk.Frame(self, bg="#0d0d3a")
        hdr.pack(fill="x")
        tk.Label(hdr, text="Open I/Q File (Beta)",
                 bg="#0d0d3a", fg="white",
                 font=("TkDefaultFont", 11, "bold"), pady=7).pack()

        body = tk.Frame(self, bg=DLG_BG)
        body.pack(fill="both", expand=True, padx=12, pady=8)
        body.columnconfigure(0, weight=1)

        # ── Disclaimer ──────────────────────────────────────────────────
        tk.Label(
            body, justify="left", anchor="w", wraplength=460,
            bg=WARN_BG, fg=WARN_FG, font=("TkDefaultFont", 8),
            text=("Beta feature. Only Vaisala rvp8/rvpts I/Q files "
                  "(vaisalaIQ) are supported right now \u2014 RadarKit "
                  "(.rkc) I/Q input isn't wired up yet. KOUN I/Q data may "
                  "load, but its moments are not expected to come out "
                  "correct.")
        ).grid(row=0, column=0, columnspan=2, sticky="ew", padx=4, pady=(0, 8))

        # ── File picker ─────────────────────────────────────────────────
        tk.Label(body, text="I/Q file:", bg=DLG_BG, fg=DLG_FG,
                 font=("TkDefaultFont", 9, "bold")
                 ).grid(row=1, column=0, sticky="w", padx=4)

        file_row = tk.Frame(body, bg=DLG_BG)
        file_row.grid(row=2, column=0, columnspan=2, sticky="ew", padx=4, pady=(0, 8))

        self._file_var = tk.StringVar(value="(no file selected)")
        tk.Entry(file_row, textvariable=self._file_var, width=46,
                 state="readonly", bg=DLG_ENTRY, relief="sunken"
                 ).pack(side="left", fill="x", expand=True)
        tk.Button(file_row, text="Browse…", command=self._browse,
                  bg=DLG_BG, fg=DLG_FG, relief="raised",
                  activebackground=DLG_ABG, activeforeground=DLG_FG,
                  padx=6).pack(side="left", padx=(6, 0))

        # ── Resolution + clutter filter ─────────────────────────────────
        opts_row = tk.Frame(body, bg=DLG_BG)
        opts_row.grid(row=3, column=0, columnspan=2, sticky="ew", padx=4, pady=(0, 8))

        tk.Label(opts_row, text="Resolution (deg):", bg=DLG_BG, fg=DLG_FG,
                 font=("TkDefaultFont", 9)).pack(side="left")
        self._az_var = tk.StringVar(value="1.0")
        tk.Entry(opts_row, textvariable=self._az_var, width=6,
                 bg=DLG_ENTRY, relief="sunken").pack(side="left", padx=(4, 16))

        self._clutter_var = tk.BooleanVar(value=False)
        tk.Checkbutton(opts_row, text="Ground clutter filter (GRF)",
                       variable=self._clutter_var,
                       bg=DLG_BG, fg=DLG_FG, selectcolor=DLG_ABG,
                       activebackground=DLG_BG, activeforeground=DLG_FG,
                       font=("TkDefaultFont", 9)).pack(side="left")

        # ── Product picker ──────────────────────────────────────────────
        tk.Label(body, text="Products to render (output field name is editable):",
                 bg=DLG_BG, fg=DLG_FG, font=("TkDefaultFont", 9, "bold"),
                 anchor="w").grid(row=4, column=0, columnspan=2, sticky="w", padx=4)

        list_frame = tk.Frame(body, bg=DLG_BG, relief="sunken", bd=1)
        list_frame.grid(row=5, column=0, columnspan=2, sticky="nsew", padx=4, pady=(2, 8))
        body.rowconfigure(5, weight=1)

        canvas = tk.Canvas(list_frame, bg=DLG_BG, highlightthickness=0,
                           width=460, height=260)
        vsb = tk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=DLG_BG)
        inner.bind("<Configure>",
                  lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        self._build_product_rows(inner)

        # ── Status line ──────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="")
        tk.Label(body, textvariable=self._status_var, bg=DLG_BG, fg=DLG_HINT,
                 font=("TkDefaultFont", 8), anchor="w", wraplength=460, justify="left"
                 ).grid(row=6, column=0, columnspan=2, sticky="w", padx=4)

        # ── Buttons ──────────────────────────────────────────────────────
        btn_row = tk.Frame(body, bg=DLG_BG)
        btn_row.grid(row=7, column=0, columnspan=2, pady=(8, 2))
        self._render_btn = tk.Button(
            btn_row, text="Render", command=self._on_render,
            bg=DLG_BG, fg=DLG_FG, relief="raised",
            activebackground=DLG_ABG, activeforeground=DLG_FG,
            font=("TkDefaultFont", 9, "bold"), padx=18, pady=4, state="disabled")
        self._render_btn.pack(side="left", padx=6)
        tk.Button(btn_row, text="Cancel", command=self.destroy,
                  bg=DLG_BG, fg=DLG_FG, relief="raised",
                  activebackground=DLG_ABG, activeforeground=DLG_FG,
                  padx=14, pady=4).pack(side="left", padx=6)

        self.update_idletasks()
        pw = app.winfo_width();  ph = app.winfo_height()
        px = app.winfo_rootx(); py = app.winfo_rooty()
        dw = self.winfo_reqwidth(); dh = self.winfo_reqheight()
        self.geometry(f"+{px + max(0, (pw - dw)//2)}+{py + max(0, (ph - dh)//2)}")

    # ── Product rows ────────────────────────────────────────────────────

    def _build_product_rows(self, parent):
        row = 0
        for group_label, specs in PRODUCT_GROUPS:
            tk.Label(parent, text=group_label, bg=DLG_BG, fg=DLG_HINT2,
                     font=("TkDefaultFont", 8, "bold"), anchor="w"
                     ).grid(row=row, column=0, columnspan=4, sticky="w",
                            padx=6, pady=(8, 1))
            row += 1
            for spec in specs:
                chk_var = tk.BooleanVar(value=spec['default_checked'])
                name_var = tk.StringVar(value=spec['default_name'])

                tk.Checkbutton(parent, text=spec['label'], variable=chk_var,
                              bg=DLG_BG, fg=DLG_FG, selectcolor=DLG_ABG,
                              activebackground=DLG_BG, activeforeground=DLG_FG,
                              font=("TkDefaultFont", 9), anchor="w", width=28,
                              justify="left"
                              ).grid(row=row, column=0, sticky="w", padx=(16, 4))

                tk.Label(parent, text="name:", bg=DLG_BG, fg=DLG_HINT2,
                         font=("TkDefaultFont", 8)).grid(row=row, column=1, sticky="e")
                tk.Entry(parent, textvariable=name_var, width=10,
                        bg=DLG_ENTRY, relief="sunken", font=("Courier", 9)
                        ).grid(row=row, column=2, sticky="w", padx=(4, 8))

                if spec['dual_pol_required']:
                    tk.Label(parent, text="(dual-pol)", bg=DLG_BG, fg=DLG_HINT2,
                             font=("TkDefaultFont", 7, "italic")
                             ).grid(row=row, column=3, sticky="w")

                self._row_vars[spec['key']] = (chk_var, name_var)
                row += 1

    # ── File selection ──────────────────────────────────────────────────

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Open Vaisala I/Q File",
            filetypes=[
                ("All files", "*.*"),
            ], parent=self)
        if not path:
            return
        self._path = path
        self._file_var.set(os.path.basename(path))
        self._render_btn.config(state="normal")
        if is_koun_file(path):
            self._status_var.set(
                "\u26a0 KOUN I/Q file detected \u2014 moments may not come "
                "out as expected (see disclaimer above).")
        else:
            self._status_var.set("")

    # ── Validation ───────────────────────────────────────────────────────

    def _collect_selection(self):
        try:
            az_deg = float(self._az_var.get())
            if az_deg <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning(
                "Open I/Q File",
                "Resolution must be a positive number of degrees.",
                parent=self)
            return None

        selected = {}
        used_names = set()
        for key, (chk_var, name_var) in self._row_vars.items():
            if not chk_var.get():
                continue
            name = name_var.get().strip()
            if not name:
                messagebox.showwarning(
                    "Open I/Q File",
                    "Every checked product needs an output name.",
                    parent=self)
                return None
            if name in used_names:
                messagebox.showwarning(
                    "Open I/Q File",
                    f"Output name '{name}' is used by more than one "
                    "selected product \u2014 names must be unique.",
                    parent=self)
                return None
            used_names.add(name)
            selected[key] = name

        if not selected:
            messagebox.showwarning(
                "Open I/Q File", "Select at least one product to render.",
                parent=self)
            return None

        return selected, az_deg

    # ── Render ───────────────────────────────────────────────────────────

    def _on_render(self):
        if not self._path:
            return
        collected = self._collect_selection()
        if collected is None:
            return
        selected, az_deg = collected
        clutter = self._clutter_var.get()
        path = self._path

        self._render_btn.config(state="disabled")
        self._status_var.set("Loading\u2026")

        def _progress(msg):
            self.after(0, lambda: self._status_var.set(msg))

        def _worker():
            try:
                volume, warn = load_iq_volume(
                    path, selected, az_deg=az_deg, clutter_filter=clutter,
                    progress_cb=_progress, verbose=False)
            except IQLoadError as e:
                self.after(0, lambda: self._fail(str(e)))
                return
            except Exception as e:
                self.after(0, lambda: self._fail(f"Unexpected error:\n{e}"))
                return
            self.after(0, lambda: self._succeed(volume, warn, path))

        threading.Thread(target=_worker, daemon=True).start()

    def _fail(self, msg):
        self._render_btn.config(state="normal")
        self._status_var.set("")
        messagebox.showerror("Open I/Q File", msg, parent=self)

    def _succeed(self, volume, warn, path):
        install_iq_volume(self.app, volume, path)
        self.destroy()
        if warn:
            messagebox.showinfo(
                "Open I/Q File \u2014 Done",
                "Rendered with the following notes:\n\n" + "\n".join(warn))


def install_iq_volume(app, volume, path):
    """Push a freshly-computed IQ volume into a running GURTVispy app.

    Mirrors the tail of GURTVispy.load_current_file's on-main-thread
    callback in gurt_gui.py, minus the file I/O step — the volume is
    already fully built by the time this is called, so there's no
    app._load_fn round trip to make.
    """
    is_first_load = app.volume is None

    app.file_list  = [os.path.abspath(path)]
    app.file_index = 0
    app.volume     = volume
    app._clear_render_cache()

    app.file_var.set(f"[1/1]  {os.path.basename(path)}  [I/Q Beta]")
    app.title(f"GURT Vispy \u2014 {os.path.basename(path)} [I/Q Beta]")

    n_sweeps = len(volume.sweeps)
    app.sweep_index.set(min(app.sweep_index.get(), max(0, n_sweeps - 1)))
    app.sweep_spin.config(to=max(0, n_sweeps - 1))

    fields_now = volume.field_names
    priority = ["DBZHC", "DBZH", "DBZ", "VEL", "VELH"]
    used = set()
    for panel in app.panels:
        panel.field_combo["values"] = fields_now
        if not fields_now:
            continue
        cur = panel.field_combo.get()
        if cur in fields_now and cur not in used:
            used.add(cur)
            continue
        chosen = next((f for f in priority if f in fields_now and f not in used),
                     next((f for f in fields_now if f not in used), fields_now[0]))
        panel.set_field(chosen)
        used.add(chosen)

    app._run_stagger_detection()
    if is_first_load:
        for panel in app.panels:
            panel.ppi.reset_camera(app.max_range.get())
    app.refresh_plot()
