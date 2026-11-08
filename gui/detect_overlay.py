# SPDX-License-Identifier: GPL-2.0-only
# GURT - Copyright (C) the GURT authors.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the
# Free Software Foundation. It is distributed WITHOUT ANY WARRANTY; see the
# file LICENSE for details.
"""DREADER vortex detection / isolation / tracking as a PPI overlay tool.

Analysis menu:
  Vortex tracking: draw search box...
        Drag a box around ONE vortex on any PPI panel. The strongest detection
        inside the box (located by azimuthal shear and/or the DREADER velocity
        difference, see Settings) is the seed. Every other sweep is searched only
        in a box that follows that vortex (last position advanced by its estimated
        motion), forward and backward through the loaded folder, so unrelated
        couplets never become part of the track.
  Re-run box tracking          -- same box and seed sweep, after changing Settings.
  Vortex detection: this sweep -- all couplets in the sweep (azimuthal-shear screened).
  Vortex detection + tracking (all vortices): folder
        DREADER's original detect -> isolate -> track over every sweep.
  Vortex detection / tracking settings

Detections that belong to the loaded file/sweep are drawn on every panel as a
circle (diameter = the detected couplet width) with the vortex number, and the
search box that produced them. The results window lists them and plots the tracks.

Scene frame: km, x east, y north, radar at the origin.
"""

import logging as _logging
_logging.getLogger("matplotlib.font_manager").setLevel(_logging.ERROR)   # a missing font in the fallback list is not an error
import base64
import io
import math
import os
import tkinter as tk
from datetime import datetime, timezone
from tkinter import filedialog, messagebox

import numpy as np
from vispy.scene import visuals

from dreader_detect import (detect_vortices, detect_in_box, isolate_detections, track_vortices,
                            detections_to_csv, BoxFollower, WindSwath, swath_to_csv, swath_to_ascii, swath_field_for_sweep, CRITERIA, latlon_to_xy_km,
                            _unix as sweep_time)
from dreader_xsect import is_velocity_field

LINE_C = (0.0, 0.0, 0.0, 1.0)
_FONT = {"family": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
         "weight": "bold"}
_COLORS = ["red", "navy", "teal", "darkorange", "purple", "darkgreen", "brown", "magenta"]
TRACK_MODES = ["Path + points", "Points only", "Off"]


def _sane_alt(v):
    """Radar altitude in metres; 0 when the file holds a garbage value (see gurt_core.sane_altitude_m)."""
    try:
        from gurt_core import sane_altitude_m
        return sane_altitude_m(v)
    except Exception:
        return float(v) if isinstance(v, (int, float)) and -500.0 <= v <= 12000.0 else 0.0


def _rgba(name):
    from matplotlib.colors import to_rgba
    return to_rgba(name)

DEFAULTS = {
    "criterion": CRITERIA[0], "polarity": "Cyclonic", "min_azshear": 0.02, "azshear_radius_m": 500.0,
    "maxdiam_m": 2000.0, "mindeltav": 30.0, "extremapct": 90.0, "maxtors": 10,
    "follow_scale": 1.0, "max_misses": 4, "predict": 1, "track_after_seed": 1,
    "maxtimediff": 600.0, "maxelevdiff": 2.0, "maxlatdiff": 0.03, "maxlondiff": 0.03,
    "mindetects": 3, "deltz": 0.0,
    "swath_smooth_m": 200.0, "swath_res_m": 100.0, "swath_min_ms": 20.0,
}
# (label, key, kind, options): kind in head / float / int / choice / bool
FIELDS = [
    ("DETECTION CRITERION", None, "head", None),
    ("Locate by", "criterion", "choice", list(CRITERIA)),
    ("Rotation sense", "polarity", "choice", ["Cyclonic", "Anticyclonic", "Either"]),
    ("Min azimuthal shear (1/s)", "min_azshear", "float", None),
    ("Azimuthal-shear window radius (m)", "azshear_radius_m", "float", None),
    ("Min velocity difference (m/s)", "mindeltav", "float", None),
    ("Max couplet diameter (m)", "maxdiam_m", "float", None),
    ("Extrema percentile (%)", "extremapct", "float", None),
    ("BOX TRACKING (one vortex)", None, "head", None),
    ("Track through folder after seeding", "track_after_seed", "bool", None),
    ("Predict motion for the follow box", "predict", "bool", None),
    ("Follow-box size (x drawn box)", "follow_scale", "float", None),
    ("Max consecutive missed sweeps", "max_misses", "int", None),
    ("Max time gap to last detection (s)", "maxtimediff", "float", None),
    ("WIND SWATH (box tracking)", None, "head", None),
    ("Velocity-magnitude smoothing (m)", "swath_smooth_m", "float", None),
    ("Swath grid resolution (m)", "swath_res_m", "float", None),
    ("Swath outline contour (m/s)", "swath_min_ms", "float", None),
    ("ALL-VORTICES MODE", None, "head", None),
    ("Max detections per sweep (0 = no limit)", "maxtors", "int", None),
    ("Isolation: max elevation difference (deg)", "maxelevdiff", "float", None),
    ("Isolation: max latitude difference (deg)", "maxlatdiff", "float", None),
    ("Isolation: max longitude difference (deg)", "maxlondiff", "float", None),
    ("Isolation: min detections per vortex", "mindetects", "int", None),
    ("Tracking layer depth (m, 0 = height quartiles)", "deltz", "float", None),
]


def _rect_segments(x0, x1, y0, y1):
    c = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    seg = []
    for i in range(4):
        a, b = c[i], c[(i + 1) % 4]
        seg += [[a[0], a[1], 0.0], [b[0], b[1], 0.0]]
    return np.asarray(seg, dtype="float32")


def _circle_segments(cx, cy, r, n=48):
    th = np.linspace(0.0, 2.0 * math.pi, n + 1)
    x, y = cx + r * np.cos(th), cy + r * np.sin(th)
    segs = []
    for i in range(n):
        segs.append([x[i], y[i], 0.0])
        segs.append([x[i + 1], y[i + 1], 0.0])
    return np.asarray(segs, dtype="float32")


class _CanvasVisuals:
    def __init__(self, scene):
        self.scene = scene
        self.circles = visuals.Line(connect="segments", color=LINE_C, width=2.0, parent=scene)
        self.rects = visuals.Line(connect="segments", color=LINE_C, width=2.0, parent=scene)
        self.pts = visuals.Markers(parent=scene)
        self.trk_line = visuals.Line(connect="segments", color=LINE_C, width=1.5, parent=scene)
        self.trk_pts = visuals.Markers(parent=scene)
        self.texts = []
        self.hide()

    def hide(self):
        self.circles.visible = False
        self.rects.visible = False
        self.pts.visible = False
        self.trk_line.visible = False
        self.trk_pts.visible = False
        for t in self.texts:
            t.visible = False

    def clear_texts(self):
        for t in self.texts:
            t.parent = None
        self.texts = []

    def draw(self, ov):
        self.clear_texts()
        self.hide()
        if ov["segments"] is not None:
            self.circles.set_data(pos=ov["segments"])
            self.pts.set_data(pos=ov["points"], face_color=[(1, 1, 0, 1)] * len(ov["points"]),
                              edge_color=[(0, 0, 0, 1)] * len(ov["points"]), size=7)
            for x, y, txt in ov["labels"]:
                self.texts.append(visuals.Text(txt, pos=[x, y, 0], color=(0, 0, 0, 1), font_size=7,
                                               bold=False, anchor_x="left", anchor_y="bottom",
                                               parent=self.scene))
            self.circles.visible = True
            self.pts.visible = True
        if ov["rects"] is not None:
            self.rects.set_data(pos=ov["rects"])
            self.rects.visible = True
        if ov.get("trk_points") is not None:
            if ov.get("trk_segments") is not None:
                self.trk_line.set_data(pos=ov["trk_segments"], color=ov["trk_seg_colors"])
                self.trk_line.visible = True
            self.trk_pts.set_data(pos=ov["trk_points"], face_color=ov["trk_face"],
                                  edge_color=[(0, 0, 0, 1)] * len(ov["trk_points"]),
                                  size=ov["trk_size"])
            self.trk_pts.visible = True


class DetectTool:
    def __init__(self, app):
        self.app = app
        self.settings = dict(DEFAULTS)
        self.dets = []              # all detections (dicts), chronological
        self.track_ids = []         # per detection: vortex id (0 = none)
        self.tracks = []            # list of lists of detection indices
        self.tracked = False
        self.notes = []             # one-line messages for the results window
        self.track_mode = TRACK_MODES[0]   # how the track is mapped on the radar canvas
        # search box
        self.box_picking = False
        self._box_start = None
        self._box_panel = None
        self._preview = None
        self.user_box = None        # (x0, x1, y0, y1) km, radar-relative, on the seed sweep
        self._box_src = None        # (path, sweep index) the box was drawn on
        self._visuals = {}
        self._win = None
        self._img = None
        self.swath = None
        self.swath_grid = None
        self._visited = []
        self._swath_meta = {}
        self._resize_job = None

    # ───────────────────────────────────────────── settings dialog
    def open_settings(self):
        win = tk.Toplevel(self.app)
        win.title("Vortex detection / tracking settings")
        win.resizable(False, False)
        win.configure(bg="#1a1a2e")
        vars_ = {}
        row = 0
        for label, key, kind, opts in FIELDS:
            if kind == "head":
                tk.Label(win, text=label, bg="#1a1a2e", fg="#9fb4ff",
                         font=("TkDefaultFont", 9, "bold")).grid(row=row, column=0, columnspan=2,
                                                                sticky="w", padx=10, pady=(8, 2))
                row += 1
                continue
            tk.Label(win, text=label, bg="#1a1a2e", fg="white", anchor="w").grid(
                row=row, column=0, sticky="w", padx=10, pady=3)
            if kind == "choice":
                v = tk.StringVar(value=str(self.settings[key]))
                tk.OptionMenu(win, v, *opts).grid(row=row, column=1, sticky="ew", padx=10, pady=2)
            elif kind == "bool":
                v = tk.IntVar(value=int(bool(self.settings[key])))
                tk.Checkbutton(win, variable=v, bg="#1a1a2e", activebackground="#1a1a2e",
                               selectcolor="#333").grid(row=row, column=1, sticky="w", padx=10)
            else:
                v = tk.StringVar(value=str(self.settings[key]))
                tk.Entry(win, textvariable=v, width=11, bg="#fff").grid(row=row, column=1, padx=10, pady=3)
            vars_[key] = (v, kind)
            row += 1

        def ok():
            for k, (v, kind) in vars_.items():
                try:
                    if kind == "choice":
                        self.settings[k] = v.get()
                    elif kind == "bool":
                        self.settings[k] = int(v.get())
                    elif kind == "int":
                        self.settings[k] = int(float(v.get()))
                    else:
                        self.settings[k] = float(v.get())
                except ValueError:
                    pass
            win.destroy()
        tk.Button(win, text="OK", command=ok, width=8).grid(row=row, column=0, columnspan=2, pady=10)

    # ───────────────────────────────────────────── helpers
    def _current_path(self):
        a = self.app
        try:
            return a.file_list[a.file_index]
        except Exception:
            return "current"

    def _pick_field(self, sweep, panel_field):
        if panel_field in sweep.fields and is_velocity_field(panel_field):
            return panel_field
        for n in sweep.fields:
            if is_velocity_field(n):
                return n
        return None

    def _panel_field(self):
        p = self.app._active_panel or (self.app.panels[0] if self.app.panels else None)
        fi = p.current_field_info() if p is not None else None
        return fi[0] if fi else None

    def _box_kwargs(self):
        st = self.settings
        return dict(criterion=st["criterion"], polarity=st["polarity"],
                    min_azshear=st["min_azshear"], azshear_radius_m=st["azshear_radius_m"],
                    maxdiam_m=st["maxdiam_m"], mindeltav=st["mindeltav"],
                    extremapct=st["extremapct"])

    def _azshear_filter(self):
        st = self.settings
        if "AzShear" not in st["criterion"]:
            return None
        return {"min_azshear": st["min_azshear"], "polarity": st["polarity"],
                "radius_m": st["azshear_radius_m"]}

    # ───────────────────────────────────────────── search box (drag on a PPI panel)
    def begin_box(self):
        if self.app.volume is None:
            messagebox.showinfo("Vortex tracking", "Open a radar file first.")
            return
        self.box_picking = True
        self._box_start = None
        self.app.cursor_var.set("Vortex tracking: drag a box around the vortex  (Esc / right-click to cancel)")
        self.app.bind("<Escape>", self.cancel_box, add="+")

    def cancel_box(self, *_):
        if self.box_picking or self._box_start is not None:
            self.box_picking = False
            self._box_start = None
            self._preview = None
            self.app.cursor_var.set("Cursor: —")
            self._redraw_all()

    def box_press(self, panel, x, y):
        self._box_panel = panel
        self._box_start = (x, y)
        self._preview = (x, x, y, y)

    def box_drag(self, x, y):
        if self._box_start is None:
            return
        import time
        now = time.monotonic()
        if now - getattr(self, "_drag_t", 0.0) < 0.03:
            return
        self._drag_t = now
        x0, y0 = self._box_start
        self._preview = (min(x0, x), max(x0, x), min(y0, y), max(y0, y))
        self._redraw_all()

    def box_release(self):
        if self._box_start is None:
            return
        box = self._preview
        panel = self._box_panel
        self._box_start = None
        self._preview = None
        self.box_picking = False
        self.app.cursor_var.set("Cursor: —")
        if box is None or (box[1] - box[0]) < 0.05 or (box[3] - box[2]) < 0.05:
            self._redraw_all()
            messagebox.showinfo("Vortex tracking", "The box is too small - drag a larger area.")
            return
        self.user_box = box
        self._box_src = (self._current_path(), int(self.app.sweep_index.get()))
        self._seed_and_track(panel, box)

    # ───────────────────────────────────────────── box tracking
    def rerun_box(self):
        if self.user_box is None or self._box_src is None:
            messagebox.showinfo("Vortex tracking", "Draw a search box first.")
            return
        self._seed_and_track(None, self.user_box)

    def _seed_and_track(self, panel, box):
        app, st = self.app, self.settings
        path, si = self._box_src
        try:
            vol = app.volume if path == self._current_path() else app._load_fn(path)
            sw = vol.sweeps[si]
        except Exception as e:
            messagebox.showerror("Vortex tracking", f"Could not load the seed sweep:\n{e}")
            return
        if getattr(sw, "scan_type", "ppi") == "rhi":
            messagebox.showinfo("Vortex tracking", "Tracking needs PPI sweeps.")
            return
        if panel is not None:
            pf = panel.current_field_info()
            pfield = pf[0] if pf else None
        else:
            pfield = self._panel_field()
        field = self._pick_field(sw, pfield)
        if field is None:
            messagebox.showinfo("Vortex tracking", "No velocity field in the seed sweep.")
            return
        kw = self._box_kwargs()
        try:
            app.cursor_var.set("Searching the box…")
            app.update_idletasks()
            why = []
            seed = detect_in_box(sw, field, box, vol.latitude, vol.longitude,
                                 file_id=path, sweep_index=si, vol=vol, why=why, **kw)
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            messagebox.showerror("Vortex tracking", f"Detection failed:\n\n{e}")
            return
        finally:
            app.cursor_var.set("Cursor: —")
        if seed is None:
            self.dets, self.track_ids, self.tracks, self.tracked = [], [], [], False
            self._redraw_all()
            messagebox.showinfo(
                "Vortex tracking",
                "No vortex found in the box with the current criteria "
                f"({st['criterion']}, {st['polarity'].lower()}, azimuthal shear >= {st['min_azshear']:g} 1/s, "
                f"dV >= {st['mindeltav']:g} m/s).\n\n"
                + ("Why: " + why[0] + ".\n\n" if why else "")
                + f"Field searched: {field}. Drag a different box or relax the thresholds in Settings.")
            return
        seed["seed"] = True
        self.notes = [f"Seed: {os.path.basename(path)} sweep {si}, field {field}, "
                      f"criterion {st['criterion']} / {st['polarity'].lower()}."]
        dets = [seed]
        self.swath = WindSwath(seed["lat"], seed["lon"], st["swath_smooth_m"])
        self.swath.add(sw, field, box, vol.latitude, vol.longitude)
        self._visited = [path]
        self._swath_meta = {"sensor": str(getattr(vol, "radar_name", "") or "Radar"),
                            "alt": _sane_alt(getattr(vol, "altitude", 0.0) or 0.0),
                            "radar_lat": float(vol.latitude), "radar_lon": float(vol.longitude),
                            "unix": float(seed["time"])}
        files = list(getattr(app, "file_list", []) or [])
        if st["track_after_seed"] and path in files:
            half_x, half_y = 0.5 * (box[1] - box[0]), 0.5 * (box[3] - box[2])
            fwd, why_f = self._walk(+1, files, files.index(path), vol, seed, si, field, half_x, half_y)
            bwd, why_b = self._walk(-1, files, files.index(path), vol, seed, si, field, half_x, half_y)
            dets = list(reversed(bwd)) + [seed] + fwd
            self.notes.append(f"Forward: {len(fwd)} detection(s), {why_f}.")
            self.notes.append(f"Backward: {len(bwd)} detection(s), {why_b}.")
        dets.sort(key=lambda d: (d["time"], d["sweep"]))
        track = list(range(len(dets)))
        track_vortices(dets, track, deltz=st["deltz"])
        self.dets, self.track_ids = dets, [1] * len(dets)
        self.tracks, self.tracked = [track], True
        self.swath_grid = self.swath.grid(st["swath_res_m"]) if self.swath else None
        self._add_swath_field()
        self._redraw_all()
        self._show_results(f"box track ({len(dets)} detections)")

    def _add_swath_field(self):
        """Put the swath on every visited volume as the plottable field VMAG_SWATH."""
        app, name = self.app, "VMAG_SWATH"
        if self.swath_grid is None:
            return
        n = 0
        for path in self._visited:
            try:
                vol = app.volume if path == self._current_path() else app._load_fn(path)
            except Exception:
                continue
            for sw in vol.sweeps:
                if getattr(sw, "scan_type", "ppi") == "rhi" or not sw.fields:
                    continue
                try:
                    sw.fields[name] = swath_field_for_sweep(sw, self.swath_grid, self.swath.ref,
                                                            vol.latitude, vol.longitude)
                    n += 1
                except Exception:
                    pass
            fn = getattr(vol, "field_names", None)
            if fn is not None and name not in fn:
                fn.append(name)
        if not n:
            return
        for panel in app.panels:
            vals = list(panel.field_combo["values"])
            if name not in vals:
                vals.append(name)
                panel.field_combo["values"] = vals
        self.notes.append(f"Field {name} added to {len(self._visited)} file(s): pick it in the field "
                          f"dropdown to plot the swath on the radar canvas.")
        app._invalidate_render_cache(field_name=name)
        app._schedule_refresh()

    def _walk(self, sign, files, seed_fi, seed_vol, seed, seed_si, field, half_x, half_y):
        """Follow the seeded vortex forward (+1) or backward (-1) through the folder."""
        app, st = self.app, self.settings
        fol = BoxFollower(seed, half_x, half_y, predict=bool(st["predict"]),
                          scale=st["follow_scale"], max_misses=int(st["max_misses"]),
                          maxtimediff=st["maxtimediff"])
        kw = self._box_kwargs()
        seed_vt = sweep_time(seed_vol.sweeps[seed_si], seed_vol, 0)
        seed_key = (seed_vt, seed_si)
        out = []
        why = "end of folder"
        indices = range(seed_fi, len(files)) if sign > 0 else range(seed_fi, -1, -1)
        for fi in indices:
            path = files[fi]
            app.cursor_var.set(f"Tracking {'forward' if sign > 0 else 'backward'}: "
                               f"{fi + 1}/{len(files)} {os.path.basename(path)}…")
            app.update_idletasks()
            try:
                vol = seed_vol if fi == seed_fi else app._load_fn(path)
            except Exception as e:
                self.notes.append(f"{os.path.basename(path)}: {e}")
                continue
            if path not in self._visited:
                self._visited.append(path)
            sweeps = []
            for si, sw in enumerate(vol.sweeps):
                if getattr(sw, "scan_type", "ppi") == "rhi":
                    continue
                key = (sweep_time(sw, vol, 0), si)
                if fi == seed_fi and ((sign > 0 and key <= seed_key) or (sign < 0 and key >= seed_key)):
                    continue
                sweeps.append((key, si, sw))
            sweeps.sort(key=lambda x: x[0], reverse=(sign < 0))
            for key, si, sw in sweeps:
                fld = field if field in sw.fields else self._pick_field(sw, None)
                if fld is None:
                    continue
                box = fol.box_for(key[0], vol.latitude, vol.longitude)
                if box is None:
                    why = f"stopped: next sweep more than {st['maxtimediff']:.0f} s from the last detection"
                    app.cursor_var.set("Cursor: —")
                    return out, why
                det = detect_in_box(sw, fld, box, vol.latitude, vol.longitude,
                                    file_id=path, sweep_index=si, vol=vol, **kw)
                if det is not None:
                    out.append(det)
                    fol.accept(det)
                    if self.swath is not None:
                        self.swath.add(sw, fld, box, vol.latitude, vol.longitude)
                elif fol.miss():
                    why = f"stopped: lost the vortex ({int(st['max_misses'])}+ sweeps in a row with no detection)"
                    app.cursor_var.set("Cursor: —")
                    return out, why
        app.cursor_var.set("Cursor: —")
        return out, why

    # ───────────────────────────────────────────── detection: this sweep (all couplets)
    def run_sweep(self):
        app, st = self.app, self.settings
        if app.volume is None:
            messagebox.showinfo("Vortex detection", "Open a radar file first.")
            return
        si = max(0, min(len(app.volume.sweeps) - 1, app.sweep_index.get()))
        sw = app.volume.sweeps[si]
        if getattr(sw, "scan_type", "ppi") == "rhi":
            messagebox.showinfo("Vortex detection", "Detection needs a PPI sweep.")
            return
        field = self._pick_field(sw, self._panel_field())
        if field is None:
            messagebox.showinfo("Vortex detection", "No velocity field in this sweep.")
            return
        try:
            app.cursor_var.set("Detecting vortices…")
            app.update_idletasks()
            dets = detect_vortices(sw, field, app.volume.latitude, app.volume.longitude,
                                   maxdiam_m=st["maxdiam_m"], mindeltav=st["mindeltav"],
                                   extremapct=st["extremapct"], maxtors=int(st["maxtors"]),
                                   file_id=self._current_path(), sweep_index=si, vol=app.volume,
                                   azshear_filter=self._azshear_filter())
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            messagebox.showerror("Vortex detection", f"Detection failed:\n\n{e}")
            return
        finally:
            app.cursor_var.set("Cursor: —")
        self.dets = dets
        self.track_ids = [0] * len(dets)
        self.tracks = []
        self.tracked = False
        self.notes = []
        self._redraw_all()
        self._show_results(f"sweep {si}  ({field})")

    # ───────────────────────────────────────────── all vortices: detect + isolate + track over the folder
    def run_folder(self):
        app, st = self.app, self.settings
        files = list(getattr(app, "file_list", []) or [])
        if not files or app._load_fn is None:
            messagebox.showinfo("Vortex tracking", "Open a folder of radar files first (Ctrl+D).")
            return
        if not messagebox.askokcancel(
                "Vortex detection + tracking (all vortices)",
                f"Run detection on every sweep of all {len(files)} file(s), then isolate and "
                "track every vortex found?\n\nTo follow a single vortex use 'Vortex tracking: draw "
                "search box…' instead - it avoids most false tracks."):
            return
        pfield = self._panel_field()
        dets, errors = [], []
        try:
            for i, path in enumerate(files):
                app.cursor_var.set(f"Detecting {i + 1}/{len(files)}: {os.path.basename(path)}…")
                app.update_idletasks()
                try:
                    vol = app._load_fn(path)
                    for si, sw in enumerate(vol.sweeps):
                        if getattr(sw, "scan_type", "ppi") == "rhi":
                            continue
                        field = self._pick_field(sw, pfield)
                        if field is None:
                            continue
                        dets += detect_vortices(
                            sw, field, vol.latitude, vol.longitude,
                            maxdiam_m=st["maxdiam_m"], mindeltav=st["mindeltav"],
                            extremapct=st["extremapct"], maxtors=int(st["maxtors"]),
                            file_id=path, sweep_index=si, vol=vol,
                            azshear_filter=self._azshear_filter())
                except Exception as e:
                    errors.append(f"{os.path.basename(path)}: {e}")
            dets.sort(key=lambda d: (d["time"], d["sweep"]))
            app.cursor_var.set("Isolating and tracking…")
            app.update_idletasks()
            tracks = isolate_detections(dets, st["maxtimediff"], st["maxelevdiff"],
                                        st["maxlatdiff"], st["maxlondiff"], int(st["mindetects"]))
            for t in tracks:
                track_vortices(dets, t, deltz=st["deltz"])
        finally:
            app.cursor_var.set("Cursor: —")
        ids = [0] * len(dets)
        for k, t in enumerate(tracks, 1):
            for i in t:
                ids[i] = k
        self.dets, self.track_ids, self.tracks, self.tracked = dets, ids, tracks, True
        self.notes = []
        self._redraw_all()
        self._show_results(f"folder ({len(files)} files)", errors)

    # ───────────────────────────────────────────── overlay
    def clear(self):
        self.dets, self.track_ids, self.tracks, self.tracked = [], [], [], False
        self.user_box = None
        self._box_src = None
        self._preview = None
        self.swath = self.swath_grid = None
        self._redraw_all()

    def sync_visibility(self, sweep_index):
        if self.dets or self._preview is not None:
            self._redraw_all()

    def _overlay_for_current(self):
        path = self._current_path()
        si = int(self.app.sweep_index.get())
        segs, pts, labels, rects = [], [], [], []
        for i, d in enumerate(self.dets):
            if d["sweep"] != si or d["file"] != path:
                continue
            segs.append(_circle_segments(d["x_km"], d["y_km"], 0.5 * d["diameter_m"] / 1000.0))
            pts.append([d["x_km"], d["y_km"], 0.0])
            tid = self.track_ids[i] if i < len(self.track_ids) else 0
            labels.append((d["x_km"], d["y_km"], f" {tid}" if tid > 0 else " ·"))
            if d.get("box"):
                rects.append(_rect_segments(*d["box"]))
        if self._preview is not None:
            rects.append(_rect_segments(*self._preview))
        ov = {"segments": np.vstack(segs) if segs else None,
              "points": np.array(pts, dtype="float32") if pts else None,
              "labels": labels,
              "rects": np.vstack(rects) if rects else None}
        ov.update(self._track_overlay())
        return ov

    def _track_overlay(self):
        """Every tracked detection (all sweeps) mapped into the loaded radar's frame.

        The seed is a large red marker; the other detections are small markers in
        their vortex's color, optionally joined by a path. Positions go through
        lat/lon, so they land correctly on whichever radar volume is displayed.
        """
        empty = {"trk_points": None, "trk_segments": None, "trk_seg_colors": None,
                 "trk_face": None, "trk_size": None}
        vol = self.app.volume
        if vol is None or self.track_mode == "Off" or not self.dets:
            return empty
        pos, face, size, segs, seg_cols = [], [], [], [], []
        groups = self.tracks if self.tracks else [[i for i, d in enumerate(self.dets)]]
        for k, idxs in enumerate(groups):
            col = _rgba(_COLORS[k % len(_COLORS)])
            prev = None
            for i in sorted(idxs, key=lambda j: self.dets[j]["time"]):
                d = self.dets[i]
                x, y = latlon_to_xy_km(d["lat"], d["lon"], vol.latitude, vol.longitude)
                pos.append([x, y, 0.0])
                if d.get("seed"):
                    face.append((1.0, 0.0, 0.0, 1.0))
                    size.append(13)
                else:
                    face.append((1.0, 1.0, 0.0, 1.0) if len(groups) == 1 else col)
                    size.append(6)
                if prev is not None and self.track_mode == "Path + points":
                    segs += [[prev[0], prev[1], 0.0], [x, y, 0.0]]
                    seg_cols += [col, col]
                prev = (x, y)
        if not pos:
            return empty
        return {"trk_points": np.array(pos, dtype="float32"),
                "trk_segments": np.array(segs, dtype="float32") if segs else None,
                "trk_seg_colors": np.array(seg_cols, dtype="float32") if segs else None,
                "trk_face": face, "trk_size": np.array(size, dtype="float32")}

    def set_track_mode(self, mode):
        self.track_mode = mode
        self._redraw_all()

    def _redraw_all(self):
        empty = {"segments": None, "points": None, "labels": [], "rects": None,
                 "trk_points": None}
        ov = self._overlay_for_current() if self.app.volume is not None else empty
        for panel in self.app.panels:
            ppi = panel.ppi
            cv = self._visuals.get(id(ppi))
            if cv is None:
                cv = self._visuals[id(ppi)] = _CanvasVisuals(ppi._view.scene)
            cv.draw(ov)
            ppi._vcanvas.update()

    # ───────────────────────────────────────────── results
    @staticmethod
    def _hms(t):
        if not t:
            return "--:--:--"
        return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%H:%M:%S")

    def _show_results(self, title, errors=None):
        if self._win is None or not self._win.winfo_exists():
            self._win = tk.Toplevel(self.app)
            self._win.title("Vortex detection")
            bar = tk.Frame(self._win)
            bar.pack(fill="x")
            tk.Button(bar, text="Save CSV…", command=self._save_csv).pack(side="left", padx=6, pady=4)
            tk.Button(bar, text="Clear overlay", command=self.clear).pack(side="left", padx=6, pady=4)
            tk.Label(bar, text="Radar track:").pack(side="left", padx=(14, 2))
            self._trk_var = tk.StringVar(value=self.track_mode)
            tk.OptionMenu(bar, self._trk_var, *TRACK_MODES,
                          command=self.set_track_mode).pack(side="left")
            tk.Label(bar, text="Plot:").pack(side="left", padx=(14, 2))
            self._plot_var = tk.StringVar(value="Tracks")
            tk.OptionMenu(bar, self._plot_var, "Tracks", "Wind swath",
                          command=lambda *_: self._replot()).pack(side="left")
            tk.Button(bar, text="Save swath CSV…", command=self._save_swath).pack(side="left", padx=6)
            tk.Button(bar, text="Export swath ASCII…", command=self._export_swath_ascii).pack(side="left", padx=6)
            tk.Button(bar, text="Export tormodel params…", command=self._export_tormodel).pack(side="left", padx=6)
            tk.Label(bar, text="Text readout:").pack(side="left", padx=(14, 2))
            self._readout_var = tk.StringVar(value="Hidden")
            tk.OptionMenu(bar, self._readout_var, "Hidden", "Shown",
                          command=lambda *_: self._apply_readout()).pack(side="left")
            self._img_lbl = tk.Label(self._win, bg="white", bd=0, highlightthickness=0)
            self._img_lbl.pack(fill="both", expand=True)
            self._img_lbl.bind("<Configure>", self._on_resize)
            self._win.geometry("1000x700")
            self._text = tk.Text(self._win, font=("Courier", 9), wrap="none", height=24)
            self._readout_shown = False
        self._trk_var.set(self.track_mode)
        self._win.title(f"Vortex detection — {title}")
        L = []
        st = self.settings
        L.append(f"VORTEX DETECTION  [{title}]   {len(self.dets)} detection(s)")
        L.append(f"criterion {st['criterion']} ({st['polarity'].lower()})   azshear >= {st['min_azshear']:g} 1/s "
                 f"(window {st['azshear_radius_m']:.0f} m)   dV >= {st['mindeltav']:.1f} m/s   "
                 f"maxdiam {st['maxdiam_m']:.0f} m   extrema {st['extremapct']:.0f}%")
        for note in self.notes:
            L.append("  " + note)
        L.append("")
        L.append(f"{'vtx':>4} {'time':>8} {'sw':>3} {'elev':>5} {'az':>7} {'range_km':>9} "
                 f"{'dV':>6} {'diam_m':>7} {'azshear':>8} {'min':>7} {'max':>7}"
                 + (f" {'spd':>5} {'dir':>5} {'incl':>5}" if self.tracked else ""))
        for i, d in enumerate(self.dets):
            tid = self.track_ids[i] if i < len(self.track_ids) else 0
            mark = "*" if d.get("seed") else " "
            azs = d.get("azshear", float("nan"))
            row = (f"{(str(tid) if tid > 0 else '-') + mark:>4} {self._hms(d['time']):>8} {d['sweep']:>3} "
                   f"{d['elevation']:>5.1f} {d['azimuth']:>7.2f} {d['range_m'] / 1000:>9.2f} "
                   f"{d['delta_v']:>6.1f} {d['diameter_m']:>7.0f} {'--' if azs != azs else f'{azs:.4f}':>8} "
                   f"{d['min_vel']:>7.1f} {d['max_vel']:>7.1f}")
            if self.tracked:
                inc = d.get("incl_deg", float("nan"))
                row += (f" {d.get('trans_speed', 0):>5.1f} {d.get('trans_dir', 0):>5.0f} "
                        f"{'--' if inc != inc else f'{inc:.1f}':>5}")
            L.append(row)
        if self.tracked:
            L.append("")
            if any(d.get("seed") for d in self.dets):
                L.append(f"TRACK  (* = seed, the first detection in the search box)")
            else:
                L.append(f"TRACKS  ({len(self.tracks)} vortex/vortices)   isolation: dt<={st['maxtimediff']:.0f}s "
                         f"dEl<={st['maxelevdiff']:.1f} dLat<={st['maxlatdiff']:.3f} dLon<={st['maxlondiff']:.3f} "
                         f"min {st['mindetects']}")
            for k, t in enumerate(self.tracks, 1):
                obs = [self.dets[i] for i in t]
                sp = [o["trans_speed"] for o in obs if o["trans_speed"] > 0]
                dr = [o["trans_dir"] for o in obs if o["trans_speed"] > 0]
                inc = [o["incl_corr_deg"] for o in obs if o["incl_corr_deg"] == o["incl_corr_deg"]]
                mdir = (math.degrees(math.atan2(np.mean(np.sin(np.radians(dr))),
                                                np.mean(np.cos(np.radians(dr))))) % 360.0) if dr else float("nan")
                L.append(f"  #{k:<3} {len(obs):>3} detections  "
                         f"{self._hms(obs[0]['time'])} - {self._hms(obs[-1]['time'])}   "
                         f"max dV {max(o['delta_v'] for o in obs):.0f} m/s   "
                         + (f"motion {np.mean(sp):.1f} m/s toward {mdir:.0f} deg   " if sp else "motion n/a   ")
                         + (f"tilt {np.mean(inc):.1f} deg" if inc else ""))
        if errors:
            L.append("")
            L.append(f"Errors ({len(errors)}):")
            L += [f"  {e}" for e in errors[:8]]
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.insert("1.0", "\n".join(L))
        self._text.configure(state="disabled")
        if self.swath_grid is None:
            self._plot_var.set("Tracks")
        self._replot()
        self._apply_readout()
        self._win.lift()

    def _plot_px(self):
        try:
            w, h = int(self._img_lbl.winfo_width()), int(self._img_lbl.winfo_height())
        except Exception:
            w = h = 0
        if w < 50 or h < 50:
            w, h = 980, 640
        return w, h

    def _on_resize(self, ev):
        if self._img is None:
            return
        if self._resize_job is not None:
            self._win.after_cancel(self._resize_job)
        self._resize_job = self._win.after(150, self._replot)

    def _replot(self):
        self._resize_job = None
        try:
            if self._plot_var.get() == "Wind swath" and self.swath_grid is not None:
                self._plot_swath()
            elif self.tracked and self.tracks:
                self._plot_tracks()
            else:
                self._img = None
                self._img_lbl.configure(image="")
        except Exception:
            import traceback
            print(traceback.format_exc())
            self._img = None
            self._img_lbl.configure(image="")
        self._apply_readout()

    def _plot_swath(self):
        """Maximum smoothed |V| swath along the track, with an outline contour, TURT style."""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        g = self.swath_grid
        M = g["M"]
        ny, nx = M.shape
        r = g["res_km"]
        wpx, hpx = self._plot_px()
        fig = Figure(figsize=(wpx / 100.0, hpx / 100.0), dpi=100)
        fig.patch.set_facecolor("white")
        FigureCanvasAgg(fig)
        ax = fig.subplots(1, 1)
        ext = [g["x0"], g["x0"] + nx * r, g["y0"], g["y0"] + ny * r]
        vmax = max(float(np.nanmax(M)), 1.0)
        im = ax.imshow(np.ma.masked_invalid(M), origin="lower", extent=ext, cmap="turbo",
                       vmin=0, vmax=max(120.0, vmax) if vmax > 100 else vmax, aspect="equal",
                       interpolation="nearest")
        lev = self.settings["swath_min_ms"]
        if np.isfinite(M).any() and np.nanmax(M) > lev:
            xs = g["x0"] + (np.arange(nx) + 0.5) * r
            ys = g["y0"] + (np.arange(ny) + 0.5) * r
            ax.contour(xs, ys, np.nan_to_num(M, nan=0.0), levels=[lev], colors="black", linewidths=1.0)
        ref = (self.swath.ref if self.swath else (0, 0))
        for k, t in enumerate(self.tracks, 1):
            ox = [latlon_to_xy_km(self.dets[i]["lat"], self.dets[i]["lon"], *ref) for i in t]
            ax.plot([p[0] for p in ox], [p[1] for p in ox], color="white", linewidth=0.9,
                    marker="^", markersize=3, markeredgecolor="black", markeredgewidth=0.4)
        cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.04)
        cb.set_label("Smoothed Velocity Magnitude (m/s)", fontsize=9, **_FONT)
        cb.ax.tick_params(labelsize=8.5)
        ax.set_xlabel("East of seed (km)", fontsize=9, **_FONT)
        ax.set_ylabel("North of seed (km)", fontsize=9, **_FONT)
        ax.set_title(f"Maximum Velocity Magnitude Swath ({self.swath.n_sweeps} sweeps, "
                     f"{self.settings['swath_smooth_m']:.0f} m smoothing)", fontsize=10.5, **_FONT)
        for sp in ax.spines.values():
            sp.set_color("black")
        ax.tick_params(axis="both", direction="out", length=4, width=1.0, labelsize=8.5)
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_fontfamily(_FONT["family"])
            lab.set_fontweight("bold")
        fig.subplots_adjust(left=0.07, right=0.93, top=0.94, bottom=0.09)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor="white")
        self._img = tk.PhotoImage(data=base64.b64encode(buf.getvalue()))
        self._img_lbl.configure(image=self._img)

    def _export_swath_ascii(self):
        if self.swath_grid is None:
            messagebox.showinfo("Wind swath", "Run a box track first.")
            return
        meta = self._swath_meta
        win = tk.Toplevel(self._win)
        win.title("Export swath ASCII")
        win.transient(self._win)
        rows = [("Sensor (sn)", meta.get("sensor", "Radar")), ("Product (pr)", "vmax"), ("Unit (un)", "m/s"),
                ("Fill value (no data)", "0"),
                ("Track start, unix s (UT)", f"{meta.get('unix', 0):.0f}"),
                ("Origin height, m (ZG)", f"{meta.get('alt', 0.0):.2f}")]
        vars_ = []
        for i, (lab, dflt) in enumerate(rows):
            tk.Label(win, text=lab).grid(row=i, column=0, sticky="w", padx=8, pady=3)
            v = tk.StringVar(value=dflt)
            tk.Entry(win, textvariable=v, width=18).grid(row=i, column=1, padx=8, pady=3)
            vars_.append(v)
        tk.Label(win, text="Row order").grid(row=len(rows), column=0, sticky="w", padx=8, pady=3)
        order = tk.StringVar(value="North to south")
        tk.OptionMenu(win, order, "North to south", "South to north").grid(row=len(rows), column=1, sticky="w", padx=8)

        def go():
            try:
                fill, ut, zg = float(vars_[3].get()), float(vars_[4].get()), float(vars_[5].get())
            except ValueError:
                messagebox.showwarning("Export swath ASCII", "Fill, UT and height must be numbers.", parent=win)
                return
            path = filedialog.asksaveasfilename(parent=win, defaultextension=".txt",
                                                filetypes=[("Text", "*.txt")],
                                                initialfile="swath.velocity.txt", title="Export swath ASCII")
            if not path:
                return
            swath_to_ascii(self.swath_grid, self.swath.ref, path, ut, zg, vars_[0].get().strip() or "Radar",
                           vars_[1].get().strip() or "vmax", vars_[2].get().strip() or "m/s", fill,
                           south_first=(order.get() == "South to north"))
            win.destroy()

        bf = tk.Frame(win)
        bf.grid(row=len(rows) + 1, column=0, columnspan=2, pady=8)
        tk.Button(bf, text="Export", command=go).pack(side="left", padx=4)
        tk.Button(bf, text="Cancel", command=win.destroy).pack(side="left", padx=4)

    def _export_tormodel(self):
        """Write a tormodel parameter file (one observation per tracked detection) from the detections."""
        import tormodel as TM
        if not self.dets:
            messagebox.showinfo("tormodel parameters", "Run a box track or a detection first.")
            return
        meta = self._swath_meta or {}
        lat, lon, alt, name = (meta.get("radar_lat"), meta.get("radar_lon"), meta.get("alt", 0.0), meta.get("sensor"))
        if lat is None:                                   # detection runs: take the first detection's radar
            try:
                path = self.dets[0]["file"]
                vol = self.app.volume if path == self._current_path() else self.app._load_fn(path)
                lat, lon, alt = float(vol.latitude), float(vol.longitude), _sane_alt(getattr(vol, "altitude", 0.0) or 0.0)
                name = str(getattr(vol, "radar_name", "") or "Radar")
            except Exception as e:
                messagebox.showerror("tormodel parameters", f"Could not read the radar position:\n{e}")
                return
        ids = sorted({t for t in self.track_ids if t > 0}) or [0]
        win = tk.Toplevel(self._win or self.app)
        win.title("Export tormodel parameters")
        rows = [("Track id (of %s)" % ",".join(map(str, ids)), str(ids[0])),
                ("Radar name", (name or "Radar").replace(" ", "")[:8]),
                ("Grid spacing, m", "100"), ("Grid margin around the track, m (blank = auto)", ""),
                ("Tangential core exponent", "1.0"), ("Tangential decay exponent", "-0.6"),
                ("Radial wind / tangential wind (negative = inflow)", "0"),
                ("Radial core / decay exponents", "1.0 -0.6"),
                ("Minimum seconds between observations", "0"),
                ("Smooth track over +- seconds (0 = raw detections)", "60"),
                ("Cut the track at jumps faster than, m/s (0 = off)", "60"),
                ("Velocity error for the Vr fit, m/s", "2.0"),
                ("Vr fit: time coupling (0 = each scan alone)", "1.0")]
        v = []
        for i, (lab, dflt) in enumerate(rows):
            tk.Label(win, text=lab).grid(row=i, column=0, sticky="w", padx=8, pady=3)
            var = tk.StringVar(value=dflt)
            tk.Entry(win, textvariable=var, width=18).grid(row=i, column=1, padx=8, pady=3)
            v.append(var)
        vrfit_on = tk.BooleanVar(value=False)
        tk.Checkbutton(win, variable=vrfit_on, justify="left", anchor="w",
                       text="Fit the radial wind (maxrad, corerad, decayrad) to the radar velocities\n"
                            "(variational: one cost over the whole track; overrides the radial values above)").grid(
            row=len(rows), column=0, columnspan=2, sticky="w", padx=8, pady=3)
        tk.Label(win, justify="left", text="Radius = half the couplet diameter; max tangential wind = half the couplet delta-V.\n"
                 "Positions are azimuth / range from the seed radar, so a moving radar is handled.").grid(
            row=len(rows) + 1, column=0, columnspan=2, padx=8, pady=4)

        def go():
            try:
                tid = int(v[0].get())
                core_r, decay_r = (float(x) for x in v[7].get().split())
                text, info = TM.params_from_track(
                    self.dets, {"name": v[1].get().strip() or "Radar", "lat": lat, "lon": lon, "elev": alt},
                    res=float(v[2].get()), margin_m=float(v[3].get()) if v[3].get().strip() else None,
                    core_tan=float(v[4].get()), decay_tan=float(v[5].get()), radial_frac=float(v[6].get()),
                    core_rad=core_r, decay_rad=decay_r, min_spacing_s=float(v[8].get()),
                    smooth_s=float(v[9].get()), max_speed=float(v[10].get()),
                    track_ids=self.track_ids, track=tid if tid else None)
            except (TM.TorError, ValueError) as e:
                messagebox.showerror("tormodel parameters", str(e), parent=win)
                return
            fit_msg = ""
            if vrfit_on.get():
                try:
                    import vrfit
                    cur = self._current_path()

                    def get_sweep(d):
                        vol = self.app.volume if d["file"] == cur else self.app._load_fn(d["file"])
                        return (vol.sweeps[int(d.get("sweep", 0))], float(vol.latitude), float(vol.longitude))
                    win.config(cursor="watch")
                    win.update_idletasks()
                    text, fi = vrfit.fit_params_text(text, self.dets, get_sweep, lat, lon, sigma=float(v[11].get()),
                                                     strength=float(v[12].get()))
                    win.config(cursor="")
                    lo_, hi_ = fi["maxrad_range"]
                    fit_msg = (f"\nVr fit: {fi['n_fit']} of {fi['n_obs']} scans had velocity data; misfit "
                               f"{fi['rms_before'] * float(v[11].get()):.1f} -> {fi['rms_after'] * float(v[11].get()):.1f} m/s rms; "
                               f"maxrad {lo_:.1f} to {hi_:.1f} m/s (negative = inflow).")
                except Exception as e:
                    win.config(cursor="")
                    messagebox.showerror("tormodel parameters", f"The Vr fit failed:\n{e}", parent=win)
                    return
            path = filedialog.asksaveasfilename(parent=win, defaultextension=".prm", initialfile="tormodel_params.txt",
                                                filetypes=[("Text", "*.txt *.prm"), ("All files", "*.*")],
                                                title="Save tormodel parameter file")
            if not path:
                return
            with open(path, "w", newline="\n") as fh:
                fh.write(text)
            messagebox.showinfo("tormodel parameters", f"Wrote {path}\n{info['n']} observations, grid {info['nx'] + 1} x "
                                f"{info['ny'] + 1} at {info['res']} m.\n{info['skipped']} detection(s) not used"
                                f" ({info.get('dropped', 0)} removed as off-track or after a jump).\n"
                                f"Fastest track motion: {info['raw_max_speed']:.0f} m/s raw -> {info['max_speed']:.0f} m/s "
                                "after smoothing.", parent=win)
            win.destroy()

        bf = tk.Frame(win)
        bf.grid(row=len(rows) + 2, column=0, columnspan=2, pady=8)
        tk.Button(bf, text="Export", command=go).pack(side="left", padx=4)
        tk.Button(bf, text="Cancel", command=win.destroy).pack(side="left", padx=4)

    def _save_swath(self):
        if self.swath_grid is None:
            messagebox.showinfo("Wind swath", "Run a box track first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                            initialfile="velmag_swath.csv", title="Save swath grid")
        if path:
            swath_to_csv(self.swath_grid, self.swath.ref, path)

    def _apply_readout(self):
        """Plot fills the window; the text table sits under it when 'Shown' (or when there is no plot)."""
        want = self._readout_var.get() == "Shown" or self._img is None
        if want and not self._readout_shown:
            self._text.configure(height=14)
            self._text.pack(side="bottom", fill="x")
            self._readout_shown = True
        elif not want and self._readout_shown:
            self._text.pack_forget()
            self._readout_shown = False

    def _plot_tracks(self):
        """Track map + dV time series, in TURT's publication style."""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        import matplotlib.ticker as mticker
        wpx, hpx = self._plot_px()
        fig = Figure(figsize=(wpx / 100.0, hpx / 100.0), dpi=100)
        fig.patch.set_facecolor("white")
        FigureCanvasAgg(fig)
        ax1, ax2 = fig.subplots(1, 2)
        lat0 = float(np.mean([d["lat"] for d in self.dets]))
        lon0 = float(np.mean([d["lon"] for d in self.dets]))
        kx = 111.32 * math.cos(math.radians(lat0))
        for k, t in enumerate(self.tracks, 1):
            obs = [self.dets[i] for i in t]
            c = _COLORS[(k - 1) % len(_COLORS)]
            x = [(o["lon"] - lon0) * kx for o in obs]
            y = [(o["lat"] - lat0) * 111.19 for o in obs]
            ax1.plot(x, y, color=c, marker="^", markersize=4, linewidth=1.0, label=f"Vortex {k}")
            tm = [(o["time"] - obs[0]["time"]) for o in obs]
            ax2.plot(tm, [o["delta_v"] for o in obs], color=c, marker="^", markersize=4,
                     linewidth=1.0, label=f"Vortex {k}")
        ax1.set_xlabel("East (km)", fontsize=9, **_FONT)
        ax1.set_ylabel("North (km)", fontsize=9, **_FONT)
        ax1.set_aspect("equal", adjustable="datalim")
        ax2.set_xlabel("Time Since First Detection (s)", fontsize=9, **_FONT)
        ax2.set_ylabel("Velocity Difference (m/s)", fontsize=9, **_FONT)
        for ax in (ax1, ax2):
            for sp in ax.spines.values():
                sp.set_color("black")
                sp.set_linewidth(1.0)
            ax.grid(False)
            ax.yaxis.grid(True, color="black", linewidth=0.8)
            ax.set_axisbelow(True)
            ax.tick_params(axis="both", direction="out", length=4, width=1.0, labelsize=8.5)
            for lab in ax.get_xticklabels() + ax.get_yticklabels():
                lab.set_fontfamily(_FONT["family"])
                lab.set_fontweight("bold")
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False,
                      prop={"family": _FONT["family"], "weight": "bold", "size": 8},
                      handlelength=1.8, columnspacing=1.5)
        ax1.set_title("Vortex Tracks", fontsize=10.5, pad=24, **_FONT)
        ax2.set_title("Detected Velocity Difference", fontsize=10.5, pad=24, **_FONT)
        fig.subplots_adjust(left=0.08, right=0.97, top=0.80, bottom=0.14, wspace=0.28)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor="white")
        self._img = tk.PhotoImage(data=base64.b64encode(buf.getvalue()))
        self._img_lbl.configure(image=self._img)

    def _save_csv(self):
        if not self.dets:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")],
            initialfile="vortex_detections.csv", title="Save detections / tracks")
        if path:
            detections_to_csv(self.dets, path, self.track_ids)
