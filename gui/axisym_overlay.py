# SPDX-License-Identifier: GPL-2.0-only
# GURT - Copyright (C) the GURT authors.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the
# Free Software Foundation. It is distributed WITHOUT ANY WARRANTY; see the
# file LICENSE for details.
"""Axisymmetric patch analysis (ported from TURT / DREADER) as a PPI overlay.

Workflow: Analysis > "Axisymmetric patch: pick center..." then click the
vortex on any PPI panel. A circular patch around the click is searched for the
strongest inbound / outbound pair of the panel's current velocity field, the
rings are centered midway between them, and the overlay shows

  * the min (blue) / max (red) points joined by a curved bar,
  * the thin circle through both points (centered on their midpoint),
  * dashed axisymmetric analysis rings, labeled with radius (km),

on every panel (they share one camera). A text window lists the patch results
(aspect-corrected velocities, F/EF equivalents, vorticity/divergence at
fractions of R ...) and the ring profile (Vt, Vr, circulation ...), which can
be saved to CSV.

All geometry is the app's scene frame: km, x east, y north, radar at the origin.
"""

import math
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from vispy.scene import visuals

from dreader_modes import (compute_dreader_patch, compute_axisymmetric, axisym_to_csv)

MIN_C = "#2255ff"
MAX_C = "#ee1111"
LINE_C = (0.0, 0.0, 0.0, 1.0)
MAX_DRAWN_RINGS = 24


# ───────────────────────────────────────────────────────────── geometry helpers
def _dashed_circle(cx, cy, r, n=96):
    """Segment pairs (2k, 3) for a dashed circle: alternate segments drawn."""
    th = np.linspace(0.0, 2.0 * math.pi, n + 1)
    x, y = cx + r * np.cos(th), cy + r * np.sin(th)
    segs = []
    for i in range(0, n, 2):
        segs.append([x[i], y[i], 0.0])
        segs.append([x[i + 1], y[i + 1], 0.0])
    return np.asarray(segs, dtype="float32")


def _circle_strip(cx, cy, r, n=96):
    th = np.linspace(0.0, 2.0 * math.pi, n + 1)
    return np.column_stack([cx + r * np.cos(th), cy + r * np.sin(th),
                            np.zeros(n + 1)]).astype("float32")


def _bar(p1, p2, bulge=0.2, n=60):
    """Quadratic Bezier between p1 and p2 bowed toward screen-up (+y)."""
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    chord = p2 - p1
    length = float(np.hypot(*chord))
    mid = 0.5 * (p1 + p2)
    if length <= 0:
        return np.array([[*p1, 0.0], [*p2, 0.0]], dtype="float32")
    normal = np.array([-chord[1], chord[0]]) / length
    if normal[1] < 0:
        normal = -normal
    ctrl = mid + normal * bulge * length
    t = np.linspace(0.0, 1.0, n)[:, None]
    curve = (1 - t) ** 2 * p1 + 2 * (1 - t) * t * ctrl + t ** 2 * p2
    return np.column_stack([curve, np.zeros(n)]).astype("float32")


# ───────────────────────────────────────────────────────────── per-canvas visuals
class _CanvasVisuals:
    """The vispy objects for one PPICanvas."""

    def __init__(self, scene):
        self.scene = scene
        self.rings = visuals.Line(connect="segments", color=LINE_C, width=2.0, parent=scene)
        self.couplet = visuals.Line(connect="strip", color=LINE_C, width=2.0, parent=scene)
        self.bar = visuals.Line(connect="strip", color=LINE_C, width=2.0, parent=scene)
        self.pts = visuals.Markers(parent=scene)
        self.texts = []
        self.hide()

    def hide(self):
        for v in (self.rings, self.couplet, self.bar, self.pts):
            v.visible = False
        for t in self.texts:
            t.visible = False

    def show(self):
        for v in (self.rings, self.couplet, self.bar, self.pts):
            v.visible = True
        for t in self.texts:
            t.visible = True

    def clear_texts(self):
        for t in self.texts:
            t.parent = None
        self.texts = []

    def draw_light(self, ov):
        """Cheap redraw during a drag: only the points, bar and couplet circle."""
        self.couplet.set_data(pos=ov["couplet"])
        self.bar.set_data(pos=ov["bar"])
        self.pts.set_data(pos=ov["points"],
                          face_color=[MIN_C, MAX_C], edge_color=[MIN_C, MAX_C], size=7)

    def draw(self, ov):
        """ov: dict from AxisymTool._overlay_geometry."""
        self.clear_texts()
        if ov["ring_segments"] is not None and len(ov["ring_segments"]):
            self.rings.set_data(pos=ov["ring_segments"])
        self.couplet.set_data(pos=ov["couplet"])
        self.bar.set_data(pos=ov["bar"])
        self.pts.set_data(pos=ov["points"],
                          face_color=[MIN_C, MAX_C], edge_color=[MIN_C, MAX_C], size=7)
        for x, y, txt in ov["labels"]:
            self.texts.append(visuals.Text(txt, pos=[x, y, 0], color=(0, 0, 0, 1),
                                           font_size=7, anchor_x="center", anchor_y="center",
                                           parent=self.scene))
        self.show()


# ───────────────────────────────────────────────────────────── controller
class AxisymTool:
    """Owns settings, the pick mode, the analysis, the overlay and the results window."""

    def __init__(self, app):
        self.app = app
        self.picking = False
        self.settings = {"radius_km": 1.0, "storm_speed": 0.0, "storm_dir": 0.0,
                         "beamwidth": 1.0, "air_density": 1.0}
        self._visuals = {}          # id(ppi) -> _CanvasVisuals
        self._overlay = None
        self._drag = None           # 'min' / 'max' while a marker is being dragged
        self._drag_t = 0.0
        self._pts = None            # {'min': (x, y), 'max': (x, y)} scene km
        self._key = None            # (id(volume), sweep index) the overlay belongs to
        self.result = None
        self._win = None

    # ----- menu actions
    def begin_pick(self):
        if self.app.volume is None:
            messagebox.showinfo("Axisymmetric patch", "Open a radar file first.")
            return
        self.picking = True
        self.app.cursor_var.set(
            "Axisymmetric patch: click the vortex center on a velocity PPI  (Esc / right-click to cancel)")
        self.app.bind("<Escape>", self._cancel_pick, add="+")

    def _cancel_pick(self, *_):
        if self.picking:
            self.picking = False
            self.app.cursor_var.set("Cursor: —")

    def run_at(self, panel, x_km, y_km):
        """Entry point from a PPI click (or the right-click menu)."""
        self.picking = False
        self.app.cursor_var.set("Cursor: —")
        try:
            self._run(panel, float(x_km), float(y_km))
        except Exception:
            import traceback
            tb = traceback.format_exc()
            print(tb)
            messagebox.showerror("Axisymmetric patch", f"Analysis failed:\n\n{tb}")

    def clear(self):
        self._overlay = None
        for cv in self._visuals.values():
            cv.hide()
            cv.clear_texts()
        self.result = None
        self._key = None
        self._redraw_all()

    def sync_visibility(self, sweep_index):
        """Called after every refresh: hide the overlay if the sweep/volume changed."""
        if self._overlay is None:
            return
        key = (id(self.app.volume), int(sweep_index))
        for cv in self._visuals.values():
            if key == self._key:
                cv.show()
            else:
                cv.hide()
        self._redraw_all()

    # ----- settings dialog
    def open_settings(self):
        win = tk.Toplevel(self.app)
        win.title("Axisymmetric patch settings")
        win.resizable(False, False)
        win.configure(bg="#1a1a2e")
        rows = [("Patch radius (km)", "radius_km"),
                ("Storm speed (m/s, 0 = none)", "storm_speed"),
                ("Storm motion toward (deg)", "storm_dir"),
                ("Radar beam width (deg)", "beamwidth"),
                ("Air density (kg/m^3)", "air_density")]
        vars_ = {}
        for i, (label, key) in enumerate(rows):
            tk.Label(win, text=label, bg="#1a1a2e", fg="white", anchor="w").grid(
                row=i, column=0, sticky="w", padx=10, pady=4)
            v = tk.StringVar(value=str(self.settings[key]))
            tk.Entry(win, textvariable=v, width=10, bg="#fff").grid(row=i, column=1, padx=10, pady=4)
            vars_[key] = v

        def ok():
            for k, v in vars_.items():
                try:
                    self.settings[k] = float(v.get())
                except ValueError:
                    pass
            win.destroy()
        tk.Button(win, text="OK", command=ok, width=8).grid(row=len(rows), column=0,
                                                            columnspan=2, pady=8)

    # ----- analysis
    def _current_sweep_field(self, panel):
        app = self.app
        si = max(0, min(len(app.volume.sweeps) - 1, app.sweep_index.get()))
        sweep = app.volume.sweeps[si]
        fi = panel.current_field_info()
        if fi is None or fi[0] not in sweep.fields:
            raise ValueError("The clicked panel has no field in this sweep.")
        if getattr(sweep, "scan_type", "ppi") == "rhi":
            raise ValueError("Axisymmetric patch analysis needs a PPI sweep.")
        return si, sweep, fi[0]

    @staticmethod
    def _window(sweep, vel, xc, yc, half_km):
        """Crop to the rays/gates within half_km of (xc, yc)."""
        az = np.asarray(sweep.azimuth, float) % 360.0
        rng = np.asarray(sweep.ranges, float) / 1000.0
        rc = math.hypot(xc, yc)
        gate_ok = np.abs(rng - rc) <= half_km + 0.5
        if rc <= half_km:
            ray_ok = np.ones(len(az), bool)
        else:
            lim = math.degrees(math.asin(min(1.0, (half_km + 0.5) / rc))) + 1.0
            bc = math.degrees(math.atan2(xc, yc)) % 360.0
            ray_ok = np.abs((az - bc + 180.0) % 360.0 - 180.0) <= lim
        if gate_ok.sum() < 3 or ray_ok.sum() < 3:
            return None
        return vel[np.ix_(ray_ok, gate_ok)], az[ray_ok], rng[gate_ok]

    def _run(self, panel, xc, yc, extrema=None):
        app, st = self.app, self.settings
        si, sweep, fname = self._current_sweep_field(panel)
        vel_full = np.ma.filled(np.ma.asarray(sweep.fields[fname], float), np.nan)
        R = max(float(st["radius_km"]), 0.05)
        half = 7.5 * R
        if extrema is not None:
            half = max(half, 3.5 * math.hypot(extrema[1][0] - extrema[0][0],
                                              extrema[1][1] - extrema[0][1]))
        crop = self._window(sweep, vel_full, xc, yc, half)
        if crop is None:
            raise ValueError("Clicked point is outside the sweep's data.")
        vel, az, rng = crop

        patch = compute_dreader_patch(
            vel, az, rng, (xc, yc), R, beamwidth_deg=st["beamwidth"], numdiam=1.0,
            storm_speed_ms=st["storm_speed"], storm_dir_deg=st["storm_dir"],
            air_density=st["air_density"], extrema=extrema)
        if patch is None:
            messagebox.showinfo("Axisymmetric patch",
                                "No inbound/outbound velocity pair inside the patch circle "
                                f"(radius {R:g} km). Click closer to the couplet or increase "
                                "the patch radius in Settings.")
            return
        mid = (0.5 * (patch["min"]["x"] + patch["max"]["x"]),
               0.5 * (patch["min"]["y"] + patch["max"]["y"]))
        b = math.radians(st["storm_dir"])
        trans = ((st["storm_speed"] * math.sin(b), st["storm_speed"] * math.cos(b))
                 if st["storm_speed"] > 0 else None)
        elev = float(np.nanmedian(sweep.elevation)) if sweep.elevation is not None else 0.0
        # ring width = the COARSER of the angular and gate spacing so every ring holds enough gates
        rc = math.hypot(*mid)
        d_az = float(np.median(np.abs(np.diff(np.unwrap(np.radians(np.sort(az)))))))
        delta = max(rc * d_az, float(np.median(np.diff(rng))))
        axi = compute_axisymmetric(vel, az, rng, mid, elev_deg=elev, delta_km=delta,
                                   max_radius_km=3.0 * patch["diam_km"], translation=trans,
                                   air_density=st["air_density"])

        self.result = {"patch": patch, "axisym": axi, "field": fname, "sweep": si,
                       "center_click": (xc, yc), "center": mid}
        self._overlay = self._overlay_geometry(patch, axi, mid)
        self._pts = {"min": (patch["min"]["x"], patch["min"]["y"]),
                     "max": (patch["max"]["x"], patch["max"]["y"])}
        self._panel = panel
        self._key = (id(app.volume), int(si))
        self._redraw_all()
        self._show_results(fname)

    @staticmethod
    def _overlay_geometry(patch, axi, mid):
        p_min = (patch["min"]["x"], patch["min"]["y"])
        p_max = (patch["max"]["x"], patch["max"]["y"])
        half = 0.5 * math.hypot(p_max[0] - p_min[0], p_max[1] - p_min[1])
        segs, labels = [], []
        if axi and axi.get("rings"):
            rings = axi["rings"]
            step = max(1, int(math.ceil(len(rings) / float(MAX_DRAWN_RINGS))))
            cx, cy = axi["center_km"]
            ang = math.radians(225.0)
            for q in rings[::step]:
                r = q["radius_km"]
                segs.append(_dashed_circle(cx, cy, r))
                labels.append((cx + r * math.sin(ang), cy + r * math.cos(ang),
                               f"{r:.2f}".rstrip("0").rstrip(".")))
        return {
            "ring_segments": np.vstack(segs) if segs else None,
            "labels": labels,
            "couplet": _circle_strip(mid[0], mid[1], half),
            "bar": _bar(p_min, p_max),
            "points": np.array([[*p_min, 0.0], [*p_max, 0.0]], dtype="float32"),
        }

    # ----- dragging the min / max markers
    def hit_test(self, x_km, y_km, tol_km):
        """Which marker ('min'/'max') is within tol_km of the point, else None."""
        if self._pts is None or self._overlay is None or self._key != (
                id(self.app.volume), int(self.app.sweep_index.get())):
            return None
        best, bd = None, tol_km
        for k, (px, py) in self._pts.items():
            d = math.hypot(px - x_km, py - y_km)
            if d <= bd:
                best, bd = k, d
        return best

    def begin_drag(self, which):
        self._drag = which

    def drag_to(self, x_km, y_km):
        """Live preview while dragging (points, bar, couplet circle only)."""
        if self._drag is None:
            return
        import time
        now = time.monotonic()
        self._pts[self._drag] = (x_km, y_km)
        if now - self._drag_t < 0.03:
            return
        self._drag_t = now
        p1, p2 = self._pts["min"], self._pts["max"]
        mid = (0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1]))
        half = 0.5 * math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        self._overlay["couplet"] = _circle_strip(mid[0], mid[1], half)
        self._overlay["bar"] = _bar(p1, p2)
        self._overlay["points"] = np.array([[*p1, 0.0], [*p2, 0.0]], dtype="float32")
        for panel in self.app.panels:
            cv = self._visuals.get(id(panel.ppi))
            if cv is not None:
                cv.draw_light(self._overlay)
                panel.ppi._vcanvas.update()

    def end_drag(self):
        """Mouse released: recompute everything with the user-placed points."""
        if self._drag is None:
            return
        self._drag = None
        p1, p2 = self._pts["min"], self._pts["max"]
        mid = (0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1]))
        try:
            self._run(self._panel, mid[0], mid[1], extrema=(p1, p2))
        except Exception:
            import traceback
            tb = traceback.format_exc()
            print(tb)
            messagebox.showerror("Axisymmetric patch", f"Analysis failed:\n\n{tb}")

    # ----- drawing on every panel
    def _redraw_all(self):
        for panel in self.app.panels:
            ppi = panel.ppi
            cv = self._visuals.get(id(ppi))
            if cv is None:
                cv = self._visuals[id(ppi)] = _CanvasVisuals(ppi._view.scene)
            if self._overlay is not None and self._key == (id(self.app.volume),
                                                           int(self.app.sweep_index.get())):
                cv.draw(self._overlay)
            else:
                cv.hide()
            ppi._vcanvas.update()

    # ----- results window
    def _show_results(self, fname):
        r = self.result
        p, a = r["patch"], r["axisym"]
        L = []
        L.append(f"AXISYMMETRIC PATCH  (field {fname}, sweep {r['sweep']})")
        L.append(f"Click ({r['center_click'][0]:.2f}, {r['center_click'][1]:.2f}) km   "
                 f"ring center ({r['center'][0]:.2f}, {r['center'][1]:.2f}) km")
        L.append("")
        mn, mx = p["min"], p["max"]
        L.append(f"min {mn['v']:.1f} m/s @ az {mn['az']:.1f} deg, {mn['rng_km']:.2f} km")
        L.append(f"max {mx['v']:.1f} m/s @ az {mx['az']:.1f} deg, {mx['rng_km']:.2f} km")
        L.append(f"Maxima diameter        {p['diam_km'] * 1000:8.0f} m   axis {p['axis_deg']:.0f} deg")
        L.append(f"Mean maxima radius     {p['mean_radius_km'] * 1000:8.0f} m")
        L.append(f"Aspect correction      {p['aspect_corr']:8.3f}")
        L.append(f"Corrected min / max    {p['min_adj']:.1f} / {p['max_adj']:.1f} m/s")
        L.append(f"Mean / asym / dV       {p['v_mean']:.1f} / {p['v_asym']:.1f} / {p['delta_v']:.1f} m/s")
        L.append(f"Ground-rel min / max   {p['min_ground']:.1f} / {p['max_ground']:.1f} m/s  "
                 f"(corr {p['min_ground_adj']:.1f} / {p['max_ground_adj']:.1f})")
        L.append(f"Circulation            {p['circulation_m2s']:.3g} m^2/s")
        L.append(f"Pressure deficit       {p['delta_p_pa'] / 100.0:.1f} hPa")
        L.append(f"Radius of max vorticity{p['max_vort_radius_km'] * 1000:8.0f} m")
        Lg = p["largest"]
        keys = list(Lg)
        L.append("Largest speed (Dop/corr/grnd/grnd+corr): " +
                 " / ".join(f"{Lg[k]:.0f}" for k in keys) + " m/s")
        L.append("F-scale:  " + " / ".join(f"{p['fscale'][k]:.1f}" for k in keys))
        L.append("EF-scale: " + " / ".join(f"{p['efscale'][k]:.1f}" for k in keys) +
                 "   (EF table from the DREADER source; equivalent intensity, not a damage rating)")
        L.append("")
        L.append("Fractions of mean radius:   R/Rm   r(m)   vort(1/s)   div(1/s)   stretch   rV(m2/s)")
        for lv in p["levels"]:
            f = lambda x: f"{x:10.2e}" if x == x else "        --"
            L.append(f"                          {lv['fraction']:5.2f} {lv['radius_km'] * 1000:6.0f} "
                     f"{f(lv['vvort'])} {f(lv['hdiv'])} {f(lv['stretch'])} {f(lv['angmom'])}")
        if a:
            L.append("")
            L.append(f"Axisymmetric rings ({len(a['rings'])}, width {a['delta_km'] * 1000:.0f} m): "
                     f"max |Vt| {abs(a['vt_max']):.1f} m/s at r = {a['r_vt_max_km'] * 1000:.0f} m")
            L.append("  r(km)    n     Vr      Vt     |V|    circ(1e3)  dp(hPa)")
            step = max(1, len(a["rings"]) // 30)
            for q in a["rings"][::step]:
                L.append(f"  {q['radius_km']:5.2f} {q['n']:4d} {q['vr']:7.1f} {q['vt']:7.1f} "
                         f"{q['vtot']:7.1f} {q['circ'] / 1e3:9.1f} {q['dp_pa'] / 100.0:8.1f}")
        else:
            L.append("")
            L.append("Ring fit: too few valid samples around the couplet.")

        if self._win is None or not self._win.winfo_exists():
            self._win = tk.Toplevel(self.app)
            self._win.title("Axisymmetric patch results")
            self._win.geometry("760x640")
            bar = tk.Frame(self._win)
            bar.pack(fill="x")
            tk.Button(bar, text="Save ring profile CSV…", command=self._save_csv).pack(side="left", padx=6, pady=4)
            tk.Button(bar, text="Clear overlay", command=self.clear).pack(side="left", padx=6, pady=4)
            self._text = tk.Text(self._win, font=("Courier", 9), wrap="none")
            self._text.pack(fill="both", expand=True)
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.insert("1.0", "\n".join(L))
        self._text.configure(state="disabled")
        self._win.lift()

    def _save_csv(self):
        if not self.result or not self.result["axisym"]:
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                            title="Save axisymmetric ring profile")
        if path:
            axisym_to_csv(self.result["axisym"], path)
