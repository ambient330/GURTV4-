# SPDX-License-Identifier: GPL-2.0-only
# GURT - Copyright (C) the GURT authors.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the
# Free Software Foundation. It is distributed WITHOUT ANY WARRANTY; see the
# file LICENSE for details.
"""DREADER cross-section tool (radial / angular) as a PPI overlay.

Analysis > "Cross-section: radial..." / "Cross-section: angular..." then click
the PPI (or use the right-click menu):

  radial   -- the ray nearest the click; every gate along it is extracted.
  angular  -- the arc at the clicked range, centered on the clicked ray and
              extending +/- the half-length from Settings (0 = full circle).
              For a velocity field the DREADER vortex dynamics (shear /
              curvature / vertical vorticity, angular momentum, circulation,
              pressure deficit) are computed along the arc.

The line / arc is drawn on every panel; a results window shows a plot, a
summary, and saves the cross-section to CSV in DREADER-style columns.

Scene frame: km, x east, y north, radar at the origin.
"""

import logging as _logging
_logging.getLogger("matplotlib.font_manager").setLevel(_logging.ERROR)   # a missing font in the fallback list is not an error
import base64
import io
import math
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from vispy.scene import visuals

from dreader_xsect import (radial_xsection, angular_xsection, xsection_to_csv,
                           is_velocity_field)

LINE_C = (0.0, 0.0, 0.0, 1.0)


def _radial_line(az_deg, r_km):
    a = math.radians(az_deg)
    return np.array([[0.0, 0.0, 0.0], [r_km * math.sin(a), r_km * math.cos(a), 0.0]],
                    dtype="float32")


def _arc(az_c, r_km, half_deg, full, n=181):
    if full:
        th = np.radians(np.linspace(0.0, 360.0, 361))
    else:
        th = np.radians(np.linspace(az_c - half_deg, az_c + half_deg, n))
    return np.column_stack([r_km * np.sin(th), r_km * np.cos(th),
                            np.zeros(th.size)]).astype("float32")


class _CanvasVisuals:
    def __init__(self, scene):
        self.line = visuals.Line(connect="strip", color=LINE_C, width=2.0, parent=scene)
        self.pts = visuals.Markers(parent=scene)
        self.hide()

    def hide(self):
        self.line.visible = False
        self.pts.visible = False

    def draw(self, ov):
        self.line.set_data(pos=ov["line"])
        self.pts.set_data(pos=ov["points"], face_color=[(0, 0, 0, 1)] * len(ov["points"]),
                          edge_color=[(1, 1, 1, 1)] * len(ov["points"]), size=7)
        self.line.visible = True
        self.pts.visible = True


class XsectTool:
    def __init__(self, app):
        self.app = app
        self.picking = None                 # None / 'radial' / 'angular'
        self.settings = {"half_km": 5.0, "air_density": 1.0}
        self._visuals = {}
        self._overlay = None
        self._key = None
        self.result = None
        self._win = None
        self._img = None

    # ----- menu actions
    def begin_pick(self, kind):
        if self.app.volume is None:
            messagebox.showinfo("Cross-section", "Open a radar file first.")
            return
        self.picking = kind
        what = ("click on the ray to extract" if kind == "radial"
                else "click the arc center (ray + range)")
        self.app.cursor_var.set(f"{kind.capitalize()} cross-section: {what}  (Esc / right-click to cancel)")
        self.app.bind("<Escape>", self._cancel_pick, add="+")

    def _cancel_pick(self, *_):
        if self.picking:
            self.picking = None
            self.app.cursor_var.set("Cursor: —")

    def run_at(self, panel, x_km, y_km, kind=None):
        kind = kind or self.picking or "angular"
        self.picking = None
        self.app.cursor_var.set("Cursor: —")
        try:
            self._run(panel, float(x_km), float(y_km), kind)
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            messagebox.showerror("Cross-section", f"Cross-section failed:\n\n{e}")

    def clear(self):
        self._overlay = None
        self.result = None
        self._key = None
        for cv in self._visuals.values():
            cv.hide()
        self._redraw_all()

    def sync_visibility(self, sweep_index):
        if self._overlay is None:
            return
        self._redraw_all()

    # ----- settings
    def open_settings(self):
        win = tk.Toplevel(self.app)
        win.title("Cross-section settings")
        win.resizable(False, False)
        win.configure(bg="#1a1a2e")
        rows = [("Angular half-length (km, 0 = full circle)", "half_km"),
                ("Air density (kg/m^3, for pressure deficit)", "air_density")]
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
    def _run(self, panel, x_km, y_km, kind):
        app, st = self.app, self.settings
        si = max(0, min(len(app.volume.sweeps) - 1, app.sweep_index.get()))
        sweep = app.volume.sweeps[si]
        if getattr(sweep, "scan_type", "ppi") == "rhi":
            raise ValueError("Cross-sections need a PPI sweep.")
        fi = panel.current_field_info()
        fname = fi[0] if fi and fi[0] in sweep.fields else next(iter(sweep.fields))
        az = math.degrees(math.atan2(x_km, y_km)) % 360.0
        rng_m = math.hypot(x_km, y_km) * 1000.0

        if kind == "radial":
            xs = radial_xsection(sweep, az)
            r_km = float(np.nanmax(xs["range_m"])) / 1000.0
            line = _radial_line(xs["azimuth"], r_km)
            pts = np.array([[0.0, 0.0, 0.0]], dtype="float32")
        else:
            vel = fname if is_velocity_field(fname) else next(
                (n for n in sweep.fields if is_velocity_field(n)), None)
            xs = angular_xsection(sweep, az, rng_m, max_radius_m=st["half_km"] * 1000.0,
                                  velocity_field=vel, air_density=st["air_density"])
            r_km = xs["range_m"] / 1000.0
            line = _arc(xs["azimuth"], r_km, xs["half_deg"], xs["full_circle"])
            a = math.radians(xs["azimuth"])
            pts = np.array([[r_km * math.sin(a), r_km * math.cos(a), 0.0], line[0], line[-1]],
                           dtype="float32")
        self.result = {"xs": xs, "field": fname, "sweep": si}
        self._overlay = {"line": line, "points": pts}
        self._key = (id(app.volume), int(si))
        self._redraw_all()
        self._show_results()

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
    def _show_results(self):
        r = self.result
        xs = r["xs"]
        if self._win is None or not self._win.winfo_exists():
            self._win = tk.Toplevel(self.app)
            self._win.title("Cross-section")
            self._win.geometry("760x940")
            bar = tk.Frame(self._win)
            bar.pack(fill="x")
            tk.Button(bar, text="Save CSV…", command=self._save_csv).pack(side="left", padx=6, pady=4)
            tk.Button(bar, text="Clear overlay", command=self.clear).pack(side="left", padx=6, pady=4)
            tk.Label(bar, text="Plot field:").pack(side="left", padx=(14, 2))
            self._plot_var = tk.StringVar()
            self._menu = tk.OptionMenu(bar, self._plot_var, "")
            self._menu.pack(side="left")
            self._img_lbl = tk.Label(self._win)
            self._img_lbl.pack()
            self._text = tk.Text(self._win, font=("Courier", 9), wrap="none", height=14)
            self._text.pack(fill="both", expand=True)
        names = list(xs["fields"])
        menu = self._menu["menu"]
        menu.delete(0, "end")
        for n in names:
            menu.add_command(label=n, command=lambda v=n: (self._plot_var.set(v), self._plot()))
        self._plot_var.set(r["field"] if r["field"] in names else names[0])
        self._plot()
        self._summary()
        self._win.lift()

    def _summary(self):
        r = self.result
        xs = r["xs"]
        L = []
        if xs["kind"] == "radial":
            L.append(f"RADIAL CROSS-SECTION  (sweep {r['sweep']})")
            L.append(f"Ray {xs['ray']}   azimuth {xs['azimuth']:.2f} deg   "
                     f"elevation {xs['elevation']:.2f} deg")
            L.append(f"Gates {len(xs['range_m'])}   range {xs['range_m'][0] / 1000:.2f}"
                     f" - {xs['range_m'][-1] / 1000:.2f} km")
            L.append("")
            L.append(f"{'field':<10}{'n good':>8}{'min':>10}{'max':>10}{'mean':>10}")
            for n, v in xs["fields"].items():
                g = v[np.isfinite(v)]
                if g.size:
                    L.append(f"{n:<10}{g.size:>8}{g.min():>10.2f}{g.max():>10.2f}{g.mean():>10.2f}")
                else:
                    L.append(f"{n:<10}{0:>8}")
        else:
            L.append(f"ANGULAR CROSS-SECTION  (sweep {r['sweep']})")
            L.append(f"Center ray {xs['center_ray']}  azimuth {xs['azimuth']:.2f} deg   "
                     f"range {xs['range_m'] / 1000:.3f} km   elevation {xs['elevation']:.2f} deg")
            span = ("full circle" if xs["full_circle"]
                    else f"+/- {xs['half_deg']:.2f} deg  ({xs['arclen_m'].max() / 1000:.2f} km)")
            L.append(f"Rays {len(xs['rays'])}   arc {span}")
            if xs.get("velocity_field"):
                vf = xs["velocity_field"]
                v = xs["fields"][vf]
                g = np.isfinite(v)
                L.append("")
                L.append(f"Dynamics from {vf}")
                if g.any():
                    L.append(f"  V min / max           {np.nanmin(v):.1f} / {np.nanmax(v):.1f} m/s")
                L.append(f"  Chord to max |V|      {xs['vmax_chord_m']:.0f} m")
                for label, key, fmt in (("Vertical vorticity", "vert_vort", "{:.4f}"),
                                        ("Shear vorticity", "shear_vort", "{:.4f}"),
                                        ("Curvature vorticity", "curv_vort", "{:.4f}")):
                    a = xs[key]
                    if np.isfinite(a).any():
                        L.append(f"  {label:<22}max {fmt.format(np.nanmax(a))}   "
                                 f"min {fmt.format(np.nanmin(a))} 1/s")
                if np.isfinite(xs["circulation"]).any():
                    L.append(f"  Circulation max       {np.nanmax(xs['circulation']):.4g} m^2/s")
                    L.append(f"  Pressure deficit max  {np.nanmax(xs['pressure_def']) / 100:.2f} hPa")
            else:
                L.append("(no velocity field in this sweep: dynamics skipped)")
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.insert("1.0", "\n".join(L))
        self._text.configure(state="disabled")

    # Same look as TURT's azimuthal cross-section plots: Times bold, boxed black
    # axes, black horizontal rules, legend centered above the axes, red "^" data line.
    _FONT = {"family": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
             "weight": "bold"}

    def _style_axis(self, ax, ylabel, xlabel=None, title=None, details="", snap=None,
                    legend=True, ncol=2):
        import matplotlib.ticker as mticker
        fnt = self._FONT
        ax.set_facecolor("white")
        for sp in ax.spines.values():
            sp.set_color("black")
            sp.set_linewidth(1.0)
        ax.grid(False)
        ax.yaxis.grid(True, color="black", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", direction="out", length=4, width=1.0,
                       labelsize=8.5, colors="black")
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_fontfamily(fnt["family"])
            lab.set_fontweight("bold")
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=9, **fnt)
        ax.set_ylabel(ylabel, fontsize=9, **fnt)
        if snap == "nonneg":
            ax.set_ylim(0, 10.0 * np.ceil(ax.get_ylim()[1] / 10.0))
            ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
        elif snap == "sym":
            lo, hi = ax.get_ylim()
            ax.set_ylim(10.0 * np.floor(lo / 10.0), 10.0 * np.ceil(hi / 10.0))
            ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
        if legend:
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol, frameon=False,
                      prop={"family": fnt["family"], "weight": "bold", "size": 9},
                      handlelength=1.8, columnspacing=2.5)
        if title:
            ax.set_title(title, fontsize=10.5, pad=24, **fnt)
        if details:
            ax.text(0.99, 0.02, details, transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=8, color="black",
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.5), **fnt)

    def _plot(self):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        xs = self.result["xs"]
        fname = self._plot_var.get()
        vals = np.asarray(xs["fields"].get(fname), float)
        dyn = xs["kind"] == "angular" and xs.get("velocity_field")
        nrows = 3 if dyn else 1
        fig = Figure(figsize=(9.0, 2.7 * nrows if dyn else 4.6), dpi=80)
        fig.patch.set_facecolor("white")
        FigureCanvasAgg(fig)
        axes = fig.subplots(nrows, 1, sharex=True) if nrows > 1 else [fig.subplots()]
        ax = axes[0]

        def line(a, y, color, marker, ms, lw, label):
            a.plot(x, y, color=color, marker=marker, markersize=ms, linewidth=lw, label=label)

        if xs["kind"] == "radial":
            x = xs["range_m"] / 1000.0
            xlabel = "Range (km)"
            title = f"Radial Cross Section of {fname}"
            details = (f"azimuth = {xs['azimuth']:.1f}\u00b0   elevation = {xs['elevation']:.1f}\u00b0")
        else:
            x = xs["arcpos_m"]
            xlabel = "Azimuthal Distance (m)"
            title = f"Azimuthal Cross Section of {fname}"
            details = (f"range = {xs['range_m'] / 1000:.2f} km   azimuth = {xs['azimuth']:.1f}\u00b0"
                       f"   rays = {len(xs['rays'])}")
        line(ax, vals, "red", "^", 4, 1.0, fname)
        ax.set_xlim(np.nanmin(x), np.nanmax(x))
        g = vals[np.isfinite(vals)]
        snap = None
        if g.size and np.max(np.abs(g)) >= 20.0:
            snap = "nonneg" if g.min() >= 0 else "sym"
        if not dyn:
            self._style_axis(ax, fname, xlabel, title, details, snap, ncol=1)
        else:
            self._style_axis(ax, fname, None, title, details, snap, ncol=1)
            line(axes[1], xs["shear_vort"], "navy", "s", 3, 1.2, "Shear")
            line(axes[1], xs["curv_vort"], "darkorange", "D", 3, 1.2, "Curvature")
            line(axes[1], xs["vert_vort"], "darkgreen", "o", 3, 1.2, "Vertical")
            axes[1].axhline(0.0, color="gray", linewidth=0.7, linestyle=":")
            self._style_axis(axes[1], "Vorticity (1/s)", None, ncol=3)
            line(axes[2], xs["circulation"] / 1e3, "purple", "v", 3, 1.2,
                 "Circulation (1e3 m$^2$/s)")
            ax2 = axes[2].twinx()
            ax2.plot(x, xs["pressure_def"] / 100.0, color="teal", marker="v", markersize=3,
                     linewidth=1.2, label="Pressure deficit (hPa)")
            self._style_axis(axes[2], "Circulation (1e3 m$^2$/s)", xlabel, legend=False)
            self._style_axis(ax2, "Pressure Deficit (hPa)", None, legend=False)
            ax2.yaxis.grid(False)
            h1, l1 = axes[2].get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            axes[2].legend(h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                           ncol=2, frameon=False, handlelength=1.8, columnspacing=2.5,
                           prop={"family": self._FONT["family"], "weight": "bold", "size": 9})
        fig.subplots_adjust(left=0.10, right=0.90 if dyn else 0.97,
                            top=0.86 if not dyn else 0.92, bottom=0.11 if not dyn else 0.07,
                            hspace=0.55)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor="white")
        self._img = tk.PhotoImage(data=base64.b64encode(buf.getvalue()))
        self._img_lbl.configure(image=self._img)

    def _save_csv(self):
        if not self.result:
            return
        xs = self.result["xs"]
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")],
            initialfile=f"{xs['kind']}_xsect.csv", title="Save cross-section")
        if path:
            xsection_to_csv(xs, path)
