#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# GURT - Copyright (C) the GURT authors.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the
# Free Software Foundation. It is distributed WITHOUT ANY WARRANTY; see the
# file LICENSE for details.
"""
gurt_gui.py — Importable GUI module for GURT Vispy.

Exports:
    GURTVispy          — main Tk application window
    PPICanvas          — VisPy PPI canvas widget
    ParamColorDialog   — field / colormap editor dialog  (from gurt_guiv3)
    EditorPanel        — floating brush editor panel     (from gurt_guiv3)
    DuplicateFieldDialog — field duplication dialog      (from gurt_guiv3)
    CfRadialExportDialog — CfRadial export dialog        (from gurt_guiv3)

    # colormap utilities (re-exported for convenience)
    GURT_CMAPS, ALL_CMAP_NAMES, resolve_cmap, resolve_mpl_cmap
    FIELD_DEFAULTS
"""

import math
import os
import threading
from datetime import datetime
from fractions import Fraction

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# Times New Roman bold for the plot colorbar only (applied via rc_context below);
# falls back through look-alike serif faces if it isn't installed.
_CB_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Liberation Serif", "Nimbus Roman",
                   "TeX Gyre Termes", "FreeSerif", "DejaVu Serif"],
    "font.weight": "bold",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
}
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image as PILImage

import os
import platform

import subprocess, shutil


def _is_wsl():
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except FileNotFoundError:
        return False
    
def ensure_mesa_glx():
    try:
        import ctypes
        ctypes.CDLL("libGL.so.1")
    except OSError:
        if shutil.which("apt"):
            subprocess.run(
                ["sudo", "apt", "install", "-y", "libgl1-mesa-glx", "libgl1-mesa-dri"],
                check=False
            )

if platform.system() == "Linux" and _is_wsl():
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    os.environ["PYOPENGL_PLATFORM"] = "glx"  # force GLX, PyOpenGL defaults to EGL on WSL

import vispy
vispy.use("tkinter")
from vispy import scene
from vispy.scene import SceneCanvas, visuals
from vispy.geometry import Rect
from vispy.gloo import VertexBuffer
from vispy.visuals.transforms import BaseTransform, STTransform

try:
    import pyart
    PYART_AVAILABLE = True
except ImportError:
    PYART_AVAILABLE = False

try:
    from PIL import ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

import os as _os, sys as _sys
_GUI_DIR = _os.path.dirname(_os.path.abspath(__file__))     # .../GURTv4/gui   (this file's folder)
_ROOT_DIR = _os.path.dirname(_GUI_DIR)                      # .../GURTv4       (edit/, readers/, exporters/ ...)
for _p in (_GUI_DIR, _ROOT_DIR):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

# Brush tools and editing functions live in their own modules so this file
# doesn't keep growing — see brushes.py and edit/ for the implementations.
# (gui/ and the GURTv4 root are put on sys.path by the guard above)
import brushes
from edit import geometry as edit_geometry
from edit import scaling as edit_scaling
from edit import geolocation as edit_geolocation
from edit import despeckle as edit_despeckle
from edit.gatefilter import GateFilterDialog, apply_gate_filter, CONDITIONS
from xsect_overlay import XsectTool

# Vortex tracking and the axisymmetric patch are optional: when their modules are
# not present (this release leaves them out, or GURT_LITE=1 is set)
# the menu entries stay and do nothing but report that they are not included.
try:
    if _os.environ.get("GURT_LITE"):
        raise ImportError("GURT_LITE")
    from axisym_overlay import AxisymTool
except ImportError:
    AxisymTool = None
try:
    if _os.environ.get("GURT_LITE"):
        raise ImportError("GURT_LITE")
    from detect_overlay import DetectTool
except ImportError:
    DetectTool = None


class _PlaceholderTool:
    """Stands in for a tool that is not part of this build.

    Every method is a no-op that only puts a note in the status bar; the flags
    the canvas checks (picking / dragging / box_picking ...) are always off.
    """
    picking = False
    box_picking = False
    _box_start = None
    _drag = None
    _overlay = None

    def __init__(self, app, label):
        self._app = app
        self._label = label
        self.settings = {"swath_smooth_m": 200.0}      # read by the smoothed |V| dialog

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        def _noop(*_a, **_k):
            try:
                self._app.cursor_var.set(f"{self._label} is not included in this build.")
            except Exception:
                pass
            return None
        return _noop

# Editor "paint" tools that operate on gates — either by dragging a
# circular brush across them (PPICanvas._on_press et al.) or by clicking
# out a polygon boundary and applying once (PPICanvas._add_boundary_point /
# GURTVispy._apply_boundary_edit). Kept as one shared tuple so the brush
# code and the boundary code never drift out of sync about which editor
# modes are "paint" modes vs. plain zoom/pan.
#
# "gatefilter_brush" is the area-limited sibling of the Edit ▸ GateFilter…
# dialog: same condition-matching idea, but scoped to whatever's under the
# brush/boundary instead of the whole sweep, and able to combine several
# conditions (across possibly different fields) with AND/OR instead of
# just one.
EDIT_BRUSH_MODES = ("unfold_brush", "deglitch_brush", "gradient_brush",
                    "delete_gate", "gatefilter_brush")

# Plain-text (no emoji) labels for the modes above, keyed the same way --
# used wherever a mode name needs to show up in a sentence rather than a
# radio button (e.g. BatchBoundaryEditDialog's summary text).
EDIT_MODE_LABELS = {
    "zoom":             "Zoom / Pan",
    "unfold_brush":     "Unfold Brush",
    "deglitch_brush":   "Deglitch Brush",
    "gradient_brush":   "Gap Fill Brush",
    "delete_gate":      "Delete Gates",
    "gatefilter_brush": "GateFilter Brush",
}


# ══════════════════════════════════════════════════════════════════════════════
# Colormaps
# ══════════════════════════════════════════════════════════════════════════════

def _make_cmap(name, hex_colors):
    rgb = [mcolors.hex2color(h) for h in hex_colors]
    return mcolors.LinearSegmentedColormap.from_list(name, rgb[::-1])


ZDR_CMAP = _make_cmap("gurt_zdr", [
    '#922E97','#B26CB6','#C998CB','#E2C7E3','#FEF9FB','#F8BEDB','#EF77B2',
    '#CF3B58','#B00000','#C80603','#DE1A0B','#EE8836','#FEF861','#5ADE64',
    '#3FE2CF','#2474B4','#0B0D9C','#D1D1D9','#7F6CA2','#453B58','#292335','#060507'])

CC_CMAP = _make_cmap("gurt_cc", [
    '#8B1E4D','#E41000','#FC7F00','#FFB600','#FFFB00','#BCE906','#87D70B',
    '#61ED6E','#719CD2','#5151E8','#2929D1','#0A0ABD','#0C0CAC','#0D0D9C',
    '#0F0F8C','#1C1C9E','#2D2D84','#404068','#454561','#4F4F4F'])

KDP_CMAP = _make_cmap("gurt_kdp", [
    '#C361F9','#6F329A','#160234','#624264','#B18596','#FAC4C5','#FF7B00',
    '#FFBC00','#FEFF00','#84DA1A','#16BA31','#3ADB94','#60FEF6','#74C7D1',
    '#8987A2','#9B507A','#EA77B8','#CE5B93','#B03D6A','#921F42','#75021B',
    '#62000E','#4B0101','#4B2828','#4B4A4A','#5F5F5F','#757575'])

SW_CMAP = _make_cmap("gurt_sw", [
    '#02A0C8','#2CA7C6','#53AEC5','#78B4C3','#9FBBC1','#C1C1C1','#DCDCDC',
    '#E6E6E6','#F2F2F2','#FFFD01','#FDC60F','#FDB313','#FC991A','#F7742D',
    '#EF6341','#E54F5B','#DE406D','#B73192','#7D26BD','#31148A','#1A0855'])

VEL_CMAP = _make_cmap("gurt_vel", [
    '#F2465B','#E0365B','#CF2646','#B71630','#9F0100','#A05060','#7C544C',
    '#885840','#966440','#A5703E','#B37B36','#C1872E','#D09326','#DE9F1E',
    '#ECAB16','#EFB70E','#EFC306','#EFCF00','#ECE4B0','#D0E4D0','#BFDCBF',
    '#A1D0A1','#85C485','#67B867','#4AAD4A','#2DA12D','#0F950F','#007D00',
    '#017100','#30855F','#48969A','#718FFE','#6A60FE','#5A24E5','#4A0EC3',
    '#3A0EAB','#2A0E94','#673A8F','#7805A3'])

VEL2_CMAP = _make_cmap("sam_vel", [
    '#2d0000',  # 200
    '#3c0000',  # 140
    '#610602',  # 120
    '#fe8950',  # 80
    '#ff9756',  # 60 (first color)
    '#ffe6a9',  # 60 (second color)
    '#ffddb0',  # 55 (first color)
    '#ff9dce',  # 55 (second color)
    '#ff8ed4',  # 40 (first color)
    '#f93a54',  # 40 (second color)
    '#f20106',  # 10 (first color)
    '#690000',  # 10 (second color)
    '#7a3039',  # 0 (first color)
    '#826a78',  # 0 (second color)
    '#6a7d69',  # -10 (first color)
    '#487047',  # -10 (second color)
    '#0f6314',  # -40 (first color)
    '#0af823',  # -40 (second color)
    '#21fd32',  # -50 (first color)
    '#b4f0f3',  # -50 (second color)
    '#aceff2',  # -70 (first color)
    '#37e2e5',  # -70 (second color)
    '#2fd7e1',  # -90 (first color)
    '#19018e',  # -90 (second color)
    '#20018d',  # -100 (first color)
    '#69028e',  # -100 (second color)
    '#72038d',  # -120 (first color)
    '#fa0482',  # -120 (second color)
    '#ff14b4',  # -140
    '#ffdcdc',  # -200
])

REF_CMAP = _make_cmap("gurt_ref", [
    '#F2465B','#E0365B','#CF2646','#B71630','#9F0100','#A05060','#7C544C',
    '#885840','#966440','#A5703E','#B37B36','#C1872E','#D09326','#DE9F1E',
    '#ECAB16','#EFB70E','#EFC306','#EFCF00','#ECE4B0','#D0E4D0','#BFDCBF',
    '#A1D0A1','#85C485','#67B867','#4AAD4A','#2DA12D','#0F950F','#007D00',
    '#017100','#30855F','#48969A','#718FFE','#6A60FE','#5A24E5','#4A0EC3',
    '#3A0EAB','#2A0E94','#673A8F','#7805A3'])

GURT_CMAPS = {
    "gurt_ref": REF_CMAP,
    "gurt_vel": VEL_CMAP,
    "sam_vel": VEL2_CMAP,
    "gurt_zdr": ZDR_CMAP,
    "gurt_kdp": KDP_CMAP,
    "gurt_sw":  SW_CMAP,
    "gurt_cc":  CC_CMAP,
}

MPL_CMAPS = [
    "viridis","plasma","inferno","magma","cividis",
    "RdBu_r","RdYlBu_r","Spectral_r","coolwarm","bwr",
    "jet","rainbow","turbo","nipy_spectral",
    "Greys","hot","bone","copper","pink",
    "PuOr","PRGn","BrBG","PiYG",
    "YlOrRd","YlOrBr","OrRd","Reds","Blues","Greens","Purples",
]

PYART_CMAPS = []
if PYART_AVAILABLE:
    try:
        for _mod in (getattr(pyart.graph, 'cm', None),
                     getattr(pyart.graph, 'cm_tables', None)):
            if _mod is None:
                continue
            for _name in dir(_mod):
                if not _name.startswith('_') and f"pyart_{_name}" not in PYART_CMAPS:
                    PYART_CMAPS.append(f"pyart_{_name}")
    except Exception:
        pass

ALL_CMAP_NAMES = list(GURT_CMAPS.keys()) + MPL_CMAPS + PYART_CMAPS


def resolve_cmap(name_or_obj):
    """Return a matplotlib Colormap from a name or object (pyart-aware)."""
    if not isinstance(name_or_obj, str):
        return name_or_obj
    if name_or_obj in GURT_CMAPS:
        return GURT_CMAPS[name_or_obj]
    if name_or_obj.startswith("pyart_") and PYART_AVAILABLE:
        bare = name_or_obj[6:]
        for mod in (getattr(pyart.graph, 'cm', None),
                    getattr(pyart.graph, 'cm_tables', None)):
            if mod is None:
                continue
            obj = getattr(mod, bare, None)
            if obj is not None:
                return obj
    try:
        return plt.get_cmap(name_or_obj)
    except Exception:
        pass
    if PYART_AVAILABLE:
        for mod in (getattr(pyart.graph, 'cm', None),
                    getattr(pyart.graph, 'cm_tables', None)):
            if mod is None:
                continue
            obj = getattr(mod, 'Carbone42', None)
            if obj is not None:
                return obj
    return REF_CMAP


# Alias used by main.py's rendering path
def resolve_mpl_cmap(name_or_obj):
    return resolve_cmap(name_or_obj)


# ── Field defaults ─────────────────────────────────────────────────────────────
# (label, vmin, vmax, cmap_name)  — merged from both scripts, pyart cmap where
# available otherwise falls back to gurt_* colormaps.

def _ref_cmap():
    return "pyart_Carbone42" if PYART_AVAILABLE else "gurt_ref"

FIELD_DEFAULTS = {
    "DBZH":            ("Z_h (dBZ)",          -20,  70,  _ref_cmap()),
    "DBZH1":           ("Z_h (dBZ)",          -28,  35,  _ref_cmap()),
    "DBZ":             ("Z_h (dBZ)",          -20,  70,  _ref_cmap()),
    "DBZVC_F":         ("Z_h (dBZ)",          -20,  70,  _ref_cmap()),
    "DBZVC":           ("Z_h (dBZ)",          -20,  70,  _ref_cmap()),
    "reflectivity":    ("Z_h (dBZ)",          -20,  70,  _ref_cmap()),
    "DBMHC":           ("Z_h (dBZ)",          -20,  70,  _ref_cmap()),
    "REF":             ("Z_h (dBZ)",          -20,  70,  _ref_cmap()),
    "VMAG_SMOOTH":     ("Smoothed |V| (m/s)",    0,  90,  "turbo"),
    "VMAG_SWATH":      ("Max smoothed |V| (m/s)", 0,  90,  "turbo"),
    "VEL":             ("Vr (m/s)",           -84,  84,  "gurt_vel"),
    "VE":              ("Vr (m/s)",           -84,  84,  "gurt_vel"),
    "velocity":        ("Vr (m/s)",           -84,  84,  "gurt_vel"),
    "VU":              ("Vr (m/s)",           -84,  84,  "gurt_vel"),
    "VELD":            ("Vr dealias (m/s)",   -60,  60,  "gurt_vel"),
    "ZDR":             ("Z_dr (dB)",           -2,   8,  "gurt_zdr"),
    "RHOHV":           ("rhoHV",               0,   1,  "gurt_cc"),
    "RHO":             ("rhoHV",               0,   1,  "gurt_cc"),
    "KDP":             ("K_dp (°/km)",         -2,  12,  "gurt_kdp"),
    "DP":              ("K_dp (°/km)",         -2,  12,  "gurt_kdp"),
    "PHIDP":           ("φ_dp (°)",            0, 360,  "gurt_sw"),
    "PHI":             ("φ_dp (°)",            0, 360,  "gurt_sw"),
    "differential_phase": ("φ_dp (°)",         0, 360,  "gurt_sw"),
    "width":           ("σ (m/s)",             0,  15,  "gurt_sw"),
    "spectrum_width":  ("σ (m/s)",             0,  15,  "gurt_sw"),
    "SW":              ("σ (m/s)",             0,  15,  "gurt_sw"),
    "WIDTH":           ("σ (m/s)",             0,  15,  "gurt_sw"),
    "SNR":             ("SNR (dB)",          -10,  50,  _ref_cmap()),
    # ── IQ_beta ("Open I/Q File (Beta)") default product names ─────────────
    # DBZVC and DBMHC already existed above; these fill in the rest of the
    # naming convention IQ_beta offers (see IQ_beta/products.py) so freshly
    # rendered I/Q moments get sensible colors/ranges out of the box instead
    # of falling back to the generic (-50, 50, gurt_ref) default.
    "DBZHC":           ("Z_h (dBZ)",          -20,  70,  _ref_cmap()),
    "VELH":            ("Vr (m/s)",           -84,  84,  "gurt_vel"),
    "VELV":            ("Vr (m/s)",           -84,  84,  "gurt_vel"),
    "SNRHC":           ("SNR (dB)",          -10,  50,  _ref_cmap()),
    "SNRVC":           ("SNR (dB)",          -10,  50,  _ref_cmap()),
    # Mirrors the existing DBMHC entry's scale above for consistency.
    "DBMVC":           ("Z_h (dBZ)",          -20,  70,  _ref_cmap()),
}

FILE_EXTENSIONS = [
    '.nc', '.h5', '.buf', '.raw', '.gz', '_V06',
    '.msg31', '.0', '.ar2v', '.RAW', '.HDF5', '.hdf5',
]
_DORADE_NAME_PREFIXES = ("swp.",)



# ══════════════════════════════════════════════════════════════════════════════
# Stagger / Nyquist detection
# ══════════════════════════════════════════════════════════════════════════════

def detect_stagger_nyquist(volume, dorade_file=None, sweep_index=0):
    """
    Infer short/long Nyquist from PRT and frequency metadata.

    Sources tried in order:
      1. DORADE RADD block  (prt1/prt2 in ms, freq1 in GHz)  via dorade_file
      2. pyart-style instrument_parameters on the volume (if present)

    Stagger detected (both PRTs valid):
      → Nyquist computed from λ/(4·PRT) for each PRT.
    Fixed PRF (one PRT, or num_ipps_trans < 2):
      → nyquist_velocity read from file (see priority below) if present.
      → Falls back to λ/(4·PRT) only if nothing stored is found.
    No PRT at all:
      → nyquist_velocity read from file used as direct fallback.

    nyquist_velocity lookup priority (used by Fixed PRF and no-PRT paths):
      1. dorade_file.nyquist_velocity      (DORADE eff_unamb_vel)
      2. volume.sweeps[sweep_index].instrument_parameters  (per-sweep, when
                                            the loader resolved it per-sweep —
                                            e.g. CfRadial files where PRF can
                                            differ between elevation cuts)
      3. volume.instrument_parameters      (standard pyart/CfRadial ip dict,
                                            volume-wide fallback)
      4. volume.sweeps[sweep_index].fields (CfRadial files that store it as a
                                            per-ray field instead of ip var)

    sweep_index selects which sweep's metadata to use when it differs from
    sweep to sweep (PRT, frequency, nyquist_velocity can all vary by
    elevation cut). Pass the currently-displayed sweep index so the result
    tracks the active sweep.

    Returns a dict:
      short_nyquist    – float | None  (m/s)
      long_nyquist     – float | None  (m/s)
      stagger_ratio    – str like "2/3" | None
      fixed_nyquist    – float | None  (m/s, non-stagger)
      extended_nyquist – float | None  (m/s)
      source           – str
    """
    result = dict(short_nyquist=None, long_nyquist=None, stagger_ratio=None,
                  fixed_nyquist=None, extended_nyquist=None, source='none')

    sweeps = getattr(volume, 'sweeps', None) or []
    sweep_index = max(0, min(sweep_index, len(sweeps) - 1)) if sweeps else 0
    cur_sweep = sweeps[sweep_index] if sweeps else None

    # Per-sweep instrument parameters take priority over the volume-wide
    # dict whenever the loader was able to resolve them per-sweep; merge so
    # any key the per-sweep dict doesn't have still falls back volume-wide.
    vol_ip   = getattr(volume, 'instrument_parameters', None) or {}
    sweep_ip = getattr(cur_sweep, 'instrument_parameters', None) or {}
    ip = {**vol_ip, **sweep_ip}

    # ── Wavelength ────────────────────────────────────────────────────────────
    wl_m = None

    if dorade_file is not None:
        try:
            freq = float(getattr(dorade_file, 'frequency', 0) or 0)
            if np.isfinite(freq):
                if freq > 1.0e6:
                    wl_m = 3e8 / freq
                elif 0.1 <= freq <= 300.0:
                    wl_m = 3e8 / (freq * 1e9)
        except Exception:
            pass

    if wl_m is None and 'frequency' in ip:
        try:
            freq_hz = float(np.ma.filled(ip['frequency']['data'],
                                         np.nan).flat[0])
            if np.isfinite(freq_hz) and freq_hz > 1e6:
                wl_m = 3e8 / freq_hz
        except Exception:
            pass

    wavelength_source = 'metadata' if wl_m is not None else 'fallback'
    if wl_m is None:
        wl_m = 0.0533  # C-band fallback, used only when no stored Nyquist exists.

    # ── PRT from DORADE RADD block (ms → s) ──────────────────────────────────
    prt_short_s = None
    prt_long_s  = None

    if dorade_file is not None:
        try:
            # DoradeFile stores these as _radd_prt1, _radd_prt2, _radd_num_ipps
            # (populated during RADD block parsing; prt values are in milliseconds,
            # num_ipps_trans >= 2 is required to declare stagger — Radx convention)
            p1_ms  = float(getattr(dorade_file, '_radd_prt1',     0) or 0)
            p2_ms  = float(getattr(dorade_file, '_radd_prt2',     0) or 0)
            n_ipps = int(  getattr(dorade_file, '_radd_num_ipps', 1) or 1)

            if p1_ms > 0 and p2_ms < 0:
                prt_short_s = p1_ms / 1000.0
            elif p2_ms > 0 and p1_ms < 0:
                prt_short_s = p2_ms / 1000.0
            elif p1_ms > 0 and p2_ms > 0:
                ps, pl = sorted((p1_ms / 1000.0, p2_ms / 1000.0))
                prt_short_s = ps
                if n_ipps >= 2:          # Radx only treats as stagger when ≥2 IPPs
                    prt_long_s = pl
                else:
                    prt_long_s = None
            elif p1_ms > 0:
                prt_short_s = p1_ms / 1000.0
        except Exception:
            pass

    # ── PRT from pyart instrument_parameters (already in seconds) ─────────────
    if prt_short_s is None and 'prt' in ip:
        try:
            arr = np.ma.filled(ip['prt']['data'], np.nan).ravel()
            prt_short_s = float(np.nanmedian(arr))
        except Exception:
            pass
        if prt_short_s is not None and 'prt_ratio' in ip:
            try:
                rarr = np.ma.filled(ip['prt_ratio']['data'], np.nan).ravel()
                r = float(np.nanmedian(rarr))
                if np.isfinite(r) and 0.0 < r < 0.99:
                    prt_long_s = prt_short_s / r
            except Exception:
                pass

    # ── Helper: read stored nyquist_velocity from any available source ────────
    def _read_stored_nyquist():
        """
        Try every known location for a stored Nyquist velocity, in priority order:
          1. dorade_file.nyquist_velocity          (DORADE eff_unamb_vel field)
          2. volume.instrument_parameters          (CfRadial / pyart ip dict)
          3. volume.sweeps[0].fields['nyquist_velocity']  (CfRadial field var)
        Returns a positive finite float or None.
        """
        # 1. DORADE attribute
        if dorade_file is not None:
            try:
                v = float(getattr(dorade_file, 'nyquist_velocity', 0) or 0)
                if np.isfinite(v) and v > 0:
                    return v
            except Exception:
                pass

        # 2. instrument_parameters dict (standard pyart location)
        if 'nyquist_velocity' in ip:
            try:
                v = float(np.ma.filled(ip['nyquist_velocity']['data'],
                                       np.nan).flat[0])
                if np.isfinite(v) and v > 0:
                    return v
            except Exception:
                pass

        # 3. sweeps[sweep_index].fields — pyart sometimes stores
        #    nyquist_velocity as a per-ray field (shape [nrays, ngates])
        #    rather than an instrument parameter.  Take the median of the
        #    first ray's finite values, from the *current* sweep so this
        #    tracks whichever sweep is being viewed.
        try:
            if cur_sweep is not None:
                nyq_arr = cur_sweep.fields.get('nyquist_velocity')
                if nyq_arr is not None:
                    flat = np.ma.filled(np.asarray(nyq_arr, dtype='float64'),
                                        np.nan).ravel()
                    v = float(np.nanmedian(flat[np.isfinite(flat)]))
                    if np.isfinite(v) and v > 0:
                        return v
        except Exception:
            pass

        return None

    # ── Fallback: no PRT at all — use stored nyquist_velocity directly ────────
    if prt_short_s is None or not (np.isfinite(prt_short_s) and prt_short_s > 0):
        nyq = _read_stored_nyquist()
        if nyq is not None:
            result['fixed_nyquist'] = nyq
            result['source'] = 'nyquist_velocity (no prt)'
        return result

    # ── Fixed PRF ─────────────────────────────────────────────────────────────
    if prt_long_s is None or not (np.isfinite(prt_long_s) and prt_long_s > 0):
        # Prefer the stored nyquist_velocity from the file over a calculated value.
        # Only fall back to λ/(4·PRT) if no stored value is found.
        stored_nyq = _read_stored_nyquist()
        if stored_nyq is not None:
            result['fixed_nyquist'] = stored_nyq
            result['source'] = (f'nyquist_velocity '
                                f'({stored_nyq:.3f} m/s from file, '
                                f'prt={prt_short_s*1000:.3f} ms)')
        else:
            nyq = wl_m / (4.0 * prt_short_s)
            result['fixed_nyquist'] = nyq if np.isfinite(nyq) else None
            result['source'] = (f'prt_fixed '
                                f'({prt_short_s*1000:.3f} ms, '
                                f'λ={wl_m*100:.2f} cm)')
        return result

    # ── Staggered PRF ─────────────────────────────────────────────────────────
    ratio  = prt_short_s / prt_long_s
    frac   = Fraction(ratio).limit_denominator(10)
    m, n   = frac.numerator, frac.denominator

    stored_nyq = _read_stored_nyquist()
    if wavelength_source != 'metadata' and stored_nyq is not None:
        result['fixed_nyquist'] = stored_nyq
        result['source'] = (f'nyquist_velocity ({stored_nyq:.3f} m/s from file; '
                            f'prt ratio {m}/{n}, no reliable frequency)')
        return result

    nyq_s  = wl_m / (4.0 * prt_short_s)
    nyq_l  = wl_m / (4.0 * prt_long_s)
    nyq_ex = m * nyq_s

    result['short_nyquist']    = nyq_s  if np.isfinite(nyq_s)  else None
    result['long_nyquist']     = nyq_l  if np.isfinite(nyq_l)  else None
    result['extended_nyquist'] = nyq_ex if np.isfinite(nyq_ex) else None
    result['stagger_ratio']    = f"{m}/{n}"
    result['source'] = (f'prt_stagger {m}/{n} '
                        f'({prt_short_s*1000:.3f}/{prt_long_s*1000:.3f} ms, '
                        f'λ={wl_m*100:.2f} cm)')
    return result


# ══════════════════════════════════════════════════════════════════════════════
# GUI theme constants
# ══════════════════════════════════════════════════════════════════════════════

BG      = "#1a1a2e"
PLOT_BG = "#191970"
BAR_BG  = "#d4d0c8"
CB_H    = 70   # colorbar strip height in pixels

DLG_BG      = "#e6e6e6"
DLG_FG      = "black"
DLG_ENTRY   = "#ffffff"
DLG_ABG     = "#cfcfcf"
DLG_HINT    = "#205080"   # info/status hint text (was light blue on navy)
DLG_HINT2   = "#555555"   # secondary/dim hint text (was light lavender on navy)


# ══════════════════════════════════════════════════════════════════════════════
# Colorbar renderer (PIL → Tk)
# ══════════════════════════════════════════════════════════════════════════════

"""
Merged colorbar rendering for the GUI.

Combines:
  - add_preset_colorbars()  -> knows how to walk `selected_fields` presets,
                               skip missing radar fields, style ticks/bbox.
  - _render_colorbar()      -> knows how to render a standalone PIL image
                               (title band / bar band / tick-label band)
                               sized for a fixed-size GUI panel.

Result: one function that takes the same `selected_fields` list used
elsewhere in the app and returns a dict of {field_name: PIL.Image} ready
to blit into a GUI, with the same look (white ticks, boxed tick labels,
title on top) as the original preset colorbars had on the matplotlib
figure.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image as PILImage

# Background color used elsewhere in the GUI for plot panels.
PLOT_BG = "black"  # replace with your app's actual PLOT_BG import/value


def _render_header_image(text: str, fontsize: float = 12.0):
    """White DejaVu Sans text in a tight black square box (matplotlib)."""
    from matplotlib.figure import Figure
    dpi = 96
    with matplotlib.rc_context({"font.family": "sans-serif",
                                "font.sans-serif": ["DejaVu Sans"],
                                "font.weight": "normal"}):
        fig = Figure(figsize=(1, 1), dpi=dpi)
        canvas = FigureCanvasAgg(fig)
        t = fig.text(0, 0, text, color="white", fontsize=fontsize,
                     fontfamily="sans-serif", fontweight="normal",
                     bbox=dict(boxstyle="square,pad=0.3", facecolor="black",
                               edgecolor="none"))
        canvas.draw()
        bb = t.get_window_extent(canvas.get_renderer())
        pad = int(np.ceil(0.3 * fontsize * dpi / 72.0)) + 1
        w, h = int(np.ceil(bb.width)) + 2 * pad, int(np.ceil(bb.height)) + 2 * pad
        fig.set_size_inches(w / dpi, h / dpi)
        t.set_position((pad / w, pad / h))
        t.set_bbox(dict(boxstyle="square,pad=0.3", facecolor="black",
                        edgecolor="none"))
        canvas.draw()
        img = PILImage.frombuffer("RGBA", canvas.get_width_height(),
                                  canvas.buffer_rgba(), "raw", "RGBA", 0, 1)
        out = PILImage.new("RGB", img.size, (0, 0, 0))
        out.paste(img, mask=img.split()[3])
    return out


def _render_colorbar(cmap, vmin: float, vmax: float, label: str, width_px: int,
                     height_px: int, **kw):
    """Colorbar image in Times New Roman bold (scoped; other matplotlib output unchanged)."""
    with matplotlib.rc_context(_CB_RC):
        return _render_colorbar_impl(cmap, vmin, vmax, label, width_px, height_px, **kw)


def _render_colorbar_impl(cmap, vmin: float, vmax: float,
                      label: str, width_px: int, height_px: int,
                      n_ticks: int = 5,
                      tick_color='white', tick_label_color=None,
                      bbox_color='black', bbox_alpha=1.0, bbox_pad=0.3,
                      tick_margin_frac: float = 0.05,
                      bar_height_frac: float = 0.55):
    """Render a single horizontal colorbar to a PIL image for the GUI.

    Ticks/tick-labels sit on TOP of the bar (each label boxed
    individually), and the bar itself is stretched to fill nearly all of
    the panel height instead of sitting as a thin strip inside a large
    black band — `bar_height_frac` controls how much of the total image
    height the bar takes up, and callers can raise/lower it if a given
    panel layout (1/2/3/4-up) has more or less vertical room to work with.

    Parameters
    ----------
    cmap : str or Colormap
    vmin, vmax : float
    label : str
        Title drawn above the ticks (e.g. the field's display title).
    width_px, height_px : int
        Size of the rendered image in pixels.
    n_ticks : int
        Number of tick marks/labels on the bar.
    tick_color : str or tuple
        Color of the tick marks (and outline color).
    tick_label_color : str or tuple, optional
        Color of the tick label text. Defaults to tick_color.
    bbox_color : str or tuple
        Background box color behind each individual tick label.
    bbox_alpha : float
        Alpha of that background box.
    bbox_pad : float
        Padding (in font-size units) around each tick label's box. Kept
        small so each box reads as a tight little tag over the bar
        rather than a big block.
    tick_margin_frac : float
        Fraction of the [vmin, vmax] range to inset from each end before
        placing ticks, so labels near the ends don't crowd/clip.
    bar_height_frac : float
        Fraction of the figure's total height given to the bar itself.
        The remainder is split between the tick-label row (just above
        the bar) and the title row (above that).
    """
    if tick_label_color is None:
        tick_label_color = tick_color

    dpi = 96
    fig = plt.Figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi,
                      facecolor=(0, 0, 0, 0))

    n_ticks = max(2, int(n_ticks))

    # Narrower per-panel bars (e.g. in a 2x2 grid) get less room per tick —
    # shrink the label font so n_ticks stays readable instead of overlapping.
    px_per_tick = width_px / n_ticks
    tick_fs = 7.0            # same 7 pt regular type as the canvas labels
    title_fs = 7.0

    # Bottom-up budget (axes fractions of the total figure height). Ticks
    # now live ABOVE the bar, so there's no need to reserve space below
    # it — that space instead goes to making the bar itself bigger:
    #   0.00–bar_bottom            small breathing room below the bar
    #   bar_bottom–(+bar_height)   the bar itself (now the dominant band)
    #   ...–title band             tick marks + tick-value labels
    #   title band–1.00            product-name title (via ax.set_title)
    bar_bottom = 0.06
    margin = 0.0   # bar runs edge to edge of the panel
    ax = fig.add_axes([margin, bar_bottom, 1.0 - 2 * margin, bar_height_frac])
    sm = plt.cm.ScalarMappable(
        norm=mcolors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax, orientation="horizontal")

    # Preset-style tick placement: inset from each end by a margin
    # fraction of the range, rather than spanning the full [vmin, vmax].
    tick_range = vmax - vmin
    inset = tick_range * tick_margin_frac
    ticks = np.linspace(vmin + inset, vmax - inset, n_ticks)
    cb.set_ticks(ticks)

    # Pick a decimal precision from the tick spacing so labels are always
    # meaningfully distinct (e.g. RHOHV 0..1 needs 2 decimals; dBZ -25..60
    # reads fine as integers) instead of a single fixed format for every
    # product's scale.
    step = (vmax - vmin) / max(1, n_ticks - 1)
    if step >= 5:
        decimals = 0
    elif step >= 0.5:
        decimals = 1
    else:
        decimals = 2
    cb.set_ticklabels([f"{t:.{decimals}f}" for t in ticks])

    # Ticks + labels on top of the bar instead of below it.
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    cb.ax.tick_params(labelsize=tick_fs, pad=2, direction="out", length=4,
                       color=tick_color, labelcolor=tick_label_color,
                       top=True, bottom=False)
    cb.outline.set_edgecolor(tick_color)
    cb.outline.set_linewidth(0.5)
    ax.set_facecolor((0, 0, 0, 0))

    # Each tick label gets its own small, tight black box (matplotlib
    # applies the bbox per-Text, so these never merge into one big strip
    # even when labels sit close together).
    bbox_props = dict(boxstyle=f"square,pad={bbox_pad}",
                       facecolor=bbox_color,
                       alpha=bbox_alpha,
                       edgecolor='none')
    for lbl in cb.ax.get_xticklabels():
        lbl.set_color(tick_label_color)
        lbl.set_fontfamily("sans-serif")
        lbl.set_fontweight("normal")
        lbl.set_bbox(bbox_props)

    canvas_agg = FigureCanvasAgg(fig)
    canvas_agg.draw()
    buf = canvas_agg.buffer_rgba()
    img = PILImage.frombuffer("RGBA", canvas_agg.get_width_height(), buf)
    plt.close(fig)
    return img


def render_preset_colorbars_for_gui(selected_fields, radar, width_px, height_px,
                                     n_ticks: int = 5,
                                     tick_color='white', tick_label_color=None,
                                     bbox_color='black', bbox_alpha=0.7,
                                     bbox_pad=0.15, bar_height_frac=0.60):
    """Render one GUI-ready colorbar image per preset field.

    This is the GUI equivalent of `add_preset_colorbars`: instead of
    adding a `fig.add_axes(...)` colorbar under each matplotlib panel,
    it renders a standalone PIL image per field (using `_render_colorbar`)
    that you can display in your GUI's image widgets — one per panel,
    same sizing/styling logic as before.

    Parameters
    ----------
    selected_fields : list of (field_name, title, vmin, vmax, cmap)
    radar : pyart.core.Radar
        Used only to skip fields that aren't present, same guard as
        `add_preset_colorbars`.
    width_px, height_px : int
        Pixel size for each rendered colorbar image (e.g. matched to
        the width and available colorbar-strip height of that panel in
        your GUI layout — pass the actual black-space height here so
        the bar can be stretched to fill it).
    bar_height_frac : float
        Fraction of height_px given to the bar itself (rest goes to the
        top-mounted ticks/title). Same value works across 1/2/3/4-panel
        layouts since it's a fraction of whatever height_px you pass in
        for that layout — just pass each layout's actual strip height.
    n_ticks, tick_color, tick_label_color, bbox_color, bbox_alpha, bbox_pad :
        Same styling knobs as `add_preset_colorbars` / `_render_colorbar`.

    Returns
    -------
    dict[str, PIL.Image.Image]
        Mapping of field_name -> rendered colorbar image, in the same
        order as `selected_fields` (skipping any field not in radar).
    """
    images = {}
    for field_name, title, vmin, vmax, cmap in selected_fields:
        if field_name not in radar.fields:
            continue

        img = _render_colorbar(
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            label=title,
            width_px=0.9,
            height_px=0.03,
            n_ticks=n_ticks,
            tick_color=tick_color,
            tick_label_color=tick_label_color,
            bbox_color=bbox_color,
            bbox_alpha=bbox_alpha,
            bbox_pad=bbox_pad,
            bar_height_frac=bar_height_frac,
        )
        images[field_name] = img

    return images


# ══════════════════════════════════════════════════════════════════════════════
# VisPy PPI geometry helpers  (kept here so gurt_gui is self-contained)
# ══════════════════════════════════════════════════════════════════════════════

def _build_ppi_geometry(azimuths_deg: np.ndarray, ranges_m: np.ndarray):
    nrays  = len(azimuths_deg)
    ngates = len(ranges_m)

    az_rad      = np.deg2rad(azimuths_deg.astype("float64"))
    az_unwrapped = np.unwrap(az_rad)
    daz          = np.diff(az_unwrapped)
    az_edges     = np.empty(nrays + 1, dtype="float64")
    az_edges[1:-1] = 0.5 * (az_unwrapped[:-1] + az_unwrapped[1:])
    az_edges[0]    = az_unwrapped[0]  - 0.5 * daz[0]
    az_edges[-1]   = az_unwrapped[-1] + 0.5 * daz[-1]

    r_km = ranges_m.astype("float64") / 1000.0
    dr   = np.diff(r_km)
    r_edges = np.empty(ngates + 1, dtype="float64")
    r_edges[1:-1] = 0.5 * (r_km[:-1] + r_km[1:])
    r_edges[0]    = max(0.0, r_km[0]  - 0.5 * (dr[0] if len(dr) else r_km[0]))
    r_edges[-1]   = r_km[-1] + 0.5 * (dr[-1] if len(dr) else r_km[-1])

    R, A = np.meshgrid(r_edges, az_edges)
    x = (R * np.sin(A)).ravel().astype("float32")
    y = (R * np.cos(A)).ravel().astype("float32")
    verts = np.column_stack([x, y, np.zeros(len(x), dtype="float32")])

    cols  = ngates + 1
    i_idx = np.repeat(np.arange(nrays,  dtype=np.uint32), ngates)
    j_idx = np.tile  (np.arange(ngates, dtype=np.uint32), nrays)
    v00 = i_idx * cols + j_idx
    v01 = v00 + 1
    v10 = v00 + cols
    v11 = v10 + 1
    tri1 = np.column_stack([v00, v01, v10])
    tri2 = np.column_stack([v01, v11, v10])
    faces = np.empty((nrays * ngates * 2, 3), dtype=np.uint32)
    faces[0::2] = tri1
    faces[1::2] = tri2
    return verts, faces


def _grid_ppi_for_image(sweep, data: np.ndarray):
    """Pre-grid PPI ray data onto a uniform azimuth grid.

    The output array has shape (n_az_bins, ngates). Row 0 = az ~360°
    and row n_az-1 = az ~0° (descending azimuth order) so that after
    applying the corrected STTransform in _ppi_image_transform the
    imap correctly maps canvas (x, y) → texture row without a
    vertical flip artifact (VisPy ImageVisual places row 0 at the
    bottom of the visual's local y-axis, which after the camera's
    y-up convention would put az 0° at the south — the opposite of
    meteorological convention).

    Returns (gridded, r0_km, dr_km, da_deg).
    """
    az_raw = np.asarray(sweep.azimuth, dtype="float64") % 360.0
    ranges_km = np.asarray(sweep.ranges, dtype="float64") / 1000.0
    nrays = az_raw.size
    ngates = ranges_km.size

    dr_km = float(np.nanmedian(np.diff(ranges_km))) if ngates > 1 else 1.0
    r0_km = float(ranges_km[0] - 0.5 * dr_km) if ngates else 0.0

    # Choose number of azimuth bins to match the native angular resolution.
    # Use the smallest-N median of sorted diffs to avoid inflating n_az
    # when a few rays are duplicated or missing (DOW-style irregular scans).
    az_sorted = np.sort(az_raw)
    diffs = np.diff(az_sorted)
    diffs = diffs[diffs > 1e-4]   # drop true duplicates / sub-0.1° jitter
    if diffs.size:
        # Use the 25th-percentile diff so that small gaps in sector scans
        # don't inflate da_deg beyond the native beam spacing.
        med_da = float(np.percentile(diffs, 25))
    else:
        med_da = 1.0
    if not np.isfinite(med_da) or med_da < 1e-4:
        med_da = 1.0
    n_az = max(nrays, int(round(360.0 / med_da)))
    da_deg = 360.0 / n_az

    # Build a native-order grid (row i = az i*da_deg, ascending).
    gridded_asc = np.full((n_az, ngates), np.nan, dtype="float32")
    bins = np.floor(az_raw / da_deg).astype(int) % n_az
    # Vectorised last-write-wins: sort by azimuth so that later rays
    # in the scan overwrite earlier ones for the same bin, matching
    # NLR's importer "last ray wins" behaviour.
    order = np.argsort(bins, kind="stable")
    gridded_asc[bins[order]] = data[order]

    # Flip to descending-azimuth order so that VisPy's ImageVisual
    # (which places row 0 at the bottom, i.e. lowest scene-y = south)
    # renders az 0° (north, highest scene-y) at the correct position.
    gridded = gridded_asc[::-1].copy()

    return gridded, r0_km, dr_km, da_deg


def _data_to_rgba(data: np.ndarray, cmap, vmin: float, vmax: float,
                  dtype=np.float32) -> np.ndarray:
    """Map a sweep data array to RGBA.

    ImageVisual uses uint8 textures to keep sweep switches cheap; MeshVisual
    still uses float colors because VisPy's mesh color path expects 0..1.
    """
    arr = np.asarray(data, dtype="float32")
    flat = arr.ravel()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    rgba = np.asarray(cmap(norm(np.nan_to_num(flat, nan=vmin))), dtype="float32")
    nan_mask = ~np.isfinite(flat)
    if hasattr(data, "mask"):
        nan_mask |= np.asarray(data.mask).ravel()
    rgba[nan_mask, 3] = 0.0
    rgba = rgba.reshape(arr.shape + (4,))
    if dtype == np.uint8:
        return np.clip(np.rint(rgba * 255.0), 0, 255).astype("uint8")
    return rgba.astype(dtype, copy=False)


def _data_to_face_colors(data: np.ndarray, cmap, vmin: float,
                          vmax: float) -> np.ndarray:
    rgba = _data_to_rgba(data, cmap, vmin, vmax, dtype=np.float32).reshape(-1, 4)
    return np.repeat(rgba, 2, axis=0)


class PPIPolarTransform(BaseTransform):
    """Polar transform matching NLR's PolarTransform exactly.
    Maps (r_km, az_deg) to (x, y).
    The imap uses the same atan convention as NLR so that the imap azimuth
    is always in [0, 360) and aligns with the pre-gridded 0-based texture.
    """
    glsl_map = """
        vec4 ppi_polar_transform_map(vec4 pos) {
            if (pos.y > 400.0) {
                pos.y = pos.y + 180.0;
            }
            return vec4(pos.x * sin(pos.y), pos.x * cos(pos.y), pos.z, 1.0);
        }
    """
    glsl_imap = """
        vec4 ppi_polar_transform_imap(vec4 pos) {
            float theta = -atan(pos.x, pos.y);
            theta = degrees(theta + 3.14159265358979323846);
            float r = length(pos.xy);
            return vec4(r, theta, pos.z, 1.0);
        }
    """
    Linear = False
    Orthogonal = False
    NonScaling = False
    Isometric = False


def _build_ring_lines(max_range_km: float, ring_interval_km: float,
                      az_interval_deg: float):
    segments = []
    theta = np.linspace(0.0, 2.0 * math.pi, 361, dtype="float32")
    r = ring_interval_km
    while r <= max_range_km * 1.01:
        xs = r * np.sin(theta)
        ys = r * np.cos(theta)
        starts = np.column_stack([xs[:-1], ys[:-1], np.zeros(len(xs)-1, "f4")])
        ends   = np.column_stack([xs[1:],  ys[1:],  np.zeros(len(xs)-1, "f4")])
        segments.append(starts)
        segments.append(ends)
        r += ring_interval_km
    az = 0.0
    while az < 360.0:
        ang = math.pi / 2.0 - math.radians(az)
        xe, ye = max_range_km * math.cos(ang), max_range_km * math.sin(ang)
        segments.append(np.array([[0, 0, 0]], dtype="f4"))
        segments.append(np.array([[xe, ye, 0]], dtype="f4"))
        az += az_interval_deg
    if not segments:
        return np.zeros((2, 3), dtype="f4")
    return np.vstack(segments)


def _ring_label_positions(max_range_km: float, ring_interval_km: float):
    labels = []
    r = ring_interval_km
    while r <= max_range_km * 1.01:
        labels.append((0.0, r, f"{int(r)} km"))
        r += ring_interval_km
    return labels


# ──────────────────────────────────────────────────────────────────────────────
# RHI geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_rhi_geometry(elevations_deg: np.ndarray, ranges_m: np.ndarray):
    """Build VisPy mesh geometry for an RHI sweep.

    In an RHI the antenna scans in *elevation* while azimuth is fixed.
    Gates are mapped to (horizontal_distance, altitude) in km:
        x = range * cos(elevation)
        y = range * sin(elevation)
    """
    nrays  = len(elevations_deg)
    ngates = len(ranges_m)

    el_rad       = np.deg2rad(elevations_deg.astype("float64"))
    el_unwrapped = np.unwrap(el_rad)
    del_         = np.diff(el_unwrapped)
    el_edges     = np.empty(nrays + 1, dtype="float64")
    el_edges[1:-1] = 0.5 * (el_unwrapped[:-1] + el_unwrapped[1:])
    el_edges[0]    = el_unwrapped[0]  - 0.5 * del_[0]
    el_edges[-1]   = el_unwrapped[-1] + 0.5 * del_[-1]

    r_km = ranges_m.astype("float64") / 1000.0
    dr   = np.diff(r_km)
    r_edges = np.empty(ngates + 1, dtype="float64")
    r_edges[1:-1] = 0.5 * (r_km[:-1] + r_km[1:])
    r_edges[0]    = max(0.0, r_km[0]  - 0.5 * (dr[0] if len(dr) else r_km[0]))
    r_edges[-1]   = r_km[-1] + 0.5 * (dr[-1] if len(dr) else r_km[-1])

    R, EL = np.meshgrid(r_edges, el_edges)
    x = (R * np.cos(EL)).ravel().astype("float32")
    y = (R * np.sin(EL)).ravel().astype("float32")
    verts = np.column_stack([x, y, np.zeros(len(x), dtype="float32")])

    cols  = ngates + 1
    i_idx = np.repeat(np.arange(nrays,  dtype=np.uint32), ngates)
    j_idx = np.tile  (np.arange(ngates, dtype=np.uint32), nrays)
    v00 = i_idx * cols + j_idx
    v01 = v00 + 1
    v10 = v00 + cols
    v11 = v10 + 1
    tri1 = np.column_stack([v00, v01, v10])
    tri2 = np.column_stack([v01, v11, v10])
    faces = np.empty((nrays * ngates * 2, 3), dtype=np.uint32)
    faces[0::2] = tri1
    faces[1::2] = tri2
    return verts, faces


def _build_rhi_ring_lines(max_range_km: float, ring_interval_km: float,
                           el_interval_deg: float):
    """Quarter-circle range arcs and elevation spokes for an RHI display."""
    segments = []
    el_angles = np.linspace(0.0, 0.5 * math.pi, 181, dtype="float32")

    r = ring_interval_km
    while r <= max_range_km * 1.01:
        xs = r * np.cos(el_angles)
        ys = r * np.sin(el_angles)
        starts = np.column_stack([xs[:-1], ys[:-1], np.zeros(len(xs) - 1, "f4")])
        ends   = np.column_stack([xs[1:],  ys[1:],  np.zeros(len(xs) - 1, "f4")])
        segments.append(starts)
        segments.append(ends)
        r += ring_interval_km

    el = 0.0
    while el <= 90.0:
        ang = math.radians(el)
        xe, ye = max_range_km * math.cos(ang), max_range_km * math.sin(ang)
        segments.append(np.array([[0, 0, 0]], dtype="f4"))
        segments.append(np.array([[xe, ye, 0]], dtype="f4"))
        el += el_interval_deg

    return np.vstack(segments) if segments else np.zeros((2, 3), dtype="f4")


def _rhi_ring_label_positions(max_range_km: float, ring_interval_km: float):
    """Label positions along the ground axis for RHI range rings."""
    labels = []
    r = ring_interval_km
    while r <= max_range_km * 1.01:
        labels.append((r, 0.2, f"{int(r)} km"))
        r += ring_interval_km
    return labels


# ══════════════════════════════════════════════════════════════════════════════
# Brush throttle — caps drag-event rate to ~30 fps so the UI stays responsive
# ══════════════════════════════════════════════════════════════════════════════

class BrushThrottle(brushes.BrushThrottle):
    """Kept as an alias for backward compatibility — implementation now
    lives in brushes.py."""
    pass


# ══════════════════════════════════════════════════════════════════════════════
# PPICanvas — VisPy canvas embedded in Tkinter
# ══════════════════════════════════════════════════════════════════════════════

def _nice_tick_step(span_km: float, target: int = 8) -> float:
    """1-2-5 tick interval giving roughly `target` ticks across `span_km`."""
    if not np.isfinite(span_km) or span_km <= 0:
        return 1.0
    raw = span_km / max(target, 1)
    mag = 10.0 ** math.floor(math.log10(raw))
    for m in (1.0, 2.0, 5.0, 10.0):
        if raw <= m * mag * 1.0001:
            return m * mag
    return 10.0 * mag


def _tick_label(v: float, step: float) -> str:
    dec = 0 if step >= 1.0 - 1e-9 else int(min(4, math.ceil(-math.log10(step) - 1e-9)))
    if dec == 0:
        return str(int(round(v)))
    return f"{v:.{dec}f}"


def compute_tick_layout(x0, x1, y0, y1, interval_km=0.0, max_per_axis=60):
    """Tick geometry for the visible scene rect (km).

    Port of the tick-mark block in the standalone GURT script
    (add_range_rings_and_azimuth_lines, `tickskm`): a small '+' at every
    grid intersection, y labels along the left edge, x labels along the top
    edge, '+' arms 1% of the smaller view span. The one change: the
    interval is not fixed but follows the zoom (1-2-5 steps, ~8 per axis)
    unless `interval_km` > 0 is given.

    Returns (step, cross line-segment points (4N,3) float32, x tick values, y tick values).
    """
    xr, yr = x1 - x0, y1 - y0
    if xr <= 0 or yr <= 0:
        return None
    span = min(xr, yr)
    step = float(interval_km) if interval_km and interval_km > 0 else _nice_tick_step(span)
    while max(xr, yr) / step > max_per_axis:           # fixed interval too fine for this zoom
        step = _nice_tick_step(step * 2.0, target=1)
    xt = np.arange(np.ceil(x0 / step) * step, x1 + 1e-9, step)
    yt = np.arange(np.ceil(y0 / step) * step, y1 + 1e-9, step)
    xt, yt = xt[(xt >= x0) & (xt <= x1)], yt[(yt >= y0) & (yt <= y1)]
    arm = span * 0.01
    if xt.size and yt.size:
        X, Y = np.meshgrid(xt, yt)
        X, Y = X.ravel(), Y.ravel()
        n = X.size
        pts = np.zeros((4 * n, 3), dtype="float32")
        pts[0::4, 0], pts[0::4, 1] = X - arm, Y
        pts[1::4, 0], pts[1::4, 1] = X + arm, Y
        pts[2::4, 0], pts[2::4, 1] = X, Y - arm
        pts[3::4, 0], pts[3::4, 1] = X, Y + arm
    else:
        pts = np.zeros((0, 3), dtype="float32")
    return step, pts, xt, yt


class PPICanvas:
    """
    Wraps a VisPy SceneCanvas embedded into a Tkinter parent frame.
    Owns all VisPy visuals and exposes a simple update_sweep() API.
    """

    def __init__(self, parent_frame: tk.Frame, app: "GURTVispy", panel=None):
        self._app   = app
        self.panel  = panel   # owning PanelState (None for legacy single-panel use)

        self._vcanvas = SceneCanvas(
            keys=None,
            bgcolor=PLOT_BG,
            parent=parent_frame,
            create_native=True,
        )
        self._vcanvas.native.pack(fill=tk.BOTH, expand=True)

        self._view = self._vcanvas.central_widget.add_view()
        self._view.camera = "panzoom"
        self._view.camera.aspect = 1.0
        # Every pan/zoom gesture is handled manually below (zoom_by/pan_by +
        # explicit broadcast to sibling panels). VisPy's own PanZoomCamera
        # also listens for mouse_wheel/mouse_move events on this view and,
        # if its native Tk bindings aren't fully overridden by the ones set
        # up further down, will ALSO call its own .zoom()/.pan() in
        # response to the same gesture — on top of (not instead of) the
        # manual call. That only happens on whichever panel is actually
        # receiving the real mouse events (the one being scrolled/dragged
        # on), never on a panel that's only receiving a synced broadcast —
        # so the directly-interacted panel ends up moving/zooming further
        # than the broadcast amount, making every other panel look like
        # it's lagging behind ("zooms/pans less"), even though the synced
        # rect math itself is correct. Disabling built-in interactivity
        # removes this second, untracked source of movement entirely.
        self._view.camera.interactive = False

        self._mesh = visuals.Mesh(shading=None, parent=self._view.scene)
        self._mesh.set_gl_state("translucent", depth_test=False, cull_face=False)
        self._image = visuals.Image(
            np.zeros((1, 1, 4), dtype="uint8"),
            interpolation="nearest",
            method="auto",
            parent=self._view.scene,
        )
        self._image.clim = (0, 255)
        self._image.set_gl_state("translucent", depth_test=False)
        self._image.visible = False

        self._rings = visuals.Line(
            connect="segments",
            color=(1.0, 1.0, 1.0, 0.22),
            width=1.0,
            parent=self._view.scene,
        )
        self._ring_texts: list[visuals.Text] = []

        # Dynamic '+' tick marks and edge labels (see compute_tick_layout).
        self._tick_lines = visuals.Line(
            connect="segments", color=(1.0, 1.0, 1.0, 1.0), width=1.0,
            parent=self._view.scene)
        self._tick_lines.visible = False
        self._tick_ytxt = visuals.Text(
            text=["0"], pos=np.zeros((1, 3), dtype="float32"), color="white",
            font_size=7, anchor_x="left", anchor_y="center",
            parent=self._view.scene)
        self._tick_xtxt = visuals.Text(
            text=["0"], pos=np.zeros((1, 3), dtype="float32"), color="white",
            font_size=7, anchor_x="center", anchor_y="top",
            parent=self._view.scene)
        self._tick_ytxt.visible = False
        self._tick_xtxt.visible = False
        self._tick_job = None
        self._preview_markers = None

        # Colorbar overlay: drawn in screen pixels on top of the radar view,
        # so the radar canvas extends behind it (the image is transparent
        # except for the bar itself and the individually boxed tick labels).
        self._cb_img = visuals.Image(
            np.zeros((1, 1, 4), dtype="uint8"), interpolation="nearest",
            method="auto", parent=self._vcanvas.scene)
        self._cb_img.clim = (0, 255)
        self._cb_img.set_gl_state("translucent", depth_test=False)
        self._cb_img.order = 100
        self._cb_img.visible = False
        self._cb_h = 0
        try:
            self._vcanvas.events.resize.connect(lambda ev: self._place_colorbar())
        except Exception:
            pass
        try:
            self._view.camera.events.transform_change.connect(lambda ev: self._schedule_ticks())
            self._vcanvas.events.resize.connect(lambda ev: self._schedule_ticks())
        except Exception:
            pass

        # Brush cursor circle (shown in brush modes).
        # Built as a Line visual with a pre-computed unit circle, scaled via
        # STTransform. Moving/resizing only changes the transform — no geometry
        # rebuild on every mouse event, unlike Ellipse which re-tessellates
        # its mesh every time .center or .radius is set.
        _n = 64
        _theta = np.linspace(0.0, 2.0 * math.pi, _n + 1, dtype="float32")
        _cx = np.cos(_theta)
        _cy = np.sin(_theta)
        _circle_pts = np.column_stack([_cx, _cy,
                                       np.zeros(_n + 1, dtype="float32")])
        self._brush_circle = visuals.Line(
            pos=_circle_pts,
            color=(1.0, 1.0, 0.0, 0.85),
            width=1.5,
            connect="strip",
            parent=self._view.scene,
        )
        self._brush_circle.transform = STTransform()
        self._brush_circle.visible = False

        # Boundary-tool polygon preview (shown while drawing in boundary
        # mode). `_boundary_points` holds the placed vertices in scene
        # (km) coordinates; the Line/Markers visuals are just a drawing
        # of that list (plus a live rubber-band segment to the cursor),
        # rebuilt on every click/move via _update_boundary_visual().
        self._boundary_points: list[tuple[float, float]] = []
        self._boundary_line = visuals.Line(
            connect="strip",
            color=(1.0, 0.25, 0.85, 0.9),
            width=2.0,
            parent=self._view.scene,
        )
        self._boundary_line.visible = False
        self._boundary_markers = visuals.Markers(parent=self._view.scene)
        self._boundary_markers.set_data(
            pos=np.zeros((0, 2), dtype="float32"),
            face_color=(1.0, 0.25, 0.85, 1.0),
            edge_color=(1.0, 1.0, 1.0, 0.9),
            size=7,
        )
        self._boundary_markers.visible = False

        self._sweep      = None
        self._verts      = None
        self._faces      = None
        self._geom_key   = None
        self._face_colors = None
        self._image_rgba  = None
        self._image_shape = None
        self._overlay_key = None
        self._max_range  = 150.0

        # Fast-path direct-GPU-buffer color patching for brush edits (see
        # _try_fast_color_patch). Invalidated (set back to None) any time
        # the mesh is fully rebuilt via update_sweep()/_mesh.set_data().
        self._color_vbo       = None
        self._unshared_colors = None

        self._drag_start_scene  = None
        self._pan_last_xy       = None
        self._dragging          = False
        self._brush_throttle   = BrushThrottle(interval_ms=33)
        # Keep pan/zoom input close to display refresh without forcing synchronous redraws.
        self._pan_throttle     = BrushThrottle(interval_ms=16)
        self._move_throttle    = BrushThrottle(interval_ms=50)

        w = self._vcanvas.native
        w.bind("<ButtonPress-1>",   self._on_press)
        w.bind("<B1-Motion>",       self._on_drag)
        w.bind("<ButtonRelease-1>", self._on_release)
        w.bind("<ButtonPress-3>",   self._on_right_click)
        w.bind("<Motion>",          self._on_mouse_move)
        w.bind("<MouseWheel>",      self._on_scroll)
        w.bind("<Button-4>",        lambda e: self._scroll_zoom(e, +1))
        w.bind("<Button-5>",        lambda e: self._scroll_zoom(e, -1))
        # Shift+scroll: resize brush, bound separately so "break" prevents
        # the normal scroll handler from also firing.
        w.bind("<Shift-MouseWheel>", self._on_shift_scroll)
        w.bind("<Shift-Button-4>",   lambda e: self._on_shift_scroll_dir(e, +1))
        w.bind("<Shift-Button-5>",   lambda e: self._on_shift_scroll_dir(e, -1))

    # ── Sweep update ──────────────────────────────────────────────────────────

    def update_sweep(self, sweep, field_name: str,
                     vmin: float, vmax: float, cmap,
                     max_range_km: float = 150.0,
                     ring_interval_km: float = 50.0,
                     az_interval_deg: float = 30.0,
                     show_rings: bool = True,
                     show_azimuths: bool = True):
        self._sweep     = sweep
        self._max_range = max_range_km
        is_rhi = getattr(sweep, "scan_type", "ppi") == "rhi"
        self._is_rhi = is_rhi

        nrays, ngates = len(sweep.azimuth), len(sweep.ranges)
        if field_name in sweep.fields:
            data = np.asarray(sweep.fields[field_name], dtype="float32")
            if data.shape != (nrays, ngates):
                tgt_r, tgt_g = nrays, ngates
                r, g = data.shape
                if r < tgt_r:
                    data = np.vstack([data,
                        np.full((tgt_r-r, g), np.nan, "float32")])
                elif r > tgt_r:
                    data = data[:tgt_r, :]
                if data.shape[1] < tgt_g:
                    data = np.hstack([data,
                        np.full((data.shape[0], tgt_g-data.shape[1]),
                                np.nan, "float32")])
                else:
                    data = data[:, :tgt_g]
        else:
            data = np.full((nrays, ngates), np.nan, dtype="float32")

        # Always use the mesh path. True per-ray azimuths from the file are used
        # directly via _build_ppi_geometry, which is correct for all data sources.
        use_image = False
        if is_rhi or not use_image:
            self._image.visible = False
            self._image_rgba = None
            self._image_shape = None
            self._mesh.visible = True
            if is_rhi:
                geom_key = self._geometry_key("rhi", sweep.elevation, sweep.ranges)
                if self._verts is None or self._geom_key != geom_key:
                    self._verts, self._faces = _build_rhi_geometry(
                        sweep.elevation, sweep.ranges)
                    self._geom_key = geom_key
            else:
                geom_key = self._geometry_key("ppi", sweep.azimuth, sweep.ranges)
                if self._verts is None or self._geom_key != geom_key:
                    self._verts, self._faces = _build_ppi_geometry(
                        sweep.azimuth, sweep.ranges)
                    self._geom_key = geom_key
            face_colors = _data_to_face_colors(data, cmap, vmin, vmax)
            self._face_colors = face_colors
            self._mesh.set_data(vertices=self._verts, faces=self._faces,
                                face_colors=face_colors)
            # A full set_data() rebuild invalidates any GPU buffer reference
            # the fast brush-patch path was holding onto.
            self._color_vbo       = None
            self._unshared_colors = None
        else:
            self._mesh.visible = False
            self._image.visible = True
            gridded, r0_km, dr_km, da_deg = _grid_ppi_for_image(sweep, data)
            self._set_image_rgba(
                _data_to_rgba(gridded, cmap, vmin, vmax, dtype=np.uint8))
            self._image.transform = self._ppi_image_transform(r0_km, dr_km, da_deg, gridded.shape[0])
            self._face_colors = None
            self._color_vbo = None
            self._unshared_colors = None
        self._update_overlays(max_range_km, ring_interval_km, az_interval_deg,
                              show_rings, show_azimuths)

    def show_cached_render(self, render, max_range_km: float,
                           ring_interval_km: float, az_interval_deg: float,
                           show_rings: bool, show_azimuths: bool):
        """Display a render product prepared by GURTVispy's sweep cache."""
        self._sweep = render["sweep"]
        self._max_range = max_range_km
        self._is_rhi = render["mode"] == "rhi"

        if render["mode"] == "image":
            self._mesh.visible = False
            self._image.visible = True
            self._set_image_rgba(render["rgba"].copy())
            self._image.transform = render["transform"]
            self._face_colors = None
            self._color_vbo = None
            self._unshared_colors = None
        else:
            self._image.visible = False
            self._image_rgba = None
            self._image_shape = None
            self._mesh.visible = True
            self._verts = render["verts"]
            self._faces = render["faces"]
            self._geom_key = render["geom_key"]
            self._face_colors = render["face_colors"].copy()
            self._mesh.set_data(vertices=self._verts, faces=self._faces,
                                face_colors=self._face_colors)
            self._color_vbo = None
            self._unshared_colors = None

        self._update_overlays(max_range_km, ring_interval_km, az_interval_deg,
                              show_rings, show_azimuths)

    def _set_image_rgba(self, rgba: np.ndarray):
        """Upload an already-colormapped RGBA image to the PPI ImageVisual."""
        rgba = np.asarray(rgba)
        if rgba.dtype == np.uint8:
            self._image.clim = (0, 255)
            upload = rgba
        else:
            self._image.clim = (0.0, 1.0)
            upload = rgba.astype("float32", copy=False)
        self._image_rgba = upload
        self._image_shape = upload.shape[:2]
        self._image.set_data(upload)

    def _can_use_ppi_image(self, sweep) -> bool:
        """Use the image path for any PPI sweep with at least 3 rays and
        uniform range spacing.  Azimuth irregularity is handled by
        _grid_ppi_for_image(), which pre-grids rays onto a uniform 0-based
        360-degree grid exactly as NLR does, so sector scans, dropped rays,
        and non-zero start azimuths are all handled correctly.
        """
        ranges = np.asarray(sweep.ranges, dtype="float64")
        az = np.asarray(sweep.azimuth, dtype="float64")
        if az.size < 3 or ranges.size < 2:
            return False
        dr = np.diff(ranges)
        dr_med = float(np.nanmedian(dr)) if dr.size else 0.0
        if not np.isfinite(dr_med) or abs(dr_med) < 1.0e-6:
            return False
        # Require reasonably uniform range spacing (same constraint as before)
        dr_tol = max(1.0, abs(dr_med) * 0.01)
        return bool(np.nanmax(np.abs(dr - dr_med)) <= dr_tol)

    def _geometry_key(self, scan_type, angles, ranges):
        angles = np.asarray(angles, dtype="float64")
        ranges = np.asarray(ranges, dtype="float64")
        if angles.size:
            angle_sig = (float(angles[0]), float(angles[-1]))
        else:
            angle_sig = (0.0, 0.0)
        if ranges.size:
            range_sig = (float(ranges[0]), float(ranges[-1]))
        else:
            range_sig = (0.0, 0.0)
        return (scan_type, int(angles.size), int(ranges.size), angle_sig, range_sig)

    def _ppi_image_transform(self, r0_km, dr_km, da_deg, n_az_bins):
        """Build the GPU transform for the pre-gridded (descending-azimuth) PPI image.

        _grid_ppi_for_image returns a row-flipped array so that VisPy's
        ImageVisual — which places row 0 at the lowest scene-y (south) —
        renders az 0° (north) at the top, matching NLR.

        The STTransform imap maps canvas (x, y) → (col, row):
          col = (r_km - r0_km) / dr_km
          row = (theta_deg - (360 - da_deg)) / (-da_deg)
              = (360 - da_deg - theta_deg) / da_deg

        Row 0   → theta = 360 - da_deg  (top of descending array, near 360°)
        Row n-1 → theta = 0°            (az 0° = north, placed at highest scene-y)
        """
        # Negative azimuth scale + translate to (360 - da_deg) maps the
        # descending-azimuth texture rows back to the correct [0, 360) range.
        return PPIPolarTransform() * STTransform(
            scale=(dr_km, -da_deg),
            translate=(r0_km, 360.0 - da_deg),
        )

    def _update_overlays(self, max_range_km, ring_interval_km,
                         az_interval_deg, show_rings, show_azimuths):
        is_rhi = getattr(self, "_is_rhi", False)
        self._schedule_ticks()
        overlay_key = (bool(is_rhi), float(max_range_km), float(ring_interval_km),
                       float(az_interval_deg), bool(show_rings), bool(show_azimuths))
        if self._overlay_key == overlay_key:
            return
        self._overlay_key = overlay_key

        if is_rhi:
            ri  = ring_interval_km if show_rings    else 1e9
            eli = az_interval_deg  if show_azimuths else 90.0
            pts = _build_rhi_ring_lines(max_range_km, ri, eli)
        else:
            ri  = ring_interval_km if show_rings    else 1e9
            azi = az_interval_deg  if show_azimuths else 360.0
            pts = _build_ring_lines(max_range_km, ri, azi)

        if len(pts) > 1:
            self._rings.set_data(pts)
            self._rings.visible = True
        else:
            self._rings.visible = False

        for t in self._ring_texts:
            t.parent = None
        self._ring_texts.clear()

        if show_rings:
            if is_rhi:
                label_positions = _rhi_ring_label_positions(max_range_km,
                                                             ring_interval_km)
            else:
                label_positions = _ring_label_positions(max_range_km,
                                                         ring_interval_km)
            for x_km, y_km, txt in label_positions:
                t = visuals.Text(
                    txt,
                    pos=[x_km + 0.5, y_km, 0],
                    color=(1, 1, 1, 0.55),
                    font_size=7,
                    anchor_x="left",
                    parent=self._view.scene,
                )
                self._ring_texts.append(t)

    # ── Edit preview (gates an edit would remove) ─────────────────────────────

    def set_preview_points(self, xy_km, color=(1.0, 0.1, 0.1, 1.0), size=4):
        """Mark gate centres (N,2 scene km) with solid squares; empty/None clears."""
        xy = None if xy_km is None else np.asarray(xy_km, dtype="float32")
        if xy is None or xy.size == 0:
            if self._preview_markers is not None:
                self._preview_markers.visible = False
            self._vcanvas.update()
            return
        if self._preview_markers is None:
            self._preview_markers = visuals.Markers(parent=self._view.scene)
            self._preview_markers.set_gl_state("translucent", depth_test=False)
        self._preview_markers.set_data(pos=xy.reshape(-1, 2), face_color=color,
                                       edge_color=(0, 0, 0, 0), edge_width=0, size=size, symbol="square")
        self._preview_markers.visible = True
        self._vcanvas.update()

    # ── Dynamic tick marks ────────────────────────────────────────────────────

    def set_colorbar(self, img):
        """Show a transparent RGBA PIL image pinned to the bottom of the
        canvas (or hide it when *img* is None)."""
        if img is None:
            self._cb_img.visible = False
            self._cb_h = 0
            self._vcanvas.update()
            return
        arr = np.ascontiguousarray(np.asarray(img.convert("RGBA"), dtype="uint8"))
        self._cb_img.set_data(arr)
        self._cb_h = arr.shape[0]
        self._cb_img.visible = True
        self._place_colorbar()

    def _place_colorbar(self):
        if not self._cb_h:
            return
        try:
            ch = self._vcanvas.size[1]
            self._cb_img.transform = STTransform(translate=(0, ch - self._cb_h, 0))
            self._vcanvas.update()
        except Exception:
            pass

    def _schedule_ticks(self):
        """Rebuild the tick marks shortly (coalesces bursts of zoom/pan events)."""
        if self._tick_job is not None:
            return
        try:
            self._tick_job = self._vcanvas.native.after(20, self._update_ticks)
        except Exception:
            self._tick_job = None
            self._update_ticks()

    def _update_ticks(self):
        self._tick_job = None
        show = True
        interval = 0.0
        try:
            show = bool(self._app.show_ticks.get())
            interval = float(self._app.tick_interval.get())
        except Exception:
            pass
        if not show or getattr(self, "_is_rhi", False):
            self._tick_lines.visible = False
            self._tick_ytxt.visible = False
            self._tick_xtxt.visible = False
            return
        w, h = self._canvas_size()
        if w <= 2 or h <= 2:
            return
        xa, ya = self._canvas_to_scene(0, 0)
        xb, yb = self._canvas_to_scene(w, h)
        x0, x1 = min(xa, xb), max(xa, xb)
        y0, y1 = min(ya, yb), max(ya, yb)
        lay = compute_tick_layout(x0, x1, y0, y1, interval)
        if lay is None:
            return
        step, pts, xt, yt = lay
        if len(pts):
            self._tick_lines.set_data(pos=pts)
            self._tick_lines.visible = True
        else:
            self._tick_lines.visible = False
        yt = yt[yt <= y1 - (y1 - y0) * 0.06]          # keep clear of the top label row
        xt = xt[xt >= x0 + (x1 - x0) * 0.08]          # keep clear of the left label column
        if yt.size:
            xoff = x0 + (x1 - x0) * 0.02
            self._tick_ytxt.text = [_tick_label(v, step) for v in yt]
            self._tick_ytxt.pos = np.column_stack(
                [np.full(yt.size, xoff), yt, np.zeros(yt.size)]).astype("float32")
            self._tick_ytxt.visible = True
        else:
            self._tick_ytxt.visible = False
        if xt.size:
            yoff = y1 - (y1 - y0) * 0.02
            self._tick_xtxt.text = [_tick_label(v, step) for v in xt]
            self._tick_xtxt.pos = np.column_stack(
                [xt, np.full(xt.size, yoff), np.zeros(xt.size)]).astype("float32")
            self._tick_xtxt.visible = True
        else:
            self._tick_xtxt.visible = False
        self._vcanvas.update()

    def set_camera_rect(self, rect):
        """Force this panel's camera to an absolute rect with no
        broadcasting. Used only for non-interactive resets (reset_camera /
        initial layout), where every panel is set independently from the
        same max_range_km rather than mirrored from one source panel."""
        self._view.camera.rect = rect
        self._schedule_ticks()
        self._force_redraw_now()

    def reset_camera(self, max_range_km=None):
        mr = max_range_km or self._max_range
        if getattr(self, "_is_rhi", False):
            # RHI: x is horizontal distance (0..mr), y is altitude (0..mr)
            margin = mr * 1.05
            self.set_camera_rect(Rect(-mr * 0.02, -mr * 0.05,
                                      margin * 1.1, margin * 0.9))
        else:
            margin = mr * 1.05
            self.set_camera_rect(Rect(-margin, -margin, margin * 2, margin * 2))

    def _force_redraw_now(self):
        """Queue a repaint without blocking the Tk input handler.

        NLR keeps panning and zooming responsive by changing only GPU-side
        transforms during the gesture and letting the event loop coalesce
        draws. GURT mirrors that here: sibling panels still receive the same
        camera transform immediately, but their actual GL redraw is queued
        instead of forced synchronously inside the mouse callback.
        """
        self._vcanvas.update()

    def _canvas_size(self):
        """This panel's own canvas size in pixels, as (width, height)."""
        try:
            w, h = self._vcanvas.native.winfo_width(), self._vcanvas.native.winfo_height()
        except Exception:
            w, h = 0, 0
        if w <= 0 or h <= 0:
            # Not yet mapped/sized by Tk; fall back to a sane default
            # rather than dividing by zero below.
            w, h = 1, 1
        return w, h

    def _canvas_to_scene(self, cx, cy):
        try:
            # Map canvas pixel -> scene (km) coords via the camera's inverse transform
            tr = self._view.camera.transform
            pos = tr.imap([cx, cy, 0, 1])
            return float(pos[0]), float(pos[1])
        except Exception:
            try:
                # Fallback: use scene node transform
                tr = self._view.scene.transform
                pos = tr.imap([cx, cy, 0, 1])
                return float(pos[0]), float(pos[1])
            except Exception:
                return 0.0, 0.0

    # ── Zoom / pan, own-panel math ───────────────────────────────────────────
    #
    # Both gestures below are computed in two parts:
    #   1. A *local* step applied to this panel's own camera, using this
    #      panel's own pixel size and current rect (zoom_by / pan_by).
    #   2. A *sync* broadcast to sibling panels, expressed as a fraction
    #      of this panel's own canvas/rect — never as an absolute pixel
    #      or world-space value — so every panel can replay the identical
    #      gesture against its own pixel size and current rect. This is
    #      what keeps multi-panel zoom/pan locked together even when
    #      panels differ slightly in pixel dimensions (which changes how
    #      VisPy's aspect=1.0 camera pads each panel's effective rect).

    def zoom_by(self, factor: float, frac_x: float = 0.5, frac_y: float = 0.5):
        """Zoom this panel's own rect by `factor`, anchored at the point
        that sits at (frac_x, frac_y) of this panel's own current rect
        (0,0 = bottom-left, 1,1 = top-right; default 0.5,0.5 = center).
        """
        r = self._view.camera.rect
        px = r.left   + frac_x * r.width
        py = r.bottom + frac_y * r.height
        new_w = r.width  / factor
        new_h = r.height / factor
        new_x = px - frac_x * new_w
        new_y = py - frac_y * new_h
        self._view.camera.rect = Rect(new_x, new_y, new_w, new_h)
        self._schedule_ticks()

    def pan_by(self, dfrac_x: float, dfrac_y: float):
        """Shift this panel's own rect by a fraction of its own current
        width/height (dfrac_x, dfrac_y), e.g. dfrac_x=0.1 pans right by
        10% of the panel's own current view width."""
        r = self._view.camera.rect
        ddx = dfrac_x * r.width
        ddy = dfrac_y * r.height
        self._view.camera.rect = Rect(r.left - ddx, r.bottom - ddy,
                                      r.width, r.height)
        self._schedule_ticks()

    def sync_zoom(self, factor: float, frac_x: float, frac_y: float):
        """Apply a zoom gesture mirrored from another panel: same factor,
        same anchor fraction of *this* panel's own current rect."""
        self.zoom_by(factor, frac_x, frac_y)
        self._force_redraw_now()

    def sync_pan(self, dfrac_x: float, dfrac_y: float):
        """Apply a pan gesture mirrored from another panel: same fractional
        shift of *this* panel's own current rect."""
        self.pan_by(dfrac_x, dfrac_y)
        self._force_redraw_now()

    # ── Mouse events ──────────────────────────────────────────────────────────

    def _on_press(self, event):
        self._app._active_panel = self.panel
        det = getattr(self._app, "_detect", None)
        if det is not None and det.box_picking:
            x_km, y_km = self._canvas_to_scene(event.x, event.y)
            det.box_press(self.panel, x_km, y_km)
            self._brush_active     = False
            self._drag_start_scene = None
            self._dragging = False
            return
        xsc = getattr(self._app, "_xsect", None)
        if xsc is not None and xsc.picking:
            x_km, y_km = self._canvas_to_scene(event.x, event.y)
            xsc.run_at(self.panel, x_km, y_km)
            self._brush_active     = False
            self._drag_start_scene = None
            self._dragging = False
            return
        axi = getattr(self._app, "_axisym", None)
        if axi is not None and axi.picking:
            x_km, y_km = self._canvas_to_scene(event.x, event.y)
            axi.run_at(self.panel, x_km, y_km)
            self._brush_active     = False
            self._drag_start_scene = None
            self._dragging = False
            return
        if axi is not None and axi._overlay is not None:
            x_km, y_km = self._canvas_to_scene(event.x, event.y)
            x1, _ = self._canvas_to_scene(event.x + 12, event.y)   # ~12 px grab radius
            hit = axi.hit_test(x_km, y_km, abs(x1 - x_km))
            if hit is not None:
                axi._panel = self.panel
                axi.begin_drag(hit)
                self._brush_active     = False
                self._drag_start_scene = None
                self._dragging = False
                return
        mode = self._app._editor_mode.get()
        if mode in EDIT_BRUSH_MODES and self._app._boundary_mode.get():
            # Boundary tool: each click adds a vertex; nothing is edited
            # until the user presses Enter / "Apply Edit".
            x_km, y_km = self._canvas_to_scene(event.x, event.y)
            self._add_boundary_point(x_km, y_km)
            self._brush_active     = False
            self._drag_start_scene = None
            self._dragging = False
            return
        if mode in EDIT_BRUSH_MODES:
            self._app._apply_brush(self.panel, event.x, event.y, mode)
            self._brush_active = True
            return
        self._brush_active     = False
        self._drag_start_scene = (event.x, event.y)
        self._dragging = False

    def _on_drag(self, event):
        det = getattr(self._app, "_detect", None)
        if det is not None and det._box_start is not None:
            x_km, y_km = self._canvas_to_scene(event.x, event.y)
            det.box_drag(x_km, y_km)
            return
        axi = getattr(self._app, "_axisym", None)
        if axi is not None and axi._drag is not None:
            x_km, y_km = self._canvas_to_scene(event.x, event.y)
            axi.drag_to(x_km, y_km)
            return
        if getattr(self, "_brush_active", False):
            x_km, y_km = self._canvas_to_scene(event.x, event.y)
            self._update_brush_circle(x_km, y_km)

            if not self._brush_throttle.ready():
                return

            mode = self._app._editor_mode.get()
            self._app._apply_brush(self.panel, event.x, event.y, mode)
            return
        if self._drag_start_scene is None:
            return
        dx = event.x - self._drag_start_scene[0]
        dy = event.y - self._drag_start_scene[1]
        if not self._dragging and (abs(dx) > 4 or abs(dy) > 4):
            self._dragging = True
        if not self._dragging:
            return
        if not self._pan_throttle.ready():
            return
        # Step delta since the last processed motion event (not since the
        # initial press), expressed as a fraction of this panel's own
        # *current rect* — but derived via the camera's real imap-based
        # transform (_canvas_to_scene), not a raw cx/cw pixel ratio.
        #
        # Why: with camera.aspect=1.0, VisPy pads the rect into a
        # `_real_rect` whose padding depends on *this panel's own* pixel
        # aspect ratio (see PanZoomCamera._update_transform in panzoom.py).
        # A raw pixel-fraction (step_dx / cw) is only equal to a fraction
        # of `rect` when a panel happens to have zero padding (i.e. its
        # widget is exactly square). Routing through _canvas_to_scene
        # (which uses the real, padding-aware transform) gives the true
        # scene-space delta, so the fraction broadcast to sibling panels
        # reproduces the same scene-space pan on every panel regardless of
        # each panel's own pixel dimensions.
        last = self._pan_last_xy or self._drag_start_scene
        x0, y0 = self._canvas_to_scene(*last)
        x1, y1 = self._canvas_to_scene(event.x, event.y)
        self._pan_last_xy = (event.x, event.y)
        if x0 == x1 and y0 == y1:
            return
        r = self._view.camera.rect
        dfrac_x = (x1 - x0) / r.width if r.width else 0.0
        dfrac_y = (y1 - y0) / r.height if r.height else 0.0
        self.pan_by(dfrac_x, dfrac_y)
        self._force_redraw_now()
        self._app._sync_pan_from(self, dfrac_x, dfrac_y)

    def _on_release(self, event):
        det = getattr(self._app, "_detect", None)
        if det is not None and det._box_start is not None:
            det.box_release()
            return
        axi = getattr(self._app, "_axisym", None)
        if axi is not None and axi._drag is not None:
            axi.end_drag()
            return
        if getattr(self, "_brush_active", False):
            self._brush_active        = False
            if self.panel is not None:
                self.panel._brush_undo_pushed = False
            return
        self._drag_start_scene = None
        self._pan_last_xy      = None
        self._dragging = False

    def _on_right_click(self, event):
        self._app._active_panel = self.panel
        det = getattr(self._app, "_detect", None)
        if det is not None and det.box_picking:
            det.cancel_box()
            return
        xsc = getattr(self._app, "_xsect", None)
        if xsc is not None and xsc.picking:
            xsc._cancel_pick()
            return
        axi = getattr(self._app, "_axisym", None)
        if axi is not None and axi.picking:
            axi._cancel_pick()
            return
        mode = self._app._editor_mode.get()
        if (mode in EDIT_BRUSH_MODES and self._app._boundary_mode.get()
                and self._boundary_points):
            # Boundary tool: right-click undoes the last placed vertex
            # instead of opening the normal context menu.
            self._remove_last_boundary_point()
            return
        self._app._show_right_click_menu(self.panel, event)

    def _on_mouse_move(self, event):
        # <Motion> also fires during <B1-Motion>. While panning, skip
        # cursor/readout work entirely; NLR keeps movement cheap by only
        # changing transforms during the gesture.
        if self._drag_start_scene is not None or getattr(self, "_brush_active", False):
            return
        _axi = getattr(self._app, "_axisym", None)
        if _axi is not None and _axi._drag is not None:
            return
        _det = getattr(self._app, "_detect", None)
        if _det is not None and _det._box_start is not None:
            return
        x_km, y_km = self._canvas_to_scene(event.x, event.y)
        self._app._update_cursor(self.panel, x_km, y_km)
        mode = self._app._editor_mode.get()
        if mode in EDIT_BRUSH_MODES and self._app._boundary_mode.get():
            self._brush_circle.visible = False
            self._update_boundary_visual(preview_xy=(x_km, y_km))
        else:
            self._update_brush_circle(x_km, y_km)

    def _update_brush_circle(self, x_km, y_km):
        mode = self._app._editor_mode.get()
        if mode in EDIT_BRUSH_MODES and not self._app._boundary_mode.get():
            r = max(0.0001, self._app._brush_radius_km.get())
            # Only update the transform — no geometry rebuild.
            self._brush_circle.transform = STTransform(
                scale=(r, r, 1), translate=(x_km, y_km, 0))
            self._brush_circle.visible = True
        else:
            self._brush_circle.visible = False
        self._vcanvas.update()

    # ── Boundary tool ─────────────────────────────────────────────────────────
    #
    # A lightweight "draw a polygon, then apply once" alternative to the
    # dragged circular brush. `_boundary_points` lives on the canvas (not
    # the app) because the boundary being drawn belongs to whichever
    # panel the user is clicking in — exactly like `_brush_active` above.

    def _add_boundary_point(self, x_km, y_km):
        self._boundary_points.append((x_km, y_km))
        self._update_boundary_visual()
        self._app._on_boundary_changed()

    def _remove_last_boundary_point(self):
        if self._boundary_points:
            self._boundary_points.pop()
        self._update_boundary_visual()
        self._app._on_boundary_changed()

    def _clear_boundary(self):
        if not self._boundary_points:
            return
        self._boundary_points = []
        self._boundary_line.visible    = False
        self._boundary_markers.visible = False
        self._vcanvas.update()
        self._app._on_boundary_changed()

    def _update_boundary_visual(self, preview_xy=None):
        """Redraw the boundary polyline + vertex markers from
        `_boundary_points`. If `preview_xy` (the live cursor position) is
        given, an extra "rubber band" segment out to the cursor is drawn
        too, without it being a committed vertex."""
        pts = list(self._boundary_points)
        if preview_xy is not None and pts:
            pts = pts + [preview_xy]

        if len(pts) >= 2:
            arr = np.array([[x, y, 0.0] for x, y in pts], dtype="float32")
            self._boundary_line.set_data(arr)
            self._boundary_line.visible = True
        else:
            self._boundary_line.visible = False

        if self._boundary_points:
            marr = np.array([[x, y, 0.0] for x, y in self._boundary_points],
                            dtype="float32")
            self._boundary_markers.set_data(
                pos=marr,
                face_color=(1.0, 0.25, 0.85, 1.0),
                edge_color=(1.0, 1.0, 1.0, 0.9),
                size=7,
            )
            self._boundary_markers.visible = True
        else:
            self._boundary_markers.visible = False
        self._vcanvas.update()

    def _on_scroll(self, event):
        self._app._active_panel = self.panel
        factor = 1.15 if event.delta > 0 else 1.0 / 1.15
        self._scroll_zoom_common(event.x, event.y, factor)

    def _scroll_zoom(self, event, direction):
        self._app._active_panel = self.panel
        factor = 1.15 if direction > 0 else 1.0 / 1.15
        self._scroll_zoom_common(event.x, event.y, factor)

    def _scroll_zoom_common(self, cx, cy, factor):
        # Anchor point for the zoom, computed via the camera's real
        # imap-based transform (_canvas_to_scene) rather than a raw
        # cx/cw, cy/ch pixel fraction.
        #
        # Why: with camera.aspect=1.0, VisPy pads `rect` into a
        # `_real_rect` whose padding depends on *this panel's own* pixel
        # aspect ratio (see PanZoomCamera._update_transform in panzoom.py).
        # cx/cw is only a correct stand-in for "fraction of rect" when a
        # panel's widget happens to be exactly square (no padding) — any
        # mismatch in pixel dimensions between panels means the same
        # frac_x/frac_y broadcast to siblings anchors at a different
        # scene location, and (combined with the analogous pan bug)
        # produces visibly different zoom/pan magnitudes per panel.
        # _canvas_to_scene already uses the padding-aware transform, so
        # deriving frac_x/frac_y from it and *then* broadcasting the
        # fraction keeps every panel zooming by the same scene-space
        # anchor and amount, regardless of its own pixel dimensions.
        r = self._view.camera.rect
        px, py = self._canvas_to_scene(cx, cy)
        frac_x = (px - r.left) / r.width if r.width else 0.5
        frac_y = (py - r.bottom) / r.height if r.height else 0.5
        self.zoom_by(factor, frac_x, frac_y)
        self._force_redraw_now()
        self._app._sync_zoom_from(self, factor, frac_x, frac_y)

    def _on_shift_scroll(self, event):
        """Shift+scroll (Windows/macOS): resize the brush, never zoom."""
        mode = self._app._editor_mode.get()
        if mode in EDIT_BRUSH_MODES:
            self._resize_brush(+1 if event.delta > 0 else -1, event.x, event.y)
        return "break"   # prevent event bubbling to normal scroll handler

    def _on_shift_scroll_dir(self, event, direction: int):
        """Shift+Button-4/5 (Linux): resize the brush, never zoom."""
        mode = self._app._editor_mode.get()
        if mode in EDIT_BRUSH_MODES:
            self._resize_brush(direction, event.x, event.y)
        return "break"

    def _resize_brush(self, direction: int, cx: int, cy: int):
        """Grow or shrink the brush radius by ~15 % per scroll tick."""
        r = self._app._brush_radius_km.get()
        r = r * 1.15 if direction > 0 else r / 1.15
        r = max(0.01, round(r, 3))   # clamp to 50 m minimum, tidy value
        self._app._brush_radius_km.set(r)
        x_km, y_km = self._canvas_to_scene(cx, cy)
        self._update_brush_circle(x_km, y_km)

    def _try_fast_color_patch(self, face_idx: np.ndarray, rgba: np.ndarray) -> bool:
        """Push only the changed face colors directly into the mesh's
        existing GPU color buffer via Buffer.set_subdata(), instead of
        going through MeshVisual.set_data().

        set_data() rebuilds a brand-new MeshData and re-unshares +
        re-uploads the colors (and vertices) for *every* face in the
        mesh on every call — cost scales with total gate count, not
        with how many gates actually changed. That's why brushing 10
        gates used to feel exactly as slow as brushing 10,000: the
        per-dab cost was O(total gates) regardless. This patches just
        the changed rows directly in the existing GPU buffer, so cost
        scales with the brush size instead.

        face_idx : flat face indices (each gate = 2 triangle faces).
        rgba     : (len(face_idx), 4) float32 colors, one row per face.

        Returns True if the GPU buffer was patched directly (no full
        rebuild needed), or False if anything about vispy's internal
        buffer layout didn't match what we expect (e.g. a vispy version
        change) — callers should fall back to the full set_data() path
        in that case. Deliberately defensive since this relies on
        private vispy attributes (verified against vispy 0.16.2:
        Visual._meshdata/_data_changed, shared_program.vert[...]).
        """
        try:
            if self._color_vbo is None or self._unshared_colors is None:
                mesh = self._mesh
                if getattr(mesh, "_data_changed", False):
                    mesh._update_data()
                vbo = mesh.shared_program.vert["base_color"]
                if not isinstance(vbo, VertexBuffer):
                    return False
                n_faces = self._faces.shape[0]
                if vbo.size != n_faces * 3:
                    return False
                self._color_vbo = vbo
                # Mirrors MeshData.get_face_colors(indexed='faces'): each
                # face contributes 3 identical consecutive rows (flat
                # shading needs one color per vertex *instance*, not per
                # shared vertex).
                self._unshared_colors = np.repeat(
                    self._face_colors, 3, axis=0).astype("float32", copy=True)

            unshared = self._unshared_colors

            order = np.argsort(face_idx)
            sorted_idx  = np.asarray(face_idx)[order]
            sorted_rgba = np.asarray(rgba)[order]

            rows = (sorted_idx[:, None] * 3 + np.arange(3)).ravel()
            unshared[rows] = np.repeat(sorted_rgba, 3, axis=0)

            # Coalesce consecutive face indices into single GPU writes so
            # a scattered brush dab doesn't turn into hundreds of tiny
            # set_subdata calls.
            breaks = np.flatnonzero(np.diff(sorted_idx) > 1) + 1
            for run in np.split(np.arange(len(sorted_idx)), breaks):
                f0 = int(sorted_idx[run[0]])
                f1 = int(sorted_idx[run[-1]]) + 1
                # offset is in elements (rows), not bytes — VertexBuffer
                # multiplies by itemsize internally.
                self._color_vbo.set_subdata(unshared[f0 * 3:f1 * 3], offset=f0 * 3)
            return True
        except Exception:
            self._color_vbo       = None
            self._unshared_colors = None
            return False

    def patch_face_colors(self, in_brush: np.ndarray, vals: np.ndarray,
                           cmap, vmin: float, vmax: float):
        """
        Recolour only the gates inside in_brush (shape nrays×ngates bool),
        using already-computed `vals` (nrays×ngates float32).
        Skips geometry rebuild and overlay update — much faster than a full refresh.
        """
        nrays, ngates = vals.shape

        # Find which (ray, gate) pairs are in the brush
        ai_arr, gi_arr = np.where(in_brush)
        if self._image.visible and self._image_rgba is not None:
            self.patch_gate_colors(ai_arr, gi_arr, vals[ai_arr, gi_arr],
                                   ngates, cmap, vmin, vmax)
            return
        if self._face_colors is None or self._faces is None:
            return
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        if len(ai_arr) == 0:
            return

        # Each gate maps to two triangle faces: 2*(ray*ngates + gate) and +1.
        face_idx = 2 * (ai_arr.astype(np.int64) * ngates + gi_arr.astype(np.int64))
        keep = face_idx + 1 < len(self._face_colors)
        if not np.any(keep):
            return
        face_idx = face_idx[keep]
        patch_vals = vals[ai_arr[keep], gi_arr[keep]]
        finite = np.isfinite(patch_vals)

        rgba = np.zeros((len(patch_vals), 4), dtype="float32")
        if np.any(finite):
            rgba[finite] = np.asarray(cmap(norm(patch_vals[finite])), dtype="float32")
        # rgba[~finite] stays all-zero (alpha 0 -> masked/NaN gates invisible)

        all_face_idx = np.concatenate([face_idx, face_idx + 1])
        all_rgba     = np.concatenate([rgba, rgba])
        self._face_colors[all_face_idx] = all_rgba

        if not self._try_fast_color_patch(all_face_idx, all_rgba):
            self._mesh.set_data(vertices=self._verts, faces=self._faces,
                                face_colors=self._face_colors)
        self._vcanvas.update()

    def patch_gate_colors(self, ai_arr: np.ndarray, gi_arr: np.ndarray,
                          patch_vals: np.ndarray, ngates: int,
                          cmap, vmin: float, vmax: float, draw: bool = True):
        """Recolour an indexed set of gates without rebuilding geometry.

        `draw` controls whether this call triggers an immediate on-screen
        repaint. Live brush painting wants draw=True (every dab should be
        visible right away). Callers that may fire in rapid bursts — e.g.
        Ctrl+Z/Ctrl+Y being held down, which auto-repeats many undo/redo
        steps per second — should pass draw=False and debounce the actual
        repaint themselves (see App._schedule_patch_flush), otherwise every
        intermediate undo step gets painted to the screen instead of just
        the final one.
        """
        if len(ai_arr) == 0:
            return

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        if self._image.visible and self._image_rgba is not None:
            ai_arr = np.asarray(ai_arr, dtype=np.int64)
            gi_arr = np.asarray(gi_arr, dtype=np.int64)
            keep = ((ai_arr >= 0) & (gi_arr >= 0) &
                    (ai_arr < self._image_rgba.shape[0]) &
                    (gi_arr < self._image_rgba.shape[1]))
            if not np.any(keep):
                return
            ai_arr = ai_arr[keep]
            gi_arr = gi_arr[keep]
            patch_vals = np.asarray(patch_vals, dtype="float32")[keep]
            finite = np.isfinite(patch_vals)
            rgba = np.zeros((len(patch_vals), 4), dtype="uint8")
            if np.any(finite):
                patch_rgba = np.asarray(cmap(norm(patch_vals[finite])), dtype="float32")
                rgba[finite] = np.clip(np.rint(patch_rgba * 255.0), 0, 255).astype("uint8")
            self._image_rgba[ai_arr, gi_arr] = rgba
            self._set_image_rgba(self._image_rgba)
            if draw:
                self._vcanvas.update()
            return

        if self._face_colors is None or self._faces is None:
            return
        face_idx = 2 * (ai_arr.astype(np.int64) * ngates + gi_arr.astype(np.int64))
        keep = face_idx + 1 < len(self._face_colors)
        if not np.any(keep):
            return

        face_idx = face_idx[keep]
        patch_vals = patch_vals[keep]
        finite = np.isfinite(patch_vals)

        rgba = np.zeros((len(patch_vals), 4), dtype="float32")
        if np.any(finite):
            rgba[finite] = np.asarray(cmap(norm(patch_vals[finite])), dtype="float32")
        # rgba[~finite] stays all-zero (alpha 0 -> masked/NaN gates invisible)

        all_face_idx = np.concatenate([face_idx, face_idx + 1])
        all_rgba     = np.concatenate([rgba, rgba])
        self._face_colors[all_face_idx] = all_rgba

        if not self._try_fast_color_patch(all_face_idx, all_rgba):
            self._mesh.set_data(vertices=self._verts, faces=self._faces,
                                face_colors=self._face_colors)
        if draw:
            self._vcanvas.update()

    def screenshot(self) -> np.ndarray:
        return self._vcanvas.render()


# ══════════════════════════════════════════════════════════════════════════════
# ParamColorDialog — field / colormap editor  (ported from gurt_guiv3)
# ══════════════════════════════════════════════════════════════════════════════

class ParamColorDialog(tk.Toplevel):
    """Edit field name, vmin/vmax, colormap, and display colours for one panel."""

    LABEL_W = 18

    def __init__(self, parent, slot, field_info, available_fields, on_apply):
        super().__init__(parent)
        fname, label, vmin, vmax, cmap_name = field_info
        self.slot     = slot
        self.on_apply = on_apply
        self.avail    = available_fields
        self.transient(parent)
        self.grab_set()
        self.resizable(False, False)
        self.title(f"Frame {slot+1} Parameter and Colors")
        self.configure(bg=DLG_BG)

        top = tk.Frame(self, bg=DLG_BG, pady=4, padx=6)
        top.pack(fill="x")
        tk.Label(top, text="17 Colors", bg=DLG_BG, relief="groove",
                 padx=8).pack(side="left", padx=4)
        tk.Button(top, text="Replot",  command=self._replot,  relief="raised",
                  bg=DLG_BG, padx=12).pack(side="left", padx=4)
        tk.Button(top, text="  OK  ",  command=self._ok,      relief="raised",
                  bg=DLG_BG, padx=12).pack(side="left", padx=4)
        tk.Button(top, text="Cancel",  command=self.destroy,  relief="raised",
                  bg=DLG_BG, padx=12).pack(side="left", padx=4)

        body = tk.Frame(self, bg=DLG_BG)
        body.pack(fill="both", expand=True, padx=6, pady=4)

        left = tk.Frame(body, bg=DLG_BG)
        left.pack(side="left", fill="y", padx=(0, 10))
        tk.Label(left, text="Parameters", bg=DLG_BG,
                 font=("TkDefaultFont", 9, "bold")).pack()
        tk.Label(left, text="(Double-click to select)", bg=DLG_BG,
                 font=("TkDefaultFont", 7), fg="#555").pack()
        self.listbox = tk.Listbox(left, width=14, height=18,
                                  bg="#ffffff", selectbackground="#000080",
                                  selectforeground="#ffffff", relief="sunken")
        sb = tk.Scrollbar(left, orient="vertical", command=self.listbox.yview)
        self.listbox.config(yscrollcommand=sb.set)
        self.listbox.pack(side="left", fill="y")
        sb.pack(side="left", fill="y")
        for f in available_fields:
            self.listbox.insert(tk.END, f)
        if fname in available_fields:
            idx = available_fields.index(fname)
            self.listbox.selection_set(idx)
            self.listbox.see(idx)
        self.listbox.bind("<Double-1>", self._list_select)

        right = tk.Frame(body, bg=DLG_BG)
        right.pack(side="left", fill="both", expand=True)

        self._fname_var   = tk.StringVar(value=fname)
        self._vmin_var    = tk.StringVar(value=str(vmin))
        self._vmax_var    = tk.StringVar(value=str(vmax))
        ctr = (vmin + vmax) / 2
        inc = (vmax - vmin) / 10
        self._ctr_var     = tk.StringVar(value=f"{ctr:.3f}")
        self._inc_var     = tk.StringVar(value=f"{inc:.3f}")
        self._label_var   = tk.StringVar(value=label)

        if not isinstance(cmap_name, str):
            cmap_name = getattr(cmap_name, 'name', 'gurt_ref')
        self._cmap_var    = tk.StringVar(value=cmap_name)

        self._bg_var      = tk.StringVar(value="midnightblue")
        self._missing_var = tk.StringVar(value="darkslateblue")
        self._exceed_var  = tk.StringVar(value="gray70")
        self._annot_var   = tk.StringVar(value="gray90")
        self._emph_var    = tk.StringVar(value="hotpink")
        self._emin_var    = tk.StringVar(value="0.000")
        self._emax_var    = tk.StringVar(value="0.000")

        rows = [
            ("Parameter Name",     self._fname_var,   False),
            ("Min",                self._vmin_var,    False),
            ("Max",                self._vmax_var,    False),
            ("Center",             self._ctr_var,     False),
            ("Increment",          self._inc_var,     False),
            ("Color Palette",      self._cmap_var,    True),
            ("Label",              self._label_var,   False),
            (None, None, None),
            ("Background Color",   self._bg_var,      False),
            ("Missing Data Color", self._missing_var, False),
            ("Exceeded Color",     self._exceed_var,  False),
            ("Annotation Color",   self._annot_var,   False),
            ("Emphasis Color",     self._emph_var,    False),
            ("Emphasis Min",       self._emin_var,    False),
            ("Emphasis Max",       self._emax_var,    False),
        ]

        for i, row in enumerate(rows):
            if row[0] is None:
                ttk.Separator(right, orient="horizontal").grid(
                    row=i, column=0, columnspan=2, sticky="ew", pady=3)
                continue
            lbl, var, is_combo = row
            tk.Label(right, text=lbl, bg=DLG_BG, anchor="e",
                     width=self.LABEL_W, font=("TkDefaultFont", 9, "bold")
                     ).grid(row=i, column=0, sticky="e", padx=4, pady=2)
            if is_combo:
                cb = ttk.Combobox(right, textvariable=var,
                                  values=ALL_CMAP_NAMES, width=24, state="normal")
                cb.grid(row=i, column=1, sticky="w", padx=4, pady=2)
                cb.bind("<<ComboboxSelected>>", self._preview_cmap)
            else:
                tk.Entry(right, textvariable=var, width=26,
                         bg=DLG_ENTRY, relief="sunken"
                         ).grid(row=i, column=1, sticky="w", padx=4, pady=2)

        tk.Label(right, text="Colormap Preview:", bg=DLG_BG,
                 font=("TkDefaultFont", 8)).grid(
            row=len(rows), column=0, columnspan=2, sticky="w", padx=4, pady=(6, 0))
        self._cmap_canvas = tk.Canvas(right, height=20, width=300,
                                      bg="#000", relief="sunken")
        self._cmap_canvas.grid(row=len(rows)+1, column=0, columnspan=2,
                               sticky="ew", padx=4, pady=2)
        self._draw_cmap_preview(cmap_name)
        self._cmap_var.trace_add("write", lambda *_: self._draw_cmap_preview(
            self._cmap_var.get()))

    def _draw_cmap_preview(self, name):
        c = self._cmap_canvas
        w = c.winfo_reqwidth() or 300
        c.delete("all")
        try:
            cm = resolve_cmap(name)
            for i in range(w):
                r, g, b, _ = cm(i / w)
                color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
                c.create_line(i, 0, i, 20, fill=color)
        except Exception:
            pass

    def _preview_cmap(self, event=None):
        self._draw_cmap_preview(self._cmap_var.get())

    def _list_select(self, event=None):
        sel = self.listbox.curselection()
        if not sel:
            return
        fname = self.avail[sel[0]]
        self._fname_var.set(fname)
        if fname in FIELD_DEFAULTS:
            lbl, mn, mx, cm = FIELD_DEFAULTS[fname]
            self._vmin_var.set(str(mn))
            self._vmax_var.set(str(mx))
            self._label_var.set(lbl)
            cname = cm if isinstance(cm, str) else getattr(cm, 'name', 'gurt_ref')
            self._cmap_var.set(cname)
            self._ctr_var.set(f"{(mn+mx)/2:.3f}")
            self._inc_var.set(f"{(mx-mn)/10:.3f}")

    def _collect(self):
        fname  = self._fname_var.get().strip()
        label  = self._label_var.get().strip() or fname
        cname  = self._cmap_var.get().strip()
        try:    vmin = float(self._vmin_var.get())
        except: vmin = -50
        try:    vmax = float(self._vmax_var.get())
        except: vmax = 50
        return fname, label, vmin, vmax, cname

    def _replot(self):
        self.on_apply(self.slot, *self._collect())

    def _ok(self):
        self.on_apply(self.slot, *self._collect())
        self.destroy()


# ══════════════════════════════════════════════════════════════════════════════
# DuplicateFieldDialog  (ported from gurt_guiv3)
# ══════════════════════════════════════════════════════════════════════════════

class DuplicateFieldDialog(tk.Toplevel):
    """Modal dialog: choose a source field and type a destination name."""

    def __init__(self, parent, available_fields, on_apply):
        super().__init__(parent)
        self.on_apply = on_apply
        self.title("Duplicate Field")
        self.configure(bg=DLG_BG)
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        pad = dict(padx=10, pady=5)

        tk.Label(self, text="Source field:", bg=DLG_BG, fg=DLG_FG,
                 anchor="w").grid(row=0, column=0, sticky="w", **pad)
        self._src_var = tk.StringVar()
        src_cb = ttk.Combobox(self, textvariable=self._src_var,
                              values=available_fields, width=18, state="readonly")
        if available_fields:
            src_cb.current(0)
        src_cb.grid(row=0, column=1, sticky="w", **pad)

        tk.Label(self, text="Output field name:", bg=DLG_BG, fg=DLG_FG,
                 anchor="w").grid(row=1, column=0, sticky="w", **pad)
        self._dst_var = tk.StringVar()
        tk.Entry(self, textvariable=self._dst_var, width=20,
                 bg=DLG_ENTRY, relief="sunken").grid(row=1, column=1, sticky="w", **pad)

        btn_frame = tk.Frame(self, bg=DLG_BG)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=8)
        tk.Button(btn_frame, text="Duplicate", command=self._ok,
                  bg=DLG_BG, relief="raised", padx=10).pack(side="left", padx=4)
        tk.Button(btn_frame, text="Cancel", command=self.destroy,
                  bg=DLG_BG, relief="raised", padx=10).pack(side="left", padx=4)

    def _ok(self):
        self.on_apply(self._src_var.get(), self._dst_var.get())
        self.destroy()


# ══════════════════════════════════════════════════════════════════════════════
# EditorPanel — floating brush toolbar  (ported from gurt_guiv3)
# ══════════════════════════════════════════════════════════════════════════════

class EditorPanel(tk.Toplevel):
    """
    Floating panel hosting editing tools (Unfold Brush, Deglitch Brush,
    Gap Fill Brush, Delete Gates) plus the Boundary Tool, which applies
    the same four edit modes to a clicked polygon instead of a dragged
    circle.

    Reads/writes shared state vars from the parent GURTVispy app:
        _editor_mode, _boundary_mode, _brush_radius_km, _brush_nyquist,
        _brush_center, _stagger_* ,  _deglitch_*
    """

    def __init__(self, app: "GURTVispy"):
        super().__init__(app)
        self.app = app
        self.title("GURT Editor")
        self.configure(bg=DLG_BG)
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build()
        self.update_idletasks()
        mx = app.winfo_x() + app.winfo_width() - self.winfo_reqwidth() - 10
        my = app.winfo_y() + 40
        self.geometry(f"+{mx}+{my}")

    def _lbl(self, parent, text):
        return tk.Label(parent, text=text, bg=DLG_BG, fg=DLG_FG,
                        font=("TkDefaultFont", 9))

    def _build(self):
        app = self.app
        pad = dict(padx=8, pady=3)

        hdr = tk.Frame(self, bg="#0d0d3a")
        hdr.pack(fill="x")
        tk.Label(hdr, text="GURT Editor Panel", bg="#0d0d3a", fg="white",
                 font=("TkDefaultFont", 10, "bold"), pady=5).pack()

        mode_frame = tk.LabelFrame(self, text=" Mode ", bg=DLG_BG, fg=DLG_FG,
                                   font=("TkDefaultFont", 9, "bold"),
                                   relief="groove", bd=1)
        mode_frame.pack(fill="x", padx=8, pady=(8, 4))

        for text, value in [
            ("🔍  Zoom / Pan  (default)", "zoom"),
            ("🖌️  Unfold Brush",          "unfold_brush"),
            ("🧹  Deglitch Brush",        "deglitch_brush"),
            ("🌊  Gap Fill Brush",        "gradient_brush"),
            ("🗑️  Delete Gates",          "delete_gate"),
            ("🚦  GateFilter Brush",      "gatefilter_brush"),
        ]:
            tk.Radiobutton(mode_frame, text=text,
                           variable=app._editor_mode, value=value,
                           bg=DLG_BG, fg=DLG_FG, selectcolor=DLG_ABG,
                           activebackground=DLG_ABG, activeforeground=DLG_FG,
                           indicatoron=True, font=("TkDefaultFont", 9),
                           command=self._on_mode_change).pack(anchor="w", **pad)

        # ── The five frames below are built here but deliberately *not*
        # packed yet. Which ones are visible (and in what combination)
        # depends on the selected Mode and on whether the Boundary Tool
        # checkbox is on — that's all decided in one place by
        # _refresh_layout(), called once at the end of _build() and again
        # on every mode/boundary-toggle change. This is what makes a tool
        # you're not using disappear instead of just greying out.

        # Boundary tool (works with whichever of the 4 edit tools above is
        # selected — applies the same edit math, but to a clicked polygon
        # instead of a dragged circle, and only once you confirm it).
        self._boundary_frame = tk.LabelFrame(
            self, text=" Boundary Tool ", bg=DLG_BG, fg=DLG_FG,
            font=("TkDefaultFont", 9, "bold"), relief="groove", bd=1)
        bnd_row = tk.Frame(self._boundary_frame, bg=DLG_BG)
        bnd_row.pack(fill="x", padx=8, pady=(4, 2))
        tk.Checkbutton(
            bnd_row, text="Use boundary (click points, Enter to apply)",
            variable=app._boundary_mode,
            bg=DLG_BG, fg=DLG_FG, selectcolor=DLG_ABG,
            activebackground=DLG_ABG, activeforeground=DLG_FG,
            font=("TkDefaultFont", 9),
            command=self._on_boundary_toggle,
        ).pack(anchor="w")
        # Sub-row: only shown once the checkbox above is actually on.
        self._boundary_apply_row = tk.Frame(self._boundary_frame, bg=DLG_BG)
        tk.Checkbutton(
            self._boundary_apply_row,
            text="Apply to all sweeps in volume",
            variable=app._boundary_all_sweeps,
            bg=DLG_BG, fg=DLG_FG, selectcolor=DLG_ABG,
            activebackground=DLG_ABG, activeforeground=DLG_FG,
            font=("TkDefaultFont", 9),
        ).pack(anchor="w", padx=8, pady=(2, 0))
        self._boundary_status_var = tk.StringVar(value="")
        tk.Label(self._boundary_apply_row, textvariable=self._boundary_status_var,
                 bg=DLG_BG, fg=DLG_HINT, font=("TkDefaultFont", 8),
                 wraplength=250, justify="left").pack(anchor="w", padx=8)

        # Custom deglitch algorithm — only ever shown/usable while the
        # Boundary Tool checkbox above is on (this whole row lives inside
        # _boundary_apply_row, which only packs in that case). Applies to
        # the Deglitch Brush and Gap Fill Brush modes.
        tk.Checkbutton(
            self._boundary_apply_row,
            text="Use custom deglitch algorithm (Deglitch / Gap Fill)",
            variable=app._custom_deglitch_enabled,
            bg=DLG_BG, fg=DLG_FG, selectcolor=DLG_ABG,
            activebackground=DLG_ABG, activeforeground=DLG_FG,
            font=("TkDefaultFont", 9),
            command=self._on_custom_deglitch_toggle,
        ).pack(anchor="w", padx=8, pady=(6, 0))

        self._custom_deglitch_settings = tk.Frame(self._boundary_apply_row, bg=DLG_BG)
        crow = tk.Frame(self._custom_deglitch_settings, bg=DLG_BG)
        crow.pack(fill="x", padx=8)
        self._lbl(crow, "Search radius (gates):").pack(side="left")
        tk.Entry(crow, textvariable=app._custom_deglitch_radius, width=5,
                 bg=DLG_ENTRY, fg="#000", relief="sunken").pack(side="left", padx=4)

        trow = tk.Frame(self._custom_deglitch_settings, bg=DLG_BG)
        trow.pack(fill="x", padx=8, pady=(2, 0))
        self._lbl(trow, "Glitch threshold (m/s):").pack(side="left")
        tk.Entry(trow, textvariable=app._custom_deglitch_thresh, width=6,
                 bg=DLG_ENTRY, fg="#000", relief="sunken").pack(side="left", padx=4)

        tk.Checkbutton(
            self._custom_deglitch_settings,
            text="Deglitch Brush: gap-fill unresolved gates (off = leave"
                 " masked, like the reference script)",
            variable=app._custom_gapfill_enabled,
            bg=DLG_BG, fg=DLG_FG, selectcolor=DLG_ABG,
            activebackground=DLG_ABG, activeforeground=DLG_FG,
            font=("TkDefaultFont", 9), wraplength=250, justify="left",
        ).pack(anchor="w", padx=8, pady=(4, 0))

        grow = tk.Frame(self._custom_deglitch_settings, bg=DLG_BG)
        grow.pack(fill="x", padx=8, pady=(2, 0))
        self._lbl(grow, "Gap fill neighbors:").pack(side="left")
        tk.Entry(grow, textvariable=app._custom_gapfill_n_neighbors, width=5,
                 bg=DLG_ENTRY, fg="#000", relief="sunken").pack(side="left", padx=4)
        self._lbl(grow, "passes:").pack(side="left", padx=(6, 0))
        tk.Entry(grow, textvariable=app._custom_gapfill_n_passes, width=5,
                 bg=DLG_ENTRY, fg="#000", relief="sunken").pack(side="left", padx=4)

        mrow = tk.Frame(self._custom_deglitch_settings, bg=DLG_BG)
        mrow.pack(fill="x", padx=8, pady=(2, 0))
        self._lbl(mrow, "Gap fill method:").pack(side="left")
        tk.OptionMenu(mrow, app._custom_gapfill_method,
                      "median", "mean").pack(side="left", padx=4)

        tk.Label(
            self._custom_deglitch_settings,
            text=("Runs the standalone deglitch.py algorithm over the "
                  "boundary region instead of the quick brush math. Gap "
                  "Fill Brush uses deglitch.py only to detect bad gates, "
                  "then fills them one at a time nearest-to-valid-first."),
            bg=DLG_BG, fg=DLG_HINT, font=("TkDefaultFont", 8),
            wraplength=250, justify="left").pack(anchor="w", padx=8, pady=(2, 4))

        tk.Button(self._boundary_apply_row, text="Apply Edit  (Enter)",
                  command=app._apply_boundary_edit,
                  bg=DLG_BG, fg=DLG_FG, relief="raised",
                  activebackground=DLG_ABG, activeforeground=DLG_FG,
                  padx=8).pack(fill="x", padx=8, pady=(2, 2))
        tk.Button(self._boundary_apply_row, text="Apply to Folder…",
                  command=app._open_batch_boundary_dialog,
                  bg=DLG_BG, fg=DLG_FG, relief="raised",
                  activebackground=DLG_ABG, activeforeground=DLG_FG,
                  padx=8).pack(fill="x", padx=8, pady=(0, 4))

        # Brush radius — only meaningful for the dragged-circle brush, not
        # the boundary tool, so it's hidden whenever Boundary is checked.
        self._brush_frame = tk.LabelFrame(
            self, text=" Brush Settings ", bg=DLG_BG, fg=DLG_FG,
            font=("TkDefaultFont", 9, "bold"), relief="groove", bd=1)
        r0 = tk.Frame(self._brush_frame, bg=DLG_BG)
        r0.pack(fill="x", **pad)
        self._lbl(r0, "Brush radius (km):").pack(side="left")
        tk.Entry(r0, textvariable=app._brush_radius_km, width=7,
                 bg=DLG_ENTRY, fg="#000", relief="sunken").pack(side="left", padx=4)

        # Unfold brush
        self._unfold_frame = tk.LabelFrame(
            self, text=" Unfold Brush Settings ", bg=DLG_BG, fg=DLG_FG,
            font=("TkDefaultFont", 9, "bold"), relief="groove", bd=1)
        r1 = tk.Frame(self._unfold_frame, bg=DLG_BG)
        r1.pack(fill="x", **pad)
        self._lbl(r1, "Nyquist velocity (m/s):").pack(side="left")
        tk.Entry(r1, textvariable=app._brush_nyquist, width=7,
                 bg=DLG_ENTRY, fg="#000", relief="sunken").pack(side="left", padx=4)
        tk.Checkbutton(
            r1, text="Auto (track sweep)",
            variable=app._brush_nyquist_auto,
            bg=DLG_BG, fg=DLG_FG, selectcolor=DLG_ABG,
            activebackground=DLG_ABG, activeforeground=DLG_FG,
            font=("TkDefaultFont", 9),
            command=app._run_stagger_detection,
        ).pack(side="left", padx=4)

        # Fold centre mode selector
        cmode_row = tk.Frame(self._unfold_frame, bg=DLG_BG)
        cmode_row.pack(fill="x", **pad)
        self._lbl(cmode_row, "Fold centre mode:").pack(side="left")
        for lbl, val in [("Manual", "manual"), ("Automatic", "auto")]:
            tk.Radiobutton(
                cmode_row, text=lbl,
                variable=app._unfold_center_mode, value=val,
                bg=DLG_BG, fg=DLG_FG, selectcolor=DLG_ABG,
                activebackground=DLG_ABG, activeforeground=DLG_FG,
                indicatoron=True, font=("TkDefaultFont", 9),
                command=self._refresh_unfold_center_display,
            ).pack(side="left", padx=4)

        # Manual sub-row (visible when mode == "manual")
        self._unfold_manual_row = tk.Frame(self._unfold_frame, bg=DLG_BG)
        self._lbl(self._unfold_manual_row, "Fold centre (m/s):").pack(side="left")
        tk.Entry(self._unfold_manual_row, textvariable=app._brush_center, width=7,
                 bg=DLG_ENTRY, fg="#000", relief="sunken").pack(side="left", padx=4)
        self._lbl(self._unfold_manual_row, "(target velocity)").pack(side="left")

        # Automatic sub-frame (visible when mode == "auto")
        self._unfold_auto_frame = tk.Frame(self._unfold_frame, bg=DLG_BG)

        mult_row = tk.Frame(self._unfold_auto_frame, bg=DLG_BG)
        mult_row.pack(fill="x", pady=1)
        self._lbl(mult_row, "Fold multiplier:").pack(side="left")
        tk.Spinbox(
            mult_row, textvariable=app._unfold_fold_multiplier,
            from_=0.5, to=10.0, increment=0.5, width=5,
            bg=DLG_ENTRY, fg="#000", relief="sunken",
            command=self._refresh_unfold_center_display,
        ).pack(side="left", padx=4)
        self._lbl(mult_row, "\u00d7 Nyquist  (x1\u21921\u00d7Nyq, x1.5\u21922\u00d7Nyq, \u2026)").pack(side="left")

        neg_row = tk.Frame(self._unfold_auto_frame, bg=DLG_BG)
        neg_row.pack(fill="x", pady=1)
        tk.Checkbutton(
            neg_row, text="Negative fold centre",
            variable=app._unfold_negative_center,
            bg=DLG_BG, fg=DLG_FG, selectcolor=DLG_ABG,
            activebackground=DLG_ABG, activeforeground=DLG_FG,
            font=("TkDefaultFont", 9),
            command=self._refresh_unfold_center_display,
        ).pack(side="left")

        ctr_row = tk.Frame(self._unfold_auto_frame, bg=DLG_BG)
        ctr_row.pack(fill="x", pady=(2, 1))
        tk.Label(ctr_row, textvariable=app._unfold_auto_center_label,
                 bg=DLG_BG, fg=DLG_HINT,
                 font=("TkDefaultFont", 9, "bold")).pack(side="left", padx=4)

        # Trace so the label updates live when Nyquist or multiplier is typed
        app._brush_nyquist.trace_add(
            "write", lambda *_: self._refresh_unfold_center_display())
        app._unfold_fold_multiplier.trace_add(
            "write", lambda *_: self._refresh_unfold_center_display())

        self._refresh_unfold_center_display()

        # Deglitch brush
        self._deglitch_frame = tk.LabelFrame(
            self, text=" Deglitch Brush Settings ", bg=DLG_BG, fg=DLG_FG,
            font=("TkDefaultFont", 9, "bold"), relief="groove", bd=1)

        stag_row = tk.Frame(self._deglitch_frame, bg=DLG_BG)
        stag_row.pack(fill="x", padx=8, pady=(4, 2))
        self._stagger_info_var = tk.StringVar(value="No file loaded")
        tk.Label(stag_row, textvariable=self._stagger_info_var,
                 bg=DLG_BG, fg=DLG_HINT,
                 font=("TkDefaultFont", 8), wraplength=250, justify="left"
                 ).pack(anchor="w")

        mode_row = tk.Frame(self._deglitch_frame, bg=DLG_BG)
        mode_row.pack(fill="x", padx=8, pady=2)
        self._lbl(mode_row, "Unfold by:").pack(side="left")
        for lbl, val in [("Short Nyq (default)", "short"),
                          ("Long Nyq", "long"), ("Manual", "manual")]:
            tk.Radiobutton(mode_row, text=lbl,
                           variable=app._deglitch_nyquist_mode, value=val,
                           bg=DLG_BG, fg=DLG_FG, selectcolor=DLG_ABG,
                           activebackground=DLG_ABG, activeforeground=DLG_FG,
                           indicatoron=True, font=("TkDefaultFont", 8),
                           command=self._refresh_stagger_display
                           ).pack(side="left", padx=4)

        man_row = tk.Frame(self._deglitch_frame, bg=DLG_BG)
        man_row.pack(fill="x", padx=8, pady=2)
        self._lbl(man_row, "Manual Nyquist (m/s):").pack(side="left")
        tk.Entry(man_row, textvariable=app._deglitch_manual_nyq,
                 width=7, bg=DLG_ENTRY, fg="#000", relief="sunken"
                 ).pack(side="left", padx=4)

        tol_row = tk.Frame(self._deglitch_frame, bg=DLG_BG)
        tol_row.pack(fill="x", padx=8, pady=2)
        self._lbl(tol_row, "Tolerance (m/s):").pack(side="left")
        tk.Entry(tol_row, textvariable=app._deglitch_tolerance,
                 width=7, bg=DLG_ENTRY, fg="#000", relief="sunken"
                 ).pack(side="left", padx=4)
        self._lbl(tol_row, "(gates > this get deglitched)").pack(side="left")

        # Gap Fill brush settings
        self._gapfill_frame = tk.LabelFrame(
            self, text=" Gap Fill Brush Settings ", bg=DLG_BG, fg=DLG_FG,
            font=("TkDefaultFont", 9, "bold"), relief="groove", bd=1)

        gf_nan_row = tk.Frame(self._gapfill_frame, bg=DLG_BG)
        gf_nan_row.pack(fill="x", padx=8, pady=(4, 2))
        tk.Checkbutton(
            gf_nan_row, text="Fill NaN / empty gates with neighbourhood estimate",
            variable=app._gapfill_fill_nan,
            bg=DLG_BG, fg=DLG_FG, selectcolor=DLG_ABG,
            activebackground=DLG_ABG, activeforeground=DLG_FG,
            font=("TkDefaultFont", 9),
        ).pack(anchor="w")

        # GateFilter Brush settings — area-limited version of the
        # Edit ▸ GateFilter… dialog: an editable list of conditions
        # ("DBZ < 20", "VEL == NaN", …), combined with AND/OR, applied
        # only to gates under the brush/boundary, masking the panel's
        # currently-displayed product — same as every other brush mode,
        # one product at a time.
        self._gatefilter_frame = tk.LabelFrame(
            self, text=" GateFilter Brush Settings ", bg=DLG_BG, fg=DLG_FG,
            font=("TkDefaultFont", 9, "bold"), relief="groove", bd=1)

        self._gf_rows_frame = tk.Frame(self._gatefilter_frame, bg=DLG_BG)
        self._gf_rows_frame.pack(fill="x", padx=4, pady=(4, 0))

        gf_add_row = tk.Frame(self._gatefilter_frame, bg=DLG_BG)
        gf_add_row.pack(fill="x", padx=8, pady=(2, 6))
        tk.Button(gf_add_row, text="+ Add condition",
                  command=self._gf_add_condition,
                  bg=DLG_BG, fg=DLG_FG, relief="raised",
                  font=("TkDefaultFont", 8)).pack(side="left")

        gf_combine_row = tk.Frame(self._gatefilter_frame, bg=DLG_BG)
        gf_combine_row.pack(fill="x", padx=8, pady=(0, 4))
        self._lbl(gf_combine_row, "Combine:").pack(side="left")
        for lbl, val in [("All match (AND)", "and"), ("Any matches (OR)", "or")]:
            tk.Radiobutton(gf_combine_row, text=lbl,
                           variable=app._gatefilter_combine, value=val,
                           bg=DLG_BG, fg=DLG_FG, selectcolor=DLG_ABG,
                           activebackground=DLG_ABG, activeforeground=DLG_FG,
                           indicatoron=True, font=("TkDefaultFont", 8)
                           ).pack(side="left", padx=4)

        tk.Label(self._gatefilter_frame,
                 text="e.g. \"DBZ < 20\" + \"VEL == NaN\" with OR masks gates\n"
                      "where DBZ<20 OR VEL is NaN; the same two conditions\n"
                      "with AND masks only gates where DBZ<20 AND VEL is NaN.\n"
                      "(Try \"Greater than\" on VEL with value 90 for\n"
                      "\"DBZ < 20 and VEL > 90\".)\n"
                      "Masks the field currently shown in this panel — "
                      "switch fields\nto run the same expression against "
                      "a different product.",
                 bg=DLG_BG, fg=DLG_HINT, font=("TkDefaultFont", 8),
                 justify="left").pack(anchor="w", padx=8, pady=(0, 4))

        self._tip_var = tk.StringVar(value="Mode: Zoom")
        self._tip_label = tk.Label(
            self, textvariable=self._tip_var, bg=DLG_BG, fg=DLG_HINT2,
            font=("TkDefaultFont", 8), wraplength=260, justify="left")
        self._tip_label.pack(fill="x", padx=8, pady=2)

        tk.Button(self, text="Undo Last Edit  (Ctrl+Z)",
                  command=app._undo_brush,
                  bg=DLG_BG, fg=DLG_FG, relief="raised",
                  activebackground=DLG_ABG, activeforeground=DLG_FG,
                  padx=8).pack(fill="x", padx=8, pady=2)

        tk.Button(self, text="Redo  (Ctrl+Y)",
                  command=app._redo_brush,
                  bg=DLG_BG, fg=DLG_FG, relief="raised",
                  activebackground=DLG_ABG, activeforeground=DLG_FG,
                  padx=8).pack(fill="x", padx=8, pady=2)

        tk.Frame(self, bg="#333366", height=1).pack(fill="x", padx=8, pady=6)

        tk.Button(self, text="Duplicate Field…",
                  command=app._duplicate_field_dialog,
                  bg=DLG_BG, fg=DLG_FG, relief="raised",
                  activebackground=DLG_ABG, activeforeground=DLG_FG,
                  padx=8).pack(fill="x", padx=8, pady=2)

        tk.Button(self, text="Close", command=self._on_close,
                  bg=DLG_BG, fg=DLG_FG, relief="raised",
                  activebackground="#550000", activeforeground=DLG_FG,
                  padx=8).pack(fill="x", padx=8, pady=(2, 8))

        self._refresh_stagger_display()
        self._refresh_gatefilter_fields()
        self._refresh_layout()

    def _refresh_stagger_display(self):
        app = self.app
        ratio     = app._stagger_ratio_str.get()
        short_nyq = app._stagger_short_nyq.get()
        long_nyq  = app._stagger_long_nyq.get()
        fixed_nyq = app._stagger_fixed_nyq.get()
        mode      = app._deglitch_nyquist_mode.get()
        source    = getattr(app, '_stagger_source', '')

        if ratio:
            ext = short_nyq * int(ratio.split('/')[0]) if short_nyq > 0 else 0
            info = (f"Stagger {ratio}  |  "
                    f"Short: {short_nyq:.2f}  Long: {long_nyq:.2f}  "
                    f"Ext: {ext:.2f} m/s")
        elif fixed_nyq > 0:
            info = f"Fixed PRF — Nyquist: {fixed_nyq:.2f} m/s"
        else:
            info = "No Nyquist detected — use Manual below."

        if source:
            info += f"\n  ({source})"

        if mode == "manual":
            man = app._deglitch_manual_nyq.get()
            info += f"\n→ Using manual: {man:.2f} m/s"
        elif mode == "short" and short_nyq > 0:
            info += f"\n→ Using short Nyquist: {short_nyq:.2f} m/s"
        elif mode == "long" and long_nyq > 0:
            info += f"\n→ Using long Nyquist: {long_nyq:.2f} m/s"
        elif fixed_nyq > 0:
            info += f"\n→ Using fixed Nyquist: {fixed_nyq:.2f} m/s"
        else:
            info += "\n→ No Nyquist available — set manually!"

        self._stagger_info_var.set(info)

    def _refresh_unfold_center_display(self, *_):
        """Show/hide manual vs automatic fold-centre sub-widgets and keep
        the computed fold-centre label up-to-date.

        Multiplier → N×Nyquist mapping (per user spec):
          x1.0  →  1 × Nyquist   (2*1.0  - 1 = 1)
          x1.5  →  2 × Nyquist   (2*1.5  - 1 = 2)
          x2.0  →  3 × Nyquist   (2*2.0  - 1 = 3)
        Formula: fold_centre = (2 * multiplier - 1) * nyquist
        Sign is flipped when "Negative fold centre" is checked.
        """
        app  = self.app
        cmode = app._unfold_center_mode.get()

        if cmode == "manual":
            self._unfold_manual_row.pack(fill="x", padx=8, pady=3)
            self._unfold_auto_frame.pack_forget()
        else:
            self._unfold_manual_row.pack_forget()
            self._unfold_auto_frame.pack(fill="x", padx=8, pady=3)
            nyq = app._brush_nyquist.get()
            if nyq <= 0:
                nyq = (app._stagger_short_nyq.get()
                       or app._stagger_fixed_nyq.get() or 0.0)
            mult     = app._unfold_fold_multiplier.get()
            negative = app._unfold_negative_center.get()
            if nyq > 0:
                n_nyq = 2.0 * mult - 1.0
                ctr   = n_nyq * nyq * (-1 if negative else 1)
                app._unfold_auto_center_label.set(
                    f"Fold centre: {ctr:+.3f} m/s  ({n_nyq:.3g}\u00d7 Nyquist)")
            else:
                app._unfold_auto_center_label.set(
                    "Fold centre: \u2014 m/s  (Nyquist not set)")

    # ── GateFilter Brush — condition rows + target checklist ──────────────

    def _gf_add_condition(self):
        """+ Add condition: append a fresh row (data lives on the app,
        not the widget, so it survives closing/reopening this window)."""
        fields = list(self.app.volume.field_names) if self.app.volume else []
        self.app._gatefilter_conditions.append({
            "field": tk.StringVar(value=fields[0] if fields else ""),
            "condition": tk.StringVar(value="Less than (<)"),
            "value": tk.StringVar(value="0.0"),
            "value2": tk.StringVar(value="0.0"),
        })
        self._rebuild_gatefilter_rows()

    def _gf_remove_condition(self, row):
        if row in self.app._gatefilter_conditions:
            self.app._gatefilter_conditions.remove(row)
        self._rebuild_gatefilter_rows()

    def _rebuild_gatefilter_rows(self):
        """Redraw the condition-row list from app._gatefilter_conditions.
        Cheap enough to just nuke and rebuild on every add/remove."""
        for w in self._gf_rows_frame.winfo_children():
            w.destroy()

        fields = list(self.app.volume.field_names) if self.app.volume else []
        for row in self.app._gatefilter_conditions:
            if fields and row["field"].get() not in fields:
                row["field"].set(fields[0])

            r = tk.Frame(self._gf_rows_frame, bg=DLG_BG)
            r.pack(fill="x", pady=2)
            ttk.Combobox(r, textvariable=row["field"], values=fields,
                        width=10, state="readonly").pack(side="left", padx=2)
            cond_cb = ttk.Combobox(r, textvariable=row["condition"],
                                   values=list(CONDITIONS.keys()), width=15,
                                   state="readonly")
            cond_cb.pack(side="left", padx=2)
            v1 = tk.Entry(r, textvariable=row["value"], width=5, bg=DLG_ENTRY)
            v1.pack(side="left", padx=2)
            v2 = tk.Entry(r, textvariable=row["value2"], width=5, bg=DLG_ENTRY)
            v2.pack(side="left", padx=2)

            def _sync(row=row, v1=v1, v2=v2):
                n = CONDITIONS.get(row["condition"].get(), 1)
                v1.config(state="normal" if n >= 1 else "disabled")
                v2.config(state="normal" if n >= 2 else "disabled")
            cond_cb.bind("<<ComboboxSelected>>", lambda e, s=_sync: s())
            _sync()

            tk.Button(r, text="✕", width=2, relief="flat",
                      bg=DLG_BG, fg=DLG_FG,
                      command=lambda row=row: self._gf_remove_condition(row)
                      ).pack(side="left", padx=(2, 0))

        if not self.app._gatefilter_conditions:
            tk.Label(self._gf_rows_frame,
                     text="No conditions yet — click \"+ Add condition\".",
                     bg=DLG_BG, fg=DLG_HINT, font=("TkDefaultFont", 8)
                     ).pack(anchor="w", pady=2)

    def _refresh_gatefilter_fields(self):
        """Re-sync the per-row field dropdowns with the current volume's
        product list. Called once at build and again on every file load
        (see GURTVispy._run_stagger_detection) so a window left open
        across loading a new file always offers the right products."""
        self._rebuild_gatefilter_rows()

    def _on_mode_change(self):
        """Radiobutton command: switching tools entirely invalidates any
        boundary polygon drawn for the previous tool, so clear it, then
        let _refresh_layout() sort out which settings frames apply."""
        self.app._clear_all_boundaries()
        self._refresh_layout()

    def _on_boundary_toggle(self):
        """Checkbox command: flipping between brush and boundary mid-task
        doesn't make sense, so clear any in-progress polygon, then let
        _refresh_layout() show/hide the Apply Edit row and brush radius.

        The custom-deglitch-algorithm option only makes sense with the
        Boundary Tool, so turning Boundary off also turns it back off
        (rather than leaving it silently armed but hidden)."""
        self.app._clear_all_boundaries()
        if not self.app._boundary_mode.get():
            self.app._custom_deglitch_enabled.set(False)
            self._custom_deglitch_settings.pack_forget()
        self._refresh_layout()

    def _on_custom_deglitch_toggle(self):
        """Checkbox command: show/hide the custom-algorithm's own settings
        (search radius, gap-fill neighbors/passes/method)."""
        if self.app._custom_deglitch_enabled.get():
            self._custom_deglitch_settings.pack(fill="x")
        else:
            self._custom_deglitch_settings.pack_forget()

    def _refresh_layout(self):
        """Single source of truth for which settings frames are visible.
        Re-run on every Mode change and every Boundary-checkbox toggle so
        a tool you're not using never lingers on screen."""
        mode     = self.app._editor_mode.get()
        boundary = self.app._boundary_mode.get()
        is_edit  = mode in EDIT_BRUSH_MODES

        # Hide everything, then re-show only what applies — each frame is
        # re-inserted directly above the tip label so the stacking order
        # stays fixed (Boundary → Brush radius → tool-specific settings)
        # no matter which subset ends up visible.
        for f in (self._boundary_frame, self._brush_frame,
                  self._unfold_frame, self._deglitch_frame, self._gapfill_frame,
                  self._gatefilter_frame):
            f.pack_forget()
        self._boundary_apply_row.pack_forget()

        fpad = dict(fill="x", padx=8, pady=4)
        if is_edit:
            self._boundary_frame.pack(before=self._tip_label, **fpad)
            if boundary:
                self._boundary_apply_row.pack(fill="x")
            else:
                # Brush radius only means anything for the dragged-circle
                # brush, not the boundary tool.
                self._brush_frame.pack(before=self._tip_label, **fpad)

            if mode == "unfold_brush":
                self._unfold_frame.pack(before=self._tip_label, **fpad)
            elif mode == "deglitch_brush":
                self._deglitch_frame.pack(before=self._tip_label, **fpad)
                self._refresh_stagger_display()
            elif mode == "gradient_brush":
                self._deglitch_frame.pack(before=self._tip_label, **fpad)
                self._gapfill_frame.pack(before=self._tip_label, **fpad)
            elif mode == "gatefilter_brush":
                self._gatefilter_frame.pack(before=self._tip_label, **fpad)
                self._refresh_gatefilter_fields()
            # delete_gate: none of unfold/deglitch/gapfill apply.

        self._update_tip(mode, boundary)
        self._update_boundary_status()

        # Hide the brush-circle / boundary previews on every panel
        # whenever we land back on plain Zoom/Pan.
        if not is_edit:
            for panel in self.app.panels:
                try:
                    panel.ppi._brush_circle.visible = False
                    panel.ppi._vcanvas.update()
                except Exception:
                    pass

    def _update_tip(self, mode, boundary):
        howto = (
            "Click to place boundary points; Enter or \"Apply Edit\" to "
            "finish.\nRight-click undoes the last point; Escape clears them."
            if boundary else
            "Left-click / drag to paint.")
        if mode == "unfold_brush":
            self._tip_var.set(
                "Unfold Brush active.\n" + howto + "\n"
                "Folds gates toward the centre velocity (set Nyquist & "
                "fold centre above).\n"
                "Press Escape or switch to Zoom to exit.")
        elif mode == "deglitch_brush":
            self._tip_var.set(
                "Deglitch Brush active.\n" + howto + "\n"
                "Gates beyond tolerance are folded toward the local median\n"
                "in steps of 2 × short Nyquist.\n"
                "Press Escape or switch to Zoom to exit.")
        elif mode == "gradient_brush":
            self._tip_var.set(
                "Gap Fill Brush active.\n" + howto + "\n"
                "Detects folded gates (same as Deglitch) then replaces them\n"
                "with the local neighbourhood median — a smooth velocity\n"
                "estimate rather than a hard integer-step correction.\n"
                "Optionally fills NaN/empty gates too (see settings above).\n"
                "Uses Tolerance from Deglitch settings.")
        elif mode == "delete_gate":
            self._tip_var.set(
                "Delete Gates active.\n" + howto + "\n"
                "NaN-masks every gate covered. Works on any field. "
                "Use Ctrl+Z to undo.\n"
                "Press Escape or switch to Zoom to exit.")
        elif mode == "gatefilter_brush":
            self._tip_var.set(
                "GateFilter Brush active.\n" + howto + "\n"
                "Masks gates matching the condition expression below, "
                "but only within the painted area/boundary — for the "
                "whole sweep, use Edit ▸ GateFilter… instead.\n"
                "Press Escape or switch to Zoom to exit.")
        else:
            self._tip_var.set("Mode: scroll to zoom, click-and-drag to pan.")

    def _update_boundary_status(self):
        """Live "N points placed" hint shown under the Apply Edit button,
        refreshed on every vertex add/remove/clear (see
        GURTVispy._on_boundary_changed)."""
        app = self.app
        if not app._boundary_mode.get() or app._editor_mode.get() not in EDIT_BRUSH_MODES:
            self._boundary_status_var.set("")
            return
        panel = app._active_panel
        n = len(panel.ppi._boundary_points) if panel is not None else 0
        if n == 0:
            self._boundary_status_var.set(
                "Click in the plot to place boundary points.")
        elif n < 3:
            self._boundary_status_var.set(f"{n} point(s) placed — need at least 3.")
        else:
            self._boundary_status_var.set(f"{n} points placed — ready to apply.")

    def _on_close(self):
        self.app._editor_mode.set("zoom")
        self.app._clear_all_boundaries()
        self.app._editor_panel_win = None
        self.destroy()


# ══════════════════════════════════════════════════════════════════════════════
# CfRadialExportDialog  (ported from gurt_guiv3)
# ══════════════════════════════════════════════════════════════════════════════

class CfRadialExportDialog(tk.Toplevel):
    """
    Modal dialog for CfRadial export: output directory + sweep/volume/all scope.
    """

    def __init__(self, app: "GURTVispy", initial_scope: str = "sweep",
                 title: str = "Export to CfRadial", file_ext: str = ".nc",
                 on_export=None):
        super().__init__(app)
        self.app = app
        self._on_export = on_export or app._do_cfradial_export
        self._file_ext  = file_ext
        self.title(title)
        self.configure(bg=DLG_BG)
        self.resizable(False, False)
        self.grab_set()
        self.transient(app)

        pad = dict(padx=10, pady=4)

        hdr = tk.Frame(self, bg="#0d0d3a")
        hdr.pack(fill="x")
        tk.Label(hdr, text=f"{title} ({file_ext})",
                 bg="#0d0d3a", fg="white",
                 font=("TkDefaultFont", 11, "bold"), pady=7).pack()

        body = tk.Frame(self, bg=DLG_BG)
        body.pack(fill="both", expand=True, padx=12, pady=8)

        tk.Label(body, text="Output directory:", bg=DLG_BG, fg=DLG_FG,
                 anchor="w", font=("TkDefaultFont", 9, "bold")
                 ).grid(row=0, column=0, sticky="w", **pad)

        dir_frame = tk.Frame(body, bg=DLG_BG)
        dir_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=2)
        body.columnconfigure(0, weight=1)

        self._dir_var = tk.StringVar(value=getattr(app, "_cfrad_export_dir", ""))
        self._dir_entry = tk.Entry(dir_frame, textvariable=self._dir_var,
                                   width=46, bg=DLG_ENTRY, relief="sunken")
        self._dir_entry.pack(side="left", fill="x", expand=True)
        tk.Button(dir_frame, text="Browse…", command=self._browse,
                  bg=DLG_BG, fg=DLG_FG, relief="raised",
                  activebackground=DLG_ABG, activeforeground=DLG_FG,
                  padx=6).pack(side="left", padx=(6, 0))

        scope_frame = tk.LabelFrame(body, text=" Export scope ",
                                    bg=DLG_BG, fg=DLG_FG,
                                    font=("TkDefaultFont", 9, "bold"),
                                    relief="groove", bd=1)
        scope_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(10, 4))

        self._scope_var = tk.StringVar(value=initial_scope)
        n_files = len(app.file_list)
        options = [
            ("sweep",  "Current sweep only"),
            ("volume", "Current volume (all sweeps in loaded file)"),
            ("all",    f"All loaded files  ({n_files} file{'s' if n_files != 1 else ''})"),
        ]
        for val, lbl in options:
            tk.Radiobutton(scope_frame, text=lbl,
                           variable=self._scope_var, value=val,
                           bg=DLG_BG, fg=DLG_FG,
                           selectcolor=DLG_ABG,
                           activebackground=DLG_ABG, activeforeground=DLG_FG,
                           indicatoron=True, font=("TkDefaultFont", 9)
                           ).pack(anchor="w", padx=8, pady=2)

        self._info_var = tk.StringVar(value="")
        tk.Label(body, textvariable=self._info_var,
                 bg=DLG_BG, fg=DLG_HINT,
                 font=("TkDefaultFont", 8), wraplength=400, justify="left"
                 ).grid(row=3, column=0, sticky="w", padx=10, pady=(2, 0))
        self._scope_var.trace_add("write", self._update_info)
        self._update_info()

        btn_row = tk.Frame(body, bg=DLG_BG)
        btn_row.grid(row=4, column=0, pady=(12, 4))
        tk.Button(btn_row, text="Export", command=self._ok,
                  bg=DLG_BG, fg=DLG_FG, relief="raised",
                  activebackground=DLG_ABG, activeforeground=DLG_FG,
                  font=("TkDefaultFont", 9, "bold"), padx=18, pady=4
                  ).pack(side="left", padx=6)
        tk.Button(btn_row, text="Cancel", command=self.destroy,
                  bg=DLG_BG, fg=DLG_FG, relief="raised",
                  activebackground=DLG_ABG, activeforeground=DLG_FG,
                  padx=14, pady=4).pack(side="left", padx=6)

        self.update_idletasks()
        pw = app.winfo_width();  ph = app.winfo_height()
        px = app.winfo_rootx(); py = app.winfo_rooty()
        dw = self.winfo_reqwidth(); dh = self.winfo_reqheight()
        self.geometry(f"+{px + (pw - dw)//2}+{py + (ph - dh)//2}")

    def _browse(self):
        d = filedialog.askdirectory(title="CfRadial output directory")
        if d:
            self._dir_var.set(d)

    def _update_info(self, *_):
        scope   = self._scope_var.get()
        out_dir = self._dir_var.get().strip() or "(not set)"
        if scope == "sweep":
            self._info_var.set(f"One .nc file will be written to:\n{out_dir}")
        elif scope == "volume":
            self._info_var.set(
                f"One .nc file (all sweeps) will be written to:\n{out_dir}")
        else:
            n = len(self.app.file_list)
            self._info_var.set(
                f"{n} .nc file(s) will be written to:\n{out_dir}\n"
                "(existing files with the same name will be overwritten)")

    def _ok(self):
        out_dir = self._dir_var.get().strip()
        if not out_dir:
            messagebox.showwarning("Export CfRadial",
                                   "Please set an output directory.", parent=self)
            return
        self.app._cfrad_export_dir = out_dir
        scope = self._scope_var.get()
        self.destroy()
        self._on_export(out_dir, scope)


class BatchBoundaryEditDialog(tk.Toplevel):
    """
    Modal dialog for running the boundary tool's current polygon + edit
    mode across every file in the currently opened folder: output
    directory + output format, then hands off to
    GURTVispy._do_batch_boundary_edit (wired in main.py, same pattern as
    CfRadial/NEXRAD export).

    Reuses whatever polygon is currently drawn on the active panel and
    whatever edit mode is selected in the Editor Panel -- both are read
    at the moment "Run" is pressed, not copied into this dialog, so
    there's nothing to keep in sync here.
    """

    def __init__(self, app: "GURTVispy"):
        super().__init__(app)
        self.app = app
        self.title("Batch Boundary Edit — Folder")
        self.configure(bg=DLG_BG)
        self.resizable(False, False)
        self.grab_set()
        self.transient(app)

        pad = dict(padx=10, pady=4)

        hdr = tk.Frame(self, bg="#0d0d3a")
        hdr.pack(fill="x")
        tk.Label(hdr, text="Batch Boundary Edit — Folder",
                 bg="#0d0d3a", fg="white",
                 font=("TkDefaultFont", 11, "bold"), pady=7).pack()

        body = tk.Frame(self, bg=DLG_BG)
        body.pack(fill="both", expand=True, padx=12, pady=8)
        body.columnconfigure(0, weight=1)

        n_files = len(app.file_list)
        mode_lbl = EDIT_MODE_LABELS.get(app._editor_mode.get(),
                                        app._editor_mode.get())
        n_pts = len(app._active_panel.ppi._boundary_points) \
            if app._active_panel is not None else 0
        tk.Label(body,
                 text=(f"Runs \"{mode_lbl}\" inside the current boundary "
                       f"({n_pts} point(s)) on every sweep of all "
                       f"{n_files} file{'s' if n_files != 1 else ''} in "
                       "the loaded folder, then saves each edited file "
                       "to the output directory below."),
                 bg=DLG_BG, fg=DLG_HINT, font=("TkDefaultFont", 8),
                 wraplength=400, justify="left"
                 ).grid(row=0, column=0, sticky="w", **pad)

        tk.Label(body, text="Output directory:", bg=DLG_BG, fg=DLG_FG,
                 anchor="w", font=("TkDefaultFont", 9, "bold")
                 ).grid(row=1, column=0, sticky="w", **pad)

        dir_frame = tk.Frame(body, bg=DLG_BG)
        dir_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=2)

        self._dir_var = tk.StringVar(value=getattr(app, "_batch_boundary_dir", ""))
        self._dir_entry = tk.Entry(dir_frame, textvariable=self._dir_var,
                                   width=46, bg=DLG_ENTRY, relief="sunken")
        self._dir_entry.pack(side="left", fill="x", expand=True)
        tk.Button(dir_frame, text="Browse…", command=self._browse,
                  bg=DLG_BG, fg=DLG_FG, relief="raised",
                  activebackground=DLG_ABG, activeforeground=DLG_FG,
                  padx=6).pack(side="left", padx=(6, 0))

        fmt_frame = tk.LabelFrame(body, text=" Output format ",
                                  bg=DLG_BG, fg=DLG_FG,
                                  font=("TkDefaultFont", 9, "bold"),
                                  relief="groove", bd=1)
        fmt_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=(10, 4))
        self._fmt_var = tk.StringVar(value="cfradial")
        for val, lbl in (("cfradial", "CfRadial (.nc)"),
                         ("nexrad",   "NEXRAD Level-II Msg31 (.ar2v)")):
            tk.Radiobutton(fmt_frame, text=lbl,
                           variable=self._fmt_var, value=val,
                           bg=DLG_BG, fg=DLG_FG, selectcolor=DLG_ABG,
                           activebackground=DLG_ABG, activeforeground=DLG_FG,
                           indicatoron=True, font=("TkDefaultFont", 9)
                           ).pack(anchor="w", padx=8, pady=2)

        # ── Optional: write edits to a new field instead of overwriting ──
        fi = app._active_panel.current_field_info() \
            if app._active_panel is not None else None
        self._src_field = fi[0] if fi else ""
        nf_frame = tk.LabelFrame(body, text=" Edited data ",
                                 bg=DLG_BG, fg=DLG_FG,
                                 font=("TkDefaultFont", 9, "bold"),
                                 relief="groove", bd=1)
        nf_frame.grid(row=4, column=0, sticky="ew", padx=10, pady=(6, 4))
        self._newfield_var = tk.BooleanVar(value=False)
        self._newfield_name = tk.StringVar(value=f"{self._src_field}_EDIT")
        self._newfield_chk = tk.Checkbutton(
            nf_frame,
            text=f"Copy edits to a new field (keep original "
                 f"{self._src_field or 'field'} unchanged)",
            variable=self._newfield_var, command=self._sync_newfield,
            bg=DLG_BG, fg=DLG_FG, selectcolor=DLG_ABG,
            activebackground=DLG_ABG, activeforeground=DLG_FG,
            font=("TkDefaultFont", 9))
        self._newfield_chk.pack(anchor="w", padx=8, pady=(2, 0))
        nrow = tk.Frame(nf_frame, bg=DLG_BG)
        nrow.pack(fill="x", padx=8, pady=(2, 4))
        tk.Label(nrow, text="New field name:", bg=DLG_BG, fg=DLG_FG,
                 font=("TkDefaultFont", 9)).pack(side="left")
        self._newfield_entry = tk.Entry(
            nrow, textvariable=self._newfield_name, width=18,
            bg=DLG_ENTRY, relief="sunken")
        self._newfield_entry.pack(side="left", padx=4)
        self._newfield_hint = tk.Label(
            nf_frame, text="", bg=DLG_BG, fg=DLG_HINT,
            font=("TkDefaultFont", 8), wraplength=400, justify="left")
        self._newfield_hint.pack(anchor="w", padx=8, pady=(0, 4))
        self._fmt_var.trace_add("write", lambda *_: self._sync_newfield())
        self._newfield_name.trace_add("write", lambda *_: self._sync_newfield())
        self._sync_newfield()

        tk.Label(body,
                 text="Existing files with the same output name will be "
                      "overwritten. This cannot be undone -- edits are "
                      "written straight to disk, not the open volume.",
                 bg=DLG_BG, fg=DLG_HINT, font=("TkDefaultFont", 8),
                 wraplength=400, justify="left"
                 ).grid(row=5, column=0, sticky="w", padx=10, pady=(4, 0))

        btn_row = tk.Frame(body, bg=DLG_BG)
        btn_row.grid(row=6, column=0, pady=(12, 4))
        tk.Button(btn_row, text="Run", command=self._ok,
                  bg=DLG_BG, fg=DLG_FG, relief="raised",
                  activebackground=DLG_ABG, activeforeground=DLG_FG,
                  font=("TkDefaultFont", 9, "bold"), padx=18, pady=4
                  ).pack(side="left", padx=6)
        tk.Button(btn_row, text="Cancel", command=self.destroy,
                  bg=DLG_BG, fg=DLG_FG, relief="raised",
                  activebackground=DLG_ABG, activeforeground=DLG_FG,
                  padx=14, pady=4).pack(side="left", padx=6)

        self.update_idletasks()
        pw = app.winfo_width();  ph = app.winfo_height()
        px = app.winfo_rootx(); py = app.winfo_rooty()
        dw = self.winfo_reqwidth(); dh = self.winfo_reqheight()
        self.geometry(f"+{px + (pw - dw)//2}+{py + (ph - dh)//2}")

    def _sync_newfield(self):
        """Enable/disable the new-field controls. NEXRAD Level-II can only
        carry its fixed moment set, so the option is CfRadial-only."""
        is_nexrad = self._fmt_var.get() == "nexrad"
        if is_nexrad:
            self._newfield_var.set(False)
        self._newfield_chk.config(state="disabled" if is_nexrad else "normal")
        on = self._newfield_var.get() and not is_nexrad
        self._newfield_entry.config(state="normal" if on else "disabled")
        if is_nexrad:
            self._newfield_hint.config(
                text="Not available for NEXRAD (fixed moment set) -- "
                     "choose CfRadial.")
        elif on:
            self._newfield_hint.config(
                text=f"The edit is applied to a copy named "
                     f"\"{self._newfield_name.get().strip()}\"; "
                     f"{self._src_field} is saved as it was loaded.")
        else:
            self._newfield_hint.config(
                text=f"Off: the edit overwrites {self._src_field} in the "
                     "saved files.")

    def _browse(self):
        d = filedialog.askdirectory(title="Batch edit output directory")
        if d:
            self._dir_var.set(d)

    def _ok(self):
        out_dir = self._dir_var.get().strip()
        if not out_dir:
            messagebox.showwarning("Batch Boundary Edit",
                                   "Please set an output directory.", parent=self)
            return
        new_field = None
        if self._newfield_var.get() and self._fmt_var.get() != "nexrad":
            new_field = self._newfield_name.get().strip().replace(" ", "_")
            if not new_field:
                messagebox.showwarning("Batch Boundary Edit",
                                       "Please enter a name for the new field.",
                                       parent=self)
                return
            if new_field == self._src_field:
                messagebox.showwarning(
                    "Batch Boundary Edit",
                    f"The new field name must differ from "
                    f"\"{self._src_field}\".", parent=self)
                return
        self.app._batch_boundary_dir = out_dir
        out_format = self._fmt_var.get()
        self.destroy()
        self.app._do_batch_boundary_edit(out_dir, out_format,
                                         new_field=new_field)


# ══════════════════════════════════════════════════════════════════════════════
# PanelState — one PPI panel + its own field/vmin/vmax/cmap selection
# ══════════════════════════════════════════════════════════════════════════════

class PanelState:
    """
    One visible PPI panel in a (possibly multi-panel) grid.

    All panels in a GURTVispy window share the same volume, sweep index,
    camera (zoom/pan), and range-ring settings — those live on GURTVispy
    itself. A PanelState only owns what's allowed to differ panel-to-panel:
    the displayed field, its vmin/vmax/colormap, and a small header strip
    of controls for picking them.
    """

    def __init__(self, app: "GURTVispy", container: tk.Frame, index: int):
        self.app   = app
        self.index = index

        self.frame = tk.Frame(container, bg=PLOT_BG,
                              highlightthickness=1,
                              highlightbackground="#444")

        # ── Field/scaling state (no longer shown as an on-panel toolbar —
        # use right-click → "Open Param & Colors…" to change field/min/max).
        # field_combo is kept as an off-screen widget purely so the rest of
        # the app can keep using its .get()/.set()/["values"] API.
        self._hidden_state = tk.Frame(self.frame)  # never packed/shown

        self.field_combo = ttk.Combobox(self._hidden_state, width=12,
                                        state="readonly", font=("Courier", 8))
        self.vmin_var = tk.StringVar(value="")
        self.vmax_var = tk.StringVar(value="")

        # The colorbar is drawn as a transparent overlay inside the PPI canvas
        # (see PPICanvas.set_colorbar), so the radar view runs behind it.
        self._cb_args = None
        self._cb_resize_id = None
        self._cb_image = None
        self._cb_key   = None

        # ── PPI canvas ───────────────────────────────────────────────────────
        self.canvas_frame = tk.Frame(self.frame, bg=PLOT_BG)
        self.canvas_frame.pack(fill="both", expand=True)

        self.cmap_override = None
        self.ppi = PPICanvas(self.canvas_frame, app, panel=self)
        self.canvas_frame.bind("<Configure>", self._on_cb_configure)
        self._brush_undo_pushed    = False
        self._boundary_undo_pushed = False

        # ── Header — floats on top of the canvas ────────────────────────────
        # Placed (not packed) directly over the canvas so the label's own
        # background box only ever wraps its text, instead of stretching
        # a full-width bar across the top of the panel and eating into the
        # plot's vertical space. Created after the canvas so it stacks
        # above it; .lift() makes that explicit.
        self.header_var = tk.StringVar(value="")
        self._header_img = None
        self.header_label = tk.Label(
            self.canvas_frame, bg="black", bd=0, padx=0, pady=0)
        self.header_var.trace_add("write", lambda *_: self._render_header())
        self.header_label.place(relx=0.5, y=2, anchor="n")
        self.header_label.lift()

    def _render_header(self):
        """Heading: matplotlib DejaVu Sans 12 pt regular, white on a tight
        black box (rendered as an image so the font is identical everywhere)."""
        text = self.header_var.get()
        if not text or not PIL_AVAILABLE:
            self.header_label.place_forget()
            return
        try:
            img = _render_header_image(text)
            self._header_img = ImageTk.PhotoImage(img)
            self.header_label.config(image=self._header_img)
            self.header_label.place(relx=0.5, y=2, anchor="n")
            self.header_label.lift()
        except Exception:
            pass

    def _on_cb_configure(self, event):
        """Panel width changed (window resize / fullscreen): re-render the bar."""
        if self._cb_args is None:
            return
        if self._cb_resize_id is not None:
            self.frame.after_cancel(self._cb_resize_id)
        self._cb_resize_id = self.frame.after(
            60, lambda: self.app._update_panel_colorbar(self, *self._cb_args))

    # ── Field info (mirrors GURTVispy._current_field_info but per-panel) ──────

    def current_field_info(self):
        fname = self.field_combo.get()
        if not fname:
            return None
        dfl       = FIELD_DEFAULTS.get(fname, (fname, -50, 50, "gurt_ref"))
        label     = dfl[0]
        cmap_name = self.cmap_override or dfl[3]
        try:
            vmin = float(self.vmin_var.get())
        except ValueError:
            vmin = dfl[1]; self.vmin_var.set(str(vmin))
        try:
            vmax = float(self.vmax_var.get())
        except ValueError:
            vmax = dfl[2]; self.vmax_var.set(str(vmax))
        return fname, label, vmin, vmax, cmap_name

    def set_field(self, fname: str):
        """Select *fname* in this panel and reset vmin/vmax/cmap to its defaults."""
        self.field_combo.set(fname)
        dfl = FIELD_DEFAULTS.get(fname, (fname, -50, 50, "gurt_ref"))
        self.vmin_var.set(str(dfl[1]))
        self.vmax_var.set(str(dfl[2]))
        self.cmap_override = None

    def destroy(self):
        try:
            self.ppi._vcanvas.close()
        except Exception:
            pass
        self.frame.destroy()


# ══════════════════════════════════════════════════════════════════════════════
# GURTVispy — main application window
# ══════════════════════════════════════════════════════════════════════════════

class GURTVispy(tk.Tk):
    """
    Main application window.

    Accepts keyword args:
        folder  — open a folder of radar files on startup
        file    — open a single radar file on startup
        volume_cls  — class implementing RadarVolume (injected by main.py)
        load_fn     — callable(path) -> RadarVolume
    """

    def __init__(self, folder=None, file=None, volume_cls=None, load_fn=None):
        super().__init__()
        self.title("GURT Vispy — Radar Viewer")
        self.configure(bg=BG)
        self.geometry("1280x860")
        self.minsize(800, 600)

        # Injected from main.py (or left None for standalone use)
        self._load_fn = load_fn

        # ── Core state ────────────────────────────────────────────────────────
        self.volume      = None
        self.file_list   = []
        self.file_index  = 0
        self.sweep_index = tk.IntVar(value=0)
        # Re-detect Nyquist/stagger metadata for whichever sweep is now
        # active. This fires on every change to sweep_index, regardless of
        # source (Up/Down keys, spinbox arrows, typed value + Tab, or the
        # programmatic .set() done after loading a file).
        self.sweep_index.trace_add(
            "write", lambda *_: self._run_stagger_detection())

        self.max_range     = tk.DoubleVar(value=150.0)
        self.ring_interval = tk.DoubleVar(value=50.0)
        self.az_interval   = tk.DoubleVar(value=30.0)
        self.show_rings    = tk.BooleanVar(value=True)
        self.show_azimuths = tk.BooleanVar(value=True)
        self.show_ticks    = tk.BooleanVar(value=True)
        self.tick_interval = tk.DoubleVar(value=0.0)   # km; 0 = automatic (follows zoom)

        self._cmap_override = None
        self._refresh_id    = None
        self._load_id       = None
        self._cfrad_export_dir = ""
        self._render_cache = {}
        self._render_cache_order = []
        self._render_cache_limit = 24
        self._geometry_cache = {}
        self._prewarm_id = None

        # ── Multi-panel layout state ──────────────────────────────────────────
        # `self.panels` holds one PanelState per visible PPI panel. All panels
        # share the same volume / sweep / camera (zoom+pan) / range-ring
        # settings; only the displayed field (and that field's vmin/vmax/cmap
        # and brush edits) differ per panel.
        self.panels        = []     # list[PanelState], left-to-right, top-to-bottom
        self._layout_rows  = 1
        self._layout_cols  = 1
        self._active_panel = None   # last panel the user interacted with
        self._detect = (DetectTool(self) if DetectTool is not None
                        else _PlaceholderTool(self, "Vortex tracking"))   # DREADER detection / tracking
        self._xsect = XsectTool(self)     # DREADER radial / angular cross-sections
        self._axisym = (AxisymTool(self) if AxisymTool is not None
                        else _PlaceholderTool(self, "Axisymmetric analysis"))   # TURT/DREADER patch overlay

        # ── Editor / brush state (used by EditorPanel) ────────────────────────
        self._editor_mode      = tk.StringVar(value="zoom")
        self._boundary_mode     = tk.BooleanVar(value=False)
        # When True, Apply Edit (boundary tool) runs on every sweep of the
        # loaded volume instead of just the active one -- see
        # brushes.apply_boundary_all_sweeps / GURTVispy._apply_boundary_edit.
        self._boundary_all_sweeps = tk.BooleanVar(value=False)
        # Boundary-only option: when True, Deglitch Brush / Gap Fill Brush
        # modes run the standalone deglitch.py algorithm (whole-sweep,
        # numba-accelerated, iterates to a stable solution) instead of the
        # quick local-median math, restricted to the polygon's gates. Only
        # meaningful -- and only ever True -- while _boundary_mode is on;
        # see brushes._custom_algo_active. Never deletes unresolved gates.
        self._custom_deglitch_enabled   = tk.BooleanVar(value=False)
        self._custom_deglitch_radius    = tk.IntVar(value=2)
        # Defaults to 15.0 to match run_deglitch.py's tuned value -- this
        # is deliberately its own setting, *not* _deglitch_tolerance (the
        # quick brush algorithm's tolerance defaults to 3.0 and means
        # something different: a fold-count cutoff, not a glitch-vs-
        # local-median threshold).
        self._custom_deglitch_thresh    = tk.DoubleVar(value=15.0)
        # Off by default, matching run_deglitch.py's gap_fill_enabled=False:
        # unresolved glitches are flagged/masked but left alone rather than
        # patched with a neighbor estimate, unless the user opts in.
        self._custom_gapfill_enabled    = tk.BooleanVar(value=False)
        self._custom_gapfill_n_neighbors = tk.IntVar(value=6)
        self._custom_gapfill_method     = tk.StringVar(value="median")
        self._custom_gapfill_n_passes   = tk.IntVar(value=10)
        self._brush_radius_km  = tk.DoubleVar(value=2.0)
        self._brush_nyquist    = tk.DoubleVar(value=0.0)
        # When True, _brush_nyquist is kept in sync with the active sweep's
        # metadata-detected Nyquist every time the sweep or file changes.
        # Typing a value into the Nyquist entry switches this off (handled
        # by the trace below) so manual overrides aren't clobbered; users
        # can re-check "Auto" in the Editor Panel to resume tracking.
        self._brush_nyquist_auto = tk.BooleanVar(value=True)
        self._setting_brush_nyquist_programmatically = False

        def _on_brush_nyquist_written(*_):
            if not self._setting_brush_nyquist_programmatically:
                self._brush_nyquist_auto.set(False)
        self._brush_nyquist.trace_add("write", _on_brush_nyquist_written)

        self._brush_center     = tk.DoubleVar(value=0.0)
        self._unfold_center_mode     = tk.StringVar(value="manual")
        self._unfold_fold_multiplier = tk.DoubleVar(value=1.0)
        self._unfold_negative_center = tk.BooleanVar(value=False)
        self._unfold_auto_center_label = tk.StringVar(value="Fold centre: \u2014 m/s")
        self._editor_panel_win = None

        # ── Stagger / Nyquist state ───────────────────────────────────────────
        self._stagger_short_nyq     = tk.DoubleVar(value=0.0)
        self._stagger_long_nyq      = tk.DoubleVar(value=0.0)
        self._stagger_fixed_nyq     = tk.DoubleVar(value=0.0)
        self._stagger_ratio_str     = tk.StringVar(value="")
        self._deglitch_nyquist_mode = tk.StringVar(value="short")
        self._deglitch_manual_nyq   = tk.DoubleVar(value=0.0)
        self._deglitch_tolerance    = tk.DoubleVar(value=3.0)

        # ── Gap Fill brush state ──────────────────────────────────────────────
        self._gapfill_fill_nan      = tk.BooleanVar(value=False)

        # ── GateFilter Brush state ────────────────────────────────────────────
        # _gatefilter_conditions: list of dicts of Tk StringVars (field,
        # condition, value, value2) — one per condition row in the Editor
        # Panel. Lives on the app (not the EditorPanel widget) so it
        # survives closing/reopening that window. Unlike the whole-sweep
        # GateFilter dialog, the brush always masks whichever field the
        # panel is currently displaying — no separate target picker.
        self._gatefilter_conditions  = []
        self._gatefilter_combine     = tk.StringVar(value="or")

        # ── Undo / redo stacks ───────────────────────────────────────────────
        self._undo_stack     = []
        self._redo_stack     = []
        self._undo_stack_max = 50

        # Debounced repaint for the undo/redo "patch changed gates" fast
        # path. Holding Ctrl+Z auto-repeats the key many times a second;
        # without this, each step would force its own immediate VisPy
        # redraw and you'd see every intermediate undo state flash by
        # instead of jumping straight to the final one.
        self._patch_flush_id      = None
        self._pending_patch_panels = set()

        self._build_menu()
        self._build_layout()
        self._bind_keys()

        if folder:
            self.after(200, lambda: self.load_folder(folder))
        elif file:
            self.file_list  = [os.path.abspath(file)]
            self.file_index = 0
            self.after(200, self.load_current_file)

    # ── Menu helpers ──────────────────────────────────────────────────────────

    def _mk_menu(self, parent):
        return tk.Menu(parent, tearoff=0,
                       bg=BAR_BG, fg="#000",
                       activebackground="#000080", activeforeground="#fff")

    def _build_menu(self):
        mb = tk.Menu(self, tearoff=0,
                     bg=BAR_BG, fg="#000",
                     activebackground="#000080", activeforeground="#fff")
        self.config(menu=mb)

        fm = self._mk_menu(mb)
        fm.add_command(label="Open File…",   command=self.open_file,
                       accelerator="Ctrl+O")
        fm.add_command(label="Open Folder…", command=self.open_folder,
                       accelerator="Ctrl+D")
        fm.add_command(label="Open I/Q File (Beta)…", command=self.open_iq_beta_file)
        fm.add_separator()
        fm.add_command(label="Save Image…",  command=self.save_image,
                       accelerator="Ctrl+S")
        fm.add_separator()
        fm.add_command(label="Quit",         command=self.quit)
        mb.add_cascade(label="File", menu=fm)

        vm = self._mk_menu(mb)
        vm.add_checkbutton(label="Range rings",   variable=self.show_rings,
                           command=self._schedule_refresh)
        vm.add_checkbutton(label="Azimuth lines", variable=self.show_azimuths,
                           command=self._schedule_refresh)
        vm.add_checkbutton(label="Tick marks (+)", variable=self.show_ticks,
                           command=self._refresh_ticks)
        vm.add_command(label="Set tick interval… (0 = auto)", command=self._ask_tick_interval)
        vm.add_separator()
        vm.add_command(label="Set max range…",     command=self._ask_max_range)
        vm.add_command(label="Set ring interval…", command=self._ask_ring_interval)
        vm.add_command(label="Reset zoom",         command=self._reset_zoom)
        mb.add_cascade(label="View", menu=vm)

        em = self._mk_menu(mb)
        em.add_command(label="Open Editor Panel…", command=self._open_editor_panel)
        em.add_command(label="Duplicate Field…",   command=self._duplicate_field_dialog)
        em.add_separator()
        em.add_command(label="Rotate Sweep…",            command=self._rotate_sweep_dialog)
        em.add_command(label="Offset and Scale Product…", command=self._offset_scale_dialog)
        em.add_command(label="Offset Elevations…",        command=self._offset_elevations_dialog)
        em.add_command(label="Set Radar Location…",       command=self._set_location_dialog)
        em.add_separator()
        em.add_command(label="Despeckle (DREADER)…", command=self._despeckle_dialog)
        em.add_command(label="GateFilter…",         command=self._open_gatefilter_dialog)
        em.add_separator()
        em.add_command(label="Export CfRadial…",   command=lambda: self._open_cfrad_export())
        em.add_command(label="Export NEXRAD (Msg31)…", command=lambda: self._open_nexrad_export())
        em.add_separator()
        em.add_command(label="Batch Boundary Edit (Folder)…",
                       command=self._open_batch_boundary_dialog)
        mb.add_cascade(label="Edit", menu=em)

        am = self._mk_menu(mb)
        am.add_command(label="Smoothed velocity magnitude field…", command=self._smooth_velmag_dialog)
        am.add_separator()
        am.add_command(label="Axisymmetric patch: pick center on PPI…", command=self._axisym.begin_pick)
        am.add_command(label="Axisymmetric patch settings…",            command=self._axisym.open_settings)
        am.add_command(label="Clear axisymmetric overlay",              command=self._axisym.clear)
        am.add_separator()
        am.add_command(label="Cross-section: radial (click ray)…",
                       command=lambda: self._xsect.begin_pick("radial"))
        am.add_command(label="Cross-section: angular (click arc center)…",
                       command=lambda: self._xsect.begin_pick("angular"))
        am.add_command(label="Cross-section settings…",                 command=self._xsect.open_settings)
        am.add_command(label="Clear cross-section overlay",             command=self._xsect.clear)
        am.add_separator()
        am.add_command(label="Vortex tracking: draw search box…",       command=self._detect.begin_box)
        am.add_command(label="Re-run box tracking",                     command=self._detect.rerun_box)
        am.add_command(label="Vortex detection: this sweep (all couplets)", command=self._detect.run_sweep)
        am.add_command(label="Vortex detection + tracking: all vortices, folder…",
                       command=self._detect.run_folder)
        am.add_command(label="Vortex detection / tracking settings…",   command=self._detect.open_settings)
        am.add_command(label="Clear vortex detections and box",         command=self._detect.clear)
        mb.add_cascade(label="Analysis", menu=am)

        self._cm_menu = self._mk_menu(mb)
        self._rebuild_cmap_menu()
        mb.add_cascade(label="Colormap", menu=self._cm_menu)

        lm = self._mk_menu(mb)
        lm.add_command(label="1 panel",            command=lambda: self.set_layout_preset("1"))
        lm.add_command(label="2 panels (side by side)", command=lambda: self.set_layout_preset("2"))
        lm.add_command(label="4 panels (grid)",     command=lambda: self.set_layout_preset("4"))
        lm.add_separator()
        rows_sub = self._mk_menu(lm)
        for n in (1, 2, 3, 4):
            rows_sub.add_command(label=str(n), command=lambda n=n: self.set_layout_rows(n))
        lm.add_cascade(label="Rows…",    menu=rows_sub)
        cols_sub = self._mk_menu(lm)
        for n in (1, 2, 3, 4):
            cols_sub.add_command(label=str(n), command=lambda n=n: self.set_layout_cols(n))
        lm.add_cascade(label="Columns…", menu=cols_sub)
        mb.add_cascade(label="Layout", menu=lm)

    def _rebuild_cmap_menu(self):
        m = self._cm_menu
        m.delete(0, "end")
        for group, names in [
            ("GURT",       list(GURT_CMAPS.keys())),
            ("Diverging",  ["RdBu_r", "RdYlBu_r", "coolwarm", "bwr", "Spectral_r"]),
            ("Sequential", ["viridis", "plasma", "inferno", "turbo", "jet", "Greys"]),
        ]:
            m.add_command(label=f"── {group} ──", state="disabled")
            for cn in names:
                m.add_command(label=cn, command=lambda c=cn: self._set_cmap(c))

    def _set_cmap(self, name):
        panel = self._active_panel or (self.panels[0] if self.panels else None)
        if panel is None:
            return
        panel.cmap_override = name
        self._schedule_refresh()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_layout(self):
        # Status bar
        sb = tk.Frame(self, bg=BAR_BG, height=22)
        sb.pack(fill="x", side="bottom")
        sb.pack_propagate(False)

        self.cursor_var = tk.StringVar(value="Cursor: —")
        tk.Label(sb, textvariable=self.cursor_var,
                 bg=BAR_BG, fg="#000", font=("Courier", 9),
                 anchor="w", padx=6).pack(side="left")

        self.file_var = tk.StringVar(value="")
        tk.Label(sb, textvariable=self.file_var,
                 bg=BAR_BG, fg="#000", font=("Courier", 9),
                 anchor="e", padx=6).pack(side="right")

        sf = tk.Frame(sb, bg=BAR_BG)
        sf.pack(side="right", padx=8)
        tk.Label(sf, text="Sweep:", bg=BAR_BG, fg="#000",
                 font=("Courier", 9)).pack(side="left")
        self.sweep_spin = tk.Spinbox(
            sf, from_=0, to=20,
            textvariable=self.sweep_index, width=3,
            command=self._schedule_refresh,
            bg="#fff", fg="#000", buttonbackground="#bbb")
        self.sweep_spin.pack(side="left", padx=2)

        # Main area: a grid container that _build_panel_grid() populates
        # with 1, 2, or 4 (or up to a 4x4) PanelState instances.
        self.plot_frame = tk.Frame(self, bg=PLOT_BG)
        self.plot_frame.pack(fill="both", expand=True)

        self.vispy_frame = tk.Frame(self.plot_frame, bg=PLOT_BG)
        self.vispy_frame.pack(fill="both", expand=True)

        self._build_panel_grid(rows=1, cols=1)

    # ── Panel grid / layout management ──────────────────────────────────────

    def _build_panel_grid(self, rows: int, cols: int):
        """
        (Re)build the panel grid as `rows` x `cols`, keeping as many existing
        panels (and their field selections) alive as possible. Extra slots
        beyond the existing panel count are filled with new panels that try
        to pick a field not already shown elsewhere.
        """
        rows = max(1, min(4, int(rows)))
        cols = max(1, min(4, int(cols)))
        count = rows * cols

        # Remember field choices from panels that will be discarded, in case
        # we need them again (e.g. user goes 4 -> 1 -> 4).
        old_fields = [p.field_combo.get() for p in self.panels]

        # Capture the current shared view (zoom/pan) from any existing
        # panel before we add new ones, so new panels can be started at
        # the same view instead of VisPy's raw default rect (0,0,1,1) —
        # which nothing else resets until the user explicitly resets zoom
        # or loads a new file.
        shared_rect = self.panels[0].ppi._view.camera.rect if self.panels else None

        # Resize self.panels to `count`, destroying surplus / creating new.
        while len(self.panels) > count:
            self.panels.pop().destroy()
        while len(self.panels) < count:
            idx = len(self.panels)
            new_panel = PanelState(self, self.vispy_frame, idx)
            if shared_rect is not None:
                new_panel.ppi._view.camera.rect = shared_rect
            self.panels.append(new_panel)

        # Re-grid every panel frame (cheap; avoids stale grid_remove state).
        #
        # IMPORTANT: grid_rowconfigure/grid_columnconfigure settings persist
        # on vispy_frame forever once set — they are NOT cleared just
        # because a row/column stops being used. Layouts cap at 4x4, so
        # always touch every row/col in that range: weight=1 for the ones
        # this layout uses, weight=0 for the rest. Without the weight=0
        # reset, going e.g. 2x2 -> 1x1 leaves the old row 1 / column 1
        # still claiming equal space, so the single panel only fills a
        # quarter of the frame instead of all of it.
        MAX_GRID_DIM = 4
        for r in range(MAX_GRID_DIM):
            if r < rows:
                self.vispy_frame.grid_rowconfigure(
                    r, weight=1, uniform="ppi_row")
            else:
                # Unused rows: weight 0 AND removed from the uniform group,
                # so they don't reserve a same-size slot alongside the rows
                # that are actually in use (this was eating ~50% of the
                # window in fullscreen with fewer than 4 rows/cols).
                self.vispy_frame.grid_rowconfigure(
                    r, weight=0, uniform="")
        for c in range(MAX_GRID_DIM):
            if c < cols:
                self.vispy_frame.grid_columnconfigure(
                    c, weight=1, uniform="ppi_col")
            else:
                self.vispy_frame.grid_columnconfigure(
                    c, weight=0, uniform="")
        for i, panel in enumerate(self.panels):
            r, c = divmod(i, cols)
            panel.frame.grid(row=r, column=c, sticky="nsew", padx=1, pady=1)

        self._layout_rows = rows
        self._layout_cols = cols
        self._active_panel = self.panels[0]

        # Populate field choices: keep what was already selected where
        # possible, otherwise pick the next not-yet-shown field so a 4-panel
        # layout defaults to 4 different fields rather than 4 copies of one.
        fields = list(self.volume.field_names) if self.volume else []
        used = set()
        for i, panel in enumerate(self.panels):
            panel.field_combo["values"] = fields
            candidate = old_fields[i] if i < len(old_fields) else ""
            if candidate not in fields or candidate in used:
                candidate = next((f for f in fields if f not in used),
                                 fields[0] if fields else "")
            if candidate:
                if candidate != panel.field_combo.get():
                    # New panel, or its old field is no longer valid/available
                    # — apply the field and reset vmin/vmax/cmap to its
                    # defaults, same as set_field() does elsewhere.
                    panel.set_field(candidate)
                else:
                    panel.field_combo.set(candidate)
                used.add(candidate)
            else:
                panel.field_combo.set("")
        # Rebuilding the grid (destroying/creating/re-gridding panel frames
        # above) only *schedules* new widget sizes with Tk — actual
        # geometry isn't recomputed until Tk's next idle pass. Forcing it
        # now means every panel's cb_label already has its correct new
        # width by the time the refresh below renders each colorbar,
        # instead of rendering against the previous layout's stale width
        # (which needed a second, unrelated refresh — e.g. changing a
        # field — to ever catch up).
        self.update_idletasks()

        self._schedule_refresh()

    def set_layout_preset(self, name: str):
        """name in {'1', '2', '4'} — the quick presets from the Layout menu."""
        presets = {"1": (1, 1), "2": (1, 2), "4": (2, 2)}
        rows, cols = presets[name]
        self._build_panel_grid(rows, cols)

    def set_layout_rows(self, n: int):
        """Stack `n` panels vertically (1 column, n rows)."""
        self._build_panel_grid(rows=n, cols=1)

    def set_layout_cols(self, n: int):
        """Lay `n` panels out horizontally (1 row, n columns)."""
        self._build_panel_grid(rows=1, cols=n)

    def _sync_zoom_from(self, src_canvas: "PPICanvas",
                        factor: float, frac_x: float, frac_y: float):
        """Replay a zoom gesture (factor, anchored at fraction frac_x/frac_y
        of the source panel's own canvas) on every other panel, against
        each panel's own current rect. See PPICanvas.zoom_by/sync_zoom."""
        for panel in self.panels:
            if panel.ppi is src_canvas:
                continue
            panel.ppi.sync_zoom(factor, frac_x, frac_y)

    def _sync_pan_from(self, src_canvas: "PPICanvas",
                       dfrac_x: float, dfrac_y: float):
        """Replay a pan gesture (shift by a fraction of the source panel's
        own current rect width/height) on every other panel, against each
        panel's own current rect. See PPICanvas.pan_by/sync_pan."""
        for panel in self.panels:
            if panel.ppi is src_canvas:
                continue
            panel.ppi.sync_pan(dfrac_x, dfrac_y)

    def _bind_keys(self):
        self.bind("<Left>",      lambda _: self._change_file(-1))
        self.bind("<Right>",     lambda _: self._change_file(+1))
        self.bind("<Up>",        lambda _: self._change_sweep(+1))
        self.bind("<Down>",      lambda _: self._change_sweep(-1))
        self.bind("<Escape>",    lambda _: self._on_escape_key())
        self.bind("<Return>",    lambda _: self._apply_boundary_edit())
        self.bind("<Control-o>", lambda _: self.open_file())
        self.bind("<Control-d>", lambda _: self.open_folder())
        self.bind("<Control-s>", lambda _: self.save_image())
        self.bind("<Control-z>", lambda _: self._undo_brush())
        self.bind("<Control-y>", lambda _: self._redo_brush())
        self.after(100, self.focus_set)

    # ── Navigation ────────────────────────────────────────────────────────────

    def _change_file(self, delta):
        if not self.file_list:
            return
        self.file_index = (self.file_index + delta) % len(self.file_list)
        self._schedule_load()

    def _change_sweep(self, delta):
        if self.volume is None:
            return
        n   = len(self.volume.sweeps)
        new = max(0, min(n - 1, self.sweep_index.get() + delta))
        self.sweep_index.set(new)
        self._schedule_refresh()

    def _schedule_load(self):
        if self._load_id:
            self.after_cancel(self._load_id)
        self._load_id = self.after(80, self.load_current_file)

    def _schedule_refresh(self):
        if self._refresh_id:
            self.after_cancel(self._refresh_id)
        self._refresh_id = self.after_idle(self.refresh_plot)

    # ── File I/O ──────────────────────────────────────────────────────────────

    def open_file(self):
        path = filedialog.askopenfilename(
            title="Open Radar File",
            filetypes=[
                ("Radar files", "*.nc *.h5 *.buf *.raw *.gz *.HDF5 *.hdf5 swp.*"),
                ("DORADE",      "swp.*"),
                ("All files",   "*.*"),
            ])
        if path:
            self.file_list  = [os.path.abspath(path)]
            self.file_index = 0
            self.file_var.set(f"[1/1]  {os.path.basename(path)}")
            self.load_current_file()

    def open_folder(self):
        d = filedialog.askdirectory(title="Open Radar Folder")
        if d:
            self.load_folder(d)

    def open_iq_beta_file(self):
        """File ▸ Open I/Q File (Beta)… — raw Vaisala rvp8/rvpts I/Q dumps,
        moments computed on the fly via calcIQ.IQMoments. Everything else
        (dialog, product list, loader) lives in the IQ_beta/ package so it
        doesn't bloat this file; imported lazily here so a missing/broken
        IQ_beta package can't break normal file loading for anyone not
        using the feature."""
        try:
            from IQ_beta import IQBetaDialog
        except Exception as e:
            messagebox.showerror(
                "Open I/Q File (Beta)",
                f"Could not load the IQ_beta package:\n{e}")
            return
        IQBetaDialog(self)

    def load_folder(self, folder):
        files = []
        for root, _, fnames in os.walk(folder):
            for f in sorted(fnames):
                is_ext    = any(f.lower().endswith(e) or f.endswith(e)
                                for e in FILE_EXTENSIONS)
                is_dorade = any(f.lower().startswith(p)
                                for p in _DORADE_NAME_PREFIXES)
                if is_ext or is_dorade:
                    files.append(os.path.join(root, f))
        self.file_list  = sorted(files)
        self.file_index = 0
        if self.file_list:
            self.load_current_file()
        else:
            messagebox.showwarning("No files",
                                   f"No radar files found in:\n{folder}")

    def load_current_file(self):
        if not self.file_list:
            return
        path = self.file_list[self.file_index]
        is_first_load = self.volume is None
        slot = self.file_index

        self.file_var.set(
            f"[{slot+1}/{len(self.file_list)}]  "
            f"{os.path.basename(path)}  [loading…]")
        self.update_idletasks()

        if self._load_fn is None:
            messagebox.showerror("Configuration error",
                                 "No load_fn provided to GURTVispy.")
            return

        # Track a load generation so stale background loads don't clobber a
        # newer one if the user flips files quickly.
        if not hasattr(self, "_load_generation"):
            self._load_generation = 0
        self._load_generation += 1
        generation = self._load_generation

        def _background():
            # ── Step 1: file I/O (can be slow for large files) ──────────────
            try:
                volume = self._load_fn(path)
            except Exception as e:
                _msg = str(e)
                self.after(0, lambda m=_msg: (
                    messagebox.showerror(
                        "Load Error",
                        f"Could not read {os.path.basename(path)}\n\n{m}"),
                    self.file_var.set(f"Error: {os.path.basename(path)}")
                ))
                return

            # ── Step 2: CPU-heavy render prep for the first sweep ────────────
            # Build geometry + face colours off the main thread so the first
            # show_cached_render call is near-instant on the main thread.
            si = max(0, min(len(volume.sweeps) - 1,
                            self.sweep_index.get()))
            sweep = volume.sweeps[si]
            is_rhi = getattr(sweep, "scan_type", "ppi") == "rhi"
            mode = "rhi" if is_rhi else "ppi"

            # Determine which fields the panels will want (current selections
            # if they exist in the new volume, else priority defaults).
            fields = list(getattr(volume, "field_names", []))
            priority = ["DBZH", "DBZH1", "DBZ", "DBZVC_F", "DBZVC",
                        "reflectivity", "REF", "VEL", "VELH", "velocity"]
            used = set()
            wanted_fields = []
            for panel in self.panels:
                cur = panel.field_combo.get()
                if cur in fields and cur not in used:
                    wanted_fields.append(cur)
                    used.add(cur)
                else:
                    chosen = next(
                        (f for f in priority if f in fields and f not in used),
                        next((f for f in fields if f not in used),
                             fields[0] if fields else None))
                    if chosen and chosen not in used:
                        wanted_fields.append(chosen)
                        used.add(chosen)

            # Pre-build geometry (shared across fields for this sweep).
            try:
                if is_rhi:
                    verts, faces = _build_rhi_geometry(sweep.elevation, sweep.ranges)
                else:
                    verts, faces = _build_ppi_geometry(sweep.azimuth, sweep.ranges)
                nrays, ngates = len(sweep.azimuth), len(sweep.ranges)
            except Exception:
                verts = faces = None
                nrays = ngates = 0

            # Pre-build face colours for each wanted field.
            prebuilt = {}  # field_name -> face_colors array
            for fname in wanted_fields:
                if fname not in sweep.fields:
                    continue
                try:
                    fi = next((p.current_field_info() for p in self.panels
                                if p.field_combo.get() == fname), None)
                    if fi is None:
                        dfl = FIELD_DEFAULTS.get(fname, (fname, -50, 50, "gurt_ref"))
                        vmin_f, vmax_f, cmap_name = dfl[1], dfl[2], dfl[3]
                    else:
                        _, _, vmin_f, vmax_f, cmap_name = fi
                    cmap = resolve_cmap(cmap_name)
                    data = np.asarray(sweep.fields[fname], dtype="float32")
                    r, g = data.shape
                    if r != nrays or g != ngates:
                        # shape-normalise
                        if r < nrays:
                            data = np.vstack([data, np.full((nrays-r, g), np.nan, "float32")])
                        elif r > nrays:
                            data = data[:nrays]
                        if data.shape[1] < ngates:
                            data = np.hstack([data, np.full((data.shape[0], ngates-data.shape[1]), np.nan, "float32")])
                        else:
                            data = data[:, :ngates]
                    prebuilt[fname] = (
                        _data_to_face_colors(data, cmap, vmin_f, vmax_f),
                        verts, faces,
                        mode,
                        vmin_f, vmax_f, cmap_name,
                    )
                except Exception:
                    pass

            # ── Step 3: hand results back to the main thread ─────────────────
            def _on_main():
                if self._load_generation != generation:
                    return  # a newer load superseded this one

                self.volume = volume
                self._clear_render_cache()

                # Seed the render cache with the pre-built renders so
                # refresh_plot doesn't have to redo the CPU work.
                if verts is not None:
                    geom_key = ("rhi" if is_rhi else "ppi",) + PPICanvas._geometry_key(
                        None, mode, sweep.elevation if is_rhi else sweep.azimuth, sweep.ranges)
                    self._geometry_cache[geom_key] = (verts, faces)

                    for fname, packed in prebuilt.items():
                        fc, _v, _f, _mode, vmin_f, vmax_f, cmap_name = packed
                        key = self._render_cache_key(si, fname, vmin_f, vmax_f, cmap_name)
                        self._render_cache[key] = {
                            "mode": mode,
                            "sweep": sweep,
                            "geom_key": geom_key,
                            "verts": verts,
                            "faces": faces,
                            "face_colors": fc,
                        }
                        self._touch_render_cache_key(key)

                self.file_var.set(
                    f"[{slot+1}/{len(self.file_list)}]  "
                    f"{os.path.basename(path)}")
                self.title(f"GURT Vispy — {os.path.basename(path)}")

                n_sweeps = len(self.volume.sweeps)
                self.sweep_index.set(min(self.sweep_index.get(),
                                         max(0, n_sweeps - 1)))
                self.sweep_spin.config(to=max(0, n_sweeps - 1))

                fields_now = self.volume.field_names
                used2 = set()
                for panel in self.panels:
                    panel.field_combo["values"] = fields_now
                    if not fields_now:
                        continue
                    cur = panel.field_combo.get()
                    if cur in fields_now:
                        used2.add(cur)
                        continue
                    chosen = next(
                        (f for f in priority if f in fields_now and f not in used2),
                        next((f for f in fields_now if f not in used2),
                             fields_now[0]))
                    panel.set_field(chosen)
                    used2.add(chosen)

                self._run_stagger_detection()
                if is_first_load:
                    for panel in self.panels:
                        panel.ppi.reset_camera(self.max_range.get())
                self.refresh_plot()

            self.after(0, _on_main)

        threading.Thread(target=_background, daemon=True).start()

    # ── Stagger detection ────────────────────────────────────────────────────

    def _run_stagger_detection(self):
        """Run detect_stagger_nyquist for the *currently active sweep* and
        push results into the Tk state vars. Called on file load and on
        every sweep_index change (see the trace set up in __init__), so
        the unfold-brush Nyquist tracks whichever sweep is on screen."""
        if self.volume is None:
            return

        sweep_index = self.sweep_index.get()
        # Pass the raw DoradeFile if the volume carries one (set by _load_dorade)
        dorade_file = getattr(self.volume, '_dorade_file', None)
        info = detect_stagger_nyquist(self.volume, dorade_file=dorade_file,
                                       sweep_index=sweep_index)

        self._stagger_short_nyq.set(info['short_nyquist'] or 0.0)
        self._stagger_long_nyq.set( info['long_nyquist']  or 0.0)
        self._stagger_fixed_nyq.set(info['fixed_nyquist'] or 0.0)
        self._stagger_ratio_str.set(info['stagger_ratio'] or "")
        self._stagger_source = info['source']

        # Mirror into unfold-brush Nyquist whenever auto-tracking is on
        # (default). Manually typing a value into the entry switches
        # tracking off so this won't clobber an intentional override.
        if self._brush_nyquist_auto.get():
            auto_nyq = (info['short_nyquist']
                        or info['fixed_nyquist']
                        or 0.0)
            self._setting_brush_nyquist_programmatically = True
            try:
                self._brush_nyquist.set(round(auto_nyq, 3))
            finally:
                self._setting_brush_nyquist_programmatically = False

        # Refresh the panel display if it is open
        if self._editor_panel_win and self._editor_panel_win.winfo_exists():
            try:
                self._editor_panel_win._refresh_stagger_display()
                self._editor_panel_win._refresh_unfold_center_display()
                self._editor_panel_win._refresh_gatefilter_fields()
            except Exception:
                pass

        # ── Rendering ─────────────────────────────────────────────────────────────

    def _current_field_info(self, panel: "PanelState" = None):
        """Convenience wrapper: field info for *panel* (defaults to the
        active panel). Most code should call panel.current_field_info()
        directly when it already has a panel reference."""
        panel = panel or self._active_panel or (self.panels[0] if self.panels else None)
        if panel is None:
            return None
        return panel.current_field_info()

    def refresh_plot(self):
        self._refresh_id = None
        if self.volume is None:
            return
        si    = max(0, min(len(self.volume.sweeps)-1, self.sweep_index.get()))
        sweep = self.volume.sweeps[si]
        mr    = self.max_range.get()

        visible_fields = set()  # only actually-rendered fields, for prewarm
        for panel in self.panels:
            fi = panel.current_field_info()
            if fi is None:
                continue
            fname, flabel, vmin, vmax, cmap_name = fi

            # Skip panels whose field doesn't exist in this sweep — avoids
            # building a full NaN render for every missing product in big files.
            if fname not in sweep.fields:
                panel.header_var.set(f"(field '{fname}' not in this sweep)")
                self._clear_panel_colorbar(panel)
                continue

            cmap = resolve_cmap(cmap_name)
            visible_fields.add(fname)

            render = self._get_render_product(si, fname, vmin, vmax, cmap_name, cmap)
            panel.ppi.show_cached_render(
                render,
                max_range_km     = mr,
                ring_interval_km = self.ring_interval.get(),
                az_interval_deg  = self.az_interval.get(),
                show_rings       = self.show_rings.get(),
                show_azimuths    = self.show_azimuths.get(),
            )

            rname = sweep.radar_name or self.volume.radar_name or "Unknown"
            t_str = (sweep.scan_time.strftime("%Y-%m-%dT%H:%M:%SZ")
                     if sweep.scan_time else "?")
            is_rhi = getattr(sweep, "scan_type", "ppi") == "rhi"
            mode_tag = "RHI" if is_rhi else "PPI"
            angle_label = (f"az={sweep.fixed_angle:.1f}°" if is_rhi
                           else f"{sweep.fixed_angle:.1f}°")
            panel.header_var.set(
                f"{t_str}   {rname}   {angle_label}  [{mode_tag}]   {flabel}")

            # Each panel gets its own colorbar, scaled to its own field —
            # not just whichever panel was last clicked.
            self._update_panel_colorbar(panel, cmap, vmin, vmax, flabel)

        # Axisymmetric overlay: draw on any new panels / hide if the sweep changed.
        try:
            self._axisym.sync_visibility(si)
            self._xsect.sync_visibility(si)
            self._detect.sync_visibility(si)
        except Exception:
            import traceback
            print(traceback.format_exc())

        # Only prewarm visible fields, not every field in the file.
        self._schedule_prewarm_adjacent(si, fields=visible_fields or None)

    def _render_cache_key(self, sweep_index: int, field_name: str,
                          vmin: float, vmax: float, cmap_name: str):
        return (int(sweep_index), field_name, round(float(vmin), 6),
                round(float(vmax), 6), str(cmap_name))

    def _touch_render_cache_key(self, key):
        try:
            self._render_cache_order.remove(key)
        except ValueError:
            pass
        self._render_cache_order.append(key)
        while len(self._render_cache_order) > self._render_cache_limit:
            old = self._render_cache_order.pop(0)
            self._render_cache.pop(old, None)

    def _clear_render_cache(self):
        self._render_cache.clear()
        self._render_cache_order.clear()
        self._geometry_cache.clear()
        if self._prewarm_id:
            try:
                self.after_cancel(self._prewarm_id)
            except Exception:
                pass
            self._prewarm_id = None

    def _invalidate_render_cache(self, sweep_index=None, field_name=None):
        keys = list(self._render_cache)
        for key in keys:
            si, fname = key[0], key[1]
            if ((sweep_index is None or si == sweep_index) and
                    (field_name is None or fname == field_name)):
                self._render_cache.pop(key, None)
                try:
                    self._render_cache_order.remove(key)
                except ValueError:
                    pass

    def _get_geometry_cached(self, mode: str, sweep):
        if mode == "rhi":
            geom_key = ("rhi",) + PPICanvas._geometry_key(None, "rhi", sweep.elevation, sweep.ranges)
        else:
            geom_key = ("ppi",) + PPICanvas._geometry_key(None, "ppi", sweep.azimuth, sweep.ranges)
        cached = self._geometry_cache.get(geom_key)
        if cached is not None:
            return geom_key, cached[0], cached[1]
        if mode == "rhi":
            verts, faces = _build_rhi_geometry(sweep.elevation, sweep.ranges)
        else:
            verts, faces = _build_ppi_geometry(sweep.azimuth, sweep.ranges)
        self._geometry_cache[geom_key] = (verts, faces)
        return geom_key, verts, faces

    def _get_render_product(self, sweep_index: int, field_name: str,
                            vmin: float, vmax: float, cmap_name: str, cmap):
        key = self._render_cache_key(sweep_index, field_name, vmin, vmax, cmap_name)
        cached = self._render_cache.get(key)
        if cached is not None:
            self._touch_render_cache_key(key)
            return cached

        sweep = self.volume.sweeps[sweep_index]
        is_rhi = getattr(sweep, "scan_type", "ppi") == "rhi"
        nrays, ngates = len(sweep.azimuth), len(sweep.ranges)
        if field_name in sweep.fields:
            data = np.asarray(sweep.fields[field_name], dtype="float32")
            data = self._normalize_field_shape(data, nrays, ngates)
        else:
            data = np.full((nrays, ngates), np.nan, dtype="float32")

        use_image = False  # always mesh path: uses true per-ray azimuths
        if use_image:
            gridded, r0_km, dr_km, da_deg = _grid_ppi_for_image(sweep, data)
            render = {
                "mode": "image",
                "sweep": sweep,
                "rgba": _data_to_rgba(gridded, cmap, vmin, vmax, dtype=np.uint8),
                "transform": PPICanvas._ppi_image_transform(None, r0_km, dr_km, da_deg, gridded.shape[0]),
            }
        else:
            mode = "rhi" if is_rhi else "ppi"
            geom_key, verts, faces = self._get_geometry_cached(mode, sweep)
            render = {
                "mode": mode,
                "sweep": sweep,
                "geom_key": geom_key,
                "verts": verts,
                "faces": faces,
                "face_colors": _data_to_face_colors(data, cmap, vmin, vmax),
            }

        self._render_cache[key] = render
        self._touch_render_cache_key(key)
        return render

    def _normalize_field_shape(self, data: np.ndarray, nrays: int, ngates: int):
        if data.shape == (nrays, ngates):
            return data
        r, g = data.shape
        if r < nrays:
            data = np.vstack([data, np.full((nrays-r, g), np.nan, "float32")])
        elif r > nrays:
            data = data[:nrays, :]
        if data.shape[1] < ngates:
            data = np.hstack([data, np.full((data.shape[0], ngates-data.shape[1]),
                                            np.nan, "float32")])
        else:
            data = data[:, :ngates]
        return data

    def _schedule_prewarm_adjacent(self, sweep_index: int, fields=None):
        if self._prewarm_id:
            try:
                self.after_cancel(self._prewarm_id)
            except Exception:
                pass
        self._prewarm_id = self.after_idle(
            lambda si=sweep_index, f=fields: self._prewarm_adjacent(si, f))

    def _prewarm_adjacent(self, sweep_index: int, fields=None):
        self._prewarm_id = None
        if self.volume is None:
            return
        n = len(self.volume.sweeps)
        wanted = [i for i in (sweep_index - 1, sweep_index + 1) if 0 <= i < n]
        if not wanted:
            return
        panel_infos = [p.current_field_info() for p in self.panels]
        panel_infos = [fi for fi in panel_infos if fi is not None]
        # If a fields set was supplied, only prewarm those fields — avoids
        # building renders for every field in large multi-product volumes.
        if fields:
            panel_infos = [fi for fi in panel_infos if fi[0] in fields]
        for si in wanted:
            for fname, _label, vmin, vmax, cmap_name in panel_infos:
                try:
                    self._get_render_product(si, fname, vmin, vmax,
                                             cmap_name, resolve_cmap(cmap_name))
                except Exception:
                    pass

    def _update_panel_colorbar(self, panel: "PanelState", cmap, vmin, vmax, label):
        """Render/refresh *panel*'s own colorbar strip, scaled to that
        panel's field — each panel keeps its own bar so different
        products (or the same product with a different vmin/vmax) never
        share one colorbar."""
        if not PIL_AVAILABLE:
            return
        try:
            panel._cb_args = (cmap, vmin, vmax, label)
            panel._cb_resize_id = None
            w = panel.ppi._vcanvas.native.winfo_width()
            h = CB_H
            if w < 10:
                cols = max(1, self._layout_cols)
                w = (self.winfo_width() or 800) // cols

            # Fewer ticks for narrower panels (e.g. a 2x2 or 4x4 grid) so
            # labels don't collide — full 12 ticks once there's room for
            # them, tapering down as the panel gets tighter.
            n_ticks = max(4, min(12, w // 45))

            # Bucket the width to the nearest 20px for the cache key so a
            # layout change (which changes panel width, and therefore
            # n_ticks) always forces a re-render, without re-rendering on
            # every 1px jitter from Tk's geometry manager.
            w_bucket = int(w)   # exact width so the bar always spans the panel
            key = (getattr(cmap, "name", repr(cmap)), round(float(vmin), 6),
                   round(float(vmax), 6), str(label), w_bucket, n_ticks)
            if panel._cb_key == key:
                return

            img    = _render_colorbar(cmap, vmin, vmax, label, w, h,
                                      n_ticks=n_ticks)
            panel.ppi.set_colorbar(img)
            panel._cb_key = key
        except Exception:
            pass

    def _clear_panel_colorbar(self, panel: "PanelState"):
        """Blank *panel*'s colorbar strip (e.g. its field isn't in the
        current sweep) so it doesn't keep showing a stale scale."""
        if panel._cb_key is None:
            return
        panel.ppi.set_colorbar(None)
        panel._cb_key = None

    # ── Cursor ────────────────────────────────────────────────────────────────

    def _update_cursor(self, panel: "PanelState", x_km: float, y_km: float):
        r_km  = math.sqrt(x_km**2 + y_km**2)
        fi    = panel.current_field_info()
        si    = self.sweep_index.get()
        sweep = (self.volume.sweeps[si]
                 if self.volume and 0 <= si < len(self.volume.sweeps) else None)
        is_rhi = sweep is not None and getattr(sweep, "scan_type", "ppi") == "rhi"

        if is_rhi:
            el = math.degrees(math.atan2(y_km, x_km))
            az = float(np.nanmedian(sweep.azimuth)) if sweep is not None else 0.0
        else:
            az = (90.0 - math.degrees(math.atan2(y_km, x_km))) % 360.0
            el = 0.0

        val_str = ""
        if fi and sweep is not None:
            fname = fi[0]
            if fname in sweep.fields and sweep.azimuth is not None:
                r_arr = sweep.ranges / 1000.0
                if is_rhi:
                    el_arr = sweep.elevation
                    el_d   = np.abs(el_arr - el)
                    ai     = int(np.argmin(el_d))
                else:
                    az_arr = sweep.azimuth
                    az_d   = np.abs((az_arr - az + 180) % 360 - 180)
                    ai     = int(np.argmin(az_d))
                ri = int(np.argmin(np.abs(r_arr - r_km)))
                try:
                    v = float(sweep.fields[fname][ai, ri])
                    val_str = (f"  val={v:.2f}" if np.isfinite(v)
                               else "  val=NaN")
                except Exception:
                    pass

        if is_rhi:
            self.cursor_var.set(
                f"Cursor: r={r_km:.1f} km  el={el:.1f}°  az={az:.1f}°"
                f"{val_str}  [{fi[1] if fi else ''}]")
        else:
            self.cursor_var.set(
                f"Cursor: {x_km:+.1f} km E  {y_km:+.1f} km N  "
                f"(r={r_km:.1f} km  az={az:.1f}°){val_str}  [{fi[1] if fi else ''}]")

    # ── Right-click menu ──────────────────────────────────────────────────────

    def _show_right_click_menu(self, panel: "PanelState", tk_event):
        menu = self._mk_menu(self)
        menu.add_command(label="Reset Zoom", command=self._reset_zoom)
        menu.add_separator()
        cmap_sub = self._mk_menu(menu)
        for group, names in [
            ("GURT",       list(GURT_CMAPS.keys())),
            ("Diverging",  ["RdBu_r", "RdYlBu_r", "coolwarm", "bwr"]),
            ("Sequential", ["viridis", "plasma", "inferno", "turbo", "jet", "Greys"]),
        ]:
            cmap_sub.add_command(label=f"── {group} ──", state="disabled")
            for cn in names:
                cmap_sub.add_command(label=cn,
                                     command=lambda c=cn: self._set_cmap(c))
        menu.add_cascade(label="Colormap", menu=cmap_sub)
        menu.add_separator()
        try:
            ax_km, ay_km = panel.ppi._canvas_to_scene(tk_event.x, tk_event.y)
            menu.add_command(label="Axisymmetric patch here",
                             command=lambda p=panel, x=ax_km, y=ay_km: self._axisym.run_at(p, x, y))
            menu.add_command(label="Track vortex: draw search box…", command=self._detect.begin_box)
            menu.add_command(label="Radial cross-section here",
                             command=lambda p=panel, x=ax_km, y=ay_km: self._xsect.run_at(p, x, y, "radial"))
            menu.add_command(label="Angular cross-section here",
                             command=lambda p=panel, x=ax_km, y=ay_km: self._xsect.run_at(p, x, y, "angular"))
        except Exception:
            pass
        menu.add_command(label="Open Editor Panel…",
                         command=self._open_editor_panel)
        menu.add_command(label="Open Param & Colors…",
                         command=lambda p=panel: self._open_param_dialog_current(p))
        try:
            rx = panel.ppi._vcanvas.native.winfo_rootx() + tk_event.x
            ry = panel.ppi._vcanvas.native.winfo_rooty() + tk_event.y
            menu.tk_popup(rx, ry)
        finally:
            menu.grab_release()
            self.after(100, self.focus_set)

    def _reset_zoom(self):
        mr = self.max_range.get()
        for panel in self.panels:
            panel.ppi.reset_camera(mr)

    # ── View helpers ──────────────────────────────────────────────────────────

    def _ask_max_range(self):
        self._ask_float("Max range (km):", self.max_range,
                        callback=lambda: (self._reset_zoom(),
                                         self._schedule_refresh()))

    def _refresh_ticks(self):
        for panel in self.panels:
            try:
                panel.ppi._schedule_ticks()
            except Exception:
                pass

    def _ask_tick_interval(self):
        self._ask_float("Tick interval (km, 0 = automatic with zoom):", self.tick_interval,
                        callback=self._refresh_ticks)

    def _ask_ring_interval(self):
        self._ask_float("Ring interval (km):", self.ring_interval,
                        callback=self._schedule_refresh)

    def _ask_float(self, prompt, var, callback=None):
        win = tk.Toplevel(self)
        win.title("Enter value")
        win.resizable(False, False)
        win.grab_set()
        win.configure(bg=BG)
        tk.Label(win, text=prompt, padx=10, pady=6, bg=BG, fg="white").pack()
        ent = tk.Entry(win, width=10, bg="#fff")
        ent.insert(0, str(var.get()))
        ent.pack(padx=10, pady=4)
        ent.focus_set()
        def ok(event=None):
            try:
                var.set(float(ent.get()))
            except ValueError:
                pass
            win.destroy()
            if callback:
                callback()
        ent.bind("<Return>", ok)
        tk.Button(win, text="OK", command=ok, width=8, bg=BAR_BG).pack(pady=4)

    # ── ParamColorDialog integration ──────────────────────────────────────────

    def _open_param_dialog_current(self, panel: "PanelState" = None):
        """Open the full field/colormap editor for *panel*'s displayed field
        (defaults to the active panel)."""
        panel = panel or self._active_panel or (self.panels[0] if self.panels else None)
        if panel is None:
            return
        fi = panel.current_field_info()
        if fi is None:
            return
        fname, label, vmin, vmax, cmap_name = fi
        available = list(self.volume.field_names) if self.volume else [fname]

        def on_apply(slot, fn, lbl, mn, mx, cn):
            panel.field_combo.set(fn)
            panel.vmin_var.set(str(mn))
            panel.vmax_var.set(str(mx))
            panel.cmap_override = cn
            self._schedule_refresh()

        ParamColorDialog(self, 0, (fname, label, vmin, vmax, cmap_name),
                         available, on_apply)

    # ── Editor panel ──────────────────────────────────────────────────────────

    def _open_editor_panel(self):
        if self._editor_panel_win and self._editor_panel_win.winfo_exists():
            self._editor_panel_win.lift()
            return
        self._editor_panel_win = EditorPanel(self)

    # ── Brush application ─────────────────────────────────────────────────────

    def _brush_gate_indices(self, sweep, x_km: float, y_km: float,
                            radius_km: float):
        return brushes.brush_gate_indices(sweep, x_km, y_km, radius_km)

    def _apply_brush(self, panel: "PanelState", canvas_x: int, canvas_y: int, mode: str):
        """Apply the active brush at the given canvas pixel position, on
        *panel*'s currently-displayed field only. (Implementation in brushes.py)"""
        return brushes.apply_brush(self, panel, canvas_x, canvas_y, mode,
                                   resolve_cmap=resolve_cmap)

    # ── Boundary tool application ───────────────────────────────────────────────
    #
    # The boundary tool is the same four edit modes as the brush (unfold /
    # deglitch / gap fill / delete gates) — only the *selection* differs:
    # a clicked polygon instead of a dragged circle, applied once instead
    # of continuously. See PPICanvas._add_boundary_point for vertex entry
    # and brushes.apply_boundary for the shared edit math.

    def _apply_boundary_edit(self):
        """Finalize the in-progress boundary polygon on the active panel
        (Enter key, or the Editor Panel's "Apply Edit" button): run the
        currently-selected edit mode over every gate inside it, then clear
        the boundary so the next click starts a fresh one."""
        if not self._boundary_mode.get():
            return
        mode = self._editor_mode.get()
        if mode not in EDIT_BRUSH_MODES:
            return
        panel = self._active_panel
        if panel is None:
            return
        points = panel.ppi._boundary_points
        if len(points) < 3:
            if points:
                messagebox.showinfo(
                    "Boundary Tool",
                    "Add at least 3 points before applying.\n"
                    "(Right-click removes the last point.)")
            return
        if self._boundary_all_sweeps.get():
            changed = brushes.apply_boundary_all_sweeps(
                self, panel, points, mode, resolve_cmap=resolve_cmap)
        else:
            changed = brushes.apply_boundary(self, panel, points, mode,
                                             resolve_cmap=resolve_cmap)
        if changed:
            panel.ppi._clear_boundary()
        return changed

    def _on_escape_key(self):
        """Escape cancels an in-progress boundary on any panel, if one
        exists; otherwise it falls back to the normal "reset zoom"."""
        for panel in self.panels:
            if panel.ppi._boundary_points:
                panel.ppi._clear_boundary()
                return
        self._reset_zoom()

    def _on_boundary_changed(self):
        """Called whenever a boundary vertex is added/removed/cleared on
        any panel, so the open Editor Panel (if any) can refresh its
        live point-count status."""
        win = self._editor_panel_win
        if win is not None and win.winfo_exists():
            win._update_boundary_status()

    def _clear_all_boundaries(self):
        for panel in self.panels:
            panel.ppi._clear_boundary()


    # ── Duplicate field ───────────────────────────────────────────────────────

    def _duplicate_field_dialog(self):
        if self.volume is None:
            messagebox.showwarning("Duplicate Field", "No radar data loaded.")
            return
        DuplicateFieldDialog(
            self,
            available_fields=list(self.volume.field_names),
            on_apply=self._do_duplicate_field,
        )

    def _do_duplicate_field(self, src_name, dst_name):
        if not dst_name:
            messagebox.showwarning("Duplicate Field", "Destination name is empty.")
            return
        if self.volume is None:
            return
        for sw in self.volume.sweeps:
            if src_name in sw.fields:
                sw.fields[dst_name] = sw.fields[src_name].copy()
        if dst_name not in self.volume.field_names:
            self.volume.field_names.append(dst_name)
        for panel in self.panels:
            cur_vals = list(panel.field_combo["values"])
            if dst_name not in cur_vals:
                cur_vals.append(dst_name)
                panel.field_combo["values"] = cur_vals
        self._invalidate_render_cache(field_name=dst_name)
        self._schedule_refresh()

    # ── Smoothed velocity magnitude ───────────────────────────────────────────

    def _smooth_velmag_dialog(self):
        if self.volume is None:
            messagebox.showwarning("Smoothed velocity magnitude", "No radar data loaded.")
            return
        names = list(self.volume.field_names)
        from dreader_xsect import is_velocity_field
        vel = [n for n in names if is_velocity_field(n)] or names
        win = tk.Toplevel(self)
        win.title("Smoothed velocity magnitude")
        win.configure(bg=DLG_BG)
        win.resizable(False, False)
        win.transient(self)
        win.grab_set()
        pad = dict(padx=10, pady=5)
        tk.Label(win, text="Velocity field:", bg=DLG_BG, fg=DLG_FG).grid(row=0, column=0, sticky="w", **pad)
        src = tk.StringVar(value=vel[0])
        ttk.Combobox(win, textvariable=src, values=names, width=18, state="readonly").grid(row=0, column=1, **pad)
        tk.Label(win, text="Smoothing length (m):", bg=DLG_BG, fg=DLG_FG).grid(row=1, column=0, sticky="w", **pad)
        sm = tk.StringVar(value=f"{self._detect.settings['swath_smooth_m']:g}")
        tk.Entry(win, textvariable=sm, width=10, bg=DLG_ENTRY).grid(row=1, column=1, sticky="w", **pad)
        tk.Label(win, text="Output field name:", bg=DLG_BG, fg=DLG_FG).grid(row=2, column=0, sticky="w", **pad)
        dst = tk.StringVar(value="VMAG_SMOOTH")
        tk.Entry(win, textvariable=dst, width=20, bg=DLG_ENTRY).grid(row=2, column=1, sticky="w", **pad)
        tk.Label(win, text="|V| averaged over a window of this size in azimuth and range;\n"
                           "applied to every sweep of the loaded volume. Also sets the\n"
                           "wind swath smoothing for the next box track.",
                 bg=DLG_BG, fg=DLG_HINT2, font=("TkDefaultFont", 8), justify="left").grid(
            row=3, column=0, columnspan=2, **pad)

        def ok():
            try:
                L = float(sm.get())
            except ValueError:
                messagebox.showwarning("Smoothed velocity magnitude", "Enter a numeric length.", parent=win)
                return
            name = dst.get().strip()
            if not name:
                messagebox.showwarning("Smoothed velocity magnitude", "Output name is empty.", parent=win)
                return
            self._detect.settings["swath_smooth_m"] = L      # the wind swath uses the same length
            win.destroy()
            self._do_smooth_velmag(src.get(), L, name)

        bf = tk.Frame(win, bg=DLG_BG)
        bf.grid(row=4, column=0, columnspan=2, pady=8)
        tk.Button(bf, text="Create", command=ok, bg=DLG_BG, padx=10).pack(side="left", padx=4)
        tk.Button(bf, text="Cancel", command=win.destroy, bg=DLG_BG, padx=10).pack(side="left", padx=4)

    def _do_smooth_velmag(self, src, smooth_m, dst):
        from velmag import smoothed_velmag
        made = 0
        for sw in self.volume.sweeps:
            if src not in sw.fields or getattr(sw, "scan_type", "ppi") == "rhi":
                continue
            try:
                sw.fields[dst] = smoothed_velmag(sw, src, smooth_m)
                made += 1
            except Exception:
                pass
        if not made:
            messagebox.showinfo("Smoothed velocity magnitude", "No sweep could be processed.")
            return
        if dst not in self.volume.field_names:
            self.volume.field_names.append(dst)
        for panel in self.panels:
            vals = list(panel.field_combo["values"])
            if dst not in vals:
                vals.append(dst)
                panel.field_combo["values"] = vals
        self._invalidate_render_cache(field_name=dst)
        self._schedule_refresh()

    # ── Despeckle (DREADER despeck_parm) ──────────────────────────────────────

    def _despeckle_dialog(self):
        if self.volume is None:
            messagebox.showwarning("Despeckle", "No radar data loaded.")
            return
        names = list(self.volume.field_names)
        fi = self._active_panel.current_field_info() if getattr(self, "_active_panel", None) else None
        win = tk.Toplevel(self)
        win.title("Despeckle (DREADER)")
        win.configure(bg=DLG_BG)
        win.resizable(False, False)
        win.transient(self)            # not modal: keep zooming / panning while previewing
        pad = dict(padx=10, pady=4)
        last = getattr(self, "_despeckle_last", {"radius": "2", "min_good": "10", "max_pass": "0"})
        fld = tk.StringVar(value=fi[0] if fi else names[0])
        tk.Label(win, text="Field:", bg=DLG_BG, fg=DLG_FG).grid(row=0, column=0, sticky="w", **pad)
        ttk.Combobox(win, textvariable=fld, values=names, width=18, state="readonly").grid(row=0, column=1, **pad)
        vs = {}
        for i, (lab, key) in enumerate([("Window radius (gates / rays):", "radius"),
                                        ("Min good neighbors:", "min_good"),
                                        ("Max passes (0 = until stable):", "max_pass")], 1):
            tk.Label(win, text=lab, bg=DLG_BG, fg=DLG_FG).grid(row=i, column=0, sticky="w", **pad)
            vs[key] = tk.StringVar(value=last[key])
            tk.Entry(win, textvariable=vs[key], width=8, bg=DLG_ENTRY).grid(row=i, column=1, sticky="w", **pad)
        tk.Label(win, text="Scope:", bg=DLG_BG, fg=DLG_FG).grid(row=4, column=0, sticky="w", **pad)
        scope = tk.StringVar(value="Current sweep")
        ttk.Combobox(win, textvariable=scope, values=["Current sweep", "All sweeps"], width=18,
                     state="readonly").grid(row=4, column=1, **pad)
        tk.Label(win, text="Output field (blank = edit in place):", bg=DLG_BG, fg=DLG_FG).grid(
            row=5, column=0, sticky="w", **pad)
        dst = tk.StringVar(value="")
        tk.Entry(win, textvariable=dst, width=20, bg=DLG_ENTRY).grid(row=5, column=1, sticky="w", **pad)
        wrap = tk.IntVar(value=1)
        tk.Checkbutton(win, text="Wrap across north (full-circle sweeps)", variable=wrap, bg=DLG_BG, fg=DLG_FG,
                       selectcolor="#333", activebackground=DLG_BG).grid(row=6, column=0, columnspan=2,
                                                                         sticky="w", **pad)
        live = tk.IntVar(value=0)
        tk.Checkbutton(win, text="Live preview (redo on every change)", variable=live, bg=DLG_BG, fg=DLG_FG,
                       selectcolor="#333", activebackground=DLG_BG).grid(row=7, column=0, columnspan=2,
                                                                         sticky="w", **pad)
        pct = tk.StringVar(value="")
        tk.Label(win, textvariable=pct, bg=DLG_BG, fg=DLG_FG, font=("TkDefaultFont", 8),
                 justify="left", wraplength=330).grid(row=8, column=0, columnspan=2, padx=10, pady=(2, 0))
        info = tk.StringVar(value="Preview marks the gates that would be removed (red) on the current sweep.")
        tk.Label(win, textvariable=info, bg=DLG_BG, fg=DLG_HINT2, font=("TkDefaultFont", 8),
                 justify="left", wraplength=330).grid(row=9, column=0, columnspan=2, **pad)

        def describe(*_):
            try:
                r, mg, _mp = params()
            except ValueError:
                pct.set("")
                return
            full = (2 * r + 1) ** 2 - 1
            edge = (r + 1) * (2 * r + 1) - 1          # neighbours of a gate on a straight edge
            if full <= 0:
                pct.set("")
                return
            pct.set(f"Min {mg} of {full} neighbours = {100.0 * mg / full:.0f}%  "
                    f"(a gate on a straight edge has about {edge} = {100.0 * edge / full:.0f}%; "
                    f"set the minimum below that or the edges erode)")

        def params():
            r, mg, mp = (int(float(vs["radius"].get())), int(float(vs["min_good"].get())),
                         int(float(vs["max_pass"].get())))
            if r < 0 or mg < 0 or mp < 0:
                raise ValueError
            return r, mg, mp

        def preview(*_):
            try:
                r, mg, mp = params()
            except ValueError:
                info.set("Radius, minimum and passes must be whole numbers >= 0.")
                return
            n, passes = self._despeckle_preview(fld.get(), r, mg, mp, bool(wrap.get()))
            if n is None:
                info.set(f"Field {fld.get()} is not in the current sweep.")
            else:
                info.set(f"Would remove {n} gate(s) in {passes} pass(es) on this sweep.")

        job = {"id": None}

        def live_changed(*_):
            if not live.get():
                return
            if job["id"] is not None:
                win.after_cancel(job["id"])
            job["id"] = win.after(300, preview)

        for v in list(vs.values()) + [fld, wrap]:
            v.trace_add("write", live_changed)
        for v in vs.values():
            v.trace_add("write", describe)
        describe()
        live.trace_add("write", lambda *_: (preview() if live.get() else None))

        def close():
            self._despeckle_clear_preview()
            win.destroy()

        def ok():
            try:
                r, mg, mp = params()
            except ValueError:
                messagebox.showwarning("Despeckle", "Radius, minimum and passes must be whole numbers >= 0.",
                                       parent=win)
                return
            self._despeckle_last = {k: v.get() for k, v in vs.items()}
            close()
            self._do_despeckle(fld.get(), r, mg, mp, scope.get() == "All sweeps", dst.get().strip(), bool(wrap.get()))

        def folder():
            try:
                r, mg, mp = params()
            except ValueError:
                messagebox.showwarning("Despeckle", "Radius, minimum and passes must be whole numbers >= 0.",
                                       parent=win)
                return
            if not getattr(self, "file_list", None):
                messagebox.showwarning("Despeckle", "No folder of files is loaded.", parent=win)
                return
            out = dst.get().strip() or f"{fld.get()}-edit"
            if out == fld.get():
                if not messagebox.askyesno(
                        "Despeckle folder",
                        f"The output field name is the same as the source ({out}).\n"
                        "The exported CfRadial files will contain the despeckled data under that name "
                        "(your original files are not changed). Continue?", parent=win):
                    return
            out_dir = filedialog.askdirectory(parent=win, title="Folder for the despeckled CfRadial files")
            if not out_dir:
                return
            self._despeckle_last = {k: v.get() for k, v in vs.items()}
            close()
            self._do_batch_despeckle(fld.get(), r, mg, mp, bool(wrap.get()), out, out_dir)

        win.protocol("WM_DELETE_WINDOW", close)
        bf = tk.Frame(win, bg=DLG_BG)
        bf.grid(row=10, column=0, columnspan=2, pady=8)
        tk.Button(bf, text="Preview", command=preview, bg=DLG_BG, padx=10).pack(side="left", padx=4)
        tk.Button(bf, text="Clear preview", command=self._despeckle_clear_preview, bg=DLG_BG,
                  padx=10).pack(side="left", padx=4)
        tk.Button(bf, text="Apply", command=ok, bg=DLG_BG, padx=10).pack(side="left", padx=4)
        tk.Button(bf, text="Close", command=close, bg=DLG_BG, padx=10).pack(side="left", padx=4)
        tk.Button(win, text="Despeckle every file in the folder → save as CfRadial…", command=folder,
                  bg=DLG_BG, padx=10).grid(row=11, column=0, columnspan=2, pady=(0, 10))

    def _do_batch_despeckle(self, fname, radius, min_good, max_pass, wrap, dst, out_dir):
        """Override in main.py: per-file load / despeckle every sweep / write CfRadial."""
        messagebox.showinfo("Despeckle folder", "(Implement _do_batch_despeckle in main.py)")

    def _despeckle_preview(self, fname, radius, min_good, max_pass, wrap):
        """Mark the gates despeckle would delete on the current sweep (all panels).
        Returns (n_removed, passes), or (None, 0) if the field is missing."""
        vol = self.volume
        si = self.sweep_index.get()
        if vol is None or not (0 <= si < len(vol.sweeps)):
            return None, 0
        sw = vol.sweeps[si]
        if fname not in sw.fields or getattr(sw, "scan_type", "ppi") == "rhi":
            self._despeckle_clear_preview()
            return None, 0
        old = sw.fields[fname]
        new, n, passes = edit_despeckle.despeckle_field(old, radius, min_good, max_pass, wrap)
        old_good = ~np.ma.getmaskarray(old)
        od = np.ma.getdata(old)
        if od.dtype.kind == "f":
            old_good &= np.isfinite(od)           # NaN gates count as already missing
        gone = old_good & np.ma.getmaskarray(new)
        ri, gi = np.nonzero(gone)
        if ri.size:
            az = np.radians(np.asarray(sw.azimuth, dtype="float64")[ri])
            rk = np.asarray(sw.ranges, dtype="float64")[gi] / 1000.0
            pts = np.column_stack([rk * np.sin(az), rk * np.cos(az)])
        else:
            pts = None
        # only the panel(s) actually showing the previewed field get the marks
        show = []
        for panel in self.panels:
            try:
                fi = panel.current_field_info()
            except Exception:
                fi = None
            if fi and fi[0] == fname:
                show.append(panel)
        if not show:
            ap = getattr(self, "_active_panel", None)
            show = [ap] if ap in self.panels else self.panels[:1]
        for panel in self.panels:
            try:
                panel.ppi.set_preview_points(pts if panel in show else None)
            except Exception:
                pass
        return int(n), int(passes)

    def _despeckle_clear_preview(self):
        for panel in self.panels:
            try:
                panel.ppi.set_preview_points(None)
            except Exception:
                pass

    def _do_despeckle(self, fname, radius, min_good, max_pass, all_sweeps, dst, wrap):
        vol = self.volume
        if vol is None:
            return
        cur = self.sweep_index.get()
        idx = (range(len(vol.sweeps)) if all_sweeps else [cur])
        target = dst or fname
        snaps, removed, done = [], 0, 0
        for si in idx:
            sw = vol.sweeps[si]
            if fname not in sw.fields or getattr(sw, "scan_type", "ppi") == "rhi":
                continue
            new, n, _ = edit_despeckle.despeckle_field(sw.fields[fname], radius, min_good, max_pass, wrap)
            old = sw.fields.get(target)
            snaps.append({"sweep_index": si, "fields": ({target: old.copy()} if old is not None else {})})
            sw.fields[target] = new
            removed += n
            done += 1
        if not done:
            messagebox.showinfo("Despeckle", f"No sweep with field {fname} to process.")
            return
        if dst and dst not in vol.field_names:
            vol.field_names.append(dst)
            for panel in self.panels:
                vals = list(panel.field_combo["values"])
                if dst not in vals:
                    vals.append(dst)
                    panel.field_combo["values"] = vals
        if not dst:
            self._push_undo_batch(snaps)
        for si in idx:
            self._invalidate_render_cache(sweep_index=si, field_name=target)
        self._schedule_refresh()
        from edit.native import status as _native_status
        self.cursor_var.set(f"Despeckle [{_native_status()}]: removed {removed} gate(s) in {done} sweep(s)")

    # ── Rotate Sweep ──────────────────────────────────────────────────────────

    def _rotate_sweep_dialog(self):
        if self.volume is None:
            messagebox.showwarning("Rotate Sweep", "No radar data loaded.")
            return
        si = self.sweep_index.get()
        if not (0 <= si < len(self.volume.sweeps)):
            return

        win = tk.Toplevel(self)
        win.title("Rotate Sweep")
        win.resizable(False, False)
        win.grab_set()
        win.configure(bg=DLG_BG)

        tk.Label(win, text="Rotate azimuth by (degrees):",
                 bg=DLG_BG, fg=DLG_FG, padx=10, pady=6).pack()
        tk.Label(win, text="Positive = clockwise. Applies to the current sweep only.",
                 bg=DLG_BG, fg=DLG_HINT2, font=("TkDefaultFont", 8),
                 padx=10).pack()
        ent = tk.Entry(win, width=10, bg=DLG_ENTRY)
        ent.insert(0, "0.0")
        ent.pack(padx=10, pady=6)
        ent.focus_set()
        ent.select_range(0, "end")

        def ok(event=None):
            try:
                deg = float(ent.get())
            except ValueError:
                messagebox.showwarning("Rotate Sweep",
                                       "Enter a numeric value.", parent=win)
                return
            win.destroy()
            self._do_rotate_sweep(si, deg)

        ent.bind("<Return>", ok)
        btns = tk.Frame(win, bg=DLG_BG)
        btns.pack(pady=(0, 8))
        tk.Button(btns, text="Rotate", command=ok, width=10,
                  bg=DLG_BG, fg=DLG_FG, relief="raised").pack(side="left", padx=4)
        tk.Button(btns, text="Cancel", command=win.destroy, width=10,
                  bg=DLG_BG, fg=DLG_FG, relief="raised").pack(side="left", padx=4)

    def _do_rotate_sweep(self, si, deg):
        if self.volume is None or not (0 <= si < len(self.volume.sweeps)):
            return
        sw = self.volume.sweeps[si]
        if sw.azimuth is None or deg == 0.0:
            return
        if self.sweep_index.get() == si:
            self._push_undo(extra={"azimuth": True})
        edit_geometry.rotate_sweep(sw, deg)
        # Geometry depends on azimuth, so force a rebuild on next draw.
        self._clear_render_cache()
        for panel in self.panels:
            panel.ppi._verts = None
        self._schedule_refresh()

    # ── Offset and Scale Product ─────────────────────────────────────────────

    def _offset_scale_dialog(self):
        if self.volume is None:
            messagebox.showwarning("Offset and Scale Product", "No radar data loaded.")
            return

        win = tk.Toplevel(self)
        win.title("Offset and Scale Product")
        win.resizable(False, False)
        win.grab_set()
        win.configure(bg=DLG_BG)

        pad = dict(padx=10, pady=4)

        tk.Label(win, text="Field:", bg=DLG_BG, fg=DLG_FG, anchor="w"
                 ).grid(row=0, column=0, sticky="w", **pad)
        fields = list(self.volume.field_names)
        field_var = tk.StringVar(
            value=(self._active_panel.field_combo.get()
                   if self._active_panel and self._active_panel.field_combo.get() in fields
                   else (fields[0] if fields else "")))
        field_cb = ttk.Combobox(win, textvariable=field_var, values=fields,
                                width=20, state="readonly")
        field_cb.grid(row=0, column=1, sticky="w", **pad)

        tk.Label(win, text="Scale (multiply by):", bg=DLG_BG, fg=DLG_FG, anchor="w"
                 ).grid(row=1, column=0, sticky="w", **pad)
        scale_var = tk.StringVar(value="1.0")
        tk.Entry(win, textvariable=scale_var, width=12, bg=DLG_ENTRY
                 ).grid(row=1, column=1, sticky="w", **pad)

        tk.Label(win, text="Offset (add):", bg=DLG_BG, fg=DLG_FG, anchor="w"
                 ).grid(row=2, column=0, sticky="w", **pad)
        offset_var = tk.StringVar(value="0.0")
        tk.Entry(win, textvariable=offset_var, width=12, bg=DLG_ENTRY
                 ).grid(row=2, column=1, sticky="w", **pad)

        tk.Label(win, text="new_value = old_value × scale + offset",
                 bg=DLG_BG, fg=DLG_HINT2, font=("TkDefaultFont", 8)
                 ).grid(row=3, column=0, columnspan=2, sticky="w", padx=10)

        scope_var = tk.StringVar(value="sweep")
        scope_frame = tk.LabelFrame(win, text=" Scope ", bg=DLG_BG, fg=DLG_FG,
                                    font=("TkDefaultFont", 9, "bold"),
                                    relief="groove", bd=1)
        scope_frame.grid(row=4, column=0, columnspan=2, sticky="ew", padx=10, pady=(8, 4))
        for val, lbl in [("sweep", "Current sweep only"),
                          ("volume", "All sweeps in this volume")]:
            tk.Radiobutton(scope_frame, text=lbl, variable=scope_var, value=val,
                           bg=DLG_BG, fg=DLG_FG, selectcolor=DLG_ABG,
                           activebackground=DLG_ABG, activeforeground=DLG_FG,
                           indicatoron=True, font=("TkDefaultFont", 9)
                           ).pack(anchor="w", padx=8, pady=2)

        def ok():
            fname = field_var.get().strip()
            if not fname:
                messagebox.showwarning("Offset and Scale Product",
                                       "Choose a field.", parent=win)
                return
            try:
                scale = float(scale_var.get())
                offset = float(offset_var.get())
            except ValueError:
                messagebox.showwarning("Offset and Scale Product",
                                       "Scale and offset must be numbers.", parent=win)
                return
            scope = scope_var.get()
            win.destroy()
            self._do_offset_scale(fname, scale, offset, scope)

        btns = tk.Frame(win, bg=DLG_BG)
        btns.grid(row=5, column=0, columnspan=2, pady=8)
        tk.Button(btns, text="Apply", command=ok, width=10,
                  bg=DLG_BG, fg=DLG_FG, relief="raised").pack(side="left", padx=4)
        tk.Button(btns, text="Cancel", command=win.destroy, width=10,
                  bg=DLG_BG, fg=DLG_FG, relief="raised").pack(side="left", padx=4)

    def _do_offset_scale(self, fname, scale, offset, scope):
        if self.volume is None:
            return
        si = self.sweep_index.get()
        if scope == "sweep":
            if not (0 <= si < len(self.volume.sweeps)):
                return
            self._push_undo([fname])
            edit_scaling.offset_scale_field(self.volume.sweeps[si], fname, scale, offset)
        else:
            # Volume-wide edit isn't covered by the single-sweep undo stack;
            # warn so it isn't mistaken for undoable.
            if not messagebox.askyesno(
                    "Offset and Scale Product",
                    f"Apply to all {len(self.volume.sweeps)} sweeps in this volume?\n"
                    "This cannot be undone with Ctrl+Z."):
                return
            edit_scaling.offset_scale_volume(self.volume.sweeps, fname, scale, offset)

        if scope == "sweep":
            self._invalidate_render_cache(sweep_index=si, field_name=fname)
        else:
            self._invalidate_render_cache(field_name=fname)
        self._schedule_refresh()

    # ── Offset Elevations / Estimate Fixed Angle ─────────────────────────────

    def _offset_elevations_dialog(self):
        if self.volume is None:
            messagebox.showwarning("Offset Elevations", "No radar data loaded.")
            return
        si = self.sweep_index.get()
        if not (0 <= si < len(self.volume.sweeps)):
            return
        sw = self.volume.sweeps[si]

        win = tk.Toplevel(self)
        win.title("Offset Elevations")
        win.resizable(False, False)
        win.grab_set()
        win.configure(bg=DLG_BG)

        pad = dict(padx=10, pady=4)

        cur_fixed = sw.fixed_angle
        tk.Label(win, text=f"Current fixed angle: {cur_fixed:.3f}°",
                 bg=DLG_BG, fg=DLG_HINT, font=("TkDefaultFont", 9)
                 ).grid(row=0, column=0, columnspan=2, sticky="w", **pad)

        tk.Label(win, text="Offset elevation by (degrees):",
                 bg=DLG_BG, fg=DLG_FG, anchor="w"
                 ).grid(row=1, column=0, sticky="w", **pad)
        offset_var = tk.StringVar(value="0.0")
        tk.Entry(win, textvariable=offset_var, width=10, bg=DLG_ENTRY
                 ).grid(row=1, column=1, sticky="w", **pad)
        tk.Label(win, text="Applies to ray elevation angles (and the fixed "
                            "angle) for the current sweep.",
                 bg=DLG_BG, fg=DLG_HINT2, font=("TkDefaultFont", 8), wraplength=320,
                 justify="left").grid(row=2, column=0, columnspan=2, sticky="w", padx=10)

        def apply_offset():
            try:
                deg = float(offset_var.get())
            except ValueError:
                messagebox.showwarning("Offset Elevations",
                                       "Enter a numeric value.", parent=win)
                return
            win.destroy()
            self._do_offset_elevations(si, deg)

        def estimate_fixed():
            self._do_estimate_fixed_angle(si)
            sw2 = self.volume.sweeps[si]
            cur_fixed_lbl.set(f"Current fixed angle: {sw2.fixed_angle:.3f}°")

        cur_fixed_lbl = tk.StringVar(value=f"Current fixed angle: {cur_fixed:.3f}°")
        # Re-point the label above at the StringVar so "Estimate Fixed Angle"
        # can update it live without rebuilding the dialog.
        for child in win.grid_slaves(row=0, column=0):
            child.destroy()
        tk.Label(win, textvariable=cur_fixed_lbl,
                 bg=DLG_BG, fg=DLG_HINT, font=("TkDefaultFont", 9)
                 ).grid(row=0, column=0, columnspan=2, sticky="w", **pad)

        tk.Button(win, text="Estimate Fixed Angle", command=estimate_fixed,
                  bg=DLG_BG, fg=DLG_FG, relief="raised"
                  ).grid(row=3, column=0, columnspan=2, sticky="ew", padx=10, pady=(8, 2))
        tk.Label(win, text="Recomputes the fixed angle from the sweep's "
                            "elevation data (fixes bad/sentinel values like -9999).",
                 bg=DLG_BG, fg=DLG_HINT2, font=("TkDefaultFont", 8), wraplength=320,
                 justify="left").grid(row=4, column=0, columnspan=2, sticky="w", padx=10)

        btns = tk.Frame(win, bg=DLG_BG)
        btns.grid(row=5, column=0, columnspan=2, pady=10)
        tk.Button(btns, text="Apply Offset", command=apply_offset, width=12,
                  bg=DLG_BG, fg=DLG_FG, relief="raised").pack(side="left", padx=4)
        tk.Button(btns, text="Close", command=win.destroy, width=10,
                  bg=DLG_BG, fg=DLG_FG, relief="raised").pack(side="left", padx=4)

    # ── GateFilter ────────────────────────────────────────────────────────────

    def _open_gatefilter_dialog(self):
        if self.volume is None:
            messagebox.showwarning("GateFilter", "No radar data loaded.")
            return
        GateFilterDialog(self)

    def _do_gatefilter(self, filter_field, condition, target_fields,
                       value, value2, scope):
        if self.volume is None:
            return
        si = self.sweep_index.get()
        if scope == "sweep":
            if not (0 <= si < len(self.volume.sweeps)):
                return
            self._push_undo(list(target_fields))
            sweeps = [self.volume.sweeps[si]]
        else:
            sweeps = self.volume.sweeps

        total = 0
        for sw in sweeps:
            try:
                total += apply_gate_filter(sw, filter_field, condition,
                                          target_fields, value, value2)
            except ValueError as e:
                messagebox.showwarning("GateFilter", str(e))
                return
        for fname in target_fields:
            if scope == "sweep":
                self._invalidate_render_cache(sweep_index=si, field_name=fname)
            else:
                self._invalidate_render_cache(field_name=fname)
        self._schedule_refresh()
        messagebox.showinfo("GateFilter", f"Masked {total} gate(s).")

    # ── GateFilter Brush — Tk-state plumbing for brushes.py ─────────────────
    #
    # Kept here (not in brushes.py) so the brush math never has to touch
    # Tk variables or show its own dialogs — it just gets back plain
    # values, or None if the expression isn't usable yet.

    def _resolve_gatefilter_spec(self, warn: bool = True):
        """Validate the GateFilter Brush expression built in the Editor
        Panel and return (conditions, combine) as plain values —
        conditions is a list of {"field", "condition", "value", "value2"}
        dicts. Returns None (after a warning dialog, unless warn=False)
        if anything about it isn't usable yet: no conditions, an unknown
        or missing filter field, or a non-numeric value. (No target-field
        check here — the brush always masks whichever field the panel is
        currently displaying, same as every other brush mode.)

        `warn=False` is used by multi-sweep/batch callers (apply-to-all-
        sweeps, folder batch edit) after the first sweep/file already
        showed the dialog once, so the same problem doesn't pop a new
        messagebox for every remaining sweep."""
        if not self._gatefilter_conditions:
            if warn:
                messagebox.showwarning(
                    "GateFilter Brush",
                    "Add at least one condition first\n"
                    "(Editor Panel ▸ GateFilter Brush Settings).")
            return None

        fields = list(self.volume.field_names) if self.volume else []
        conditions = []
        for row in self._gatefilter_conditions:
            field = row["field"].get().strip()
            condition = row["condition"].get()
            if not field or field not in fields:
                if warn:
                    messagebox.showwarning(
                        "GateFilter Brush", f"Invalid filter field: '{field}'.")
                return None
            n_vals = CONDITIONS.get(condition, 1)
            try:
                value = float(row["value"].get()) if n_vals >= 1 else 0.0
                value2 = float(row["value2"].get()) if n_vals >= 2 else 0.0
            except ValueError:
                if warn:
                    messagebox.showwarning(
                        "GateFilter Brush",
                        f"Value(s) for '{field}' must be numeric.")
                return None
            conditions.append({"field": field, "condition": condition,
                              "value": value, "value2": value2})

        return conditions, self._gatefilter_combine.get()

    def _do_offset_elevations(self, si, deg):
        if self.volume is None or not (0 <= si < len(self.volume.sweeps)):
            return
        sw = self.volume.sweeps[si]
        if deg == 0.0:
            return
        self._push_undo(extra={"elevation": True, "fixed_angle": True})
        edit_geometry.offset_elevations(sw, deg)
        self._clear_render_cache()
        self._schedule_refresh()

    def _do_estimate_fixed_angle(self, si):
        if self.volume is None or not (0 <= si < len(self.volume.sweeps)):
            return
        sw = self.volume.sweeps[si]
        self._push_undo(extra={"fixed_angle": True})
        try:
            new_fixed = edit_geometry.estimate_fixed_angle(sw)
        except ValueError as e:
            messagebox.showwarning("Estimate Fixed Angle", str(e))
            return
        self._invalidate_render_cache(sweep_index=si)
        self._schedule_refresh()
        n_valid = int(np.count_nonzero(
            np.isfinite(np.asarray(sw.elevation, dtype="float64"))))
        messagebox.showinfo("Estimate Fixed Angle",
                            f"Fixed angle set to {new_fixed:.3f}° "
                            f"(median of valid elevation values).")

    # ── Geolocation (fix missing/incorrect radar lat/lon/alt) ──────────────────

    def _set_location_dialog(self):
        if self.volume is None:
            messagebox.showwarning("Set Radar Location", "No radar data loaded.")
            return
        vol = self.volume

        win = tk.Toplevel(self)
        win.title("Set Radar Location")
        win.resizable(False, False)
        win.grab_set()
        win.configure(bg=DLG_BG)

        pad = dict(padx=10, pady=4)

        status = ("Currently geolocated." if edit_geolocation.is_geolocated(vol)
                  else "Not geolocated (lat/lon/alt are 0,0,0) — set the real "
                       "site location below.")
        tk.Label(win, text=status, bg=DLG_BG, fg=DLG_HINT,
                 font=("TkDefaultFont", 8), wraplength=320, justify="left"
                 ).grid(row=0, column=0, columnspan=2, sticky="w", **pad)

        tk.Label(win, text="Latitude (°N):", bg=DLG_BG, fg=DLG_FG, anchor="w"
                 ).grid(row=1, column=0, sticky="w", **pad)
        lat_var = tk.StringVar(value=f"{vol.latitude:.6f}")
        tk.Entry(win, textvariable=lat_var, width=14, bg=DLG_ENTRY
                 ).grid(row=1, column=1, sticky="w", **pad)

        tk.Label(win, text="Longitude (°E):", bg=DLG_BG, fg=DLG_FG, anchor="w"
                 ).grid(row=2, column=0, sticky="w", **pad)
        lon_var = tk.StringVar(value=f"{vol.longitude:.6f}")
        tk.Entry(win, textvariable=lon_var, width=14, bg=DLG_ENTRY
                 ).grid(row=2, column=1, sticky="w", **pad)

        tk.Label(win, text="Altitude (m MSL):", bg=DLG_BG, fg=DLG_FG, anchor="w"
                 ).grid(row=3, column=0, sticky="w", **pad)
        alt_var = tk.StringVar(value=f"{vol.altitude:.2f}")
        tk.Entry(win, textvariable=alt_var, width=14, bg=DLG_ENTRY
                 ).grid(row=3, column=1, sticky="w", **pad)

        tk.Label(win, text="Use negative values for South / West.\n"
                            "This sets the radar site location for the whole "
                            "volume (used by CfRadial / NEXRAD export).\n"
                            "Not covered by Ctrl+Z.",
                 bg=DLG_BG, fg=DLG_HINT2, font=("TkDefaultFont", 8),
                 wraplength=320, justify="left"
                 ).grid(row=4, column=0, columnspan=2, sticky="w", padx=10)

        def ok():
            try:
                lat = float(lat_var.get())
                lon = float(lon_var.get())
                alt = float(alt_var.get())
            except ValueError:
                messagebox.showwarning("Set Radar Location",
                                       "Latitude, longitude, and altitude "
                                       "must be numbers.", parent=win)
                return
            win.destroy()
            self._do_set_location(lat, lon, alt)

        btns = tk.Frame(win, bg=DLG_BG)
        btns.grid(row=5, column=0, columnspan=2, pady=10)
        tk.Button(btns, text="Apply", command=ok, width=10,
                  bg=DLG_BG, fg=DLG_FG, relief="raised").pack(side="left", padx=4)
        tk.Button(btns, text="Cancel", command=win.destroy, width=10,
                  bg=DLG_BG, fg=DLG_FG, relief="raised").pack(side="left", padx=4)

    def _do_set_location(self, lat, lon, alt):
        if self.volume is None:
            return
        try:
            edit_geolocation.set_location(self.volume, lat, lon, alt)
        except ValueError as e:
            messagebox.showwarning("Set Radar Location", str(e))
            return
        messagebox.showinfo("Set Radar Location",
                            f"Radar location set to "
                            f"{lat:.6f}°, {lon:.6f}°, {alt:.1f} m.")


    # ── CfRadial export stub (logic lives in main.py) ─────────────────────────

    def _open_cfrad_export(self, scope="sweep"):
        CfRadialExportDialog(self, initial_scope=scope)

    def _do_cfradial_export(self, out_dir, scope):
        """Override in main.py subclass or monkeypatch to implement export."""
        messagebox.showinfo("Export",
                            f"Export ({scope}) to:\n{out_dir}\n\n"
                            "(Implement _do_cfradial_export in main.py)")

    def _open_nexrad_export(self, scope="sweep"):
        CfRadialExportDialog(self, initial_scope=scope,
                            title="Export to NEXRAD (Msg31)",
                            file_ext=".ar2v",
                            on_export=self._do_nexrad_export)

    def _do_nexrad_export(self, out_dir, scope):
        """Override in main.py subclass or monkeypatch to implement export."""
        messagebox.showinfo("Export",
                            f"Export ({scope}) to:\n{out_dir}\n\n"
                            "(Implement _do_nexrad_export in main.py)")

    # ── Batch boundary edit (folder) stub (logic lives in main.py) ────────────

    def _open_batch_boundary_dialog(self):
        if not self._boundary_mode.get():
            messagebox.showinfo(
                "Batch Boundary Edit",
                "Enable \"Use boundary\" and draw a polygon on the active "
                "panel first.")
            return
        mode = self._editor_mode.get()
        if mode not in EDIT_BRUSH_MODES:
            messagebox.showinfo(
                "Batch Boundary Edit",
                "Pick one of the edit modes (Unfold, Deglitch, Gap Fill, "
                "Delete Gates, GateFilter) first.")
            return
        panel = self._active_panel
        points = panel.ppi._boundary_points if panel is not None else []
        if len(points) < 3:
            messagebox.showinfo(
                "Batch Boundary Edit",
                "Add at least 3 boundary points before running a batch "
                "edit.\n(Right-click removes the last point.)")
            return
        if not self.file_list:
            messagebox.showwarning(
                "Batch Boundary Edit",
                "No folder of files is loaded. Open a folder first "
                "(Ctrl+D).")
            return
        BatchBoundaryEditDialog(self)

    def _do_batch_boundary_edit(self, out_dir, out_format, new_field=None):
        """Override in main.py subclass or monkeypatch to implement the
        actual per-file load / edit / export loop."""
        messagebox.showinfo(
            "Batch Boundary Edit",
            f"Batch edit ({out_format}) to:\n{out_dir}\n\n"
            "(Implement _do_batch_boundary_edit in main.py)")

    # ── Deglitch helpers ──────────────────────────────────────────────────────

    def _get_deglitch_nyquist(self):
        """Return the short Nyquist to use as the deglitch step (2 * short_nyq).

        For staggered-PRF deglitching the *short* Nyquist is always the folding
        interval — the long Nyquist is used only as a hint (the local median of
        the brush acts as the long-Nyquist-unfolded velocity estimate).

        Priority:
          1. manual override  (deglitch_nyquist_mode == "manual")
          2. short Nyquist from stagger detection  (default / recommended)
          3. long  Nyquist from stagger detection  (if user selects "long" mode)
          4. fixed Nyquist (non-staggered radar)
          5. brush Nyquist setting / radar metadata fallback
        Returns float > 0 or None on failure.
        """
        mode = self._deglitch_nyquist_mode.get()
        if mode == "manual":
            v = self._deglitch_manual_nyq.get()
            return v if v > 0 else None
        elif mode == "short":
            v = self._stagger_short_nyq.get()
            if v > 0:
                return v
            # Fall through to fixed (non-staggered radar)
            v = self._stagger_fixed_nyq.get()
            if v > 0:
                return v
        elif mode == "long":
            v = self._stagger_long_nyq.get()
            if v > 0:
                return v
            v = self._stagger_fixed_nyq.get()
            if v > 0:
                return v
        else:  # "fixed" or unknown
            v = self._stagger_fixed_nyq.get()
            if v > 0:
                return v
        # Final fallback: brush Nyquist widget or radar instrument_parameters
        v = self._brush_nyquist.get()
        if v > 0:
            return v
        return None

    # ── Undo / brush stubs ────────────────────────────────────────────────────

    def _push_undo(self, field_names=(), extra=None):
        """Save current field data (and optionally azimuth/elevation/
        fixed_angle) for the current sweep onto the undo stack.

        `extra` may include any of 'azimuth', 'elevation', 'fixed_angle' —
        used by Rotate Sweep / Offset Elevations / Estimate Fixed Angle so
        Ctrl+Z can undo those too, not just brush edits.
        """
        if self.volume is None:
            return
        si = self.sweep_index.get()
        if not (0 <= si < len(self.volume.sweeps)):
            return
        sw = self.volume.sweeps[si]
        snapshot = {
            "sweep_index": si,
            "fields": {fn: sw.fields[fn].copy()
                       for fn in field_names if fn in sw.fields},
        }
        if extra:
            if "azimuth" in extra and sw.azimuth is not None:
                snapshot["azimuth"] = np.asarray(sw.azimuth).copy()
            if "elevation" in extra and sw.elevation is not None:
                snapshot["elevation"] = np.asarray(sw.elevation).copy()
            if "fixed_angle" in extra:
                snapshot["fixed_angle"] = sw.fixed_angle
        self._undo_stack.append(snapshot)
        if len(self._undo_stack) > self._undo_stack_max:
            self._undo_stack.pop(0)
        # A fresh edit invalidates any previously-undone redo history.
        self._redo_stack.clear()

    def _push_undo_batch(self, snapshots):
        """Push a list of per-sweep snapshots (same shape as one
        _push_undo() snapshot each) as a single grouped undo entry, so
        one Ctrl+Z undoes every sweep a multi-sweep edit touched instead
        of just the last one. Used by the boundary tool's "apply to all
        sweeps" option (see brushes.apply_boundary_all_sweeps)."""
        if not snapshots:
            return
        self._undo_stack.append(list(snapshots))
        if len(self._undo_stack) > self._undo_stack_max:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def _restore_entry(self, entry):
        """Restore either a single undo snapshot (a dict) or a grouped
        batch of them (a list, from _push_undo_batch), returning the
        matching inverse (dict or list) for the opposite stack."""
        if isinstance(entry, list):
            inverses = [self._restore_snapshot(s) for s in entry]
            inverses = [i for i in inverses if i is not None]
            return inverses or None
        return self._restore_snapshot(entry)

    def _undo_brush(self):
        """Restore the most recent undo snapshot (or grouped batch)."""
        if not self._undo_stack:
            return
        entry = self._undo_stack.pop()
        inverse = self._restore_entry(entry)
        if inverse is not None:
            self._redo_stack.append(inverse)
            if len(self._redo_stack) > self._undo_stack_max:
                self._redo_stack.pop(0)

    def _redo_brush(self):
        """Re-apply the most recently undone snapshot (or grouped batch)."""
        if not self._redo_stack:
            return
        entry = self._redo_stack.pop()
        inverse = self._restore_entry(entry)
        if inverse is not None:
            self._undo_stack.append(inverse)
            if len(self._undo_stack) > self._undo_stack_max:
                self._undo_stack.pop(0)

    def _record_brush_touch(self, fname, ray_idx, gate_idx):
        """Accumulate the exact gate indices touched by the in-progress
        brush stroke onto the most recent undo snapshot. This lets
        undo/redo recolour precisely those gates afterward instead of
        diffing (or rebuilding) the whole field — mirrors the same
        ai_arr/gi_arr bookkeeping brushes.apply_brush already does for
        its own live painting."""
        if not self._undo_stack:
            return
        touched = self._undo_stack[-1].setdefault("touched", {})
        ray_idx  = np.asarray(ray_idx, dtype=np.intp)
        gate_idx = np.asarray(gate_idx, dtype=np.intp)
        prev = touched.get(fname)
        if prev is None:
            touched[fname] = (ray_idx.copy(), gate_idx.copy())
        else:
            touched[fname] = (np.concatenate([prev[0], ray_idx]),
                               np.concatenate([prev[1], gate_idx]))

    def _restore_snapshot(self, snapshot):
        """Apply a saved snapshot (from either the undo or redo stack) to
        the volume, returning a snapshot of the state *before* restoring
        so the caller can push it onto the opposite stack.

        If the snapshot carries recorded "touched" gate indices (set by
        `_record_brush_touch` while a brush stroke was applied), only
        those exact gates are recoloured via `patch_gate_colors` — no
        mesh rebuild, no whole-array diffing. Edits that don't know their
        own footprint in advance (Duplicate Field, Estimate Fixed Angle,
        Rotate Sweep, etc.) fall back to the previous full-refresh
        behaviour, which is unchanged for those. Azimuth/elevation
        changes always need a full geometry rebuild, since those move
        vertex positions, not just colour.
        """
        if self.volume is None:
            return None
        si = snapshot.get("sweep_index", self.sweep_index.get())
        if not (0 <= si < len(self.volume.sweeps)):
            return None
        sw = self.volume.sweeps[si]

        inverse = {"sweep_index": si, "fields": {}}
        touched_in = snapshot.get("touched", {})
        inverse_touched = {}
        changed_fields = []   # (fname, ray_idx, gate_idx, new_arr)
        needs_full_refresh = False

        for fn, new_arr in snapshot.get("fields", {}).items():
            old_arr = sw.fields.get(fn)
            if old_arr is not None:
                inverse["fields"][fn] = old_arr.copy()
            sw.fields[fn] = new_arr
            self._invalidate_render_cache(sweep_index=si, field_name=fn)

            touch = touched_in.get(fn)
            if touch is not None and len(touch[0]):
                inverse_touched[fn] = touch
                changed_fields.append((fn, touch[0], touch[1], new_arr))
            else:
                # No recorded per-gate footprint for this field (a
                # whole-field edit, or an older code path) -> can't know
                # cheaply which gates changed, so fall back to a full
                # refresh rather than diffing the entire array.
                needs_full_refresh = True

        if inverse_touched:
            inverse["touched"] = inverse_touched

        geometry_changed = False
        if "azimuth" in snapshot:
            inverse["azimuth"] = np.asarray(sw.azimuth).copy()
            sw.azimuth = snapshot["azimuth"]
            geometry_changed = True
        if "elevation" in snapshot:
            inverse["elevation"] = np.asarray(sw.elevation).copy()
            sw.elevation = snapshot["elevation"]
            geometry_changed = True
        if "fixed_angle" in snapshot:
            inverse["fixed_angle"] = sw.fixed_angle
            sw.fixed_angle = snapshot["fixed_angle"]

        if si == self.sweep_index.get():
            if geometry_changed:
                # Azimuth/elevation changed -> geometry must be rebuilt,
                # not just recoloured.
                self._geometry_cache.clear()
                for panel in self.panels:
                    panel.ppi._verts = None
                self._schedule_refresh()
            elif needs_full_refresh:
                self._schedule_refresh()
            elif changed_fields:
                self._patch_changed_fields(changed_fields)

        return inverse

    def _patch_changed_fields(self, changed_fields):
        """Recolour only the gates touched by an undo/redo on every panel
        currently displaying one of the affected fields, without rebuilding
        geometry — mirrors the brush fast-path in brushes.apply_brush.

        The GPU-side colour buffers are updated synchronously here (cheap),
        but the actual on-screen repaint is deferred via
        `_schedule_patch_flush` rather than fired immediately. Ctrl+Z/Ctrl+Y
        auto-repeat when held down, so a single keypress-and-hold can call
        this many times per second; painting each intermediate step to the
        screen looks like every undone edit briefly reappearing/flickering
        by instead of a single clean jump to the final state.
        """
        try:
            for panel in self.panels:
                fi = panel.current_field_info()
                if fi is None:
                    continue
                pf_name, _, vmin, vmax, cmap_name = fi
                for fn, ray_idx, gate_idx, new_arr in changed_fields:
                    if fn != pf_name:
                        continue
                    cmap = resolve_cmap(cmap_name)
                    # patch_vals must be flat, one value per (ray, gate)
                    # pair in ray_idx/gate_idx — NOT the full 2-D field.
                    patch_vals = np.ma.filled(
                        new_arr[ray_idx, gate_idx], np.nan
                    ).astype("float32", copy=False)
                    panel.ppi.patch_gate_colors(
                        ray_idx, gate_idx, patch_vals, new_arr.shape[1],
                        cmap, vmin, vmax, draw=False)
                    self._schedule_patch_flush(panel)
        except Exception:
            # Fall back to a full refresh rather than leaving stale colours.
            self._schedule_refresh()

    def _schedule_patch_flush(self, panel):
        """Debounce the actual VisPy repaint for `_patch_changed_fields`.

        Coalesces bursts of undo/redo steps (e.g. Ctrl+Z auto-repeating
        while held down) into a single repaint on the next idle tick,
        instead of one repaint per keystroke."""
        self._pending_patch_panels.add(panel)
        if self._patch_flush_id is not None:
            self.after_cancel(self._patch_flush_id)
        self._patch_flush_id = self.after_idle(self._flush_patch_draws)

    def _flush_patch_draws(self):
        self._patch_flush_id = None
        panels, self._pending_patch_panels = self._pending_patch_panels, set()
        for panel in panels:
            try:
                panel.ppi._vcanvas.update()
            except Exception:
                pass

    # ── Save image ────────────────────────────────────────────────────────────

    def save_image(self):
        path = filedialog.asksaveasfilename(
            title="Save Image",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("All", "*.*")],
        )
        if not path:
            return
        if not PIL_AVAILABLE:
            messagebox.showerror("PIL missing",
                                 "Install Pillow: pip install Pillow")
            return
        try:
            rgba = self.ppi.screenshot()
            PILImage.fromarray(rgba).save(path)
        except Exception as e:
            messagebox.showerror("Save Error", str(e))