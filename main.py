#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# GURT - Copyright (C) the GURT authors.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the
# Free Software Foundation. It is distributed WITHOUT ANY WARRANTY; see the
# file LICENSE for details.
"""
GURT Vispy — Radar viewer using VisPy for GPU-accelerated PPI rendering.

Supports (all reading routed through readers.detect_format + the readers/
package — no pyart dependency):
  - DORADE sweepfiles          (readers/dorade_reader.py)
  - GRIB2 (JMA radar)          (readers/grib2_reader.py)
  - CfRadial / CF-NetCDF       (readers/cfradial_reader.py)
  - ODIM_H5 (EUMETNET OPERA)   (readers/odim_h5_reader.py)
  - UKMO polar files           (readers/ukmo_reader.py)
  - NEXRAD Level-II MSG31/MSG1 (readers/nexrad_level2.py)

Rendering stack:
  - VisPy SceneCanvas (OpenGL) embedded in a Tkinter frame
  - Matplotlib colorbar strip (tiny figure, below the canvas)
  - Tkinter for menus, status bar, field controls

Controls:
  Left/Right arrows  — previous/next file in folder
  Up/Down arrows     — previous/next sweep
  Scroll wheel       — zoom in/out at cursor
  Left-drag          — pan
  Right-click        — colormap / reset zoom / param editor menu
  Escape             — reset zoom
  Ctrl+O             — open file
  Ctrl+D             — open folder
  Ctrl+S             — save image
  Ctrl+Z             — undo last brush edit
  Ctrl+Y             — redo last undone edit

Dependencies:
  pip install vispy pyopengltk netCDF4 matplotlib numpy pillow
"""

import os
from datetime import datetime, timezone

import numpy as np

# Format sniffing lives in readers/detect_format.py but isn't re-exported
# from readers/__init__.py yet, so it's pulled from the submodule directly.
from readers.detect_format import detect_filetype
from readers import (
    DoradeFile,
    Grib2File,
    read_cfradial,
    read_odim_h5, is_odim_h5,
    read_ukmo, is_ukmo,
    read_nexrad_level2, is_nexrad_level2,
)
from exporters import (
    NexradMsg31Writer, Sweep as NexradSweep, Ray as NexradRay,
    write_cfradial, write_dorade,
)

import os as _os, sys as _sys
_GUI_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "gui")   # brushes.py, gurt_gui.py ...
if _GUI_DIR not in _sys.path:
    _sys.path.insert(0, _GUI_DIR)

import brushes

# ── GUI module (importable) ───────────────────────────────────────────────────
from gurt_gui import (
    GURTVispy,
    PPICanvas,
    ParamColorDialog,
    EditorPanel,
    DuplicateFieldDialog,
    CfRadialExportDialog,
    FIELD_DEFAULTS,
    GURT_CMAPS,
    ALL_CMAP_NAMES,
    resolve_cmap,
    FILE_EXTENSIONS,
    _DORADE_NAME_PREFIXES,
)


# Data model and file loaders live in gurt_core.py (GUI-free, shared with gurt_cli.py).
from gurt_core import SweepData, RadarVolume, _load_dorade, _load_grib2, load_radar_file

# ══════════════════════════════════════════════════════════════════════════════
# Volume cache + background prefetch
# ══════════════════════════════════════════════════════════════════════════════

import threading
import sys

_CACHE_MAX_BYTES   = 512 * 1024 * 1024   # 512 MB ceiling
_CACHE_MAX_ENTRIES = 8                    # never keep more than 8 volumes
_PREFETCH_AHEAD    = 3                    # files to prefetch in each direction


def _volume_bytes(vol: RadarVolume) -> int:
    """Rough in-memory size estimate for a RadarVolume."""
    total = 0
    for sw in vol.sweeps:
        for arr in sw.fields.values():
            try:
                total += arr.nbytes
            except AttributeError:
                pass
        for attr in ("azimuth", "elevation", "ranges"):
            a = getattr(sw, attr, None)
            if a is not None:
                try:
                    total += a.nbytes
                except AttributeError:
                    pass
    return max(total, 1)


class VolumeCache:
    """
    LRU cache for RadarVolume objects with a hard memory ceiling.

    Thread-safe: background prefetch threads write through the same lock
    as the foreground loader.
    """

    def __init__(self, load_fn, max_bytes=_CACHE_MAX_BYTES,
                 max_entries=_CACHE_MAX_ENTRIES):
        self._load_fn     = load_fn
        self._max_bytes   = max_bytes
        self._max_entries = max_entries
        self._cache       = {}           # path -> RadarVolume
        self._order       = []           # LRU order (most-recent last)
        self._bytes_used  = 0
        self._lock        = threading.Lock()
        self._inflight    = set()        # paths currently being fetched
        self._queue       = []           # paths waiting for the prefetch worker
        self._worker      = None

    # ── public ────────────────────────────────────────────────────────────────

    def get(self, path: str) -> RadarVolume:
        """Return volume for *path*, loading synchronously if not cached."""
        with self._lock:
            if path in self._cache:
                self._touch(path)
                return self._cache[path]
        # Not in cache — load on the calling thread (foreground)
        vol = self._load_fn(path)
        with self._lock:
            self._insert(path, vol)
        return vol

    def prefetch(self, paths):
        """
        Fire-and-forget background load for each path not already cached. One worker thread
        drains a queue (file readers are serialized by readers.nclock anyway, and a thread per
        file of a large folder just piles up memory); a new call replaces what is still queued.
        """
        with self._lock:
            self._queue = [p for p in paths if p not in self._cache and p not in self._inflight]
            if self._worker is not None and self._worker.is_alive():
                return
            self._worker = threading.Thread(target=self._drain, daemon=True, name="prefetch")
            self._worker.start()

    def _drain(self):
        while True:
            with self._lock:
                while self._queue and (self._queue[0] in self._cache or self._queue[0] in self._inflight):
                    self._queue.pop(0)
                if not self._queue:
                    self._worker = None
                    return
                path = self._queue.pop(0)
                self._inflight.add(path)
            self._bg_load(path)

    def invalidate(self, path: str):
        """Remove a single entry (e.g. after in-place edit & save)."""
        with self._lock:
            if path in self._cache:
                self._bytes_used -= _volume_bytes(self._cache.pop(path))
                try:
                    self._order.remove(path)
                except ValueError:
                    pass

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._order.clear()
            self._bytes_used = 0

    @property
    def stats(self):
        with self._lock:
            return {
                "entries":    len(self._cache),
                "bytes_used": self._bytes_used,
                "max_bytes":  self._max_bytes,
            }

    # ── internal ──────────────────────────────────────────────────────────────

    def _bg_load(self, path: str):
        try:
            vol = self._load_fn(path)
        except Exception:
            return          # silently skip; foreground will show the error
        finally:
            with self._lock:
                self._inflight.discard(path)
        with self._lock:
            # Only insert if still needed (foreground may have loaded it already)
            if path not in self._cache:
                self._insert(path, vol)

    def _touch(self, path: str):
        """Move *path* to most-recently-used end (call with lock held)."""
        try:
            self._order.remove(path)
        except ValueError:
            pass
        self._order.append(path)

    def _insert(self, path: str, vol: RadarVolume):
        """Insert *vol* and evict LRU entries as needed (call with lock held)."""
        size = _volume_bytes(vol)
        # Evict until we have room for the new entry
        while self._order and (
            self._bytes_used + size > self._max_bytes
            or len(self._cache) >= self._max_entries
        ):
            lru_path = self._order.pop(0)
            old = self._cache.pop(lru_path, None)
            if old is not None:
                self._bytes_used -= _volume_bytes(old)
        self._cache[path]  = vol
        self._order.append(path)
        self._bytes_used  += size

# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="GURT Vispy — GPU-accelerated radar viewer")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--folder", "-d", metavar="DIR")
    grp.add_argument("--file",   "-f", metavar="FILE")
    parser.add_argument("--cache-mb", type=int, default=512,
                        metavar="MB",
                        help="Volume cache ceiling in MB (default: 512)")
    args = parser.parse_args()

    cache = VolumeCache(
        load_fn    = load_radar_file,
        max_bytes  = args.cache_mb * 1024 * 1024,
    )

    app = GURTVispy(
        folder=args.folder,
        file=args.file,
        load_fn=cache.get,
    )

    # ── Hook load_current_file to trigger prefetch after every navigation ──────
    _orig_load = app.load_current_file

    def _patched_load():
        _orig_load()
        # After a successful load, schedule prefetch of adjacent files
        fl = app.file_list
        fi = app.file_index
        if len(fl) < 2:
            return
        candidates = []
        for delta in range(1, _PREFETCH_AHEAD + 1):
            candidates.append(fl[(fi + delta) % len(fl)])
            candidates.append(fl[(fi - delta) % len(fl)])
        cache.prefetch(candidates)

    app.load_current_file = _patched_load

    # ── CfRadial export: wire the real implementation onto the GUI's stub ──────
    def _set_status(msg):
        try:
            app.file_var.set(msg)
            app.update_idletasks()
        except Exception:
            pass

    def _do_cfradial_export(out_dir, scope):
        def _export(vol, out_d, sweep_indices=None):
            return write_cfradial(vol, out_d, sweep_indices=sweep_indices)

        from tkinter import messagebox
        try:
            os.makedirs(out_dir, exist_ok=True)

            if scope == "sweep":
                if app.volume is None:
                    messagebox.showinfo("Export CfRadial", "No radar file loaded.")
                    return
                si   = app.sweep_index.get()
                dest = _export(app.volume, out_dir, sweep_indices=[si])
                messagebox.showinfo("Export CfRadial", f"Saved sweep {si}:\n{dest}")

            elif scope == "volume":
                if app.volume is None:
                    messagebox.showinfo("Export CfRadial", "No radar file loaded.")
                    return
                dest = _export(app.volume, out_dir)
                messagebox.showinfo("Export CfRadial", f"Saved volume:\n{dest}")

            elif scope == "all":
                if not app.file_list:
                    messagebox.showwarning("Export CfRadial", "No files loaded.")
                    return
                orig_idx = app.file_index
                saved, errors = [], []
                for i, fpath in enumerate(app.file_list):
                    try:
                        _set_status(f"Exporting {i+1}/{len(app.file_list)}: "
                                    f"{os.path.basename(fpath)}…")
                        v    = cache.get(fpath)
                        dest = _export(v, out_dir)
                        saved.append(dest)
                    except Exception as e:
                        errors.append(f"{os.path.basename(fpath)}: {e}")
                app.file_index = orig_idx
                app.load_current_file()
                msg = f"Exported {len(saved)} of {len(app.file_list)} files to:\n{out_dir}"
                if errors:
                    msg += f"\n\nErrors ({len(errors)}):\n" + "\n".join(errors[:5])
                    if len(errors) > 5:
                        msg += f"\n… and {len(errors)-5} more"
                messagebox.showinfo("Export CfRadial — Done", msg)
        except Exception as e:
            messagebox.showerror("Export CfRadial", f"Export failed:\n{e}")
        finally:
            _set_status("")

    app._do_cfradial_export = _do_cfradial_export

    def _do_dorade_export(out_dir, scope):
        """One DORADE sweepfile per sweep (swp.<time>.<radar>.<msec>.<angle>_<scan>_v<vol>)."""
        from tkinter import messagebox
        compat = "dreader" if app._dorade_dreader.get() else "standard"
        try:
            os.makedirs(out_dir, exist_ok=True)
            if scope in ("sweep", "volume"):
                if app.volume is None:
                    messagebox.showinfo("Export DORADE", "No radar file loaded.")
                    return
                idx = [app.sweep_index.get()] if scope == "sweep" else None
                paths = write_dorade(app.volume, out_dir, sweep_indices=idx, compat=compat)
                messagebox.showinfo("Export DORADE", f"Wrote {len(paths)} sweepfile(s) to:\n{out_dir}")
            else:
                if not app.file_list:
                    messagebox.showwarning("Export DORADE", "No files loaded.")
                    return
                orig_idx = app.file_index
                n_files, errors = 0, []
                for i, fpath in enumerate(app.file_list):
                    try:
                        _set_status(f"Exporting {i+1}/{len(app.file_list)}: {os.path.basename(fpath)}…")
                        write_dorade(cache.get(fpath), out_dir, volume_number=i + 1, compat=compat)
                        n_files += 1
                    except Exception as e:
                        errors.append(f"{os.path.basename(fpath)}: {e}")
                app.file_index = orig_idx
                app.load_current_file()
                msg = f"Exported {n_files} of {len(app.file_list)} files to:\n{out_dir}"
                if errors:
                    msg += f"\n\nErrors ({len(errors)}):\n" + "\n".join(errors[:5])
                messagebox.showinfo("Export DORADE — Done", msg)
        except Exception as e:
            messagebox.showerror("Export DORADE", f"Export failed:\n{e}")
        finally:
            _set_status("")

    app._do_dorade_export = _do_dorade_export

    # ── NEXRAD (Msg31) export: wire the real implementation onto the GUI's stub ──

    def _sweep_to_nexrad_sweep(sw, vol):
        """Convert one SweepData (+ its parent VolumeData) into a
        nexrad_writer.Sweep of Ray objects. Field selection (one VEL,
        one REF) is handled automatically inside nexrad_writer when the
        file is written — we just pass through whatever fields exist."""
        ip = getattr(vol, "instrument_parameters", None) or {}

        def _ip_scalar(key, default=0.0):
            try:
                return float(np.asarray(ip[key]["data"]).flat[0])
            except (KeyError, IndexError, TypeError, ValueError):
                return default

        # Per-sweep instrument_parameters (populated per-cut by the reader,
        # from each cut's own nyquist_spd) must win over the volume-level
        # dict -- vol.instrument_parameters only ever holds a single
        # frequency entry, so reading nyquist from it here silently fell
        # back to the 29.1 default for every sweep in the file.
        sw_ip = getattr(sw, "instrument_parameters", None) or {}

        def _sw_ip_scalar(key, default=None):
            try:
                return float(np.asarray(sw_ip[key]["data"]).flat[0])
            except (KeyError, IndexError, TypeError, ValueError):
                return default

        nyquist = _sw_ip_scalar("nyquist_velocity") or _ip_scalar("nyquist_velocity", 29.1) or 29.1
        unamb_km = float(getattr(vol, "unambiguous_range", 0.0) or 0.0) / 1000.0 or 230.0

        ranges     = sw.ranges
        start_range_m = int(round(float(ranges[0]))) if ranges is not None and len(ranges) else 0
        gate_width_m  = (int(round(float(ranges[1] - ranges[0])))
                         if ranges is not None and len(ranges) > 1 else 250)

        # Per-ray unix timestamps, if the reader exposed them (see
        # the reader's sw.ray_time). Falls back to one
        # timestamp per sweep only for readers that don't provide per-ray
        # times -- previously this fallback was the *only* path, so every
        # ray in every sweep shared one timestamp.
        ray_times = getattr(sw, "ray_time", None)
        has_ray_times = ray_times is not None and len(ray_times) > 0

        scan_time = sw.scan_time or vol.scan_time or datetime.utcnow()
        fallback_time_unix = scan_time.replace(tzinfo=timezone.utc).timestamp() \
            if scan_time.tzinfo is None else scan_time.timestamp()

        rf_by_field = getattr(sw, "range_folded", None) or {}

        # nexrad_writer.FIELD_PARAMS only recognizes the exact keys REF,
        # VEL, SW, ZDR, PHI, RHO, CFP -- anything else is silently dropped
        # (ray.fields.get(fname) just returns None). Readers across our
        # formats don't agree on spelling (see gurt_gui.FIELD_DEFAULTS for
        # the full alias zoo), so normalize here rather than relying on
        # every field to already be spelled the NEXRAD way.
        _NEXRAD_FIELD_ALIASES = {
            "REF": ("REF", "DBZH", "DBZ", "DBZHC", "DBZVC_F", "DBZVC",
                    "DBMHC", "reflectivity"),
            "VEL": ("VEL", "VE", "VU", "VELH", "velocity"),
            "SW":  ("SW", "WIDTH", "width", "spectrum_width"),
            "ZDR": ("ZDR",),
            "PHI": ("PHI", "PHIDP", "differential_phase"),
            "RHO": ("RHO", "RHOHV"),
            "CFP": ("CFP",),
        }
        _ALIAS_TO_CANON = {alias: canon
                           for canon, aliases in _NEXRAD_FIELD_ALIASES.items()
                           for alias in aliases}

        def _canon_fields(d):
            out = {}
            for fname, arr in d.items():
                canon = _ALIAS_TO_CANON.get(fname)
                if canon is None:
                    continue  # not one of the 7 NEXRAD moments -- drop
                if canon in out:
                    continue  # already have this moment from another alias
                out[canon] = arr
            return out

        rays = []
        nrays = len(sw.azimuth) if sw.azimuth is not None else 0
        for i in range(nrays):
            ray_fields = _canon_fields({
                fname: arr[i] for fname, arr in sw.fields.items()
                if arr is not None and i < arr.shape[0]
            })
            ray_rf = _canon_fields({
                fname: rfarr[i] for fname, rfarr in rf_by_field.items()
                if rfarr is not None and i < rfarr.shape[0] and rfarr[i].any()
            })
            if has_ray_times and i < len(ray_times):
                time_unix = float(ray_times[i])
            else:
                time_unix = fallback_time_unix
            rays.append(NexradRay(
                azimuth=float(sw.azimuth[i]),
                elevation=float(sw.elevation[i]) if sw.elevation is not None else float(sw.fixed_angle),
                time_unix=time_unix,
                start_range_m=start_range_m,
                gate_width_m=gate_width_m,
                fields=ray_fields,
                nyquist_mps=nyquist,
                unamb_range_km=unamb_km,
                range_folded=ray_rf or None,
            ))

        return NexradSweep(fixed_elevation=float(sw.fixed_angle), rays=rays)

    def _write_nexrad_volume(vol, out_dir, sweep_indices=None):
        """Write *vol* out as one NEXRAD Msg31 (.ar2v) file in *out_dir*.
        Hoisted out of _do_nexrad_export (out_dir passed explicitly
        rather than captured via closure) so _do_batch_boundary_edit can
        reuse it too."""
        idxs = sweep_indices if sweep_indices is not None else range(len(vol.sweeps))
        nexrad_sweeps = [_sweep_to_nexrad_sweep(vol.sweeps[i], vol) for i in idxs]
        site_id = (vol.radar_name or "KXXX")[:4].upper().ljust(4)
        freq_hz = 0.0
        ip = getattr(vol, "instrument_parameters", None) or {}
        if "frequency" in ip:
            try:
                freq_hz = float(np.asarray(ip["frequency"]["data"]).flat[0])
            except (KeyError, IndexError, TypeError, ValueError):
                freq_hz = 0.0
        writer = NexradMsg31Writer(
            site_id=site_id,
            latitude=vol.latitude,
            longitude=vol.longitude,
            height_m=int(round(vol.altitude)),
            frequency_hz=freq_hz,
        )
        fname = f"{site_id.strip()}_{(vol.scan_time or datetime.utcnow()):%Y%m%d_%H%M%S}.ar2v"
        dest = os.path.join(out_dir, fname)
        writer.write(dest, nexrad_sweeps)
        return dest

    def _do_nexrad_export(out_dir, scope):
        from tkinter import messagebox
        try:
            os.makedirs(out_dir, exist_ok=True)

            if scope == "sweep":
                if app.volume is None:
                    messagebox.showinfo("Export NEXRAD", "No radar file loaded.")
                    return
                si   = app.sweep_index.get()
                dest = _write_nexrad_volume(app.volume, out_dir, sweep_indices=[si])
                messagebox.showinfo("Export NEXRAD", f"Saved sweep {si}:\n{dest}")

            elif scope == "volume":
                if app.volume is None:
                    messagebox.showinfo("Export NEXRAD", "No radar file loaded.")
                    return
                dest = _write_nexrad_volume(app.volume, out_dir)
                messagebox.showinfo("Export NEXRAD", f"Saved volume:\n{dest}")

            elif scope == "all":
                if not app.file_list:
                    messagebox.showwarning("Export NEXRAD", "No files loaded.")
                    return
                orig_idx = app.file_index
                saved, errors = [], []
                for i, fpath in enumerate(app.file_list):
                    try:
                        _set_status(f"Exporting {i+1}/{len(app.file_list)}: "
                                    f"{os.path.basename(fpath)}…")
                        v    = cache.get(fpath)
                        dest = _write_nexrad_volume(v, out_dir)
                        saved.append(dest)
                    except Exception as e:
                        errors.append(f"{os.path.basename(fpath)}: {e}")
                app.file_index = orig_idx
                app.load_current_file()
                msg = f"Exported {len(saved)} of {len(app.file_list)} files to:\n{out_dir}"
                if errors:
                    msg += f"\n\nErrors ({len(errors)}):\n" + "\n".join(errors[:5])
                    if len(errors) > 5:
                        msg += f"\n… and {len(errors)-5} more"
                messagebox.showinfo("Export NEXRAD — Done", msg)
        except Exception as e:
            messagebox.showerror("Export NEXRAD", f"Export failed:\n{e}")
        finally:
            _set_status("")

    app._do_nexrad_export = _do_nexrad_export

    # ── Batch boundary edit (folder): wire the real implementation onto the ────
    # GUI's stub. Reuses whatever polygon + edit mode the user set up
    # interactively (see gurt_gui.BatchBoundaryEditDialog), loads every file in
    # the currently opened folder through the same cache used for navigation,
    # applies brushes.apply_boundary_to_volume() to each, and writes the result
    # out in the chosen format -- same per-file error collection pattern as
    # _do_cfradial_export/_do_nexrad_export above.

    def _do_batch_boundary_edit(out_dir, out_format, new_field=None):
        """*new_field*: if set, edits are written to a copy of the active
        field under that name (original left untouched). CfRadial only --
        the NEXRAD writer can't store extra moments."""
        from tkinter import messagebox
        panel = app._active_panel
        if panel is None:
            return
        points = list(panel.ppi._boundary_points)
        mode = app._editor_mode.get()
        fi = panel.current_field_info()
        if fi is None:
            messagebox.showinfo("Batch Boundary Edit",
                                "No field selected on the active panel.")
            return
        fname = fi[0]
        if new_field and out_format == "nexrad":
            messagebox.showinfo(
                "Batch Boundary Edit",
                "NEXRAD Level-II can't hold extra fields. Use CfRadial to "
                "save edits as a new field.")
            return
        if new_field and new_field == fname:
            messagebox.showinfo(
                "Batch Boundary Edit",
                "The new field name must differ from the field being "
                f"edited ({fname}).")
            return

        try:
            os.makedirs(out_dir, exist_ok=True)
            orig_idx = app.file_index
            saved, errors = [], []
            for i, fpath in enumerate(app.file_list):
                try:
                    _set_status(f"Editing {i+1}/{len(app.file_list)}: "
                                f"{os.path.basename(fpath)}…")
                    v = cache.get(fpath)
                    n_sweeps = brushes.apply_boundary_to_volume(
                        app, v, fname, points, mode, dest_fname=new_field)
                    if out_format == "nexrad":
                        dest = _write_nexrad_volume(v, out_dir)
                    else:
                        dest = write_cfradial(v, out_dir)
                    saved.append(
                        f"{os.path.basename(dest)}  ({n_sweeps} sweep(s) edited)")
                    cache.invalidate(fpath)
                except Exception as e:
                    errors.append(f"{os.path.basename(fpath)}: {e}")
            app.file_index = orig_idx
            app.load_current_file()
            msg = (f"Edited and saved {len(saved)} of {len(app.file_list)} "
                  f"files to:\n{out_dir}")
            if errors:
                msg += f"\n\nErrors ({len(errors)}):\n" + "\n".join(errors[:5])
                if len(errors) > 5:
                    msg += f"\n… and {len(errors)-5} more"
            messagebox.showinfo("Batch Boundary Edit — Done", msg)
        except Exception as e:
            messagebox.showerror("Batch Boundary Edit", f"Batch edit failed:\n{e}")
        finally:
            _set_status("")

    app._do_batch_boundary_edit = _do_batch_boundary_edit

    # ── Despeckle the whole folder -> CfRadial ─────────────────────────────────
    # Every PPI sweep of every file in the open folder is despeckled (DREADER
    # despeck_parm, libgurt when available), written to *dst* (a new field, so the
    # source field survives next to it), and each volume is saved as CfRadial in
    # out_dir. Source files are never modified; the cache entry is dropped after
    # each file so the in-memory copy is not left edited.

    def _do_batch_despeckle(fname, radius, min_good, max_pass, wrap, dst, out_dir):
        from tkinter import messagebox
        from edit import despeckle as _desp
        if not app.file_list:
            messagebox.showinfo("Despeckle folder", "No files loaded.")
            return
        try:
            os.makedirs(out_dir, exist_ok=True)
            orig_idx = app.file_index
            saved, errors = [], []
            for i, fpath in enumerate(app.file_list):
                try:
                    _set_status(f"Despeckling {i+1}/{len(app.file_list)}: "
                                f"{os.path.basename(fpath)}…")
                    v = cache.get(fpath)
                    nsw, removed = 0, 0
                    for sw in v.sweeps:
                        if fname not in sw.fields or getattr(sw, "scan_type", "ppi") == "rhi":
                            continue
                        new, n, _ = _desp.despeckle_field(sw.fields[fname], radius, min_good,
                                                          max_pass, wrap)
                        sw.fields[dst] = new
                        nsw += 1
                        removed += n
                    if nsw == 0:
                        raise ValueError(f"no sweep has field {fname}")
                    if dst not in v.field_names:
                        v.field_names.append(dst)
                    out = write_cfradial(v, out_dir)
                    saved.append(f"{os.path.basename(out)}  ({nsw} sweep(s), {removed} gates removed)")
                    cache.invalidate(fpath)
                except Exception as e:
                    errors.append(f"{os.path.basename(fpath)}: {e}")
            app.file_index = orig_idx
            app.load_current_file()
            msg = (f"Despeckled {fname} -> {dst} and saved {len(saved)} of "
                   f"{len(app.file_list)} files to:\n{out_dir}")
            if errors:
                msg += f"\n\nErrors ({len(errors)}):\n" + "\n".join(errors[:5])
                if len(errors) > 5:
                    msg += f"\n… and {len(errors)-5} more"
            messagebox.showinfo("Despeckle folder — Done", msg)
        except Exception as e:
            messagebox.showerror("Despeckle folder", f"Despeckle failed:\n{e}")
        finally:
            _set_status("")

    app._do_batch_despeckle = _do_batch_despeckle

    app.mainloop()


if __name__ == "__main__":
    main()