#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# GURT - Copyright (C) the GURT authors.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the
# Free Software Foundation. It is distributed WITHOUT ANY WARRANTY; see the
# file LICENSE for details.
"""
brushes.py — Brush + Boundary tools for GURT Vispy.

Holds the pure(-ish) logic for the PPI editor's five edit modes (Unfold,
Deglitch, Gap Fill, Delete Gates, GateFilter Brush), plus the
BrushThrottle helper used to limit how often a brush is re-applied while
dragging.

There are two ways to *select* which gates an edit mode runs on:
  • brush_gate_indices()    — a dragged circle (continuous, while painting)
  • boundary_gate_indices() — a clicked polygon (one-shot, on Apply Edit)
Both feed the same per-mode math in _compute_edit_values(), so "unfold via
brush" and "unfold via boundary" always behave identically — only the
selection shape differs. GateFilter Brush is no exception: it always
masks the field it's run on (fname/data below), same as every other
mode here — it just decides *which* of the selected gates to mask by
evaluating a compound condition expression (from the Editor Panel)
against the sweep, instead of doing folding/median math.

The boundary polygon can also be applied beyond just the active sweep:
  • apply_boundary()             — active sweep only (in-app, undoable)
  • apply_boundary_all_sweeps()  — every sweep of the loaded volume
                                    (in-app, one grouped undo entry)
  • apply_boundary_to_volume()   — every sweep of an arbitrary volume,
                                    headless (used by the folder batch-
                                    edit path in main.py, no undo)
All three recompute boundary_gate_indices() per sweep, since each sweep
has its own azimuth/range geometry even though the polygon is drawn once
in shared (x_km, y_km) scene coordinates.

gurt_gui.py keeps thin wrapper methods that simply delegate to the
functions below, so the edit math lives in one place and the GUI module
stays focused on widgets/menus.
"""

import math
import time

import numpy as np
from tkinter import messagebox
from numba import njit

from edit.gatefilter import evaluate_conditions
from edit.deglitch import _run_core as _deglitch_run_core, gap_fill as _deglitch_gap_fill


class BrushThrottle:
    """Returns True at most once per interval_ms milliseconds."""
    def __init__(self, interval_ms: int = 33):
        self._interval_ms = interval_ms
        self._last_ts: float = 0.0

    def ready(self) -> bool:
        now = time.monotonic() * 1000.0
        if now - self._last_ts >= self._interval_ms:
            self._last_ts = now
            return True
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Gate selection — circular brush
# ══════════════════════════════════════════════════════════════════════════════

def brush_gate_indices(sweep, x_km: float, y_km: float, radius_km: float):
    """Return (ray_indices, gate_indices) for gates of *sweep* under the
    circular brush centred at (x_km, y_km) with radius radius_km.

    For PPI sweeps the canvas is (E km, N km) — x = r*sin(az), y = r*cos(az).
    For RHI sweeps the canvas is (horiz km, alt km) — x = r*cos(el), y = r*sin(el).
    scan_type is read from sweep.scan_type ("rhi" or "ppi").
    """
    is_rhi  = getattr(sweep, "scan_type", "ppi") == "rhi"
    ranges  = np.asarray(sweep.ranges, dtype="float32") / 1000.0
    if ranges.size == 0:
        return (np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp))

    center_r = math.hypot(x_km, y_km)

    # --- gate pre-filter: keep only gates within [center_r ± radius_km] ---
    gate_idx = np.flatnonzero(
        (ranges >= center_r - radius_km) &
        (ranges <= center_r + radius_km)
    )
    if gate_idx.size == 0:
        return (np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp))

    if is_rhi:
        # RHI: scanning angle is elevation
        el = np.asarray(sweep.elevation, dtype="float32")
        if el.size == 0:
            return (np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp))

        if center_r <= radius_km:
            ray_idx = np.arange(el.size, dtype=np.intp)
        else:
            # centre elevation from brush position
            center_el = math.degrees(math.atan2(y_km, x_km))
            ray_pad = 1.0
            if el.size > 1:
                ray_pad += float(np.nanmedian(np.abs(np.diff(np.sort(el)))))
            half_width = math.degrees(
                math.asin(min(1.0, radius_km / max(center_r, 1.0e-6)))
            ) + ray_pad
            el_diff = np.abs(el - center_el)
            ray_idx = np.flatnonzero(el_diff <= half_width)
            if ray_idx.size == 0:
                return (np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp))

        el_rad     = np.deg2rad(el[ray_idx])
        sub_ranges = ranges[gate_idx]
        # RHI gate positions: x = r*cos(el), y = r*sin(el)
        sub_gx = np.multiply.outer(np.cos(el_rad), sub_ranges)
        sub_gy = np.multiply.outer(np.sin(el_rad), sub_ranges)

    else:
        # PPI: scanning angle is azimuth
        az = np.asarray(sweep.azimuth, dtype="float32")
        if az.size == 0:
            return (np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp))

        if center_r <= radius_km:
            ray_idx = np.arange(az.size, dtype=np.intp)
        else:
            center_az = (math.degrees(math.atan2(x_km, y_km)) + 360.0) % 360.0
            ray_pad = 1.0
            if az.size > 1:
                ray_pad += float(np.nanmedian(np.abs(np.diff(np.sort(az)))))
            half_width = math.degrees(
                math.asin(min(1.0, radius_km / max(center_r, 1.0e-6)))
            ) + ray_pad
            az_diff = np.abs(((az - center_az + 180.0) % 360.0) - 180.0)
            ray_idx = np.flatnonzero(az_diff <= half_width)
            if ray_idx.size == 0:
                return (np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp))

        az_rad     = np.deg2rad(az[ray_idx])
        sub_ranges = ranges[gate_idx]
        # PPI gate positions: x = r*sin(az), y = r*cos(az)
        sub_gx = np.multiply.outer(np.sin(az_rad), sub_ranges)
        sub_gy = np.multiply.outer(np.cos(az_rad), sub_ranges)

    hits = np.nonzero((sub_gx - x_km) ** 2 + (sub_gy - y_km) ** 2
                      <= radius_km ** 2)
    if hits[0].size == 0:
        return (np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp))
    return ray_idx[hits[0]], gate_idx[hits[1]]


# ══════════════════════════════════════════════════════════════════════════════
# Gate selection — polygon boundary
# ══════════════════════════════════════════════════════════════════════════════

def _points_in_polygon(px: np.ndarray, py: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorized even-odd (ray-casting) point-in-polygon test.

    `px`/`py` are 1-D arrays of test-point coordinates; `poly` is an
    (N, 2) array of polygon vertices (does not need to repeat the first
    vertex at the end — the edge from the last vertex back to the first
    is added automatically)."""
    n = poly.shape[0]
    inside = np.zeros(px.shape, dtype=bool)
    x1, y1 = poly[-1, 0], poly[-1, 1]
    for i in range(n):
        x2, y2 = poly[i, 0], poly[i, 1]
        # +1e-15 sidesteps a divide-by-zero on horizontal edges; cond is
        # already False for those ((py<y1) == (py<y2)), so the perturbed
        # value it would otherwise multiply by never gets used anyway.
        xint = x1 + (py - y1) * (x2 - x1) / (y2 - y1 + 1.0e-15)
        cond = ((py < y1) != (py < y2)) & (px < xint)
        inside ^= cond
        x1, y1 = x2, y2
    return inside


def boundary_gate_indices(sweep, polygon_xy):
    """Return (ray_indices, gate_indices) for gates of *sweep* whose
    centre point falls inside the closed polygon described by
    *polygon_xy* (an iterable of (x_km, y_km) vertices; at least 3 are
    required — the polygon is closed automatically, no need to repeat
    the first vertex at the end).

    For PPI: gate positions are x = r*sin(az), y = r*cos(az).
    For RHI: gate positions are x = r*cos(el), y = r*sin(el).
    """
    poly = np.asarray(polygon_xy, dtype="float64")
    if poly.shape[0] < 3:
        return (np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp))

    is_rhi = getattr(sweep, "scan_type", "ppi") == "rhi"
    ranges = np.asarray(sweep.ranges, dtype="float32") / 1000.0
    if ranges.size == 0:
        return (np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp))

    xmin, ymin = poly.min(axis=0)
    xmax, ymax = poly.max(axis=0)

    max_r = math.hypot(max(abs(xmin), abs(xmax)), max(abs(ymin), abs(ymax)))
    range_idx = np.flatnonzero(ranges <= max_r + 1.0e-6)
    if range_idx.size == 0:
        return (np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp))

    sub_ranges = ranges[range_idx]

    if is_rhi:
        el = np.asarray(sweep.elevation, dtype="float32")
        if el.size == 0:
            return (np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp))
        el_rad = np.deg2rad(el)
        gx = np.multiply.outer(np.cos(el_rad), sub_ranges)
        gy = np.multiply.outer(np.sin(el_rad), sub_ranges)
    else:
        az = np.asarray(sweep.azimuth, dtype="float32")
        if az.size == 0:
            return (np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp))
        az_rad = np.deg2rad(az)
        gx = np.multiply.outer(np.sin(az_rad), sub_ranges)
        gy = np.multiply.outer(np.cos(az_rad), sub_ranges)

    in_bbox = (gx >= xmin) & (gx <= xmax) & (gy >= ymin) & (gy <= ymax)
    cand_ray, cand_sub = np.nonzero(in_bbox)
    if cand_ray.size == 0:
        return (np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp))

    inside = _points_in_polygon(gx[cand_ray, cand_sub], gy[cand_ray, cand_sub], poly)
    if not np.any(inside):
        return (np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp))

    return cand_ray[inside], range_idx[cand_sub[inside]]


# ══════════════════════════════════════════════════════════════════════════════
# Shared preconditions + edit math
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_panel_target(app, panel):
    """Shared precondition checks for both apply_brush and apply_boundary.
    Returns (sweep, field_name, data_array), or None if there's nothing
    editable right now (no volume loaded, no field selected on this
    panel, sweep index out of range, or the field isn't present on this
    sweep)."""
    if app.volume is None:
        return None
    fi = panel.current_field_info()
    if fi is None:
        return None
    fname = fi[0]
    si = app.sweep_index.get()
    if not (0 <= si < len(app.volume.sweeps)):
        return None
    sw = app.volume.sweeps[si]
    if fname not in sw.fields:
        return None
    return sw, fname, sw.fields[fname]


def _custom_algo_active(app) -> bool:
    """True only when the Boundary Tool is in use *and* the "Use custom
    deglitch algorithm" checkbox is on. The circular brush never uses
    the standalone deglitch.py algorithm -- it's whole-sweep and multi-
    pass, which doesn't make sense to re-run on every drag frame -- so
    this deliberately also gates on `app._boundary_mode` rather than
    just the checkbox's own value."""
    return bool(
        app._boundary_mode.get()
        and getattr(app, "_custom_deglitch_enabled", None) is not None
        and app._custom_deglitch_enabled.get()
    )


def _apply_custom_deglitch_gapfill(app, sw, fname, data, ai_arr, gi_arr, mode,
                                   warn: bool = True):
    """Boundary-only alternative to the built-in "deglitch_brush" /
    "gradient_brush" math: runs the standalone deglitch.py algorithm
    (deglitch._run_core, the numba port of qualitycontrol.c's
    deglitch_parm) over the *whole* sweep -- it needs full neighbourhood
    context to find each gate's local median -- and, for the Gap Fill
    mode, follows it with deglitch.gap_fill's multi-pass nearest-
    neighbour filler. Only the gates selected by the boundary polygon
    (ai_arr/gi_arr) are ever reported back as changed, matching the
    (changed_ai, changed_gi, changed_vals) contract every other branch
    of _compute_edit_values returns.

    Invalid/unresolved gates are flagged (masked) the same way
    run_deglitch.py's delete_bad=True does; whether they then get patched
    with a gap-fill pass or stay masked is controlled separately by
    `app._custom_gapfill_enabled` (off by default, matching
    run_deglitch.py's own gap_fill_enabled=False).
    """
    if len(ai_arr) == 0:
        return None

    nyq = app._get_deglitch_nyquist()
    if nyq is None or nyq <= 0:
        if warn:
            messagebox.showwarning(
                "Custom Deglitch",
                "Could not determine short Nyquist velocity.\n"
                "Set it manually in the Editor Panel → Deglitch Settings.")
        return None

    bad_val = -9999.0
    orig_mask = np.ma.getmaskarray(data).copy()
    filled = np.ma.filled(data, bad_val).astype(np.float64)
    nrays = filled.shape[0]
    nyquist_arr = np.full(nrays, float(nyq), dtype=np.float64)

    radius = max(1, int(app._custom_deglitch_radius.get()))
    thresh = float(app._custom_deglitch_thresh.get())

    # delete_bad=True lets the algorithm flag a glitch it can't
    # fold-correct (matches run_deglitch.py's own default). Whether that
    # flagged gate then gets patched by gap_fill or stays masked is a
    # separate, explicit choice below -- not tied to which brush mode
    # triggered this.
    corrected = _deglitch_run_core(
        filled, bad_val, nyquist_arr, radius, thresh,
        0,      # max_pass = 0 -> iterate until stable, like the C default
        True,   # delete_bad -- flag unresolved glitches
        numprt=1,
    )

    if app._custom_gapfill_enabled.get():
        # Patch every gate the deglitch pass flagged (and nothing that was
        # already missing beforehand -- orig_mask) with a neighbor-based
        # estimate instead of leaving it masked.
        result = np.ma.masked_where(
            np.abs(corrected - bad_val) <= 1.0e-5, corrected)
        result = _deglitch_gap_fill(
            result, orig_mask=orig_mask,
            n_neighbors=max(1, int(app._custom_gapfill_n_neighbors.get())),
            method=app._custom_gapfill_method.get(),
            n_passes=max(1, int(app._custom_gapfill_n_passes.get())))
        corrected = np.ma.filled(result, bad_val).astype(np.float64)

    new_full = np.ma.masked_where(
        np.abs(corrected - bad_val) <= 1.0e-5, corrected)

    sel_orig = np.ma.filled(data[ai_arr, gi_arr], np.nan).astype("float64", copy=False)
    sel_new = np.ma.filled(new_full[ai_arr, gi_arr], np.nan).astype("float64", copy=False)

    changed = ~np.isclose(sel_orig, sel_new, equal_nan=True, atol=1.0e-3)
    if not np.any(changed):
        return None

    changed_ai = ai_arr[changed]
    changed_gi = gi_arr[changed]
    changed_vals = sel_new[changed].astype("float32", copy=False)
    return changed_ai, changed_gi, changed_vals


@njit(cache=True)
def _ordered_local_fill_core(data, bad_val, target_mask, use_median, max_passes):
    """
    data        : 2D float64 array (nrays, ngates), mutated in place.
                  Every gate where target_mask is True starts at bad_val.
    target_mask : 2D bool array, same shape -- True for gates this fill
                  is allowed to touch (everything else, valid or not, is
                  only ever read as a potential neighbour, never written).
    use_median  : True = median of the valid 3x3 neighbours, False = mean.
    max_passes  : safety cap on the number of full raster sweeps.

    Repeatedly scans every target gate in raster order; a gate is filled
    as soon as it has at least 2 valid immediate (3x3, self excluded)
    neighbours -- "valid" meaning not bad_val *right now*, which includes
    gates this same pass already filled. That immediate reuse is what
    gives the nearest-to-valid-first behaviour: a gate touching the
    original valid data gets filled on pass 1, which can make its own
    neighbours fillable later in pass 1 or on pass 2, and so on outward,
    layer by layer, same idea as deglitch.gap_fill's along-ray filler but
    in the full 2D (ray, gate) neighbourhood the Gap Fill Brush itself
    uses. Gates that never pick up 2 valid neighbours (fully isolated
    within their local window) are left at bad_val.
    """
    nrays, ngates = data.shape
    vals = np.empty(8, dtype=np.float64)

    for _pass in range(max_passes):
        changed_any = False
        for ray in range(nrays):
            for gate in range(ngates):
                if not target_mask[ray, gate]:
                    continue
                if abs(data[ray, gate] - bad_val) > 1.0e-5:
                    continue  # already filled on an earlier pass

                r0 = ray - 1
                r1 = ray + 2
                if r0 < 0:
                    r0 = 0
                if r1 > nrays:
                    r1 = nrays
                g0 = gate - 1
                g1 = gate + 2
                if g0 < 0:
                    g0 = 0
                if g1 > ngates:
                    g1 = ngates

                n = 0
                for rr in range(r0, r1):
                    for gg in range(g0, g1):
                        if rr == ray and gg == gate:
                            continue
                        v = data[rr, gg]
                        if abs(v - bad_val) > 1.0e-5:
                            vals[n] = v
                            n += 1

                if n < 2:
                    continue  # not enough valid neighbours yet -- try a later pass

                if use_median:
                    newv = np.median(vals[:n])
                else:
                    s = 0.0
                    for i in range(n):
                        s += vals[i]
                    newv = s / n

                data[ray, gate] = newv
                changed_any = True

        if not changed_any:
            break

    return data


def _apply_custom_ordered_gapfill(app, sw, fname, data, ai_arr, gi_arr,
                                  warn: bool = True):
    """Gap Fill Brush's boundary-only custom algorithm: uses the
    standalone deglitch.py algorithm purely as a *detector* (any gate it
    would flag -- fold-correctable or not -- counts as "caught"), then
    fills every caught gate, plus every gate already missing beforehand,
    one at a time with the Gap Fill Brush's own local-3x3-median math --
    never the deglitch algorithm's own Nyquist-fold correction. Gates are
    filled nearest-to-already-valid-data first (only counting valid data
    *inside* the boundary as a neighbour -- valid gates just outside the
    polygon are never used), and each fill immediately counts as valid
    data for whichever gate gets processed next, so a filled boundary
    "grows inward" from its edges until every gate in it is valid (or,
    for a gate with no valid data anywhere nearby inside the boundary,
    left alone).
    """
    if len(ai_arr) == 0:
        return None

    nyq = app._get_deglitch_nyquist()
    if nyq is None or nyq <= 0:
        if warn:
            messagebox.showwarning(
                "Custom Gap Fill",
                "Could not determine short Nyquist velocity.\n"
                "Set it manually in the Editor Panel → Deglitch Settings.")
        return None

    bad_val = -9999.0
    orig_mask = np.ma.getmaskarray(data).copy()
    filled = np.ma.filled(data, bad_val).astype(np.float64)
    nrays = filled.shape[0]
    nyquist_arr = np.full(nrays, float(nyq), dtype=np.float64)

    radius = max(1, int(app._custom_deglitch_radius.get()))
    thresh = float(app._custom_deglitch_thresh.get())

    # Detection only: run the deglitch algorithm on a scratch copy so we
    # can see which gates it would have touched (folded, or flagged and
    # masked) without ever keeping its fold-correction as the answer --
    # that correction gets thrown away below in favour of the local-
    # median fill.
    detect_scratch = filled.copy()
    detect_result = _deglitch_run_core(
        detect_scratch, bad_val, nyquist_arr, radius, thresh,
        0, True, numprt=1)
    caught = (np.abs(detect_result - filled) > 1.0e-6) & (~orig_mask)

    boundary_mask = np.zeros(filled.shape, dtype=np.bool_)
    boundary_mask[ai_arr, gi_arr] = True

    # Everything to fill: gates the deglitch pass caught, plus gates that
    # were already missing/masked -- restricted to the boundary polygon,
    # since that's the only region this custom algorithm is allowed to
    # touch. "do that for everything in the boundary until everything is
    # valid" -- both kinds of invalid gate are treated the same way.
    target_mask = boundary_mask & (orig_mask | caught)
    if not np.any(target_mask):
        return None

    filled[target_mask] = bad_val   # discard any fold-correction on caught gates

    # Only gates *inside* the boundary may ever act as a valid neighbour
    # for the fill -- blank out everything outside it on a scratch copy so
    # the ordered fill below can't reach past the polygon's edge. Nothing
    # outside the boundary is reported back as changed regardless (only
    # ai_arr/gi_arr gets compared at the end), so this scratch copy is the
    # only place that matters.
    work = filled.copy()
    work[~boundary_mask] = bad_val

    max_passes = max(1, int(app._custom_gapfill_n_passes.get()))
    use_median = app._custom_gapfill_method.get() == "median"
    filled = _ordered_local_fill_core(work, bad_val, target_mask,
                                      use_median, max_passes)

    new_full = np.ma.masked_where(np.abs(filled - bad_val) <= 1.0e-5, filled)

    sel_orig = np.ma.filled(data[ai_arr, gi_arr], np.nan).astype("float64", copy=False)
    sel_new = np.ma.filled(new_full[ai_arr, gi_arr], np.nan).astype("float64", copy=False)

    changed = ~np.isclose(sel_orig, sel_new, equal_nan=True, atol=1.0e-3)
    if not np.any(changed):
        return None

    changed_ai = ai_arr[changed]
    changed_gi = gi_arr[changed]
    changed_vals = sel_new[changed].astype("float32", copy=False)
    return changed_ai, changed_gi, changed_vals


def _compute_edit_values(app, sw, fname, data, ai_arr, gi_arr, mode,
                         warn: bool = True):
    """Pure data-level edit math shared by every caller that selects a
    set of (ray, gate) indices on some sweep's field and wants the
    mode-specific math applied to them -- the dragged brush, the
    single-sweep boundary tool, "apply to all sweeps", and the folder
    batch-edit path all funnel through here.

    Deliberately does not touch undo state, the GPU buffer, or any
    panel/app rendering -- those belong to whichever caller actually
    knows what "this sweep" means for its situation (interactively
    displayed vs. one of many sweeps in a background volume).

    `warn` controls whether a failed precondition (e.g. no Nyquist set,
    an invalid GateFilter expression) pops a messagebox -- callers that
    loop over many sweeps/files pass warn=False after the first failure
    so the user isn't shown the same dialog dozens of times.

    Returns (changed_ai, changed_gi, changed_vals) -- the subset of
    ai_arr/gi_arr that actually changed, and their new values -- or
    None if nothing changed or a precondition failed.
    """
    if len(ai_arr) == 0:
        return None

    # Boundary-only override: when the "Use custom deglitch algorithm"
    # checkbox is on (only selectable while the Boundary Tool itself is
    # in use -- see _custom_algo_active):
    #   • Deglitch Brush  -> deglitch.py's own fold-correction algorithm
    #   • Gap Fill Brush  -> deglitch.py used only for detection, with
    #     every caught/missing gate filled one at a time by the Gap Fill
    #     Brush's own local-median math instead
    if mode == "deglitch_brush" and _custom_algo_active(app):
        return _apply_custom_deglitch_gapfill(app, sw, fname, data, ai_arr, gi_arr,
                                              mode, warn=warn)
    if mode == "gradient_brush" and _custom_algo_active(app):
        return _apply_custom_ordered_gapfill(app, sw, fname, data, ai_arr, gi_arr,
                                             warn=warn)

    brushed = data[ai_arr, gi_arr]
    brush_vals = np.ma.filled(brushed, np.nan).astype("float32", copy=False)
    new_vals = brush_vals.copy()
    changed = np.zeros(brush_vals.shape, dtype=bool)

    if mode == "unfold_brush":
        nyq = app._brush_nyquist.get()
        if nyq <= 0:
            nyq = (app._stagger_short_nyq.get()
                   or app._stagger_fixed_nyq.get() or 0.0)
        if nyq <= 0:
            if warn:
                messagebox.showwarning(
                    "Unfold",
                    "Nyquist velocity is 0.\n"
                    "Set a Nyquist value in the Editor Panel before applying.")
            return None
        # Fold centre: manual entry, or computed from multiplier when in auto mode.
        # Multiplier formula: fold_centre = (2 * mult - 1) * nyq
        #   x1.0 -> 1*nyq, x1.5 -> 2*nyq, x2.0 -> 3*nyq, …
        center_mode = getattr(app, "_unfold_center_mode", None)
        if center_mode is not None and center_mode.get() == "auto":
            mult     = app._unfold_fold_multiplier.get()
            negative = app._unfold_negative_center.get()
            n_nyq    = 2.0 * mult - 1.0
            ctr      = n_nyq * nyq * (-1 if negative else 1)
        else:
            ctr = app._brush_center.get()
        finite = np.isfinite(brush_vals)
        diff = np.where(finite, brush_vals - ctr, 0.0)
        n_folds = np.round(diff / (2.0 * nyq)).astype(int)
        candidate_vals = np.where(
            finite, brush_vals - n_folds * 2.0 * nyq, brush_vals)
        changed = finite & (candidate_vals != brush_vals)
        new_vals[changed] = candidate_vals[changed]

    elif mode == "deglitch_brush":
        # Staggered-PRF deglitch:
        #   • Folding interval  = 2 * short_nyquist  (gates alias at this step)
        #   • Long Nyquist acts as a *hint* via the local median:
        #       - the median of brushed gates is assumed to lie within the
        #         long-Nyquist unambiguous interval (i.e. it is the true vel)
        #       - each gate is then unfolded toward the median in units of
        #         2 * short_nyq (the short-Nyquist ambiguity interval)
        #   This matches the standard Sachidananda/Zrnic staggered-PRF
        #   unfolding convention: short-Nyquist PRF is unfolded by default,
        #   long-Nyquist used as reference.
        short_nyq = app._get_deglitch_nyquist()
        if short_nyq is None or short_nyq <= 0:
            if warn:
                messagebox.showwarning(
                    "Deglitch",
                    "Could not determine short Nyquist velocity.\n"
                    "Set it manually in the Editor Panel → Deglitch Settings.")
            return None

        tol = app._deglitch_tolerance.get()
        step = 2.0 * short_nyq   # one stagger-PRF folding interval

        # The local median over the selection is our long-Nyquist velocity
        # hint: most gates selected are (hopefully) correct, so the median
        # represents the true velocity that glitched gates should be
        # folded toward.
        finite_mask = np.isfinite(brush_vals)
        valid_vals = brush_vals[finite_mask]
        if len(valid_vals) < 3:
            return None   # not enough data for a reliable median
        median_val = float(np.median(valid_vals))

        # For each gate that deviates from the median by more than
        # tolerance, compute how many short-Nyquist folding intervals away
        # it is and shift it by that many steps toward the median.  Only
        # accept the shift if it actually reduces the deviation (prevents
        # over-correction at the boundary of the valid range).
        diff = np.where(finite_mask, brush_vals - median_val, 0.0)
        n_steps = np.round(diff / step).astype(int)
        candidate_vals = brush_vals - n_steps * step
        changed = (
            finite_mask &
            (np.abs(diff) > tol) &
            (n_steps != 0) &
            (np.abs(candidate_vals - median_val) < np.abs(diff))
        )
        new_vals[changed] = candidate_vals[changed]

    elif mode == "gradient_brush":
        # Gap Fill: detect folded gates the same way as the deglitch brush
        # (local median over the selection as hint, step = 2 * short_nyq),
        # but instead of snapping by an integer number of steps, replace
        # the offending gate with the *local neighbourhood median* — a
        # spatially-smooth velocity estimate from the surrounding gates.
        # This is better for regions where the true velocity is uncertain
        # or noisy. Optionally also fills NaN/masked gates with the
        # neighbourhood estimate when _gapfill_fill_nan is enabled.
        short_nyq = app._get_deglitch_nyquist()
        if short_nyq is None or short_nyq <= 0:
            if warn:
                messagebox.showwarning(
                    "Gap Fill",
                    "Could not determine short Nyquist velocity.\n"
                    "Set it manually in the Editor Panel → Deglitch Settings.")
            return None

        tol = app._deglitch_tolerance.get()
        step = 2.0 * short_nyq
        fill_nan = app._gapfill_fill_nan.get()
        nrays, ngates = data.shape

        # --- selection-wide median as the long-Nyquist hint (same as deglitch) ---
        finite_mask = np.isfinite(brush_vals)
        valid_brush = brush_vals[finite_mask]
        if len(valid_brush) < 3 and not fill_nan:
            return None
        brush_median = float(np.median(valid_brush)) if len(valid_brush) >= 3 else 0.0

        for i in range(len(ai_arr)):
            r_idx = int(ai_arr[i])
            g_idx = int(gi_arr[i])
            cur = brush_vals[i]
            is_finite = bool(finite_mask[i])

            # Decide whether this gate needs fixing
            needs_fix = False
            if is_finite:
                diff = cur - brush_median
                n_steps = int(np.round(diff / step))
                if n_steps != 0 and abs(diff) > tol:
                    needs_fix = True
            elif fill_nan:
                needs_fix = True

            if not needs_fix:
                continue

            # Build a local neighbourhood (3×3 in ray/gate space) to
            # estimate the true velocity, excluding the gate itself so a
            # badly folded centre gate doesn't poison the estimate.
            r0_ = max(0, r_idx - 1);  r1_ = min(nrays,  r_idx + 2)
            g0_ = max(0, g_idx - 1);  g1_ = min(ngates, g_idx + 2)
            nbhd = data[r0_:r1_, g0_:g1_].ravel()
            # exclude the gate itself
            ci = (r_idx - r0_) * (g1_ - g0_) + (g_idx - g0_)
            nbhd_vals = []
            for j, nv in enumerate(nbhd):
                if j == ci:
                    continue
                fv = float(np.ma.filled(nv, np.nan))
                if np.isfinite(fv):
                    nbhd_vals.append(fv)

            if len(nbhd_vals) < 2:
                continue   # not enough neighbours to estimate

            local_median = float(np.median(nbhd_vals))
            new_vals[i] = local_median
            changed[i] = True

    elif mode == "delete_gate":
        # Set every selected gate to NaN (masked), regardless of its
        # current value.
        new_vals[:] = np.nan
        changed[:] = True

    elif mode == "gatefilter_brush":
        # Area-limited sibling of edit/gatefilter.py's whole-sweep
        # apply_gate_filter(): evaluate the compound condition expression
        # built in the Editor Panel's GateFilter Brush Settings against
        # this sweep, then NaN-mask whichever brushed/boundary gates
        # match it — always in *this* field (fname/data), same as every
        # other brush mode here, never a separately-chosen target.
        spec = app._resolve_gatefilter_spec() if warn else \
               _resolve_gatefilter_spec_silent(app)
        if spec is None:
            return None
        conditions, combine = spec
        try:
            match_mask = evaluate_conditions(sw, conditions, combine)
        except ValueError as e:
            if warn:
                messagebox.showwarning("GateFilter Brush", str(e))
            return None
        matched = match_mask[ai_arr, gi_arr]
        new_vals[matched] = np.nan
        changed[matched] = True

    if not np.any(changed):
        return None

    changed_ai = ai_arr[changed]
    changed_gi = gi_arr[changed]
    changed_vals = new_vals[changed].astype("float32", copy=False)
    return changed_ai, changed_gi, changed_vals


def _resolve_gatefilter_spec_silent(app):
    """Same as app._resolve_gatefilter_spec() but never pops a
    messagebox -- used by multi-sweep/batch callers after the first
    sweep/file has already shown the warning once."""
    try:
        return app._resolve_gatefilter_spec(warn=False)
    except TypeError:
        # Older GURTVispy without the warn= kwarg -- fall back to the
        # normal (possibly-warning) call rather than hard failing.
        return app._resolve_gatefilter_spec()


def _apply_edit_core(app, panel, sw, fname, data, ai_arr, gi_arr, mode,
                     resolve_cmap, undo_flag_attr: str) -> bool:
    """Shared tail end of both apply_brush and apply_boundary: given an
    already-resolved set of (ray, gate) indices to operate on, run the
    mode-specific edit math via _compute_edit_values(), then push undo
    and patch the GPU buffer for *panel*'s currently-displayed sweep.

    `undo_flag_attr` is the name of the per-panel bool attribute used to
    avoid pushing a fresh undo snapshot for every dab of a single
    continuous brush stroke (apply_brush uses "_brush_undo_pushed";
    apply_boundary's one-shot edits use "_boundary_undo_pushed", reset
    fresh before every call).

    Returns True if any gate was actually changed.
    """
    result = _compute_edit_values(app, sw, fname, data, ai_arr, gi_arr, mode)
    if result is None:
        return False
    changed_ai, changed_gi, changed_vals = result

    if not getattr(panel, undo_flag_attr, False):
        app._push_undo([fname])
        setattr(panel, undo_flag_attr, True)

    data[changed_ai, changed_gi] = changed_vals

    try:
        app._invalidate_render_cache(sweep_index=app.sweep_index.get(),
                                     field_name=fname)
    except Exception:
        pass

    app._record_brush_touch(fname, changed_ai, changed_gi)

    # Fast path: patch only the affected triangles directly on the GPU
    # buffer of the panel being edited in. Any *other* panel currently
    # displaying the same field also needs a refresh, since editing in
    # one panel can change data another panel is showing.
    other_panels_show_field = any(
        p is not panel and p.field_combo.get() == fname for p in app.panels)

    fi2 = panel.current_field_info()
    if fi2 and fi2[0] == fname:
        _, _, vmin, vmax, cmap_name = fi2
        cmap = resolve_cmap(cmap_name) if resolve_cmap is not None else cmap_name
        panel.ppi.patch_gate_colors(
            changed_ai, changed_gi, changed_vals, data.shape[1],
            cmap, vmin, vmax)
        if other_panels_show_field:
            app._schedule_refresh()
    else:
        app._schedule_refresh()

    return True


# ══════════════════════════════════════════════════════════════════════════════
# Public entry points
# ══════════════════════════════════════════════════════════════════════════════

def apply_brush(app, panel, canvas_x: int, canvas_y: int, mode: str, resolve_cmap=None):
    """Apply the active brush at the given canvas pixel position, on
    *panel*'s currently-displayed field only. Called continuously while
    dragging — see PPICanvas._on_press / _on_drag.

    `app` is the GURTVispy instance (read for shared brush state / volume /
    undo / refresh helpers); `panel` is the PanelState being painted in.
    """
    target = _resolve_panel_target(app, panel)
    if target is None:
        return False
    sw, fname, data = target

    x_km, y_km = panel.ppi._canvas_to_scene(canvas_x, canvas_y)
    r_brush = max(0.0001, app._brush_radius_km.get())

    ai_arr, gi_arr = brush_gate_indices(sw, x_km, y_km, r_brush)
    return _apply_edit_core(app, panel, sw, fname, data, ai_arr, gi_arr, mode,
                            resolve_cmap, "_brush_undo_pushed")


def apply_boundary(app, panel, vertices_km, mode: str, resolve_cmap=None):
    """Apply *mode*'s edit math to every gate of *panel*'s currently-
    displayed field whose centre falls inside the closed polygon
    *vertices_km* (an iterable of (x_km, y_km) vertices).

    Unlike apply_brush (called continuously while dragging a circle),
    this is meant to be called once, when the user finalizes a boundary
    (Enter key, or the Editor Panel's "Apply Edit" button) — see
    GURTVispy._apply_boundary_edit.

    Returns True if any gate was actually changed, so the caller knows
    it's safe to clear the boundary; False on any precondition failure
    (no data loaded, fewer than 3 vertices, nothing inside the polygon,
    or nothing within it needed changing for *mode*).
    """
    target = _resolve_panel_target(app, panel)
    if target is None:
        return False
    sw, fname, data = target

    if len(vertices_km) < 3:
        return False

    ai_arr, gi_arr = boundary_gate_indices(sw, vertices_km)
    # Boundary edits are one-shot (applied once per Enter / button click),
    # not a continuous stroke, so always allow a fresh undo snapshot
    # rather than reusing the brush's "only push once per stroke" flag.
    panel._boundary_undo_pushed = False
    return _apply_edit_core(app, panel, sw, fname, data, ai_arr, gi_arr,
                            mode, resolve_cmap, "_boundary_undo_pushed")


def apply_boundary_all_sweeps(app, panel, vertices_km, mode: str,
                              resolve_cmap=None) -> bool:
    """Like apply_boundary, but runs *mode*'s edit math on every sweep of
    the currently loaded volume that carries *panel*'s displayed field,
    not just the active one.

    Each sweep has its own azimuth/range geometry, so the polygon's gate
    membership is recomputed per sweep via boundary_gate_indices() rather
    than reusing one index set. Global settings that aren't per-sweep
    (Nyquist, deglitch tolerance, the GateFilter Brush expression) are
    read once from the Editor Panel and applied uniformly to every sweep.

    All touched sweeps are pushed as a single grouped undo entry (see
    GURTVispy._push_undo_batch) so Ctrl+Z undoes every sweep at once
    rather than just the last one.

    Returns True if any sweep was actually changed.
    """
    if app.volume is None:
        return False
    fi = panel.current_field_info()
    if fi is None:
        return False
    fname = fi[0]
    if len(vertices_km) < 3:
        return False

    active_si = app.sweep_index.get()
    snapshots = []
    warn = True

    for si, sw in enumerate(app.volume.sweeps):
        data = sw.fields.get(fname)
        if data is None:
            continue
        ai_arr, gi_arr = boundary_gate_indices(sw, vertices_km)
        if len(ai_arr) == 0:
            continue
        result = _compute_edit_values(app, sw, fname, data, ai_arr, gi_arr,
                                      mode, warn=warn)
        if result is None:
            warn = False   # already told the user once; don't repeat it
            continue
        changed_ai, changed_gi, changed_vals = result

        snapshots.append({
            "sweep_index": si,
            "fields": {fname: data.copy()},
            "touched": {fname: (changed_ai.copy(), changed_gi.copy())},
        })

        data[changed_ai, changed_gi] = changed_vals
        app._invalidate_render_cache(sweep_index=si, field_name=fname)

        if si == active_si:
            app._record_brush_touch(fname, changed_ai, changed_gi)
            fi2 = panel.current_field_info()
            if fi2 and fi2[0] == fname:
                _, _, vmin, vmax, cmap_name = fi2
                cmap = resolve_cmap(cmap_name) if resolve_cmap is not None else cmap_name
                panel.ppi.patch_gate_colors(
                    changed_ai, changed_gi, changed_vals, data.shape[1],
                    cmap, vmin, vmax)

    if not snapshots:
        return False

    app._push_undo_batch(snapshots)

    other_panels_show_field = any(
        p is not panel and p.field_combo.get() == fname for p in app.panels)
    if other_panels_show_field:
        app._schedule_refresh()

    return True


def apply_boundary_to_volume(app, vol, fname: str, vertices_km, mode: str,
                             dest_fname: str = None) -> int:
    """Headless sibling of apply_boundary_all_sweeps for the folder
    batch-edit path (see main.py's _do_batch_boundary_edit): applies
    *mode*'s edit math to every sweep of *vol* -- an arbitrary
    RadarVolume, not necessarily app.volume -- that carries *fname*,
    wherever the boundary polygon *vertices_km* covers it.

    *vol* is typically a background-loaded file on its way to being
    written back out, so there's no panel, GPU buffer, or undo stack
    involved -- just the raw field arrays. `app` is only used for its
    global edit settings (Nyquist, deglitch tolerance, GateFilter Brush
    expression), same as every other mode here.

    If *dest_fname* is given (and differs from *fname*), each sweep's
    *fname* array is first copied to ``sw.fields[dest_fname]`` and the edit
    is applied to that copy only, leaving the original field untouched.
    Sweeps the polygon doesn't touch still get an unmodified copy so the
    new field exists on every sweep.

    Returns the number of sweeps actually changed.
    """
    if len(vertices_km) < 3:
        return 0

    copy_mode = bool(dest_fname) and dest_fname != fname
    n_changed = 0
    warn = True
    for sw in vol.sweeps:
        data = sw.fields.get(fname)
        if data is None:
            continue
        if copy_mode:
            data = data.copy()
            sw.fields[dest_fname] = data
        ai_arr, gi_arr = boundary_gate_indices(sw, vertices_km)
        if len(ai_arr) == 0:
            continue
        result = _compute_edit_values(app, sw, fname, data, ai_arr, gi_arr,
                                      mode, warn=warn)
        if result is None:
            warn = False
            continue
        changed_ai, changed_gi, changed_vals = result
        data[changed_ai, changed_gi] = changed_vals
        n_changed += 1

    return n_changed