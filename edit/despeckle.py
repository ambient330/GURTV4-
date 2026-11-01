# SPDX-License-Identifier: GPL-2.0-only
# GURT - Copyright (C) the GURT authors.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the
# Free Software Foundation. It is distributed WITHOUT ANY WARRANTY; see the
# file LICENSE for details.
"""Iterative despeckle, a port of DREADER's `despeck_parm` (qualitycontrol.c).

Every good gate counts the good gates in the (2r+1) x (2r+1) window around it
(ray x gate, itself excluded). If the count is below the minimum, the gate is
deleted. Passes repeat until a pass removes nothing, or `max_pass` is reached
(0 = until stable). Each pass reads the data as it stood at the start of the
pass, as the C code does (it writes into a temporary array and copies back).

The minimum scales at sweep edges: the C code keeps the ratio
min_good / (full window - 1) and rounds up for the clipped window. Differences
from the C code:
  * The clipped-window size is counted exactly (the C edge arithmetic is off
    at the far ray/gate edges).
  * `wrap` treats the ray axis as circular for full-circle sweeps, so rays
    next to north are not handled as an edge. Turn it off for the C behavior.
"""

import math

import numpy as np


def _box_sum(X, r, wrap):
    """(2r+1)^2 window sum of X (float 2-D), zeros outside; circular in rays if wrap."""
    ny, nx = X.shape
    if r <= 0:
        return X.copy()
    # along gates (zeros outside)
    P = np.pad(X, ((0, 0), (r, r)), mode="constant")
    c = np.concatenate([np.zeros((ny, 1)), np.cumsum(P, axis=1)], axis=1)
    G = c[:, 2 * r + 1:2 * r + 1 + nx] - c[:, 0:nx]
    # along rays (circular if wrap and the sweep is long enough, else zeros outside)
    mode = "wrap" if (wrap and ny >= 2 * r + 1) else "constant"
    P = np.pad(G, ((r, r), (0, 0)), mode=mode)
    c = np.concatenate([np.zeros((1, nx)), np.cumsum(P, axis=0)], axis=0)
    return c[2 * r + 1:2 * r + 1 + ny] - c[0:ny]


def despeckle_mask(good, radius=2, min_good=10, max_pass=0, wrap=True):
    """Return (kept_mask, n_removed, passes) for a boolean good-gate array."""
    good = np.asarray(good, bool).copy()
    r = int(radius)
    if r < 0 or min_good < 0:
        raise ValueError("radius and minimum good gates must be >= 0")
    maxgood = (2 * r + 1) ** 2 - 1
    frac = (min_good / maxgood) if maxgood else 0.0
    ny, nx = good.shape
    # window size (clipped at the sweep edges) minus the gate itself
    ones = np.ones(good.shape, float)
    reduced = _box_sum(ones, r, wrap) - 1.0
    need = np.ceil(frac * reduced - 1e-9)
    removed_total, passes = 0, 0
    while True:
        cnt = _box_sum(good.astype(float), r, wrap) - good
        bad = good & (cnt < need)
        n = int(bad.sum())
        passes += 1
        if n == 0:
            break
        good &= ~bad
        removed_total += n
        if max_pass > 0 and passes >= max_pass:
            break
    return good, removed_total, passes


def _delete_gates(arr, new_mask):
    """Copy of *arr* with the gates in *new_mask* deleted: masked AND, for float data,
    set to NaN, because GURT's renderer and tools treat NaN as the missing marker and
    would otherwise still draw the value hiding under the mask."""
    data = np.array(np.ma.getdata(arr), copy=True)
    if data.dtype.kind == "f":
        data[new_mask] = np.nan
    return np.ma.array(data, mask=new_mask)


def despeckle_field(data, radius=2, min_good=10, max_pass=0, wrap=True):
    """Despeckle a masked (nrays, ngates) array. Returns (new masked array, n_removed, passes)."""
    arr = np.ma.asarray(data)
    from edit.native import native
    nat = native()
    if nat is not None and arr.ndim == 2:
        out, n, passes = nat.despeckle(arr, radius, min_good, max_pass, wrap)
        return _delete_gates(arr, np.ma.getmaskarray(out)), n, passes
    good = ~np.ma.getmaskarray(arr)
    vals = np.ma.getdata(arr)
    if vals.dtype.kind == "f":
        good &= np.isfinite(vals)
    kept, n, passes = despeckle_mask(good, radius, min_good, max_pass, wrap)
    return _delete_gates(arr, np.ma.getmaskarray(arr) | ~kept), n, passes
