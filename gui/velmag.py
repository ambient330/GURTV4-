# SPDX-License-Identifier: GPL-2.0-only
# GURT - Copyright (C) the GURT authors.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the
# Free Software Foundation. It is distributed WITHOUT ANY WARRANTY; see the
# file LICENSE for details.
"""Smoothed velocity magnitude (|V| averaged over a window of a given length).

Self-contained (numpy only) so it ships in every build, independent of the
vortex-detection modules that also use these helpers.
"""

import math
import types

import numpy as np


def _sorted_view(sweep, field):
    """Azimuth-sorted view of one sweep / field (None if unusable)."""
    az_all = np.asarray(sweep.azimuth, float) % 360.0
    n = az_all.size
    rng = np.asarray(sweep.ranges, float)
    if n < 8 or rng.size < 3 or field not in sweep.fields:
        return None
    order = np.argsort(az_all, kind="stable")
    az = az_all[order]
    V = np.ma.filled(np.ma.asarray(sweep.fields[field], float), np.nan)[order][:, :rng.size]
    gaps = np.diff(np.r_[az, az[0] + 360.0])
    med_gap = float(np.median(gaps[gaps > 0])) if np.any(gaps > 0) else 1.0
    return types.SimpleNamespace(az=az, rng=rng, V=V, order=order, n=n, ng=rng.size,
                                 circ=bool(gaps.max() <= 3.0 * med_gap), med_gap=med_gap)


def _win_sum(X, k, axis):
    """Centered window sum of half-size k along *axis* (zeros outside the array)."""
    if k <= 0:
        return X.copy()
    L = X.shape[axis]
    pad = [(0, 0)] * X.ndim
    pad[axis] = (k + 1, k)
    cs = np.cumsum(np.pad(X, pad), axis=axis)
    hi = [slice(None)] * X.ndim
    lo = [slice(None)] * X.ndim
    hi[axis] = slice(2 * k + 1, 2 * k + 1 + L)
    lo[axis] = slice(0, L)
    return cs[tuple(hi)] - cs[tuple(lo)]


def _half_win(length_m, spacing_m):
    """Half-size (cells) of a boxcar whose full length is about *length_m*; 0 = no smoothing."""
    return max(int(round((length_m / max(spacing_m, 1.0) - 1.0) / 2.0)), 0)


def nan_box_mean(A, m, n, min_frac=0.5):
    """Mean of the good values in a (2m+1) x (2n+1) window; NaN if under *min_frac* are good."""
    good = np.isfinite(A)
    v = np.where(good, A, 0.0)
    s = _win_sum(_win_sum(v, m, 0), n, 1)
    c = _win_sum(_win_sum(good.astype(float), m, 0), n, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = s / c
    out[c < min_frac * (2 * m + 1) * (2 * n + 1)] = np.nan
    return out


def smoothed_velmag(sweep, field, smooth_m=200.0, min_frac=0.5):
    """Smoothed |velocity| for a whole sweep, as a masked (nrays, ngates) array.

    A boxcar of about *smooth_m* in azimuth (arc length, so more rays at long
    range is fewer) and in range, averaging the good |V| values only; cells
    with under *min_frac* of the window good are masked. Wraps through north
    for full-circle sweeps.
    """
    P = _sorted_view(sweep, field)
    if P is None:
        raise ValueError("Sweep has no usable data in that field.")
    A = np.abs(P.V)
    dr = float(np.median(np.diff(P.rng)))
    nn = _half_win(smooth_m, dr)
    out = np.full(A.shape, np.nan)
    edges = np.linspace(0, P.ng, 17).astype(int)
    for g0, g1 in zip(edges[:-1], edges[1:]):
        if g1 <= g0:
            continue
        Rc = float(np.median(P.rng[g0:g1]))
        m = min(_half_win(smooth_m, Rc * math.radians(P.med_gap)), 40)
        lo, hi = max(g0 - nn, 0), min(g1 + nn, P.ng)
        blk = A[:, lo:hi]
        if P.circ and m > 0:
            blk = np.concatenate([blk[-m:], blk, blk[:m]], axis=0)
            S = nan_box_mean(blk, m, nn, min_frac)[m:-m]
        else:
            S = nan_box_mean(blk, m, nn, min_frac)
        out[:, g0:g1] = S[:, g0 - lo:g1 - lo]
    res = np.full(A.shape, np.nan)
    res[P.order] = out                                   # back to the sweep's own ray order
    full = np.full(np.asarray(sweep.fields[field]).shape, np.nan)
    full[:, :res.shape[1]] = res
    return np.ma.masked_invalid(full)
