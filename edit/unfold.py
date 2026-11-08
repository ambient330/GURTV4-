# SPDX-License-Identifier: GPL-2.0-only
# GURT - Copyright (C) the GURT authors.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the
# Free Software Foundation. It is distributed WITHOUT ANY WARRANTY; see the
"""DREADER unfold (dealias_parm): Bargen-Brown style along-ray unfolding.

For each good gate (ray by ray, range increasing, edited in place) the reference
is the median of the good gates among the previous ``radius + 1`` gates on the
same ray (already unfolded). A gate that differs from it by more than
``thresh * nyquist`` is stepped by +/-2*nyquist toward the reference for as long
as it stays on the same side; the closest value seen is kept. Gates with no
reference are left alone. With a reference sweep (``ref``) the reference is
taken from the matching gates of that sweep instead (falls back to the same-ray
rule when the reference has nothing there).

Arithmetic is float32 like the C original; libgurt (gurt_unfold) gives identical
results and is used when built.
"""

import numpy as np

try:
    from numba import njit
except ImportError:                                  # pure Python fallback (slow)
    def njit(*a, **k):
        return a[0] if a and callable(a[0]) else (lambda f: f)


@njit(cache=True)
def _median_f(buf, n):
    v = np.sort(buf[:n])
    if n % 2 == 0:
        return np.float32((np.float64(v[n // 2 - 1]) + np.float64(v[n // 2])) / 2.0)
    return v[(n - 1) // 2]


@njit(cache=True)
def _unfold_core(data, nyq, radius, thresh, have_ref, ref, ray_map, gate_map, region, use_region):
    nrays, ngates = data.shape
    rng = ref.shape[1]
    good = np.empty(2 * radius + 4, dtype=np.float32)
    f_thresh = np.float32(thresh)
    two = np.float32(2.0)
    changed = 0
    for ray in range(nrays):
        nq = np.float32(nyq[ray])
        if not (nq > np.float32(0.0)) or not np.isfinite(nq):
            continue
        for gate in range(ngates):
            orig = data[ray, gate]
            if np.isnan(orig) or (use_region and not region[ray, gate]):
                continue
            abort = False
            numgood = 0
            locvalue = np.float32(0.0)
            if have_ref:
                rr = ray_map[ray]
                rg = gate_map[gate]
                if rr < 0 or rg < 0:
                    abort = True
                else:
                    lg = rg
                    count = 0
                    while lg >= 0 and count <= radius:
                        if not np.isnan(ref[rr, lg]):
                            good[numgood] = ref[rr, lg]
                            numgood += 1
                        count += 1
                        lg -= 1
                    lg = rg + 1
                    count = 0
                    while lg < rng and count <= radius:
                        if not np.isnan(ref[rr, lg]):
                            good[numgood] = ref[rr, lg]
                            numgood += 1
                        count += 1
                        lg += 1
                    if numgood == 0:
                        while lg < rng and numgood == 0:
                            if not np.isnan(ref[rr, lg]):
                                good[numgood] = ref[rr, lg]
                                numgood += 1
                            lg += 1
                    if numgood == 0:
                        lg = rg - radius
                        while lg >= 0 and numgood == 0:
                            if not np.isnan(ref[rr, lg]):
                                good[numgood] = ref[rr, lg]
                                numgood += 1
                            lg -= 1
                    if numgood > 0:
                        locvalue = _median_f(good, numgood)
                    else:
                        abort = True
            if (not have_ref) or abort:
                lg = gate - 1
                count = 0
                numgood = 0
                while lg >= 0 and count <= radius:
                    v = data[ray, lg]
                    if not np.isnan(v):
                        good[numgood] = v
                        numgood += 1
                    count += 1
                    lg -= 1
                if numgood > 0:
                    locvalue = _median_f(good, numgood)
                    abort = False
                else:
                    abort = True
            newvalue = orig
            mindiff = np.abs(newvalue - locvalue)
            if (not abort) and mindiff > f_thresh * nq:
                unf = 0
                prev = 0
                steps = 0
                while (not abort) and steps < 100000:
                    steps += 1
                    if unf == prev or prev == 0:
                        prev = unf
                        if newvalue < locvalue:
                            newvalue = newvalue + two * nq
                            unf = 1
                        else:
                            newvalue = newvalue - two * nq
                            unf = -1
                        newdiff = np.abs(newvalue - locvalue)
                        if newdiff < mindiff:
                            data[ray, gate] = newvalue
                            mindiff = newdiff
                    else:
                        abort = True
            if data[ray, gate] != orig:
                changed += 1
    return changed


def match_reference(az, ranges, ref_az, ref_ranges, az_tol=0.5):
    """Map each ray / gate of the current sweep to the ray / gate of the reference sweep
    (DREADER locate_closeray / locate_gate). Returns (ray_map, gate_map) with -1 = none."""
    az = np.asarray(az, float)
    ref_az = np.asarray(ref_az, float)
    d = np.abs(az[:, None] - ref_az[None, :])
    d = np.minimum(d, 360.0 - d)
    j = np.argmin(d, axis=1)
    ray_map = np.where(d[np.arange(len(az)), j] <= abs(az_tol), j, -1).astype(np.int32)
    ranges = np.asarray(ranges, float)
    ref_ranges = np.asarray(ref_ranges, float)
    if len(ref_ranges) > 1:
        spacing = abs(float(np.median(np.diff(ref_ranges))))
    else:
        spacing = 0.0
    k = np.clip(np.searchsorted(ref_ranges, ranges), 1, max(len(ref_ranges) - 1, 1))
    k0 = np.clip(k - 1, 0, len(ref_ranges) - 1)
    k1 = np.clip(k, 0, len(ref_ranges) - 1)
    pick = np.where(np.abs(ref_ranges[k0] - ranges) <= np.abs(ref_ranges[k1] - ranges), k0, k1)
    ok = np.abs(ref_ranges[pick] - ranges) < spacing
    gate_map = np.where(ok, pick, -1).astype(np.int32)
    return ray_map, gate_map


def unfold_field(data, nyquist, radius=2, thresh=0.8, ref=None, ray_map=None, gate_map=None,
                 region=None):
    """Unfold *data* (masked array, nrays x ngates). `nyquist`: scalar or per-ray.
    Optional reference sweep `ref` with `ray_map` / `gate_map` (see match_reference).
    Returns (masked array, n_unfolded)."""
    from edit.native import native
    nat = native()
    if nat is not None and nat.has_unfold():
        return nat.unfold(data, nyquist, radius, thresh, ref, ray_map, gate_map, region)
    m = np.ma.getmaskarray(data)
    x = np.array(np.ma.getdata(data), dtype=np.float32, copy=True)
    x[m] = np.nan
    nr, ng = x.shape
    nq = np.broadcast_to(np.asarray(nyquist, np.float64), (nr,)).copy()
    have_ref = ref is not None
    if have_ref:
        rm = np.ma.getmaskarray(ref)
        r = np.array(np.ma.getdata(ref), dtype=np.float32, copy=True)
        r[rm] = np.nan
        rmap = np.asarray(ray_map, np.int32)
        gmap = np.asarray(gate_map, np.int32)
    else:
        r = np.zeros((1, 1), np.float32)
        rmap = np.zeros(1, np.int32)
        gmap = np.zeros(1, np.int32)
    use_region = region is not None
    reg = np.asarray(region, bool) if use_region else np.zeros((1, 1), bool)
    n = _unfold_core(x, nq, int(radius), float(thresh), have_ref, r, rmap, gmap, reg, use_region)
    return np.ma.masked_invalid(x), int(n)
