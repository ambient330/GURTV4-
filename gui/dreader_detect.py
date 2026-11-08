# SPDX-License-Identifier: GPL-2.0-only
# GURT - Copyright (C) the GURT authors.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the
# Free Software Foundation. It is distributed WITHOUT ANY WARRANTY; see the
# file LICENSE for details.
"""DREADER vortex detection, isolation and tracking (extractor modes 10-12) for GURT.

Ported from the DREADER C source (`detect_center`, `isolate_center`,
`track_center` in algorithms.c; `calc_percentile`, `calc_median`,
`calc_greatcirc`, `calc_latlon`, `eval_linearspline` helpers).

detect_vortices  one sweep, one velocity field -> list of vortex detections.
                 Finds the largest gate-to-gate / ray+-k velocity difference
                 (|V1 - V2| with the two gates no farther apart than the max
                 diameter), then refines the center from the median positions
                 of the strongest outbound / inbound gates in a box around it,
                 marks that box as searched and repeats.
isolate_detections
                 chains detections (from successive sweeps / volumes) into
                 separate vortices using time, elevation and lat/lon proximity.
track_vortices   per vortex: horizontal translation (linear-spline path,
                 great-circle motion, per elevation layer) and the tilt of the
                 vortex axis between successive sweeps.

Deviations from the C code, all so the port behaves on real GURT data:
  * Rays are ordered by azimuth and neighbours wrap around 360 deg for full
    circles (the C code works on array indices and cannot pair across the seam).
  * The search box uses an azimuth window instead of ray indices, so a box that
    straddles the seam is not turned into "every ray in between".
  * Median / mean azimuths are taken relative to the detection azimuth.
  * Duplicate integer-second observation times inside one tracking layer are
    collapsed (the C spline divides by zero there).
"""

import csv
import os
import math
import types

import numpy as np

from velmag import _win_sum, _half_win, nan_box_mean, smoothed_velmag  # noqa: F401  (shared helpers)

TOL = 1.0e-5
EARTHRAD = 6371000.0
APPEARTHRAD = 8483000.0
MINLOCS = 4          # min observations in a layer to compute a translation
MAXTRAN = 25.0       # translation speeds at or above this are discarded (m/s)


# ───────────────────────────────────────────────────────────── small helpers
def _wrap180(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def calc_percentile(values, percent):
    """DREADER percentile: mean of the floor / ceil ranked values (no interpolation)."""
    v = np.sort(np.asarray(values, float))
    if v.size == 0:
        return float("nan")
    idx = (percent / 100.0) * (v.size - 1)
    return float((v[int(math.floor(idx))] + v[int(math.ceil(idx))]) / 2.0)


def calc_median(values):
    v = np.sort(np.asarray(values, float))
    n = v.size
    if n == 0:
        return float("nan")
    return float(v[n // 2]) if n % 2 else float((v[n // 2 - 1] + v[n // 2]) / 2.0)


def _beam_geometry(rng_m, el_deg):
    """hdist, vdist (4/3-earth beam height) as in calc_cartesian."""
    e = math.radians(el_deg)
    h = rng_m * math.cos(e)
    z = rng_m * math.sin(e)
    v = math.sqrt(rng_m ** 2 + APPEARTHRAD ** 2 + 2.0 * APPEARTHRAD * z) - APPEARTHRAD
    return h, v


def calc_latlon(rng_m, az_deg, el_deg, radar_lat, radar_lon):
    """Gate latitude / longitude (deg) from the radar position (calc_latlon)."""
    h, v = _beam_geometry(rng_m, el_deg)
    gc = math.asin(min(1.0, h / (v + EARTHRAD)))
    p0, l0, az = map(math.radians, (radar_lat, radar_lon, az_deg))
    lat = math.asin(math.sin(p0) * math.cos(gc) + math.cos(p0) * math.sin(gc) * math.cos(az))
    lon = l0 + math.atan2(math.sin(gc) * math.cos(p0) * math.sin(az),
                          math.cos(gc) - math.sin(p0) * math.sin(lat))
    while lon > math.pi:
        lon -= 2 * math.pi
    while lon < -math.pi:
        lon += 2 * math.pi
    return math.degrees(lat), math.degrees(lon)


def calc_greatcirc(lat0, lon0, lat1, lon1):
    """Great-circle distance (m) and compass bearing (deg) from point 0 to point 1."""
    p0, p1 = math.radians(lat0), math.radians(lat1)
    dl = math.radians(lon1 - lon0)
    c = math.sin(p0) * math.sin(p1) + math.cos(p0) * math.cos(p1) * math.cos(dl)
    dist = EARTHRAD * math.acos(max(-1.0, min(1.0, c)))
    ang = math.degrees(math.atan2(math.cos(p0) * math.sin(p1) - math.sin(p0) * math.cos(p1) * math.cos(dl),
                                  math.sin(dl) * math.cos(p1)))
    if ang < 0:
        ang += 360.0
    ang = 90.0 - ang                    # convert_angle: math angle -> compass bearing
    if ang < 0:
        ang += 360.0
    return dist, ang


# ───────────────────────────────────────────────────────────── detection
def _geo(az_deg, el_deg):
    """Per-ray direction cosines (x east, y north, z up) of a unit slant range."""
    a, e = np.radians(az_deg), np.radians(el_deg)
    return np.sin(a) * np.cos(e), np.cos(a) * np.cos(e), np.sin(e)


def _scan_cells(ri, gi, V, geo, rng, skip, n, circ, maxdiam):
    """Best (|dV|, chord, shear, k) over the radius loop for each (ray, gate) cell.

    k = 0 pairs (ray, ray-1); k >= 1 pairs (ray-k, ray+k). The loop for a cell
    stops once the two gates are farther apart than *maxdiam*.
    """
    sx, cy, se = geo
    m = ri.size
    best_dv = np.zeros(m)
    best_d = np.zeros(m)
    best_sh = np.zeros(m)
    best_k = np.zeros(m, int)
    dist = np.zeros(m)
    idx = np.arange(m)
    r = ri.copy()
    g = gi.copy()
    kmax = (n - 1) // 2 if circ else n
    k = 0
    while idx.size and k <= kmax:
        if k == 0:
            a = r
            b = r - 1
            geom_ok = np.ones(r.size, bool) if circ else (b >= 0)
            b = np.where(geom_ok, b, r) % n
            free = ~skip[a, g] & ~skip[b, g]
            b = np.where(free, b, a)                      # degenerate pair (dist 0)
            free = ~skip[a, g] & ~skip[b, g]
        else:
            if circ:
                a, b = (r + k) % n, (r - k) % n
            else:
                keep = (r + k < n) & (r - k >= 0)
                idx, r, g = idx[keep], r[keep], g[keep]
                if not idx.size:
                    break
                a, b = r + k, r - k
            free = ~skip[a, g] & ~skip[b, g]
        d = rng[g] * np.sqrt((sx[a] - sx[b]) ** 2 + (cy[a] - cy[b]) ** 2 + (se[a] - se[b]) ** 2)
        cur = np.where(free, d, dist[idx])
        dist[idx] = cur
        va, vb = V[a, g], V[b, g]
        good = free & np.isfinite(va) & np.isfinite(vb)
        dv = np.where(good, np.abs(np.where(good, va, 0.0) - np.where(good, vb, 0.0)), 0.0)
        cand = good & (dv > best_dv[idx]) & (cur <= maxdiam)
        ci = idx[cand]
        best_dv[ci] = dv[cand]
        best_d[ci] = d[cand]
        best_sh[ci] = np.where(d[cand] > TOL, 2.0 * dv[cand] / np.maximum(d[cand], TOL), 0.0)
        best_k[ci] = k
        alive = cur <= maxdiam
        idx, r, g = idx[alive], r[alive], g[alive]
        k += 1
    return best_dv, best_d, best_sh, best_k


def _unix(sweep, vol=None, ray=None):
    """UTC unix time of a ray. Per-ray times are absolute in some formats but seconds since the sweep start in
    others (DORADE sweepfiles); values below 1e8 are offsets and are added to the sweep / volume start time."""
    st = getattr(sweep, "scan_time", None) or (getattr(vol, "scan_time", None) if vol is not None else None)
    base = 0.0
    if st is not None:
        try:
            if st.tzinfo is None:
                from datetime import timezone
                st = st.replace(tzinfo=timezone.utc)
            base = float(st.timestamp())
        except Exception:
            base = 0.0
    rt = getattr(sweep, "ray_time", None)
    if rt is not None and ray is not None and len(rt) > ray:
        t = float(rt[ray])
        return t if abs(t) >= 1.0e8 else base + t
    return base


def _prep(sweep, field):
    """Azimuth-sorted view of one sweep / field (None if unusable)."""
    az_all = np.asarray(sweep.azimuth, float) % 360.0
    n = az_all.size
    rng = np.asarray(sweep.ranges, float)
    if n < 8 or rng.size < 3 or field not in sweep.fields:
        return None
    el_all = (np.asarray(sweep.elevation, float) if getattr(sweep, "elevation", None) is not None
              else np.full(n, float(getattr(sweep, "fixed_angle", 0.0) or 0.0)))
    order = np.argsort(az_all, kind="stable")
    az, el = az_all[order], el_all[order]
    V = np.ma.filled(np.ma.asarray(sweep.fields[field], float), np.nan)[order][:, :rng.size]
    gaps = np.diff(np.r_[az, az[0] + 360.0])
    med_gap = float(np.median(gaps[gaps > 0])) if np.any(gaps > 0) else 1.0
    return types.SimpleNamespace(
        sweep=sweep, field=field, az=az, el=el, rng=rng, V=V, order=order, n=n, ng=rng.size,
        circ=bool(gaps.max() <= 3.0 * med_gap), med_gap=med_gap, geo=_geo(az, el))


# ───────────────────────────────────────────────────────────── azimuthal shear
def azimuthal_shear(P, rays, gates, radius_m=1000.0):
    """Local linear least-squares azimuthal shear (1/s) at the given rays x gates.

    A plane V = a + b*s + c*r is fit to the good radial velocities in a window
    around each cell (s = distance along the azimuthal direction, r = along the
    beam); azimuthal shear is b. Positive = cyclonic (radial velocity increases
    with azimuth, i.e. outbound to the right of inbound). The window half-size is
    about *radius_m* in both directions, measured at the median range of the
    requested cells (so it is a fixed number of rays / gates for the call).
    Returns an array of shape (len(rays), len(gates)); NaN where under half the
    window holds good data.
    """
    rays = np.asarray(rays, int)
    gates = np.asarray(gates, int)
    RI, GJ = np.meshgrid(rays, gates, indexing="ij")
    ri, gj = RI.ravel(), GJ.ravel()
    n, ng, rng = P.n, P.ng, P.rng
    sx, cy, _ = P.geo
    Rc = float(np.median(rng[gates]))
    dth = math.radians(P.med_gap)
    dr = float(np.median(np.diff(rng))) if ng > 1 else 250.0
    m = int(min(8, max(1, round(radius_m / max(Rc * dth, 1.0)))))
    nn = int(min(4, max(1, round(radius_m / max(dr, 1.0)))))
    L = max(dr, Rc * dth)                                    # length scale: keeps the 3x3 solve well conditioned
    th = np.radians(P.az)[ri]
    cth, sth = np.cos(th), np.sin(th)
    xc, yc = rng[gj] * sx[ri], rng[gj] * cy[ri]
    S = np.zeros((9, ri.size))
    total = (2 * m + 1) * (2 * nn + 1)
    for di in range(-m, m + 1):
        rn = ri + di
        if P.circ:
            rn = rn % n
            vr = np.ones(ri.size, bool)
        else:
            vr = (rn >= 0) & (rn < n)
            rn = np.clip(rn, 0, n - 1)
        for dj in range(-nn, nn + 1):
            gn = gj + dj
            vg = (gn >= 0) & (gn < ng)
            gn = np.clip(gn, 0, ng - 1)
            v = P.V[rn, gn]
            w = vr & vg & np.isfinite(v)
            dx = rng[gn] * sx[rn] - xc
            dy = rng[gn] * cy[rn] - yc
            s = (dx * cth - dy * sth) / L
            r = (dx * sth + dy * cth) / L
            wf = w.astype(float)
            v = np.where(w, v, 0.0)
            S[0] += wf
            S[1] += wf * s
            S[2] += wf * r
            S[3] += wf * s * s
            S[4] += wf * r * r
            S[5] += wf * s * r
            S[6] += v
            S[7] += v * s
            S[8] += v * r
    out = np.full(ri.size, np.nan)
    ok = S[0] >= 0.5 * total
    if ok.any():
        A = np.empty((ok.sum(), 3, 3))
        A[:, 0, 0], A[:, 0, 1], A[:, 0, 2] = S[0][ok], S[1][ok], S[2][ok]
        A[:, 1, 0], A[:, 1, 1], A[:, 1, 2] = S[1][ok], S[3][ok], S[5][ok]
        A[:, 2, 0], A[:, 2, 1], A[:, 2, 2] = S[2][ok], S[5][ok], S[4][ok]
        b = np.stack([S[6][ok], S[7][ok], S[8][ok]], axis=1)[..., None]
        try:
            sol = np.linalg.solve(A, b)[..., 0]
        except np.linalg.LinAlgError:
            sol = (np.linalg.pinv(A) @ b)[..., 0]
        out[ok] = sol[:, 1] / L
    return out.reshape(len(rays), len(gates))


def _polarity_score(az, polarity):
    p = str(polarity).lower()
    if p.startswith("cyc"):
        return az
    if p.startswith("anti"):
        return -az
    return np.abs(az)


# ───────────────────────────────────────────────────────────── characterization
def _characterize(P, ray0, gate0, maxdv, chord, shear, kk, extremapct, skip,
                  file_id, sweep_index, radar_lat, radar_lon, vol):
    """DREADER's refinement of one candidate couplet into a detection dict.

    Returns (detection or None, rays, gates) where rays / gates are the search
    box the caller should mark as searched.
    """
    az, el, rng, V, n, order = P.az, P.el, P.rng, P.V, P.n, P.order
    R0 = float(rng[gate0])
    min_g = int(np.argmin(np.abs(rng - max(R0 - chord, 0.0))))
    max_g = int(np.argmin(np.abs(rng - (R0 + chord))))
    half = math.degrees(chord / R0) if R0 > 0 else 180.0
    d_az = _wrap180(az - az[ray0])
    rays = np.where(np.abs(d_az) <= half)[0] if half < 180.0 else np.arange(n)
    if ray0 not in rays:
        rays = np.append(rays, ray0)
    gates = np.arange(min_g, max_g + 1)

    box = V[np.ix_(rays, gates)]
    good = np.isfinite(box)
    vels = box[good]
    if not vels.size:
        return None, rays, gates
    maxvel = calc_percentile(vels, extremapct)
    minvel = calc_percentile(vels, 100.0 - extremapct)
    rr, gg = np.where(good)
    bv = box[rr, gg]
    mx = bv >= maxvel
    mn = bv <= minvel
    az_off = d_az[rays][rr]
    rg = rng[gates][gg]
    med_max_r, med_max_o = calc_median(rg[mx]), calc_median(az_off[mx])
    med_min_r, med_min_o = calc_median(rg[mn]), calc_median(az_off[mn])
    c_r = 0.5 * (med_max_r + med_min_r)
    c_off = 0.5 * (med_max_o + med_min_o)
    c_az = (az[ray0] + c_off) % 360.0
    cg = int(np.argmin(np.abs(rng - c_r)))
    cr = int(np.argmin(np.abs(_wrap180(az - c_az))))
    if skip is not None and skip[cr, cg]:
        return None, rays, gates

    if kk > 0:
        i1, i2 = ray0 - kk, ray0 + kk
    else:
        i1, i2 = ray0 - 1, ray0
    i1, i2 = i1 % n, i2 % n
    v1, v2 = float(V[i1, gate0]), float(V[i2, gate0])
    if v2 >= v1:
        imin, imax, vmin_i, vmax_i = i1, i2, v1, v2
    else:
        imin, imax, vmin_i, vmax_i = i2, i1, v2, v1
    el_c = float(el[cr])
    h, vd = _beam_geometry(float(rng[cg]), el_c)
    lat, lon = calc_latlon(float(rng[cg]), float(az[cr]), el_c, radar_lat, radar_lon)
    t = _unix(P.sweep, vol, int(order[cr]))
    det = {
        "file": file_id, "sweep": sweep_index, "field": P.field, "time": t,
        "azimuth": float(az[cr]), "elevation": el_c, "range_m": float(rng[cg]),
        "hdist_m": h, "vdist_m": vd, "lat": lat, "lon": lon,
        "x_km": float(rng[cg]) / 1000.0 * math.sin(math.radians(az[cr])),
        "y_km": float(rng[cg]) / 1000.0 * math.cos(math.radians(az[cr])),
        "init_min": vmin_i, "init_min_az": float(az[imin]), "init_min_rng": float(rng[gate0]),
        "init_max": vmax_i, "init_max_az": float(az[imax]), "init_max_rng": float(rng[gate0]),
        "delta_v": maxdv, "diameter_m": chord, "shear": shear,
        "first_az": float(az[ray0]), "first_rng": R0,
        "min_pct": 100.0 - extremapct, "min_vel": minvel, "n_min": int(mn.sum()),
        "med_min_az": float((az[ray0] + med_min_o) % 360.0), "med_min_rng": med_min_r,
        "max_pct": extremapct, "max_vel": maxvel, "n_max": int(mx.sum()),
        "med_max_az": float((az[ray0] + med_max_o) % 360.0), "med_max_rng": med_max_r,
        "n_total": int(mn.sum() + mx.sum()),
        "azshear": float("nan"), "seed": False, "box": None,
        "_ray": cr, "_gate": cg,
    }
    return det, rays, gates


def couplet_sign(det):
    """+1 if the detected couplet is cyclonic (outbound at the larger azimuth), else -1."""
    return 1.0 if _wrap180(det["init_max_az"] - det["init_min_az"]) > 0 else -1.0


def _polarity_ok(det, polarity):
    p = str(polarity).lower()
    if p.startswith("cyc"):
        return couplet_sign(det) > 0
    if p.startswith("anti"):
        return couplet_sign(det) < 0
    return True


def _center_azshear(P, det, radius_m, polarity):
    """Signed azimuthal shear at a detection's center (strongest of the 3x3 cells around it)."""
    cr, cg = det["_ray"], det["_gate"]
    rays = np.array([(cr + d) % P.n for d in (-1, 0, 1)]) if P.circ else \
        np.clip(np.array([cr - 1, cr, cr + 1]), 0, P.n - 1)
    gates = np.clip(np.array([cg - 1, cg, cg + 1]), 0, P.ng - 1)
    a = azimuthal_shear(P, rays, gates, radius_m)
    if not np.isfinite(a).any():
        return float("nan")
    sc = np.where(np.isfinite(a), _polarity_score(a, polarity), -np.inf)
    i, j = np.unravel_index(int(np.argmax(sc)), sc.shape)
    return float(a[i, j])


def detect_vortices(sweep, field, radar_lat, radar_lon, maxdiam_m=2000.0, mindeltav=30.0,
                    extremapct=90.0, maxtors=0, file_id="", sweep_index=0, vol=None,
                    progress=None, azshear_filter=None):
    """Run DREADER's detection on one sweep. Returns a list of detection dicts.

    azshear_filter: optional dict(min_azshear, polarity, radius_m); detections whose
    center azimuthal shear fails it are dropped (a false-positive screen).
    """
    P = _prep(sweep, field)
    if P is None:
        return []
    n, ng, rng, V = P.n, P.ng, P.rng, P.V
    skip = np.zeros((n, ng), bool)
    G, R = np.meshgrid(np.arange(ng - 1), np.arange(n), indexing="ij")      # gate-major order, as in C
    cell_g, cell_r = G.ravel(), R.ravel()
    best_dv, best_d, best_sh, best_k = _scan_cells(cell_r, cell_g, V, P.geo, rng, skip, n, P.circ, maxdiam_m)

    out = []
    numtor = 0
    while maxtors == 0 or numtor < maxtors:
        if progress:
            progress(numtor)
        c = int(np.argmax(best_dv))
        maxdv = float(best_dv[c])
        ray0, gate0 = int(cell_r[c]), int(cell_g[c])
        chord, shear, kk = float(best_d[c]), float(best_sh[c]), int(best_k[c])
        passed = maxdv >= mindeltav and chord <= maxdiam_m and maxdv > 0.0
        if passed:
            det, rays, gates = _characterize(P, ray0, gate0, maxdv, chord, shear, kk, extremapct, skip,
                                             file_id, sweep_index, radar_lat, radar_lon, vol)
            if det is not None:
                keep = True
                if azshear_filter:
                    az_c = _center_azshear(P, det, azshear_filter["radius_m"], azshear_filter["polarity"])
                    det["azshear"] = az_c
                    keep = bool(np.isfinite(az_c) and
                                _polarity_score(az_c, azshear_filter["polarity"]) >= azshear_filter["min_azshear"]
                                and _polarity_ok(det, azshear_filter["polarity"]))
                else:
                    det["azshear"] = _center_azshear(P, det, 1000.0, "either")
                if keep:
                    out.append(det)
                    numtor += 1
        else:
            R0 = float(rng[gate0])
            min_g = int(np.argmin(np.abs(rng - max(R0 - chord, 0.0))))
            max_g = int(np.argmin(np.abs(rng - (R0 + chord))))
            half = math.degrees(chord / R0) if R0 > 0 else 180.0
            d_az = _wrap180(P.az - P.az[ray0])
            rays = np.where(np.abs(d_az) <= half)[0] if half < 180.0 else np.arange(n)
            gates = np.arange(min_g, max_g + 1)

        # mark the box searched and refresh the cells it invalidated
        skip[np.ix_(rays, gates)] = True
        if not passed:
            break
        live = best_dv > 0
        ia = np.where(best_k > 0, cell_r + best_k, cell_r)
        ib = np.where(best_k > 0, cell_r - best_k, cell_r - 1)
        stale = live & (skip[ia % n, cell_g] | skip[ib % n, cell_g])
        if stale.any():
            s = np.where(stale)[0]
            dv, dd, ss, kq = _scan_cells(cell_r[s], cell_g[s], V, P.geo, rng, skip, n, P.circ, maxdiam_m)
            best_dv[s], best_d[s], best_sh[s], best_k[s] = dv, dd, ss, kq
        if not (best_dv > 0).any():
            break
    return out


# ───────────────────────────────────────────────────────────── boxed detection
CRITERIA = ("AzShear + dV", "AzShear", "dV (DREADER)")


def detect_in_box(sweep, field, box_km, radar_lat, radar_lon, criterion="AzShear + dV",
                  polarity="Cyclonic", min_azshear=0.02, azshear_radius_m=1000.0,
                  maxdiam_m=2000.0, mindeltav=30.0, extremapct=90.0,
                  file_id="", sweep_index=0, vol=None, why=None):
    """Find THE vortex inside a user box: at most one detection, or None.

    box_km = (x0, x1, y0, y1) in the radar-relative scene frame (km, x east, y north).

    criterion
      "AzShear + dV"  locate the strongest azimuthal shear in the box (sign set by
                      *polarity*), then require its couplet's |dV| >= mindeltav.
      "AzShear"       locate by azimuthal shear only (|dV| is reported, not required).
      "dV (DREADER)"  the original DREADER search, limited to the box.
    In every case the located couplet is refined with DREADER's percentile / median
    center procedure and must be no wider than *maxdiam_m*.

    why: optional list; when nothing is found, one line saying which test failed and by how much
    is appended (what the vortex dialog shows).
    """
    def no(msg):
        if why is not None:
            why.append(msg)
        return None

    P = _prep(sweep, field)
    if P is None:
        return no("the sweep has no usable data for this field")
    x0, x1, y0, y1 = box_km
    rkm = P.rng / 1000.0
    azr = np.radians(P.az)
    xs = np.outer(np.sin(azr), rkm)
    ys = np.outer(np.cos(azr), rkm)
    inbox = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
    if not inbox.any():
        return no("the box holds no radar gates (it is outside the sweep or in the cone of silence)")
    rr, gg = np.where(inbox)
    rays, gates = np.unique(rr), np.unique(gg)
    sub = inbox[np.ix_(rays, gates)]
    n, ng = P.n, P.ng
    skip = np.zeros((n, ng), bool)
    use_az = criterion in ("AzShear + dV", "AzShear")
    peak = float("nan")

    if use_az:
        a = azimuthal_shear(P, rays, gates, azshear_radius_m)
        score = np.where(sub & np.isfinite(a), _polarity_score(a, polarity), -np.inf)
        i, j = np.unravel_index(int(np.argmax(score)), score.shape)
        if not np.isfinite(score[i, j]) or score[i, j] < min_azshear:
            av = np.where(sub & np.isfinite(a), a, np.nan)
            if not np.isfinite(av).any():
                return no(f"no azimuthal shear could be computed in the box (too few good {field} gates around each cell)")
            return no(f"strongest azimuthal shear in the box is {np.nanmax(av):+.3f} 1/s cyclonic and "
                      f"{-np.nanmin(av):+.3f} 1/s anticyclonic; the {str(polarity).lower()} one must reach {min_azshear:g} 1/s")
        peak = float(a[i, j])
        r0, g0 = int(rays[i]), int(gates[j])
        cr_ = (r0 + np.arange(-2, 3)) % n if P.circ else np.clip(r0 + np.arange(-2, 3), 0, n - 1)
        cg_ = np.clip(g0 + np.arange(-2, 3), 0, ng - 2)
        CR, CG = np.meshgrid(np.unique(cr_), np.unique(cg_), indexing="ij")
        cells_r, cells_g = CR.ravel(), CG.ravel()
    else:
        ii, jj = np.where(sub & (gates[None, :] < ng - 1))
        cells_r, cells_g = rays[ii], gates[jj]
        if not cells_r.size:
            return no("the box holds no usable gates")

    dv, dd, sh, kq = _scan_cells(cells_r, cells_g, P.V, P.geo, P.rng, skip, n, P.circ, maxdiam_m)
    c = int(np.argmax(dv))
    if dv[c] <= 0.0:
        return no("no velocity couplet (inbound next to outbound) around the shear peak")
    maxdv, chord = float(dv[c]), float(dd[c])
    if criterion != "AzShear" and maxdv < mindeltav:
        return no(f"the couplet's delta-V is {maxdv:.1f} m/s, below the required {mindeltav:g} m/s "
                  f"(couplets are only searched up to Max couplet diameter = {maxdiam_m:g} m, so a wider vortex reads low - raise it; "
                  "also check the velocity is not folded / aliased)")
    if chord > maxdiam_m:
        return no(f"the couplet is {chord:.0f} m wide, wider than the maximum diameter {maxdiam_m:g} m "
                  "(raise Max couplet diameter in Settings for a large vortex)")
    det, _, _ = _characterize(P, int(cells_r[c]), int(cells_g[c]), maxdv, chord, float(sh[c]),
                              int(kq[c]), extremapct, None, file_id, sweep_index,
                              radar_lat, radar_lon, vol)
    if det is None:
        return no("the couplet could not be refined (missing velocity around the extrema)")
    if use_az and not _polarity_ok(det, polarity):
        return no(f"the couplet found has the opposite rotation sense to {str(polarity).lower()}"
                  f" (delta-V {maxdv:.1f} m/s, {chord:.0f} m wide); set Rotation sense to Either or the other sign")
    det["azshear"] = peak if use_az else _center_azshear(P, det, azshear_radius_m, polarity)
    det["box"] = (float(x0), float(x1), float(y0), float(y1))
    return det


# ───────────────────────────────────────────────────────────── following a boxed vortex
def latlon_to_xy_km(lat, lon, ref_lat, ref_lon):
    return ((lon - ref_lon) * 111.32 * math.cos(math.radians(ref_lat)), (lat - ref_lat) * 111.19)


def xy_km_to_latlon(x, y, ref_lat, ref_lon):
    return (ref_lat + y / 111.19, ref_lon + x / (111.32 * math.cos(math.radians(ref_lat))))


class BoxFollower:
    """Carries the one tracked vortex from sweep to sweep.

    Starts from the seed detection (the first detection inside the user's box).
    For every new sweep it builds a search box around where the vortex should be
    (last detection, advanced by the motion estimated from the recent detections),
    the same size as the user's box (optionally scaled), widened 25% for each
    sweep in a row that found nothing. It gives up after *max_misses* misses in a
    row or when the next sweep is more than *maxtimediff* seconds from the last
    detection. Positions are lat/lon, so a moving (mobile) radar is handled.
    """

    def __init__(self, seed, half_x_km, half_y_km, predict=True, scale=1.0,
                 max_misses=4, maxtimediff=600.0):
        self.ref = (seed["lat"], seed["lon"])
        self.hist = [seed]
        self.half = (half_x_km * scale, half_y_km * scale)
        self.predict = predict
        self.max_misses = max_misses
        self.maxtimediff = maxtimediff
        self.misses = 0
        self.v = (0.0, 0.0)                       # km/s east, north

    def _enu(self, d):
        return latlon_to_xy_km(d["lat"], d["lon"], *self.ref)

    def box_for(self, t, radar_lat, radar_lon):
        last = self.hist[-1]
        dt = t - last["time"]
        if abs(dt) > self.maxtimediff:
            return None
        ex, ey = self._enu(last)
        if self.predict:
            ex += self.v[0] * dt
            ey += self.v[1] * dt
        plat, plon = xy_km_to_latlon(ex, ey, *self.ref)
        cx, cy = latlon_to_xy_km(plat, plon, radar_lat, radar_lon)
        grow = 1.0 + 0.25 * self.misses
        hx, hy = self.half[0] * grow, self.half[1] * grow
        return (cx - hx, cx + hx, cy - hy, cy + hy)

    def accept(self, det):
        self.hist.append(det)
        self.misses = 0
        recent = self.hist[-4:]
        old, new = recent[0], recent[-1]
        span = new["time"] - old["time"]
        if len(recent) >= 2 and abs(span) >= 20.0:
            (x0, y0), (x1, y1) = self._enu(old), self._enu(new)
            vx, vy = (x1 - x0) / span, (y1 - y0) / span
            if math.hypot(vx, vy) * 1000.0 < MAXTRAN:
                self.v = (vx, vy)

    def miss(self):
        """Register a sweep with no detection; True when the follower should stop."""
        self.misses += 1
        return self.misses > self.max_misses


# ───────────────────────────────────────────────────────────── wind swath



class WindSwath:
    """Maximum smoothed Doppler velocity magnitude |V| seen at each point along a track.

    Every sweep of the tracked vortex contributes |V| (smoothed over about
    *smooth_m* in azimuth and range) for the gates inside that sweep's search
    box. Values from all sweeps are placed on one common east / north grid
    (relative to the seed position, through lat/lon so a moving radar works)
    and the per-cell maximum is kept: a wind swath like a time-maximum
    velocity-magnitude product.
    """

    def __init__(self, ref_lat, ref_lon, smooth_m=200.0):
        self.ref = (ref_lat, ref_lon)
        self.smooth_m = smooth_m
        self.x, self.y, self.v = [], [], []
        self.gate_m = []
        self.n_sweeps = 0

    def add(self, sweep, field, box_km, radar_lat, radar_lon):
        P = _prep(sweep, field)
        if P is None:
            return 0
        x0, x1, y0, y1 = box_km
        mar = self.smooth_m / 1000.0 + 0.5
        rkm = P.rng / 1000.0
        azr = np.radians(P.az)
        xs = np.outer(np.sin(azr), rkm)
        ys = np.outer(np.cos(azr), rkm)
        wide = (xs >= x0 - mar) & (xs <= x1 + mar) & (ys >= y0 - mar) & (ys <= y1 + mar)
        if not wide.any():
            return 0
        rr, gg = np.where(wide)
        rays, gates = np.unique(rr), np.unique(gg)
        if P.circ and rays.size > 1 and rays[0] == 0 and rays[-1] == P.n - 1:
            k = int(np.argmax(np.diff(rays)))                # box straddles the 0/360 seam
            if np.diff(rays)[k] > 1:
                rays = np.r_[rays[k + 1:], rays[:k + 1]]
        A = np.abs(P.V[np.ix_(rays, gates)])
        Rc = float(np.median(P.rng[gates]))
        dr = float(np.median(np.diff(P.rng)))
        m = _half_win(self.smooth_m, Rc * math.radians(P.med_gap))
        nn = _half_win(self.smooth_m, dr)
        S = nan_box_mean(A, min(m, 40), min(nn, 20))
        X, Y = xs[np.ix_(rays, gates)], ys[np.ix_(rays, gates)]
        sel = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1) & np.isfinite(S)
        if not sel.any():
            return 0
        ox, oy = latlon_to_xy_km(radar_lat, radar_lon, *self.ref)
        self.x.append(X[sel] + ox)
        self.y.append(Y[sel] + oy)
        self.v.append(S[sel])
        self.gate_m.append(dr)
        self.n_sweeps += 1
        return int(sel.sum())

    def grid(self, res_m=100.0):
        """Rasterize to {'M': (ny, nx) array (NaN = no data), 'x0', 'y0' (km), 'res_km'}."""
        if not self.x:
            return None
        x, y, v = np.concatenate(self.x), np.concatenate(self.y), np.concatenate(self.v)
        res = max(res_m, 10.0) / 1000.0
        x0, y0 = float(x.min()), float(y.min())
        nx = int((x.max() - x0) / res) + 1
        ny = int((y.max() - y0) / res) + 1
        while nx * ny > 4.0e6:
            res *= 2.0
            nx = int((x.max() - x0) / res) + 1
            ny = int((y.max() - y0) / res) + 1
        ix = np.clip(((x - x0) / res).astype(int), 0, nx - 1)
        iy = np.clip(((y - y0) / res).astype(int), 0, ny - 1)
        M = np.full(nx * ny, -np.inf)
        np.maximum.at(M, iy * nx + ix, v)
        M = M.reshape(ny, nx)
        # bridge the gaps between gate centers (gates wider than a grid cell)
        reps = int(min(3, math.ceil(0.5 * float(np.median(self.gate_m)) / (res * 1000.0))))
        for _ in range(max(reps, 0)):
            P = np.pad(M, 1, constant_values=-np.inf)
            M = np.max([P[i:i + ny, j:j + nx] for i in range(3) for j in range(3)], axis=0)
        M[~np.isfinite(M)] = np.nan
        return {"M": M, "x0": x0, "y0": y0, "res_km": res}



def swath_field_for_sweep(sweep, g, ref, radar_lat, radar_lon):
    """Sample a swath grid (from WindSwath.grid) onto one sweep's gates.

    Returns a masked (nrays, ngates) array in the sweep's own shape: the swath
    maximum where the gate lies inside the swath, masked elsewhere.
    """
    az = np.radians(np.asarray(sweep.azimuth, float) % 360.0)
    rkm = np.asarray(sweep.ranges, float) / 1000.0
    ox, oy = latlon_to_xy_km(radar_lat, radar_lon, *ref)
    x = np.outer(np.sin(az), rkm) + ox
    y = np.outer(np.cos(az), rkm) + oy
    M = g["M"]
    ny, nx = M.shape
    ix = np.floor((x - g["x0"]) / g["res_km"]).astype(int)
    iy = np.floor((y - g["y0"]) / g["res_km"]).astype(int)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    out = np.full(x.shape, np.nan)
    out[ok] = M[iy[ok], ix[ok]]
    shape = np.asarray(sweep.fields[next(iter(sweep.fields))]).shape if sweep.fields else out.shape
    if shape != out.shape:
        full = np.full(shape, np.nan)
        h, w = min(shape[0], out.shape[0]), min(shape[1], out.shape[1])
        full[:h, :w] = out[:h, :w]
        out = full
    return np.ma.masked_invalid(out)


def swath_to_ascii(g, ref, path, unix, alt_m=0.0, sensor="Radar", product="vmax",
                   unit="m/s", fill=0.0, south_first=False):
    """Write the swath grid as a swath ASCII file (named swath.<name>: the prefix is added when missing;
    returns the path written).

    Header, one 'KEY value' per line: UT (unix s), XG / YG (grid geographic
    origin, lon / lat), ZG (origin height, m), X0 / Y0 (grid cell origin, m east /
    north of XG, YG), nx, ny, dx, dy (cell size, m), sn (sensor), pr (product),
    un (unit). Then ny rows of nx space-delimited values; row 0 is the northern
    row unless *south_first*. Cells with no data get *fill*. CRLF line endings.
    """
    d, b = os.path.split(str(path))
    path = os.path.join(d, b if b.startswith("swath.") else "swath." + b)
    M = np.where(np.isfinite(g["M"]), g["M"], fill)
    if not south_first:
        M = M[::-1]
    ny, nx = M.shape
    res_m = g["res_km"] * 1000.0

    def num(v):
        return f"{int(round(v))}" if abs(v - round(v)) < 1e-6 else f"{v:.3f}"

    y0 = g["y0"] * 1000.0                        # south edge, whichever row comes first
    head = [("UT", f"{int(round(unix))}"), ("XG", f"{ref[1]:.6f}"), ("YG", f"{ref[0]:.6f}"),
            ("ZG", f"{alt_m:.2f}"), ("X0", str(int(round(g["x0"] * 1000.0)))), ("Y0", str(int(round(y0)))),
            ("nx", str(nx)), ("ny", str(ny)), ("dx", num(res_m)), ("dy", num(res_m)),
            ("sn", sensor), ("pr", product), ("un", unit)]
    with open(path, "w", newline="") as fh:
        for k, v in head:
            fh.write(f"{k} {v}\r\n")
        for row in M:
            fh.write(" ".join(f"{x:.4f}" for x in row) + "\r\n")
    return path


def swath_to_csv(g, ref, path):
    """Write the swath grid cells that hold data: East/North (km from the seed), lat, lon, |V|."""
    M = g["M"]
    iy, ix = np.where(np.isfinite(M))
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["East(km)", "North(km)", "Lat(deg)", "Lon(deg)", "VelMagMax(m/s)"])
        for j, i in zip(iy, ix):
            x = g["x0"] + (i + 0.5) * g["res_km"]
            y = g["y0"] + (j + 0.5) * g["res_km"]
            lat, lon = xy_km_to_latlon(x, y, *ref)
            w.writerow([f"{x:.3f}", f"{y:.3f}", f"{lat:.6f}", f"{lon:.6f}", f"{M[j, i]:.1f}"])


# ───────────────────────────────────────────────────────────── isolation
def isolate_detections(dets, maxtimediff=600.0, maxelevdiff=2.0, maxlatdiff=0.03,
                       maxlondiff=0.03, mindetects=3):
    """Group detections (in chronological order) into vortices.

    Returns a list of lists of detection indices (one list per vortex, ids
    1..N in order). Detections that never reach *mindetects* are discarded.
    Faithful to the C control flow: a candidate must come from a different
    sweep than the current member, a time gap above *maxtimediff* ends the
    chain, and a different-elevation candidate may link with 2x lat/lon slack.
    """
    n = len(dets)
    lin = [0] * n
    numtors = 0
    while any(v == 0 for v in lin):
        center = next(i for i, v in enumerate(lin) if v == 0)
        cur = dets[center]
        numdetects = 1
        numtors += 1
        ncenter = center
        cur_file = (cur["file"], cur["sweep"])
        next_file = cur_file
        notfound = False
        while ncenter < n - 1:
            # advance to the next unprocessed detection from a different sweep
            while ncenter < n - 1 and (cur_file == next_file or lin[ncenter] != 0 or notfound):
                ncenter += 1
                next_file = (dets[ncenter]["file"], dets[ncenter]["sweep"])
                notfound = False
            if lin[ncenter] != 0 or next_file == cur_file:
                break
            nd = dets[ncenter]
            dt = abs(nd["time"] - cur["time"])
            de = abs(nd["elevation"] - cur["elevation"])
            dlat, dlon = abs(nd["lat"] - cur["lat"]), abs(nd["lon"] - cur["lon"])
            if dt <= maxtimediff and de <= maxelevdiff:
                if dlat <= maxlatdiff and dlon <= maxlondiff:
                    link = True
                else:
                    link = False
            elif dt <= maxtimediff and de > maxelevdiff:
                link = dlat <= 2 * maxlatdiff and dlon <= 2 * maxlondiff
            else:
                break                                   # time gap: this vortex is finished
            if link:
                numdetects += 1
                lin[center] = numtors
                lin[ncenter] = numtors
                center = ncenter
                cur = nd
                cur_file = next_file
                notfound = False
            else:
                notfound = True
        lin[center] = numtors
        if numdetects < mindetects:
            for i in range(n):
                if lin[i] == numtors:
                    lin[i] = -1
            numtors -= 1
    tracks = [[] for _ in range(numtors)]
    for i, v in enumerate(lin):
        if v > 0:
            tracks[v - 1].append(i)
    return tracks


# ───────────────────────────────────────────────────────────── tracking
def track_vortices(dets, track, deltz=0.0):
    """Translation and tilt for the detections in *track* (indices into *dets*).

    Writes into each detection dict: trans_speed (m/s), trans_dir (deg),
    vert_diff_m, horiz_diff_m, incl_deg, orient_deg, incl_corr_deg, orient_corr_deg.
    """
    obs = [dets[i] for i in track]
    m = len(obs)
    for o in obs:
        o.update({"trans_speed": 0.0, "trans_dir": 0.0, "vert_diff_m": float("nan"),
                  "horiz_diff_m": float("nan"), "incl_deg": float("nan"), "orient_deg": float("nan"),
                  "incl_corr_deg": float("nan"), "orient_corr_deg": float("nan")})
    if m == 0:
        return obs

    h = np.array([o["vdist_m"] for o in obs])
    minh, maxh = min(0.0, float(h.min())), max(0.0, float(h.max()))
    if abs(deltz) > TOL:
        nlay = max(1, int(math.ceil((maxh - minh) / deltz)))
        edges = [(minh + i * deltz, minh + (i + 1) * deltz) for i in range(nlay)]
    else:
        q1, q2, q3 = (calc_percentile(h, p) for p in (25, 50, 75))
        edges = [(minh, q1), (q1, q2), (q2, q3), (q3, maxh)]

    for bot, top in edges:
        sel = [i for i in range(m) if bot <= h[i] <= top]
        if not sel:
            continue
        # collapse duplicate integer-second times
        times, keep = [], []
        for i in sel:
            t = int(round(obs[i]["time"]))
            if times and t == times[-1]:
                continue
            times.append(t)
            keep.append(i)
        if len(keep) < 2:
            continue
        t_arr = np.array(times)
        lat_arr = np.array([obs[i]["lat"] for i in keep])
        lon_arr = np.array([obs[i]["lon"] for i in keep])
        grid = np.arange(t_arr[0], t_arr[-1] + 1)
        lat_g = np.interp(grid, t_arr, lat_arr)
        lon_g = np.interp(grid, t_arr, lon_arr)
        for loc, i in enumerate(keep):
            ti = int(t_arr[loc] - t_arr[0])
            if loc == 0:
                j0, j1 = ti, ti + 1
            elif loc == len(keep) - 1:
                j0, j1 = ti - 1, ti
            else:
                j0, j1 = ti - 1, ti + 1
            dtime = float(grid[j1] - grid[j0])
            if len(keep) >= MINLOCS and dtime > 0:
                dist, ang = calc_greatcirc(lat_g[j0], lon_g[j0], lat_g[j1], lon_g[j1])
                speed = dist / dtime
            else:
                speed, ang = 0.0, 0.0
            if speed >= MAXTRAN:
                speed, ang = 0.0, 0.0
            obs[i]["trans_speed"], obs[i]["trans_dir"] = speed, ang

    # tilt between successive observations (same volume, higher beam, < 60 s apart)
    for j in range(1, m):
        a, b = obs[j - 1], obs[j]
        ta, tb = int(round(a["time"])), int(round(b["time"]))
        if b["vdist_m"] > a["vdist_m"] and abs(tb - ta) < 60:
            xa, ya = a["hdist_m"] * math.sin(math.radians(a["azimuth"])), a["hdist_m"] * math.cos(math.radians(a["azimuth"]))
            xb, yb = b["hdist_m"] * math.sin(math.radians(b["azimuth"])), b["hdist_m"] * math.cos(math.radians(b["azimuth"]))
            vdiff = b["vdist_m"] - a["vdist_m"]
            hdiff = math.hypot(xb - xa, yb - ya)
            incl = math.degrees(math.atan(abs(hdiff / vdiff)))
            orient = math.degrees(math.atan2(xb - xa, yb - ya)) % 360.0
            # motion-corrected: move the later position back to the earlier time
            sp, ang = b["trans_speed"], math.radians(b["trans_dir"])
            xc = xb + (ta - tb) * sp * math.sin(ang)
            yc = yb + (ta - tb) * sp * math.cos(ang)
            hdiff_c = math.hypot(xc - xa, yc - ya)
            incl_c = math.degrees(math.atan(abs(hdiff_c / vdiff)))
            orient_c = math.degrees(math.atan2(xc - xa, yc - ya)) % 360.0
            b.update({"vert_diff_m": vdiff, "horiz_diff_m": hdiff, "incl_deg": incl,
                      "orient_deg": orient, "incl_corr_deg": incl_c, "orient_corr_deg": orient_c})
    return obs


# ───────────────────────────────────────────────────────────── CSV output
_DET_COLS = [
    ("Track", "track"), ("File", "file"), ("Sweep", "sweep"), ("Field", "field"),
    ("UnixEpochTime(s)", "time"), ("AzimuthAngle(deg)", "azimuth"), ("ElevationAng(deg)", "elevation"),
    ("Range(m)", "range_m"), ("HorizDist(m)", "hdist_m"), ("VertDist(m)", "vdist_m"),
    ("GateLat(deg)", "lat"), ("GateLon(deg)", "lon"),
    ("FGuessMin(m/s)", "init_min"), ("FGuessMinAz(deg)", "init_min_az"), ("FGuessMinRa(m)", "init_min_rng"),
    ("FGuessMax(m/s)", "init_max"), ("FGuessMaxAz(deg)", "init_max_az"), ("FGuessMaxRa(m)", "init_max_rng"),
    ("DeltaVFixRa(m/s)", "delta_v"), ("DiameterFixRa(m)", "diameter_m"), ("SymVertVort(1/s)", "shear"),
    ("AzShear(1/s)", "azshear"),
    ("FirstGuessAz(deg)", "first_az"), ("FirstGuessRa(m)", "first_rng"),
    ("MinVPercent(%)", "min_pct"), ("MinVPercntl(m/s)", "min_vel"), ("NumMinV", "n_min"),
    ("MedianMinAz(deg)", "med_min_az"), ("MedianMinRa(m)", "med_min_rng"),
    ("MaxVPercent(%)", "max_pct"), ("MaxVPercntl(m/s)", "max_vel"), ("NumMaxV", "n_max"),
    ("MedianMaxAz(deg)", "med_max_az"), ("MedianMaxRa(m)", "med_max_rng"), ("TotMax", "n_total"),
    ("TransSpeed(m/s)", "trans_speed"), ("TransDirect(deg)", "trans_dir"),
    ("VertCentDiff(m)", "vert_diff_m"), ("HorizCentDiff(m)", "horiz_diff_m"),
    ("InclinAngle(deg)", "incl_deg"), ("OrientAngle(deg)", "orient_deg"),
    ("CorrInclinA(deg)", "incl_corr_deg"), ("CorrOrientA(deg)", "orient_corr_deg"),
]


def detections_to_csv(dets, path, track_ids=None):
    """Write detections (DREADER detection-file columns; tracking columns when present)."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([c[0] for c in _DET_COLS])
        for i, d in enumerate(dets):
            row = []
            for _, key in _DET_COLS:
                if key == "track":
                    row.append("" if not track_ids or track_ids[i] <= 0 else track_ids[i])
                    continue
                v = d.get(key, "")
                if isinstance(v, float):
                    row.append("" if not np.isfinite(v) else f"{v:.6f}" if key in ("lat", "lon")
                               else f"{v:.6g}" if key in ("shear", "azshear") else f"{v:.2f}")
                else:
                    row.append(v)
            w.writerow(row)
