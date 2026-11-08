# SPDX-License-Identifier: GPL-2.0-only
# GURT - Copyright (C) the GURT authors.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the
# Free Software Foundation. It is distributed WITHOUT ANY WARRANTY; see the
# file LICENSE for details.
"""Ports of two DREADER `extractor` modes (C. Alexander, SourceForge "dreader"):

  * patch dynamics  (extract_patch + calc_univdynam): couplet extrema inside a
    circular patch, aspect-ratio (beam smoothing) correction, ground-relative
    and corrected peak velocities, radius of maximum vorticity from an annulus
    scan, vorticity / divergence / stretching / angular momentum at fixed
    fractions of the mean radius, F- and EF-scale equivalents.
  * axisymmetric mode (extract_axisym): concentric rings about the vortex
    centre; in each ring a 2-parameter least-squares fit of
        VD = Vr*cos(phi - az) + Vt*sin(phi - az)
    gives the mean radial and tangential wind (phi = compass bearing centre ->
    gate, az = radar azimuth), then circulation, angular momentum, pressure
    deficit and flow angle.

Differences from the C code (deliberate): angles use ONE convention (compass
bearings) instead of mixing math and compass angles; translation is projected
onto each gate's beam direction; VD is divided (not multiplied) by cos(elev).
No GUI code; works on full-sweep arrays.
"""

import numpy as np

try:                                       # optional C implementation (libgurt, full build)
    from edit.native import native as _native
except Exception:                          # pragma: no cover
    def _native():
        return None


def _nat():
    m = _native()
    return m if (m is not None and getattr(m, "has_axisym", lambda: False)()) else None

PATCH_RADII_FRACTIONS = (0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00)
MIN_SAMPLES = 5            # DREADER MINSAMP
KM = 1000.0


# ---------------------------------------------------------------- helpers
def aspect_ratio_correction(beamwidth_deg, range_km, diam_km):
    """DREADER calc_aspectcorr: factor restoring peak velocities smoothed by
    the finite beam width. 1/(1-0.24*(2*bw/D)) for bw <= D/4, else 1.14."""
    if beamwidth_deg is None or beamwidth_deg <= 0 or diam_km <= 0 or range_km <= 0:
        return 1.0
    bw_km = range_km * np.deg2rad(beamwidth_deg)
    if bw_km <= 0.25 * diam_km:
        return 1.0 / (1.0 - 0.24 * (2.0 * bw_km / diam_km))
    return 1.14


def f_scale(v):
    """DREADER calc_fscale (Fujita): (|v|/6.3)^(2/3) - 2."""
    return float((abs(v) / 6.3) ** (2.0 / 3.0) - 2.0)


_EF_EDGES = (29.0, 38.0, 49.2, 60.4, 73.8, 89.4, 104.6)


def ef_scale(v):
    """DREADER calc_efscale: continuous EF number from 3-s gust speed (m/s),
    0 below 29 m/s extrapolating negative, saturating at 6."""
    v = abs(v)
    if v <= 29.0:
        return float((v - 29.0) / 29.0)
    if v > _EF_EDGES[-1]:
        return 6.0
    for k in range(len(_EF_EDGES) - 1):
        lo, hi = _EF_EDGES[k], _EF_EDGES[k + 1]
        if lo < v <= hi:
            return float(k + (v - lo) / (hi - lo))
    return 6.0


def _sweep_xy(az_deg, rng_km):
    az = np.deg2rad(np.asarray(az_deg, float))[:, None]
    r = np.asarray(rng_km, float)[None, :]
    return r * np.sin(az), r * np.cos(az)


def annulus_width_km(az_deg, rng_km, center_range_km):
    """DREADER calc_annwidth: the smaller of the angular spacing (at the
    centre range) and the gate spacing."""
    az = np.unwrap(np.deg2rad(np.asarray(az_deg, float)))
    d_az = float(np.median(np.abs(np.diff(az)))) if len(az) > 1 else 0.0
    d_r = float(np.median(np.abs(np.diff(rng_km)))) if len(rng_km) > 1 else 0.0
    cands = [c for c in (center_range_km * d_az, d_r) if c > 0]
    return min(cands) if cands else 0.0


def _bearing_deg(dx, dy):
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def _ring_extrema(vel, X, Y, xc, yc, radius, width):
    """Min/max of the valid velocities in the annulus |d - radius| <= width/2."""
    d = np.hypot(X - xc, Y - yc)
    m = np.isfinite(vel) & (d >= radius - width / 2.0) & (d <= radius + width / 2.0)
    if not m.any():
        return None
    idx = np.flatnonzero(m.ravel())
    v = vel.ravel()[idx]
    return idx[int(np.argmin(v))], idx[int(np.argmax(v))]


# ------------------------------------------------------------ patch mode
def _patch_py(vel, az_deg, rng_km, center_km, base_diam_km, beamwidth_deg=1.0,
                          numdiam=1.0, storm_speed_ms=0.0, storm_dir_deg=0.0, air_density=1.0,
                          extrema=None):
    """DREADER patch mode on a full sweep.

    extrema: optional ((x_min, y_min), (x_max, y_max)) km. If given, the inbound/
    outbound points are the valid gates nearest these positions (user-placed)
    instead of the patch circle's extreme values.

    vel: (n_az, n_rng) masked/NaN Doppler velocity (m/s); center_km: (x, y)
    north-up km; base_diam_km: couplet diameter whose multiple (numdiam) sets
    the circular patch radius. Returns a dict (SI-ish units, distances in km)
    or None if the patch holds no valid inbound+outbound pair.
    """
    vel = np.ma.filled(np.ma.asarray(vel, float), np.nan)
    az = np.asarray(az_deg, float) % 360.0
    X, Y = _sweep_xy(az, rng_km)
    xc, yc = center_km
    rc = float(np.hypot(xc, yc))
    bc = float(_bearing_deg(xc, yc))
    patch_radius = float(numdiam) * float(base_diam_km) if numdiam and base_diam_km else float(base_diam_km)
    if patch_radius <= 0:
        return None

    d = np.hypot(X - xc, Y - yc)
    if extrema is not None:
        valid = np.flatnonzero(np.isfinite(vel).ravel())
        if valid.size < 2:
            return None
        Xv, Yv = X.ravel()[valid], Y.ravel()[valid]
        def nearest(p):
            return valid[int(np.argmin((Xv - p[0]) ** 2 + (Yv - p[1]) ** 2))]
        i_min, i_max = nearest(extrema[0]), nearest(extrema[1])
        if i_min == i_max:
            return None
    else:
        m = np.isfinite(vel) & (d <= patch_radius)
        if m.sum() < 2:
            return None
        idx = np.flatnonzero(m.ravel())
        vals = vel.ravel()[idx]
        i_min, i_max = idx[int(np.argmin(vals))], idx[int(np.argmax(vals))]
        if vel.ravel()[i_min] >= 0 or vel.ravel()[i_max] <= 0:
            return None

    def info(i):
        ia, ir = np.unravel_index(i, vel.shape)
        return {"v": float(vel[ia, ir]), "az": float(az[ia]), "rng_km": float(rng_km[ir]),
                "x": float(X[ia, ir]), "y": float(Y[ia, ir])}

    mn, mx = info(i_min), info(i_max)
    min_radius = float(np.hypot(mn["x"] - xc, mn["y"] - yc))
    max_radius = float(np.hypot(mx["x"] - xc, mx["y"] - yc))
    diam_km = float(np.hypot(mx["x"] - mn["x"], mx["y"] - mn["y"]))
    if diam_km <= 0:
        return None
    axis_ang = float(_bearing_deg(mx["x"] - mn["x"], mx["y"] - mn["y"]))   # inbound -> outbound
    mean_radius = 0.5 * (min_radius + max_radius)

    asp = aspect_ratio_correction(beamwidth_deg, rc, diam_km)
    min_adj, max_adj = mn["v"] * asp, mx["v"] * asp
    v_mean = 0.5 * (abs(min_adj) + abs(max_adj))
    v_asym = abs(max_adj) - abs(min_adj)
    delta_v = max_adj - min_adj

    d_m = diam_km * KM
    circ = (np.pi / 2.0) * delta_v * d_m          # DREADER: (pi/2)*dV*D  (m^2/s)
    delta_p = air_density * v_mean ** 2           # Pa, cyclostrophic

    sin_term = np.sin(np.deg2rad(axis_ang - bc))
    cos_term = np.cos(np.deg2rad(axis_ang - bc))

    # radius of maximum vorticity: scan annuli, ring min/max pair -> bulk vorticity
    width = annulus_width_km(az, rng_km, rc)
    vvort_max_radius = 0.0
    if width > 0:
        best = -1.0
        for rad in np.arange(max(width, 0.25 * mean_radius), 4.0 * mean_radius + 1e-9, width):
            ex = _ring_extrema(vel, X, Y, xc, yc, rad, width)
            if ex is None:
                continue
            a, b = ex
            dia = float(np.hypot(X.ravel()[b] - X.ravel()[a], Y.ravel()[b] - Y.ravel()[a]))
            if dia < 1.5 * width:      # skip degenerate (near-coincident) pairs
                continue
            ang = _bearing_deg(X.ravel()[b] - X.ravel()[a], Y.ravel()[b] - Y.ravel()[a])
            vv = 2.0 * (vel.ravel()[b] - vel.ravel()[a]) / (dia * KM) * np.sin(np.deg2rad(ang - bc))
            if abs(vv) > best:
                best, vvort_max_radius = abs(vv), dia / 2.0

    # vorticity / divergence / stretching / angular momentum at fractions of mean radius
    levels = []
    for frac in PATCH_RADII_FRACTIONS:
        rad = frac * mean_radius
        row = {"fraction": frac, "radius_km": rad, "vvort": np.nan, "hdiv": np.nan,
               "stretch": np.nan, "angmom": np.nan}
        if abs(frac - 1.0) < 1e-9:
            row["vvort"] = 2.0 * delta_v / d_m * sin_term
            row["hdiv"] = 2.0 * delta_v / d_m * cos_term
            row["stretch"] = -row["vvort"] * row["hdiv"]
            row["angmom"] = mean_radius * KM * v_mean
        else:
            ex = _ring_extrema(vel, X, Y, xc, yc, rad, width) if width > 0 else None
            if ex is not None:
                a, b = ex
                xa, ya, xb, yb = X.ravel()[a], Y.ravel()[a], X.ravel()[b], Y.ravel()[b]
                dia = float(np.hypot(xb - xa, yb - ya))
                if dia > 0:
                    ang = _bearing_deg(xb - xa, yb - ya)
                    dv = vel.ravel()[b] - vel.ravel()[a]
                    row["vvort"] = 2.0 * dv / (dia * KM) * np.sin(np.deg2rad(ang - bc))
                    row["hdiv"] = 2.0 * dv / (dia * KM) * np.cos(np.deg2rad(ang - bc))
                    row["stretch"] = -row["vvort"] * row["hdiv"]
                    row["angmom"] = rad * KM * 0.5 * (abs(vel.ravel()[a]) + abs(vel.ravel()[b]))
        levels.append(row)

    # ground-relative velocities (same convention as TURT's Patch: + C sin(beta - az))
    beta = np.deg2rad(storm_dir_deg)
    min_gr = mn["v"] + storm_speed_ms * np.sin(beta - np.deg2rad(mn["az"]))
    max_gr = mx["v"] + storm_speed_ms * np.sin(beta - np.deg2rad(mx["az"]))
    min_gr_adj = min_adj + storm_speed_ms * np.sin(beta - np.deg2rad(mn["az"]))
    max_gr_adj = max_adj + storm_speed_ms * np.sin(beta - np.deg2rad(mx["az"]))
    largest = {
        "doppler": max(abs(mn["v"]), abs(mx["v"])),
        "adjusted": max(abs(min_adj), abs(max_adj)),
        "ground": max(abs(min_gr), abs(max_gr)),
        "ground_adjusted": max(abs(min_gr_adj), abs(max_gr_adj)),
    }
    return {
        "patch_radius_km": patch_radius, "min": mn, "max": mx,
        "min_radius_km": min_radius, "max_radius_km": max_radius, "mean_radius_km": mean_radius,
        "diam_km": diam_km, "axis_deg": axis_ang, "aspect_corr": asp,
        "min_adj": float(min_adj), "max_adj": float(max_adj), "v_mean": float(v_mean),
        "v_asym": float(v_asym), "delta_v": float(delta_v),
        "circulation_m2s": float(circ), "delta_p_pa": float(delta_p),
        "max_vort_radius_km": float(vvort_max_radius), "levels": levels,
        "min_ground": float(min_gr), "max_ground": float(max_gr),
        "min_ground_adj": float(min_gr_adj), "max_ground_adj": float(max_gr_adj),
        "largest": largest,
        "fscale": {k: f_scale(v) for k, v in largest.items()},
        "efscale": {k: ef_scale(v) for k, v in largest.items()},
        "storm_speed_ms": storm_speed_ms, "storm_dir_deg": storm_dir_deg,
    }


# --------------------------------------------------------- axisymmetric mode
def _axisym_py(vel, az_deg, rng_km, center_km, elev_deg=0.0, delta_km=None,
                         max_radius_km=None, translation=None, air_density=1.0):
    """DREADER axisymmetric mode on a full sweep.

    Rings of width delta_km are stepped by delta_km/2 (overlapping, as in
    DREADER). translation: optional (u, v) m/s of the vortex, removed from VD
    along each gate's beam. Returns {'rings': [...], 'delta_km', ...} or None.
    Each ring dict: radius_km, n, vr, vt, vtot, rvr (radial momentum m^2/s),
    angmom (m^2/s), circ (m^2/s), dp_pa, flow_angle_deg.
    """
    vel = np.ma.filled(np.ma.asarray(vel, float), np.nan)
    az = np.asarray(az_deg, float) % 360.0
    X, Y = _sweep_xy(az, rng_km)
    xc, yc = center_km
    rc = float(np.hypot(xc, yc))
    if delta_km is None or delta_km <= 0:
        delta_km = annulus_width_km(az, rng_km, rc)
    if delta_km <= 0:
        return None
    if not max_radius_km or max_radius_km <= 0:
        max_radius_km = float(np.max(rng_km))

    az_g = np.broadcast_to(np.deg2rad(az)[:, None], vel.shape)
    vd = vel / max(np.cos(np.deg2rad(elev_deg)), 1e-3)
    if translation is not None:
        u, v = translation
        vd = vd - (u * np.sin(az_g) + v * np.cos(az_g))
    dx, dy = X - xc, Y - yc
    dist = np.hypot(dx, dy)
    phi = np.arctan2(dx, dy)               # compass bearing centre -> gate (rad)
    t_coef = np.sin(phi - az_g)
    r_coef = np.cos(phi - az_g)
    good = np.isfinite(vd)

    rings, ring_r, prev_n = [], 0.0, 0
    while ring_r < max_radius_km:
        ring_r += delta_km / 2.0
        m = good & (dist > ring_r - delta_km / 2.0) & (dist < ring_r + delta_km / 2.0)
        n = int(m.sum())
        if n >= MIN_SAMPLES:
            A = np.column_stack([r_coef[m], t_coef[m]])
            sol, *_ = np.linalg.lstsq(A, vd[m], rcond=None)
            if np.linalg.cond(A) < 1e6:
                vr, vt = float(sol[0]), float(sol[1])
                r_m = ring_r * KM
                rings.append({
                    "radius_km": ring_r, "n": n, "vr": vr, "vt": vt,
                    "vtot": float(np.hypot(vr, vt)), "rvr": r_m * vr, "angmom": r_m * vt,
                    "circ": 2.0 * np.pi * r_m * vt, "dp_pa": air_density * vt * vt,
                    "flow_angle_deg": float(np.degrees(np.arctan2(vr, vt))),
                })
        # DREADER stop rule: stop when sample count collapses (edge of data)
        if prev_n and 2 * n < prev_n and ring_r > 2 * delta_km:
            break
        prev_n = n
    if not rings:
        return None
    vts = np.array([r["vt"] for r in rings])
    k = int(np.argmax(np.abs(vts)))
    return {"rings": rings, "delta_km": float(delta_km), "center_km": (xc, yc),
            "vt_max": float(vts[k]), "r_vt_max_km": rings[k]["radius_km"],
            "translation": translation}


def compute_dreader_patch(vel, az_deg, rng_km, center_km, base_diam_km, beamwidth_deg=1.0,
                          numdiam=1.0, storm_speed_ms=0.0, storm_dir_deg=0.0, air_density=1.0,
                          extrema=None):
    """Patch dynamics (see _patch_py for the full description); runs in libgurt when available."""
    nat = _nat()
    if nat is None:
        return _patch_py(vel, az_deg, rng_km, center_km, base_diam_km, beamwidth_deg, numdiam,
                         storm_speed_ms, storm_dir_deg, air_density, extrema)
    az = np.asarray(az_deg, float) % 360.0
    res = nat.patch(np.ma.asarray(vel, float), az, rng_km, center_km[0], center_km[1], base_diam_km,
                    beamwidth_deg, numdiam, storm_speed_ms, storm_dir_deg, air_density, extrema)
    if res is None:
        return None
    r, lev = res
    levels = [{"fraction": float(l[0]), "radius_km": float(l[1]), "vvort": float(l[2]),
               "hdiv": float(l[3]), "stretch": float(l[4]), "angmom": float(l[5])} for l in lev]
    largest = {"doppler": r["largest_doppler"], "adjusted": r["largest_adjusted"],
               "ground": r["largest_ground"], "ground_adjusted": r["largest_ground_adjusted"]}
    return {
        "patch_radius_km": r["patch_radius_km"],
        "min": {"v": r["v_min"], "az": r["az_min"], "rng_km": r["rng_min"], "x": r["x_min"], "y": r["y_min"]},
        "max": {"v": r["v_max"], "az": r["az_max"], "rng_km": r["rng_max"], "x": r["x_max"], "y": r["y_max"]},
        "min_radius_km": r["min_radius_km"], "max_radius_km": r["max_radius_km"],
        "mean_radius_km": r["mean_radius_km"], "diam_km": r["diam_km"], "axis_deg": r["axis_deg"],
        "aspect_corr": r["aspect_corr"], "min_adj": r["min_adj"], "max_adj": r["max_adj"],
        "v_mean": r["v_mean"], "v_asym": r["v_asym"], "delta_v": r["delta_v"],
        "circulation_m2s": r["circulation_m2s"], "delta_p_pa": r["delta_p_pa"],
        "max_vort_radius_km": r["max_vort_radius_km"], "levels": levels,
        "min_ground": r["min_ground"], "max_ground": r["max_ground"],
        "min_ground_adj": r["min_ground_adj"], "max_ground_adj": r["max_ground_adj"],
        "largest": largest,
        "fscale": {k: f_scale(v) for k, v in largest.items()},
        "efscale": {k: ef_scale(v) for k, v in largest.items()},
        "storm_speed_ms": storm_speed_ms, "storm_dir_deg": storm_dir_deg,
    }


def compute_axisymmetric(vel, az_deg, rng_km, center_km, elev_deg=0.0, delta_km=None,
                         max_radius_km=None, translation=None, air_density=1.0):
    """Axisymmetric mode (see _axisym_py for the full description); runs in libgurt when available."""
    nat = _nat()
    if nat is None:
        return _axisym_py(vel, az_deg, rng_km, center_km, elev_deg, delta_km, max_radius_km,
                          translation, air_density)
    az = np.asarray(az_deg, float) % 360.0
    xc, yc = center_km
    if delta_km is None or delta_km <= 0:
        delta_km = nat.annulus_width(az, rng_km, float(np.hypot(xc, yc)))
    if delta_km <= 0:
        return None
    if not max_radius_km or max_radius_km <= 0:
        max_radius_km = float(np.max(rng_km))
    tab = nat.axisym(np.ma.asarray(vel, float), az, rng_km, xc, yc, elev_deg, delta_km, max_radius_km,
                     translation, air_density, MIN_SAMPLES)
    if not len(tab):
        return None
    rings = [dict(zip(nat.AXI_COLS, (float(x) for x in row))) for row in tab]
    for r in rings:
        r["n"] = int(r["n"])
    vts = tab[:, 3]
    k = int(np.argmax(np.abs(vts)))
    return {"rings": rings, "delta_km": float(delta_km), "center_km": (xc, yc),
            "vt_max": float(vts[k]), "r_vt_max_km": rings[k]["radius_km"],
            "translation": translation}


def axisym_to_csv(res, path):
    """Write the ring table in DREADER-like columns."""
    cols = ["radius_km", "n", "vr", "vt", "vtot", "rvr", "angmom", "circ", "dp_pa", "flow_angle_deg"]
    names = ["Radius(km)", "NumSamp", "RadialVel(m/s)", "TangentVel(m/s)", "HorizVel(m/s)",
             "RadialMom(m^2/s)", "AngMom(m^2/s)", "Circul(m^2/s)", "PressureDef(Pa)", "FlowAngle(deg)"]
    with open(path, "w") as fh:
        fh.write(",".join(names) + "\n")
        for r in res["rings"]:
            fh.write(",".join(f"{r[c]:.4f}" if c != "n" else str(r[c]) for c in cols) + "\n")
