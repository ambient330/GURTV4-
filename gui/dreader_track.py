# SPDX-License-Identifier: GPL-2.0-only
# GURT - Copyright (C) the GURT authors.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the
# Free Software Foundation. It is distributed WITHOUT ANY WARRANTY; see the
# file LICENSE for details.
"""DREADER circulation detection, isolation and tracking for GURT.

Ported from the DREADER C source (extractor modes -detect, -isolate, -track:
`detect_center`, `isolate_center`, `track_center` in algorithms.c, with
`calc_elevcorr`, `calc_latlon`, `calc_greatcirc`, `calc_percentile`,
`eval_linearspline` helpers).

detect_sweep   One sweep -> list of circulation detections.  Repeatedly finds the
               largest azimuthal velocity difference (gate-to-gate, then ray pairs
               +/-k beams apart) whose separation is <= max diameter, refines the
               center from the median positions of the strongest inbound and
               outbound gates in a box around it, masks that box, and repeats
               until the difference falls under the minimum.
isolate        Chains detections from different sweeps (in time order) into
               individual circulations by time / elevation / lat-lon proximity.
track          Per isolated circulation: translation speed and direction from the
               piecewise-linear center positions in layers of height, and the
               vertical tilt (inclination / orientation, with and without a
               motion correction) between successive tilts of the same volume.

Deliberate differences from the C code:
  * The refinement box is selected by azimuth offset instead of ray-index range
    (the C index range collapses to the whole circle when a box straddles the
    start of the sweep), and median azimuths are taken relative to the center ray
    so the 0/360 seam does not corrupt them.
  * Great-circle distances use the haversine form in double precision (the C code
    keeps lat/lon in float32, which is noisy over a 1-2 s displacement).
  * A detection whose center cannot be matched to a gate is dropped (the C code
    aborts the whole detection run).
  * The detection loop is capped so a zero threshold cannot spin forever.
"""

import csv
import math

import numpy as np

TOL = 1.0e-6
EARTHRAD = 6371000.0
APPEARTHRAD = 8483000.0          # 4/3 earth radius
MINLOCS = 4                      # fewer observations -> no translation estimate
MAXTRAN = 25.0                   # m/s; faster "translation" is discarded
MAX_PASSES = 5000


# ───────────────────────────────────────────────────────────── geometry helpers
def _wrap180(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def _unit(az_deg, el_deg):
    a, e = np.radians(az_deg), np.radians(el_deg)
    return np.stack([np.sin(a) * np.cos(e), np.cos(a) * np.cos(e), np.sin(e)], axis=1)


def _erfcx(x):
    """exp(x^2) * erfc(x), safe for large x."""
    if x < 25.0:
        return math.exp(x * x) * math.erfc(x)
    return 1.0 / (x * math.sqrt(math.pi)) * (1.0 - 1.0 / (2.0 * x * x))


def correct_elevation(elev_deg, beamwidth_deg):
    """calc_elevcorr: beam-pattern weighted elevation correction."""
    if beamwidth_deg is None or beamwidth_deg <= 0:
        return float(elev_deg)
    e = math.radians(elev_deg)
    b = abs(2.0 * math.sqrt(math.log(2.0)) / math.radians(beamwidth_deg))
    if e <= 0 or e >= math.pi:
        if e >= math.pi:
            e -= 2.0 * math.pi
        x = abs(b * e)
        term = 1.0 / (math.sqrt(math.pi) * b * _erfcx(x)) * math.exp(-0.0) \
            if False else _inv_term_neg(b, e)
    else:
        term = _inv_term_pos(b, e)
    return math.degrees(e + term)


def _inv_term_neg(b, e):
    # 1 / (sqrt(pi) b exp(b^2 e^2) erfc(|b e|))  = 1 / (sqrt(pi) b erfcx(|b e|) * exp(0))
    # exp(x^2) erfc(x) = erfcx(x) so the product in the C source is exactly erfcx.
    return 1.0 / (math.sqrt(math.pi) * b * _erfcx(abs(b * e)))


def _inv_term_pos(b, e):
    x2 = (b * e) ** 2
    if x2 > 700.0:
        return 0.0
    return 1.0 / (math.sqrt(math.pi) * b * math.exp(x2) * (2.0 - math.erfc(abs(b * e))))


def gate_position(rng_m, az_deg, el_deg):
    """calc_cartesian: hdist, vdist (4/3 earth) of a gate."""
    e = math.radians(el_deg)
    z = rng_m * math.sin(e)
    hdist = rng_m * math.cos(e)
    vdist = math.sqrt(rng_m * rng_m + APPEARTHRAD ** 2 + 2.0 * APPEARTHRAD * z) - APPEARTHRAD
    return hdist, vdist


def gate_latlon(rng_m, az_deg, el_deg, lat0, lon0):
    """calc_latlon."""
    h, v = gate_position(rng_m, az_deg, el_deg)
    g = math.asin(max(-1.0, min(1.0, h / (v + EARTHRAD))))
    p0, l0, a = math.radians(lat0), math.radians(lon0), math.radians(az_deg)
    lat = math.asin(math.sin(p0) * math.cos(g) + math.cos(p0) * math.sin(g) * math.cos(a))
    lon = l0 + math.atan2(math.sin(g) * math.cos(p0) * math.sin(a),
                          math.cos(g) - math.sin(p0) * math.sin(lat))
    while lon > math.pi:
        lon -= 2 * math.pi
    while lon < -math.pi:
        lon += 2 * math.pi
    return math.degrees(lat), math.degrees(lon)


def great_circle(lat1, lon1, lat2, lon2):
    """Distance (m) and meteorological bearing (deg from north, clockwise)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    a = (math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    d = 2.0 * EARTHRAD * math.asin(min(1.0, math.sqrt(a)))
    brg = math.degrees(math.atan2(math.sin(dl) * math.cos(p2),
                                  math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)))
    return d, brg % 360.0


def percentile(values, pct):
    """calc_percentile: mean of the floor / ceil ranks."""
    v = np.sort(np.asarray(values, float))
    if v.size == 0:
        return float("nan")
    idx = (pct / 100.0) * (v.size - 1)
    return float(0.5 * (v[int(math.floor(idx))] + v[int(math.ceil(idx))]))


def _median_spacing(az):
    d = _wrap180(np.diff(az))
    d = d[np.abs(d) > TOL]
    return float(abs(np.median(d))) if d.size else 1.0


def _nearest(arr, x):
    return int(np.argmin(np.abs(np.asarray(arr) - x)))


# ───────────────────────────────────────────────────────────── detection
def _best_pair(vel, U, rng, skip, maxdiam):
    """Largest |dV| over (ray+k, ray-k) pairs (k=0: ray and ray-1) with chord <= maxdiam.
    Returns best_dv, best_k, best_chord arrays shaped (nr, ng-1)."""
    nr, ng = vel.shape
    G = ng - 1
    bd = np.zeros((nr, G))
    bk = np.full((nr, G), -1, int)
    bc = np.zeros((nr, G))
    rg = rng[:G][None, :]
    k = 0
    while True:
        if k == 0:
            r0, r1 = 1, nr
            ia, ib = np.arange(1, nr), np.arange(0, nr - 1)
        else:
            r0, r1 = k, nr - k
            if r1 <= r0:
                break
            ia, ib = np.arange(r0, r1) + k, np.arange(r0, r1) - k
        norm = np.linalg.norm(U[ia] - U[ib], axis=1)
        chord = norm[:, None] * rg
        active = chord <= maxdiam
        if not active.any():
            break
        va, vb = vel[ia, :G], vel[ib, :G]
        ok = (active & np.isfinite(va) & np.isfinite(vb)
              & ~skip[ia, :G] & ~skip[ib, :G])
        dv = np.where(ok, np.abs(va - vb), 0.0)
        sub_d, sub_k, sub_c = bd[r0:r1], bk[r0:r1], bc[r0:r1]
        upd = ok & (dv > sub_d)
        sub_d[upd] = dv[upd]
        sub_k[upd] = k
        sub_c[upd] = chord[upd]
        k += 1
    return bd, bk, bc


def detect_sweep(az, el, rng_m, vel, ray_time, radar, params):
    """Find circulations in one sweep.

    az, el  : per-ray azimuth / elevation (deg);   rng_m: per-gate slant range (m)
    vel     : (rays, gates) velocity, NaN = bad     ray_time: per-ray unix time
    radar   : {"lat", "lon"}
    params  : maxdiam_m, mindeltav, extremapct, maxtors (0 = no limit), beamwidth
    Returns a list of detection dicts, strongest first.
    """
    az = np.asarray(az, float) % 360.0
    el = np.asarray(el, float)
    rng = np.asarray(rng_m, float)
    vel = np.asarray(vel, float)
    nr, ng = vel.shape
    if nr < 4 or ng < 3:
        return []
    maxdiam = float(params["maxdiam_m"])
    mindv = float(params["mindeltav"])
    pct = float(params["extremapct"])
    maxtors = int(params.get("maxtors", 0) or 0)
    bw = float(params.get("beamwidth", 1.0))
    if mindv <= 0:
        raise ValueError("Minimum delta-V must be > 0.")

    U = _unit(az, el)
    skip = np.zeros((nr, ng), bool)
    angdist = _median_spacing(az)
    med_gate = float(np.median(np.diff(rng))) if ng > 1 else 1.0
    out = []
    numtor = 0
    passes = 0
    while (maxtors == 0 or numtor < maxtors) and passes < MAX_PASSES:
        passes += 1
        bd, bk, bc = _best_pair(vel, U, rng, skip, maxdiam)
        flat = int(np.argmax(bd.T.ravel()))          # first max in (gate, ray) order, like the C loops
        gc, rc = divmod(flat, nr)
        maxdv = float(bd[rc, gc])
        if maxdv < mindv:
            break
        k, chord = int(bk[rc, gc]), float(bc[rc, gc])
        rang_c, az_c = float(rng[gc]), float(az[rc])

        # initial (first-guess) pair
        if k > 0:
            i1, i2 = rc - k, rc + k
        else:
            i1, i2 = rc - 1, rc
        v1, v2 = float(vel[i1, gc]), float(vel[i2, gc])
        if v2 >= v1:
            imin, imax, initmin, initmax = i1, i2, v1, v2
        else:
            imin, imax, initmin, initmax = i2, i1, v2, v1

        # refinement box
        half = math.degrees(chord / rang_c) if rang_c > 0 else 360.0
        d_az = _wrap180(az - az_c)
        rays = np.where(np.abs(d_az) <= half)[0] if half < 180.0 else np.arange(nr)
        if rc not in rays:
            rays = np.append(rays, rc)
        g0 = _nearest(rng, max(rang_c - chord, 0.0))
        g1 = _nearest(rng, rang_c + chord)
        gates = np.arange(min(g0, g1), max(g0, g1) + 1)
        box = np.ix_(rays, gates)
        box_v = vel[box]
        good = np.isfinite(box_v)
        pre_skip = skip.copy() if False else None     # (checked below without a full copy)
        numtor_inc = False

        if good.any():
            vals = box_v[good]
            maxvel = percentile(vals, pct)
            minvel = percentile(vals, 100.0 - pct)
            rr = np.broadcast_to(rng[gates][None, :], box_v.shape)
            dd = np.broadcast_to(d_az[rays][:, None], box_v.shape)
            hi = good & (box_v >= maxvel)
            lo = good & (box_v <= minvel)
            n_hi, n_lo = int(hi.sum()), int(lo.sum())
            med_max_r, med_max_d = float(np.median(rr[hi])), float(np.median(dd[hi]))
            med_min_r, med_min_d = float(np.median(rr[lo])), float(np.median(dd[lo]))
            cen_r = 0.5 * (med_max_r + med_min_r)
            cen_az = (az_c + 0.5 * (med_max_d + med_min_d)) % 360.0
            ray = _nearest(_wrap180(az - cen_az), 0.0)
            gate = _nearest(rng, cen_r)
            matched = (abs(_wrap180(az[ray] - cen_az)) <= abs(10.0 * angdist)
                       and abs(rng[gate] - cen_r) < abs(med_gate))
            already = bool(skip[ray, gate]) if matched else True
            if matched and not already:
                out.append(_make_detection(
                    az, el, rng, vel, ray_time, radar, bw, ray, gate, rc, gc,
                    initmin, imin, initmax, imax, maxdv, chord,
                    pct, minvel, n_lo, med_min_d + az_c, med_min_r,
                    maxvel, n_hi, med_max_d + az_c, med_max_r))
            numtor += 1
        skip[box] = True
    out.sort(key=lambda d: -d["delta_v"])
    return out


def _make_detection(az, el, rng, vel, ray_time, radar, bw, ray, gate, rc, gc,
                    initmin, imin, initmax, imax, maxdv, chord,
                    pct, minvel, n_lo, med_min_az, med_min_r,
                    maxvel, n_hi, med_max_az, med_max_r):
    r, a, e = float(rng[gate]), float(az[ray]), float(el[ray])
    ce = correct_elevation(e, bw)
    h, v = gate_position(r, a, e)
    ch, cv = gate_position(r, a, ce)
    lat, lon = gate_latlon(r, a, e, radar["lat"], radar["lon"])
    clat, clon = gate_latlon(r, a, ce, radar["lat"], radar["lon"])
    return {
        "time": float(ray_time[ray]), "az": a, "elev": e, "range_m": r, "corr_elev": ce,
        "hdist": h, "vdist": v, "corr_hdist": ch, "corr_vdist": cv,
        "lat": lat, "lon": lon, "corr_lat": clat, "corr_lon": clon,
        "init_min": initmin, "init_min_az": float(az[imin]), "init_min_rng": float(rng[gc]),
        "init_max": initmax, "init_max_az": float(az[imax]), "init_max_rng": float(rng[gc]),
        "delta_v": maxdv, "diameter_m": chord, "shear": (2.0 * maxdv / chord) if chord > TOL else float("nan"),
        "fg_az": float(az[rc]), "fg_rng": float(rng[gc]),
        "min_pct": 100.0 - pct, "min_vel": minvel, "n_min": n_lo,
        "med_min_az": med_min_az % 360.0, "med_min_rng": med_min_r,
        "max_pct": pct, "max_vel": maxvel, "n_max": n_hi,
        "med_max_az": med_max_az % 360.0, "med_max_rng": med_max_r,
        "n_total": n_lo + n_hi,
        "trans_speed": float("nan"), "trans_dir": float("nan"),
        "vert_diff": float("nan"), "horiz_diff": float("nan"),
        "incl": float("nan"), "orient": float("nan"),
        "corr_incl": float("nan"), "corr_orient": float("nan"),
        "track": 0,
    }


# ───────────────────────────────────────────────────────────── isolation
def isolate(dets, maxtimediff, maxelevdiff, maxlatdiff, maxlondiff, mindetects=0):
    """Assign a circulation id (1..N) to each detection, 0 = rejected.

    ``dets`` must be in time order and carry ``file`` (a hashable sweep id),
    ``time``, ``elev``, ``corr_lat``, ``corr_lon``.  Mirrors isolate_center: from
    the first unassigned detection, walk forward over detections in OTHER sweeps
    and link the first one close enough in time and space; a larger elevation
    jump is allowed twice the lat/lon tolerance.  The walk stops at the first
    candidate that is too far away in time.
    """
    n = len(dets)
    mark = [0] * n                       # 0 unprocessed, -1 rejected, >0 circulation id
    tor = 0
    while 0 in mark:
        c = mark.index(0)
        tor += 1
        ndet = 1
        cur = c
        i = c + 1
        while i < n:
            if dets[i]["file"] == dets[cur]["file"] or mark[i] != 0:
                i += 1
                continue
            d, p = dets[i], dets[cur]
            if abs(d["time"] - p["time"]) <= maxtimediff:
                near_elev = abs(d["elev"] - p["elev"]) <= maxelevdiff
                f = 1.0 if near_elev else 2.0
                if (abs(d["corr_lat"] - p["corr_lat"]) <= f * maxlatdiff
                        and abs(d["corr_lon"] - p["corr_lon"]) <= f * maxlondiff):
                    ndet += 1
                    mark[cur] = tor
                    mark[i] = tor
                    cur = i
                i += 1
            else:
                break
        mark[cur] = tor
        if ndet < mindetects:
            for j in range(n):
                if mark[j] == tor:
                    mark[j] = -1
            tor -= 1
    for d, m in zip(dets, mark):
        d["track"] = max(m, 0)
    return tor


# ───────────────────────────────────────────────────────────── tracking
def track(dets, deltz=0.0):
    """Translation and tilt for ONE isolated circulation (list of its detections,
    time ordered).  Fills trans_speed/dir and vert_diff ... corr_orient in place."""
    n = len(dets)
    if n == 0:
        return dets
    hv = np.array([d["corr_vdist"] for d in dets])
    minel = min(0.0, float(hv.min()))                 # C code starts the minimum at 0
    maxel = max(0.0, float(hv.max()))
    for d in dets:
        d["trans_speed"] = 0.0
        d["trans_dir"] = 0.0

    if abs(deltz) > TOL:
        nlay = max(1, int(math.ceil((maxel - minel) / deltz)))
        bounds = [(minel + i * deltz, minel + (i + 1) * deltz) for i in range(nlay)]
    else:
        q1, med, q3 = (percentile(hv, p) for p in (25, 50, 75))
        bounds = [(minel, q1), (q1, med), (med, q3), (q3, maxel)]

    for bot, top in bounds:
        idx = [i for i in range(n) if bot <= dets[i]["corr_vdist"] <= top]
        if len(idx) < 2:
            continue
        times = np.array([round(dets[i]["time"]) for i in idx], float)
        lats = np.array([dets[i]["corr_lat"] for i in idx])
        lons = np.array([dets[i]["corr_lon"] for i in idx])
        t = np.arange(times[0], times[-1] + 1.0)
        ilat = np.interp(t, times, lats)
        ilon = np.interp(t, times, lons)
        nloc = len(idx)
        for loc, i in enumerate(idx):
            s = int(times[loc] - times[0])
            if loc == 0:
                a, b = s, min(s + 1, len(t) - 1)
            elif loc == nloc - 1:
                a, b = max(s - 1, 0), s
            else:
                a, b = max(s - 1, 0), min(s + 1, len(t) - 1)
            dt = t[b] - t[a]
            if nloc >= MINLOCS and dt > 0:
                dist, brg = great_circle(ilat[a], ilon[a], ilat[b], ilon[b])
                spd = dist / dt
            else:
                spd, brg = 0.0, 0.0
            if spd >= MAXTRAN:
                spd, brg = 0.0, 0.0
            dets[i]["trans_speed"], dets[i]["trans_dir"] = float(spd), float(brg)

    # tilt between successive tilts of the same volume
    for i in range(1, n):
        a, b = dets[i - 1], dets[i]
        if b["corr_vdist"] > a["corr_vdist"] and abs(round(b["time"]) - round(a["time"])) < 60:
            t0, t1 = round(a["time"]), round(b["time"])
            h0, az0 = a["corr_hdist"], math.radians(a["az"])
            h1, az1 = b["corr_hdist"], math.radians(b["az"])
            x0, y0 = h0 * math.sin(az0), h0 * math.cos(az0)
            x1, y1 = h1 * math.sin(az1), h1 * math.cos(az1)
            vdiff = b["corr_vdist"] - a["corr_vdist"]
            hdiff = math.hypot(x1 - x0, y1 - y0)
            incl = math.degrees(math.atan(abs(hdiff / vdiff)))
            orient = math.degrees(math.atan2(x1 - x0, y1 - y0)) % 360.0
            # move the upper center back to the lower center's time with the (upper) translation
            tv, ta = b["trans_speed"], math.radians(b["trans_dir"])
            x1c = x1 + (t0 - t1) * tv * math.sin(ta)
            y1c = y1 + (t0 - t1) * tv * math.cos(ta)
            hdc = math.hypot(x1c - x0, y1c - y0)
            b["vert_diff"], b["horiz_diff"] = vdiff, hdiff
            b["incl"], b["orient"] = incl, orient
            b["corr_incl"] = math.degrees(math.atan(abs(hdc / vdiff)))
            b["corr_orient"] = math.degrees(math.atan2(x1c - x0, y1c - y0)) % 360.0
    return dets


def run_pipeline(dets, p):
    """isolate + track every circulation.  ``p``: maxtimediff, maxelevdiff, maxlatdiff,
    maxlondiff, mindetects, deltz.  Returns {id: [detections]} (detections time ordered)."""
    dets.sort(key=lambda d: (d["time"], d["elev"]))
    ntor = isolate(dets, p["maxtimediff"], p["maxelevdiff"], p["maxlatdiff"],
                   p["maxlondiff"], int(p.get("mindetects", 0)))
    tracks = {}
    for d in dets:
        if d["track"] > 0:
            tracks.setdefault(d["track"], []).append(d)
    for tid, lst in tracks.items():
        track(lst, p.get("deltz", 0.0))
    return tracks


# ───────────────────────────────────────────────────────────── CSV
CSV_COLS = [
    ("Track", "track", "{:d}"), ("File", "file_name", "{}"), ("Sweep", "sweep", "{:d}"),
    ("UnixEpochTime(s)", "time", "{:.3f}"),
    ("AzimuthAngle(deg)", "az", "{:.2f}"), ("ElevationAng(deg)", "elev", "{:.2f}"),
    ("Range(m)", "range_m", "{:.1f}"), ("CorrElevAng(deg)", "corr_elev", "{:.2f}"),
    ("HorizDist(m)", "hdist", "{:.1f}"), ("VertDist(m)", "vdist", "{:.1f}"),
    ("CorrHorizDist(m)", "corr_hdist", "{:.1f}"), ("CorrVertDist(m)", "corr_vdist", "{:.1f}"),
    ("GateLat(deg)", "lat", "{:.6f}"), ("GateLon(deg)", "lon", "{:.6f}"),
    ("CorrGateLat(deg)", "corr_lat", "{:.6f}"), ("CorrGateLon(deg)", "corr_lon", "{:.6f}"),
    ("FGuessMin(m/s)", "init_min", "{:.1f}"), ("FGuessMinAz(deg)", "init_min_az", "{:.2f}"),
    ("FGuessMinRa(m)", "init_min_rng", "{:.1f}"),
    ("FGuessMax(m/s)", "init_max", "{:.1f}"), ("FGuessMaxAz(deg)", "init_max_az", "{:.2f}"),
    ("FGuessMaxRa(m)", "init_max_rng", "{:.1f}"),
    ("DeltaVFixRa(m/s)", "delta_v", "{:.1f}"), ("DiameterFixRa(m)", "diameter_m", "{:.1f}"),
    ("SymVertVort(1/s)", "shear", "{:.6f}"),
    ("FirstGuessAz(deg)", "fg_az", "{:.2f}"), ("FirstGuessRa(m)", "fg_rng", "{:.1f}"),
    ("MinVPercent(%)", "min_pct", "{:.2f}"), ("MinVPercntl(m/s)", "min_vel", "{:.1f}"),
    ("NumMinV", "n_min", "{:d}"), ("MedianMinAz(deg)", "med_min_az", "{:.2f}"),
    ("MedianMinRa(m)", "med_min_rng", "{:.1f}"),
    ("MaxVPercent(%)", "max_pct", "{:.2f}"), ("MaxVPercntl(m/s)", "max_vel", "{:.1f}"),
    ("NumMaxV", "n_max", "{:d}"), ("MedianMaxAz(deg)", "med_max_az", "{:.2f}"),
    ("MedianMaxRa(m)", "med_max_rng", "{:.1f}"), ("TotMax", "n_total", "{:d}"),
    ("TransSpeed(m/s)", "trans_speed", "{:.1f}"), ("TransDirect(deg)", "trans_dir", "{:.1f}"),
    ("VertCentDiff(m)", "vert_diff", "{:.1f}"), ("HorizCentDiff(m)", "horiz_diff", "{:.1f}"),
    ("InclinAngle(deg)", "incl", "{:.1f}"), ("OrientAngle(deg)", "orient", "{:.1f}"),
    ("CorrInclinA(deg)", "corr_incl", "{:.1f}"), ("CorrOrientA(deg)", "corr_orient", "{:.1f}"),
]


def _cell(fmt, v):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ""
    try:
        return fmt.format(v)
    except (ValueError, TypeError):
        return str(v)


def detections_to_csv(dets, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([c[0] for c in CSV_COLS])
        for d in dets:
            w.writerow([_cell(f, d.get(k)) for _, k, f in CSV_COLS])
