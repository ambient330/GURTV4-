# SPDX-License-Identifier: GPL-2.0-only
# GURT - Copyright (C) the GURT authors.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the
# Free Software Foundation. It is distributed WITHOUT ANY WARRANTY; see the
# file LICENSE for details.
"""DREADER cross-section extraction (extractor modes 4 and 5) for GURT.

Ported from the DREADER C source (`extract_radxsec` and `extract_angxsec` in
procmatrix.c, `calc_dist` / `calc_cartesian` in coordinates.c).

radial_xsection   -- every gate along one ray (DREADER "radial cross-section").
angular_xsection  -- every ray at one range, centered on a chosen ray, with the
                     DREADER arc quantities (arc position / length / chord) and,
                     for a velocity field, the vortex dynamics:

    normalized distance   |chord / chord-to-max-|V||
    shear vorticity       (V[i+1] - V[i-1]) / dist(gate[i+1], gate[i-1])
    curvature vorticity   V / chord      (sign flips on the low-azimuth side)
    vertical vorticity    shear + curvature
    angular momentum      |V| * chord
    circulation           2 pi |V| * chord
    pressure deficit      rho V^2

Differences from the C code, all deliberate:
  * Rays are ordered by azimuth rather than by array index, so the arc is
    contiguous for sweeps that start mid-circle and for counter-clockwise scans
    (the C code's direct = -1 case collapses to the same result).
  * Neighbour rays are looked up circularly, but a neighbour more than 3x the
    median beam spacing away (a sector-scan edge) counts as missing.
  * Slant range / flat geometry as in calc_cartesian; a 4/3-earth beam height
    is added for the radial output.
"""

import csv
import math
import re

import numpy as np

TOL = 1.0e-6
EARTH_43 = 8483000.0          # DREADER's APPEARTHRAD (4/3 earth radius)

VELOCITY_NAMES = {"VEL", "VE", "VU", "VR", "V", "VELH", "VELOCITY"}


def is_velocity_field(name):
    """True for a velocity field: VEL, VE, VELH ... and derived names built on one, e.g. VE-climo,
    VEL_despeckled, VE.filtered (any leading word that is a velocity name)."""
    n = str(name).upper()
    if n in VELOCITY_NAMES or n.startswith("VEL") or "VELOCITY" in n:
        return True
    return re.split(r"[^A-Z0-9]+", n)[0] in VELOCITY_NAMES


def _wrap180(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def _field_col(arr, idx_rays, gate):
    col = np.ma.filled(np.ma.asarray(arr[:, gate], float), np.nan)
    return col[idx_rays] if idx_rays is not None else col


def _cart(rng_m, az_deg, el_deg):
    a, e = np.radians(az_deg), np.radians(el_deg)
    return np.array([rng_m * np.sin(a) * np.cos(e),
                     rng_m * np.cos(a) * np.cos(e),
                     rng_m * np.sin(e)])


def _dist(r1, a1, e1, r2, a2, e2):
    return float(np.linalg.norm(_cart(r2, a2, e2) - _cart(r1, a1, e1)))


def _elev(sweep, n):
    el = getattr(sweep, "elevation", None)
    if el is None:
        return np.full(n, float(getattr(sweep, "fixed_angle", 0.0) or 0.0))
    return np.asarray(el, float)


def _nearest_ray(az, az0):
    return int(np.argmin(np.abs(_wrap180(az - az0))))


def _nearest_gate(rng_m, r0):
    return int(np.argmin(np.abs(rng_m - r0)))


# ───────────────────────────────────────────────────────────────── radial
def radial_xsection(sweep, azimuth_deg):
    """All gates along the ray nearest *azimuth_deg*.

    Returns a dict: ray, azimuth, elevation, range_m (array), horiz_m, height_m,
    and ``fields`` {name: values with NaN for bad/missing}.
    """
    az = np.asarray(sweep.azimuth, float) % 360.0
    rng = np.asarray(sweep.ranges, float)
    if az.size == 0 or rng.size == 0:
        raise ValueError("Sweep has no data.")
    el = _elev(sweep, az.size)
    ray = _nearest_ray(az, azimuth_deg)
    e = float(el[ray])
    horiz = rng * math.cos(math.radians(e))
    z = rng * math.sin(math.radians(e))
    height = np.sqrt(rng ** 2 + EARTH_43 ** 2 + 2.0 * EARTH_43 * z) - EARTH_43
    fields = {}
    for name, arr in sweep.fields.items():
        if arr is None or arr.shape[0] != az.size:
            continue
        fields[name] = np.ma.filled(np.ma.asarray(arr[ray], float), np.nan)[:rng.size]
    return {"kind": "radial", "ray": ray, "azimuth": float(az[ray]), "elevation": e,
            "range_m": rng, "horiz_m": horiz, "height_m": height, "fields": fields}


# ───────────────────────────────────────────────────────────────── angular
def angular_xsection(sweep, azimuth_deg, range_m, max_radius_m=0.0,
                     velocity_field=None, air_density=1.0):
    """All rays within *max_radius_m* (arc length) of the center ray, at one range.

    *max_radius_m* <= 0 uses every ray. *velocity_field* names the field the
    vortex dynamics are computed from (None = skip them).
    """
    az = np.asarray(sweep.azimuth, float) % 360.0
    rng = np.asarray(sweep.ranges, float)
    n = az.size
    if n < 3 or rng.size == 0:
        raise ValueError("Sweep has too few rays.")
    el = _elev(sweep, n)
    rc = _nearest_ray(az, azimuth_deg)
    gc = _nearest_gate(rng, range_m)
    R = float(rng[gc])
    az_c, el_c = float(az[rc]), float(el[rc])

    d_az = _wrap180(az - az_c)                    # signed offset from the center ray
    if max_radius_m and max_radius_m > TOL:
        half = math.degrees(max_radius_m / R)
        win = np.where(np.abs(d_az) <= half)[0] if half < 180.0 else np.arange(n)
    else:
        win = np.arange(n)
    win = win[np.argsort(d_az[win], kind="stable")]       # contiguous arc, low -> high azimuth
    if win.size < 3:
        raise ValueError("Fewer than 3 rays inside the arc window.")

    # circular azimuth ordering of the WHOLE sweep, for the +/-1 neighbours
    order = np.argsort(az, kind="stable")
    pos = np.empty(n, int)
    pos[order] = np.arange(n)
    gaps = np.diff(np.r_[az[order], az[order][0] + 360.0])
    med_gap = float(np.median(gaps[gaps > 0])) if np.any(gaps > 0) else 1.0

    def neighbour(r, step):
        p = pos[r]
        q = (p + step) % n
        gap = gaps[p] if step > 0 else gaps[(p - 1) % n]
        return int(order[q]) if gap <= 3.0 * med_gap else None

    arcang = d_az[win]
    arcpos = np.radians(arcang) * R
    arclen = np.abs(arcpos)
    chord = np.array([_dist(R, az_c, el_c, R, az[r], el[r]) for r in win])

    fields = {}
    for name, arr in sweep.fields.items():
        if arr is None or arr.shape[0] != n:
            continue
        fields[name] = _field_col(arr, win, gc)

    out = {"kind": "angular", "center_ray": rc, "center_gate": gc,
           "azimuth": az_c, "elevation": el_c, "range_m": R,
           "azimuths": az[win], "elevations": el[win],
           "arcpos_m": arcpos, "arclen_m": arclen, "chord_m": chord,
           "rays": win, "fields": fields, "velocity_field": None,
           "half_deg": float(np.max(np.abs(arcang))) if win.size else 0.0,
           "full_circle": bool(win.size == n)}

    if velocity_field and velocity_field in sweep.fields:
        v_all = np.ma.filled(np.ma.asarray(sweep.fields[velocity_field][:, gc], float), np.nan)
        v = v_all[win]
        good = np.isfinite(v)

        # chord to the strongest |V| in the window (first one wins ties, as in the C code)
        vmax_len = 0.0
        if good.any():
            k = int(np.nanargmax(np.where(good, np.abs(v), -np.inf)))
            vmax_len = float(chord[k])
        normd = (np.abs(chord / vmax_len) if vmax_len > TOL
                 else np.full(win.size, np.nan))

        shear = np.full(win.size, np.nan)
        curv = np.full(win.size, np.nan)
        vert = np.full(win.size, np.nan)
        for i, r in enumerate(win):
            if r == rc or chord[i] <= TOL or not good[i]:
                continue
            rm, rp = neighbour(r, -1), neighbour(r, +1)
            if rm is None or rp is None:
                continue
            vm, vp = v_all[rm], v_all[rp]
            if not (np.isfinite(vm) and np.isfinite(vp)):
                continue
            dx = _dist(R, az[rp], el[rp], R, az[rm], el[rm])
            if dx <= TOL:
                continue
            shear[i] = (vp - vm) / dx
            c = v[i] / chord[i]
            curv[i] = -c if arcang[i] < 0 else c
            vert[i] = shear[i] + curv[i]

        angmom = np.where(good, np.abs(v) * chord, np.nan)
        circ = np.where(good, 2.0 * math.pi * np.abs(v) * chord, np.nan)
        dp = np.where(good, air_density * v * v, np.nan)
        out.update({"velocity_field": velocity_field, "norm_dist": normd,
                    "shear_vort": shear, "curv_vort": curv, "vert_vort": vert,
                    "ang_mom": angmom, "circulation": circ, "pressure_def": dp,
                    "vmax_chord_m": vmax_len})
    return out


# ───────────────────────────────────────────────────────────────── CSV
def xsection_to_csv(xs, path):
    """Write a cross-section in DREADER's column style (values only, comma-delimited)."""
    names = list(xs["fields"])
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        if xs["kind"] == "radial":
            w.writerow(["AzimuthAngle(deg)", "ElevationAng(deg)", "Range(m)",
                        "HorizDist(m)", "BeamHeightAGL(m)"] + names)
            for g in range(len(xs["range_m"])):
                w.writerow([f"{xs['azimuth']:.2f}", f"{xs['elevation']:.2f}",
                            f"{xs['range_m'][g]:.1f}", f"{xs['horiz_m'][g]:.1f}",
                            f"{xs['height_m'][g]:.1f}"] +
                           [_f(xs["fields"][n][g]) for n in names])
            return
        dyn = []
        if xs.get("velocity_field"):
            dyn = [("NormRadMaxTanVel", "norm_dist"), ("ShearVort(1/s)", "shear_vort"),
                   ("CurvitVort(1/s)", "curv_vort"), ("VertVort(1/s)", "vert_vort"),
                   ("AngMom(m^2/s)", "ang_mom"), ("Circul(m^2/s)", "circulation"),
                   ("PressureDef(Pa)", "pressure_def")]
        w.writerow(["AzimuthAngle(deg)", "ElevationAng(deg)", "Range(m)",
                    "ArcPosition(m)", "ArcLength(m)", "ChordLength(m)"] + names +
                   [d[0] for d in dyn])
        for i in range(len(xs["rays"])):
            w.writerow([f"{xs['azimuths'][i]:.2f}", f"{xs['elevations'][i]:.2f}",
                        f"{xs['range_m']:.1f}", f"{xs['arcpos_m'][i]:.1f}",
                        f"{xs['arclen_m'][i]:.1f}", f"{xs['chord_m'][i]:.1f}"] +
                       [_f(xs["fields"][n][i]) for n in names] +
                       [_f(xs[k][i], 6) for _, k in dyn])


def _f(x, nd=2):
    return "" if not np.isfinite(x) else f"{x:.{nd}f}"
