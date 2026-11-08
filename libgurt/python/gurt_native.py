# SPDX-License-Identifier: GPL-2.0-only
# GURT - Copyright (C) the GURT authors.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the
# Free Software Foundation. It is distributed WITHOUT ANY WARRANTY; see the
# file LICENSE for details.
"""ctypes binding for libgurt (see ../include/gurt.h).

    import gurt_native as gn
    out, n_removed, n_passes = gn.despeckle(masked_array, radius=2, min_good=10)

Every function takes / returns numpy masked arrays (or plain float arrays with
NaN for missing), never modifies its input, and accepts an optional boolean
`region` of the same shape: gates outside it are left untouched (the brush /
polygon footprint), though neighbours there still count as context.

`available()` tells whether the shared library was found; GURT can fall back
to its Python implementations when it was not.
"""

import ctypes as C
import os
import sys

import numpy as np

_LIB = None
_ERR = None


def _candidates():
    here = os.path.dirname(os.path.abspath(__file__))
    names = {"win32": ["gurt.dll", "libgurt.dll"], "darwin": ["libgurt.dylib"]}.get(
        sys.platform, ["libgurt.so"])
    root = os.path.join(here, "..")
    dirs = [os.environ.get("LIBGURT_DIR", ""), here, root, os.path.join(root, "build"),
            os.path.join(root, "build", "Release"), os.path.join(root, "build", "Debug")]
    if sys.platform == "win32":
        dirs.append(os.path.join(root, "prebuilt", "windows-x64"))
    for d in dirs:
        for n in names:
            if d:
                yield os.path.join(d, n)


def _load():
    global _LIB, _ERR
    if _LIB is not None or _ERR is not None:
        return _LIB
    for path in _candidates():
        if os.path.exists(path):
            try:
                lib = C.CDLL(path)
                _bind(lib)
                _LIB = lib
                return _LIB
            except OSError as e:
                _ERR = str(e)
    _ERR = _ERR or ("libgurt not found for this platform (" + sys.platform + "); build it per libgurt/README.md")
    return None


def available():
    return _load() is not None


def _bind(lib):
    f32p = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS,WRITEABLE")
    f32c = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
    f64c = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    f64p = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS,WRITEABLE")
    u8c = np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS")
    opt = lambda t: C.c_void_p            # noqa: E731  (NULL-able pointer argument)
    lib.gurt_version.restype = C.c_char_p
    lib.gurt_despeckle.argtypes = [f32p, C.c_int32, C.c_int32, C.c_int32, C.c_int32, C.c_int32,
                                   C.c_int32, C.c_void_p, C.POINTER(C.c_int64), C.POINTER(C.c_int32)]
    lib.gurt_despeckle.restype = C.c_int
    lib.gurt_deglitch.argtypes = [f32p, C.c_int32, C.c_int32, f64c, C.c_int32, C.c_double,
                                  C.c_int32, C.c_int32, C.c_int32, C.c_void_p, C.c_void_p,
                                  C.c_void_p, C.c_void_p, C.c_void_p,
                                  C.POINTER(C.c_int64), C.POINTER(C.c_int32)]
    lib.gurt_deglitch.restype = C.c_int
    lib.gurt_gapfill.argtypes = [f32p, C.c_int32, C.c_int32, C.c_int32, C.c_int32, C.c_int32,
                                 C.c_void_p, C.c_void_p, C.POINTER(C.c_int64)]
    lib.gurt_gapfill.restype = C.c_int
    try:                                   # tormodel (newer builds)
        i32c = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
        i32 = C.c_int32
        lib.gurt_tor_interp.argtypes = [i32, i32c, f32c, i32, i32, i32, f32p]
        lib.gurt_tor_interp.restype = C.c_int
        lib.gurt_tor_grid.argtypes = [f32c, i32, i32, i32, i32, i32, i32, i32, i32,
                                      C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p]
        lib.gurt_tor_grid.restype = C.c_int
        lib.gurt_tor_maxgrid.argtypes = [f32c, i32, i32, i32, i32, i32, i32, i32, i32, f32p]
        lib.gurt_tor_maxgrid.restype = C.c_int
        lib.gurt_tor_point.argtypes = [f32c, i32, C.c_float, C.c_float, f32p]
        lib.gurt_tor_point.restype = C.c_int
        lib.gurt_tor_latlon.argtypes = [C.c_float, C.c_float, C.c_float, C.c_float,
                                        C.POINTER(C.c_float), C.POINTER(C.c_float)]
        lib.gurt_tor_latlon.restype = C.c_int
        lib.gurt_tor_polar.argtypes = [C.c_float, C.c_float, C.POINTER(C.c_float), C.POINTER(C.c_float)]
        lib.gurt_tor_polar.restype = C.c_int
        lib.gurt_tor_cart.argtypes = [C.c_float, C.c_float, C.POINTER(C.c_float), C.POINTER(C.c_float)]
        lib.gurt_tor_cart.restype = C.c_int
    except AttributeError:
        pass
    try:                                   # axisymmetric analysis (full GURT builds only)
        i32 = C.c_int32
        lib.gurt_annulus_width.argtypes = [f64c, i32, f64c, i32, C.c_double]
        lib.gurt_annulus_width.restype = C.c_double
        lib.gurt_axisym.argtypes = [f64c, i32, i32, f64c, f64c] + [C.c_double] * 5 + \
            [i32, C.c_double, C.c_double, C.c_double, i32, f64p, i32, C.POINTER(C.c_int32)]
        lib.gurt_axisym.restype = C.c_int
        lib.gurt_patch.argtypes = [f64c, i32, i32, f64c, f64c] + [C.c_double] * 8 + \
            [i32] + [C.c_double] * 4 + [f64p, f64p, C.POINTER(C.c_int32)]
        lib.gurt_patch.restype = C.c_int
    except AttributeError:
        pass
    try:                                   # older builds of the library have no unfold
        lib.gurt_unfold.argtypes = [f32p, C.c_int32, C.c_int32, f64c, C.c_int32, C.c_double,
                                    C.c_void_p, C.c_int32, C.c_int32, C.c_void_p, C.c_void_p,
                                    C.c_void_p, C.POINTER(C.c_int64)]
        lib.gurt_unfold.restype = C.c_int
    except AttributeError:
        pass


def has_tormodel():
    lib = _load()
    return lib is not None and hasattr(lib, "gurt_tor_interp")


def has_axisym():
    lib = _load()
    return lib is not None and hasattr(lib, "gurt_axisym")


def has_unfold():
    lib = _load()
    return lib is not None and hasattr(lib, "gurt_unfold")


def version():
    lib = _load()
    return lib.gurt_version().decode() if lib else None


# ── helpers ────────────────────────────────────────────────────────────────
def _to_nan(a):
    """Masked/NaN array -> C-contiguous float32 copy with NaN for missing."""
    m = np.ma.getmaskarray(a)
    x = np.array(np.ma.getdata(a), dtype=np.float32, order="C", copy=True)
    x[m] = np.nan
    return x


def _to_masked(x):
    return np.ma.masked_invalid(x)


def _u8(mask, shape):
    if mask is None:
        return None
    m = np.ascontiguousarray(np.asarray(mask, dtype=bool).astype(np.uint8))
    if m.shape != shape:
        raise ValueError("region/protect mask must have the same shape as the field")
    return m


def _ptr(a):
    return None if a is None else a.ctypes.data_as(C.c_void_p)


def _check(rc, what):
    if rc != 0:
        raise ValueError(f"libgurt {what} failed (code {rc})")


def _need():
    lib = _load()
    if lib is None:
        raise RuntimeError(_ERR)
    return lib


# ── API ────────────────────────────────────────────────────────────────────
def despeckle(data, radius=2, min_good=10, max_pass=0, wrap=True, region=None):
    """DREADER despeck_parm. Returns (masked array, n_removed, n_passes)."""
    lib = _need()
    x = _to_nan(data)
    nr, ng = x.shape
    reg = _u8(region, x.shape)
    nrem, npass = C.c_int64(0), C.c_int32(0)
    _check(lib.gurt_despeckle(x, nr, ng, int(radius), int(min_good), int(max_pass), int(bool(wrap)),
                              _ptr(reg), C.byref(nrem), C.byref(npass)), "despeckle")
    return _to_masked(x), int(nrem.value), int(npass.value)


def deglitch(data, nyquist, radius=2, thresh=15.0, max_pass=0, delete_bad=True,
             numprt=1, prt1=None, prt2=None, nyq1=None, nyq2=None, region=None):
    """DREADER deglitch_parm. `nyquist` (and nyq1/nyq2) may be a scalar or a per-ray array.
    Returns (masked array, n_glitch_last_pass, n_passes)."""
    lib = _need()
    x = _to_nan(data)
    nr, ng = x.shape

    def per_ray(v):
        if v is None:
            return None
        return np.ascontiguousarray(np.broadcast_to(np.asarray(v, np.float64), (nr,)).copy())

    nq, q1, q2 = per_ray(nyquist), per_ray(nyq1), per_ray(nyq2)
    p1 = None if prt1 is None else _to_nan(prt1)
    p2 = None if prt2 is None else _to_nan(prt2)
    reg = _u8(region, x.shape)
    ngl, npass = C.c_int64(0), C.c_int32(0)
    _check(lib.gurt_deglitch(x, nr, ng, nq, int(radius), float(thresh), int(max_pass),
                             int(bool(delete_bad)), int(numprt), _ptr(p1), _ptr(p2), _ptr(q1),
                             _ptr(q2), _ptr(reg), C.byref(ngl), C.byref(npass)), "deglitch")
    return _to_masked(x), int(ngl.value), int(npass.value)


def gapfill(data, n_neighbors=3, method="median", n_passes=5, protect=None, region=None):
    """Fill missing gates along each ray. Returns (masked array, n_filled)."""
    if method not in ("median", "mean"):
        raise ValueError("method must be 'median' or 'mean'")
    lib = _need()
    x = _to_nan(data)
    nr, ng = x.shape
    prot, reg = _u8(protect, x.shape), _u8(region, x.shape)
    nf = C.c_int64(0)
    _check(lib.gurt_gapfill(x, nr, ng, int(n_neighbors), int(method == "median"), int(n_passes),
                            _ptr(prot), _ptr(reg), C.byref(nf)), "gapfill")
    return _to_masked(x), int(nf.value)


def unfold(data, nyquist, radius=2, thresh=0.8, ref=None, ray_map=None, gate_map=None, region=None):
    """DREADER dealias_parm. `nyquist` scalar or per-ray. Optional reference sweep `ref`
    (2-D, NaN/masked = missing) with `ray_map` (len nrays) / `gate_map` (len ngates) giving
    the matching ref ray / gate index (-1 = none). Returns (masked array, n_unfolded)."""
    lib = _need()
    if not hasattr(lib, "gurt_unfold"):
        raise RuntimeError("this libgurt build has no unfold; rebuild it (make)")
    x = _to_nan(data)
    nr, ng = x.shape
    nq = np.ascontiguousarray(np.broadcast_to(np.asarray(nyquist, np.float64), (nr,)).copy())
    reg = _u8(region, x.shape)
    r = rm = gm = None
    rnr = rng_ = 0
    if ref is not None:
        r = _to_nan(ref)
        rnr, rng_ = r.shape
        rm = np.ascontiguousarray(np.asarray(ray_map, np.int32))
        gm = np.ascontiguousarray(np.asarray(gate_map, np.int32))
        if rm.shape != (nr,) or gm.shape != (ng,):
            raise ValueError("ray_map / gate_map must have lengths nrays / ngates")
    n = C.c_int64(0)
    _check(lib.gurt_unfold(x, nr, ng, nq, int(radius), float(thresh), _ptr(r), rnr, rng_,
                           _ptr(rm), _ptr(gm), _ptr(reg), C.byref(n)), "unfold")
    return _to_masked(x), int(n.value)


# ----------------------------------------------------------------------------- tormodel
TOR_NOBS, TOR_NSTATE = 9, 13
# observation columns / state columns (same order as gurt.h GURT_OBS_* / GURT_ST_*)
TOR_OBS_COLS = ("azim", "rang", "radius", "maxtan", "coretan", "decaytan", "maxrad", "corerad", "decayrad")
TOR_STATE_COLS = ("x", "y", "azim", "rang", "radius", "transvel", "transang", "maxtan", "coretan",
                  "decaytan", "maxrad", "corerad", "decayrad")


def _tor_lib():
    lib = _need()
    if not hasattr(lib, "gurt_tor_interp"):
        raise RuntimeError("this libgurt build has no tormodel; rebuild it (make)")
    return lib


def tor_interp(tobs, obs, numt, t0, dt):
    """Interpolate observations to numt steps t0 + k*dt. tobs: seconds (strictly increasing,
    int32 range -- pass them relative to any origin, t0 on the same origin); obs: [numobs, 9]
    (TOR_OBS_COLS). Returns float32 [numt, 13] (TOR_STATE_COLS)."""
    lib = _tor_lib()
    tobs = np.ascontiguousarray(tobs, np.int32)
    obs = np.ascontiguousarray(obs, np.float32)
    out = np.zeros((int(numt), TOR_NSTATE), np.float32)
    _check(lib.gurt_tor_interp(len(tobs), tobs, obs, int(numt), int(t0), int(dt), out), "tormodel interpolation")
    return out


def tor_grid(state, tstep, x0, dx, numx, y0, dy, numy, want=("speed",)):
    """Wind on the grid at one time step. Returns {name: float32 [numy, numx]} for the names in
    `want` (speed, direc, tordist, posang)."""
    lib = _tor_lib()
    state = np.ascontiguousarray(state, np.float32)
    arrs = {n: np.zeros((int(numy), int(numx)), np.float32) for n in want}
    ptr = lambda n: arrs[n].ctypes.data if n in arrs else None      # noqa: E731
    _check(lib.gurt_tor_grid(state, len(state), int(tstep), int(x0), int(dx), int(numx), int(y0),
                             int(dy), int(numy), ptr("speed"), ptr("direc"), ptr("tordist"),
                             ptr("posang")), "tormodel grid")
    return arrs


def tor_maxgrid(state, nsteps, x0, dx, numx, y0, dy, numy, threads=None):
    """Maximum wind over the first *nsteps* steps at every grid cell, float32 [numy, numx]. Large grids are split
    into strips of rows that run on separate threads (ctypes releases the GIL during the C call), so the work
    scales with the number of cores; the result is identical to a single call. threads: None = automatic
    (GURT_THREADS environment variable, else the CPU count), 1 = no splitting."""
    lib = _tor_lib()
    state = np.ascontiguousarray(state, np.float32)
    numy, numx = int(numy), int(numx)
    out = np.zeros((numy, numx), np.float32)
    if threads is None:
        try:
            threads = int(os.environ.get("GURT_THREADS", "") or (os.cpu_count() or 1))
        except ValueError:
            threads = os.cpu_count() or 1
    work = float(numy) * numx * max(int(nsteps), 1)
    n = max(1, min(int(threads), numy, int(work // 2.0e6) + 1))       # no point splitting small jobs

    def strip(r0, r1):
        sub = out[r0:r1]                                              # contiguous rows
        _check(lib.gurt_tor_maxgrid(state, len(state), int(nsteps), int(x0), int(dx), numx,
                                    int(y0) + r0 * int(dy), int(dy), r1 - r0, sub), "tormodel max grid")

    if n == 1:
        strip(0, numy)
        return out
    from concurrent.futures import ThreadPoolExecutor
    cuts = [numy * i // (n * 4) for i in range(n * 4 + 1)]            # 4 strips per thread: even load
    with ThreadPoolExecutor(max_workers=n) as ex:
        for f in [ex.submit(strip, cuts[i], cuts[i + 1]) for i in range(n * 4) if cuts[i + 1] > cuts[i]]:
            f.result()
    return out


def tor_point(state, xpos, ypos):
    """Time series at a point: float32 [numt, 4] = tornado distance, tornado direction, speed, direction."""
    lib = _tor_lib()
    state = np.ascontiguousarray(state, np.float32)
    out = np.zeros((len(state), 4), np.float32)
    _check(lib.gurt_tor_point(state, len(state), float(xpos), float(ypos), out), "tormodel point")
    return out


def tor_latlon(radar_lat, radar_lon, xpos, ypos):
    lib = _tor_lib()
    la, lo = C.c_float(), C.c_float()
    _check(lib.gurt_tor_latlon(float(radar_lat), float(radar_lon), float(xpos), float(ypos),
                               C.byref(la), C.byref(lo)), "tormodel lat/lon")
    return la.value, lo.value


def tor_polar(x, y):
    """(range, azimuth) of a point x, y; DREADER's convert_polar (azimuth is meteorological)."""
    lib = _tor_lib()
    m, a = C.c_float(), C.c_float()
    _check(lib.gurt_tor_polar(float(x), float(y), C.byref(m), C.byref(a)), "tormodel polar")
    return m.value, a.value


def tor_cart(rang, azim):
    lib = _tor_lib()
    x, y = C.c_float(), C.c_float()
    _check(lib.gurt_tor_cart(float(rang), float(azim), C.byref(x), C.byref(y)), "tormodel cart")
    return x.value, y.value


# ── axisymmetric analysis / patch dynamics (full builds) ──────────────────
AXI_COLS = ("radius_km", "n", "vr", "vt", "vtot", "rvr", "angmom", "circ", "dp_pa", "flow_angle_deg")
PATCH_KEYS = ("patch_radius_km", "v_min", "az_min", "rng_min", "x_min", "y_min",
              "v_max", "az_max", "rng_max", "x_max", "y_max", "min_radius_km", "max_radius_km",
              "mean_radius_km", "diam_km", "axis_deg", "aspect_corr", "min_adj", "max_adj", "v_mean",
              "v_asym", "delta_v", "circulation_m2s", "delta_p_pa", "max_vort_radius_km",
              "min_ground", "max_ground", "min_ground_adj", "max_ground_adj",
              "largest_doppler", "largest_adjusted", "largest_ground", "largest_ground_adjusted")


def _axi_lib():
    lib = _load()
    if lib is None or not hasattr(lib, "gurt_axisym"):
        raise RuntimeError("this libgurt build has no axisymmetric analysis (full GURT build only)")
    return lib


def _vel64(vel):
    m = np.ma.getmaskarray(vel)
    x = np.array(np.ma.getdata(vel), dtype=np.float64, order="C", copy=True)
    x[m] = np.nan
    return x


def annulus_width(az_deg, rng_km, center_range_km):
    lib = _axi_lib()
    az = np.ascontiguousarray(az_deg, np.float64)
    rg = np.ascontiguousarray(rng_km, np.float64)
    return float(lib.gurt_annulus_width(az, len(az), rg, len(rg), float(center_range_km)))


def axisym(vel, az_deg, rng_km, xc, yc, elev_deg, delta_km, max_radius_km, translation=None,
           air_density=1.0, min_samples=5):
    """Ring table [n, 10] (AXI_COLS) of the axisymmetric analysis."""
    lib = _axi_lib()
    v = _vel64(vel)
    az = np.ascontiguousarray(az_deg, np.float64)
    rg = np.ascontiguousarray(rng_km, np.float64)
    cap = int(max_radius_km / (delta_km / 2.0)) + 4
    rings = np.zeros((cap, 10), np.float64)
    n = C.c_int32()
    u, w = translation if translation is not None else (0.0, 0.0)
    _check(lib.gurt_axisym(v, v.shape[0], v.shape[1], az, rg, float(xc), float(yc), float(elev_deg),
                           float(delta_km), float(max_radius_km), int(translation is not None),
                           float(u), float(w), float(air_density), int(min_samples), rings, cap,
                           C.byref(n)), "axisymmetric analysis")
    return rings[:n.value].copy()


def patch(vel, az_deg, rng_km, xc, yc, base_diam_km, beamwidth_deg, numdiam, storm_speed_ms,
          storm_dir_deg, air_density, extrema=None):
    """Patch dynamics. Returns None (no couplet) or (dict of PATCH_KEYS, levels[8, 6])."""
    lib = _axi_lib()
    v = _vel64(vel)
    az = np.ascontiguousarray(az_deg, np.float64)
    rg = np.ascontiguousarray(rng_km, np.float64)
    out = np.zeros(len(PATCH_KEYS), np.float64)
    lev = np.zeros((8, 6), np.float64)
    st = C.c_int32()
    (x0, y0), (x1, y1) = extrema if extrema is not None else ((0.0, 0.0), (0.0, 0.0))
    _check(lib.gurt_patch(v, v.shape[0], v.shape[1], az, rg, float(xc), float(yc), float(base_diam_km),
                          float(beamwidth_deg or 0.0), float(numdiam or 0.0), float(storm_speed_ms),
                          float(storm_dir_deg), float(air_density), int(extrema is not None),
                          float(x0), float(y0), float(x1), float(y1), out, lev, C.byref(st)),
           "patch dynamics")
    if st.value:
        return None
    return dict(zip(PATCH_KEYS, out.tolist())), lev
