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
