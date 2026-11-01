# SPDX-License-Identifier: GPL-2.0-only
# GURT - Copyright (C) the GURT authors.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the
# Free Software Foundation. It is distributed WITHOUT ANY WARRANTY; see the
# file LICENSE for details.
"""GURT's own entry points give the same answers with and without libgurt.
Run from anywhere:  python libgurt/tests/test_integration.py"""
import os, sys, types, importlib
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
try:
    import numba  # noqa: F401
except ImportError:
    nb = types.ModuleType("numba"); nb.njit = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
    sys.modules["numba"] = nb

from edit import deglitch as pg, despeckle as pd, native as nv

rng = np.random.default_rng(11)
BAD = -9999.0
v = (np.sin(np.linspace(0, 6, 90))[:, None] * 12 + rng.normal(0, 1.0, (90, 120)))      # float64 data
g = rng.random(v.shape) < 0.04
v[g] += rng.choice([-30.0, 30.0], g.sum())
f = np.ma.array(v, mask=rng.random(v.shape) < 0.3)
filled = f.filled(BAD)
nyq = np.full(90, 15.0)

def both(fn):
    os.environ.pop("GURT_NO_NATIVE", None); nv._state.update(mod=None, tried=False)
    a = fn(); assert nv.native() is not None, "libgurt not loaded"
    os.environ["GURT_NO_NATIVE"] = "1"; nv._state.update(mod=None, tried=False)
    b = fn(); assert nv.native() is None
    os.environ.pop("GURT_NO_NATIVE", None); nv._state.update(mod=None, tried=False)
    return a, b

fails = 0
def check(name, ok):
    global fails; fails += (not ok); print(("PASS " if ok else "FAIL ") + name)

a, b = both(lambda: pg._run_core(filled.copy(), BAD, nyq, 2, 15.0, 0, True, numprt=1))
check("_run_core (brush deglitch) native == python", np.allclose(a, b, atol=1e-4) and
      ((np.abs(a - filled) > 1e-6) == (np.abs(b - filled) > 1e-6)).all())
check("_run_core keeps exact float64 for untouched gates", (a[(np.abs(a - filled) <= 1e-6)] == filled[(np.abs(a - filled) <= 1e-6)]).all())

dg = np.ma.masked_where(np.abs(b - BAD) <= 1e-5, b)
orig_mask = np.ma.getmaskarray(f)
a, b = both(lambda: pg.gap_fill(dg, orig_mask=orig_mask, n_neighbors=3, method="median", n_passes=5))
check("gap_fill native == python", (np.ma.getmaskarray(a) == np.ma.getmaskarray(b)).all() and
      np.allclose(a.filled(0), b.filled(0), atol=1e-4))

a, b = both(lambda: pd.despeckle_field(f, 2, 10, 0, True))
check("despeckle_field native == python", (np.ma.getmaskarray(a[0]) == np.ma.getmaskarray(b[0])).all()
      and a[1:] == b[1:] and a[0].dtype == b[0].dtype)

print("\nALL PASSED" if not fails else f"\n{fails} FAILED"); sys.exit(1 if fails else 0)
