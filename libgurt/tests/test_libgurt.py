# SPDX-License-Identifier: GPL-2.0-only
# GURT - Copyright (C) the GURT authors.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the
# Free Software Foundation. It is distributed WITHOUT ANY WARRANTY; see the
# file LICENSE for details.
"""Check libgurt against the pure-Python implementations GURT already ships
(edit/despeckle.py, edit/deglitch.py). Run:  python tests/test_libgurt.py"""
import os, sys, types
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))           # GURTv4/
sys.path.insert(0, os.path.join(HERE, "..", "python"))
sys.path.insert(0, ROOT)

try:                                                              # numba is optional for the oracle
    import numba                                                  # noqa: F401
except ImportError:
    nb = types.ModuleType("numba")
    nb.njit = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
    sys.modules["numba"] = nb

import gurt_native as gn
from edit import despeckle as pd
from edit import deglitch as pg

assert gn.available(), "build libgurt first (make)"
print("libgurt", gn.version())
rng = np.random.default_rng(7)
fails = 0


def check(name, ok, extra=""):
    global fails
    fails += (not ok)
    print(("PASS " if ok else "FAIL ") + name, extra)


def field(nr=90, ng=120, bad=0.3, nyq=15.0):
    v = (np.sin(np.linspace(0, 6, nr))[:, None] * 12 + rng.normal(0, 1.0, (nr, ng))).astype(np.float32)
    g = rng.random((nr, ng)) < 0.04
    v[g] += rng.choice([-2 * nyq, 2 * nyq], g.sum()).astype(np.float32)       # folded glitches
    m = rng.random((nr, ng)) < bad
    return np.ma.array(v, mask=m)


# despeckle: exact match
for r, mg, mp, wrap in [(1, 4, 0, True), (2, 10, 0, False), (2, 10, 2, True), (3, 20, 0, True), (2, 0, 0, True)]:
    f = field()
    a, n, p = gn.despeckle(f, r, mg, mp, wrap)
    b, nb_, pb = pd.despeckle_field(f, r, mg, mp, wrap)
    check(f"despeckle r={r} min={mg} max_pass={mp} wrap={wrap}",
          (np.ma.getmaskarray(a) == np.ma.getmaskarray(b)).all() and n == nb_ and p == pb, f"removed={n} passes={p}")

# despeckle with a region: only gates inside may be deleted
f = field(bad=0.55); reg = np.zeros(f.shape, bool); reg[20:50, 30:80] = True
a, n, _ = gn.despeckle(f, 2, 10, 0, True, region=reg)
full, _, _ = gn.despeckle(f, 2, 10, 0, True)
changed = np.ma.getmaskarray(a) != np.ma.getmaskarray(f)
check("despeckle region confines edits", not changed[~reg].any() and changed[reg].any())

# deglitch vs the Python core (numprt 1 and 3)
BAD = -9999.0
for numprt, thr, dele in [(1, 15.0, True), (1, 10.0, False), (3, 15.0, True)]:
    f = field(bad=0.1)
    nyq = np.full(f.shape[0], 15.0)
    nq1, nq2 = np.full(f.shape[0], 11.0), np.full(f.shape[0], 20.0)
    filled = f.filled(BAD).astype(np.float64)
    ref = pg._run_core(filled.copy(), BAD, nyq, 2, thr, 0 if dele else 3, dele, numprt=numprt,
                       nyq1=nq1 if numprt == 3 else None, nyq2=nq2 if numprt == 3 else None)
    ref = np.ma.masked_where(np.abs(ref - BAD) <= 1e-5, ref)
    a, ng, npas = gn.deglitch(f, 15.0, 2, thr, 0 if dele else 3, dele, numprt,
                              nyq1=nq1 if numprt == 3 else None, nyq2=nq2 if numprt == 3 else None)
    same_mask = (np.ma.getmaskarray(a) == np.ma.getmaskarray(ref)).all()
    close = np.allclose(a.filled(0), np.ma.getdata(ref) * (~np.ma.getmaskarray(ref)), atol=1e-4)
    check(f"deglitch numprt={numprt} thresh={thr} delete={dele}", same_mask and close, f"passes={npas}")

# deglitch region
f = field(bad=0.1); reg = np.zeros(f.shape, bool); reg[10:40, 10:60] = True
a, _, _ = gn.deglitch(f, 15.0, 2, 15.0, 0, True, 1, region=reg)
diff = (np.ma.getmaskarray(a) != np.ma.getmaskarray(f)) | (a.filled(0) != f.filled(0))
check("deglitch region confines edits", not diff[~reg].any() and diff[reg].any())

# gap fill
for method, k, passes in [("median", 3, 5), ("mean", 2, 3)]:
    f = field(bad=0.15)
    orig = np.ma.getmaskarray(f).copy()
    f2 = f.copy(); f2[rng.random(f.shape) < 0.05] = np.ma.masked           # new gaps
    a, nfill = gn.gapfill(f2, k, method, passes, protect=orig)
    b = pg.gap_fill(f2, orig_mask=orig, n_neighbors=k, method=method, n_passes=passes)
    same = (np.ma.getmaskarray(a) == np.ma.getmaskarray(b)).all() and np.allclose(
        a.filled(0), b.filled(0), atol=1e-4)
    check(f"gapfill {method} k={k} passes={passes}", same, f"filled={nfill}")

print("\nALL PASSED" if not fails else f"\n{fails} FAILED")
sys.exit(1 if fails else 0)
