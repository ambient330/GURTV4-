# SPDX-License-Identifier: GPL-2.0-only
# GURT - Copyright (C) the GURT authors.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the
# Free Software Foundation. It is distributed WITHOUT ANY WARRANTY; see the
"""Unfold: libgurt (C) vs the numba/Python port vs a literal transcription of DREADER's
dealias_parm. Run:  python tests/test_unfold.py   (needs libgurt built for the C checks)"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
sys.path.insert(0, ROOT)
os.environ["GURT_NO_NATIVE"] = "1"                      # edit.unfold -> numba path
from edit import unfold as uf
import gurt_native as gn

f32 = np.float32


def literal(data, nyq, radius, thresh, ref=None, rmap=None, gmap=None):
    """Statement-by-statement port of dealias_parm (float32 like the C)."""
    d = np.array(data, dtype=f32, copy=True)
    nr, ng = d.shape
    bad = np.isnan
    for ray in range(nr):
        nq = f32(nyq[ray])
        for gate in range(ng):
            if bad(d[ray, gate]):
                continue
            unf = prev = 0
            abort = 0
            good = []
            locvalue = f32(0)
            if ref is not None:
                rr, rg = rmap[ray], gmap[gate]
                if rr < 0 or rg < 0:
                    abort = 1
                if not abort:
                    lg, c = rg, 0
                    while lg >= 0 and c <= radius:
                        if not bad(ref[rr, lg]): good.append(ref[rr, lg])
                        c += 1; lg -= 1
                    lg, c = rg + 1, 0
                    while lg < ref.shape[1] and c <= radius:
                        if not bad(ref[rr, lg]): good.append(ref[rr, lg])
                        c += 1; lg += 1
                    if not good:
                        while lg < ref.shape[1] and not good:
                            if not bad(ref[rr, lg]): good.append(ref[rr, lg])
                            lg += 1
                    if not good:
                        lg = rg - radius
                        while lg >= 0 and not good:
                            if not bad(ref[rr, lg]): good.append(ref[rr, lg])
                            lg -= 1
                    if good:
                        s = sorted(good); n = len(s)
                        locvalue = f32((float(s[n // 2 - 1]) + float(s[n // 2])) / 2.0) if n % 2 == 0 else s[(n - 1) // 2]
                    else:
                        abort = 1
            if ref is None or abort:
                good = []
                lg, c = gate - 1, 0
                while lg >= 0 and c <= radius:
                    if not bad(d[ray, lg]): good.append(d[ray, lg])
                    c += 1; lg -= 1
                if good:
                    s = sorted(good); n = len(s)
                    locvalue = f32((float(s[n // 2 - 1]) + float(s[n // 2])) / 2.0) if n % 2 == 0 else s[(n - 1) // 2]
                    abort = 0
                else:
                    abort = 1
            newvalue = d[ray, gate]
            mindiff = abs(f32(newvalue - locvalue))
            if mindiff > f32(thresh) * nq and not abort:
                while not abort:
                    if unf == prev or prev == 0:
                        prev = unf
                        if newvalue < locvalue:
                            newvalue = f32(newvalue + f32(2) * nq); unf = 1
                        else:
                            newvalue = f32(newvalue - f32(2) * nq); unf = -1
                        newdiff = abs(f32(newvalue - locvalue))
                        if newdiff < mindiff:
                            d[ray, gate] = newvalue; mindiff = newdiff
                    else:
                        abort = 1
    return d


rng = np.random.default_rng(3)
fails = 0
def check(name, ok, extra=""):
    global fails
    fails += (not ok); print(("PASS " if ok else "FAIL ") + name, extra)

def folded(nr=60, ng=90, nyq=15.0, bad=0.15, amp=45.0):
    az = np.linspace(0, 2 * np.pi, nr, endpoint=False)[:, None]
    true = (amp * np.sin(az) * np.exp(-np.linspace(0, 1.5, ng))[None, :] + rng.normal(0, 0.7, (nr, ng))).astype(f32)
    v = ((true + nyq) % (2 * nyq) - nyq).astype(f32)
    v[rng.random((nr, ng)) < bad] = np.nan
    return true, v

for nyq, rad, th in [(15.0, 2, 0.8), (10.0, 1, 0.5), (26.3, 3, 0.8)]:
    true, v = folded(nyq=nyq)
    nq = np.full(v.shape[0], nyq)
    want = literal(v, nq, rad, th)
    a, n = uf.unfold_field(np.ma.masked_invalid(v), nq, rad, th)
    check(f"numba == literal (nyq={nyq} r={rad} thr={th})",
          np.array_equal(np.ma.filled(a, np.nan), want, equal_nan=True), f"changed={n}")
    if gn.available():
        b, nb = gn.unfold(np.ma.masked_invalid(v), nq, rad, th)
        check(f"C     == literal (nyq={nyq} r={rad} thr={th})",
              np.array_equal(np.ma.filled(b, np.nan), want, equal_nan=True), f"changed={nb}")
        check("  counts agree", nb == n == int(np.sum(~np.isnan(v) & (want != v))))
    ok = np.isfinite(want)
    print(f"      folded gates fixed: {np.mean(np.abs(want[ok]-true[ok]) < 1.5):.3f} (raw {np.mean(np.abs(v[ok]-true[ok]) < 1.5):.3f})")

# reference sweep
true, v = folded()
_, refv = folded(bad=0.4)
refv = literal(refv, np.full(refv.shape[0], 15.0), 2, 0.8)
az = np.linspace(0, 360, v.shape[0], endpoint=False) + 0.2
raz = np.linspace(0, 360, refv.shape[0], endpoint=False)
ranges = 500 + np.arange(v.shape[1]) * 100.0
rmap, gmap = uf.match_reference(az, ranges, raz, ranges, 0.5)
nq = np.full(v.shape[0], 15.0)
want = literal(v, nq, 2, 0.8, refv, rmap, gmap)
a, n = uf.unfold_field(np.ma.masked_invalid(v), nq, 2, 0.8, np.ma.masked_invalid(refv), rmap, gmap)
check("numba == literal with reference sweep", np.array_equal(np.ma.filled(a, np.nan), want, equal_nan=True), f"changed={n}")
if gn.available():
    b, nb = gn.unfold(np.ma.masked_invalid(v), nq, 2, 0.8, np.ma.masked_invalid(refv), rmap, gmap)
    check("C     == literal with reference sweep", np.array_equal(np.ma.filled(b, np.nan), want, equal_nan=True), f"changed={nb}")
rmap2 = rmap.copy(); rmap2[::3] = -1
want = literal(v, nq, 2, 0.8, refv, rmap2, gmap)
a, _ = uf.unfold_field(np.ma.masked_invalid(v), nq, 2, 0.8, np.ma.masked_invalid(refv), rmap2, gmap)
check("unmatched rays fall back to same-ray rule", np.array_equal(np.ma.filled(a, np.nan), want, equal_nan=True))

# region: only gates inside may change
true, v = folded(); reg = np.zeros(v.shape, bool); reg[10:40, 20:60] = True
full, _ = uf.unfold_field(np.ma.masked_invalid(v), 15.0, 2, 0.8)
r_, _ = uf.unfold_field(np.ma.masked_invalid(v), 15.0, 2, 0.8, region=reg)
ch = np.ma.filled(r_, np.nan) != v; ch &= ~(np.isnan(np.ma.filled(r_, np.nan)) & np.isnan(v))
check("region: nothing outside the mask changes", not (ch & ~reg).any(), f"inside changed={int(ch.sum())}")
if gn.available():
    c_, _ = gn.unfold(np.ma.masked_invalid(v), 15.0, 2, 0.8, region=reg)
    check("region: C == numba", np.array_equal(np.ma.filled(c_, np.nan), np.ma.filled(r_, np.nan), equal_nan=True))

# degenerate inputs must not hang
z, _ = uf.unfold_field(np.ma.masked_invalid(v), 0.0, 2, 0.8)
check("nyquist 0 is a no-op", np.array_equal(np.ma.filled(z, np.nan), v, equal_nan=True))
print("ALL PASSED" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
