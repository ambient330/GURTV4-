# SPDX-License-Identifier: GPL-2.0-only
"""C (libgurt) vs Python reference for the axisymmetric / patch analyses on a synthetic vortex.
Run from GURTv4/:  python tests/test_axisym.py"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [ROOT]
os.environ.setdefault("LIBGURT_DIR", os.path.join(ROOT, "libgurt"))
from edit.native import native
from gui import dreader_modes as M

nat = native()
if nat is None or not nat.has_axisym():
    print("libgurt with axisym not available: SKIP")
    sys.exit(0)
fails = 0


def check(name, ok, extra=""):
    global fails
    fails += not ok
    print(("PASS " if ok else "FAIL ") + name, extra)


def vortex(rmax=0.4, vt=60.0, vr=-8.0, center=(6.0, 14.0), trans=(5.0, 3.0), seed=1,
           naz=720, ngates=300, dr=0.1, hole=True, elev=0.5):
    rng = np.random.default_rng(seed)
    az = (np.arange(naz) * (360.0 / naz) + 0.17) % 360.0
    r = (np.arange(ngates) + 1) * dr
    a = np.deg2rad(az)[:, None]
    X, Y = r[None] * np.sin(a), r[None] * np.cos(a)
    dx, dy = X - center[0], Y - center[1]
    d = np.hypot(dx, dy) + 1e-9
    prof = np.where(d < rmax, d / rmax, rmax / d)
    u = trans[0] + (vt * prof) * (-dy / d) + (vr * prof) * (dx / d)
    v = trans[1] + (vt * prof) * (dx / d) + (vr * prof) * (dy / d)
    vel = (u * np.sin(a) + v * np.cos(a)) * np.cos(np.deg2rad(elev)) + rng.normal(0, 0.3, (naz, ngates))
    vel = vel.astype(np.float32).astype(float)
    if hole:
        vel[rng.random(vel.shape) < 0.05] = np.nan
    return vel, az, r, center


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def same(x, y, tol=1e-6):
    if isinstance(x, dict):
        return x.keys() == y.keys() and all(same(x[k], y[k], tol) for k in x)
    if isinstance(x, (list, tuple)):
        return len(x) == len(y) and all(same(p, q, tol) for p, q in zip(x, y))
    if x is None or y is None:
        return x is None and y is None
    if isinstance(x, float) and np.isnan(x):
        return isinstance(y, float) and np.isnan(y)
    return close(float(x), float(y), tol)


for seed, kw in [(1, {}), (2, dict(center=(-9.0, 5.0), vt=45.0, rmax=0.8)),
                 (3, dict(center=(3.0, -20.0), vt=30.0, rmax=1.5, naz=360, dr=0.25)),
                 (4, dict(center=(0.5, 0.5), rmax=0.3))]:
    vel, az, r, c = vortex(seed=seed, **kw)
    for trans in (None, (5.0, 3.0)):
        a = M._axisym_py(vel, az, r, c, 0.5, None, 12.0, trans, 1.1)
        b = M.compute_axisymmetric(vel, az, r, c, 0.5, None, 12.0, trans, 1.1)
        check(f"axisym seed {seed} trans={trans is not None}", same(a, b, 1e-6),
              f"({len(a['rings']) if a else 0} rings)")
    for kwp in (dict(base_diam_km=1.2, numdiam=1.5), dict(base_diam_km=0.8, numdiam=1.0, storm_speed_ms=12.0, storm_dir_deg=250.0)):
        a = M._patch_py(vel, az, r, c, beamwidth_deg=1.0, air_density=1.1, **kwp)
        b = M.compute_dreader_patch(vel, az, r, c, beamwidth_deg=1.0, air_density=1.1, **kwp)
        check(f"patch seed {seed} {kwp['base_diam_km']}", same(a, b, 1e-6), "" if a else "(None both)")
    ex = ((c[0] - 0.3, c[1]), (c[0] + 0.3, c[1]))
    a = M._patch_py(vel, az, r, c, 1.0, extrema=ex)
    b = M.compute_dreader_patch(vel, az, r, c, 1.0, extrema=ex)
    check(f"patch extrema seed {seed}", same(a, b, 1e-6))

# degenerate inputs
empty = np.full((360, 100), np.nan)
az = np.arange(360.0); r = np.arange(1, 101) * 0.25
check("axisym on empty sweep is None", M.compute_axisymmetric(empty, az, r, (5, 5), 0, 0.5, 20) is None)
check("patch on empty sweep is None", M.compute_dreader_patch(empty, az, r, (5, 5), 1.0) is None)
check("masked input accepted", M.compute_axisymmetric(np.ma.masked_invalid(vortex()[0]), *vortex()[1:3], vortex()[3], 0.5, None, 12.0) is not None)
print("ALL PASSED" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
