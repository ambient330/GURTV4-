# SPDX-License-Identifier: GPL-2.0-only
"""Variational radial-wind fit. Run from GURTv4/:  python tests/test_vrfit.py"""
import math, os, sys, types
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "gui")]
import tormodel as TM
import vrfit as VF

fails = 0
def check(name, ok, extra=""):
    global fails
    fails += 0 if ok else 1
    print(("PASS " if ok else "FAIL ") + name, extra if not ok else "")

rng_ = np.random.default_rng(7)
RLAT, RLON = 35.2358, -97.4619
az = np.arange(0.0, 360.0, 0.5)
rg = np.arange(1000.0, 50000.0, 75.0)
K = 10
t0 = 1306270715.0
true_f = -0.25 + 0.05 * np.sin(np.arange(K) / 3.0)             # maxrad / maxtan, drifting slowly
true_core, true_decay = 1.0, -0.6
maxtan, radius = 45.0, 500.0
bg = (8.0, -5.0)                                                # environmental wind u, v

def make_sweep(xc, yc, f, noise=1.5):
    A, R = np.meshgrid(np.radians(az), rg, indexing="ij")
    el = np.radians(0.5)
    S = {"x": R * math.cos(el) * np.sin(A), "y": R * math.cos(el) * np.cos(A), "sinaz": np.sin(A), "cosaz": np.cos(A),
         "cosel": np.full(A.shape, math.cos(el))}
    v = VF.model_doppler(S, xc, yc, radius, maxtan, 1.0, -0.6, f * maxtan, true_core, true_decay)
    v = v + (bg[0] * np.sin(A) + bg[1] * np.cos(A)) * math.cos(el) + rng_.normal(0, noise, A.shape)
    return types.SimpleNamespace(azimuth=az, ranges=rg, elevation=np.full(az.size, 0.5), fields={"VEL": v})

xs = -6000.0 + 250.0 * np.arange(K)
ys = 21000.0 + 120.0 * np.arange(K)
dets, sweeps = [], {}
for k in range(K):
    lat = RLAT + ys[k] / 111190.0
    lon = RLON + xs[k] / (111320.0 * math.cos(math.radians(RLAT)))
    d = {"time": t0 + 60 * k, "lat": lat, "lon": lon, "diameter_m": 2 * radius, "delta_v": 2 * maxtan, "file": f"f{k}", "sweep": 0, "field": "VEL"}
    dets.append(d)
    sweeps[f"f{k}"] = make_sweep(xs[k], ys[k], true_f[k])
text, _ = TM.params_from_track(dets, {"name": "TEST", "lat": RLAT, "lon": RLON, "elev": 300.0}, res=100,
                               radial_frac=0.0, smooth_s=0.0)
check("starting file has maxrad 0", all(ln.split()[12] == "0.00" for ln in text.splitlines() if len(ln.split()) >= 15 and ln[:4].isdigit()))

get = lambda d: (sweeps[d["file"]], RLAT, RLON)
new, info = VF.fit_params_text(text, dets, get, RLAT, RLON, sigma=2.0, strength=1.0)
mr = np.array([float(ln.split()[12]) for ln in new.splitlines() if len(ln.split()) >= 15 and ln[:4].isdigit()])
mc = np.array([float(ln.split()[13]) for ln in new.splitlines() if len(ln.split()) >= 15 and ln[:4].isdigit()])
true_mr = true_f * maxtan
print("true maxrad", np.round(true_mr, 1), "\nfit  maxrad", mr, "\ncore", mc, "rms", info["rms_before"], info["rms_after"])
check("fitted maxrad is inflow near the truth (mean error < 2 m/s)", abs(np.mean(mr - true_mr)) < 2.0 and np.abs(mr - true_mr).max() < 4.0)
check("the fit reduces the misfit", info["rms_after"] < 0.8 * info["rms_before"])
check("file keeps its other columns and layout", len(new.splitlines()) == len(text.splitlines()) and
      all(a.split()[:12] == b.split()[:12] for a, b in zip(new.splitlines(), text.splitlines()) if a[:4].isdigit()))
check("exponents stay inside their bounds", all(0.2 <= c <= 3.0 for c in mc))

# smoothness: one scan with wildly wrong data should be pulled toward its neighbours
bad = dict(sweeps); bad["f4"] = make_sweep(xs[4], ys[4], +0.9)
new_s, _ = VF.fit_params_text(text, dets, lambda d: (bad[d["file"]], RLAT, RLON), RLAT, RLON, strength=30.0)
new_0, _ = VF.fit_params_text(text, dets, lambda d: (bad[d["file"]], RLAT, RLON), RLAT, RLON, strength=0.0)
v = lambda s: float([ln for ln in s.splitlines() if len(ln.split()) >= 15 and ln[:4].isdigit()][4].split()[12])
check("time coupling holds a bad scan near its neighbours", abs(v(new_s) - true_mr[4]) < abs(v(new_0) - true_mr[4]) - 3.0, (v(new_s), v(new_0)))

# a scan without data (no sweep) follows the smoothness term only
miss = lambda d: None if d["file"] == "f5" else (sweeps[d["file"]], RLAT, RLON)
new_m, info_m = VF.fit_params_text(text, dets, miss, RLAT, RLON)
check("missing scan is filled in from its neighbours", info_m["n_fit"] == K - 1 and abs(v(new_m) - true_mr[4]) < 4.0)
print("ALL PASSED" if not fails else f"{fails} FAILED"); sys.exit(1 if fails else 0)
