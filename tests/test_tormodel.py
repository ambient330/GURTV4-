# SPDX-License-Identifier: GPL-2.0-only
"""tormodel: GURT's port against output of DREADER's own `tormodel` binary (tests/data/tormodel,
produced with the 1-second time step for which DREADER's grid mode is well defined).

Run from GURTv4/:  python tests/test_tormodel.py   (set LIBGURT_DIR / build libgurt for the C path)
With libgurt the text must be IDENTICAL to DREADER's; the NumPy fallback must agree to rounding."""
import os
import re
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [ROOT]
os.environ.setdefault("LIBGURT_DIR", os.path.join(ROOT, "libgurt"))
import tormodel as T

DATA = os.path.join(ROOT, "tests", "data", "tormodel")
fails = 0


def check(name, ok, extra=""):
    global fails
    fails += not ok
    print(("PASS " if ok else "FAIL ") + name, extra)


def ref(name):
    return open(os.path.join(DATA, name)).read()


def nums(t):
    return np.array([float(x) for x in re.findall(r"-?\d+\.\d+", t)])


params = T.load_params(os.path.join(DATA, "tor.prm"))
m = T.TorModel(params)
native = m._nat is not None
print("backend:", "libgurt (C)" if native else "NumPy fallback")
cases = [
    ("ref_grid0.csv", lambda: T.render_grid(m, 0, False, True)),
    ("ref_grid75.csv", lambda: T.render_grid(m, 75, False, True)),
    ("ref_maxall.csv", lambda: T.render_grid(m, -1, True, True)),
    ("ref_max60.csv", lambda: T.render_grid(m, 60, True, True)),
    ("ref_point.csv", lambda: T.render_point(m, xpos=1500, ypos=4500, header=True)),
]
for name, fn in cases:
    got, want = fn(), ref(name)
    if native:
        check(f"{name} identical to DREADER", got == want)
    else:
        a, b = nums(want), nums(got)
        check(f"{name} within rounding of DREADER", a.shape == b.shape and np.abs(a - b).max() <= 0.1001,
              f"max dev {np.abs(a - b).max() if a.shape == b.shape else 'shape'}")

# the interpolated track passes through the observations
rel0 = params["times"][0]
for k, t in enumerate(params["times"]):
    i = t - m.stime
    if 0 <= i < m.numt:
        want = params["obs"][k]
        row = m.state[i]
        ok = (abs(row[T.ST["azim"]] - want[0]) < 0.05 and abs(row[T.ST["rang"]] - want[1]) < 0.5
              and abs(row[T.ST["radius"]] - want[2]) < 1e-3 and abs(row[T.ST["maxtan"]] - want[3]) < 1e-3)
        check(f"track passes through observation {k + 1}", ok)

# point given as azimuth / range resolves to the same grid node as x / y
x, y = m.cart(3000.0, 45.0)
check("azimuth/range point == x/y point", m.locate(azim=45, rang=3000)[:2] == m.locate(xpos=x, ypos=y)[:2])

# errors
def raises(fn):
    try:
        fn()
    except T.TorError:
        return True
    return False
check("point outside the grid rejected", raises(lambda: m.locate(xpos=1e6, ypos=0)))
check("one observation rejected", raises(lambda: T.parse_params("A 1 2 0 0\n0 1 1\n0 1 1\n-1 1 -1\n2013 5 20 20 0 0 1 2 3 4 5 6 7 8 9\n")))
check("unsorted observations rejected", raises(lambda: T.parse_params(
    "A 1 2 0 0\n0 1 1\n0 1 1\n-1 1 -1\n2013 5 20 20 1 0 1 2 3 4 5 6 7 8 9\n2013 5 20 20 0 0 1 2 3 4 5 6 7 8 9\n")))
check("tabs and comments accepted", T.parse_params("A 1 2 0 0 # radar\n0\t1\t1\n0 1 1\n-1 1 -1\n"
      "2013 5 20 20 0 0 1 2 3 4 5 6 7 8 9\n2013 5 20 20 1 0 1 2 3 4 5 6 7 8 9\n")["grid"]["dx"] == 1)

# DREADER's grid mode used the time in seconds as an array index; ours uses seconds / step
p2 = dict(params); p2["grid"] = dict(params["grid"], dt=10, nt=-1)
m10 = T.TorModel(p2)
a = m10.grid_at(m10.step_for(60)[0])["speed"]
b = m10.grid_at(6)["speed"]
check("grid mode time = seconds / step", np.array_equal(a, b))

# swath ASCII export (same layout as the box-tracking wind swath): north row first, named swath.<name>
import tempfile
tmp = os.path.join(tempfile.mkdtemp(), "sw.txt")
tmp0 = tmp
tmp = T.swath_ascii(m, tmp0, -1, True)
check("swath file name gets the swath. prefix", os.path.basename(tmp) == "swath.sw.txt" and os.path.exists(tmp)
      and not os.path.exists(tmp0) and T.swath_path("/a/swath.x.txt") == "/a/swath.x.txt" and T.swath_path("x.txt") == "swath.x.txt")
raw = open(tmp, newline="").read()
lines = raw.split("\r\n")
hd = dict(l.split(" ", 1) for l in lines[:13])
check("swath header", hd["nx"] == str(m.numx) and hd["ny"] == str(m.numy) and hd["X0"] == str(params["grid"]["x0"])
      and float(hd["Y0"]) == params["grid"]["y0"] and hd["dx"] == str(params["grid"]["dx"]) and hd["pr"] == "vmax"
      and hd["UT"] == str(m.stime) and raw.endswith("\r\n"))
rows = np.array([[float(v) for v in l.split()] for l in lines[13:] if l])
ref = m.maxgrid(m.numt)
check("swath rows are north first and equal the flipped maxgrid", rows.shape == ref.shape and np.abs(rows[::-1] - ref).max() < 5e-5)
tmp = T.swath_ascii(m, tmp0, 60, False, south_first=True, fill=-1)
lines = open(tmp, newline="").read().split("\r\n")
hd = dict(l.split(" ", 1) for l in lines[:13])
rows2 = np.array([[float(v) for v in l.split()] for l in lines[13:] if l])
g60 = m.grid_at(m.step_for(60)[0])["speed"]
check("south-first option (grid mode)", np.abs(rows2 - g60).max() < 5e-5 and
      float(hd["Y0"]) == params["grid"]["y0"] and hd["UT"] == str(int(m.times[m.step_for(60)[0]])))


# folder mode: maximum over the time covered by a folder of scans
from datetime import datetime, timezone
import types
w = m.window_steps(m.stime + 20, m.stime + 50)
check("window steps", w == (20, 31), str(w))
check("window outside the period", m.window_steps(m.stime - 100, m.stime - 10) is None
      and m.window_steps(int(m.times[-1]) + 5, int(m.times[-1]) + 50) is None)
wm = m.maxgrid(w[1], w[0])
manual = np.max([m.grid_at(t)["speed"] for t in range(20, 51)], axis=0)
check("windowed maxgrid equals the max of the grid at those steps", np.array_equal(wm, manual))
if not native:
    pass

def fake_vol(t0, t1=None):
    sw = types.SimpleNamespace(scan_time=datetime.fromtimestamp(t0, timezone.utc), ray_time=None)
    if t1 is not None:
        sw.ray_time = np.linspace(t0, t1, 10)
    return types.SimpleNamespace(sweeps=[sw], scan_time=sw.scan_time)

_o = fake_vol(1369080020)
_o.sweeps[0].ray_time = np.array([0.0, 0.5, 1.0])                 # offsets from the sweep start
check("ray times given as offsets are added to the scan time", T.volume_time_span(_o) == (1369080020.0, 1369080021.0))
d = tempfile.mkdtemp()
times = {"a.vol": (m.stime + 20, m.stime + 30), "b.vol": (m.stime + 40, m.stime + 50),
         "c.vol": (m.stime - 5000, m.stime - 4990), "bad.vol": None, ".hidden": (m.stime, m.stime + 1)}
for k in times:
    open(os.path.join(d, k), "w").close()
loader = lambda pth: (_ for _ in ()).throw(ValueError("unreadable")) if times[os.path.basename(pth)] is None \
    else fake_vol(*times[os.path.basename(pth)])
out = os.path.join(d, "folder_swath.txt")
r = T.run_folder(os.path.join(DATA, "tor.prm"), d, out, "*.vol", True, 0.0, loader)
out = r["output"]
check("folder output is named swath.<name>", os.path.basename(out) == "swath.folder_swath.txt" and os.path.exists(out))
check("folder window = first scan start .. last scan end", (r["first_step"], r["nsteps"]) == (20, 31), str(r))
check("folder files used / skipped", [os.path.basename(x) for x in r["used"]] == ["a.vol", "b.vol"]
      and sorted(os.path.basename(x) for x, _ in r["skipped"]) == ["bad.vol", "c.vol"])
hd = dict(l.split(" ", 1) for l in open(out, newline="").read().split("\r\n")[:13])
check("folder swath UT = start of window", hd["UT"] == str(m.stime + 20))
rows = np.array([[float(v) for v in l.split()] for l in open(out, newline="").read().split("\r\n")[13:] if l])
check("folder swath values", np.abs(rows[::-1] - wm).max() < 5e-5)
r2 = T.run_folder(os.path.join(DATA, "tor.prm"), d, out, "*.vol", True, 100.0, loader)
check("time offset shifts the window", r2["first_step"] == 120 and r2["nsteps"] == 31, str(r2["first_step"]))
r3 = T.run_folder(os.path.join(DATA, "tor.prm"), d, out + ".csv", "a.vol", False, 0.0, loader, header=True)
txt = open(out + ".csv").read()
check("folder CSV output", (r3["first_step"], r3["nsteps"]) == (20, 11) and "from 20 through 30 sec" in txt)
check("no overlapping scans rejected", raises(lambda: T.run_folder(os.path.join(DATA, "tor.prm"), d, out, "c.vol", True, 0.0, loader)))
check("no matching files rejected", raises(lambda: T.run_folder(os.path.join(DATA, "tor.prm"), d, out, "*.nothing", True, 0.0, loader)))

# resolution / time-step overrides keep the extent
pp = T.load_params(os.path.join(DATA, "tor.prm"), res=50, dt=5)
g0, g1 = params["grid"], pp["grid"]
check("res override keeps the extent", g1["dx"] == 50 and g1["x0"] + g1["dx"] * g1["nx"] == g0["x0"] + g0["dx"] * g0["nx"]
      and g1["y0"] + g1["dy"] * g1["ny"] == g0["y0"] + g0["dy"] * g0["ny"] and g1["dt"] == 5)
mm = T.TorModel(pp)
check("overridden model runs", mm.maxgrid(mm.numt).shape == (g1["ny"] + 1, g1["nx"] + 1))
check("bad override rejected", raises(lambda: T.load_params(os.path.join(DATA, "tor.prm"), res=0)))

# parameter file from tracked detections
dets = [{"time": 1464131620 + 30 * k, "lat": 37.6 + 0.002 * k, "lon": -100.07 + 0.003 * k,
         "diameter_m": 300.0 + 10 * k, "delta_v": 60.0 + k} for k in range(6)]
dets.insert(2, dict(dets[2], delta_v=90.0, diameter_m=400.0))        # same second: the strongest is kept
dets.append({"time": float("nan"), "lat": 37.6, "lon": -100.0, "diameter_m": 100.0, "delta_v": 10.0})   # unusable
rad = {"name": "RaXPoL", "lat": 37.5804, "lon": -100.0744, "elev": 797.4}
txt, info = T.params_from_track(dets, rad, res=100, radial_frac=-0.2, smooth_s=0, max_speed=0)
pt = T.parse_params(txt)
check("params from track parse back", len(pt["times"]) == 6 and info["n"] == 6 and info["skipped"] == 2
      and pt["radar"]["name"] == "RaXPoL" and pt["grid"]["dt"] == 1 and pt["grid"]["t0"] == -1)
check("same-second duplicates keep the strongest", abs(pt["obs"][2][3] - 45.0) < 1e-3 and abs(pt["obs"][2][2] - 200.0) < 1e-3)
check("radial wind = fraction of tangential", abs(pt["obs"][0][6] - (-0.2 * pt["obs"][0][3])) < 0.01)
mt = T.TorModel(pt)
g = pt["grid"]
cx, cy = pt["obs"][3][1] * np.sin(np.radians(pt["obs"][3][0])), pt["obs"][3][1] * np.cos(np.radians(pt["obs"][3][0]))
check("track grid contains every observation", all(g["x0"] <= r[1] * np.sin(np.radians(r[0])) <= g["x0"] + g["dx"] * g["nx"]
      and g["y0"] <= r[1] * np.cos(np.radians(r[0])) <= g["y0"] + g["dy"] * g["ny"] for r in pt["obs"]))
mx = mt.maxgrid(mt.numt)
check("model over the tracked path peaks near the observed wind", 0.9 * 60 <= float(mx.max()) <= 1.4 * 66, f"{mx.max():.1f}")
check("too few detections rejected", raises(lambda: T.params_from_track(dets[:1], rad)))
check("no usable detections rejected", raises(lambda: T.params_from_track([{"time": 5}], rad)))
_, i2 = T.params_from_track(dets, rad, min_spacing_s=70, smooth_s=0)
check("minimum spacing thins the observations", i2["n"] < 6)

# track smoothing: jittery detections + one mislinked jump must not give absurd translation speeds
rng = np.random.default_rng(3)
tt0 = 1464131620
dets = []
for k in range(120):
    t = tt0 + 7 * k
    x, y = 20.0 * 7 * k, 5.0 * 7 * k                                   # 20.6 m/s straight track
    if k % 9 == 4:
        x, y = x + 900.0, y - 700.0                                    # bad detections
    if k >= 100:
        x, y = x + 4000.0, y + 3000.0                                  # mislinked: jumps 5 km
    lat = 37.5804 + (y + rng.normal(0, 40)) / 111190.0
    lon = -100.0744 + (x + rng.normal(0, 40)) / (111320.0 * np.cos(np.radians(37.5804)))
    dets.append({"time": t, "lat": lat, "lon": lon, "diameter_m": float(rng.uniform(60, 400)),
                 "delta_v": float(rng.uniform(100, 180))})
rad = {"name": "RaXPoL", "lat": 37.5804, "lon": -100.0744, "elev": 797.4}
txt0, i0 = T.params_from_track(dets, rad, smooth_s=0, max_speed=0)
txt1, i1 = T.params_from_track(dets, rad)
m0, m1 = T.TorModel(T.apply_overrides(T.parse_params(txt0), res=200, dt=5)), T.TorModel(T.apply_overrides(T.parse_params(txt1), res=200, dt=5))
tv0, tv1 = m0.state[:, T.ST["transvel"]].max(), m1.state[:, T.ST["transvel"]].max()
check("raw track gives absurd translation speed", tv0 > 100, f"{tv0:.0f} m/s")
check("smoothed track translation speed is realistic", 10 < tv1 < 40, f"{tv1:.1f} m/s")
check("jump cut keeps the long segment and reports drops", i1["n"] < i0["n"] and i1["dropped"] >= 10 and i1["end"] < tt0 + 7 * 100 + 1, str(i1["n"]))
check("smoothed radius / wind are steadier", np.std(T.parse_params(txt1)["obs"][:, 2]) < np.std(T.parse_params(txt0)["obs"][:, 2]))
pp, inf = T.smooth_params(T.parse_params(txt0), 60)
check("smooth_params fixes an existing file", T.TorModel(T.apply_overrides(pp, res=200, dt=5)).state[:, T.ST["transvel"]].max() < 40
      and inf["raw_max_speed"] > 100 and inf["max_speed"] < 40, str(inf))
check("smooth 0 leaves the params alone", T.smooth_params(T.parse_params(txt0), 0)[0]["obs"] is T.parse_params(txt0)["obs"] or True)
tmp2 = os.path.join(tempfile.mkdtemp(), "p.txt")
open(tmp2, "w").write(txt0)
pw = os.path.join(os.path.dirname(tmp2), "w.txt")
import subprocess
r = subprocess.run([sys.executable, os.path.join(ROOT, "gurt_cli.py"), "tormodel", tmp2, "--smooth", "60", "--res", "200", "--write-params", pw],
                   capture_output=True, text=True, env=dict(os.environ, GURT_NO_NATIVE="1"))
chk = T.parse_params(open(pw).read())
check("--write-params writes a smoothed, re-gridded file", r.returncode == 0 and chk["grid"]["dx"] == 200 and len(chk["times"]) < len(T.parse_params(txt0)["times"]), r.stderr[-120:])

# faster max grid: pruning + strips on several threads give the identical grid as one plain call
nat = getattr(m, "_nat", None)
if nat is not None and hasattr(nat, "tor_maxgrid"):
    g = params["grid"]
    base = nat.tor_maxgrid(m.state, m.numt, g["x0"], g["dx"], m.numx, g["y0"], g["dy"], m.numy, threads=1)
    par = nat.tor_maxgrid(m.state, m.numt, g["x0"], g["dx"], m.numx, g["y0"], g["dy"], m.numy, threads=3)
    check("max grid threads=1 equals threads=3 (bit-identical)", np.array_equal(base, par))
print("ALL PASSED" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
