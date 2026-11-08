# SPDX-License-Identifier: GPL-2.0-only
"""detect_in_box explains why nothing was found. Run from GURTv4/:  python tests/test_box_why.py"""
import os, sys, types
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "gui")]
import dreader_detect as dd

fails = 0
def check(name, ok, extra=""):
    global fails
    fails += 0 if ok else 1
    print(("PASS " if ok else "FAIL ") + name, extra if not ok else "")

def sweep(diam_m, vmax, sign=1):
    """Rankine-like couplet centred 20 km north of the radar, 0.5 deg x 250 m grid."""
    az = np.arange(0.0, 360.0, 0.5)
    rng = np.arange(2000.0, 40000.0, 250.0)
    X = np.outer(np.sin(np.radians(az)), rng)
    Y = np.outer(np.cos(np.radians(az)), rng)
    dx, dy = X - 0.0, Y - 20000.0
    r = np.hypot(dx, dy)
    R = diam_m / 2.0
    vt = np.where(r < R, vmax * r / R, vmax * R / np.maximum(r, 1.0))
    vt = np.where(r > 4 * R, 0.0, vt)
    # radial component of a counter-clockwise (cyclonic) vortex seen from the origin
    vr = sign * vt * (-dy * X + dx * Y) / np.maximum(r, 1.0) / np.maximum(np.hypot(X, Y), 1.0)
    return types.SimpleNamespace(azimuth=az, ranges=rng, elevation=np.full(az.size, 0.5), fixed_angle=0.5,
                                 fields={"VEL": vr}, scan_time=None, ray_time=None)

box = (-4.0, 4.0, 16.0, 24.0)
for sign in (1, -1):
    w = []
    d = dd.detect_in_box(sweep(800, 40, sign), "VEL", box, 37.0, -88.0, why=w)
    cyc = d is not None
    w2 = []
    d2 = dd.detect_in_box(sweep(800, 40, sign), "VEL", box, 37.0, -88.0, polarity="Anticyclonic", min_azshear=0.005, why=w2)
    check(f"sign {sign}: exactly one rotation sense is found", cyc != (d2 is not None), str((w, w2)))
    bad = w if not cyc else w2
    check("opposite sense is explained", bool(bad) and "rotation sense" in bad[0] or "shear" in bad[0], str(bad))
w = []
check("too wide for Max couplet diameter is explained", dd.detect_in_box(sweep(8000, 40), "VEL", box, 37.0, -88.0, polarity="Either", maxdiam_m=2000.0, min_azshear=0.005, why=w) is None
      and bool(w) and "Max couplet diameter" in w[0], str(w))
w = []
check("weak delta-V is explained", dd.detect_in_box(sweep(800, 8), "VEL", box, 37.0, -88.0, polarity="Either", min_azshear=0.005, why=w) is None
      and bool(w), str(w))
w = []
check("empty box is explained", dd.detect_in_box(sweep(800, 40), "VEL", (200, 210, 200, 210), 37.0, -88.0, why=w) is None
      and "no radar gates" in "".join(w), str(w))
from datetime import datetime, timezone
sw = sweep(800, 40)
sw.scan_time = datetime(2021, 12, 11, 3, 17, 59, tzinfo=timezone.utc)
sw.ray_time = np.arange(sw.azimuth.size) * 0.01                      # offsets from the sweep start (DORADE style)
d = dd.detect_in_box(sw, "VEL", box, 37.0, -88.0, polarity="Either", min_azshear=0.005)
t0 = sw.scan_time.timestamp()
check("detection time = sweep start + ray offset (not the bare offset)", d is not None and t0 <= d["time"] <= t0 + 4.0, str(d and d["time"]))
sw.ray_time = sw.ray_time + t0                                      # absolute ray times stay as they are
d = dd.detect_in_box(sw, "VEL", box, 37.0, -88.0, polarity="Either", min_azshear=0.005)
check("absolute ray times are kept", d is not None and t0 <= d["time"] <= t0 + 4.0, str(d and d["time"]))
from dreader_xsect import is_velocity_field
check("derived velocity names are velocity fields", all(is_velocity_field(n) for n in
      ("VE", "VE-climo", "VEL_despeckled", "VE.filtered", "VELH")) and not any(is_velocity_field(n) for n in ("REF", "DBZ", "SW", "VEHICLE")))
print("ALL PASSED" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
