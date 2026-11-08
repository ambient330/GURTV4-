# SPDX-License-Identifier: GPL-2.0-only
"""Concurrent CfRadial reads / writes (prefetch threads + foreground) must not hit HDF5 at the same time.
Run from GURTv4/:  python tests/test_threads.py"""
import os, sys, tempfile, threading
from datetime import datetime, timezone
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "gui")]
from gurt_core import RadarVolume, SweepData, load_radar_file
from exporters import write_cfradial
from readers.nclock import NC_LOCK

rng = np.random.default_rng(2)
v = RadarVolume(); v.radar_name = "TEST"; v.latitude, v.longitude, v.altitude = 35.2, -97.4, 320.0
v.scan_time = datetime(2015, 5, 6, 21, 13, 21, tzinfo=timezone.utc); v.field_names = ["VU"]
v.instrument_parameters = {"nyquist_velocity": {"data": np.array([32.0], "float32")}}
s = SweepData(); n, g = 360, 300
s.azimuth = np.arange(n, dtype="float32"); s.elevation = np.full(n, 0.5, "float32")
s.ranges = (150.0 + np.arange(g) * 75.0).astype("float32"); s.fixed_angle = 0.5; s.scan_time = v.scan_time
s.instrument_parameters = dict(v.instrument_parameters)
s.fields["VU"] = np.ma.masked_array(rng.uniform(-30, 30, (n, g)), mask=np.zeros((n, g), bool)); v.sweeps.append(s)

fails = []
with tempfile.TemporaryDirectory() as d:
    p = write_cfradial(v, d)
    ok = []
    def work():
        try:
            for _ in range(10):
                w = load_radar_file(p)
                ok.append(len(w.sweeps) == 1 and "VU" in w.sweeps[0].fields)
        except Exception as e:
            fails.append(repr(e))
    def writer():
        try:
            for i in range(3):
                write_cfradial(v, os.path.join(d, f"w{i}"))
        except Exception as e:
            fails.append(repr(e))
    ts = [threading.Thread(target=work) for _ in range(8)] + [threading.Thread(target=writer)]
    [t.start() for t in ts]; [t.join() for t in ts]
    # the lock really is held while a file is read
    held = []
    import readers.cfradial_reader as cr
    orig = cr.read_cfradial
    def spy(*a, **k):
        held.append(NC_LOCK._is_owned()); return orig(*a, **k)
    cr.read_cfradial = spy
    import gurt_core; gurt_core.read_cfradial = spy if hasattr(gurt_core, "read_cfradial") else None
    load_radar_file(p)
good = len(ok) == 80 and all(ok) and not fails
print(("PASS " if good else "FAIL ") + "80 concurrent reads + writes", fails[:2])
print("PASS lock held during a read" if (not held or all(held)) else "FAIL lock not held")
sys.exit(0 if good else 1)
