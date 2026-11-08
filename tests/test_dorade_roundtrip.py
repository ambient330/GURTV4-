# SPDX-License-Identifier: GPL-2.0-only
"""Write a synthetic volume as DORADE and read it back with GURT's reader. Run from GURTv4/:
    python tests/test_dorade_roundtrip.py"""
import os, sys, tempfile
from datetime import datetime, timezone
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "gui")]
from gurt_core import RadarVolume, SweepData, load_radar_file
from exporters import write_dorade

rng = np.random.default_rng(0)
vol = RadarVolume(); vol.radar_name = "TESTRDR"; vol.latitude, vol.longitude, vol.altitude = 35.2, -97.4, 320.0
vol.scan_time = datetime(2024, 5, 24, 0, 30, tzinfo=timezone.utc); vol.field_names = ["VE", "DBZ", "a_long_field_name"]
vol.instrument_parameters = {"nyquist_velocity": {"data": np.array([26.0], "float32")}}
for el, g in ((0.5, 301), (1.5, 1832)):   # 1832 gates > 1500 exercises CSFD
    s = SweepData(); n = 361                       # odd gate count exercises the 4-byte padding
    s.azimuth = (np.arange(n) * 360.0 / n).astype("float32"); s.elevation = np.full(n, el, "float32")
    s.ranges = (150.0 + np.arange(g) * 75.0).astype("float32"); s.fixed_angle = el; s.radar_name = "TESTRDR"
    s.scan_time = vol.scan_time; s.instrument_parameters = dict(vol.instrument_parameters)
    for f, (lo, hi) in {"VE": (-26, 26), "DBZ": (-10, 70), "a_long_field_name": (0, 1)}.items():
        a = rng.uniform(lo, hi, (n, g)); s.fields[f] = np.ma.masked_array(a, mask=rng.random((n, g)) < 0.25)
    vol.sweeps.append(s)

from gurt_core import sane_altitude_m
assert sane_altitude_m(797.4) == 797.4 and sane_altitude_m(-431602094.0) == 0.0 and sane_altitude_m(float("nan")) == 0.0
assert sane_altitude_m(None) == 0.0 and sane_altitude_m(-50.0) == -50.0 and sane_altitude_m(99999.0) == 0.0
fails = 0
with tempfile.TemporaryDirectory() as d:
    for compat in ("standard", "dreader"):
        paths = write_dorade(vol, os.path.join(d, compat), compat=compat)
        assert len(paths) == 2
        for si, p in enumerate(paths):
            r = load_radar_file(p); s = r.sweeps[0]; o = vol.sweeps[si]
            ok = (s.fields["VE"].shape == o.fields["VE"].shape and np.allclose(s.azimuth, o.azimuth, atol=1e-4)
                  and np.allclose(s.ranges, o.ranges) and abs(s.fixed_angle - o.fixed_angle) < 1e-4
                  and abs(r.latitude - 35.2) < 1e-4 and abs(r.altitude - 320.0) < 1e-2)
            names = {"VE": "VE", "DBZ": "DBZ", "a_long_field_name": "a_long_f"}
            for f, dn in names.items():
                a = np.ma.filled(o.fields[f], np.nan); b = np.ma.filled(s.fields[dn], np.nan)
                ok &= np.array_equal(np.isnan(a), np.isnan(b)) and np.nanmax(np.abs(a - b)) < 0.006
            print(("PASS " if ok else "FAIL ") + f"{compat} sweep {si}  {os.path.basename(p)}")
            fails += not ok
# sector that crosses north, stored sorted by azimuth: must be written in scan order
sec = SweepData(); n, g = 60, 120
az = np.sort(np.mod(np.arange(n) * 1.0 - 30.0, 360.0)).astype("float32")      # 0..29, 330..359
sec.azimuth = az; sec.elevation = np.full(n, 0.5, "float32"); sec.ranges = (500 + np.arange(g) * 100.0).astype("float32")
sec.fixed_angle = 0.5; sec.scan_time = vol.scan_time; sec.radar_name = "TESTRDR"
sec.fields["VE"] = np.ma.masked_array(rng.uniform(-20, 20, (n, g)), mask=np.zeros((n, g), bool))
sv = RadarVolume(); sv.radar_name = "TESTRDR"; sv.latitude, sv.longitude, sv.altitude = 35.2, -97.4, 320.0
sv.scan_time = vol.scan_time; sv.field_names = ["VE"]; sv.sweeps = [sec]
with tempfile.TemporaryDirectory() as d2:
    r2 = load_radar_file(write_dorade(sv, d2)[0]).sweeps[0]
    ok = abs(r2.azimuth[0] - 330.0) < 1e-3 and abs(r2.azimuth[-1] - 29.0) < 1e-3 and "PPI" in os.listdir(d2)[0]
    ok &= np.allclose(np.ma.filled(r2.fields["VE"], 0)[0], sec.fields["VE"][30], atol=0.006)
    print(("PASS " if ok else "FAIL ") + "north-crossing sector written in scan order"); fails += not ok
print("ALL PASSED" if not fails else f"{fails} FAILED"); sys.exit(1 if fails else 0)
