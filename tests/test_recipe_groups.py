# SPDX-License-Identifier: GPL-2.0-only
"""One-sweep-per-file DORADE input: sweepfiles of a volume are processed together and written once, also with
parallel workers. Run from GURTv4/:  python tests/test_recipe_groups.py"""
import os, shutil, subprocess, sys, tempfile
from datetime import datetime, timezone, timedelta
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "gui")]
from gurt_core import RadarVolume, SweepData, load_radar_file
from exporters import write_dorade
import gurt_recipe as gr

fails = 0
def check(name, ok, extra=""):
    global fails
    fails += 0 if ok else 1
    print(("PASS " if ok else "FAIL ") + name, extra if not ok else "")

rng = np.random.default_rng(1)
def volume(t):
    v = RadarVolume(); v.radar_name = "MWR_05XP"; v.latitude, v.longitude, v.altitude = 35.2, -97.4, 320.0
    v.scan_time = t; v.field_names = ["VU", "CR"]
    v.instrument_parameters = {"nyquist_velocity": {"data": np.array([32.0], "float32")}}
    for el in (0.8, 1.6, 2.4):
        s = SweepData(); n, g = 120, 200
        s.azimuth = (np.arange(n) * 3.0).astype("float32"); s.elevation = np.full(n, el, "float32")
        s.ranges = (150.0 + np.arange(g) * 75.0).astype("float32"); s.fixed_angle = el; s.radar_name = v.radar_name
        s.scan_time = t; s.instrument_parameters = dict(v.instrument_parameters)
        s.fields["VU"] = np.ma.masked_array(rng.uniform(-30, 30, (n, g)), mask=rng.random((n, g)) < 0.2)
        s.fields["CR"] = np.ma.masked_array(rng.uniform(0, 60, (n, g)), mask=np.zeros((n, g), bool))
        v.sweeps.append(s)
    return v

with tempfile.TemporaryDirectory() as d:
    inp, out = os.path.join(d, "in"), os.path.join(d, "out")
    os.makedirs(inp)
    for k in range(3):
        stage = os.path.join(d, f"stage{k}")
        for p in write_dorade(volume(datetime(2015, 5, 6, 21, 13, 21, tzinfo=timezone.utc) + timedelta(seconds=16 * k)), stage):
            shutil.move(p, inp)
    files = gr.list_input_files({"input": {"glob": "swp.*", "recursive": False}}, inp)
    groups = gr.group_inputs(files)
    check("9 sweepfiles form 3 volumes of 3", len(files) == 9 and len(groups) == 3 and all(len(g) == 3 for g in groups),
          str([os.path.basename(f) for f in files]))
    rec = gr.new_recipe([{"op": "copy", "field": "VU", "out": "VE-edit"}])
    rec["input"]["glob"] = "swp.*"
    rp = os.path.join(d, "r.json"); gr.save_recipe(rec, rp)
    for jobs in (1, 3):
        shutil.rmtree(out, ignore_errors=True)
        p = subprocess.run([sys.executable, os.path.join(ROOT, "gurt_cli.py"), "run", rp, inp, "-o", out, "-j", str(jobs)],
                           capture_output=True, text=True)
        outs = sorted(os.listdir(out)) if os.path.isdir(out) else []
        check(f"-j {jobs}: exit 0, three volumes written once each", p.returncode == 0 and len(outs) == 3, p.stdout[-400:] + p.stderr[-400:])
        if len(outs) == 3:
            v = load_radar_file(os.path.join(out, outs[0]))
            check(f"-j {jobs}: output holds all three sweeps with the new field",
                  len(v.sweeps) == 3 and sorted(round(float(s.fixed_angle), 1) for s in v.sweeps) == [0.8, 1.6, 2.4]
                  and all("VE-edit" in s.fields for s in v.sweeps), str([(s.fixed_angle, list(s.fields)) for s in v.sweeps]))
print("ALL PASSED" if not fails else f"{fails} FAILED"); sys.exit(1 if fails else 0)
