# SPDX-License-Identifier: GPL-2.0-only
# GURT - Copyright (C) the GURT authors.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the
# Free Software Foundation. It is distributed WITHOUT ANY WARRANTY; see the
"""Headless GURT command line: run an editing recipe over a folder of radar files.

    python gurt_cli.py run recipe.json /data/in -o /data/out [-j 4] [--dry-run]
    python gurt_cli.py validate recipe.json
    python gurt_cli.py ops
    python gurt_cli.py init recipe.json      # write an example recipe
    python gurt_cli.py tormodel params.txt --grid --time 60 -o wind.csv   # DREADER tormodel (see tormodel.py)
    python gurt_cli.py tormodel params.txt --folder /data/scans --maxgrid --swath -o swath.txt   # max over a folder's scans
    python gurt_cli.py tormodel --example > params.txt                  # example tormodel parameter file

No Tk / vispy needed. Set GURT_NO_NATIVE=1 to skip libgurt and use the Python fallback.
"""

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "gui")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gurt_recipe as gr

EXAMPLE = gr.new_recipe([
    {"op": "despeckle", "field": "VE", "radius": 2, "min_good": 12, "max_pass": 3, "out": "VE-edit"},
    {"op": "deglitch", "field": "VE-edit", "radius": 2, "thresh": 15.0, "gapfill": True},
    {"op": "smooth_velmag", "field": "VE-edit", "length_m": 200, "out": "VMAG"},
])


def _worker(args):
    path, rec, out_dir = args
    t0 = time.time()
    try:
        r = gr.process_file(path, rec, out_dir)
        r["seconds"] = round(time.time() - t0, 2)
        return r
    except Exception as e:                      # keep going; report at the end
        return {"input": path if isinstance(path, str) else path[0], "error": f"{type(e).__name__}: {e}"}


def cmd_run(a):
    rec = gr.load_recipe(a.recipe)
    if not os.path.isdir(a.folder):
        sys.exit(f"error: {a.folder} is not a folder")
    files = gr.list_input_files(rec, a.folder)
    if not files:
        sys.exit(f"error: no files match {rec['input']['glob']!r} in {a.folder}")
    print(f"Recipe: {a.recipe}  ({len(rec['steps'])} steps)")
    for i, s in enumerate(rec["steps"], 1):
        print(f"  {i}. {gr.describe_step(s)}")
    print(f"{len(files)} input file(s) in {a.folder}")
    if a.dry_run:
        from gurt_core import load_radar_file
        try:
            vol = load_radar_file(files[0])
            have = gr.check_fields(vol.field_names, rec["steps"])
        except Exception as e:
            sys.exit(f"dry run failed on {os.path.basename(files[0])}: {e}")
        print(f"Dry run OK on {os.path.basename(files[0])}; output fields: {', '.join(have)}")
        return 0
    out_dir = a.output
    os.makedirs(out_dir, exist_ok=True)
    groups = gr.group_inputs(files)
    if len(groups) != len(files):
        print(f"{len(files)} sweepfile(s) form {len(groups)} volume(s); each volume is processed and written once")
    jobs = [(g if len(g) > 1 else g[0], rec, out_dir) for g in groups]
    results = []
    t0 = time.time()
    if a.jobs > 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            for n, r in enumerate(ex.map(_worker, jobs), 1):
                results.append(r); _report(n, len(jobs), r)
    else:
        for n, j in enumerate(jobs, 1):
            r = _worker(j); results.append(r); _report(n, len(jobs), r)
    bad = [r for r in results if "error" in r]
    print(f"Done: {len(results) - len(bad)} written, {len(bad)} failed, "
          f"{time.time() - t0:.1f} s. Output: {out_dir}")
    if a.log:
        with open(a.log, "w", encoding="utf-8") as f:
            json.dump({"recipe": rec, "results": results}, f, indent=2)
        print(f"Log: {a.log}")
    return 1 if bad else 0


def _report(n, total, r):
    name = os.path.basename(r["input"]) + (f" (+{len(r['inputs']) - 1} more sweepfile(s))" if len(r.get("inputs", [])) > 1 else "")
    if "error" in r:
        print(f"[{n}/{total}] FAILED {name}: {r['error']}")
    else:
        gates = sum(s["gates"] for s in r["steps"])
        outs = r.get("outputs")
        dest = f"{len(outs)} DORADE sweepfile(s)" if outs else os.path.basename(r["output"])
        print(f"[{n}/{total}] {name} -> {dest}  ({gates} gates changed, {r['seconds']} s)")


def cmd_validate(a):
    rec = gr.load_recipe(a.recipe)
    print(f"{a.recipe}: OK, {len(rec['steps'])} step(s)")
    for i, s in enumerate(rec["steps"], 1):
        print(f"  {i}. {gr.describe_step(s)}")
    return 0


def cmd_ops(a):
    for op, (req, opt) in gr.OPS.items():
        print(f"{op}\n    {gr.OP_HELP[op]}")
        print(f"    required: {', '.join(req)}")
        if opt:
            print("    optional: " + ", ".join(f"{k}={v!r}" for k, v in opt.items()))
    return 0


def cmd_init(a):
    if os.path.exists(a.recipe) and not a.force:
        sys.exit(f"error: {a.recipe} exists (use --force)")
    gr.save_recipe(EXAMPLE, a.recipe)
    print(f"Wrote example recipe to {a.recipe}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gurt_cli", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a recipe over a folder")
    r.add_argument("recipe"); r.add_argument("folder")
    r.add_argument("-o", "--output", required=False, default="gurt_out", help="output folder (default ./gurt_out)")
    r.add_argument("-j", "--jobs", type=int, default=1, help="parallel files (processes)")
    r.add_argument("--dry-run", action="store_true", help="validate against the first file, write nothing")
    r.add_argument("--log", help="write a JSON log of the run")
    r.set_defaults(fn=cmd_run)
    v = sub.add_parser("validate", help="check a recipe file"); v.add_argument("recipe"); v.set_defaults(fn=cmd_validate)
    o = sub.add_parser("ops", help="list available steps"); o.set_defaults(fn=cmd_ops)
    i = sub.add_parser("init", help="write an example recipe"); i.add_argument("recipe")
    i.add_argument("--force", action="store_true"); i.set_defaults(fn=cmd_init)
    t = sub.add_parser("tormodel", help="analytic tornado wind model (DREADER tormodel): point / grid / maxgrid",
                       description="Pass --example to print an example parameter file.")
    import tormodel as _tm
    _tm.add_arguments(t)
    t.set_defaults(fn=lambda a: _tm.run_args(a))
    if (argv if argv is not None else sys.argv[1:])[:2] == ["tormodel", "--example"]:
        sys.stdout.write(_tm.EXAMPLE_PARAMS)
        return 0
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except gr.RecipeError as e:
        sys.exit(f"recipe error: {e}")


if __name__ == "__main__":
    sys.exit(main())
