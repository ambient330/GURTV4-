# SPDX-License-Identifier: GPL-2.0-only
# GURT - Copyright (C) the GURT authors.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the
# Free Software Foundation. It is distributed WITHOUT ANY WARRANTY; see the
"""GURT recipes: a JSON description of an editing pipeline for a folder of radar files.

No GUI dependencies. Used by gurt_cli.py (``python gurt_cli.py run recipe.json DIR -o OUT``)
and written by the GURT GUI ("Recipe" menu).

Recipe layout::

    {
      "gurt_recipe": 1,
      "input":  {"glob": "*", "recursive": false},
      "steps":  [ {"op": "despeckle", "field": "VE", "radius": 2, "min_good": 12,
                   "max_pass": 3, "out": "VE-edit"}, ... ],
      "output": {"format": "cfradial"}
    }

Every step works on every PPI sweep that has the step's ``field`` (RHI sweeps are
skipped). ``out`` names the result field; when omitted the step overwrites ``field``
in memory (the input files themselves are never modified).
"""

import json
import os
import fnmatch

import numpy as np

RECIPE_VERSION = 1

# op -> (required params, {optional param: default})
OPS = {
    "despeckle": (["field"], {"out": None, "radius": 2, "min_good": 10, "max_pass": 0, "wrap": True}),
    "deglitch": (["field"], {"out": None, "radius": 2, "thresh": 15.0, "max_pass": 0,
                             "delete_bad": True, "nyquist": None, "gapfill": False,
                             "gapfill_n_neighbors": 3, "gapfill_method": "median",
                             "gapfill_n_passes": 5}),
    "unfold": (["field"], {"out": None, "radius": 2, "thresh": 0.8, "nyquist": None,
                           "reference": "none", "az_tol": 0.5, "fixed_angle_tol": 10.0}),
    "gapfill": (["field"], {"out": None, "n_neighbors": 3, "method": "median", "n_passes": 5}),
    "smooth_velmag": (["field"], {"out": None, "length_m": 200.0, "min_frac": 0.5}),
    "gatefilter": (["field", "by", "condition"],
                   {"out": None, "value": 0.0, "value2": 0.0}),
    "copy": (["field", "out"], {}),
    "drop": (["fields"], {}),
}

OP_HELP = {
    "despeckle": "DREADER iterative despeckle: remove gates with too few good neighbours.",
    "deglitch": "DREADER deglitch: fold velocity toward the local median by multiples of 2*Nyquist; "
                "optionally gap-fill the gates it could not fix.",
    "unfold": "DREADER unfold (dealias): along-ray Bargen-Brown unfolding against a local median; "
              "optional reference='previous_sweep'.",
    "gapfill": "Fill masked gates from the nearest good gates along each ray.",
    "smooth_velmag": "Smoothed |V| (boxcar of length_m in range and arc length).",
    "gatefilter": "Mask gates of 'field' wherever the 'by' field meets a condition "
                  "(<, <=, >, >=, ==, !=, between, outside, nan), like Edit > GateFilter.",
    "copy": "Copy a field under a new name.",
    "drop": "Remove fields (list) from the output.",
}


# recipe symbol -> label used by edit/gatefilter.py (and the GUI dialog)
GATE_CONDITIONS = {
    "<": "Less than (<)", "<=": "Less than or equal (<=)",
    ">": "Greater than (>)", ">=": "Greater than or equal (>=)",
    "==": "Equal to (==)", "!=": "Not equal to (!=)",
    "between": "Between (inclusive)", "outside": "Outside range", "nan": "Is NaN / missing",
}


class RecipeError(ValueError):
    pass


# ── load / validate ───────────────────────────────────────────────────────────

def load_recipe(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if path.lower().endswith((".yml", ".yaml")):
        try:
            import yaml
        except ImportError as e:
            raise RecipeError("YAML recipes need PyYAML (pip install pyyaml); JSON needs nothing.") from e
        rec = yaml.safe_load(text)
    else:
        try:
            rec = json.loads(text)
        except json.JSONDecodeError as e:
            raise RecipeError(f"{path}: not valid JSON ({e})") from e
    return validate_recipe(rec)


def validate_recipe(rec):
    """Return a normalised copy (defaults filled in) or raise RecipeError."""
    if not isinstance(rec, dict):
        raise RecipeError("recipe must be a JSON object")
    ver = rec.get("gurt_recipe", RECIPE_VERSION)
    if ver != RECIPE_VERSION:
        raise RecipeError(f"unsupported recipe version {ver!r} (this GURT reads {RECIPE_VERSION})")
    steps = rec.get("steps")
    if not isinstance(steps, list) or not steps:
        raise RecipeError("recipe needs a non-empty 'steps' list")
    out_steps = []
    for i, st in enumerate(steps, 1):
        if not isinstance(st, dict) or "op" not in st:
            raise RecipeError(f"step {i}: needs an 'op'")
        op = st["op"]
        if op not in OPS:
            raise RecipeError(f"step {i}: unknown op {op!r} (known: {', '.join(OPS)})")
        req, opt = OPS[op]
        extra = set(st) - {"op", "note"} - set(req) - set(opt)
        if extra:
            raise RecipeError(f"step {i} ({op}): unknown parameter(s) {sorted(extra)}")
        for k in req:
            if k not in st or st[k] in (None, "", []):
                raise RecipeError(f"step {i} ({op}): missing required parameter '{k}'")
        s = {"op": op}
        if st.get("note"):
            s["note"] = str(st["note"])
        for k in req:
            s[k] = st[k]
        for k, d in opt.items():
            s[k] = st.get(k, d)
        _check_values(i, s)
        out_steps.append(s)
    inp = dict({"glob": "*", "recursive": False}, **(rec.get("input") or {}))
    outp = dict({"format": "cfradial"}, **(rec.get("output") or {}))
    if outp["format"] not in ("cfradial", "dorade"):
        raise RecipeError("output.format must be 'cfradial' or 'dorade'")
    if outp["format"] == "dorade":
        outp = dict({"compat": "standard", "fields": None}, **outp)
        if outp["compat"] not in ("standard", "dreader"):
            raise RecipeError("output.compat must be 'standard' or 'dreader'")
        if outp["fields"] is not None and not (isinstance(outp["fields"], list)
                                               and all(isinstance(x, str) for x in outp["fields"])):
            raise RecipeError("output.fields must be a list of field names")
    return {"gurt_recipe": RECIPE_VERSION, "input": inp, "steps": out_steps, "output": outp}


def _check_values(i, s):
    op = s["op"]
    def pos(k, allow_zero=False):
        v = s.get(k)
        if v is None:
            return
        if not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0 or (v == 0 and not allow_zero):
            raise RecipeError(f"step {i} ({op}): '{k}' must be a {'non-negative' if allow_zero else 'positive'} number")
    if op == "despeckle":
        pos("radius"); pos("min_good", True); pos("max_pass", True)
    elif op == "deglitch":
        pos("radius"); pos("thresh"); pos("max_pass", True); pos("nyquist")
        pos("gapfill_n_neighbors"); pos("gapfill_n_passes")
        if s["gapfill_method"] not in ("median", "mean"):
            raise RecipeError(f"step {i} (deglitch): gapfill_method must be 'median' or 'mean'")
    elif op == "unfold":
        pos("radius", True); pos("thresh"); pos("nyquist"); pos("az_tol", True); pos("fixed_angle_tol", True)
        if s["reference"] not in ("none", "previous_sweep"):
            raise RecipeError(f"step {i} (unfold): reference must be 'none' or 'previous_sweep'")
    elif op == "gapfill":
        pos("n_neighbors"); pos("n_passes")
        if s["method"] not in ("median", "mean"):
            raise RecipeError(f"step {i} (gapfill): method must be 'median' or 'mean'")
    elif op == "smooth_velmag":
        pos("length_m")
        if not (0 < float(s["min_frac"]) <= 1):
            raise RecipeError(f"step {i} (smooth_velmag): min_frac must be in (0, 1]")
    elif op == "gatefilter":
        if s["condition"] not in GATE_CONDITIONS:
            raise RecipeError(f"step {i} (gatefilter): condition must be one of "
                              f"{', '.join(repr(c) for c in GATE_CONDITIONS)}")
        for k in ("value", "value2"):
            if not isinstance(s[k], (int, float)) or isinstance(s[k], bool):
                raise RecipeError(f"step {i} (gatefilter): '{k}' must be a number")
    elif op == "drop":
        if not isinstance(s["fields"], list) or not all(isinstance(x, str) for x in s["fields"]):
            raise RecipeError(f"step {i} (drop): 'fields' must be a list of field names")


def step_out_name(s):
    if s["op"] == "drop":
        return None
    if s.get("out"):
        return s["out"]
    if s["op"] == "smooth_velmag":
        return s["field"] + "-vmag"
    return s["field"]


def describe_step(s):
    op = s["op"]
    if op == "drop":
        return "drop " + ", ".join(s["fields"])
    if op == "gatefilter":
        return (f"gatefilter {s['field']}: mask where {s['by']} {s['condition']} {s['value']}"
                + (f" .. {s['value2']}" if s["condition"] in ("between", "outside") else "")
                + f" -> {step_out_name(s)}")
    params = ", ".join(f"{k}={v}" for k, v in s.items()
                       if k not in ("op", "field", "out", "note") and v is not None)
    return f"{op} {s['field']} -> {step_out_name(s)}" + (f"  ({params})" if params else "")


def list_input_files(rec, folder):
    inp = rec["input"]
    pat = inp.get("glob") or "*"
    found = []
    if inp.get("recursive"):
        for d, _, fs in os.walk(folder):
            found += [os.path.join(d, f) for f in fs if fnmatch.fnmatch(f, pat)]
    else:
        found = [os.path.join(folder, f) for f in os.listdir(folder)
                 if fnmatch.fnmatch(f, pat) and os.path.isfile(os.path.join(folder, f))]
    return sorted(p for p in found if not os.path.basename(p).startswith("."))


def group_inputs(files):
    """Group input files into volumes. DORADE sweepfiles hold one sweep each, named
    swp.<yyyymmddhhmmss-ish>.<radar>.<ms>.<angle>_<mode>_v<n>; those sharing swp.<time>.<radar> are one volume
    and are processed together (so the volume is written once, not once per sweep). Any other file is its own
    group. Returns a list of path lists, in input order."""
    groups, order = {}, []
    for p in files:
        parts = os.path.basename(p).split(".")
        key = (os.path.dirname(p),) + tuple(parts[:3]) if parts[0] == "swp" and len(parts) >= 6 else (p,)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(p)
    return [groups[k] for k in order]


def _load_group(paths):
    """Load one volume from one or more files; sweepfiles are merged, ordered by fixed angle."""
    from gurt_core import load_radar_file
    vols = [load_radar_file(p) for p in paths]
    vol = vols[0]
    if len(vols) > 1:
        sweeps = [sw for v in vols for sw in v.sweeps]
        sweeps.sort(key=lambda sw: (str(getattr(sw, "scan_type", "ppi")), float(getattr(sw, "fixed_angle", 0.0) or 0.0)))
        vol.sweeps = sweeps
        names = list(vol.field_names)
        for v in vols[1:]:
            names += [n for n in v.field_names if n not in names]
        vol.field_names = names
    return vol


# ── execution ────────────────────────────────────────────────────────────────

def _sweep_nyquist(vol, sw, override):
    if override:
        return float(override)
    for ip in (getattr(sw, "instrument_parameters", None), getattr(vol, "instrument_parameters", None)):
        try:
            d = np.asarray(ip["nyquist_velocity"]["data"], dtype=float)
            d = d[np.isfinite(d) & (d > 0)]
            if d.size:
                return float(np.median(d))
        except (KeyError, TypeError, IndexError):
            pass
    raise RecipeError("no Nyquist velocity in the file; set 'nyquist' in the deglitch step")


def _as_masked(a):
    a = np.ma.masked_invalid(np.ma.asanyarray(a))
    return a


def _apply_step(vol, sw, s, prev=None):
    """Run one step on one sweep; returns (new_field_array or None, gates_changed)."""
    op, f = s["op"], s["field"]
    data = sw.fields[f]
    if op == "despeckle":
        from edit import despeckle
        new, n, _ = despeckle.despeckle_field(data, s["radius"], s["min_good"], s["max_pass"], s["wrap"])
        return new, int(n)
    if op == "deglitch":
        from edit.deglitch import _run_core, gap_fill
        bad = -9999.0
        m = _as_masked(data)
        orig_mask = np.ma.getmaskarray(m).copy()
        filled = np.ma.filled(m, bad).astype(np.float64)
        nyq = np.full(filled.shape[0], _sweep_nyquist(vol, sw, s["nyquist"]), dtype=np.float64)
        corr = _run_core(filled, bad, nyq, int(s["radius"]), float(s["thresh"]),
                         int(s["max_pass"]), bool(s["delete_bad"]), numprt=1)
        res = np.ma.masked_where(np.abs(corr - bad) <= 1e-5, corr)
        if s["gapfill"]:
            res = gap_fill(res, orig_mask=orig_mask, n_neighbors=int(s["gapfill_n_neighbors"]),
                           method=s["gapfill_method"], n_passes=int(s["gapfill_n_passes"]))
        res = res.astype("float32")
        changed = int(np.sum(~np.isclose(np.ma.filled(m, np.nan), np.ma.filled(res, np.nan),
                                          equal_nan=True, atol=1e-3)))
        return res, changed
    if op == "unfold":
        from edit.unfold import unfold_field, match_reference
        nyq = _sweep_nyquist(vol, sw, s["nyquist"])
        m = _as_masked(data)
        ref = rmap = gmap = None
        if s["reference"] == "previous_sweep" and prev is not None:
            ref = _as_masked(prev.fields[step_out_name(s)] if step_out_name(s) in prev.fields
                             else prev.fields[f])
            rmap, gmap = match_reference(sw.azimuth, sw.ranges, prev.azimuth, prev.ranges, s["az_tol"])
        new, n = unfold_field(m, nyq, int(s["radius"]), float(s["thresh"]), ref, rmap, gmap)
        return new.astype("float32"), int(n)
    if op == "gapfill":
        from edit.deglitch import gap_fill
        m = _as_masked(data)
        res = gap_fill(m, orig_mask=None, n_neighbors=int(s["n_neighbors"]),
                       method=s["method"], n_passes=int(s["n_passes"])).astype("float32")
        return res, int(np.ma.getmaskarray(m).sum() - np.ma.getmaskarray(res).sum())
    if op == "smooth_velmag":
        from velmag import smoothed_velmag
        return smoothed_velmag(sw, f, s["length_m"], s["min_frac"]), 0
    if op == "gatefilter":
        from edit.gatefilter import build_filter
        cond = GATE_CONDITIONS[s["condition"]]
        if s["by"] not in sw.fields:
            raise RecipeError(f"gatefilter: filter field '{s['by']}' is missing in a sweep that has '{f}'")
        gf = build_filter(sw, s["by"], cond, float(s["value"]), float(s["value2"]))
        vals = np.ma.filled(data, np.nan).astype("float32", copy=True)
        n = int(np.count_nonzero(gf._excluded & np.isfinite(vals)))
        vals[gf._excluded] = np.nan
        return np.ma.masked_invalid(vals), n
    if op == "copy":
        return data.copy(), 0
    raise RecipeError(f"unhandled op {op}")


def run_steps(vol, steps):
    """Apply *steps* to every PPI sweep of *vol* in place. Returns a list of per-step stats."""
    stats = []
    for i, s in enumerate(steps, 1):
        op = s["op"]
        if op == "drop":
            for sw in vol.sweeps:
                for k in s["fields"]:
                    sw.fields.pop(k, None)
            vol.field_names = [n for n in vol.field_names if n not in s["fields"]]
            stats.append({"step": i, "op": op, "sweeps": len(vol.sweeps), "gates": 0})
            continue
        dst = step_out_name(s)
        nsw = gates = 0
        for si, sw in enumerate(vol.sweeps):
            if s["field"] not in sw.fields or getattr(sw, "scan_type", "ppi") == "rhi":
                continue
            prev = None
            if op == "unfold" and s["reference"] == "previous_sweep":
                for pj in range(si - 1, -1, -1):          # nearest earlier sweep at a similar angle
                    ps = vol.sweeps[pj]
                    if (s["field"] in ps.fields and getattr(ps, "scan_type", "ppi") != "rhi"
                            and abs(float(ps.fixed_angle) - float(sw.fixed_angle)) < s["fixed_angle_tol"]):
                        prev = ps
                        break
            new, n = _apply_step(vol, sw, s, prev)
            sw.fields[dst] = new
            nsw += 1
            gates += n
        if nsw == 0:
            raise RecipeError(f"step {i} ({op}): no PPI sweep has field '{s['field']}'")
        if dst not in vol.field_names:
            vol.field_names.append(dst)
        stats.append({"step": i, "op": op, "sweeps": nsw, "gates": gates})
    return stats


def check_fields(field_names, steps):
    """Dry-run the field bookkeeping; raises RecipeError if a step reads a missing field."""
    have = list(field_names)
    for i, s in enumerate(steps, 1):
        if s["op"] == "drop":
            have = [f for f in have if f not in s["fields"]]
            continue
        if s["field"] not in have:
            raise RecipeError(f"step {i} ({s['op']}): field '{s['field']}' is not available here "
                              f"(have: {', '.join(have)})")
        if s["op"] == "gatefilter" and s["by"] not in have:
            raise RecipeError(f"step {i} (gatefilter): filter field '{s['by']}' is not available here "
                              f"(have: {', '.join(have)})")
        d = step_out_name(s)
        if d not in have:
            have.append(d)
    return have


def process_file(path, rec, out_dir):
    """Load *path* (one file, or a list of DORADE sweepfiles of one volume), run the recipe, write CfRadial
    into *out_dir*. Returns a result dict."""
    from exporters import write_cfradial
    paths = [path] if isinstance(path, str) else list(path)
    path = paths[0]
    vol = _load_group(paths)
    stats = run_steps(vol, rec["steps"])
    os.makedirs(out_dir, exist_ok=True)
    if rec["output"]["format"] == "dorade":
        from exporters import write_dorade
        paths_out = write_dorade(vol, out_dir, fields=rec["output"].get("fields"),
                             compat=rec["output"].get("compat", "standard"))
        outs = paths_out
        return {"input": path, "inputs": paths, "output": outs[0] if len(outs) == 1 else f"{len(outs)} sweepfiles in {out_dir}",
                "outputs": outs, "steps": stats}
    out = write_cfradial(vol, out_dir)
    if os.path.abspath(out) == os.path.abspath(path):
        raise RecipeError("output would overwrite the input file; choose a different output folder")
    try:
        import netCDF4
        with netCDF4.Dataset(out, "a") as nc:
            nc.history = (str(getattr(nc, "history", "") or "") +
                          "GURT recipe: " + json.dumps(rec["steps"], separators=(",", ":")) +
                          f"; input={os.path.basename(path)}")
    except Exception:
        pass
    return {"input": path, "inputs": paths, "output": out, "steps": stats}


def new_recipe(steps=None):
    return {"gurt_recipe": RECIPE_VERSION, "input": {"glob": "*", "recursive": False},
            "steps": list(steps or []), "output": {"format": "cfradial"}}


def save_recipe(rec, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)
        f.write("\n")
