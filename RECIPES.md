# GURT recipes (command-line batch processing)

A recipe is a small JSON file listing editing steps. GURT can record one while you work
(**Recipe** menu), and `gurt_cli.py` replays it over a whole folder with no GUI, so it also
runs over SSH or on a cluster.

```bash
python gurt_cli.py ops                                  # list steps and their parameters
python gurt_cli.py init my_recipe.json                  # write an example recipe
python gurt_cli.py validate my_recipe.json
python gurt_cli.py run my_recipe.json /data/in -o /data/out --dry-run
python gurt_cli.py run my_recipe.json /data/in -o /data/out -j 4 --log run.json
```

Needs numpy, netCDF4 and numba (Tk and vispy are **not**
needed). If libgurt is built (`libgurt/README.md`) the despeckle / deglitch / gap-fill steps use
it; `GURT_NO_NATIVE=1` forces the Python versions (identical results, slower).

Worked examples with notes are in [`examples/README.md`](examples/README.md).

## Recording from the GUI

1. Keep **Recipe → Record applied steps** ticked.
2. Apply tools as usual. Recorded so far: **Despeckle** and **Unfold** (when applied to *All sweeps*), **GateFilter** (scope
   *volume*) and **Smoothed velocity magnitude**.
3. **Recipe → Show recipe… / Remove last step / Clear recipe**, then **Save recipe…**.

Brush strokes and boundary edits are hand-drawn on one sweep, so they are not recorded.

See `examples/ve_cleanup.json` for a worked example (gate-filter VE on DBMHC < -100, copy to
VE-edit, despeckle, deglitch, write CfRadial).

## Format

```json
{
  "gurt_recipe": 1,
  "input":  {"glob": "*", "recursive": false},
  "steps": [
    {"op": "despeckle", "field": "VE", "radius": 2, "min_good": 12, "max_pass": 3, "out": "VE-edit"},
    {"op": "deglitch",  "field": "VE-edit", "radius": 2, "thresh": 15.0, "gapfill": true},
    {"op": "smooth_velmag", "field": "VE-edit", "length_m": 200, "out": "VMAG"}
  ],
  "output": {"format": "cfradial"}
}
```

* Each step runs on every PPI sweep that has `field` (RHI sweeps are skipped). A step whose
  field exists in no sweep stops that file with an error.
* `out` is the result field. Leave it out to overwrite `field` in memory. The input files are
  never modified; results are written as CfRadial into the `-o` folder (never into the input
  file itself).
* Steps run in order, so later steps can read earlier results (`VE-edit` above).
* `"output"` can also write DORADE sweepfiles (one `swp.*` file per sweep):
  `{"format": "dorade", "compat": "dreader", "fields": ["VE", "VE-edit"]}` (`fields` optional = all;
  `compat` `standard` or `dreader`, see below). See `examples/ve_cleanup_dorade.json`.
* The output file's `history` attribute records the recipe that produced it.
* `.yaml` recipes work too if PyYAML is installed.
* Any step may carry a `"note"` string (ignored by the runner) to document the recipe.
* Deglitch takes the Nyquist velocity from the file; set `"nyquist": 26.3` to override it.

| op | what it does | key parameters (defaults) |
|---|---|---|
| `despeckle` | DREADER iterative despeckle | `radius` 2, `min_good` 10, `max_pass` 0 (until stable), `wrap` true |
| `deglitch` (GUI: Edit ▸ Deglitch, recorded when applied to all sweeps) | DREADER deglitch (fold toward local median by 2×Nyquist) | `radius` 2, `thresh` 15, `max_pass` 0, `delete_bad` true, `nyquist`, `gapfill` false, `gapfill_n_neighbors` 3, `gapfill_method` median, `gapfill_n_passes` 5 |
| `unfold` | DREADER unfold/dealias: along-ray median reference, steps of 2×Nyquist | `radius` 2, `thresh` 0.8 (fraction of Nyquist), `nyquist` (from file), `reference` `none` or `previous_sweep`, `az_tol` 0.5, `fixed_angle_tol` 10 |
| `gapfill` | fill masked gates from nearest good gates along the ray | `n_neighbors` 3, `method` median, `n_passes` 5 |
| `smooth_velmag` | smoothed \|V\| | `length_m` 200, `min_frac` 0.5 |
| `gatefilter` | mask gates of `field` where the `by` field meets a condition (like Edit ▸ GateFilter) | `field`, `by`, `condition` (`<` `<=` `>` `>=` `==` `!=` `between` `outside` `nan`), `value`, `value2`, `out` |
| `copy` | copy a field under a new name | `field`, `out` |
| `drop` | remove fields from the output | `fields` (list) |

Exit status is 1 if any file failed, so scripts can check it. Failed files are reported and the
rest still run.

## DORADE export

**Edit ▸ Export DORADE sweepfiles…** (and `"output": {"format": "dorade"}` in recipes) writes one big-endian
sweepfile per sweep: `swp.<YYMMDDHHMMSS, year-1900>.<radar>.<msec>.<angle>.<tenths>_<PPI|RHI>_v<volume>`.
Fields are stored as 16-bit integers with a scale chosen per field (0.01 for velocities and dBZ, finer for
ratios), bad = -32768. DORADE limits field names to 8 characters (longer ones are shortened and kept
unique) and a CELV range block to 1500 gates (uniformly spaced longer sweeps use a CSFD cell-spacing block).

`compat: "dreader"` (the **DREADER-compatible range block** menu option, on by default in the GUI) writes
the CELV block with 8192 slots, which is what the DREADER build in this source tree expects. Other DORADE
readers prefer the standard block (`compat: "standard"`, the default for recipes and the Python API).
Verified: GURT's reader round-trips the data (to the 0.01 quantisation), and DREADER's own `editor`
reads the files, copies a field and rewrites them.
