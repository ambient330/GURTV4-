# Example recipes

Four recipes to copy and edit. They all do the same kind of job: take a folder of radar files, clean the
velocity, write the result. The full reference for every step is in `../RECIPES.md`.

Run one like this (from the GURT folder):

```
python gurt_cli.py validate examples/ve_cleanup.json
python gurt_cli.py run examples/ve_cleanup.json /data/scans -o /data/cleaned --dry-run
python gurt_cli.py run examples/ve_cleanup.json /data/scans -o /data/cleaned -j 4 --log run.json
```

`validate` just checks the recipe and prints the steps. `--dry-run` lists what would be processed and
writes nothing. Do that once before a long run. Your input files are never touched, the output goes in the
`-o` folder.

## What is in here

**ve_cleanup.json** is the one I would start from. It assumes the velocity field is
called `VE` and that there is a power field `DBMHC`. It masks velocity wherever power is below -100 dBm, copies the
result to `VE-edit`, despeckles that, then deglitches it. The original `VE` stays in the output next to
`VE-edit`, so you can flip between them in the viewer.

**ve_cleanup_dorade.json** is the same recipe but the output is DORADE sweepfiles instead of CfRadial, with
the DREADER-compatible range block. Use it when the next stop is DREADER.

**unfold_clean.json** is for velocity that is folded and needs dealiasing. Field name is `VEL` (change it for yours).
Despeckle first, because the unfolder believes whatever gates it is given and a lone bad gate will start
a wrong unfold. Then `unfold` with `previous_sweep` as the reference, then a deglitch with gap fill to
mop up. If your scans are a single sweep, set `reference` to `none`.

**velmag_swath_prep.json** cleans velocity, makes the smoothed |V| field `VMAG` (200 m boxcar) and drops the raw
velocity so the files are small. VMAG is what the wind swath is built from.

## Changing a recipe

Field names are the thing you will change most. Check what your files hold first. `ncdump -h file.nc`
works for CfRadial, or open the file in GURT and look at the field menu. A step that names a field no
sweep has stops that file with an error and the run moves on to the next file.

Steps run top to bottom and each one sees the result of the one above, so the order is part of the recipe.
`out` names the result. Leave it off and the step overwrites `field` in memory (still not in your input file).

Any step can carry a `"note"`. It does nothing, it is just there so you remember in a month why you picked a
threshold.

A few settings that come up:

* `despeckle`: `radius` 2 and `min_good` 10 to 12 is what DREADER users usually run. Raising `min_good`
  removes more. `max_pass` 0 means keep going until nothing changes; a small number stops early.
* `deglitch`: `thresh` is in m/s and the Nyquist comes from the file. If a file has no Nyquist, or a wrong
  one, give `"nyquist": 26.3` (your value) in the step.
* `gatefilter`: `condition` can be `<` `<=` `>` `>=` `==` `!=` `between` `outside` or `nan`. `between` and
  `outside` use `value` and `value2`.
* `unfold`: `thresh` is a fraction of the Nyquist. 0.8 is a fairly strict test.

## Folders of one-sweep files

Some radars (DORADE `swp.*` files) write one file per sweep. GURT groups the files that share a
`swp.<time>.<radar>` name and processes them as one volume, so the output is one file per volume with all its
sweeps in it. The run prints a line saying how many files it turned into how many volumes. Before this was
fixed, each sweep wrote to the same output name, and with `-j 4` they fought over it, so if you ran
into permission errors on an older copy, that was why.

## Recording a recipe in the GUI instead

Recipe menu, tick Record applied steps, then work on one file as usual. Despeckle, unfold (to all sweeps), gate filter
(volume scope) and smoothed velocity magnitude get recorded. Hand edits (brush, boundaries) do not. Save with
Recipe > Save recipe, then run it from the command line on the whole folder.

## Tips

* Try a recipe on three or four files before you start on 3000. The log file (`--log run.json`) shows how many gates each step changed
  on each file, which is a quick way to spot a threshold that does nothing or deletes everything.
* `-j` is the number of processes. More than the number of cores does not help, and on a laptop disk the
  limit is often reading and writing, not the maths.
* GURT uses the C code (libgurt) if it is built, and the Python code if it is not. Results are the same, the
  Python is just slower. `GURT_NO_NATIVE=1` forces Python if you ever want to compare.
* Exit status is 1 if any file failed, so a shell script can tell.
