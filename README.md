# GURT

GURT is a vispy + Tk radar viewer/editor (with libgurt, an optional C backend for the
despeckle / deglitch / gap-fill / unfold / tormodel and, in the full release, axisymmetric-analysis routines).

This is the full release, including DREADER-derived vortex detection/tracking and
axisymmetric analysis. (Set `GURT_LITE=1` to disable those two tools and show empty
placeholders instead.)

## Batch processing

Record editing steps in the GUI (**Recipe** menu) or write a JSON recipe, then run it over a
folder headlessly with `python gurt_cli.py run recipe.json IN_DIR -o OUT_DIR`. See
[RECIPES.md](RECIPES.md).

## Tornado wind model (tormodel)

The full write-up (parameter file, the model itself, swaths, speed, troubleshooting) is in [`TORMODEL.md`](TORMODEL.md). What follows is the short version.

A port of DREADER's `tormodel`: reads an ASCII parameter file of radar tornado observations plus
output-grid specs and writes a CSV of the interpolated analytic wind field. Three modes: `point`
(time series at an x,y or azimuth/range point), `grid` (x,y grid at a time) and `maxgrid` (x,y grid of
the maximum at each node). Use **Analysis > Tornado wind model (tormodel)...** or
`python gurt_cli.py tormodel PARAMS --grid --time 60 -o out.csv` (`tormodel --example` prints a sample
parameter file). With libgurt the output is identical to DREADER's. Deliberate differences: grid-mode time
is in seconds regardless of the time step, `maxgrid` covers the whole period by default, and output files
are overwritten rather than appended.

**Swaths.** Choose *Swath ASCII (.txt)* (CLI: `--swath`) to write the `grid` or `maxgrid` result in the same
swath text layout as the box-tracking wind swath export (UT/XG/YG/ZG/X0/Y0/nx/ny/dx/dy/sn/pr/un header, then rows
north to south, CRLF; X0 / Y0 are always the grid's south-west corner; `--south-first` flips it). Every swath file is named
`swath.<name>`: the prefix is added when missing, so `-o 25mtornado.txt` writes `swath.25mtornado.txt`. A `maxgrid` swath is the maximum wind through the chosen time, or the whole period.

**Folder run.** *Run over folder* (CLI: `tormodel params.txt --folder DIR --maxgrid --swath -o folder_swath.txt`,
optional `--glob` and `--time-offset SECONDS`) reads the UTC time of every radar file in the folder and writes ONE file: the
maximum model wind over the time span the scans cover (first ray of the first scan to last ray of the last scan), as a swath
.txt or a grid CSV. Scans outside the parameter file's observation period, or unreadable files, are skipped and reported;
it is an error if none overlap.

**Resolution.** The grid comes from the parameter file, so set a small `dx` / `dy` (and a larger `nx` / `ny`) for a
fine swath, or leave the file alone and use *Grid spacing override* in the dialog (CLI: `--res 50 --dt 5`), which keeps the
grid's extent. Cost is cells x time steps: a 641 x 641 grid over 1926 one-second steps took about 80 s with libgurt;
`--dt 5` cuts that five-fold and changes the swath very little because the tornado moves slowly.

**Parameter file from tracking (full build).** In the Vortex detection window, *Export tormodel params...* writes a
tormodel parameter file with one observation per tracked detection, so the model follows the tracked vortex instead of a
hand-typed track. Position = the detection's azimuth / range from the seed radar (via lat/lon, so a moving radar works),
radius = half the couplet diameter, maximum tangential wind = half the couplet delta-V; radial wind (fraction of tangential,
0 = none) and the exponents are set in the dialog. The grid covers the track plus a margin. Observations in the same second
keep the strongest; a minimum spacing can thin them. Then run the file (or a whole folder) in the tormodel dialog.

**Variational Vr fit (optional, full build).** Tick *Fit the radial wind to the radar velocities* in the export dialog and
the radial columns (`maxrad`, `corerad`, `decayrad`) are fitted to the Doppler velocities of every tracked scan instead of
being typed in. The tangential profile and the track stay as tracked. One cost function is minimised over the whole track
(`vrfit.py`): the squared misfit between the observed velocity and the model's beam-component velocity near each vortex
(gates within 3 radii, at least 1.2 km), plus a penalty on scan-to-scan change of the parameters (*time coupling*; `0` =
each scan alone, larger = stiffer) and a weak pull toward the dialog's starting values. A uniform background wind is
fitted per scan, so the environmental flow and the vortex translation are not mistaken for inflow. A scan whose file has no
usable velocity follows its neighbours. The dialog reports the misfit before / after and the fitted `maxrad` range
(negative = inflow). Exponents are kept inside 0.2-3 (core) and -3 to -0.05 (decay).

**Track smoothing.** tormodel adds the interpolated track's translation speed to the wind, so a jittery or mislinked track
gives absurd winds (a raw detection track once produced 420 m/s). The track export therefore removes detections far off the
path, cuts the track at jumps faster than 60 m/s (keeping the longest continuous part) and smooths position, radius and wind
over +-60 s by default (dialog fields; `0` = raw). The same smoothing can be applied to any existing parameter file:
*Smooth the observed track* in the tormodel dialog, or `--smooth 60` on the command line; `--write-params FILE` saves the
smoothed (and re-gridded, with `--res` / `--dt`) file instead of running the model.

**Speed.** The max grid skips time steps that cannot beat the best value found so far (exactly, without changing a
digit) and splits the grid into strips that run on all CPU cores. Set `GURT_THREADS=n` to use `n` threads
(`1` = single thread). A 50 m grid over 1925 steps takes about 2 s on 2 cores; use a larger `--dt` (steps of 3-5 s are
usually indistinguishable) and a coarser `--res` for previews, and the full resolution for the final file.

## License

GURT is free software, licensed under the **GNU General Public License, version 2 only**
(`GPL-2.0-only`). See [LICENSE](LICENSE). Each GURT source file carries an
`SPDX-License-Identifier: GPL-2.0-only` header.

## Credits and third-party code

* **DREADER** (GPL-2.0, per its SourceForge page) - the despeckle, deglitch, threshold and cross-section algorithms
  (`edit/despeckle.py`, `edit/deglitch.py`, `libgurt/`, `gui/velmag.py`, `gui/dreader_xsect.py`,
  `gui/dreader_detect.py`, `gui/dreader_track.py`, `gui/dreader_modes.py`, `gui/axisym_overlay.py`)
  are ports of DREADER's C code (`qualitycontrol.c` and related files).
* `libgurt/src/tormodel.c`, `tormodel.py` - port of DREADER's `tormodel`; `libgurt/src/axisym.c` ports the
  extractor axisymmetric / patch modes (GPL-2.0, C. Alexander).
* `readers/nexrad_level2.py` - derived from Py-ART, BSD-3-Clause (UChicago Argonne, LLC). Its
  original notice is kept in the file.
* `exporters/nexrad_writer.py` - modified from NCAR LROSE code (LROSE BSD license, GPL-compatible).
  Marked as modified; full license text in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
* Runtime dependencies (vispy, matplotlib, numpy, Py-ART, netCDF4, Pillow, ...) keep their own
  licenses. 
