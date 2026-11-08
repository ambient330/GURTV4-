# tormodel

tormodel builds a wind field around a tornado from a handful of radar observations. You give it the
vortex position, size and peak winds at a few times, it fills in the times between, and it tells you the
wind at any point on a grid. The usual use is a damage swath: the highest wind every grid cell saw while
the tornado went by.

It is a port of the `tormodel` program from DREADER. Run it with libgurt built and the numbers match
DREADER's, digit for digit. Without libgurt a NumPy version runs instead and agrees to rounding
(the last printed digit can differ by 1 now and then).

## Running it

```
python gurt_cli.py tormodel params.txt --maxgrid --swath -o 25mtornado.txt
```

or use Analysis > Tornado wind model (tormodel)... in the GUI. `tormodel --example` prints a small
parameter file you can edit.

Three things it can compute:

| mode | what you get |
|---|---|
| `--point` | wind versus time at one place (give `--xpos --ypos`, or `--azim --rang`) |
| `--grid` | wind speed over the grid at one moment (`--time` in seconds from the start) |
| `--maxgrid` | highest wind speed at each cell up to `--time`; leave `--time` out for the whole period |

Without `-o` the CSV goes to the terminal.

## The parameter file

Plain text, whitespace separated, `#` starts a comment. Order matters.

```
# radar  lat      lon       heading  elevation(m)
KOUN     35.2358  -97.4619  0.0      379
# x0   dx   nx
-4000  200  40
# y0   dy   ny
-1000  200  40
# t0 (epoch s, -1 = first obs)   dt (s)   nt (-1 = to the last obs)
-1  1  -1
# yyyy mm dd HH MM SS  azim  rang    radius  maxtan coretan decaytan maxrad corerad decayrad
2013 5 20 20 0 0       30.0  5000.0  150.0   60.0   1.0     -0.6      -10.0  1.0     -0.6
2013 5 20 20 1 0       32.0  5400.0  200.0   75.0   1.0     -0.6      -12.0  1.0     -0.6
```

The radar line is only used to label output and to turn grid points into latitude and longitude. The
heading is carried along and not used for anything.

The three grid lines are `start step count`. x is metres east of the radar, y is metres north. The grid has
`nx` steps, so the output has `nx + 1` columns (it includes both ends). The time line works the same way, in
seconds. `-1 1 -1` means start at the first observation, step one second, run to the last observation, and
that is what you want nearly all the time.

Each observation line is a UTC time followed by nine numbers:

| column | meaning |
|---|---|
| `azim`, `rang` | where the vortex centre is: degrees clockwise from north, metres from the radar |
| `radius` | radius of maximum wind, metres |
| `maxtan` | peak tangential wind at that radius, m/s |
| `coretan`, `decaytan` | exponent of the tangential wind inside / outside the radius |
| `maxrad` | peak radial wind at that radius, m/s (negative is inflow) |
| `corerad`, `decayrad` | the same two exponents for the radial wind |

Times must be strictly increasing and you need at least two observations.

## What the model does

Position is a cubic spline through the observations. Radius, peak winds and exponents are interpolated
linearly between observations. At each second it takes the 3-point derivative of the position and calls that the
translation speed.

The wind at a grid point a distance r from the centre is

```
Vt = maxtan * (r / radius) ^ coretan       r <= radius
Vt = maxtan * (r / radius) ^ decaytan      r  > radius
```

and the same for the radial part with `maxrad`, `corerad`, `decayrad`. Vt turns counterclockwise, Vr points
out. The translation vector is then added on top of both. The speed printed is the length of that sum.

Two consequences that surprise people:

1. The translation speed is added to the wind everywhere, not only near the vortex. If the track is jittery
   (detections that wander a few hundred metres from scan to scan) the model sees a tornado racing around
   at 100 m/s and the swath shows it. One raw track gave 420 m/s. This is why GURT smooths tracks before it
   writes a parameter file, see below.
2. With a decay exponent near -0.6 the wind falls off slowly. At 20 radii it is still about 17 percent of the peak.
   A single bad second in the track can lift winds over a few kilometres. A steeper exponent
   (-1 or so) keeps the far field small if that matches the tornado better.

## Output

CSV by default. The grid modes write a title line, a blank line and then `ny + 1` rows of `nx + 1` values, northern row first
(the way a map is drawn).
`--header` adds the radar and grid description at the top, for tools that expect DREADER's layout.

Where this differs from DREADER, on purpose: `--grid` takes its time in seconds whatever the time step is
(DREADER used it as an array index, which only works for a 1 s step), `--maxgrid` with no time covers the
whole period (DREADER did only the first step), the point header shows the position actually used, and the
output file is overwritten instead of appended to.

### Swath files

`--swath` writes the grid or maxgrid result as a swath text file, the same layout GURT's box-tracking wind
swath export uses:

```
UT  (unix time)
XG  radar longitude
YG  radar latitude
ZG  radar altitude, m
X0  Y0  west and south edge of the grid, m from the radar
nx  ny
dx  dy
sn  sensor name
pr  product name
un  unit
```

then `ny` rows of `nx` numbers, `%.4f`, CRLF line endings. Rows run north to south. `X0` and `Y0` are always
the south-west corner of the grid, whichever way the rows run. `--south-first` writes the rows
south to north instead, and `--sensor`, `--product`, `--fill` set the name fields and what goes in cells with
no wind.

The file name always starts with `swath.`. If you ask for `25mtornado.txt` you get `swath.25mtornado.txt`.
Ask for `swath.whatever.txt` and it is left alone. `ZG` is taken from the radar elevation, and anything
outside -500 to 12000 m is replaced with 0, because some radar files carry junk there.

## Running over a folder

```
python gurt_cli.py tormodel params.txt --folder /data/scans --maxgrid --swath -o folder.txt
```

Reads the time of every radar file in the folder and writes one file: the highest wind over the whole period
the scans cover, from the first ray of the first scan to the last ray of the last scan. Scans outside the
observation period and files that will not load are skipped and listed. If nothing overlaps it stops with an error.
`--glob` picks which files, `--time-offset SECONDS` fixes a radar clock that is off.

DORADE sweepfiles keep ray times as seconds from the start of the sweep. GURT adds the sweep start for you, so
`swp.*` folders work.

## Grid resolution and speed

The grid comes from the parameter file. To change it without editing the file, use `--res 50` (metres, x and y)
and `--dt 5` (seconds). The extent stays the same.

Work is cells times time steps. A 10 m grid over a 17 km track is a lot of cells. Some numbers from one
machine, a 50 m grid and 1925 steps:

| | time |
|---|---|
| first version | 12 s |
| after the pruning in libgurt, 1 core | 4.5 s |
| 2 cores | 2.3 s |

The max grid skips any time step that cannot beat what a cell already has, which does not change a single
digit of the output, and splits the rows over all your cores. Set `GURT_THREADS=1` to turn the threads off
(or `=n` to cap them). Going from 1 s to 5 s steps cuts the work five-fold and the swath hardly changes,
because a tornado does not move far in 5 s. A good habit is `--res 50 --dt 5` while you are tuning
a parameter file and the real resolution once at the end.

## Making a parameter file from GURT tracking

In the vortex detection window, Export tormodel params... writes one observation per tracked detection:

* position is the detection's azimuth and range from the seed radar (through latitude and longitude, so a
  moving radar is fine),
* radius is half the couplet diameter,
* maxtan is half the couplet delta-V,
* maxrad is a fraction of maxtan you pick in the dialog (0 means no radial wind).

The grid covers the track plus a margin of three times the largest radius (at least 1 km).

Track smoothing happens here, and it matters. The export throws out detections that are far off the path, cuts the
track at any jump faster than 60 m/s and keeps the longest piece that is left, then smooths position, radius and wind
over +-60 s with a robust local-linear fit. Set the smoothing to 0 to see the raw detections. The same smoothing is
available for any existing file: `--smooth 60`, and `--write-params out.txt` saves the smoothed, re-gridded file
without running the model.

If you ever see winds far above what the radar measured, look at the translation speed first. In a track
that has one mislinked detection it jumps for a few scans and the model adds all of it to the wind.

### Fitting the radial wind to the radar (optional)

The radial columns are normally guesses. Tick the fit option in the export dialog and GURT fits them to the
Doppler velocities of every scan instead. Only the radial profile (`maxrad`, `corerad`, `decayrad`) is touched.

It minimises one cost over the whole track: the squared difference between observed velocity and the
model's beam velocity near each vortex, plus a penalty on change from one scan to the next (the time
coupling field, 0 fits each scan alone), plus a weak pull toward the values you started with. A uniform
background wind is solved for each scan so the surrounding flow is not read as inflow. Scans without usable
velocity follow their neighbours. The dialog shows the misfit before and after and the fitted `maxrad` range.
The code is `vrfit.py`, and `tests/test_vrfit.py` shows how it behaves on made-up vortices.

Treat the result as a starting point. It only knows the beam component of the wind, and a single Doppler
velocity cannot separate every combination of the three parameters.

## Odd things to check when a swath looks wrong

* Swath file sits far from where you expect. Check that `X0` and `Y0` in the header are the south-west corner. A viewer that
  reads rows the other way will put the whole grid one grid height off.
* No data on one side of the track. The grid may be too small. Widen the margin or the `x`/`y` lines.
* One huge blob around a spot on the track. Look at the track near that time (see translation above).
* Blank or flat output. `t0` and `nt` might put the time window outside the observations.
* Very slow. Check `nx * ny * nt` first. Raise `--dt`.
