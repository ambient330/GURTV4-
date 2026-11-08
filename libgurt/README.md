# libgurt

GURT's radar editing and QC algorithms in portable C99, with a ctypes binding so
the Python GURT can use them today and a future C GURT can link them directly.

## Build

    make                 # libgurt.so / libgurt.dylib / gurt.dll
    # or: cmake -B build && cmake --build build

### Windows

`make` produces a Linux library (`libgurt.so`); a Windows Python needs `gurt.dll`.
A prebuilt 64-bit `prebuilt/windows-x64/gurt.dll` is included and found
automatically. To rebuild it: `pip install ziglang` then `sh build_windows_dll.sh`
(works from WSL), or use Visual Studio Build Tools:

    cmake -B build
    cmake --build build --config Release

If you run GURT from WSL / Linux Python instead, `make` is the right build.

## Test (compares against GURT's pure-Python implementations)

    python tests/test_libgurt.py
    python tests/test_tormodel.py      # vs. output of DREADER's own tormodel binary
    python tests/test_axisym.py        # full tree: C vs Python reference

## Contents (v0.1)

| function | source | notes |
|---|---|---|
| `gurt_despeckle` | DREADER `despeck_parm` | iterative neighbour-count removal; optional circular rays |
| `gurt_deglitch`  | DREADER `deglitch_parm` / `edit/deglitch.py` | fold toward local median; single-PRT, two-PRT, same-field staggered retry |
| `gurt_gapfill`   | `edit/deglitch.py: gap_fill` | nearest-neighbour fill along each ray |
| `gurt_unfold`    | DREADER `dealias_parm` | Bargen-Brown along-ray median reference |
| `gurt_tor_*`     | DREADER `tormodel` | observation interpolation (cubic-spline track), wind grid / max grid / point time series; output identical to DREADER's |
| `gurt_axisym`, `gurt_patch`, `gurt_annulus_width` | DREADER `extractor` axisymmetric / patch modes | **full build only** (`src/axisym.c`, `include/gurt_axisym.h`); the lite tree omits these files and the build picks them up only when present |

## Brush design

Every routine takes an optional `region` mask (uint8 `[nrays][ngates]`). Only
gates inside it are modified; neighbours outside still count as context. A brush
circle or boundary polygon becomes a mask, and the same routine runs on it, so
brush editing and whole-sweep editing share one code path.

Planned next on the same pattern: brush footprint -> mask (circle / polygon in
km, using sweep geometry), unfold (fold multiplier / center modes), gradient
(gap fill by ordered neighbours), GateFilter expressions, then azimuthal shear,
vortex detection / tracking, cross-sections and the swath.

## Python

`python/gurt_native.py` wraps these (masked arrays in / out, `region=` keyword).
Set `LIBGURT_DIR` or put the library next to it; `gurt_native.available()` says
whether it was found, so GURT can fall back to the Python code.

## Use inside GURT

GURT picks libgurt up automatically if it has been built (`make` in `libgurt/`):
Edit -> Despeckle, the Deglitch / Gap Fill brushes (custom algorithm) and the
boundary deglitch all call it, and print `[GURT] libgurt 0.1.0 loaded` at startup.
Without the library they fall back to the Python code. `GURT_NO_NATIVE=1` forces
the Python path. `python libgurt/tests/test_integration.py` checks both paths agree.
