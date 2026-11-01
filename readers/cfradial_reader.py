#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-only
# GURT - Copyright (C) the GURT authors.
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the
# Free Software Foundation. It is distributed WITHOUT ANY WARRANTY; see the
# file LICENSE for details.
"""
readers/cfradial_reader.py — Pure-netCDF4 CFRadial reader for GURT.

No pyart dependency.  Lives in the ``readers/`` package alongside
``dorade_reader.py`` and ``grib2_reader.py``.  Re-exported from
``readers/__init__.py`` so ``main.py`` can simply do:

    from readers import read_cfradial

``read_cfradial(path)`` returns a ``RadarVolume`` populated with
``SweepData`` objects — exactly the same types that ``_load_dorade`` and
``_load_nexrad`` produce, so the rest of GURT works without any changes.

Time resolution (in order of preference)
-----------------------------------------
1. ``time`` variable units string  ("seconds since 2024-01-01T00:00:00Z")
2. ``start_datetime`` global attribute
3. ``time_coverage_start`` global attribute
4. Fallback: 2000-01-01T00:00:00

RHI detection
-------------
``sw.scan_type`` is set to ``"rhi"`` when the ``sweep_mode`` variable for
that sweep contains ``"rhi"`` (case-insensitive).  GURT's rendering code
reads this attribute to choose the correct geometry builder.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import numpy as np

try:
    import netCDF4
    NETCDF4_AVAILABLE = True
except ImportError:
    NETCDF4_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def _parse_datetime(s: str) -> Optional[datetime]:
    """Try several ISO-8601 / CF-like datetime strings."""
    if not s:
        return None
    s = str(s).strip().rstrip("Z").strip()
    for fmt in (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y%m%dT%H%M%S",
        "%Y-%m-%dT%H:%M:%S.%f",
    ):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _nc_scalar(nc, name: str) -> Optional[float]:
    if name not in nc.variables:
        return None
    try:
        v = float(np.ma.filled(np.asarray(nc.variables[name][:]), np.nan).ravel()[0])
        return v if np.isfinite(v) else None
    except Exception:
        return None


def _nc_median_positive(nc, name: str) -> Optional[float]:
    """Median of finite, positive values of a netCDF4 variable, or None."""
    if name not in nc.variables:
        return None
    try:
        arr = np.ma.filled(
            np.asarray(nc.variables[name][:], dtype="float64"), np.nan
        ).ravel()
        valid = arr[np.isfinite(arr) & (arr > 0)]
        if len(valid) == 0:
            return None
        v = float(np.nanmedian(valid))
        return v if np.isfinite(v) else None
    except Exception:
        return None


def _decode_char_strings(raw_var) -> list[str]:
    """
    Decode a netCDF4 fixed-width char variable (e.g. sweep_mode with dims
    (sweep, string_length_32)) into a list of Python strings, one per
    leading-dimension entry.

    Robust to every shape netCDF4-python can hand back depending on
    version / auto_chartostring setting:
      - already-decoded 1-D array of bytes/np.bytes_ (chartostring applied)
      - 2-D array of individual 'S1' chars, plain or masked
      - masked entries, which show up as None (or numpy.ma.masked) when
        pulled through .tolist() -- these are treated as blank chars
        rather than raising, since a mask here just means "not written",
        not "reading failed".
    """
    try:
        data = raw_var[:]
    except Exception:
        return []

    out: list[str] = []
    for row in data:
        if isinstance(row, (bytes, np.bytes_)):
            out.append(row.decode(errors="ignore"))
            continue
        if hasattr(row, "tolist"):
            chars = row.tolist()
        elif isinstance(row, (list, tuple)):
            chars = list(row)
        else:
            out.append(str(row))
            continue
        pieces = []
        for c in chars:
            if c is None or c is np.ma.masked:
                continue
            if isinstance(c, (bytes, np.bytes_)):
                pieces.append(c)
            elif isinstance(c, str):
                pieces.append(c.encode(errors="ignore"))
            else:
                # unexpected scalar type (e.g. masked constant) — skip it
                continue
        out.append(b"".join(pieces).decode(errors="ignore"))
    return out


# ──────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────

def read_cfradial(path: str, RadarVolume=None, SweepData=None):
    """
    Read a CFRadial (.nc / .h5) file and return a ``RadarVolume``.

    ``RadarVolume`` and ``SweepData`` are imported from ``main`` by default.
    Pass them explicitly when using this module standalone (e.g. in tests).

    Parameters
    ----------
    path : str
        Path to the CFRadial file.
    RadarVolume, SweepData : class, optional
        Data-model classes.  Imported from ``main`` when omitted.

    Returns
    -------
    RadarVolume
    """
    if not NETCDF4_AVAILABLE:
        raise RuntimeError(
            "netCDF4 is required to read CFRadial files (pip install netCDF4)."
        )

    # Import data model from main unless caller provided them
    if RadarVolume is None or SweepData is None:
        from main import RadarVolume, SweepData  # type: ignore[assignment]

    nc  = netCDF4.Dataset(path, "r")
    vol = RadarVolume()

    # ── Site metadata ──────────────────────────────────────────────────────
    vol.radar_name = (
        str(getattr(nc, "instrument_name", "") or "").strip()
        or str(getattr(nc, "site_name",        "") or "").strip()
        or "Unknown"
    )
    vol.latitude  = _nc_scalar(nc, "latitude")  or 0.0
    vol.longitude = _nc_scalar(nc, "longitude") or 0.0
    vol.altitude  = _nc_scalar(nc, "altitude")  or 0.0

    # ── Base time (ordered preference) ────────────────────────────────────
    base_dt: Optional[datetime] = None

    # 1. time variable units  ("seconds since …")
    if "time" in nc.variables:
        units_str = getattr(nc.variables["time"], "units", "") or ""
        if "since" in units_str.lower():
            after = units_str.split("since", 1)[-1].strip()
            base_dt = _parse_datetime(after)

    # 2. start_datetime / time_coverage_start global attributes
    if base_dt is None:
        for attr in ("start_datetime", "time_coverage_start"):
            raw = getattr(nc, attr, None)
            if raw:
                base_dt = _parse_datetime(str(raw))
                if base_dt:
                    break

    if base_dt is None:
        base_dt = datetime(2000, 1, 1)

    vol.scan_time = base_dt

    # ── Shared geometry variables ──────────────────────────────────────────
    if "time" not in nc.variables:
        nc.close()
        raise ValueError(f"CFRadial file has no 'time' variable: {path}")

    ray_times    = np.asarray(nc.variables["time"][:],      dtype="float64")
    azimuth      = np.asarray(nc.variables["azimuth"][:],   dtype="float32")
    elevation    = np.asarray(nc.variables["elevation"][:], dtype="float32")
    ranges       = np.asarray(nc.variables["range"][:],     dtype="float32")
    start_idxs   = np.asarray(nc.variables["sweep_start_ray_index"][:], dtype=int)
    end_idxs     = np.asarray(nc.variables["sweep_end_ray_index"][:],   dtype=int)
    fixed_angles = np.asarray(nc.variables["fixed_angle"][:],           dtype="float32")

    nrays_total = len(azimuth)
    ngates      = len(ranges)

    # ── Sweep mode strings ────────────────────────────────────────────────
    sweep_modes: list[str] = []
    if "sweep_mode" in nc.variables:
        sweep_modes = [
            s.strip().lower() for s in _decode_char_strings(nc.variables["sweep_mode"])
        ]

    # ── Field names: 2-D (nrays_total × ngates) variables ─────────────────
    field_names = [
        v for v, var in nc.variables.items()
        if var.ndim == 2 and var.shape == (nrays_total, ngates)
    ]
    vol.field_names = field_names

    # ── Instrument parameters ─────────────────────────────────────────────
    vol.instrument_parameters = {}
    for ip_key in ("nyquist_velocity", "prt", "prt_ratio", "frequency"):
        v = _nc_median_positive(nc, ip_key)
        if v is not None:
            dtype = "float32" if ip_key == "nyquist_velocity" else "float64"
            vol.instrument_parameters[ip_key] = {
                "data": np.array([v], dtype=dtype)
            }

    # ── Per-sweep extraction ───────────────────────────────────────────────
    for si in range(len(start_idxs)):
        s = int(start_idxs[si])
        e = int(end_idxs[si])

        sw = SweepData()
        sw.radar_name  = vol.radar_name
        sw.fixed_angle = float(fixed_angles[si])
        sw.azimuth     = azimuth[s : e + 1]
        sw.elevation   = elevation[s : e + 1]
        sw.ranges      = ranges

        # RHI detection from sweep_mode variable
        mode_str = sweep_modes[si] if si < len(sweep_modes) else ""
        sw.scan_type = "rhi" if "rhi" in mode_str else "ppi"

        # Scan time = base time + offset of the sweep's first ray
        try:
            sw.scan_time = base_dt + timedelta(seconds=float(ray_times[s]))
        except Exception:
            sw.scan_time = base_dt

        # Field data
        for fname in field_names:
            raw = nc.variables[fname][s : e + 1, :]
            arr = (
                raw.filled(np.nan)
                if hasattr(raw, "filled")
                else np.asarray(raw, dtype="float32")
            )
            sw.fields[fname] = np.ma.masked_invalid(arr.astype("float32"))

        vol.sweeps.append(sw)

    nc.close()
    return vol